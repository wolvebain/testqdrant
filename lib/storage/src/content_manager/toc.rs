use std::collections::HashMap;
use std::fs::{create_dir_all, read_dir, remove_dir_all};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::runtime::Runtime;
use tokio::sync::{RwLock, RwLockReadGuard};

use collection::config::{CollectionConfig, CollectionParams};
use collection::operations::config_diff::DiffConfig;
use collection::operations::types::{
    PointRequest, RecommendRequest, Record, ScrollRequest, ScrollResult, SearchRequest,
    UpdateResult,
};
use collection::operations::CollectionUpdateOperations;
use collection::Collection;
use segment::types::ScoredPoint;

use crate::content_manager::collection_meta_ops::{
    AliasOperations, ChangeAliasesOperation, CollectionMetaOperations, CreateAlias,
    CreateAliasOperation, CreateCollection, DeleteAlias, DeleteAliasOperation, RenameAlias,
    RenameAliasOperation, UpdateCollection,
};
use crate::content_manager::alias_mapping::AliasPersistence;
use crate::content_manager::collections_ops::{Checker, Collections};
use crate::content_manager::errors::StorageError;
use crate::types::StorageConfig;
use collection::collection_manager::collection_managers::CollectionSearcher;
use collection::collection_manager::simple_collection_searcher::SimpleCollectionSearcher;

#[cfg(feature = "consensus")]
use raft::{eraftpb::Entry as RaftEntry, RaftState};
use std::ops::Deref;
#[cfg(feature = "consensus")]
use wal::Wal;

const COLLECTIONS_DIR: &str = "collections";
#[cfg(feature = "consensus")]
const COLLECTIONS_META_WAL_DIR: &str = "collections_meta_wal";

/// The main object of the service. It holds all objects, required for proper functioning.
/// In most cases only one `TableOfContent` is enough for service. It is created only once during
/// the launch of the service.
pub struct TableOfContent {
    collections: Arc<RwLock<Collections>>,
    storage_config: StorageConfig,
    search_runtime: Runtime,
    collection_management_runtime: Runtime,
    alias_persistence: AliasPersistence,
    segment_searcher: Box<dyn CollectionSearcher + Sync + Send>,

    #[cfg(feature = "consensus")]
    collection_meta_wal: Arc<std::sync::Mutex<Wal>>,
    #[cfg(feature = "consensus")]
    raft_state: RaftState,
}

impl TableOfContent {
    pub fn new(storage_config: &StorageConfig, search_runtime: Runtime) -> Self {
        let collections_path = Path::new(&storage_config.storage_path).join(&COLLECTIONS_DIR);
        let collection_management_runtime = Runtime::new().unwrap();

        create_dir_all(&collections_path).expect("Can't create Collections directory");

        let collection_paths =
            read_dir(&collections_path).expect("Can't read Collections directory");

        let mut collections: HashMap<String, Collection> = Default::default();

        for entry in collection_paths {
            let collection_path = entry
                .expect("Can't access of one of the collection files")
                .path();
            let collection_name = collection_path
                .file_name()
                .expect("Can't resolve a filename of one of the collection files")
                .to_str()
                .expect("A filename of one of the collection files is not a valid UTF-8")
                .to_string();

            let collection = collection_management_runtime
                .block_on(Collection::load(collection_name.clone(), &collection_path));

            collections.insert(collection_name, collection);
        }

        let alias_path = Path::new(&storage_config.storage_path).join("aliases");

        let alias_persistence =
            AliasPersistence::open(alias_path).expect("Can't open database by the provided config");

        #[cfg(feature = "consensus")]
        let collection_meta_wal = {
            let collections_meta_wal_path =
                Path::new(&storage_config.storage_path).join(&COLLECTIONS_META_WAL_DIR);
            create_dir_all(&collections_meta_wal_path)
                .expect("Can't create Collections meta Wal directory");
            Arc::new(std::sync::Mutex::new(
                Wal::open(collections_meta_wal_path).unwrap(),
            ))
        };

        TableOfContent {
            collections: Arc::new(RwLock::new(collections)),
            storage_config: storage_config.clone(),
            search_runtime,
            alias_persistence,
            segment_searcher: Box::new(SimpleCollectionSearcher::new()),
            collection_management_runtime,
            #[cfg(feature = "consensus")]
            collection_meta_wal,
            #[cfg(feature = "consensus")]
            raft_state: RaftState::default(),
        }
    }

    fn get_collection_path(&self, collection_name: &str) -> PathBuf {
        Path::new(&self.storage_config.storage_path)
            .join(&COLLECTIONS_DIR)
            .join(collection_name)
    }

    fn create_collection_path(&self, collection_name: &str) -> Result<PathBuf, StorageError> {
        let path = self.get_collection_path(collection_name);

        create_dir_all(&path).map_err(|err| StorageError::ServiceError {
            description: format!(
                "Can't create directory for collection {}. Error: {}",
                collection_name, err
            ),
        })?;

        Ok(path)
    }

    /// Finds the original name of the collection
    ///
    /// # Arguments
    ///
    /// * `collection_name` - Name of the collection or alias to resolve
    ///
    /// # Result
    ///
    /// If the collection exists - return its name
    /// If alias exists - returns the original collection name
    /// If neither exists - returns [`StorageError`]
    async fn resolve_name(&self, collection_name: &str) -> Result<String, StorageError> {
        let alias_collection_name = self.alias_persistence.get(collection_name)?;

        let resolved_name = match alias_collection_name {
            None => collection_name.to_string(),
            Some(resolved_alias) => resolved_alias,
        };
        self.collections
            .read()
            .await
            .validate_collection_exists(&resolved_name)
            .await?;
        Ok(resolved_name)
    }

    pub async fn create_collection(
        &self,
        collection_name: &str,
        operation: CreateCollection,
    ) -> Result<bool, StorageError> {
        let CreateCollection {
            vector_size,
            distance,
            shard_number,
            hnsw_config: hnsw_config_diff,
            wal_config: wal_config_diff,
            optimizers_config: optimizers_config_diff,
        } = operation;

        self.collections
            .read()
            .await
            .validate_collection_not_exists(collection_name)
            .await?;

        let collection_path = self.create_collection_path(collection_name)?;

        let collection_params = CollectionParams {
            vector_size,
            distance,
            shard_number: NonZeroU32::new(shard_number).ok_or(StorageError::BadInput {
                description: "`shard_number` cannot be 0".to_string(),
            })?,
        };
        let wal_config = match wal_config_diff {
            None => self.storage_config.wal.clone(),
            Some(diff) => diff.update(&self.storage_config.wal)?,
        };

        let optimizers_config = match optimizers_config_diff {
            None => self.storage_config.optimizers.clone(),
            Some(diff) => diff.update(&self.storage_config.optimizers)?,
        };

        let hnsw_config = match hnsw_config_diff {
            None => self.storage_config.hnsw_index,
            Some(diff) => diff.update(&self.storage_config.hnsw_index)?,
        };

        let collection = Collection::new(
            collection_name.to_string(),
            Path::new(&collection_path),
            &CollectionConfig {
                wal_config,
                params: collection_params,
                optimizer_config: optimizers_config,
                hnsw_config,
            },
        )
        .await?;

        let mut write_collections = self.collections.write().await;
        write_collections
            .validate_collection_not_exists(collection_name)
            .await?;
        write_collections.insert(collection_name.to_string(), collection);
        Ok(true)
    }

    pub async fn update_collection(
        &self,
        collection_name: &str,
        operation: UpdateCollection,
    ) -> Result<bool, StorageError> {
        match operation.optimizers_config {
            None => {}
            Some(new_optimizers_config) => {
                let collection = self.get_collection(collection_name).await?;
                collection
                    .update_optimizer_params(new_optimizers_config)
                    .await?
            }
        }
        Ok(true)
    }

    pub async fn delete_collection(&self, collection_name: &str) -> Result<bool, StorageError> {
        if let Some(mut removed) = self.collections.write().await.remove(collection_name) {
            removed.before_drop().await;
            let path = self.get_collection_path(collection_name);
            remove_dir_all(path).map_err(|err| StorageError::ServiceError {
                description: format!(
                    "Can't delete collection {}, error: {}",
                    collection_name, err
                ),
            })?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub async fn update_aliases(
        &self,
        operation: ChangeAliasesOperation,
    ) -> Result<bool, StorageError> {
        // Lock all collections for alias changes
        // Prevent search on partially switched collections
        let collection_lock = self.collections.write().await;
        for action in operation.actions {
            match action {
                AliasOperations::CreateAlias(CreateAliasOperation {
                    create_alias:
                        CreateAlias {
                            collection_name,
                            alias_name,
                        },
                }) => {
                    collection_lock
                        .validate_collection_exists(&collection_name)
                        .await?;
                    collection_lock
                        .validate_collection_not_exists(&alias_name)
                        .await?;

                    self.alias_persistence.insert(alias_name, collection_name)?;
                }
                AliasOperations::DeleteAlias(DeleteAliasOperation {
                    delete_alias: DeleteAlias { alias_name },
                }) => {
                    self.alias_persistence.remove(&alias_name)?;
                }
                AliasOperations::RenameAlias(RenameAliasOperation {
                    rename_alias:
                        RenameAlias {
                            old_alias_name,
                            new_alias_name,
                        },
                }) => {
                    if !self.alias_persistence.contains_alias(&old_alias_name)? {
                        return Err(StorageError::NotFound {
                            description: format!("Alias {} does not exists!", old_alias_name),
                        });
                    }

                    // safe Option.unwrap as the alias mapping is currently locked exclusively
                    let collection = self.alias_persistence.remove(&old_alias_name)?.unwrap();
                    // remove + insert is not transactional
                    self.alias_persistence.insert(new_alias_name, collection)?
                }
            };
        }
        Ok(true)
    }

    pub async fn perform_collection_operation(
        &self,
        operation: CollectionMetaOperations,
    ) -> Result<bool, StorageError> {
        match operation {
            CollectionMetaOperations::CreateCollection(operation) => {
                self.create_collection(&operation.collection_name, operation.create_collection)
                    .await
            }
            CollectionMetaOperations::UpdateCollection(operation) => {
                self.update_collection(&operation.collection_name, operation.update_collection)
                    .await
            }
            CollectionMetaOperations::DeleteCollection(operation) => {
                self.delete_collection(&operation.0).await
            }
            CollectionMetaOperations::ChangeAliases(operation) => {
                self.update_aliases(operation).await
            }
        }
    }

    pub async fn get_collection<'a>(
        &'a self,
        collection_name: &str,
    ) -> Result<RwLockReadGuard<'a, Collection>, StorageError> {
        let read_collection = self.collections.read().await;
        let real_collection_name = self.resolve_name(collection_name).await?;
        // resolve_name already checked collection existence, unwrap is safe here
        Ok(RwLockReadGuard::map(read_collection, |collection| {
            collection.get(&real_collection_name).unwrap()
        }))
    }

    /// Recommend points using positive and negative example from the request
    ///
    /// # Arguments
    ///
    /// * `collection_name` - for what collection do we recommend
    /// * `request` - [`RecommendRequest`]
    ///
    /// # Result
    ///
    /// Points with recommendation score
    pub async fn recommend(
        &self,
        collection_name: &str,
        request: RecommendRequest,
    ) -> Result<Vec<ScoredPoint>, StorageError> {
        let collection = self.get_collection(collection_name).await?;
        collection
            .recommend_by(
                request,
                self.segment_searcher.deref(),
                self.search_runtime.handle(),
            )
            .await
            .map_err(|err| err.into())
    }

    /// Search for the closest points using vector similarity with given restrictions defined
    /// in the request
    ///
    /// # Arguments
    ///
    /// * `collection_name` - in what collection do we search
    /// * `request` - [`SearchRequest`]
    ///
    /// # Result
    ///
    /// Points with search score
    pub async fn search(
        &self,
        collection_name: &str,
        request: SearchRequest,
    ) -> Result<Vec<ScoredPoint>, StorageError> {
        let collection = self.get_collection(collection_name).await?;
        collection
            .search(
                request,
                self.segment_searcher.as_ref(),
                self.search_runtime.handle(),
            )
            .await
            .map_err(|err| err.into())
    }

    /// Return specific points by IDs
    ///
    /// # Arguments
    ///
    /// * `collection_name` - select from this collection
    /// * `request` - [`PointRequest`]
    ///
    /// # Result
    ///
    /// List of points with specified information included
    pub async fn retrieve(
        &self,
        collection_name: &str,
        request: PointRequest,
    ) -> Result<Vec<Record>, StorageError> {
        let collection = self.get_collection(collection_name).await?;
        collection
            .retrieve(request, self.segment_searcher.as_ref())
            .await
            .map_err(|err| err.into())
    }

    /// List of all collections
    pub async fn all_collections(&self) -> Vec<String> {
        self.collections.read().await.keys().cloned().collect()
    }

    /// List of all aliases for a given collection
    pub fn collection_aliases(&self, collection_name: &str) -> Result<Vec<String>, StorageError> {
        let result = self.alias_persistence.collection_aliases(collection_name)?;
        Ok(result)
    }

    /// Paginate over all stored points with given filtering conditions
    ///
    /// # Arguments
    ///
    /// * `collection_name` - which collection to use
    /// * `request` - [`ScrollRequest`]
    ///
    /// # Result
    ///
    /// List of points with specified information included
    pub async fn scroll(
        &self,
        collection_name: &str,
        request: ScrollRequest,
    ) -> Result<ScrollResult, StorageError> {
        let collection = self.get_collection(collection_name).await?;
        collection
            .scroll_by(request, self.segment_searcher.deref())
            .await
            .map_err(|err| err.into())
    }

    pub async fn update(
        &self,
        collection_name: &str,
        operation: CollectionUpdateOperations,
        wait: bool,
    ) -> Result<UpdateResult, StorageError> {
        let collection = self.get_collection(collection_name).await?;
        collection
            .update(operation, wait)
            .await
            .map_err(|err| err.into())
    }

    #[cfg(feature = "consensus")]
    pub fn collection_wal_entry(&self, id: u64) -> raft::Result<RaftEntry> {
        <RaftEntry as prost::Message>::decode(
            self.collection_meta_wal
                .lock()
                .map_err(consensus::raft_error_other)?
                .entry(id)
                .ok_or(raft::Error::Store(raft::StorageError::Unavailable))?
                .as_ref(),
        )
        .map_err(consensus::raft_error_other)
    }

    pub fn this_peer_id(&self) -> u32 {
        0
    }

    #[cfg(feature = "consensus")]
    async fn collection_meta_snapshot(&self) -> consensus::CollectionMetaSnapshot {
        let collections: HashMap<collection::CollectionId, collection::State> = self
            .collections
            .read()
            .await
            .iter()
            .map(|(id, collection)| (id.clone(), collection.state(self.this_peer_id())))
            .collect();
        consensus::CollectionMetaSnapshot {
            collections,
            // TODO: fill aliases
            aliases: HashMap::new(),
        }
    }

    #[cfg(feature = "consensus")]
    #[allow(dead_code)]
    async fn apply_entry(&self, entry: &RaftEntry) -> Result<(), StorageError> {
        let operation: CollectionMetaOperations = entry.try_into()?;
        self.perform_collection_operation(operation).await?;
        Ok(())
    }
}

// `TableOfContent` should not be dropped from async context.
impl Drop for TableOfContent {
    fn drop(&mut self) {
        self.collection_management_runtime.block_on(async {
            for (_, mut collection) in self.collections.write().await.drain() {
                collection.before_drop().await;
            }
        });
    }
}

#[cfg(feature = "consensus")]
mod consensus {
    use std::{collections::HashMap, sync::Arc};

    use collection::CollectionId;
    use raft::{eraftpb::Entry as RaftEntry, storage::Storage as RaftStorage, RaftState};
    use serde::{Deserialize, Serialize};

    use super::TableOfContent;

    impl RaftStorage for TableOfContent {
        fn initial_state(&self) -> raft::Result<RaftState> {
            Ok(RaftState::default())
        }

        fn entries(
            &self,
            low: u64,
            high: u64,
            max_size: impl Into<Option<u64>>,
        ) -> raft::Result<Vec<RaftEntry>> {
            let max_size: Option<_> = max_size.into();
            (low..high)
                .take(max_size.unwrap_or(high - low + 1) as usize)
                .map(|id| self.collection_wal_entry(id))
                .collect()
        }

        fn term(&self, idx: u64) -> raft::Result<u64> {
            Ok(self.collection_wal_entry(idx)?.term)
        }

        fn first_index(&self) -> raft::Result<u64> {
            Ok(self
                .collection_meta_wal
                .lock()
                .map_err(raft_error_other)?
                .first_index()
                + 1)
        }

        fn last_index(&self) -> raft::Result<u64> {
            Ok(self
                .collection_meta_wal
                .lock()
                .map_err(raft_error_other)?
                .num_entries())
        }

        fn snapshot(&self, request_index: u64) -> raft::Result<raft::eraftpb::Snapshot> {
            let snapshot = self
                .collection_management_runtime
                .block_on(self.collection_meta_snapshot());
            if self.raft_state.hard_state.commit >= request_index {
                Ok(raft::eraftpb::Snapshot {
                    data: serde_cbor::to_vec(&snapshot).map_err(raft_error_other)?,
                    metadata: Some(raft::eraftpb::SnapshotMetadata {
                        conf_state: Some(self.raft_state.conf_state.clone()),
                        index: self.raft_state.hard_state.commit,
                        term: self.raft_state.hard_state.term,
                    }),
                })
            } else {
                Err(raft::Error::Store(
                    raft::StorageError::SnapshotTemporarilyUnavailable,
                ))
            }
        }
    }

    #[derive(Clone)]
    struct TableOfContentRef(Arc<TableOfContent>);

    impl RaftStorage for TableOfContentRef {
        fn initial_state(&self) -> raft::Result<raft::RaftState> {
            self.0.initial_state()
        }

        fn entries(
            &self,
            low: u64,
            high: u64,
            max_size: impl Into<Option<u64>>,
        ) -> raft::Result<Vec<raft::eraftpb::Entry>> {
            self.0.entries(low, high, max_size)
        }

        fn term(&self, idx: u64) -> raft::Result<u64> {
            self.0.term(idx)
        }

        fn first_index(&self) -> raft::Result<u64> {
            self.0.first_index()
        }

        fn last_index(&self) -> raft::Result<u64> {
            self.0.last_index()
        }

        fn snapshot(&self, request_index: u64) -> raft::Result<raft::eraftpb::Snapshot> {
            self.0.snapshot(request_index)
        }
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct CollectionMetaSnapshot {
        pub collections: HashMap<CollectionId, collection::State>,
        pub aliases: HashMap<String, CollectionId>,
    }

    impl TryFrom<&raft::eraftpb::Snapshot> for CollectionMetaSnapshot {
        type Error = serde_cbor::Error;

        fn try_from(snapshot: &raft::eraftpb::Snapshot) -> Result<Self, Self::Error> {
            serde_cbor::from_slice(snapshot.get_data())
        }
    }

    #[derive(thiserror::Error, Debug)]
    #[error("{0}")]
    struct StrError(String);

    pub fn raft_error_other(e: impl std::error::Error) -> raft::Error {
        raft::Error::Store(raft::StorageError::Other(Box::new(StrError(e.to_string()))))
    }
}
