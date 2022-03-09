use collection::operations::config_diff::{HnswConfigDiff, OptimizersConfigDiff, WalConfigDiff};
use schemars::JsonSchema;
use segment::types::Distance;
use serde::{Deserialize, Serialize};

// *Operation wrapper structure is only required for better OpenAPI generation

/// Create alternative name for a collection.
/// Collection will be available under both names for search, retrieve,
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct CreateAlias {
    pub collection_name: String,
    pub alias_name: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct CreateAliasOperation {
    pub create_alias: CreateAlias,
}

/// Delete alias if exists
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct DeleteAlias {
    pub alias_name: String,
}

/// Delete alias if exists
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct DeleteAliasOperation {
    pub delete_alias: DeleteAlias,
}

/// Change alias to a new one
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct RenameAlias {
    pub old_alias_name: String,
    pub new_alias_name: String,
}

/// Change alias to a new one
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct RenameAliasOperation {
    pub rename_alias: RenameAlias,
}

/// Group of all the possible operations related to collection aliases
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[serde(untagged)]
pub enum AliasOperations {
    CreateAlias(CreateAliasOperation),
    DeleteAlias(DeleteAliasOperation),
    RenameAlias(RenameAliasOperation),
}

impl From<CreateAlias> for AliasOperations {
    fn from(create_alias: CreateAlias) -> Self {
        AliasOperations::CreateAlias(CreateAliasOperation { create_alias })
    }
}

impl From<DeleteAlias> for AliasOperations {
    fn from(delete_alias: DeleteAlias) -> Self {
        AliasOperations::DeleteAlias(DeleteAliasOperation { delete_alias })
    }
}

impl From<RenameAlias> for AliasOperations {
    fn from(rename_alias: RenameAlias) -> Self {
        AliasOperations::RenameAlias(RenameAliasOperation { rename_alias })
    }
}

/// Operation for creating new collection and (optionally) specify index params
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct CreateCollection {
    pub vector_size: usize,
    pub distance: Distance,
    #[serde(default = "default_shard_number")]
    pub shard_number: u32,
    /// Custom params for HNSW index. If none - values from service configuration file are used.
    pub hnsw_config: Option<HnswConfigDiff>,
    /// Custom params for WAL. If none - values from service configuration file are used.
    pub wal_config: Option<WalConfigDiff>,
    /// Custom params for Optimizers.  If none - values from service configuration file are used.
    pub optimizers_config: Option<OptimizersConfigDiff>,
}

const fn default_shard_number() -> u32 {
    1
}

/// Operation for creating new collection and (optionally) specify index params
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct CreateCollectionOperation {
    pub collection_name: String,
    #[serde(flatten)]
    pub create_collection: CreateCollection,
}

/// Operation for updating parameters of the existing collection
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct UpdateCollection {
    /// Custom params for Optimizers.  If none - values from service configuration file are used.
    /// This operation is blocking, it will only proceed ones all current optimizations are complete
    pub optimizers_config: Option<OptimizersConfigDiff>, // ToDo: Allow updates for other configuration params as well
}

/// Operation for updating parameters of the existing collection
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct UpdateCollectionOperation {
    pub collection_name: String,
    #[serde(flatten)]
    pub update_collection: UpdateCollection,
}

/// Operation for performing changes of collection aliases.
/// Alias changes are atomic, meaning that no collection modifications can happen between
/// alias operations.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ChangeAliasesOperation {
    pub actions: Vec<AliasOperations>,
}

/// Operation for deleting collection with given name
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct DeleteCollectionOperation(pub String);

/// Enumeration of all possible collection update operations
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StorageOperations {
    CreateCollection(CreateCollectionOperation),
    UpdateCollection(UpdateCollectionOperation),
    DeleteCollection(DeleteCollectionOperation),
    ChangeAliases(ChangeAliasesOperation),
}
