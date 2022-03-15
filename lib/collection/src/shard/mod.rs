pub mod local_shard;
pub mod remote_shard;

use crate::collection_manager::holders::segment_holder::SegmentHolder;
use crate::shard::remote_shard::RemoteShard;
use crate::{
    CollectionInfo, CollectionResult, CollectionSearcher, CollectionUpdateOperations, LocalShard,
    OptimizersConfigDiff, Record, UpdateResult,
};
use async_trait::async_trait;
use parking_lot::RwLock;
use segment::types::{ExtendedPointId, Filter, WithPayloadInterface};
use std::sync::Arc;

pub type ShardId = u32;

pub type PeerId = u32;

/// Shard
///
/// A shard can either be local or remote
///
#[allow(clippy::large_enum_variant)]
pub enum Shard {
    Local(LocalShard),
    Remote(RemoteShard),
}

impl Shard {
    pub fn get(&self) -> Arc<dyn ShardOperation + Sync + Send + '_> {
        match self {
            Shard::Local(local_shard) => Arc::new(local_shard),
            Shard::Remote(remote_shard) => Arc::new(remote_shard),
        }
    }

    pub async fn before_drop(&mut self) {
        match self {
            Shard::Local(local_shard) => local_shard.before_drop().await,
            Shard::Remote(_) => (),
        }
    }
}

#[async_trait]
pub trait ShardOperation {
    async fn update(
        &self,
        operation: CollectionUpdateOperations,
        wait: bool,
    ) -> CollectionResult<UpdateResult>;

    fn segments(&self) -> &RwLock<SegmentHolder>;

    async fn scroll_by(
        &self,
        segment_searcher: &(dyn CollectionSearcher + Sync),
        offset: Option<ExtendedPointId>,
        limit: usize,
        with_payload_interface: &WithPayloadInterface,
        with_vector: bool,
        filter: Option<&Filter>,
    ) -> CollectionResult<Vec<Record>>;

    async fn update_optimizer_params(
        &self,
        optimizer_config_diff: OptimizersConfigDiff,
    ) -> CollectionResult<()>;

    async fn info(&self) -> CollectionResult<CollectionInfo>;
}
