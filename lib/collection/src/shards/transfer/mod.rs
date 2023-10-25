use async_trait::async_trait;
use common::defaults;

use super::channel_service::ChannelService;
use super::shard::PeerId;
use crate::operations::types::CollectionResult;

pub mod shard_transfer;
pub mod transfer_tasks_pool;

/// Interface to consensus for shard transfer operations.
#[async_trait]
pub trait ShardTransferConsensus: Send + Sync {
    /// Get the current consensus commit and term state.
    ///
    /// Returns `(commit, term)`.
    fn consensus_commit_term(&self) -> (u64, u64);

    /// Wait for all other peers to reach the current consensus
    ///
    /// This will take the current consensus state of this node. It then explicitly awaits on all
    /// other nodes to reach this consensus state.
    ///
    /// # Errors
    ///
    /// This errors if:
    /// - any of the peers is not on the same term
    /// - waiting takes longer than the specified timeout
    /// - any of the peers cannot be reached
    async fn await_consensus_sync(
        &self,
        this_peer_id: PeerId,
        channel_service: &ChannelService,
    ) -> CollectionResult<()> {
        let (commit, term) = self.consensus_commit_term();
        channel_service
            .await_commit_on_all_peers(this_peer_id, commit, term, defaults::CONSENSUS_META_OP_WAIT)
            .await
    }
}
