use super::*;

impl TableOfContent {
    pub fn snapshots_path(&self) -> &str {
        &self.storage_config.snapshots_path
    }

    pub fn collection_snapshots_path(snapshots_path: &Path, collection_name: &str) -> PathBuf {
        snapshots_path.join(collection_name)
    }

    pub fn snapshots_path_for_collection(&self, collection_name: &str) -> PathBuf {
        Self::collection_snapshots_path(
            Path::new(&self.storage_config.snapshots_path),
            collection_name,
        )
    }

    pub async fn create_snapshots_path(
        &self,
        collection_name: &str,
    ) -> Result<PathBuf, StorageError> {
        let snapshots_path = self.snapshots_path_for_collection(collection_name);
        tokio::fs::create_dir_all(&snapshots_path)
            .await
            .map_err(|err| {
                StorageError::service_error(format!(
                    "Can't create directory for snapshots {collection_name}. Error: {err}"
                ))
            })?;

        Ok(snapshots_path)
    }

    pub async fn create_snapshot(
        &self,
        collection_name: &str,
    ) -> Result<SnapshotDescription, StorageError> {
        let collection = self.get_collection(collection_name).await?;
        // We want to use temp dir inside the temp_path (storage if not specified), because it is possible, that
        // snapshot directory is mounted as network share and multiple writes to it could be slow
        let temp_dir = self.optional_temp_or_storage_temp_path()?;
        Ok(collection
            .create_snapshot(&temp_dir, self.this_peer_id)
            .await?)
    }

    pub fn send_set_replica_state_proposal(
        &self,
        collection_name: String,
        peer_id: PeerId,
        shard_id: ShardId,
        state: ReplicaState,
        from_state: Option<ReplicaState>,
    ) -> Result<(), StorageError> {
        if let Some(operation_sender) = &self.consensus_proposal_sender {
            Self::send_set_replica_state_proposal_op(
                operation_sender,
                collection_name,
                peer_id,
                shard_id,
                state,
                from_state,
            )?;
        }
        Ok(())
    }

    pub fn request_remove_replica(
        &self,
        collection_name: String,
        shard_id: ShardId,
        peer_id: PeerId,
    ) -> Result<(), StorageError> {
        if let Some(proposal_sender) = &self.consensus_proposal_sender {
            Self::send_remove_replica_proposal_op(
                proposal_sender,
                collection_name,
                peer_id,
                shard_id,
            )?;
        }
        Ok(())
    }

    fn send_remove_replica_proposal_op(
        proposal_sender: &OperationSender,
        collection_name: String,
        peer_id: PeerId,
        shard_id: ShardId,
    ) -> Result<(), StorageError> {
        let operation = ConsensusOperations::remove_replica(collection_name, shard_id, peer_id);
        proposal_sender.send(operation)
    }

    pub fn request_shard_transfer(
        &self,
        collection_name: String,
        shard_id: ShardId,
        from_peer: PeerId,
        to_peer: PeerId,
        sync: bool,
    ) -> Result<(), StorageError> {
        if let Some(proposal_sender) = &self.consensus_proposal_sender {
            let transfer_request = ShardTransfer {
                shard_id,
                from: from_peer,
                to: to_peer,
                sync,
            };
            let operation = ConsensusOperations::start_transfer(collection_name, transfer_request);
            proposal_sender.send(operation)?;
        }
        Ok(())
    }
}
