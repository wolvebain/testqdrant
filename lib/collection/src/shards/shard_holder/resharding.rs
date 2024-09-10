use std::collections::HashMap;
use std::fmt;
use std::ops::Deref as _;
use std::sync::Arc;

use segment::types::{Condition, Filter, ShardKey};

use super::ShardHolder;
use crate::hash_ring::{self, HashRingRouter};
use crate::operations::cluster_ops::ReshardingDirection;
use crate::operations::types::{CollectionError, CollectionResult, UpdateResult};
use crate::shards::replica_set::{ReplicaState, ShardReplicaSet};
use crate::shards::resharding::{ReshardKey, ReshardStage, ReshardState};
use crate::shards::shard::ShardId;

impl ShardHolder {
    pub fn check_start_resharding(&mut self, resharding_key: &ReshardKey) -> CollectionResult<()> {
        let ReshardKey {
            direction,
            peer_id: _,
            shard_id,
            shard_key,
        } = resharding_key;

        let ring = get_ring(&mut self.rings, shard_key)?;

        {
            let state = self.resharding_state.read();
            assert_resharding_state_consistency(&state, ring, shard_key);

            if let Some(state) = state.deref() {
                if state.matches(resharding_key) {
                    return Err(CollectionError::bad_request(format!(
                        "resharding {resharding_key} is already in progress:\n{state:#?}"
                    )));
                } else {
                    return Err(CollectionError::bad_request(format!(
                        "another resharding is in progress:\n{state:#?}"
                    )));
                }
            }
        }

        // Don't remove the last shard if resharding down
        if matches!(direction, ReshardingDirection::Down) {
            let shard_count = match shard_key {
                Some(shard_key) => self
                    .get_shard_key_to_ids_mapping()
                    .get(shard_key)
                    .map_or(0, |shards| shards.len()),
                None => self.shards.len(),
            };
            if shard_count <= 1 {
                return Err(CollectionError::bad_request(format!(
                    "cannot remove shard {shard_id} by resharding down, it is the last shard",
                )));
            }
        }

        let has_shard = self.shards.contains_key(shard_id);
        match resharding_key.direction {
            ReshardingDirection::Up => {
                if has_shard {
                    return Err(CollectionError::bad_request(format!(
                        "shard holder already contains shard {shard_id} replica set",
                    )));
                }
            }
            ReshardingDirection::Down => {
                if !has_shard {
                    return Err(CollectionError::bad_request(format!(
                        "shard holder does not contain shard {shard_id} replica set",
                    )));
                }
            }
        }

        // TODO(resharding): Check that peer exists!?

        Ok(())
    }

    // TODO: do not leave broken intermediate state if this fails midway?
    pub fn start_resharding_unchecked(
        &mut self,
        resharding_key: ReshardKey,
        new_shard: Option<ShardReplicaSet>,
    ) -> CollectionResult<()> {
        let ReshardKey {
            direction,
            peer_id,
            shard_id,
            shard_key,
        } = resharding_key;

        // TODO(resharding): Delete shard on error!?

        let ring = get_ring(&mut self.rings, &shard_key)?;
        ring.start_resharding(shard_id, direction);

        // Add new shard if resharding up
        if let Some(new_shard) = new_shard {
            debug_assert_eq!(direction, ReshardingDirection::Up);
            self.add_shard(shard_id, new_shard, shard_key.clone())?;
        }

        self.resharding_state.write(|state| {
            debug_assert!(
                state.is_none(),
                "resharding is already in progress:\n{state:#?}",
            );

            *state = Some(ReshardState::new(direction, peer_id, shard_id, shard_key));
        })?;

        Ok(())
    }

    pub fn commit_read_hashring(&mut self, resharding_key: ReshardKey) -> CollectionResult<()> {
        self.check_resharding(&resharding_key, check_stage(ReshardStage::MigratingPoints))?;

        self.resharding_state.write(|state| {
            let Some(state) = state else {
                unreachable!();
            };

            state.stage = ReshardStage::ReadHashRingCommitted;
        })?;

        Ok(())
    }

    pub fn commit_write_hashring(&mut self, resharding_key: ReshardKey) -> CollectionResult<()> {
        self.check_resharding(
            &resharding_key,
            check_stage(ReshardStage::ReadHashRingCommitted),
        )?;

        let ring = get_ring(&mut self.rings, &resharding_key.shard_key)?;
        ring.commit_resharding();

        self.resharding_state.write(|state| {
            let Some(state) = state else {
                unreachable!();
            };

            state.stage = ReshardStage::WriteHashRingCommitted;
        })?;

        Ok(())
    }

    pub fn check_finish_resharding(&mut self, resharding_key: &ReshardKey) -> CollectionResult<()> {
        self.check_resharding(
            resharding_key,
            check_stage(ReshardStage::WriteHashRingCommitted),
        )?;

        Ok(())
    }

    pub fn finish_resharding_unchecked(&mut self, _: &ReshardKey) -> CollectionResult<()> {
        self.resharding_state.write(|state| {
            debug_assert!(state.is_some(), "resharding is not in progress");
            *state = None;
        })?;

        Ok(())
    }

    fn check_resharding(
        &mut self,
        resharding_key: &ReshardKey,
        check_state: impl Fn(&ReshardState) -> CollectionResult<()>,
    ) -> CollectionResult<()> {
        let ReshardKey {
            shard_id,
            shard_key,
            ..
        } = resharding_key;

        let ring = get_ring(&mut self.rings, shard_key)?;

        let state = self.resharding_state.read();
        assert_resharding_state_consistency(&state, ring, &resharding_key.shard_key);

        match state.deref() {
            Some(state) if state.matches(resharding_key) => {
                check_state(state)?;
            }

            Some(state) => {
                return Err(CollectionError::bad_request(format!(
                    "another resharding is in progress:\n{state:#?}"
                )));
            }

            None => {
                return Err(CollectionError::bad_request(
                    "resharding is not in progress",
                ));
            }
        }

        debug_assert!(
            self.shards.contains_key(shard_id),
            "shard holder does not contain shard {shard_id} replica set"
        );

        // TODO(resharding): Assert that peer exists!?

        Ok(())
    }

    pub fn check_abort_resharding(&mut self, resharding_key: &ReshardKey) -> CollectionResult<()> {
        let state = self.resharding_state.read();

        // `abort_resharding` designed to be self-healing, so...
        //
        // - it's safe to run, if resharding is *not* in progress
        let Some(state) = state.deref() else {
            return Ok(());
        };

        // - it's safe to run, if *another* resharding is in progress
        if !state.matches(resharding_key) {
            return Ok(());
        }

        // - it's safe to run, if write hash ring was not committed yet
        if state.stage < ReshardStage::ReadHashRingCommitted {
            return Ok(());
        }

        // - but resharding can't be aborted, after read hash ring has been committed
        Err(CollectionError::bad_request(format!(
            "can't abort resharding {resharding_key}, \
             because read hash ring has been committed already, \
             resharding must be completed",
        )))
    }

    pub async fn abort_resharding(
        &mut self,
        resharding_key: ReshardKey,
        force: bool,
    ) -> CollectionResult<()> {
        let ReshardKey {
            direction,
            peer_id,
            shard_id,
            ref shard_key,
        } = resharding_key;

        let is_in_progress = match self.resharding_state.read().deref() {
            Some(state) if state.matches(&resharding_key) => {
                if !force && state.stage >= ReshardStage::ReadHashRingCommitted {
                    return Err(CollectionError::bad_request(format!(
                        "can't abort resharding {resharding_key}, \
                         because read hash ring has been committed already, \
                         resharding must be completed",
                    )));
                }

                true
            }

            Some(state) => {
                log::warn!(
                    "aborting resharding {resharding_key}, \
                     but another resharding is in progress:\n\
                     {state:#?}"
                );

                false
            }

            None => {
                log::warn!(
                    "aborting resharding {resharding_key}, \
                     but resharding is not in progress"
                );

                false
            }
        };

        // Cleanup existing shards if resharding down
        if is_in_progress && direction == ReshardingDirection::Down {
            for (&id, shard) in self.shards.iter() {
                // Skip shards that does not belong to resharding shard key
                if self.shard_id_to_key_mapping.get(&id) != shard_key.as_ref() {
                    continue;
                }

                // Skip target shard
                if id == shard_id {
                    continue;
                }

                // Revert replicas in `Resharding` state back into `Active` state
                for (peer, state) in shard.peers() {
                    if state == ReplicaState::Resharding {
                        shard.set_replica_state(&peer, ReplicaState::Active)?;
                    }
                }

                // We only cleanup local shards
                if !shard.is_local().await {
                    continue;
                }

                // Remove any points that might have been transferred from target shard
                let filter = self.hash_ring_filter(id).expect("hash ring filter");
                let filter = Filter::new_must(Condition::CustomIdChecker(Arc::new(filter)));
                shard.cleanup_local_shard(filter).await?;
            }
        }

        if let Some(ring) = self.rings.get_mut(shard_key) {
            log::debug!("reverting resharding hashring for shard {shard_id}");
            ring.abort_resharding(shard_id, direction);
        } else {
            log::warn!(
                "aborting resharding {resharding_key}, \
                 but {shard_key:?} hashring does not exist"
            );
        }

        // Remove new shard if resharding up
        if direction == ReshardingDirection::Up {
            if let Some(shard) = self.get_shard(&shard_id) {
                match shard.peer_state(&peer_id) {
                    Some(ReplicaState::Resharding) => {
                        log::debug!("removing peer {peer_id} from {shard_id} replica set");
                        shard.remove_peer(peer_id).await?;
                    }

                    Some(ReplicaState::Dead) if is_in_progress => {
                        log::debug!("removing dead peer {peer_id} from {shard_id} replica set");
                        shard.remove_peer(peer_id).await?;
                    }

                    Some(state) => {
                        return Err(CollectionError::bad_request(format!(
                            "peer {peer_id} is in {state:?} state"
                        )));
                    }

                    None => {
                        log::warn!(
                            "aborting resharding {resharding_key}, \
                             but peer {peer_id} does not exist in {shard_id} replica set"
                        );
                    }
                }

                if shard.peers().is_empty() {
                    log::debug!("removing {shard_id} replica set, because replica set is empty");

                    if let Some(shard_key) = shard_key {
                        self.key_mapping.write_optional(|key_mapping| {
                            if !key_mapping.contains_key(shard_key) {
                                return None;
                            }

                            let mut key_mapping = key_mapping.clone();
                            key_mapping.get_mut(shard_key).unwrap().remove(&shard_id);
                            Some(key_mapping)
                        })?;
                    }

                    self.drop_and_remove_shard(shard_id).await?;
                    self.shard_id_to_key_mapping.remove(&shard_id);
                }
            } else {
                log::warn!(
                    "aborting resharding {resharding_key}, \
                     but shard holder does not contain {shard_id} replica set",
                );
            }
        }

        if is_in_progress {
            self.resharding_state.write(|state| {
                debug_assert!(
                    state
                        .as_ref()
                        .map_or(false, |state| state.matches(&resharding_key)),
                    "resharding {resharding_key} is not in progress:\n{state:#?}"
                );

                state.take();
            })?;
        }

        Ok(())
    }

    pub async fn cleanup_local_shard(&self, shard_id: ShardId) -> CollectionResult<UpdateResult> {
        let shard = self.get_shard(&shard_id).ok_or_else(|| {
            CollectionError::not_found(format!("shard {shard_id} does not exist"))
        })?;

        if !shard.is_local().await {
            return Err(CollectionError::bad_shard_selection(format!(
                "shard {shard_id} is not a local shard"
            )))?;
        }

        let filter = self.hash_ring_filter(shard_id).expect("hash ring filter");
        let filter = Filter::new_must(Condition::CustomIdChecker(Arc::new(filter)));
        shard.cleanup_local_shard(filter).await
    }

    pub fn resharding_filter(&self) -> Option<hash_ring::HashRingFilter> {
        let shard_id = self.resharding_state.read().as_ref()?.shard_id;
        self.hash_ring_filter(shard_id)
    }

    pub fn hash_ring_filter(&self, shard_id: ShardId) -> Option<hash_ring::HashRingFilter> {
        if !self.contains_shard(&shard_id) {
            return None;
        }

        let shard_key = self.shard_id_to_key_mapping.get(&shard_id).cloned();
        let router = self.rings.get(&shard_key).expect("hashring exists");
        let ring = match router {
            HashRingRouter::Single(ring) => ring,
            HashRingRouter::Resharding { old, new } => {
                if new.len() > old.len() {
                    new
                } else {
                    old
                }
            }
        };

        Some(hash_ring::HashRingFilter::new(ring.clone(), shard_id))
    }
}

fn get_ring<'a>(
    rings: &'a mut HashMap<Option<ShardKey>, HashRingRouter>,
    shard_key: &'_ Option<ShardKey>,
) -> CollectionResult<&'a mut HashRingRouter> {
    rings.get_mut(shard_key).ok_or_else(|| {
        CollectionError::bad_request(format!(
            "{} hashring does not exist",
            shard_key_fmt(shard_key)
        ))
    })
}

fn assert_resharding_state_consistency(
    state: &Option<ReshardState>,
    ring: &HashRingRouter,
    shard_key: &Option<ShardKey>,
) {
    match state.as_ref().map(|state| state.stage) {
        Some(ReshardStage::MigratingPoints | ReshardStage::ReadHashRingCommitted) => {
            debug_assert!(
                ring.is_resharding(),
                "resharding is in progress, \
                 but {shard_key:?} hashring is not a resharding hashring:\n\
                 {state:#?}"
            );
        }

        Some(ReshardStage::WriteHashRingCommitted) => {
            debug_assert!(
                !ring.is_resharding(),
                "resharding is in progress, \
                 and write hashring has already been committed, \
                 but {shard_key:?} hashring is a resharding hashring:\n\
                 {state:#?}"
            );
        }

        None => {
            debug_assert!(
                !ring.is_resharding(),
                "resharding is not in progress, \
                 but {shard_key:?} hashring is a resharding hashring"
            );
        }
    }
}

fn check_stage(stage: ReshardStage) -> impl Fn(&ReshardState) -> CollectionResult<()> {
    move |state| {
        if state.stage == stage {
            Ok(())
        } else {
            Err(CollectionError::bad_request(format!(
                "expected resharding stage {stage:?}, but resharding is at stage {:?}",
                state.stage,
            )))
        }
    }
}

fn shard_key_fmt(key: &Option<ShardKey>) -> &dyn fmt::Display {
    match key {
        Some(key) => key,
        None => &"default",
    }
}
