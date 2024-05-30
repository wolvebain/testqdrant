use std::sync::Arc;

use futures::{future, TryFutureExt};
use itertools::{Either, Itertools};
use segment::types::Order;
use segment::utils::scored_point_ties::ScoredPointTies;

use super::Collection;
use crate::common::transpose_iterator::{transpose, transposed_iter};
use crate::operations::consistency_params::ReadConsistency;
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::types::CollectionResult;
use crate::operations::universal_query::shard_query::{
    Fusion, ScoringQuery, ShardQueryRequest, ShardQueryResponse,
};

struct IntermediateQueryInfo<'a> {
    scoring_query: Option<&'a ScoringQuery>,
    take: usize,
}

impl Collection {
    /// Returns a vector of shard responses for the given query.
    async fn query_shards_concurrently(
        &self,
        request: Arc<ShardQueryRequest>,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
    ) -> CollectionResult<Vec<ShardQueryResponse>> {
        // query all shards concurrently
        let shard_holder = self.shards_holder.read().await;
        let target_shards = shard_holder.select_shards(shard_selection)?;
        let all_searches = target_shards.iter().map(|(shard, shard_key)| {
            let shard_key = shard_key.cloned();
            shard
                .query(
                    Arc::clone(&request),
                    read_consistency,
                    shard_selection.is_shard_id(),
                )
                .and_then(move |mut records| async move {
                    if shard_key.is_none() {
                        return Ok(records);
                    }
                    for batch in &mut records {
                        for point in batch {
                            point.shard_key.clone_from(&shard_key);
                        }
                    }
                    Ok(records)
                })
        });
        future::try_join_all(all_searches).await
    }

    /// To be called on the remote instance. Only used for the internal service.
    ///
    /// If the root query is a Fusion, the returned results correspond to each the prefetches.
    /// Otherwise, it will be a list with a single list of scored points.
    pub async fn query_internal(
        &self,
        request: ShardQueryRequest,
        read_consistency: Option<ReadConsistency>,
        shard_selection: &ShardSelectorInternal,
    ) -> CollectionResult<ShardQueryResponse> {
        let request = Arc::new(request);

        // Results from all shards
        // Shape: [num_shards, num_internal_queries, num_scored_points]
        let all_shards_results = self
            .query_shards_concurrently(Arc::clone(&request), read_consistency, shard_selection)
            .await?;

        let query_infos = intermediate_query_infos(&request);
        let results_len = query_infos.len();
        let mut results = ShardQueryResponse::with_capacity(results_len);
        debug_assert!(all_shards_results
            .iter()
            .all(|shard_results| shard_results.len() == results_len));

        let collection_params = self.collection_config.read().await.params.clone();

        // Time to merge the results in each shard for each intermediate query.
        // In order to do this, we need to iterate over columns of the all_shards_results matrix.
        //
        // [ [shard1_result1, shard1_result2],
        //          ↓               ↓
        //   [shard2_result1, shard2_result2] ]
        //
        // = [merged_result1, merged_result2]

        // Shape: [num_internal_queries, num_shards, num_scored_points]
        let all_shards_result_by_transposed = transposed_iter(all_shards_results);

        for (query_info, shards_results) in
            query_infos.into_iter().zip(all_shards_result_by_transposed)
        {
            // `shards_results` shape: [num_shards, num_scored_points]
            let order = ScoringQuery::order(query_info.scoring_query, &collection_params)?;

            // Equivalent to:
            //
            // shards_results
            //     .into_iter()
            //     .kmerge_by(match order {
            //         Order::LargeBetter => |a, b| ScoredPointTies(a) > ScoredPointTies(b),
            //         Order::SmallBetter => |a, b| ScoredPointTies(a) < ScoredPointTies(b),
            //     })
            //
            // if the `kmerge_by` function were able to work with reference predicates.
            // Either::Left and Either::Right are used to allow type inference to work.
            //
            let intermediate_result = match order {
                Order::LargeBetter => Either::Left(
                    shards_results
                        .into_iter()
                        .kmerge_by(|a, b| ScoredPointTies(a) > ScoredPointTies(b)),
                ),
                Order::SmallBetter => Either::Right(
                    shards_results
                        .into_iter()
                        .kmerge_by(|a, b| ScoredPointTies(a) < ScoredPointTies(b)),
                ),
            }
            .dedup()
            .take(query_info.take)
            .collect();

            results.push(intermediate_result);
        }

        Ok(results)
    }
}

/// Returns a list of the query that corresponds to each of the results in each shard.
///
/// Example: `[info1, info2, info3]` corresponds to `[result1, result2, result3]` of each shard
fn intermediate_query_infos(request: &ShardQueryRequest) -> Vec<IntermediateQueryInfo<'_>> {
    if let Some(ScoringQuery::Fusion(Fusion::Rrf)) = request.query {
        // In case of RRF, expect the propagated intermediate results
        request
            .prefetches
            .iter()
            .map(|prefetch| IntermediateQueryInfo {
                scoring_query: prefetch.query.as_ref(),
                take: prefetch.limit,
            })
            .collect_vec()
    } else {
        // Otherwise, we expect the root result
        vec![IntermediateQueryInfo {
            scoring_query: request.query.as_ref(),
            take: request.offset + request.limit,
        }]
    }
}
