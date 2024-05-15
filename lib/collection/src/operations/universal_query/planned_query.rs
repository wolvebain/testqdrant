//! Types used within `LocalShard` to represent a planned `ShardQueryRequest`

use std::sync::Arc;

use segment::types::{WithPayloadInterface, WithVector};

use super::shard_query::{ScoringQuery, ShardPrefetch, ShardQueryRequest};
use crate::operations::types::{
    CollectionError, CollectionResult, CoreSearchRequest, CoreSearchRequestBatch,
};

pub struct PlannedQuery {
    pub merge_plan: PrefetchPlan,
    pub batch: Arc<CoreSearchRequestBatch>,
    pub offset: usize,
    pub with_vector: WithVector,
    pub with_payload: WithPayloadInterface,
}

pub struct PrefetchMerge {
    /// Alter the scores before selecting the best limit
    pub rescore: Option<ScoringQuery>,

    /// Keep this much points from the top
    pub limit: usize,
}

pub enum PrefetchSource {
    /// A reference offset into the main search batch
    BatchIdx(usize),

    /// A nested prefetch
    Prefetch(PrefetchPlan),
}

pub struct PrefetchPlan {
    /// Gather all these sources
    pub sources: Vec<PrefetchSource>,

    /// How to merge the sources
    pub merge: PrefetchMerge,
}

// TODO(universal-query): Maybe just return a CoreSearchRequest if there is no prefetch?
impl TryFrom<ShardQueryRequest> for PlannedQuery {
    type Error = CollectionError;

    fn try_from(request: ShardQueryRequest) -> CollectionResult<Self> {
        let ShardQueryRequest {
            query,
            filter,
            score_threshold,
            limit,
            offset: req_offset,
            with_vector,
            with_payload,
            prefetches: prefetch,
            params,
        } = request;

        let mut core_searches = Vec::new();
        let sources;
        let rescore;
        let offset;

        if !prefetch.is_empty() {
            sources = recurse_prefetches(&mut core_searches, prefetch);
            rescore = Some(query);
            offset = req_offset;
        } else {
            #[allow(clippy::infallible_destructuring_match)]
            // TODO(universal-query): remove once there are more variants
            let query = match query {
                ScoringQuery::Vector(query) => query,
                // TODO(universal-query): return error for fusion queries without prefetch
            };
            let core_search = CoreSearchRequest {
                query,
                filter,
                score_threshold,
                with_vector: None,
                with_payload: None,
                offset: req_offset,
                params,
                limit,
            };
            core_searches.push(core_search);

            sources = vec![PrefetchSource::BatchIdx(0)];
            rescore = None;
            offset = 0;
        }

        Ok(Self {
            merge_plan: PrefetchPlan {
                sources,
                merge: PrefetchMerge { rescore, limit },
            },
            batch: Arc::new(CoreSearchRequestBatch {
                searches: core_searches,
            }),
            offset,
            with_vector,
            with_payload,
        })
    }
}

fn recurse_prefetches(
    core_searches: &mut Vec<CoreSearchRequest>,
    prefetches: Vec<ShardPrefetch>,
) -> Vec<PrefetchSource> {
    let mut sources = Vec::with_capacity(prefetches.len());

    for prefetch in prefetches {
        let ShardPrefetch {
            prefetches,
            query,
            limit,
            params,
            filter,
            score_threshold,
        } = prefetch;

        let source = if prefetches.is_empty() {
            match query {
                ScoringQuery::Vector(query_enum) => {
                    let core_search = CoreSearchRequest {
                        query: query_enum,
                        filter,
                        params,
                        limit,
                        offset: 0,
                        with_payload: None,
                        with_vector: None,
                        score_threshold,
                    };

                    let idx = core_searches.len();
                    core_searches.push(core_search);

                    PrefetchSource::BatchIdx(idx)
                }
            }
        } else {
            let sources = recurse_prefetches(core_searches, prefetches);

            let prefetch_plan = PrefetchPlan {
                sources,
                merge: PrefetchMerge {
                    rescore: Some(query),
                    limit,
                },
            };
            PrefetchSource::Prefetch(prefetch_plan)
        };
        sources.push(source);
    }

    sources
}
