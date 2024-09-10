use std::sync::Arc;

use actix_web::{post, web, Responder};
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::types::{
    CountRequestInternal, PointRequestInternal, ScrollRequestInternal,
};
use collection::shards::shard::ShardId;
use segment::types::{Condition, Filter};
use storage::content_manager::errors::{StorageError, StorageResult};
use storage::dispatcher::Dispatcher;
use storage::rbac::{Access, AccessRequirements};

use crate::actix::api::read_params::ReadParams;
use crate::actix::auth::ActixAccess;
use crate::actix::helpers;
use crate::common::points;

// Configure services
pub fn config_local_shard_api(cfg: &mut web::ServiceConfig) {
    cfg.service(get_points)
        .service(scroll_points)
        .service(count_points)
        .service(cleanup_shard);
}

#[post("/collections/{collection}/shards/{shard}/points")]
async fn get_points(
    dispatcher: web::Data<Dispatcher>,
    ActixAccess(access): ActixAccess,
    path: web::Path<CollectionShard>,
    request: web::Json<PointRequestInternal>,
    params: web::Query<ReadParams>,
) -> impl Responder {
    helpers::time(async move {
        let records = points::do_get_points(
            dispatcher.toc(&access),
            &path.collection,
            request.into_inner(),
            params.consistency,
            params.timeout(),
            ShardSelectorInternal::ShardId(path.shard),
            access,
        )
        .await?;

        let records: Vec<_> = records.into_iter().map(api::rest::Record::from).collect();
        Ok(records)
    })
    .await
}

#[post("/collections/{collection}/shards/{shard}/points/scroll")]
async fn scroll_points(
    dispatcher: web::Data<Dispatcher>,
    ActixAccess(access): ActixAccess,
    path: web::Path<CollectionShard>,
    request: web::Json<WithFilter<ScrollRequestInternal>>,
    params: web::Query<ReadParams>,
) -> impl Responder {
    helpers::time(async move {
        let WithFilter {
            mut request,
            hash_ring_filter,
        } = request.into_inner();

        let hash_ring_filter = match hash_ring_filter {
            Some(filter) => get_hash_ring_filter(
                &dispatcher,
                &access,
                &path.collection,
                AccessRequirements::new(),
                filter.expected_shard_id,
            )
            .await?
            .into(),

            None => None,
        };

        request.filter = merge_with_optional_filter(request.filter.take(), hash_ring_filter);

        dispatcher
            .toc(&access)
            .scroll(
                &path.collection,
                request,
                params.consistency,
                params.timeout(),
                ShardSelectorInternal::ShardId(path.shard),
                access,
            )
            .await
    })
    .await
}

#[post("/collections/{collection}/shards/{shard}/points/count")]
async fn count_points(
    dispatcher: web::Data<Dispatcher>,
    ActixAccess(access): ActixAccess,
    path: web::Path<CollectionShard>,
    request: web::Json<WithFilter<CountRequestInternal>>,
    params: web::Query<ReadParams>,
) -> impl Responder {
    helpers::time(async move {
        let WithFilter {
            mut request,
            hash_ring_filter,
        } = request.into_inner();

        let hash_ring_filter = match hash_ring_filter {
            Some(filter) => get_hash_ring_filter(
                &dispatcher,
                &access,
                &path.collection,
                AccessRequirements::new(),
                filter.expected_shard_id,
            )
            .await?
            .into(),

            None => None,
        };

        request.filter = merge_with_optional_filter(request.filter.take(), hash_ring_filter);

        points::do_count_points(
            dispatcher.toc(&access),
            &path.collection,
            request,
            params.consistency,
            params.timeout(),
            ShardSelectorInternal::ShardId(path.shard),
            access,
        )
        .await
    })
    .await
}

#[post("/collections/{collection}/shards/{shard}/cleanup")]
async fn cleanup_shard(
    dispatcher: web::Data<Dispatcher>,
    ActixAccess(access): ActixAccess,
    path: web::Path<CollectionShard>,
) -> impl Responder {
    helpers::time(async move {
        let path = path.into_inner();
        dispatcher
            .toc(&access)
            .cleanup_local_shard(&path.collection, path.shard, access)
            .await
    })
    .await
}

#[derive(serde::Deserialize, validator::Validate)]
struct CollectionShard {
    #[validate(length(min = 1, max = 255))]
    collection: String,
    shard: ShardId,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct WithFilter<T> {
    #[serde(flatten)]
    request: T,
    #[serde(default)]
    hash_ring_filter: Option<SerdeHelper>,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct SerdeHelper {
    expected_shard_id: ShardId,
}

async fn get_hash_ring_filter(
    dispatcher: &Dispatcher,
    access: &Access,
    collection: &str,
    reqs: AccessRequirements,
    expected_shard_id: ShardId,
) -> StorageResult<Filter> {
    let pass = access.check_collection_access(collection, reqs)?;

    let shard_holder = dispatcher
        .toc(access)
        .get_collection(&pass)
        .await?
        .shards_holder();

    let hash_ring_filter = shard_holder
        .read()
        .await
        .hash_ring_filter(expected_shard_id)
        .ok_or_else(|| {
            StorageError::bad_request(format!(
                "shard {expected_shard_id} does not exist in collection {collection}"
            ))
        })?;

    let condition = Condition::CustomIdChecker(Arc::new(hash_ring_filter));
    let filter = Filter::new_must(condition);

    Ok(filter)
}

fn merge_with_optional_filter(filter: Option<Filter>, hash_ring: Option<Filter>) -> Option<Filter> {
    match (filter, hash_ring) {
        (Some(filter), Some(hash_ring)) => hash_ring.merge_owned(filter).into(),
        (Some(filter), None) => filter.into(),
        (None, Some(hash_ring)) => hash_ring.into(),
        _ => None,
    }
}
