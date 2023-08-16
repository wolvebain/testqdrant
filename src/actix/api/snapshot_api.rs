use std::io;

use actix_files::NamedFile;
use actix_multipart::form::tempfile::TempFile;
use actix_multipart::form::MultipartForm;
use actix_web::rt::time::Instant;
use actix_web::{delete, get, post, put, web, Responder, Result};
use actix_web_validator::{Json, Path, Query};
use collection::collection::Collection;
use collection::operations::snapshot_ops::{SnapshotPriority, SnapshotRecover};
use collection::shards::replica_set::ReplicaState;
use collection::shards::shard::ShardId;
use reqwest::Url;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use storage::content_manager::errors::StorageError;
use storage::content_manager::snapshots::recover::{activate_shard, do_recover_from_snapshot};
use storage::content_manager::snapshots::{
    self, do_create_full_snapshot, do_delete_collection_snapshot, do_delete_full_snapshot,
    do_list_full_snapshots, get_full_snapshot_path,
};
use storage::content_manager::toc::{TableOfContent, SNAPSHOTS_TEMP_DIR};
use storage::dispatcher::Dispatcher;
use tokio::sync::RwLockReadGuard;
use uuid::Uuid;
use validator::Validate;

use super::CollectionPath;
use crate::actix::helpers;
use crate::actix::helpers::{
    accepted_response, collection_into_actix_error, process_response, storage_into_actix_error,
};
use crate::common::collections::*;

#[derive(Deserialize, Validate)]
struct SnapshotPath {
    #[serde(rename = "snapshot_name")]
    #[validate(length(min = 1))]
    name: String,
}

#[derive(Deserialize, Serialize, JsonSchema, Validate)]
pub struct SnapshotUploadingParam {
    pub wait: Option<bool>,
    pub priority: Option<SnapshotPriority>,
}

#[derive(Deserialize, Serialize, JsonSchema, Validate)]
pub struct SnapshottingParam {
    pub wait: Option<bool>,
}

#[derive(MultipartForm)]
pub struct SnapshottingForm {
    snapshot: TempFile,
}

// Actix specific code
pub async fn do_get_full_snapshot(toc: &TableOfContent, snapshot_name: &str) -> Result<NamedFile> {
    let file_name = get_full_snapshot_path(toc, snapshot_name)
        .await
        .map_err(storage_into_actix_error)?;

    Ok(NamedFile::open(file_name)?)
}

pub async fn do_save_uploaded_snapshot(
    toc: &TableOfContent,
    collection_name: &str,
    snapshot: TempFile,
) -> std::result::Result<Url, StorageError> {
    let filename = snapshot
        .file_name
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let collection_snapshot_path = toc.snapshots_path_for_collection(collection_name);
    if !collection_snapshot_path.exists() {
        log::debug!(
            "Creating missing collection snapshots directory for {}",
            collection_name
        );
        toc.create_snapshots_path(collection_name).await?;
    }

    let path = collection_snapshot_path.join(filename);

    snapshot.file.persist(&path)?;

    let absolute_path = path.canonicalize()?;

    let snapshot_location = Url::from_file_path(&absolute_path).map_err(|_| {
        StorageError::service_error(format!(
            "Failed to convert path to URL: {}",
            absolute_path.display()
        ))
    })?;

    Ok(snapshot_location)
}

// Actix specific code
pub async fn do_get_snapshot(
    toc: &TableOfContent,
    collection_name: &str,
    snapshot_name: &str,
) -> Result<NamedFile> {
    let collection = toc
        .get_collection(collection_name)
        .await
        .map_err(storage_into_actix_error)?;

    let file_name = collection
        .get_snapshot_path(snapshot_name)
        .await
        .map_err(collection_into_actix_error)?;

    Ok(NamedFile::open(file_name)?)
}

#[get("/collections/{name}/snapshots")]
async fn list_snapshots(toc: web::Data<TableOfContent>, path: web::Path<String>) -> impl Responder {
    let collection_name = path.into_inner();
    let timing = Instant::now();

    let response = do_list_snapshots(&toc, &collection_name).await;
    process_response(response, timing)
}

#[post("/collections/{name}/snapshots")]
async fn create_snapshot(
    dispatcher: web::Data<Dispatcher>,
    path: web::Path<String>,
    params: Query<SnapshottingParam>,
) -> impl Responder {
    let collection_name = path.into_inner();
    let wait = params.wait.unwrap_or(true);

    let timing = Instant::now();
    let response = do_create_snapshot(dispatcher.get_ref(), &collection_name, wait).await;
    match response {
        Err(_) => process_response(response, timing),
        Ok(_) if wait => process_response(response, timing),
        Ok(_) => accepted_response(timing),
    }
}

#[post("/collections/{name}/snapshots/upload")]
async fn upload_snapshot(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    MultipartForm(form): MultipartForm<SnapshottingForm>,
    params: Query<SnapshotUploadingParam>,
) -> impl Responder {
    let timing = Instant::now();
    let snapshot = form.snapshot;
    let wait = params.wait.unwrap_or(true);

    let snapshot_location =
        match do_save_uploaded_snapshot(dispatcher.get_ref(), &collection.name, snapshot).await {
            Ok(location) => location,
            Err(err) => return process_response::<()>(Err(err), timing),
        };

    let snapshot_recover = SnapshotRecover {
        location: snapshot_location,
        priority: params.priority,
    };

    let response = do_recover_from_snapshot(
        dispatcher.get_ref(),
        &collection.name,
        snapshot_recover,
        wait,
    )
    .await;
    match response {
        Err(_) => process_response(response, timing),
        Ok(_) if wait => process_response(response, timing),
        Ok(_) => accepted_response(timing),
    }
}

#[put("/collections/{name}/snapshots/recover")]
async fn recover_from_snapshot(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<SnapshotRecover>,
    params: Query<SnapshottingParam>,
) -> impl Responder {
    let timing = Instant::now();
    let snapshot_recover = request.into_inner();
    let wait = params.wait.unwrap_or(true);

    let response = do_recover_from_snapshot(
        dispatcher.get_ref(),
        &collection.name,
        snapshot_recover,
        wait,
    )
    .await;
    match response {
        Err(_) => process_response(response, timing),
        Ok(_) if wait => process_response(response, timing),
        Ok(_) => accepted_response(timing),
    }
}

#[get("/collections/{name}/snapshots/{snapshot_name}")]
async fn get_snapshot(
    toc: web::Data<TableOfContent>,
    path: web::Path<(String, String)>,
) -> impl Responder {
    let (collection_name, snapshot_name) = path.into_inner();
    do_get_snapshot(&toc, &collection_name, &snapshot_name).await
}
#[get("/snapshots")]
async fn list_full_snapshots(toc: web::Data<TableOfContent>) -> impl Responder {
    let timing = Instant::now();
    let response = do_list_full_snapshots(toc.get_ref()).await;
    process_response(response, timing)
}

#[post("/snapshots")]
async fn create_full_snapshot(
    dispatcher: web::Data<Dispatcher>,
    params: Query<SnapshottingParam>,
) -> impl Responder {
    let timing = Instant::now();
    let wait = params.wait.unwrap_or(true);
    let response = do_create_full_snapshot(dispatcher.get_ref(), wait).await;
    match response {
        Err(_) => process_response(response, timing),
        Ok(_) if wait => process_response(response, timing),
        Ok(_) => accepted_response(timing),
    }
}

#[get("/snapshots/{snapshot_name}")]
async fn get_full_snapshot(
    toc: web::Data<TableOfContent>,
    path: web::Path<String>,
) -> impl Responder {
    let snapshot_name = path.into_inner();
    do_get_full_snapshot(&toc, &snapshot_name).await
}

#[delete("/snapshots/{snapshot_name}")]
async fn delete_full_snapshot(
    dispatcher: web::Data<Dispatcher>,
    path: web::Path<String>,
    params: Query<SnapshottingParam>,
) -> impl Responder {
    let snapshot_name = path.into_inner();
    let timing = Instant::now();
    let wait = params.wait.unwrap_or(true);
    let response = do_delete_full_snapshot(dispatcher.get_ref(), &snapshot_name, wait).await;
    match response {
        Err(_) => process_response(response, timing),
        Ok(_) if wait => process_response(response, timing),
        Ok(_) => accepted_response(timing),
    }
}

#[delete("/collections/{name}/snapshots/{snapshot_name}")]
async fn delete_collection_snapshot(
    dispatcher: web::Data<Dispatcher>,
    path: web::Path<(String, String)>,
    params: Query<SnapshottingParam>,
) -> impl Responder {
    let (collection_name, snapshot_name) = path.into_inner();
    let timing = Instant::now();
    let wait = params.wait.unwrap_or(true);
    let response =
        do_delete_collection_snapshot(dispatcher.get_ref(), &collection_name, &snapshot_name, wait)
            .await;
    match response {
        Err(_) => process_response(response, timing),
        Ok(_) if wait => process_response(response, timing),
        Ok(_) => accepted_response(timing),
    }
}

#[get("/collections/{collection}/shards/{shard}/snapshots")]
async fn list_shard_snapshots(
    toc: web::Data<TableOfContent>,
    path: web::Path<(String, ShardId)>,
) -> impl Responder {
    let future = async move {
        let (collection, shard) = path.into_inner();
        let collection = toc.get_collection(&collection).await?;
        let snapshots = collection.list_shard_snapshots(shard).await?;
        Ok(snapshots)
    };

    helpers::time(future).await
}

#[post("/collections/{collection}/shards/{shard}/snapshots")]
async fn create_shard_snapshot(
    toc: web::Data<TableOfContent>,
    path: web::Path<(String, ShardId)>,
    query: web::Query<SnapshottingParam>,
) -> impl Responder {
    let future = async move {
        let (collection, shard) = path.into_inner();
        let collection = toc.get_collection(&collection).await?;
        let snapshot = collection
            .create_shard_snapshot(shard, &toc.temp_snapshots_path().join(SNAPSHOTS_TEMP_DIR))
            .await?;

        Ok(snapshot)
    };

    helpers::time_or_accept(future, query.wait.unwrap_or(true)).await
}

// TODO: `POST` or `PUT` (same as `recover_from_snapshot`)!?
#[post("/collections/{collection}/shards/{shard}/snapshots/recover")]
async fn recover_shard_snapshot(
    toc: web::Data<TableOfContent>,
    path: web::Path<(String, ShardId)>,
    query: web::Query<SnapshottingParam>,
    web::Json(request): web::Json<SnapshotRecover>,
) -> impl Responder {
    let future = async move {
        let (collection, shard) = path.into_inner();
        let collection = toc.get_collection(&collection).await?;
        let snapshots_dir = collection.get_snapshots_path_for_shard(shard).await?;

        // TODO: Handle cleanup on download failure (e.g., using `tempfile`)!

        let snapshot_path =
            snapshots::download::download_snapshot(request.location, &snapshots_dir).await?;

        recover_shard_snapshot_impl(
            &toc,
            &collection,
            shard,
            &snapshot_path,
            request.priority.unwrap_or_default(),
        )
        .await?;

        Ok(())
    };

    helpers::time_or_accept(future, query.wait.unwrap_or(true)).await
}

// TODO: `PUT` or `POST` (same as `upload_snapshot`)!?
#[put("/collections/{collection}/shards/{shard}/snapshots/upload")]
async fn upload_shard_snapshot(
    toc: web::Data<TableOfContent>,
    path: web::Path<(String, ShardId, String)>,
    query: web::Query<SnapshotUploadingParam>,
    MultipartForm(form): MultipartForm<SnapshottingForm>,
) -> impl Responder {
    let SnapshotUploadingParam { wait, priority } = query.into_inner();

    let future = async move {
        let (collection, shard, snapshot) = path.into_inner();
        let collection = toc.get_collection(&collection).await?;
        let snapshots_dir = collection.get_snapshots_path_for_shard(shard).await?;

        if !snapshots_dir.exists() {
            std::fs::create_dir_all(&snapshots_dir)?;
        }

        let snapshot_path = snapshots_dir.join(snapshot);

        form.snapshot
            .file
            .persist(&snapshot_path)
            .map_err(io::Error::from)?;

        recover_shard_snapshot_impl(
            &toc,
            &collection,
            shard,
            &snapshot_path,
            priority.unwrap_or_default(),
        )
        .await?;

        Ok(())
    };

    helpers::time_or_accept(future, wait.unwrap_or(true)).await
}

#[get("/collections/{collection}/shards/{shard}/snapshots/{snapshot}")]
async fn download_shard_snapshot(
    toc: web::Data<TableOfContent>,
    path: web::Path<(String, ShardId, String)>,
) -> Result<impl Responder, helpers::HttpError> {
    let (collection, shard, snapshot) = path.into_inner();
    let collection = toc.get_collection(&collection).await?;
    let snapshot_path = collection.get_shard_snapshot_path(shard, &snapshot).await?;

    Ok(NamedFile::open(snapshot_path))
}

#[delete("/collections/{collection}/shards/{shard}/snapshots/{snapshot}")]
async fn delete_shard_snapshot(
    toc: web::Data<TableOfContent>,
    path: web::Path<(String, ShardId, String)>,
    query: web::Query<SnapshottingParam>,
) -> impl Responder {
    let future = async move {
        let (collection, shard, snapshot) = path.into_inner();
        let collection = toc.get_collection(&collection).await?;
        let snapshot_path = collection.get_shard_snapshot_path(shard, &snapshot).await?;

        // TODO: Do we need these explicit checks/errors?
        // TODO: `std::fs::remove_file` would fail with roughly the same errors.

        if !snapshot_path.exists() {
            let description = format!("Snapshot {snapshot} not found");
            return Err(StorageError::NotFound { description }.into());
        } else if !snapshot_path.is_file() {
            let description = format!("{} is not a file", snapshot_path.display());
            return Err(StorageError::service_error(description).into());
        }

        std::fs::remove_file(&snapshot_path)?;

        Ok(())
    };

    helpers::time_or_accept(future, query.wait.unwrap_or(true)).await
}

async fn recover_shard_snapshot_impl(
    toc: &TableOfContent,
    collection: &RwLockReadGuard<'_, Collection>,
    shard: ShardId,
    snapshot_path: &std::path::Path,
    priority: SnapshotPriority,
) -> Result<(), StorageError> {
    // TODO: Check snapshot compatibility?
    // TODO: Switch replica into `Partial` state?

    collection
        .restore_shard_snapshot(
            shard,
            snapshot_path,
            toc.this_peer_id,
            toc.is_distributed(),
            &toc.temp_snapshots_path().join(SNAPSHOTS_TEMP_DIR),
        )
        .await?;

    let state = collection.state().await;
    let shard_info = state.shards.get(&shard).unwrap(); // TODO: Handle `unwrap`?..

    let other_active_replicas: Vec<_> = shard_info
        .replicas
        .iter()
        .map(|(&peer, &state)| (peer, state))
        .filter(|&(peer, state)| peer != toc.this_peer_id && state == ReplicaState::Active)
        .collect();

    if other_active_replicas.is_empty() {
        activate_shard(toc, collection, toc.this_peer_id, &shard).await?;
    } else {
        match priority {
            SnapshotPriority::Snapshot => {
                activate_shard(toc, collection, toc.this_peer_id, &shard).await?;

                let replicas_to_keep = state.config.params.replication_factor.get() - 1;

                let replicas_to_remove = other_active_replicas
                    .len()
                    .saturating_sub(replicas_to_keep as usize);

                let mut other_active_replicas = other_active_replicas.iter().copied();

                for (peer, _) in other_active_replicas.by_ref().take(replicas_to_remove) {
                    // Is it right to remove *active* replicas!?
                    // Why not remove some `Dead` or `Partial` replica instead?
                    // What if there's no active replicas? Are we going *over* the replication factor then?
                    toc.request_remove_replica(collection.name(), shard, peer)?;
                }

                for (peer, _) in other_active_replicas {
                    toc.send_set_replica_state_proposal(
                        collection.name(),
                        peer,
                        shard,
                        ReplicaState::Dead,
                        None,
                    )?;
                }
            }

            SnapshotPriority::Replica => {
                toc.send_set_replica_state_proposal(
                    collection.name(),
                    toc.this_peer_id,
                    shard,
                    ReplicaState::Dead,
                    None,
                )?;
            }
        }
    }

    Ok(())
}

// Configure services
pub fn config_snapshots_api(cfg: &mut web::ServiceConfig) {
    cfg.service(list_snapshots)
        .service(create_snapshot)
        .service(upload_snapshot)
        .service(recover_from_snapshot)
        .service(get_snapshot)
        .service(list_full_snapshots)
        .service(create_full_snapshot)
        .service(get_full_snapshot)
        .service(delete_full_snapshot)
        .service(delete_collection_snapshot)
        .service(list_shard_snapshots)
        .service(create_shard_snapshot)
        .service(recover_shard_snapshot)
        .service(upload_shard_snapshot)
        .service(download_shard_snapshot)
        .service(delete_shard_snapshot);
}
