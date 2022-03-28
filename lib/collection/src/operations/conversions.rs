use crate::operations::config_diff::{HnswConfigDiff, WalConfigDiff};
use crate::operations::point_ops::PointsSelector::PointIdsSelector;
use crate::operations::point_ops::{FilterSelector, PointIdsList, PointStruct, PointsSelector};
use crate::operations::types::{CollectionStatus, OptimizersStatus};
use crate::{CollectionInfo, OptimizersConfigDiff, Record, UpdateResult};
use api::grpc::conversions::{payload_to_proto, proto_to_payloads};
use tonic::Status;

impl From<api::grpc::qdrant::HnswConfigDiff> for HnswConfigDiff {
    fn from(value: api::grpc::qdrant::HnswConfigDiff) -> Self {
        Self {
            m: value.m.map(|v| v as usize),
            ef_construct: value.ef_construct.map(|v| v as usize),
            full_scan_threshold: value.full_scan_threshold.map(|v| v as usize),
        }
    }
}

impl From<api::grpc::qdrant::WalConfigDiff> for WalConfigDiff {
    fn from(value: api::grpc::qdrant::WalConfigDiff) -> Self {
        Self {
            wal_capacity_mb: value.wal_capacity_mb.map(|v| v as usize),
            wal_segments_ahead: value.wal_segments_ahead.map(|v| v as usize),
        }
    }
}

impl From<api::grpc::qdrant::OptimizersConfigDiff> for OptimizersConfigDiff {
    fn from(value: api::grpc::qdrant::OptimizersConfigDiff) -> Self {
        Self {
            deleted_threshold: value.deleted_threshold,
            vacuum_min_vector_number: value.vacuum_min_vector_number.map(|v| v as usize),
            default_segment_number: value.default_segment_number.map(|v| v as usize),
            max_segment_size: value.max_segment_size.map(|v| v as usize),
            memmap_threshold: value.memmap_threshold.map(|v| v as usize),
            indexing_threshold: value.indexing_threshold.map(|v| v as usize),
            payload_indexing_threshold: value.payload_indexing_threshold.map(|v| v as usize),
            flush_interval_sec: value.flush_interval_sec,
            max_optimization_threads: value.max_optimization_threads.map(|v| v as usize),
        }
    }
}

impl From<CollectionInfo> for api::grpc::qdrant::CollectionInfo {
    fn from(value: CollectionInfo) -> Self {
        let CollectionInfo {
            status,
            optimizer_status,
            vectors_count,
            segments_count,
            disk_data_size,
            ram_data_size,
            config,
            payload_schema,
        } = value;

        api::grpc::qdrant::CollectionInfo {
            status: match status {
                CollectionStatus::Green => api::grpc::qdrant::CollectionStatus::Green,
                CollectionStatus::Yellow => api::grpc::qdrant::CollectionStatus::Yellow,
                CollectionStatus::Red => api::grpc::qdrant::CollectionStatus::Red,
            }
            .into(),
            optimizer_status: Some(match optimizer_status {
                OptimizersStatus::Ok => api::grpc::qdrant::OptimizerStatus {
                    ok: true,
                    error: "".to_string(),
                },
                OptimizersStatus::Error(error) => {
                    api::grpc::qdrant::OptimizerStatus { ok: false, error }
                }
            }),
            vectors_count: vectors_count as u64,
            segments_count: segments_count as u64,
            disk_data_size: disk_data_size as u64,
            ram_data_size: ram_data_size as u64,
            config: Some(api::grpc::qdrant::CollectionConfig {
                params: Some(api::grpc::qdrant::CollectionParams {
                    vector_size: config.params.vector_size as u64,
                    distance: match config.params.distance {
                        segment::types::Distance::Cosine => api::grpc::qdrant::Distance::Cosine,
                        segment::types::Distance::Euclid => api::grpc::qdrant::Distance::Euclid,
                        segment::types::Distance::Dot => api::grpc::qdrant::Distance::Dot,
                    }
                    .into(),
                    shard_number: config.params.shard_number.get(),
                }),
                hnsw_config: Some(api::grpc::qdrant::HnswConfigDiff {
                    m: Some(config.hnsw_config.m as u64),
                    ef_construct: Some(config.hnsw_config.ef_construct as u64),
                    full_scan_threshold: Some(config.hnsw_config.full_scan_threshold as u64),
                }),
                optimizer_config: Some(api::grpc::qdrant::OptimizersConfigDiff {
                    deleted_threshold: Some(config.optimizer_config.deleted_threshold),
                    vacuum_min_vector_number: Some(
                        config.optimizer_config.vacuum_min_vector_number as u64,
                    ),
                    default_segment_number: Some(
                        config.optimizer_config.default_segment_number as u64,
                    ),
                    max_segment_size: Some(config.optimizer_config.max_segment_size as u64),
                    memmap_threshold: Some(config.optimizer_config.memmap_threshold as u64),
                    indexing_threshold: Some(config.optimizer_config.indexing_threshold as u64),
                    payload_indexing_threshold: Some(
                        config.optimizer_config.payload_indexing_threshold as u64,
                    ),
                    flush_interval_sec: Some(config.optimizer_config.flush_interval_sec as u64),
                    max_optimization_threads: Some(
                        config.optimizer_config.max_optimization_threads as u64,
                    ),
                }),
                wal_config: Some(api::grpc::qdrant::WalConfigDiff {
                    wal_capacity_mb: Some(config.wal_config.wal_capacity_mb as u64),
                    wal_segments_ahead: Some(config.wal_config.wal_segments_ahead as u64),
                }),
            }),
            payload_schema: payload_schema
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
        }
    }
}

impl From<Record> for api::grpc::qdrant::RetrievedPoint {
    fn from(record: Record) -> Self {
        Self {
            id: Some(record.id.into()),
            payload: record.payload.map(payload_to_proto).unwrap_or_default(),
            vector: record.vector.unwrap_or_default(),
        }
    }
}

impl TryFrom<api::grpc::qdrant::PointStruct> for PointStruct {
    type Error = Status;

    fn try_from(value: api::grpc::qdrant::PointStruct) -> Result<Self, Self::Error> {
        let api::grpc::qdrant::PointStruct {
            id,
            vector,
            payload,
        } = value;

        let converted_payload = proto_to_payloads(payload)?;

        Ok(Self {
            id: id
                .ok_or_else(|| Status::invalid_argument("Empty ID is not allowed"))?
                .try_into()?,
            vector,
            payload: Some(converted_payload),
        })
    }
}

impl TryFrom<api::grpc::qdrant::PointsSelector> for PointsSelector {
    type Error = Status;

    fn try_from(value: api::grpc::qdrant::PointsSelector) -> Result<Self, Self::Error> {
        match value.points_selector_one_of {
            Some(api::grpc::qdrant::points_selector::PointsSelectorOneOf::Points(points)) => {
                Ok(PointIdsSelector(PointIdsList {
                    points: points
                        .ids
                        .into_iter()
                        .map(|p| p.try_into())
                        .collect::<Result<Vec<_>, _>>()?,
                }))
            }
            Some(api::grpc::qdrant::points_selector::PointsSelectorOneOf::Filter(f)) => {
                Ok(PointsSelector::FilterSelector(FilterSelector {
                    filter: f.try_into()?,
                }))
            }
            _ => Err(Status::invalid_argument("Malformed PointsSelector type")),
        }
    }
}

impl From<UpdateResult> for api::grpc::qdrant::UpdateResult {
    fn from(value: UpdateResult) -> Self {
        Self {
            operation_id: value.operation_id,
            status: value.status as i32,
        }
    }
}
