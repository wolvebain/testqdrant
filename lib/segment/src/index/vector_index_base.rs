use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use common::cpu::CpuPermit;
use common::types::{PointOffsetType, ScoredPointOffset, TelemetryDetail};
use half::f16;
use sparse::common::types::DimId;
use sparse::index::inverted_index::inverted_index_compressed_immutable_ram::InvertedIndexCompressedImmutableRam;
use sparse::index::inverted_index::inverted_index_compressed_mmap::InvertedIndexCompressedMmap;
use sparse::index::inverted_index::inverted_index_immutable_ram::InvertedIndexImmutableRam;
use sparse::index::inverted_index::inverted_index_mmap::InvertedIndexMmap;
use sparse::index::inverted_index::inverted_index_ram::InvertedIndexRam;

use super::hnsw_index::graph_links::{GraphLinksMmap, GraphLinksRam};
use super::hnsw_index::hnsw::HNSWIndex;
use super::plain_payload_index::PlainIndex;
use super::sparse_index::sparse_vector_index::SparseVectorIndex;
use crate::common::operation_error::OperationResult;
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::{QueryVector, VectorRef};
use crate::telemetry::VectorIndexSearchesTelemetry;
use crate::types::{Filter, SearchParams};

/// Trait for vector searching
pub trait VectorIndex {
    /// Return list of Ids with fitting
    fn search(
        &self,
        vectors: &[&QueryVector],
        filter: Option<&Filter>,
        top: usize,
        params: Option<&SearchParams>,
        query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>>;

    /// Force internal index rebuild.
    fn build_index(&mut self, permit: Arc<CpuPermit>, stopped: &AtomicBool) -> OperationResult<()> {
        self.build_index_with_progress(permit, stopped, || ())
    }

    /// Force internal index rebuild.
    fn build_index_with_progress(
        &mut self,
        permit: Arc<CpuPermit>,
        stopped: &AtomicBool,
        tick_progress: impl FnMut(),
    ) -> OperationResult<()>;

    fn get_telemetry_data(&self, detail: TelemetryDetail) -> VectorIndexSearchesTelemetry;

    fn files(&self) -> Vec<PathBuf>;

    /// The number of indexed vectors, currently accessible
    fn indexed_vector_count(&self) -> usize;

    /// Update index for a single vector
    fn update_vector(&mut self, id: PointOffsetType, vector: VectorRef) -> OperationResult<()>;
}

pub enum VectorIndexEnum {
    Plain(PlainIndex),
    HnswRam(HNSWIndex<GraphLinksRam>),
    HnswMmap(HNSWIndex<GraphLinksMmap>),
    SparseRam(SparseVectorIndex<InvertedIndexRam>),
    SparseImmRam(SparseVectorIndex<InvertedIndexImmutableRam>),
    SparseMmap(SparseVectorIndex<InvertedIndexMmap>),
    SparseCompImmRamF32(SparseVectorIndex<InvertedIndexCompressedImmutableRam<f32>>),
    SparseCompImmRamF16(SparseVectorIndex<InvertedIndexCompressedImmutableRam<f16>>),
    SparseCompMmapF32(SparseVectorIndex<InvertedIndexCompressedMmap<f32>>),
    SparseCompMmapF16(SparseVectorIndex<InvertedIndexCompressedMmap<f16>>),
}

impl VectorIndexEnum {
    pub fn is_index(&self) -> bool {
        match self {
            Self::Plain(_) => false,
            Self::HnswRam(_) => true,
            Self::HnswMmap(_) => true,
            Self::SparseRam(_) => true,
            Self::SparseImmRam(_) => true,
            Self::SparseMmap(_) => true,
            Self::SparseCompImmRamF32(_) => true,
            Self::SparseCompImmRamF16(_) => true,
            Self::SparseCompMmapF32(_) => true,
            Self::SparseCompMmapF16(_) => true,
        }
    }

    pub fn fill_idf_statistics(&self, idf: &mut HashMap<DimId, usize>) {
        match self {
            Self::Plain(_) | Self::HnswRam(_) | Self::HnswMmap(_) => (),
            Self::SparseRam(index) => index.fill_idf_statistics(idf),
            Self::SparseImmRam(index) => index.fill_idf_statistics(idf),
            Self::SparseMmap(index) => index.fill_idf_statistics(idf),
            Self::SparseCompImmRamF32(index) => index.fill_idf_statistics(idf),
            Self::SparseCompImmRamF16(index) => index.fill_idf_statistics(idf),
            Self::SparseCompMmapF32(index) => index.fill_idf_statistics(idf),
            Self::SparseCompMmapF16(index) => index.fill_idf_statistics(idf),
        }
    }
}

impl VectorIndex for VectorIndexEnum {
    fn search(
        &self,
        vectors: &[&QueryVector],
        filter: Option<&Filter>,
        top: usize,
        params: Option<&SearchParams>,
        query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        match self {
            VectorIndexEnum::Plain(index) => {
                index.search(vectors, filter, top, params, query_context)
            }
            VectorIndexEnum::HnswRam(index) => {
                index.search(vectors, filter, top, params, query_context)
            }
            VectorIndexEnum::HnswMmap(index) => {
                index.search(vectors, filter, top, params, query_context)
            }
            VectorIndexEnum::SparseRam(index) => {
                index.search(vectors, filter, top, params, query_context)
            }
            VectorIndexEnum::SparseImmRam(index) => {
                index.search(vectors, filter, top, params, query_context)
            }
            VectorIndexEnum::SparseMmap(index) => {
                index.search(vectors, filter, top, params, query_context)
            }
            VectorIndexEnum::SparseCompImmRamF32(index) => {
                index.search(vectors, filter, top, params, query_context)
            }
            VectorIndexEnum::SparseCompImmRamF16(index) => {
                index.search(vectors, filter, top, params, query_context)
            }
            VectorIndexEnum::SparseCompMmapF32(index) => {
                index.search(vectors, filter, top, params, query_context)
            }
            VectorIndexEnum::SparseCompMmapF16(index) => {
                index.search(vectors, filter, top, params, query_context)
            }
        }
    }

    fn build_index_with_progress(
        &mut self,
        permit: Arc<CpuPermit>,
        stopped: &AtomicBool,
        tick_progress: impl FnMut(),
    ) -> OperationResult<()> {
        match self {
            VectorIndexEnum::Plain(index) => {
                index.build_index_with_progress(permit, stopped, tick_progress)
            }
            VectorIndexEnum::HnswRam(index) => {
                index.build_index_with_progress(permit, stopped, tick_progress)
            }
            VectorIndexEnum::HnswMmap(index) => {
                index.build_index_with_progress(permit, stopped, tick_progress)
            }
            VectorIndexEnum::SparseRam(index) => {
                index.build_index_with_progress(permit, stopped, tick_progress)
            }
            VectorIndexEnum::SparseImmRam(index) => {
                index.build_index_with_progress(permit, stopped, tick_progress)
            }
            VectorIndexEnum::SparseMmap(index) => {
                index.build_index_with_progress(permit, stopped, tick_progress)
            }
            VectorIndexEnum::SparseCompImmRamF32(index) => {
                index.build_index_with_progress(permit, stopped, tick_progress)
            }
            VectorIndexEnum::SparseCompImmRamF16(index) => {
                index.build_index_with_progress(permit, stopped, tick_progress)
            }
            VectorIndexEnum::SparseCompMmapF32(index) => {
                index.build_index_with_progress(permit, stopped, tick_progress)
            }
            VectorIndexEnum::SparseCompMmapF16(index) => {
                index.build_index_with_progress(permit, stopped, tick_progress)
            }
        }
    }

    fn get_telemetry_data(&self, detail: TelemetryDetail) -> VectorIndexSearchesTelemetry {
        match self {
            VectorIndexEnum::Plain(index) => index.get_telemetry_data(detail),
            VectorIndexEnum::HnswRam(index) => index.get_telemetry_data(detail),
            VectorIndexEnum::HnswMmap(index) => index.get_telemetry_data(detail),
            VectorIndexEnum::SparseRam(index) => index.get_telemetry_data(detail),
            VectorIndexEnum::SparseImmRam(index) => index.get_telemetry_data(detail),
            VectorIndexEnum::SparseMmap(index) => index.get_telemetry_data(detail),
            VectorIndexEnum::SparseCompImmRamF32(index) => index.get_telemetry_data(detail),
            VectorIndexEnum::SparseCompImmRamF16(index) => index.get_telemetry_data(detail),
            VectorIndexEnum::SparseCompMmapF32(index) => index.get_telemetry_data(detail),
            VectorIndexEnum::SparseCompMmapF16(index) => index.get_telemetry_data(detail),
        }
    }

    fn files(&self) -> Vec<PathBuf> {
        match self {
            VectorIndexEnum::Plain(index) => index.files(),
            VectorIndexEnum::HnswRam(index) => index.files(),
            VectorIndexEnum::HnswMmap(index) => index.files(),
            VectorIndexEnum::SparseRam(index) => index.files(),
            VectorIndexEnum::SparseImmRam(index) => index.files(),
            VectorIndexEnum::SparseMmap(index) => index.files(),
            VectorIndexEnum::SparseCompImmRamF32(index) => index.files(),
            VectorIndexEnum::SparseCompImmRamF16(index) => index.files(),
            VectorIndexEnum::SparseCompMmapF32(index) => index.files(),
            VectorIndexEnum::SparseCompMmapF16(index) => index.files(),
        }
    }

    fn indexed_vector_count(&self) -> usize {
        match self {
            Self::Plain(index) => index.indexed_vector_count(),
            Self::HnswRam(index) => index.indexed_vector_count(),
            Self::HnswMmap(index) => index.indexed_vector_count(),
            Self::SparseRam(index) => index.indexed_vector_count(),
            Self::SparseImmRam(index) => index.indexed_vector_count(),
            Self::SparseMmap(index) => index.indexed_vector_count(),
            Self::SparseCompImmRamF32(index) => index.indexed_vector_count(),
            Self::SparseCompImmRamF16(index) => index.indexed_vector_count(),
            Self::SparseCompMmapF32(index) => index.indexed_vector_count(),
            Self::SparseCompMmapF16(index) => index.indexed_vector_count(),
        }
    }

    fn update_vector(&mut self, id: PointOffsetType, vector: VectorRef) -> OperationResult<()> {
        match self {
            Self::Plain(index) => index.update_vector(id, vector),
            Self::HnswRam(index) => index.update_vector(id, vector),
            Self::HnswMmap(index) => index.update_vector(id, vector),
            Self::SparseRam(index) => index.update_vector(id, vector),
            Self::SparseImmRam(index) => index.update_vector(id, vector),
            Self::SparseMmap(index) => index.update_vector(id, vector),
            Self::SparseCompImmRamF32(index) => index.update_vector(id, vector),
            Self::SparseCompImmRamF16(index) => index.update_vector(id, vector),
            Self::SparseCompMmapF32(index) => index.update_vector(id, vector),
            Self::SparseCompMmapF16(index) => index.update_vector(id, vector),
        }
    }
}
