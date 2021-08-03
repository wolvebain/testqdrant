use std::path::{Path, PathBuf};

use segment::types::{HnswConfig, Indexes, PayloadIndexType, SegmentType, StorageType};

use crate::config::CollectionParams;
use crate::segment_manager::holders::segment_holder::{
    LockedSegment, LockedSegmentHolder, SegmentId,
};
use crate::segment_manager::optimizers::segment_optimizer::{
    OptimizerThresholds, SegmentOptimizer,
};

pub struct IndexingOptimizer {
    thresholds_config: OptimizerThresholds,
    segments_path: PathBuf,
    collection_temp_dir: PathBuf,
    collection_params: CollectionParams,
    hnsw_config: HnswConfig,
}

impl IndexingOptimizer {
    pub fn new(
        thresholds_config: OptimizerThresholds,
        segments_path: PathBuf,
        collection_temp_dir: PathBuf,
        collection_params: CollectionParams,
        hnsw_config: HnswConfig,
    ) -> Self {
        IndexingOptimizer {
            thresholds_config,
            segments_path,
            collection_temp_dir,
            collection_params,
            hnsw_config,
        }
    }

    fn worst_segment(&self, segments: LockedSegmentHolder) -> Option<(SegmentId, LockedSegment)> {
        segments
            .read()
            .iter()
            .filter_map(|(idx, segment)| {
                let segment_entry = segment.get();
                let read_segment = segment_entry.read();
                let vector_count = read_segment.vectors_count();

                let segment_config = read_segment.config();

                if read_segment.segment_type() == SegmentType::Special {
                    return None; // Never optimize already optimized segment
                }

                // Apply indexing to plain segments which have grown too big
                let is_vector_indexed = match segment_config.index {
                    Indexes::Plain { .. } => false,
                    Indexes::Hnsw(_) => true,
                };

                let is_payload_indexed = match segment_config.payload_index.unwrap_or_default() {
                    PayloadIndexType::Plain => false,
                    PayloadIndexType::Struct => true,
                };

                let is_memmaped = match segment_config.storage_type {
                    StorageType::InMemory => false,
                    StorageType::Mmap => true,
                };

                let big_for_mmap = vector_count >= self.thresholds_config.memmap_threshold;
                let big_for_index = vector_count >= self.thresholds_config.indexing_threshold;
                let big_for_payload_index =
                    vector_count >= self.thresholds_config.payload_indexing_threshold;

                let has_payload = !read_segment.get_indexed_fields().is_empty();

                let require_indexing = (big_for_mmap && !is_memmaped)
                    || (big_for_index && !is_vector_indexed)
                    || (has_payload && big_for_payload_index && !is_payload_indexed);

                match require_indexing {
                    true => Some((*idx, vector_count)),
                    false => None,
                }
            })
            .max_by_key(|(_, num_vectors)| *num_vectors)
            .map(|(idx, _)| (idx, segments.read().get(idx).unwrap().clone()))
    }
}

impl SegmentOptimizer for IndexingOptimizer {
    fn collection_path(&self) -> &Path {
        self.segments_path.as_path()
    }

    fn temp_path(&self) -> &Path {
        self.collection_temp_dir.as_path()
    }

    fn collection_params(&self) -> CollectionParams {
        self.collection_params.clone()
    }

    fn hnsw_config(&self) -> HnswConfig {
        self.hnsw_config
    }

    fn threshold_config(&self) -> &OptimizerThresholds {
        &self.thresholds_config
    }

    fn check_condition(&self, segments: LockedSegmentHolder) -> Vec<SegmentId> {
        match self.worst_segment(segments) {
            None => vec![],
            Some((segment_id, _segment)) => vec![segment_id],
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Deref;
    use std::sync::Arc;

    use itertools::Itertools;
    use parking_lot::lock_api::RwLock;
    use tempdir::TempDir;

    use segment::types::StorageType;

    use crate::operations::point_ops::{PointInsertOperations, PointOperations};
    use crate::operations::FieldIndexOperations;
    use crate::segment_manager::fixtures::random_segment;
    use crate::segment_manager::holders::segment_holder::SegmentHolder;
    use crate::segment_manager::simple_segment_updater::{
        process_field_index_operation, process_point_operation,
    };

    use super::*;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_indexing_optimizer() {
        init();

        let mut holder = SegmentHolder::default();

        let payload_field = "number".to_owned();

        let dim = 4;

        let segments_dir = TempDir::new("segments_dir").unwrap();
        let segments_temp_dir = TempDir::new("segments_temp_dir").unwrap();
        let mut opnum = 101..1000000;

        let small_segment = random_segment(segments_dir.path(), opnum.next().unwrap(), 25, dim);
        let middle_segment = random_segment(segments_dir.path(), opnum.next().unwrap(), 100, dim);
        let large_segment = random_segment(segments_dir.path(), opnum.next().unwrap(), 200, dim);

        let segment_config = small_segment.segment_config.clone();

        let small_segment_id = holder.add(small_segment);
        let middle_segment_id = holder.add(middle_segment);
        let large_segment_id = holder.add(large_segment);

        let mut index_optimizer = IndexingOptimizer::new(
            OptimizerThresholds {
                memmap_threshold: 1000,
                indexing_threshold: 1000,
                payload_indexing_threshold: 50,
            },
            segments_dir.path().to_owned(),
            segments_temp_dir.path().to_owned(),
            CollectionParams {
                vector_size: segment_config.vector_size,
                distance: segment_config.distance,
            },
            Default::default(),
        );

        let locked_holder = Arc::new(RwLock::new(holder));

        // ---- check condition for MMap optimization
        let suggested_to_optimize = index_optimizer.check_condition(locked_holder.clone());
        assert!(suggested_to_optimize.is_empty());

        index_optimizer.thresholds_config.memmap_threshold = 150;
        index_optimizer.thresholds_config.indexing_threshold = 50;

        let suggested_to_optimize = index_optimizer.check_condition(locked_holder.clone());
        assert!(suggested_to_optimize.contains(&large_segment_id));

        // ----- CREATE AN INDEXED FIELD ------
        process_field_index_operation(
            locked_holder.deref(),
            opnum.next().unwrap(),
            &FieldIndexOperations::CreateIndex(payload_field.clone()),
        )
        .unwrap();

        // ------ Plain -> Mmap & Indexed payload
        let suggested_to_optimize = index_optimizer.check_condition(locked_holder.clone());
        assert!(suggested_to_optimize.contains(&large_segment_id));
        eprintln!("suggested_to_optimize = {:#?}", suggested_to_optimize);
        index_optimizer
            .optimize(locked_holder.clone(), suggested_to_optimize)
            .unwrap();
        eprintln!("Done");

        // ------ Plain -> Indexed payload
        let suggested_to_optimize = index_optimizer.check_condition(locked_holder.clone());
        assert!(suggested_to_optimize.contains(&middle_segment_id));
        index_optimizer
            .optimize(locked_holder.clone(), suggested_to_optimize)
            .unwrap();

        // ------- Keep smallest segment without changes
        let suggested_to_optimize = index_optimizer.check_condition(locked_holder.clone());
        assert!(suggested_to_optimize.is_empty());

        assert_eq!(
            locked_holder.read().len(),
            3,
            "Testing no new segments were created"
        );

        let infos = locked_holder
            .read()
            .iter()
            .map(|(_sid, segment)| segment.get().read().info())
            .collect_vec();
        let configs = locked_holder
            .read()
            .iter()
            .map(|(_sid, segment)| segment.get().read().config())
            .collect_vec();

        let indexed_count = infos
            .iter()
            .filter(|info| info.segment_type == SegmentType::Indexed)
            .count();
        assert_eq!(
            indexed_count, 2,
            "Testing that 2 segments are actually indexed"
        );

        let mmap_count = configs
            .iter()
            .filter(|config| config.storage_type == StorageType::Mmap)
            .count();
        assert_eq!(
            mmap_count, 1,
            "Testing that only largest segment is not Mmap"
        );

        let segment_dirs = segments_dir.path().read_dir().unwrap().collect_vec();
        assert_eq!(
            segment_dirs.len(),
            locked_holder.read().len(),
            "Testing that new segments are persisted and old data is removed"
        );

        for info in infos.iter() {
            assert!(
                info.schema.contains_key(&payload_field),
                "Testing that payload is not lost"
            );
            assert!(
                info.schema[&payload_field].indexed,
                "Testing that payload index is not lost"
            );
        }

        let insert_point_ops = PointOperations::UpsertPoints(PointInsertOperations::BatchPoints {
            ids: vec![501, 502, 503],
            vectors: vec![
                vec![1.0, 0.0, 0.5, 0.0],
                vec![1.0, 0.0, 0.5, 0.5],
                vec![1.0, 0.0, 0.5, 1.0],
            ],
            payloads: None,
        });

        let smallest_size = infos
            .iter()
            .min_by_key(|info| info.num_vectors)
            .unwrap()
            .num_vectors;

        process_point_operation(
            locked_holder.deref(),
            opnum.next().unwrap(),
            insert_point_ops,
        )
        .unwrap();

        let new_infos = locked_holder
            .read()
            .iter()
            .map(|(_sid, segment)| segment.get().read().info())
            .collect_vec();
        let new_smallest_size = new_infos
            .iter()
            .min_by_key(|info| info.num_vectors)
            .unwrap()
            .num_vectors;

        assert_eq!(
            new_smallest_size,
            smallest_size + 3,
            "Testing that new data is added to an appendable segment only"
        );

        // ---- New appendable segment should be created if none left

        // Index even the smallest segment
        index_optimizer.thresholds_config.payload_indexing_threshold = 20;
        let suggested_to_optimize = index_optimizer.check_condition(locked_holder.clone());
        assert!(suggested_to_optimize.contains(&small_segment_id));
        index_optimizer
            .optimize(locked_holder.clone(), suggested_to_optimize)
            .unwrap();

        let new_infos2 = locked_holder
            .read()
            .iter()
            .map(|(_sid, segment)| segment.get().read().info())
            .collect_vec();

        assert!(
            new_infos2.len() > new_infos.len(),
            "Check that new appendable segment was created"
        );

        let insert_point_ops = PointOperations::UpsertPoints(PointInsertOperations::BatchPoints {
            ids: vec![601, 602, 603],
            vectors: vec![
                vec![0.0, 1.0, 0.5, 0.0],
                vec![0.0, 1.0, 0.5, 0.5],
                vec![0.0, 1.0, 0.5, 1.0],
            ],
            payloads: None,
        });

        process_point_operation(
            locked_holder.deref(),
            opnum.next().unwrap(),
            insert_point_ops,
        )
        .unwrap();
    }
}
