use std::sync::Arc;

use bitvec::vec::BitVec;
use common::types::PointOffsetType;
use parking_lot::lock_api::RwLock;
use parking_lot::Mutex;
use rand::Rng;

use super::cpu_graph_builder::CpuGraphBuilder;
use super::gpu_graph_builder::GpuGraphBuilder;
use crate::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;
use crate::vector_storage::{RawScorer, VectorStorageEnum};

pub const CPU_POINTS_COUNT_MULTIPLICATOR: usize = 8;
pub const CANDIDATES_CAPACITY_DIV: usize = 8;

pub struct CombinedGraphBuilder<'a, TFabric>
where
    TFabric: Fn() -> Box<dyn RawScorer + 'a> + Send + Sync + 'a,
{
    pub cpu_builder: Arc<CpuGraphBuilder<'a, TFabric>>,
    pub cpu_threads: usize,
    pub gpu_builder: Arc<Mutex<GpuGraphBuilder>>,
    pub gpu_threads: usize,
}

impl<'a, TFabric> CombinedGraphBuilder<'a, TFabric>
where
    TFabric: Fn() -> Box<dyn RawScorer + 'a> + Send + Sync + 'a,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new<R>(
        num_vectors: usize,
        m: usize,
        m0: usize,
        ef_construct: usize,
        entry_points_num: usize,
        scorer_fabric: TFabric,
        vector_storage: &VectorStorageEnum,
        dim: usize,
        rng: &mut R,
        cpu_threads: usize,
        gpu_threads: usize,
    ) -> Self
    where
        R: Rng + ?Sized,
    {
        let cpu_builder = Arc::new(CpuGraphBuilder::new(
            num_vectors,
            m,
            m0,
            ef_construct,
            entry_points_num,
            scorer_fabric,
            rng,
        ));

        let gpu_builder = Arc::new(Mutex::new(GpuGraphBuilder::new(
            num_vectors,
            m,
            m0,
            ef_construct,
            vector_storage,
            dim,
            cpu_builder.point_levels.clone(),
            gpu_threads,
        )));
        gpu_builder.lock().clear_links();

        Self {
            cpu_builder,
            cpu_threads,
            gpu_builder,
            gpu_threads,
        }
    }

    pub fn into_graph_layers_builder(self) -> GraphLayersBuilder {
        let mut links_layers = vec![];
        let num_vectors = self.cpu_builder.graph_layers_builder.links_layers.len();
        for point_levels in &self.cpu_builder.graph_layers_builder.links_layers {
            let mut layers = vec![];
            for level in point_levels {
                let links = level.read().clone();
                layers.push(parking_lot::RwLock::new(links));
            }
            links_layers.push(layers);
        }
        GraphLayersBuilder {
            max_level: std::sync::atomic::AtomicUsize::new(
                self.cpu_builder
                    .graph_layers_builder
                    .max_level
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            m: self.cpu_builder.graph_layers_builder.m,
            m0: self.cpu_builder.graph_layers_builder.m0,
            ef_construct: self.cpu_builder.graph_layers_builder.ef_construct,
            level_factor: self.cpu_builder.graph_layers_builder.level_factor,
            use_heuristic: self.cpu_builder.graph_layers_builder.use_heuristic,
            links_layers,
            entry_points: Mutex::new(
                self.cpu_builder
                    .graph_layers_builder
                    .entry_points
                    .lock()
                    .clone(),
            ),
            visited_pool: crate::index::visited_pool::VisitedPool::new(),
            ready_list: RwLock::new(BitVec::repeat(false, num_vectors)),
        }
    }

    fn download_links(
        cpu_builder: Arc<CpuGraphBuilder<'a, TFabric>>,
        gpu_builder: Arc<Mutex<GpuGraphBuilder>>,
        level: usize,
    ) {
        let gpu_builder = gpu_builder.lock();
        for idx in 0..cpu_builder.num_vectors() as PointOffsetType {
            if level <= cpu_builder.get_point_level(idx) {
                let links = gpu_builder.get_links(idx);
                cpu_builder.set_links(level, idx, links);
            }
        }
    }

    fn upload_links(
        cpu_builder: Arc<CpuGraphBuilder<'a, TFabric>>,
        gpu_builder: Arc<Mutex<GpuGraphBuilder>>,
        level: usize,
        count: usize,
    ) {
        let mut gpu_builder = gpu_builder.lock();
        let mut links = vec![];
        gpu_builder.clear_links();
        for idx in 0..count {
            links.clear();
            cpu_builder.links_map(level, idx as PointOffsetType, |link| {
                links.push(link);
            });
            gpu_builder.set_links(idx as PointOffsetType, &links);
        }
    }

    pub fn build(&self) {
        struct GpuStartData {
            level: usize,
            start_idx: PointOffsetType,
            entries: Vec<Option<u32>>,
        }

        let timer = std::time::Instant::now();
        let pool = rayon::ThreadPoolBuilder::new()
            .thread_name(|idx| format!("hnsw-build-{idx}"))
            .num_threads(self.cpu_threads)
            .build()
            .unwrap();

        let max_level = self.cpu_builder.max_level();
        let cpu_count = (self.gpu_threads * self.cpu_builder.m * CPU_POINTS_COUNT_MULTIPLICATOR)
            as PointOffsetType;

        let (sender, receiver) = std::sync::mpsc::channel::<GpuStartData>();
        rayon::scope(|s| {
            // spawn CPU thread
            s.spawn(move |_| {
                for level in (0..=max_level).rev() {
                    let timer = std::time::Instant::now();
                    let gpu_start = self.cpu_builder.build_level(&pool, level, cpu_count);
                    println!("CPU level {} build time = {:?}", level, timer.elapsed());

                    if gpu_start < self.cpu_builder.num_vectors() as u32 {
                        let entries = self.cpu_builder.entries.lock().clone();
                        sender
                            .send(GpuStartData {
                                level,
                                start_idx: gpu_start,
                                entries,
                            })
                            .unwrap();
                    }
                }
            });

            // spawn GPU thread
            s.spawn(move |_| {
                while let Ok(m) = receiver.recv() {
                    let timer = std::time::Instant::now();
                    Self::upload_links(
                        self.cpu_builder.clone(),
                        self.gpu_builder.clone(),
                        m.level,
                        m.start_idx as usize,
                    );
                    self.gpu_builder
                        .lock()
                        .build_level(m.entries, m.level, m.start_idx);
                    Self::download_links(
                        self.cpu_builder.clone(),
                        self.gpu_builder.clone(),
                        m.level,
                    );
                    println!("GPU level {} build time = {:?}", m.level, timer.elapsed());
                }
            });
        });
        println!("GPU+CPU total build time = {:?}", timer.elapsed());
    }
}

#[cfg(test)]
mod tests {
    use common::fixed_length_priority_queue::FixedLengthPriorityQueue;
    use common::types::ScoredPointOffset;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;
    use crate::common::rocksdb_wrapper::{open_db, DB_VECTOR_CF};
    use crate::fixtures::index_fixtures::{
        random_vector, FakeFilterContext, TestRawScorerProducer,
    };
    use crate::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;
    use crate::index::hnsw_index::graph_links::GraphLinksRam;
    use crate::index::hnsw_index::point_scorer::FilteredScorer;
    use crate::spaces::simple::CosineMetric;
    use crate::types::Distance;
    use crate::vector_storage::dense::simple_dense_vector_storage::open_simple_dense_vector_storage;
    use crate::vector_storage::VectorStorage;

    #[test]
    fn test_gpu_hnsw_equal() {
        let num_vectors = 10_000;
        let dim = 16;
        let m = 8;
        let m0 = 16;
        let ef_construct = 16;
        let entry_points_num = 10;
        let gpu_threads_count = 1;
        let cpu_threads_count = 1;

        let mut rng = StdRng::seed_from_u64(42);
        let vector_holder = TestRawScorerProducer::<CosineMetric>::new(dim, num_vectors, &mut rng);
        let dir = tempfile::Builder::new().prefix("db_dir").tempdir().unwrap();
        let db = open_db(dir.path(), &[DB_VECTOR_CF]).unwrap();
        let storage = open_simple_dense_vector_storage(
            db,
            DB_VECTOR_CF,
            dim,
            Distance::Cosine,
            &false.into(),
        )
        .unwrap();
        {
            let mut borrowed_storage = storage.borrow_mut();
            for idx in 0..(num_vectors as PointOffsetType) {
                borrowed_storage
                    .insert_vector(idx, vector_holder.vectors.get(idx).into())
                    .unwrap();
            }
        }

        let added_vector = vector_holder.vectors.get(0).to_vec();
        let graph_layers_2 = CombinedGraphBuilder::new(
            num_vectors,
            m,
            m0,
            ef_construct,
            entry_points_num,
            || vector_holder.get_raw_scorer(added_vector.clone()).unwrap(),
            &storage.borrow(),
            dim,
            &mut rng,
            cpu_threads_count,
            gpu_threads_count,
        );
        graph_layers_2.build();

        let mut graph_layers_1 = GraphLayersBuilder::new_with_params(
            num_vectors,
            m,
            m0,
            ef_construct,
            entry_points_num,
            true,
            true,
        );

        for idx in 0..(num_vectors as PointOffsetType) {
            let fake_filter_context = FakeFilterContext {};
            let added_vector = vector_holder.vectors.get(idx).to_vec();
            let raw_scorer = vector_holder.get_raw_scorer(added_vector.clone()).unwrap();

            let scorer = FilteredScorer::new(raw_scorer.as_ref(), Some(&fake_filter_context));
            graph_layers_1.set_levels(idx, graph_layers_2.cpu_builder.get_point_level(idx));
            graph_layers_1.link_new_point(idx, scorer);
        }

        let graph_layers_2 = graph_layers_2.into_graph_layers_builder();

        for (point_id, layer_1) in graph_layers_1.links_layers.iter().enumerate() {
            for (level, links_1) in layer_1.iter().enumerate().rev() {
                let links_1 = links_1.read().clone();
                let links_2 = graph_layers_2.links_layers[point_id][level].read().clone();
                assert_eq!(links_1.as_slice(), links_2);
            }
        }
    }

    #[test]
    fn test_gpu_hnsw_quality() {
        let num_vectors = 10_000;
        let dim = 32;
        let m = 8;
        let m0 = 16;
        let ef_construct = 16;
        let entry_points_num = 10;
        let gpu_threads_count = 6;
        let cpu_threads_count = 4;

        let mut rng = StdRng::seed_from_u64(42);
        let vector_holder = TestRawScorerProducer::<CosineMetric>::new(dim, num_vectors, &mut rng);
        let dir = tempfile::Builder::new().prefix("db_dir").tempdir().unwrap();
        let db = open_db(dir.path(), &[DB_VECTOR_CF]).unwrap();
        let storage = open_simple_dense_vector_storage(
            db,
            DB_VECTOR_CF,
            dim,
            Distance::Cosine,
            &false.into(),
        )
        .unwrap();
        {
            let mut borrowed_storage = storage.borrow_mut();
            for idx in 0..(num_vectors as PointOffsetType) {
                borrowed_storage
                    .insert_vector(idx, vector_holder.vectors.get(idx).into())
                    .unwrap();
            }
        }

        let added_vector = vector_holder.vectors.get(0).to_vec();
        let graph_layers_2 = CombinedGraphBuilder::new(
            num_vectors,
            m,
            m0,
            ef_construct,
            entry_points_num,
            || vector_holder.get_raw_scorer(added_vector.clone()).unwrap(),
            &storage.borrow(),
            dim,
            &mut rng,
            cpu_threads_count,
            gpu_threads_count,
        );
        graph_layers_2.build();

        let mut graph_layers_1 = GraphLayersBuilder::new_with_params(
            num_vectors,
            m,
            m0,
            ef_construct,
            entry_points_num,
            true,
            true,
        );

        let timer = std::time::Instant::now();
        for idx in 0..(num_vectors as PointOffsetType) {
            let fake_filter_context = FakeFilterContext {};
            let added_vector = vector_holder.vectors.get(idx).to_vec();
            let raw_scorer = vector_holder.get_raw_scorer(added_vector.clone()).unwrap();

            let scorer = FilteredScorer::new(raw_scorer.as_ref(), Some(&fake_filter_context));
            graph_layers_1.set_levels(idx, graph_layers_2.cpu_builder.get_point_level(idx));
            graph_layers_1.link_new_point(idx, scorer);
        }
        println!("CPU total build time = {:?}", timer.elapsed());

        let graph_layers_2 = graph_layers_2.into_graph_layers_builder();

        let graph_1 = graph_layers_1
            .into_graph_layers::<GraphLinksRam>(None)
            .unwrap();
        let graph_2 = graph_layers_2
            .into_graph_layers::<GraphLinksRam>(None)
            .unwrap();

        let attempts = 10;
        let top = 10;
        let ef = 16;
        let mut total_sames_1 = 0;
        let mut total_sames_2 = 0;
        for _ in 0..attempts {
            let query = random_vector(&mut rng, dim);
            let fake_filter_context = FakeFilterContext {};
            let raw_scorer = vector_holder.get_raw_scorer(query).unwrap();

            let mut reference_top = FixedLengthPriorityQueue::<ScoredPointOffset>::new(top);
            for idx in 0..vector_holder.vectors.len() as PointOffsetType {
                reference_top.push(ScoredPointOffset {
                    idx,
                    score: raw_scorer.score_point(idx),
                });
            }
            let brute_top = reference_top.into_vec();

            let scorer = FilteredScorer::new(raw_scorer.as_ref(), Some(&fake_filter_context));
            let graph_search_1 = graph_1.search(top, ef, scorer, None);
            let sames_1 = sames_count(&brute_top, &graph_search_1);
            total_sames_1 += sames_1;

            let scorer = FilteredScorer::new(raw_scorer.as_ref(), Some(&fake_filter_context));
            let graph_search_2 = graph_2.search(top, ef, scorer, None);
            let sames_2 = sames_count(&brute_top, &graph_search_2);
            total_sames_2 += sames_2;
        }
        let min_sames = top as f32 * 0.7 * attempts as f32;
        println!("total_sames_1 = {}", total_sames_1);
        println!("total_sames_2 = {}", total_sames_2);
        println!("min_sames = {}", min_sames);
        assert!(total_sames_1 as f32 > min_sames);
        assert!(total_sames_2 as f32 > min_sames);
    }

    fn sames_count(a: &[ScoredPointOffset], b: &[ScoredPointOffset]) -> usize {
        let mut count = 0;
        for a_item in a {
            for b_item in b {
                if a_item.idx == b_item.idx {
                    count += 1;
                }
            }
        }
        count
    }
}