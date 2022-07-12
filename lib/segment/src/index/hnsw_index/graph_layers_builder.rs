use crate::index::hnsw_index::entry_points::EntryPoints;
use crate::index::hnsw_index::graph_layers::{GraphLayers, GraphLayersBase, LinkContainer};
use crate::index::hnsw_index::point_scorer::FilteredScorer;
use crate::index::visited_pool::{VisitedList, VisitedPool};
use crate::spaces::tools::FixedLengthPriorityQueue;
use crate::types::{PointOffsetType, ScoreType};
use crate::vector_storage::ScoredPointOffset;
use parking_lot::{Mutex, RwLock};
use rand::distributions::Uniform;
use rand::Rng;
use std::cmp::min;
use std::collections::BinaryHeap;
use std::sync::atomic::AtomicUsize;

pub type LockedLinkContainer = RwLock<LinkContainer>;
pub type LockedLayersContainer = Vec<LockedLinkContainer>;

/// Same as `GraphLayers`,  but allows to build in parallel
/// Convertable to `GraphLayers`
pub struct GraphLayersBuilder {
    max_level: AtomicUsize,
    m: usize,
    m0: usize,
    ef_construct: usize,
    // Factor of level probability
    level_factor: f64,
    // Exclude points according to "not closer than base" heuristic?
    use_heuristic: bool,
    links_layers: Vec<LockedLayersContainer>,
    entry_points: Mutex<EntryPoints>,

    // Fields used on construction phase only
    visited_pool: VisitedPool,
}

impl GraphLayersBase for GraphLayersBuilder {
    fn get_visited_list_from_pool(&self) -> VisitedList {
        self.visited_pool.get(self.num_points())
    }

    fn return_visited_list_to_pool(&self, visited_list: VisitedList) {
        self.visited_pool.return_back(visited_list);
    }

    fn links_map<F>(&self, point_id: PointOffsetType, level: usize, mut f: F)
    where
        F: FnMut(PointOffsetType),
    {
        let links = self.links_layers[point_id as usize][level].read();
        for link in links.iter() {
            f(*link);
        }
    }

    fn get_m(&self, level: usize) -> usize {
        if level == 0 {
            self.m0
        } else {
            self.m
        }
    }
}

impl GraphLayersBuilder {
    pub fn into_graph_layers(self) -> GraphLayers {
        let unlocker_links_layers = self
            .links_layers
            .into_iter()
            .map(|l| l.into_iter().map(|l| l.into_inner()).collect())
            .collect();

        GraphLayers {
            max_level: self.max_level.load(std::sync::atomic::Ordering::Relaxed),
            m: self.m,
            m0: self.m0,
            ef_construct: self.ef_construct,
            links_layers: unlocker_links_layers,
            entry_points: self.entry_points.into_inner(),
            visited_pool: self.visited_pool,
        }
    }

    pub fn new_with_params(
        num_vectors: usize, // Initial number of points in index
        m: usize,           // Expected M for non-first layer
        m0: usize,          // Expected M for first layer
        ef_construct: usize,
        entry_points_num: usize, // Depends on number of points
        use_heuristic: bool,
        reserve: bool,
    ) -> Self {
        let mut links_layers: Vec<LockedLayersContainer> = vec![];

        for _i in 0..num_vectors {
            let mut links = Vec::new();
            if reserve {
                links.reserve(m0);
            }
            links_layers.push(vec![RwLock::new(links)]);
        }

        Self {
            max_level: AtomicUsize::new(0),
            m,
            m0,
            ef_construct,
            level_factor: 1.0 / (m as f64).ln(),
            use_heuristic,
            links_layers,
            entry_points: Mutex::new(EntryPoints::new(entry_points_num)),
            visited_pool: VisitedPool::new(),
        }
    }

    pub fn new(
        num_vectors: usize, // Initial number of points in index
        m: usize,           // Expected M for non-first layer
        m0: usize,          // Expected M for first layer
        ef_construct: usize,
        entry_points_num: usize, // Depends on number of points
        use_heuristic: bool,
    ) -> Self {
        Self::new_with_params(
            num_vectors,
            m,
            m0,
            ef_construct,
            entry_points_num,
            use_heuristic,
            true,
        )
    }

    fn num_points(&self) -> usize {
        self.links_layers.len()
    }

    /// Generate random level for a new point, according to geometric distribution
    pub fn get_random_layer<R>(&self, rng: &mut R) -> usize
    where
        R: Rng + ?Sized,
    {
        let distribution = Uniform::new(0.0, 1.0);
        let sample: f64 = rng.sample(distribution);
        let picked_level = -sample.ln() * self.level_factor;
        picked_level.round() as usize
    }

    fn get_point_level(&self, point_id: PointOffsetType) -> usize {
        self.links_layers[point_id as usize].len() - 1
    }

    pub fn set_levels(&mut self, point_id: PointOffsetType, level: usize) {
        if self.links_layers.len() <= point_id as usize {
            while self.links_layers.len() <= point_id as usize {
                self.links_layers.push(vec![]);
            }
        }
        let point_layers = &mut self.links_layers[point_id as usize];
        while point_layers.len() <= level {
            let mut links = vec![];
            links.reserve(self.m);
            point_layers.push(RwLock::new(links));
        }
        self.max_level
            .fetch_max(level, std::sync::atomic::Ordering::Relaxed);
    }

    /// Connect new point to links, so that links contains only closest points
    fn connect_new_point<F>(
        links: &mut LinkContainer,
        new_point_id: PointOffsetType,
        target_point_id: PointOffsetType,
        level_m: usize,
        mut score_internal: F,
    ) where
        F: FnMut(PointOffsetType, PointOffsetType) -> ScoreType,
    {
        // ToDo: binary search here ? (most likely does not worth it)
        let new_to_target = score_internal(target_point_id, new_point_id);

        let mut id_to_insert = links.len();
        for (i, &item) in links.iter().enumerate() {
            let target_to_link = score_internal(target_point_id, item);
            if target_to_link < new_to_target {
                id_to_insert = i;
                break;
            }
        }

        if links.len() < level_m {
            links.insert(id_to_insert, new_point_id);
        } else if id_to_insert != links.len() {
            links.pop();
            links.insert(id_to_insert, new_point_id);
        }
    }

    /// <https://github.com/nmslib/hnswlib/issues/99>
    fn select_candidate_with_heuristic_from_sorted<F>(
        candidates: impl Iterator<Item = ScoredPointOffset>,
        m: usize,
        mut score_internal: F,
    ) -> Vec<PointOffsetType>
    where
        F: FnMut(PointOffsetType, PointOffsetType) -> ScoreType,
    {
        let mut result_list = vec![];
        result_list.reserve(m);
        for current_closest in candidates {
            if result_list.len() >= m {
                break;
            }
            let mut is_good = true;
            for &selected_point in &result_list {
                let dist_to_already_selected = score_internal(current_closest.idx, selected_point);
                if dist_to_already_selected > current_closest.score {
                    is_good = false;
                    break;
                }
            }
            if is_good {
                result_list.push(current_closest.idx);
            }
        }

        result_list
    }

    /// <https://github.com/nmslib/hnswlib/issues/99>
    fn select_candidates_with_heuristic<F>(
        candidates: FixedLengthPriorityQueue<ScoredPointOffset>,
        m: usize,
        score_internal: F,
    ) -> Vec<PointOffsetType>
    where
        F: FnMut(PointOffsetType, PointOffsetType) -> ScoreType,
    {
        let closest_iter = candidates.into_iter();
        Self::select_candidate_with_heuristic_from_sorted(closest_iter, m, score_internal)
    }

    pub fn link_new_point(&self, point_id: PointOffsetType, mut points_scorer: FilteredScorer) {
        // Check if there is an suitable entry point
        //   - entry point level if higher or equal
        //   - it satisfies filters

        let level = self.get_point_level(point_id);

        let entry_point_opt = self
            .entry_points
            .lock()
            .new_point(point_id, level, |point_id| {
                points_scorer.check_point(point_id)
            });
        match entry_point_opt {
            // New point is a new empty entry (for this filter, at least)
            // We can't do much here, so just quit
            None => {}

            // Entry point found.
            Some(entry_point) => {
                let mut level_entry = if entry_point.level > level {
                    // The entry point is higher than a new point
                    // Let's find closest one on same level

                    // greedy search for a single closest point
                    self.search_entry(
                        entry_point.point_id,
                        entry_point.level,
                        level,
                        &mut points_scorer,
                    )
                } else {
                    ScoredPointOffset {
                        idx: entry_point.point_id,
                        score: points_scorer.score_internal(point_id, entry_point.point_id),
                    }
                };
                // minimal common level for entry points
                let linking_level = min(level, entry_point.level);

                for curr_level in (0..=linking_level).rev() {
                    let level_m = self.get_m(curr_level);

                    let nearest_points = {
                        let existing_links =
                            self.links_layers[point_id as usize][curr_level].read();
                        self.search_on_level(
                            level_entry,
                            curr_level,
                            self.ef_construct,
                            &mut points_scorer,
                            &existing_links,
                        )
                    };

                    let scorer = |a, b| points_scorer.score_internal(a, b);

                    if self.use_heuristic {
                        let selected_nearest =
                            Self::select_candidates_with_heuristic(nearest_points, level_m, scorer);
                        self.links_layers[point_id as usize][curr_level]
                            .write()
                            .clone_from(&selected_nearest);

                        for &other_point in &selected_nearest {
                            let mut other_point_links =
                                self.links_layers[other_point as usize][curr_level].write();
                            if other_point_links.len() < level_m {
                                // If linked point is lack of neighbours
                                other_point_links.push(point_id);
                            } else {
                                let mut candidates = BinaryHeap::with_capacity(level_m + 1);
                                candidates.push(ScoredPointOffset {
                                    idx: point_id,
                                    score: scorer(point_id, other_point),
                                });
                                for other_point_link in
                                    other_point_links.iter().take(level_m).copied()
                                {
                                    candidates.push(ScoredPointOffset {
                                        idx: other_point_link,
                                        score: scorer(other_point_link, other_point),
                                    });
                                }
                                let selected_candidates =
                                    Self::select_candidate_with_heuristic_from_sorted(
                                        candidates.into_sorted_vec().into_iter().rev(),
                                        level_m,
                                        scorer,
                                    );
                                other_point_links.clear(); // this do not free memory, which is good
                                for selected in selected_candidates.iter().copied() {
                                    other_point_links.push(selected);
                                }
                            }
                        }
                    } else {
                        for nearest_point in &nearest_points {
                            {
                                let mut links =
                                    self.links_layers[point_id as usize][curr_level].write();
                                Self::connect_new_point(
                                    &mut links,
                                    nearest_point.idx,
                                    point_id,
                                    level_m,
                                    scorer,
                                );
                            }

                            {
                                let mut links = self.links_layers[nearest_point.idx as usize]
                                    [curr_level]
                                    .write();
                                Self::connect_new_point(
                                    &mut links,
                                    point_id,
                                    nearest_point.idx,
                                    level_m,
                                    scorer,
                                );
                            }
                            if nearest_point.score > level_entry.score {
                                level_entry = *nearest_point;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::fixtures::index_fixtures::{
        random_vector, FakeFilterContext, TestRawScorerProducer,
    };

    use super::*;
    use crate::index::hnsw_index::tests::create_graph_layer_fixture;
    use crate::spaces::metric::Metric;
    use crate::spaces::simple::{CosineMetric, EuclidMetric};
    use crate::types::VectorElementType;
    use crate::vector_storage::RawScorer;
    use itertools::Itertools;
    use rand::prelude::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};

    const M: usize = 8;

    fn parallel_graph_build<TMetric: Metric + Sync + Send, R>(
        num_vectors: usize,
        dim: usize,
        use_heuristic: bool,
        rng: &mut R,
    ) -> (TestRawScorerProducer<TMetric>, GraphLayersBuilder)
    where
        R: Rng + ?Sized,
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        let m = M;
        let ef_construct = 16;
        let entry_points_num = 10;

        let vector_holder = TestRawScorerProducer::<TMetric>::new(dim, num_vectors, rng);

        let mut graph_layers = GraphLayersBuilder::new(
            num_vectors,
            m,
            m * 2,
            ef_construct,
            entry_points_num,
            use_heuristic,
        );

        for idx in 0..(num_vectors as PointOffsetType) {
            let level = graph_layers.get_random_layer(rng);
            graph_layers.set_levels(idx, level);
        }
        pool.install(|| {
            (0..(num_vectors as PointOffsetType))
                .into_par_iter()
                .for_each(|idx| {
                    let fake_filter_context = FakeFilterContext {};
                    let added_vector = vector_holder.vectors.get(idx).to_vec();
                    let raw_scorer = vector_holder.get_raw_scorer(added_vector);
                    let scorer = FilteredScorer::new(&raw_scorer, Some(&fake_filter_context));
                    graph_layers.link_new_point(idx, scorer);
                });
        });

        (vector_holder, graph_layers)
    }

    fn create_graph_layer<TMetric: Metric, R>(
        num_vectors: usize,
        dim: usize,
        use_heuristic: bool,
        rng: &mut R,
    ) -> (TestRawScorerProducer<TMetric>, GraphLayersBuilder)
    where
        R: Rng + ?Sized,
    {
        let m = M;
        let ef_construct = 16;
        let entry_points_num = 10;

        let vector_holder = TestRawScorerProducer::<TMetric>::new(dim, num_vectors, rng);

        let mut graph_layers = GraphLayersBuilder::new(
            num_vectors,
            m,
            m * 2,
            ef_construct,
            entry_points_num,
            use_heuristic,
        );

        for idx in 0..(num_vectors as PointOffsetType) {
            let level = graph_layers.get_random_layer(rng);
            graph_layers.set_levels(idx, level);
        }

        for idx in 0..(num_vectors as PointOffsetType) {
            let fake_filter_context = FakeFilterContext {};
            let added_vector = vector_holder.vectors.get(idx).to_vec();
            let raw_scorer = vector_holder.get_raw_scorer(added_vector.clone());
            let scorer = FilteredScorer::new(&raw_scorer, Some(&fake_filter_context));
            graph_layers.link_new_point(idx, scorer);
        }

        (vector_holder, graph_layers)
    }

    #[test]
    fn test_parallel_graph_build() {
        let num_vectors = 1000;
        let dim = 8;

        let mut rng = StdRng::seed_from_u64(42);
        type M = CosineMetric;

        // let (vector_holder, graph_layers_builder) =
        //     create_graph_layer::<M, _>(num_vectors, dim, false, &mut rng);

        let (vector_holder, graph_layers_builder) =
            parallel_graph_build::<M, _>(num_vectors, dim, false, &mut rng);

        let main_entry = graph_layers_builder
            .entry_points
            .lock()
            .get_entry_point(|_x| true)
            .expect("Expect entry point to exists");

        assert!(main_entry.level > 0);

        let num_levels = graph_layers_builder
            .links_layers
            .iter()
            .map(|x| x.len())
            .max()
            .unwrap();
        assert_eq!(main_entry.level + 1, num_levels);

        let total_links_0: usize = graph_layers_builder
            .links_layers
            .iter()
            .map(|x| x[0].read().len())
            .sum();

        assert!(total_links_0 > 0);

        eprintln!("total_links_0 = {:#?}", total_links_0);
        eprintln!("num_vectors = {:#?}", num_vectors);

        assert!(total_links_0 as f64 / num_vectors as f64 > M as f64);

        let top = 5;
        let query = random_vector(&mut rng, dim);
        let processed_query = M::preprocess(&query).unwrap_or_else(|| query.clone());
        let mut reference_top = FixedLengthPriorityQueue::new(top);
        for idx in 0..vector_holder.vectors.len() as PointOffsetType {
            let vec = &vector_holder.vectors.get(idx);
            reference_top.push(ScoredPointOffset {
                idx,
                score: M::similarity(vec, &processed_query),
            });
        }

        let graph = graph_layers_builder.into_graph_layers();

        let fake_filter_context = FakeFilterContext {};
        let raw_scorer = vector_holder.get_raw_scorer(query);
        let scorer = FilteredScorer::new(&raw_scorer, Some(&fake_filter_context));
        let ef = 16;
        let graph_search = graph.search(top, ef, scorer);

        assert_eq!(reference_top.into_vec(), graph_search);
    }

    #[test]
    fn test_add_points() {
        let num_vectors = 1000;
        let dim = 8;

        let mut rng = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);

        type M = CosineMetric;

        let (vector_holder, graph_layers_builder) =
            create_graph_layer::<M, _>(num_vectors, dim, false, &mut rng);

        let (_vector_holder_orig, graph_layers_orig) =
            create_graph_layer_fixture::<M, _>(num_vectors, M, dim, false, &mut rng2);

        // check is graph_layers_builder links are equeal to graph_layers_orig
        let orig_len = graph_layers_orig.links_layers[0].len();
        let builder_len = graph_layers_builder.links_layers[0].len();

        assert_eq!(orig_len, builder_len);

        for idx in 0..builder_len {
            let links_orig = &graph_layers_orig.links_layers[0][idx];
            let links_builder = graph_layers_builder.links_layers[0][idx].read();
            let link_container_from_builder = links_builder.iter().copied().collect::<Vec<_>>();
            assert_eq!(links_orig, &link_container_from_builder);
        }

        let main_entry = graph_layers_builder
            .entry_points
            .lock()
            .get_entry_point(|_x| true)
            .expect("Expect entry point to exists");

        assert!(main_entry.level > 0);

        let num_levels = graph_layers_builder
            .links_layers
            .iter()
            .map(|x| x.len())
            .max()
            .unwrap();
        assert_eq!(main_entry.level + 1, num_levels);

        let total_links_0: usize = graph_layers_builder
            .links_layers
            .iter()
            .map(|x| x[0].read().len())
            .sum();

        assert!(total_links_0 > 0);

        eprintln!("total_links_0 = {:#?}", total_links_0);
        eprintln!("num_vectors = {:#?}", num_vectors);

        assert!(total_links_0 as f64 / num_vectors as f64 > M as f64);

        let top = 5;
        let query = random_vector(&mut rng, dim);
        let processed_query = M::preprocess(&query).unwrap_or_else(|| query.clone());
        let mut reference_top = FixedLengthPriorityQueue::new(top);
        for idx in 0..vector_holder.vectors.len() as PointOffsetType {
            let vec = &vector_holder.vectors.get(idx);
            reference_top.push(ScoredPointOffset {
                idx,
                score: M::similarity(vec, &processed_query),
            });
        }

        let graph = graph_layers_builder.into_graph_layers();

        let fake_filter_context = FakeFilterContext {};
        let raw_scorer = vector_holder.get_raw_scorer(query);
        let scorer = FilteredScorer::new(&raw_scorer, Some(&fake_filter_context));
        let ef = 16;
        let graph_search = graph.search(top, ef, scorer);

        assert_eq!(reference_top.into_vec(), graph_search);
    }

    #[test]
    #[ignore]
    fn test_hnsw_graph_properties() {
        const NUM_VECTORS: usize = 5_000;
        const DIM: usize = 16;
        const M: usize = 16;
        const EF_CONSTRUCT: usize = 64;
        const USE_HEURISTIC: bool = true;

        let mut rng = StdRng::seed_from_u64(42);

        let vector_holder = TestRawScorerProducer::<CosineMetric>::new(DIM, NUM_VECTORS, &mut rng);
        let mut graph_layers_builder =
            GraphLayersBuilder::new(NUM_VECTORS, M, M * 2, EF_CONSTRUCT, 10, USE_HEURISTIC);
        let fake_filter_context = FakeFilterContext {};
        for idx in 0..(NUM_VECTORS as PointOffsetType) {
            let added_vector = vector_holder.vectors.get(idx).to_vec();
            let raw_scorer = vector_holder.get_raw_scorer(added_vector);
            let scorer = FilteredScorer::new(&raw_scorer, Some(&fake_filter_context));
            let level = graph_layers_builder.get_random_layer(&mut rng);
            graph_layers_builder.set_levels(idx, level);
            graph_layers_builder.link_new_point(idx, scorer);
        }
        let graph_layers = graph_layers_builder.into_graph_layers();

        let number_layers = graph_layers.links_layers.len();
        eprintln!("number_layers = {:#?}", number_layers);

        let max_layers = graph_layers.links_layers.iter().map(|x| x.len()).max();
        eprintln!("max_layers = {:#?}", max_layers);

        eprintln!(
            "graph_layers.links_layers[910] = {:#?}",
            graph_layers.links_layers[910]
        );

        let total_edges: usize = graph_layers.links_layers.iter().map(|x| x[0].len()).sum();
        let avg_connectivity = total_edges as f64 / NUM_VECTORS as f64;
        eprintln!("avg_connectivity = {:#?}", avg_connectivity);
    }

    #[test]
    #[ignore]
    fn test_candidate_selection_heuristics() {
        const NUM_VECTORS: usize = 100;
        const DIM: usize = 16;
        const M: usize = 16;

        let mut rng = StdRng::seed_from_u64(42);

        let vector_holder = TestRawScorerProducer::<EuclidMetric>::new(DIM, NUM_VECTORS, &mut rng);

        let mut candidates: FixedLengthPriorityQueue<ScoredPointOffset> =
            FixedLengthPriorityQueue::new(NUM_VECTORS);

        let new_vector_to_insert = random_vector(&mut rng, DIM);

        let scorer = vector_holder.get_raw_scorer(new_vector_to_insert);

        for i in 0..NUM_VECTORS {
            candidates.push(ScoredPointOffset {
                idx: i as PointOffsetType,
                score: scorer.score_point(i as PointOffsetType),
            });
        }

        let sorted_candidates = candidates.into_vec();

        for x in sorted_candidates.iter().take(M) {
            eprintln!("sorted_candidates = ({}, {})", x.idx, x.score);
        }

        let selected_candidates = GraphLayersBuilder::select_candidate_with_heuristic_from_sorted(
            sorted_candidates.into_iter(),
            M,
            |a, b| scorer.score_internal(a, b),
        );

        for x in selected_candidates.iter() {
            eprintln!("selected_candidates = {}", x);
        }
    }

    #[test]
    fn test_connect_new_point() {
        let num_points = 10;
        let m = 6;
        let ef_construct = 32;

        // See illustration in docs
        let points: Vec<Vec<VectorElementType>> = vec![
            vec![21.79, 7.18],  // Target
            vec![20.58, 5.46],  // 1  B - yes
            vec![21.19, 4.51],  // 2  C
            vec![24.73, 8.24],  // 3  D - yes
            vec![24.55, 9.98],  // 4  E
            vec![26.11, 6.85],  // 5  F
            vec![17.64, 11.14], // 6  G - yes
            vec![14.97, 11.52], // 7  I
            vec![14.97, 9.60],  // 8  J
            vec![16.23, 14.32], // 9  H
            vec![12.69, 19.13], // 10 K
        ];

        let scorer = |a: PointOffsetType, b: PointOffsetType| {
            -((points[a as usize][0] - points[b as usize][0]).powi(2)
                + (points[a as usize][1] - points[b as usize][1]).powi(2))
            .sqrt()
        };

        let mut insert_ids = (1..points.len() as PointOffsetType).collect_vec();

        let mut candidates = FixedLengthPriorityQueue::new(insert_ids.len());
        for &id in &insert_ids {
            candidates.push(ScoredPointOffset {
                idx: id,
                score: scorer(0, id),
            });
        }

        let res = GraphLayersBuilder::select_candidates_with_heuristic(candidates, m, scorer);

        assert_eq!(&res, &vec![1, 3, 6]);

        let mut rng = StdRng::seed_from_u64(42);

        let graph_layers_builder = GraphLayersBuilder::new(num_points, m, m, ef_construct, 1, true);
        insert_ids.shuffle(&mut rng);
        for &id in &insert_ids {
            let level_m = graph_layers_builder.get_m(0);
            let mut links = graph_layers_builder.links_layers[0][0].write();
            GraphLayersBuilder::connect_new_point(&mut links, id, 0, level_m, scorer)
        }
        let mut result = Vec::new();
        graph_layers_builder.links_map(0, 0, |link| result.push(link));
        assert_eq!(&result, &vec![1, 2, 3, 4, 5, 6]);
    }
}
