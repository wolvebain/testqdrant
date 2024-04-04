use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use common::cpu::CpuPermit;
use common::types::{PointOffsetType, TelemetryDetail};
use itertools::Itertools;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use rstest::rstest;
use segment::data_types::vectors::{only_default_multi_vector, QueryVector, DEFAULT_VECTOR_NAME};
use segment::entry::entry_point::SegmentEntry;
use segment::fixtures::payload_fixtures::{random_int_payload, random_multi_vector};
use segment::index::hnsw_index::graph_links::GraphLinksRam;
use segment::index::hnsw_index::hnsw::HNSWIndex;
use segment::index::hnsw_index::num_rayon_threads;
use segment::index::{PayloadIndex, VectorIndex};
use segment::segment_constructor::build_segment;
use segment::types::{
    Condition, Distance, FieldCondition, Filter, HnswConfig, Indexes, MultiVectorConfig, Payload,
    PayloadSchemaType, Range, SearchParams, SegmentConfig, SeqNumberType, VectorDataConfig,
    VectorStorageType,
};
use segment::vector_storage::query::context_query::ContextPair;
use segment::vector_storage::query::discovery_query::DiscoveryQuery;
use segment::vector_storage::query::reco_query::RecoQuery;
use serde_json::json;
use tempfile::Builder;

use crate::utils::path;

const MAX_EXAMPLE_PAIRS: usize = 4;

enum QueryVariant {
    Nearest,
    RecommendBestScore,
    Discovery,
}

fn random_multi_vec_discovery_query<R: Rng + ?Sized>(
    rnd: &mut R,
    dim: usize,
    num_vector_per_points: usize,
) -> QueryVector {
    let num_pairs: usize = rnd.gen_range(1..MAX_EXAMPLE_PAIRS);

    let target = random_multi_vector(rnd, dim, num_vector_per_points).into();

    let pairs = (0..num_pairs)
        .map(|_| {
            let positive = random_multi_vector(rnd, dim, num_vector_per_points).into();
            let negative = random_multi_vector(rnd, dim, num_vector_per_points).into();
            ContextPair { positive, negative }
        })
        .collect_vec();

    DiscoveryQuery::new(target, pairs).into()
}

fn random_multi_vec_reco_query<R: Rng + ?Sized>(
    rnd: &mut R,
    dim: usize,
    num_vector_per_points: usize,
) -> QueryVector {
    let num_examples: usize = rnd.gen_range(1..MAX_EXAMPLE_PAIRS);

    let positive = (0..num_examples)
        .map(|_| random_multi_vector(rnd, dim, num_vector_per_points).into())
        .collect_vec();
    let negative = (0..num_examples)
        .map(|_| random_multi_vector(rnd, dim, num_vector_per_points).into())
        .collect_vec();

    RecoQuery::new(positive, negative).into()
}

fn random_multi_vec_query<R: Rng + ?Sized>(
    variant: &QueryVariant,
    rnd: &mut R,
    dim: usize,
    num_vector_per_points: usize,
) -> QueryVector {
    match variant {
        QueryVariant::Nearest => random_multi_vector(rnd, dim, num_vector_per_points).into(),
        QueryVariant::Discovery => {
            random_multi_vec_discovery_query(rnd, dim, num_vector_per_points)
        }
        QueryVariant::RecommendBestScore => {
            random_multi_vec_reco_query(rnd, dim, num_vector_per_points)
        }
    }
}

#[rstest]
#[case::nearest(QueryVariant::Nearest, 32, 5)]
#[case::discovery(QueryVariant::Discovery, 128, 10)] // tests that check better precision are in `hnsw_discover_test.rs`
#[case::recommend(QueryVariant::RecommendBestScore, 64, 10)]
fn test_multi_filterable_hnsw(
    #[case] query_variant: QueryVariant,
    #[case] ef: usize,
    #[case] max_failures: usize, // out of 100
) {
    _test_multi_filterable_hnsw(query_variant, ef, max_failures);
}

fn _test_multi_filterable_hnsw(
    query_variant: QueryVariant,
    ef: usize,
    max_failures: usize, // out of 100
) {
    let stopped = AtomicBool::new(false);

    let vector_dim = 8;
    let m = 8;
    let num_points: u64 = 5_000;
    let num_vector_per_points = 2;
    let ef_construct = 16;
    let distance = Distance::Cosine;
    let full_scan_threshold = 16; // KB
    let indexing_threshold = 500; // num vectors
    let num_payload_values = 2;

    let mut rnd = StdRng::seed_from_u64(42);

    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
    let hnsw_dir = Builder::new().prefix("hnsw_dir").tempdir().unwrap();

    let config = SegmentConfig {
        vector_data: HashMap::from([(
            DEFAULT_VECTOR_NAME.to_owned(),
            VectorDataConfig {
                size: vector_dim,
                distance,
                storage_type: VectorStorageType::Memory,
                index: Indexes::Plain {}, // uses plain index
                quantization_config: None,
                multi_vec_config: Some(MultiVectorConfig::default()), // uses multivec config
            },
        )]),
        sparse_vector_data: Default::default(),
        payload_storage_type: Default::default(),
    };

    let int_key = "int";

    let mut segment = build_segment(dir.path(), &config, true).unwrap();
    for n in 0..num_points {
        let idx = n.into();
        let multi_vec = random_multi_vector(&mut rnd, vector_dim, num_vector_per_points);

        let int_payload = random_int_payload(&mut rnd, num_payload_values..=num_payload_values);
        let payload: Payload = json!({int_key:int_payload,}).into();

        let named_vectors = only_default_multi_vector(&multi_vec);
        segment
            .upsert_point(n as SeqNumberType, idx, named_vectors)
            .unwrap();
        segment
            .set_full_payload(n as SeqNumberType, idx, &payload)
            .unwrap();
    }

    let payload_index_ptr = segment.payload_index.clone();

    let hnsw_config = HnswConfig {
        m,
        ef_construct,
        full_scan_threshold,
        max_indexing_threads: 2,
        on_disk: Some(false),
        payload_m: None,
    };

    let permit_cpu_count = num_rayon_threads(hnsw_config.max_indexing_threads);
    let permit = Arc::new(CpuPermit::dummy(permit_cpu_count as u32));

    let vector_storage = &segment.vector_data[DEFAULT_VECTOR_NAME].vector_storage;
    let quantized_vectors = &segment.vector_data[DEFAULT_VECTOR_NAME].quantized_vectors;
    let mut hnsw_index = HNSWIndex::<GraphLinksRam>::open(
        hnsw_dir.path(),
        segment.id_tracker.clone(),
        vector_storage.clone(),
        quantized_vectors.clone(),
        payload_index_ptr.clone(),
        hnsw_config,
    )
    .unwrap();

    hnsw_index.build_index(permit.clone(), &stopped).unwrap();

    payload_index_ptr
        .borrow_mut()
        .set_indexed(&path(int_key), PayloadSchemaType::Integer.into())
        .unwrap();
    let borrowed_payload_index = payload_index_ptr.borrow();
    let blocks = borrowed_payload_index
        .payload_blocks(&path(int_key), indexing_threshold)
        .collect_vec();
    for block in blocks.iter() {
        assert!(
            block.condition.range.is_some(),
            "only range conditions should be generated for this type of payload"
        );
    }

    let mut coverage: HashMap<PointOffsetType, usize> = Default::default();
    let px = payload_index_ptr.borrow();
    for block in &blocks {
        let filter = Filter::new_must(Condition::Field(block.condition.clone()));
        let points = px.query_points(&filter);
        for point in points {
            coverage.insert(point, coverage.get(&point).unwrap_or(&0) + 1);
        }
    }
    let expected_blocks = num_points as usize / indexing_threshold * 2;

    eprintln!("blocks.len() = {:#?}", blocks.len());
    assert!(
        (blocks.len() as i64 - expected_blocks as i64).abs() <= 3,
        "real number of payload blocks is too far from expected"
    );

    assert_eq!(
        coverage.len(),
        num_points as usize,
        "not all points are covered by payload blocks"
    );

    hnsw_index.build_index(permit, &stopped).unwrap();

    let top = 3;
    let mut hits = 0;
    let attempts = 100;
    for i in 0..attempts {
        let query =
            random_multi_vec_query(&query_variant, &mut rnd, vector_dim, num_vector_per_points);

        let range_size = 40;
        let left_range = rnd.gen_range(0..400);
        let right_range = left_range + range_size;

        let filter = Filter::new_must(Condition::Field(FieldCondition::new_range(
            path(int_key),
            Range {
                lt: None,
                gt: None,
                gte: Some(left_range as f64),
                lte: Some(right_range as f64),
            },
        )));

        let filter_query = Some(&filter);

        let index_result = hnsw_index
            .search(
                &[&query],
                filter_query,
                top,
                Some(&SearchParams {
                    hnsw_ef: Some(ef),
                    ..Default::default()
                }),
                &false.into(),
            )
            .unwrap();

        // check that search was performed using HNSW index
        assert_eq!(
            hnsw_index
                .get_telemetry_data(TelemetryDetail::default())
                .filtered_large_cardinality
                .count,
            i + 1
        );

        // segment uses a plain index by configuration
        let plain_result = segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow()
            .search(&[&query], filter_query, top, None, &false.into())
            .unwrap();

        if plain_result == index_result {
            hits += 1;
        }
    }
    assert!(
        attempts - hits <= max_failures,
        "hits: {hits} of {attempts}"
    ); // Not more than X% failures
    eprintln!("hits = {hits:#?} out of {attempts}");
}
