mod fixtures;


#[cfg(test)]
mod tests {
    use crate::fixtures::segment::build_segment_1;
    use segment::entry::entry_point::SegmentEntry;
    use std::collections::HashSet;
    use segment::types::{Filter, Condition};

    #[test]
    fn test_point_exclusion() {
        let segment = build_segment_1();

        assert!(segment.has_point(3));

        let query_vector = vec![1.0, 1.0, 1.0, 1.0];

        let res = segment.search(&query_vector, None, 1, None).unwrap();

        let best_match = res.get(0).expect("Non-empty result");
        assert_eq!(best_match.idx, 3);


        let ids: HashSet<_> = vec![3].into_iter().collect();


        let frt = Filter {
            should: None,
            must: None,
            must_not: Some(vec![Condition::HasId(ids)]),
        };


        let res = segment.search(&query_vector, Some(&frt), 1, None).unwrap();

        let best_match = res.get(0).expect("Non-empty result");
        assert_ne!(best_match.idx, 3);
    }
}