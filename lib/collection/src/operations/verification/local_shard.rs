use super::StrictModeVerification;
use crate::operations::types::{CountRequestInternal, ScrollRequestInternal};

impl StrictModeVerification for ScrollRequestInternal {
    fn query_limit(&self) -> Option<usize> {
        self.limit
    }

    fn timeout(&self) -> Option<usize> {
        None
    }

    fn indexed_filter_read(&self) -> Option<&segment::types::Filter> {
        self.filter.as_ref()
    }

    fn indexed_filter_write(&self) -> Option<&segment::types::Filter> {
        None
    }

    fn request_exact(&self) -> Option<bool> {
        None
    }

    fn request_search_params(&self) -> Option<&segment::types::SearchParams> {
        None
    }
}

impl StrictModeVerification for CountRequestInternal {
    fn query_limit(&self) -> Option<usize> {
        None
    }

    fn timeout(&self) -> Option<usize> {
        None
    }

    fn indexed_filter_read(&self) -> Option<&segment::types::Filter> {
        self.filter.as_ref()
    }

    fn indexed_filter_write(&self) -> Option<&segment::types::Filter> {
        None
    }

    fn request_exact(&self) -> Option<bool> {
        Some(self.exact)
    }

    fn request_search_params(&self) -> Option<&segment::types::SearchParams> {
        None
    }
}
