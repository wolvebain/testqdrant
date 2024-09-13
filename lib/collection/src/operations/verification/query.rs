use api::rest::{Prefetch, QueryGroupsRequestInternal, QueryRequestInternal};

use super::StrictModeVerification;
use crate::collection::Collection;
use crate::operations::config_diff::StrictModeConfig;

impl StrictModeVerification for QueryRequestInternal {
    fn check_custom(
        &self,
        collection: &Collection,
        strict_mode_config: &StrictModeConfig,
    ) -> Result<(), crate::operations::types::CollectionError> {
        if let Some(prefetch) = &self.prefetch {
            for prefetch in prefetch {
                prefetch.check_strict_mode(collection, strict_mode_config)?;
            }
        }

        Ok(())
    }

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
        self.params.as_ref()
    }
}

impl StrictModeVerification for Prefetch {
    fn check_custom(
        &self,
        collection: &Collection,
        strict_mode_config: &StrictModeConfig,
    ) -> Result<(), crate::operations::types::CollectionError> {
        // Prefetch.prefetch is of type Prefetch (recursive type)
        if let Some(prefetch) = &self.prefetch {
            for prefetch in prefetch {
                prefetch.check_strict_mode(collection, strict_mode_config)?;
            }
        }

        Ok(())
    }

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
        self.params.as_ref()
    }
}

impl StrictModeVerification for QueryGroupsRequestInternal {
    fn query_limit(&self) -> Option<usize> {
        self.group_request.limit
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
        self.params.as_ref()
    }
}
