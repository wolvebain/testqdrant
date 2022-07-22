pub mod actix_telemetry;
pub mod api;
#[allow(dead_code)] // May contain functions used in different binaries. Not actually dead
pub mod helpers;

use std::sync::Arc;

use ::api::grpc::models::{ApiResponse, ApiStatus, VersionInfo};
use actix_cors::Cors;
use actix_web::middleware::{Condition, Logger};
use actix_web::web::Data;
use actix_web::{error, get, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use storage::dispatcher::Dispatcher;

use crate::actix::api::cluster_api::config_cluster_api;
use crate::actix::api::collections_api::config_collections_api;
use crate::actix::api::count_api::count_points;
use crate::actix::api::recommend_api::recommend_points;
use crate::actix::api::retrieve_api::{get_point, get_points, scroll_points};
use crate::actix::api::search_api::search_points;
use crate::actix::api::snapshot_api::config_snapshots_api;
use crate::actix::api::telemetry_api::config_telemetry_api;
use crate::actix::api::update_api::config_update_api;
use crate::common::telemetry::TelemetryCollector;
use crate::settings::{max_web_workers, Settings};

fn json_error_handler(err: error::JsonPayloadError, _req: &HttpRequest) -> error::Error {
    use actix_web::error::JsonPayloadError;

    let detail = err.to_string();
    let mut resp_b = match &err {
        JsonPayloadError::ContentType => HttpResponse::UnsupportedMediaType(),
        JsonPayloadError::Deserialize(json_err) if json_err.is_data() => {
            HttpResponse::UnprocessableEntity()
        }
        _ => HttpResponse::BadRequest(),
    };
    let response = resp_b.json(ApiResponse::<()> {
        result: None,
        status: ApiStatus::Error(detail),
        time: 0.0,
    });
    error::InternalError::from_response(err, response).into()
}

#[get("/")]
pub async fn index() -> impl Responder {
    HttpResponse::Ok().json(VersionInfo::default())
}

#[allow(dead_code)]
pub fn init(
    dispatcher: Arc<Dispatcher>,
    telemetry_collector: Arc<parking_lot::Mutex<TelemetryCollector>>,
    settings: Settings,
) -> std::io::Result<()> {
    actix_web::rt::System::new().block_on(async {
        let toc_data = web::Data::new(dispatcher.toc().clone());
        let dispatcher_data = web::Data::new(dispatcher);
        let telemetry_data = web::Data::new(telemetry_collector.clone());
        HttpServer::new(move || {
            let cors = Cors::default()
                .allow_any_origin()
                .allow_any_method()
                .allow_any_header();

            App::new()
                .wrap(Condition::new(settings.service.enable_cors, cors))
                .wrap(actix_telemetry::ActixTelemetryTransform::new(
                    telemetry_collector.clone(),
                ))
                .wrap(Logger::default())
                .app_data(dispatcher_data.clone())
                .app_data(toc_data.clone())
                .app_data(telemetry_data.clone())
                .app_data(Data::new(
                    web::JsonConfig::default()
                        .limit(32 * 1024 * 1024)
                        .error_handler(json_error_handler),
                )) // 32 Mb
                .service(index)
                .configure(config_collections_api)
                .configure(config_snapshots_api)
                .configure(config_update_api)
                .configure(config_cluster_api)
                .configure(config_telemetry_api)
                .service(get_point)
                .service(get_points)
                .service(scroll_points)
                .service(search_points)
                .service(recommend_points)
                .service(count_points)
        })
        .workers(max_web_workers(&settings))
        .bind(format!(
            "{}:{}",
            settings.service.host, settings.service.http_port
        ))?
        .run()
        .await
    })
}

#[cfg(test)]
mod tests {
    use ::api::grpc::api_crate_version;

    #[test]
    fn test_version() {
        assert_eq!(
            api_crate_version(),
            env!("CARGO_PKG_VERSION"),
            "Qdrant and lib/api crate versions are not same"
        );
    }
}
