mod domain;
pub use domain::ModelState;
pub mod port;


mod adapter;
pub use adapter::ndarray_engine::NdArrayEngine;
#[cfg(feature = "server")]
pub use adapter::file_repository::FileModelRepository;
#[cfg(feature = "server")]
pub use adapter::file_repository::JsonModelRepository;
#[cfg(feature = "server")]
pub use adapter::async_ndarray_engine::AsyncNdArrayEngine;


#[cfg(feature = "server")]
mod application;
#[cfg(feature = "server")]
pub use application::digit_classifier_service::DigitClassifierService;