mod adapter;
#[cfg(feature = "server")]
mod application;
mod domain;
pub mod port;

#[cfg(feature = "server")]
pub use adapter::file_repository::FileModelRepository;

pub use adapter::ndarray_engine::NdArrayEngine;

#[cfg(feature = "server")]
pub use adapter::async_ndarray_engine::AsyncNdArrayEngine;

#[cfg(feature = "server")]
pub use application::digit_classifier_service::DigitClassifierService;