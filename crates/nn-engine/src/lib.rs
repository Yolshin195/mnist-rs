pub mod domain;
pub mod port;
pub mod adapter;

pub use adapter::ndarray_engine::NdArrayEngine;
pub use port::classifier::DigitClassifier;
pub use domain::prediction::Prediction;
pub use domain::error::NNError;
pub use domain::model_state::ModelState;
pub use adapter::file_repository::FileModelRepository;
pub use port::model_repository::ModelRepository;
pub use domain::TrainMetrics;