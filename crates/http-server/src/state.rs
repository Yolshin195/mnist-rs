use nn_engine::port::async_classifier::AsyncDigitClassifier;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub classifier: Arc<dyn AsyncDigitClassifier>,
}
