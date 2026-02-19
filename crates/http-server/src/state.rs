use std::sync::Arc;
use nn_engine::DigitClassifier;


#[derive(Clone)]
pub struct AppState {
    pub classifier: Arc<dyn DigitClassifier>,
}