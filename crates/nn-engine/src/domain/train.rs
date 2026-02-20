use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingStepResult {
    pub loss: f32,
    pub correct: bool,
}
