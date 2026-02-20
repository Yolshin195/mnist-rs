use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Prediction {
    pub digit: u8,
    pub confidence: f32,
}
