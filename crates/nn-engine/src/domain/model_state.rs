use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct ModelState {
    pub w1: Vec<f32>, // 128 * 784
    pub b1: Vec<f32>, // 128
    pub w2: Vec<f32>, // 10 * 128
    pub b2: Vec<f32>, // 10
}
