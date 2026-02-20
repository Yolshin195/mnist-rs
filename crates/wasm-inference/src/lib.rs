use nn_engine::{NdArrayEngine, Prediction};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmModel {
    engine: NdArrayEngine,
}

#[wasm_bindgen]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            engine: NdArrayEngine::new(),
        }
    }

    #[wasm_bindgen]
    pub fn load_model(&mut self, bytes: &[u8]) {
        let state = bincode::deserialize(bytes).unwrap();
        self.engine.import_state_sync(state);
    }

    #[wasm_bindgen]
    pub fn predict(&self, pixels: &[u8]) -> JsValue {
        let prediction = self.engine.predict_sync(pixels);

        serde_wasm_bindgen::to_value(&prediction).unwrap()
    }
}
