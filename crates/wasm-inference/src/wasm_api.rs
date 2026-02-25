use wasm_bindgen::prelude::*;
use std::cell::RefCell;

use nn_engine::NdArrayEngine;
use nn_engine::ModelState;
use nn_engine::port::classifier::{
    DigitPredictor, DigitTrainer, ModelStateImporter
};

thread_local! {
    static ENGINE: RefCell<NdArrayEngine> = RefCell::new(NdArrayEngine::new());
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn predict(pixels: Vec<u8>) -> JsValue {
    ENGINE.with(|engine| {
        let engine = engine.borrow();
        let result = engine.predict(&pixels).unwrap();

        serde_wasm_bindgen::to_value(&result).unwrap()
    })
}

#[wasm_bindgen]
pub fn train(label: u8, pixels: Vec<u8>) -> JsValue {
    ENGINE.with(|engine| {
        let mut engine = engine.borrow_mut();
        let result = engine.train(label, &pixels).unwrap();

        serde_wasm_bindgen::to_value(&result).unwrap()
    })
}

#[wasm_bindgen]
pub fn create_empty_model() {
    ENGINE.with(|engine| {
        *engine.borrow_mut() = NdArrayEngine::new();
    });
}

#[wasm_bindgen]
pub fn create_model_from_state(state: JsValue) -> Result<(), JsValue> {
    let model_state: ModelState =
        serde_wasm_bindgen::from_value(state)
            .map_err(|e| JsValue::from_str(&format!("Deserialize error: {e}")))?;

    ENGINE.with(|engine| {
        let mut new_engine = NdArrayEngine::new();

        new_engine
            .import_state(model_state)
            .map_err(|e| JsValue::from_str(&format!("Import error: {e:?}")))?;

        *engine.borrow_mut() = new_engine;

        Ok(())
    })
}