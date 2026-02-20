use wasm_bindgen::prelude::*;
use std::cell::RefCell;

use nn_engine::NdArrayEngine;
use nn_engine::port::classifier::{
    DigitPredictor, DigitTrainer
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