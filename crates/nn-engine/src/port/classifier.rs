use crate::domain::{ModelState, Prediction, TrainingStepResult, error::NNError};

pub trait DigitPredictor {
    fn predict(&self, pixels: &[u8]) -> Result<Prediction, NNError>;
}

pub trait DigitTrainer {
    fn train(&mut self, label: u8, pixels: &[u8]) -> Result<TrainingStepResult, NNError>;
}

pub trait ModelStateExporter {
    fn export_state(&self) -> Result<ModelState, NNError>;
}

pub trait ModelStateImporter {
    fn import_state(&mut self, state: ModelState) -> Result<(), NNError>;
}