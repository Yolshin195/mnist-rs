use std::fmt;

#[derive(Debug)]
pub enum NNError {
    InvalidInput,
    IoError(String),
    SerializationError,
    PersistenceError,
}

impl fmt::Display for NNError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NNError::InvalidInput => write!(f, "Invalid input data"),
            NNError::IoError(msg) => write!(f, "I/O error: {}", msg),
            NNError::SerializationError => write!(f, "Serialization error"),
            NNError::PersistenceError => write!(f, "Persistence error"),
        }
    }
}

impl std::error::Error for NNError {}