//! App-wide services owning worker threads: the language model, speech
//! recognition, and the periodic review built on the model.

pub mod model;
pub mod reviewer;
pub mod speech;
