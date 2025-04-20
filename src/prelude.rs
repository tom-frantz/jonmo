//! The jonmo prelude.
//! 
//! This module re-exports the most commonly used items from the jonmo library.
//! Import this module to get convenient access to the core types and traits.
//! 
//! ```
//! use jonmo::prelude::*;
//! ```

pub use crate::signal::{Signal, SignalBuilder, SignalExt, SignalHandle};
pub use crate::signal_vec::{MutableVec, SignalVec, SignalVecBuilder, SignalVecExt, VecDiff};
pub use crate::tree::{dedupe, TERMINATE}; // Assuming dedupe and TERMINATE are in tree.rs
