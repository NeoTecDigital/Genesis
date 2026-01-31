//! Genesis Categorical Type System
//!
//! This library implements the categorical type system that enforces
//! the Genesis Instantiation Principle (GIP) formalism.

pub mod category;
pub mod waveform;

#[cfg(feature = "gpu")]
pub mod ffi;

#[cfg(feature = "gpu")]
pub mod gpu;

// Re-export main types for convenience
pub use category::{
    Empty, Unit, Numeric,
    Morphism, Genesis, Instantiation,
    MorphismBuilder,
};

pub use waveform::{
    WaveformNode, Complex64,
    GenesisParams, InstantiationParams,
};

#[cfg(feature = "gpu")]
pub use gpu::{GpuContext, GpuPipeline, GpuBuffer, GpuError};

#[cfg(test)]
mod tests;