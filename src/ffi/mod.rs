//! Foreign Function Interface (FFI) module
//!
//! Provides bindings to external C libraries, primarily Nova for GPU operations.

pub mod nova;

pub use nova::{NovaApi, BufferUsage, MemoryInfo};