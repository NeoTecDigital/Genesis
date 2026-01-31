//! GPU integration module for Genesis categorical computation
//!
//! This module provides the interface between Genesis categorical types
//! and Nova's Vulkan compute infrastructure for GPU acceleration.

pub mod vulkan;
pub mod shaders;
pub mod pipeline;
pub mod buffers;

#[cfg(test)]
mod tests;

pub use vulkan::{GpuContext, GpuError};
pub use pipeline::GpuPipeline;
pub use buffers::{GpuBuffer, BufferUsage};
pub use shaders::{ShaderModule, ShaderStage};

/// Re-export key types for convenience
pub use self::vulkan::NovaContext;