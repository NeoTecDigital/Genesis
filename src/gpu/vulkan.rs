//! Vulkan context management and Nova integration

use std::ffi::{CString, c_void};
use std::sync::Arc;
use std::ptr;

use crate::category::{Genesis, Instantiation};
use crate::waveform::{WaveformNode, GenesisParams, InstantiationParams};

/// Error types for GPU operations
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("Failed to initialize Nova context: {0}")]
    InitializationError(String),

    #[error("Shader compilation failed: {0}")]
    ShaderCompilationError(String),

    #[error("Buffer allocation failed: {0}")]
    BufferAllocationError(String),

    #[error("Pipeline creation failed: {0}")]
    PipelineCreationError(String),

    #[error("Command execution failed: {0}")]
    ExecutionError(String),

    #[error("Memory transfer failed: {0}")]
    TransferError(String),

    #[error("FFI error: {0}")]
    FFIError(String),
}

/// Opaque handle to Nova's Vulkan context
#[repr(C)]
pub struct NovaContext {
    /// Private pointer to Nova's internal Vulkan state
    handle: *mut c_void,
}

// Safety: NovaContext manages thread-safe Vulkan resources
unsafe impl Send for NovaContext {}
unsafe impl Sync for NovaContext {}

/// GPU execution context managing Nova/Vulkan resources
pub struct GpuContext {
    nova_context: Arc<NovaContext>,
    device_id: u32,
    queue_family_index: u32,
    compute_queue: *mut c_void,
    command_pool: *mut c_void,
}

impl GpuContext {
    /// Initialize GPU context through Nova
    pub fn new() -> Result<Self, GpuError> {
        unsafe {
            // Initialize Nova context
            let nova_handle = nova_init_context();
            if nova_handle.is_null() {
                return Err(GpuError::InitializationError(
                    "Nova context initialization failed".into()
                ));
            }

            // Query device properties
            let device_id = nova_get_device_id(nova_handle);
            let queue_family = nova_get_compute_queue_family(nova_handle);

            // Get compute queue handle
            let queue = nova_get_compute_queue(nova_handle);
            if queue.is_null() {
                nova_destroy_context(nova_handle);
                return Err(GpuError::InitializationError(
                    "Failed to get compute queue".into()
                ));
            }

            // Create command pool for compute commands
            let pool = nova_create_command_pool(nova_handle, queue_family);
            if pool.is_null() {
                nova_destroy_context(nova_handle);
                return Err(GpuError::InitializationError(
                    "Failed to create command pool".into()
                ));
            }

            Ok(GpuContext {
                nova_context: Arc::new(NovaContext { handle: nova_handle }),
                device_id,
                queue_family_index: queue_family,
                compute_queue: queue,
                command_pool: pool,
            })
        }
    }

    /// Get handle to Nova context for FFI calls
    pub fn nova_handle(&self) -> *mut c_void {
        self.nova_context.handle
    }

    /// Submit compute commands to GPU
    pub fn submit_compute(&self, commands: *mut c_void) -> Result<(), GpuError> {
        unsafe {
            let result = nova_submit_compute(
                self.nova_context.handle,
                self.compute_queue,
                commands
            );

            if result != 0 {
                return Err(GpuError::ExecutionError(
                    format!("Compute submission failed with code {}", result)
                ));
            }

            Ok(())
        }
    }

    /// Wait for GPU operations to complete
    pub fn wait_idle(&self) -> Result<(), GpuError> {
        unsafe {
            let result = nova_queue_wait_idle(self.compute_queue);
            if result != 0 {
                return Err(GpuError::ExecutionError(
                    "Failed to wait for queue idle".into()
                ));
            }
            Ok(())
        }
    }

    /// Create a new command buffer for recording commands
    pub fn create_command_buffer(&self) -> Result<*mut c_void, GpuError> {
        unsafe {
            let cmd_buffer = nova_allocate_command_buffer(
                self.nova_context.handle,
                self.command_pool
            );

            if cmd_buffer.is_null() {
                return Err(GpuError::ExecutionError(
                    "Failed to allocate command buffer".into()
                ));
            }

            Ok(cmd_buffer)
        }
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            if !self.command_pool.is_null() {
                nova_destroy_command_pool(self.nova_context.handle, self.command_pool);
            }
        }
    }
}

impl Drop for NovaContext {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                nova_destroy_context(self.handle);
            }
        }
    }
}

// Nova C FFI bindings
extern "C" {
    // Context management
    fn nova_init_context() -> *mut c_void;
    fn nova_destroy_context(context: *mut c_void);

    // Device queries
    fn nova_get_device_id(context: *mut c_void) -> u32;
    fn nova_get_compute_queue_family(context: *mut c_void) -> u32;
    fn nova_get_compute_queue(context: *mut c_void) -> *mut c_void;

    // Command management
    fn nova_create_command_pool(context: *mut c_void, queue_family: u32) -> *mut c_void;
    fn nova_destroy_command_pool(context: *mut c_void, pool: *mut c_void);
    fn nova_allocate_command_buffer(context: *mut c_void, pool: *mut c_void) -> *mut c_void;
    fn nova_free_command_buffer(context: *mut c_void, pool: *mut c_void, buffer: *mut c_void);

    // Execution
    fn nova_submit_compute(context: *mut c_void, queue: *mut c_void, commands: *mut c_void) -> i32;
    fn nova_queue_wait_idle(queue: *mut c_void) -> i32;
}