//! Nova C FFI bindings for Vulkan compute operations
//!
//! This module provides safe Rust wrappers around Nova's C API
//! for GPU compute operations via Vulkan.

use std::ffi::{c_void, c_char, CStr, CString};
use std::ptr;

/// Nova API wrapper providing safe access to C functions
pub struct NovaApi;

impl NovaApi {
    /// Initialize Nova Vulkan context
    pub unsafe fn init() -> *mut c_void {
        nova_init_context()
    }

    /// Destroy Nova context
    pub unsafe fn destroy(context: *mut c_void) {
        if !context.is_null() {
            nova_destroy_context(context);
        }
    }

    /// Create a compute buffer
    pub unsafe fn create_buffer(
        context: *mut c_void,
        size: usize,
        usage: BufferUsage,
    ) -> *mut c_void {
        nova_create_buffer(context, size, usage as u32)
    }

    /// Load a compiled shader module
    pub unsafe fn load_shader(
        context: *mut c_void,
        path: &str,
    ) -> Result<*mut c_void, String> {
        let c_path = CString::new(path)
            .map_err(|e| format!("Invalid path: {}", e))?;

        let shader = nova_load_shader(context, c_path.as_ptr());
        if shader.is_null() {
            Err(format!("Failed to load shader: {}", path))
        } else {
            Ok(shader)
        }
    }

    /// Dispatch compute workgroups
    pub unsafe fn dispatch_compute(
        context: *mut c_void,
        shader: *mut c_void,
        x: u32,
        y: u32,
        z: u32,
    ) {
        nova_dispatch_compute(context, shader, x, y, z);
    }

    /// Wait for all GPU operations to complete
    pub unsafe fn wait_idle(context: *mut c_void) {
        nova_device_wait_idle(context);
    }

    /// Get device name
    pub unsafe fn get_device_name(context: *mut c_void) -> String {
        let name_ptr = nova_get_device_name(context);
        if name_ptr.is_null() {
            String::from("Unknown Device")
        } else {
            CStr::from_ptr(name_ptr)
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Get device memory info
    pub unsafe fn get_memory_info(context: *mut c_void) -> MemoryInfo {
        let mut info = MemoryInfo::default();
        nova_get_memory_info(
            context,
            &mut info.total_memory,
            &mut info.available_memory,
            &mut info.dedicated_memory,
        );
        info
    }
}

/// Buffer usage flags
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum BufferUsage {
    TransferSrc = 0x00000001,
    TransferDst = 0x00000002,
    UniformBuffer = 0x00000010,
    StorageBuffer = 0x00000020,
    IndexBuffer = 0x00000040,
    VertexBuffer = 0x00000080,
}

/// Device memory information
#[derive(Debug, Default, Clone)]
pub struct MemoryInfo {
    pub total_memory: u64,
    pub available_memory: u64,
    pub dedicated_memory: u64,
}

// Raw FFI declarations - link with Nova C library
#[link(name = "nova_compute")]
extern "C" {
    // Context management
    fn nova_init_context() -> *mut c_void;
    fn nova_destroy_context(context: *mut c_void);

    // Device queries
    fn nova_get_device_name(context: *mut c_void) -> *const c_char;
    fn nova_get_memory_info(
        context: *mut c_void,
        total: *mut u64,
        available: *mut u64,
        dedicated: *mut u64,
    );
    fn nova_get_device_id(context: *mut c_void) -> u32;
    fn nova_get_compute_queue_family(context: *mut c_void) -> u32;
    fn nova_get_compute_queue(context: *mut c_void) -> *mut c_void;

    // Buffer management
    fn nova_create_buffer(context: *mut c_void, size: usize, usage: u32) -> *mut c_void;
    fn nova_destroy_buffer(context: *mut c_void, buffer: *mut c_void);
    fn nova_map_buffer(context: *mut c_void, buffer: *mut c_void) -> *mut c_void;
    fn nova_unmap_buffer(context: *mut c_void, buffer: *mut c_void);

    // Shader management
    fn nova_load_shader(context: *mut c_void, path: *const c_char) -> *mut c_void;
    fn nova_create_shader_module(
        context: *mut c_void,
        spirv_code: *const u32,
        code_size: usize,
        stage: u32,
    ) -> *mut c_void;
    fn nova_destroy_shader_module(context: *mut c_void, module: *mut c_void);

    // Pipeline management
    fn nova_create_descriptor_set_layout(
        context: *mut c_void,
        binding_count: u32,
    ) -> *mut c_void;
    fn nova_destroy_descriptor_set_layout(context: *mut c_void, layout: *mut c_void);
    fn nova_create_compute_pipeline(
        context: *mut c_void,
        shader: *mut c_void,
        layout: *mut c_void,
    ) -> *mut c_void;
    fn nova_destroy_pipeline(context: *mut c_void, pipeline: *mut c_void);

    // Descriptor sets
    fn nova_allocate_descriptor_set(
        context: *mut c_void,
        layout: *mut c_void,
    ) -> *mut c_void;
    fn nova_free_descriptor_set(context: *mut c_void, descriptor_set: *mut c_void);
    fn nova_update_descriptor_set(
        context: *mut c_void,
        descriptor_set: *mut c_void,
        binding: u32,
        buffer: *mut c_void,
    );

    // Command management
    fn nova_create_command_pool(context: *mut c_void, queue_family: u32) -> *mut c_void;
    fn nova_destroy_command_pool(context: *mut c_void, pool: *mut c_void);
    fn nova_allocate_command_buffer(context: *mut c_void, pool: *mut c_void) -> *mut c_void;
    fn nova_free_command_buffer(context: *mut c_void, pool: *mut c_void, buffer: *mut c_void);

    // Command recording
    fn nova_cmd_bind_pipeline(cmd_buffer: *mut c_void, pipeline: *mut c_void);
    fn nova_cmd_bind_descriptor_set(cmd_buffer: *mut c_void, pipeline: *mut c_void, descriptor_set: *mut c_void);
    fn nova_cmd_dispatch(cmd_buffer: *mut c_void, x: u32, y: u32, z: u32);

    // Execution
    fn nova_dispatch_compute(
        context: *mut c_void,
        shader: *mut c_void,
        x: u32,
        y: u32,
        z: u32,
    );
    fn nova_submit_compute(
        context: *mut c_void,
        queue: *mut c_void,
        commands: *mut c_void,
    ) -> i32;
    fn nova_queue_wait_idle(queue: *mut c_void) -> i32;
    fn nova_device_wait_idle(context: *mut c_void);

    // VMA (Vulkan Memory Allocator) functions
    fn nova_vma_allocate_buffer(
        context: *mut c_void,
        size: usize,
        usage_flags: u32,
    ) -> *mut c_void;
    fn nova_vma_free(context: *mut c_void, allocation: *mut c_void);
    fn nova_vma_get_buffer(allocation: *mut c_void) -> *mut c_void;
    fn nova_vma_map(allocation: *mut c_void) -> *mut c_void;
    fn nova_vma_unmap(allocation: *mut c_void);
    fn nova_vma_flush(allocation: *mut c_void, offset: usize, size: usize);
    fn nova_vma_invalidate(allocation: *mut c_void, offset: usize, size: usize);
}