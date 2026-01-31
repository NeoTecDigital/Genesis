//! GPU buffer management using Nova's VMA (Vulkan Memory Allocator)

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem;
use std::slice;

use crate::gpu::{GpuContext, GpuError};
use crate::waveform::WaveformNode;

/// Buffer usage flags
#[derive(Debug, Clone, Copy)]
pub enum BufferUsage {
    /// Buffer can be used as uniform buffer
    Uniform,
    /// Buffer can be used as storage buffer
    Storage,
    /// Buffer can be used as staging for CPU-GPU transfers
    Staging,
    /// Buffer used for image data
    Image,
}

/// GPU buffer with VMA allocation
pub struct GpuBuffer<T> {
    /// VMA allocation handle
    allocation: *mut c_void,
    /// Vulkan buffer handle
    buffer: *mut c_void,
    /// CPU-accessible staging buffer (if applicable)
    staging_data: Option<Vec<T>>,
    /// Size in bytes
    size_bytes: usize,
    /// Number of elements
    count: usize,
    /// Usage flags
    usage: BufferUsage,
    /// Tracks if staging buffer has uncommitted changes
    dirty: bool,
    /// Nova context reference
    context: *mut c_void,
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

impl<T: Copy + Clone> GpuBuffer<T> {
    /// Create a new GPU buffer
    pub fn new(
        context: &GpuContext,
        usage: BufferUsage,
        count: usize,
    ) -> Result<Self, GpuError> {
        let size_bytes = count * mem::size_of::<T>();

        // Determine VMA allocation flags based on usage
        let vma_flags = match usage {
            BufferUsage::Uniform => VMA_USAGE_GPU_ONLY | VMA_BUFFER_USAGE_UNIFORM,
            BufferUsage::Storage => VMA_USAGE_GPU_ONLY | VMA_BUFFER_USAGE_STORAGE,
            BufferUsage::Staging => VMA_USAGE_CPU_TO_GPU | VMA_BUFFER_USAGE_TRANSFER_SRC,
            BufferUsage::Image => VMA_USAGE_GPU_ONLY | VMA_BUFFER_USAGE_STORAGE,
        };

        unsafe {
            // Allocate buffer through Nova's VMA wrapper
            let allocation = nova_vma_allocate_buffer(
                context.nova_handle(),
                size_bytes,
                vma_flags,
            );

            if allocation.is_null() {
                return Err(GpuError::BufferAllocationError(
                    format!("Failed to allocate {} bytes", size_bytes)
                ));
            }

            // Get Vulkan buffer handle
            let buffer = nova_vma_get_buffer(allocation);
            if buffer.is_null() {
                nova_vma_free(context.nova_handle(), allocation);
                return Err(GpuError::BufferAllocationError(
                    "Failed to get buffer handle".into()
                ));
            }

            // Create staging buffer for CPU-accessible usage
            let staging_data = match usage {
                BufferUsage::Staging => Some(Vec::with_capacity(count)),
                _ => None,
            };

            Ok(GpuBuffer {
                allocation,
                buffer,
                staging_data,
                size_bytes,
                count,
                usage,
                dirty: false,
                context: context.nova_handle(),
                _phantom: PhantomData,
            })
        }
    }

    /// Upload data from staging buffer to GPU
    pub fn upload(&mut self, data: &[T]) -> Result<(), GpuError> {
        if data.len() > self.count {
            return Err(GpuError::TransferError(
                format!("Data size {} exceeds buffer capacity {}", data.len(), self.count)
            ));
        }

        unsafe {
            // Map GPU memory for writing
            let mapped_ptr = nova_vma_map(self.allocation);
            if mapped_ptr.is_null() {
                return Err(GpuError::TransferError(
                    "Failed to map GPU memory".into()
                ));
            }

            // Copy data to mapped memory
            let src_ptr = data.as_ptr() as *const u8;
            let dst_ptr = mapped_ptr as *mut u8;
            let copy_size = data.len() * mem::size_of::<T>();

            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, copy_size);

            // Unmap memory
            nova_vma_unmap(self.allocation);

            // Flush to ensure GPU visibility
            nova_vma_flush(self.allocation, 0, copy_size);
        }

        // Update staging buffer if present
        if let Some(ref mut staging) = self.staging_data {
            staging.clear();
            staging.extend_from_slice(data);
            self.dirty = false;
        }

        Ok(())
    }

    /// Download data from GPU to staging buffer
    pub fn download(&mut self, output: &mut Vec<T>) -> Result<(), GpuError> {
        output.clear();
        output.reserve(self.count);

        unsafe {
            // Map GPU memory for reading
            let mapped_ptr = nova_vma_map(self.allocation);
            if mapped_ptr.is_null() {
                return Err(GpuError::TransferError(
                    "Failed to map GPU memory for reading".into()
                ));
            }

            // Invalidate cache before reading
            nova_vma_invalidate(self.allocation, 0, self.size_bytes);

            // Copy data from mapped memory
            let src_ptr = mapped_ptr as *const T;
            let data_slice = slice::from_raw_parts(src_ptr, self.count);
            output.extend_from_slice(data_slice);

            // Unmap memory
            nova_vma_unmap(self.allocation);
        }

        // Update staging buffer
        if let Some(ref mut staging) = self.staging_data {
            staging.clear();
            staging.extend_from_slice(output);
            self.dirty = false;
        }

        Ok(())
    }

    /// Get Vulkan buffer handle for binding
    pub fn handle(&self) -> *mut c_void {
        self.buffer
    }

    /// Get buffer size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Get element count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Check if staging buffer has uncommitted changes
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Access staging buffer (if available)
    pub fn staging_data(&self) -> Option<&[T]> {
        self.staging_data.as_deref()
    }

    /// Mutable access to staging buffer
    pub fn staging_data_mut(&mut self) -> Option<&mut Vec<T>> {
        self.dirty = true;
        self.staging_data.as_mut()
    }
}

impl<T> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        if !self.allocation.is_null() {
            unsafe {
                nova_vma_free(self.context, self.allocation);
            }
        }
    }
}

// VMA usage flags (matching Nova/Vulkan definitions)
const VMA_USAGE_GPU_ONLY: u32 = 0;
const VMA_USAGE_CPU_TO_GPU: u32 = 1;
const VMA_USAGE_GPU_TO_CPU: u32 = 2;

const VMA_BUFFER_USAGE_UNIFORM: u32 = 0x10;
const VMA_BUFFER_USAGE_STORAGE: u32 = 0x20;
const VMA_BUFFER_USAGE_TRANSFER_SRC: u32 = 0x40;
const VMA_BUFFER_USAGE_TRANSFER_DST: u32 = 0x80;

// Nova VMA FFI bindings
extern "C" {
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