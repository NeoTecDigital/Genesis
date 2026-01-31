//! GPU compute pipeline for categorical operations

use std::ffi::c_void;
use std::sync::Arc;
use std::mem;

use crate::gpu::{GpuContext, GpuBuffer, ShaderModule, BufferUsage, GpuError};
use crate::waveform::{WaveformNode, GenesisParams, InstantiationParams};
use crate::category::{Genesis, Instantiation};

/// GPU compute pipeline for Genesis operations
pub struct GpuPipeline {
    context: Arc<GpuContext>,

    // Compute pipelines
    genesis_pipeline: *mut c_void,
    instantiate_pipeline: *mut c_void,
    fft_forward_pipeline: *mut c_void,
    fft_inverse_pipeline: *mut c_void,

    // Descriptor sets for binding resources
    genesis_descriptor_set: *mut c_void,
    instantiate_descriptor_set: *mut c_void,
    fft_descriptor_set: *mut c_void,

    // Pipeline layouts
    genesis_layout: *mut c_void,
    instantiate_layout: *mut c_void,
    fft_layout: *mut c_void,
}

impl GpuPipeline {
    /// Create a new GPU pipeline with pre-compiled shaders
    pub fn new(context: Arc<GpuContext>) -> Result<Self, GpuError> {
        // Load compiled shaders (using new standardized names)
        let genesis_shader = ShaderModule::load_spirv(
            &context,
            std::path::Path::new("shaders/gamma_genesis.spv"),
            crate::gpu::ShaderStage::Compute,
            "main",
        )?;

        let instantiate_shader = ShaderModule::load_spirv(
            &context,
            std::path::Path::new("shaders/iota_instantiation.spv"),
            crate::gpu::ShaderStage::Compute,
            "main",
        )?;

        let fft_shader = ShaderModule::load_spirv(
            &context,
            std::path::Path::new("shaders/fft.spv"),
            crate::gpu::ShaderStage::Compute,
            "main",
        )?;

        unsafe {
            // Create descriptor set layouts
            let genesis_layout = nova_create_descriptor_set_layout(
                context.nova_handle(),
                2, // 2 bindings: params + output
            );

            let instantiate_layout = nova_create_descriptor_set_layout(
                context.nova_handle(),
                3, // 3 bindings: params + input + output
            );

            let fft_layout = nova_create_descriptor_set_layout(
                context.nova_handle(),
                3, // 3 bindings: params + input + output
            );

            // Create compute pipelines
            let genesis_pipeline = nova_create_compute_pipeline(
                context.nova_handle(),
                genesis_shader.handle(),
                genesis_layout,
            );

            let instantiate_pipeline = nova_create_compute_pipeline(
                context.nova_handle(),
                instantiate_shader.handle(),
                instantiate_layout,
            );

            let fft_forward_pipeline = nova_create_compute_pipeline(
                context.nova_handle(),
                fft_shader.handle(),
                fft_layout,
            );

            let fft_inverse_pipeline = nova_create_compute_pipeline(
                context.nova_handle(),
                fft_shader.handle(),
                fft_layout,
            );

            // Allocate descriptor sets
            let genesis_descriptor = nova_allocate_descriptor_set(
                context.nova_handle(),
                genesis_layout,
            );

            let instantiate_descriptor = nova_allocate_descriptor_set(
                context.nova_handle(),
                instantiate_layout,
            );

            let fft_descriptor = nova_allocate_descriptor_set(
                context.nova_handle(),
                fft_layout,
            );

            Ok(GpuPipeline {
                context,
                genesis_pipeline,
                instantiate_pipeline,
                fft_forward_pipeline,
                fft_inverse_pipeline,
                genesis_descriptor_set: genesis_descriptor,
                instantiate_descriptor_set: instantiate_descriptor,
                fft_descriptor_set: fft_descriptor,
                genesis_layout,
                instantiate_layout,
                fft_layout,
            })
        }
    }

    /// Execute genesis morphism on GPU (γ: ∅ → 𝟙)
    pub fn execute_genesis(
        &mut self,
        params: &GenesisParams,
        output: &mut GpuBuffer<WaveformNode>,
    ) -> Result<(), GpuError> {
        // Create parameter buffer
        let params_buffer: GpuBuffer<GenesisParams> = GpuBuffer::new(&self.context, BufferUsage::Uniform, 1)?;

        // Upload parameters to GPU
        let params_slice = std::slice::from_ref(params);
        params_buffer.upload(params_slice)?;

        unsafe {
            // Update descriptor set with buffers
            nova_update_descriptor_set(
                self.context.nova_handle(),
                self.genesis_descriptor_set,
                0, // binding 0: params
                params_buffer.handle(),
            );

            nova_update_descriptor_set(
                self.context.nova_handle(),
                self.genesis_descriptor_set,
                1, // binding 1: output
                output.handle(),
            );

            // Record compute commands
            let cmd_buffer = self.context.create_command_buffer()?;

            // Begin command buffer recording
            if nova_cmd_begin(cmd_buffer) != 0 {
                return Err(GpuError::ExecutionError(
                    "Failed to begin command buffer".into()
                ));
            }

            nova_cmd_bind_pipeline(
                cmd_buffer,
                self.genesis_pipeline,
            );

            nova_cmd_bind_descriptor_set(
                cmd_buffer,
                self.genesis_pipeline,
                self.genesis_descriptor_set,
            );

            // Dispatch with appropriate workgroup size
            let workgroup_size = 256;
            let num_elements = output.count();
            let num_workgroups = (num_elements + workgroup_size - 1) / workgroup_size;

            nova_cmd_dispatch(
                cmd_buffer,
                num_workgroups as u32,
                1,
                1,
            );

            // End command buffer recording
            if nova_cmd_end(cmd_buffer) != 0 {
                return Err(GpuError::ExecutionError(
                    "Failed to end command buffer".into()
                ));
            }

            // Submit compute commands to GPU
            self.context.submit_compute(cmd_buffer)?;
            
            // Wait for completion
            self.context.wait_idle()?;
        }

        Ok(())
    }

    /// Execute instantiation morphism on GPU (ι_n: 𝟙 → n)
    pub fn execute_instantiation(
        &mut self,
        params: &InstantiationParams,
        input: &GpuBuffer<WaveformNode>,
        output: &mut GpuBuffer<WaveformNode>,
    ) -> Result<(), GpuError> {
        // Create parameter buffer
        let mut params_buffer: GpuBuffer<InstantiationParams> = GpuBuffer::new(&self.context, BufferUsage::Uniform, 1)?;

        // Upload parameters to GPU
        let params_slice = std::slice::from_ref(params);
        params_buffer.upload(params_slice)?;

        unsafe {
            // Update descriptor set
            nova_update_descriptor_set(
                self.context.nova_handle(),
                self.instantiate_descriptor_set,
                0, // binding 0: params
                params_buffer.handle(),
            );

            nova_update_descriptor_set(
                self.context.nova_handle(),
                self.instantiate_descriptor_set,
                1, // binding 1: input
                input.handle(),
            );

            nova_update_descriptor_set(
                self.context.nova_handle(),
                self.instantiate_descriptor_set,
                2, // binding 2: output
                output.handle(),
            );

            // Record commands
            let cmd_buffer = self.context.create_command_buffer()?;

            // Begin command buffer recording
            if nova_cmd_begin(cmd_buffer) != 0 {
                return Err(GpuError::ExecutionError(
                    "Failed to begin command buffer".into()
                ));
            }

            nova_cmd_bind_pipeline(
                cmd_buffer,
                self.instantiate_pipeline,
            );

            nova_cmd_bind_descriptor_set(
                cmd_buffer,
                self.instantiate_pipeline,
                self.instantiate_descriptor_set,
            );

            // Dispatch
            let workgroup_size = 256;
            let num_elements = output.count();
            let num_workgroups = (num_elements + workgroup_size - 1) / workgroup_size;

            nova_cmd_dispatch(
                cmd_buffer,
                num_workgroups as u32,
                1,
                1,
            );

            // End command buffer recording
            if nova_cmd_end(cmd_buffer) != 0 {
                return Err(GpuError::ExecutionError(
                    "Failed to end command buffer".into()
                ));
            }

            // Submit compute commands to GPU
            self.context.submit_compute(cmd_buffer)?;
            
            // Wait for completion
            self.context.wait_idle()?;
        }

        Ok(())
    }

    /// Execute FFT on GPU (forward or inverse)
    pub fn execute_fft(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        inverse: bool,
    ) -> Result<(), GpuError> {
        let size = input.count() / 2; // Complex numbers (real, imag pairs)

        // FFT parameters
        #[repr(C)]
        struct FFTParams {
            size: u32,
            log2_size: u32,
            direction: u32,
            normalization: f32,
        }

        let log2_size = (size as f32).log2() as u32;
        let params = FFTParams {
            size: size as u32,
            log2_size,
            direction: if inverse { 1 } else { 0 },
            normalization: 1.0 / (size as f32).sqrt(),
        };

        let mut params_buffer: GpuBuffer<FFTParams> = GpuBuffer::new(&self.context, BufferUsage::Uniform, 1)?;

        // Upload parameters to GPU
        let params_slice = std::slice::from_ref(&params);
        params_buffer.upload(params_slice)?;

        unsafe {
            // Update descriptor set
            nova_update_descriptor_set(
                self.context.nova_handle(),
                self.fft_descriptor_set,
                0, // binding 0: params
                params_buffer.handle(),
            );

            nova_update_descriptor_set(
                self.context.nova_handle(),
                self.fft_descriptor_set,
                1, // binding 1: input
                input.handle(),
            );

            nova_update_descriptor_set(
                self.context.nova_handle(),
                self.fft_descriptor_set,
                2, // binding 2: output
                output.handle(),
            );

            // Record commands
            let cmd_buffer = self.context.create_command_buffer()?;

            // Begin command buffer recording
            if nova_cmd_begin(cmd_buffer) != 0 {
                return Err(GpuError::ExecutionError(
                    "Failed to begin command buffer".into()
                ));
            }

            let pipeline = if inverse {
                self.fft_inverse_pipeline
            } else {
                self.fft_forward_pipeline
            };

            nova_cmd_bind_pipeline(cmd_buffer, pipeline);
            nova_cmd_bind_descriptor_set(cmd_buffer, pipeline, self.fft_descriptor_set);

            // Dispatch
            let workgroup_size = 256;
            let num_workgroups = (size + workgroup_size - 1) / workgroup_size;

            nova_cmd_dispatch(
                cmd_buffer,
                num_workgroups as u32,
                1,
                1,
            );

            // End command buffer recording
            if nova_cmd_end(cmd_buffer) != 0 {
                return Err(GpuError::ExecutionError(
                    "Failed to end command buffer".into()
                ));
            }

            // Submit compute commands to GPU
            self.context.submit_compute(cmd_buffer)?;
            
            // Wait for completion
            self.context.wait_idle()?;
        }

        Ok(())
    }
}

impl Drop for GpuPipeline {
    fn drop(&mut self) {
        unsafe {
            if !self.genesis_pipeline.is_null() {
                nova_destroy_pipeline(self.context.nova_handle(), self.genesis_pipeline);
            }
            if !self.instantiate_pipeline.is_null() {
                nova_destroy_pipeline(self.context.nova_handle(), self.instantiate_pipeline);
            }
            if !self.fft_forward_pipeline.is_null() {
                nova_destroy_pipeline(self.context.nova_handle(), self.fft_forward_pipeline);
            }
            if !self.fft_inverse_pipeline.is_null() {
                nova_destroy_pipeline(self.context.nova_handle(), self.fft_inverse_pipeline);
            }

            // Clean up descriptor sets and layouts
            if !self.genesis_descriptor_set.is_null() {
                nova_free_descriptor_set(self.context.nova_handle(), self.genesis_descriptor_set);
            }
            if !self.instantiate_descriptor_set.is_null() {
                nova_free_descriptor_set(self.context.nova_handle(), self.instantiate_descriptor_set);
            }
            if !self.fft_descriptor_set.is_null() {
                nova_free_descriptor_set(self.context.nova_handle(), self.fft_descriptor_set);
            }

            if !self.genesis_layout.is_null() {
                nova_destroy_descriptor_set_layout(self.context.nova_handle(), self.genesis_layout);
            }
            if !self.instantiate_layout.is_null() {
                nova_destroy_descriptor_set_layout(self.context.nova_handle(), self.instantiate_layout);
            }
            if !self.fft_layout.is_null() {
                nova_destroy_descriptor_set_layout(self.context.nova_handle(), self.fft_layout);
            }
        }
    }
}

// Nova pipeline FFI bindings
extern "C" {
    fn nova_create_descriptor_set_layout(
        context: *mut c_void,
        binding_count: u32,
    ) -> *mut c_void;

    fn nova_destroy_descriptor_set_layout(
        context: *mut c_void,
        layout: *mut c_void,
    );

    fn nova_create_compute_pipeline(
        context: *mut c_void,
        shader: *mut c_void,
        layout: *mut c_void,
    ) -> *mut c_void;

    fn nova_destroy_pipeline(
        context: *mut c_void,
        pipeline: *mut c_void,
    );

    fn nova_allocate_descriptor_set(
        context: *mut c_void,
        layout: *mut c_void,
    ) -> *mut c_void;

    fn nova_free_descriptor_set(
        context: *mut c_void,
        descriptor_set: *mut c_void,
    );

    fn nova_update_descriptor_set(
        context: *mut c_void,
        descriptor_set: *mut c_void,
        binding: u32,
        buffer: *mut c_void,
    );

    fn nova_cmd_bind_pipeline(
        cmd_buffer: *mut c_void,
        pipeline: *mut c_void,
    );

    fn nova_cmd_bind_descriptor_set(
        cmd_buffer: *mut c_void,
        descriptor_set: *mut c_void,
    );

    fn nova_cmd_dispatch(
        cmd_buffer: *mut c_void,
        x: u32,
        y: u32,
        z: u32,
    );

    fn nova_cmd_begin(cmd_buffer: *mut c_void) -> i32;
    fn nova_cmd_end(cmd_buffer: *mut c_void) -> i32;
}