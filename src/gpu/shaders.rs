//! Shader module loading and compilation

use std::ffi::{CString, c_void};
use std::path::Path;
use std::fs;

use crate::gpu::{GpuContext, GpuError};

/// Shader stage type
#[derive(Debug, Clone, Copy)]
pub enum ShaderStage {
    Compute,
    Vertex,
    Fragment,
}

/// Compiled shader module
pub struct ShaderModule {
    /// Vulkan shader module handle
    module: *mut c_void,
    /// Shader stage
    stage: ShaderStage,
    /// Entry point name
    entry_point: CString,
    /// Nova context reference
    context: *mut c_void,
}

impl ShaderModule {
    /// Load a compiled SPIR-V shader from file
    pub fn load_spirv(
        context: &GpuContext,
        path: &Path,
        stage: ShaderStage,
        entry_point: &str,
    ) -> Result<Self, GpuError> {
        // Read SPIR-V binary
        let spirv_data = fs::read(path).map_err(|e| {
            GpuError::ShaderCompilationError(
                format!("Failed to read shader file {:?}: {}", path, e)
            )
        })?;

        Self::from_spirv(context, &spirv_data, stage, entry_point)
    }

    /// Create shader module from SPIR-V bytes
    pub fn from_spirv(
        context: &GpuContext,
        spirv_data: &[u8],
        stage: ShaderStage,
        entry_point: &str,
    ) -> Result<Self, GpuError> {
        // Validate SPIR-V magic number
        if spirv_data.len() < 4 {
            return Err(GpuError::ShaderCompilationError(
                "Invalid SPIR-V data: too short".into()
            ));
        }

        let magic = u32::from_le_bytes([
            spirv_data[0], spirv_data[1], spirv_data[2], spirv_data[3]
        ]);

        // SPIR-V magic number: 0x07230203
        if magic != 0x07230203 && magic != 0x03022307 {
            return Err(GpuError::ShaderCompilationError(
                format!("Invalid SPIR-V magic number: 0x{:08x}", magic)
            ));
        }

        let entry_cstring = CString::new(entry_point).map_err(|e| {
            GpuError::ShaderCompilationError(
                format!("Invalid entry point name: {}", e)
            )
        })?;

        unsafe {
            let stage_flag = match stage {
                ShaderStage::Compute => VK_SHADER_STAGE_COMPUTE,
                ShaderStage::Vertex => VK_SHADER_STAGE_VERTEX,
                ShaderStage::Fragment => VK_SHADER_STAGE_FRAGMENT,
            };

            let module = nova_create_shader_module(
                context.nova_handle(),
                spirv_data.as_ptr() as *const u32,
                spirv_data.len(),
                stage_flag,
            );

            if module.is_null() {
                return Err(GpuError::ShaderCompilationError(
                    "Failed to create shader module".into()
                ));
            }

            Ok(ShaderModule {
                module,
                stage,
                entry_point: entry_cstring,
                context: context.nova_handle(),
            })
        }
    }

    /// Compile GLSL to SPIR-V using shaderc
    pub fn compile_glsl(
        context: &GpuContext,
        glsl_source: &str,
        stage: ShaderStage,
        entry_point: &str,
        include_dirs: &[&Path],
    ) -> Result<Self, GpuError> {
        let stage_flag = match stage {
            ShaderStage::Compute => "comp",
            ShaderStage::Vertex => "vert",
            ShaderStage::Fragment => "frag",
        };

        // Use shaderc-rs for compilation (would need to add dependency)
        // For now, assume pre-compiled SPIR-V
        Err(GpuError::ShaderCompilationError(
            "GLSL compilation not implemented - use pre-compiled SPIR-V".into()
        ))
    }

    /// Get shader module handle
    pub fn handle(&self) -> *mut c_void {
        self.module
    }

    /// Get entry point name
    pub fn entry_point(&self) -> &CString {
        &self.entry_point
    }

    /// Get shader stage
    pub fn stage(&self) -> ShaderStage {
        self.stage
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                nova_destroy_shader_module(self.context, self.module);
            }
        }
    }
}

// Vulkan shader stage flags
const VK_SHADER_STAGE_VERTEX: u32 = 0x00000001;
const VK_SHADER_STAGE_FRAGMENT: u32 = 0x00000010;
const VK_SHADER_STAGE_COMPUTE: u32 = 0x00000020;

// Nova shader FFI bindings
extern "C" {
    fn nova_create_shader_module(
        context: *mut c_void,
        spirv_code: *const u32,
        code_size: usize,
        stage: u32,
    ) -> *mut c_void;

    fn nova_destroy_shader_module(context: *mut c_void, module: *mut c_void);
}