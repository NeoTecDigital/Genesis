//! Build script for Genesis
//!
//! Compiles GLSL shaders to SPIR-V and configures Nova library linking

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shaders/");
    // Legacy batch_pipeline.cpp removed - using Nova library directly
    // println!("cargo:rerun-if-changed=src/gpu/batch_pipeline.cpp");

    // Only compile shaders if GPU feature is enabled
    if env::var("CARGO_FEATURE_GPU").is_ok() {
        compile_shaders();
        // Legacy C++ pipeline compilation removed - using Nova directly
        // compile_cpp_pipeline();
        link_nova_library();
    }
}

// Legacy function - kept for reference but no longer used
// The batch_pipeline.cpp has been replaced with direct Nova library usage
#[allow(dead_code)]
fn compile_cpp_pipeline() {
    // This function compiled the legacy batch_pipeline.cpp
    // Now we use Nova library directly for GPU operations
    // Keeping this commented for historical reference

    // let nova_path = PathBuf::from("../Nova");
    // if !nova_path.exists() {
    //     println!("cargo:warning=Nova library not found, cannot compile C++ pipeline.");
    //     return;
    // }
    //
    // cc::Build::new()
    //     .file("src/gpu/batch_pipeline.cpp")
    //     .cpp(true)
    //     .flag("-std=c++17")
    //     .include(&nova_path)
    //     .compile("genesis_batch_pipeline");
    //
    // println!("cargo:rustc-link-lib=static=genesis_batch_pipeline");
}

fn compile_shaders() {
    let shader_dir = PathBuf::from("shaders");
    let _out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Shaders to compile
    let shaders = [
        ("genesis.comp", "genesis.spv"),
        ("instantiate.comp", "instantiate.spv"),
        ("fft.comp", "fft.spv"),
    ];

    // Check for glslc compiler (from Vulkan SDK)
    let glslc = which_glslc();

    if glslc.is_none() {
        println!("cargo:warning=glslc not found - shaders will not be compiled");
        println!("cargo:warning=Install Vulkan SDK or ensure glslc is in PATH");
        println!("cargo:warning=Pre-compiled SPIR-V files will be used if available");
        return;
    }

    let glslc = glslc.unwrap();

    for (source, output) in &shaders {
        let input_path = shader_dir.join(source);
        let output_path = shader_dir.join(output);

        println!("cargo:rerun-if-changed={}", input_path.display());

        // Compile GLSL to SPIR-V
        let status = Command::new(&glslc)
            .arg(&input_path)
            .arg("-o")
            .arg(&output_path)
            .arg("--target-env=vulkan1.2")
            .arg("-O") // Optimize
            .status();

        match status {
            Ok(status) => {
                if !status.success() {
                    panic!("Failed to compile shader: {}", source);
                }
                println!("Compiled shader: {} -> {}", source, output);
            }
            Err(e) => {
                panic!("Failed to run glslc: {}", e);
            }
        }
    }
}

fn link_nova_library() {
    // Nova library location (relative to Genesis project)
    let nova_path = PathBuf::from("../Nova");

    // Check if Nova exists
    if !nova_path.exists() {
        println!("cargo:warning=Nova library not found at {:?}", nova_path);
        println!("cargo:warning=GPU features will not link correctly");
        return;
    }

    // Add Nova build directory to library search path
    let nova_build = nova_path.join("build");
    if nova_build.exists() {
        println!("cargo:rustc-link-search=native={}", nova_build.display());
    }

    // Link against Nova compute library
    // Note: Actual library name depends on Nova's build output
    println!("cargo:rustc-link-lib=static=nova_compute");

    // Link against Vulkan
    println!("cargo:rustc-link-lib=vulkan");

    // Platform-specific linking
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=pthread");
    }
}

fn which_glslc() -> Option<PathBuf> {
    // Try to find glslc in PATH
    if let Ok(output) = Command::new("which").arg("glslc").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout);
            return Some(PathBuf::from(path.trim()));
        }
    }

    // Try common Vulkan SDK locations
    let common_paths = [
        "/usr/bin/glslc",
        "/usr/local/bin/glslc",
        "$VULKAN_SDK/bin/glslc",
        "$HOME/.vulkan_sdk/bin/glslc",
    ];

    for path_str in &common_paths {
        let path = PathBuf::from(shellexpand::tilde(path_str).to_string());
        if path.exists() {
            return Some(path);
        }
    }

    None
}

// Simple tilde expansion (minimal implementation)
mod shellexpand {
    use std::env;

    pub fn tilde(path: &str) -> String {
        if path.starts_with("$HOME") {
            if let Ok(home) = env::var("HOME") {
                return path.replace("$HOME", &home);
            }
        }
        if path.starts_with("$VULKAN_SDK") {
            if let Ok(sdk) = env::var("VULKAN_SDK") {
                return path.replace("$VULKAN_SDK", &sdk);
            }
        }
        path.to_string()
    }
}