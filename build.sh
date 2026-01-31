#!/usr/bin/env bash
#
# Unified build script for the Genesis project.
#
# This script orchestrates the entire build process, including:
# 1. Initializing and updating git submodules.
# 2. Compiling the Rust core library and C++ components.
# 3. Compiling shaders.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Git Submodules ---
echo "Initializing and updating git submodules..."
git submodule update --init --recursive

# --- Rust Build ---
echo "Building Rust core and C++ pipeline..."
cargo build --release

# --- Shader Compilation ---
# The shader compilation is handled by the build.rs script, 
# so we don't need to call compile_shaders.sh explicitly.
# If we wanted to decouple it, we could call it here.
# ./compile_shaders.sh

echo "Build complete."
echo "The final library can be found at target/release/libgenesis.so"
