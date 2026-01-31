#!/bin/bash
# Build GPU-optimized batch pipeline

set -e

echo "=== Building Genesis GPU Batch Pipeline ==="

# Configuration
BUILD_DIR="build"
NOVA_DIR="../Nova"
SHADER_DIR="shaders"

# Create build directory
mkdir -p "$BUILD_DIR"

# Compile shaders (if not already done)
echo "Compiling shaders..."
cd "$SHADER_DIR"
for shader in gamma_genesis.comp iota_instantiation.comp tau_reduction.comp epsilon_erasure.comp \
              gamma_revelation.comp iota_abstraction.comp tau_expansion.comp epsilon_preservation.comp; do
    if [ ! -f "${shader%.comp}.spv" ]; then
        echo "  Compiling $shader..."
        glslc "$shader" -o "${shader%.comp}.spv" 2>&1 || glslangValidator -V "$shader" -o "${shader%.comp}.spv" 2>&1
    fi
done
cd ..

# Copy shaders to build directory
echo "Copying shaders to build directory..."
mkdir -p "$BUILD_DIR/shaders"
cp "$SHADER_DIR"/*.spv "$BUILD_DIR/shaders/"

# Compile C++ batch pipeline
echo "Compiling batch pipeline..."
g++ -std=c++17 \
    -O3 \
    -fPIC \
    -shared \
    -o "$BUILD_DIR/libgenesis_gpu.so" \
    src/gpu/batch_pipeline.cpp \
    -I"$NOVA_DIR" \
    -L"$NOVA_DIR/build" \
    -lnova_compute \
    -Wl,-rpath,"$NOVA_DIR/build"

echo "✅ Build complete!"
echo ""
echo "Library: $BUILD_DIR/libgenesis_gpu.so"
echo "Shaders: $BUILD_DIR/shaders/*.spv"
echo ""
echo "Test with: python genesis_gpu.py"
