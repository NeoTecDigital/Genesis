#!/bin/bash

# Build script for Genesis GPU batch pipeline

set -e

echo "=== Building Genesis GPU Batch Pipeline ==="
echo

# Check if Nova library exists
NOVA_LIB="../../../Nova/build/libnova_compute.so"
if [ ! -f "$NOVA_LIB" ]; then
    echo "Nova library not found. Building Nova first..."
    cd ../../../Nova
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    cd -
    echo "✅ Nova library built"
    echo
fi

# Build batch pipeline
echo "Building batch pipeline..."
make clean
make

echo
echo "✅ Build complete. Run './test_batch_pipeline' to test."