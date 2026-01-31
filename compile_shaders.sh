#!/bin/bash
# Compile GLSL shaders to SPIR-V for Genesis GPU operations

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Genesis Shader Compilation${NC}"
echo "================================"

# Check for glslc
if ! command -v glslc &> /dev/null; then
    echo -e "${RED}ERROR: glslc not found${NC}"
    echo "Please install the Vulkan SDK or ensure glslc is in your PATH"
    echo "On Arch Linux: sudo pacman -S vulkan-tools shaderc"
    exit 1
fi

# Shader directory
SHADER_DIR="shaders"

if [ ! -d "$SHADER_DIR" ]; then
    echo -e "${RED}ERROR: Shader directory not found${NC}"
    exit 1
fi

# Compile each shader
compile_shader() {
    local source=$1
    local output=$2

    echo -n "Compiling $source... "

    if glslc \
        "$SHADER_DIR/$source" \
        -o "$SHADER_DIR/$output" \
        --target-env=vulkan1.2 \
        -O \
        -g \
        -Werror 2>&1; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        return 1
    fi
}

# Compile all shaders
echo ""
compile_shader "genesis.comp" "genesis.spv"
compile_shader "instantiate.comp" "instantiate.spv"
compile_shader "fft.comp" "fft.spv"

echo ""
echo -e "${GREEN}Shader compilation complete!${NC}"
echo ""

# Verify SPIR-V files
echo "Verifying SPIR-V files..."
for spv in "$SHADER_DIR"/*.spv; do
    if [ -f "$spv" ]; then
        size=$(stat -c%s "$spv" 2>/dev/null || stat -f%z "$spv" 2>/dev/null)
        echo "  $(basename $spv): ${size} bytes"
    fi
done

echo ""
echo -e "${GREEN}All shaders compiled successfully!${NC}"