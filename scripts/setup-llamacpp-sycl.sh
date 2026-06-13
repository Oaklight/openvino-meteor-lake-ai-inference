#!/bin/bash
# Build llama.cpp with SYCL backend for Intel GPU
# Tested on: Arch Linux, Intel Core Ultra 7 155H

set -euo pipefail

INSTALL_DIR="${1:-$HOME/llama.cpp-sycl}"

echo "=== Installing oneAPI packages ==="
sudo pacman -S --noconfirm --needed \
	intel-oneapi-dpcpp-cpp \
	intel-oneapi-mkl-sycl \
	intel-oneapi-tbb \
	cmake

echo ""
echo "=== Enabling oneAPI environment ==="
source /opt/intel/oneapi/setvars.sh

echo ""
echo "=== Verifying SYCL devices ==="
sycl-ls

echo ""
echo "=== Cloning llama.cpp ==="
if [ -d "$INSTALL_DIR" ]; then
	echo "Directory $INSTALL_DIR exists, pulling latest..."
	cd "$INSTALL_DIR"
	git pull
else
	git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$INSTALL_DIR"
	cd "$INSTALL_DIR"
fi

echo ""
echo "=== Building with SYCL backend ==="
cmake -B build \
	-DGGML_SYCL=ON \
	-DCMAKE_C_COMPILER=icx \
	-DCMAKE_CXX_COMPILER=icpx \
	-DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j"$(nproc)"

echo ""
echo "=== Verifying SYCL device detection ==="
./build/bin/llama-ls-sycl-device

echo ""
echo "=== Build complete ==="
echo "Binary at: $INSTALL_DIR/build/bin/"
echo ""
echo "Usage:"
echo "  source /opt/intel/oneapi/setvars.sh"
echo "  $INSTALL_DIR/build/bin/llama-bench -m <model.gguf> -ngl 99 -sm none -mg 0"
