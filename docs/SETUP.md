# Environment Setup Guide

Setup instructions for Intel Meteor Lake AI inference on Arch Linux.

## Prerequisites

- Arch Linux (rolling release)
- Intel Core Ultra (Meteor Lake) CPU with integrated Arc Graphics
- `xe` kernel module loaded (check: `lsmod | grep xe`)
- User in `render` group (check: `groups $USER`)

```bash
# Add user to render group if needed
sudo usermod -aG render $USER
# Logout/login for changes to take effect
```

## 1. Intel GPU Compute Runtime

Required for both OpenVINO and llama.cpp SYCL to access the iGPU.

```bash
sudo pacman -S intel-compute-runtime level-zero-loader
```

Verify:

```bash
# Should see the GPU listed
clinfo -l
```

## 2. OpenVINO Environment (Embedding + LLM)

### Create conda environment

```bash
conda create -n openvino python=3.12 -y
conda activate openvino
```

### Install packages

```bash
pip install openvino optimum[openvino] sentence-transformers openvino-genai
```

### Verify GPU detection

```python
from openvino import Core
core = Core()
print("Devices:", core.available_devices)
for dev in core.available_devices:
    print(f"  {dev}: {core.get_property(dev, 'FULL_DEVICE_NAME')}")
```

Expected output:

```
Devices: ['CPU', 'GPU']
  CPU: Intel(R) Core(TM) Ultra 7 155H
  GPU: Intel(R) Arc(TM) Graphics (iGPU)
```

## 3. llama.cpp SYCL Environment (LLM)

### Install oneAPI

```bash
sudo pacman -S intel-oneapi-dpcpp-cpp intel-oneapi-mkl-sycl intel-oneapi-tbb cmake
```

### Verify SYCL devices

```bash
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

Expected output:

```
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Arc(TM) Graphics ...
[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Core(TM) Ultra 7 155H ...
```

### Build llama.cpp

```bash
source /opt/intel/oneapi/setvars.sh

git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

cmake -B build \
    -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

### Verify SYCL device detection from llama.cpp

```bash
./build/bin/llama-ls-sycl-device
```

Expected output:

```
Found 1 SYCL devices:
|ID|        Device Type|                   Name|Version|compute units|
|--|-------------------|-----------------------|-------|-------------|
| 0| [level_zero:gpu:0]|     Intel Arc Graphics|  12.71|          128|
```

## Troubleshooting

### `sycl-ls` not found after installing oneAPI

Always source the environment first:

```bash
source /opt/intel/oneapi/setvars.sh
```

### OpenVINO only shows CPU, no GPU

Ensure `intel-compute-runtime` and `level-zero-loader` are installed. The `xe` kernel module must be loaded.

### llama.cpp out of device memory

Meteor Lake iGPU shares system RAM. For 8B Q4 models (~5GB), ensure at least 8GB free system memory. Use `-c 8192` to limit context size.

### Model loading slow on first run with OpenVINO

First run exports the model to OpenVINO IR format. Use pre-converted models from HuggingFace (see README) to skip this step.
