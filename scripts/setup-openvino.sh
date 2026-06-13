#!/bin/bash
# Setup OpenVINO environment for Intel Meteor Lake AI inference
# Tested on: Arch Linux, Intel Core Ultra 7 155H

set -euo pipefail

echo "=== Installing Intel GPU compute runtime ==="
sudo pacman -S --noconfirm --needed intel-compute-runtime level-zero-loader

echo ""
echo "=== Creating conda environment ==="
conda create -n openvino python=3.12 -y
eval "$(conda shell.bash hook)"
conda activate openvino

echo ""
echo "=== Installing OpenVINO packages ==="
pip install openvino optimum[openvino] sentence-transformers openvino-genai

echo ""
echo "=== Verifying GPU detection ==="
python3 -c "
from openvino import Core
core = Core()
print('Available devices:', core.available_devices)
for dev in core.available_devices:
    print(f'  {dev}: {core.get_property(dev, \"FULL_DEVICE_NAME\")}')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate openvino"
