# AI Inference on Intel Meteor Lake iGPU

Benchmarks and setup scripts for running AI inference workloads on Intel Meteor Lake integrated Arc Graphics — covering embedding models, rerankers, and LLM generation using OpenVINO and llama.cpp SYCL.

## System Overview

| Component | Specification |
|-----------|--------------|
| Laptop | ThinkPad X1 Carbon Gen 12 (21KC0005US) |
| CPU | Intel Core Ultra 7 155H (6P + 8E + 2LPE, 22 threads) |
| GPU | Intel Arc Graphics (Meteor Lake, 128 EU) |
| Memory | 32 GB DDR5 (shared CPU/GPU) |
| OS | Arch Linux (rolling) |
| Kernel | 7.0.11-arch1-1 |
| GPU Driver | xe (kernel module) |
| Compute Runtime | intel-compute-runtime 26.18.38308.1 |
| OpenVINO | 2026.2.0 |
| oneAPI | 2026.0.0 |

## Benchmark Summary

### Embedding: BGE-M3 (568M params)

| Configuration | Single (samples/s) | Batch 16 (samples/s) |
|---------------|--------------------:|---------------------:|
| FP32 CPU | 23.5 | 27.0 |
| FP32 GPU | 41.1 | 179.2 |
| **INT8 CPU** | **82.9** | **128.3** |
| **INT8 GPU** | 67.6 | **245.4** |

**Best**: INT8 GPU batch (245 samples/s) for throughput, INT8 CPU (83 samples/s, 12ms) for latency.

### Reranker: BGE Reranker v2 M3 (568M params)

| Configuration | Single (pairs/s) | Batch 16 (pairs/s) |
|---------------|------------------:|-------------------:|
| FP16 CPU | 6.9 | 6.4 |
| **FP16 GPU** | **27.4** | 41.8 |
| INT8 CPU | 16.6 | 19.2 |
| **INT8 GPU** | **33.0** | **43.5** |

**Best**: INT8 GPU (33 pairs/s single, 44 pairs/s batch). GPU dominates for reranking — 4-5x faster than CPU.

### LLM Generation: Qwen3 8B

Comparison using each framework's native model format:

| Backend | Format | Quantization | Prompt (tok/s) | Generation (tok/s) |
|---------|--------|-------------|---------------:|-------------------:|
| llama.cpp SYCL GPU | GGUF | Q4_K_M | 70.2 | 6.9 |
| llama.cpp SYCL CPU | GGUF | Q4_K_M | 88.5 | 3.9 |
| llama.cpp OpenVINO CPU | GGUF | Q4_K_M | 34.5 | 5.3 |
| llama.cpp OpenVINO GPU | GGUF | Q4_K_M | ❌ OOM | ❌ |
| **OpenVINO GenAI CPU** | **OV IR** | **INT4** | — | **8.5** |
| OpenVINO GenAI GPU | OV IR | INT4 | — | 7.2 |

**Best**: OpenVINO GenAI CPU with native INT4 IR format (8.5 tok/s).

## Key Findings

1. **Embedding/Reranker → OpenVINO INT8 GPU** is the clear winner. Batch processing at 245 samples/s (embedding) and 44 pairs/s (reranker) makes it viable for production RAG pipelines on a laptop.

2. **LLM generation → OpenVINO GenAI CPU** beats all other options including llama.cpp SYCL. The iGPU doesn't help for autoregressive generation because CPU and GPU share the same memory bandwidth — CPU's larger L3 cache and VNNI instructions give it an edge.

3. **INT8 quantization** provides 2-3x speedup on CPU thanks to Intel VNNI instructions, with negligible quality loss for embedding/reranker models.

4. **llama.cpp OpenVINO backend is immature** — GPU crashes with OOM on 8B models, CPU performance is worse than SYCL. The SYCL backend is the better llama.cpp option on Intel GPUs today.

5. **Native format matters for LLMs** — OpenVINO GenAI with its own INT4 IR format (8.5 tok/s) outperforms llama.cpp SYCL with GGUF Q4_K_M (6.9 tok/s GPU, 3.9 tok/s CPU). Each framework performs best with its own optimized format.

6. **Reranker is GPU-bound** — unlike LLM generation (memory-bandwidth bound), cross-encoder reranking benefits massively from GPU parallelism, even on a shared-memory iGPU.

## Quick Start

### OpenVINO (Embedding + Reranker + LLM)

```bash
# 1. Install Intel GPU compute runtime
sudo pacman -S intel-compute-runtime level-zero-loader

# 2. Create conda environment
conda create -n openvino python=3.12 -y
conda activate openvino
pip install openvino optimum[openvino] sentence-transformers openvino-genai

# 3. Verify GPU detection
python -c "from openvino import Core; c = Core(); print(c.available_devices)"
# Expected: ['CPU', 'GPU']

# 4. Run benchmarks
python scripts/bench-embedding.py
python scripts/bench-reranker.py
python scripts/bench-llm-openvino.py
```

### llama.cpp SYCL (LLM)

```bash
# 1. Install oneAPI
sudo pacman -S intel-oneapi-dpcpp-cpp intel-oneapi-mkl-sycl intel-oneapi-tbb

# 2. Build llama.cpp
source /opt/intel/oneapi/setvars.sh
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# 3. Run benchmark
./build/bin/llama-bench -m <model.gguf> -ngl 99 -sm none -mg 0
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/SETUP.md](docs/SETUP.md) | Detailed environment setup for Arch Linux |
| [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) | Full benchmark results with analysis |
| [results/](results/) | Raw benchmark outputs per machine |

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/setup-openvino.sh](scripts/setup-openvino.sh) | Install GPU runtime + OpenVINO conda env |
| [scripts/setup-llamacpp-sycl.sh](scripts/setup-llamacpp-sycl.sh) | Install oneAPI + build llama.cpp with SYCL |
| [scripts/bench-embedding.py](scripts/bench-embedding.py) | Embedding model benchmark (OpenVINO) |
| [scripts/bench-reranker.py](scripts/bench-reranker.py) | Reranker model benchmark (OpenVINO) |
| [scripts/bench-llm-openvino.py](scripts/bench-llm-openvino.py) | LLM benchmark (OpenVINO GenAI) |
| [scripts/bench-llm-llamacpp.sh](scripts/bench-llm-llamacpp.sh) | LLM benchmark (llama.cpp SYCL) |

## Pre-converted OpenVINO Models

| Model | HuggingFace Repo | Use Case |
|-------|-----------------|----------|
| BGE-M3 INT8 | [stellars/bge-m3-openvino-int8](https://huggingface.co/stellars/bge-m3-openvino-int8) | Embedding |
| BGE Reranker v2 M3 FP16 | [J-Parker/bge-reranker-v2-m3-openvino](https://huggingface.co/J-Parker/bge-reranker-v2-m3-openvino) | Reranking |
| BGE Reranker v2 M3 INT8 | [stellars/bge-reranker-v2-m3-openvino-int8](https://huggingface.co/stellars/bge-reranker-v2-m3-openvino-int8) | Reranking |
| Qwen3 Embedding 0.6B INT8 | [OpenVINO/Qwen3-Embedding-0.6B-int8-ov](https://huggingface.co/OpenVINO/Qwen3-Embedding-0.6B-int8-ov) | Embedding |
| Qwen3 8B INT4 | [OpenVINO/Qwen3-8B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-8B-int4-ov) | LLM |

## License

MIT
