# Detailed Benchmark Results

All benchmarks run on ThinkPad X1 Carbon Gen 12 with Intel Core Ultra 7 155H.

## 1. Embedding: BGE-M3 (568M params)

### FP32 (auto-exported from HuggingFace)

Model: `BAAI/bge-m3`, exported to OpenVINO IR on first load.

| Device | Single Text (samples/s) | Latency (ms) | Batch 16 (samples/s) | Batch Latency (ms) |
|--------|------------------------:|-------------:|---------------------:|--------------------:|
| CPU | 23.5 | 42.6 | 27.0 | 591.8 |
| GPU | 41.1 | 24.3 | 179.2 | 89.3 |

### INT8 (pre-converted from HuggingFace)

Model: [`stellars/bge-m3-openvino-int8`](https://huggingface.co/stellars/bge-m3-openvino-int8)

| Device | Single Text (samples/s) | Latency (ms) | Batch 16 (samples/s) | Batch Latency (ms) |
|--------|------------------------:|-------------:|---------------------:|--------------------:|
| CPU | 82.9 | 12.1 | 128.3 | 124.7 |
| GPU | 67.6 | 14.8 | 245.4 | 65.2 |

### Key Takeaways

- **INT8 on CPU is 3.5x faster than FP32** for single inference — Intel VNNI instructions accelerate INT8 natively.
- **GPU batch processing scales well**: FP32 GPU goes from 41→179 (4.4x), INT8 GPU from 68→245 (3.6x).
- **For latency-sensitive single queries**: INT8 CPU (12ms) beats INT8 GPU (15ms).
- **For batch throughput**: INT8 GPU (245/s) wins decisively.
- Model loading: INT8 pre-converted loads in 8s (GPU) vs 77s for FP32 auto-export.

## 2. LLM: Qwen3 8B

### llama.cpp SYCL Backend

Model: `Qwen3-8B-Q4_K_M.gguf` (4.68 GiB)

| Device | Layers on GPU | Prompt Processing (tok/s) | Generation (tok/s) |
|--------|:-------------:|--------------------------:|-------------------:|
| GPU | 99 (all) | 70.18 ± 31.71 | 6.87 ± 1.77 |
| CPU | 0 | 88.48 ± 6.53 | 3.86 ± 0.81 |

### OpenVINO GenAI

Model: [`OpenVINO/Qwen3-8B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3-8B-int4-ov)

| Device | Generation (tok/s) | Model Load Time (s) |
|--------|-------------------:|--------------------:|
| CPU | ~8.5 | 2.0 |
| GPU | ~7.2 | 18.6 |

### Key Takeaways

- **OpenVINO GenAI CPU is the fastest** for LLM generation (8.5 tok/s), beating both llama.cpp SYCL options.
- **GPU doesn't help for LLM generation** on shared-memory iGPU — the bottleneck is memory bandwidth (102 GB/s), which is shared between CPU and GPU. CPU benefits from L3 cache proximity.
- **llama.cpp SYCL GPU has higher prompt processing** (70 tok/s) but slower generation (6.9 tok/s).
- **For interactive chat**: 8.5 tok/s ≈ typing speed, usable but not fast.

## 3. Comparison: iGPU vs Discrete GPU

For context, here's how Meteor Lake iGPU compares to discrete Intel GPUs:

| Hardware | Memory | Bandwidth | Qwen3 8B Gen (tok/s) |
|----------|--------|-----------|---------------------:|
| Meteor Lake iGPU (128 EU) | 32GB shared | 102 GB/s | 8.5 (OV CPU) |
| Arc A770 (512 EU) | 16GB GDDR6 | 560 GB/s | ~37 (SYCL) |
| Max 1550 (PVC) | 128GB HBM2e | 1600 GB/s | ~73 (SYCL) |

The memory bandwidth gap explains why discrete GPUs are 4-8x faster for LLM inference.

## Methodology

- **Embedding benchmarks**: 100 iterations single, 20 iterations batch (after 5 warmup iterations).
- **llama.cpp benchmarks**: `llama-bench` with `-p 512 -n 128` (512 token prompt, 128 token generation).
- **OpenVINO GenAI benchmarks**: `max_new_tokens=256`, `do_sample=False`, measured wall-clock time.
- All tests run with minimal background load. Laptop on AC power, performance governor.
