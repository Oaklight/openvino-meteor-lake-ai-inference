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

## 2. Reranker: BGE Reranker v2 M3 (568M params)

### FP16 (pre-converted from HuggingFace)

Model: [`J-Parker/bge-reranker-v2-m3-openvino`](https://huggingface.co/J-Parker/bge-reranker-v2-m3-openvino)

| Device | Single (pairs/s) | Latency (ms) | Batch 16 (pairs/s) | Batch Latency (ms) |
|--------|------------------:|-------------:|-------------------:|--------------------:|
| CPU | 6.9 | 143.9 | 6.4 | 2509.8 |
| GPU | 27.4 | 36.5 | 41.8 | 382.7 |

### INT8 (pre-converted from HuggingFace)

Model: [`stellars/bge-reranker-v2-m3-openvino-int8`](https://huggingface.co/stellars/bge-reranker-v2-m3-openvino-int8)

| Device | Single (pairs/s) | Latency (ms) | Batch 16 (pairs/s) | Batch Latency (ms) |
|--------|------------------:|-------------:|-------------------:|--------------------:|
| CPU | 16.6 | 60.1 | 19.2 | 833.9 |
| GPU | 33.0 | 30.3 | 43.5 | 367.9 |

### Key Takeaways

- **GPU dominates reranking** — 4-5x faster than CPU across all configurations.
- **INT8 CPU is 2.4x faster** than FP16 CPU (16.6 vs 6.9 pairs/s) thanks to VNNI.
- **GPU INT8 vs FP16 difference is small** (33 vs 27 pairs/s single) — GPU handles FP16 well natively.
- **Batch scaling on GPU is good**: FP16 27→42 (1.5x), INT8 33→44 (1.3x).
- **CPU batch scaling is poor**: FP16 actually gets *worse* in batch (6.9→6.4), INT8 only marginal (16.6→19.2). Cross-encoder pairs are longer sequences that stress CPU memory.
- For a RAG pipeline reranking 20 documents: INT8 GPU takes ~0.6s, FP16 CPU takes ~2.9s.

## 3. LLM: Qwen3 8B

### Framework Comparison: Same GGUF Q4_K_M Model

Model: `Qwen3-8B-Q4_K_M.gguf` (4.68 GiB)

| Backend | Device | Prompt pp512 (tok/s) | Generation tg128 (tok/s) |
|---------|--------|---------------------:|-------------------------:|
| llama.cpp SYCL | GPU | 70.18 ± 31.71 | **6.87 ± 1.77** |
| llama.cpp SYCL | CPU | **88.48 ± 6.53** | 3.86 ± 0.81 |
| llama.cpp OpenVINO | CPU | 34.46 ± 11.08 | 5.29 ± 1.69 |
| llama.cpp OpenVINO | GPU | ❌ OOM crash | ❌ |

### Native Format: OpenVINO GenAI with INT4 IR

Model: [`OpenVINO/Qwen3-8B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3-8B-int4-ov)

| Device | Generation (tok/s) | Model Load Time (s) |
|--------|-------------------:|--------------------:|
| CPU | **~8.5** | 2.0 |
| GPU | ~7.2 | 18.6 |

### Key Takeaways

- **OpenVINO GenAI CPU with native INT4 IR is fastest** (8.5 tok/s) — each framework runs best with its own optimized format.
- **llama.cpp SYCL GPU generation** (6.9 tok/s) is decent but still behind OpenVINO GenAI CPU.
- **llama.cpp OpenVINO backend is immature** — GPU crashes with OOM on 8B models, CPU performance is worse than SYCL backend. Not recommended on Meteor Lake iGPU.
- **Prompt processing**: llama.cpp SYCL CPU leads (88.5 tok/s), but this matters less for interactive chat.
- **GPU doesn't help for LLM generation** on shared-memory iGPU — the bottleneck is memory bandwidth (102 GB/s shared between CPU and GPU). CPU benefits from larger L3 cache.
- **For interactive chat**: 8.5 tok/s ≈ typing speed, usable but not fast.

## 4. Comparison: iGPU vs Discrete GPU

For context, here's how Meteor Lake iGPU compares to discrete Intel GPUs:

| Hardware | Memory | Bandwidth | Qwen3 8B Gen (tok/s) |
|----------|--------|-----------|---------------------:|
| Meteor Lake iGPU (128 EU) | 32GB shared | 102 GB/s | 8.5 (OV CPU) |
| Arc A770 (512 EU) | 16GB GDDR6 | 560 GB/s | ~37 (SYCL) |
| Max 1550 (PVC) | 128GB HBM2e | 1600 GB/s | ~73 (SYCL) |

The memory bandwidth gap explains why discrete GPUs are 4-8x faster for LLM inference.

## Methodology

- **Embedding benchmarks**: 100 iterations single, 20 iterations batch (after 5 warmup iterations). OpenVINO via `optimum-intel`.
- **Reranker benchmarks**: 100 iterations single, 20 iterations batch (after 5 warmup iterations). OpenVINO via `optimum-intel`. Input: query + ~50-word document per pair.
- **llama.cpp benchmarks**: `llama-bench` with `-p 512 -n 128` (512 token prompt, 128 token generation).
- **OpenVINO GenAI benchmarks**: `max_new_tokens=256`, `do_sample=False`, measured wall-clock time. Token count estimated from word count × 1.3.
- All tests run with minimal background load. Laptop on AC power, performance governor.
