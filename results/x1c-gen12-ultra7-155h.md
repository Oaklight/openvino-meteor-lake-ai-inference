# Raw Results: ThinkPad X1 Carbon Gen 12

**Date**: 2026-06-12
**Operator**: Peng Ding

## System Info

```
CPU:    Intel(R) Core(TM) Ultra 7 155H (6P+8E+2LPE, 22 threads)
GPU:    Intel Arc Graphics (Meteor Lake, 128 EU, xe driver)
Memory: 32 GB DDR5
OS:     Arch Linux, kernel 7.0.11-arch1-1
GPU RT: intel-compute-runtime 26.18.38308.1-2
L0:     level-zero-loader 1.28.2-1
oneAPI: intel-oneapi-dpcpp-cpp 2026.0.0_947-1
OV:     openvino 2026.2.0
```

## Embedding: BGE-M3 FP32 (auto-export)

Model: `BAAI/bge-m3`

```
--- CPU ---
Model loaded in 77.2s
Single text: 23.5 samples/sec (42.6 ms/sample)
Batch 16: 27.0 samples/sec (591.8 ms/batch)

--- GPU ---
Model loaded in 35.2s
Single text: 41.1 samples/sec (24.3 ms/sample)
Batch 16: 179.2 samples/sec (89.3 ms/batch)
```

## Embedding: BGE-M3 INT8 (pre-converted)

Model: `stellars/bge-m3-openvino-int8`

```
--- CPU ---
Model loaded in 26.3s
Single: 82.9 samples/s (12.1 ms)
Batch16: 128.3 samples/s (124.7 ms/batch)

--- GPU ---
Model loaded in 7.9s
Single: 67.6 samples/s (14.8 ms)
Batch16: 245.4 samples/s (65.2 ms/batch)
```

## LLM: Qwen3 8B — llama.cpp SYCL

Model: `Qwen3-8B-Q4_K_M.gguf` (4.68 GiB)

```
| model                          |       size |     params | backend    | ngl |     sm |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --------------: | -------------------: |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | SYCL       |  99 |   none |           pp512 |        70.18 ± 31.71 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | SYCL       |  99 |   none |           tg128 |          6.87 ± 1.77 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | SYCL       |   0 |   none |           pp512 |         88.48 ± 6.53 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | SYCL       |   0 |   none |           tg128 |          3.86 ± 0.81 |
```

## LLM: Qwen3 8B — OpenVINO GenAI

Model: `OpenVINO/Qwen3-8B-int4-ov`

```
=== CPU ===
Model loaded in 2.0s
Words: 209, est tokens: 271
Total time: 32.0s
Est generation speed: 8.5 tok/s

=== GPU ===
Model loaded in 18.6s
Words: 209, est tokens: 271
Total time: 37.6s
Est generation speed: 7.2 tok/s
```
