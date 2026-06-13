#!/bin/bash
# Benchmark LLM inference using llama.cpp SYCL backend
#
# Usage:
#   ./bench-llm-llamacpp.sh <model.gguf>
#   ./bench-llm-llamacpp.sh ~/models/Qwen3-8B-Q4_K_M.gguf

set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf>}"
LLAMA_BENCH="${LLAMA_BENCH:-$(dirname "$0")/../llama.cpp-sycl/build/bin/llama-bench}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-128}"

if [ ! -f "$LLAMA_BENCH" ]; then
	echo "llama-bench not found at: $LLAMA_BENCH"
	echo "Set LLAMA_BENCH env var or run setup-llamacpp-sycl.sh first"
	exit 1
fi

echo "=== Enabling oneAPI environment ==="
source /opt/intel/oneapi/setvars.sh

echo ""
echo "=== Model: $MODEL ==="
echo "=== Prompt tokens: $PROMPT_TOKENS, Generation tokens: $GEN_TOKENS ==="

echo ""
echo "=== GPU Benchmark (all layers on GPU) ==="
ZES_ENABLE_SYSMAN=1 "$LLAMA_BENCH" \
	-m "$MODEL" \
	-ngl 99 \
	-sm none \
	-mg 0 \
	-p "$PROMPT_TOKENS" \
	-n "$GEN_TOKENS"

echo ""
echo "=== CPU Benchmark (no GPU offload) ==="
ZES_ENABLE_SYSMAN=1 "$LLAMA_BENCH" \
	-m "$MODEL" \
	-ngl 0 \
	-p "$PROMPT_TOKENS" \
	-n "$GEN_TOKENS"
