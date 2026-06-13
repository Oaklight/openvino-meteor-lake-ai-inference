#!/usr/bin/env python3
"""Benchmark LLM inference using OpenVINO GenAI.

Usage:
    python bench-llm-openvino.py
    python bench-llm-openvino.py --model OpenVINO/Qwen3-8B-int4-ov --device CPU
"""

import argparse
import time

import openvino_genai as ov_genai
from huggingface_hub import snapshot_download


def benchmark_generation(pipe, prompt, max_new_tokens=256):
    """Benchmark text generation speed.

    Args:
        pipe: OpenVINO GenAI LLMPipeline.
        prompt: Input prompt text.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Dict with generation results and timing.
    """
    start = time.time()
    result = pipe.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.time() - start

    words = len(result.split())
    est_tokens = int(words * 1.3)

    return {
        "text": result,
        "est_tokens": est_tokens,
        "elapsed": elapsed,
        "tok_per_sec": est_tokens / elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM with OpenVINO GenAI")
    parser.add_argument(
        "--model",
        default="OpenVINO/Qwen3-8B-int4-ov",
        help="HuggingFace model ID (default: OpenVINO/Qwen3-8B-int4-ov)",
    )
    parser.add_argument(
        "--device",
        nargs="+",
        default=["CPU", "GPU"],
        help="Devices to benchmark (default: CPU GPU)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--prompt",
        default="Write a detailed explanation of how neural networks learn, covering backpropagation, gradient descent, and loss functions.",
        help="Prompt for generation",
    )
    args = parser.parse_args()

    # Download model if needed
    print(f"Model: {args.model}")
    model_path = snapshot_download(args.model)
    print(f"Path: {model_path}")
    print()

    results = []
    for device in args.device:
        print(f"--- {device} ---")
        t0 = time.time()
        pipe = ov_genai.LLMPipeline(model_path, device)
        load_time = time.time() - t0
        print(f"Model loaded in {load_time:.1f}s")

        # Warmup
        pipe.generate("Hi", max_new_tokens=10)

        # Benchmark
        result = benchmark_generation(pipe, args.prompt, args.max_tokens)
        result["device"] = device
        result["load_time"] = load_time
        results.append(result)

        print(f"Est tokens: {result['est_tokens']}")
        print(f"Time: {result['elapsed']:.1f}s")
        print(f"Speed: {result['tok_per_sec']:.1f} tok/s")
        print(f"Preview: {result['text'][:200]}...")
        print()

        del pipe

    # Summary
    print("=" * 50)
    print(f"{'Device':<8} {'Load (s)':>10} {'Tokens':>8} {'Time (s)':>10} {'tok/s':>8}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['device']:<8} {r['load_time']:>10.1f} {r['est_tokens']:>8} "
            f"{r['elapsed']:>10.1f} {r['tok_per_sec']:>8.1f}"
        )


if __name__ == "__main__":
    main()
