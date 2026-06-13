#!/usr/bin/env python3
"""Benchmark reranker models on OpenVINO (CPU vs GPU, FP16 vs INT8).

Usage:
    python bench-reranker.py
    python bench-reranker.py --model J-Parker/bge-reranker-v2-m3-openvino --device GPU
"""

import argparse
import time

from optimum.intel import OVModelForSequenceClassification
from transformers import AutoTokenizer


def benchmark(model, tokenizer, device, n_single=100, n_batch=20, batch_size=16):
    """Run single and batch reranker benchmarks.

    Args:
        model: OpenVINO model for sequence classification (reranking).
        tokenizer: HuggingFace tokenizer.
        device: Device name (CPU/GPU).
        n_single: Number of single-pair iterations.
        n_batch: Number of batch iterations.
        batch_size: Number of query-document pairs per batch.

    Returns:
        Dict with benchmark results.
    """
    query = "What is deep learning?"
    document = (
        "Deep learning is a subset of machine learning that uses neural networks "
        "with multiple layers to learn representations of data. It has achieved "
        "state-of-the-art results in many tasks including image recognition, "
        "natural language processing, and speech recognition."
    )

    # Single pair input
    inputs = tokenizer(
        query,
        document,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    # Warmup
    for _ in range(5):
        model(**inputs)

    # Single pair benchmark
    start = time.time()
    for _ in range(n_single):
        model(**inputs)
    single_elapsed = time.time() - start
    single_rate = n_single / single_elapsed
    single_latency = single_elapsed / n_single * 1000

    # Batch benchmark
    queries = [query] * batch_size
    documents = [
        f"Document {i}: {document} Additional context for document {i}."
        for i in range(batch_size)
    ]
    inputs_batch = tokenizer(
        queries,
        documents,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    for _ in range(3):
        model(**inputs_batch)

    start = time.time()
    for _ in range(n_batch):
        model(**inputs_batch)
    batch_elapsed = time.time() - start
    batch_rate = n_batch * batch_size / batch_elapsed
    batch_latency = batch_elapsed / n_batch * 1000

    return {
        "device": device,
        "single_rate": single_rate,
        "single_latency_ms": single_latency,
        "batch_rate": batch_rate,
        "batch_latency_ms": batch_latency,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark reranker models on OpenVINO"
    )
    parser.add_argument(
        "--model",
        default="J-Parker/bge-reranker-v2-m3-openvino",
        help="HuggingFace model ID (default: J-Parker/bge-reranker-v2-m3-openvino)",
    )
    parser.add_argument(
        "--device",
        nargs="+",
        default=["CPU", "GPU"],
        help="Devices to benchmark (default: CPU GPU)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export model to OpenVINO format (for non-OV models)",
    )
    parser.add_argument(
        "--n-single",
        type=int,
        default=100,
        help="Single-pair iterations (default: 100)",
    )
    parser.add_argument(
        "--n-batch", type=int, default=20, help="Batch iterations (default: 20)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: 16)"
    )
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Export: {args.export}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    results = []

    for device in args.device:
        print(f"--- {device} ---")
        t0 = time.time()
        if args.export:
            model = OVModelForSequenceClassification.from_pretrained(
                args.model, export=True, device=device
            )
        else:
            model = OVModelForSequenceClassification.from_pretrained(
                args.model, device=device
            )
        load_time = time.time() - t0
        print(f"Model loaded in {load_time:.1f}s")

        result = benchmark(
            model,
            tokenizer,
            device,
            n_single=args.n_single,
            n_batch=args.n_batch,
            batch_size=args.batch_size,
        )
        results.append(result)

        print(
            f"Single: {result['single_rate']:.1f} pairs/s ({result['single_latency_ms']:.1f} ms)"
        )
        print(
            f"Batch {args.batch_size}: {result['batch_rate']:.1f} pairs/s ({result['batch_latency_ms']:.1f} ms/batch)"
        )
        print()

        del model

    # Summary table
    print("=" * 70)
    print(
        f"{'Device':<8} {'Single (p/s)':>14} {'Latency (ms)':>14} {'Batch (p/s)':>14} {'Batch ms':>10}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['device']:<8} {r['single_rate']:>14.1f} {r['single_latency_ms']:>14.1f} "
            f"{r['batch_rate']:>14.1f} {r['batch_latency_ms']:>10.1f}"
        )


if __name__ == "__main__":
    main()
