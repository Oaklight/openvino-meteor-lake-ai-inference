#!/usr/bin/env python3
"""Convert HuggingFace models to OpenVINO IR format using NNCF.

Bypasses optimum-cli limitations for newer model architectures by
directly using openvino.convert_model() + NNCF weight compression.

Usage:
    # INT4 quantization (default)
    python convert-to-openvino.py --model openbmb/MiniCPM4.1-8B --output ./MiniCPM4.1-8B-int4

    # INT8 quantization
    python convert-to-openvino.py --model openbmb/MiniCPM4.1-8B --weight-format int8 --output ./MiniCPM4.1-8B-int8

    # FP16 (no quantization)
    python convert-to-openvino.py --model openbmb/MiniCPM4.1-8B --weight-format fp16 --output ./MiniCPM4.1-8B-fp16

    # Local model path
    python convert-to-openvino.py --model /path/to/model --weight-format int4 --output ./output

    # Auto-name output directory
    python convert-to-openvino.py --model Qwen/Qwen3.5-9B --weight-format int4
    # -> outputs to ./Qwen_Qwen3.5-9B-int4/
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch


def get_output_dir(model_id: str, weight_format: str, output: str | None) -> Path:
    """Determine output directory path.

    Args:
        model_id: HuggingFace model ID or local path.
        weight_format: Quantization format (int4, int8, fp16).
        output: User-specified output path, or None for auto-naming.

    Returns:
        Path to the output directory.
    """
    if output:
        return Path(output)
    name = model_id.replace("/", "_")
    return Path(f"{name}-{weight_format}")


def _try_optimum_export(
    model_id: str, weight_format: str, output_dir: Path, trust_remote_code: bool
):
    """Try converting with optimum-intel (handles DynamicCache, KV cache, etc.).

    Args:
        model_id: HuggingFace model ID or local path.
        weight_format: Target weight format.
        output_dir: Output directory.
        trust_remote_code: Whether to trust remote code.

    Returns:
        True if optimum succeeded, None otherwise.
    """
    try:
        from optimum.intel import OVModelForCausalLM

        print(f"Trying optimum-intel export (weight_format={weight_format})...")
        t0 = time.time()
        ov_model = OVModelForCausalLM.from_pretrained(
            model_id,
            export=True,
            trust_remote_code=trust_remote_code,
        )
        # Apply quantization via optimum's save
        ov_model.save_pretrained(output_dir)
        print(f"optimum-intel export done in {time.time() - t0:.1f}s")

        # If weight format is int4/int8, need to compress after
        if weight_format in ("int4", "int8"):
            import nncf
            import openvino as ov

            print(f"Applying {weight_format} compression via NNCF...")
            core = ov.Core()
            model = core.read_model(output_dir / "openvino_model.xml")

            if weight_format == "int4":
                model = nncf.compress_weights(
                    model,
                    mode=nncf.CompressWeightsMode.INT4_ASYM,
                    group_size=128,
                )
            else:
                model = nncf.compress_weights(
                    model,
                    mode=nncf.CompressWeightsMode.INT8_SYM,
                )
            ov.save_model(model, output_dir / "openvino_model.xml")
            print(f"NNCF {weight_format} compression done")

        return True
    except Exception as e:
        print(f"optimum-intel failed: {e}")
        return None


def _write_metadata(
    output_dir: Path,
    model_id: str,
    weight_format: str,
    group_size: int,
    dtype: str,
    converter: str,
):
    """Write conversion metadata to output directory.

    Args:
        output_dir: Output directory.
        model_id: Source model ID.
        weight_format: Weight format used.
        group_size: Group size for INT4.
        dtype: Torch dtype used.
        converter: Name of the converter used.
    """
    import nncf
    import openvino as ov

    metadata = {
        "source_model": model_id,
        "weight_format": weight_format,
        "group_size": group_size if weight_format == "int4" else None,
        "converter": converter,
        "openvino_version": ov.__version__,
        "nncf_version": nncf.__version__,
        "torch_dtype": dtype,
    }
    with open(output_dir / "conversion_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _verify_output(output_dir: Path):
    """Verify the conversion output files exist.

    Args:
        output_dir: Output directory to check.
    """
    model_file = output_dir / "openvino_model.xml"
    bin_file = output_dir / "openvino_model.bin"
    if model_file.exists() and bin_file.exists():
        bin_size = bin_file.stat().st_size / (1024**3)
        print(f"\nSuccess! Output: {output_dir}")
        print(f"  openvino_model.xml + .bin ({bin_size:.2f} GB)")
    else:
        print(f"\nError: expected files not found in {output_dir}")
        sys.exit(1)


def copy_tokenizer_files(model_id: str, output_dir: Path, trust_remote_code: bool):
    """Save tokenizer and config files to the output directory.

    Args:
        model_id: HuggingFace model ID or local path.
        output_dir: Directory to save tokenizer files.
        trust_remote_code: Whether to trust remote code.
    """
    from transformers import AutoConfig, AutoTokenizer

    print("Saving tokenizer and config...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(output_dir)

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    config.save_pretrained(output_dir)


def convert_model(
    model_id: str,
    weight_format: str,
    output_dir: Path,
    trust_remote_code: bool,
    group_size: int,
    dtype: str,
):
    """Convert a HuggingFace model to OpenVINO IR with optional quantization.

    Args:
        model_id: HuggingFace model ID or local path.
        weight_format: Target weight format (int4, int8, fp16).
        output_dir: Directory to save the converted model.
        trust_remote_code: Whether to trust remote code.
        group_size: Group size for INT4 quantization.
        dtype: Torch dtype for model loading (auto, float16, bfloat16).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import nncf
    import openvino as ov

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine torch dtype
    torch_dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = torch_dtype_map.get(dtype, "auto")

    # Try optimum-intel first (handles DynamicCache, KV cache, etc.)
    ov_model = _try_optimum_export(
        model_id, weight_format, output_dir, trust_remote_code
    )

    if ov_model is not None:
        # optimum handled everything including quantization and saving
        copy_tokenizer_files(model_id, output_dir, trust_remote_code)
        _write_metadata(
            output_dir, model_id, weight_format, group_size, dtype, "optimum-intel"
        )
        _verify_output(output_dir)
        return

    # Fallback: direct ov.convert_model + NNCF
    print("optimum-intel failed, falling back to ov.convert_model + NNCF...")

    # Load tokenizer for example input
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )

    # Load PyTorch model
    print(f"Loading model from {model_id} (dtype={dtype})...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Prepare example input for tracing
    example_text = "Hello, how are you?"
    example_input = tokenizer(example_text, return_tensors="pt")
    example_input = {
        "input_ids": example_input["input_ids"],
        "attention_mask": example_input["attention_mask"],
    }

    # Convert to OpenVINO — try with example_input first, fallback without
    print("Converting to OpenVINO IR...")
    t0 = time.time()
    try:
        ov_model = ov.convert_model(model, example_input=example_input)
    except Exception as e:
        print(f"Tracing with example_input failed ({e}), trying without...")
        ov_model = ov.convert_model(model)
    print(f"Conversion done in {time.time() - t0:.1f}s")

    # Free PyTorch model memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Apply weight compression if needed
    if weight_format == "fp16":
        print("Saving as FP16 (no quantization)...")
        ov.save_model(
            ov_model, output_dir / "openvino_model.xml", compress_to_fp16=True
        )
    elif weight_format == "int8":
        print("Applying INT8 weight compression...")
        t0 = time.time()
        ov_model = nncf.compress_weights(
            ov_model,
            mode=nncf.CompressWeightsMode.INT8_SYM,
        )
        print(f"INT8 compression done in {time.time() - t0:.1f}s")
        ov.save_model(ov_model, output_dir / "openvino_model.xml")
    elif weight_format == "int4":
        print(f"Applying INT4 weight compression (group_size={group_size})...")
        t0 = time.time()
        ov_model = nncf.compress_weights(
            ov_model,
            mode=nncf.CompressWeightsMode.INT4_ASYM,
            group_size=group_size,
        )
        print(f"INT4 compression done in {time.time() - t0:.1f}s")
        ov.save_model(ov_model, output_dir / "openvino_model.xml")
    else:
        raise ValueError(f"Unknown weight format: {weight_format}")

    # Save tokenizer and config
    copy_tokenizer_files(model_id, output_dir, trust_remote_code)

    _write_metadata(
        output_dir,
        model_id,
        weight_format,
        group_size,
        dtype,
        "ov.convert_model + NNCF",
    )
    _verify_output(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to OpenVINO IR using NNCF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model Qwen/Qwen3-8B --weight-format int4
  %(prog)s --model openbmb/MiniCPM4.1-8B --weight-format int8 --output ./output
  %(prog)s --model /local/path/model --weight-format fp16
        """,
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--weight-format",
        "-w",
        choices=["int4", "int8", "fp16"],
        default="int4",
        help="Weight format (default: int4)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory (default: auto-named from model + format)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code (default: True)",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
        help="Do not trust remote code",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for INT4 quantization (default: 128)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        default="auto",
        help="Torch dtype for loading (default: auto)",
    )
    args = parser.parse_args()

    output_dir = get_output_dir(args.model, args.weight_format, args.output)

    # Check if already done
    if (output_dir / "openvino_model.xml").exists():
        print(f"Output already exists: {output_dir}")
        print("Delete it first if you want to reconvert.")
        sys.exit(0)

    print(f"Model:         {args.model}")
    print(f"Weight format: {args.weight_format}")
    print(f"Output:        {output_dir}")
    print(f"Torch dtype:   {args.dtype}")
    if args.weight_format == "int4":
        print(f"Group size:    {args.group_size}")
    print()

    convert_model(
        model_id=args.model,
        weight_format=args.weight_format,
        output_dir=output_dir,
        trust_remote_code=args.trust_remote_code,
        group_size=args.group_size,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
