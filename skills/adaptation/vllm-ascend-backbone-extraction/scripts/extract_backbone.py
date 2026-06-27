#!/usr/bin/env python3
"""
Extract LLM backbone weights from a custom-architecture checkpoint.

Usage:
    python extract_backbone.py \
        --input /path/to/original_model \
        --output /path/to/extracted_backbone \
        --strip-prefix "point_backbone. point_proj." \
        --target-arch LlamaForCausalLM \
        --target-model-type llama
"""

import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file


def extract_weights(input_dir: str, output_dir: str, strip_prefixes: list):
    """Load safetensors, filter out non-backbone keys, save to output."""
    safetensor_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(".safetensors")
    ])

    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {input_dir}")

    # If single file, extract directly; if sharded, preserve shards
    for fname in safetensor_files:
        src_path = os.path.join(input_dir, fname)
        state_dict = load_file(src_path)

        backbone_keys = [
            k for k in state_dict.keys()
            if not any(k.startswith(p) for p in strip_prefixes)
        ]
        backbone_state = {k: state_dict[k] for k in backbone_keys}

        dst_path = os.path.join(output_dir, fname)
        save_file(backbone_state, dst_path)
        print(f"  {fname}: {len(backbone_state)} / {len(state_dict)} keys retained")


def rewrite_config(input_dir: str, output_dir: str, target_arch: str,
                   target_model_type: str, extra_remove_fields: list = None):
    """Rewrite config.json for standard backbone architecture."""
    config_path = os.path.join(input_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    config["model_type"] = target_model_type
    config["architectures"] = [target_arch]
    config.pop("auto_map", None)

    # Remove multimodal-related fields
    remove_keywords = ["point", "vision", "image", "pixel", "visual"]
    if extra_remove_fields:
        remove_keywords.extend(extra_remove_fields)

    for key in list(config.keys()):
        if any(x in key.lower() for x in remove_keywords):
            del config[key]
            print(f"  Removed config key: {key}")

    output_config_path = os.path.join(output_dir, "config.json")
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json -> model_type={target_model_type}, architectures=[{target_arch}]")


def copy_tokenizer_files(input_dir: str, output_dir: str):
    """Copy tokenizer and generation config files."""
    files_to_copy = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "added_tokens.json",
    ]
    for fname in files_to_copy:
        src = os.path.join(input_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, output_dir)
            print(f"  Copied {fname}")


def main():
    parser = argparse.ArgumentParser(description="Extract LLM backbone from custom model")
    parser.add_argument("--input", required=True, help="Path to original model directory")
    parser.add_argument("--output", required=True, help="Path to output extracted backbone directory")
    parser.add_argument("--strip-prefix", required=True,
                        help="Space-separated list of key prefixes to strip (e.g. 'point_backbone. point_proj.')")
    parser.add_argument("--target-arch", required=True,
                        help="Target architecture class name (e.g. LlamaForCausalLM)")
    parser.add_argument("--target-model-type", required=True,
                        help="Target model_type value (e.g. llama)")
    parser.add_argument("--remove-field", action="append", default=[],
                        help="Additional config.json fields to remove (can be specified multiple times)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    strip_prefixes = args.strip_prefix.split()

    print("=== Step 1: Extract backbone weights ===")
    extract_weights(args.input, args.output, strip_prefixes)

    print("\n=== Step 2: Rewrite config.json ===")
    rewrite_config(args.input, args.output, args.target_arch,
                   args.target_model_type, args.remove_field)

    print("\n=== Step 3: Copy tokenizer files ===")
    copy_tokenizer_files(args.input, args.output)

    print(f"\n=== Extraction complete ===")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
