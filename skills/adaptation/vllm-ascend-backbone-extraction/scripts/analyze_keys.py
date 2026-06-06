#!/usr/bin/env python3
"""
Analyze checkpoint keys to identify backbone vs non-backbone weights.

Usage:
    python analyze_keys.py --model-dir /path/to/model
"""

import argparse
import json
import os

from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description="Analyze model checkpoint keys")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(args.model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print("=== config.json ===")
        print(f"  model_type: {config.get('model_type', 'N/A')}")
        print(f"  architectures: {config.get('architectures', 'N/A')}")
        print()

    # Find safetensors files
    safetensor_files = [
        f for f in os.listdir(args.model_dir)
        if f.endswith(".safetensors")
    ]
    if not safetensor_files:
        print("No .safetensors files found in model directory")
        return

    # Load and analyze keys
    all_keys = []
    for fname in sorted(safetensor_files):
        path = os.path.join(args.model_dir, fname)
        state_dict = load_file(path)
        all_keys.extend(state_dict.keys())

    print(f"=== Total keys: {len(all_keys)} ===")
    print()

    # Group by prefix
    prefix_counts = {}
    for k in all_keys:
        prefix = k.split(".")[0]
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    print("=== Key prefixes ===")
    for prefix, count in sorted(prefix_counts.items(), key=lambda x: -x[1]):
        print(f"  {prefix:30s} {count:4d} keys")
    print()

    # Common backbone indicators
    backbone_indicators = ["model.", "transformer.", "language_model.", "qwen2."]
    backbone_keys = [k for k in all_keys if any(k.startswith(p) for p in backbone_indicators)]
    print(f"=== Probable backbone keys: {len(backbone_keys)} ===")
    for k in backbone_keys[:10]:
        print(f"  {k}")
    if len(backbone_keys) > 10:
        print(f"  ... and {len(backbone_keys) - 10} more")
    print()

    # Common multimodal prefixes to strip
    mm_prefixes = ["point_", "vision_", "image_", "pixel_", "visual_", "merger_", "multi_modal_"]
    mm_keys = [k for k in all_keys if any(k.startswith(p) for p in mm_prefixes)]
    if mm_keys:
        print(f"=== Probable multimodal keys to strip: {len(mm_keys)} ===")
        prefix_set = sorted(set(k.split(".")[0] for k in mm_keys))
        print(f"  Prefixes: {', '.join(prefix_set)}")
        for k in mm_keys[:5]:
            print(f"  {k}")
        if len(mm_keys) > 5:
            print(f"  ... and {len(mm_keys) - 5} more")
    else:
        print("=== No obvious multimodal keys detected ===")


if __name__ == "__main__":
    main()
