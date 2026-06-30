#!/usr/bin/env python3
"""<ModelName> NPU Smoke Test — import, loading, forward pass verification.

Replace <Placeholder> markers with actual values for the target model.

Usage:
    cp smoke_test_template.py test_<model_name>.py
    # Edit: replace all <...> placeholders
    python3 test_<model_name>.py
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch_npu  # noqa: F401 — registers NPU device

warnings.filterwarnings("ignore")

RESULT_FILE = os.path.join(os.path.dirname(__file__), "<model_name>_result.json")


def check_npu() -> dict:
    """Verify NPU availability and return device info."""
    if not torch.npu.is_available():
        raise RuntimeError("NPU not available")
    return {
        "available": True,
        "count": torch.npu.device_count(),
        "name": torch.npu.get_device_name(0),
    }


def save_and_exit(results: dict) -> None:
    """Write results JSON and exit with appropriate code."""
    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {RESULT_FILE}")
    sys.exit(0 if results.get("overall") else 1)


def main():
    results: dict = {
        "model": "<ModelName>",
        "variant": "<variant>",
        "repo": "<https://github.com/org/repo>",
        "implementation": "<timm|transformers|torchvision|opencv|modelscope>",
        "npu_available": False,
        "tests": {},
    }

    # ── 1. NPU check ──────────────────────────────────────────
    try:
        npu = check_npu()
        results["npu_available"] = True
        results["npu_device"] = npu["name"]
    except Exception as e:
        results["tests"]["npu_check"] = {"status": "failed", "error": str(e)}
        return save_and_exit(results)

    device = "npu:0"

    # ── 2. Import test ────────────────────────────────────────
    t0 = time.time()
    try:
        # TODO: Replace with actual import
        # from <package> import <ModelClass>
        results["tests"]["import"] = {
            "status": "passed",
            "time_s": round(time.time() - t0, 3),
        }
    except Exception as e:
        results["tests"]["import"] = {
            "status": "failed",
            "error": str(e),
            "time_s": round(time.time() - t0, 3),
        }
        return save_and_exit(results)

    # ── 3. Model loading test ─────────────────────────────────
    t0 = time.time()
    model = None
    try:
        # TODO: Replace with actual model loading
        # model = ModelClass(...).to(device).eval()
        results["tests"]["model_loading"] = {
            "status": "passed",
            "device": device,
            "time_s": round(time.time() - t0, 3),
        }
    except Exception as e:
        results["tests"]["model_loading"] = {
            "status": "failed",
            "error": str(e),
            "time_s": round(time.time() - t0, 3),
        }
        return save_and_exit(results)

    # ── 4. Forward pass test (single input) ───────────────────
    try:
        # TODO: Replace with actual input shape
        input_shape = [1, 3, 224, 224]  # <batch, channels, height, width>
        dummy = torch.randn(*input_shape, device=device)

        t0 = time.time()
        with torch.no_grad():
            output = model(dummy)
        fwd_time = time.time() - t0

        # Handle output format (tuple, dict, tensor, etc.)
        if isinstance(output, (tuple, list)):
            out = output[0]
        elif isinstance(output, dict):
            out = list(output.values())[0]
        else:
            out = output

        results["tests"]["forward_single"] = {
            "status": "passed",
            "input_shape": input_shape,
            "output_shape": list(out.shape),
            "forward_time_s": round(fwd_time, 3),
        }
    except Exception as e:
        results["tests"]["forward_single"] = {
            "status": "failed",
            "error": str(e),
        }

    # ── Compute overall ──────────────────────────────────────
    results["overall"] = all(
        t.get("status") in ("passed", "passed_limited", "skipped")
        for t in results["tests"].values()
    )
    save_and_exit(results)


if __name__ == "__main__":
    main()
