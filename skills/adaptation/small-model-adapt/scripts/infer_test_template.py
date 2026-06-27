#!/usr/bin/env python3
"""<ModelName> NPU Inference Test — real data forward pass + result visualization.

For CV/audio/video models that need real input data and output validation.

Usage:
    cp infer_test_template.py infer_<model_name>.py
    # Edit: replace all <...> placeholders, add test media files
    python3 infer_<model_name>.py
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch_npu  # noqa: F401

warnings.filterwarnings("ignore")

RESULT_FILE = os.path.join(os.path.dirname(__file__), "infer_result.json")


def load_test_input(device: str) -> torch.Tensor:
    """Load real test data appropriate for the model type.

    TODO: Replace with actual data loading logic.
    Examples:
        CV model:   read image with PIL/cv2, apply transforms
        Audio model: load wav with torchaudio, resample
        Video model: read frames with decord/cv2
    """
    # ── CV model example ──
    # from PIL import Image
    # from torchvision import transforms
    # img = Image.open("test.jpg").convert("RGB")
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # return transform(img).unsqueeze(0).to(device)

    # ── Audio model example ──
    # import torchaudio
    # waveform, sr = torchaudio.load("test_audio.wav")
    # if sr != 16000:
    #     waveform = torchaudio.functional.resample(waveform, sr, 16000)
    # return waveform.to(device)

    # ── Placeholder ──
    return torch.randn(1, 3, 224, 224, device=device)


def process_output(output, input_data) -> dict:
    """Process and optionally visualize model output.

    TODO: Replace with actual output handling.
    Returns dict with output metadata.
    """
    # Handle different output formats
    if isinstance(output, (tuple, list)):
        out = output[0]
    elif isinstance(output, dict):
        out = list(output.values())[0]
    else:
        out = output

    # ── CV: save visualization ──
    # from torchvision.utils import save_image
    # save_image(out, "viz.png")

    # ── Audio: save wav ──
    # import torchaudio
    # torchaudio.save("output.wav", out.cpu(), 16000)

    return {
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
        "output_device": str(out.device),
        "output_min": round(float(out.min().cpu()), 4),
        "output_max": round(float(out.max().cpu()), 4),
        "output_mean": round(float(out.mean().cpu()), 4),
    }


def main():
    results: dict = {
        "model": "<ModelName>",
        "variant": "<variant>",
        "test_type": "real_inference",
        "npu_available": False,
        "tests": {},
    }

    # ── 1. NPU check ──────────────────────────────────────────
    try:
        if not torch.npu.is_available():
            raise RuntimeError("NPU not available")
        results["npu_available"] = True
        results["npu_device"] = torch.npu.get_device_name(0)
        results["npu_count"] = torch.npu.device_count()
    except Exception as e:
        results["tests"]["npu_check"] = {"status": "failed", "error": str(e)}
        with open(RESULT_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        sys.exit(1)

    device = "npu:0"

    # ── 2. Model loading ──────────────────────────────────────
    t0 = time.time()
    model = None
    try:
        # TODO: Replace with actual model loading (with pretrained weights)
        # model = ModelClass.from_pretrained("weights_path").to(device).eval()
        results["tests"]["model_loading"] = {
            "status": "passed",
            "time_s": round(time.time() - t0, 3),
        }
    except Exception as e:
        results["tests"]["model_loading"] = {
            "status": "failed",
            "error": str(e),
            "time_s": round(time.time() - t0, 3),
        }
        with open(RESULT_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        sys.exit(1)

    # ── 3. Load test input ────────────────────────────────────
    t0 = time.time()
    try:
        test_input = load_test_input(device)
        results["tests"]["input_loading"] = {
            "status": "passed",
            "input_shape": list(test_input.shape),
            "input_dtype": str(test_input.dtype),
            "time_s": round(time.time() - t0, 3),
        }
    except Exception as e:
        results["tests"]["input_loading"] = {
            "status": "failed",
            "error": str(e),
        }
        with open(RESULT_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        sys.exit(1)

    # ── 4. Forward pass ───────────────────────────────────────
    try:
        t0 = time.time()
        with torch.no_grad():
            output = model(test_input)
        fwd_time = time.time() - t0

        output_meta = process_output(output, test_input)

        results["tests"]["forward"] = {
            "status": "passed",
            "forward_time_s": round(fwd_time, 3),
            **output_meta,
        }
    except Exception as e:
        results["tests"]["forward"] = {
            "status": "failed",
            "error": str(e),
        }

    # ── Compute overall ──────────────────────────────────────
    results["overall"] = all(
        t.get("status") in ("passed", "passed_limited", "skipped")
        for t in results["tests"].values()
    )

    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {RESULT_FILE}")
    print(f"Overall: {'PASSED' if results['overall'] else 'FAILED'}")

    sys.exit(0 if results["overall"] else 1)


if __name__ == "__main__":
    main()
