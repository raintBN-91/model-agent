#!/usr/bin/env python3
"""
Chronos-2 Ascend NPU Inference Script
======================================
Amazon Chronos-2 time series forecasting model inference on Huawei Ascend NPU.
Supports CPU baseline, NPU FP32, NPU FP16/AMP, and batch inference.

Usage:
    python inference.py --model_path /path/to/model --mode cpu
    python inference.py --model_path /path/to/model --mode npu --prediction_length 64
    python inference.py --model_path /path/to/model --mode npu --batch_size 128
    python inference.py --model_path /path/to/model --mode validate
    python inference.py --model_path /path/to/model --mode benchmark --num_runs 100
"""

import argparse
import json
import logging
import os
import time
import warnings
import sys
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch

warnings.filterwarnings("ignore")

# Silence torch_dtype deprecation by pre-importing chronos with env suppression
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
IS_NPU_AVAILABLE = False
NPU_COUNT = 0
NPU_NAME = "N/A"
CANONICAL_MODEL_PATH = "/tmp/chronos2_model/amazon/chronos-2"

try:
    import torch_npu

    if torch.npu.is_available():
        IS_NPU_AVAILABLE = True
        NPU_COUNT = torch.npu.device_count()
        NPU_NAME = torch.npu.get_device_name(0)
except Exception:
    pass


def has_jemalloc() -> bool:
    """Detect if jemalloc is loaded via LD_PRELOAD."""
    try:
        import ctypes
        ctypes.CDLL("libjemalloc.so.2")
        return True
    except Exception:
        return False


def setup_npu_env() -> dict:
    """Configure NPU runtime environment. Returns active optimizations."""
    active = {}
    if not IS_NPU_AVAILABLE:
        return active

    if not os.environ.get("TASK_QUEUE_ENABLE"):
        os.environ["TASK_QUEUE_ENABLE"] = "1"
    active["TASK_QUEUE_ENABLE"] = os.environ.get("TASK_QUEUE_ENABLE", "0")

    if os.environ.get("CPU_AFFINITY_CONF"):
        active["CPU_AFFINITY_CONF"] = os.environ["CPU_AFFINITY_CONF"]

    if has_jemalloc():
        active["jemalloc"] = True
    else:
        active["jemalloc"] = False

    return active


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_pipeline(model_path: str, device: str = "cpu"):
    """Load Chronos2Pipeline from local path and move to specified device."""
    from chronos import Chronos2Pipeline

    logger.info(f"Loading Chronos-2 from {model_path} ...")
    t0 = time.time()

    pipeline = Chronos2Pipeline.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map=None,
    )

    if device == "npu":
        pipeline.model = pipeline.model.to("npu:0")
        logger.info(f"Model moved to NPU:0 ({NPU_NAME})")
    elif device == "cuda":
        pipeline.model = pipeline.model.to("cuda:0")
        logger.info("Model moved to CUDA:0")
    else:
        pipeline.model = pipeline.model.to("cpu")
        logger.info("Model on CPU")

    pipeline.model.eval()
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.2f}s")
    return pipeline, load_time


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------
def generate_test_data(
    batch_size: int = 1,
    context_length: int = 512,
    seed: int = 42,
) -> torch.Tensor:
    """Generate reproducible synthetic time series test data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = torch.randn(batch_size, 1, context_length)
    return data


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(
    pipeline,
    inputs: torch.Tensor,
    prediction_length: int,
    batch_size: int = 1,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    warmup: int = 3,
    synchronize: bool = True,
) -> tuple[list[torch.Tensor], float]:
    """Run inference and return predictions + wall time in seconds."""
    device_type = str(pipeline.model.device)

    autocast_ctx = nullcontext()
    if use_amp and device_type.startswith("npu"):
        autocast_ctx = torch.npu.amp.autocast(dtype=amp_dtype)
    elif use_amp and device_type.startswith("cuda"):
        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)

    # Warmup
    for _ in range(warmup):
        _ = pipeline.predict(
            inputs, prediction_length=prediction_length, batch_size=batch_size
        )

    if synchronize:
        if device_type.startswith("npu"):
            torch.npu.synchronize()
        elif device_type.startswith("cuda"):
            torch.cuda.synchronize()

    t0 = time.time()
    with autocast_ctx:
        predictions = pipeline.predict(
            inputs,
            prediction_length=prediction_length,
            batch_size=batch_size,
        )

    if synchronize:
        if device_type.startswith("npu"):
            torch.npu.synchronize()
        elif device_type.startswith("cuda"):
            torch.cuda.synchronize()

    elapsed = time.time() - t0
    return predictions, elapsed


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def benchmark_runs(
    pipeline,
    inputs: torch.Tensor,
    prediction_length: int,
    batch_size: int = 1,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    num_runs: int = 100,
    warmup: int = 5,
) -> dict:
    """Run timed benchmark and return latency statistics."""
    device_type = str(pipeline.model.device)

    autocast_ctx = nullcontext()
    if use_amp and device_type.startswith("npu"):
        autocast_ctx = torch.npu.amp.autocast(dtype=amp_dtype)
    elif use_amp and device_type.startswith("cuda"):
        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)

    # Warmup
    for _ in range(warmup):
        _ = pipeline.predict(
            inputs, prediction_length=prediction_length, batch_size=batch_size
        )

    latencies = []
    for _ in range(num_runs):
        if device_type.startswith("npu"):
            torch.npu.synchronize()
        elif device_type.startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.time()
        with autocast_ctx:
            _ = pipeline.predict(
                inputs,
                prediction_length=prediction_length,
                batch_size=batch_size,
            )

        if device_type.startswith("npu"):
            torch.npu.synchronize()
        elif device_type.startswith("cuda"):
            torch.cuda.synchronize()

        latencies.append(time.time() - t0)

    arr = np.array(latencies) * 1000  # ms
    return {
        "num_runs": num_runs,
        "avg_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "throughput_items_s": float(batch_size / (np.mean(arr) / 1000)),
    }


# ---------------------------------------------------------------------------
# Accuracy comparison
# ---------------------------------------------------------------------------
def compare_outputs(
    cpu_preds: list[torch.Tensor],
    npu_preds: list[torch.Tensor],
) -> dict:
    """Compute accuracy metrics between CPU and NPU predictions."""
    all_max_rel_err = []
    all_mean_rel_err = []
    all_max_abs = []

    for i, (cpu_p, npu_p) in enumerate(zip(cpu_preds, npu_preds)):
        cpu_a = cpu_p.float().numpy()
        npu_a = npu_p.float().numpy()

        abs_diff = np.abs(cpu_a - npu_a)
        max_abs_err = float(np.max(abs_diff))
        denominator = np.maximum(np.abs(cpu_a), 1e-8)
        rel_err = abs_diff / denominator
        max_rel_err = float(np.max(rel_err))
        mean_rel_err = float(np.mean(rel_err))

        all_max_rel_err.append(max_rel_err)
        all_mean_rel_err.append(mean_rel_err)
        all_max_abs.append(max_abs_err)

    return {
        "max_abs_error": float(np.max(all_max_abs)),
        "max_rel_error_pct": float(np.max(all_max_rel_err) * 100),
        "mean_rel_error_pct": float(np.mean(all_mean_rel_err) * 100),
        "per_sample": [
            {"max_rel_error_pct": e * 100, "max_abs_error": a}
            for e, a in zip(all_max_rel_err, all_max_abs)
        ],
    }


# ---------------------------------------------------------------------------
# Accuracy validation
# ---------------------------------------------------------------------------
def accuracy_validation(model_path: str, prediction_length: int = 64) -> dict:
    """Compare CPU vs NPU outputs across multiple context lengths."""
    logger.info("=" * 60)
    logger.info("ACCURACY VALIDATION: CPU vs NPU")
    logger.info("=" * 60)

    cpu_pipeline, _ = load_pipeline(model_path, device="cpu")

    test_configs = [
        {"name": "short_ctx128_pred16", "context_length": 128, "prediction_length": 16},
        {"name": "medium_ctx512_pred64", "context_length": 512, "prediction_length": 64},
        {"name": "long_ctx2048_pred128", "context_length": 2048, "prediction_length": 128},
    ]

    results = []
    for cfg in test_configs:
        logger.info(f"\n  Test: {cfg['name']}")
        data = generate_test_data(batch_size=1, context_length=cfg["context_length"])

        cpu_preds, cpu_t = run_inference(
            cpu_pipeline, data, prediction_length=cfg["prediction_length"]
        )
        logger.info(f"    CPU: {cpu_t*1000:.1f}ms, shape: {cpu_preds[0].shape}")

        if IS_NPU_AVAILABLE:
            setup_npu_env()
            npu_pipeline, _ = load_pipeline(model_path, device="npu")
            npu_preds, npu_t = run_inference(
                npu_pipeline, data, prediction_length=cfg["prediction_length"]
            )
            logger.info(f"    NPU: {npu_t*1000:.1f}ms, shape: {npu_preds[0].shape}")

            metrics = compare_outputs(cpu_preds, npu_preds)
            r = {
                "name": cfg["name"],
                "context_length": cfg["context_length"],
                "prediction_length": cfg["prediction_length"],
                "cpu_time_ms": cpu_t * 1000,
                "npu_time_ms": npu_t * 1000,
                "speedup": cpu_t / npu_t if npu_t > 0 else 0,
                "max_rel_error_pct": metrics["max_rel_error_pct"],
                "max_abs_error": metrics["max_abs_error"],
                "pass": metrics["max_rel_error_pct"] < 1.0,
            }
            results.append(r)
            logger.info(
                f"    max_rel_error={metrics['max_rel_error_pct']:.4f}%, "
                f"speedup={r['speedup']:.2f}x, PASS={r['pass']}"
            )
            del npu_pipeline
            torch.npu.empty_cache()
        else:
            results.append({**cfg, "cpu_time_ms": cpu_t * 1000, "error": "NPU not available"})

    del cpu_pipeline
    return {
        "configs": results,
        "overall_pass": all(r.get("pass", False) for r in results),
        "np_available": IS_NPU_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Chronos-2 Ascend NPU Inference")
    parser.add_argument(
        "--model_path", type=str, default=CANONICAL_MODEL_PATH,
        help=f"Path to Chronos-2 model dir (default: {CANONICAL_MODEL_PATH})"
    )
    parser.add_argument(
        "--mode", type=str, default="npu",
        choices=["cpu", "npu", "benchmark", "validate"],
    )
    parser.add_argument("--prediction_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_sync", action="store_true", help="Disable device sync in timing")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(
            f"Model not found at {args.model_path}. "
            f"Download via: python download_model.py "
            f"or: from modelscope import snapshot_download; "
            f"snapshot_download('amazon/chronos-2', cache_dir='/tmp/chronos2_model')"
        )

    logger.info(f"Model path: {args.model_path}")
    logger.info(f"NPU available: {IS_NPU_AVAILABLE}")
    if IS_NPU_AVAILABLE:
        logger.info(f"NPU: {NPU_NAME} x {NPU_COUNT}")
    logger.info(f"jemalloc: {has_jemalloc()}")

    results = {}

    # ---- CPU mode ----
    if args.mode == "cpu":
        pipeline, load_time = load_pipeline(args.model_path, device="cpu")
        data = generate_test_data(batch_size=args.batch_size, context_length=args.context_length)
        preds, elapsed = run_inference(
            pipeline, data, prediction_length=args.prediction_length,
            synchronize=not args.no_sync,
        )
        logger.info(f"CPU: {elapsed*1000:.1f}ms, output shape: {preds[0].shape}")
        results["cpu"] = {"latency_ms": elapsed * 1000, "output_shape": list(preds[0].shape)}
        del pipeline

    # ---- NPU mode ----
    elif args.mode == "npu":
        if not IS_NPU_AVAILABLE:
            logger.error("NPU not available!")
            sys.exit(1)
        active_env = setup_npu_env()
        logger.info(f"Active optimizations: {active_env}")

        pipeline, load_time = load_pipeline(args.model_path, device="npu")
        data = generate_test_data(batch_size=args.batch_size, context_length=args.context_length)

        preds, elapsed = run_inference(
            pipeline, data,
            prediction_length=args.prediction_length,
            batch_size=args.batch_size,
            use_amp=args.fp16,
            synchronize=not args.no_sync,
        )
        logger.info(
            f"NPU {'FP16' if args.fp16 else 'FP32'}: "
            f"{elapsed*1000:.1f}ms, output shape: {preds[0].shape}"
        )
        results["npu"] = {
            "latency_ms": elapsed * 1000,
            "output_shape": list(preds[0].shape),
            "fp16": args.fp16,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "optimizations": active_env,
        }
        del pipeline
        torch.npu.empty_cache()

    # ---- Benchmark mode ----
    elif args.mode == "benchmark":
        if IS_NPU_AVAILABLE:
            active_env = setup_npu_env()
            logger.info(f"NPU optimizations: {active_env}")

        configs = [
            {"label": "CPU_FP32", "device": "cpu", "use_amp": False},
        ]
        if IS_NPU_AVAILABLE:
            configs += [
                {"label": "NPU_FP32", "device": "npu", "use_amp": False},
                {"label": "NPU_FP16", "device": "npu", "use_amp": True},
            ]

        bench_results = []
        for cfg in configs:
            logger.info(f"\n--- Benchmark: {cfg['label']} ---")
            pipeline, load_time = load_pipeline(args.model_path, device=cfg["device"])
            data = generate_test_data(
                batch_size=args.batch_size, context_length=args.context_length
            )
            stats = benchmark_runs(
                pipeline, data,
                prediction_length=args.prediction_length,
                batch_size=args.batch_size,
                use_amp=cfg["use_amp"],
                num_runs=args.num_runs,
            )
            stats["label"] = cfg["label"]
            stats["device"] = cfg["device"]
            stats["load_time_s"] = load_time
            stats["batch_size"] = args.batch_size
            stats["context_length"] = args.context_length
            bench_results.append(stats)
            logger.info(
                f"  {cfg['label']}: avg={stats['avg_ms']:.1f}ms, "
                f"p50={stats['p50_ms']:.1f}ms, p95={stats['p95_ms']:.1f}ms, "
                f"throughput={stats['throughput_items_s']:.2f}/s"
            )
            del pipeline
            if cfg["device"] == "npu":
                torch.npu.empty_cache()

        results["benchmark"] = bench_results

        # Speedup vs CPU
        cpu_avg = bench_results[0]["avg_ms"] if bench_results else 1
        for b in bench_results[1:]:
            b["speedup_vs_cpu"] = cpu_avg / b["avg_ms"] if b["avg_ms"] > 0 else 0

    # ---- Validate mode ----
    elif args.mode == "validate":
        acc = accuracy_validation(args.model_path, prediction_length=args.prediction_length)
        results["accuracy_validation"] = acc

        # Print summary
        print("\n" + "=" * 60)
        print("ACCURACY VALIDATION SUMMARY")
        print("=" * 60)
        for r in acc["configs"]:
            status = "PASS" if r.get("pass") else "FAIL"
            print(
                f"  {r['name']}: max_rel_err={r.get('max_rel_error_pct', 'N/A'):.4f}%, "
                f"NPU={r.get('npu_time_ms', 'N/A'):.1f}ms, "
                f"speedup={r.get('speedup', 0):.1f}x, {status}"
            )
        print(f"  Overall: {'PASS' if acc['overall_pass'] else 'FAIL'}")

    # Save output
    output_path = args.output or f"inference_results_{args.mode}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    print("\n" + json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
