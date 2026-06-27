#!/usr/bin/env python3
"""
Chronos-2 Ascend NPU Evaluation Suite
=====================================
Comprehensive evaluation: accuracy validation, performance benchmark,
batch scaling, and context length scaling.

Generates:
  - evaluation/evaluation_report.json
  - evaluation/evaluation_report.md
  - evaluation/benchmark_results.json

Usage:
    python evaluate.py --model_path /path/to/model --output_dir ./evaluation
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

IS_NPU_AVAILABLE = False
NPU_NAME = "N/A"
try:
    import torch_npu

    if torch.npu.is_available():
        IS_NPU_AVAILABLE = True
        NPU_NAME = torch.npu.get_device_name(0)
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import (
    load_pipeline,
    generate_test_data,
    run_inference,
    benchmark_runs,
    compare_outputs,
    setup_npu_env,
    has_jemalloc,
)


def print_section(title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def get_model_info(pipeline) -> dict:
    model = pipeline.model
    config = model.config
    chronos_cfg = model.chronos_config
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "architecture": config.architectures[0] if config.architectures else "unknown",
        "d_model": getattr(config, "d_model", None),
        "num_layers": getattr(config, "num_layers", None),
        "num_heads": getattr(config, "num_heads", None),
        "num_params": n_params,
        "num_params_M": round(n_params / 1e6, 1),
        "context_length": chronos_cfg.context_length,
        "input_patch_size": chronos_cfg.input_patch_size,
        "output_patch_size": chronos_cfg.output_patch_size,
        "max_output_patches": chronos_cfg.max_output_patches,
        "default_prediction_length": chronos_cfg.max_output_patches * chronos_cfg.output_patch_size,
        "quantiles": chronos_cfg.quantiles,
    }


# ---------------------------------------------------------------------------
# 1. Accuracy validation
# ---------------------------------------------------------------------------
def run_accuracy_test(model_path: str) -> dict:
    print_section("1. ACCURACY VALIDATION (CPU vs NPU)")
    print("  Requirement: max relative error < 1%")

    results = {
        "test": "accuracy_validation",
        "requirement": "max_rel_error < 1%",
        "timestamp": datetime.now().isoformat(),
    }

    cpu_pipeline, cpu_load = load_pipeline(model_path, device="cpu")
    results["model_info"] = get_model_info(cpu_pipeline)

    test_configs = [
        {"name": "short_ctx128_pred16", "ctx": 128, "pred": 16},
        {"name": "medium_ctx512_pred64", "ctx": 512, "pred": 64},
        {"name": "long_ctx2048_pred128", "ctx": 2048, "pred": 128},
    ]

    acc_entries = []
    for cfg in test_configs:
        print(f"\n  --- {cfg['name']} ---")
        data = generate_test_data(batch_size=1, context_length=cfg["ctx"], seed=42)

        cpu_preds, cpu_t = run_inference(cpu_pipeline, data, prediction_length=cfg["pred"])
        print(f"    CPU: {cpu_t*1000:.1f}ms, shape={cpu_preds[0].shape}")

        if IS_NPU_AVAILABLE:
            setup_npu_env()
            npu_pipeline, npu_load = load_pipeline(model_path, device="npu")

            # NPU FP32
            npu_preds, npu_t = run_inference(npu_pipeline, data, prediction_length=cfg["pred"])
            print(f"    NPU FP32: {npu_t*1000:.1f}ms, shape={npu_preds[0].shape}")

            # NPU FP16
            npu_preds_fp16, npu_t_fp16 = run_inference(
                npu_pipeline, data, prediction_length=cfg["pred"], use_amp=True
            )
            print(f"    NPU FP16: {npu_t_fp16*1000:.1f}ms")

            metrics_fp32 = compare_outputs(cpu_preds, npu_preds)
            metrics_fp16 = compare_outputs(cpu_preds, npu_preds_fp16)

            entry = {
                "name": cfg["name"],
                "context_length": cfg["ctx"],
                "prediction_length": cfg["pred"],
                "cpu_time_ms": round(cpu_t * 1000, 3),
                "npu_fp32_time_ms": round(npu_t * 1000, 3),
                "npu_fp16_time_ms": round(npu_t_fp16 * 1000, 3),
                "speedup_fp32": round(cpu_t / npu_t, 2) if npu_t > 0 else 0,
                "speedup_fp16": round(cpu_t / npu_t_fp16, 2) if npu_t_fp16 > 0 else 0,
                "fp32_max_rel_error_pct": round(metrics_fp32["max_rel_error_pct"], 6),
                "fp16_max_rel_error_pct": round(metrics_fp16["max_rel_error_pct"], 4),
                "fp32_pass": metrics_fp32["max_rel_error_pct"] < 1.0,
                "fp16_pass": metrics_fp16["max_rel_error_pct"] < 1.0,
            }
            acc_entries.append(entry)
            print(
                f"    FP32 max_rel_err={metrics_fp32['max_rel_error_pct']:.4f}% "
                f"({'PASS' if entry['fp32_pass'] else 'FAIL'})"
            )
            print(
                f"    FP16 max_rel_err={metrics_fp16['max_rel_error_pct']:.2f}% "
                f"({'PASS' if entry['fp16_pass'] else 'FAIL'})"
            )
            del npu_pipeline
            torch.npu.empty_cache()
        else:
            acc_entries.append({
                "name": cfg["name"],
                "cpu_time_ms": round(cpu_t * 1000, 3),
                "error": "NPU not available",
            })

    del cpu_pipeline
    results["entries"] = acc_entries
    results["fp32_overall_pass"] = all(
        e.get("fp32_pass", False) for e in acc_entries if "fp32_pass" in e
    )
    results["fp16_overall_pass"] = all(
        e.get("fp16_pass", False) for e in acc_entries if "fp16_pass" in e
    )
    return results


# ---------------------------------------------------------------------------
# 2. Performance benchmark
# ---------------------------------------------------------------------------
def run_performance_test(model_path: str, num_runs: int = 100) -> dict:
    print_section("2. PERFORMANCE BENCHMARK")
    print(f"  {num_runs} runs per config, warmup=5, context=512, pred=64")

    results = {
        "test": "performance_benchmark",
        "num_runs": num_runs,
        "timestamp": datetime.now().isoformat(),
        "jemalloc": has_jemalloc(),
    }

    test_data = generate_test_data(batch_size=1, context_length=512, seed=42)

    configs = []

    # CPU baseline
    cpu_pipeline, cpu_load = load_pipeline(model_path, device="cpu")
    cpu_stats = benchmark_runs(
        cpu_pipeline, test_data, prediction_length=64, num_runs=num_runs
    )
    cpu_stats.update({"name": "CPU_FP32", "device": "cpu", "precision": "fp32"})
    configs.append(cpu_stats)
    print(
        f"  CPU_FP32: avg={cpu_stats['avg_ms']:.1f}ms, "
        f"p50={cpu_stats['p50_ms']:.1f}ms, p95={cpu_stats['p95_ms']:.1f}ms"
    )
    del cpu_pipeline

    if IS_NPU_AVAILABLE:
        active_env = setup_npu_env()
        results["optimizations"] = active_env

        for label, use_amp, precision in [
            ("NPU_FP32", False, "fp32"),
            ("NPU_FP16", True, "fp16"),
        ]:
            npu_pipeline, npu_load = load_pipeline(model_path, device="npu")
            stats = benchmark_runs(
                npu_pipeline, test_data,
                prediction_length=64,
                use_amp=use_amp,
                num_runs=num_runs,
            )
            stats.update({
                "name": label,
                "device": "npu",
                "precision": precision,
                "npu_load_time_s": round(npu_load, 2),
            })
            configs.append(stats)
            print(
                f"  {label}: avg={stats['avg_ms']:.1f}ms, "
                f"p50={stats['p50_ms']:.1f}ms, p95={stats['p95_ms']:.1f}ms, "
                f"throughput={stats['throughput_items_s']:.2f}/s"
            )
            del npu_pipeline
            torch.npu.empty_cache()

    # Speedup vs CPU
    cpu_avg = configs[0]["avg_ms"]
    for c in configs[1:]:
        c["speedup_vs_cpu"] = round(cpu_avg / c["avg_ms"], 2) if c["avg_ms"] > 0 else 0

    results["configurations"] = configs
    results["cpu_baseline_ms"] = round(cpu_avg, 2)
    return results


# ---------------------------------------------------------------------------
# 3. Batch scaling test (FP32, BS=1..256)
# ---------------------------------------------------------------------------
def run_batch_scaling_test(model_path: str) -> dict:
    print_section("3. BATCH SIZE SCALING (NPU FP32, context=512, pred=64)")
    print("  Model: T5 encoder-decoder with cross-attention batch sharing")

    results = {
        "test": "batch_scaling",
        "timestamp": datetime.now().isoformat(),
    }

    if not IS_NPU_AVAILABLE:
        results["error"] = "NPU not available"
        return results

    setup_npu_env()
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    batch_entries = []

    for bs in batch_sizes:
        print(f"\n  BS={bs}: ", end="", flush=True)
        data = generate_test_data(batch_size=bs, context_length=512, seed=42)
        pipeline, load_time = load_pipeline(model_path, device="npu")

        stats = benchmark_runs(
            pipeline, data,
            prediction_length=64,
            batch_size=bs,
            num_runs=min(50, max(10, 200 // max(bs, 1))),
            warmup=3,
        )
        per_sample_ms = stats["avg_ms"] / bs

        entry = {
            "batch_size": bs,
            "avg_total_ms": round(stats["avg_ms"], 2),
            "p50_total_ms": round(stats["p50_ms"], 2),
            "p95_total_ms": round(stats["p95_ms"], 2),
            "per_sample_ms": round(per_sample_ms, 3),
            "throughput_items_s": round(stats["throughput_items_s"], 2),
        }
        batch_entries.append(entry)
        print(
            f"total={stats['avg_ms']:.1f}ms, per_sample={per_sample_ms:.2f}ms, "
            f"throughput={stats['throughput_items_s']:.0f}/s"
        )

        del pipeline
        torch.npu.empty_cache()

    # Compute speedups vs BS=1
    bs1_throughput = batch_entries[0]["throughput_items_s"] if batch_entries else 1
    for e in batch_entries:
        e["speedup_vs_bs1"] = (
            round(e["throughput_items_s"] / bs1_throughput, 1)
            if bs1_throughput > 0 else 0
        )

    results["batch_entries"] = batch_entries
    results["peak_throughput"] = max(
        e["throughput_items_s"] for e in batch_entries
    ) if batch_entries else 0
    return results


# ---------------------------------------------------------------------------
# 4. Context length scaling test
# ---------------------------------------------------------------------------
def run_context_scaling_test(model_path: str) -> dict:
    print_section("4. CONTEXT LENGTH SCALING (NPU FP32, BS=1, pred=64)")

    results = {
        "test": "context_length_scaling",
        "timestamp": datetime.now().isoformat(),
    }

    if not IS_NPU_AVAILABLE:
        results["error"] = "NPU not available"
        return results

    setup_npu_env()
    ctx_lengths = [128, 256, 512, 1024, 2048, 4096]
    entries = []

    for ctx in ctx_lengths:
        print(f"\n  ctx={ctx}: ", end="", flush=True)
        data = generate_test_data(batch_size=1, context_length=ctx, seed=42)
        pipeline, _ = load_pipeline(model_path, device="npu")

        stats = benchmark_runs(
            pipeline, data,
            prediction_length=64,
            batch_size=1,
            num_runs=30,
            warmup=3,
        )
        entry = {
            "context_length": ctx,
            "avg_ms": round(stats["avg_ms"], 2),
            "p50_ms": round(stats["p50_ms"], 2),
            "p95_ms": round(stats["p95_ms"], 2),
            "throughput_items_s": round(stats["throughput_items_s"], 2),
        }
        entries.append(entry)
        print(f"avg={stats['avg_ms']:.1f}ms, p95={stats['p95_ms']:.1f}ms")

        del pipeline
        torch.npu.empty_cache()

    results["entries"] = entries
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_markdown_report(
    accuracy: dict,
    performance: dict,
    batch: dict,
    context: dict,
    env_info: dict,
) -> str:
    lines = []
    lines.append("# Chronos-2 Ascend NPU Evaluation Report")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**NPU**: {NPU_NAME} | **jemalloc**: {env_info.get('jemalloc', False)}")
    lines.append(f"**Active Optimizations**: TASK_QUEUE_ENABLE={env_info.get('TASK_QUEUE_ENABLE', '0')}")

    # Model info
    mi = accuracy.get("model_info", {})
    if mi:
        lines.append("\n## Model Information")
        lines.append(f"- Architecture: {mi.get('architecture', 'N/A')}")
        lines.append(f"- Parameters: {mi.get('num_params_M', 'N/A')}M")
        lines.append(f"- d_model: {mi.get('d_model', 'N/A')} | Layers: {mi.get('num_layers', 'N/A')}")
        lines.append(f"- Context Length: {mi.get('context_length', 'N/A')}")
        lines.append(f"- Default Prediction Length: {mi.get('default_prediction_length', 'N/A')}")

    # Accuracy
    lines.append("\n## 1. Accuracy Validation")
    lines.append("**Requirement**: Max Relative Error < 1%\n")
    entries = accuracy.get("entries", [])
    if entries and "npu_fp32_time_ms" in entries[0]:
        lines.append(
            "| Test | CPU (ms) | NPU FP32 (ms) | NPU FP16 (ms) | "
            "FP32 Error % | FP16 Error % | FP32 | FP16 |"
        )
        lines.append(
            "|------|----------|---------------|---------------|"
            "-------------|-------------|------|------|"
        )
        for e in entries:
            fp32_stat = "PASS" if e.get("fp32_pass") else "FAIL"
            fp16_stat = "PASS" if e.get("fp16_pass") else "FAIL"
            lines.append(
                f"| {e['name']} | {e['cpu_time_ms']:.1f} | "
                f"{e['npu_fp32_time_ms']:.1f} | {e['npu_fp16_time_ms']:.1f} | "
                f"{e['fp32_max_rel_error_pct']:.4f} | {e['fp16_max_rel_error_pct']:.2f} | "
                f"{fp32_stat} | {fp16_stat} |"
            )
        fp32_overall = "PASS" if accuracy.get("fp32_overall_pass") else "FAIL"
        fp16_overall = "PASS" if accuracy.get("fp16_overall_pass") else "FAIL"
        lines.append(f"\n**Overall FP32**: {fp32_overall} | **Overall FP16**: {fp16_overall}")

    # Performance
    lines.append("\n## 2. Performance Benchmark")
    configs = performance.get("configurations", [])
    if configs:
        lines.append(
            "| Configuration | Avg (ms) | P50 (ms) | P95 (ms) | "
            "Std (ms) | Throughput (/s) | Speedup vs CPU |"
        )
        lines.append(
            "|---------------|----------|----------|----------|"
            "----------|-----------------|----------------|"
        )
        for c in configs:
            su = c.get("speedup_vs_cpu", 1)
            lines.append(
                f"| {c['name']} | {c['avg_ms']:.1f} | {c['p50_ms']:.1f} | "
                f"{c['p95_ms']:.1f} | {c['std_ms']:.1f} | "
                f"{c['throughput_items_s']:.2f} | {su:.1f}x |"
            )

    # Batch scaling
    lines.append("\n## 3. Batch Size Scaling (NPU FP32)")
    batch_entries = batch.get("batch_entries", [])
    if batch_entries:
        lines.append(
            "| BS | Total (ms) | P50 (ms) | P95 (ms) | "
            "Per-Sample (ms) | Throughput (/s) | Speedup vs BS=1 |"
        )
        lines.append(
            "|----|-----------|----------|----------|"
            "----------------|-----------------|-----------------|"
        )
        for e in batch_entries:
            lines.append(
                f"| {e['batch_size']} | {e['avg_total_ms']:.1f} | "
                f"{e['p50_total_ms']:.1f} | {e['p95_total_ms']:.1f} | "
                f"{e['per_sample_ms']:.2f} | {e['throughput_items_s']:.0f} | "
                f"{e['speedup_vs_bs1']:.1f}x |"
            )

    # Context length scaling
    lines.append("\n## 4. Context Length Scaling (NPU FP32, BS=1, pred=64)")
    ctx_entries = context.get("entries", [])
    if ctx_entries:
        lines.append("| Context Length | Avg (ms) | P50 (ms) | P95 (ms) | Throughput (/s) |")
        lines.append("|----------------|----------|----------|----------|-----------------|")
        for e in ctx_entries:
            lines.append(
                f"| {e['context_length']} | {e['avg_ms']:.1f} | "
                f"{e['p50_ms']:.1f} | {e['p95_ms']:.1f} | {e['throughput_items_s']:.1f} |"
            )

    # Environment
    lines.append("\n## 5. Environment")
    lines.append(f"- torch: {env_info.get('torch_version', 'N/A')}")
    lines.append(f"- torch_npu: {env_info.get('torch_npu_version', 'N/A')}")
    lines.append(f"- CANN: {env_info.get('ascend_home', 'N/A')}")
    lines.append(f"- NPU: {NPU_NAME}")
    lines.append(f"- jemalloc: {env_info.get('jemalloc', False)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chronos-2 NPU Evaluation Suite")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to Chronos-2 model directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./evaluation",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--num_runs", type=int, default=100,
        help="Number of benchmark runs (default: 100)",
    )
    parser.add_argument(
        "--skip_context_scale", action="store_true",
        help="Skip context length scaling test",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect environment info
    env_info = {
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__ if IS_NPU_AVAILABLE else "N/A",
        "ascend_home": os.environ.get("ASCEND_HOME", os.environ.get("ASCEND_TOOLKIT_HOME", "N/A")),
        "jemalloc": has_jemalloc(),
        "TASK_QUEUE_ENABLE": os.environ.get("TASK_QUEUE_ENABLE", "0"),
    }

    print("=" * 70)
    print("  Chronos-2 Ascend NPU Evaluation Suite")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  NPU: {IS_NPU_AVAILABLE} ({NPU_NAME})")
    print(f"  jemalloc: {has_jemalloc()}")
    print("=" * 70)

    print("\n[Phase 1/4] Accuracy Validation: CPU vs NPU")
    accuracy = run_accuracy_test(args.model_path)

    print("\n[Phase 2/4] Performance Benchmark")
    performance = run_performance_test(args.model_path, num_runs=args.num_runs)

    print("\n[Phase 3/4] Batch Scaling Test")
    batch = run_batch_scaling_test(args.model_path)

    print("\n[Phase 4/4] Context Length Scaling Test")
    context = {}
    if not args.skip_context_scale:
        context = run_context_scaling_test(args.model_path)
    else:
        context = {"test": "context_length_scaling", "skipped": True}

    # Build full report
    full_report = {
        "accuracy": accuracy,
        "performance": performance,
        "batch_scaling": batch,
        "context_length_scaling": context,
        "environment": env_info,
    }

    # Save JSON
    json_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\nJSON report: {json_path}")

    # Save Markdown
    md_content = generate_markdown_report(accuracy, performance, batch, context, env_info)
    md_path = os.path.join(args.output_dir, "evaluation_report.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown report: {md_path}")

    # Save benchmark-only JSON for easy comparison
    bench_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(bench_path, "w") as f:
        json.dump({
            "performance": performance,
            "batch_scaling": batch,
            "context_length_scaling": context,
        }, f, indent=2, default=str)
    print(f"Benchmark results: {bench_path}")

    # Terminal summary
    print_section("EVALUATION SUMMARY")
    acc_entries = accuracy.get("entries", [])
    if acc_entries and "fp32_max_rel_error_pct" in acc_entries[0]:
        max_err = max(e["fp32_max_rel_error_pct"] for e in acc_entries)
        print(f"  Accuracy FP32: {'PASS' if accuracy.get('fp32_overall_pass') else 'FAIL'} "
              f"(max rel error = {max_err:.4f}%)")

    for c in performance.get("configurations", []):
        su = c.get("speedup_vs_cpu", 1)
        print(f"  {c['name']}: avg={c['avg_ms']:.1f}ms, "
              f"p95={c['p95_ms']:.1f}ms, throughput={c['throughput_items_s']:.2f}/s, "
              f"{su:.1f}x vs CPU")

    batch_entries = batch.get("batch_entries", [])
    if batch_entries:
        peak = max(e["throughput_items_s"] for e in batch_entries)
        print(f"  Batch scaling: peak throughput = {peak:.0f} items/s")

    ctx_entries = context.get("entries", [])
    if ctx_entries:
        lat_range = f"{min(e['avg_ms'] for e in ctx_entries):.1f} - {max(e['avg_ms'] for e in ctx_entries):.1f}ms"
        print(f"  Context length scaling (128-4096): latency range = {lat_range}")


if __name__ == "__main__":
    main()
