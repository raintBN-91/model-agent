#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchCore 精度与性能评测脚本
============================
对比 NPU 与 CPU 在每个推理阶段的数值精度，以及端到端异常分数一致性。
输出 JSON 报告与终端日志，便于存档与截图。

用法:
    # 完整评测（精度 + 性能）
    python benchmark.py --data data/mvtec --category bottle --runs 100

    # 仅精度测试
    python benchmark.py --data data/mvtec --category bottle --precision

    # 仅性能测试（NPU）
    python benchmark.py --data data/mvtec --category bottle --device npu --runs 100
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

from inference import PatchCoreInference


# ============================================================
# 阶段级精度对比
# ============================================================

def stage_wise_precision(
    data_dir: str,
    category: str,
    num_images: int = 10,
    coreset_ratio: float = 0.1,
) -> Dict[str, float]:
    """
    分阶段精度对比：分别在 CPU 和 NPU 上运行同一输入，
    计算特征提取、KNN 距离计算、异常分数的绝对/相对误差。
    """
    print("[INFO] 阶段级精度对比 ...")

    # 共享的 transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载测试图片
    test_dir = os.path.join(data_dir, category, "test", "good")
    image_paths = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])[:num_images]

    if not image_paths:
        raise FileNotFoundError(f"无测试图片: {test_dir}")

    print(f"[INFO] 测试图片数: {len(image_paths)}")

    results = {}

    # ---- 1. 特征提取精度 ----
    print("\n  [1/3] 特征提取精度 (CPU vs NPU)")
    engine_cpu = PatchCoreInference(device_type="cpu")
    engine_npu = PatchCoreInference(device_type="npu")

    # CPU 构建内存库并拷贝到 NPU
    engine_cpu.build_memory_bank(data_dir, category, coreset_ratio)
    engine_npu.memory_bank = engine_cpu.memory_bank.to(engine_npu.device)
    engine_npu._memory_bank_norm_sq = (engine_npu.memory_bank ** 2).sum(dim=-1)

    feat_errors = []
    knn_errors = []
    score_errors = []

    for img_path in tqdm(image_paths, desc="精度对比"):
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        # CPU 特征 + KNN + 分数
        with torch.no_grad():
            cpu_feat = engine_cpu.extract_features_tensor(tensor.to(engine_cpu.device))
            cpu_dists = engine_cpu.compute_knn_distances(cpu_feat)
            cpu_patch_scores = cpu_dists[:, :engine_cpu.n_neighbors].mean(dim=-1)
            cpu_image_score = cpu_patch_scores.max().item()

        # NPU 特征 + KNN + 分数
        npu_tensor = tensor.to(engine_npu.device)
        with torch.no_grad():
            npu_feat = engine_npu.extract_features_tensor(npu_tensor)
            npu_dists = engine_npu.compute_knn_distances(npu_feat)
            npu_patch_scores = npu_dists[:, :engine_npu.n_neighbors].mean(dim=-1)
            npu_image_score = npu_patch_scores.max().item()

        # 特征误差
        feat_diff = torch.abs(cpu_feat.cpu() - npu_feat.cpu())
        feat_errors.append({
            "max_abs": float(feat_diff.max()),
            "mean_abs": float(feat_diff.mean()),
            "rel_pct": float(feat_diff.max() / (cpu_feat.abs().max().item() + 1e-9) * 100),
        })

        # KNN 距离误差
        dist_diff = torch.abs(cpu_dists.cpu() - npu_dists.cpu())
        knn_errors.append({
            "max_abs": float(dist_diff.max()),
            "mean_abs": float(dist_diff.mean()),
            "rel_pct": float(dist_diff.max() / (cpu_dists.abs().max().item() + 1e-9) * 100),
        })

        # 异常分数误差
        abs_err = abs(cpu_image_score - npu_image_score)
        score_errors.append({
            "cpu_score": cpu_image_score,
            "npu_score": npu_image_score,
            "abs_error": abs_err,
            "rel_error_pct": abs_err / (abs(cpu_image_score) + 1e-9) * 100,
        })

    # 汇总
    results["feature_extraction"] = {
        "max_abs_error": float(np.max([e["max_abs"] for e in feat_errors])),
        "mean_abs_error": float(np.mean([e["mean_abs"] for e in feat_errors])),
        "max_rel_error_pct": float(np.max([e["rel_pct"] for e in feat_errors])),
    }
    results["knn_distance"] = {
        "max_abs_error": float(np.max([e["max_abs"] for e in knn_errors])),
        "mean_abs_error": float(np.mean([e["mean_abs"] for e in knn_errors])),
        "max_rel_error_pct": float(np.max([e["rel_pct"] for e in knn_errors])),
    }
    results["anomaly_score"] = {
        "max_abs_error": float(np.max([e["abs_error"] for e in score_errors])),
        "mean_abs_error": float(np.mean([e["abs_error"] for e in score_errors])),
        "max_rel_error_pct": float(np.max([e["rel_error_pct"] for e in score_errors])),
        "mean_rel_error_pct": float(np.mean([e["rel_error_pct"] for e in score_errors])),
        "per_image": score_errors,
    }
    results["precision_passed"] = bool(
        results["anomaly_score"]["max_rel_error_pct"] < 1.0
    )

    print(f"\n  特征提取 max_abs_error: {results['feature_extraction']['max_abs_error']:.8f}")
    print(f"  KNN 距离 max_abs_error: {results['knn_distance']['max_abs_error']:.8f}")
    print(f"  异常分数 max_rel_error: {results['anomaly_score']['max_rel_error_pct']:.4f}%")
    print(f"  精度通过: {results['precision_passed']}")

    return results


# ============================================================
# 性能基准测试
# ============================================================

def benchmark_performance(
    device_type: str = "npu",
    data_dir: str = "data/mvtec",
    category: str = "bottle",
    num_runs: int = 100,
    warmup_runs: int = 10,
    compile_model: bool = False,
    coreset_ratio: float = 0.1,
) -> Dict[str, float]:
    """
    性能基准测试。
    """
    print(f"\n[INFO] 性能基准测试 ({device_type}, {num_runs} runs)")

    engine = PatchCoreInference(device_type=device_type, compile_model=compile_model)
    engine.build_memory_bank(data_dir, category, coreset_ratio=coreset_ratio)

    dummy = torch.randn(1, 3, 224, 224, device=engine.device)

    # Warmup
    print(f"[INFO] Warmup ({warmup_runs} runs) ...")
    for _ in range(warmup_runs):
        engine.predict_tensor(dummy)

    # Timed runs
    print(f"[INFO] 计时 ({num_runs} runs) ...")
    times = []
    for _ in tqdm(range(num_runs)):
        _, _, elapsed = engine.predict_tensor(dummy)
        times.append(elapsed)

    arr = np.array(times)
    stats = {
        "device": device_type,
        "compile": compile_model,
        "runs": num_runs,
        "avg_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "throughput_img_per_s": float(1000.0 / np.mean(arr)),
    }

    print(f"\n  平均时延: {stats['avg_ms']:.2f} ms")
    print(f"  P50: {stats['p50_ms']:.2f} ms")
    print(f"  P90: {stats['p90_ms']:.2f} ms")
    print(f"  P99: {stats['p99_ms']:.2f} ms")
    print(f"  吞吐量: {stats['throughput_img_per_s']:.2f} img/s")

    return stats


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PatchCore 精度与性能评测")
    parser.add_argument("--data", type=str, default="data/mvtec", help="数据目录")
    parser.add_argument("--category", type=str, default="bottle", help="类别名")
    parser.add_argument("--device", type=str, default="npu", choices=["cpu", "npu"], help="测试设备")
    parser.add_argument("--precision", action="store_true", help="运行精度测试")
    parser.add_argument("--num_images", type=int, default=10, help="精度测试图片数")
    parser.add_argument("--runs", type=int, default=100, help="性能测试运行次数")
    parser.add_argument("--compile", action="store_true", help="启用 torch.compile")
    parser.add_argument("--coreset_ratio", type=float, default=0.1, help="Coreset 采样比例")
    parser.add_argument("--output", type=str, default=None, help="JSON 报告输出路径")
    args = parser.parse_args()

    report = {}

    # 1. 精度测试
    if args.precision:
        print("=" * 70)
        print("PatchCore 精度评测 (CPU vs NPU)")
        print("=" * 70)
        precision_result = stage_wise_precision(
            args.data, args.category, args.num_images, args.coreset_ratio
        )
        report["precision"] = precision_result

    # 2. 性能测试
    if args.device:
        print("\n" + "=" * 70)
        print(f"PatchCore 性能评测 ({args.device})")
        print("=" * 70)
        perf_result = benchmark_performance(
            device_type=args.device,
            data_dir=args.data,
            category=args.category,
            num_runs=args.runs,
            compile_model=args.compile,
            coreset_ratio=args.coreset_ratio,
        )
        report["performance"] = perf_result

    # 3. 保存报告
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] 评测报告已保存: {args.output}")

    print("\n" + "=" * 70)
    print("评测完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
