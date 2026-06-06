#!/usr/bin/env python3
"""
模型 dtype 配置检查脚本。
检查 config.json 中的 torch_dtype、NPU 型号、推荐 dtype。

Usage:
    python dtype_check.py /path/to/model
"""

import json
import os
import subprocess
import sys


def get_npu_model() -> str:
    """获取 NPU 型号。"""
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if "910" in line:
                    return "910B/910B4"
                if "310P" in line:
                    return "310P"
                if "310" in line:
                    return "310"
            return "unknown"
    except Exception:
        pass
    return "unknown"


def check_dtype_support(npu_model: str) -> dict:
    """返回各 dtype 在目标 NPU 上的支持状态。"""
    support = {
        "910B/910B4": {"bfloat16": True, "float16": True, "float32": True},
        "310P": {"bfloat16": True, "float16": True, "float32": True},
        "310": {"bfloat16": False, "float16": True, "float32": True},
        "unknown": {"bfloat16": False, "float16": True, "float32": True},
    }
    return support.get(npu_model, support["unknown"])


def recommend_dtype(config_dtype: str, npu_support: dict) -> str:
    """根据 config dtype 和 NPU 支持情况推荐 dtype。"""
    if config_dtype == "bfloat16" and npu_support["bfloat16"]:
        return "bfloat16"
    if config_dtype == "float16" and npu_support["float16"]:
        return "float16"
    if npu_support["bfloat16"]:
        return "bfloat16"
    if npu_support["float16"]:
        return "float16"
    return "float32"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/model")
        sys.exit(1)

    model_path = sys.argv[1]
    config_path = os.path.join(model_path, "config.json")

    print("=== vLLM-Ascend dtype 配置检查 ===")
    print()

    # 检查 config.json
    if not os.path.exists(config_path):
        print(f"[FAIL] 未找到 config.json: {config_path}")
        sys.exit(1)

    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        print(f"[FAIL] config.json 解析失败: {e}")
        sys.exit(1)

    config_dtype = config.get("torch_dtype", "未设置")
    print(f"[INFO] config.json torch_dtype: {config_dtype}")

    # 检查 NPU
    npu_model = get_npu_model()
    if npu_model == "unknown":
        print("[WARN] 无法识别 NPU 型号（npu-smi 不可用）")
    else:
        print(f"[OK]   NPU 型号: {npu_model}")

    # dtype 支持情况
    support = check_dtype_support(npu_model)
    print()
    print("--- NPU dtype 支持情况 ---")
    for dtype, ok in support.items():
        status = "[OK]" if ok else "[FAIL]"
        print(f"{status} {dtype}")

    # 推荐 dtype
    recommended = recommend_dtype(str(config_dtype), support)
    print()
    print(f"--- 推荐启动参数 ---")
    print(f"--dtype {recommended}")

    # 警告
    if str(config_dtype) == "float32" and support.get("bfloat16"):
        print()
        print("[WARN] config.json 中 torch_dtype 为 float32，可能导致 OOM")
        print("       建议在 910B 系列上显式指定 --dtype bfloat16")


if __name__ == "__main__":
    main()
