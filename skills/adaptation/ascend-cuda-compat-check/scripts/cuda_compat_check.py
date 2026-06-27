#!/usr/bin/env python3
"""
CUDA 兼容性检查脚本。
扫描模型仓库中的 CUDA-only 依赖、代码关键字、权重结构。

Usage:
    python cuda_compat_check.py /path/to/model_or_repo
"""

import ast
import os
import sys


CUDA_ONLY_PACKAGES = {
    "torchsparse", "torchsparse.nn", "MinkowskiEngine",
    "flash_attn", "flashattention", "xformers",
    "pytorch3d", "open3d.cuda", "bitsandbytes",
    "auto_gptq", "auto_awq", "vllm",  # vllm 需配合 vllm-ascend
}

CUDA_KEYWORDS = [
    ".cuda()", "torch.cuda", "cuda:",
    "flash_attn", "xformers.ops",
    "torchsparse", "MinkowskiEngine",
]

NON_LLM_PREFIXES = [
    "point_backbone", "point_proj", "point_encoder",
    "vision_encoder", "vision_tower", "image_encoder",
    "visual_encoder", "clip_encoder", "qformer",
]


def scan_requirements(repo_path: str) -> list[str]:
    """扫描 requirements.txt 中的 CUDA-only 依赖。"""
    req_path = os.path.join(repo_path, "requirements.txt")
    if not os.path.exists(req_path):
        return []

    found = []
    with open(req_path) as f:
        for line in f:
            line = line.strip().split("#")[0].lower()
            if not line:
                continue
            pkg = line.split("==")[0].split(">=")[0].split("[")[0].strip()
            if pkg in CUDA_ONLY_PACKAGES:
                found.append(pkg)
    return found


def scan_python_code(repo_path: str) -> list[tuple[str, str]]:
    """扫描 Python 代码中的 CUDA 关键字。"""
    matches = []
    for root, _, files in os.walk(repo_path):
        # 跳过隐藏目录和缓存
        if any(part.startswith(".") for part in root.split(os.sep)):
            continue
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for kw in CUDA_KEYWORDS:
                    if kw in content:
                        matches.append((fpath, kw))
                        break  # 每个文件只报一次
            except Exception:
                pass
    return matches


def scan_weight_keys(model_path: str) -> list[str]:
    """扫描模型权重中的非 LLM key 前缀。"""
    try:
        from safetensors import safe_open
        files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        if not files:
            return []
        keys = []
        for sf in files[:1]:  # 检查第一个文件即可
            with safe_open(os.path.join(model_path, sf), framework="pt") as f:
                keys = list(f.keys())
        non_llm = []
        for k in keys:
            for prefix in NON_LLM_PREFIXES:
                if prefix in k:
                    non_llm.append(k)
                    break
        return non_llm
    except ImportError:
        print("[WARN] 未安装 safetensors，跳过权重扫描")
        return []
    except Exception as e:
        print(f"[WARN] 权重扫描失败: {e}")
        return []


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/model_or_repo")
        sys.exit(1)

    target = sys.argv[1]
    print("=== 昇腾 NPU CUDA 兼容性检查 ===")
    print(f"目标路径: {target}")
    print()

    # 1. requirements.txt
    print("--- requirements.txt 扫描 ---")
    cuda_pkgs = scan_requirements(target)
    if cuda_pkgs:
        for pkg in cuda_pkgs:
            print(f"[WARN] 发现 CUDA-only 依赖: {pkg}")
    else:
        print("[OK]   未发现明显的 CUDA-only 依赖")

    # 2. Python 代码扫描
    print()
    print("--- Python 代码扫描 ---")
    code_matches = scan_python_code(target)
    if code_matches:
        seen_files = set()
        for fpath, kw in code_matches:
            rel = os.path.relpath(fpath, target)
            if rel not in seen_files:
                seen_files.add(rel)
                print(f"[WARN] {rel} 包含关键字: {kw}")
    else:
        print("[OK]   未发现 CUDA 相关代码关键字")

    # 3. 权重 key 扫描（如果是模型目录）
    print()
    print("--- 模型权重扫描 ---")
    if os.path.exists(os.path.join(target, "config.json")):
        non_llm = scan_weight_keys(target)
        if non_llm:
            print(f"[WARN] 发现 {len(non_llm)} 个非 LLM 权重 key，例如:")
            for k in non_llm[:5]:
                print(f"       - {k}")
        else:
            print("[OK]   权重结构为标准 LLM，无非标准 key")
    else:
        print("[INFO] 未找到 config.json，跳过权重扫描")

    # 4. 综合评估
    print()
    print("--- 兼容性评估 ---")
    issues = len(cuda_pkgs) + len(code_matches) + len(non_llm)
    if issues == 0:
        print("[PASS] 未发现明显的 CUDA-only 依赖，可尝试全量适配")
    elif len(non_llm) > 0 and not cuda_pkgs:
        print("[INFO] 权重含非 LLM key，但无 CUDA-only 依赖")
        print("       建议检查这些 key 对应的模块是否为 NPU 支持的标准算子")
    elif len(cuda_pkgs) > 0 or len(code_matches) > 0:
        print("[WARN] 发现 CUDA-only 组件，建议评估局部提取策略")
        print("       参考: 提取 LLM backbone，移除 CUDA 编码器")


if __name__ == "__main__":
    main()
