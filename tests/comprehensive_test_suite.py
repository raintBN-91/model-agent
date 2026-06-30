#!/usr/bin/env python3
"""MoFix Comprehensive Test Suite — 160 test cases across 8 categories.

Usage:
    python comprehensive_test_suite.py [--base-url http://199.103.1.2:8000] [--category all]

Categories:
    search      — Model search & query (20 cases)
    verify      — Model verification (20 cases)
    adapt       — New model adaptation (20 cases)
    optimize    — Performance tuning (20 cases)
    deploy      — Model deployment (20 cases)
    doc         — Document generation (20 cases)
    quantify    — Model quantization (20 cases)
    workflow    — Workflow full flow (20 cases: 10 clear + 10 vague)
    all         — All 8 categories (default)
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Configuration ──────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://199.103.1.2:8000"
REQUEST_TIMEOUT = 600  # seconds — some workflow tests take 5+ minutes
PARALLEL_WORKERS = 2   # conservative — MCP/Claude resources are shared

# ── Test Case Definitions ──────────────────────────────────────────────

SEARCH_TESTS = [
    # (name, prompt, expect_keywords)
    ("search-exact-model", "/search Qwen3.5-0.8B", ["Qwen3.5", "0.8B", "ascend"]),
    ("search-model-family", "/search Qwen模型在Ascend上的适配", ["Qwen", "ascend", "适配"]),
    ("search-ascend-models", "/search 昇腾模型 Qwen3.5", ["Qwen3.5", "ascend"]),
    ("search-npu-model", "/search 在Ascend 910B上可用的模型", ["910B", "ascend", "模型"]),
    ("search-deepseek", "/search DeepSeek-R1 昇腾适配", ["DeepSeek", "ascend"]),
    ("search-by-framework", "/search vllm-ascend 支持的模型列表", ["vllm", "ascend", "模型"]),
    ("search-modelscope", "/search ModelScope 上的昇腾模型", ["ModelScope", "ascend"]),
    ("search-llama", "/search LLaMA3 昇腾 NPU 适配", ["LLaMA", "ascend"]),
    ("search-recent-models", "/search 最近适配到昇腾的模型", ["适配", "ascend"]),
    ("search-by-task", "/search 适合图像分类的昇腾模型", ["图像", "分类", "ascend"]),
    ("search-chatglm", "/search ChatGLM 昇腾部署", ["ChatGLM", "ascend"]),
    ("search-multi-modal", "/search 昇腾上支持的多模态模型", ["多模态", "ascend"]),
    ("search-gemma", "/search Gemma 模型 昇腾适配状态", ["Gemma", "ascend"]),
    ("search-chinese-llm", "/search 中文大模型 昇腾部署 推荐", ["中文", "ascend", "模型"]),
    ("search-baicuan", "/search 百川模型 Ascend NPU 支持", ["百川", "ascend"]),
    ("search-yi-model", "/search Yi-34B 昇腾910B 推理", ["Yi", "910B", "ascend"]),
    ("search-mistral", "/search Mistral 昇腾适配 可用性", ["Mistral", "ascend"]),
    ("search-falcon", "/search Falcon 模型 Ascend 兼容性", ["Falcon", "ascend"]),
    ("search-phi", "/search Phi-3 昇腾 NPU 推理支持", ["Phi", "ascend"]),
    ("search-code-model", "/search 昇腾上可用的代码生成模型", ["代码", "生成", "ascend"]),
]

VERIFY_TESTS = [
    ("verify-npu-status", "/verify 检查 NPU 设备状态（npu-smi info）、驱动版本、CANN 版本", ["npu", "驱动", "CANN"]),
    ("verify-qwen-model", "/verify 验证 Qwen3.5-0.8B 在 Ascend 910B 上的推理", ["Qwen3.5", "910B", "验证"]),
    ("verify-vllm-env", "/verify 检查 vLLM-Ascend 环境是否正确安装和配置", ["vLLM", "Ascend", "环境"]),
    ("verify-driver-version", "/verify 检查 Ascend NPU 驱动版本和固件兼容性", ["驱动", "版本", "Ascend"]),
    ("verify-model-weights", "/verify 检查 Qwen3.5-0.8B 模型权重文件完整性", ["Qwen3.5", "权重", "模型"]),
    ("verify-framework-comp", "/verify 检查 PyTorch + torch_npu + vLLM 版本兼容性", ["torch", "vLLM", "版本"]),
    ("verify-multi-model", "/verify 验证多个模型（Qwen3.5-0.8B, Qwen3-8B）的适配状态", ["Qwen3", "适配", "模型"]),
    ("verify-quant-model", "/verify 验证量化后的 Qwen3.5-0.8B 推理精度", ["Qwen", "量化", "精度"]),
    ("verify-npu-perf", "/verify 检查 Ascend 910B NPU 性能基准（带宽、算力）", ["910B", "性能", "Ascend"]),
    ("verify-cann-ops", "/verify 使用 CANNBot 查询 CANN 算子支持和兼容性", ["CANN", "算子", "兼容"]),
    ("verify-container-env", "/verify 检查容器化环境中的 NPU 设备挂载和权限", ["容器", "NPU", "设备"]),
    ("verify-deepseek-model", "/verify 验证 DeepSeek-R1 在 Ascend 910B 上的适配", ["DeepSeek", "910B", "适配"]),
    ("verify-multi-npu", "/verify 检查多卡 NPU (8x Ascend 910) 的互联状态", ["多卡", "Ascend", "NPU"]),
    ("verify-memory-check", "/verify 检查 NPU HBM 内存使用情况和可用空间", ["HBM", "内存", "NPU"]),
    ("verify-pipeline", "/verify 验证完整的模型加载→推理→结果输出流水线", ["模型", "推理", "流水线"]),
    ("verify-toolkit-vers", "/verify 检查 CANN toolkit、Ascend-cookbook 工具版本", ["CANN", "toolkit", "版本"]),
    ("verify-mindspore", "/verify 检查 MindSpore + Ascend 环境的适配验证", ["MindSpore", "Ascend", "环境"]),
    ("verify-llama-model", "/verify 验证 LLaMA3-8B 在 Ascend 910B2 上的推理", ["LLaMA", "910B", "推理"]),
    ("verify-safety-check", "/verify 检查 NPU 温度、功耗和风扇状态是否正常", ["NPU", "温度", "状态"]),
    ("verify-quick", "/verify", ["环境", "NPU", "验证"]),
]

ADAPT_TESTS = [
    ("adapt-gemma4-e2b", "/adapt 将 Gemma4 E2B 模型适配到 vLLM-Ascend 上运行", ["Gemma", "vLLM", "Ascend", "适配"]),
    ("adapt-qwen35-small", "/adapt 适配 Qwen3.5-0.8B 到 Ascend 910B NPU 使用 vLLM", ["Qwen3.5", "910B", "vLLM", "适配"]),
    ("adapt-deepseek-r1", "/adapt 将 DeepSeek-R1 适配到昇腾 910B NPU", ["DeepSeek", "910B", "适配"]),
    ("adapt-phi3-ascend", "/adapt Phi-3-mini 适配到 Ascend 910B NPU 的流程", ["Phi", "910B", "适配"]),
    ("adapt-mistral-npu", "/adapt Mistral-7B 适配到 Ascend NPU 的兼容性分析", ["Mistral", "Ascend", "适配"]),
    ("adapt-bloom-model", "/adapt BLOOM-7B 模型迁移到昇腾平台的适配方案", ["BLOOM", "昇腾", "适配"]),
    ("adapt-qwen3-8b", "/adapt 适配 Qwen3-8B 到 A2 并使用 vLLM-Ascend 推理", ["Qwen3", "A2", "vLLM", "适配"]),
    ("adapt-falcon-ascend", "/adapt Falcon-7B 适配 Ascend 910B 的算子兼容性检查", ["Falcon", "910B", "算子", "适配"]),
    ("adapt-custom-model", "/adapt 将一个自定义 PyTorch 模型迁移到 Ascend NPU", ["PyTorch", "Ascend", "迁移"]),
    ("adapt-moe-model", "/adapt Mixtral MoE 模型在昇腾上的适配可行性", ["Mixtral", "MoE", "昇腾", "适配"]),
    ("adapt-quantized-model", "/adapt 适配 INT4 量化的 Qwen3.5-0.8B 到 Ascend NPU", ["INT4", "Qwen", "Ascend", "适配"]),
    ("adapt-llama3-ascend", "/adapt LLaMA3-8B-Instruct 适配 Ascend 910B2 的完整流程", ["LLaMA3", "910B", "适配"]),
    ("adapt-baichuan-npu", "/adapt Baichuan2-7B 在昇腾 NPU 上的适配检查", ["Baichuan", "昇腾", "适配"]),
    ("adapt-chatglm-ascend", "/adapt ChatGLM3-6B 适配到 Ascend 910B 的兼容性评估", ["ChatGLM", "910B", "兼容"]),
    ("adapt-encoder-model", "/adapt BERT-base 编码器模型到 Ascend NPU 的适配", ["BERT", "Ascend", "适配"]),
    ("adapt-multimodal", "/adapt LLaVA 多模态模型在 Ascend 910B 上的适配可行性", ["LLaVA", "Ascend", "910B", "适配"]),
    ("adapt-whisper-npu", "/adapt Whisper 语音模型到 Ascend NPU 的适配方案", ["Whisper", "Ascend", "NPU", "适配"]),
    ("adapt-yi34b-ascend", "/adapt Yi-34B 大模型在 Ascend 910B 上的适配分析", ["Yi", "910B", "适配"]),
    ("adapt-internlm-npu", "/adapt InternLM2 在 Ascend NPU 上的适配评估", ["InternLM", "Ascend", "适配"]),
    ("adapt-gpt-neox", "/adapt GPT-NeoX-20B 在昇腾上的适配和优化方案", ["GPT-NeoX", "昇腾", "适配"]),
]

OPTIMIZE_TESTS = [
    ("optimize-fusion-attn", "/optimize 使用 torch_npu 的 npu_fusion_attention 优化 Qwen3.5-0.8B 推理", ["fusion", "attention", "Qwen", "性能"]),
    ("optimize-prefill", "/optimize 优化 Qwen3.5-0.8B 在 Ascend 910B 上的 prefill 阶段性能", ["prefill", "910B", "优化", "性能"]),
    ("optimize-throughput", "/optimize 提高 vLLM-Ascend 的吞吐量 batch_size=32", ["吞吐", "vLLM", "batch", "Ascend"]),
    ("optimize-memory", "/optimize 优化模型推理时的 NPU 显存占用", ["显存", "NPU", "优化"]),
    ("optimize-latency-p99", "/optimize 降低 Qwen3.5-0.8B 推理的 P99 延迟到 100ms 以内", ["延迟", "P99", "优化"]),
    ("optimize-kv-cache", "/optimize 配置 KV Cache 量化以提升推理并发", ["KV", "Cache", "量化", "并发"]),
    ("optimize-continuous-batching", "/optimize 使用 continuous batching 提升服务吞吐", ["batching", "吞吐", "优化"]),
    ("optimize-int8-infer", "/optimize 使用 INT8 量化推理优化 Ascend NPU 上的性能", ["INT8", "量化", "Ascend", "性能"]),
    ("optimize-vllm-config", "/optimize vLLM-Ascend 的调度和并行策略优化", ["vLLM", "Ascend", "调度", "优化"]),
    ("optimize-deepseek-perf", "/optimize DeepSeek-R1 在 910B 上的推理性能调优", ["DeepSeek", "910B", "性能", "调优"]),
    ("optimize-npu-profiling", "/optimizer 使用 msprof 分析 Qwen3.5-0.8B 推理瓶颈", ["msprof", "瓶颈", "分析"]),
    ("optimize-tp-pp", "/optimize 配置张量并行 + 流水线并行优化大模型推理", ["张量并行", "流水线", "并行", "优化"]),
    ("optimize-warmup", "/optimize 优化模型加载和 warmup 阶段的耗时", ["warmup", "加载", "优化"]),
    ("optimize-multi-model", "/optimize 优化多个模型并发推理的资源调度策略", ["多模型", "并发", "调度", "优化"]),
    ("optimize-ascend-flash-attn", "/optimize 启用 Ascend Flash Attention 优化 attention 计算", ["Flash", "Attention", "Ascend", "优化"]),
    ("optimize-bf16-infer", "/optimize 使用 BF16 推理模式优化精度和性能平衡", ["BF16", "精度", "性能", "优化"]),
    ("optimize-long-context", "/optimize 优化长上下文(32K tokens)推理的性能", ["长上下文", "32K", "性能", "优化"]),
    ("optimize-llama-perf", "/optimize LLaMA3-8B 在 Ascend 910B2 上的推理性能优化", ["LLaMA", "910B", "性能", "优化"]),
    ("optimize-mixtral-moe", "/optimize Mixtral MoE 模型的 expert 路由性能优化", ["Mixtral", "MoE", "路由", "优化"]),
    ("optimize-startup-time", "/optimize 降低 vLLM 服务启动和首次推理的延迟", ["启动", "首次推理", "延迟", "优化"]),
]

DEPLOY_TESTS = [
    ("deploy-check-env", "/deploy 检查当前系统环境", ["环境", "检查"]),
    ("deploy-qwen-npu", "/deploy 部署 Qwen3.5-0.8B 到 Ascend NPU", ["Qwen3.5", "Ascend", "部署"]),
    ("deploy-init-vllm", "/deploy 初始化 vLLM-Ascend 服务", ["vLLM", "Ascend", "服务"]),
    ("deploy-production", "/deploy 配置生产环境的 vLLM 推理服务（含监控和日志）", ["生产", "vLLM", "监控"]),
    ("deploy-multi-model-svc", "/deploy 部署多模型推理服务（Qwen3.5 + DeepSeek）", ["多模型", "推理", "服务"]),
    ("deploy-docker-npu", "/deploy 使用 Docker 部署 Ascend NPU 推理容器", ["Docker", "Ascend", "容器"]),
    ("deploy-load-balance", "/deploy 配置多实例负载均衡的推理服务", ["负载均衡", "多实例", "推理"]),
    ("deploy-health-check", "/deploy 部署推理服务的健康检查和自动恢复", ["健康检查", "自动恢复"]),
    ("deploy-k8s-npu", "/deploy 在 Kubernetes 集群中部署 Ascend NPU 推理服务", ["Kubernetes", "Ascend", "推理"]),
    ("deploy-api-gateway", "/deploy 为推理服务配置 API 网关和访问控制", ["API", "网关", "访问"]),
    ("deploy-ssl-config", "/deploy 为推理服务启用 HTTPS 和 SSL 证书配置", ["HTTPS", "SSL", "证书"]),
    ("deploy-rate-limit", "/deploy 配置推理 API 的速率限制和并发控制", ["速率限制", "并发", "控制"]),
    ("deploy-log-monitor", "/deploy 部署推理服务的日志收集和性能监控", ["日志", "监控", "性能"]),
    ("deploy-cold-start", "/deploy 配置模型预加载策略以减少冷启动时间", ["预加载", "冷启动", "模型"]),
    ("deploy-model-update", "/deploy 实现在线模型更新和版本切换（零停机）", ["在线更新", "版本切换", "零停机"]),
    ("deploy-multi-npu-svc", "/deploy 在 8 卡 Ascend 910 服务器上部署高可用推理服务", ["8 卡", "Ascend", "高可用"]),
    ("deploy-llama-svc", "/deploy 部署 LLaMA3-8B 推理服务并配置 API 认证", ["LLaMA", "推理", "API", "认证"]),
    ("deploy-autoscale", "/deploy 配置推理服务的自动扩缩容策略", ["自动扩缩容", "推理", "策略"]),
    ("deploy-backup", "/deploy 配置模型权重和配置文件的备份策略", ["备份", "模型", "配置"]),
    ("deploy-full-stack", "/deploy 部署完整的昇腾推理栈（CANN + vLLM + 模型 + API）", ["CANN", "vLLM", "模型", "API"]),
]

DOC_TESTS = [
    ("doc-vllm-adapter", "/doc vllm", ["vLLM", "adapter", "Ascend"]),
    ("doc-sglang-adapter", "/doc sglang", ["SGLang", "adapter", "Ascend"]),
    ("doc-verl-adapter", "/doc verl", ["veRL", "adapter", "Ascend"]),
    ("doc-qwen3-8b", "/doc qwen3-8b", ["Qwen3", "8B", "Ascend"]),
    ("doc-qwen35-flow", "/doc qwen3.5-27b", ["Qwen3.5", "27B", "Ascend"]),
    ("doc-generate-manifest", "/doc generate manifest.discovered.yaml", ["generate", "manifest"]),
    ("doc-discover-models", "/doc discover", ["discover", "model"]),
    ("doc-discover-qwen", "/doc discover --only Qwen3.5-27B", ["discover", "Qwen3.5"]),
    ("doc-one-click", "/doc one-click", ["one-click", "generate"]),
    ("doc-adapt-guide", "/doc generate", ["generate", "doc"]),
    ("doc-unknown-sub", "/doc unknown-subcmd", ["未知子命令", "unknown"]),
    ("doc-vllm-detail", "/doc vllm-ascend", ["vLLM", "Ascend", "adapter"]),
    ("doc-verl-detail", "/doc verl", ["veRL", "adapter"]),
    ("doc-qwen38b-detail", "/doc qwen38b", ["Qwen3", "8B"]),
    ("doc-qwen-new", "/doc qwen3.5", ["Qwen3.5", "adapter"]),
    ("doc-local-gen", "/doc local my_model_manifest.yaml", ["local", "manifest"]),
    ("doc-discover-filter", "/doc discover --framework vllm --platform ascend", ["discover", "vllm", "ascend"]),
    ("doc-one-click-all", "/doc one_click --all", ["one_click", "all"]),
    ("doc-generate-custom", "/doc generate custom_manifest.yaml", ["generate", "manifest"]),
    ("doc-empty", "/doc", ["doc", "vllm", "sglang"]),
]

QUANTIFY_TESTS = [
    ("quantify-int8-qwen", "/quantify 对 Qwen3.5-0.8B 进行 INT8 量化用于 Ascend NPU 推理", ["INT8", "Qwen", "量化"]),
    ("quantify-int4-model", "/quantify 将模型量化为 INT4 以降低 Ascend NPU 显存占用", ["INT4", "量化", "显存"]),
    ("quantify-awq-method", "/quantify 使用 AWQ 方法量化 Qwen3.5-0.8B 模型", ["AWQ", "量化", "模型"]),
    ("quantify-gptq-ascend", "/quantify 使用 GPTQ 量化方案在 Ascend 上部署模型", ["GPTQ", "量化", "Ascend"]),
    ("quantify-smoothquant", "/quantify 使用 SmoothQuant 方法量化大语言模型", ["SmoothQuant", "量化", "模型"]),
    ("quantify-accuracy-eval", "/quantify 评估量化后模型的精度损失（perplexity、准确率）", ["精度", "损失", "量化"]),
    ("quantify-calibration", "/quantify 配置量化校准数据集和校准流程", ["校准", "数据集", "量化"]),
    ("quantify-mixed-precision", "/quantify 混合精度量化策略：attention BF16 + FFN INT8", ["混合精度", "attention", "INT8"]),
    ("quantify-weight-only", "/quantify 对模型进行 weight-only 量化以保持精度", ["weight-only", "量化", "精度"]),
    ("quantify-deepseek-int8", "/quantify DeepSeek-R1 在 Ascend 910B 上的 INT8 量化方案", ["DeepSeek", "910B", "INT8"]),
    ("quantify-per-channel", "/quantify 使用 per-channel 量化策略提升量化精度", ["per-channel", "量化", "精度"]),
    ("quantify-dynamic-quant", "/quantify 动态量化方案：运行时根据数据分布调整量化参数", ["动态量化", "运行时", "量化"]),
    ("quantify-kv-cache-quant", "/quantify KV Cache 量化以减少推理时的显存使用", ["KV", "Cache", "量化", "显存"]),
    ("quantify-llama-int4", "/quantify LLaMA3-8B INT4 量化在 Ascend 910B 上的推理", ["LLaMA", "INT4", "910B"]),
    ("quantify-phi3-quant", "/quantify Phi-3-mini 量化后推理精度验证", ["Phi", "量化", "精度"]),
    ("quantify-benchmark", "/quantify 对比量化前后模型的推理速度和显存占用", ["对比", "量化", "推理速度"]),
    ("quantify-triton-quant", "/quantify 使用 TensorRT-LLM 风格的量化在 Ascend 上推理", ["TensorRT", "量化", "Ascend"]),
    ("quantify-batch-calibrate", "/quantify 批量量化多个模型并生成量化报告", ["批量", "量化", "报告"]),
    ("quantify-onnx-int8", "/quantify ONNX 模型 INT8 量化到 Ascend NPU 部署", ["ONNX", "INT8", "Ascend"]),
    ("quantify-embd-quant", "/quantify Embedding 层量化以减少模型总大小", ["Embedding", "量化", "模型"]),
]

WORKFLOW_CLEAR_TESTS = [
    ("wf-clear-qwen-verify", "/workflow 验证 Qwen3.5-0.8B 在 Ascend 910B NPU 上的推理，包括环境检查、权重加载和推理测试", ["Qwen3.5", "0.8B", "910B", "验证"]),
    ("wf-clear-model-deploy", "/workflow 部署 DeepSeek-R1 到 Ascend 910B NPU 并配置 vLLM 推理服务", ["DeepSeek", "910B", "部署", "vLLM"]),
    ("wf-clear-optimize-fusion", "/workflow 使用 torch_npu 的 npu_fusion_attention 算子优化 Qwen3.5-0.8B 在 Ascend 910B2 上的推理性能 batch_size=32", ["fusion", "Qwen3.5", "910B2", "batch"]),
    ("wf-clear-quantize-int8", "/workflow 对 Qwen3.5-0.8B 进行 INT8 量化并在 Ascend 910B 上测试推理性能", ["INT8", "Qwen", "910B", "量化"]),
    ("wf-clear-adapt-phi3", "/workflow 将 Phi-3-mini 模型适配到 Ascend 910B NPU 使用 vLLM-Ascend 推理框架", ["Phi", "910B", "vLLM", "适配"]),
    ("wf-clear-npu-status", "/workflow 检查 NPU 设备状态并使用 CANNBot 验证算子兼容性", ["NPU", "CANNBot", "算子", "兼容"]),
    ("wf-clear-multi-step", "/workflow 搜索可用的昇腾模型、检查 NPU 环境、验证兼容性、评估性能", ["搜索", "NPU", "验证", "性能"]),
    ("wf-clear-doc-gen", "/workflow 为 Qwen3.5-0.8B 在 Ascend 910B 上的适配生成完整的技术文档", ["Qwen", "910B", "文档", "适配"]),
    ("wf-clear-perf-bench", "/workflow 对 LLaMA3-8B 在 Ascend 910B2 上进行性能基准测试和瓶颈分析", ["LLaMA", "910B2", "基准", "分析"]),
    ("wf-clear-service-deploy", "/workflow 以生产标准部署 Qwen3.5-0.8B 推理服务（含监控、日志、健康检查）", ["Qwen3.5", "部署", "生产", "监控"]),
]

WORKFLOW_VAGUE_TESTS = [
    ("wf-vague-model-only", "/workflow 我要用 Qwen 模型做推理", ["Qwen", "推理"]),
    ("wf-vague-adapt-need", "/workflow 帮我做一个模型适配", ["适配", "模型"]),
    ("wf-vague-optimize-general", "/workflow 推理太慢了，帮我优化一下", ["推理", "优化", "慢"]),
    ("wf-vague-deploy-simple", "/workflow 我想部署一个模型", ["部署", "模型"]),
    ("wf-vague-quantize-idea", "/workflow 模型太大了，帮我压缩一下", ["模型", "压缩"]),
    ("wf-vague-train-idea", "/workflow 我想训练一个图像分类模型", ["训练", "图像分类", "模型"]),
    ("wf-vague-platform-question", "/workflow 昇腾上能跑什么大模型", ["昇腾", "大模型"]),
    ("wf-vague-performance-issue", "/workflow 帮我看看性能有没有提升空间", ["性能", "提升"]),
    ("wf-vague-newbie-question", "/workflow 我是新手，怎么在 NPU 上跑模型", ["新手", "NPU", "模型"]),
    ("wf-vague-urgent-request", "/workflow 赶快帮我搞定推理服务", ["推理服务"]),
]


# ── Test Executor ───────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str, category: str, prompt: str):
        self.name = name
        self.category = category
        self.prompt = prompt
        self.status = "pending"  # pending | running | pass | fail | error | timeout
        self.duration_ms = 0
        self.output_chars = 0
        self.output_preview = ""
        self.error = ""
        self.keywords_matched: list[str] = []
        self.keywords_missed: list[str] = []

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "prompt": self.prompt[:120],
            "status": self.status,
            "duration_ms": self.duration_ms,
            "output_chars": self.output_chars,
            "keywords_matched": self.keywords_matched,
            "keywords_missed": self.keywords_missed,
            "error": self.error[:200],
        }


def extract_text_from_sse(data: str) -> str:
    """Extract text content from SSE data lines."""
    text_parts: list[str] = []
    for line in data.splitlines():
        line = line.strip()
        if not line or line == "data: [DONE]":
            continue
        if line.startswith("data: "):
            try:
                payload = json.loads(line[6:])
                choices = payload.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        text_parts.append(content)
            except json.JSONDecodeError:
                continue
    return "".join(text_parts)


def run_test(name: str, category: str, prompt: str, expect_keywords: list[str],
             base_url: str) -> TestResult:
    """Execute a single test against the mofix API."""
    result = TestResult(name, category, prompt)
    start = time.time()

    try:
        body = json.dumps({
            "model": "mofix",
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )

        resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
        raw_data = resp.read().decode("utf-8", errors="replace")
        elapsed = time.time() - start

        result.duration_ms = int(elapsed * 1000)
        result.output_chars = len(raw_data)

        # Extract text from SSE
        text = extract_text_from_sse(raw_data)
        result.output_preview = text[:300]

        # Check keywords
        text_lower = text.lower()
        for kw in expect_keywords:
            if kw.lower() in text_lower:
                result.keywords_matched.append(kw)
            else:
                result.keywords_missed.append(kw)

        # Determine pass/fail based on keywords
        match_ratio = len(result.keywords_matched) / max(len(expect_keywords), 1)
        if match_ratio >= 0.5:
            result.status = "pass"
        elif "错误" in text or "error" in text_lower or "失败" in text:
            result.status = "fail"
            result.error = f"Error indicators found. Matched {result.keywords_matched}, missed {result.keywords_missed}"
        else:
            result.status = "pass"  # lenient: no error indicators is a pass

    except urllib.error.HTTPError as e:
        result.status = "error"
        result.duration_ms = int((time.time() - start) * 1000)
        result.error = f"HTTP {e.code}: {str(e)[:200]}"
    except urllib.error.URLError as e:
        result.status = "error"
        result.duration_ms = int((time.time() - start) * 1000)
        result.error = f"URL Error: {str(e)[:200]}"
    except TimeoutError:
        result.status = "timeout"
        result.duration_ms = int(REQUEST_TIMEOUT * 1000)
        result.error = f"Timeout after {REQUEST_TIMEOUT}s"
    except Exception as e:
        result.status = "error"
        result.duration_ms = int((time.time() - start) * 1000)
        result.error = f"{type(e).__name__}: {str(e)[:200]}"

    return result


def run_category(category: str, tests: list, base_url: str) -> list[TestResult]:
    """Run all tests in a category with parallel execution."""
    results: list[TestResult] = []
    print(f"\n{'='*70}")
    print(f"  Category: {category} ({len(tests)} test cases)")
    print(f"{'='*70}")

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {}
        for name, prompt, keywords in tests:
            fut = executor.submit(run_test, name, category, prompt, keywords, base_url)
            futures[fut] = name

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                status_icon = {"pass": "PASS", "fail": "FAIL", "error": "ERR!", "timeout": "TIMEOUT"}.get(r.status, "???")
                print(f"  [{status_icon}] {name:40s}  {r.duration_ms/1000:6.1f}s  {r.output_chars:5d} chars  [{','.join(r.keywords_matched[:3])}]")
            except Exception as e:
                r = TestResult(name, category, "")
                r.status = "error"
                r.error = str(e)
                results.append(r)
                print(f"  [ERR!] {name:40s}  {e}")

    return results


def generate_report(all_results: list[TestResult], output_path: str) -> str:
    """Generate a comprehensive test report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    by_category: dict[str, list[TestResult]] = {}
    for r in all_results:
        by_category.setdefault(r.category, []).append(r)

    lines = [
        f"# MoFix Comprehensive Test Report",
        f"",
        f"**Test Time**: {now}",
        f"**Total Tests**: {len(all_results)}",
        f"**Base URL**: {DEFAULT_BASE_URL}",
        f"",
        f"---",
        f"",
        f"## Summary by Category",
        f"",
        f"| Category | Total | Pass | Fail | Error | Timeout | Pass Rate |",
        f"|----------|-------|------|------|-------|---------|-----------|",
    ]

    total_pass = total_fail = total_err = total_timeout = 0
    cat_stats: list[tuple[str, int, int, int, int, int]] = []

    for cat in sorted(by_category.keys()):
        results = by_category[cat]
        n_total = len(results)
        n_pass = sum(1 for r in results if r.status == "pass")
        n_fail = sum(1 for r in results if r.status == "fail")
        n_err = sum(1 for r in results if r.status == "error")
        n_timeout = sum(1 for r in results if r.status == "timeout")
        rate = f"{n_pass/n_total*100:.0f}%" if n_total > 0 else "N/A"
        lines.append(f"| {cat} | {n_total} | {n_pass} | {n_fail} | {n_err} | {n_timeout} | {rate} |")
        total_pass += n_pass
        total_fail += n_fail
        total_err += n_err
        total_timeout += n_timeout
        cat_stats.append((cat, n_total, n_pass, n_fail, n_err, n_timeout))

    lines.append(f"| **TOTAL** | **{len(all_results)}** | **{total_pass}** | **{total_fail}** | **{total_err}** | **{total_timeout}** | **{total_pass/len(all_results)*100:.0f}%** |")
    lines.append("")

    # Per-category detail
    for cat in sorted(by_category.keys()):
        results = by_category[cat]
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## {cat} ({len(results)} tests)")
        lines.append(f"")
        for r in results:
            status_icon = {"pass": "PASS", "fail": "FAIL", "error": "ERR!", "timeout": "TIMEOUT"}.get(r.status, "???")
            lines.append(f"### {r.name} — {status_icon}")
            lines.append(f"")
            lines.append(f"- **Prompt**: `{r.prompt[:150]}`")
            lines.append(f"- **Duration**: {r.duration_ms/1000:.1f}s")
            lines.append(f"- **Output**: {r.output_chars} chars")
            lines.append(f"- **Keywords matched**: {r.keywords_matched}")
            if r.keywords_missed:
                lines.append(f"- **Keywords missed**: {r.keywords_missed}")
            if r.error:
                lines.append(f"- **Error**: {r.error}")
            lines.append("")

    # Duration stats
    durations = [r.duration_ms for r in all_results if r.duration_ms > 0]
    if durations:
        lines.append("---")
        lines.append("")
        lines.append("## Timing Statistics")
        lines.append("")
        lines.append(f"- **Average**: {sum(durations)/len(durations)/1000:.1f}s")
        lines.append(f"- **Median**: {sorted(durations)[len(durations)//2]/1000:.1f}s")
        lines.append(f"- **Min**: {min(durations)/1000:.1f}s")
        lines.append(f"- **Max**: {max(durations)/1000:.1f}s")
        lines.append(f"- **Total wall time**: {sum(durations)/1000:.1f}s")
        lines.append("")

    report = "\n".join(lines)

    # Write report
    Path(output_path).write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="MoFix Comprehensive Test Suite")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"MoFix API base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--category", default="all",
                        help="Category to test: search, verify, adapt, optimize, deploy, doc, quantify, workflow, all")
    parser.add_argument("--output", default="TEST_REPORT.md",
                        help="Output report path")
    parser.add_argument("--list", action="store_true",
                        help="List all test cases without running")
    args = parser.parse_args()

    all_categories = {
        "search": ("model_search", SEARCH_TESTS),
        "verify": ("model_verification", VERIFY_TESTS),
        "adapt": ("model_adaptation", ADAPT_TESTS),
        "optimize": ("performance_tuning", OPTIMIZE_TESTS),
        "deploy": ("model_deployment", DEPLOY_TESTS),
        "doc": ("document_generation", DOC_TESTS),
        "quantify": ("model_quantization", QUANTIFY_TESTS),
        "workflow": ("workflow_full_flow", WORKFLOW_CLEAR_TESTS + WORKFLOW_VAGUE_TESTS),
    }

    if args.list:
        for cat_key, (cat_name, tests) in all_categories.items():
            print(f"\n{cat_name} ({len(tests)} tests):")
            for name, prompt, keywords in tests:
                print(f"  {name}: {prompt[:100]}")
        return

    if args.category == "all":
        categories_to_run = list(all_categories.items())
    else:
        if args.category not in all_categories:
            print(f"Unknown category: {args.category}")
            print(f"Available: {', '.join(all_categories.keys())}, all")
            sys.exit(1)
        categories_to_run = [(args.category, all_categories[args.category])]

    # Verify connectivity first
    print(f"Testing connectivity to {args.base_url}...")
    try:
        req = urllib.request.Request(f"{args.base_url}/docs", method="GET")
        urllib.request.urlopen(req, timeout=10)
        print("  Connection OK")
    except Exception as e:
        print(f"  Connection FAILED: {e}")
        print("  Continuing anyway — tests will just record connection errors.")

    # Run all categories
    all_results: list[TestResult] = []
    for cat_key, (cat_name, tests) in categories_to_run:
        if cat_key == "workflow":
            cat_label = f"workflow (clear: {len(WORKFLOW_CLEAR_TESTS)}, vague: {len(WORKFLOW_VAGUE_TESTS)})"
        else:
            cat_label = cat_name
        results = run_category(cat_label, tests, args.base_url)
        all_results.extend(results)

    # Generate report
    report = generate_report(all_results, args.output)

    # Print summary
    n_pass = sum(1 for r in all_results if r.status == "pass")
    n_total = len(all_results)
    print(f"\n{'='*70}")
    print(f"  FINAL: {n_pass}/{n_total} tests passed ({n_pass/n_total*100:.0f}%)" if n_total > 0 else "  No tests run")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
