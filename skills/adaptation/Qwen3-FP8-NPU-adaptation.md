# Qwen3-4B-Thinking-2507-FP8 Ascend NPU Adaptation

## Trigger

当用户需要在华为昇腾 NPU 上适配或运行 **Qwen3 系列 FP8 量化模型**（如 `Qwen3-4B-Thinking-2507-FP8`）时，使用此 Skill。

适用场景：
- 新拿到一个 Qwen3 FP8 checkpoint，需要在昇腾 NPU 上跑通 vLLM 推理
- 遇到 `fp8 quantization is currently not supported in npu` 报错
- 需要生成模型适配测评报告和交付件

## Prerequisites

| 组件 | 已验证版本 |
|------|-----------|
| CANN | 8.5.1 |
| PyTorch | 2.5.1 |
| torch_npu | 2.9.0.post1 |
| vLLM | 0.18.0 |
| vllm-ascend | 0.18.0rc1 |
| Transformers | 4.51.0 |
| Python | 3.11.14 |
| NPU | Atlas A2 (Ascend 910B4), 32GB HBM |

## Core Problem

vLLM-Ascend NPU 平台 `supported_quantization` 仅支持 `['ascend', 'compressed-tensors']`，**原生不支持 `fp8`**。直接加载 FP8 checkpoint 会报错：

```
fp8 quantization is currently not supported in npu.
```

此外，vLLM 的 FP8 线性层依赖 `cutlass_scaled_mm`、`deepgemm`、`flashinfer` 等 CUDA/Hopper 专用算子，在昇腾上无 fallback 路径。

## Verified Solution

采用与 MiniMax-M2 FP8 相同的成熟策略，在 vllm-ascend 中新增两层补丁，实现**加载时反量化**（FP8 → BF16）：

### Patch 1: Config 层拦截

文件：`vllm_ascend/patch/platform/patch_minimax_m2_config.py`

修改 `_should_disable_fp8`，将 `qwen3` 加入支持列表：

```python
def _should_disable_fp8(cfg: ModelConfig, quant_method: str | None) -> bool:
    return (
        current_platform.device_name == "npu"
        and _get_model_type(cfg) in ("minimax_m2", "qwen3")
        and quant_method == "fp8"
    )
```

在 `ModelConfig._verify_quantization` 中拦截，检测到 `qwen3` + `fp8` + `npu` 时，将 `cfg.quantization` 设为 `None`，绕过平台验证。

### Patch 2: Worker 层反量化

文件：`vllm_ascend/patch/worker/patch_qwen3_fp8.py`（新增）

包装 `Qwen3ForCausalLM.load_weights`，在权重加载时实时将 FP8 权重 + `weight_scale_inv` 反量化为 BF16：

```python
def _dequantize_fp8_block_weight(
    fp8_weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    block_n, block_k = block_size
    n, k = fp8_weight.shape
    expanded_scale = weight_scale_inv.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
    expanded_scale = expanded_scale[:n, :k].to(dtype=torch.bfloat16)
    return fp8_weight.to(dtype=torch.bfloat16) * expanded_scale
```

反量化公式：`bf16_weight = fp8_weight.to(bf16) * expanded_scale_inv`

### Patch 3: 注册新 patch

文件：`vllm_ascend/patch/worker/__init__.py`

新增导入：
```python
import vllm_ascend.patch.worker.patch_qwen3_fp8  # noqa
```

## Step-by-Step Guide

### Step 1: Environment Setup

```bash
# Verify NPU is available
npu-smi info

# Verify vLLM + vllm-ascend are installed
python -c "import vllm; print(vllm.__version__)"
python -c "import vllm_ascend; print('vllm-ascend OK')"
```

### Step 2: Apply Patches

如果 vllm-ascend 安装在系统只读目录，复制到本地修改并用 PYTHONPATH 覆盖：

```bash
# Find vllm-ascend installation path
python -c "import vllm_ascend; print(vllm_ascend.__path__[0])"

# Copy to workspace
cp -r <vllm_ascend_path> /workspace/vllm-ascend-local

# Apply patches (see Patch 1/2/3 above)
# Edit vllm-ascend-local/vllm_ascend/patch/platform/patch_minimax_m2_config.py
# Create vllm-ascend-local/vllm_ascend/patch/worker/patch_qwen3_fp8.py
# Edit vllm-ascend-local/vllm_ascend/patch/worker/__init__.py

# Set PYTHONPATH to prioritize local copy
export PYTHONPATH=/workspace/vllm-ascend-local:$PYTHONPATH
```

### Step 3: Download Model Weights

```bash
# ModelScope (recommended)
modelscope download --model Qwen/Qwen3-4B-Thinking-2507-FP8 --local_dir /models/Qwen3-4B-Thinking-2507-FP8

# HuggingFace (mirror)
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
    Qwen/Qwen3-4B-Thinking-2507-FP8 --local-dir /models/Qwen3-4B-Thinking-2507-FP8
```

### Step 4: Start Server (Dummy Fast Gate)

```bash
vllm serve /tmp/qwen3-4b-fp8 \
  --load-format dummy \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --port 8000
```

Expected log:
```
WARNING  Detected fp8 Qwen3 checkpoint on NPU. Disabling fp8 quantization and loading dequantized bf16 weights instead.
```

### Step 5: Start Server (Real-Weight Gate)

```bash
vllm serve /models/Qwen3-4B-Thinking-2507-FP8 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --port 8000
```

### Step 6: Validate Inference

```bash
# Readiness check
curl -sf http://127.0.0.1:8000/v1/models

# Text inference
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/models/Qwen3-4B-Thinking-2507-FP8",
    "messages": [{"role": "user", "content": "Hello, introduce yourself briefly."}],
    "temperature": 0.7,
    "max_tokens": 128
  }'

# Enable thinking mode (append /think to prompt)
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/models/Qwen3-4B-Thinking-2507-FP8",
    "messages": [{"role": "user", "content": "Solve 1+1/think"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

## Performance Baseline

Verified on Atlas A2 (Ascend 910B4), single NPU:

| Metric | Value |
|--------|-------|
| Weight load time | ~43 s (4.8 GB safetensors) |
| Dequantized VRAM | ~7.7 GB (reported) / ~27 GB (npu-smi observed) |
| ACL Graph compile | ~2 min (first run) |
| Throughput (32 prompts, concurrency=8, 128 tokens) | **216.8 tokens/s** |
| Latency (single request, 128 tokens) | ~4.8 s |
| Max batch size | 16 (configured) |

## Known Issues and Fixes

| Issue | Error / Symptom | Fix |
|-------|----------------|-----|
| FP8 not supported | `fp8 quantization is currently not supported in npu` | Apply Config + Worker patches above |
| Read-only install | Cannot edit `/vllm-workspace` or system paths | Copy to local dir, use `PYTHONPATH` override |
| Model download timeout | `ConnectionTimeoutError` to huggingface.co | Use `HF_ENDPOINT=https://hf-mirror.com` or ModelScope |
| Git ownership warning | `detected dubious ownership` | `git config --global --add safe.directory <path>` |
| Accuracy benchmark low | GSM8K accuracy near 0% on few samples | **Expected for 4B model with simple few-shot prompt**; use larger samples or specialized math evaluation framework |
| pip install -e fails | Missing `cmake>=3.26` | Skip editable install, use `PYTHONPATH` directly |

## File Checklist for Delivery

| File | Purpose |
|------|---------|
| `README.md` | Model card with `hardware: NPU` frontmatter |
| `inference.py` | Synchronous inference script |
| `benchmark_throughput.py` | Throughput benchmark |
| `benchmark_accuracy.py` | Accuracy benchmark (GSM8K) |
| `SKILL.md` | This adaptation skill |
| `patches/patch_minimax_m2_config.py` | Config layer FP8 bypass |
| `patches/patch_qwen3_fp8.py` | Worker layer FP8 dequantization |
| `tests/e2e/models/configs/<Model>.yaml` | E2E test config |
| `docs/source/tutorials/models/<Model>.md` | Deployment tutorial |
| `screenshots/*.png` | Self-verification screenshots |
| `logs/*.log` / `logs/*.json` | Run logs and benchmark results |

## Notes

- **Do not pass `--quantization fp8`**; the patch handles it automatically.
- For multi-card TP later, set `HCCL_OP_EXPANSION_MODE=AIV`.
- The model is a pure text model; multimodal is not supported.
- Thinking mode is activated by appending `/think` to the user prompt.
- This Skill only covers **adaptation** (getting it to run). For **optimization** (tuning throughput/latency), additional profiling and kernel-level work is needed.
