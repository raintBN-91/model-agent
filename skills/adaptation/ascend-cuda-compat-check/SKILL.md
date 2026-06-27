---
name: ascend-cuda-compat-check
description: >
  昇腾 NPU CUDA 组件兼容性检查与适配范围决策 Skill。
  帮助识别模型中的 CUDA-specific 依赖（如 TorchSparse、FlashAttention、xFormers 等），
  判断哪些组件可在 NPU 上原生运行，哪些需要提取/替换/绕过，从而确定合理的适配范围。
  当用户提到 CUDA-only、算子不支持、多模态编码器、模型架构不支持、
  自定义算子、适配范围评估等问题时触发。
metadata:
  short-description: 昇腾 NPU CUDA 组件兼容性检查与适配范围决策
  category: NPU-Model-Adaptation
  tags: [ascend, npu, cuda, compatibility, torchsparse, flashattention, adaptation, multimodal]
---

# 昇腾 NPU CUDA 组件兼容性检查与适配范围决策 Skill

本 Skill 帮助在将模型迁移到 Ascend NPU 之前，快速评估其 CUDA 依赖的兼容性，
确定合理的适配范围与策略。
以 `SpatialLM-Llama-1B` 的适配经验为核心案例：
其 `TorchSparse` 点云编码器为 CUDA-specific，最终适配范围缩小为仅 LLM backbone。

## 问题背景

许多 HuggingFace 模型并非纯 Transformer，而是包含 CUDA-specific 组件：
- **稀疏卷积**（如 TorchSparse）用于点云/3D 处理
- **FlashAttention** 用于高效注意力计算
- **xFormers** 用于显存优化
- **自定义 CUDA kernel** 用于特定算子加速

这些组件在 Ascend NPU 上通常：
- 直接报错 `ImportError` 或 `RuntimeError`
- 导致 `torch.cuda` 调用失败
- 使模型无法加载或推理

在投入大量适配精力前，必须预先识别这些依赖并制定策略。

---

## 常见 CUDA-only 组件清单

| 组件/库 | 典型用途 | NPU 支持状态 | 替代方案 |
|---------|---------|-------------|---------|
| **TorchSparse** | 3D 稀疏卷积（点云编码） | 不支持 | 跳过编码器，仅提取 LLM backbone |
| **FlashAttention** | 高效注意力（LLM/ViT） | 部分支持 | vLLM-Ascend 已内置 NPU 优化注意力；或使用原生 attention |
| **xFormers** | 显存优化 attention/FFN | 不支持 | 禁用 xFormers，回退到 PyTorch 原生实现 |
| **Open3D (CUDA)** | 3D 数据处理 | 不支持 | 使用 Open3D CPU 版本；或预处理后在 NPU 上推理 |
| **PyTorch3D** | 3D 深度学习 | 不支持 | 仅提取文本/图像 backbone |
| **MinkowskiEngine** | 稀疏卷积神经网络 | 不支持 | 同 TorchSparse |
| **自定义 CUDA kernel** | 模型作者编写的 `.cu` 文件 | 不支持 | 联系作者获取 NPU 版本；或重写为 PyTorch 原生算子 |
| **bitsandbytes (8-bit)** | 量化加载 | 不支持 | 使用 msmodelslim 进行 NPU 量化 |
| **auto-gptq / auto-awq** | GPTQ/AWQ 量化 | 不支持 | 使用 msmodelslim 进行 NPU 量化 |
| **vllm (原生 CUDA)** | LLM 推理引擎 | 需使用 **vllm-ascend** | 安装 `vllm-ascend` 插件 |

---

## 兼容性检查流程

### 步骤 1：识别模型架构中的非标准组件

阅读模型仓库的 `README.md` 和代码，关注以下关键词：

```
cuda
cuda()
.cuda()
torchsparse
flash_attn
xformers
custom kernel
point cloud encoder
3D convolution
sparse convolution
```

### 步骤 2：检查模型权重中的 key 前缀

```python
from safetensors import safe_open

with safe_open("model.safetensors", framework="pt") as f:
    keys = list(f.keys())

# 查找非 LLM 相关的 key
non_llm_prefixes = [k for k in keys if any(
    p in k for p in ["point_backbone", "point_proj", "vision_encoder", "image_encoder"]
)]
print("非 LLM key:", non_llm_prefixes[:10])
```

### 步骤 3：尝试加载并观察报错

```python
import torch
from transformers import AutoModel

# 强制在 CPU 上加载，观察是否有 cuda 调用
try:
    model = AutoModel.from_pretrained("path/to/model", torch_dtype=torch.float32)
    print("[OK] 模型可加载")
except Exception as e:
    print(f"[FAIL] 加载失败: {e}")
```

### 步骤 4：评估适配策略

根据检查结果，选择以下策略之一：

| 策略 | 适用场景 | 工作量 | 输出 |
|------|---------|--------|------|
| **全量适配** | 纯 Transformer，无 CUDA 依赖 | 低 | 完整模型 |
| **局部提取** | 有 CUDA 编码器，但 LLM backbone 可独立运行 | 中 | 提取 backbone + 重写 config |
| **算子替换** | 少量自定义 CUDA kernel，可用 PyTorch 重写 | 高 | 修改模型代码 |
| **方案放弃** | 核心组件完全 CUDA-specific 且无替代 | — | 建议更换模型 |

---

## 局部提取：以 SpatialLM 为例

### 背景

SpatialLM 的架构为：Point Cloud Encoder (`TorchSparse`) → MLP Projector → Llama Backbone。
其中 `TorchSparse` 是 CUDA-only 的，但 Llama backbone 是纯 Transformer，可在 NPU 上运行。

### 提取步骤

1. **识别可提取部分**
   - 检查权重 key，找出所有 `llm.*` 或 `model.*` 相关的 key
   - 确认 backbone 是标准 `LlamaForCausalLM` 结构

2. **过滤权重**
   - 移除所有 `point_backbone.*` 和 `point_proj.*` key
   - 保留 `llm.*` 或 `model.*` key

3. **重写 config.json**
   - 将 `model_type` 改为 `"llama"`
   - 将 `architectures` 改为 `["LlamaForCausalLM"]`
   - 移除多模态相关配置项

4. **验证提取结果**
   - 使用 `transformers.AutoModelForCausalLM` 加载
   - 确认无 CUDA 相关报错

### 提取脚本示例

详见本 Skill 的 `scripts/extract_backbone.py`：

```bash
python scripts/extract_backbone.py \
  --input model_weights/SpatialLM-Llama-1B \
  --output model_weights/SpatialLM-Llama-1B-LLM \
  --backbone-prefix llm \
  --target-arch LlamaForCausalLM
```

---

## 常见报错与修复

| 报错信息 | 原因 | 修复方案 |
|---------|------|---------|
| `No module named 'torchsparse'` | CUDA-only 依赖未安装 | 如非核心组件，尝试移除相关 import；如为核心组件，考虑提取 backbone |
| `RuntimeError: CUDA error` | 代码中硬编码 `.cuda()` | 替换为 `.to(device)`，并设置 `device="npu"` |
| `NotImplementedError: Could not run 'aten::xxx' with arguments from the 'CUDA' backend` | 算子注册了 CUDA 实现但未注册 NPU | 回退到 CPU 实现；或寻找 NPU 替代算子 |
| `flash_attn is only supported on CUDA` | FlashAttention 不支持 NPU | 禁用 FlashAttention，使用原生 attention |
| `xformers.ops.memory_efficient_attention` 报错 | xFormers 不支持 NPU | 设置环境变量禁用 xFormers，或修改代码回退 |
| `ImportError: cannot import name 'xxx' from 'torchsparse'` | TorchSparse 编译失败或版本不兼容 | 放弃该组件，提取可运行部分 |

---

## 决策检查清单

在决定适配策略前，确认以下问题：

- [ ] 模型 README 是否提到 CUDA 或特定 GPU 要求？
- [ ] 模型是否包含 3D/点云/图像编码器？
- [ ] `requirements.txt` 是否包含 `torchsparse`、`flash-attn`、`xformers` 等？
- [ ] 权重文件中是否有大量非 Transformer 的 key？
- [ ] 模型的核心能力是否依赖 CUDA-only 组件（如 3D 理解依赖点云编码器）？
- [ ] 是否可降级为文本-only 推理（如只使用 LLM backbone）？
- [ ] 是否有替代模型可实现相同功能且原生支持 NPU？

---

## 一键检查脚本

```bash
python scripts/cuda_compat_check.py /path/to/model/repo
```

脚本功能：
- 扫描 `requirements.txt` 中的 CUDA-only 依赖
- 检查 Python 代码中的 `.cuda()`、`torchsparse` 等关键字
- 检查权重 key 前缀，识别非 LLM 组件
- 输出兼容性评估报告

---

## 完整适配范围评估模板

```markdown
## 模型兼容性评估报告

### 模型信息
- 名称：xxx
- 架构：xxx
- 原始仓库：xxx

### CUDA 依赖检查
| 检查项 | 结果 |
|--------|------|
| requirements.txt 含 CUDA-only 库 | 是/否 |
| 代码中含 .cuda() 调用 | 是/否 |
| 权重含非 LLM key | 是/否 |
| 核心能力依赖 CUDA 组件 | 是/否 |

### 适配策略
- [ ] 全量适配
- [ ] 局部提取（backbone only）
- [ ] 算子替换
- [ ] 建议放弃

### 适配后功能
- 支持任务：文本生成 / 文本+图像 / 不支持
- 已知限制：xxx
```

---

## 参考

- TorchSparse 文档：<https://github.com/mit-han-lab/torchsparse>
- FlashAttention 文档：<https://github.com/Dao-AILab/flash-attention>
- vLLM-Ascend 架构支持列表：<https://docs.vllm.ai/projects/ascend/zh-cn/v0.18.0/>
- 本 Skill 提取脚本：`scripts/extract_backbone.py`
