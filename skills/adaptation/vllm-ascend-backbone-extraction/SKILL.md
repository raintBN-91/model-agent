---
name: vllm-ascend-backbone-extraction
description: >
  从自定义架构（如多模态模型）中提取标准 LLM backbone 权重，
  使其可在 vLLM-Ascend 上直接部署的 Skill。
  适用于 backbone 为 Llama/Qwen 等标准架构，但外层被自定义 wrapper
  包裹导致 vLLM 无法识别的场景。
  当用户提到多模态模型部署昇腾、自定义架构适配、backbone 提取时触发。
metadata:
  short-description: 自定义架构模型 backbone 提取与 vLLM-Ascend 适配
  category: NPU-Model-Adaptation
  tags: [ascend, npu, vllm, adaptation, backbone, multimodal, extraction]
---

# vLLM-Ascend 自定义架构 Backbone 提取 Skill

本 Skill 指导如何从 vLLM 不原生支持的自定义架构中，提取标准的 LLM backbone
权重并重新打包，使其可在 vLLM-Ascend 上直接部署。

以 `manycore-research/SpatialLM-Llama-1B`（自定义 `spatiallm_llama` 架构，
底层 backbone 为 Llama 3.2-1B）在 Atlas 800 A2 上的适配为参考案例。

## 适用场景

| 场景 | 说明 |
|------|------|
| 多模态模型 | vision/point-cloud + LLM，encoder 为 CUDA 专用，但 LLM backbone 标准 |
| 指令微调变种 | `model_type` 被改为自定义值，但权重结构与 Llama/Qwen 一致 |
| MoE/Adapter 包装 | 额外添加了 router/adapter 权重，core LLM 部分完全标准 |

## 前置条件

| 项目 | 要求 |
|------|------|
| 模型格式 | HuggingFace Transformers 格式（含 `config.json` 和 `*.safetensors`） |
| 依赖 | `safetensors` Python 库 |
| 分析能力 | 能够查看 checkpoint keys（本 Skill 提供脚本） |

## 流程总览

```
0. 判断是否需要提取
→ 1. 分析 checkpoint 结构
→ 2. 识别 backbone 权重前缀
→ 3. 提取并过滤权重
→ 4. 重写 config.json
→ 5. 复制 tokenizer 文件
→ 6. 验证提取结果
```

---

## 0. 判断是否需要提取

**需要提取的信号**：
- `vllm serve` 报错 `ValueError: Model architecture <xxx> is not supported`
- `config.json` 中 `model_type` 不是 `llama` / `qwen2` / `gemma` 等标准值
- `config.json` 中 `architectures` 包含自定义类名（如 `SpatialLMForCausalLM`）

**不需要提取的情况**：
- 模型为原生支持的架构（如 `LlamaForCausalLM`、`Qwen2ForCausalLM`）
- 报错为算子不支持或 dtype 问题（属于其他适配问题）

---

## 1. 分析 Checkpoint 结构

使用本 Skill 提供的分析脚本查看所有权重 keys：

```bash
python scripts/analyze_keys.py --model-dir /path/to/model
```

输出示例（SpatialLM-Llama-1B）：

```
point_backbone.encoder.0.weight       (CUDA encoder，NPU 不支持)
point_backbone.encoder.1.weight
point_proj.linear.weight              (多模态 projector)
point_proj.linear.bias
model.embed_tokens.weight             (Llama backbone)
model.layers.0.self_attn.q_proj.weight
model.layers.0.mlp.gate_proj.weight
...
model.norm.weight
lm_head.weight
```

**判断规则**：
- 以 `model.` 或 `transformer.` 开头的 keys → **标准 backbone**
- `embed_tokens`, `layers`, `norm`, `lm_head` → **确认是 LLM 核心**
- 以 `point_`, `vision_`, `image_` 等开头的 keys → **多模态组件，需剥离**

---

## 2. 识别 Backbone 权重前缀

常见模型结构的 backbone 前缀对照：

| 原始架构 | Backbone 前缀 | 需剥离前缀 | 目标架构 |
|---------|--------------|-----------|---------|
| `spatiallm_llama` | `model.` | `point_backbone.*`, `point_proj.*` | `LlamaForCausalLM` |
| `llava` / `llava_next` | `language_model.` | `vision_tower.*`, `multi_modal_projector.*` | `LlamaForCausalLM` |
| `qwen2_vl` | `model.` | `visual.*`, `merger.*` | `Qwen2ForCausalLM` |
| 自定义 wrapper | `transformer.` / `model.` | 视具体情况 | 视 backbone 类型 |

---

## 3. 提取并过滤权重

使用提取脚本：

```bash
python scripts/extract_backbone.py \
  --input /path/to/original_model \
  --output /path/to/extracted_backbone \
  --strip-prefix "point_backbone. point_proj." \
  --target-arch LlamaForCausalLM \
  --target-model-type llama
```

脚本逻辑核心：

```python
from safetensors.torch import load_file, save_file

state_dict = load_file("model.safetensors")

# 过滤掉非 backbone 的 keys
exclude_prefixes = ["point_backbone.", "point_proj."]
backbone_keys = [
    k for k in state_dict.keys()
    if not any(k.startswith(p) for p in exclude_prefixes)
]
backbone_state = {k: state_dict[k] for k in backbone_keys}

save_file(backbone_state, "extracted/model.safetensors")
```

---

## 4. 重写 config.json

提取脚本会自动完成以下修改：

```python
with open("config.json", "r") as f:
    config = json.load(f)

# 改为标准架构
config["model_type"] = "llama"
config["architectures"] = ["LlamaForCausalLM"]

# 移除 auto_map（如果有）
config.pop("auto_map", None)

# 移除多模态相关字段
for key in list(config.keys()):
    if any(x in key.lower() for x in ["point", "vision", "image", "pixel"]):
        del config[key]

with open("extracted/config.json", "w") as f:
    json.dump(config, f, indent=2)
```

**关键修改点**：
- `model_type` → 标准值（`llama` / `qwen2` / `gemma`）
- `architectures` → 标准类名数组
- 移除 `auto_map`（防止 Transformers 自动加载自定义类）
- 移除多模态专属字段（避免 vLLM 解析报错）

---

## 5. 复制 Tokenizer 文件

以下文件通常可直接复用，无需修改：

```bash
cp original/tokenizer.json extracted/
cp original/tokenizer_config.json extracted/
cp original/special_tokens_map.json extracted/
cp original/generation_config.json extracted/  # 可选
```

> **注意**：如果 tokenizer 也是自定义的（如加入了 image tokens），可能需要清理 `added_tokens_decoder` 中多模态相关的条目。

---

## 6. 验证提取结果

### 6.1 结构检查

```bash
python scripts/analyze_keys.py --model-dir /path/to/extracted_backbone
```

**通过标准**：
- 所有 keys 均以标准 backbone 前缀开头
- 无多模态组件残留
- key 总数合理（应与同规模标准模型相近）

### 6.2 vLLM 加载测试

```bash
vllm serve /path/to/extracted_backbone \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --served-model-name extracted-llm
```

**通过标准**：
- vLLM 启动无 `architecture not supported` 报错
- `/v1/models` 返回 200
- `/v1/chat/completions` 可正常生成文本

### 6.3 输出质量检查

由于剥离了多模态 projector，模型**只能做文本生成**，无法处理原生的多模态输入。
验证时仅测试文本 prompt：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "extracted-llm", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

## 验收确认

- [ ] `analyze_keys.py` 确认 backbone 前缀和需剥离前缀
- [ ] `extract_backbone.py` 执行成功，输出目录生成 `model.safetensors` + `config.json`
- [ ] `config.json` 中 `model_type` 和 `architectures` 为标准值
- [ ] vLLM 启动不再报架构不支持错误
- [ ] API 端点可正常响应文本 prompt

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `RuntimeError: shape mismatch` | backbone 权重被错误过滤 | 检查 `--strip-prefix`，不要误删 `model.layers.*` |
| tokenizer 加载报错 | 自定义 added_tokens 残留 | 清理 `tokenizer_config.json` 中多模态相关 tokens |
| 生成结果为空或乱码 | `lm_head` 被误删 | 确保 `lm_head.weight` 保留在提取结果中 |
| vLLM 仍报不支持 | config.json 未正确改写 | 确认 `architectures` 为标准类名，且 `auto_map` 已移除 |

---

## 参考

- 本 Skill 脚本模板：`scripts/analyze_keys.py`、`scripts/extract_backbone.py`
- 验证案例：`manycore-research/SpatialLM-Llama-1B`（`spatiallm_llama` → `LlamaForCausalLM`）
