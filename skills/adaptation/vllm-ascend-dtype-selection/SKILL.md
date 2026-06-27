---
name: vllm-ascend-dtype-selection
description: >
  vLLM-Ascend 数据类型（dtype）选择与 NPU 兼容性决策 Skill。
  解决部署时因 dtype 选择不当导致的 OOM、精度异常、算子不支持等问题，
  提供 Ascend NPU 上 bfloat16 / float16 / float32 / auto 的完整支持矩阵与选择策略。
  当用户提到 dtype、bfloat16、float16、精度、OOM、数据类型选择、
  --dtype 参数等问题时触发。
metadata:
  short-description: vLLM-Ascend dtype 选择与 NPU 兼容性指南
  category: NPU-Model-Adaptation
  tags: [ascend, npu, vllm, dtype, bfloat16, float16, precision, datatype, adaptation]
---

# vLLM-Ascend 数据类型选择与 NPU 兼容性决策 Skill

本 Skill 帮助在 Ascend NPU 上部署 vLLM 时，选择正确的 `dtype` 参数。
以 `google/gemma-3-270m-it` 和 `SpatialLM-Llama-1B-LLM`
在 Atlas 800 A2 (NPU 910B4) 上的部署经验为参考。

## 问题背景

`--dtype` 是 vLLM serve 的关键参数，选择不当会导致：
- **OOM**：fp32 占用内存是 bf16 的两倍
- **精度异常**：fp16 在某些算子上容易溢出
- **运行报错**：部分算子在特定 dtype 下不被 NPU 支持
- **性能下降**：auto 模式在 NPU 上的行为可能与 CUDA 不同

---

## Ascend NPU dtype 支持矩阵

### 各代 NPU 硬件支持

| NPU 型号 | bf16 | fp16 | fp32 | 说明 |
|----------|------|------|------|------|
| 910B / 910B4 | 支持 | 支持 | 支持 | 推荐 bf16，兼具精度与效率 |
| 310P | 支持 | 支持 | 支持 | 边缘推理场景，内存受限时可用 fp16 |
| 310 | 不支持 | 支持 | 支持 | 仅支持 fp16 / fp32 |

### vLLM-Ascend dtype 行为

| dtype 参数 | NPU 行为 | 适用场景 |
|-----------|---------|---------|
| `bfloat16` | 直接启用 bf16 计算 | **强烈推荐**（910B 系列） |
| `float16` | 启用 fp16 计算 | 内存极度受限，或模型原生 fp16 |
| `float32` | 全 fp32 计算 | 调试精度问题，或极小模型 |
| `auto` | 读取 config.json 的 `torch_dtype` 字段 | 若 config 写的是 fp32 可能意外导致 OOM |

### 内存占用对比

以 1B 参数模型为例，加载时显存占用（近似值）：

| dtype | 权重占用 | KV Cache 占用 | 总占用（单 batch） |
|-------|---------|--------------|------------------|
| fp32  | ~4 GB   | ~2x          | 最大 |
| bf16  | ~2 GB   | ~1x          | 中等 |
| fp16  | ~2 GB   | ~1x          | 中等（与 bf16 相近） |

---

## dtype 选择决策树

```
开始
  │
  ▼
NPU 型号是 910B/910B4？
  ├── 是 ──> 模型原生支持 bf16？
  │             ├── 是 ──> 推荐 --dtype bfloat16
  │             └── 否 ──> 检查 config.json torch_dtype
  │                           ├── float32 ──> 考虑 --dtype bfloat16（强制转换）
  │                           └── float16 ──> --dtype float16
  │
  └── 否（如 310）──> 仅支持 fp16/fp32
                    ├── 内存充足 ──> --dtype float32（精度优先）
                    └── 内存紧张 ──> --dtype float16
```

### 快速选择指南

| 场景 | 推荐 dtype | 理由 |
|------|-----------|------|
| 910B 系列，追求均衡 | `bfloat16` | 精度接近 fp32，内存接近 fp16 |
| 910B 系列，大模型部署 | `bfloat16` | 节省内存，避免 fp16 溢出 |
| 边缘设备（310P），内存受限 | `float16` | 内存最小化 |
| 调试精度问题 | `float32` | 排除 dtype 导致的精度损失 |
| 模型 config 未标明 dtype | `auto` | 让 vLLM 自动推断（注意检查推断结果） |

---

## 各 dtype 详细说明

### bfloat16（推荐）

**优势：**
- 动态范围与 fp32 相同（8 位指数），不易溢出
- 内存占用仅为 fp32 的一半
- 在 910B 系列上有原生硬件加速

**注意：**
- 部分旧版 CANN 对 bf16 的支持可能不完整，建议使用 CANN 8.0+
- 极少数自定义算子可能未实现 bf16 kernel

**启动示例：**

```bash
vllm serve /path/to/model \
  --dtype bfloat16 \
  --tensor-parallel-size 1
```

### float16

**适用场景：**
- 边缘设备内存极度受限
- 模型权重本身就是 fp16 格式

**风险：**
- 动态范围小（5 位指数），大数值容易溢出
- LayerNorm、Softmax 等算子在 fp16 下可能出现 `NaN` 或 `Inf`

**启动示例：**

```bash
vllm serve /path/to/model \
  --dtype float16 \
  --tensor-parallel-size 1
```

### float32

**适用场景：**
- 调试阶段排除 dtype 因素
- 极小模型（如 < 1B），内存不是瓶颈

**劣势：**
- 内存占用翻倍
- 计算吞吐量低于 bf16

**启动示例：**

```bash
vllm serve /path/to/model \
  --dtype float32 \
  --tensor-parallel-size 1
```

### auto

vLLM 会读取 `config.json` 中的 `torch_dtype` 字段自动选择。

**潜在问题：**
- 部分模型 config 写的是 `float32`，会导致意外 OOM
- 建议显式指定 `--dtype`，避免依赖 auto

**检查 auto 实际选择的 dtype：**

```bash
# 启动日志中会显示类似：
# "Using dtype=torch.bfloat16"
# 或在加载模型前打印确认
python -c "
import json
with open('config.json') as f:
    c = json.load(f)
print('config torch_dtype:', c.get('torch_dtype', 'not set'))
"
```

---

## 常见报错与修复

| 报错信息 | 原因 | 修复方案 |
|---------|------|---------|
| `RuntimeError: bfloat16 is not supported on this device` | NPU 硬件或 CANN 版本不支持 bf16 | 降级到 `--dtype float16`；升级 CANN 到 8.0+ |
| `Out of memory during model loading` | dtype 占用内存过大 | 改用 `--dtype bfloat16` 或 `--dtype float16` |
| `NaN/Inf detected in output` | fp16 溢出 | 改用 `--dtype bfloat16` 或 `--dtype float32` |
| `not implemented for Half` | 某算子未实现 fp16 kernel | 改用 `--dtype bfloat16` 或 `--dtype float32` |
| `dtype mismatch` | 权重 dtype 与运行 dtype 不一致 | 确保 `--dtype` 与权重实际格式匹配，或让 vLLM 自动转换 |
| `auto selected float32 causing OOM` | config.json 中 torch_dtype 为 float32 | 显式指定 `--dtype bfloat16` |

---

## 验证当前 dtype 配置

### 启动时观察日志

vLLM 启动日志中会输出实际使用的 dtype，查找关键词：

```
Using dtype=torch.bfloat16
```

### 运行时检查

```python
import torch

# 检查模型参数 dtype
for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")
    break  # 打印第一层即可确认
```

### 一键检查脚本

```bash
python scripts/dtype_check.py /path/to/model
```

输出示例：

```
[OK]   config.json torch_dtype: bfloat16
[INFO] 推荐启动参数: --dtype bfloat16
[OK]   NPU 型号: 910B4 (支持 bf16)
```

---

## 完整配置模板

### 910B 系列推荐启动脚本

```bash
#!/bin/bash
# Atlas 800 A2 (NPU 910B4) 推荐配置

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

vllm serve /opt/atomgit/weights/gemma-3-270m-it \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \          # 910B 系列强烈推荐
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

### 边缘设备（310P）启动脚本

```bash
#!/bin/bash
# Atlas 300I Pro (NPU 310P) 内存受限配置

vllm serve /path/to/model \
  --tensor-parallel-size 1 \
  --dtype float16 \           # 内存最小化
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80
```

---

## 参考

- CANN 数据类型支持文档：<https://www.hiascend.com/document/detail/en/CANNCommunityEdition/80RC1alpha003/devguide/aclpythondevg/>
- vLLM dtype 参数文档：<https://docs.vllm.ai/en/latest/serving/env_vars.html>
- BFloat16 论文：<https://arxiv.org/abs/1905.12322>
