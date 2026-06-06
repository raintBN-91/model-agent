---
name: chronos-2-npu
description: >
  Amazon Chronos-2 时间序列预测模型在华为昇腾 Ascend 910 NPU 上的适配与性能优化。
  8步已验证流程：环境初始化→模型下载→NPU推理→精度验证→性能基准→批次扩展(BS=1→256)→FP16/BF16评估→报告生成。
  NPU FP32 时延 23.6ms (9.6x vs CPU), 峰值吞吐 2457/s (BS=128), 精度误差 0.093% (<1%)。
  触发词: Chronos-2 + Ascend/NPU, 时间序列预测 + torch_npu, T5 + 昇腾优化,
  chronos-forecasting + 部署, batch scaling + NPU, chronos model adaptation.
metadata:
  short-description: Chronos-2 时间序列预测模型昇腾 NPU 推理适配与优化 (verified on Ascend 910)
  category: NPU-Model-Optimization
  tags: [ascend, npu, chronos-2, time-series, forecasting, pytorch, inference, optimization,
    t5, batch-scaling, torch_npu, huawei, model-adaptation, verified]
  hardware: [Ascend910, Ascend910B]
  benchmark_single_latency_ms: 23.6
  benchmark_peak_throughput: 2457
  speedup_vs_cpu: 9.6
  model_type: time-series-forecasting
  pipeline_tag: time-series-forecasting
compatibility: >
  Python 3.11+, PyTorch 2.9.0, torch_npu 2.9.0.post1,
  CANN 8.5.1+, Ascend 910/910B NPU, chronos-forecasting 2.2.2.
license: Apache-2.0
---

# Chronos-2 Ascend NPU 推理适配与优化

## 流程总览

```
Step 0 → Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6
环境检查   模型下载   NPU推理   精度验证   性能基准   批次扩展   FP16评估
```

| 你想做什么 | 去哪步 |
|-----------|--------|
| 首次跑通 NPU | Step 0→1→2 |
| 验证精度 | Step 3 |
| 测性能/最优配置 | Step 4+5 |
| 排查 FP16 不准 | Step 6 |
| 一键复现 | `scripts/run_all.sh` |

## TL;DR

Amazon Chronos-2 (119.5M, T5-based) 时间序列预测模型迁移到昇腾 910 NPU。

| 阶段 | 操作 | 时延 (ms) | 吞吐 (items/s) | 加速比 vs CPU |
|:----|:----|:--------:|:-------------:|:-----------:|
| 0 | CPU FP32 基线 | 226.8 | 4.4 | 1.0x |
| 1 | NPU FP32 迁移 | 23.6 | 42.3 | 9.6x |
| 2 | NPU FP16 (AMP) | 28.0 | 35.7 | 8.1x (变慢) |
| **3** | **+ 批处理 BS=128** | **52.1** | **2,457** | **58x vs BS=1** |

**精度**: CPU vs NPU FP32 最大相对误差 **0.093%** (< 1% 要求) ✅  
**FP16 不可行**: 分位数预测敏感 + 自回归误差累积 → 3.15%-73.48% 误差 ❌

## 一键快速开始

```bash
git clone https://gitcode.com/weixin_43499674/chronos-2-npu.git
cd chronos-2-npu && bash run_all.sh
```
> `scripts/run_all.sh` 自动完成：环境检查→下载→精度→性能→批次扩展 全流程。

## 问题 → 方案速查

| 问题 | 看 | 操作 |
|------|----|------|
| NPU 不可用 | Step 0 | `npu-smi info`, 检查 CANN 版本 |
| 模型下载失败 | Step 1 | 切换 HF 镜像或换 cache_dir |
| 推理 shape 不对 | Step 2 | `dtype=torch.float32`, `device_map=None` |
| 精度超标 (>1%) | Step 3 | 必须 FP32，禁用 FP16/BF16 |
| 性能不达预期 | Step 4+5 | TASK_QUEUE_ENABLE=1, BS≥32 |
| FP16 比 FP32 慢 | Step 6 | autocast 开销 > 计算节省 |
| NPU OOM | Step 0+2 | `pkill python3`, 设置 PYTORCH_NPU_ALLOC_CONF |
| 首次推理慢 | 限制#6 | CANN JIT 缓存，预热后正常 |

## 前置条件

- **硬件**: Ascend 910/910B NPU
- **软件**: CANN 8.5.1+, Python 3.11+, PyTorch 2.9.0 + torch_npu 2.9.0.post1
- **依赖**: `pip install chronos-forecasting>=2.0 transformers safetensors modelscope numpy`
- **权重**: ModelScope `amazon/chronos-2` (~456MB)

## 适配步骤

### Step 0: 环境初始化 ⏱️ <1min

```bash
npu-smi info                         # 期望: Ascend910/910B, Health=OK
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TASK_QUEUE_ENABLE=1 CPU_AFFINITY_CONF=0
python3 -c "import torch; import torch_npu; print(torch.npu.is_available())"
# 期望: True
```

> ✅ **通过**: `npu-smi info | grep -c OK` ≥1 且 `python3 -c "import torch_npu; assert torch.npu.device_count()>=1"` 不报错  
> ❌ **阻断**: 不满足 → 停止，检查驱动/CANN版本

**异常处理**:
- `npu-smi info` 无输出 → NPU 驱动未安装或硬件不可用
- `torch.npu.is_available()` False → CANN/torch_npu 版本不匹配 (`pip list|grep torch_npu`)
- `aclInit Invalid device ID` → `unset ASCEND_RT_VISIBLE_DEVICES`
- CANN < 8.0 → 升级到 8.5.1+

### Step 1: 下载模型 ⏱️ ~2min（取决于网速）

```bash
pip install chronos-forecasting>=2.0 transformers safetensors modelscope numpy
python3 -c "from modelscope import snapshot_download; print(snapshot_download('amazon/chronos-2', cache_dir='/tmp/chronos2_model'))"
```

> ✅ **检查点**: 模型目录下存在 `config.json` + `model.safetensors`（~456MB）

**异常处理**:
- 下载超时 → `export HF_ENDPOINT=https://hf-mirror.com` 用 HuggingFace 镜像
- 磁盘不足 → 清 `/tmp/chronos2_model/` 或换 `cache_dir`

> 📁 `scripts/download_model.py`, `references/verified_performance.json`

### Step 2: NPU 推理 ⏱️ ~30s

```python
import os, torch
os.environ['TASK_QUEUE_ENABLE'] = '1'
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained(
    '/tmp/chronos2_model/amazon/chronos-2',
    dtype=torch.float32, device_map=None)
pipeline.model = pipeline.model.to('npu:0')
pipeline.model.eval()

inputs = torch.randn(1, 1, 512)
predictions = pipeline.predict(inputs, prediction_length=64)
# shape: (1, 21, 64) = (variates, quantiles, pred_length)
```

> ✅ **通过**: `assert predictions[0].shape == (1,21,64)` 不报错，输出 `PASS: shape=(1,21,64)`  
> ❌ **阻断**: shape 不对→回 Step 0；`KeyError:'torch_dtype'`→升级 chronos-forecasting≥2.0

**预期输出**: `Model moved to NPU:0` → `predictions[0].shape = (1, 21, 64)`  
**异常处理**:
- `KeyError: 'torch_dtype'` → 升级 chronos-forecasting >=2.0
- NPU OOM → `npu-smi info` 查显存, `pkill python3` 清理, 或设 `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128`

> 📁 `scripts/inference.py`, `scripts/requirements.txt`

### Step 3: 精度验证 ⏱️ ~2min（CPU+NPU 各一次推理）

```python
cpu_p = Chronos2Pipeline.from_pretrained(model_path, dtype=torch.float32, device_map=None)
cpu_p.model = cpu_p.model.to('cpu'); cpu_p.model.eval()
with torch.no_grad():
    cpu_out = cpu_p.predict(data, prediction_length=64)[0].float().numpy()
    npu_out = npu_p.predict(data, prediction_length=64)[0].float().numpy()
max_rel_err = float(np.max(np.abs(cpu_out-npu_out) / np.maximum(np.abs(cpu_out), 1e-8)) * 100)
print(f'Max relative error: {max_rel_err:.4f}%')  # 实测: 0.0926% ✅
assert max_rel_err < 1.0, 'FAIL'
```

> ✅ **通过**: `assert max_rel_err < 1.0` 不触发，输出 `PASS: max_rel_error=0.09%`  
> ❌ **阻断**: 误差≥1%→确认 FP32 非 FP16, 检查 `torch.manual_seed(42)`

**预期输出**: `Max relative error: 0.0926%` → `PASS ✅`

> 📁 `scripts/evaluate.py` (含 3 场景精度测试)

### Step 4: 性能基准 ⏱️ ~5min

```python
import time, numpy as np
for _ in range(5): _ = pipeline.predict(inputs, prediction_length=64)
torch.npu.synchronize()
lats = []
for _ in range(100):
    torch.npu.synchronize(); t0 = time.time()
    _ = pipeline.predict(inputs, prediction_length=64)
    torch.npu.synchronize()
    lats.append((time.time()-t0)*1000)
arr = np.array(lats)
print(f'Avg={np.mean(arr):.1f}ms P50={np.percentile(arr,50):.1f}ms P95={np.percentile(arr,95):.1f}ms')
```

**预期输出**: `Avg=23.6ms P50=23.6ms P95=23.8ms throughput=42.3/s 9.6x vs CPU`

> 📁 `references/verified_performance.json`

### Step 5: 批次扩展 ⏱️ ~10min（BS=1→256 共 9 档）

Chronos-2 的 T5 架构天然批友好：多序列通过跨注意力共享编码器计算。

```python
for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    batch = torch.randn(bs, 1, 512)
    # benchmark (same pattern as Step 4)
    print(f'BS={bs}: total={elapsed:.1f}ms throughput={bs/elapsed:.0f}/s')
```

| BS | Total (ms) | Per-Sample (ms) | Throughput (/s) | Speedup |
|----|-----------|-----------------|-----------------|---------|
| 1 | 23.6 | 23.60 | 42 | 1.0x |
| 32 | 24.2 | 0.76 | 1,322 | 31.2x |
| **128** | **52.1** | **0.41** | **2,457** | **58.0x ★** |
| 256 | 106.6 | 0.42 | 2,403 | 56.7x |

> 吞吐饱和点: BS=128。BS=256 不再增长。NPU 内存恒定 ~456MB。

### Step 6: FP16/BF16 精度评估 ⏱️ ~2min

```python
with torch.npu.amp.autocast(dtype=torch.float16):
    fp16_out = npu_p.predict(data, prediction_length=64)
# 实测: FP16 max_rel_err=3.15% ❌, BF16 max_rel_err=34.89% ❌
```

**根因**: 分位数预测（21 级 0.01-0.99）对数值精度高度敏感；T5 解码器自回归生成，每步误差累积。

## 已知限制 & 昇腾调优建议

### 限制
1. **FP16/BF16 不可用**: 分位数预测敏感，必须 FP32。`set_float32_matmul_precision()` 无效
2. **torch.compile 不支持**: CANN 8.5.1 `torch.isnan` GE 未实现（升级 CANN 后可重试）
3. **torch.jit.trace 不支持**: 模型含动态控制流
4. **多 NPU 并行无收益**: 双卡开销 > 单卡批处理

### 昇腾调优
5. **jemalloc**: `LD_PRELOAD=libjemalloc.so.2` 可减 3-5% 内存抖动
6. **防 OOM**: `export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128`
7. **kernel_meta 缓存**: 若首次推理每次重启都慢，`chmod 777 kernel_meta/` 或 `export ASCEND_CACHE_PATH=/tmp/ascend_cache`
8. **监控**: 另开终端 `watch -n 1 npu-smi info` 实时看显存和温度

## 资源文件

| 文件 | 用途 |
|------|------|
| `scripts/inference.py` | 推理脚本 (CPU/NPU/批量/精度/基准) |
| `scripts/evaluate.py` | 评测脚本 (精度+性能+批次+上下文) |
| `scripts/requirements.txt` | Python 依赖清单 |
| `scripts/run_all.sh` | 一键评测流水线 |
| `references/verified_performance.json` | 已验证性能数据 (机器可读) |

## 性能基线

| 硬件平台 | 精度 | BS | 时延 (ms) | P95 (ms) | 吞吐 (/s) | 内存 (MB) |
|---------|------|----|---------|---------|----------|---------|
| Intel Xeon (CPU) | FP32 | 1 | 226.8 | 228.6 | 4.4 | N/A |
| Ascend 910 | FP32 | 1 | 23.6 | 23.8 | 42.3 | 456 |
| Ascend 910 | FP16 | 1 | 28.0 | 28.5 | 35.7 | 456 |

## 精度基线

| 测试场景 | CPU (ms) | NPU FP32 (ms) | 最大相对误差 | 达标 |
|---------|---------|---------------|------------|------|
| short (ctx=128, pred=16) | 128.9 | 23.3 | 0.0036% | ✅ |
| medium (ctx=512, pred=64) | 234.4 | 23.6 | 0.0023% | ✅ |
| long (ctx=2048, pred=128) | 605.6 | 23.8 | 0.0926% | ✅ |

## 自验报告

- **日期**: 2026-05-20 | **硬件**: Ascend 910 × 1 (Ascend910_9362)
- **软件**: CANN 8.5.1, torch 2.9.0, torch_npu 2.9.0.post1, chronos-forecasting 2.2.2
- **验证**: NPU 3/3 测试通过, max_rel_error 0.093%, 9.6x speedup
- **复现**: `bash run_all.sh` | **仓库**: https://gitcode.com/weixin_43499674/chronos-2-npu
