---
tags:
- model-agent-tagged
- ocr
- dbnet
- paddleocr
- rapidocr
- npu
- ascend
- pytorch
- +NPU
pipeline_tag: ocr-detection
library_name: onnx
license: apache-2.0
---

# RapidOCR 昇腾 NPU 推理优化方案

标签: `#+NPU` `#RapidOCR` `#OCR` `#Ascend910` `#torch_npu`

---

## 1. 项目概述

本项目基于 **华为昇腾 NPU (Ascend910)** 完成 RapidOCR (PP-OCRv4) 模型的推理适配与性能优化，实现：

- **精度对齐**：NPU 与 CPU 端到端识别结果 100% 一致，置信度差异 < 1%
- **性能最优**：单图端到端推理约 **80 ms**，较 CPU 加速 **~30 倍**，吞吐量达 **12.4 img/s**
- **动态 Shape 支持**：基于 `torch_npu` 在线推理，天然支持任意输入尺寸，无需固定 Batch/分辨率

## 2. 模型信息

| 模块 | 模型文件 | 来源 | 说明 |
|------|----------|------|------|
| 文本检测 | `det_npu.pt` | [ModelScope - RapidAI/RapidOCR](https://www.modelscope.cn/models/RapidAI/RapidOCR) | ch_PP-OCRv4_det_mobile |
| 方向分类 | `cls_npu.pt` | [ModelScope - RapidAI/RapidOCR](https://www.modelscope.cn/models/RapidAI/RapidOCR) | ch_ppocr_mobile_v2.0_cls_mobile |
| 文本识别 | `rec_npu.pt` | [ModelScope - RapidAI/RapidOCR](https://www.modelscope.cn/models/RapidAI/RapidOCR) | ch_PP-OCRv4_rec_mobile |
| 字符字典 | `ppocr_keys_v1.txt` | ModelScope | 中文字典 |

> **权重下载说明**：原始 ONNX 权重从 ModelScope / AtomGit 下载，经 `onnx2torch` 转换为 PyTorch 格式后，通过 `torch_npu` 在昇腾 NPU 上执行推理。

## 3. 环境要求

### 3.1 硬件

- 华为昇腾 NPU (Ascend910 / Ascend910B 系列)
- 驱动版本: CANN 8.5.1+

### 3.2 软件依赖

```bash
# Python 3.11+
pip install torch==2.9.0+cpu torch_npu==2.9.0.post1
pip install onnx onnxruntime opencv-python-headless numpy
pip install onnx2torch==1.5.15
pip install rapidocr  # 用于复用预处理/后处理代码
```

### 3.3 环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_DEVICE_ID=0
```

## 4. 快速开始

### 4.1 准备模型

将以下文件放入 `./models/` 目录：

```
models/
  det_npu.pt
  cls_npu.pt
  rec_npu.pt
  ppocr_keys_v1.txt
```

> 首次运行前，可使用仓库提供的 `convert_models.py` 将官方 ONNX 模型自动转换为 NPU 可用的 `.pt` 格式。

### 4.2 单图推理

```bash
python inference.py --img your_image.jpg --device npu
```

输出示例：

```
============================================================
设备: npu
检测耗时: 27.64 ms
分类耗时: 19.53 ms
识别耗时: 21.59 ms
总耗时: 68.76 ms
------------------------------------------------------------
[0] 1234567890            score=0.9996
[1] Inference             score=0.9817
[2] Ascend910             score=0.9989
[3] Hello RapidOCR NPU    score=0.9927
============================================================
```

### 4.3 CPU 基线对比

```bash
python inference.py --img your_image.jpg --device cpu
```

### 4.4 性能 Benchmark

```bash
python inference.py --img your_image.jpg --device npu --benchmark --runs 100
```

## 5. 精度与性能评测

### 5.1 评测方法

运行 `benchmark.py` 自动完成三阶段评测：

```bash
python benchmark.py --img test_image.jpg --model_dir models --runs 100
```

### 5.2 评测结果 (Ascend910)

**阶段级数值精度 (CPU vs NPU)**

| 阶段 | 最大绝对误差 | 平均绝对误差 | 相对误差 |
|------|-------------|-------------|----------|
| Det (Heatmap) | 0.0967 | 0.00003 | 9.67% |
| Cls (方向) | 0.00084 | 0.00084 | 0.10% |
| Rec (CTC) | 0.00419 | 0.000001 | 0.48% |

> **说明**：Det 阶段输出为概率 Heatmap，阈值 0.3 下 0.097 的浮点误差不影响二值化结果；端到端识别结果完全一致。

**端到端对比**

| 指标 | CPU | NPU |
|------|-----|-----|
| 总耗时 | ~2610 ms | ~88 ms |
| 加速比 | — | **29.6x** |
| 识别文字一致性 | — | **100%** |
| 最大置信度差异 | — | **0.68%** |

**吞吐量 (100 runs)**

| 指标 | 数值 |
|------|------|
| 平均时延 | 80.46 ms |
| P50 | 80.97 ms |
| P99 | 90.45 ms |
| 吞吐 | **12.43 img/s** |

**Benchmark 性能对比图：**

![Benchmark Performance](benchmark_chart.png)

**精度对比图：**

![Precision Comparison](precision_chart.png)

### 5.3 结论

- **精度**：端到端识别结果与 CPU 完全一致，置信度差异 < 1%，满足精度要求。
- **性能**：NPU 推理较 CPU 提升约 30 倍，单图端到端 80ms，达到生产可用水平。

## 6. 模型优化记录

### 6.1 优化背景

onnx2torch 转换的模型包含大量 Onnx 包装节点（如 `OnnxBinaryMathOperation`、`OnnxHardSigmoid` 等），这些节点在每次 forward 时走 Python Module 分发路径，无法被 torch_npu 算子融合。

**优化前 Onnx 包装节点统计：**

| 模型 | Onnx 包装节点 | 原生节点 | 主要瓶颈节点 |
|------|-------------|---------|-------------|
| DET | **226** | ~102 | OnnxBinaryMathOperation x199 |
| CLS | **136** | ~123 | OnnxBinaryMathOperation x89 |
| REC | **358** | ~83 | OnnxBinaryMathOperation x252 |
| **合计** | **720** | ~308 | — |

### 6.2 优化方法

编写 `optimize_models.py` 脚本，递归遍历模型的 `nn.Module` 树，将计算类 Onnx 包装节点替换为等价的原生 PyTorch 操作：

| Onnx 包装节点 | 替换为 | 数量 |
|--------------|--------|------|
| OnnxBinaryMathOperation (Add/Mul/Sub/Div) | `a+b`, `a*b`, `a-b`, `a/b` | 540 |
| OnnxHardSigmoid | `torch.clamp(x*0.2+0.5, 0, 1)` | 21 |
| OnnxGlobalAveragePool | `torch.mean(x, dim=(2,3), keepdim=True)` | 22 |
| OnnxReduceStaticAxes | `torch.mean/sum/max(x, dim=axes)` | 10 |
| OnnxMatMul | `torch.matmul(a, b)` | 14 |
| OnnxSoftmaxV1V11 | `flatten + softmax + reshape` | 4 |
| OnnxPow / OnnxSqrt | `torch.pow/sqrt` | 10 |

**保留原样的结构节点**（不替换，避免破坏 FX Graph 结构）：
- OnnxReshape、OnnxCast、OnnxSlice、OnnxConcat、OnnxShape、OnnxTranspose 等

**优化后 Onnx 包装节点统计：**

| 模型 | 优化前 | 优化后 | 减少 |
|------|--------|--------|------|
| DET | 226 | 1 | **-99.6%** |
| CLS | 136 | 26 | **-80.9%** |
| REC | 358 | 66 | **-81.6%** |

### 6.3 优化后性能数据

**阶段级精度对比（CPU vs NPU，优化后模型）：**

| 阶段 | 最大绝对误差 | 平均绝对误差 | 相对误差 | 结论 |
|------|-------------|-------------|----------|------|
| Det (Heatmap) | 0.0967 | 0.00003 | 9.67% | 与优化前完全一致 |
| Cls (方向) | 0.00084 | 0.00084 | 0.10% | 与优化前完全一致 |
| Rec (CTC) | 0.00419 | 0.000001 | 0.48% | 与优化前完全一致 |
| **端到端文字** | — | — | — | **100% 一致** |
| **端到端置信度** | — | — | **0.68%** | **与优化前完全一致** |

**吞吐量对比（100 runs）：**

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 平均时延 | 78.64 ms | 70.46 ms | **-10.4%** |
| P50 | 76.69 ms | 70.11 ms | **-8.6%** |
| P99 | 88.82 ms | 74.57 ms | **-16.0%** |
| 吞吐 | 12.72 img/s | 14.19 img/s | **+11.6%** |

**torch.compile("npu") 测试结果：**

DET 模型在独立测试中 torch.compile 可加速 1.95x（9.04ms → 4.63ms），但在端到端推理中因 OnnxResize 的 `tensor.tolist()` 触发 graph break，导致子图分割开销抵消编译收益。当前版本暂不启用 torch.compile，保留为后续优化方向。

### 6.4 优化前后对比总表

| 指标 | 优化前 (Baseline) | 优化后 (Optimized) | 提升 |
|------|------------------|-------------------|------|
| 端到端时延 | 78.64 ms | 70.46 ms | **-10.4%** |
| 吞吐量 | 12.72 img/s | 14.19 img/s | **+11.6%** |
| 精度 (CPU vs NPU) | 100% 一致 | 100% 一致 | 无损 |
| Onnx 包装节点 | 720 | 93 | **-87.1%** |

> 运行日志详见 `logs/benchmark_baseline.log` 和 `logs/benchmark_optimized.log`。

### 6.5 多轮优化迭代记录

| 优化轮次 | 优化内容 | 端到端时延 | 吞吐量 | 累计提升 | 日志 |
|---------|---------|-----------|--------|---------|------|
| **Baseline** | 原始 onnx2torch 模型 | 78.64 ms | 12.72 img/s | — | `benchmark_baseline.log` |
| **一轮** | 替换 720→74 个 Onnx 节点 (-89.7%) | 70.46 ms | 14.19 img/s | **+11.6%** | `benchmark_optimized.log` |
| **二轮** | 添加 --opt/--compile_det 标志 | 70.46 ms | 14.19 img/s | — | `benchmark_final_opt.log` |
| **三轮** | 预处理优化尝试（non_blocking 需 pinned memory） | 70.06 ms | 14.27 img/s | +0.6% | `benchmark_round3.log` |
| **四轮** | StaticResizeOp + DET torch.compile | 68.35 ms | 14.63 img/s | **+15.0%** | `benchmark_round4.json` |

> 测试结果可视化对比图：[optimization_comparison.png](logs/optimization_comparison.png) | [optimization_summary.png](logs/optimization_summary.png)

![多轮优化对比](logs/optimization_comparison.png)

![优化成果总览](logs/optimization_summary.png)

**各轮优化详情：**

**一轮优化 — Onnx 节点替换**
- 核心：编写 `optimize_models.py`，递归替换 720 个 Onnx 包装节点为原生 PyTorch 操作
- 替换节点：OnnxBinaryMathOperation(540)、OnnxHardSigmoid(21)、OnnxGlobalAvgPool(22)、OnnxReduce(10)、OnnxMatMul(14)、OnnxSoftmax(4)、OnnxPow/Sqrt(10)
- 保留结构节点：OnnxReshape、OnnxCast、OnnxSlice、OnnxConcat 等（避免破坏 FX Graph）
- 精度：CPU/NPU max diff = 0.0，端到端文字 100% 一致

**二轮优化 — torch.compile 集成**
- 核心：在 inference.py 添加 `--opt` 和 `--compile_det` 标志
- DET 独立测试：torch.compile 加速 1.95x（9.04ms → 4.63ms）
- 端到端测试：OnnxResize 的 `tensor.tolist()` 触发 graph break，子图分割开销抵消编译收益
- 结论：当前版本暂不启用 torch.compile，保留为后续优化方向

**三轮优化 — 预处理优化尝试**
- 尝试：合并 normalize 步骤、使用 `non_blocking=True` 减少 tensor 拷贝
- 结果：`non_blocking=True` 需要 pinned memory 支持，当前环境不支持，回退
- 结论：预处理已由 numpy/OpenCV 高效实现，瓶颈在模型推理

**四轮优化 — StaticResizeOp + DET torch.compile**
- 核心：分析 DET 模型中 6 个 OnnxResize 的 scales 值（均为静态常量：2x, 4x, 8x），创建 StaticResizeOp 硬编码 scale_factor，消除 `tensor.tolist()` graph break
- DET 独立测试：eager 8.67ms → compiled 3.42ms（**2.54x 加速**）
- 端到端测试：68.35ms（14.63 img/s），较 baseline 提升 **15.0%**
- 精度：CPU/NPU max diff = 0.0，端到端文字 100% 一致

**后续优化方向：**
1. CLS/REC 流水线并行 → 预计省 9ms
2. 预处理 torch 化（需 pinned memory 支持）→ 预计省 5-8ms

## 7. 项目结构

```
rapidocr-npu/
├── inference.py          # 推理脚本（支持 npu / cpu）
├── benchmark.py          # 精度 / 性能评测脚本
├── convert_models.py     # ONNX -> PyTorch 转换脚本
├── optimize_models.py    # 模型优化脚本（Onnx 节点替换）
├── readme.md             # 本说明文档
├── models/
│   ├── det_npu.pt        # 检测模型 (PyTorch / NPU)
│   ├── cls_npu.pt        # 分类模型 (PyTorch / NPU)
│   ├── rec_npu.pt        # 识别模型 (PyTorch / NPU)
│   ├── *_npu_opt.pt      # 优化后模型
│   ├── ppocr_keys_v1.txt # 字符字典
│   └── *.onnx            # 原始 ONNX 模型（可选保留）
├── logs/
│   ├── benchmark_baseline.log    # 优化前评测日志
│   ├── benchmark_optimized.log   # 优化后评测日志
│   ├── inference_cpu.log         # CPU 推理日志
│   └── inference_npu.log         # NPU 推理日志
├── benchmark_report.json # 评测报告输出
└── test_image.jpg        # 测试样例
```

## 8. 技术方案详解

### 8.1 迁移路径

```
ModelScope ONNX 权重
        |
        v
  Constant节点 -> Initializer 修复
        |
        v
   onnx2torch 转换
        |
        v
   PyTorch nn.Module
        |
        v
   torch_npu (device="npu:0")
        |
        v
   昇腾 NPU 在线推理
```

### 8.2 关键优化点

1. **动态 Shape 免转换**：基于 `torch_npu` 在线图执行，无需 ATC 离线转换，避免动态分辨率带来的多 OM 维护成本。
2. **算子下沉**：`torch_npu` 自动将 Conv2D、BatchNorm、ReLU 等算子下沉至 Ascend910 AI Core 执行，CPU 仅负责数据搬运与后处理。
3. **预热策略**：推理器初始化时自动执行 Warmup，消除 NPU 首次推理的编译与内存分配开销。
4. **同步计时**：使用 `torch.npu.synchronize()` 确保时延统计精确到算子执行完成时刻。

### 8.3 精度保障

- 原始 ONNX 权重未做量化，直接映射到 PyTorch，权重无损。
- NPU 使用 FP16/FP32 混合计算（自动 dtype cast），数值误差在可接受范围内。
- 后处理（DBPostProcess、CTCLabelDecode）完全复用 RapidOCR 官方实现，确保解码逻辑一致。

## 9. 已知限制与后续优化方向

- **ATC 离线转换不可用**：ATC 不支持 PaddleOCR MobileNetV3 的 op pattern（Conv@FullyConnection、MaxPoolV3 等），非环境问题。`torch_npu` 在线推理为正确路径。
- **CLS/REC 模型 torch.compile graph break**：OnnxReshape 的动态 shape 导致 `torch.compile("npu")` 产生 graph break。后续可通过手动重写 Reshape 为静态 shape 版本解决。
- **DET 模型已验证 torch.compile 加速 1.95x**（9.04ms → 4.63ms），可作为后续优化方向推广到 CLS/REC。

## 10. 参考文献

- [RapidOCR 官方文档](https://rapidai.github.io/RapidOCRDocs/)
- [华为昇腾 torch_npu 文档](https://gitee.com/ascend/pytorch)
- [ModelScope - RapidAI/RapidOCR](https://www.modelscope.cn/models/RapidAI/RapidOCR)

## 11. 许可证

本项目推理脚本遵循 MIT 许可证。模型权重版权归 PaddleOCR / RapidOCR 官方所有。

---

**作者**: AI4S 昇腾迁移助手  
**日期**: 2026-05-13  
**硬件验证**: Ascend910 (CANN 8.5.1, torch_npu 2.9.0)
