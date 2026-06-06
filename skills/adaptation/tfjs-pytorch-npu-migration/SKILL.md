---
name: tfjs-pytorch-npu-migration
description: 将 TensorFlow.js (TF.js) 模型转换为 PyTorch 并在华为昇腾 (Ascend) NPU 上完成推理验证与发布的端到端 Skill。适用于非 transformers 架构的小型模型（如 CNN 音频分类器、图像分类器）从 TF.js 生态迁移到 PyTorch + torch_npu 的场景。涵盖权重解析与维度映射、PyTorch 模型重建、NPU 功能/性能验证、以及 GitCode 模型仓库发布。
---

# TF.js 到 PyTorch 昇腾 NPU 迁移 (TF.js to PyTorch Ascend NPU Migration)

本 Skill 提供从 TensorFlow.js (tfjs-layers) 模型到 PyTorch 的完整转换链路，并基于 `torch_npu` 在华为昇腾 NPU 上完成推理验证与发布。

## 适用场景

- 需要将 TF.js 训练/导出的模型（如音频分类、图像分类）迁移到 PyTorch 生态
- 目标运行环境为华为昇腾 NPU（Ascend910 系列）
- 模型不属于 transformers/vLLM 大模型链路，需使用原生 PyTorch 推理
- 需要将适配后的模型发布为 GitCode 模型仓库

## 核心流程

```
1. 解析 TF.js 权重 ──→ 2. 重建 PyTorch 模型 ──→ 3. NPU 推理验证 ──→ 4. 自动化验证报告 ──→ 5. 发布到 GitCode
```

---

## 预检检查点

在开始迁移前，请确认以下前提条件已满足：

| 检查项 | 确认方式 | 通过标准 |
|--------|---------|---------|
| TF.js 模型文件完整 | `ls model.json weights_manifest.json weights.bin` | 三个文件均存在 |
| Python 环境就绪 | `python -c "import torch, numpy"` | 无 ImportError |
| NPU 驱动可用 | `npu-smi info` | 至少 1 张 NPU 卡状态为 OK |
| torch_npu 已安装 | `python -c "import torch_npu; print(torch_npu.__version__)"` | 正常输出版本号 |

> **暂停确认模式**：若任意检查项未通过，请暂停并修复环境问题后再继续下一步。

---

## Step 1: 解析 TF.js 权重

### 检查点 1-A：确认权重文件存在且大小合理

```bash
test -f weights.bin && test -f model.json && test -f weights_manifest.json && echo "PASS" || echo "FAIL: 缺少 TF.js 模型文件"
```

TF.js 模型通常包含以下文件：
- `model.json`：模型拓扑（层结构、参数形状）
- `weights_manifest.json`：权重清单（名称、形状、dtype）
- `weights.bin`：二进制权重数据

### 读取二进制权重

```python
import numpy as np

with open('weights.bin', 'rb') as f:
    weights_buffer = f.read()

offset = 0
state_dict = {}
for entry in weights_manifest:
    name = entry['name']      # e.g. "conv2d_1/kernel"
    shape = entry['shape']
    num_elements = np.prod(shape)
    bytes_needed = num_elements * 4  # float32
    arr = np.frombuffer(weights_buffer[offset:offset+bytes_needed], np.float32)
    arr = arr.reshape(shape)
    offset += bytes_needed
```

### 权重维度映射规则

| TF.js 层 | TF.js 权重名 | PyTorch 对应 | 维度变换 |
|----------|-------------|-------------|---------|
| Conv2D | `conv2d_N/kernel` | `conv2d_N.weight` | `[H, W, C_in, C_out]` → `[C_out, C_in, H, W]` |
| Dense | `dense_1/kernel` | `dense_1.weight` | `[in, out]` → `[out, in]`（transpose）|
| Bias | `*/bias` | `*.bias` | 直接拷贝 |

**关键细节**：
- TF.js Conv2D kernel 为 `NHWC` 格式（`[H, W, C_in, C_out]`），需 permute 为 PyTorch `NCHW`（`[C_out, C_in, H, W]`）
- TF.js Dense kernel 为 `[in_features, out_features]`，PyTorch `Linear.weight` 为 `[out_features, in_features]`，需 transpose
- Bias 向量无需变换，直接按名称映射即可

### 检查点 1-B：验证权重加载后 shape 与 manifest 一致

```python
for name, tensor in state_dict.items():
    expected = next(e['shape'] for e in weights_manifest if e['name'] == name)
    assert list(tensor.shape) == expected, f"Shape mismatch: {name}"
print("CHECKPOINT 1-B PASSED: 所有权重 shape 与 manifest 一致")
```

---

## Step 2: 重建 PyTorch 模型

### 检查点 2-A：逐层核对拓扑与参数

根据 `model.json` 中的拓扑逐层重建 `nn.Module`。重建完成后，运行以下检查：

```python
# 检查 state_dict 键名与模型定义完全匹配
missing, unexpected = model.load_state_dict(state_dict, strict=False)
assert not missing, f"Missing keys: {missing}"
assert not unexpected, f"Unexpected keys: {unexpected}"
print("CHECKPOINT 2-A PASSED: 权重与模型定义严格匹配")
```

根据 `model.json` 中的拓扑逐层重建 `nn.Module`。

### 典型 CNN 音频分类器结构

```python
import torch.nn as nn

class SoundModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(1, 8, kernel_size=(2, 8))
        self.max_pooling2d_1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv2d_2 = nn.Conv2d(8, 32, kernel_size=(2, 4))
        self.max_pooling2d_2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        # ... 更多层
        self.dense_1 = nn.Linear(704, 2000)
        self.NewHeadDense = nn.Linear(2000, 2)

    def forward(self, x):
        # x: [batch, channels, height, width]
        x = torch.relu(self.conv2d_1(x))
        x = self.max_pooling2d_1(x)
        # ...
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.dense_1(x))
        x = self.NewHeadDense(x)
        x = torch.softmax(x, dim=-1)
        return x
```

### 加载与验证

```python
model = SoundModel()
model.load_state_dict(state_dict, strict=True)
model.eval()

# 形状验证
dummy = torch.randn(1, 1, 43, 232)
with torch.no_grad():
    out = model(dummy)
assert out.shape == (1, 2)
assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-5)
```

### 保存 PyTorch 权重

```python
torch.save(model.state_dict(), 'model/sound_model.pth')
```

---

## Step 3: NPU 推理验证

### 检查点 3-A：确认推理设备与预期一致

```bash
python -c "import torch; import torch_npu; print('NPU available:', torch.npu.is_available()); print('Device count:', torch.npu.device_count())"
```

> **暂停确认模式**：如果预期在 NPU 上推理但 `torch.npu.is_available()` 为 `False`，请检查 `ASCEND_RT_VISIBLE_DEVICES` 环境变量及 CANN 驱动状态，修复后再继续。

### 推理脚本 (`inference.py`)

提供支持 CPU/NPU 切换、音频文件输入、numpy 输入、批量推理和性能统计的脚本：

```bash
# NPU 单 batch 推理
python inference.py --device npu --repeats 10 --warmup 3

# 音频文件推理（需 librosa）
python inference.py --device npu --audio sample.wav

# 批量推理
python inference.py --device npu --batch-size 16 --repeats 10
```

### 环境变量建议

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
```

---

## Step 4: 自动化验证报告

### 检查点 4-A：验证通过后才允许发布

运行 `verify_sound_model.py` 前，确认以下前置检查已就绪：

| 检查项 | 命令 | 通过标准 |
|--------|------|---------|
| 模型文件存在 | `test -f model/sound_model.pth` | 返回 0 |
| NPU 健康 | `npu-smi info | grep OK` | 至少 1 条 OK |
| 日志目录可写 | `mkdir -p logs && touch logs/.write_test && rm logs/.write_test` | 返回 0 |

> **暂停确认模式**：`verify_sound_model.py` 执行后，请检查输出的 `validation_report.json`，确认 `success` 字段为 `true` 且所有功能验证项通过，再进入发布阶段。若验证失败，请根据 `verification.log` 定位问题并修复后重新运行。

运行 `verify_sound_model.py` 执行五阶段验证流水线：

| 阶段 | 内容 |
|------|------|
| 环境预检 | `npu-smi info`、torch_npu 版本、NPU 设备识别 |
| 模型部署 | checkpoint 存在性、CPU/NPU 加载验证 |
| 功能验证 | 确定性一致性、Softmax 合法性、输出形状、CPU-NPU 数值一致性 |
| 性能基准 | 多 batch (1/4/8/16) 延迟与吞吐量测试 |
| 资源清理 | NPU cache 清理 |

### 验证产物

- `validation_report.json`：结构化 JSON 报告
- `verification.log`：带时间戳的文本日志

---

## Step 5: 发布到 GitCode 模型仓库

### 仓库结构要求

```
sound_model/
├── README.md                    # 模型卡（含 YAML frontmatter）
├── inference.py                 # NPU 推理脚本
├── convert_tfjs_to_pytorch.py   # 转换脚本
├── verify_sound_model.py        # 验证脚本
├── model/
│   ├── sound_model.pth          # PyTorch 权重
│   ├── model.json               # TF.js 原始拓扑
│   ├── weights_manifest.json
│   └── weights.bin
├── validation_report.json
└── verification.log
```

### README frontmatter 示例

```yaml
---
license: bsd-3-clause
language:
  - zh
pipeline_tag: audio-classification
tags:
  - model-agent-tagged
  - NPU
  - Ascend
  - torch-npu
  - audio-classification
hardware: NPU
---
```

### 检查点 5-A：发布前最终确认清单

```bash
echo "□ README.md 已包含 model-agent-tagged 标签"
echo "□ 所有路径均为相对路径（无绝对路径）"
echo "□ .gitignore 已排除 __pycache__ 和 .claude/"
echo "□ validation_report.json 存在且 success=true"
echo "□ 模型权重已放入 ./model/ 目录"
```

> **暂停确认模式**：以上清单全部勾选后，方可执行推送。若 `validation_report.json` 中 `success` 为 `false`，禁止发布，必须回退到 Step 4 修复问题。

### 发布步骤

1. 确保所有代码和文档使用**相对路径**（`./model/`、`./logs/` 等）
2. 通过 GitCode API 创建 `repository_type: "model"` 的仓库
3. 初始化 git、添加 `.gitignore`、提交并推送

```bash
curl -X POST -H "PRIVATE-TOKEN: <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"name":"sound_model","repository_type":"model","visibility":"public"}' \
  https://api.gitcode.com/api/v5/user/repos
```

---

## 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_dir` | string | ✅ | TF.js 模型目录（含 model.json / weights.bin） |
| `output_dir` | string | ❌ | 输出目录，默认 `./` |
| `device` | string | ❌ | 推理设备，`npu` 或 `cpu`，默认 `npu` |
| `batch_sizes` | array | ❌ | 性能测试 batch 列表，默认 `[1, 4, 8, 16]` |

---

## 注意事项

1. **输入维度对齐**
   - TF.js 默认 `channels_last`：`[batch, H, W, C]`
   - PyTorch 默认 `channels_first`：`[batch, C, H, W]`
   - 预处理脚本需显式完成 reshape

2. **预处理一致性**
   - 若原始模型训练时使用特定的特征提取逻辑（如自定义 MFCC 参数），推理时的预处理必须与训练时完全一致，否则精度会下降

3. **NPU 日志目录权限**
   - 若出现 `can not create directory: ~/ascend/log`，属于非致命警告，不影响推理结果

4. **模型体积**
   - 本 Skill 针对轻量级模型（参数量 < 1000 万），可直接加载到 NPU 内存，无需分片

---

## 边界条件与异常处理

### 常见异常场景与 Fallback 策略

| 异常场景 | 现象/报错 | Fallback 策略 |
|---------|----------|--------------|
| 权重文件缺失 | `FileNotFoundError: weights.bin` | 检查原始模型导出目录，确认 TF.js 导出时包含权重文件；若缺失需重新导出 |
| 维度映射错误 | `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | 逐层打印 shape，核对 Conv2D permute 和 Dense transpose 是否正确；对比 TF.js 与 PyTorch 每层输出 shape |
| NPU 不可用 | `torch.npu.is_available() == False` | 回退到 CPU 推理（`--device cpu`），同时检查 CANN 驱动和 `ASCEND_RT_VISIBLE_DEVICES` |
| NPU 内存不足 | `NPU out of memory` | 减小 `--batch-size`；设置 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`；如仍不足，使用 CPU 推理 |
| 预处理维度缺失 | `Conv2d` 接收 3D 而非 4D 输入 | 在 `inference.py` 中添加 `unsqueeze(1)` 或 `reshape`，确保输入为 `[batch, channels, H, W]` |
| librosa 不可用 | `ImportError: No module named librosa` | 安装 `pip install librosa`；若环境限制无法安装，改用 `--numpy` 传入预处理的 `.npy` 文件 |
| 验证报告生成失败 | `PermissionError: cannot write logs/` | 检查目录写权限，或修改 `LOG_DIR` 为当前用户有写权限的路径 |
| GitCode 推送冲突 | `failed to push some refs` | 先 `git pull` 合并远程变更，或强制推送（仅空仓库初始化时可用） |
| API 创建仓库失败 | `Project with same name already exists` | 检查该名称仓库是否已存在；若已删除需等待垃圾回收，或改用其他路径名 |

### 紧急回退命令

当 NPU 推理出现不可恢复错误时，可立即回退到 CPU 验证模型逻辑正确性：

```bash
python inference.py --device cpu --repeats 1
```

当验证流水线任何阶段失败时，可单独重试该阶段：

```bash
# 仅运行功能验证（跳过性能基准）
python verify_sound_model.py --skip-perf
```

---

## 资源清单

### 必需文件（由用户准备）

| 文件 | 来源 | 说明 |
|------|------|------|
| `model.json` | TF.js 导出 | 模型拓扑定义 |
| `weights_manifest.json` | TF.js 导出 | 权重名称、形状、dtype 清单 |
| `weights.bin` | TF.js 导出 | 二进制权重数据 |

### 本 Skill 生成的文件

| 文件 | 生成阶段 | 说明 |
|------|---------|------|
| `convert_tfjs_to_pytorch.py` | Step 1-2 | 权重解析与 PyTorch 模型重建脚本 |
| `inference.py` | Step 3 | CPU/NPU 推理脚本 |
| `verify_sound_model.py` | Step 4 | 自动化验证流水线脚本 |
| `model/sound_model.pth` | Step 2 | 转换后的 PyTorch 权重 |
| `validation_report.json` | Step 4 | 结构化验证报告 |
| `verification.log` | Step 4 | 带时间戳的验证日志 |
| `README.md` | Step 5 | 模型卡文档（含 YAML frontmatter） |
| `.gitignore` | Step 5 | Git 忽略规则 |

### 外部依赖与资源引用

| 资源 | 用途 | 获取方式 |
|------|------|---------|
| `torch` >= 2.0 | PyTorch 推理后端 | `pip install torch` |
| `torch-npu` | Ascend NPU 支持 | 参考 [昇腾 PyTorch 安装指南](https://gitee.com/ascend/pytorch) |
| `numpy` | 权重解析 | `pip install numpy` |
| `librosa`（可选） | 音频预处理 | `pip install librosa` |
| CANN >= 8.0 | NPU 驱动与运行时 | [昇腾 CANN 下载](https://www.hiascend.com/software/cann) |
| GitCode API Token | 模型仓库创建 | GitCode 个人设置 → Access Tokens |

### 参考仓库

- 本 Skill 的完整适配示例：`https://gitcode.com/Nice_try/sound_20260322-1`
- vLLM-Ascend 模型适配 Skill：`./vllm-ascend-model-adapter`
- 昇腾模型验证 Skill：`./ascend-model-verification`

---

## 参考脚本

| 脚本 | 功能 |
|------|------|
| `convert_tfjs_to_pytorch.py` | TF.js 权重解析与 PyTorch 模型重建 |
| `inference.py` | CPU/NPU 推理、音频输入、批量推理、性能统计 |
| `verify_sound_model.py` | 五阶段自动化验证流水线 |

---

## 参考文档

- [PyTorch Conv2d 文档](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [torch_npu 使用指南](https://gitee.com/ascend/pytorch)
- [GitCode 模型仓库发布](https://gitcode.com/gh_mirrors/gi/gitcode-help-docs/)
