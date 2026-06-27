# NPU 适配类型分类

根据模型接入方式（不是任务类型）分为四种，从最简单到最复杂。

---

## 类型 A：纯 PyTorch 模型（最简单）

**特征**：标准 `nn.Module`，没有自定义 CUDA kernel。

**涵盖**：timm、torchvision、transformers（大部分）、uniformer_pytorch

**适配方法**：一行代码。
```python
model = SomeModel(...)      # 正常构造
model = model.to("npu:0")   # 搬到 NPU
model.eval()                # 推理模式
```

**关注点**：只有权重下载。导入和推理都不是问题。

**通用经验**：
- `pretrained=True/weights=xxx` 优先，下载失败就 `pretrained=False/weights=None`
- 如果模型 API 不支持 `from_config`，直接构造模型类再 `load_state_dict` 也可行
- 只要模型是纯 Python + PyTorch 算子，NPU 兼容性有保证

---

## 类型 B：框架封装模型

**特征**：被框架包装，不一定暴露 `nn.Module` 接口，有自己的 device 指定方式。

**涵盖**：modelscope pipeline、insightface、rfdetr、openai-whisper、kornia

**适配方法**：找到框架的 device 参数，传入 `"npu:0"`。
```python
# modelscope
pipeline(task, model="xxx", device="npu:0")

# insightface
FaceAnalysis(providers=[("CUDAExecutionProvider", {"device_id": 0})])

# rfdetr
RFDETRBase(device="npu:0")

# openai-whisper
whisper.load_model("tiny", device="npu:0")

# kornia
model = DISK().to("npu:0")
```

**通用经验**：
- 每种框架有自己的 device 约定，**先看文档找 device 参数**
- 如果框架内部用 onnxruntime，`"npu:0"` 可能不适用
- 包装层可能把参数放 CPU 上，推理时再搬到 NPU，这是正常的

---

## 类型 C：CUDA 专用模型（需要替换）

**特征**：原始模型依赖 CUDA kernel（`.cu`、`cupy`、`cuda_extension`），NPU 上无法加载。

**涵盖**：CudaSift、FAST（原始 C++）、PnLCalib、Mamba（官方 mamba-ssm）

**适配方法**：不跑原模型，跑功能等价物。
```
原始模型（无法运行）
    ↓
找到功能等价的替代实现
    ↓
替代实现在 CPU 上做核心计算
    ↓
中间结果搬到 NPU 做张量运算
    ↓
验证 NPU 的计算能力（不是验证原始模型）
```

**通用经验**：
- 经典 CV 算法 → OpenCV 是最可靠的替代（SIFT、FAST、calibrateCamera）
- 有开源 PyTorch 复现的 → 优先用 PyTorch 复现版本
- 只有 CUDA 实现的 → 判断是否值得写 NPU kernel，大多数情况下不值得
- **评估标准不是"原始模型跑通了没"，而是"NPU 能否参与这个任务的计算管线"**

---

## 类型 D：有 NPU 算子 bug 的模型

**特征**：GPU/CPU 正常，NPU 上部分算子行为不一致。

**典型案例**：LightGlue — `F.max_pool2d(return_indices=True)` 在 NPU 上返回 int8。

**适配流程**：
```
1. 定位报错点    → 哪个 op 挂了？
2. 分析语义差异  → NPU 上的行为和 GPU/CPU 有什么不同？
3. 寻找等价实现  → 同一个数学计算，有没有不用这个 op 的写法？
4. Monkey-patch  → 在 import 后替换掉有问题的函数
```

**通用经验**：
- **不要改模型源码**，用 monkey-patch 注入修复
- 修复应该是**数学等价**的，不是近似
- 记录详细的 workaround 说明，方便 NPU 驱动更新后移除
- 常见的 NPU 算子限制：int8 截断、某些 dtype 不支持、linalg 回退 CPU、fp16 精度问题
