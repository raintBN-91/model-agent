# 代码模板与 Schema

## JSON 输出 Schema

所有适配脚本产出的 JSON 遵循统一 schema：

```json
{
  "model": "模型名（人类可读）",
  "variant": "具体变体名",
  "repo": "源仓库地址",
  "implementation": "加载方式（timm|transformers|torchvision|opencv|modelscope|...）",
  "npu_available": true,
  "npu_device": "NPU 设备名",
  "tests": {
    "<测试名>": {
      "status": "passed | failed | passed_limited | skipped",
      "input_shape": [1, 3, 224, 224],
      "output_shape": [1, 1000],
      "forward_time_s": 0.123,
      "error": "报错信息（仅失败时）"
    }
  },
  "overall": true
}
```

**设计原则**：
- `status` 四态覆盖：通过/失败/降级通过/跳过
- 每个测试独立记录，即使上一步失败，后续也尝试执行
- `overall` = all status is "passed"（`passed_limited` 和 `skipped` 也算通过）

---

## 冒烟测试模板

完整模板见 `scripts/smoke_test_template.py`。核心结构：

```
1. NPU 环境探测  → torch.npu.is_available(), device_count()
2. 模型导入       → import ModelClass
3. 模型加载       → ModelClass(...).to("npu:0").eval()
4. 前向推理       → torch.randn(batch, *shape).to("npu:0") → model(input)
5. 结果保存       → JSON 写入 <model>_result.json
```

## 推理测试模板

完整模板见 `scripts/infer_test_template.py`。与冒烟测试的区别：
- 使用真实输入数据（图片/音频文件）
- 加载预训练权重
- 输出结果可视化（CV 模型）或保存音频（语音模型）
- 记录预处理+推理全流程耗时

---

## Type A 快速模板（纯 PyTorch）

```python
import torch, torch_npu

# 1. 检查 NPU
assert torch.npu.is_available(), "NPU not available"
device = torch.device("npu:0")

# 2. 加载模型
model = YourModel.from_pretrained("model_name").to(device).eval()

# 3. 推理
dummy = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    output = model(dummy)

print(f"Output shape: {output.shape}")
print("NPU 适配成功!")
```

## Type C 快速模板（CUDA 替代）

```python
import cv2, numpy as np, torch

# 1. 先用 OpenCV 做核心计算（替代 CUDA）
img = cv2.imread("test.jpg")
result = cv2.some_cv_algorithm(img)

# 2. 中间结果搬到 NPU 做后续处理
tensor = torch.from_numpy(result).to("npu:0")
# ... NPU 上的后续计算
```
