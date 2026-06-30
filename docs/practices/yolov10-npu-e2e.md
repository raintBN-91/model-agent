# [内测挑战] YOLOv10n 昇腾 NPU 目标检测端到端部署实践

> 赛道：Agent 跑测实践 ｜ 预估积分：200（PR 合并后）
> 关联 Skill: `skills/yolov10-npu/`

## 1. 背景与目标

YOLOv10n 是清华大学开源的轻量级目标检测模型（260 万参数，COCO 80 类）。本实践将其从 CPU 迁移到华为昇腾 NPU (Ascend 910)，通过 6 阶段优化实现从 3.2 FPS 到 69.5 FPS 的端到端推理加速，加速比达 **21.5 倍**。

## 2. 环境准备

| 项目 | 版本 |
| --- | --- |
| 硬件 | Ascend 910/910B NPU（至少 1 卡空闲） |
| OS | Ubuntu 22.04 / openEuler |
| Python | 3.10+ |
| PyTorch | 2.3.1 |
| torch_npu | 2.3.1.post1 |
| CANN | 8.0.RC3+ |
| 依赖包 | pillow, numpy, torchvision |

## 3. 初始化与 NPU 检测

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0
python -c "import torch,torch_npu; print(f'NPU available: {torch.npu.is_available()}')"
```

## 4. 安装依赖与获取权重

```bash
pip install pillow numpy torchvision
wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt -O yolov10n.pt
```

**踩坑记录：**
1. `torch.load(weights, weights_only=False)` 必填，否则报安全错误
2. ultralytics 版本需 >= 8.2.0，低版本不支持 YOLOv10 架构

## 5. NPU 适配关键步骤

### 5.1 基础 NPU 推理

```python
import torch, torch_npu
from ultralytics import YOLO
model = YOLO("yolov10n.pt")
model.to("npu:0")
results = model("test.jpg")
```

### 5.2 FP16 半精度优化

```python
model = YOLO("yolov10n.pt").to("npu:0")
model.model.half()
```

**精度验证**：

```python
m_cpu = YOLO('yolov10n.pt')
m_npu = YOLO('yolov10n.pt').to('npu:0'); m_npu.model.half()
img = Image.open('test.jpg')
diff = (m_cpu(img)[0].boxes.data - m_npu(img)[0].boxes.data.cpu()).abs().max().item()
assert diff < 0.01
```

### 5.3 torch.compile NPU 后端加速

```python
model.model = torch.compile(model.model, backend="npu")
for _ in range(3): _ = model("test.jpg")  # warmup
```

### 5.4 运行时环境变量优化

```bash
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export COMBINED_ENABLE=1
export ACL_OP_COMPILER_CACHE_MODE=1
export ACL_OP_COMPILER_CACHE_DIR=/tmp/npu_op_cache
```

### 5.5 NPU 图像缩放与后处理

```python
import torch.nn.functional as F
img_tensor = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to("npu:0")
img_tensor = F.interpolate(img_tensor, size=(640,640))
```

## 6. 精度与性能验证

| 阶段 | 配置 | E2E 延迟 (ms) | FPS | 加速比 |
|:----:|:---|:-----------:|:---:|:------:|
| 0 | CPU FP32 基线 | 309.73 | 3.2 | 1.0x |
| 1 | NPU FP32 推理 | 20.67 | 48.4 | 15.1x |
| 2 | + FP16 半精度 | 17.53 | 57.0 | 17.8x |
| 3 | + torch.compile | 16.60 | 60.2 | 18.8x |
| **4** | **+ 流式流水线** | **14.38** | **69.5** | **21.5x** |

CPU FP32 与 NPU FP16 推理的检测框坐标最大绝对误差 < 0.01，满足生产部署精度要求。

## 7. 流式流水线（可选进阶）

```python
from queue import Queue
import threading

def predict_stream(self, images):
    q = Queue(maxsize=2)
    def _prefetch():
        for img in images: q.put(self.preprocess(img))
        q.put(None)
    threading.Thread(target=_prefetch, daemon=True).start()
    for item in iter(q.get, None):
        tensor, shape, pad = item
        with torch.no_grad(): preds = self.model(tensor)
        yield self.postprocess(preds, shape, pad)
```

流式流水线可额外带来 **+15%** 吞吐提升。

## 8. FAQ

- **torch.load 报 weights_only 错误** -> 使用 `weights_only=False`
- **首次 compile 慢 (~20s)** -> 预热 3 次；设置 `ACL_OP_COMPILER_CACHE_MODE=1`
- **NPU 内存不足** -> `torch.npu.empty_cache()` 或启用 FP16
- **检测框精度偏差大** -> 确认输入尺寸 640x640，归一化参数与训练一致

## 9. 参考

- YOLOv10 官方: https://github.com/THU-MIG/yolov10
- 本仓库 Skill: `skills/yolov10-npu/SKILL.md`
- 仓库内脚本: `skills/yolov10-npu/scripts/yolov10_npu_benchmark.py`
