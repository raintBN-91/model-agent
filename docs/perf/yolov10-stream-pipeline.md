# [内测挑战] YOLOv10n NPU 流式流水线推理优化报告

> 赛道：性能优化实践 ｜ 预估积分：200（PR 合并后）
> 关联 Skill: `skills/yolov10-npu/`
> 关联实践文档: `docs/practices/yolov10-npu-e2e.md`

## 1. 问题背景

在视频流或批量图片推理场景中，YOLOv10n 在昇腾 NPU 上的推理瓶颈不仅是模型计算本身，还包括：
- **CPU 预处理阻塞**：图片解码、resize、归一化串行执行，NPU 空闲等待
- **CPU-NPU 传输开销**：每轮推理需将预处理后的张量从 CPU 内存拷贝到 NPU 显存
- **后处理回传延迟**：NMS 和框解码在 CPU 执行，大张量回传耗时

本优化通过**双缓冲线程预取 + NPU 端到端处理**，消除 CPU 阻塞，实现流水线并行。

## 2. 环境配置

- NPU: Ascend 910_9362
- CANN: 8.0.RC3+
- torch: 2.3.1, torch_npu: 2.3.1.post1
- 模型: YOLOv10n（260 万参数）
- 输入: COCO 验证集 640x640 RGB

## 3. Baseline（串行推理）

```python
model = YOLO("yolov10n.pt").to("npu:0")
model.model.half()
model.model = torch.compile(model.model, backend="npu")

for img_path in image_list:
    img = Image.open(img_path)
    result = model(img)  # 预处理 -> NPU -> 后处理，全串行
```

| 配置 | E2E 延迟 (ms/图) | FPS | NPU 利用率 |
|:---|:---:|:---:|:---:|
| CPU FP32 基线 | 309.73 | 3.2 | - |
| NPU FP16 + compile | 17.53 | 57.0 | ~65% |
| + 运行时优化 | 16.60 | 60.2 | ~72% |

瓶颈分析：NPU 实际推理仅 2.78ms，但 CPU 预处理 + 传输 + 后处理占用 13.8ms，NPU 大量时间空闲等待。

## 4. 优化方案：流式流水线

### 4.1 双缓冲线程预取

在独立线程中提前完成下一张图的预处理，主线程专注 NPU 推理：

```python
from queue import Queue
import threading

def predict_stream(self, images):
    q = Queue(maxsize=2)  # 双缓冲
    n = len(images)

    def _prefetch():
        for i in range(n):
            q.put(self.preprocess(images[i]))
        q.put(None)  # EOF

    threading.Thread(target=_prefetch, daemon=True).start()

    for i in range(n):
        item = q.get()
        if item is None:
            break
        tensor, shape, pad = item
        with torch.no_grad():
            preds = self.model(tensor)
        yield self.postprocess(preds, shape, pad)
```

### 4.2 NPU 端到端处理

- 预处理：使用 `F.interpolate` 在 NPU 上完成 resize
- 后处理：NMS / Box decode 在 NPU 上执行，仅最终小结果回传 CPU

```python
import torch.nn.functional as F

img_tensor = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to("npu:0")
img_tensor = F.interpolate(img_tensor, size=(640,640))
# NPU 上执行后处理
preds = preds.transpose(-1, -2)
bboxes, scores, labels = ops.v10postprocess(preds, max_det, nc)
filtered_cpu = filtered.float().cpu()  # 仅小结果回传
```

## 5. 优化效果对比

| 指标 | Baseline (串行) | Optimized (流式流水线) | 提升 |
|:---|:---:|:---:|:---:|
| E2E 延迟 (ms/图) | 16.60 | 14.38 | -13.4% |
| FPS | 60.2 | 69.5 | +15.4% |
| NPU 利用率 | ~72% | ~91% | +26.4% |
| CPU 预处理耗时 | 8.2ms | 0ms (隐藏) | 完全重叠 |
| 端到端加速比 (vs CPU) | 18.8x | **21.5x** | +14.4% |

> 以上数据来自仓库 Skill `skills/yolov10-npu/SKILL.md` 的基准测试声明，待独立 NPU 环境复测确认。

## 6. 复现代码

```bash
# 1. 环境准备
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install pillow numpy torchvision ultralytics
wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt

# 2. 运行基准测试
python scripts/yolov10_npu_benchmark.py --model yolov10n.pt --runs 50

# 3. 运行流式优化版本
python scripts/yolov10_npu_benchmark.py --model yolov10n.pt --runs 50 --stream
```

## 7. 适用场景与限制

- **适用**：视频流实时检测、批量图片离线处理、多摄像头并发推理
- **限制**：单图推理无收益；`Queue(maxsize=2)` 内存占用与 batch size 成正比；线程安全需确保预处理和模型共享只读数据

## 8. 参考

- 本仓库 Skill: `skills/yolov10-npu/SKILL.md`
- 仓库内基准脚本: `skills/yolov10-npu/scripts/yolov10_npu_benchmark.py`
- YOLOv10 论文: https://arxiv.org/abs/2405.14458
