---
name: npu-adapt
description: 将 HuggingFace / ModelScope 图片分类模型适配为单卡昇腾 NPU 可运行的提交工程。当用户提到"适配模型到昇腾NPU"、"NPU适配"、"把xxx模型放到NPU上跑"、"Ascend适配"、或给出模型名要求做图片分类适配时，使用此Skill。执行完整流水线：模型预判、可用性检查、工程创建、环境检查、依赖、测试图、推理、精度验证、性能测试、README生成、截图、创建GitCode仓库、推送、生成竞赛提交信息。适用于 timm/*、google/*、facebook/*、microsoft/* 等标准图片分类模型，明确拒绝 DINO/MAE/CLIP/embedding/backbone/OCR/检测/分割/生成/多模态模型。
---

# 昇腾 NPU 图片分类模型适配

将 ModelScope / HuggingFace 图片分类模型适配为可在单卡昇腾 NPU（Ascend910B）上运行的提交工程。不使用 vLLM、TP、EP、DP、MoE、文本生成、检测、分割、OCR、多模态。

## 输入参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `model_name` | 是 | ModelScope 模型名，如 `timm/resnet50.a1_in1k` |
| `git_repo` | 否 | GitCode 仓库地址 |
| `git_token` | 否 | GitCode push token |
| `task_type` | 否 | 默认 `image-classification` |
| `skip_push` | 否 | 设为 `true` 则跳过阶段 11（Git 推送），仅完成阶段 0-10。默认 `false` |
| `special_notes` | 否 | 特殊注意事项 |

**关于 skip_push**：默认行为是阶段 0-11 一气呵成直接完成，不需要用户说"git push"就会自动推送。只有当用户明确说"先不要 git push"、"不推送"、"仅适配不提交"等类似指令时，才将 `skip_push` 设为 `true`，完成阶段 0-10 后直接输出结果，不要停在阶段 10 等待用户手动触发阶段 11。

---

## 绝对规则（不可违反）

1. **不 fallback**：目标模型不可用时停止并生成 `README_FAIL_REASON.md`
2. **不伪造**：只有真实 NPU 推理成功才能写"适配成功"
3. **不提交权重**：`*.bin`, `*.safetensors`, `*.pth`, `*.pt`, `*.ckpt`, `*.onnx` 不提交
4. **自动 git push（默认直接完成 0-11）**：适配成功后自动创建 GitCode `model` 类型仓库并 push，阶段 0-11 一气呵成，不需要用户额外说"git push"。只有用户明确说"先不要 git push"时，设 `skip_push=true`，完成阶段 0-10 后直接输出结果
5. **禁止 HuggingFace 自动下载**：`timm/*` 严禁 `timm.create_model(..., pretrained=True)`，必须通过 ModelScope snapshot_download + 本地权重加载
6. **不兼容即失败**：`iic/cv_*`、DINO/MAE/CLIP/embedding/backbone/检测/分割/生成/多模态 直接失败
7. **精度验证必须达标**：CPU-NPU 相对误差 `relative_error < 1.0%`，否则视为适配失败，生成 `README_FAIL_REASON.md`
8. **README 必须包含推理正常输出证据**：README 第 3 节"推理运行"必须列出 NPU Top-5 预测结果（类别 + 概率），不能只写"日志保存在 logs/inference.log"。无推理输出证据视为适配不完整

### 已完成模型（禁止重复适配）

  nvidia/mit-b0, google/vit-base-patch16-224, facebook/deit-base-patch16-224,
  microsoft/swin-base-patch4-window12-384-in22k, openai/whisper-tiny, timm/convnext_base.clip_laion2b,
  facebook/convnext-tiny-224, microsoft/resnet-50,
  google/mobilenet_v2_1.0_224, google/efficientnet-b0, apple/mobilevit-small, facebook/regnet-y-040,
  google/efficientnet-b1, facebook/regnet-x-080, google/efficientnet-b2, google/efficientnet-b3, google/efficientnet-b4,
  google/efficientnet-b5,
  google/efficientnet-b6, google/efficientnet-b7, facebook/regnet-x-040, facebook/regnet-x-032, facebook/regnet-y-032,
  facebook/regnet-y-016, google/beit-base-patch16-224, microsoft/resnet-18, facebook/densenet121, microsoft/resnet-34,
  microsoft/swin-tiny-patch4-window7-224, facebook/levit-128S, facebook/deit-small-patch16-224,
  facebook/deit-tiny-patch16-224, microsoft/swin-small-patch4-window7-224, facebook/convnext-small-224,
  microsoft/resnet-101, facebook/levit-192,
  facebook/levit-256, facebook/levit-128, apple/mobilevit-xx-small, apple/mobilevit-x-small, facebook/convnext-base-224,
  snap-research/efficientformer-l1-300, snap-research/efficientformer-l3-300, facebook/deit-tiny-distilled-patch16-224,
  facebook/deit-small-distilled-patch16-224, microsoft/resnet-152, facebook/regnet-y-080, facebook/regnet-x-120,
  facebook/regnet-y-120, facebook/regnet-x-160, facebook/regnet-y-160, facebook/regnet-y-320, facebook/regnet-x-320,
  facebook/regnet-y-640-seer-in1k, facebook/regnet-y-1280-seer-in1k, facebook/regnet-y-320-seer-in1k,
  facebook/deit-base-distilled-patch16-224, facebook/convnext-large-224, facebook/convnext-xlarge-224-22k-1k,
  microsoft/swin-large-patch4-window7-224, microsoft/swin-large-patch4-window12-384, facebook/levit-384,
  google/vit-large-patch16-224, google/vit-large-patch32-384,
  timm/resnet50.a1_in1k,
  timm/regnety_080_tv.tv2_in1k,
  timm/efficientnet_el_pruned.in1k,
  timm/wide_resnet50_2.racm_in1k,
  timm/mobilenetv2_100.ra_in1k,
  timm/wide_resnet101_2.tv_in1k,
  timm/mobilenetv3_small_100.lamb_in1k,
  timm/res2net50_48w_2s.in1k,
  timm/deit3_small_patch16_224.fb_in22k_ft_in1k,
  timm/mobilenetv4_conv_small.e2400_r224_in1k,
  timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k,
  timm/vit_large_patch16_224.augreg_in21k,
  timm/vit_base_patch16_224.augreg_in21k,
  timm/resnet34.a1_in1k,
  timm/swin_tiny_patch4_window7_224.ms_in22k_ft_in1k,
  timm/tresnet_l.miil_in1k,
  timm/vit_large_patch16_224.augreg_in21k_ft_in1k,
  timm/tiny_vit_5m_224.in1k,
  timm/deit_base_patch16_224.fb_in1k,
  timm/convnext_tiny.fb_in22k_ft_in1k,
  timm/vit_base_patch32_clip_384.openai_ft_in12k_in1k,
  timm/vit_base_patch16_224.augreg_in21k_ft_in1k,
  timm/mobilenetv3_large_100.miil_in21k_ft_in1k,
  timm/efficientnet_b4.ra2_in1k,
  timm/efficientnet_b0.ra_in1k,
  timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k,
  timm/inception_v3.tf_adv_in1k,
  timm/tf_efficientnetv2_s.in21k,
  timm/convnextv2_base.fcmae_ft_in22k_in1k,
  timm/vit_small_patch16_224.augreg_in1k,
  timm/caformer_m36.sail_in22k_ft_in1k_384,
  timm/mobilenetv3_large_100.ra_in1k,
  timm/densenet201.tv_in1k,
  timm/mobilenetv4_hybrid_large.ix_e600_r384_in1k,
  timm/resnet50d.ra2_in1k,
  timm/tf_efficientnet_b0.ns_jft_in1k,
  timm/resnet50_gn.a1h_in1k,
  timm/regnety_040.pycls_in1k,
  timm/hgnet_tiny.paddle_in1k,
  timm/maxvit_tiny_tf_224.in1k,
  timm/poolformer_s12.sail_in1k,
  timm/poolformer_s36.sail_in1k,
  timm/tf_efficientnet_lite0.in1k,
  timm/deit3_base_patch16_224.fb_in22k_ft_in1k,
  timm/maxvit_tiny_rw_224.sw_in1k,
  timm/vit_small_patch16_384.augreg_in21k_ft_in1k,
  timm/xcit_small_24_p8_224.fb_dist_in1k,
  timm/resnetv2_50x1_bit.goog_in21k_ft_in1k,
  timm/mobilenetv3_large_100.miil_in21k,
  timm/eva02_large_patch14_448.mim_in22k_ft_in1k,
  timm/coatnet_2_rw_224.sw_in12k,
  timm/vit_base_patch16_224_miil.in21k,
  timm/swin_base_patch4_window12_384.ms_in22k_ft_in1k,
  timm/tf_efficientnet_b7.ns_jft_in1k,
  timm/resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k,
  timm/vgg16.tv_in1k,
  timm/convnext_tiny.fb_in22k_ft_in1k_384,
  timm/swin_small_patch4_window7_224.ms_in22k,
  timm/flexivit_base.1000ep_in21k

**每次适配成功后，在列表末尾追加 `, {new_model_name}`。失败不更新。**

---

## 模型兼容性优先级

| 优先级 | 类型 | 说明 |
|--------|------|------|
| A | `timm/*` | 最佳。结构标准、权重完整、无 HF 依赖。`pretrained=False` + 本地权重加载。 |
| B | `google/*`, `facebook/*`, `microsoft/*` | Transformers 标准分类。HF 仅作备选，主路径必须是 ModelScope + 本地路径加载。 |
| C | `iic/cv_*` | 高风险。必须能迁移内部 PyTorch 模型到 `npu:0` 并获得 logits，否则失败。 |
| D | DINO/MAE/CLIP/embedding/backbone/检测/分割/生成/多模态 | 直接失败。 |

---

## 精简 11 阶段流程

### 阶段 0：模型预判与可用性检查

根据 `model_name` 前缀直接判断兼容性。等级 D 直接生成 `README_FAIL_REASON.md`。

**ModelScope 下载与验证（一次完成，不要分多次脚本）：**

```python
from modelscope import snapshot_download
model_dir = snapshot_download(model_name)
```

timm/* 专用：
```python
import os, torch, timm
from safetensors.torch import load_file as safetensors_load

timm_name = model_name.replace("timm/", "")
model = timm.create_model(timm_name, pretrained=False)

# 查找权重
weight_files = [os.path.join(model_dir, f) for f in sorted(os.listdir(model_dir))
                if f.endswith(('.safetensors', '.bin', '.pth', '.pt', '.ckpt'))]
weights_path = next((p for p in weight_files if os.path.exists(p)), None)

# 加载权重（注意：旧权重可能不兼容 weights_only=True，需回退）
if weights_path.endswith('.safetensors'):
    state_dict = safetensors_load(weights_path)
else:
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception:
        state_dict = torch.load(weights_path, map_location="cpu")

for key in ['state_dict', 'model', 'module', 'ema', 'model_state']:
    if key in state_dict and isinstance(state_dict[key], dict):
        state_dict = state_dict[key]
        break
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

missing, unexpected = model.load_state_dict(state_dict, strict=False)
if len(missing) > 100 and len(unexpected) > 100:
    # 失败：权重与模型结构严重不匹配

# Dummy forward CPU + NPU
model.eval()
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out_cpu = model(dummy)          # shape [1, 1000]
    out_npu = model.to("npu:0")(dummy.to("npu:0"))  # shape [1, 1000]
```

- 输出 shape 必须是 `[batch, num_classes]`
- 阶段 0 的所有验证写到一个 Python 脚本里直接执行，不要在外层目录创建孤立的 stage0_check.py
- 下载成功后将 `model_dir` 和 `weights_path` 保存到 `logs/paths.txt` 供后续脚本复用

### 阶段 1：创建工程目录

目录名：`{model_name.replace('/', '-')}-NPU/`

```
{model_name}-NPU/
├── logs/
├── screenshots/
├── assets/
├── model_utils.py       # 公共模型加载模块（关键：避免三个脚本重复）
├── inference.py
├── eval_accuracy.py
├── benchmark.py
├── requirements.txt
├── .gitignore
└── README.md
```

**关键改进：用 `model_utils.py` 抽取公共逻辑，禁止三个主脚本各自复制粘贴 load_model。**

### 阶段 2：环境检查

仅在首次适配或环境变化时执行。同一 NPU 容器连续适配多个模型时，可复用已有 `logs/env_check.log`。

```bash
npu-smi info > logs/env_check.log
python -c "import torch; import torch_npu; print(f'npu_available={torch.npu.is_available()}, device={torch.npu.get_device_name(0)}')" >> logs/env_check.log
```

NPU 不可用 → `README_FAIL_REASON.md` 并停止。

### 阶段 3：requirements.txt

```
torch
torchvision
transformers
pillow
requests
numpy
huggingface_hub
accelerate
modelscope
timm
safetensors
```

### 阶段 4：测试图片

```python
# 直接内联在适配脚本中，不要单独创建 scripts/download_test_image.py
import requests, time
from PIL import Image

url = "https://picsum.photos/400/300"
for attempt in range(3):
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            open("assets/test.jpg", "wb").write(r.content)
            Image.open("assets/test.jpg").convert("RGB").save("assets/test.jpg")
            break
    except Exception:
        time.sleep(2)
else:
    Image.new("RGB", (400, 300), (128,128,128)).save("assets/test.jpg")
    open("assets/test_image_note.txt", "w").write("占位图\n")
```

### 阶段 5：model_utils.py（新增核心模块）

**这是减少重复工作的核心。所有脚本通过 import model_utils 加载模型，禁止各自复制粘贴 load_model。**

```python
# model_utils.py
import os, torch, timm
from modelscope import snapshot_download
from safetensors.torch import load_file as safetensors_load

MODEL_NAME = "timm/resnet50.a1_in1k"
TIMM_NAME = MODEL_NAME.replace("timm/", "")

def load_model(pretrained=False):
    model_dir = snapshot_download(MODEL_NAME)
    model = timm.create_model(TIMM_NAME, pretrained=False)

    weight_files = [os.path.join(model_dir, f) for f in sorted(os.listdir(model_dir))
                    if f.endswith(('.safetensors', '.bin', '.pth', '.pt', '.ckpt'))]
    weights_path = next((p for p in weight_files if os.path.exists(p)), None)
    if weights_path is None:
        raise RuntimeError(f"No local weights in {model_dir}")

    if weights_path.endswith('.safetensors'):
        state_dict = safetensors_load(weights_path)
    else:
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception:
            state_dict = torch.load(weights_path, map_location="cpu")

    for key in ['state_dict', 'model', 'module', 'ema', 'model_state']:
        if key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
            break
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return model, model_dir, weights_path, missing, unexpected

def preprocess(image, model):
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return transform(image).unsqueeze(0)
```

### 阶段 6：inference.py

```python
import torch.nn.functional as F
from PIL import Image
import model_utils

model, model_dir, weights_path, missing, unexpected = model_utils.load_model()
model = model.to("npu:0").eval()

image = Image.open("assets/test.jpg").convert("RGB")
inputs = model_utils.preprocess(image, model).to("npu:0")

with torch.no_grad():
    torch.npu.synchronize()
    outputs = model(inputs)
    torch.npu.synchronize()

probs = F.softmax(outputs, dim=-1)
top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)
# ... 输出和日志
```

**要求：**
1. `pretrained=False` + 本地权重
2. `timm.data.resolve_model_data_config` + `create_transform`
3. `npu:0` 真实推理
4. 无 id2label 时用 `class_x`
5. 日志写入 `logs/inference.log`

### 阶段 7：eval_accuracy.py

```python
import os, torch, torch.nn.functional as F
from PIL import Image
import model_utils

model, _, _, _, _ = model_utils.load_model()
model.eval()

image = Image.open("assets/test.jpg").convert("RGB")
inputs = model_utils.preprocess(image, model)

with torch.no_grad():
    out_cpu = model(inputs)

model_npu = model.to("npu:0")
inputs_npu = inputs.to("npu:0")
with torch.no_grad():
    torch.npu.synchronize()
    out_npu = model_npu(inputs_npu)
    torch.npu.synchronize()

# 官方 Ascend 精度一致性指标
diff = (out_cpu - out_npu.cpu()).abs()
max_abs_error = diff.max().item()
mean_abs_error = diff.mean().item()
mean_cpu_abs = out_cpu.abs().mean().item()
relative_error = (mean_abs_error / mean_cpu_abs) * 100 if mean_cpu_abs != 0 else float('inf')
cosine_similarity = torch.nn.functional.cosine_similarity(
    out_cpu.flatten(), out_npu.cpu().flatten(), dim=0
).item()

result = "PASS" if relative_error < 1.0 else "FAIL"

log_path = "logs/accuracy.log"
os.makedirs("logs", exist_ok=True)
with open(log_path, "w") as f:
    f.write("=== CPU-NPU Accuracy Consistency Check ===\n")
    f.write(f"max_abs_error     {max_abs_error:.6f}\n")
    f.write(f"mean_abs_error    {mean_abs_error:.6f}\n")
    f.write(f"relative_error    {relative_error:.4f}%\n")
    f.write(f"cosine_similarity {cosine_similarity:.6f}\n")
    f.write(f"threshold         1.0%\n")
    f.write(f"结果              {result}\n")
    f.write("\n")
    f.write("Note: This is a smoke consistency check on a single test image,\n")
    f.write("not an official ImageNet full validation set evaluation.\n")

if result == "FAIL":
    raise RuntimeError(f"Accuracy check FAILED: relative_error={relative_error:.4f}% >= 1.0%")
```

日志写入 `logs/accuracy.log`，包含官方 Ascend 精度一致性指标：`max_abs_error`、`mean_abs_error`、`relative_error`、`cosine_similarity`、`threshold=1.0%`、`result=PASS/FAIL`。`relative_error < 1.0%` 为通过阈值，否则视为适配失败并生成 `README_FAIL_REASON.md`。

### 阶段 8：benchmark.py

```python
import time
from PIL import Image
import model_utils

model, _, _, _, _ = model_utils.load_model()
model = model.to("npu:0").eval()

image = Image.open("assets/test.jpg").convert("RGB")
inputs = model_utils.preprocess(image, model).to("npu:0")

# 预热 2 次 + 正式 10 次（ResNet/ViT 级别）
# 每次前后 torch.npu.synchronize()
# 统计 avg/min/max/p50/p90/p95/images_per_sec
```

日志写入 `logs/benchmark.log`。

### 阶段 9：运行验证

依次执行：
1. `python inference.py` → 失败则生成 `README_FAIL_REASON.md`
2. `python eval_accuracy.py`
3. `python benchmark.py`

### 阶段 10：README.md + 最终检查 + 自验证

**README.md 模板：**
```markdown
---
hardware: NPU
---

# {model} on Ascend NPU

## 1. 简介
## 2. 验证环境
## 3. 推理运行

```bash
pip install -r requirements.txt
python inference.py
```

推理结果 (NPU Top-5):
- Top-1: class_x (0.xxxx)
- Top-2: class_x (0.xxxx)
- Top-3: class_x (0.xxxx)
- Top-4: class_x (0.xxxx)
- Top-5: class_x (0.xxxx)

日志保存在 `logs/inference.log`。

## 4. 精度验证

对单张测试图片进行 CPU 与 NPU 一致性验证：

| 指标 | 数值 |
|------|------|
| max_abs_error | {val:.6f} |
| mean_abs_error | {val:.6f} |
| relative_error | {val:.4f}% |
| cosine_similarity | {val:.6f} |
| threshold | 1.0% |
| **结果** | **PASS/FAIL** |

- CPU Top-1: class_x
- NPU Top-1: class_x
- CPU Top-5: class_a, class_b, ...
- NPU Top-5: class_a, class_b, ...
- Top-1 match: True/False
- Top-5 match: True/False

## 5. 性能参考
## 6. 精度评测说明

本项目包含单图 smoke consistency 验证，非官方 ImageNet 完整验证集评测。详细指标见第 4 节。

## 7. 自验证截图
## 8. 日志文件
## 9. 注意事项
## 10. 标签 #NPU
```

**最终检查（合并为一个步骤）：**
```bash
find . -maxdepth 2 -type f | sort
find . -type f -size +50M -print
find . \( -name "*.bin" -o -name "*.safetensors" -o -name "*.pth" -o -name "*.pt" -o -name "*.ckpt" -o -name "*.onnx" \) -print
rm -f fusion_result.json
rm -rf kernel_meta/
```

**自验证截图：**
- 从 `logs/` 提取关键内容写入 `screenshots/self_verification.txt`
- Python + PIL 生成 `screenshots/self_verification.png`（必须包含 NPU 环境、推理结果、精度验证、性能测试摘要）

---

## 失败处理

任何阶段失败，生成 `README_FAIL_REASON.md`，包含：
- 模型名、兼容性等级、加载路径
- ModelScope 下载状态、文件清单
- 是否使用 `pretrained=False`、是否加载成功
- 是否能输出 logits、是否能迁移到 `npu:0`
- 失败阶段、失败命令、失败原因
- **是否 fallback：否**

---

## .gitignore

```
__pycache__/
*.pyc
.cache/
hf_cache/
pytorch_model.bin
*.bin
model.safetensors
*.safetensors
model-*/
models--*/
fusion_result.json
kernel_meta/
.modelscope/
modelscope_cache/
ms_cache/
hub/
*.pth
*.pt
*.ckpt
*.onnx
```

---

### 阶段 11：创建 GitCode 仓库 + 推送 + 竞赛提交信息

**仅当 `skip_push != true` 时执行。** 适配成功后自动执行，无需用户额外要求，阶段 0-11 一气呵成。

#### 11a. 创建 GitCode 仓库

Token 从环境变量 `ATOMGIT_USER_TOKEN` 读取。仓库名根据模型名自动生成：

```
REPO_NAME = model_name.replace('/', '-') + '-NPU'   # 如 timm-efficientnet_b4-ra2_in1k-NPU
```

调用 API 创建仓库，**仓库类型必须是 `model`**（展示模型卡片，方便竞赛提交）：
```bash
curl -s -X POST 'https://api.gitcode.com/api/v5/user/repos' \
  -H "private-token: $ATOMGIT_USER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"name\": \"${REPO_NAME}\", \"path\": \"${REPO_NAME}\", \"private\": false, \"repository_type\": \"model\", \"topics\": [\"Hardware: NPU\", \"NPU\", \"Ascend\", \"Ascend910\"]}"
```

**关键：从 API 响应中提取正确路径，不要自己拼接。** 响应包含 `namespace.path`（用户名）和 `path`（仓库名）：

```python
# Python 解析响应
import json, subprocess
result = subprocess.run(['curl', '-s', '-X', 'POST', ...], capture_output=True, text=True)
resp = json.loads(result.stdout)
NAMESPACE = resp['namespace']['path']   # 如 "gyccc"
REPO_FULL_PATH = f"{NAMESPACE}/{resp['path']}"  # 如 "gyccc/timm-resnet50-NPU"
GIT_URL = resp['http_url_to_repo']     # 完整 HTTPS URL
```

- 返回 201 → 创建成功
- 返回 422（已存在）→ 调用 GET API 获取已有仓库信息，提取 `namespace.path` 和 `http_url_to_repo`
- 其他错误 → 停止并提示用户

#### 11b. Git 推送

**必须使用从 API 响应中提取的 `GIT_URL`，不要拼接路径：**

```bash
cd {project_dir}
git init && git branch -m main
git remote add origin {GIT_URL} || git remote set-url origin {GIT_URL}
git add . && git commit -m "Adapt {model_name} for Ascend NPU"
git push -u origin main
```

**推送失败的常见原因及解决**：
GitCode 新建仓库时可能自动生成初始提交（如 README/License），导致本地与远程历史不共祖，push 报 `rejected`。

此时执行：
```bash
git pull origin main --rebase
git push -u origin main
```
用 `--rebase` 将本地提交变基到远程历史之上，再推送即可通过。

**常见错误**：直接用 `https://gitcode.com/{REPO_NAME}.git` 会 403，因为缺少用户名前缀。正确格式是 `https://gitcode.com/{NAMESPACE}/{REPO_NAME}.git`。

#### 11c. 生成竞赛提交信息

推送成功后，输出以下三行供用户直接复制粘贴到竞赛提交表单：

```
=== 竞赛提交信息 ===

原始权重模型地址：
https://modelscope.cn/models/{model_name}

提交项目地址：
https://gitcode.com/{REPO_FULL_PATH}

提交说明：
适配 {model_name} 到昇腾 NPU (Ascend910)。使用 ModelScope snapshot_download 下载权重，timm.create_model(pretrained=False) 加载本地权重，包含推理验证、CPU-NPU 精度一致性检查和性能基准测试。{accuracy_summary} {benchmark_summary}
```

其中：
- `REPO_FULL_PATH`：从 API 响应提取的完整路径，如 `gyccc/timm-resnet50-NPU`
- `accuracy_summary`：从 `logs/accuracy.log` 提取，如 "CPU-NPU Top-1/Top-5 完全匹配"
- `benchmark_summary`：从 `logs/benchmark.log` 提取，如 "吞吐量 51.85 images/sec"
