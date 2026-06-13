---
name: small_model_adapt
description: NPU 适配验证工具包——将任意 PyTorch/CV/音频/视频模型迁移到 Ascend NPU 并验证兼容性。当用户要求"把模型跑到 NPU 上"、"NPU 适配"、"验证 NPU 兼容性"、"批量验证模型"、模型在 NPU 上报错需要排查修复时使用此 skill。
version: 1.0.0
metadata:
  bundled_resources:
    - references/    # 适配类型分类、决策树、模板代码
    - scripts/       # 冒烟测试模板、推理测试模板
---

# Small Model Adapt — 模型 NPU 适配验证

## 概述

基于 22 个模型的实战适配经验，沉淀的 NPU 适配方法论。覆盖图像分类、检测、分割、特征匹配、语音、视频、OCR、语言模型。

核心认知：适配一个新模型到 NPU，本质只做四件事：

| 步骤 | 问题 | 复杂度 |
|------|------|--------|
| **导入** | 这个包的 import 能过吗？ | 低 |
| **加载** | 模型能初始化并搬到 NPU 上吗？ | 中（权重下载是主要障碍） |
| **前向** | 造一个假输入扔进去，能跑通吗？ | 中高 |
| **真实推理** | 用真实数据跑，输出合理吗？ | 高 |

**前向跑通 ≠ 模型可用**，但**对于 NPU pipeline 验证，前向通过就够了**。

## 适配流程

### 第一步：判断适配类型

根据模型特征判断属于哪种适配类型（详细参考 `references/adapter_types.md`）：

| 类型 | 特征 | 适配难度 |
|------|------|---------|
| **A — 纯 PyTorch** | 标准 nn.Module，无 CUDA kernel | `.to("npu:0")` 一行搞定 |
| **B — 框架封装** | modelscope/insightface/whisper 等包装 | 找框架的 device 参数 |
| **C — CUDA 依赖** | 依赖 .cu 文件/cupy/cuda_extension | 找 OpenCV/PyTorch 等价实现 |
| **D — NPU 算子 bug** | GPU 正常，NPU 行为不一致 | 定位算子 → monkey-patch |

不确时：先按类型 A 尝试，失败后看报错信息判断。

### 第二步：跑冒烟测试

使用 `scripts/smoke_test_template.py` 模板（替换 `<ModelName>` 和其他占位符后使用），验证四步：

1. **NPU 环境探测**：`torch.npu.is_available()`, `device_count()`, `get_device_name(0)`
2. **模型导入**：import 不报错
3. **模型加载**：构造 → `.to("npu:0")` → `.eval()`
4. **前向推理**：合成数据 forward → 检查输出 shape/dtype/数值范围

### 第三步：跑真实推理（可选）

如果需要验证完整管线，使用 `scripts/infer_test_template.py`，改用真实输入数据。

### 第四步：记录结果

结果保存为结构化 JSON（参考 `references/templates.md` 中的 schema）。

## 决策树

遇到问题时阅读 `references/decision_tree.md`，包括：
- 输入 shape 排查
- 权重下载回退策略  
- 优雅降级三层回退
- 适配成功三层次判定（冒烟/pipeline/可用）

## 已有案例速查

所有代码在 `/home/qiuqi/small/code/scanner-res/other/`：

| 类型 | 模型 | 适配要点 |
|------|------|---------|
| A | ResNet, ViT, MobileViT, UniFormer, ViViT, VideoMAE, Wav2Vec2, MMS | `.to("npu:0")` + 权重下载回退 |
| B | SCRFD, RF-DETR, FunASR, GOT-OCR2, Whisper | 找框架 device 参数 |
| C | CudaSift, FAST, PnLCalib, VideoSwin, Mamba | OpenCV代理/PyTorch复现 |
| D | LightGlue | max_pool2d return_indices int8 截断 → 值比较 NMS |

## Bundled Resources

### scripts/
- `scripts/smoke_test_template.py` — 通用冒烟测试模板（导入+加载+前向），替换占位符即可
- `scripts/infer_test_template.py` — 真实推理测试模板

### references/
- `references/adapter_types.md` — A/B/C/D 四种适配类型详解
- `references/decision_tree.md` — 决策树、工程决策、输入 shape 排查
- `references/templates.md` — 完整代码模板、JSON schema、目录结构
