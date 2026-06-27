---
name: power-load-forecast
description: 通过时序大模型对电力负荷进行推理预测，输出未来一天或多天的负荷预测结果。当用户要求推理明天的负荷预测结果、预测明天的负荷、或说"帮我推一下明天的负荷预测"时，应使用此技能。
---

# Power Load Forecast

该 Skill 用于执行每日负荷预测，包括运行推理脚本完成电力负荷预测，并对多模型结果进行融合。

## 目录结构

```
power-load-forecast/
├── SKILL.md                    # 技能描述文件（本文件）
└── references/
    └── fusion_formula.md       # 预测结果融合公式
```

## 核心概念

### 起报时间

- **起报时间** = 预报目标日期的前一天 09:30
- 例如：预报 **2026 年 5 月 15 日** 的负荷 → 起报时间 = **2026 年 5 月 14 日 09:30**
- 起报时间通过 `--pred_start_time` 参数传入，格式 `YYYYMMDDHHMM`

### `{date}` 占位符

- `{date}` 统一表示**预报时间的第一天**的日期，格式 `YYYYMMDD`
- 例如：预报 **2026 年 5 月 15 日** 的负荷 → `{date}` = `20260515`
- 预报 **2026 年 5 月 15 日 ~ 17 日**（3天）→ `{date}` = `20260515`（仍为第一天）
- `{date}` 用于文件名模板中，Agent 在执行时将其替换为实际值

## 模型信息

模型配置文件位于 `/data/predict-agent/config/model_info.md`，包含各模型的推理脚本路径、原始文件名和目标文件名。

## 融合公式

详见 [references/fusion_formula.md](references/fusion_formula.md)

## 执行流程

### Step 1: 多模型并行推理

从 `/data/predict-agent/config/model_info.md` 读取各模型的推理脚本路径和脚本参数，同时启动多个推理进程（每个模型一个），由 Agent 统一管理并行和等待。

> **注意：** 实际执行时，使用 bash 工具的 `run_in_background=true` 参数分别启动每个模型，然后等待所有后台任务完成。推理脚本必须在各自所在目录下执行（因为脚本内部使用相对路径定位 dataset/、geo/ 等资源）。

**执行步骤：**

1. 读取 `/data/predict-agent/config/model_info.md`，获取所有模型的推理脚本路径和默认参数
2. 为每个模型并发启动一个推理进程（使用 `run_in_background=true`，工作目录设为脚本所在目录）
3. 等待所有后台推理进程完成
4. 检查每个模型的推理结果（返回码），记录成功/失败状态
5. 如果有模型推理失败，向用户报告失败情况

推理脚本执行结束之后，查看推理结果。
负荷预测 csv 文件中数据为间隔 15 分钟的时序数据（每天 96 个点，若预测多天则总点数为 天数 * 96）：

```
timestamp,OT
2026/04/22 00:00,46121.18920438817
2026/04/22 00:15,45828.569408295254
2026/04/22 00:30,45641.127130267145
2026/04/22 00:45,45506.277158298384
2026/04/22 01:00,45666.61497914909
2026/04/22 01:15,45557.057997557225
2026/04/22 01:30,45258.15787467807
2026/04/22 01:45,44741.04234548837
2026/04/22 02:00,44484.74938861859
......
```

### Step 2: 复制推理结果到统一目录（Agent 执行）

并行推理完成后，将各模型的推理结果复制到统一的结果目录。

**规则：**
- 读取 `/data/predict-agent/config/model_info.md` 中每个模型的配置
- 每个模型有 **原始文件名** 和 **目标文件名** 两个字段，其中 `{date}` 为**预报时间第一天日期**占位符
- `{date}` 统一表示**预报时间的第一天**，格式：`YYYYMMDD`
  - 例如：预报 2026-05-15 的负荷，则 `{date}` = `20260515`
- 将 `{date}` 替换为实际预报日期后，比较原始文件路径和目标文件路径是否完全一致（**目录和文件名均一致**）：
  - **一致** → 无需任何操作（原始文件已在目标位置，跳过复制）
  - **不一致** → 先确保目标目录 `/data/infer_result/{date}/` 存在，然后将原始文件**复制**到目标文件路径
- 复制后，后续融合步骤使用**目标文件名**读取数据

> **注意**：此步骤在推理脚本执行完毕后完成。采用复制而非重命名，保留原始文件不动。

### Step 3: 结果融合（Agent 动态生成融合代码）

根据配置文件动态生成融合代码并执行，实现推理结果的融合。

**执行方式：**

读取两个配置文件，动态生成 Python 融合代码并执行：

1. **读取 `/data/predict-agent/config/fusion_model_info.md`** — 获取所有融合算法的定义（融合名称、参与模型、权重、输出文件路径）
2. **读取 `/data/predict-agent/config/model_info.md`** — 获取各模型的目标文件路径和 CSV 列名（时间列名、预测值列名）
3. **动态生成融合代码** — 根据配置生成 Python 代码，写入临时文件并执行

**动态生成代码的模板逻辑：**

```python
import pandas as pd
import os

date = "{date}"
result_dir = f"/data/infer_result/{date}"

# 1. 读取各模型结果（根据 model_info.md 中的 CSV 列名配置）
models = {}  # 模型名 -> DataFrame
# 对每个模型：
#   - 读取目标文件路径（替换 {date}）
#   - 根据 CSV 时间列名和预测值列名解析
#   - 统一时间列名为 'timestamp'，预测值列名为模型名

# 2. 按时间合并所有模型
# merged = models[0] 依次 merge 其他模型

# 3. 对每个融合算法（从 fusion_model_info.md 读取）：
#   - 获取参与模型列表和对应权重
#   - 计算加权融合结果
#   - 保存到输出文件路径（替换 {date}）
```

**融合规则：**
- 融合结果的时间列名为 `timestamp`，负荷值列名为`OT` 
- 如果用户明确指定了某种融合方式，则只执行对应的算法
- **如果用户没有指定**，则 `fusion_model_info.md` 中定义的所有融合算法都运行一次，分别输出结果
- 最终将所有融合结果都展示给用户，让用户自行选择或参考

> **注意**：融合算法的增删改统一在 `/data/predict-agent/config/fusion_model_info.md` 中维护，无需修改 SKILL.md 或任何脚本。

### Step 4: 结果展示（Agent 动态生成绘图代码）

将单模型预测结果和融合结果绘制到一张折线图中，清晰直观地呈现负荷变化趋势（支持单天或多天预测），绘图必须严格遵守绘图规则。

**执行方式：**

1. 读取 `model_info.md` 获取各模型的目标文件路径
2. 读取 `fusion_model_info.md` 获取融合结果的输出文件路径
3. 根据以下绘图配置动态生成 Python 代码，写入临时文件并执行

**绘图规则：**
1. **中文字体**：绘图时必须指定中文字体为**文泉驿微米黑**（`WenQuanYi Micro Hei`），字体文件路径为 `/usr/share/fonts/wqy-microhei/wqy-microhei.ttc`（注意是 `.ttc` 后缀，不是 `.ttf`）。通过 `matplotlib` 的 `font_manager.FontProperties` 加载该字体，并设置 `plt.rcParams['font.family']`，确保标题、坐标轴标签、图例等所有中文文本正常显示。
2. **图例**：单模型图例名称使用上表中的"显示名称"；融合结果图例名称使用 `fusion_model_info.md` 中的"融合名称"。
3. **标题**：使用中文，格式为 `{预报日期} 负荷预测结果`，例如 `2026年5月15日 负荷预测结果`。
4. **坐标轴**：X 轴标签为"时间"，Y 轴标签为"负荷（MW）"，均使用中文。
5. **单模型与融合结果同图**：所有单模型预测曲线和所有融合结果曲线绘制在同一张图中，使用不同颜色和线型区分，确保图例清晰可辨。
6. **时间轴（严格 15 分钟刻度）**：
   - 数据为间隔 15 分钟的时序数据（每天 96 个点，若预测多天则总点数为 天数 * 96）
   - **必须根据实际数据的天数动态生成 X 轴刻度**，确保严格每 15 分钟一个刻度：
     ```python
     # 根据实际数据总行数计算天数
     total_points = len(df)  # 数据总行数
     days = total_points // 96  # 预测天数
     
     # 生成时间标签：每天 00:00 ~ 23:45
     time_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
     
     # 如果是多天预测，在标签前加上日期前缀
     if days > 1:
         from datetime import datetime, timedelta
         base_date = datetime.strptime("{date}", "%Y%m%d")
         all_labels = []
         for d in range(days):
             date_prefix = (base_date + timedelta(days=d)).strftime("%m/%d")
             all_labels.extend([f"{date_prefix}\n{t}" for t in time_labels])
         labels = all_labels[:total_points]
     else:
         labels = time_labels[:total_points]
     
     # 设置 X 轴刻度位置
     plt.xticks(ticks=range(total_points), labels=labels, rotation=45, fontsize=8)
     ```
   - 标签倾斜 45° 显示，字体大小适当（建议 fontsize=8 或更小）以避免重叠
   - **禁止使用 `mdates`、`AutoDateLocator` 等自动刻度定位器**，必须手动设置刻度
7. **图片保存**：图片保存到 `/data/infer_result/{date}/` 目录下，文件名为 `forecast_{date}.png`。
8. **图片查看**：禁止查看图片。
9. **图片发送**：绘图代码执行完毕后，**必须调用 `send_file_to_user` 工具**，将生成的图片文件路径（绝对路径）作为参数发送给用户。

> **注意**：新增模型时，只需在 SKILL.md 的绘图配置表中添加一行，并在 `model_info.md` 中添加模型信息。新增融合算法时，只需在 `fusion_model_info.md` 中添加一行。Agent 会自动读取配置并生成正确的绘图代码。
