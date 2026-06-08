---
name: cv-ascend-adapt
description: 面向开源 PyTorch CV 模型的昇腾 NPU 通用适配技能，用于训练、分布式训练、ONNX 到 OM 导出、推理一致性验证与问题排查。Use when users need Ascend migration checklists, HCCL/autocast/dtype compatibility fixes, or adaptation report generation without binding to one specific model.

---

# CV 昇腾适配

使用此技能以可复用方式执行 CV 模型昇腾适配。

## 快速开始

1. 先判定当前任务属于哪一类：
   - 若用户只要求验证一个能力点、一个链路片段或一个输出模板，走“单一效果验证”路径。
   - 若用户目标是完成训练、分布式、导出、推理、交付中的多阶段闭环，走“整体适配流程”路径。

2. 执行前必须先说明：
   - 当前选择的是“单一效果验证”还是“整体适配流程”
   - 当前处于哪个阶段或哪个验证用例

3. 若为“单一效果验证”：
   - 读取 `references/cn-prompts.md`
   - 只选择一个验证目标执行
   - 按“动作 + 证据 + 结论 + 失败修复建议”输出

4. 若为“整体适配流程”：
   - 先阅读下面“工作流”和对应阶段文件
   - 执行 `python scripts/scaffold_docs.py --out <目录> --model <模型名>` 生成交付文档骨架
   - 按阶段顺序逐步推进，不跳阶段，不混阶段收尾

5. 出现报错时：
   - 执行 `python scripts/triage_log.py --log <日志> --out <报告.md>`
   - 排障时再读取 `references/error-signatures.md`

6. 完成前必须用证据补全文档；缺少证据视为未完成。

## 工作流

本章节仅用于“整体适配流程”。

进入条件：

- 用户目标是完成整体适配，而不是只验证单一能力点
- 任务需要跨多个阶段闭环推进
- 任务最终需要形成交付物、结论或完整验收结果

执行规则：

- 必须按阶段顺序推进：`阶段0 -> 阶段1 -> 阶段1.5 -> 阶段1.8 -> 阶段2 -> 阶段3 -> 阶段4 -> 阶段5`
- 每进入新阶段前，先说明当前阶段
- 每个阶段的完成标准未满足前，不进入下一阶段
- 若阶段失败，按对应阶段文件中的“失败处理”执行
- 若任务中途被收敛为“只验证一个点”，可切换到“单一效果验证”路径，但不得再宣称已完成整体适配

### 阶段0：需求分析与信息提取

- 详见 [references/workflow-stages/stage-0-requirement-analysis.md](references/workflow-stages/stage-0-requirement-analysis.md)

### 阶段1：环境收敛

- 详见 [references/workflow-stages/stage-1-environment-convergence.md](references/workflow-stages/stage-1-environment-convergence.md)

### 阶段1.5：数据集前置条件确认

- 详见 [references/workflow-stages/stage-1.5-dataset-prerequisites.md](references/workflow-stages/stage-1.5-dataset-prerequisites.md)

### 阶段1.8：依赖安装收敛（训练前置）

- 详见 [references/workflow-stages/stage-1.8-dependency-convergence.md](references/workflow-stages/stage-1.8-dependency-convergence.md)

### 阶段2：训练链路打通

- 详见 [references/workflow-stages/stage-2-training-enablement.md](references/workflow-stages/stage-2-training-enablement.md)

### 阶段3：分布式与精度兼容

- 详见 [references/workflow-stages/stage-3-distributed-and-precision.md](references/workflow-stages/stage-3-distributed-and-precision.md)

### 阶段4：导出与推理验证

- 详见 [references/workflow-stages/stage-4-export-and-inference.md](references/workflow-stages/stage-4-export-and-inference.md)

### 阶段5：报告与交付

- 详见 [references/workflow-stages/stage-5-report-and-delivery.md](references/workflow-stages/stage-5-report-and-delivery.md)
