# 阶段4：导出与推理验证

## 输入信息

- 已通过训练或已确认可用的推理入口
- 模型权重、导出脚本、后处理逻辑
- 目标 SoC、ATC 工具、ais_bench 工具
- 基线推理样例与后处理要求

## 必做动作

- 建立确定性流水线：`PyTorch -> ONNX -> OM`，并使用 OM 模型完成推理，默认使用的工具如下：
  - `PyTorch -> ONNX`：PyTorch 框架
  - `ONNX -> OM`：ATC
  - OM 模型推理：[ais_bench工具](https://gitee.com/ascend/tools/blob/master/ais-bench_workload/tool/ais_bench/README.md)、[API参考](https://gitee.com/ascend/tools/blob/master/ais-bench_workload/tool/ais_bench/API_GUIDE.md)
- 固化 ATC 参数与目标 SoC
- SoC 判定优先级：
  - P0：优先读取 `source set_env.sh` 后环境中的 `SOC_VERSION`（或等价环境变量）
  - P1：若 P0 为空，则解析 `npu-smi info` 的芯片型号作为候选
  - P2：若仍无法判定，使用用户显式传入值；再为空才使用项目默认值
- SoC 规范化规则：
  - 对 `ascend*` 前缀统一转换为 `Ascend*`（首字母大写）
- 一致性校验规则：
  - 若 `set_env.sh` 推导 SoC 与 `npu-smi info` 不一致，默认以 `set_env.sh` 为准并记录冲突
- 导出元信息必须记录：
  - `runtime_soc_version`
  - `atc_soc_version`
  - `soc_source`（`set_env` / `npu-smi` / `user` / `default`）
- 验证预处理、后处理一致性与输出结构
- 推理后处理必须满足：
  - 同一 `label` 仅保留最高置信度框（top1 per label）
  - 同类其余框不保留
  - 若存在 mask，必须与 bbox 同步按索引过滤
- 推理结果必须输出以下字段：
  - `dedup_policy`（固定值：`keep_top1_per_label`）
  - `raw_num_detections`
  - `num_detections`
  - `num_kept`
- 必须产出性能统计：
  - warmup 次数
  - 有效迭代次数
  - 单阶段耗时
  - 端到端耗时
  - mean / p50 / p90 / p99（ms）
  - FPS
  - 吞吐（img/s）
- 必须产出精度统计：
  - 基线：同输入、同后处理配置下的 PyTorch 结果
  - 检测数差异
  - bbox 坐标误差（L1/最大值）
  - score 误差（MAE/最大值）
  - label 一致率
  - mask IoU（如有）
- 导出交付文档必须补齐：
  - 数据集路径与标注格式（含字段示例）
  - 单卡与多卡可执行命令（真实路径）
  - 导出后文件清单（meta / onnx / om / log）
  - 依赖安装命令（标准安装 + 失败回退 + 安装后验证）
- 推荐输出文件：
  - `perf_summary.json`
  - `accuracy_summary.json`
  - `infer_validation_report.md`

## 禁止动作

- 跳过 `PyTorch -> ONNX -> OM` 的确定性流水线设计
- 未固化 SoC 来源就执行 ATC 导出
- 在 `set_env.sh` 与 `npu-smi info` 冲突时不记录来源冲突
- 未时候用 OM 模型进行推理
- 推理后处理不执行同 label top1 去重
- 缺少性能或精度统计就宣称导出与推理已完成
- 忽略预处理、后处理一致性直接对比结果

## 完成标准

- 同一输入可稳定复现 OM 构建结果
- 导出元信息完整，SoC 来源可追溯
- 推理输出结构、阈值行为、去重策略已验证
- 性能统计与精度统计已完整产出
- 导出交付文档补充项已齐备

## 失败处理

- 若 SoC 无法直接判定，按 P0/P1/P2 顺序回退，并记录 `soc_source`
- 若 `set_env.sh` 与 `npu-smi info` 冲突，默认以 `set_env.sh` 为准并记录冲突
- 若精度不一致，优先回查预处理、后处理、输出结构与去重逻辑
- 若性能统计不稳定，补充分阶段耗时与 warmup 信息后重测

## 下一阶段

- 导出与推理验证完成后，进入阶段5

## 输出结果

- ONNX、OM、meta、log 等导出产物
- `perf_summary.json`
- `accuracy_summary.json`
- `infer_validation_report.md`
- SoC 判定与一致性记录
