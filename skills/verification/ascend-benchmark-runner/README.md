# ascend-benchmark-runner

vLLM-Ascend 性能基准测试参数自适应与标准化报告生成 Skill。

## 适用场景

在昇腾 NPU 上完成模型适配后，需要对模型进行性能基准测试。但 vLLM 不同版本的 `bench` 子命令参数差异大：

- v0.4.x：使用 `python -m vllm.benchmarks.benchmark_latency`
- v0.5.x：引入 `vllm bench latency`，参数为 `--num-batches`
- v0.6.x+：统一为 `vllm bench`，`--num-prompts` 替代 `--num-batches`，新增 `vllm bench serve`

手动试错成本高，本 Skill 自动完成版本检测 → 参数推断 → 测试执行 → 报告生成。

## 触发方式

- "跑 benchmark"
- "性能测试"
- "吞吐"
- "延迟"
- "vllm bench 参数报错"
- "测一下性能"

## 核心能力

| 能力 | 说明 |
|------|------|
| 版本自适应 | 自动检测 vLLM 版本，选择对应 bench 命令和合法参数 |
| 服务感知 | 检测本地是否有 vLLM serve 运行，自动选择离线/在线测试模式 |
| 标准测试套 | 内置单 batch 延迟、离线吞吐、在线服务压测三套配置 |
| 结构化报告 | 自动生成 `benchmark_report.md` + `benchmark_report.json` |
| 异常兜底 | 参数不存在时自动回退到兼容命令，OOM 时自动降配重试 |

## 目录说明

| 路径 | 说明 |
|------|------|
| `SKILL.md` | Claude Code Skill 定义，包含完整工作流与参数推断逻辑 |
| `README.md` | 本文件 |
| `configs/single_batch_latency.json` | 单 batch 延迟测试标准配置 |
| `configs/offline_throughput.json` | 离线吞吐测试标准配置 |
| `configs/online_serving.json` | 在线服务压测标准配置 |

## 测试模式

### 1. 单 Batch 延迟测试
衡量最小并发下的端到端推理延迟，用于获取延迟基线。

### 2. 离线吞吐测试
使用大批量输入，衡量 GPU/NPU 利用率最大化的吞吐能力。

### 3. 在线服务压测
在已有 vLLM serve 服务基础上，使用指定 request_rate 模拟并发请求，
输出 TTFT（Time To First Token）和 TPOT（Time Per Output Token）。

## 报告输出

每次 benchmark 完成后，当前目录下会生成：

- `benchmark_latency_single.log` — 延迟测试原始日志
- `benchmark_throughput_offline.log` — 吞吐测试原始日志
- `benchmark_serve_online.log` — 在线压测原始日志
- `benchmark_report.md` — Markdown 性能报告
- `benchmark_report.json` — 结构化 JSON 数据

## 依赖

- 昇腾 NPU 环境（CANN + torch_npu）
- vLLM-Ascend 已安装
- `npu-smi` 工具可用
- benchmark 数据集（默认使用 sharegpt，可自动下载）

## 与现有生态的关系

- `verify-agent`：提供底层 benchmark 脚本执行
- `ascend-resource-scheduler`：若在线压测前检测到 NPU 资源冲突，可自动调度
- `ascend-smoke-validator`：benchmark 前可先执行 smoke 测试确认服务健康

## 贡献

提交 PR 至：https://gitcode.com/Ascend/model-agent
