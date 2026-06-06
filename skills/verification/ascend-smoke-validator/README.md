# ascend-smoke-validator

vLLM-Ascend 在线服务一键 Smoke 验证 Skill。

## 适用场景

vLLM serve 启动后，需要快速验证服务是否真正可用。手动 curl 步骤零散，容易遗漏，且没有结构化记录。

本 Skill 自动探测服务端口，执行标准化 API 测试，生成报告，并在失败时自动诊断原因。

## 触发方式

- "smoke 测试"
- "服务健康检查"
- "curl 测试"
- "验证服务是否可用"
- "测试 API"
- "服务通了吗"

## 核心能力

| 能力 | 说明 |
|------|------|
| 端口自发现 | 通过进程扫描、pid 文件、默认端口三层探测，自动发现 vLLM 服务端点 |
| 标准测试套 | 覆盖 4 个核心用例：模型列表、chat、completion、function calling |
| 结构化报告 | 生成 `smoke_test.log`（人类可读）+ `smoke_report.json`（机器可读） |
| 自动诊断 | 失败时自动诊断 5 类常见原因：端口未监听、模型名不匹配、OOM、启动中、网络问题 |
| 容错设计 | function calling 不支持时标记 SKIP 而非 FAIL，避免误报 |

## 测试用例

| # | 端点 | 方法 | 验证内容 | 是否强制 |
|---|------|------|---------|---------|
| 1 | `/v1/models` | GET | 能正确返回模型列表 | ✅ 强制 |
| 2 | `/v1/chat/completions` | POST | 能正确返回 chat 响应 | ✅ 强制 |
| 3 | `/v1/completions` | POST | 能正确返回 completion 响应 | ✅ 强制 |
| 4 | `/v1/chat/completions` (tools) | POST | 支持 function calling | ⚪ 可选 |

## 目录说明

| 路径 | 说明 |
|------|------|
| `SKILL.md` | Claude Code Skill 定义，包含完整测试流程与诊断逻辑 |
| `README.md` | 本文件 |
| `testcases/chat.json` | chat completions 标准请求体模板 |
| `testcases/completion.json` | completions 标准请求体模板 |
| `testcases/function_calling.json` | function calling 标准请求体模板 |

## 报告输出

每次 smoke 测试后，当前目录下生成：

- `smoke_test.log` — 人类可读的测试结果与诊断信息
- `smoke_report.json` — 结构化 JSON，包含每项测试的状态、错误信息、总体结果

## 依赖

- vLLM serve 已启动（本地或指定主机）
- `curl` 可用
- `python3` 可用（用于 JSON 解析）

## 与现有生态的关系

- `verify-agent`：偏离线验证（模型下载、bench、NPU 状态），本 skill 偏在线服务启动后的快速 API 健康检查，形成互补
- `ascend-resource-scheduler`：如诊断发现 OOM 或服务冲突，可调用进行资源调度
- `ascend-benchmark-runner`：smoke 验证通过后，可继续执行性能基准测试

## 贡献

提交 PR 至：https://gitcode.com/Ascend/model-agent
