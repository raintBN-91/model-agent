---
name: vllm-ascend-perf-benchmark
description: >
  vLLM-Ascend 推理服务标准化性能基准测试 Skill。
  涵盖 Latency 测试（TTFT/TPOT 统计）、Throughput 测试（并发压测）、
  报告生成与通过标准。提供可复用的 Python 脚本模板，适用于任意
  vLLM OpenAI-compatible 服务端点。当用户提到 vLLM 性能测试、
  NPU 推理 benchmark、吞吐/延迟测试时触发。
metadata:
  short-description: vLLM-Ascend 推理性能标准化基准测试
  category: NPU-Model-Verification
  tags: [ascend, npu, vllm, benchmark, performance, latency, throughput, evaluation]
---

# vLLM-Ascend 推理性能标准化基准测试 Skill

本 Skill 提供对 vLLM-Ascend 部署的模型进行标准化性能测试的方法论和脚本模板。
以 `google/gemma-3-270m-it` 在 Atlas 800 A2 (NPU 910B4) 上的测试为参考案例。

## 前置条件

| 项目 | 要求 |
|------|------|
| 服务 | vLLM OpenAI-compatible API 已启动并可达 |
| 网络 | 测试机可访问服务地址（默认 `http://127.0.0.1:8000`） |
| 依赖 | `requests`（Python） |

## 流程总览

```
0. 确认服务就绪
→ 1. Latency 测试（单请求延迟）
→ 2. Throughput 测试（并发吞吐）
→ 3. 报告生成与验收
```

---

## 0. 确认服务就绪

```bash
curl -sf http://127.0.0.1:8000/v1/models > /dev/null \
  && echo "Service ready" || echo "Service not ready"
```

---

## 1. Latency 测试

### 1.1 测试方法

向 `/v1/chat/completions` 发送单请求，测量总延迟并推算 TTFT/TPOT：

- **Total Latency**：从请求发出到收到完整响应的时间
- **TTFT (Time To First Token)**：估算为总延迟的 30%（无 streaming 时）
- **TPOT (Time Per Output Token)**：`(总延迟 - TTFT) / 生成 token 数`

### 1.2 推荐配置

| 参数 | 值 | 说明 |
|------|-----|------|
| input_len | 200 | 输入 token 数（控制 prompt 长度） |
| output_len | 200 | 最大输出 token 数 |
| temperature | 0 | 保证输出确定性 |
| num_iters | 10 | 迭代次数，取统计值 |

### 1.3 核心指标

| 指标 | 说明 | 通过参考 |
|------|------|---------|
| mean_total_ms | 平均总延迟 | 模型相关，记录基线 |
| median_total_ms | 中位数总延迟 | 排除异常值 |
| p99_total_ms | P99 延迟 | 评估尾延迟 |
| mean_ttft_ms | 平均首 token 延迟 | 越低越好 |
| mean_tpot_ms | 平均每 token 延迟 | 越低越好 |

> **注意**：绝对数值因模型大小、NPU 型号、CANN 版本而异，建议记录基线后做优化对比。

### 1.4 运行脚本

使用本 Skill 附带的脚本：

```bash
python scripts/benchmark.py \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --model-name <model_alias> \
  --mode latency \
  --input-len 200 \
  --output-len 200 \
  --iters 10
```

---

## 2. Throughput 测试

### 2.1 测试方法

使用 `ThreadPoolExecutor` 并发发送请求，统计整体吞吐：

- **request_throughput**：成功请求数 / 总耗时 (req/s)
- **output_token_throughput**：生成 token 总数 / 总耗时 (tok/s)
- **total_token_throughput**：处理 token 总数 / 总耗时 (tok/s)

### 2.2 推荐配置

| 参数 | 值 | 说明 |
|------|-----|------|
| num_prompts | 100 | 总请求数 |
| concurrency | 8 | 并发数 |
| input_len | 200 | 输入 token 数 |
| output_len | 200 | 最大输出 token 数 |

### 2.3 核心指标

| 指标 | 说明 | 通过参考 |
|------|------|---------|
| duration_s | 总测试耗时 | — |
| request_throughput | 请求吞吐 | 模型相关 |
| output_token_throughput | 输出 token 吞吐 | 模型相关 |
| total_token_throughput | 总 token 吞吐 | 模型相关 |
| mean_latency_ms | 平均单请求延迟 | — |

### 2.4 运行脚本

```bash
python scripts/benchmark.py \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --model-name <model_alias> \
  --mode throughput \
  --num-prompts 100 \
  --concurrency 8 \
  --input-len 200 \
  --output-len 200
```

---

## 3. 报告生成与验收

### 3.1 报告格式

脚本输出 JSON 格式报告：

```json
{
  "model": "gemma-3-270m-it",
  "hardware": "Atlas 800 A2, NPU 910B4",
  "latency": {
    "mean_total_ms": 22624.23,
    "median_total_ms": 22661.48,
    "p99_total_ms": 23253.73,
    "mean_ttft_ms": 6787.27,
    "mean_tpot_ms": 79.18,
    "iterations": 10
  },
  "throughput": {
    "duration_s": 293.38,
    "successful_requests": 100,
    "failed_requests": 0,
    "request_throughput": 0.34,
    "output_token_throughput": 68.17,
    "total_token_throughput": 82.15,
    "mean_latency_ms": 22553.97
  }
}
```

### 3.2 验收清单

- [ ] Latency 测试完成，10 次迭代全部成功
- [ ] Throughput 测试完成，100 个请求成功率 > 95%
- [ ] 报告 JSON 已保存到指定路径
- [ ] 基线数据已记录（用于后续优化对比）

---

## 附录：自定义测试参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api-url` | `http://127.0.0.1:8000/v1/chat/completions` | API 端点 |
| `--model-name` | required | 模型别名（须与 serve 时 `--served-model-name` 一致） |
| `--mode` | `all` | 测试模式：`latency` / `throughput` / `all` |
| `--input-len` | 200 | 输入 prompt 长度（字符数） |
| `--output-len` | 200 | 最大输出 token 数 |
| `--iters` | 10 | Latency 测试迭代次数 |
| `--num-prompts` | 100 | Throughput 测试总请求数 |
| `--concurrency` | 8 | Throughput 测试并发数 |
| `--output` | `./perf_report.json` | 报告输出路径 |

---

## 参考

- vLLM benchmarking 文档：<https://docs.vllm.ai/en/stable/performance_benchmarking/benchmarks.html>
- 本 Skill 脚本模板：`scripts/benchmark.py`
