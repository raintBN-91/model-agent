---
name: ascend-benchmark-runner
description: >
  vLLM-Ascend 性能基准测试参数自适应与标准化报告生成专家。
  当用户需要在昇腾 NPU 上跑 vLLM 性能测试时，自动检测当前 vLLM 版本并推断
  bench latency/throughput/serve 的正确 CLI 参数，执行标准化测试套，
  最后生成 Markdown + JSON 结构化性能报告。
  触发场景包括："跑benchmark"、"性能测试"、"吞吐"、"延迟"、
  "vllm bench参数报错"、"测一下性能"、"benchmark一下"。
---

# ascend-benchmark-runner — vLLM-Ascend 性能基准测试标准化

## 核心工作流程

### 阶段 1：环境与版本诊断

**任务 1.1：检测 vLLM 版本**

```bash
vllm --version 2>&1 || python3 -c "import vllm; print(vllm.__version__)" 2>&1
```

记录主版本号（如 `0.6.1` → `0.6`），后续参数推断依赖此版本。

**任务 1.2：检测 NPU 状态**

```bash
npu-smi info
npu-smi info -t memory
```

确认目标 NPU 卡号、可用内存，避免 benchmark 因 OOM 失败。

**任务 1.3：检测当前 vLLM 服务状态**

```bash
# 检查本地是否有 serve 在运行
curl -s http://localhost:8000/v1/models > /dev/null 2>&1 && echo "service_up" || echo "service_down"
```

- 若服务在运行：优先使用 **在线压测模式**（`vllm bench serve` 或 `benchmark_serving.py`）
- 若服务未运行：使用 **离线测试模式**（`vllm bench latency` / `vllm bench throughput`）

### 阶段 2：参数自适应推断

**任务 2.1：版本-参数映射表**

根据检测到的 vLLM 版本，选择对应的参数集：

| 版本范围 | bench 子命令 | 关键参数差异 |
|---------|-------------|-------------|
| v0.4.x | `vllm benchmark_latency` / `benchmark_throughput` | 独立脚本，参数为 `--model`, `--input-len`, `--output-len`, `--batch-size` |
| v0.5.x | `vllm bench latency` / `vllm bench throughput` | 新增 `--num-iters`, `--num-batches`；`--tensor-parallel-size` 变为 `-tp` |
| v0.6.x+ | `vllm bench latency` / `vllm bench throughput` / `vllm bench serve` | 统一为 `vllm bench` 子命令；`--num-prompts` 替代 `--num-batches`；新增 `--request-rate` |

**任务 2.2：自动参数推断逻辑**

```python
# 伪代码逻辑，实际通过 shell 条件判断实现
version = detect_vllm_version()

if version < "0.5.0":
    latency_cmd = "python -m vllm.benchmarks.benchmark_latency"
    throughput_cmd = "python -m vllm.benchmarks.benchmark_throughput"
    serve_cmd = None  # 无原生 serve bench
elif version < "0.6.0":
    latency_cmd = "vllm bench latency"
    throughput_cmd = "vllm bench throughput"
    serve_cmd = None
else:
    latency_cmd = "vllm bench latency"
    throughput_cmd = "vllm bench throughput"
    serve_cmd = "vllm bench serve"
```

**任务 2.3：测试配置选择**

根据用户意图选择内置配置（见 `configs/` 目录）：

| 配置名称 | 适用场景 | 关键参数 |
|---------|---------|---------|
| `single_batch_latency.json` | 单 batch 延迟基线 | batch_size=1, input_len=128, output_len=128 |
| `offline_throughput.json` | 离线吞吐测试 | batch_size=16/32/64, input_len=512, output_len=256 |
| `online_serving.json` | 在线服务压测 | num_prompts=1000, request_rate=inf/10/5 |

### 阶段 3：执行标准化测试

#### 测试 A：单 batch 延迟测试

```bash
# v0.6.x+ 示例
vllm bench latency \
    --model $MODEL_PATH \
    --input-len 128 \
    --output-len 128 \
    --batch-size 1 \
    --tensor-parallel-size $TP_SIZE \
    --device npu \
    2>&1 | tee benchmark_latency_single.log

# v0.4.x/v0.5.x 兼容命令
python -m vllm.benchmarks.benchmark_latency \
    --model $MODEL_PATH \
    --input-len 128 \
    --output-len 128 \
    --batch-size 1 \
    --tensor-parallel-size $TP_SIZE \
    2>&1 | tee benchmark_latency_single.log
```

#### 测试 B：离线吞吐测试

```bash
# v0.6.x+ 示例
vllm bench throughput \
    --model $MODEL_PATH \
    --dataset sharegpt \
    --num-prompts 1000 \
    --max-model-len 4096 \
    --tensor-parallel-size $TP_SIZE \
    --device npu \
    2>&1 | tee benchmark_throughput_offline.log

# v0.5.x 兼容命令（使用 --num-batches）
vllm bench throughput \
    --model $MODEL_PATH \
    --dataset sharegpt \
    --num-batches 1000 \
    --max-model-len 4096 \
    --tensor-parallel-size $TP_SIZE \
    --device npu \
    2>&1 | tee benchmark_throughput_offline.log
```

#### 测试 C：在线服务压测

```bash
# 首先确认服务已启动（若未启动，提示用户先启动或调用 ascend-resource-scheduler）
curl -s http://localhost:8000/v1/models > /dev/null || echo "ERROR: No vLLM serve detected"

# v0.6.x+ 使用 bench serve
vllm bench serve \
    --model $MODEL_NAME \
    --dataset sharegpt \
    --num-prompts 1000 \
    --request-rate 10 \
    --endpoint /v1/completions \
    2>&1 | tee benchmark_serve_online.log

# 旧版本使用 benchmark_serving.py
python -m vllm.benchmarks.benchmark_serving \
    --model $MODEL_NAME \
    --dataset sharegpt \
    --num-prompts 1000 \
    --request-rate 10 \
    --endpoint /v1/completions \
    2>&1 | tee benchmark_serve_online.log
```

### 阶段 4：结果解析与报告生成

**任务 4.1：解析 latency 输出**

```bash
# 提取关键指标
LATENCY_MEAN=$(grep -E "Avg latency|Mean latency|mean" benchmark_latency_single.log | tail -1 | awk '{print $NF}')
LATENCY_P50=$(grep -E "P50|50th percentile" benchmark_latency_single.log | tail -1 | awk '{print $NF}')
LATENCY_P99=$(grep -E "P99|99th percentile" benchmark_latency_single.log | tail -1 | awk '{print $NF}')
TOKEN_THROUGHPUT=$(grep -E "Throughput.*token|tokens/sec" benchmark_latency_single.log | tail -1 | awk '{print $NF}')
```

**任务 4.2：解析 throughput 输出**

```bash
REQ_THROUGHPUT=$(grep -E "Throughput.*request|requests/s" benchmark_throughput_offline.log | tail -1 | awk '{print $NF}')
TOKEN_THROUGHPUT=$(grep -E "Throughput.*token|tokens/s" benchmark_throughput_offline.log | tail -1 | awk '{print $NF}')
```

**任务 4.3：解析 serve 输出**

```bash
# 在线压测指标
REQ_THROUGHPUT=$(grep -E "Request throughput|requests/s" benchmark_serve_online.log | tail -1 | awk '{print $NF}')
TOKEN_THROUGHPUT=$(grep -E "Token throughput|tokens/s" benchmark_serve_online.log | tail -1 | awk '{print $NF}')
TTFT_P50=$(grep -E "TTFT.*P50|Time to first token.*50th" benchmark_serve_online.log | tail -1 | awk '{print $NF}')
TPOT_P50=$(grep -E "TPOT.*P50|Time per output token.*50th" benchmark_serve_online.log | tail -1 | awk '{print $NF}')
```

**任务 4.4：生成结构化报告**

输出以下文件：

1. `benchmark_report.md` — Markdown 性能报告
2. `benchmark_report.json` — 结构化 JSON，便于后续自动化处理

```markdown
# vLLM-Ascend 性能基准测试报告

## 测试环境
| 项目 | 内容 |
|------|------|
| 测试时间 | YYYY-MM-DD HH:mm:ss |
| vLLM 版本 | X.X.X |
| 模型 | model_name |
| NPU 设备 | Ascend-910B (卡号 X) |
| Tensor Parallel | TP=X |
| 测试模式 | 离线延迟 / 离线吞吐 / 在线压测 |

## 测试结果

### 单 Batch 延迟测试
| 指标 | 数值 |
|------|------|
| Mean Latency | XX ms |
| P50 Latency | XX ms |
| P99 Latency | XX ms |
| Token Throughput | XX tokens/s |

### 离线吞吐测试
| 指标 | 数值 |
|------|------|
| Request Throughput | XX req/s |
| Token Throughput | XX tokens/s |

### 在线服务压测 (request_rate=10)
| 指标 | 数值 |
|------|------|
| Request Throughput | XX req/s |
| Token Throughput | XX tokens/s |
| TTFT P50 | XX ms |
| TPOT P50 | XX ms |

## 原始日志
- `benchmark_latency_single.log`
- `benchmark_throughput_offline.log`
- `benchmark_serve_online.log`
```

```json
{
  "meta": {
    "test_time": "YYYY-MM-DD HH:mm:ss",
    "vllm_version": "X.X.X",
    "model": "model_name",
    "npu_device": "Ascend-910B",
    "tensor_parallel": 1,
    "test_mode": "latency|throughput|serve"
  },
  "latency": {
    "mean_ms": 0,
    "p50_ms": 0,
    "p99_ms": 0,
    "token_throughput": 0
  },
  "throughput": {
    "request_throughput": 0,
    "token_throughput": 0
  },
  "serve": {
    "request_throughput": 0,
    "token_throughput": 0,
    "ttft_p50_ms": 0,
    "tpot_p50_ms": 0
  }
}
```

## 异常处理规则

| 异常情况 | 处理方案 |
|---------|---------|
| `vllm bench` 命令不存在 | 回退到 `python -m vllm.benchmarks.*` |
| 参数报错（如 `--num-iterations` 未知） | 根据版本映射表自动替换为对应版本的合法参数 |
| OOM / 内存不足 | 减小 batch_size 或 max-model-len，重试一次 |
| 模型未找到 | 检查 MODEL_PATH 是否存在，尝试从 HF_CACHE 推断 |
| serve 压测时服务未启动 | 提示用户先启动 vllm serve，或调用 ascend-resource-scheduler 调度 |
| benchmark 脚本不存在 | 提示用户安装完整 vLLM：`pip install vllm[benchmark]` |
