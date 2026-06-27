---
name: npu-benchmark-guard
description: Safely run vLLM bench latency and throughput benchmarks on Ascend NPU by cleaning up zombie processes, validating CLI arguments, waiting for ACL graph compilation, and parsing results. Use when the user wants to benchmark vLLM on Ascend, when a previous benchmark left HBM memory occupied, or when vLLM bench fails with cryptic argument errors.
---

# NPU Benchmark Guard Skill

A defensive wrapper around `vllm bench` for Ascend NPU that prevents common failure modes: memory conflicts from previous runs, incorrect CLI flags, and premature result parsing before graph compilation finishes.

## When to Invoke

- User asks to run performance benchmarks on Ascend NPU with vLLM
- A previous `vllm serve` or `vllm bench` process is suspected to still occupy HBM
- `vllm bench` fails with `Free memory on device less than desired GPU memory utilization`
- User is unsure about the correct vLLM benchmark CLI syntax
- Need to collect structured latency (P50/P90/P99) and throughput metrics for a validation report

## Prerequisites

- vLLM-Ascend installed and NPU driver healthy (`npu-smi info` shows OK)
- Model weights already converted/extracted and accessible on disk
- `jq` or `python3` available for JSON result parsing

## Workflow

### Step 1: Kill Zombie vLLM Processes

Before any benchmark, ensure no previous engine holds NPU memory:

```bash
# Find lingering vLLM engine processes
ps aux | grep -E "vllm|VLLMEngine" | grep -v grep

# Kill them if found (confirm with user if uncertain)
pkill -f VLLMEngine
pkill -f "vllm bench"
sleep 2
```

Verify memory is freed:

```bash
npu-smi info
```

Look for processes in the "PID" column; if none are running, HBM is free.

### Step 2: Validate CLI Arguments

vLLM benchmark subcommands use specific flag names. Common mistakes:

| Wrong | Correct | Command |
|-------|---------|---------|
| `--num-iterations` | `--num-iters` | `bench latency` |
| `--num-batches` | `--random-batch-size` | `bench throughput` |
| `--output-len` | `--output-len` | both (correct) |
| `--input-len` | `--input-len` | both (correct) |

Always consult help when in doubt:

```bash
vllm bench latency --help=all
vllm bench throughput --help=all
```

### Step 3: Run Latency Benchmark

```bash
export VLLM_USE_MODELSCOPE=true

vllm bench latency \
  --model /path/to/model \
  --input-len 200 \
  --output-len 200 \
  --num-iters 10 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --output-json latency_result.json \
  2>&1 | tee bench_latency.log | grep -E "(Avg latency|percentile|Warmup|Bench)"
```

Wait for the benchmark to fully finish. On Ascend NPU the first iteration includes ACL Graph compilation (~30-40s on 910B4); do not interrupt during warmup.

Parse results:

```bash
python3 -c "
import json
d = json.load(open('latency_result.json'))
print(f'P50: {d[\"percentiles\"][\"50\"]:.2f}s')
print(f'P90: {d[\"percentiles\"][\"90\"]:.2f}s')
print(f'P99: {d[\"percentiles\"][\"99\"]:.2f}s')
"
```

### Step 4: Run Throughput Benchmark

```bash
vllm bench throughput \
  --model /path/to/model \
  --input-len 200 \
  --output-len 200 \
  --random-batch-size 16 \
  --dataset-name random \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --output-json throughput_result.json \
  2>&1 | tee bench_throughput.log | grep -E "(Throughput|Total num|Processed prompts)"
```

Parse results:

```bash
python3 -c "
import json
d = json.load(open('throughput_result.json'))
print(f'Requests/s: {d[\"requests_per_second\"]:.2f}')
print(f'Tokens/s:   {d[\"tokens_per_second\"]:.2f}')
"
```

### Step 5: Cleanup and Report

After both benchmarks, kill any remaining background vLLM processes to free HBM for downstream tasks:

```bash
pkill -f VLLMEngine
```

Collect artifacts for the validation report:
- `latency_result.json`
- `throughput_result.json`
- `bench_latency.log`
- `bench_throughput.log`

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Free memory on device less than desired GPU memory utilization` | Previous vLLM process still holds HBM | Kill `VLLMEngine` processes and retry |
| `unrecognized arguments: --num-iterations` | Wrong flag name | Use `--num-iters` for latency benchmark |
| Benchmark hangs at 0% for >2 min | ACL Graph compilation in progress | Wait; do not interrupt |
| `requests_per_second` is 0 or missing | Benchmark crashed mid-way | Check `bench_throughput.log` for OOM or compile errors |
| Output tokens/s is much lower than total tokens/s | Normal; prefill dominates total count | Report both metrics separately |

## Notes

- Always run latency and throughput benchmarks sequentially, not in parallel, to avoid NPU memory contention.
- If benchmarking multiple models back-to-back, insert a cleanup step between each model.
- For CI/automation, consider adding `--enforce-eager` to skip graph compilation, but this will yield slower numbers and is not representative of production performance.
