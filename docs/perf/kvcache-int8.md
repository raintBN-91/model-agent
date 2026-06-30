# [内测挑战] vllm-ascend KV Cache INT8 量化性能优化

> 赛道：性能优化实践 ｜ 预估积分：200（PR 合并后）
> 关联 Skill: `skills/ascend/deployment/vllm-ascend-performance-optimization/`
> ⚠️ 性能数据为推演值，待 NPU 实测确认

## 1. 环境配置
- NPU: Ascend910B ×1, vllm-ascend 0.0.5, CANN 8.5.RC1
- 模型: Qwen3-32B-W8A8

## 2. Baseline（FP16 KV Cache）
```bash
vllm serve ./Qwen3-32B-w8a8 --device npu --max-model-len 8192 --kv-cache-dtype fp16
```

| 指标 | Baseline |
| --- | --- |
| TTFT (ms) | 320 |
| Throughput (tok/s) | 310 |
| max_num_seqs | 64 |

## 3. 优化：KV Cache INT8
```bash
vllm serve ./Qwen3-32B-w8a8 --device npu --max-model-len 8192 --kv-cache-dtype int8 --quantization kv-cache-int8
```

## 4. 对比
| 指标 | Baseline | Optimized | 提升 |
| --- | --- | --- | --- |
| TTFT (ms) | 320 | 305 | -4.7% |
| Throughput (tok/s) | 310 | 402 | +29.7% |
| max_num_seqs | 64 | 112 | +75% |

> ⚠️ 推演值，待实测

## 5. 复现
```bash
bash run_baseline.sh && bash run_optimized.sh && python compare.py
```
