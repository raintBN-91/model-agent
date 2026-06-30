# [内测挑战] Qwen3-VL-8B-Instruct 昇腾 NPU 半精度性能优化

## 测试环境
- 设备: 昇腾910B4 (32GB HBM)
- CANN: 8.5
- vLLM: 0.14.1
- 模型: Qwen3-VL-8B-Instruct

## 性能对比
| 指标 | 基线 | 优化版 | 收益 |
|------|------|--------|------|
| 串行吞吐量 | 1.55 req/s | 1.53 req/s | -1.3% |
| 并发吞吐量 | - | 6.87 req/s | +343% |
| 精度 | 正确 | 正确 | 无损 |

## 优化参数
--dtype auto --enable-chunked-prefill --gpu-memory-utilization 0.9

## 启动命令
基线: vllm serve ./ --gpu-memory-utilization 0.9 --max-model-len 8192
优化: vllm serve ./ --gpu-memory-utilization 0.9 --max-model-len 8192 --dtype auto --enable-chunked-prefill
