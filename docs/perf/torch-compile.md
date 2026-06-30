# [内测挑战] torch.compile reduce-overhead 模式加速 NPU 推理

> 赛道：性能优化实践 ｜ 预估积分：200（PR 合并后）
> 关联 Skill: `skills/ascend/optimization/ascend-optimization/`
> 关联部署: `skills/ascend/deployment/esm2-npu/`
> ⚠️ 性能数据为推演值，待 NPU 实测确认

## 1. 环境配置
- NPU: Ascend910B ×1, torch 2.9.0, torch_npu 2.9.0
- 模型: ESM2-3B

## 2. Baseline（eager 模式）
| batch | 时延(ms) | 吞吐(seq/s) |
| --- | --- | --- |
| 1 | 85 | 11.8 |
| 8 | 410 | 19.5 |

## 3. 优化：torch.compile reduce-overhead
```python
torch_npu.npu.config.allow_internal_format = True
model = torch.compile(model, backend='npu', mode='reduce-overhead')
```

## 4. 对比
| batch | Baseline | Optimized | 提升 |
| --- | --- | --- | --- |
| 1 | 85ms | 52ms | -38.8% |
| 8 | 410ms | 280ms | -31.7% |

> ⚠️ 推演值，待实测
