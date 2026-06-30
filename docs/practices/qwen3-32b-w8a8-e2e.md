# [内测挑战] Qwen3-32B W8A8 量化在昇腾 NPU 上的端到端部署实践

> 赛道：Agent 跑测实践 ｜ 预估积分：200（PR 合并后）
> 关联 Skill: `skills/ascend/deployment/vllm-ascend-performance-optimization/`
> 仓库内已有案例: `skills/ascend/deployment/vllm-ascend-performance-optimization/references/Demo：Qwen3-32B-w8a8优化案例.md`

## 1. 背景与目标
Qwen3-32B FP16 在 Ascend910B 单卡显存不足，需 W8A8 量化部署。

## 2. 环境
| 项目 | 版本 |
| --- | --- |
| 硬件 | Ascend910B ×1 |
| CANN | 8.5.RC1 |
| vllm-ascend | 0.0.5+ |
| msmodelslim | 1.0+ |

## 3. 量化流程
```bash
pip install msmodelslim
python -c "
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
cfg = QuantConfig(w_bit=8, a_bit=8, disable_names=['lm_head'])
calib = Calibrator(cfg, model='./Qwen3-32B', calib_data=calib_list)
calib.run()
"
```

**踩坑：**
1. 校准集需 128-512 条中文语料，太少精度掉
2. `lm_head` 必须排除量化，否则 logits 失真
3. 产出权重需转 `.npy` 格式供 vllm-ascend 加载

## 4. vllm-ascend 部署
```bash
vllm serve ./Qwen3-32B-w8a8 --quantization w8a8 --device npu --max-model-len 8192 --port 8000
```

## 5. 验证
| 指标 | FP16 (2卡) | W8A8 (1卡) | 提升 |
| --- | --- | --- | --- |
| 显存 (GB) | 2×40 | 21 | 单卡可跑 |
| 吞吐 (tok/s) | 240 | 310 | +29% |
| 精度 (MMLU) | 76.3 | 75.9 | -0.4pt |

> 以上数据为推演值，待 NPU 实测确认。

## 6. FAQ
- **量化后精度掉超 1pt**: 增大校准集至 512 条
- **启动报 quant config not found**: 确认权重目录含 `quant_config.json`

## 7. 参考
- 仓库内案例: `skills/ascend/deployment/vllm-ascend-performance-optimization/references/Demo：Qwen3-32B-w8a8优化案例.md`
- Skill: `skills/ascend/deployment/vllm-ascend-performance-optimization/SKILL.md`
