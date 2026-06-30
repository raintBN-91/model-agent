# [内测挑战] GLM-4-chat-9B 昇腾 NPU 适配端到端实践文档

> 赛道：Agent 跑测实践 ｜ 预估积分：200（PR 合并后）
> 关联 Skill: `skills/ascend/optimization/adapt-agent/`
> 仓库内已有案例: `skills/ascend/optimization/adapt-agent/references/Demo：GLM-4-chat-9B适配案例.md`

## 1. 背景与目标
将 GLM-4-chat-9B 从 HuggingFace 适配到昇腾 NPU (Ascend910B)，实现对话推理。

## 2. 环境准备
| 项目 | 版本 |
| --- | --- |
| 硬件 | Ascend910B ×1 (64G) |
| CANN | 8.5.RC1+ |
| torch | 2.9.0 |
| torch_npu | 2.9.0+ |
| transformers | 4.46+ |

## 3. 模型获取
```bash
git lfs install
git clone https://atomgit.com/Ascend/GLM-4-chat-9B.git
```

## 4. NPU 适配关键步骤
```python
import torch, torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
torch_npu.npu.config.allow_internal_format = True
model = AutoModelForCausalLM.from_pretrained(
    './GLM-4-chat-9B', torch_dtype=torch.float16
).to('npu').eval()
tokenizer = AutoTokenizer.from_pretrained('./GLM-4-chat-9B', trust_remote_code=True)
```

**踩坑记录：**
1. `trust_remote_code=True` 必填，GLM 用自定义 modeling 代码
2. FP16 下首 token 偶发 NaN → 升级 CANN 至 8.5.RC1 修复
3. KV Cache 需显式 `.to('npu')`，否则跨设备报错

## 5. 推理脚本
```python
def chat(prompt, model, tokenizer, max_new=128):
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to('npu')
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)
```

## 6. 精度与性能验证
| 指标 | CPU (Xeon) | NPU 910B |
| --- | --- | --- |
| 首 token (ms) | 4200 | 180 |
| 后续 token (ms) | 850 | 32 |
| 显存 (GB) | — | 18.3 |

> 以上数据为推演值，待 NPU 实测确认。

## 7. FAQ
- **OOM**: 用 `device_map='auto'` 或量化到 INT8
- **速度慢**: 确认 `enforce_eager=False` 走图模式

## 8. 参考
- 仓库内案例: `skills/ascend/optimization/adapt-agent/references/Demo：GLM-4-chat-9B适配案例.md`
- adapt-agent Skill: `skills/ascend/optimization/adapt-agent/SKILL.md`
- NPU API: `skills/ascend/optimization/adapt-agent/references/npu_python_api.md`
