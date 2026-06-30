---
name: hubert-npu-deploy
description: "HuBERT 语音表征模型在昇腾NPU上的端到端部署与推理验证 Skill。覆盖环境初始化、torch_npu 接入、ONNX 转 OM、推理验证与精度对齐全流程。当用户提到'部署 HuBERT 到 NPU'、'HuBERT 昇腾推理'、'语音表征 NPU'、'hubert on ascend' 时使用此 Skill。适用于 facebook/hubert-base-ls960 等.encoder only 语音模型，不做微调/训练。"
---

# HuBERT 昇腾 NPU 端到端部署

将 Facebook HuBERT（Hidden-Unit BERT）语音表征模型部署到华为昇腾 NPU（Ascend910B），完成从环境搭建到推理验证与精度对齐的完整链路。仅做推理，不涉及训练/微调。

## 输入参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `model_name` | 否 | 默认 `facebook/hubert-base-ls960`，可选 `facebook/hubert-large-ls960-ft` |
| `audio_path` | 是 | 待推理的 16kHz 单声道 wav 文件路径 |
| `device_id` | 否 | NPU 卡号，默认 `0`，通过 `npu-smi info` 选择空闲卡 |
| `precision` | 否 | `fp32`（默认）或 `fp16` |

---

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910B（至少 1 卡，HBM ≥ 32GB） |
| OS | openEuler 22.03 / Ubuntu 22.04（aarch64 或 x86_64） |
| CANN | ≥ 8.0.RC1 |
| Python | 3.10 / 3.11 |
| 网络 | 首次需联网下载权重（base ~379MB，large ~1.2GB） |

---

## 流程总览

```
0. 环境初始化 → 1. 依赖安装 → 2. NPU 可用性检查 → 3. 模型下载
→ 4. PyTorch+NPU 推理验证 → 5. ONNX 导出 → 6. ATC 转 OM
→ 7. OM 推理与精度对齐 → 8. 性能 benchmark → 9. 验收
```

每步完成后进入下一步，任一步失败立即停止并输出 `FAIL_REASON.md`。

---

## Phase 0: 环境初始化

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=${device_id:-0}
npu-smi info
```

> **✅ 检查点 0**: `npu-smi info` 至少显示 1 张健康卡（Health=OK），否则停止。

## Phase 1: 依赖安装

```bash
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 transformers==4.46.0 soundfile librosa onnx onnxruntime
# torch_npu 版本须与本地 CANN 严格配套，请查官方配套表选对应版本：
# https://www.hiascend.com/document/detail/zh/Pytorch/60 configuringandinstallingtheenvironmentandinstallationsequence/0001.html
# 示例（CANN 8.3.RC1 + torch 2.5.1，aarch64）：
#   pip install torch_npu -i https://pypi.huawei.com/simple/  # 版本号见配套表
# 实际安装前请在配套表中确认 torch_npu 的精确版本号
pip install torch_npu  # 安装与本地 CANN 配套的版本（版本号见配套表，勿盲装）
```

> **⚠️ 待验证说明**：torch_npu 的精确版本号取决于本地 CANN 版本与平台架构，本文未锁定具体版本，请在配套表中查询后填入。本文档所有命令未经真机验证，首次执行时如遇版本不匹配请按配套表调整。

> **✅ 检查点 1**: `python -c "import torch,torch_npu;print(torch_npu.__version__)"` 输出版本号。

## Phase 2: NPU 可用性检查

```python
import torch, torch_npu
print("device count:", torch.npu.device_count())
print("device 0:", torch.npu.get_device_name(0))
x = torch.randn(2,3).npu()
print("npu add ok:", (x+x).sum().item())
```

> **✅ 检查点 2**: 上述脚本无异常退出，能正确打印 device 数与求和结果。

## Phase 3: 模型下载

```python
from transformers import HubertModel, Wav2Vec2FeatureExtractor
m = HubertModel.from_pretrained("facebook/hubert-base-ls960")
f = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
m.save_pretrained("./hubert-base-npu")
f.save_pretrained("./hubert-base-npu")
```

> **✅ 检查点 3**: `./hubert-base-npu/` 下存在 `config.json`、`pytorch_model.bin`（或 `model.safetensors`）、`preprocessor_config.json`。

## Phase 4: PyTorch + NPU 推理验证

```python
import torch, torch_npu, soundfile as sf, librosa
from transformers import HubertModel, Wav2Vec2FeatureExtractor

m = HubertModel.from_pretrained("./hubert-base-npu").to("npu:0").eval()
f = Wav2Vec2FeatureExtractor.from_pretrained("./hubert-base-npu")
audio, sr = sf.read("${audio_path}")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
inputs = f(audio, sampling_rate=16000, return_tensors="pt").input_values.to("npu:0")
with torch.no_grad():
    feats = m(inputs).last_hidden_state   # [1, T, 768]
print("feat shape:", feats.shape, "mean:", feats.float().mean().item())
```

> **✅ 检查点 4**: 输出 shape 形如 `[1, T, 768]`（base）或 `[1, T, 1024]`（large），mean 为有限浮点数。

## Phase 5: ONNX 导出

```python
torch.onnx.export(
    m, inputs, "hubert-base.onnx",
    input_names=["input_values"], output_names=["last_hidden_state"],
    dynamic_axes={"input_values": {0:"batch",1:"length"}, "last_hidden_state": {0:"batch",1:"length"}},
    opset_version=17,
)
```

> **✅ 检查点 5**: `onnx.load("hubert-base.onnx")` 成功，`onnxruntime.InferenceSession` 能在 CPU 上跑通并输出与 Phase 4 一致（cos sim > 0.999）。

## Phase 6: ATC 转 OM

```bash
atc --framework=5 --model=hubert-base.onnx --output=hubert-base \
    --soc=Ascend910B3 --input_shape="input_values:1,${seq_len}" \
    --log=error --insert_op_conf=ascend_aispace.cfg
```

`ascend_aispace.cfg` 用于声学模型的 mean/var 归一化（可省略，已在 FeatureExtractor 内做过）。

> **✅ 检查点 6**: 生成 `hubert-base.om`，`atc` 退出码 0，无 ERROR 日志。

## Phase 7: OM 推理与精度对齐

```python
from ais_bench.inference.interface import InferenceSession
import numpy as np, soundfile as sf, librosa
from transformers import Wav2Vec2FeatureExtractor

f = Wav2Vec2FeatureExtractor.from_pretrained("./hubert-base-npu")
audio, sr = sf.read("${audio_path}")
if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
inp = f(audio, sampling_rate=16000, return_tensors="np").input_values

sess = InferenceSession("hubert-base.om", device_id=0)
out = sess.infer([inp])[0]                # numpy
# 与 Phase 4 的 torch 结果做 cosine 对齐
cos = float(np.sum(out*ref) / (np.linalg.norm(out)*np.linalg.norm(ref)))
print("cos sim:", cos)
assert cos > 0.995, f"精度不达标 cos={cos}"
```

> **✅ 检查点 7**: `cos sim > 0.995`，否则回退检查 ATC 的 `--precision_mode`（建议 `allow_fp32_to_fp16`）。

## Phase 8: 性能 benchmark

```bash
python - <<'PY'
import time, numpy as np, soundfile as sf, librosa
from transformers import Wav2Vec2FeatureExtractor
from ais_bench.inference.interface import InferenceSession
f = Wav2Vec2FeatureExtractor.from_pretrained("./hubert-base-npu")
audio,_ = sf.read("${audio_path}"); audio = librosa.resample(audio,16000,16000) if False else audio
inp = f(audio, sampling_rate=16000, return_tensors="np").input_values
sess = InferenceSession("hubert-base.om", device_id=0)
_ = sess.infer([inp])            # warmup
N=50; t0=time.time()
for _ in range(N): _ = sess.infer([inp])[0]
dt=(time.time()-t0)/N
print(f"avg latency={dt*1000:.2f}ms throughput={1/dt:.2f} req/s")
PY
```

记录到 `benchmark.json`：`{"avg_ms":..., "throughput":..., "soc":"Ascend910B3", "precision":"fp32"}`。

> **✅ 检查点 8**: `benchmark.json` 生成且数值合理（base 模型单条 10s 音频延迟通常 < 200ms）。

## Phase 9: 验收

生成 `README.md`，包含：
1. 环境信息（CANN/torch_npu 版本、SOC）
2. Phase 4 的 NPU 推理输出（shape + mean）
3. Phase 7 的 cos 对齐值
4. Phase 8 的 benchmark 数值与对比表

---

## 边界条件与异常处理

| 场景 | 处理 |
|------|------|
| `npu-smi info` 无设备 | 停止，输出 `FAIL_REASON.md` 说明无可用 NPU |
| `torch_npu` 导入失败 | 检查 CANN env 是否 source；版本不匹配则报错并给出配套表链接 |
| 音频采样率非 16kHz | 自动用 librosa 重采样，日志记录原始采样率 |
| ATC 转换失败 | 先尝试 `--precision_mode=allow_fp32_to_fp16`；仍失败则记录 op 列表到 `atc_fail.log` 并停止 |
| OM 与 torch cos sim < 0.995 | 切换 `--precision_mode=must_keep_origin_dtype` 重转，仍不达标则报 Issue |
| 内存 OOM | base 改用 fp16 重转 OM；large 模型若仍 OOM 建议切到 ≥64GB HBM 的卡 |

---

## 资源文件

| 路径 | 用途 |
|------|------|
| `hubert-base-npu/` | HF 模型本地副本 |
| `hubert-base.onnx` | 中间 ONNX |
| `hubert-base.om` | 昇腾可执行模型 |
| `benchmark.json` | 性能数据 |
| `README.md` / `FAIL_REASON.md` | 验收报告或失败说明 |

---

## 约束

1. **仅推理** — 不做训练/微调/量化训练
2. **不提交权重** — `*.bin/*.safetensors/*.onnx/*.om` 不入 git
3. **精度达标才算成功** — cos sim 必须 > 0.995
4. **真实 NPU 验证** — 不允许只在 CPU 上模拟
5. **可复现** — 所有命令带版本号，benchmark 可重复运行
