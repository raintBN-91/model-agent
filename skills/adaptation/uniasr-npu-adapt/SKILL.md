---
name: uniasr-npu-adapt
description: >
  Adapt ModelScope UniASR speech recognition models (TF1 frozen graph wrapped as PyTorch weights)
  to Huawei Ascend NPU. Handles component name mapping between config.yaml and funasr registry,
  model construction, weight loading, NPU deployment, and accuracy verification.
  Triggers: UniASR NPU, speech_UniASR Ascend, uniasr 昇腾适配, funasr NPU inference,
  ASR model Ascend, TF1-to-PyTorch Ascend, ModelScope ASR NPU.
tags:
  - ascend
  - npu
  - asr
  - uniasr
  - funasr
  - modelscope
  - speech-recognition
  - torch_npu
  - tf-to-pytorch
---

# UniASR NPU Adaptation Skill

## Trigger Scenarios

Use this skill when:
- Adapting `iic/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline` or similar UniASR models to Ascend NPU
- A ModelScope ASR model's `model.pb` is actually a PyTorch ZIP archive with pre-converted weights
- Component names in `config.yaml` don't match funasr's registered class names
- Need to verify NPU inference accuracy against CPU baseline

## Verified Environment

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11.14 | |
| torch | 2.9.0 | |
| torch_npu | 2.9.0.post1 | Ascend NPU backend |
| CANN | 8.5.1 | |
| funasr | 1.3.1 | UniASR PyTorch implementation |
| modelscope | 1.35.3 | Model download |
| soundfile | latest | Audio loading (ffmpeg unavailable on aarch64) |
| NPU | 2x Ascend 910 | Atlas 800 A2 |

## Adaptation Steps

### Step 1: Download Model

```python
from modelscope import snapshot_download
model_dir = snapshot_download(
    'iic/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline',
    cache_dir='./'
)
```

### Step 2: Discover Weight Format — Critical

**The `model.pb` is NOT a TensorFlow frozen graph.** It is a ZIP archive containing pre-converted PyTorch weights.

```bash
file model.pb   # "Zip archive data"
xxd model.pb | head -1  # "PK" (ZIP magic)
```

Load directly:
```python
import torch
state_dict = torch.load('model.pb', map_location='cpu', weights_only=False)
# 1103 keys, including stride_conv.conv.weight: [880, 880, 2]
```

### Step 3: Map Component Names

config.yaml uses old naming. Map to funasr 1.3.1 registered names:

| config.yaml | funasr 1.3.1 |
|-------------|-------------|
| `sanm_chunk_opt` | `SANMEncoderChunkOpt` |
| `fsmn_scama_opt` | `FsmnDecoderSCAMAOpt` |
| `cif_predictor_v2` | `CifPredictorV2` |
| `specaug_lfr` | `SpecAugLFR` |
| `wav_frontend` | `WavFrontend` (matches) |

Apply mappings and remove unused keys:
```python
name_map = {
    'sanm_chunk_opt': 'SANMEncoderChunkOpt',
    'fsmn_scama_opt': 'FsmnDecoderSCAMAOpt',
    'cif_predictor_v2': 'CifPredictorV2',
    'specaug_lfr': 'SpecAugLFR',
}
for key in ['encoder', 'encoder2', 'decoder', 'decoder2',
             'predictor', 'predictor2', 'specaug']:
    old = config.get(key)
    if old in name_map:
        config[key] = name_map[old]
config.pop('stride_conv', None)
if config.get('normalize') is None:
    config.pop('normalize', None)
```

### Step 4: Build Model

```python
from funasr.register import tables
from funasr.models.uniasr.model import UniASR

# Frontend (handles FBank extraction + LFR + CMVN)
frontend = tables.frontend_classes.get('WavFrontend')(
    n_mels=80, fs=16000, frame_length=25, frame_shift=10,
    lfr_m=7, lfr_n=6, cmvn_file='am.mvn'
)
input_size = frontend.output_size()  # 560 = 80 * 7

model = UniASR(
    specaug='SpecAugLFR', specaug_conf=config['specaug_conf'],
    normalize=None, normalize_conf={},
    encoder='SANMEncoderChunkOpt', encoder_conf=config['encoder_conf'],
    encoder2='SANMEncoderChunkOpt', encoder2_conf=config['encoder2_conf'],
    decoder='FsmnDecoderSCAMAOpt', decoder_conf=config['decoder_conf'],
    decoder2='FsmnDecoderSCAMAOpt', decoder2_conf=config['decoder2_conf'],
    predictor='CifPredictorV2', predictor_conf=config['predictor_conf'],
    predictor_bias=0, predictor_weight=1.0,
    predictor2='CifPredictorV2', predictor2_conf=config['predictor2_conf'],
    predictor2_bias=0, predictor2_weight=1.0,
    stride_conv_conf={'kernel_size': 2, 'stride': 2, 'pad': [0, 1]},
    ctc=None, ctc_conf={}, ctc_weight=0.0,
    ctc2=None, ctc2_conf={}, ctc2_weight=0.0,
    decoder_attention_chunk_type='chunk',
    decoder_attention_chunk_type2='chunk',
    loss_weight_model1=0.5, input_size=input_size, vocab_size=8359,
    ignore_id=0, blank_id=0, sos=1, eos=2,
    lsm_weight=0.1, length_normalized_loss=True, share_embedding=False,
)
```

**Architecture note**: `stride_conv` idim/odim = `input_size + encoder_output_size` = 560 + 320 = **880**. This matches the checkpoint weight shape `[880, 880, 2]`.

### Step 5: Load Weights and Move to NPU

```python
state_dict = torch.load('model.pb', map_location='cpu', weights_only=False)
model.load_state_dict(state_dict, strict=True)  # 1103/1103 perfect

import torch_npu
model = model.to('npu:0')
model.eval()
```

### Step 6: Inference

```python
import soundfile as sf
from funasr.utils.load_utils import extract_fbank

# Load audio (soundfile avoids ffmpeg dependency)
audio_data, sr = sf.read('audio.wav', dtype='float32')
if audio_data.ndim > 1:
    audio_data = audio_data.mean(axis=1)
audio_tensor = torch.from_numpy(audio_data.copy())
if sr != 16000:
    import torchaudio.functional as F
    audio_tensor = F.resample(audio_tensor.unsqueeze(0), sr, 16000).squeeze(0)

# Extract features
feats, feat_lens = extract_fbank(audio_tensor, data_type='sound', frontend=frontend)
feats = feats.to(torch.float32).to('npu:0')
feat_lens = feat_lens.to('npu:0')

# Tokenizer wrapper
class CharTokenizer:
    def __init__(self, token_list):
        self.token_list = token_list
    def ids2tokens(self, ids):
        return [self.token_list[i] if i < len(self.token_list) else '<UNK>' for i in ids]
    def tokens2text(self, tokens):
        return ''.join(tokens)

result = model.inference(
    data_in=feats, data_lengths=feat_lens,
    tokenizer=CharTokenizer(token_list),
    key=['audio'], device='npu:0',
    data_type='fbank', decoding_model='offline',
    beam_size=5, lm_weight=0.7,
    token_list=token_list, nbest=1,
    frontend=frontend, penalty=0.0,
    maxlenratio=0.0, minlenratio=0.0,
    ctc_weight=0.0, token_num_relax=5, decoding_ind=1,
)
text = result[0][0]['text']
```

### Step 7: Accuracy Verification

Compare NPU vs CPU with identical seed:
```python
torch.manual_seed(42)
with torch.no_grad():
    _, enc_cpu, _ = model_cpu.encode(feats_cpu, feat_lens_cpu, ind=0)
    enc2_cpu, _ = model_cpu.encode2(enc_cpu, enc_lens, feats_cpu, feat_lens_cpu, ind=0)

torch.manual_seed(42)
with torch.no_grad():
    _, enc_npu, _ = model_npu.encode(feats_npu, feat_lens_npu, ind=0)
    enc2_npu, _ = model_npu.encode2(enc_npu, enc_lens_npu, feats_npu, feat_lens_npu, ind=0)

cos1 = F.cosine_similarity(enc_cpu.reshape(-1), enc_npu.cpu().reshape(-1), dim=0)
cos2 = F.cosine_similarity(enc2_cpu.reshape(-1), enc2_npu.cpu().reshape(-1), dim=0)
mae1 = (enc_cpu - enc_npu.cpu()).abs().mean()
mae2 = (enc2_cpu - enc2_npu.cpu()).abs().mean()
```

**Expected**: cosine similarity > 0.999999, relative error < 0.03%.

## Verified Results

| Metric | Value |
|--------|-------|
| Weight keys | 1103/1103 (100%) |
| Encoder1 cosine sim | 1.00000060 |
| Encoder2 cosine sim | 1.00000060 |
| Encoder1 rel error | 0.023% |
| Encoder2 rel error | 0.029% |
| ASR text match | 100% (NPU = CPU) |
| NPU inference (5.6s audio) | 1.35s, RTF 0.24 |
| CPU inference (same audio) | 3.10s |
| NPU speedup | 2.48x |

## Common Issues

### `TypeError: 'NoneType' object is not callable` on model build
**Root cause**: Config name doesn't match funasr registry. **Fix**: Apply Step 3 name mapping.

### `FileNotFoundError: ffmpeg`
**Root cause**: torchaudio 2.9.0 requires torchcodec or ffmpeg, neither on aarch64.
**Fix**: Use `soundfile` for audio loading.

### `argument after ** must be a mapping, not NoneType` for stride_conv
**Root cause**: `stride_conv_conf` missing from model kwargs.
**Fix**: Always include `stride_conv_conf={'kernel_size': 2, 'stride': 2, 'pad': [0, 1]}`.

### `got multiple values for keyword argument 'decoding_mode'`
**Root cause**: Passing both `decoding_mode` and `decoding_model` to inference.
**Fix**: Only pass `decoding_model` ('offline'/'fast'/'normal').

### `'NoneType' object is not subscriptable` at `key[0]`
**Root cause**: `key` parameter not passed to inference.
**Fix**: Always pass `key=['name']` to inference.
