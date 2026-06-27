# Config Name → funasr Registry Mapping

## Why This Matters

ModelScope ASR models (especially older UniASR variants) use config.yaml with component names from older funasr versions. Current funasr (1.3.1) registers classes under different names. Without mapping, model construction fails with `'NoneType' object is not callable`.

## Complete Mapping Table

| config.yaml key | config.yaml value | funasr 1.3.1 registered name | funasr table |
|-----------------|-------------------|------------------------------|-------------|
| `encoder` | `sanm_chunk_opt` | `SANMEncoderChunkOpt` | `encoder_classes` |
| `encoder2` | `sanm_chunk_opt` | `SANMEncoderChunkOpt` | `encoder_classes` |
| `decoder` | `fsmn_scama_opt` | `FsmnDecoderSCAMAOpt` | `decoder_classes` |
| `decoder2` | `fsmn_scama_opt` | `FsmnDecoderSCAMAOpt` | `decoder_classes` |
| `predictor` | `cif_predictor_v2` | `CifPredictorV2` | `predictor_classes` |
| `predictor2` | `cif_predictor_v2` | `CifPredictorV2` | `predictor_classes` |
| `specaug` | `specaug_lfr` | `SpecAugLFR` | `specaug_classes` |
| `frontend` | `wav_frontend` | `WavFrontend` | `frontend_classes` |
| `model` | `uniasr` | `UniASR` | `model_classes` |
| `stride_conv` | `stride_conv1d` | *(remove - not used)* | - |
| `normalize` | `null` | *(remove - not used)* | - |

## Architecture Decoded from Weights

```
Frontend (WavFrontend): raw_audio → FBank(80) + LFR(m=7,n=6) + CMVN → [1, T, 560]

StrideConv (Conv1dSubsampling): [1, T, 880] → [1, T/2, 880]
  └─ Applied AFTER encoder1, BEFORE encoder2
  └─ idim = input_size + encoder_output_size = 560 + 320 = 880
  └─ odim = 880 (same), kernel=2, stride=2

Encoder1 (SANMEncoderChunkOpt, 35 blocks):
  └─ encoders0: pre-block [560→320] with FSMN kernel=11
  └─ encoders.0..33: 34× SANM blocks [320], kernel_size=11, chunk=[20,60]

Encoder2 (SANMEncoderChunkOpt, 20 blocks):
  └─ encoders0: pre-block [880→320] with FSMN kernel=21
  └─ encoders.0..18: 19× SANM blocks [320], kernel_size=21, chunk=[45,70]

Decoder1 (FsmnDecoderSCAMAOpt, 6 blocks):
  └─ embed [8359, 256], attn_dim=256, linear_units=1024
  └─ 6× SCAMA decoder blocks with kernel_size=11

Decoder2 (FsmnDecoderSCAMAOpt, 6 blocks):
  └─ embed [8359, 320], attn_dim=320, linear_units=1280
  └─ 6× SCAMA decoder blocks with kernel_size=11

Predictor1 (CifPredictorV2): idim=320, threshold=1.0
Predictor2 (CifPredictorV2): idim=320, threshold=1.0
```

## Key Architecture Insights

1. **num_blocks in config represents total blocks** (pre-block + regular blocks):
   - encoder_conf.num_blocks=35 → 1 pre-block (encoders0) + 34 regular blocks
   - encoder2_conf.num_blocks=20 → 1 pre-block (encoders0) + 19 regular blocks

2. **stride_conv input dimension** = frontend output (560) + encoder1 output (320) = 880
   - This is computed internally as `input_size + encoder_output_size`

3. **Decoder block count** is 6, matching `att_layer_num=6` in config
   - `num_blocks=12` with `att_layer_num=6` creates 6 attention + 6 FFN blocks

4. **No CTC layers** — ctc_weight=0.0, so no CTC-related weights in checkpoint
