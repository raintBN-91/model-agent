---
name: diffusers-ascend-adaptation
description: Adaptation and verification skill for Diffusers-based image/video generation models (e.g., FLUX, Stable Diffusion) on Huawei Ascend NPU. Use when users want to run a Diffusers pipeline on Ascend NPU, validate image generation compatibility, scan operator compatibility for diffusion transformers/VAE, or generate adaptation documentation for non-LLM generative models.
---

# Diffusers Ascend Adaptation

An AI-assisted skill for adapting Diffusers-based image/video generation models to run on Ascend NPU. This skill complements `vllm-ascend-model-adapter` (which focuses on LLM/VLM) by covering the diffusion model lifecycle: pipeline loading, text encoder compatibility, transformer denoising, VAE decoding, and image generation validation.

## When to Use This Skill

- Adapting a Diffusers image generation model (FLUX, SD, SDXL, etc.) to Ascend NPU
- Validating if an existing Diffusers checkpoint runs correctly on Ascend NPU
- Scanning diffusion transformer or VAE code for CUDA/Triton operators that may block Ascend support
- Troubleshooting pipeline loading or image generation failures on Ascend
- Generating adaptation documentation for GitCode model cards

## When NOT to Use This Skill

- The model is a pure LLM/VLM served via vLLM-Ascend -> use `vllm-ascend-model-adapter` instead
- The model is a pure PyTorch model without Diffusers pipeline structure -> use `ai4s-basic` or `ascend-optimization`

## Adaptation Playbook

### Step 1 -- Collect Context

Confirm with the user:

- **Model path**: HuggingFace path, ModelScope path, or local checkpoint directory
- **Model family**: FLUX, Stable Diffusion, SDXL, LDM, or custom Diffusers pipeline
- **Key components** (from `model_index.json`): text encoder, transformer/UNet, VAE, scheduler
- **Hardware**: NPU device count and memory (e.g., Ascend 910B4 32GB)
- **Expected resolutions**: 512x512, 1024x1024, or custom

> **[CHECKPOINT]** Confirm with user before proceeding: Is the model path accessible and does the hardware match expectations?

#### Pause and Confirm Protocol

At each `[CHECKPOINT]` in this playbook, the agent must:

1. Report current status to the user
2. Wait for explicit approval or apply the documented fallback
3. Log the decision point in the validation report

### Step 2 -- Analyze Model Structure

Inspect `model_index.json` and sub-component `config.json` files:

```
Diffusers Pipeline
├── text_encoder     -> transformers model (e.g., Qwen3ForCausalLM, T5EncoderModel, CLIPTextModel)
├── tokenizer        -> tokenizer config
├── transformer/UNet -> diffusion backbone (e.g., Flux2Transformer2DModel, UNet2DConditionModel)
├── vae              -> AutoencoderKL or variant
└── scheduler        -> FlowMatchEulerDiscreteScheduler, DDPMScheduler, etc.
```

**Checklist:**
- Read `model_index.json` to identify pipeline class and component types
- Read `transformer/config.json` or `unet/config.json` for architecture details
- Check if pipeline class exists in installed `diffusers` version
- Check if text encoder class exists in installed `transformers` version
- Verify all `.safetensors` or `.bin` weight files are present

### Step 3 -- Operator Compatibility Gate

Scan the transformer/UNet and VAE modeling code for operators:

| Operator Type | Ascend Compatibility | Action |
|--------------|----------------------|--------|
| **Torch** native | Functional | Monitor performance |
| **Triton** kernel | Uncertain | Verify on NPU; may fail |
| **CUDA** with fallback | Use fallback | Document fallback path |
| **CUDA** without fallback | **BLOCKED** | Early exit; file issue |
| **Custom ATen ops** | Check `torch_npu` mapping | Test explicitly |

**Diffusion-specific scan targets:**
- `nn.Conv2d` / `nn.ConvTranspose2d` in VAE
- `nn.Linear`, `nn.LayerNorm`, `nn.RMSNorm` in transformer
- `F.scaled_dot_product_attention` or custom attention
- RoPE / positional embedding implementations
- `torch.einsum` and `torch.bmm` in attention modules

**Early-exit rule:** If any core denoising operator is pure CUDA with no Torch fallback, stop and document the blocker.

> **[CHECKPOINT]** If any core denoising operator is pure CUDA with no Torch fallback, STOP and report blocker to user. Do not proceed to Step 5.

### Step 4 -- Environment Preparation

```bash
pip install diffusers transformers accelerate safetensors
export ASCEND_RT_VISIBLE_DEVICES=0
```

Verify versions:
```python
import diffusers, transformers, torch, torch_npu
print(diffusers.__version__, transformers.__version__, torch_npu.__version__)
```

### Step 5 -- Stage A: Pipeline Loading Gate

Load the pipeline on CPU first to verify structure integrity:

```python
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=False,
)
print(type(pipe).__name__)
print(list(pipe.components.keys()))
```

**Success criteria:**
- All 5 components load without error
- Pipeline class name matches `model_index.json`
- Weight loading completes (check for missing shards)

> **[CHECKPOINT]** Verify all 5 components loaded. If any component missing or class mismatch, STOP and investigate before moving to NPU.

### Step 6 -- Stage B: NPU Inference Gate

Move pipeline to NPU and run a minimal inference:

```python
import torch, torch_npu
device = torch.device("npu:0")
pipe = pipe.to(device)

image = pipe(
    prompt="a photo of an astronaut riding a horse",
    num_inference_steps=20,
    guidance_scale=3.5,
    height=512,
    width=512,
    max_sequence_length=512,
).images[0]
image.save("test_output.png")
```

**Success criteria:**
- Pipeline moves to NPU without OOM
- Inference completes all denoising steps
- Output image is valid and saved

> **[CHECKPOINT]** Confirm pipeline moved to NPU without OOM. If OOM occurs, apply Fallback Ladder Step 1-2 before retry.

### Step 7 -- Feature Validation Matrix

Run a comprehensive validation covering:

| Test Case | Purpose |
|-----------|---------|
| 512x512 @ 20 steps | Baseline quality/speed |
| 1024x1024 @ 20 steps | High-resolution support |
| 512x512 @ 10 steps | Fast generation |
| 512x512 @ 50 steps | Quality ceiling |
| guidance_scale = 1.0 | Unconditional-like behavior |
| guidance_scale = 7.0 | Strong prompt adherence |
| Long prompt (512 tokens) | Text encoder capacity |
| Chinese / non-English prompt | Multilingual support |
| 3 consecutive runs | Stability and memory leak check |

**Record:** time per run, peak NPU memory, output image integrity.

### Step 8 -- Validation Test Prompts & Expected Metrics

Use the exact prompts below in validation scripts to ensure reproducible evaluation.

#### Standard Test Prompts

| Test Case | Prompt | Expected Behavior |
|-----------|--------|-------------------|
| Baseline 512x512 | `a photo of an astronaut riding a horse` | Coherent image, 512x512 output |
| High-res 1024x1024 | `a photo of an astronaut riding a horse` | Coherent image, 1024x1024 output |
| Fast generation | `a red apple on a wooden table` | Lower step count, faster inference |
| Quality ceiling | `a serene lake surrounded by mountains` | Higher step count, finer details |
| Unconditional-like | `a futuristic city skyline at sunset, digital art` | guidance_scale=1.0, less prompt adherence |
| Strong guidance | `a futuristic city skyline at sunset, digital art` | guidance_scale=7.0, strong prompt adherence |
| Long prompt | A highly detailed digital painting of a magical forest with glowing mushrooms, ancient trees, a misty atmosphere, fireflies, and a small wooden cabin in the background, fantasy art style, trending on artstation, 8k resolution, ultra detailed | No truncation errors, full encoding |
| Chinese prompt | `一只可爱的熊猫在吃竹子，中国画风格` | Correct Chinese character handling |

#### Performance Baseline (Ascend 910B4, bfloat16)

| Resolution | Steps | Expected Time | Peak Memory | Pass Threshold |
|-----------|-------|---------------|-------------|----------------|
| 512x512 | 20 | ~4-5 s (warm) | ~15 GB | < 10 s, < 20 GB |
| 1024x1024 | 20 | ~14-15 s (warm) | ~16 GB | < 25 s, < 20 GB |
| 512x512 | 10 | ~2-3 s (warm) | ~15 GB | < 6 s |
| 512x512 | 50 | ~10-12 s (warm) | ~15 GB | < 20 s |

> **Note:** First-run includes CPU->NPU weight migration, which adds ~15-20 s overhead.

#### Stability Criteria

- 3 consecutive runs with standard deviation of time < 0.5 s
- Memory usage delta between runs < 1 GB
- No `RuntimeError` across all runs

#### Validation Command

```bash
cd <repo-dir>
python3 flux_npu_validate.py > flux_validation_log.txt 2>&1
```

Expected artifacts: `flux_validation_log.txt`, `flux_validation_report.json`

### Step 9 -- Operator Scan & Report

Use `named_modules()` to enumerate all modules in the transformer and VAE:

```python
transformer = pipe.transformer
ops = {}
for name, module in transformer.named_modules():
    cls = type(module).__name__
    ops[cls] = ops.get(cls, 0) + 1
```

Document:
- Total module count and unique types
- Top 10 most frequent module types
- Any CUDA or Triton operators found
- Ascend compatibility verdict

### Step 10 -- Generate Artifacts

**Required files:**
1. `README.md` -- adaptation documentation with YAML frontmatter
2. `flux_npu_inference.py` -- minimal runnable inference script
3. `flux_npu_validate.py` -- self-validation script covering Steps 5-7
4. `flux_validation_report.json` -- structured JSON report
5. `flux_validation_log.txt` -- raw terminal log (via shell redirection)

**README.md frontmatter:**
```yaml
---
tags:
- model-agent-tagged
- diffusers
- NPU
- text-to-image
library_name: diffusers
pipeline_tag: text-to-image
license: apache-2.0
hardware: NPU
---
```

### Step 11 -- Deliver & Publish

1. Backport all files to the delivery repo
2. Ensure exactly one commit or a clean commit history
3. Push to GitCode with `gitcode-publish` or manual push
4. Verify the model card renders correctly on GitCode

## Boundary Conditions & Exception Handling

### Exception Decision Table

| Symptom | Likely Cause | Immediate Action | Fallback |
|---------|-------------|------------------|----------|
| `OutOfMemoryError` during `.to(device)` | Model > NPU HBM | Reduce resolution to 256x256 or use CPU offload | Fallback Ladder Step 6 |
| `ValueError: Provide either prompt or prompt_embeds` | Positional arg passed as `image` | Use keyword argument `prompt="..."` | -- |
| `AttributeError: module 'diffusers' has no attribute 'Flux2KleinPipeline'` | diffusers version too old | Upgrade diffusers >= 0.38.0 | Reinstall dependencies |
| `KeyError` in `model_index.json` loading | Corrupt checkpoint or missing shards | Re-download weights | Verify file manifest |
| RuntimeError on `scaled_dot_product_attention` | torch_npu sdpa incompatibility | Export `TORCH_NPU_SDT_FA_ENABLE=0` | Environment variable toggle |
| Output is pure noise / black image | bfloat16 unsupported on NPU | Switch to `torch.float16` | Dtype fallback |
| `ImportError: cannot import name 'Qwen3ForCausalLM'` | transformers < 4.57.0 | Upgrade transformers | Reinstall dependencies |
| Pipeline loads but NPU inference hangs | CANN driver / runtime mismatch | Check CANN version >= 8.0 | Reinstall CANN toolkit |

### Resolution & Memory Bounds

| Resolution | Approx. Peak Memory | Status |
|-----------|---------------------|--------|
| 256x256 | ~8 GB | Minimum tested, safe on 16 GB cards |
| 512x512 | ~15 GB | Baseline, safe on 32 GB cards |
| 1024x1024 | ~16 GB | Maximum tested, safe on 32 GB cards |
| > 2048x2048 | > 32 GB | Likely OOM on 32 GB cards; not recommended |

### Step-wise Failure Isolation

- **If Stage A (Pipeline Loading) fails:** Do not proceed to Stage B. Investigate `model_index.json`, missing weights, or version mismatches first.
- **If operator scan finds CUDA ops:** Document blockers and STOP before NPU inference. File an issue with the operator list.
- **If Stage B fails at 512x512:** Fallback to 256x256 resolution before declaring the model incompatible.
- **If feature validation fails on a single case:** Run the Fallback Ladder for that specific case; do not abort the entire validation.

## Fallback Ladder

When inference fails, follow this ordered approach:

```
1. Reproduce once to confirm deterministic failure
        ↓
2. Reduce resolution to 256x256 to isolate memory issues
        ↓
3. Reduce num_inference_steps to 10 to isolate stepwise failures
        ↓
4. Use torch.float16 instead of bfloat16 for dtype compatibility
        ↓
5. Set torch.npu.set_per_process_memory_fraction(0.8) to limit memory
        ↓
6. Check if text_encoder fails independently (load only text_encoder to NPU)
        ↓
7. Apply targeted code fix, loop back to Step 5
```

## Quick Reference

### Common Diffusers Pipelines on Ascend

| Model Family | Pipeline Class | Key Component | Typical Memory |
|-------------|----------------|---------------|----------------|
| FLUX.1/2 | FluxPipeline / Flux2KleinPipeline | Flux2Transformer2DModel | ~15 GB (bf16) |
| SD 1.5 | StableDiffusionPipeline | UNet2DConditionModel | ~4-6 GB (fp16) |
| SDXL | StableDiffusionXLPipeline | UNet2DConditionModel | ~6-8 GB (fp16) |

### Performance Baselines (Ascend 910B4)

| Resolution | 20 steps | Typical Time |
|-----------|----------|--------------|
| 512x512 | 20 | ~4-5 s (pipeline warm) |
| 1024x1024 | 20 | ~14-15 s (pipeline warm) |

> First-run includes CPU->NPU weight migration, which adds ~15-20 s overhead.

## Related Resources & References

### Related Skills

| Skill | When to Use |
|-------|-------------|
| `vllm-ascend-model-adapter` | For LLM/VLM models served via vLLM-Ascend (not Diffusers pipelines) |
| `ai4s-basic` | For pure PyTorch models without Diffusers pipeline structure |
| `ascend-optimization` | For performance tuning after successful adaptation |

### External Reference Documentation

- [Diffusers FLUX pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux)
- [CANN QuickStart Guide](https://www.hiascend.com/document/detail/zh/canncommercial/800/quickstart/quickstart_0001.html)
- [torch_npu GitHub](https://github.com/Ascend/pytorch)
- [ModelScope FLUX.2-klein-base-4B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-4B)
- [HuggingFace FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)

### File Manifest Template

Each adaptation must generate the following files:

| File | Purpose | Generated By |
|------|---------|-------------|
| `README.md` | Adaptation documentation with YAML frontmatter | Agent (Step 10) |
| `*_npu_inference.py` | Minimal runnable inference script | Agent (Step 10) |
| `*_npu_validate.py` | Self-validation script (Steps 5-7) | Agent (Step 10) |
| `*_validation_report.json` | Structured JSON report | Validation script |
| `*_validation_log.txt` | Raw terminal log (shell redirection) | User / CI |

## Output Requirements

Every adaptation must produce:
- Minimal diff or standalone scripts (no changes to diffusers/transformers source)
- `README.md` with YAML frontmatter and complete validation results
- `*_npu_validate.py` self-validation script
- `*_validation_report.json` structured report
- `*_validation_log.txt` raw log via shell redirection
- Optional: `*_npu_inference.py` minimal example script
