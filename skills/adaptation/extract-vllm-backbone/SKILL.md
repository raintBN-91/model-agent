---
name: extract-vllm-backbone
description: Extract the LLM backbone from a non-standard or multimodal model checkpoint so it can be served by vLLM. Use when vLLM fails to load a model due to unsupported architecture (e.g. spatiallm_llama, llava, qwen-vl), when the user only needs text generation from the LLM component, or when the model contains multimodal keys (point_backbone, vision_tower, mm_projector) that are incompatible with vLLM.
---

# Extract vLLM Backbone Skill

Extract the pure LLM backbone weights and configuration from a multimodal or custom-architecture checkpoint, repackage it as a standard vLLM-compatible model, and verify it loads successfully.

## When to Invoke

- vLLM throws `Unsupported model type` or `No architecture named XXX` for a custom `model_type`
- The model checkpoint contains non-LLM keys such as `point_backbone.*`, `vision_tower.*`, `mm_projector.*`, `image_encoder.*`
- User only needs text generation capability and does not require the multimodal encoder on NPU
- The model is based on a known backbone (Llama, Qwen2, etc.) but wrapped in a custom architecture class

## Prerequisites

- Python 3.11+
- `safetensors` library installed (`pip install safetensors`)
- Original model directory containing `config.json` and `model.safetensors`

## Workflow

### Step 1: Analyze the Original Config

Read `config.json` to identify the custom architecture and backbone type:

```bash
python3 -c "import json; c=json.load(open('config.json')); print('model_type:', c.get('model_type')); print('architectures:', c.get('architectures'))"
```

Common patterns:
| Custom Type | Likely Backbone | Keys to Remove |
|-------------|-----------------|----------------|
| `spatiallm_llama` | LlamaForCausalLM | `point_backbone.*`, `point_proj.*` |
| `llava` | LlamaForCausalLM | `vision_tower.*`, `mm_projector.*` |
| `qwen2_vl` | Qwen2ForCausalLM | `visual.*`, `merger.*` |

### Step 2: Filter Safetensors Keys

Load the checkpoint and drop all non-LLM keys. Typical LLM key prefixes to **keep**:
- `model.embed_tokens.*`
- `model.layers.*`
- `model.norm.*`
- `lm_head.*`

Run the extraction script:

```python
from safetensors.torch import load_file, save_file
import json, os

src_dir = "original_model"
dst_dir = "llm_backbone"
os.makedirs(dst_dir, exist_ok=True)

state = load_file(os.path.join(src_dir, "model.safetensors"))
# Remove multimodal / custom keys
llm_keys = [k for k in state.keys() if not k.startswith(("point_backbone.", "point_proj."))]
llm_state = {k: state[k] for k in llm_keys}
save_file(llm_state, os.path.join(dst_dir, "model.safetensors"))
print(f"Kept {len(llm_state)} / {len(state)} keys")
```

### Step 3: Rewrite Config for Standard Architecture

```python
import json

with open(os.path.join(src_dir, "config.json")) as f:
    cfg = json.load(f)

cfg["model_type"] = "llama"          # or "qwen2", "gemma", etc.
cfg["architectures"] = ["LlamaForCausalLM"]
cfg.pop("auto_map", None)

# Remove multimodal-specific fields
for key in list(cfg.keys()):
    if "point" in key.lower() or "vision" in key.lower() or "image" in key.lower():
        cfg.pop(key)

with open(os.path.join(dst_dir, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
```

### Step 4: Copy Tokenizer Files

```bash
cp original_model/tokenizer*.json original_model/special_tokens_map.json \
   original_model/generation_config.json llm_backbone/ 2>/dev/null || true
```

### Step 5: vLLM Dummy Load Gate

Verify vLLM recognizes the new package before wasting time on full graph compilation:

```bash
vllm serve llm_backbone \
  --load-format dummy \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --dtype bfloat16
```

If this starts without `Unsupported model type`, the extraction is structurally correct.

### Step 6: Full Weight Validation

Stop the dummy server and run with real weights:

```bash
vllm serve llm_backbone \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --served-model-name MyModel-LLM
```

Wait for ACL Graph compilation to finish (~30-60s on Ascend 910B), then test:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MyModel-LLM","messages":[{"role":"user","content":"Hello"}]}'
```

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `KeyError: point_backbone.encoder` during load | Forgot to filter keys | Re-run extraction script |
| `size mismatch for lm_head.weight` | Config vocab_size mismatch with tokenizer | Copy original tokenizer.json, verify `vocab_size` in config |
| `torch.compile` fails after load | Hidden multimodal config keys left behind | Strip all fields containing `vision`, `point`, `image` from config |
| Output is garbled / random | Wrong `torch_dtype` or missing `lm_head` | Ensure `lm_head.*` keys were kept in extraction |

## Known Limitations

- This skill only repackages the LLM backbone. The multimodal encoder (e.g. TorchSparse, CLIP ViT) is discarded and cannot be recovered.
- If the custom architecture reuses the LLM layers with modified forward logic (not just added encoders), simple key filtering may not suffice.
