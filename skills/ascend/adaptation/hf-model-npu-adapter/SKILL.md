---
name: hf-model-npu-adapter
description: "Universal Hugging Face model to Ascend NPU adaptation skill. Use when the user wants to adapt an arbitrary Hugging Face model (text/vision/audio/multimodal) to run on Huawei Ascend NPUs, and no model-specific adaptation skill exists. This skill provides a generic 7-phase adaptation pipeline: model architecture analysis, operator compatibility gating, adapter template matching, framework integration (vLLM-Ascend/MindSpeed/MindIE), inference validation, performance profiling, and delivery of a comprehensive adaptation report. Acts as the universal fallback entry point for models not covered by existing specialized adaptation skills."
---

# HuggingFace Model NPU Adapter

A universal adaptation skill for running arbitrary Hugging Face models on Huawei Ascend NPUs. This skill serves as the **generic fallback adapter** when no model-specific adaptation skill exists in the repository.

## When to Use This Skill

- User provides a Hugging Face model ID or URL and wants to run it on Ascend NPU
- No existing model-specific skill (e.g., `boltz2`, `chronos-2-npu`, `winclip`) covers the target model
- User wants to assess whether a model can be adapted before committing engineering effort
- User needs a structured adaptation report showing feasibility, risks, and estimated effort

## How to Use (Agent Execution Guide)

### Triggering the Skill

The skill is activated when the user expresses an intent to adapt a model to Ascend NPU. Typical trigger phrases:

```
"帮我把 meta-llama/Llama-3-8B 适配到 NPU"
"这个模型能跑在昇腾上吗？https://huggingface.co/mistralai/Mistral-7B"
"检查一下 Qwen/Qwen2-7B 的 NPU 兼容性"
"评估 google/gemma-7b 迁移到 Ascend 的工作量"
```

### Agent Execution Flow

Once triggered, the Agent executes the 7-phase pipeline automatically. The user does **not** need to guide individual steps — the Agent reads this SKILL.md and follows it:

| Phase | What the Agent Does | User Involvement |
|---|---|---|
| 1. Model Identification | Fetches `config.json` from HF / local path; classifies modality, attention type, dtype | May need to confirm model ID or provide local path |
| 2. Operator Compatibility | Scans model source code for CUDA/Triton/PyTorch ops; gates on blocking ops | **EARLY EXIT if blocked** — Agent reports which ops block adaptation and why |
| 3. Template Matching | Selects Template A-F based on model architecture | None (automatic) |
| 4. Framework Integration | Creates/modifies vLLM-Ascend or MindIE files; registers model | None (automatic) |
| 5. Inference Validation | Smoke test → cosine similarity vs CPU baseline → top-k accuracy | None (automatic) |
| 6. Performance Profiling | TTFT / TPOT / throughput / memory measurements | None (automatic) |
| 7. Adaptation Report | Generates structured Markdown report with all findings | User reviews final report |

### Expected Output

After the pipeline completes, the Agent delivers a structured **Adaptation Report** containing:

- **Status**: ✅ Adapted / ⚠️ With caveats / ❌ Blocked
- **Architecture Summary**: model type, parameters, attention mechanism, dtype
- **Operator Compatibility Matrix**: passed / warning / blocked counts with details
- **Validation Results**: cosine similarity score, precision check
- **Performance Metrics**: TTFT, TPOT, throughput, NPU memory
- **Files Created/Modified**: exact paths in the repository
- **Known Limitations & Next Steps**

### Example Session

```
User:  "帮我把 google/gemma-7b 适配到 Ascend NPU"

Agent: [Phase 1] 正在获取 google/gemma-7b 的模型配置...
       → 识别为 Decoder-Only LLM，GQA 注意力，bf16，8.5B 参数

Agent: [Phase 2] 扫描算子兼容性...
       → 7 个 PyTorch 原生 op ✅
       → 2 个 Triton kernel ⚠️ (需精度验证)
       → 0 个 CUDA-only op 🔴
       → 兼容性关卡：PASS，进入 Phase 3

Agent: [Phase 3] 匹配适配模板 → Template A (标准 Decoder-Only LLM)
Agent: [Phase 4] 框架集成 → vLLM-Ascend 通道，注册模型...
Agent: [Phase 5] 推理验证 → 余弦相似度 0.998 ✅
Agent: [Phase 6] 性能分析 → TTFT 45ms, TPOT 12ms, 吞吐 89 token/s

Agent: [Phase 7] 生成适配报告:
       # NPU Adaptation Report: google/gemma-7b
       ## Summary
       - Status: ✅ Adapted
       - Framework: vLLM-Ascend
       - Template: A
       - Effort: 3h
       
       ...（完整报告）
```

### Quick Scan Mode

If the user only wants a feasibility check without full adaptation, add `--scan-only` in the user request:

| Request | Agent Behavior |
|---|---|
| `"评估 Qwen/Qwen2.5-72B 能不能上昇腾"` | Run Phase 1-2 only, output feasibility verdict + gating result |
| `"检查 deepseek-ai/DeepSeek-V3 兼容性"` | Same as above |
| `"完整适配 meta-llama/Llama-3-8B"` | Run all 7 phases |

## Adaptation Phases

### Phase 1 — Model Identification & Retrieval

Extract the model identifier from user input. Accepts the following formats:

| Input Format | Example |
|---|---|
| HF model ID | `meta-llama/Llama-3-8B` |
| HF URL | `https://huggingface.co/Qwen/Qwen2-7B` |
| Local path | `/data/models/bert-finetuned` |

**Retrieval Actions:**
1. Fetch `config.json` from the model repository or local path
2. Extract: `model_type`, `architectures`, `torch_dtype`, `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`
3. Classify the model modality:

```
Model Modality
├─ Text (LLM)           → Llama/Qwen/GLM/Mistral/GPT-NeoX/Phi/Gemma/Bloom/...
├─ Text (Encoder)       → BERT/RoBERTa/DeBERTa/ALBERT/XLMRoBERTa/...
├─ Vision               → ViT/Swin/ConvNeXt/DINO/CLIP-Vision/...
├─ Speech/Audio         → Whisper/Wav2Vec2/HuBERT/EnCodec/...
├─ Multimodal           → Qwen2-VL/Llava/Llama-3.2-Vision/InternVL/...
└─ Diffusion            → StableDiffusion/Flux/DALL-E/...
```

4. Classify attention mechanism:

| Attention Type | Detection Cue |
|---|---|
| Standard MHA | `num_key_value_heads == num_attention_heads` |
| GQA (Grouped Query) | `num_key_value_heads < num_attention_heads` |
| MQA (Multi-Query) | `num_key_value_heads == 1` |
| MLA (Multi-Latent) | `attention_bias` present or MLA-specific config keys |
| Sliding Window | `sliding_window` or `use_sliding_window` in config |
| Mamba/SSM | `model_type` contains `mamba` or `ssm` |
| Flash Attention | Uses `_flash_attn` or `attn_implementation == "flash_attention_2"` |

5. Check quantization:
   - `quantization_config` present → note method (GPTQ/AWQ/FP8/BitsAndBytes)
   - `torch_dtype` → note (float16/bfloat16/float32)

### Phase 2 — Operator Compatibility Gating

Scan model architecture for operators and classify against Ascend NPU compatibility:

| Operator Category | Ascend Status | Handler |
|---|---|---|
| Native PyTorch ops (`torch.nn.*`, `torch.*`) | ✅ Pass | Note potential performance gaps |
| `torch._C._nn.*` (canonical) | ✅ Pass | Standard aten fallback |
| Triton kernels (`triton.jit`, `triton.*`) | ⚠️ Warning | Flag for correctness verification |
| Custom CUDA kernels (`torch.cuda.*`, `CUDA_HOME`) | 🔴 Blocked | Must have PyTorch fallback |
| `einops` / `torch.einsum` | ✅ Pass | Standard PyTorch backend |
| `flash_attn` (CUDA-only) | 🔴 Blocked | Must switch to `sdpa` or Ascend flash attn |
| `xformers` | 🔴 Blocked | Must remove or replace |

**Gating Decision Matrix:**

```
All ops pass → Proceed to Phase 3
Contains ⚠️ Warning ops → Proceed to Phase 3 with caution flags
Contains 🔴 Blocked ops but fallback exists → Proceed with fallback, document
Contains 🔴 Blocked ops with NO fallback → EARLY EXIT: report blocked, suggest alternatives
```

**Early Exit Protocol:** When blocked ops have no fallback, produce a structured report:
```
MODEL: {model_id}
STATUS: ❌ Blocked
BLOCKING_OPS:
  - {op_name}: CUDA kernel in {file}:{line} — no PyTorch fallback
  - {op_name}: Uses flash_attn CUDA — no sdpa alternative
RECOMMENDATION:
  - Option 1: Replace {op} with PyTorch equivalent
  - Option 2: Use AscendC to implement {op} as custom Ascend op
  - Option 3: Wait for Ascend ecosystem support
ESTIMATED EFFORT: {estimate} engineering hours
```

### Phase 3 — Adapter Template Matching

Based on model architecture and modality, select the appropriate adaptation strategy from the template matrix:

#### Template A: Standard Decoder-Only LLM
**Applicable to:** Llama/Qwen/GLM/Mistral/GPT-NeoX/Phi/Gemma/Bloom
**Framework:** vLLM-Ascend (primary), MindIE (alternative)
**Strategy:**
- Register model in `vllm/model_executor/models/registry.py`
- Add `{ModelName}ForCausalLM` class inheriting from vLLM model base
- Override `load_weights` for Ascend weight mapping
- Configure Ascend-specific attention backend (ACLGraph)
- Add dtype conversion hooks for Ascend native precision

**Key Files to Create/Modify:**
```
vllm/model_executor/models/{model_name_lower}.py   # New model file
vllm/model_executor/models/registry.py             # Register entry
vllm/transformers_utils/configs/{model_name}.py    # Config adapter (if needed)
```

**Reference Implementation:**
Follow the pattern of existing vLLM-Ascend adapted models:
- `vllm/model_executor/models/qwen2.py` → Qwen architecture pattern
- `vllm/model_executor/models/llama.py` → Llama architecture pattern

#### Template B: BERT-Style Encoder
**Applicable to:** BERT/RoBERTa/DeBERTa/ALBERT/XLMRoBERTa
**Framework:** Custom torch-npu pipeline
**Strategy:**
- Wrap model in ONNX export workflow
- Convert via ATC (Ascend Tensor Compiler) to OM format
- Build inference pipeline with ACL (Ascend Computing Language) runtime
- Validate accuracy against CPU/GPU baseline

**Key Components:**
```python
# 1. ONNX Export
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)

# 2. ATC Conversion
# atc --model=model.onnx --framework=5 --output=model --soc_version=Ascend910B

# 3. ACL Inference
# Load OM model via acl.mdl.load_from_file()
# Execute via acl.mdl.execute()
```

#### Template C: Vision Model
**Applicable to:** ViT/Swin/ConvNeXt/DINO
**Framework:** torch-npu + ATC
**Strategy:**
- Verify `torch_npu` backend compatibility
- Trace model with `torch.jit.trace` for graph capture
- Apply Ascend fusion passes: Conv+BatchNorm, Linear+GELU
- Profile with `torch_npu.profiler` for hotspots

#### Template D: Audio/Speech Model
**Applicable to:** Whisper/Wav2Vec2/HuBERT
**Framework:** MindIE or custom torch-npu pipeline
**Strategy:**
- Identify encoder-decoder boundary
- Adapt encoder for NPU static graph optimization
- Handle variable-length audio input with Ascend dynamic shape
- Verify timestamps and alignment accuracy

#### Template E: Multimodal (VLM)
**Applicable to:** Qwen2-VL/Llava/InternVL/Phi-3-Vision
**Framework:** vLLM-Ascend with multimodal extension
**Strategy:**
- Separate vision encoder and language backbone
- Adapt vision encoder via Template C
- Adapt language backbone via Template A
- Bridge multimodal projector with Ascend-compatible ops
- Validate image preprocessing pipeline on Ascend

#### Template F: Diffusion Model
**Applicable to:** StableDiffusion/Flux/SDXL
**Framework:** torch-npu with diffusion pipeline
**Strategy:**
- Replace CUDA-specific scheduler ops with PyTorch ops
- Adapt VAE decoder for Ascend
- Apply UNet operator fusion via Ascend graph capture
- Validate output image quality (FID/CLIP score)

### Phase 4 — Framework Integration

Based on the matched template, execute framework-specific integration:

#### vLLM-Ascend Pathway

1. **Register model architecture:**
   ```python
   # In vllm/model_executor/models/registry.py
   _MODELS = {
       # ... existing entries ...
       "{model_name_capitalized}ForCausalLM": ("{model_file}", "{class_name}"),
   }
   ```

2. **Implement model file** with Ascend-specific hooks:
   ```python
   class {Model}ForCausalLM(LlamaForCausalLM):  # or appropriate base
       def load_weights(self, weights):
           # Handle Ascend weight loading, dtype conversion
           ...
       def forward(self, *args, **kwargs):
           # Ensure Ascend-compatible attention
           ...
   ```

3. **Attention backend configuration:**
   - Default: `ACLGraph` (Ascend native)
   - Fallback: `torch_sdpa` for attention patterns not yet supported by ACLGraph

4. **Weight mapping:** Map HuggingFace weight keys to vLLM weight keys, handling:
   - `model.layers.X.self_attn.*` → Ascend attention layout
   - `model.layers.X.mlp.*` → Ascend MLP layout
   - Quantized weights (GPTQ/AWQ) → Ascend quantized format

#### MindIE Pathway

1. **MindIE model registration** via `mindie.register_model()`
2. **Dynamic shape configuration** for variable batch size
3. **Ascend graph compilation** settings for optimal throughput

### Phase 5 — Inference Validation

Execute a staged validation:

1. **Smoke Test:** Load model, run single forward pass, verify no crash
2. **Correctness Test:** Compare output logits/tokens against CPU reference (cosine similarity > 0.99)
3. **Precision Test:** Run on known inputs, compare top-k accuracy against GPU baseline
4. **Functional Test:** Test generation (LLM) / classification (Encoder) / detection (Vision) with expected outputs

**Validation Checklist:**
```
[ ] Model loads without OOM on target NPU
[ ] Single inference produces non-NaN output
[ ] Cosine similarity vs CPU/GPU baseline > 0.99
[ ] Token generation deterministic with same seed
[ ] Batch inference produces consistent results
[ ] FP16/BF16 precision matches expected range (no underflow/overflow)
```

### Phase 6 — Performance Profiling

Profile the adapted model and generate a performance report:

| Metric | Measurement Method |
|---|---|
| First Token Latency (TTFT) | `torch_npu.profiler` or `time.perf_counter()` around prefill |
| Per-Token Latency (TPOT) | Average over 100 decode steps |
| Throughput (tokens/s) | Total tokens / wall clock time |
| NPU Memory Usage | `torch_npu.memory_allocated()` / `torch_npu.max_memory_allocated()` |
| Memory Utilization | Allocated / Total NPU memory |

**Optimization Levers (if performance is suboptimal):**
- Increase batch size to improve NPU utilization
- Enable `torch.compile` with Ascend backend
- Apply KV cache optimization (prefix caching)
- Enable continuous batching for serving scenarios
- Use NPU fusion passes for operator fusion

### Phase 7 — Adaptation Report

Generate a structured report summarizing the entire adaptation:

```markdown
# NPU Adaptation Report: {model_id}

## Summary
- **Status:** ✅ Adapted / ⚠️ Adapted with caveats / ❌ Blocked
- **Framework:** {vLLM-Ascend / MindIE / custom}
- **Template Used:** {A/B/C/D/E/F}
- **Estimated Engineering Effort:** {hours}h

## Architecture Analysis
- Type: {LLM/Encoder/Vision/Audio/VLM/Diffusion}
- Parameters: {N}B
- Attention: {MHA/GQA/MQA/MLA/Sliding/Mamba}
- Dtype: {float16/bfloat16/float32}
- Quantization: {none/GPTQ/AWQ/FP8}

## Operator Compatibility
- Passed: {N} ops
- Warnings: {N} ops (list with file locations)
- Blocked: {N} ops (list with fallback suggestions)

## Inference Validation
- Cosine Similarity: {value} (threshold: 0.99)
- Top-1 Accuracy: {value}
- Generation Consistency: {pass/fail}

## Performance Profile (on 910B)
- TTFT: {value}ms
- TPOT: {value}ms
- Throughput: {value} tokens/s
- NPU Memory: {value}GB / {total}GB

## Files Created/Modified
- {list of files with paths}

## Known Limitations
- {limitation 1}
- {limitation 2}

## Next Steps
- {recommended action 1}
- {recommended action 2}
```

## Cross-Skill Collaboration

This skill interoperates with other adaptation skills in the ecosystem:

| Preceding Skill | Relationship |
|---|---|
| `model-series-vendor-detector` | Determine vendor/model family before adaptation |
| `ascend-cuda-compat-check` | Pre-scan for CUDA dependencies before starting |
| `hardware-check-principle` | Validate NPU hardware capability for model size |

| Following Skill | Relationship |
|---|---|
| `ascend-model-verification` | Deeper correctness validation post-adaptation |
| `vllm-ascend-performance-optimization` | Performance tuning after basic adaptation |
| `ascend-profiling` | Detailed profiling of adapted model |

## Error Handling

| Error | Cause | Resolution |
|---|---|---|
| `config.json` not found | Invalid model ID or network issue | Verify model ID, check HF availability |
| OOM during loading | Model too large for NPU memory | Suggest model sharding, quantization, or CPU offloading |
| `torch_npu` not available | NPU driver/CANN not installed | Return to Phase 2 early exit with setup guide |
| Operator not supported on NPU | Custom CUDA kernel | Report early exit with blocking ops |
| NaN in output | Numerical instability on NPU | Check dtype, suggest mixed precision or specific op fixes |
| Shape mismatch | Dynamic shape not handled | Add dynamic shape configuration or padding |
