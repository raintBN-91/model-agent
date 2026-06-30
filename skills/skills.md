# Model Agent Skills

昇腾Model **Agent Skills** **导航表**，所有skills 均经过skills-Eval测评系统验证

### 快速体验 [skills-eval](https://relenting-speed-octopus.ngrok-free.dev/)

在「仓库 URL」里贴上 **Git 仓库地址**，「校验规则」选 **skills-eval 默认** 或 **Anthropic skill-creator 加强**，点 **开始评测** 即可.


## Skills汇总：场景分类

| 场景 | Skills 数量 | 说明 |
| --- | --- | --- |
| [模型适配](#模型适配) | 45 | 模型迁移与适配：GPU→NPU迁移、框架转换、AI for Science模型移植、Megatron/MindSpeed迁移 |
| [模型部署](#模型部署) | 114 | 单模型NPU部署管线：部署、推理、精度对比、性能基线、环境配置、驱动安装 |
| [性能优化](#性能优化) | 89 | 性能优化与算子开发：AscendC/Triton/Catlass/TileLang/PyPTO算子、模型推理优化、profiling |
| [部署验证](#部署验证) | 10 | 部署就绪验证：smoke test、benchmark验证、consistency check、HCCL测试、profiling异常检测 |
| [通用工具](#通用工具) | 19 | 通用工程工具：SSH、代码理解、测试生成、bug修复、coverage、重构、skill管理 |
| [文档生成](#文档生成) | 10 | 文档生成：算子文档、Docker指南、CI/CD、实验追踪、GitCode发布 |
| [PTA Pipeline](#PTA-Pipeline) | 6 | PTA Pipeline智能体：adapt-agent、verify-agent、optimizer-agent、quantify-agent |
| [模型量化](#模型量化) | 2 | 量化工具与NPU架构知识 |
| [知识检索](#知识检索) | 1 | 文档检索与知识查询 |
| [其他/归档](#其他-归档) | 5 | 归档及难以归类项 |
| **合计** | **301** | - |


## 模型适配

共 **45** 个 skill。


| Agent 名称 | 类型 | 链接 | 验证（skills-eval） |
| --- | --- | --- | --- |
| npu-adapter-reviewer | 模型适配 | ./adaptation/adapt-agent | PASS |
| adapter-check-principle | 模型适配 | ./adaptation/adapter-check-principle | PASS |
| ai-for-science-ascend-tf-community | 模型适配 | ./adaptation/ai-for-science-ascend-tf-community | PASS |
| ai-for-science-boltz2 | 模型适配 | ./adaptation/ai-for-science-boltz2 | PASS |
| ai-for-science-boltzgen | 模型适配 | ./adaptation/ai-for-science-boltzgen | PASS |
| ai-for-science-deepfri | 模型适配 | ./adaptation/ai-for-science-deepfri | PASS |
| ai-for-science-deepfri-tf-npu | 模型适配 | ./adaptation/ai-for-science-deepfri-tf-npu | PASS |
| ai-for-science-diffsbdd | 模型适配 | ./adaptation/ai-for-science-diffsbdd | PASS |
| ai-for-science-generator | 模型适配 | ./adaptation/ai-for-science-generator | PASS |
| ai-for-science-goedel-prover | 模型适配 | ./adaptation/ai-for-science-goedel-prover | PASS |
| ai-for-science-oligoformer | 模型适配 | ./adaptation/ai-for-science-oligoformer | PASS |
| ai-for-science-proteinbert | 模型适配 | ./adaptation/ai-for-science-proteinbert | PASS |
| ai-for-science-tf-to-pytorch | 模型适配 | ./adaptation/ai-for-science-tf-to-pytorch | PASS |
| ai-for-science-ai4s-basic | 模型适配 | ./adaptation/ai4s-basic | PASS |
| ai4s-main | 模型适配 | ./adaptation/ai4s-main | PASS |
| ascend-cuda-compat-check | 模型适配 | ./adaptation/ascend-cuda-compat-check | PASS |
| chronos-2-npu | 模型适配 | ./adaptation/chronos-2-npu | PASS |
| cv-ascend-adapt | 模型适配 | ./adaptation/cv-ascend-adapt | PASS |
| ai-for-science-deepfri-tf-npu | 模型适配 | ./adaptation/deepfri-tf-npu | PASS |
| ai-for-science-diffsbdd | 模型适配 | ./adaptation/diffsbdd | PASS |
| diffusers-ascend-adaptation | 模型适配 | ./adaptation/diffusers-ascend-adaptation | PASS |
| ascend-model-migration | 模型适配 | ./adaptation/drivingsdk-ascend-model-migration | PASS |
| extract-vllm-backbone | 模型适配 | ./adaptation/extract-vllm-backbone | PASS |
| hardware-check-principle | 模型适配 | ./adaptation/hardware-check-principle | PASS |
| megatron-change-analyzer | 模型适配 | ./adaptation/megatron-change-analyzer | PASS |
| megatron-impact-mapper | 模型适配 | ./adaptation/megatron-impact-mapper | PASS |
| megatron-migration-generator | 模型适配 | ./adaptation/megatron-migration-generator | PASS |
| mindspeed-fsdp2-config-migration | 模型适配 | ./adaptation/mindspeed-fsdp2-config-migration | PASS |
| mindspeed-fsdp2-data-migration | 模型适配 | ./adaptation/mindspeed-fsdp2-data-migration | PASS |
| mindspeed-fsdp2-migration-main | 模型适配 | ./adaptation/mindspeed-fsdp2-migration-main | PASS |
| mindspeed-fsdp2-model-migration | 模型适配 | ./adaptation/mindspeed-fsdp2-model-migration | PASS |
| model-migration | 模型适配 | ./adaptation/model-migration | PASS |
| model-series-vendor-detector | 模型适配 | ./adaptation/model-series-vendor-detector | PASS |
| msverl-daily-regression-triage | 模型适配 | ./adaptation/msverl-daily-regression-triage | PASS |
| npu-adapt | 模型适配 | ./adaptation/npu-adapt-skill | PASS |
| rapidocr-npu | 模型适配 | ./adaptation/rapidocr-npu | PASS |
| small_model_adapt | 模型适配 | ./adaptation/small-model-adapt | PASS |
| ai-for-science-tf-to-pytorch | 模型适配 | ./adaptation/tf-to-pytorch | PASS |
| tfjs-pytorch-npu-migration | 模型适配 | ./adaptation/tfjs-pytorch-npu-migration | PASS |
| uniasr-npu-adapt | 模型适配 | ./adaptation/uniasr-npu-adapt | PASS |
| uv-torch-adaptation | 模型适配 | ./adaptation/uv-torch-adaptation | PASS |
| vllm-ascend-backbone-extraction | 模型适配 | ./adaptation/vllm-ascend-backbone-extraction | PASS |
| vllm-ascend-dtype-selection | 模型适配 | ./adaptation/vllm-ascend-dtype-selection | PASS |
| vllm-ascend-model-adapter | 模型适配 | ./adaptation/vllm-ascend-model-adapter | PASS |
| winclip-ascend | 模型适配 | ./adaptation/winclip | PASS |

**这一组里**：PASS 45 / FAIL 0。

## 模型部署

共 **114** 个 skill。


| Agent 名称 | 类型 | 链接 | 验证（skills-eval） |
| --- | --- | --- | --- |
| acestep-v15-xl-npu-deploy | 模型部署 | ./deployment/acestep-v15-xl-npu | PASS |
| aimv2-adapt-npu | 模型部署 | ./deployment/aimv2-adapt-npu | PASS |
| aimv2-npu-deploy | 模型部署 | ./deployment/aimv2-npu | PASS |
| ascend-inference-repos-copilot | 模型部署 | ./deployment/ascend-inference-repos-copilot | PASS |
| ascend-mmlab-install-suite | 模型部署 | ./deployment/ascend-mmlab-install-suite | PASS |
| ascend-modelscope-mirror | 模型部署 | ./deployment/ascend-modelscope-mirror | PASS |
| ascend-npu-driver-install | 模型部署 | ./deployment/ascend-npu-driver-install | PASS |
| ascend-resource-scheduler | 模型部署 | ./deployment/ascend-resource-scheduler | PASS |
| ascend-vllm-env-config | 模型部署 | ./deployment/ascend-vllm-env-config | PASS |
| atc-model-converter | 模型部署 | ./deployment/atc-model-converter | PASS |
| batch21-spnasnet100-npu-deploy | 模型部署 | ./deployment/batch21-spnasnet100-npu | PASS |
| bigvgan-v2-npu | 模型部署 | ./deployment/bigvgan-v2-npu | PASS |
| blendmask-npu | 模型部署 | ./deployment/blendmask-npu | PASS |
| c-radio-v2-npu | 模型部署 | ./deployment/c-radio-v2-npu | PASS |
| c-radio-v4-npu | 模型部署 | ./deployment/c-radio-v4-npu | PASS |
| convformer-npu | 模型部署 | ./deployment/convformer-npu | PASS |
| convnext-npu | 模型部署 | ./deployment/convnext-npu | PASS |
| convnext-test-npu-deploy | 模型部署 | ./deployment/convnext-test-npu | PASS |
| convnext_base.clip_laion2b-npu | 模型部署 | ./deployment/convnext_base.clip_laion2b-npu | PASS |
| cosyvoice-npu-deploy | 模型部署 | ./deployment/cosyvoice-npu-deploy | PASS |
| cubeai-vit-classification-npu-deploy | 模型部署 | ./deployment/cubeai-vit-classification-npu-deploy | PASS |
| ascend-detectron2-install | 模型部署 | ./deployment/detectron2 | PASS |
| detr-npu-deploy | 模型部署 | ./deployment/detr-npu | PASS |
| dino-vits8-npu | 模型部署 | ./deployment/dino-vits8-npu | PASS |
| dinov2-npu-deploy | 模型部署 | ./deployment/dinov2-npu | PASS |
| dm-nfnet-npu-deploy | 模型部署 | ./deployment/dm-nfnet-npu | PASS |
| efficientnet-b14-npu | 模型部署 | ./deployment/efficientnet-b14-npu | PASS |
| efficientnet-npu-adapt | 模型部署 | ./deployment/efficientnet-npu-adapt | PASS |
| efficientnet-npu-skill | 模型部署 | ./deployment/efficientnet-npu-skill | PASS |
| esm2-npu-deploy | 模型部署 | ./deployment/esm2-npu | PASS |
| flan-t5-base-npu | 模型部署 | ./deployment/flan-t5-base-npu | PASS |
| flan-t5-ascend-npu-agent | 模型部署 | ./deployment/flan-t5-npu | PASS |
| git-npu-deploy | 模型部署 | ./deployment/git-npu | PASS |
| gitcode-repo-bootstrap | 模型部署 | ./deployment/gitcode-repo-bootstrap | PASS |
| granite-4.1-npu-deploy | 模型部署 | ./deployment/granite-4.1-npu | PASS |
| granite-speech-npu-deploy | 模型部署 | ./deployment/granite-speech-npu | PASS |
| infinity-instruct-baai-npu | 模型部署 | ./deployment/infinity-instruct-baai-npu | PASS |
| kani-tts-400m-npu | 模型部署 | ./deployment/kani-tts-400m-npu | PASS |
| kotoba-whisper-npu-deploy | 模型部署 | ./deployment/kotoba-whisper-npu | PASS |
| levit-npu-deploy | 模型部署 | ./deployment/levit-npu-deploy | PASS |
| ascend-mmcv-install | 模型部署 | ./deployment/mmcv | PASS |
| ascend-mmdet-install | 模型部署 | ./deployment/mmdet | PASS |
| ascend-mmdet3d-install | 模型部署 | ./deployment/mmdet3d | PASS |
| mms-300m-1130-forced-aligner-npu | 模型部署 | ./deployment/mms-300m-1130-forced-aligner-npu | PASS |
| mms-tts-npu-deploy | 模型部署 | ./deployment/mms-tts-npu | PASS |
| mobilevit-npu-deploy | 模型部署 | ./deployment/mobilevit-npu | PASS |
| model-training | 模型部署 | ./deployment/model-training | PASS |
| modelscope-npu-deploy | 模型部署 | ./deployment/modelscope-npu-deploy | PASS |
| moss-tts-npu-deployment | 模型部署 | ./deployment/moss-tts-npu-deployment | PASS |
| neutts-nano-npu-deploy | 模型部署 | ./deployment/neutts-nano-npu | PASS |
| npu-engine-core-init-skill | 模型部署 | ./deployment/npu-engine-core-init-skill | PASS |
| npu-smi | 模型部署 | ./deployment/npu-smi | PASS |
| openjourney-npu-deploy | 模型部署 | ./deployment/openjourney-npu | PASS |
| outetts-npu-deploy | 模型部署 | ./deployment/outetts-npu | PASS |
| paraformer-npu-deploy | 模型部署 | ./deployment/paraformer-npu | PASS |
| patchcore-npu | 模型部署 | ./deployment/patchcore-npu | PASS |
| patchcore-npu | 模型部署 | ./deployment/patchcore-npu-root | PASS |
| phi-npu | 模型部署 | ./deployment/phi-npu | PASS |
| phikon-v2-npu-deploy | 模型部署 | ./deployment/phikon-v2-npu | PASS |
| plant-classify-npu-deploy | 模型部署 | ./deployment/plant-classify-npu | PASS |
| pp-ocrv3-npu | 模型部署 | ./deployment/pp-ocrv3-npu | PASS |
| pp-ocrv4-npu-deploy | 模型部署 | ./deployment/pp-ocrv4-npu-deploy | PASS |
| ai-for-science-proteinbert | 模型部署 | ./deployment/proteinbert | PASS |
| qwen-asr-npu-deploy | 模型部署 | ./deployment/qwen-asr-npu-deploy | PASS |
| qwen3-asr-npu-deploy | 模型部署 | ./deployment/qwen3-asr-npu | PASS |
| rapidocr-npu-model | 模型部署 | ./deployment/rapidocr-npu-model | PASS |
| reader-lm-npu-deploy | 模型部署 | ./deployment/reader-lm-npu | PASS |
| sambert-hifigan-tts-npu | 模型部署 | ./deployment/sambert-hifigan-tts-npu | PASS |
| selecsls-npu | 模型部署 | ./deployment/selecsls-npu | PASS |
| semnasnet-npu | 模型部署 | ./deployment/semnasnet-npu-skill | PASS |
| senet-npu | 模型部署 | ./deployment/senet-npu | PASS |
| sequencer2d-npu | 模型部署 | ./deployment/sequencer2d-npu | PASS |
| siglip-npu-deploy | 模型部署 | ./deployment/siglip-npu | PASS |
| sk-resnet-npu | 模型部署 | ./deployment/sk-resnet-npu | PASS |
| soprano-80m-npu-deploy | 模型部署 | ./deployment/soprano-80m-npu | PASS |
| speaker-diarization-npu-deploy | 模型部署 | ./deployment/speaker-diarization-npu | PASS |
| swin-batch-npu | 模型部署 | ./deployment/swin-batch-npu | PASS |
| swin-transformer-npu | 模型部署 | ./deployment/swin-transformer-npu | PASS |
| swinv2-npu-skill | 模型部署 | ./deployment/swinv2-npu-skill | PASS |
| timm-mixnet-npu | 模型部署 | ./deployment/timm-mixnet-npu | PASS |
| timm-npu-adapt | 模型部署 | ./deployment/timm-npu-adapt | PASS |
| timm-se-npu-batch24 | 模型部署 | ./deployment/timm-se-npu-batch24 | PASS |
| tiny-vit-npu | 模型部署 | ./deployment/tiny-vit-npu | PASS |
| tinynet-npu | 模型部署 | ./deployment/tinynet-npu-skill | PASS |
| trocr-npu-deploy | 模型部署 | ./deployment/trocr-npu | PASS |
| twins-npu | 模型部署 | ./deployment/twins-npu | PASS |
| vgg-npu-deploy | 模型部署 | ./deployment/vgg-npu | PASS |
| vit-base-patch16-224-in21k-npu | 模型部署 | ./deployment/vit-base-patch16-224-in21k-npu | PASS |
| vit_base_patch16_224-dino-npu-deploy | 模型部署 | ./deployment/vit_base_patch16_224-dino-npu | PASS |
| vit_small_patch16_dinov3.lvd1689m-npu | 模型部署 | ./deployment/vit_small_patch16_dinov3.lvd1689m-npu | PASS |
| vllm-ascend-native-deploy | 模型部署 | ./deployment/vllm-ascend-native-deploy | PASS |
| vllm-ascend-troubleshooter | 模型部署 | ./deployment/vllm-ascend-troubleshooter | PASS |
| voice-activity-detection-npu | 模型部署 | ./deployment/voice-activity-detection-npu | PASS |
| volo-npu-deploy | 模型部署 | ./deployment/volo-npu | PASS |
| volo-npu-deploy | 模型部署 | ./deployment/volo-npu-skill | PASS |
| vtp-npu-adapation | 模型部署 | ./deployment/vtp-npu-adapation | PASS |
| vtp-small-f16d64-npu-deploy | 模型部署 | ./deployment/vtp-small-f16d64-npu | PASS |
| wav2vec2-indonesian-javanese-sundanese-npu | 模型部署 | ./deployment/wav2vec2-indonesian-javanese-sundanese-npu | PASS |
| wav2vec2-urdu-npu-deploy | 模型部署 | ./deployment/wav2vec2-large-xls-r-300m-urdu-npu | PASS |
| wav2vec2-large-xlsr-53-greek-npu | 模型部署 | ./deployment/wav2vec2-large-xlsr-53-greek | PASS |
| wav2vec2-xlsr-53-npu-deploy | 模型部署 | ./deployment/wav2vec2-xlsr-53-npu-deploy | PASS |
| wavyfusion-npu-deploy | 模型部署 | ./deployment/wavyfusion-npu | PASS |
| webssl-mae-npu-deploy | 模型部署 | ./deployment/webssl-mae-npu | PASS |
| whisper-large-v3-npu | 模型部署 | ./deployment/whisper-large-v3-npu | PASS |
| whisper-medium-npu-deploy | 模型部署 | ./deployment/whisper-medium-npu | PASS |
| wide-resnet-npu-deploy | 模型部署 | ./deployment/wide-resnet-npu | PASS |
| winclip-npu | 模型部署 | ./deployment/winclip-npu | PASS |
| xception-npu | 模型部署 | ./deployment/xception-npu-deployment | PASS |
| xcit-models-npu-deployment | 模型部署 | ./deployment/xcit-models-npu-deployment | PASS |
| xcit-npu | 模型部署 | ./deployment/xcit-npu | PASS |
| xcit-npu-deploy | 模型部署 | ./deployment/xcit-npu-deploy | PASS |
| yolov10-npu | 模型部署 | ./deployment/yolov10-npu | PASS |
| yolov10-npu-contrib | 模型部署 | ./deployment/yolov10-npu-contrib | PASS |
| yolov10-m-ascend-optimization | 模型部署 | ./deployment/yolov10-npu-model | PASS |

**这一组里**：PASS 114 / FAIL 0。

## 性能优化

共 **89** 个 skill。


| Agent 名称 | 类型 | 链接 | 验证（skills-eval） |
| --- | --- | --- | --- |
| ai-for-science-ai4s-perf-tuning | 性能优化 | ./optimization/ai4s-perf-tuning | PASS |
| ascend-ai4s-precision-alignment | 性能优化 | ./optimization/ai4s-precision-alignment | PASS |
| ai-for-science-ai4s-profiling | 性能优化 | ./optimization/ai4s-profiling | PASS |
| ascend-affinity-operator | 性能优化 | ./optimization/ascend-affinity-operator | PASS |
| ascend-history-to-skill | 性能优化 | ./optimization/ascend-history-to-skill | PASS |
| ascend-optimization | 性能优化 | ./optimization/ascend-optimization | PASS |
| agent-profiling | 性能优化 | ./optimization/ascend-profiling | PASS |
| ascendc-api-best-practices | 性能优化 | ./optimization/ascendc-api-best-practices | PASS |
| ascendc-code-review | 性能优化 | ./optimization/ascendc-code-review | PASS |
| ascendc-direct-invoke-template | 性能优化 | ./optimization/ascendc-direct-invoke-template | PASS |
| ascendc-env-check | 性能优化 | ./optimization/ascendc-env-check | PASS |
| ascendc-operator-code-gen | 性能优化 | ./optimization/ascendc-operator-code-gen | PASS |
| ascendc-operator-code-review | 性能优化 | ./optimization/ascendc-operator-code-review | PASS |
| ascendc-operator-compile-debug | 性能优化 | ./optimization/ascendc-operator-compile-debug | PASS |
| ascendc-operator-design | 性能优化 | ./optimization/ascendc-operator-design | PASS |
| ascendc-operator-dev | 性能优化 | ./optimization/ascendc-operator-dev | PASS |
| ascendc-operator-doc-gen | 性能优化 | ./optimization/ascendc-operator-doc-gen | PASS |
| ascendc-mssanitizer | 性能优化 | ./optimization/ascendc-operator-mssanitizer | PASS |
| ascendc-operator-performance-eval | 性能优化 | ./optimization/ascendc-operator-performance-eval | PASS |
| ascendc-operator-performance-optim | 性能优化 | ./optimization/ascendc-operator-performance-optim | PASS |
| ascendc-operator-precision-debug | 性能优化 | ./optimization/ascendc-operator-precision-debug | PASS |
| ascendc-operator-precision-eval | 性能优化 | ./optimization/ascendc-operator-precision-eval | PASS |
| ascendc-operator-project-init | 性能优化 | ./optimization/ascendc-operator-project-init | PASS |
| ascendc-operator-testcase-gen | 性能优化 | ./optimization/ascendc-operator-testcase-gen | PASS |
| ascendc-precision-debug | 性能优化 | ./optimization/ascendc-precision-debug | PASS |
| ascendc-registry-invoke-to-direct-invoke | 性能优化 | ./optimization/ascendc-registry-invoke-to-direct-invoke | PASS |
| ascendc-runtime-debug | 性能优化 | ./optimization/ascendc-runtime-debug | PASS |
| ascendc-st-design | 性能优化 | ./optimization/ascendc-st-design | PASS |
| ascendc-task-focus | 性能优化 | ./optimization/ascendc-task-focus | PASS |
| ascendc-tiling-design | 性能优化 | ./optimization/ascendc-tiling-design | PASS |
| ascendc-whitebox-design | 性能优化 | ./optimization/ascendc-whitebox-design | PASS |
| cann-operator-env-config | 性能优化 | ./optimization/cann-operator-env-config | PASS |
| catlass-operator-code-gen | 性能优化 | ./optimization/catlass-operator-code-gen | PASS |
| catlass-operator-design | 性能优化 | ./optimization/catlass-operator-design | PASS |
| catlass-operator-dev | 性能优化 | ./optimization/catlass-operator-dev | PASS |
| catlass-operator-performance-optim | 性能优化 | ./optimization/catlass-operator-performance-optim | PASS |
| data-pipeline-optimizer | 性能优化 | ./optimization/data-pipeline-optimizer | PASS |
| distributed-optimizer | 性能优化 | ./optimization/distributed-optimizer | PASS |
| memory-optimizer | 性能优化 | ./optimization/memory-optimizer | PASS |
| mixed-precision-optimizer | 性能优化 | ./optimization/mixed-precision-optimizer | PASS |
| model-infer-fusion | 性能优化 | ./optimization/model-infer-fusion | PASS |
| model-infer-graph-mode | 性能优化 | ./optimization/model-infer-graph-mode | PASS |
| model-infer-kvcache | 性能优化 | ./optimization/model-infer-kvcache | PASS |
| model-infer-migrator | 性能优化 | ./optimization/model-infer-migrator | PASS |
| model-infer-multi-stream | 性能优化 | ./optimization/model-infer-multi-stream | PASS |
| model-infer-optimize | 性能优化 | ./optimization/model-infer-optimize | PASS |
| model-infer-parallel-analysis | 性能优化 | ./optimization/model-infer-parallel-analysis | PASS |
| model-infer-parallel-impl | 性能优化 | ./optimization/model-infer-parallel-impl | PASS |
| model-infer-precision-debug | 性能优化 | ./optimization/model-infer-precision-debug | PASS |
| model-infer-prefetch | 性能优化 | ./optimization/model-infer-prefetch | PASS |
| model-infer-runtime-debug | 性能优化 | ./optimization/model-infer-runtime-debug | PASS |
| model-infer-superkernel | 性能优化 | ./optimization/model-infer-superkernel | PASS |
| msprof-optimizer | 性能优化 | ./optimization/msprof-optimizer | PASS |
| npu-adapter-reviewer | 性能优化 | ./optimization/npu-adapter-reviewer | PASS |
| ops-precision-standard | 性能优化 | ./optimization/ops-precision-standard | PASS |
| ops-profiling | 性能优化 | ./optimization/ops-profiling | PASS |
| ops-simulator | 性能优化 | ./optimization/ops-simulator | PASS |
| perf-analyzer | 性能优化 | ./optimization/perf-analyzer | PASS |
| pypto-api-explore | 性能优化 | ./optimization/pypto-api-explore | PASS |
| pypto-golden-generate | 性能优化 | ./optimization/pypto-golden-generate | PASS |
| pypto-intent-understand | 性能优化 | ./optimization/pypto-intent-understand | PASS |
| pypto-op-design | 性能优化 | ./optimization/pypto-op-design | PASS |
| pypto-op-develop | 性能优化 | ./optimization/pypto-op-develop | PASS |
| pypto-op-perf-tune | 性能优化 | ./optimization/pypto-op-perf-tune | PASS |
| pypto-precision-compare | 性能优化 | ./optimization/pypto-precision-compare | PASS |
| pypto-precision-debug | 性能优化 | ./optimization/pypto-precision-debug | PASS |
| simple-vector-triton-gpu-to-npu | 性能优化 | ./optimization/simple-vector-triton-gpu-to-npu | PASS |
| tilelang-api-best-practices | 性能优化 | ./optimization/tilelang-api-best-practices | PASS |
| tilelang-op-design | 性能优化 | ./optimization/tilelang-op-design | PASS |
| tilelang-op-developer | 性能优化 | ./optimization/tilelang-op-developer | PASS |
| tilelang-programming-model-guide | 性能优化 | ./optimization/tilelang-programming-model-guide | PASS |
| tilelang-review | 性能优化 | ./optimization/tilelang-review | PASS |
| tilelang-vector-ascend-ops-migration | 性能优化 | ./optimization/tilelang-vector-ascend-ops-migration | PASS |
| triton-operator-code-gen | 性能优化 | ./optimization/triton-operator-code-gen | PASS |
| triton-operator-code-review | 性能优化 | ./optimization/triton-operator-code-review | PASS |
| triton-operator-design | 性能优化 | ./optimization/triton-operator-design | PASS |
| triton-operator-dev | 性能优化 | ./optimization/triton-operator-dev | PASS |
| triton-operator-env-config | 性能优化 | ./optimization/triton-operator-env-config | PASS |
| triton-operator-performance-eval | 性能优化 | ./optimization/triton-operator-performance-eval | PASS |
| triton-operator-performance-optim | 性能优化 | ./optimization/triton-operator-performance-optim | PASS |
| triton-operator-precision-eval | 性能优化 | ./optimization/triton-operator-precision-eval | PASS |
| tune-frontend | 性能优化 | ./optimization/tune-frontend | PASS |
| tune-incore | 性能优化 | ./optimization/tune-incore | PASS |
| tune-swimlane | 性能优化 | ./optimization/tune-swimlane | PASS |
| vector-triton-ascend-ops-optimizer | 性能优化 | ./optimization/vector-triton-ascend-ops-optimizer | PASS |
| verl-async-dapo | 性能优化 | ./optimization/verl-async-dapo | PASS |
| vllm-ascend-acl-graph-warmup | 性能优化 | ./optimization/vllm-ascend-acl-graph-warmup | PASS |
| vllm-ascend-performance-optimization | 性能优化 | ./optimization/vllm-ascend-performance-optimization | PASS |
| vLLM-ascend_FAQ_Generator | 性能优化 | ./optimization/vllm-ascend_faq_generator | PASS |

**这一组里**：PASS 89 / FAIL 0。

## 部署验证

共 **10** 个 skill。


| Agent 名称 | 类型 | 链接 | 验证（skills-eval） |
| --- | --- | --- | --- |
| ascend-benchmark-runner | 部署验证 | ./verification/ascend-benchmark-runner | PASS |
| ascend-model-verification | 部署验证 | ./verification/ascend-model-verification | PASS |
| ascend-profiling-anomaly | 部署验证 | ./verification/ascend-profiling-anomaly | PASS |
| ascend-smoke-validator | 部署验证 | ./verification/ascend-smoke-validator | PASS |
| ascend_model_verifier | 部署验证 | ./verification/ascend_model_verifier | PASS |
| hccl-test | 部署验证 | ./verification/hccl-test | PASS |
| mindspeed-fsdp2-verification | 部署验证 | ./verification/mindspeed-fsdp2-verification | PASS |
| npu-benchmark-guard | 部署验证 | ./verification/npu-benchmark-guard | PASS |
| vllm-ascend-consistency-check | 部署验证 | ./verification/vllm-ascend-consistency-check | PASS |
| vllm-ascend-perf-benchmark | 部署验证 | ./verification/vllm-ascend-perf-benchmark | PASS |

**这一组里**：PASS 10 / FAIL 0。

## 通用工具

共 **19** 个 skill。


| Agent 名称 | 类型 | 链接 | 验证（skills-eval） |
| --- | --- | --- | --- |
| analyse-coverage | 通用工具 | ./common/analyse-coverage | PASS |
| auto-bug-fixer | 通用工具 | ./common/auto-bug-fixer | PASS |
| auto-develop-test-gen | 通用工具 | ./common/auto-develop-test-gen | PASS |
| code-comprehension | 通用工具 | ./common/code-comprehension | PASS |
| ssh-dev-suite/connect | 通用工具 | ./common/connect | PASS |
| coverage | 通用工具 | ./common/coverage | PASS |
| ssh-dev-suite/debug | 通用工具 | ./common/debug | PASS |
| ssh-dev-suite/deploy | 通用工具 | ./common/deploy | PASS |
| generate-unit-test | 通用工具 | ./common/generate-unit-test | PASS |
| issue_autoreply | 通用工具 | ./common/issue_autoreply | PASS |
| ssh-dev-suite/long-task | 通用工具 | ./common/long-task | PASS |
| pytest-writer | 通用工具 | ./common/pytest-writer | PASS |
| python-refactoring | 通用工具 | ./common/python-refactoring | PASS |
| repo-reader | 通用工具 | ./common/repo-reader | PASS |
| ops-easyasc-dsl | 通用工具 | ./common/skill | PASS |
| skill-auditor | 通用工具 | ./common/skill-auditor | PASS |
| ssh-dev-suite | 通用工具 | ./common/ssh-connection | PASS |
| ssh-dev-suite/tunnel | 通用工具 | ./common/tunnel | PASS |
| unittest-writer | 通用工具 | ./common/unittest-writer | PASS |

**这一组里**：PASS 19 / FAIL 0。

## 文档生成

共 **10** 个 skill。


| Agent 名称 | 类型 | 链接 | 验证（skills-eval） |
| --- | --- | --- | --- |
| ascend-delivery-pipeline | 文档生成 | ./documentation/ascend-delivery-pipeline | PASS |
| ascend-docker | 文档生成 | ./documentation/ascend-docker | PASS |
| ascendc-docs-gen | 文档生成 | ./documentation/ascendc-docs-gen | PASS |
| ascendc-regbase-best-practice | 文档生成 | ./documentation/ascendc-regbase-best-practice | PASS |
| ascendc-ut-develop | 文档生成 | ./documentation/ascendc-ut-develop | PASS |
| git-commit-author-fix | 文档生成 | ./documentation/git-commit-author-fix | PASS |
| gitcode-publish | 文档生成 | ./documentation/gitcode-publish | PASS |
| model-agent-pr-submit | 文档生成 | ./documentation/pr-submit | PASS |
| swanlab-setup | 文档生成 | ./documentation/swanlab-setup | PASS |
| triton-operator-doc-gen | 文档生成 | ./documentation/triton-operator-doc-gen | PASS |

**这一组里**：PASS 10 / FAIL 0。

## PTA Pipeline

共 **6** 个 skill。


| Agent 名称 | 类型 | 链接 | 验证（skills-eval） |
| --- | --- | --- | --- |
| adapt-agent | PTA Pipeline | ./pta/adapt-agent | PASS |
| optimizer-agent | PTA Pipeline | ./pta/optimizer-agent | PASS |
| optimizer-agent-plus | PTA Pipeline | ./pta/optimizer-agent-plus | PASS |
| quantify-agent | PTA Pipeline | ./pta/quantify-agent | PASS |
| verify-agent | PTA Pipeline | ./pta/verify-agent | PASS |
| verify-agent-plus | PTA Pipeline | ./pta/verify-agent-plus | PASS |

**这一组里**：PASS 6 / FAIL 0。

## 模型量化

共 **2** 个 skill。


| Agent 名称 | 类型 | 链接 | 验证（skills-eval） |
| --- | --- | --- | --- |
| ascendc-npu-arch | 模型量化 | ./quantization/ascendc-npu-arch | PASS |
| awesome-llm-model | 模型量化 | ./quantization/awesome-llm-model | PASS |

**这一组里**：PASS 2 / FAIL 0。

## 知识检索

共 **1** 个 skill。


| Agent 名称 | 类型 | 链接 | 验证（skills-eval） |
| --- | --- | --- | --- |
| ascendc-docs-search | 知识检索 | ./search/ascendc-docs-search | PASS |

**这一组里**：PASS 1 / FAIL 0。

## 其他/归档

共 **5** 个 skill。


| Agent 名称 | 类型 | 链接 | 验证（skills-eval） |
| --- | --- | --- | --- |
| archived | 其他/归档 | ./other/archived | PASS |
| generative-recommendation-verification | 其他/归档 | ./other/generative-recommendation-verification | PASS |
| vLLM-ascend_FAQ_Generator | 其他/归档 | ./other/issue_solver | PASS |
| megatron-commit-tracker | 其他/归档 | ./other/megatron-commit-tracker | PASS |
| run-mindspeed-llm-test | 其他/归档 | ./other/run-mindspeed-llm-test | PASS |

**这一组里**：PASS 5 / FAIL 0。
