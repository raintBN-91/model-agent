# Model Agent Skills

昇腾Model **Agent Skills** **导航表**，所有skills 均经过skills-Eval测评系统验证

### 快速体验 [skills-eval](https://relenting-speed-octopus.ngrok-free.dev/)

在「仓库 URL」里贴上 **Git 仓库地址**，「校验规则」选 **skills-eval 默认** 或 **Anthropic skill-creator 加强**，点 **开始评测** 即可.

## Skills汇总：场景分类


| 场景                  | Skills 数量 | 占比       |
| ------------------- | --------- | -------- |
| [性能优化](#性能优化)       | 69        | 42.6%    |
| [质量验证](#质量验证)       | 51        | 31.5%    |
| [模型部署](#模型部署)       | 15        | 9.3%     |
| [模型适配](#模型适配)       | 14        | 8.6%     |
| [文档生成](#文档生成)       | 10        | 6.2%     |
| [模型量化](#模型量化)       | 2         | 1.2%     |
| [知识检索](#知识检索)       | 1         | 0.6%     |
| **合计**              | **162**   | **100%** |


## 性能优化

共 **69** 个 skill。


| Agent 名称                                 | 类型           | 链接                                         | 验证（skills-eval）       |
| ---------------------------------------- | ------------ | ------------------------------------------ | --------------------- |
| adapt-agent                              | Optimization | ./adapt-agent                              | PASS                  |
| ai4s-basic                               | Optimization | ./ai4s-basic                               | FAIL（body_line_limit） |
| ai4s-main                                | Optimization | ./ai4s-main                                | PASS                  |
| ai4s-perf-tuning                         | Optimization | ./ai4s-perf-tuning                         | FAIL（body_line_limit） |
| ai4s-precision-alignment                 | Optimization | ./ai4s-precision-alignment                 | PASS                  |
| ai4s-profiling                           | Optimization | ./ai4s-profiling                           | PASS                  |
| analyse-coverage                         | Optimization | ./analyse-coverage                         | PASS                  |
| ascend-affinity-operator                 | Optimization | ./ascend-affinity-operator                 | PASS                  |
| ascend-history-to-skill                  | Optimization | ./ascend-history-to-skill                  | PASS                  |
| ascend-optimization                      | Optimization | ./ascend-optimization                      | PASS                  |
| ascend-profiling                         | Optimization | ./ascend-profiling                         | PASS                  |
| ascendc-api-best-practices               | Optimization | ./ascendc-api-best-practices               | PASS                  |
| ascendc-code-review                      | Optimization | ./ascendc-code-review                      | PASS                  |
| ascendc-direct-invoke-template           | Optimization | ./ascendc-direct-invoke-template           | PASS                  |
| ascendc-operator-code-gen                | Optimization | ./ascendc-operator-code-gen                | FAIL（document_parse）  |
| ascendc-operator-design                  | Optimization | ./ascendc-operator-design                  | PASS                  |
| ascendc-operator-performance-eval        | Optimization | ./ascendc-operator-performance-eval        | PASS                  |
| ascendc-operator-performance-optim       | Optimization | ./ascendc-operator-performance-optim       | PASS                  |
| ascendc-registry-invoke-to-direct-invoke | Optimization | ./ascendc-registry-invoke-to-direct-invoke | FAIL（body_line_limit） |
| ascendc-runtime-debug                    | Optimization | ./ascendc-runtime-debug                    | PASS                  |
| ascendc-task-focus                       | Optimization | ./ascendc-task-focus                       | PASS                  |
| ascendc-tiling-design                    | Optimization | ./ascendc-tiling-design                    | PASS                  |
| catlass-operator-dev                     | Optimization | ./catlass-operator-dev                     | PASS                  |
| catlass-operator-performance-optim       | Optimization | ./catlass-operator-performance-optim       | PASS                  |
| code-comprehension                       | Optimization | ./code-comprehension                       | PASS                  |
| debug                                    | Optimization | ./debug                                    | PASS                  |
| deepfri-tf-npu                           | Optimization | ./deepfri-tf-npu                           | PASS                  |
| diffsbdd                                 | Optimization | ./diffsbdd                                 | PASS                  |
| generative-recommendation-verification   | Optimization | ./generative-recommendation-verification   | FAIL（body_line_limit） |
| issue_solver                             | Optimization | ./issue_solver                             | FAIL（missing_name）    |
| long-task                                | Optimization | ./long-task                                | PASS                  |
| model-infer-fusion                       | Optimization | ./model-infer-fusion                       | PASS                  |
| model-infer-graph-mode                   | Optimization | ./model-infer-graph-mode                   | PASS                  |
| model-infer-kvcache                      | Optimization | ./model-infer-kvcache                      | PASS                  |
| model-infer-migrator                     | Optimization | ./model-infer-migrator                     | PASS                  |
| model-infer-multi-stream                 | Optimization | ./model-infer-multi-stream                 | PASS                  |
| model-infer-optimize                     | Optimization | ./model-infer-optimize                     | PASS                  |
| model-infer-parallel-analysis            | Optimization | ./model-infer-parallel-analysis            | FAIL（body_line_limit） |
| model-infer-precision-debug              | Optimization | ./model-infer-precision-debug              | PASS                  |
| model-infer-prefetch                     | Optimization | ./model-infer-prefetch                     | FAIL（body_line_limit） |
| model-infer-runtime-debug                | Optimization | ./model-infer-runtime-debug                | PASS                  |
| model-infer-superkernel                  | Optimization | ./model-infer-superkernel                  | PASS                  |
| npu-adapter-reviewer                     | Optimization | ./npu-adapter-reviewer                     | PASS                  |
| ops-profiling                            | Optimization | ./ops-profiling                            | PASS                  |
| perf-analyzer                            | Optimization | ./perf-analyzer                            | PASS                  |
| pypto-intent-understand                  | Optimization | ./pypto-intent-understand                  | FAIL（body_line_limit） |
| pypto-op-perf-tune                       | Optimization | ./pypto-op-perf-tune                       | FAIL（body_line_limit） |
| pypto-precision-debug                    | Optimization | ./pypto-precision-debug                    | PASS                  |
| pytest-writer                            | Optimization | ./pytest-writer                            | PASS                  |
| python-refactoring                       | Optimization | ./python-refactoring                       | PASS                  |
| simple-vector-triton-gpu-to-npu          | Optimization | ./simple-vector-triton-gpu-to-npu          | FAIL（body_line_limit） |
| tilelang-api-best-practices              | Optimization | ./tilelang-api-best-practices              | PASS                  |
| tilelang-op-developer                    | Optimization | ./tilelang-op-developer                    | PASS                  |
| tilelang-programming-model-guide         | Optimization | ./tilelang-programming-model-guide         | PASS                  |
| tilelang-vector-ascend-ops-migration     | Optimization | ./tilelang-vector-ascend-ops-migration     | FAIL（document_parse）  |
| triton-operator-code-review              | Optimization | ./triton-operator-code-review              | PASS                  |
| triton-operator-design                   | Optimization | ./triton-operator-design                   | PASS                  |
| triton-operator-dev                      | Optimization | ./triton-operator-dev                      | PASS                  |
| triton-operator-performance-eval         | Optimization | ./triton-operator-performance-eval         | PASS                  |
| triton-operator-performance-optim        | Optimization | ./triton-operator-performance-optim        | PASS                  |
| triton-operator-precision-eval           | Optimization | ./triton-operator-precision-eval           | PASS                  |
| tune-frontend                            | Optimization | ./tune-frontend                            | PASS                  |
| tune-incore                              | Optimization | ./tune-incore                              | PASS                  |
| tune-swimlane                            | Optimization | ./tune-swimlane                            | PASS                  |
| tunnel                                   | Optimization | ./tunnel                                   | PASS                  |
| unittest-writer                          | Optimization | ./unittest-writer                          | PASS                  |
| vector-triton-ascend-ops-optimizer       | Optimization | ./vector-triton-ascend-ops-optimizer       | PASS                  |
| verl-async-dapo                          | Optimization | ./verl-async-dapo                          | PASS                  |
| vLLM-ascend_FAQ_Generator                | Optimization | ./vllm-ascend_faq_generator                | FAIL（missing_name）    |


**这一组里**：PASS 56 / FAIL 13 / 报告里没出现的 0。

## 质量验证

共 **51** 个 skill。


| Agent 名称                         | 类型           | 链接                                 | 验证（skills-eval）       |
| -------------------------------- | ------------ | ---------------------------------- | --------------------- |
| ascend-mmlab-install-suite       | Verification | ./ascend-mmlab-install-suite       | PASS                  |
| ascend-profiling-anomaly         | Verification | ./ascend-profiling-anomaly         | PASS                  |
| ascend-tf-community              | Verification | ./ascend-tf-community              | PASS                  |
| ascendc-env-check                | Verification | ./ascendc-env-check                | PASS                  |
| ascendc-operator-code-review     | Verification | ./ascendc-operator-code-review     | PASS                  |
| ascendc-operator-compile-debug   | Verification | ./ascendc-operator-compile-debug   | FAIL（document_parse）  |
| ascendc-operator-dev             | Verification | ./ascendc-operator-dev             | PASS                  |
| ascendc-operator-doc-gen         | Verification | ./ascendc-operator-doc-gen         | PASS                  |
| ascendc-operator-mssanitizer     | Verification | ./ascendc-operator-mssanitizer     | FAIL（body_line_limit） |
| ascendc-operator-precision-debug | Verification | ./ascendc-operator-precision-debug | PASS                  |
| ascendc-operator-precision-eval  | Verification | ./ascendc-operator-precision-eval  | PASS                  |
| ascendc-operator-project-init    | Verification | ./ascendc-operator-project-init    | PASS                  |
| ascendc-operator-testcase-gen    | Verification | ./ascendc-operator-testcase-gen    | PASS                  |
| ascendc-precision-debug          | Verification | ./ascendc-precision-debug          | PASS                  |
| ascendc-st-design                | Verification | ./ascendc-st-design                | PASS                  |
| ascendc-whitebox-design          | Verification | ./ascendc-whitebox-design          | PASS                  |
| auto-bug-fixer                   | Verification | ./auto-bug-fixer                   | PASS                  |
| auto-develop-test-gen            | Verification | ./auto-develop-test-gen            | PASS                  |
| boltzgen                         | Verification | ./boltzgen                         | FAIL（body_line_limit） |
| cann-operator-env-config         | Verification | ./cann-operator-env-config         | PASS                  |
| catlass-operator-code-gen        | Verification | ./catlass-operator-code-gen        | PASS                  |
| catlass-operator-design          | Verification | ./catlass-operator-design          | PASS                  |
| deepfri                          | Verification | ./deepfri                          | PASS                  |
| detectron2                       | Verification | ./detectron2                       | PASS                  |
| generate-unit-test               | Verification | ./generate-unit-test               | FAIL（document_parse）  |
| goedel-prover                    | Verification | ./goedel-prover                    | PASS                  |
| hccl-test                        | Verification | ./hccl-test                        | PASS                  |
| issue_autoreply                  | Verification | ./issue_autoreply                  | FAIL（document_parse）  |
| megatron-commit-tracker          | Verification | ./megatron-commit-tracker          | PASS                  |
| mindspeed-fsdp2-config-migration | Verification | ./mindspeed-fsdp2-config-migration | PASS                  |
| mindspeed-fsdp2-data-migration   | Verification | ./mindspeed-fsdp2-data-migration   | PASS                  |
| mindspeed-fsdp2-model-migration  | Verification | ./mindspeed-fsdp2-model-migration  | PASS                  |
| mindspeed-fsdp2-verification     | Verification | ./mindspeed-fsdp2-verification     | PASS                  |
| mmcv                             | Verification | ./mmcv                             | PASS                  |
| mmdet                            | Verification | ./mmdet                            | PASS                  |
| mmdet3d                          | Verification | ./mmdet3d                          | PASS                  |
| model-infer-parallel-impl        | Verification | ./model-infer-parallel-impl        | PASS                  |
| model-training                   | Verification | ./model-training                   | PASS                  |
| oligoformer                      | Verification | ./oligoformer                      | PASS                  |
| ops-precision-standard           | Verification | ./ops-precision-standard           | PASS                  |
| ops-simulator                    | Verification | ./ops-simulator                    | PASS                  |
| pypto-api-explore                | Verification | ./pypto-api-explore                | PASS                  |
| pypto-golden-generate            | Verification | ./pypto-golden-generate            | PASS                  |
| pypto-op-design                  | Verification | ./pypto-op-design                  | PASS                  |
| pypto-op-develop                 | Verification | ./pypto-op-develop                 | PASS                  |
| pypto-precision-compare          | Verification | ./pypto-precision-compare          | PASS                  |
| quantify-agent                   | Verification | ./quantify-agent                   | FAIL（document_parse）  |
| tilelang-op-design               | Verification | ./tilelang-op-design               | PASS                  |
| tilelang-review                  | Verification | ./tilelang-review                  | FAIL（body_line_limit） |
| triton-operator-code-gen         | Verification | ./triton-operator-code-gen         | PASS                  |
| triton-operator-env-config       | Verification | ./triton-operator-env-config       | PASS                  |


**这一组里**：PASS 44 / FAIL 7 / 报告里没出现的 0。

## 模型部署

共 **15** 个 skill。


| Agent 名称                             | 类型         | 链接                                     | 验证（skills-eval）       |
| ------------------------------------ | ---------- | -------------------------------------- | --------------------- |
| Archived                             | Deployment | ./archived                             | 未报告                   |
| ascend-inference-repos-copilot       | Deployment | ./ascend-inference-repos-copilot       | PASS                  |
| ascend-npu-driver-install            | Deployment | ./ascend-npu-driver-install            | PASS                  |
| Ascend_Model_Verifier                | Deployment | ./ascend_model_verifier                | FAIL（document_parse）  |
| atc-model-converter                  | Deployment | ./atc-model-converter                  | FAIL（body_line_limit） |
| deploy                               | Deployment | ./deploy                               | PASS                  |
| drivingsdk-ascend-model-migration    | Deployment | ./drivingsdk-ascend-model-migration    | PASS                  |
| esm2-npu                             | Deployment | ./esm2-npu                             | PASS                  |
| npu-smi                              | Deployment | ./npu-smi                              | PASS                  |
| proteinbert                          | Deployment | ./proteinbert                          | PASS                  |
| repo-reader                          | Deployment | ./repo-reader                          | PASS                  |
| skill-auditor                        | Deployment | ./skill-auditor                        | PASS                  |
| ssh-connection                       | Deployment | ./ssh-connection                       | PASS                  |
| tf-to-pytorch                        | Deployment | ./tf-to-pytorch                        | PASS                  |
| vllm-ascend-performance-optimization | Deployment | ./vllm-ascend-performance-optimization | PASS                  |


**这一组里**：PASS 12 / FAIL 2 / 报告里没出现的 1。

## 模型适配

共 **14** 个 skill。


| Agent 名称                       | 类型         | 链接                               | 验证（skills-eval）      |
| ------------------------------ | ---------- | -------------------------------- | -------------------- |
| adapter-check-principle        | Adaptation | ./adapter-check-principle        | PASS                 |
| ascend-model-verification      | Adaptation | ./ascend-model-verification      | PASS                 |
| boltz2                         | Adaptation | ./boltz2                         | PASS                 |
| generator                      | Adaptation | ./generator                      | PASS                 |
| hardware-check-principle       | Adaptation | ./hardware-check-principle       | PASS                 |
| megatron-change-analyzer       | Adaptation | ./megatron-change-analyzer       | PASS                 |
| megatron-impact-mapper         | Adaptation | ./megatron-impact-mapper         | PASS                 |
| megatron-migration-generator   | Adaptation | ./megatron-migration-generator   | PASS                 |
| mindspeed-fsdp2-migration-main | Adaptation | ./mindspeed-fsdp2-migration-main | PASS                 |
| model-migration                | Adaptation | ./model-migration                | PASS                 |
| model-series-vendor-detector   | Adaptation | ./model-series-vendor-detector   | PASS                 |
| msverl-daily-regression-triage | Adaptation | ./msverl-daily-regression-triage | PASS                 |
| uv-torch-adaptation            | Adaptation | ./uv-torch-adaptation            | FAIL（document_parse） |
| vllm-ascend-model-adapter      | Adaptation | ./vllm-ascend-model-adapter      | PASS                 |


**这一组里**：PASS 13 / FAIL 1 / 报告里没出现的 0。

## 文档生成

共 **10** 个 skill。


| Agent 名称                      | 类型            | 链接                              | 验证（skills-eval） |
| ----------------------------- | ------------- | ------------------------------- | --------------- |
| ascend-docker                 | Documentation | ./ascend-docker                 | PASS            |
| ascendc-docs-gen              | Documentation | ./ascendc-docs-gen              | PASS            |
| ascendc-regbase-best-practice | Documentation | ./ascendc-regbase-best-practice | PASS            |
| ascendc-ut-develop            | Documentation | ./ascendc-ut-develop            | PASS            |
| connect                       | Documentation | ./connect                       | PASS            |
| coverage                      | Documentation | ./coverage                      | PASS            |
| run-mindspeed-llm-test        | Documentation | ./run-mindspeed-llm-test        | PASS            |
| skill                         | Documentation | ./skill                         | PASS            |
| swanlab-setup                 | Documentation | ./swanlab-setup                 | PASS            |
| triton-operator-doc-gen       | Documentation | ./triton-operator-doc-gen       | PASS            |


**这一组里**：PASS 10 / FAIL 0 / 报告里没出现的 0。

## 模型量化

共 **2** 个 skill。


| Agent 名称          | 类型           | 链接                  | 验证（skills-eval） |
| ----------------- | ------------ | ------------------- | --------------- |
| ascendc-npu-arch  | Quantization | ./ascendc-npu-arch  | PASS            |
| awesome-llm-model | Quantization | ./awesome-llm-model | PASS            |


**这一组里**：PASS 2 / FAIL 0 / 报告里没出现的 0。

## 知识检索

共 **1** 个 skill。


| Agent 名称            | 类型     | 链接                    | 验证（skills-eval） |
| ------------------- | ------ | --------------------- | --------------- |
| ascendc-docs-search | Search | ./ascendc-docs-search | PASS            |


**这一组里**：PASS 1 / FAIL 0 / 报告里没出现的 0。