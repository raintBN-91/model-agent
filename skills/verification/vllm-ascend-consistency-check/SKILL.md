---
name: vllm-ascend-consistency-check
description: >
  vLLM-Ascend 模型轻量级一致性验证 Skill。
  适用于小模型或快速验证场景，替代重量级 benchmark（如 GSM8K）。
  涵盖 temperature=0 一致性测试、基础推理测试（算术/事实/逻辑）、
  多语言测试、报告生成与通过标准。提供可复用的 Python 脚本模板。
  当用户提到模型精度验证、一致性检查、轻量级模型测试、
  NPU 推理结果对比时触发。
metadata:
  short-description: vLLM-Ascend 模型轻量级一致性验证
  category: NPU-Model-Verification
  tags: [ascend, npu, vllm, accuracy, consistency, verification, lightweight, evaluation]
---

# vLLM-Ascend 模型轻量级一致性验证 Skill

本 Skill 提供对 vLLM-Ascend 部署的模型进行轻量级一致性验证的方法论和脚本模板。
特别适用于小参数模型（如 270M-1B）或部署链路快速验证，替代耗时的
完整 benchmark（如 GSM8K）。

以 `google/gemma-3-270m-it` 在 Atlas 800 A2 (NPU 910B4) 上的验证为参考案例。

## 前置条件

| 项目 | 要求 |
|------|------|
| 服务 | vLLM OpenAI-compatible API 已启动并可达 |
| 网络 | 测试机可访问服务地址（默认 `http://127.0.0.1:8000`） |
| 依赖 | `requests`（Python） |

## 流程总览

```
0. 确认服务就绪
→ 1. 一致性测试（temperature=0）
→ 2. 基础推理测试
→ 3. 多语言测试
→ 4. 报告生成与验收
```

---

## 0. 确认服务就绪

```bash
curl -sf http://127.0.0.1:8000/v1/models > /dev/null \
  && echo "Service ready" || echo "Service not ready"
```

---

## 1. 一致性测试

### 1.1 测试目的

验证模型在固定输入下输出是否稳定，排除随机性导致的推理不一致。

### 1.2 测试方法

使用 `temperature=0` 对同一 prompt 连续调用 3 次，对比输出是否完全一致。

```python
prompt = "What is the capital of France? Answer in one word."
# 调用 3 次，temperature=0, max_tokens=16
# 检查 3 次输出是否完全相同
```

### 1.3 通过标准

3 次输出**完全一致**（字符串级别）。

> **原理**：temperature=0 时采样是确定性的（贪婪解码）。若输出不一致，
> 可能原因：服务未完全预热、并发干扰、或模型本身存在非确定性算子。

---

## 2. 基础推理测试

### 2.1 测试目的

验证模型具备基本的算术、事实和逻辑推理能力。

### 2.2 测试用例

| 类别 | Prompt | max_tokens |
|------|--------|-----------|
| 算术 | `What is 15 + 27?` | 16 |
| 事实 | `Which planet is known as the Red Planet?` | 16 |
| 逻辑 | `If all cats have tails, and Tom is a cat, does Tom have a tail? Answer yes or no.` | 16 |

### 2.3 通过标准

- 模型对每条 prompt 均返回非空响应
- 响应内容与问题相关（无需严格答案匹配，小模型可能回答不完整）

> **注意**：对于 270M 级别的小模型，生成质量受参数量限制，
> 测试目的是验证**推理链路完整**而非评估模型智商。

---

## 3. 多语言测试

### 3.1 测试目的

验证模型在非英语输入下也能正常生成，确认 tokenizer 和推理链路
对多语言的支持。

### 3.2 测试用例

| 语言 | Prompt | max_tokens |
|------|--------|-----------|
| 中文 | `用一句中文说明 TCP 和 UDP 的核心区别。` | 64 |

### 3.3 通过标准

- 模型返回非空响应
- 响应包含中文字符（确认 tokenizer 未截断或乱码）

---

## 4. 报告生成与验收

### 4.1 报告格式

脚本输出 JSON 格式报告：

```json
{
  "model": "gemma-3-270m-it",
  "hardware": "Atlas 800 A2, NPU 910B4",
  "consistency": {
    "prompt": "What is the capital of France? Answer in one word.",
    "outputs": ["Paris", "Paris", "Paris"],
    "consistent": true
  },
  "reasoning": [
    {"name": "Arithmetic", "prompt": "What is 15 + 27?", "output": "42"},
    {"name": "Fact", "prompt": "Which planet is known as the Red Planet?", "output": "Mars"},
    {"name": "Logic", "prompt": "...", "output": "Yes"}
  ],
  "chinese": {
    "prompt": "用一句中文说明 TCP 和 UDP 的核心区别。",
    "output": "TCP是面向连接的可靠传输协议，UDP是无连接的不可靠传输协议。"
  }
}
```

### 4.2 验收清单

- [ ] 一致性测试通过（3 次输出一致）
- [ ] 基础推理测试全部返回非空响应
- [ ] 多语言测试返回包含目标语言字符的响应
- [ ] 报告 JSON 已保存到指定路径
- [ ] 推理链路端到端打通确认

---

## 运行脚本

使用本 Skill 附带的脚本：

```bash
python scripts/check.py \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --model-name gemma-3-270m-it \
  --output ./accuracy_report.json
```

## 与重量级 Benchmark 的对比

| 验证方式 | 适用场景 | 耗时 | 资源 |
|---------|---------|------|------|
| **本 Skill（轻量验证）** | 小模型、快速部署验证、CI/CD | 1-2 分钟 | 低 |
| GSM8K / MMLU | 大模型、学术评估、发布报告 | 数小时 | 高 |

> **建议**：先用本 Skill 快速验证部署链路，再按需决定是否跑完整 benchmark。

---

## 参考

- 本 Skill 脚本模板：`scripts/check.py`
