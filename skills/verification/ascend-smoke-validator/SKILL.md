---
name: ascend-smoke-validator
description: >
  vLLM-Ascend 在线服务一键 Smoke 验证专家。当 vLLM serve 启动后，
  自动探测服务端口，执行标准化的 API smoke 测试用例集（模型列表、chat、completion、function calling），
  捕获响应并生成结构化报告。如遇失败，自动诊断常见原因（端口未监听、模型名不匹配、OOM 等）。
  触发场景包括："smoke测试"、"服务健康检查"、"curl测试"、
  "验证服务是否可用"、"测试API"、"服务通了吗"。
---

# ascend-smoke-validator — vLLM-Ascend 一键 Smoke 验证

## 核心工作流程

### 阶段 1：服务端口自发现

**任务 1.1：探测 vLLM serve 进程与端口**

```bash
echo "=== 探测 vLLM 服务 ==="

# 方法1：通过进程查找监听端口
VLLM_PID=$(pgrep -f "vllm.*serve" | head -1)
if [ -n "$VLLM_PID" ]; then
    PORT=$(ss -tlnp | grep -E "pid=$VLLM_PID|vllm" | awk '{print $4}' | cut -d: -f2 | head -1)
fi

# 方法2：读取 pid 文件旁常见端口
cat vllm_serve.log 2>/dev/null | grep -oE "http://[^:]+:[0-9]+" | head -1

# 方法3：默认回退
PORT="${PORT:-8000}"
HOST="${VLLM_HOST:-localhost}"
BASE_URL="http://${HOST}:${PORT}"

echo "Detected base URL: $BASE_URL"
```

**任务 1.2：健康检查**

```bash
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health" 2>/dev/null || echo "000")
MODELS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/v1/models" 2>/dev/null || echo "000")

if [ "$HEALTH_STATUS" = "200" ]; then
    echo "✅ Health endpoint: OK (200)"
elif [ "$MODELS_STATUS" = "200" ]; then
    echo "⚠️ Health endpoint unavailable, but /v1/models responds (200)"
else
    echo "❌ Service not responding. Health: $HEALTH_STATUS, Models: $MODELS_STATUS"
fi
```

### 阶段 2：标准化 Smoke 测试

**任务 2.1：测试用例 1 — 模型列表查询**

```bash
echo ""
echo "=== Smoke Test 1: /v1/models ==="
MODELS_RESP=$(curl -s "${BASE_URL}/v1/models" 2>/dev/null)
echo "$MODELS_RESP" > /tmp/smoke_test_1_models.json

if echo "$MODELS_RESP" | grep -q '"id"'; then
    MODEL_NAME=$(echo "$MODELS_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null)
    echo "✅ PASS | Model: $MODEL_NAME"
    TEST1_STATUS="PASS"
else
    echo "❌ FAIL | Cannot retrieve model list"
    TEST1_STATUS="FAIL"
    TEST1_ERROR="Invalid response or no models found"
fi
```

**任务 2.2：测试用例 2 — Chat Completions**

```bash
echo ""
echo "=== Smoke Test 2: /v1/chat/completions ==="
CHAT_RESP=$(curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME:-model}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Hello, are you working?\"}],
    \"max_tokens\": 50,
    \"temperature\": 0.1
  }" 2>/dev/null)

echo "$CHAT_RESP" > /tmp/smoke_test_2_chat.json

if echo "$CHAT_RESP" | grep -q '"choices"'; then
    CONTENT=$(echo "$CHAT_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null | head -c 100)
    echo "✅ PASS | Response: ${CONTENT}..."
    TEST2_STATUS="PASS"
else
    echo "❌ FAIL | No choices in response"
    TEST2_STATUS="FAIL"
    TEST2_ERROR=$(echo "$CHAT_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error','Unknown error'))" 2>/dev/null || echo "Invalid JSON response")
fi
```

**任务 2.3：测试用例 3 — Completions**

```bash
echo ""
echo "=== Smoke Test 3: /v1/completions ==="
COMP_RESP=$(curl -s "${BASE_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME:-model}\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 20,
    \"temperature\": 0.1
  }" 2>/dev/null)

echo "$COMP_RESP" > /tmp/smoke_test_3_completion.json

if echo "$COMP_RESP" | grep -q '"choices"'; then
    TEXT=$(echo "$COMP_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'])" 2>/dev/null | head -c 100)
    echo "✅ PASS | Response: ${TEXT}..."
    TEST3_STATUS="PASS"
else
    echo "❌ FAIL | No choices in response"
    TEST3_STATUS="FAIL"
    TEST3_ERROR=$(echo "$COMP_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error','Unknown error'))" 2>/dev/null || echo "Invalid JSON response")
fi
```

**任务 2.4：测试用例 4 — Function Calling（如模型支持）**

```bash
echo ""
echo "=== Smoke Test 4: /v1/chat/completions (function calling) ==="
FC_RESP=$(curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME:-model}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What's the weather in Beijing?\"}],
    \"tools\": [{
      \"type\": \"function\",
      \"function\": {
        \"name\": \"get_weather\",
        \"description\": \"Get weather info\",
        \"parameters\": {
          \"type\": \"object\",
          \"properties\": {\"location\": {\"type\": \"string\"}},
          \"required\": [\"location\"]
        }
      }
    }],
    \"max_tokens\": 100
  }" 2>/dev/null)

echo "$FC_RESP" > /tmp/smoke_test_4_function_calling.json

# Function calling 是可选能力，不强制 PASS
if echo "$FC_RESP" | grep -q '"tool_calls"\|"function_call"'; then
    echo "✅ PASS | Function calling supported"
    TEST4_STATUS="PASS"
elif echo "$FC_RESP" | grep -q '"choices"'; then
    echo "⚠️ SKIP | Model responds but does not use tools (may be expected)"
    TEST4_STATUS="SKIP"
else
    echo "❌ FAIL | Error in function calling request"
    TEST4_STATUS="FAIL"
    TEST4_ERROR=$(echo "$FC_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error','Unknown error'))" 2>/dev/null || echo "Invalid JSON response")
fi
```

### 阶段 3：自动诊断（失败时）

**任务 3.1：诊断引擎**

如果任一测试失败，执行以下诊断：

```bash
echo ""
echo "=== 自动诊断 ==="

# D1: 端口未监听
if [ "$MODELS_STATUS" != "200" ]; then
    echo "[D1] 端口未监听或服务未启动"
    echo "     检查: npu-smi info -t processes | grep vllm"
    echo "     检查: cat vllm_serve.log | tail -20"
fi

# D2: 模型名不匹配
if [ "$TEST2_STATUS" = "FAIL" ] || [ "$TEST3_STATUS" = "FAIL" ]; then
    if echo "${TEST2_ERROR}${TEST3_ERROR}" | grep -qi "model.*not\|invalid.*model"; then
        echo "[D2] 模型名不匹配"
        echo "     可用模型: $MODEL_NAME"
        echo "     请确认请求中的 model 字段与 /v1/models 返回的 id 一致"
    fi
fi

# D3: OOM / 内存不足
if echo "${TEST2_ERROR}${TEST3_ERROR}" | grep -qi "out.*memory\|OOM\|cuda\|npu.*memory"; then
    echo "[D3] NPU 内存不足 (OOM)"
    echo "     检查: npu-smi info -t memory"
    echo "     建议: 减小 max_model_len 或降低 tensor-parallel-size"
fi

# D4: 服务启动中 / 模型加载未完成
if echo "${TEST2_ERROR}${TEST3_ERROR}" | grep -qi "loading\|warmup\|not.*ready"; then
    echo "[D4] 服务正在启动或模型加载中"
    echo "     建议: 等待 1-2 分钟后重试"
    echo "     检查: tail -f vllm_serve.log"
fi

# D5: 网络/防火墙
if [ "$MODELS_STATUS" = "000" ]; then
    echo "[D5] 无法连接到服务端口"
    echo "     检查: curl -v ${BASE_URL}/v1/models"
    echo "     检查: ss -tlnp | grep $PORT"
fi
```

### 阶段 4：结构化报告生成

**任务 4.1：生成 smoke_test.log**

```bash
cat > smoke_test.log <<EOF
=== vLLM-Ascend Smoke Test Log ===
Timestamp: $(date -Iseconds)
Service URL: ${BASE_URL}
Model Name: ${MODEL_NAME:-N/A}

Test Results:
----------------------------------------
[1] /v1/models        | ${TEST1_STATUS} | ${TEST1_ERROR:-N/A}
[2] /v1/chat/completions | ${TEST2_STATUS} | ${TEST2_ERROR:-N/A}
[3] /v1/completions   | ${TEST3_STATUS} | ${TEST3_ERROR:-N/A}
[4] Function Calling  | ${TEST4_STATUS} | ${TEST4_ERROR:-N/A}
----------------------------------------
Overall: $(if [ "$TEST1_STATUS" = "PASS" ] && [ "$TEST2_STATUS" = "PASS" ] && [ "$TEST3_STATUS" = "PASS" ]; then echo "PASS"; else echo "FAIL"; fi)
EOF

cat smoke_test.log
```

**任务 4.2：生成 smoke_report.json**

```bash
python3 <<PYEOF
import json
from datetime import datetime

report = {
    "meta": {
        "test_time": datetime.now().isoformat(),
        "service_url": "${BASE_URL}",
        "model_name": "${MODEL_NAME:-N/A}"
    },
    "results": [
        {
            "id": 1,
            "endpoint": "/v1/models",
            "method": "GET",
            "status": "${TEST1_STATUS}",
            "error": "${TEST1_ERROR:-}" if "${TEST1_ERROR:-}" else None
        },
        {
            "id": 2,
            "endpoint": "/v1/chat/completions",
            "method": "POST",
            "status": "${TEST2_STATUS}",
            "error": "${TEST2_ERROR:-}" if "${TEST2_ERROR:-}" else None
        },
        {
            "id": 3,
            "endpoint": "/v1/completions",
            "method": "POST",
            "status": "${TEST3_STATUS}",
            "error": "${TEST3_ERROR:-}" if "${TEST3_ERROR:-}" else None
        },
        {
            "id": 4,
            "endpoint": "/v1/chat/completions (function calling)",
            "method": "POST",
            "status": "${TEST4_STATUS}",
            "error": "${TEST4_ERROR:-}" if "${TEST4_ERROR:-}" else None
        }
    ],
    "overall": "PASS" if all(r["status"] == "PASS" for r in [
        {"status": "${TEST1_STATUS}"},
        {"status": "${TEST2_STATUS}"},
        {"status": "${TEST3_STATUS}"}
    ]) else "FAIL"
}

with open("smoke_report.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("Report saved: smoke_report.json")
PYEOF
```

**任务 4.3：终端汇总输出**

```bash
echo ""
echo "========================================"
echo "   Smoke Test Complete"
echo "========================================"
echo "Log:  smoke_test.log"
echo "JSON: smoke_report.json"
echo ""
python3 -c "import json; r=json.load(open('smoke_report.json')); print('Overall:', r['overall'])"
```

## 异常处理规则

| 异常情况 | 处理方案 |
|---------|---------|
| 无法找到服务端口 | 遍历 8000-8010 端口尝试，全部失败则提示手动指定 `VLLM_HOST`/`PORT` |
| 模型名未知 | 先从 `/v1/models` 获取，若失败则使用占位符 `model` |
| 服务响应超时 | curl 设置 `--max-time 30`，超时标记为 FAIL |
| 返回非 JSON | 记录原始响应前 500 字符，标记为 FAIL |
| Function calling 不支持 | 标记为 SKIP（非 FAIL），不影响 overall 判定 |
| 诊断后仍无法解决 | 提示用户查看 `vllm_serve.log` 并调用 `ascend-resource-scheduler` 检查资源 |
