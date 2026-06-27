---
name: verify-agent
description: 昇腾模型适配验证 (Ascend Model Adaptation Verification) - 自动化验证华为昇腾设备上 vLLM-Ascend 模型部署、工具。当用户提及在昇腾 NPU (Atlas 800 A2/A3) 上部署大语言模型、时，使用此 Skill。适用于 Qwen3.5-0.8B、DeepSeek-V3 等 vLLM-Ascend 支持的模型。思考和输出都以中文进行
---

# 昇腾模型适配验证 (Ascend Model Adaptation Verification)

基于 vLLM-Ascend 后端的自动化验证流水线，用于验证大语言模型在华为昇腾 NPU 设备上的部署能力。

## 核心流程

```
1.检查环境 ──→ 2. 启动服务 ──→ 3. 执行验证 ──→ 4. 资源清理(可选) ──→ 5. 生成报告
```
**无需用户确认，直接执行**

---

## 输入参数
不需要默认参数，直接使用模型的默认地址：~/.cache/modelscope/hub/models/Qwen/Qwen3.5-0.8B

## 验证步骤

### Step 1: 环境预检 
 
 
依次执行以下检查	 
1. 检查NPU设备和驱动 
```bash 
npu-smi info 
```

2. 检查vLLM-Ascend 
```bash 
pip list | grep vllm 
```

3. 检查是否有vllm进程残留 
```bash 
pkill -f "vllm serve" 
```

### Step 2: 启动服务

根据默认参数在后台直接启动临时 vLLM API 服务器，启动命令如下，直接使用这个命令启动：

```bash
nohup vllm serve ~/.cache/modelscope/hub/models/Qwen/Qwen3.5-0.8B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  > /tmp/vllm_serve.log 2>&1 &
```

**重要：为支持流式反馈，轮询必须分步执行，每次探测使用独立的 Bash 调用，不要写成一个长循环脚本。**
1. 使用`tail -f` 阻塞等待就绪关键字，匹配后立即执行 HTTP 确认：
# 前台实时流式监听，逐行输出
```bash
timeout 300 bash << 'EOF'
 	     tail -n0 -f /tmp/vllm_serve.log | while read line; do
 	         echo "$line"
 	         if echo "$line" | grep -qE "Application startup complete|Uvicorn running"; then
 	             break
 	         fi
 	     done
 	 EOF
```

2. 若最终超时，查看日志：
```bash
tail -n 100 /tmp/vllm_serve.log
```

若探测失败，可通过日志排查原因：
```bash
tail -n 100 /tmp/vllm_serve.log
```

### Step 3: 执行验证测试

#### 3.1 功能验证

使用curl请求
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "{model_path}",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_completion_tokens": 4096
  }'
```

### Step 4: 资源清理
验证完成后自动停止服务进程：

```bash
pkill -f "vllm serve"
```

### Step 5: 生成验证报告
输出结构化 JSON 报告：

```json
{
  "success": true,
  "summary": "验证完成，耗时 320.5 秒",
  "detailed_report": {
    "start_command": "vllm serve ~/.cache/modelscope/hub/models/Qwen/Qwen3.5-0.8B --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --max-model-len 8192 --gpu-memory-utilization 0.85 --enforce-eager",
    "logs": ["步骤1完成", "服务启动成功", "精度测试通过"]
  },
  "report_path": "./validation_report.json"
}
```
**操作步骤：**

1. 在当前目录下使用 touch 创建一个空的、名为`{model_name}-validation_report.json`的json文件

2. 用 Write 工具将报告写入 `{model_name}-validation_report.json`

3. 仅告知用户：验证方案报告已生成，并给出报告的绝对路径，**不用在回复中输出报告全文**

---

## 输出格式

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | boolean | 整个流程是否成功 |
| `summary` | string | 人类可读的结果摘要 |
| `detailed_report` | object | 详细测试结果 |
| `detailed_report.start_command` | string | 实际使用的 vLLM 启动命令 |
| `detailed_report.logs` | array | 关键步骤日志 |
| `report_path` | string | 报告文件绝对路径 |

---

## 错误处理

- **环境检查失败**：记录错误信息，直接返回失败报告
- **服务启动失败**：捕获异常，尝试清理进程，报告失败原因
- **API 调用超时**：设置 60 秒超时，自动重试 2 次
- **测试执行失败**：记录失败环节，继续执行后续测试（可选）

---
