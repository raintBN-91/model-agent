# ascend-skills-eval 使用指南

`ascend-skills-eval` 是面向昇腾 Skills 的结构评测系统，提供 9 维评分、Markdown 报告与可视化成果卡。本指南覆盖从启动到批量评测的端到端流程。

## 前置条件

| 项目 | 要求 |
|------|------|
| Python | 3.10+ |
| Node.js | 18+（成果卡渲染依赖 puppeteer/playwright，**不能用 pip 安装**，需用 nvm 或系统包管理器安装） |
| git | 任意版本（仓库评测需要 clone） |

## 1. 启动服务

```bash
cd ascend-skills-eval/web-service
pip install -r requirements.txt
# Node.js 需独立安装（pip 装不了），任选一种：
#   nvm install 18 && nvm use 18         # 跨平台推荐
#   winget install OpenJS.NodeJS          # Windows
#   Ubuntu/Debian 用 NodeSource 官方源装 18+（apt 默认源版本太旧）：
#     curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
#     sudo apt-get install -y nodejs
node -v   # 确认 >= 18
# 本地调试建议绑 127.0.0.1，避免暴露公网；确需外网访问再改 0.0.0.0 并加防火墙
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

健康检查：
```bash
curl http://localhost:8000/health
# {"status":"ok","service":"ascend-skills-eval"}
```

## 2. 单 Skill 评测

直接粘贴 SKILL.md 内容。注意：用 `jq -Rs` 安全读取文件嵌入 JSON，避免换行/引号破坏 JSON 格式；路径要相对于当前工作目录（已在 web-service 下，故用 `../../` 回到仓库根）：

```bash
# 从仓库根目录运行（推荐），或在 web-service 下用 ../../ 前缀
SKILL_MD=$(jq -Rs . < ../../skills/deployment/npu-smi/SKILL.md)
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d "{\"skill_name\":\"npu-smi\",\"skill_markdown\":$SKILL_MD}"
```

> ⚠️ 若 SKILL.md 含双引号/换行/反斜杠，**必须**用 `jq -Rs` 转义，直接 `$(cat ...)` 会破坏 JSON。

返回 JSON 包含 `total_score`（0-100）、9 个 `dimensions`（含权重/分数/说明）与 `suggestions`。

## 3. 仓库评测

给定 git 仓库链接，服务自动 `git clone --depth 1` 并评测：

```bash
curl -X POST http://localhost:8000/evaluate-repo \
  -H "Content-Type: application/json" \
  -d "{\"repo_url\":\"https://gitcode.com/Ascend/model-agent.git\",\"skill_path\":\"skills/deployment/npu-smi/SKILL.md\"}"
```

返回 `eval`、`card`（base64 PNG 成果卡）、`report_markdown`、`picked_file`。

## 4. 批量仓库评测

并发评估多个仓库并生成排行榜：

```bash
curl -X POST http://localhost:8000/evaluate-repos \
  -H "Content-Type: application/json" \
  -d "{\"items\":[{\"repo_url\":\"https://github.com/A/B.git\"},{\"repo_url\":\"https://gitcode.com/C/D.git\"}]}"
```

返回 `ranking`（按分数降序）、`results`、`failed`、`batch_report_markdown`、`top_card`。

> 性能优化后（见 PR #03）：10 仓库并发耗时 ~7s（原串行 ~25s），批量总超时 180s。

## 5. 成果卡生成

```bash
curl -X POST http://localhost:8000/render-card \
  -H "Content-Type: application/json" \
  -d "{\"data\":{\"skill-name\":\"demo\",\"score-before\":70,\"score-after\":88,\"improve-1\":\"a\",\"improve-2\":\"b\",\"improve-3\":\"c\",\"dims\":[]},\"open_image\":false}"
```

返回 `image_base64`，可直接 `<img src="data:image/png;base64,...">` 嵌入。渲染超时 30s，超时返回 504。

## 6. 端到端示例

`POST /evaluate-and-render` 一步完成「评测 + 成果卡」：

```bash
curl -X POST http://localhost:8000/evaluate-and-render \
  -H "Content-Type: application/json" \
  -d "{\"skill_markdown\":\"---\\nname: demo\\n---\\n# Demo\"}"
```

## 常见问题

| 问题 | 处理 |
|------|------|
| `render-card` 返回 500 找不到脚本 | 检查 `ascend-skills-eval/skills/skills-eval/scripts/render-card.mjs` 是否存在 |
| `evaluate-repo` 400 仓库拉取失败 | 确认仓库公开可访问，或指定正确的 `branch` |
| 评分维度全是低分 | 检查 SKILL.md 是否含 frontmatter、步骤序号、异常处理关键词、昇腾术语 |
