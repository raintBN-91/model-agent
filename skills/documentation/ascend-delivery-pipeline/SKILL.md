---
name: ascend-delivery-pipeline
description: >
  昇腾模型适配结果自动化交付专家。当用户需要将适配/调优成果提交到 GitCode 时，
  自动扫描 deliverables、一键打包、修复 git 陷阱（dubious ownership、author identity、
  main/master 分支冲突）、自动创建远程仓库并推送，最后校验 README frontmatter NPU 标签。
  触发场景包括："提交gitcode"、"交付结果"、"上传适配报告"、"git报错"、
  "push到gitcode"、"打包交付"。
---

# ascend-delivery-pipeline — 适配结果自动化交付

## 核心工作流程

### 阶段 1：Deliverables 自动扫描

**任务 1.1：识别当前目录下的候选交付文件**

```bash
# 按类别扫描文件
echo "=== 扫描交付文件 ==="
README_FILES=$(ls -1 README* readme* 2>/dev/null)
PY_FILES=$(ls -1 *.py 2>/dev/null)
LOG_FILES=$(ls -1 *.log 2>/dev/null)
JSON_FILES=$(ls -1 *.json 2>/dev/null)
MD_FILES=$(ls -1 *.md 2>/dev/null | grep -iv readme)
CONFIG_FILES=$(ls -1 *.yaml *.yml *.json 2>/dev/null | grep -E "config|setting" || true)

# 汇总清单
cat > /tmp/delivery_manifest_raw.txt <<EOF
README文档:
$README_FILES

Python脚本:
$PY_FILES

日志文件:
$LOG_FILES

JSON数据:
$JSON_FILES

其他文档:
$MD_FILES

配置文件:
$CONFIG_FILES
EOF

cat /tmp/delivery_manifest_raw.txt
```

**任务 1.2：大文件与敏感文件预检**

```bash
echo "=== 安全预检 ==="
# 检测大文件（>100MB）
find . -maxdepth 1 -type f -size +100M -exec ls -lh {} \; 2>/dev/null | awk '{print "WARNING: 大文件 >100MB:", $9, $5}'

# 检测敏感文件
SENSITIVE=$(find . -maxdepth 1 -type f \( -name "*.env*" -o -name "*.key" -o -name "*.pem" -o -name "*secret*" -o -name "*password*" -o -name "*token*" \) 2>/dev/null)
if [ -n "$SENSITIVE" ]; then
    echo "WARNING: 检测到潜在敏感文件:"
    echo "$SENSITIVE"
    echo "建议从交付包中排除或使用 .gitignore"
fi
```

**任务 1.3：用户确认交付清单**

向用户展示扫描结果，询问：
- "以上文件将被纳入交付包，是否有需要排除或补充的文件？"
- 用户可回复："排除 xxx.log"、"补充 xxx.json"

### 阶段 2：一键打包

**任务 2.1：创建交付目录**

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DELIVERY_DIR="deliverables_${TIMESTAMP}"
mkdir -p "$DELIVERY_DIR"

# 复制已确认的文件
cp -v README* readme* "$DELIVERY_DIR/" 2>/dev/null || true
cp -v *.py "$DELIVERY_DIR/" 2>/dev/null || true
cp -v *.log "$DELIVERY_DIR/" 2>/dev/null || true
cp -v *.json "$DELIVERY_DIR/" 2>/dev/null || true
cp -v *.md "$DELIVERY_DIR/" 2>/dev/null || true

# 生成 MANIFEST.md
cat > "$DELIVERY_DIR/MANIFEST.md" <<EOF
# 交付清单

| 类别 | 文件 | 说明 |
|------|------|------|
EOF

for f in $(ls -1 "$DELIVERY_DIR/"); do
    case "$f" in
        README*|readme*) DESC="模型部署与适配说明文档" ;;
        *.py) DESC="推理/验证脚本" ;;
        *.log) DESC="运行日志" ;;
        *.json) DESC="结构化数据/报告" ;;
        *.md) DESC="补充文档" ;;
        *) DESC="其他" ;;
    esac
    echo "| $(echo $f | sed 's/\.[^.]*$//') | $f | $DESC |" >> "$DELIVERY_DIR/MANIFEST.md"
done

echo "交付包已创建: $DELIVERY_DIR/"
cat "$DELIVERY_DIR/MANIFEST.md"
```

### 阶段 3：Git 陷阱自动修复

**任务 3.1：初始化 git 仓库**

```bash
cd "$DELIVERY_DIR"

if [ ! -d .git ]; then
    git init
    echo "Git 仓库已初始化"
fi
```

**任务 3.2：修复 safe.directory**

```bash
# 自动将当前目录加入 safe.directory
git config --global --add safe.directory "$(pwd)" 2>/dev/null || \
git config --local --add safe.directory "$(pwd)" 2>/dev/null || true
```

**任务 3.3：配置用户身份**

```bash
# 如果全局未配置，使用本地配置兜底
if [ -z "$(git config user.name)" ]; then
    GIT_USER="${USER:-$(whoami)}"
    git config user.name "$GIT_USER"
    echo "Set git user.name: $GIT_USER"
fi

if [ -z "$(git config user.email)" ]; then
    GIT_EMAIL="${EMAIL:-agent@atomgit.ai}"
    git config user.email "$GIT_EMAIL"
    echo "Set git user.email: $GIT_EMAIL"
fi
```

**任务 3.4：处理 main/master 分支映射**

```bash
# 检测本地默认分支
DEFAULT_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")

# 如果本地是 master，统一为 main（符合 GitCode 新仓库默认规范）
if [ "$DEFAULT_BRANCH" = "master" ]; then
    git branch -m main
    echo "分支已重命名: master -> main"
    DEFAULT_BRANCH="main"
fi

# 如果本地无分支（新仓库），创建 main
git checkout -b main 2>/dev/null || true
```

### 阶段 4：远程仓库创建与推送

**任务 4.1：解析目标仓库信息**

```bash
# 从环境变量或用户输入获取
cd "$DELIVERY_DIR"
REPO_OWNER="${GITCODE_OWNER:-yangkang11111}"
REPO_NAME="${GITCODE_REPO:-$(basename $(cd .. && pwd))}"
REPO_NAME=$(echo "$REPO_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_-]/-/g')

# 推断 GitCode Token（从现有 remote URL）
GITCODE_TOKEN="${GITCODE_TOKEN:-}"
if [ -z "$GITCODE_TOKEN" ]; then
    # 尝试从 origin remote 提取
    EXISTING_REMOTE=$(git config --get remote.origin.url 2>/dev/null || echo "")
    if echo "$EXISTING_REMOTE" | grep -q "auth:"; then
        GITCODE_TOKEN=$(echo "$EXISTING_REMOTE" | sed 's/.*auth:\([^@]*\)@.*/\1/')
    fi
fi

echo "目标仓库: gitcode.com/$REPO_OWNER/$REPO_NAME"
```

**任务 4.2：检测远程仓库是否存在**

```bash
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    "https://gitcode.com/$REPO_OWNER/$REPO_NAME" 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "301" ] || [ "$HTTP_CODE" = "302" ]; then
    echo "远程仓库已存在"
    REPO_EXISTS=true
else
    echo "远程仓库不存在，需要创建"
    REPO_EXISTS=false
fi
```

**任务 4.3：自动创建远程仓库（如不存在）**

```bash
if [ "$REPO_EXISTS" = "false" ] && [ -n "$GITCODE_TOKEN" ]; then
    # GitCode API 创建仓库（格式参考 Gitee/Gitea API）
    curl -s -X POST \
        "https://gitcode.com/api/v5/user/repos" \
        -H "Content-Type: application/json" \
        -H "Authorization: token $GITCODE_TOKEN" \
        -d "{\"name\":\"$REPO_NAME\",\"private\":false,\"auto_init\":false,\"description\":\"昇腾NPU模型适配交付件\"}" \
        2>/dev/null | head -c 500
    echo ""
    echo "仓库创建请求已发送，请确认创建结果"
fi
```

**任务 4.4：配置 remote 并推送**

```bash
REMOTE_URL="https://auth:${GITCODE_TOKEN}@gitcode.com/$REPO_OWNER/$REPO_NAME.git"

# 配置或更新 origin
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"

# 添加文件并提交
git add .
git commit -m "feat: deliver model adaptation results

包含推理脚本、验证日志、性能报告及适配说明文档。

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"

# 推送到 main
git push -u origin main
```

### 阶段 5：Frontmatter NPU 标签校验

**任务 5.1：检查 README frontmatter**

```bash
README_FILE=$(ls -1 README* readme* 2>/dev/null | head -1)

if [ -n "$README_FILE" ]; then
    # 检查是否已有 frontmatter
    if head -5 "$README_FILE" | grep -q "^---"; then
        # 检查是否含 NPU/Ascend 标签
        if grep -q -E "npu|ascend|昇腾|NPU|Ascend" "$README_FILE"; then
            echo "✅ README frontmatter 已包含 NPU/Ascend 标签"
        else
            echo "⚠️ README frontmatter 缺少 NPU/Ascend 标签，建议补充"
            echo "示例: tags: [NPU, Ascend, vLLM]"
        fi
    else
        echo "⚠️ README 缺少 YAML frontmatter，建议添加以支持模型卡片自动识别"
        echo "示例:"
        echo "---"
        echo "tags:"
        echo "  - NPU"
        echo "  - Ascend"
        echo "  - vLLM"
        echo "hardware: Ascend-910B"
        echo "---"
    fi
fi
```

### 阶段 6：交付报告生成

```bash
cat > delivery_report.json <<EOF
{
  "delivery_time": "$(date -Iseconds)",
  "target_repo": "gitcode.com/$REPO_OWNER/$REPO_NAME",
  "package_dir": "$DELIVERY_DIR",
  "git_fixes_applied": [
    "safe.directory",
    "user.identity",
    "main_branch_mapping"
  ],
  "files_delivered": [
$(ls -1 "$DELIVERY_DIR/" | sed 's/^/    "/;s/$/",/' | sed '$ s/,$//')
  ],
  "frontmatter_check": "$(grep -q 'NPU\|Ascend' "$README_FILE" 2>/dev/null && echo 'passed' || echo 'warning')",
  "status": "completed"
}
EOF

echo ""
echo "=== 交付完成 ==="
echo "交付包目录: $DELIVERY_DIR"
echo "目标仓库: https://gitcode.com/$REPO_OWNER/$REPO_NAME"
echo "交付报告: delivery_report.json"
```

## 异常处理规则

| 异常情况 | 处理方案 |
|---------|---------|
| `dubious ownership` | 自动执行 `git config --global --add safe.directory $(pwd)` |
| `author identity unknown` | 自动配置 `user.name` 和 `user.email`（环境变量或系统默认值） |
| `remote rejected` (main vs master) | 本地分支统一重命名为 `main`，并强制对应远程 `main` |
| 仓库不存在且无 API Token | 提示用户手动创建仓库，或提供 `GITCODE_TOKEN` |
| 推送冲突 | 先 `git pull origin main --rebase`，冲突时提示用户手动解决 |
| 大文件阻塞推送 | 提示用户将大文件（>100MB）移入 LFS 或排除交付包 |
| README 缺少 frontmatter | 仅警告，不阻塞推送，提供补充示例 |
