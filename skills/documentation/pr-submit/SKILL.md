---
name: model-agent-pr-submit
description: >
  向 Ascend/model-agent 仓库提交跨仓库 Merge Request 的标准化流程。
  涵盖 fork 配置、本地开发、commit 规范、跨仓库 MR 创建与 CLA 修复。
  当用户提到提交 skill、创建 MR、跨仓库 PR、CLA 修复、fork 提交时触发。
metadata:
  short-description: Ascend model-agent 跨仓库 MR 提交工作流
  category: Workflow
  tags: [git, gitcode, merge-request, cla, fork, workflow, pr-submit]
---

# model-agent-pr-submit

向 `Ascend/model-agent` 仓库提交跨仓库 PR 的完整工作流。

## 固定配置

| 项目 | 值 |
|---|---|
| 上游仓库 | `Ascend/model-agent`（项目 ID: `9646177`） |
| Fork 仓库 | `gcw_C8PI9e90/model-agent_6443`（项目 ID: `9850257`） |
| 目标分支 | `master` |
| Token 环境变量 | `ATOMGIT_USER_TOKEN` |
| 提交者身份 | `git config user.name gcw_C8PI9e90`、`git config user.email 13290192271@163.com` |

> **关键**：commit 的 author 和 committer 邮箱必须与 CLA 签署邮箱一致（`13290192271@163.com`）。CLA 检查以 committer 邮箱为准。

## 流程总览

```
0. 环境准备（首次）
→ 1. 创建功能分支
→ 2. 添加文件并提交
→ 3. 推送到 Fork
→ 4. 创建跨仓库 MR
→ 5. CLA 问题修复（如需要）
→ 6. 验收确认
```

---

## 0. 环境准备（首次）

0. 配置 Git 身份
1. 添加 upstream 远程
2. 添加 fork 远程
3. 拉取最新代码

```bash
# 配置 Git 身份（必须与 CLA 签署邮箱一致）
git config --global user.name "gcw_C8PI9e90"
git config --global user.email "13290192271@163.com"

# 添加 upstream（上游仓库）
git remote add upstream https://gitcode.com/Ascend/model-agent.git

# 添加 fork 远程（使用 token 认证）
git remote add fork "https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/gcw_C8PI9e90/model-agent_6443.git"

# 拉取最新代码
git fetch upstream master
git fetch fork
```

**输入/输出定义**：
- 输入：Git 环境、`ATOMGIT_USER_TOKEN` 环境变量
- 输出：配置好的 upstream 和 fork remote
- 异常：如果 `git remote add` 报错 remote 已存在，可先用 `git remote remove` 删除旧 remote。

### 0.5 验证环境

```bash
# 验证 upstream 和 fork remote 是否正确配置
git remote -v
```

**通过标准**：upstream 指向 `Ascend/model-agent.git`，fork 指向个人 fork 仓库。

---

## 1. 创建功能分支

0. 切换到 upstream/master 基础
1. 创建功能分支
2. 验证分支

```bash
git checkout -b feature/<标识>-npu-skill upstream/master
```

**输入/输出定义**：
- 输入：功能标识名
- 输出：基于 upstream/master 的新本地分支
- 异常：如果分支已存在，请使用 `git branch -D` 删除或换用其他名称。

### 1.1 验证分支

```bash
# 验证当前分支是否正确
git branch --show-current
```

**通过标准**：输出应为 `feature/<标识>-npu-skill`。

---

## 2. 添加文件并提交

0. 添加 skill 文件到暂存区
1. 创建 commit
2. 验证 commit 身份
3. 修正提交（如需要）

```bash
# 添加 skill 文件
git add skills/<category>/<skill-name>/

# 提交（author 和 committer 自动使用 git config 中的身份）
git commit -m "Add <skill-name> deployment skill"

# 如果提交后发现身份不对，可修正最后一次提交
git commit --amend --reset-author --no-edit
```

**输入/输出定义**：
- 输入：skill 文件路径、commit message
- 输出：本地 commit
- 异常：如果 `git commit` 后身份不对，请使用 `--amend --reset-author` 修正。

### 2.1 验证 commit 信息

```bash
# 验证 author 和 committer 邮箱
git log -1 --pretty=fuller
```

**通过标准**：author 和 committer 邮箱均为 `13290192271@163.com`。

---

## 3. 推送到 Fork

0. 推送到 fork 远程
1. 验证推送结果

```bash
git push fork feature/<标识>-npu-skill
```

**输入/输出定义**：
- 输入：本地分支名
- 输出：fork 远程上的同名分支
- 异常：如果 push 失败（403），请检查 token 是否有效或 remote URL 是否正确。

### 3.1 验证推送

```bash
# 验证远程分支是否存在
git ls-remote fork feature/<标识>-npu-skill
```

**通过标准**：返回远程分支的 commit hash，与本地一致。

---

## 4. 创建跨仓库 MR

0. 准备 MR 参数
1. 调用 GitCode API 创建 MR
2. 验证 MR 创建结果

```bash
curl --location "https://api.gitcode.com/api/v4/projects/9850257/merge_requests" \
  --header "private-token: ${ATOMGIT_USER_TOKEN}" \
  --data '{
    "source_branch": "feature/<标识>-npu-skill",
    "target_branch": "master",
    "target_project_id": 9646177,
    "title": "Add <skill-name> deployment skill",
    "description": "Skill 用途说明"
  }'
```

### 4.1 验证 MR

```bash
# 验证 MR 创建结果（使用返回的 iid）
curl --location "https://api.gitcode.com/api/v4/projects/9646177/merge_requests/<iid>" \
  --header "private-token: ${ATOMGIT_USER_TOKEN}"
```

**通过标准**：返回 state 为 "opened"。

**输入/输出定义**：
- 输入：源分支名、目标分支名、MR 标题和描述
- 输出：新创建的 MR 链接
- 异常：如果 API 返回 401，请检查 `ATOMGIT_USER_TOKEN` 是否已导出；如果返回 422，请确认分支名和项目 ID 是否正确。

---

## 5. CLA 问题修复

如果 MR 出现红色 `ascend-cla/no` 标签，按以下步骤处理：

0. 确认 CLA 签署状态
1. 检查 commit 邮箱
2. 修正 commit 身份
3. 强制推送到 Fork

### 5.1 确认 CLA 签署状态

提交 PR 后访问评论区提供的 CLA 签署地址：
- 个人贡献者 → 签署个人 CLA
- 企业贡献者（已签企业 CLA）→ 法人贡献者登记

### 5.2 检查 commit 邮箱

```bash
git log --pretty=fuller
```

CLA 检查以 **committer 邮箱** 为准。

### 5.3 修正 commit 身份

**场景 A：邮箱不一致**

```bash
# 修正最后一次提交（可选 --no-edit 保留原提交消息）
git commit --amend --author="gcw_C8PI9e90 <13290192271@163.com>" --no-edit

# 或者重置 author 和 committer 到当前 git config
git -c user.name="gcw_C8PI9e90" -c user.email="13290192271@163.com" commit --amend --reset-author --no-edit
```

**场景 B：需要修正多个提交**

```bash
git rebase -i <commit_id>~n
# 将 pick 改为 edit，逐一执行：
git commit --amend --author="gcw_C8PI9e90 <13290192271@163.com>" --no-edit
git rebase --continue
```

### 5.4 强制推送到 Fork

```bash
git push fork feature/<标识>-npu-skill --force
```

推送后 MR 会自动更新，CLA 标签自动重新检查。注意：force push 可能被保护分支规则阻止。

---

## 6. 异常边界与 Fallback 策略

| 异常场景 | 触发条件 | Fallback 策略 | 恢复操作 |
|---|---|---|---|
| CLA 检查失败 | committer 邮箱与 CLA 签署邮箱不一致 | 暂停，修正 commit 身份 | `git commit --amend --reset-author` 后 force push |
| Push 失败 403 | Token 无效或权限不足 | 失败，需用户确认 | 检查 `ATOMGIT_USER_TOKEN` 是否有效 |
| MR 创建 401 | Token 未提供或过期 | 失败，需用户确认 | 导出 `ATOMGIT_USER_TOKEN` 环境变量 |
| MR 创建 422 | 分支名或项目 ID 错误 | 失败，需用户确认 | 检查 source_branch 和 target_project_id |
| 远程已存在 | `git remote add` 重复执行 | 可忽略或移除旧 remote | `git remote remove <remote-name>` 后重新添加 |
| Force push 被阻止 | 分支受保护 | 失败，需用户确认 | 检查 fork 仓库的分支保护设置 |
| Rebase 冲突 | 多个 commit 修正时冲突 | 暂停，手动解决 | `git rebase --continue` 或 `--abort` |

---

## 7. 检查点与验收确认

完成以下检查清单即为提交成功。每步完成后请**用户确认**再继续下一步：

| # | 检查项 | 验证方法 | 通过标准 | 操作说明 |
|---|--------|---------|---------|---------|
| 1 | Git 身份配置 | `git config user.email` | 与 CLA 签署邮箱一致（13290192271@163.com） | 用户确认邮箱正确 |
| 2 | Commit 推送 | `git push fork` | 推送到 fork 分支成功 | 用户确认推送无错误 |
| 3 | MR 创建 | GitCode API 返回 | MR 创建成功且 URL 有效 | 用户确认 MR 链接可访问 |
| 4 | CLA 标签 | MR 页面标签状态 | 显示 `ascend-cla/yes` 绿色标签 | 用户确认 CLA 通过 |
| 5 | MR 完整性 | MR 标题和描述 | 标题清晰描述新增内容，说明完整 | 用户确认 MR 信息完善 |

---

## 8. 资源速查表

| 资源类型 | 路径 / 说明 |
|----------|-------------|
| 上游仓库 | `https://gitcode.com/Ascend/model-agent.git` |
| Fork 仓库 | `https://gitcode.com/gcw_C8PI9e90/model-agent_6443.git` |
| 上游项目 ID | `9646177` |
| Fork 项目 ID | `9850257` |
| MR API | `https://api.gitcode.com/api/v4/projects/9850257/merge_requests` |
| Token 变量 | `ATOMGIT_USER_TOKEN` |
| CLA 签署邮箱 | `13290192271@163.com` |
| 分支规范 | `feature/{标识}-npu-skill` |
| 分支模板 | `templates/branch-naming.md`（参考） |
| MR 模板 | `templates/mr-description.md`（参考） |
| Commit 模板 | `templates/commit-message.md`（参考） |
| Skill 目录模板 | `references/skill-directory.md`（参考） |
| 评测脚本参考 | `references/eval-scripts.md`（参考） |
| 模板目录 | `templates/`（Skill 标准化模板） |
| 参考文档 | `references/`（开发规范参考） |

---

## 关键要点

1. **跨仓库 MR**：通过 `target_project_id: 9646177` 让 fork 分支指向上游仓库发起合并请求
2. **认证方式**：`private-token` header 配合 `auth:${TOKEN}@` URL 认证
3. **commit 身份**：author/committer 邮箱必须与 CLA 签署邮箱一致（`13290192271@163.com`），否则会触发 `ascend-cla/no`
4. **CLA 检查**：以 committer 邮箱为准，修正后强制推送到 fork 分支即可自动重检
5. **分支规范**：使用 `feature/{标识}-npu-skill` 格式
