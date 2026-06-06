---
name: git-commit-author-fix
description: >
  修复 GitCode / GitLab Merge Request 中提交作者信息缺失、邮箱不正确或未签署 CLA 的问题。
  适用于用户说"PR/MR 没有签署 CLA"、"提交邮箱不对"、"需要重新提交"、
  "作者信息是 example.com"等场景时触发。
metadata:
  short-description: GitCode MR 提交作者信息修复与重新推送
  category: Documentation
  tags: [git, gitcode, cla, commit, author, mr, pr, force-push]
---

# GitCode MR 提交作者信息修复 Skill

本 Skill 提供在 GitCode（AtomGit）平台上修复 Merge Request 提交作者信息、
邮箱地址及 CLA 签署状态的标准操作流程。适用于 Ascend/model-agent 仓库及类似项目。

### 适用场景

在 Ascend/model-agent 仓库的 MR 流程中，当用户使用 `adapt-agent` 工具自动提交 NPU 模型 Skill 时，
可能因 git 配置缺失导致作者邮箱为 `adapt-agent@example.com`，触发 GitCode CLA 检查失败。
本 Skill 专门用于修正此类问题，使 MR 顺利通过合规检查。

## 触发场景

- 用户反馈"这个 PR 没有签署 CLA"
- 用户反馈"提交时没有邮箱信息"
- 提交记录中作者邮箱为 `example.com` 等占位符
- 需要重新提交 MR/PR 以修正作者信息
- Git Hooks 检查失败且提示"Author identity not verified"

## 前置确认

### 信息收集

开始修复前，必须向用户确认（或从环境变量/历史记录中获取）以下信息：

| 信息项 | 说明 | 示例 |
|--------|------|------|
| 主仓库地址 | 目标项目仓库 URL | `https://gitcode.com/Ascend/model-agent.git` |
| Fork 仓库地址 | 用户个人 fork 的 URL | `https://gitcode.com/<username>/model-agent_6443.git` |
| 源分支名 | MR 的源分支（通常不是 `master`） | `feat/granite-4.1-npu-skill` |
| 正确用户名 | git `user.name` | `gcw_C8PI9e90` |
| 正确邮箱 | git `user.email` | `13290192271@163.com` |
| GitCode Token | 用于推送，可从环境变量读取 | `ATOMGIT_USER_TOKEN` |

### 前置检查

1. [确认] 用户是否已提供正确的用户名和邮箱
2. [确认] 用户是否已提供 fork 仓库地址
3. [确认] 环境变量 `ATOMGIT_USER_TOKEN` 或等价 Token 是否可用
4. [确认] 本地已安装 git（版本 >= 2.0）
5. [确认] 确认 MR 页面显示的"未签署 CLA"状态是否因作者信息导致

若前置检查项不满足，先收集缺失信息再继续。

## 工作流程

### 流程概览

```
1. 获取仓库 → 2. 添加 Fork Remote → 3. 定位问题提交
→ 用户确认：提交信息确认有误
→ 4. 修复提交作者（amend / rebase / filter-branch）
→ 用户确认：修复后验证
→ 5. 验证修改结果
→ 6. 强制推送到 Fork
→ 7. 提醒用户签署 CLA
→ 8. 验收确认
```

按以下步骤顺序执行。**每步完成后检查确认标准**，不通过则排查修复。

---

## 1. 获取仓库

若本地没有主仓库，先 clone：

```bash
git clone https://gitcode.com/<owner>/<repo>.git /path/to/repo
cd /path/to/repo
```

### 错误处理

- 若 clone 超时：检查网络连接，尝试使用 `GIT_SSL_NO_VERIFY=1` 跳过 SSL 验证
- 若已存在本地仓库：跳过 clone，直接进入下一步
- 若 `remote: repository not found`：确认仓库 URL 正确且有访问权限

### Checkpoint：仓库获取确认

- [确认] `git status` 在本地仓库目录中可正常执行
- [确认] `git remote -v` 显示主仓库 remote

---

## 2. 添加 Fork Remote 并获取分支

```bash
git remote add fork https://gitcode.com/<username>/<fork-repo>.git
git fetch fork <source-branch>
```

### 错误处理

- 若 remote 已存在：先 `git remote remove fork` 再重新添加
- 若 fetch 失败：确认 fork 地址和分支名正确，确认用户对该 fork 有推送权限
- 若 `could not read Username`：后续推送时需要使用 Token 认证
- 若分支名不确定：用 `git branch -r | grep fork` 列出所有 fork 远程分支

### Checkpoint：Fork Remote 确认

- [确认] `git remote -v | grep fork` 显示 fork 地址
- [确认] `git branch -r | grep fork` 显示目标分支

---

## 3. 定位问题提交

检查 fork 分支相比主仓库多出的提交，确认作者信息：

```bash
git log --format="%h %an <%ae> %s" origin/master..fork/<source-branch>
```

**常见异常**：
- 作者名为 `Adapt Agent`
- 邮箱为 `adapt-agent@example.com`
- 用户名/邮箱与用户提供的正确信息不符

### 错误处理

- 若 `origin/master..fork/<source-branch>` 为空：检查源分支是否基于 master，使用 `git merge-base origin/master fork/<source-branch>` 确认共同祖先
- 若输出太多提交：使用 `--oneline` 缩小输出，或加上 `-5` 限制条数
- 若 fork 分支包含非预期的合并提交：确认只需要修改本 MR 相关的提交

### Checkpoint：提交定位确认

- [确认] 已确认需要修复的提交列表
- [确认] 记录下旧的错误作者信息供后续验证
- [确认] 与用户确认这些提交确实需要修复

---

## 4. 修复提交作者信息

### 4.1 单提交场景（最常见）

若仅有一个提交需要修复：

```bash
git checkout -b <source-branch> fork/<source-branch>
git commit --amend --author="<正确用户名> <正确邮箱>" --no-edit
```

### 4.2 多提交场景

若有多个提交需要统一修改作者，方案一：`rebase -i`

```bash
git checkout -b <source-branch> fork/<source-branch>
git rebase -i origin/master
# 在交互式 rebase 中，将需要修改的提交标记为 edit
# 对每个 edit 的提交执行：
#   git commit --amend --author="<正确用户名> <正确邮箱>" --no-edit
#   git rebase --continue
```

方案二：`git filter-branch`（适用于大量提交）：

```bash
git filter-branch --env-filter '
    export GIT_AUTHOR_NAME="<正确用户名>"
    export GIT_AUTHOR_EMAIL="<正确邮箱>"
    export GIT_COMMITTER_NAME="<正确用户名>"
    export GIT_COMMITTER_EMAIL="<正确邮箱>"
' origin/master..HEAD
```

### 4.3 GitHub CLI 替代方案（若 gh 可用）

```bash
gh pr checkout <pr-number>
git commit --amend --author="<正确用户名> <正确邮箱>" --no-edit
```

### 错误处理

- 若 `--amend` 失败说 nothing to commit：确认已 checkout 到正确分支
- 若 rebase 冲突：解决冲突后 `git add <file>` 再 `git rebase --continue`
- 若 `filter-branch` 失败：检查是否在裸仓库中运行，或使用 `git filter-repo` 替代
- 若误操作：使用 `git reflog` 找回原分支状态

### Checkpoint：修复确认

- [确认] `git log --format="%h %an <%ae> %s" -1` 显示作者信息已更新

---

## 5. 验证修改结果

```bash
git log --format="%h %an <%ae> %s" -1
```

确认输出中的作者名和邮箱已更新为正确值。

同时验证与主仓库的差异：

```bash
git log --format="%h %an <%ae> %s" origin/master..HEAD
```

- 确认所有提交的作者信息均已修正
- 确认提交内容未意外变更

### Checkpoint：验证确认

- [确认] 作者名和邮箱与用户提供的正确信息一致
- [确认] 提交内容完整未丢失

---

## 6. 强制推送到 Fork

使用 Token 推送（避免交互式密码输入）：

```bash
git push "https://<username>:${ATOMGIT_USER_TOKEN}@gitcode.com/<username>/<fork-repo>.git" <source-branch> --force
```

### 6.1 --force-with-lease 降级

若 `--force-with-lease` 因 stale info 失败，改用 `--force`（先 `fetch` 确认远程无他人新提交）：

```bash
git fetch fork <source-branch>
git push "https://<username>:${ATOMGIT_USER_TOKEN}@gitcode.com/<username>/<fork-repo>.git" <source-branch> --force
```

### 6.2 推送验证

**推送成功后**：
- 远程会返回 `[forced update]`
- MR 页面会自动更新为新的 commit hash
- Git Hooks 检查通常显示 `[PASSED]`

### 错误处理

- 若 `could not read Username`：确认 Token 已设置到环境变量且格式正确
- 若 `remote: Git Hooks Checking [FAILED]`：检查 hooks 错误详情，确认 CLA/作者信息已通过
- 若 Token 过期：重新生成 Token 或申请更新
- 若 `403 Forbidden`：确认 Token 有 `write_repository` 权限

### Checkpoint：推送确认

- [确认] 终端显示 `[forced update]` 且无错误
- [确认] MR 页面 commit hash 已更新

---

## 7. 提醒用户签署 CLA

**CLA 签署无法通过 git 命令完成**。推送后必须提醒用户：

> 请前往 MR 页面手动点击"签署 CLA"按钮，平台上才能通过合规检查。

### 附加说明

- CLA 签署状态由 GitCode 平台管理，与提交作者信息无关
- 修正作者信息仅解决 `Author identity` 相关的 hooks 检查
- 若推送后 CLA 仍显示未签署：确认用户是否已签署过 CLA

---

## 8. 验收确认

### 检查清单

完成以下检查清单即为修复成功：

1. [ ] 已确认正确的用户名、邮箱、fork 地址和源分支名
2. [ ] `git log` 显示作者信息已更新为正确值
3. [ ] 成功推送到 fork 并看到 `[forced update]`
4. [ ] 远程 Git Hooks 检查通过
5. [ ] 已提醒用户在 MR 页面手动签署 CLA

### 回退策略

若修复过程中出现问题：
1. 使用 `git reflog` 找回修改前的提交状态
2. 重新 checkout 到 fork 的原始分支状态：`git checkout -b recovery fork/<source-branch>`
3. 重试修复步骤

### 最终输出

修复成功的交付物：
1. 修正后的提交记录（推送到 fork）
2. 更新后的 MR（自动同步）
3. 给用户的 CLA 签署提醒

---

## 参考脚本

| 资源 | 用途 | 路径 |
|------|------|------|
| 修复脚本 | 单次提交自动化修复 | `scripts/fix-author.sh` |
| 测试用例 | MR 作者修复验证 | `test-prompts.json` |
| 参考文档 | git commit 修改参考 | `references/git-amend-guide.md` |

### 验证数据

- 修复前后的提交对比结果可记录到 `results.tsv` 用于审计
- 多 MR 修复的测评数据可汇总到 `eval.json` 格式的报告

### 参考脚本

修复脚本位于 `scripts/fix-author.sh`，可实现单次提交的自动化修复：

```bash
# scripts/fix-author.sh 用法示例
# ./fix-author.sh <fork-url> <branch> <name> <email>
```

### 测试用例

测试用例文件位于 `test-prompts.json`，包含 MR 作者修复的验证用例，覆盖单提交修复、多提交 rebase、filter-branch 批量修复和推送验证等场景。

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `could not read Username` | 未提供 Token | 使用 `https://username:token@...` 格式推送 |
| `stale info` / force-with-lease 失败 | fetch 后远程又有变化 | 重新 fetch，确认安全后用 `--force` |
| 分支名不是 `master` | MR 源分支可能是 feature 分支 | 用 `git branch -r \| grep fork` 列出所有远程分支 |
| fork/master 与 origin/master 无差异 | MR 源分支在其他分支上 | 检查 `fork/*` 所有分支 |
| 推送后 MR 未更新 | 源分支与 push 的分支不匹配 | 确认推送的分支名与 MR 上的源分支一致 |
| rebase 冲突 | 两个分支有相同文件的不同修改 | 解决冲突后 `git rebase --continue` |
| filter-branch 后 refs 混乱 | filter-branch 修改了提交链 | 检查 `.git/refs/original/` 备份，或用 `git reset --hard ORIG_HEAD` 回退 |
| Git Hooks 仍报错 | CLA 未签署或作者信息仍有问题 | 检查 hooks 具体错误信息，确认作者信息已完全修正 |
