# Git 提交作者修改参考

## 常用命令

- 修改最近一次提交的作者：`git commit --amend --author="Name <email>" --no-edit`
- 使用 filter-branch 批量修改：`git filter-branch --env-filter 'export GIT_AUTHOR_NAME="..."; export GIT_AUTHOR_EMAIL="..."'`
- 查看提交作者：`git log --format="%h %an <%ae> %s"`

## 注意事项

1. `--amend` 和 `rebase` 会改写提交历史，需要 `--force` 推送
2. 确保使用正确的邮箱地址以避免 CLA 检查失败
3. CLA 签署需要用户手动在 MR 页面完成
