# ascend-delivery-pipeline

昇腾模型适配结果自动化交付 Skill。

## 适用场景

模型适配/调优完成后，需要将成果提交到 GitCode 归档。此过程常遇到 git 陷阱：

- `fatal: detected dubious ownership in repository`
- `*** Please tell me who you are`
- `remote rejected: main vs master branch conflict`
- 目标仓库不存在，需要手动创建

本 Skill 自动完成打包 → 修复 → 创建仓库 → 推送的全流程。

## 触发方式

- "提交 gitcode"
- "交付结果"
- "上传适配报告"
- "git 报错"
- "push 到 gitcode"
- "打包交付"

## 核心能力

| 能力 | 说明 |
|------|------|
| 自动扫描 | 识别 README、推理脚本、验证脚本、日志、报告、配置文件 |
| 一键打包 | 按类别整理到 `deliverables_YYYYMMDD_HHMMSS/` 目录，生成 MANIFEST |
| Git 陷阱修复 | 自动处理 safe.directory、user.identity、main/master 分支映射 |
| 仓库自动创建 | 通过 GitCode API 检测并创建缺失的远程仓库 |
| 安全预检 | 大文件警告（>100MB）、敏感文件检测（.env、.key 等） |
| Frontmatter 校验 | 检查 README 是否含 NPU/Ascend 标签，提示补充 |
| 交付报告 | 生成结构化交付报告，记录所有修复操作和文件清单 |

## 工作流程

```
用户请求交付
    │
    ▼
扫描当前目录 deliverables
    │
    ▼
展示清单 → 用户确认/调整
    │
    ▼
打包到 deliverables_YYYYMMDD_HHMMSS/
    │
    ▼
Git 陷阱自动修复
├── safe.directory
├── user.name / user.email
└── main/master 分支统一
    │
    ▼
检测远程仓库 → 不存在则自动创建
    │
    ▼
推送交付件
    │
    ▼
Frontmatter NPU 标签校验
    │
    ▼
生成交付报告
```

## 目录说明

| 路径 | 说明 |
|------|------|
| `SKILL.md` | Claude Code Skill 定义，包含完整工作流与修复脚本 |
| `README.md` | 本文件 |
| `scripts/package.sh` | 可选的独立打包脚本，供流水线直接调用 |

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `GITCODE_TOKEN` | GitCode API Token | 从 git remote URL 自动推断 |
| `GITCODE_OWNER` | 目标仓库所属用户/组织 | 当前 git remote 推断 |
| `GITCODE_REPO` | 目标仓库名 | 当前目录名 |
| `USER` | Git commit 用户名 | 系统用户名 |
| `EMAIL` | Git commit 邮箱 | `agent@atomgit.ai` |

## 与现有生态的关系

- `gitcode-publish`：本 skill 处理完整的 deliverables 打包与 git 修复后，可调用 `gitcode-publish` 进一步美化 README frontmatter 和模型卡片标签
- `ascend-resource-scheduler`：交付前若 NPU 上仍有 vLLM serve 运行，可先调度停服
- `ascend-benchmark-runner`：交付的 benchmark_report.json / .md 由本 skill 自动纳入打包清单

## 贡献

提交 PR 至：https://gitcode.com/Ascend/model-agent
