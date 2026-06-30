# 贡献指南 - Contributing to Model-Agent

感谢您对 Model-Agent 项目的关注！本文档说明如何为项目贡献 SKILL.md 和相关资源。

## SKILL.md 标准格式

每个 SKILL.md 必须包含 YAML frontmatter：

```yaml
---
name: skill-name           # skill 名称（小写连字符）
description: "简短描述"     # 一句话说明 skill 用途
---
```

## 引用外部文件的路径规范

SKILL.md 中引用本地文件时，**必须使用相对于 SKILL.md 所在目录的相对路径**。

### 标准目录结构

每个 skill 目录建议包含以下子目录：

```
skills/<category>/<skill-name>/
├── SKILL.md              # 主文档
├── references/           # 参考资料
│   ├── *.md
│   └── *.json
├── templates/            # 模板文件
│   └── *.md
├── scripts/              # 可执行脚本
│   └── *.py
└── assets/               # 静态资源
    └── *.png
```

### 引用格式

引用 references/ 下的文件：

```markdown
详见 [参考文档](references/example.md)
```

引用 templates/ 下的文件：

```markdown
使用 [模板](templates/template.md) 生成报告
```

引用 scripts/ 下的文件：

```markdown
执行 `scripts/run.py` 进行验证
```

### 跨模块引用

如需引用其他 skill 或项目根目录的文件，使用 `../../../` 形式并明确标注：

```markdown
参考 [评测系统](../../../ascend-skills-eval/skills/skills-eval/SKILL.md)
```

### 禁止事项

- ❌ 不要引用不存在的文件
- ❌ 不要使用绝对路径
- ❌ 不要硬编码敏感信息（Token、密码）

## PR 提交前自检清单

提交 PR 前请确认：

- [ ] SKILL.md 包含正确的 YAML frontmatter
- [ ] 所有引用的本地文件都实际存在
- [ ] 引用路径使用规范的相对路径格式
- [ ] 无硬编码的敏感信息
- [ ] 已运行 `python tools/check_skill_refs.py` 校验断链

## CI 校验

项目 CI 会自动运行 `tools/check_skill_refs.py` 检查所有 SKILL.md 的引用有效性，断链将导致 CI 失败。

## 添加新的 references/、templates/ 文件

1. 在 skill 目录下创建对应子目录（references/、templates/ 等）
2. 添加文件到子目录
3. 在 SKILL.md 中引用该文件
4. 运行 `python tools/check_skill_refs.py` 验证

## 联系与支持

- 提交 Issue: https://gitcode.com/Ascend/model-agent/issues
- 提交 PR: https://gitcode.com/Ascend/model-agent/pulls

感谢您的贡献！🎉
