# Tools 辅助工具目录

本目录存放项目维护用的辅助脚本。

## 脚本清单

| 脚本 | 用途 | 使用方式 |
|------|------|---------|
| `check_skill_refs.py` | 检查 SKILL.md 中引用的本地文件是否存在 | `python tools/check_skill_refs.py` |

## 使用建议

- 运行前请确保已安装 Python 3.8+
- 敏感信息（如 API Token）请通过环境变量传入，不要硬编码在脚本中

## 贡献规范

新增辅助脚本时请：
1. 放置到本目录下
2. 更新本 README 的脚本清单
3. 在脚本头部添加 docstring 说明用途
4. 不要硬编码敏感信息，使用环境变量

## 安全提醒

⚠️ 请勿在脚本中硬编码 API Token、密码等敏感信息。请使用环境变量：

```python
import os
token = os.environ.get('GITCODE_TOKEN')
```
