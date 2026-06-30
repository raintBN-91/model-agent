# [内测挑战] Adapt-Agent system prompt 渐进式引用降低 Token 消耗

> 赛道：性能优化实践 ｜ 预估积分：200（PR 合并后）
> 关联 Skill: `skills/ascend/optimization/adapt-agent/`
> ⚠️ 性能数据为推演值，待实测确认

## 1. 环境配置
- Agent: Model-Agent adapt-agent Skill
- 模型: GLM-5.2（Pro 套餐）
- 任务: 单次模型适配咨询（10 轮对话）

## 2. Baseline（全量注入 references）
| 轮次 | input_tokens | 耗时(s) |
| --- | --- | --- |
| 1 | 28640 | 8.2 |
| 10 | 68900 | 11.4 |
| **累计** | 412300 | 87.6 |

## 3. 优化：渐进式引用
- system prompt 仅含 SKILL.md 主体 + references 索引
- 按需读取命中的 reference，已注入不重复

## 4. 对比
| 指标 | Baseline | Optimized | 提升 |
| --- | --- | --- | --- |
| 累计 input_tokens | 412300 | 71400 | -82.7% |
| 累计耗时 (s) | 87.6 | 31.5 | -64.0% |

> ⚠️ 推演值，待实测
