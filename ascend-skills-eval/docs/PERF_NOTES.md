# ascend-skills-eval 批量评估性能优化

本变更将 `/evaluate-repos` 的串行 clone+评估改为线程池并发执行，并为 `render_card` 的 node 子进程增加超时保护与进程树清理。

## 改动

- `ascend-skills-eval/web-service/app/main.py`
  - `evaluate_repos` 用 `ThreadPoolExecutor(max_workers=4)` 并发，总超时 180s，未完成任务结果丢弃并记入 `failed`
  - 新增 `_handle` 内置 cache（key 为 `(repo_url, branch, skill_path)` 三元组，带 `threading.Lock`），避免同仓库+同分支重复 clone
  - `render_card` 改用 `subprocess.Popen` + 进程组启动，超时 30s 后用 `taskkill /T`（Windows）或 `os.killpg`（Unix）杀整个进程树，避免 puppeteer/chromium 孤儿进程
  - `_has_frontmatter` 归一化 CRLF/LF，兼容 Windows 换行的 SKILL.md
  - `_pick_repo_file` 的 `rglob` 结果加 `sorted()`，保证跨平台确定性
  - `git clone` 命令追加 `--` 分隔符，防御性编程
- `ascend-skills-eval/web-service/requirements.txt` — 声明 Python >= 3.10

## 性能对比

10 个相同仓库（本地 mock git server）。下表为**理论预期值**（基于 4 worker 并发的 IO 等待模型推算），实际加速比受网络/磁盘影响，真实数据请在目标环境用 `tests/test_eval.py` 同样的批量请求实测后填入。

| 指标 | 优化前（串行） | 优化后（4 并发，理论） | 备注 |
|------|--------|--------|------|
| 10 仓库总耗时 | ~25s 估算 | ~7s 估算 | 待实测，以目标环境为准 |
| 单仓库 clone 超时 | 120s | 120s（未改） | 保持原值，由总超时 180s 兜底 |
| render_card 无超时 | 可永久挂死 | 30s → 504 + 进程树清理 | 已实测（见 test_eval.py） |

> ⚠️ 说明：本 PR 的测试仅验证"并发改造不破坏单条评估逻辑"与"frontmatter 归一化"，**未对真实加速比做基准测试**。上表数字为推算，提交前请在真实环境跑一次并替换为实测值。

## 验收
- `python -m pytest ascend-skills-eval/tests/test_eval.py` 通过（当前 17 用例）
- 批量总超时 180s，超时后未完成任务进 `failed` 且标注 `结果已丢弃`
- render_card 超时返回 504，子进程树被清理
- cache 使用 per-key 锁，同 key 串行化防 thundering herd，不同 key 真正并发
