# Trading Agent

基于大语言模型的 A 股交易 Agent：让 LLM 自主查询行情、计算 alpha 因子，并据此决定买卖，在带撮合 / 滑点 / 手续费的回测环境中执行决策。

> ⚠️ 本项目仅用于教学演示，主要演示 Agentic LLM 在多轮工具调用与状态管理下的自主交易决策能力（research → 下单 → 观测 → 收尾的完整闭环）。**本项目不构成任何投资建议，也不适合实盘使用。股市有风险，投资需谨慎。**

## ⚠️ 前置数据准备

本项目依赖 **Qlib 格式的 A 股日频数据**，运行前必须先准备好，否则所有行情类工具会返回 `[错误] ...`。Qlib 官方自带的示例数据（`GetData().qlib_data`）已较旧，推荐使用持续更新的社区数据：

```bash
# 方式一（推荐，开箱即用）：从 ModelScope 下载预构建数据集（已转为实际价格并补全指数 / ETF）
pip install modelscope
modelscope download --dataset chenshaohon/trade-agent-data --local_dir ./data
unzip -d ./data/qlib_data ./data/qlib_data.zip
mv data/qlib_data/qlib_data/* data/qlib_data/ && rm -rf data/qlib_data/qlib_data/
```

方式二（从源构建）：基于按发布日期持续更新 Qlib bin release 的社区项目 [`chenditc/investment_data`](https://github.com/chenditc/investment_data) 下载个股数据，再将前复权价转回实际价、并用 AKShare 补充指数 / ETF（可参考其配套的 `setup_qlib.py` 流程）。

数据目录由 `EnvConfig.qlib_data_dir` 配置，默认 `data/qlib_data`，准备好后指向实际目录即可。此外还需一个 **OpenAI 兼容推理端点**，可以使用 vLLM-Ascend 部署微调后的 Qwen3.5 推理模型，或使用其他兼容 OpenAI 的 API，Agent 通过 `AsyncOpenAI` 客户端访问 API。

## 安装

```bash
uv sync          # 依赖：openai-agents / pyqlib / streamlit；Python ≥ 3.11
```

## 模块说明

| 模块 | 说明 |
| --- | --- |
| `schemas.py` | 领域数据类型：`EnvConfig` / `Position` / `Scenario` / `TradeContext` / `EpisodeResult` 等 |
| `data/` | Qlib 数据访问层（带超时保护的 `safe_features`）+ 技术指标表达式映射 |
| `backtest/` | `Portfolio` 撮合引擎、单日 `run_backtest_episode`、多日 `run_multi_day_backtest` |
| `agent/` | OpenAI Agents SDK 接入：`build_trade_agent` 构造挂好 4 个工具的交易 Agent |
| `app.py` | Streamlit 可视化面板，交互式运行 Agent 并查看决策 / 成交 / 净值 |

Agent 挂载 4 个工具：`get_price_data`（行情+技术指标）、`get_macro_indicators`（大盘指数）、`compute_alpha_factor`（自定义 Qlib 因子表达式）、`TakeAction`（buy / sell / stop）。

## 启动

```bash
# 在本目录（trading_agent/，即项目根目录）执行
streamlit run app.py
```

在左侧边栏填入推理端点 `base_url` / `api_key` / 模型名与 Qlib 数据目录，主面板设置交易日、股票池、初始资金后点击「运行」。

也可在代码中直接调用：

```python
# 在 trading_agent/ 根目录下运行（agent / data / backtest / schemas 均为顶层模块）
from openai import AsyncOpenAI
from agent import build_trade_agent
from backtest import run_backtest_episode
from schemas import Scenario, EnvConfig

agent = build_trade_agent()
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
scenario = Scenario(start_date="2023-06-01", universe=["SH600519", "SZ000001"])
result = await run_backtest_episode(scenario, agent, client, env_config=EnvConfig(), model="Qwen3")
```
