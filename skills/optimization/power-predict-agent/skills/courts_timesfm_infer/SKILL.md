---
skill_name: courts_forecasting_optimizer
display_name: 台区负荷预测自动调参优化
description: |
  自动探索TimesFM台区负荷预测最优参数。支持用户给出大致方向后自行探索，
  自适应聚焦最优区间，8卡NPU并行，用户随时可喊停。

version: 3.0.0
author: user
tags: [timesfm, forecasting, optimize, ascend, npu, auto-tune, adaptive, self-explore]

triggers:
  - regex: "(优化|调参|搜索|找最优|自动跑|跑.*实验).*台区.*预测"
  - regex: "courts.*optimiz|auto.*tune|self.*explore"
  - regex: "帮我跑|开始.*调参|你自行|你发挥|你决定"
  - regex: "无气象|没有天气|without.*weather|no weather"
  - intent: "forecasting_auto_optimization"

environment:
  python: ">=3.11"
  venv: "/root/timesfm_env/timesfm/bin/python"
  working_dir: "/data/power_pred_all/models/timesFM/courts_load_forecasting_skill"

resources:
  npu:
    type: ascend
    count: 8

parameters:
  - name: user_intent
    type: string
    required: true
  
  - name: base_config_dir
    type: path
    default: "/data/power_pred_all/models/timesFM/courts_load_forecasting_skill/config"
  
  - name: true_csv
    type: path
    default: "/data/power_pred_all/models/timesFM/courts_load_forecasting_skill/data/usr/all_usrs_high_quality_interpolate.csv"
  
  - name: top_n
    type: integer
    default: 3
  
  - name: area_top_n
    type: integer
    default: 5

outputs:
  - name: final_report
    type: markdown
  
  - name: running_log
    type: stream

stop_conditions:
  - user_command: ["停", "停止", "stop", "够了", "结束", "停!", "别跑了", "pause"]
  - exception: "any_unhandled_exception"
  - npu_error: "aclError|rtError|memory_error"
  - all_npu_faulty: true
---

# 台区负荷预测自动调参优化 Skill

## 核心能力

| 能力 | 说明 |
|-----|------|
| **自行探索** | 用户给大致方向，Agent自动设计实验计划 |
| **自适应采样** | 3轮聚焦：粗筛→细搜→精搜 |
| **8卡并行** | 自动调度，故障卡自动隔离 |
| **随时停止** | 用户喊"停"立即汇总当前最优 |
| **无气象优化** | 自动处理weather_enabled=false场景 |

## 用户使用方式

### 方式1：给方向，Agent自行探索（推荐）