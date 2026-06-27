# 训练参数说明

参数主要在模型目录下 `train.py` 文件中修改。修改前先备份，修改完后请用户确认。

## 时间参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `default_test_start` | `"2026-04-10 9:30"` | 默认测试开始时间 |
| `default_test_end` | `"2026-04-10 9:30"` | 默认测试结束时间 |
| `--train_start` | `"2021-1-16 9:30"` | 训练集开始时间，9:30是启推时间 |
| `--train_end` | `"2024-12-30 9:30"` | 训练集结束时间，9:30是启推时间 |
| `--val_start` | `"2024-12-31 9:30"` | 评测集开始时间 |
| `--val_end` | `"2025-09-29 9:30"` | 评测集结束时间 |
| `--test_start` | 同 default_test_start | 测试集开始时间 |
| `--test_end` | 同 default_test_end | 测试集结束时间 |
| `--finetune_start` | `"2021-1-16 9:30"` | 微调集开始时间，不早于训练集开始 |
| `--finetune_end` | `"2024-12-30 9:30"` | 微调集结束时间，不晚于训练集结束 |

## 模型架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | model_name | 模型名称 |
| `--attention_type` | attention_type | 注意力类型 |
| `--e_layers` | `4` | Encoder层数 |
| `--patch_len` | patch_len | Patch长度 |
| `--activation` | act | 激活函数 |
| `--glb_num` | `24` | 全局参数 |
| `--n_heads` | `8` | 注意力头数 |
| `--d_model` | `512` | 模型维度 |
| `--d_ff` | `1024` | FFN维度 |
| `--dropout` | `0.0` | Dropout比率 |
| `--use_norm` | `1` | 是否使用归一化 |

## 序列长度参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seq_len` | `710` | 输入序列长度。710/96≈7.39天，710 mod 96=38，代表参考历史7天+当天9:15的数据 |
| `--pred_len` | `154` | 预测序列长度。154/96≈1.6天，154 mod 96=58，代表预测1天96点+当天9:30~23:45 |
| `--weather_len` | `216` | 气象序列长度 |

## 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--is_training` | `"0"` | 训练时改成 `"1"` |
| `--batch_size` | `32` | 批次大小 |
| `--train_epochs` | `100` | 训练轮数 |
| `--warmup_epochs` | `10` | Warmup轮数 |
| `--learning_rate` | `0.0001` | 学习率 |
| `--finetune_lr` | `1e-5` | 微调学习率 |
| `--patience` | `10` | Early stopping耐心值 |
| `--ema_start` | `0` | EMA起始轮数 |
| `--weight_decay` | `0.5` | 权重衰减 |
| `--itr` | `1` | 运行次数 |

## 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--root_path` | `"./dataset"` | 负荷数据路径（相对train.py） |
| `--data_path` | `"load.csv"` | 负荷数据文件名（相对train.py） |
| `--weather_csv_path` | `"weather_true.csv"` | 气象数据文件名 |
| `--use_weather` | `"1"` | 是否使用气象数据（1或0） |
| `--use_time` | `"1"` | 是否使用时间特征 |
| `--use_holiday` | `"1"` | 是否使用节假日特征 |
| `--weather_column_regex` | `r"(_t2m\|_q)"` | 从气象文件提取的气象要素正则 |
| `--weather_dim` | `32` | 气象特征维度 = 气象要素数 × 城市数 |
| `--use_pangu_weather` | `"0"` | 是否使用盘古气象 |
| `--pangu_root` | `"../pangu_weather"` | 盘古气象路径 |
| `--pangu_issue_backoff_points` | `"1"` | 盘古气象回退点数 |

## 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu` | `"0"` | 使用的NPU核编号，用 `npu-smi info` 查看空闲核 |
| `--pred_only` | `"0"` | 仅预测模式 |
| `--inverse` | (flag) | 是否反归一化 |
| `--features` | `"MS"` | 特征类型 |
| `--data` | `"energy"` | 数据集名称 |
| `--task_name` | `"long_term_forecast"` | 任务名称 |
| `--des` | f"{des}{run_index}" | 实验描述 |
