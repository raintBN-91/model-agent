# 负荷预测模型信息

> `{date}` 为**预报时间第一天**日期占位符，格式 `YYYYMMDD`。
> 例如：预报 2026-05-15 的负荷 → `{date}` = `20260515`

## 云羲模型
- **推理脚本**：`/data/power_pred_all/models/fuhe-huabei-v7/infer_v1.py`
- **原始文件名**：`/data/power_pred_all/models/fuhe-huabei-v7/pred_Energy0.csv`
- **目标文件名**：`/data/infer_result/{date}/pred_Energy0_{date}.csv`
- **推理参数**：
  - `--pred_start_time`：起报时间，格式 `YYYYMMDDHHMM`，例如 `202605140930`
  - `--pred_days`：预报天数，默认 1
- **调用示例**：
  ```bash
  cd /data/power_pred_all/models/fuhe-huabei-v7 && \
  python infer_v1.py \
    --pred_start_time 202605140930 \
    --pred_days 1
  ```
  > 注意：必须在脚本所在目录下执行（脚本内部使用相对路径），`--pred_start_time` 替换为实际起报时间。

## MoE模型
- **推理脚本**：`/data/power_pred_all/models/power_moe_jjt_v3_0319/infer_moe_future_jjt_v1.py`
- **原始文件名**：`/data/power_pred_all/models/power_moe_jjt_v3_0319/preds_jjt_hmoe_{date}.csv`
- **目标文件名**：`/data/infer_result/{date}/preds_jjt_hmoe_{date}.csv`
- **推理参数**：
  - `--pred_start_time`：起报时间，格式 `YYYYMMDDHHMM`，例如 `202605140930`
  - `--pred_days`：预报天数，默认 1
- **调用示例**：
  ```bash
  cd /data/power_pred_all/models/power_moe_jjt_v3_0319 && \
  python infer_moe_future_jjt_v1.py \
    --pred_start_time 202605140930 \
    --pred_days 1
  ```
  > 注意：必须在脚本所在目录下执行（脚本内部使用相对路径），`--pred_start_time` 替换为实际起报时间。

## 天权模型
- **推理脚本**：`/data/power_pred_all/models/TQ_Foundation_model_39/infer_jjt_v1.py`
- **原始文件名**：`/data/power_pred_all/models/TQ_Foundation_model_39/xh_preds_{date}.csv`
- **目标文件名**：`/data/infer_result/{date}/xh_preds_{date}.csv`
- **推理参数**：
  - `--pred_start_time`：起报时间，格式 `YYYYMMDDHHMM`，例如 `202605140930`
  - `--pred_days`：预报天数，默认 1
- **调用示例**：
  ```bash
  cd /data/power_pred_all/models/TQ_Foundation_model_39 && \
  python infer_jjt_v1.py \
    --pred_start_time 202605140930 \
    --pred_days 1
  ```
  > 注意：必须在脚本所在目录下执行（脚本内部使用相对路径），`--pred_start_time` 替换为实际起报时间。
