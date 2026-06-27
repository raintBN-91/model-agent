# 阶段1.8：依赖安装收敛（训练前置）

## 输入信息

- 项目 requirements 文件或等价依赖定义
- 环境矩阵与版本约束
- 可能需要的补丁信息

## 必做动作

- 默认优先执行 `pip install -r requirements.txt`（或项目等价 requirements 文件）
- 若 requirements 安装失败，先区分失败包类型：
  - 纯 Python 包
  - 编译扩展包
  - NPU 相关包
- 对失败包执行源码安装回退：
  - 优先 `pip wheel` 构建本地 wheel
  - 离线 `pip install` 本地 wheel
- 若存在 NPU 或算子补丁，先应用补丁再构建 wheel
- 必须保留安装证据：
  - requirements 安装日志
  - 失败包与失败原因
  - 回退命令与结果
  - 产物路径和 SHA256
- 安装后必须执行最小导入验证：
  - `torch`、`torch_npu`、关键扩展包导入通过
  - 至少一个最小 NPU 算子 smoke test 通过

## 禁止动作

- requirements 安装失败后直接跳过依赖问题进入训练
- 未完成源码安装回退就宣称依赖已收敛
- 未保存日志、wheel 路径和 SHA256 就关闭安装问题
- 未做导入验证和最小 NPU smoke test 就进入训练链路

## 完成标准

- 依赖安装完成或回退路径执行完成
- 安装日志、失败原因、回退动作、产物路径、SHA256 已固化
- `torch`、`torch_npu`、关键扩展包导入验证通过
- 最小 NPU 算子 smoke test 通过

## 失败处理

- 若标准安装失败，转入源码安装回退
- 若补丁缺失导致构建失败，先记录缺失点并补丁后重试
- 若最小导入验证或 smoke test 失败，停止进入阶段2并输出失败归因

## 下一阶段

- 依赖验证通过后，进入阶段2

## 输出结果

- 依赖安装日志
- 失败包与回退记录
- wheel 产物路径与 SHA256
- 导入验证与 smoke test 记录
