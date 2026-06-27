# 错误签名参考

排障时优先匹配错误签名，再按映射路径执行检查与修复。

| 错误签名 | 可能根因 | 优先检查项 | 常见修复 |
| --- | --- | --- | --- |
| `HcclCommInitRootInfo` 或 `error code is 7` | 端口冲突或 HCCL 配置/映射异常 | `master-port`、可见设备、`hccn.conf` 挂载 | 调整 `--master-port`，修正挂载或配置后重试 |
| `No backend type associated with device type cpu` | 分布式路径回退到 CPU | rank 到 device 的映射逻辑、`cfg.device` 解析 | 在模型初始化前按 rank 显式绑定 NPU |
| `Unsupported data type at::kDouble` | HCCL collective 路径使用了 `float64` | 日志/统计同步张量、allreduce 张量 dtype | 将统计张量切换为 `float32` |
| `unsupported autocast device_type` | AMP 的 `device_type` 与运行时不兼容 | autocast 上下文构造逻辑 | 映射到受支持的 `device_type` 或运行时 API |
| `fatal error: 'cstdint' file not found` | CANN/ATC 头文件或环境变量未加载 | toolkit 环境变量、include 路径 | 先 source CANN 环境脚本，再重新执行 ATC |
| `Out of memory` 或 `OOM` | batch 或输入尺寸过大 | batch size、输入尺寸、AMP 状态 | 降低 batch，启用 AMP，必要时做梯度累积 |
| 多卡启动卡住 | 进程组、网络或可见设备映射异常 | `torchrun` 参数、device map、端口与 rank 环境变量 | 统一启动参数并显式设置可见设备 |
| 第三方评测器 CUDA 报错 | 外部库存在 CUDA 硬编码 | 第三方库中的 `all_gather`、device tensor 路径 | 用隔离适配层或局部 monkey patch 处理 |
| `dcmi module initialize failed. ret is -8005` 或 `npu-smi info` 在当前会话失败但本地 shell 正常 | 当前执行环境存在额外沙箱，导致 `/dev` 视图或 NPU 设备节点被屏蔽 | 当前进程是否运行在额外沙箱内、`/dev/davinci*` 是否可见、是否需要绕过沙箱执行 | 切换到真实容器/宿主机视角执行 `npu-smi` 与 `torch.npu` 探测，不要直接判定容器环境失败 |

## 升级处理规则

若无签名匹配：
- 标记为 `unknown`
- 提取首个异常附近前后 40 行日志
- 记录完整命令行与环境矩阵
- 在根因确认后新增一条签名记录
