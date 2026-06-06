# RapidOCR ONNX → NPU 转换流程

## 完整 pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 下载原始权重 (ModelScope / AtomGit)                │
│  rapidocr 包会自动下载 ONNX 到 site-packages/rapidocr/models/ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Constant → Initializer 修复                         │
│  原因: Paddle2ONNX 将权重放在 Constant 节点                  │
│  工具: scripts/fix_constant_nodes.py                         │
│  输出: *_const.onnx                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: ONNX → PyTorch (onnx2torch)                        │
│  输入: *_const.onnx                                          │
│  注意: rec 模型需先打 BatchNorm patch (见 references/)        │
│  输出: *_npu.pt                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: NPU 推理 (torch_npu)                               │
│  device = torch.device("npu:0")                             │
│  model = torch.load("*_npu.pt").eval().to(device)           │
│  output = model(input_tensor)                               │
└─────────────────────────────────────────────────────────────┘
```

## 各阶段踩坑记录

| 阶段 | 问题 | 现象 | 解决方案 |
|------|------|------|----------|
| Step 2 | Constant 节点 | `KeyError: 'conv1_weights'` | 提取为 initializer |
| Step 3 | BatchNorm rank | `spatial_rank == -2` | patch fallback to 2D |
| Step 3 | 动态 shape | ATC 无法转换 | 改用 torch_npu 在线推理 |
| Step 4 | 首次推理慢 | 时延异常高 | warmup 预热 |
| Step 4 | 输入尺寸 | `RuntimeError: size mismatch` | 严格 32 对齐 |

## 性能对比

- CPU (Intel Xeon): ~2600 ms/图
- NPU (Ascend910): ~80 ms/图
- **加速比: ~30x**
