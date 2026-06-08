#!/usr/bin/env python3
"""Generate standard adaptation documentation skeletons."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

TEMPLATES = {
    "env_matrix.md": """# 环境矩阵\n\n- 生成时间：{date}\n- 模型：{model}\n\n## 硬件信息\n- 机器：\n- Ascend SoC：\n- 卡数：\n- 单卡显存/HBM：\n\n## 运行时版本\n- 操作系统：\n- Python：\n- PyTorch：\n- torchvision：\n- torch_npu：\n- CANN：\n- 驱动：\n- 固件：\n\n## 版本匹配检查\n- torchvision 与 torch 是否匹配：\n- torch_npu 与 torch 是否匹配：\n- 判定依据：\n\n## 容器与挂载\n- 容器镜像：\n- 启动命令：\n- 设备节点：\n- 关键挂载：\n- hccn.conf 挂载：\n- 是否需要绕过沙箱执行 NPU 检查：\n\n## 真实设备验证\n- npu-smi 版本：\n- npu-smi 输出摘要：\n- torch.npu.is_available()：\n- torch.npu.device_count()：\n- current_device：\n- NPU 可见性结论：\n- 备注：\n""",
    "adaptation_checklist.md": """# 适配检查清单\n\n- 生成时间：{date}\n- 模型：{model}\n\n## 阶段状态\n- [ ] 基线已固化\n- [ ] 环境已收敛\n- [ ] 依赖安装已收敛（requirements 优先 + 失败源码回退）\n- [ ] 单卡训练 smoke 通过\n- [ ] 多卡训练 smoke 通过\n- [ ] 多卡真实 NPU 训练通过\n- [ ] 多卡 AMP 长周期训练通过\n- [ ] HCCL 与 dtype 兼容通过\n- [ ] AMP 兼容通过\n- [ ] loss 收敛证据已固化\n- [ ] 最终评估指标已固化\n- [ ] checkpoint 路径已固化\n- [ ] ONNX 导出通过\n- [ ] OM 导出通过\n- [ ] 推理一致性通过\n- [ ] 推理去重策略通过（top1 per label）\n- [ ] 推理性能统计已固化\n- [ ] 推理精度统计已固化\n- [ ] 交付报告完成\n\n## 证据链接\n- 环境命令：\n- requirements 安装日志：\n- 源码安装回退日志：\n- wheel 与 SHA256：\n- 训练日志：\n- checkpoint：\n- loss 曲线或标量文件：\n- 导出日志：\n- 导出产物清单：\n- 推理输出：\n- 去重前后统计：\n- 性能统计：\n- 精度统计：\n- 指标摘要：\n""",
    "issue_matrix.md": """# 问题矩阵\n\n- 生成时间：{date}\n- 模型：{model}\n\n| 现象 | 根因 | 优先检查项 | 修复动作 | 预防措施 | 状态 |\n| --- | --- | --- | --- | --- | --- |\n| | | | | | |\n""",
    "adaptation_report.md": """# 适配报告\n\n- 生成时间：{date}\n- 模型：{model}\n\n## 1. 任务范围\n- 任务类型：\n- 数据集：\n- 目标硬件：\n\n## 2. 环境矩阵\n- 链接：env_matrix.md\n- 环境收敛结论：\n\n## 3. 代码改动\n- 修改文件：\n- 关键兼容改动：\n\n## 4. 训练验证\n\n### 4.1 数据集\n- 数据集名称：\n- 数据根目录（真实路径）：\n- 标注文件（真实路径）：\n- 图像目录（真实路径）：\n- 标注格式（COCO/VOC/自定义）：\n- 标注字段示例：\n- 样本数：\n- train/val 关系：\n\n### 4.2 训练命令\n- 单卡命令（真实路径）：\n- 多卡命令（真实路径）：\n- 分布式启动脚本命令（真实路径）：\n- 本次实际执行命令：\n\n### 4.3 环境信息\n- 操作系统：\n- Python：\n- PyTorch：\n- torchvision：\n- torch_npu：\n- CANN：\n- 驱动：\n- 固件：\n- npu-smi：\n- torch.npu.is_available()：\n- torch.npu.device_count()：\n- 当前设备：\n- 是否绕过沙箱执行真实验证：\n\n### 4.4 项目/框架信息\n- 项目名称：\n- 训练框架：\n- 框架版本：\n- 训练入口脚本：\n- 分布式启动脚本：\n\n### 4.5 训练产物\n- 配置文件或训练脚本：\n- work_dir 或输出目录：\n- 日志路径：\n- checkpoint 路径：\n\n### 4.6 关键 loss 点\n- epoch 1：\n- epoch 10：\n- epoch 20：\n- epoch 40：\n- epoch 60：\n- epoch 80：\n- epoch 100：\n- 前 10 轮平均 loss：\n- 后 10 轮平均 loss：\n\n### 4.7 最终指标\n- 关键评估指标 1：\n- 关键评估指标 2：\n- 关键评估指标 3：\n\n### 4.8 结论\n- 是否收敛：\n- 是否完成真实训练验证：\n- 非阻塞 warning：\n\n## 5. 依赖安装记录\n- 标准安装命令：\n- 失败回退命令（源码/wheel）：\n- 安装后验证命令：\n- 关键依赖版本：\n- wheel 与 SHA：\n\n## 6. 导出验证\n- ONNX 导出结果：\n- OM 导出结果：\n- SoC 与 shape 参数：\n- runtime_soc_version：\n- atc_soc_version：\n- soc_source（set_env / npu-smi / user / default）：\n- 导出产物清单（meta/onnx/om/log）：\n\n## 7. 推理后处理与性能\n- dedup_policy：\n- raw_num_detections：\n- num_detections：\n- num_kept：\n- 测试输入集：\n- warmup 次数：\n- 有效迭代次数：\n- backbone/rpn/roi 平均耗时（ms）：\n- e2e mean/p50/p90/p99（ms）：\n- FPS（img/s）：\n- 吞吐（img/s）：\n\n## 8. 推理精度统计\n- 对齐基线（PyTorch）：\n- num_dets 差异：\n- bbox 误差（L1/Max）：\n- score 误差（MAE/Max）：\n- label 一致率：\n- mask IoU（如有）：\n- 通过阈值与结论：\n\n## 9. 问题与处置\n- 链接：issue_matrix.md\n\n## 10. 风险与后续动作\n- 当前风险：\n- 跟进计划：\n""",
}


def write_file(path: Path, content: str, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        return f"skip {path.name} (exists)"
    path.write_text(content, encoding="utf-8")
    return f"write {path.name}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold adaptation documentation files")
    parser.add_argument("--out",
                        required=True,
                        help="Output directory for generated markdown files")
    parser.add_argument("--model",
                        default="unknown-model",
                        help="Model name used in template headers")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Overwrite files if they already exist")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for filename, template in TEMPLATES.items():
        content = template.format(date=now, model=args.model)
        status = write_file(out_dir / filename, content, args.overwrite)
        print(status)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
