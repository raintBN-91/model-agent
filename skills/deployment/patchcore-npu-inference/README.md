# PatchCore — 昇腾 NPU 工业异常检测智能体

## 项目概述
基于昇腾 800I A2 NPU 构建的 PatchCore 工业异常检测智能体，在 MVTec-AD 数据集上实现端到端推理。包含特征提取、记忆库构建（Coreset 采样）、异常评分与分割全流程，精度误差 < 1%（vs GPU/CPU 基线）。

## 核心功能
- **端到端推理**：特征提取 → 记忆库构建 → 异常评分 → 分割定位
- **国产化算力支持**：适配昇腾 800I A2 NPU，使用 BaihuNN 替代 FaissNN 做近邻搜索
- **高精度**：图像级 AUROC ≥ 97.5%，像素级 AUROC ≥ 96.0%（15 类平均）
- **灵活配置**：支持单类别/全类别推理，可调节 backbone、采样率、batch size 等参数

## 快速使用

### 1. 环境依赖
```bash
# Python ≥ 3.9
# torch ≥ 2.1.0 + torch_npu（昇腾 CANN）
# torchvision, timm, faiss-cpu, scikit-learn, scikit-image
```

### 2. 数据准备
下载 MVTec-AD 数据集并按类别目录存放。

### 3. 技能调用
```bash
python /path/to/Patchcore/inference.py \
    --data_dir /data/mvtec_anomaly_detection \
    --device npu:0
```

## 技能文件
- 技能定义：`skills/patchcore-npu-inference/SKILL.md`
- 推理脚本：见 [quzhi_1981/Patchcore](https://gitcode.com/quzhi_1981/Patchcore) 仓库

## 项目信息
- **项目位置**：`models/contribution/patchcore-npu-inference`
- **项目类型**：昇腾 Model-Agent 社区行业垂直场景贡献
- **应用领域**：工业缺陷检测、视觉异常检测
- **精度指标**：ImgAUROC ≥ 97.5%, PixAUROC ≥ 96.0%
