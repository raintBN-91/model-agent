# ModelAgent

> [English](./OVERVIEW.md) | 中文

## 快速参考

- ModelAgent 由 [modelagent community](https://gitcode.com/Ascend/model-agent) 维护

- 从哪里获取帮助
  - [ModelAgent 开源项目](https://gitcode.com/Ascend/model-agent)
  - [问题反馈](https://gitcode.com/Ascend/model-agent/issues)

---

## ModelAgent

ModelAgent 是面向昇腾 AI 处理器的模型智能体体验镜像，可以在镜像中体验 ModelAgent 提供的各项自动化能力，包括模型适配、迁移、量化、优化等功能。

---

## 支持的 Tags 及镜像链接

### Tag 规范

Tag 遵循以下格式：

```
<版本号>-<芯片系列>-<操作系统>-py<python版本>
```

| 字段 | 示例值 | 说明 |
|---|---|---|
| `版本号` | `1.0` | ModelAgent 版本号 |
| `芯片系列` | `a2`、`a3` | 目标昇腾芯片系列 |
| `操作系统` | `ubuntu22.04` | 基础操作系统 |
| `python版本` | `py3.11` | Python 版本 |

### ModelAgent 1.0
| Tag | dockerfile | 镜像内容 |
|-----|----------|----------|
| `1.0-a2-ubuntu22.04-py3.11` | [dockerfile](https://gitcode.com/Ascend/model-agent/blob/master/docker/1.0-a2-ubuntu22.04-py3.11/Dockerfile) | modelagent/torch_npu 2.9 |
| `1.0-a3-ubuntu22.04-py3.11` | [dockerfile](https://gitcode.com/Ascend/model-agent/blob/master/docker/1.0-a3-ubuntu22.04-py3.11/Dockerfile) | modelagent/torch_npu 2.9 |

---

## 快速开始

### 前置要求（可选）

#### 安装驱动

主机上必须安装与容器内 CANN 版本兼容的昇腾 NPU 驱动。请参阅 [CANN 兼容性矩阵](https://www.hiascend.com/document) 了解驱动与 CANN 版本的对应关系。

---

### 运行 ModelAgent 容器

```bash
docker run \
  --name modelagent_container \
  --shm-size=1g \
  --net=host \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -it {model-agent_tag}:latest bash
```

### 如何本地构建
```bash
docker buildx build -t modelagent:1.0-a2-ubuntu22.04-py3.11 -f Dockerfile .
```

### 如何二次开发
```bash
# 以 ModelAgent 镜像为基础镜像，叠加用户软件
FROM quay.io/linxishuixin/model-agent/model-agent-a2:v0.0

RUN apt update -y && \
 apt install gcc ...

...
```

---

## model agent使用

进入容器后，输入命令编辑 Claude 配置文件：

```bash
vim ~/.claude/settings.json
```

按 `i` 键进入编辑模式，先删除原先内容，然后输入如下内容替换原内容（以 Kimi K2.6 为例）：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.kimi.com/coding/",
    "ANTHROPIC_AUTH_TOKEN": "xxx",
    "ANTHROPIC_MODEL": "kimi-k2.6",
    "ANTHROPIC_SMALL_FAST_MODEL": "kimi-k2.6",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "API_TIMEOUT_MS": "600000"
  },
  "extraKnownMarketplaces": {
    "ascend-model-agent-plugin": {
      "source": {
        "source": "git",
        "url": "https://gitcode.com/gmq123/ascend-model-agent-plugin.git"
      }
    }
  },
  "enabledPlugins": {
    "ascend-model-agent-plugin@ascend-model-agent-plugin": true
  }
}
```

> **注意**：将 `ANTHROPIC_AUTH_TOKEN` 的值 `"xxx"` 替换成你自己的实际 Token。

输入 `:wq` 保存并退出编辑，然后启动 Claude Code：

```bash
claude
```

一直按回车进入到对话界面，即可体验 Model Agent Skills，例如：

- `/verify-agent 帮我适配 qwen3.5-0.8B 模型`
- `/optimizer-agent 帮我优化 qwen3.5-0.8B 模型`

---

## 镜像软件信息

| 软件 | 版本 |
|---|---|
| OS | Ubuntu 22.04.5 |
| Python | 3.11 |
| Package Manager | apt |
| Torch_npu | 2.9 |

---

## 支持的硬件

| 芯片系列 | 产品示例 | 架构 |
|---|---|---|
| 昇腾 910 | Atlas 800T A2、Atlas 900 A2 PoD | ARM64 / x86_64 |
| 昇腾 A3 | Atlas 800T A3 | ARM64 / x86_64 |

---

## 许可证

查看这些镜像中包含的 ModelAgent 和相关软件的[许可证信息](https://gitcode.com/Ascend/model-agent)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。