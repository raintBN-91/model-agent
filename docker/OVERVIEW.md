# ModelAgent

> English | [中文](./OVERVIEW.zh.md)

## Quick Reference

- ModelAgent is maintained by the [modelagent community](https://gitcode.com/Ascend/model-agent)

- Where to get help
  - [ModelAgent Open Source Project](https://gitcode.com/Ascend/model-agent)
  - [Issue Feedback](https://gitcode.com/Ascend/model-agent/issues)

---

## ModelAgent

ModelAgent is an intelligent model agent experience image for Atlas AI processors. Within the image, you can experience various automation capabilities provided by ModelAgent, including model adaptation, migration, quantization, optimization, and more.

---

## Supported Tags and Image Links

### Tag Convention

Tags follow the format below:

```
<version>-<chip_series>-<os>-py<python_version>
```

| Field | Example | Description |
|---|---|---|
| `version` | `1.0` | ModelAgent version |
| `chip_series` | `a2`, `a3` | Target Atlas chip series |
| `os` | `ubuntu22.04` | Base operating system |
| `python_version` | `py3.11` | Python version |

### ModelAgent 1.0
| Tag | Image Address | Image Contents |
|-----|----------|----------|
| `1.0-a2-ubuntu22.04-py3.11` | [dockerfile](https://gitcode.com/Ascend/model-agent/blob/master/docker/1.0-a2-ubuntu22.04-py3.11/Dockerfile) | modelagent/torch_npu 2.9 |
| `1.0-a3-ubuntu22.04-py3.11` | [dockerfile](https://gitcode.com/Ascend/model-agent/blob/master/docker/1.0-a3-ubuntu22.04-py3.11/Dockerfile) | modelagent/torch_npu 2.9 |

---

## Quick Start

### Prerequisites (Optional)

#### Install Driver

The host machine must have an Atlas NPU driver installed that is compatible with the CANN version inside the container. Please refer to the [CANN Compatibility Matrix](https://www.hiascend.com/document) for the mapping between drivers and CANN versions.

---

### Run ModelAgent Container

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

### How to Build Locally
```bash
docker buildx build -t modelagent:0.0-a2-ubuntu22.04-py3.11 -f Dockerfile .
```

### How to Develop Further
```bash
# Use the ModelAgent image as the base image and add your own software on top
FROM quay.io/linxishuixin/model-agent/model-agent-a2:v0.0

RUN apt update -y && \
 apt install gcc ...

...
```

---

## Using Model Agent

After entering the container, edit the Claude configuration file with the following command:

```bash
vim ~/.claude/settings.json
```

Press `i` to enter edit mode, delete the existing content, and then enter the following to replace it (using Kimi K2.6 as an example):

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

> **Note**: Replace the value of `ANTHROPIC_AUTH_TOKEN`, `"xxx"`, with your actual token.

Type `:wq` to save and exit, then start Claude Code:

```bash
claude
```

Keep pressing Enter until you reach the conversation interface, then you can experience Model Agent Skills, for example:

- `/verify-agent Help me adapt the qwen3.5-0.8B model`
- `/optimizer-agent Help me optimize the qwen3.5-0.8B model`

---

## Image Software Information

| Software | Version |
|---|---|
| OS | Ubuntu 22.04.5 |
| Python | 3.11 |
| Package Manager | apt |
| Torch_npu | 2.9 |

---

## Supported Hardware

| Chip Series | Product Examples | Architecture |
|---|---|---|
| Atlas 910 | Atlas 800T A2, Atlas 900 A2 PoD | ARM64 / x86_64 |
| Atlas A3 | Atlas 800T A3 | ARM64 / x86_64 |

---

## License

See the [license information](https://gitcode.com/Ascend/model-agent) for ModelAgent and related software contained in these images.

As with all container images, pre-installed software packages (Python, system libraries, etc.) may be subject to their own license terms.
