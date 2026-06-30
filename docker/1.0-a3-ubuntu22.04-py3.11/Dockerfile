FROM quay.io/ascend/vllm-ascend:v0.18.0-a3

USER root


# 替换APT源为华为云
RUN echo "配置华为云APT源..." && \
cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
sed -i 's@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g' /etc/apt/sources.list && \
sed -i 's@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g' /etc/apt/sources.list && \
echo "华为云APT源配置完成"

# 安装基础系统工具
RUN apt-get update && \
    apt-get install -y \
    sudo \
    zip \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ARG TARGETARCH
RUN echo "Building for architecture: $TARGETARCH" && \
    case "$TARGETARCH" in \
      amd64|x86_64) TTYD_URL="https://github.com/tsl0922/ttyd/releases/latest/download/ttyd.x86_64" ;; \
      arm64|aarch64) TTYD_URL="https://github.com/tsl0922/ttyd/releases/latest/download/ttyd.aarch64" ;; \
      *) echo "Unsupported arch: $TARGETARCH"; exit 1 ;; \
    esac \
    && curl -fsSL "$TTYD_URL" -o /usr/local/bin/ttyd \
    && chmod +x /usr/local/bin/ttyd

WORKDIR /repositories
ARG CACHEBUST=1

# 安装 Node.js 20.x（Claude Code CLI 需要）
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 安装 Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

RUN echo "创建 atomgit 用户与用户组..." && \
    groupadd -g 1000 -r atomgitgroup && \
    useradd -u 1000 -m -g atomgitgroup -s /bin/bash atomgit && \
    usermod -aG sudo atomgit && \
    echo "atomgit ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/atomgit && \
    chmod 0440 /etc/sudoers.d/atomgit && \
    mkdir -p /opt/atomgit && \
    chown -R atomgit:atomgitgroup /opt/atomgit && \
    echo "✅ atomgit 用户已创建并授予 sudo 权限"



# ⭐ 切换到 atomgit 用户
USER atomgit

# 创建必要的目录
RUN mkdir -p ~/.claude
# 安装插件
RUN claude plugin marketplace add https://gitcode.com/gmq123/ascend-model-agent-plugin.git
RUN claude plugin install ascend-model-agent-plugin@ascend-model-agent-plugin

# 创建工作目录
WORKDIR /opt/atomgit

