# MiniCPM5 RTX5080 开发镜像
# 基础镜像使用本机已有的 CUDA 12.8.1 devel（含 nvcc，支持 sm_120 / compute_120），
# 对应《MiniCPM5 RTX5080 Rust CUDA 实施计划》3.3 节软件栈要求。
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

# 构建期代理（走宿主机 mihomo，构建网络需为 host）；仅构建期生效，不留在最终镜像
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ENV http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY}

ENV DEBIAN_FRONTEND=noninteractive

# C++20/CMake 构建链、Rust 前置依赖、离线 Python 工具链基础、sshd 与调试工具
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        wget \
        ca-certificates \
        pkg-config \
        libssl-dev \
        python3 \
        python3-pip \
        python3-venv \
        openssh-server \
        gosu \
        gdb \
        vim \
        jq \
        unzip \
        htop \
    && rm -rf /var/lib/apt/lists/*

# Rust stable 工具链装到 /opt，运行期以 named volume 持久化，容器重建不丢缓存
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:${PATH}

RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal \
    && chmod -R a+rX /opt/rustup /opt/cargo

# 清除构建期代理，避免其固化进镜像；运行期代理由 compose 注入
ENV http_proxy= \
    https_proxy=

# 与宿主机 nuists517 (1000:1000) 同 uid 的开发用户，保证挂载目录文件属主一致；
# 基础镜像自带 ubuntu (1000:1000) 默认用户，先移除避免 UID/GID 冲突
RUN existing_user=$(getent passwd ${USER_UID} | cut -d: -f1) \
    && { [ -z "$existing_user" ] || userdel --remove "$existing_user"; } \
    && existing_group=$(getent group ${USER_GID} | cut -d: -f1) \
    && { [ -z "$existing_group" ] || groupdel "$existing_group"; } \
    && groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
    && mkdir -p /opt/rustup /opt/cargo /home/${USERNAME}/.ssh \
    && chown -R ${USER_UID}:${USER_GID} /opt/rustup /opt/cargo /home/${USERNAME}/.ssh \
    && chmod 700 /home/${USERNAME}/.ssh

# sshd 监听 2301（host 网络下与宿主机 22 不冲突），仅公钥认证
RUN ssh-keygen -A \
    && printf 'Port 2301\nPubkeyAuthentication yes\nPasswordAuthentication no\nPermitRootLogin no\n' \
        > /etc/ssh/sshd_config.d/00-container.conf

# sudo 免密提权：dev 用户容器内可执行 root 操作（nvidia-smi 调整、系统包安装等）
RUN apt-get update && apt-get install -y --no-install-recommends sudo \
    && rm -rf /var/lib/apt/lists/* \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 440 /etc/sudoers.d/${USERNAME}

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
