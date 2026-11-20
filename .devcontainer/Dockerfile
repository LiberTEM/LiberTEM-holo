# Dev container for LiberTEM-holo (see .devcontainer/README.md).
#
# No CUDA base image on purpose: like the tox `py3XX-cupy` envs, the CUDA
# toolkit comes from pip wheels (`cuda-toolkit[...]`), and the driver is
# injected by the NVIDIA container toolkit at runtime. Works on amd64 and arm64.
FROM mcr.microsoft.com/devcontainers/base:ubuntu26.04

ARG NVIM_VERSION=0.12.6

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        build-essential ripgrep fd-find less htop \
    && rm -rf /var/lib/apt/lists/*

# Nsight Systems (nsys) and Nsight Compute (ncu) for profiling CUDA code,
# from NVIDIA's CUDA apt repo. The pip cuda-toolkit doesn't ship them.
# The cuda-nsight-*-X-Y metapackages pick the versions that go with CUDA X.Y
# and put nsys/ncu into /usr/local/cuda-X.Y/bin.
ARG CUDA_VERSION=13.5
RUN case "$(uname -m)" in \
        x86_64) arch=x86_64 ;; aarch64) arch=sbsa ;; *) exit 1 ;; \
    esac \
    && ver="$(echo "${CUDA_VERSION}" | tr . -)" \
    && curl -fsSL -o /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/${arch}/cuda-keyring_1.1-1_all.deb" \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        "cuda-nsight-systems-${ver}" \
        "cuda-nsight-compute-${ver}" \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}"

# uv / uvx (also used for tox, pre-commit, ruff)
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /usr/local/bin/

# neovim (Ubuntu's package is too old); VS Code users can ignore it
RUN case "$(uname -m)" in \
        x86_64) arch=x86_64 ;; aarch64) arch=arm64 ;; *) exit 1 ;; \
    esac \
    && curl -fsSL "https://github.com/neovim/neovim/releases/download/v${NVIM_VERSION}/nvim-linux-${arch}.tar.gz" \
        | tar -xz -C /opt \
    && ln -s "/opt/nvim-linux-${arch}/bin/nvim" /usr/local/bin/nvim

USER vscode

# Claude Code, native installer (self-updating, installs to ~/.local/bin)
RUN curl -fsSL https://claude.ai/install.sh | bash

# Python interpreters for uv / tox; 3.14 is the dev default
RUN uv python install 3.14 3.13 3.12 3.11 3.10

# pre-commit on PATH: the git hook in the shared .git falls back to it when
# the interpreter it was installed with only exists on the host
RUN uv tool install pre-commit

ENV PATH="/home/vscode/.local/bin:${PATH}"
