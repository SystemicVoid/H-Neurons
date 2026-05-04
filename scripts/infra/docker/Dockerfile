# syntax=docker/dockerfile:1.7
ARG CUDA_BASE=nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.8
ARG PYTHON_VERSION=3.11

FROM --platform=linux/amd64 ${UV_IMAGE} AS uv

FROM --platform=linux/amd64 ${CUDA_BASE} AS builder
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_TORCH_BACKEND=cu130
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /opt/h-neurons
COPY requirements.txt pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install "${PYTHON_VERSION}" \
 && uv venv --python "${PYTHON_VERSION}" /opt/h-neurons/.venv \
 && uv pip install \
        --python /opt/h-neurons/.venv/bin/python \
        --torch-backend cu130 \
        -r requirements.txt

FROM --platform=linux/amd64 ${CUDA_BASE} AS final
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/h-neurons/.venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    PATH=/opt/h-neurons/.venv/bin:/usr/local/cuda/bin:$PATH \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    PYTHONUNBUFFERED=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        cuda-nvcc-13-0 \
        curl \
        git \
        openssh-server \
        pigz \
        ripgrep \
        rsync \
        tmux \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd /root/.ssh \
 && chmod 700 /root/.ssh \
 && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
 && sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
COPY --from=builder /opt/uv/python /opt/uv/python
COPY --from=builder /opt/h-neurons/.venv /opt/h-neurons/.venv
COPY --from=uv /uv /uvx /usr/local/bin/
COPY scripts/infra/docker/start.sh /start.sh
COPY scripts/infra/docker/smoke_triton.py /opt/smoke/triton.py
RUN chmod +x /start.sh
EXPOSE 22
CMD ["/start.sh"]
