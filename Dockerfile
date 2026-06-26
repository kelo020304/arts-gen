# syntax=docker/dockerfile:1.6

ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_MIRROR=https://mirrors.ivolces.com/ubuntu
ARG PIP_INDEX_URL=https://mirrors.ivolces.com/pypi/simple/
ARG PIP_TRUSTED_HOST=mirrors.ivolces.com
ARG VENV_DIR=/opt/venvs/arts-gen
ARG TORCH_CUDA_ARCH_LIST=9.0
ARG INSTALL_SAM3D_ENV=0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/venvs/arts-gen/bin:/opt/conda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CUDA=1 \
    MAX_JOBS=2 \
    MAKEFLAGS=-j2 \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    SPCONV_ALGO=native \
    ATTN_BACKEND=sdpa \
    LIDRA_SKIP_INIT=true \
    ARTS_GEN_ENV_DIR=/opt/venvs/arts-gen \
    ARTS_GEN_PYTHON=/opt/venvs/arts-gen/bin/python \
    SAM3D_DEPS_DIR=/workspace/arts-gen/sam3d_cu118_deps \
    SAM3D_PIPELINE_YAML=/weights/pipeline.yaml \
    PYTHONPATH=/workspace/arts-gen/TRELLIS-arts

WORKDIR /workspace/arts-gen

RUN if [ -n "${APT_MIRROR}" ]; then \
        printf '%s\n' \
          "deb ${APT_MIRROR} jammy main restricted universe multiverse" \
          "deb ${APT_MIRROR} jammy-updates main restricted universe multiverse" \
          "deb ${APT_MIRROR} jammy-backports main restricted universe multiverse" \
          "deb ${APT_MIRROR} jammy-security main restricted universe multiverse" \
          > /etc/apt/sources.list; \
    fi \
    && find /etc/apt/sources.list.d -type f -name '*.list' -exec mv {} {}.disabled \; 2>/dev/null || true \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        g++-11 \
        gcc-11 \
        git \
        libegl1 \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ninja-build \
        pkg-config \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN python --version \
    && python -c 'import sys; assert sys.version_info[:2] == (3, 10), f"expected Python 3.10, got {sys.version}"'
RUN mkdir -p /opt/venvs \
    && ln -sfn /opt/conda "${VENV_DIR}" \
    && ln -sf "$(command -v python)" /usr/local/bin/python3.10 \
    && ln -sf "$(command -v pip)" /usr/local/bin/pip3.10

COPY README.md CLAUDE.md run_pipeline.sh ./
COPY scripts/ scripts/
COPY pipeline/ pipeline/
COPY TRELLIS-arts/ TRELLIS-arts/
COPY submodules/TRELLIS.1/ submodules/TRELLIS.1/
COPY submodules/sam3d-stage/ submodules/sam3d-stage/

RUN if [ "${INSTALL_SAM3D_ENV}" = "1" ]; then \
        test -d "${SAM3D_DEPS_DIR}" || { \
          echo "SAM3D_DEPS_DIR not found: ${SAM3D_DEPS_DIR}"; \
          echo "Mount or copy the offline SAM3D CUDA 11.8 bundle before building with INSTALL_SAM3D_ENV=1."; \
          exit 2; \
        }; \
        PYTHON_BIN=python3.10 \
        SAM3D_DEPS_DIR="${SAM3D_DEPS_DIR}" \
        AUTO_CUDA_118=0 \
        LOCAL_CUDA_118=/usr/local/cuda \
        LOCAL_GCC11_HOME=/usr \
        PREFER_GCC11=1 \
        ALLOW_UNSUPPORTED_HOST_COMPILER=1 \
        VENV_SYSTEM_SITE_PACKAGES=1 \
        ALLOW_ONLINE_WHEELHOUSE=1 \
        FLASH_ATTN_HARD=0 \
        KAOLIN_HARD=1 \
        XFORMERS_HARD=0 \
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        bash scripts/ops/setup/setup_sam3d_env_cu118.sh "${VENV_DIR}"; \
    fi

RUN if [ -x "${ARTS_GEN_PYTHON}" ]; then \
        "${ARTS_GEN_PYTHON}" -c 'import sys, torch; import easydict, safetensors, trimesh; print("runtime OK", sys.version.split()[0], torch.__version__)'; \
    fi

CMD ["bash"]
