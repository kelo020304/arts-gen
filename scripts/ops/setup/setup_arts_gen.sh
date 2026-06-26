#!/usr/bin/env bash
# setup_arts_gen.sh — install the `arts-gen` conda environment for arts-reconstruction.
#
# Replaces the legacy `trellis` / `artsvox` envs. Mirrors TRELLIS-arts/setup.sh's
# command logic but pinned to a single env name (`arts-gen`) and one platform
# (Linux + CUDA 11.8 + PyTorch 2.4.0). Adds dataset_toolkits's deps.
#
# Usage:
#   bash scripts/ops/setup/setup_arts_gen.sh [options]
#
# Modes:
#   --check                 Dry-run: print every command, execute nothing.
#   --force                 Re-create env even if it exists (conda env remove + recreate).
#
# Stage skips (for incremental recovery if a stage fails):
#   --skip-pytorch          Skip stage 1 (env + PyTorch)
#   --skip-basic            Skip stage 2 (basic deps)
#   --skip-train            Skip stage 3 (train deps)
#   --skip-xformers         Skip xformers
#   --skip-flashattn        Skip flash-attn (often fails — use --skip-flashattn first)
#   --skip-spconv           Skip spconv-cu120
#   --skip-kaolin           Skip kaolin
#   --skip-nvdiffrast       Skip nvdiffrast (cloned + pip install)
#   --skip-diffoctreerast   Skip diffoctreerast (cloned + pip install)
#   --skip-vox2seq          Skip vox2seq (cloned from upstream microsoft/TRELLIS)
#
# Recovery example (flash-attn fails, want to resume rest):
#   bash scripts/ops/setup/setup_arts_gen.sh --skip-pytorch --skip-basic --skip-train --skip-flashattn
#
# Verify after install:
#   conda activate arts-gen
#   python -c "import torch, trimesh, yaml, transformers, spconv; print(torch.__version__)"

set -euo pipefail

# Limit compile parallelism to 2 cores (project-wide convention).
# Affects: flash-attn (Stage 4b), nvdiffrast / diffoctreerast / vox2seq (Stage 5).
# Pre-built wheels (xformers, spconv-cu120, kaolin, torch) ignore these vars.
export MAX_JOBS="${MAX_JOBS:-2}"
export MAKEFLAGS="${MAKEFLAGS:--j2}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

ENV_NAME="arts-gen"
PYTHON_VERSION="3.10"
PYTORCH_VERSION="2.4.0"
TORCHVISION_VERSION="0.19.0"
CUDA_TAG="cu124"  # PyTorch wheel flavor (cu124 wheels work with CUDA 12.4-12.6 toolkit)
EXT_TMP="/tmp/arts-gen-extensions"

# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
CHECK=0
FORCE=0
declare -A SKIP=(
  [pytorch]=0 [basic]=0 [train]=0 [xformers]=0 [flashattn]=0
  [spconv]=0 [kaolin]=0 [nvdiffrast]=0 [diffoctreerast]=0 [vox2seq]=0
)

usage() { sed -n '2,28p' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --check)               CHECK=1; shift ;;
    --force)               FORCE=1; shift ;;
    --skip-pytorch)        SKIP[pytorch]=1;        shift ;;
    --skip-basic)          SKIP[basic]=1;          shift ;;
    --skip-train)          SKIP[train]=1;          shift ;;
    --skip-xformers)       SKIP[xformers]=1;       shift ;;
    --skip-flashattn)      SKIP[flashattn]=1;      shift ;;
    --skip-spconv)         SKIP[spconv]=1;         shift ;;
    --skip-kaolin)         SKIP[kaolin]=1;         shift ;;
    --skip-nvdiffrast)     SKIP[nvdiffrast]=1;     shift ;;
    --skip-diffoctreerast) SKIP[diffoctreerast]=1; shift ;;
    --skip-vox2seq)        SKIP[vox2seq]=1;        shift ;;
    -h|--help)             usage; exit 0 ;;
    *)  echo "[error] unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# -----------------------------------------------------------------------------
# run() — wrapper that prints + (optionally) executes
# -----------------------------------------------------------------------------
run() {
  echo "  + $*"
  if [ "$CHECK" -eq 0 ]; then
    eval "$@"
  fi
}

# pip via the env's python (PIP_BASE for uninstall/show, PIP for install)
PIP_BASE="conda run -n $ENV_NAME --no-capture-output pip"
PIP="$PIP_BASE install"

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------
echo "=== arts-gen env setup ==="
echo "  env_name        : $ENV_NAME"
echo "  python_version  : $PYTHON_VERSION"
echo "  pytorch_version : $PYTORCH_VERSION + $CUDA_TAG"
echo "  mode            : $([ "$CHECK" -eq 1 ] && echo DRY-RUN || echo EXECUTE)"
echo "  force           : $FORCE"
echo "  compile_jobs    : MAX_JOBS=$MAX_JOBS  MAKEFLAGS=$MAKEFLAGS  CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"
echo "  skip            : $(for k in "${!SKIP[@]}"; do [ "${SKIP[$k]}" -eq 1 ] && echo -n "$k "; done)"
echo ""

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda not in PATH" >&2; exit 3
fi

# -----------------------------------------------------------------------------
# Stage 1: env + PyTorch (pip wheel, NOT conda)
# -----------------------------------------------------------------------------
# Why pip-torch instead of `conda install pytorch -c pytorch -c nvidia`:
# Conda channel torch 2.4.0 has a known incompatibility with newer MKL where
# libtorch_cpu.so references `iJIT_NotifyEvent` (Intel ITT API) that newer MKL
# doesn't expose -> ImportError. Pip wheels bundle their own MKL/ITT and avoid
# this. See https://github.com/pytorch/pytorch/issues/123097
if [ "${SKIP[pytorch]}" -eq 0 ]; then
  echo "--- Stage 1: create env + install PyTorch (pip wheel) ---"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [ "$FORCE" -eq 1 ]; then
      run conda env remove -n "$ENV_NAME" -y
      run conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
    else
      echo "  [skip] env $ENV_NAME already exists (use --force to recreate)"
      # If conda-installed torch lingers, pip install below will replace it.
    fi
  else
    run conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
  fi
  # Uninstall any existing conda-installed torch first, then install pip wheel.
  run "$PIP_BASE uninstall -y torch torchvision 2>/dev/null || true"
  run "$PIP torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/$CUDA_TAG"
fi

# -----------------------------------------------------------------------------
# Stage 2: basic deps (data processing + image / mesh)
# -----------------------------------------------------------------------------
if [ "${SKIP[basic]}" -eq 0 ]; then
  echo "--- Stage 2: basic deps ---"
  run "$PIP pillow imageio imageio-ffmpeg tqdm easydict \
       opencv-python-headless scipy ninja rembg onnxruntime \
       trimesh open3d xatlas pyvista pymeshfix igraph transformers \
       pyyaml pandas matplotlib"
  run "$PIP git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
fi

# -----------------------------------------------------------------------------
# Stage 3: train deps + pillow-simd
# -----------------------------------------------------------------------------
if [ "${SKIP[train]}" -eq 0 ]; then
  echo "--- Stage 3: train deps ---"
  run "$PIP tensorboard lpips"
  echo "  (pillow-simd is optional; reinstalls Pillow-as-SIMD; uncomment line below to enable)"
  echo "  # run $PIP --force-reinstall pillow-simd"
fi

# -----------------------------------------------------------------------------
# Stage 4: package-level CUDA extensions
# -----------------------------------------------------------------------------
if [ "${SKIP[xformers]}" -eq 0 ]; then
  echo "--- Stage 4a: xformers (torch $PYTORCH_VERSION + $CUDA_TAG) ---"
  # --force-reinstall ensures we replace any existing xformers built against a different CUDA tag
  run "$PIP --force-reinstall --no-deps xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/$CUDA_TAG"
fi

if [ "${SKIP[flashattn]}" -eq 0 ]; then
  echo "--- Stage 4b: flash-attn (often fails to compile — has --skip-flashattn) ---"
  # --no-build-isolation: flash-attn's setup.py imports torch at build time;
  # without this flag, pip creates an isolated build env without torch and fails immediately.
  # Requires arts-gen to already have torch + setuptools + wheel + packaging + ninja (Stage 1+2).
  run "$PIP flash-attn --no-build-isolation"
fi

if [ "${SKIP[spconv]}" -eq 0 ]; then
  echo "--- Stage 4c: spconv-cu120 (closest cu12.x variant, forward-compat with cu124/126) ---"
  run "$PIP spconv-cu120"
fi

if [ "${SKIP[kaolin]}" -eq 0 ]; then
  echo "--- Stage 4d: kaolin (NVIDIA pre-built wheel for torch 2.4.0 cu124) ---"
  run "$PIP kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${PYTORCH_VERSION}_cu124.html"
fi

# -----------------------------------------------------------------------------
# Stage 5: source-clone CUDA extensions (nvdiffrast / diffoctreerast / vox2seq)
# -----------------------------------------------------------------------------
mkdir_ext() { run "mkdir -p $EXT_TMP"; }

if [ "${SKIP[nvdiffrast]}" -eq 0 ]; then
  echo "--- Stage 5a: nvdiffrast ---"
  mkdir_ext
  if [ -d "$EXT_TMP/nvdiffrast" ]; then
    echo "  [skip] $EXT_TMP/nvdiffrast already cloned"
  else
    run "git clone https://github.com/NVlabs/nvdiffrast.git $EXT_TMP/nvdiffrast"
  fi
  run "$PIP $EXT_TMP/nvdiffrast"
fi

if [ "${SKIP[diffoctreerast]}" -eq 0 ]; then
  echo "--- Stage 5b: diffoctreerast ---"
  mkdir_ext
  if [ -d "$EXT_TMP/diffoctreerast" ]; then
    echo "  [skip] $EXT_TMP/diffoctreerast already cloned"
  else
    run "git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git $EXT_TMP/diffoctreerast"
  fi
  run "$PIP $EXT_TMP/diffoctreerast"
fi

if [ "${SKIP[vox2seq]}" -eq 0 ]; then
  echo "--- Stage 5c: vox2seq (clone upstream TRELLIS, install extension subdir) ---"
  mkdir_ext
  if [ -d "$EXT_TMP/trellis_upstream" ]; then
    echo "  [skip] $EXT_TMP/trellis_upstream already cloned"
  else
    run "git clone --depth 1 https://github.com/microsoft/TRELLIS.git $EXT_TMP/trellis_upstream"
  fi
  if [ -d "$EXT_TMP/trellis_upstream/extensions/vox2seq" ] || [ "$CHECK" -eq 1 ]; then
    run "$PIP $EXT_TMP/trellis_upstream/extensions/vox2seq"
  else
    echo "  [warn] $EXT_TMP/trellis_upstream/extensions/vox2seq not found; skipping"
  fi
fi

# -----------------------------------------------------------------------------
# Verify
# -----------------------------------------------------------------------------
echo ""
echo "--- Stage 6: verify ---"
if [ "$CHECK" -eq 0 ]; then
  conda run -n "$ENV_NAME" --no-capture-output python -c "
import importlib, sys
mods = ['torch', 'torchvision', 'trimesh', 'yaml', 'transformers', 'open3d']
for m in mods:
    try:
        v = importlib.import_module(m).__version__
    except Exception as e:
        v = f'IMPORT_FAIL: {e}'
    print(f'  {m:20s}: {v}')
import torch
print(f'  cuda_available     : {torch.cuda.is_available()}')
print(f'  cuda_version       : {torch.version.cuda}')
" || {
    echo "[warn] verification import failed — check skipped stages"
    exit 5
  }
else
  echo "  [dry-run] would run: python -c 'import torch, trimesh, yaml, transformers ...'"
fi

echo ""
echo "=== arts-gen setup complete ==="
echo "Activate: conda activate $ENV_NAME"
