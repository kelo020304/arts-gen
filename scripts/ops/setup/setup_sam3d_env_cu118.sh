#!/usr/bin/env bash
# =============================================================================
# setup_sam3d_env_cu118.sh — cu118 build of the ONE UNIFIED venv (offline H20).
# =============================================================================
# This is the CUDA 11.8 sibling of setup_sam3d_env.sh. Same goal: a single venv
# for ALL 4 full-pipeline-inference stages (ss / part / slat / assemble) AND the
# eval platform — sam3d_objects and TRELLIS-arts (part flow) coexist here, with
# the heavy deps shared. The cu121 script targets the DEV cu121 GPU box; THIS one
# targets the offline H20 + CUDA 11.8 dev boxes where the cu121 pins do NOT work.
#
# WHY A SEPARATE cu118 SCRIPT
#   sam-3d-objects pins the WHOLE stack to CUDA 12.1 (torch 2.5.1+cu121,
#   torchaudio 2.5.1+cu121, spconv-cu121, xformers 0.0.28.post3, kaolin cu121,
#   nvidia-cuda-nvcc-cu12 / cuda-python 12.1, ...). On a cu118 box those wheels
#   either don't import (ABI/driver mismatch) or aren't on the offline mirror.
#   So here the ENTIRE stack is cu118, matched to the proven trellis env
#   'arts-gen': torch 2.4.0+cu118, torchvision 0.19.0+cu118, spconv-cu118==2.3.8,
#   xformers==0.0.27.post2+cu118, nvdiffrast 0.4.0, NO flash_attn, NO kaolin
#   prebuilt — trellis part flow runs on exactly this. We then ADD the deps
#   sam3d's wrappers hard-require (kaolin + pytorch3d + gsplat), built/fetched
#   for cu118.
#
# DEP HANDLING (per the sam3d hard-dep probe)
#   - flash_attn  : OPTIONAL on cu118. It is only imported when the GPU is
#                   A100/H100/H200; on everything else it is dead code. Building
#                   flash-attn for cu118+torch2.4 is the painful one, so this
#                   script SKIPS it by default (FLASH_ATTN_HARD=0). Set
#                   FLASH_ATTN_HARD=1 to attempt a source build (note printed on
#                   failure). NOTE: arts-gen itself ships NO flash_attn.
#   - kaolin      : HARD (top-level import via flexicubes; only check_tensor is
#                   used at runtime, but the import must succeed). Fetched as a
#                   prebuilt cu118 wheel from NVIDIA's kaolin index for
#                   (torch 2.4.0, cu118). KAOLIN_HARD=1 by default.
#   - xformers    : OPTIONAL (only used if ATTN_BACKEND=xformers; default sdpa).
#                   Still pinned to the arts-gen wheel (0.0.27.post2+cu118) so
#                   the env matches the proven trellis setup.
#   - nvdiffrast  : OPTIONAL (lazy import, try/except guarded) — not installed.
#   - pytorch3d   : HARD (top-level import in inference_pipeline_pointmap),
#                   built from the pinned git ref against cu118 torch.
#   - gsplat      : HARD (top-level in gaussian_render), built from the pinned
#                   git ref against cu118 torch.
#   - MoGe        : pure python + torch, no cu pin — installed as-is from base.
#
# WHERE THIS RUNS
#   *** OFFLINE H20 / CUDA 11.8 DEV BOXES. *** The cu121 variant is
#   setup_sam3d_env.sh (DEV cu121 GPU box only). pytorch3d / gsplat (and
#   flash_attn if you force it) build CUDA extensions and want a GPU visible AT
#   BUILD TIME, so build this on a GPU node, not a CPU-only login shell. Offline
#   means: only an internal pip mirror is reachable. Point PIP_INDEX_URL /
#   PIP_EXTRA_INDEX_URL at the mirror that hosts the cu118 wheels (see overrides).
#
# CONDA -> VENV
#   Upstream ships a conda env (CUDA 12.1 toolkit + compilers). We want a *venv*,
#   so those come from the SYSTEM instead:
#     - a python3.11 interpreter on PATH (match upstream's pinned 3.11; the cu118
#       wheels below are built for the 3.11 ABI, so do NOT downgrade to 3.10),
#     - a CUDA 11.8 toolkit with nvcc on PATH (to compile pytorch3d/gsplat). This
#       script warns (not aborts) if nvcc is not >= 11.8, and accepts ANY nvcc
#       (no 12.x hard-fail, unlike the cu121 script).
#
# WHAT IT BUILDS
#   A python3.11 venv with:
#     A. torch 2.4.0+cu118 / torchvision 0.19.0+cu118 / torchaudio 2.4.0+cu118
#        (== arts-gen; overridable via TORCH_SPEC), from the cu118 wheel index.
#     B. sam3d base deps WITHOUT the cu121 pins — requirements.txt rewritten on
#        the fly (spconv-cu121 -> spconv-cu118, +cu121 -> +cu118, torch/torchaudio
#        /torchvision + nvidia-cuda-nvcc-cu12 / cuda-python lines dropped). plus
#        xformers==0.0.27.post2+cu118 pinned explicitly.
#     C. pytorch3d + gsplat compiled from their pinned git refs against cu118
#        torch (+ optionally flash_attn / kaolin per the probe flags above).
#     D. sam-3d-objects + the 3 stage wrappers, editable (--no-deps for sam3d:
#        its cu121-pinned deps are handled in B/C).
#     E. the upstream hydra patch (./patching/hydra).
#     F. smoke import of sam3d_objects + the stage wrappers + trellis part flow.
#
# WEIGHTS / CONFIG (not installed here — runtime assets, pulled separately)
#   weights : /robot/data-lab/jzh/art-gen/weights   (== sam-3d-objects checkpoints/hf)
#   config  : <that weights dir>/pipeline.yaml
#
# SS-LATENT CONTRACT (for the glue that consumes this env, not used by setup)
#   sample_sparse_structure() returns return_dict["shape"] [bs,4096,8]; reshaped
#   to z_global via .permute(0,2,1).contiguous().view(bs,8,16,16,16). ss_latent.npy
#   is a plain (8,16,16,16) float32 array. return_dict["coords"] is [M,4] int.
#
# CONVENTIONS
#   - set -eo pipefail (NOT -u): this shell's startup snapshot references an
#     unbound $ZSH_VERSION, and -u makes $(...) sub-shells crash silently.
#   - No silent fallbacks: every step either succeeds or aborts with set -e.
#   - Compile rule: heavy builds run under MAX_JOBS=2 + 2-core make/cmake.
#
# USAGE
#   bash scripts/ops/setup/setup_sam3d_env_cu118.sh [VENV_DIR]
#   VENV_DIR        positional, default ".venv/sam3d-cu118" under sam-3d-objects.
#   env overrides:
#     PYTHON_BIN         python interpreter   (default python3.11)
#     TORCH_CUDA_INDEX   cu118 wheel index    (default download.pytorch.org/whl/cu118)
#     TORCH_SPEC         torch trio spec      (default torch/tv/ta 2.4.0+cu118)
#     XFORMERS_SPEC      xformers pin         (default xformers==0.0.27.post2+cu118)
#     KAOLIN_FIND_LINKS  kaolin wheel index   (default nvidia-kaolin torch-2.4.0_cu118)
#     FLASH_ATTN_HARD=1  force a flash_attn source build (default 0 = skip)
#     KAOLIN_HARD=0      skip kaolin (default 1 = install; it is a HARD import)
#     XFORMERS_HARD=1    install xformers --no-deps (default 0 = skip; sdpa default)
#     AUTO_CUDA_118=0    do not auto-prefer $HOME/.local/cuda-11.8 when present
#     LOCAL_GCC11_HOME   user-local gcc-11 root, default
#                        $HOME/.local/toolchains/gcc-11-ubuntu24/usr
#     PREFER_GCC11=0     do not auto-use gcc-11/g++-11 when present
#     ALLOW_UNSUPPORTED_HOST_COMPILER=0
#                        fail instead of adding nvcc --allow-unsupported-compiler
#                        when CUDA 11.8 only sees a newer host compiler
#     SKIP_NVCC_CHECK=1  bypass the nvcc presence/version warning
#     VENV_SYSTEM_SITE_PACKAGES=1
#                        create the venv with --system-site-packages (Docker
#                        builds can reuse the base image's torch/cu118 stack
#                        instead of duplicating it inside the venv)
# =============================================================================

set -eo pipefail

# --- compile rule: cap parallelism to 2 cores for any build-from-source ------
export MAX_JOBS=2
export MAKEFLAGS="-j2"
export CMAKE_BUILD_PARALLEL_LEVEL=2

# --- paths -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SAM3D_STAGE_DIR="$REPO_ROOT/submodules/sam3d-stage"
SAM3D_OBJ_DIR="$SAM3D_STAGE_DIR/submodules/sam-3d-objects"

MASK_PKG_DIR="$SAM3D_STAGE_DIR/generate_mask"
SURFACE_VOXEL_PKG_DIR="$SAM3D_STAGE_DIR/generate_surface_voxel"
TEXTURE_PKG_DIR="$SAM3D_STAGE_DIR/generate_texture"
HYDRA_PATCH="$SAM3D_OBJ_DIR/patching/hydra"
LOCAL_HYDRA_CORE_UTILS="$REPO_ROOT/scripts/vendor_patches/hydra_core_utils.py"

REQ_BASE="$SAM3D_OBJ_DIR/requirements.txt"
REQ_P3D="$SAM3D_OBJ_DIR/requirements.p3d.txt"
REQ_INFER="$SAM3D_OBJ_DIR/requirements.inference.txt"

# --- args / overrides --------------------------------------------------------
VENV_DIR="${1:-$SAM3D_OBJ_DIR/.venv/sam3d-cu118}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu118}"
NGC_INDEX="${NGC_INDEX:-https://pypi.ngc.nvidia.com}"
# torch trio: 2.5.1+cu118 — chosen so the BUNDLED kaolin wheel
# (kaolin-0.17.0-cp311, built for torch-2.5.1_cu118) matches the torch ABI.
# (arts-gen runs 2.4.0+cu118; trellis works on 2.5.1 too — minor bump.)
TORCH_SPEC="${TORCH_SPEC:-torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118}"
# OFFLINE deps bundle (pulled by tos_pull_sam3d_cu118_deps.sh): the kaolin cu118
# wheel + pytorch3d/gsplat/MoGe source that an offline box can't get from
# NVIDIA-S3 / github. pytorch3d/gsplat build here against the box's cu118 nvcc.
SAM3D_DEPS_DIR="${SAM3D_DEPS_DIR:-$REPO_ROOT/sam3d_cu118_deps}"
# offline wheelhouse inside the bundle: torch+cu118 trio + spconv + their full
# dep closure (nvidia-*-cu11 etc.) + pip/setuptools/wheel — installed --no-index.
WHEELHOUSE="$SAM3D_DEPS_DIR/wheelhouse"
# H20 = Hopper sm_90 — build pytorch3d/gsplat kernels for it (cu118 nvcc 11.8
# supports sm_90). Override for other GPUs (e.g. "8.0;8.6;9.0").
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
# xformers is OPTIONAL (default ATTN_BACKEND=sdpa) but we pin the arts-gen wheel
# so the env matches the proven trellis setup exactly.
XFORMERS_SPEC="${XFORMERS_SPEC:-xformers==0.0.27.post2+cu118}"
# kaolin publishes per-(torch,cuda) wheel indices. This one matches torch 2.4.0
# + cu118 (the arts-gen torch). kaolin is a HARD import via flexicubes.
KAOLIN_FIND_LINKS="${KAOLIN_FIND_LINKS:-https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html}"
# dep-hardness toggles (defaults from the sam3d cu118 probe):
#   flash_attn = OPTIONAL on cu118 (dead code off A100/H100/H200) -> skip.
#   kaolin     = HARD (top-level import) -> install.
FLASH_ATTN_HARD="${FLASH_ATTN_HARD:-0}"
KAOLIN_HARD="${KAOLIN_HARD:-1}"
XFORMERS_HARD="${XFORMERS_HARD:-0}"
AUTO_CUDA_118="${AUTO_CUDA_118:-1}"
LOCAL_GCC11_HOME="${LOCAL_GCC11_HOME:-${HOME:-}/.local/toolchains/gcc-11-ubuntu24/usr}"
PREFER_GCC11="${PREFER_GCC11:-1}"
ALLOW_UNSUPPORTED_HOST_COMPILER="${ALLOW_UNSUPPORTED_HOST_COMPILER:-auto}"
VENV_SYSTEM_SITE_PACKAGES="${VENV_SYSTEM_SITE_PACKAGES:-0}"
ALLOW_ONLINE_WHEELHOUSE="${ALLOW_ONLINE_WHEELHOUSE:-0}"

# Prefer a user-installed CUDA 11.8 toolkit when available, even if the shell
# already points CUDA_HOME at a system CUDA 12.x install. This keeps the venv
# build local and avoids accidentally compiling cu118 torch extensions with the
# wrong toolkit. Set AUTO_CUDA_118=0 to opt out.
LOCAL_CUDA_118="${LOCAL_CUDA_118:-${HOME:-}/.local/cuda-11.8}"
if [ "$AUTO_CUDA_118" = "1" ] && [ -x "$LOCAL_CUDA_118/bin/nvcc" ]; then
    export CUDA_HOME="$LOCAL_CUDA_118"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

append_nvcc_flag() {
    case " ${NVCC_FLAGS:-} " in
        *" $1 "*) ;;
        *) export NVCC_FLAGS="${NVCC_FLAGS:+$NVCC_FLAGS }$1" ;;
    esac
}

# CUDA 11.8's nvcc officially supports host GCC <= 11. If gcc-11/g++-11 are
# available, prefer them for both direct C++ compilation and nvcc's -ccbin.
# Otherwise keep the system compiler visible, but make the unsupported compiler
# choice explicit in NVCC_FLAGS so pytorch3d/gsplat do not fail at nvcc's guard.
HOST_COMPILER_NOTE="system default"
if [ "$PREFER_GCC11" = "1" ] && [ -x "$LOCAL_GCC11_HOME/bin/gcc-11" ] && [ -x "$LOCAL_GCC11_HOME/bin/g++-11" ]; then
    export PATH="$LOCAL_GCC11_HOME/bin:$PATH"
fi
if [ "$PREFER_GCC11" = "1" ] && command -v gcc-11 >/dev/null 2>&1 && command -v g++-11 >/dev/null 2>&1; then
    export CC="${CC:-$(command -v gcc-11)}"
    export CXX="${CXX:-$(command -v g++-11)}"
    export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
    append_nvcc_flag "-ccbin=$CC"
    HOST_COMPILER_NOTE="gcc-11/g++-11"
else
    HOST_CC="${CC:-$(command -v gcc || true)}"
    HOST_CC_MAJOR=""
    if [ -n "$HOST_CC" ]; then
        HOST_CC_MAJOR="$("$HOST_CC" -dumpfullversion -dumpversion 2>/dev/null | cut -d. -f1 || true)"
    fi
    if [ -n "$HOST_CC_MAJOR" ] && [ "$HOST_CC_MAJOR" -gt 11 ]; then
        case "$ALLOW_UNSUPPORTED_HOST_COMPILER" in
            1|true|yes|on|auto)
                append_nvcc_flag "--allow-unsupported-compiler"
                HOST_COMPILER_NOTE="system gcc $HOST_CC_MAJOR with nvcc --allow-unsupported-compiler"
                ;;
            *)
                echo "[error] CUDA 11.8 nvcc supports host GCC <= 11, but current gcc is $HOST_CC_MAJOR and gcc-11 is not on PATH." >&2
                echo "        Install gcc-11/g++-11 or set ALLOW_UNSUPPORTED_HOST_COMPILER=1." >&2
                exit 1
                ;;
        esac
    fi
fi

export FORCE_CUDA="${FORCE_CUDA:-1}"

echo "=============================================================="
echo "[setup_sam3d_env_cu118] cu118 / GPU env — OFFLINE H20 BOXES"
echo "  venv dir          : $VENV_DIR"
echo "  python            : $PYTHON_BIN"
echo "  cuda home         : ${CUDA_HOME:-<unset>}"
echo "  host compiler     : $HOST_COMPILER_NOTE"
echo "  CC/CXX            : ${CC:-<unset>} / ${CXX:-<unset>}"
echo "  NVCC_FLAGS        : ${NVCC_FLAGS:-<unset>}"
echo "  force cuda build  : $FORCE_CUDA"
echo "  system site pkgs  : $VENV_SYSTEM_SITE_PACKAGES"
echo "  online wheelhouse : $ALLOW_ONLINE_WHEELHOUSE"
echo "  torch index       : $TORCH_CUDA_INDEX"
echo "  torch spec        : $TORCH_SPEC"
echo "  xformers spec     : $XFORMERS_SPEC"
echo "  ngc index         : $NGC_INDEX"
echo "  kaolin find-links : $KAOLIN_FIND_LINKS"
echo "  flash_attn hard   : $FLASH_ATTN_HARD  (0 = skip, dead code off A100/H100/H200)"
echo "  kaolin hard       : $KAOLIN_HARD  (1 = install, top-level import)"
echo "  xformers hard     : $XFORMERS_HARD  (0 = skip; default ATTN_BACKEND=sdpa)"
echo "  sam-3d-objects    : $SAM3D_OBJ_DIR"
echo "=============================================================="

# --- sanity: required source trees + tools exist -----------------------------
for d in "$SAM3D_OBJ_DIR" "$MASK_PKG_DIR" "$SURFACE_VOXEL_PKG_DIR" "$TEXTURE_PKG_DIR"; do
    [ -d "$d" ] || { echo "[error] missing package dir: $d" >&2; exit 1; }
done
[ -f "$SAM3D_OBJ_DIR/pyproject.toml" ] || {
    echo "[error] sam-3d-objects pyproject.toml missing under $SAM3D_OBJ_DIR" >&2; exit 1; }
[ -f "$REQ_BASE" ]  || { echo "[error] requirements.txt missing: $REQ_BASE" >&2; exit 1; }
[ -f "$REQ_P3D" ]   || { echo "[error] requirements.p3d.txt missing: $REQ_P3D" >&2; exit 1; }
[ -f "$REQ_INFER" ] || { echo "[error] requirements.inference.txt missing: $REQ_INFER" >&2; exit 1; }
[ -x "$HYDRA_PATCH" ] || {
    echo "[error] hydra patch script missing/not executable: $HYDRA_PATCH" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "[error] interpreter '$PYTHON_BIN' not found on PATH" >&2
    echo "        install python3.11 (upstream pins 3.11) or set PYTHON_BIN=." >&2
    exit 1; }

# Offline deps bundle must be pulled first (tos_pull_sam3d_cu118_deps.sh).
KAOLIN_WHL="$(ls "$SAM3D_DEPS_DIR"/kaolin-*.whl 2>/dev/null | head -1 || true)"
TORCH_WHL="$(ls "$SAM3D_DEPS_DIR"/wheelhouse/torch-*.whl 2>/dev/null | head -1 || true)"
if [ ! -d "$SAM3D_DEPS_DIR" ] || [ -z "$KAOLIN_WHL" ] || \
   [ ! -d "$SAM3D_DEPS_DIR/pytorch3d" ] || [ ! -d "$SAM3D_DEPS_DIR/gsplat" ] || \
   [ ! -d "$SAM3D_DEPS_DIR/MoGe" ] || [ ! -d "$SAM3D_DEPS_DIR/utils3d" ]; then
    echo "[error] sam3d cu118 deps bundle missing/incomplete at: $SAM3D_DEPS_DIR" >&2
    echo "        expected: kaolin-*.whl + pytorch3d/ + gsplat/ + MoGe/ + utils3d/" >&2
    echo "        pull it first:  bash scripts/ops/tos/tos_pull_sam3d_cu118_deps.sh" >&2
    echo "        (or set SAM3D_DEPS_DIR=/path/to/bundle)" >&2
    exit 1
fi
if [ -z "$TORCH_WHL" ] && [ "$ALLOW_ONLINE_WHEELHOUSE" != "1" ]; then
    echo "[error] offline wheelhouse missing/incomplete at: $WHEELHOUSE" >&2
    echo "        expected: wheelhouse/torch-*.whl for offline install." >&2
    echo "        set ALLOW_ONLINE_WHEELHOUSE=1 only for Docker/online builds that can use PIP_INDEX_URL." >&2
    exit 1
fi
echo "[info] deps bundle    : $SAM3D_DEPS_DIR (kaolin wheel: $(basename "$KAOLIN_WHL"))"
echo "[info] arch list      : TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (H20=sm_90)"

# nvcc is needed to compile pytorch3d / gsplat (and flash_attn if forced). A
# conda env would bundle the CUDA toolkit; a venv relies on the system one.
# The target is cu118, and torch extension builds reject a newer toolkit with:
# "detected CUDA version (...) mismatches the version used to compile PyTorch".
# So CUDA 11.8 is required for local source builds.
if [ "${SKIP_NVCC_CHECK:-0}" != "1" ]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "[warn] nvcc not found on PATH — pytorch3d/gsplat CANNOT compile." >&2
        echo "       Install the CUDA 11.8 toolkit (or 'module load cuda/11.8') and re-run," >&2
        echo "       or set SKIP_NVCC_CHECK=1 if a toolkit is provided another way." >&2
    else
        nvcc_ver="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
        echo "[info] nvcc: $(nvcc --version | tail -1)  (CUDA ${nvcc_ver:-?})"
        nvcc_major="${nvcc_ver%%.*}"
        nvcc_minor="${nvcc_ver#*.}"
        if [ "${nvcc_major:-0}" -ne 11 ] || [ "${nvcc_minor:-0}" -ne 8 ]; then
            echo "[error] nvcc is CUDA ${nvcc_ver:-<unknown>}, but this cu118 venv must build extensions with CUDA 11.8." >&2
            echo "        Install a local toolkit or point CUDA_HOME/PATH at one, e.g.:" >&2
            echo "          export CUDA_HOME=\$HOME/.local/cuda-11.8" >&2
            echo "          export PATH=\$CUDA_HOME/bin:\$PATH" >&2
            echo "          export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}" >&2
            echo "        Set SKIP_NVCC_CHECK=1 only if you are not building pytorch3d/gsplat." >&2
            exit 1
        fi
    fi
fi

# -----------------------------------------------------------------------------
# 1. Create venv + upgrade pip tooling
# -----------------------------------------------------------------------------
echo "[1/7] creating venv at $VENV_DIR"
if [ "$VENV_SYSTEM_SITE_PACKAGES" = "1" ]; then
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
# OFFLINE bootstrap: pip/setuptools/wheel from the bundle's wheelhouse, NOT any
# index. The box's mirror was observed to lack 'wheel', and there is no public
# internet — so do NOT touch download.pytorch.org or pypi.org here.
# Keep setuptools below 81: lightning 2.3.3 still imports pkg_resources, which
# setuptools 81+ no longer vendors. If the offline wheelhouse has no older
# setuptools, preserve the venv seed version (Python 3.11 uv seed is 79.x).
if [ -n "$TORCH_WHL" ]; then
    python -m pip install --no-index --find-links "$WHEELHOUSE" -U pip wheel
else
    python -m pip install -U pip wheel
fi
if [ -d "$WHEELHOUSE" ] && find "$WHEELHOUSE" -maxdepth 1 -type f -name 'setuptools-*.whl' | grep -Eq 'setuptools-([0-7][0-9]|80)\\.'; then
    python -m pip install --no-index --find-links "$WHEELHOUSE" -U "setuptools<81"
elif python -c "import pkg_resources" >/dev/null 2>&1; then
    echo "[1/7] setuptools: preserving venv seed with pkg_resources"
else
    echo "[error] setuptools lacks pkg_resources and no setuptools<81 wheel exists in $WHEELHOUSE" >&2
    exit 1
fi
echo "[1/7] python: $(python --version 2>&1)  pip: $(pip --version)"

# -----------------------------------------------------------------------------
# A. torch — cu118 trio (== arts-gen). Install FIRST so every later build
#    (pytorch3d / gsplat / flash_attn) and the kaolin wheel resolve against the
#    already-present torch 2.4.0+cu118 rather than dragging in a cu12* build.
# -----------------------------------------------------------------------------
echo "[2/7] (A) torch trio + spconv-cu118 — FULLY OFFLINE from the wheelhouse"
# torch + spconv-cu118 AND their full dep closure (nvidia-*-cu11, cumm-cu118,
# pure-python) are all in the wheelhouse, so install with --no-index: NOTHING is
# fetched from any index here (no public net, mirror lacks the +cu118 wheels).
if [ -n "$TORCH_WHL" ]; then
    pip install --no-index --find-links "$WHEELHOUSE" torch torchvision torchaudio
    echo "[2/7] spconv-cu118 (offline)"
    pip install --no-index --find-links "$WHEELHOUSE" spconv-cu118
else
    echo "[2/7] torch trio — using system/inherited packages or configured pip indexes"
    pip install torch torchvision torchaudio --extra-index-url "$TORCH_CUDA_INDEX"
    echo "[2/7] spconv-cu118 (online/index)"
    pip install spconv-cu118
fi

# -----------------------------------------------------------------------------
# B. sam3d base deps WITHOUT the cu121 pins. requirements.txt is rewritten on the
#    fly into a cu118 variant:
#      - spconv-cu121      -> spconv-cu118   (2.3.x cu118 wheels exist)
#      - +cu121            -> +cu118         (any local-version pin)
#      - drop torch/torchaudio/torchvision   (installed in step A)
#      - drop nvidia-cuda-nvcc-cu12 / cuda-python==  (cu12 toolchain pins; on
#        cu118 we rely on the system 11.8 nvcc, not a pip-shipped cu12 nvcc)
#      - drop nvidia-pyindex (install-time side effect that edits pip config;
#        not a runtime dependency, and it fails under isolated wheel builds)
#      - drop flash_attn here only if it lives in this file (it does not — it is
#        in requirements.p3d.txt; handled in step C). Same for kaolin (inference).
#    Then pip install -r the filtered file with the caller's configured pip
#    index, and pin xformers explicitly to the arts-gen cu118 wheel.
# -----------------------------------------------------------------------------
echo "[3/7] (B) sam3d base deps — cu118-filtered requirements.txt"
REQ_CU118="$(mktemp -t sam3d_req_cu118.XXXXXX.txt)"
trap 'rm -f "$REQ_CU118"' EXIT
sed -e 's/spconv-cu121/spconv-cu118/' \
    -e 's/+cu121/+cu118/g' \
    "$REQ_BASE" \
  | grep -vE '^[[:space:]]*nvidia-cuda-nvcc-cu12' \
  | grep -vE '^[[:space:]]*nvidia-pyindex([=<>!+ ].*)?$' \
  | grep -vE '^[[:space:]]*cuda-python==' \
  | grep -vE '^[[:space:]]*torch(audio|vision)?([=<>!+ ].*)?$' \
  | grep -vE '^[[:space:]]*xformers([=<>!+ ].*)?$' \
  | grep -vE '^[[:space:]]*spconv-cu1[0-9]+' \
  | grep -vE '^[[:space:]]*MoGe[[:space:]]*@' \
  > "$REQ_CU118"
echo "[3/7] filtered requirements -> $REQ_CU118 (cu121 pins removed):"
echo "      removed: nvidia-cuda-nvcc-cu12*, cuda-python==*, torch/torchaudio/torchvision, xformers"
echo "      removed: nvidia-pyindex (pip config helper, not runtime)"
echo "      rewrote: spconv-cu121 -> spconv-cu118, +cu121 -> +cu118"
pip install -r "$REQ_CU118"
# xformers is OPTIONAL (default attention backend is torch SDPA). The cu118
# xformers wheel is coupled to torch 2.4.0 and would otherwise downgrade the
# torch 2.5.1 stack that the bundled kaolin wheel expects, so skip by default.
if [ "$XFORMERS_HARD" = "1" ]; then
    echo "[3/7] XFORMERS_HARD=1 — installing --no-deps: $XFORMERS_SPEC"
    # shellcheck disable=SC2086
    pip install --no-deps $XFORMERS_SPEC --extra-index-url "$TORCH_CUDA_INDEX" \
      || echo "[warn] xformers not installed (optional; default ATTN_BACKEND=sdpa works)."
else
    echo "[3/7] xformers — SKIPPED (XFORMERS_HARD=0; default ATTN_BACKEND=sdpa)."
fi

# -----------------------------------------------------------------------------
# C. compiled-from-source against cu118 nvcc (consistent with cu118 torch):
#      - pytorch3d  : HARD. The pinned git ref from requirements.p3d.txt, built
#                     --no-build-isolation so it sees the already-installed
#                     cu118 torch (2-core via MAX_JOBS/MAKEFLAGS).
#      - gsplat     : HARD. The pinned git ref from requirements.inference.txt,
#                     same build flags.
#      - flash_attn : OPTIONAL — only attempted if FLASH_ATTN_HARD=1; dead code
#                     off A100/H100/H200. On failure we print a clear note and
#                     CONTINUE (it is optional), so set -e is locally relaxed.
#      - kaolin     : HARD — fetched as a prebuilt cu118 wheel from NVIDIA's
#                     kaolin index for (torch 2.4.0, cu118). If absent, note it.
#    Git refs are pulled straight out of the upstream requirements files so they
#    stay in lock-step with the pins (no hard-coded SHAs to drift).
# -----------------------------------------------------------------------------
echo "[4/7] (C) source builds from the OFFLINE bundle — pytorch3d + gsplat + MoGe"
echo "      (compile with the box's cu118 nvcc for sm_90; slow, 2-core)"
# MoGe first (pure python+torch, fast) so a typo here fails before the long builds.
# MoGe declares utils3d as a GitHub URL dependency; install the bundled utils3d
# source and MoGe --no-deps so offline/Docker builds never clone github.com.
echo "[4/7] utils3d (bundled source for MoGe)"
pip install --no-build-isolation "$SAM3D_DEPS_DIR/utils3d"
echo "[4/7] MoGe dependencies (bundled requirements, GitHub URL removed)"
MOGE_REQ="$(mktemp -t moge_req.XXXXXX.txt)"
grep -vE '^[[:space:]]*(git\+https://github.com/EasternJournalist/utils3d.git|utils3d[[:space:]]*@)' \
    "$SAM3D_DEPS_DIR/MoGe/requirements.txt" > "$MOGE_REQ"
pip install -r "$MOGE_REQ"
rm -f "$MOGE_REQ"
echo "[4/7] MoGe (bundled source, deps already handled)"
pip install --no-build-isolation --no-deps "$SAM3D_DEPS_DIR/MoGe"
echo "[4/7] pytorch3d (bundled source @ sam3d's pinned commit) — COMPILES, slow"
pip install --no-build-isolation "$SAM3D_DEPS_DIR/pytorch3d"
echo "[4/7] gsplat (bundled source @ sam3d's pinned commit, glm submodule incl.) — COMPILES"
pip install --no-build-isolation "$SAM3D_DEPS_DIR/gsplat"

# flash_attn — OPTIONAL on cu118. Default: SKIP (dead code off A100/H100/H200).
if [ "$FLASH_ATTN_HARD" = "1" ]; then
    FA_SPEC="$(grep -E '^[[:space:]]*flash_attn==' "$REQ_P3D" | head -1 | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    FA_SPEC="${FA_SPEC:-flash_attn==2.8.3}"
    echo "[4/7] FLASH_ATTN_HARD=1 — attempting source build: $FA_SPEC (cu118 torch)"
    if pip install --no-build-isolation "$FA_SPEC"; then
        echo "[4/7] flash_attn installed."
    else
        echo "[warn] flash_attn build FAILED. It is OPTIONAL on cu118 (only used on" >&2
        echo "       A100/H100/H200; default ATTN_BACKEND=sdpa) so the stack still works." >&2
        echo "       If you truly need it: use a cu118 PREBUILT wheel, or a flash_attn" >&2
        echo "       version matching torch 2.4.0+cu118 (the pinned 2.8.3 targets newer" >&2
        echo "       torch). Re-run with FLASH_ATTN_HARD=0 to skip it cleanly." >&2
    fi
else
    echo "[4/7] flash_attn — SKIPPED (FLASH_ATTN_HARD=0; arts-gen ships none either)."
fi

# kaolin — HARD import (flexicubes). Prebuilt cu118 wheel from the OFFLINE bundle.
echo "[4/7] kaolin (bundled cu118 wheel): $(basename "$KAOLIN_WHL")"
KAOLIN_INSTALL_WHL="$KAOLIN_WHL"
if ! python - <<'PY' "$KAOLIN_INSTALL_WHL"
from packaging.utils import parse_wheel_filename
from pathlib import Path
import sys
parse_wheel_filename(Path(sys.argv[1]).name)
PY
then
    KAOLIN_FIXED_WHL="$(mktemp -t kaolin-0.17.0-cp311-cp311-linux_x86_64.XXXXXX.whl)"
    rm -f "$KAOLIN_FIXED_WHL"
    KAOLIN_FIXED_WHL="${KAOLIN_FIXED_WHL%.whl}.whl"
    cp "$KAOLIN_WHL" "$KAOLIN_FIXED_WHL"
    KAOLIN_INSTALL_WHL="$KAOLIN_FIXED_WHL"
    echo "      normalized wheel filename for pip: $(basename "$KAOLIN_INSTALL_WHL")"
fi
# The bundled kaolin wheel is installed --no-deps because pip cannot resolve the
# non-standard local wheel filename directly. Install its pure-python runtime
# deps explicitly so top-level `import kaolin` can reach kaolin.io.gltf.
python -m pip install "ipycanvas" "ipyevents" "jupyter-client<8" "pygltflib" "warp-lang"
python -m pip install --no-deps "$KAOLIN_INSTALL_WHL"

# -----------------------------------------------------------------------------
# D. sam-3d-objects + the 3 stage wrappers, editable.
#    sam3d_objects is installed --no-deps ON PURPOSE: its requirements are the
#    cu121-pinned set already handled in B/C, and a plain editable install would
#    re-resolve them (dragging cu121 wheels back in). The stage wrappers' deps
#    (numpy/torch/hydra/omegaconf/trimesh/pillow) are already satisfied.
# -----------------------------------------------------------------------------
echo "[5/7] (D) editable installs — sam3d_objects (--no-deps) + stage wrappers"
pip install -e "$SAM3D_OBJ_DIR" --no-deps
pip install -e "$MASK_PKG_DIR"
pip install -e "$SURFACE_VOXEL_PKG_DIR"
pip install -e "$TEXTURE_PKG_DIR"

# -----------------------------------------------------------------------------
# E. hydra patch (same as the cu121 script). patching/hydra swaps in a hydra
#    core/utils.py fix not yet on PyPI that instantiate(config) relies on. It
#    asserts hydra==1.3.2 and downloads the file over the network.
# -----------------------------------------------------------------------------
echo "[6/7] (E) hydra patch"
if [ -f "$LOCAL_HYDRA_CORE_UTILS" ]; then
    python - "$LOCAL_HYDRA_CORE_UTILS" <<'PY'
import os
import shutil
import sys

import hydra

if hydra.__version__ != "1.3.2":
    raise RuntimeError("different hydra version has been found, cannot patch")

src = sys.argv[1]
dst = os.path.join(os.path.dirname(hydra.__file__), "core", "utils.py")
shutil.copyfile(src, dst)
print(f"[6/7] hydra core/utils.py patched from local vendor file: {src}")
PY
else
    python "$HYDRA_PATCH"
fi

# -----------------------------------------------------------------------------
# F. Smoke imports — sam3d wrappers AND trellis part flow, all in this one venv.
#    LIDRA_SKIP_INIT=true short-circuits the missing sam3d_objects.init module.
#    Top-up the couple of small pure-python trellis deps defensively first
#    (safetensors/easydict/trimesh) — no-ops if already pulled by sam3d's reqs.
# -----------------------------------------------------------------------------
echo "[7/7] (F) trellis top-up + smoke imports (sam3d + trellis, one venv)"
pip install "safetensors" "easydict" "trimesh"   # harmless no-ops if already present
LIDRA_SKIP_INIT=true python -c "import sam3d_objects; from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap; print('sam3d_objects OK')"
LIDRA_SKIP_INIT=true python -c "from surface_voxel.pipeline import SurfaceVoxelPipeline; from texture.pipeline import TexturePipeline; print('stage wrappers OK')"
PYTHONPATH="$REPO_ROOT/TRELLIS-arts" ATTN_BACKEND=sdpa python -c "import inference; from trellis.models.part_flow import PartSSLatentFlowModel; from trellis.models.sparse_structure_vae import SparseStructureDecoder; print('trellis part-flow OK')"

echo "[done] UNIFIED cu118 venv at $VENV_DIR  (runs ALL 4 infer stages: ss/part/slat/assemble + the platform)"
echo "       activate : source \"$VENV_DIR/bin/activate\""
echo "       run anything trellis with:  PYTHONPATH=$REPO_ROOT/TRELLIS-arts python ..."
echo "       weights  : /robot/data-lab/jzh/art-gen/weights  (= checkpoints/hf)"
echo "       config   : <weights>/pipeline.yaml"
echo "       flash_attn: ${FLASH_ATTN_HARD} (0 = not installed, dead code off A100/H100/H200)"
echo "       NOTE: the cu121 variant is scripts/ops/setup/setup_sam3d_env.sh (DEV cu121 box only)."
