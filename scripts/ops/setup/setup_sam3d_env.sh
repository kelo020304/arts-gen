#!/usr/bin/env bash
# =============================================================================
# setup_sam3d_env.sh — build the ONE UNIFIED venv for the whole inference stack.
# =============================================================================
# This is the single environment for ALL 4 full-pipeline-inference stages
# (ss / part / slat / assemble) AND the eval platform — NO separate arts-gen
# conda env, NO second sam3d venv, NO cross-env subprocess. sam3d_objects and
# TRELLIS-arts (part flow) coexist here on cu121 / torch 2.5.1; the heavy deps
# part flow needs (spconv-cu121, flash_attn, xformers, ...) are exactly the ones
# sam3d already pins, so they are shared. TRELLIS-arts is used via PYTHONPATH.
#
# WHERE THIS RUNS
#   *** DEV MACHINE ONLY (cu121 GPU). DO NOT RUN ON THE cu118 WORKSTATION. ***
#   The whole sam3d_objects stack is pinned to CUDA 12.1 (torch 2.5.1+cu121,
#   spconv-cu121, xformers 0.0.28.post3, flash_attn 2.8.3, kaolin cu121, etc.).
#   The arts-gen workstation is CUDA 11.8 — installing this there produces a
#   broken env (ABI / driver mismatch) that fails at import or first CUDA call.
#   pytorch3d / flash_attn build CUDA extensions and want a GPU visible AT BUILD
#   TIME (upstream doc/setup.md: "you may have to build on a compute node with
#   GPU"), so build this on a GPU node, not a CPU-only login shell.
#
# CONDA -> VENV
#   Upstream ships a conda env (environments/default.yml) that bundles
#   python=3.11 + the CUDA 12.1 toolkit + c/cxx compilers. We want a *venv*
#   (to ship to dev), so those come from the SYSTEM instead:
#     - a python3.11 interpreter on PATH (match upstream's pinned 3.11; every
#       cu121 wheel below — torch/spconv-cu121/xformers/flash_attn/kaolin — is
#       built for the 3.11 ABI, so DO NOT silently downgrade to 3.10),
#     - a CUDA 12.1 toolkit with nvcc on PATH (to compile pytorch3d/flash_attn/
#       gsplat). This script checks for nvcc and aborts if it is missing.
#
# WHAT IT BUILDS  (follows upstream doc/setup.md ordering exactly)
#   A python3.11 venv with:
#     - core sam-3d-objects + requirements.txt + dev extras   (pip install -e '.[dev]')
#       torch 2.5.1+cu121 lands here, pulled via PIP_EXTRA_INDEX_URL.
#     - pytorch3d + flash_attn                                 (pip install -e '.[p3d]')
#       compiled against the torch installed by the [dev] step (2-step install:
#       pytorch3d's torch dep resolves badly as a transitive — doc/setup.md).
#     - kaolin (prebuilt cu121 wheel) + gsplat + gradio        (pip install -e '.[inference]')
#       kaolin grabbed from PIP_FIND_LINKS (nvidia-kaolin cu121/torch2.5.1).
#     - the upstream hydra patch                               (./patching/hydra)
#     - the 3 stage wrapper packages: generate_mask / surface_voxel / texture
#   Then a smoke import of InferencePipelinePointMap + the stage wrappers.
#
# WEIGHTS / CONFIG (not installed here — runtime assets, pulled separately)
#   weights : /robot/data-lab/jzh/art-gen/weights   (== sam-3d-objects checkpoints/hf)
#             fetched by scripts/ops/tos/tos_pull_sam3d_weights.sh.
#   config  : <that weights dir>/pipeline.yaml       (passed to SurfaceVoxel/
#             TexturePipeline as config_path; instantiate() reads it via hydra).
#   The sam3d wrappers load the pipeline with:
#       os.environ['LIDRA_SKIP_INIT']='true'  (BEFORE `import sam3d_objects`)
#       config = OmegaConf.load(pipeline.yaml)
#       config.rendering_engine = 'pytorch3d'
#       config.compile_model    = False
#       config.workspace_dir    = str(config_path.parent)
#       config.device           = device
#       pipe = hydra.utils.instantiate(config)
#
# SS-LATENT CONTRACT (for the glue that consumes this env, not used by setup)
#   sample_sparse_structure() returns return_dict["shape"] of shape [bs,4096,8];
#   reshaped to z_global via .permute(0,2,1).contiguous().view(bs,8,16,16,16).
#   The dataset's z_global (reconstruction/ss_latents_expanded/<id>/angle_<a>/
#   latent.npz, key "mean") is (8,16,16,16) float32 — so the SS-stage output
#   ss_latent.npy is a plain (8,16,16,16) float32 array (bs=1, squeezed).
#   return_dict["coords"] is the decoded voxel [M,4] int (batch + xyz).
#
# CONVENTIONS
#   - set -eo pipefail (NOT -u): this shell's startup snapshot references an
#     unbound $ZSH_VERSION, and -u makes $(...) sub-shells crash silently. See
#     the project gotchas. Hard failures still surface because set -e is on.
#   - No silent fallbacks: every step either succeeds or aborts with set -e.
#   - Compile rule: heavy builds run under MAX_JOBS=2 + 2-core make/cmake.
#
# USAGE
#   bash scripts/ops/setup/setup_sam3d_env.sh [VENV_DIR]
#   VENV_DIR        positional, default ".venv/sam3d" under the sam-3d-objects root.
#   env overrides:
#     PYTHON_BIN         python interpreter   (default python3.11)
#     TORCH_CUDA_INDEX   cu121 wheel index    (default download.pytorch.org/whl/cu121)
#     NGC_INDEX          NVIDIA NGC pip index (default pypi.ngc.nvidia.com)
#     KAOLIN_FIND_LINKS  kaolin wheel index   (default nvidia-kaolin cu121/torch2.5.1)
#     SKIP_NVCC_CHECK=1  bypass the nvcc presence check (not recommended)
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

# --- args / overrides --------------------------------------------------------
VENV_DIR="${1:-$SAM3D_OBJ_DIR/.venv/sam3d}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"
NGC_INDEX="${NGC_INDEX:-https://pypi.ngc.nvidia.com}"
# kaolin publishes per-(torch,cuda) wheel indices. This one matches torch 2.5.1
# + cu121, exactly what doc/setup.md pins via PIP_FIND_LINKS.
KAOLIN_FIND_LINKS="${KAOLIN_FIND_LINKS:-https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html}"

echo "=============================================================="
echo "[setup_sam3d_env] cu121 / GPU env — DEV MACHINE ONLY"
echo "  venv dir          : $VENV_DIR"
echo "  python            : $PYTHON_BIN"
echo "  torch index       : $TORCH_CUDA_INDEX"
echo "  ngc index         : $NGC_INDEX"
echo "  kaolin find-links : $KAOLIN_FIND_LINKS"
echo "  sam-3d-objects    : $SAM3D_OBJ_DIR"
echo "=============================================================="

# --- sanity: required source trees + tools exist -----------------------------
for d in "$SAM3D_OBJ_DIR" "$MASK_PKG_DIR" "$SURFACE_VOXEL_PKG_DIR" "$TEXTURE_PKG_DIR"; do
    [ -d "$d" ] || { echo "[error] missing package dir: $d" >&2; exit 1; }
done
[ -f "$SAM3D_OBJ_DIR/pyproject.toml" ] || {
    echo "[error] sam-3d-objects pyproject.toml missing under $SAM3D_OBJ_DIR" >&2; exit 1; }
[ -x "$HYDRA_PATCH" ] || {
    echo "[error] hydra patch script missing/not executable: $HYDRA_PATCH" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "[error] interpreter '$PYTHON_BIN' not found on PATH" >&2
    echo "        install python3.11 (upstream pins 3.11) or set PYTHON_BIN=." >&2
    exit 1; }

# nvcc is needed to compile pytorch3d / flash_attn / gsplat. A conda env would
# bundle the CUDA toolkit; a venv relies on the system one. Fail early with a
# clear message rather than deep inside a 20-minute compile.
if [ "${SKIP_NVCC_CHECK:-0}" != "1" ]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "[error] nvcc not found on PATH — cannot compile pytorch3d/flash_attn/gsplat." >&2
        echo "        Install the CUDA 12.1 toolkit (or 'module load cuda/12.1'), then re-run." >&2
        echo "        Override with SKIP_NVCC_CHECK=1 only if a toolkit is provided another way." >&2
        exit 1
    fi
    nvcc_ver="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
    echo "[info] nvcc: $(nvcc --version | tail -1)  (CUDA ${nvcc_ver:-?})"
    nvcc_major="${nvcc_ver%%.*}"
    if [ "${nvcc_major:-0}" -lt 12 ]; then
        echo "[error] nvcc is CUDA ${nvcc_ver:-<unknown>}, but this stack needs CUDA 12.x" >&2
        echo "        (torch 2.5.1+cu121; H20/Hopper sm_90 is unsupported by CUDA 11.x)." >&2
        echo "        Point PATH/CUDA_HOME at a 12.x toolkit, e.g.:" >&2
        echo "          ls -d /usr/local/cuda*        # find a 12.x install" >&2
        echo "          export CUDA_HOME=/usr/local/cuda-12.1" >&2
        echo "          export PATH=\$CUDA_HOME/bin:\$PATH" >&2
        echo "          export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >&2
        echo "        then re-run. Override with SKIP_NVCC_CHECK=1 only if you know better." >&2
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# 1. Create venv + upgrade pip tooling
# -----------------------------------------------------------------------------
echo "[1/6] creating venv at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
# Some pip mirrors intermittently fail to serve 'wheel' ("No matching
# distribution"). Fall back to the official PyPI index for the bootstrap.
python -m pip install -U pip setuptools wheel \
  || python -m pip install -U pip setuptools wheel -i https://pypi.org/simple/
echo "[1/6] python: $(python --version 2>&1)  pip: $(pip --version)"

# -----------------------------------------------------------------------------
# 2. Core install: `pip install -e '.[dev]'`  (doc/setup.md step 1)
#    Pulls requirements.txt (base) + requirements.dev.txt. torch 2.5.1+cu121,
#    spconv-cu121, xformers and the MoGe git dep resolve here, which is why the
#    cu121 + NGC indices must be visible via PIP_EXTRA_INDEX_URL (PIP_EXTRA_*
#    ADDS indices alongside PyPI, unlike --index-url which replaces it).
# -----------------------------------------------------------------------------
echo "[2/6] pip install -e '.[dev]'  (core + torch 2.5.1+cu121)"
export PIP_EXTRA_INDEX_URL="$NGC_INDEX $TORCH_CUDA_INDEX"
pip install -e "$SAM3D_OBJ_DIR[dev]"

# -----------------------------------------------------------------------------
# 3. pytorch3d + flash_attn: `pip install -e '.[p3d]'`  (doc/setup.md step 2)
#    Separate step ON PURPOSE — pytorch3d's torch dependency resolves badly as a
#    transitive, so it is installed after torch is already present. These COMPILE
#    CUDA extensions (slow); the 2-core cap from the MAX_JOBS/MAKEFLAGS exports
#    above applies here.
# -----------------------------------------------------------------------------
echo "[3/6] pip install -e '.[p3d]'  (pytorch3d + flash_attn — COMPILES, slow)"
pip install -e "$SAM3D_OBJ_DIR[p3d]"

# -----------------------------------------------------------------------------
# 4. Inference extras: `pip install -e '.[inference]'`  (doc/setup.md step 3)
#    kaolin==0.17.0 (prebuilt cu121/torch2.5.1 wheel via PIP_FIND_LINKS) +
#    gsplat (git, compiles) + gradio + seaborn. If kaolin's wheel is missing for
#    this ABI, this fails loudly (set -e) — fix KAOLIN_FIND_LINKS / the python
#    version, do NOT add a source-build fallback.
# -----------------------------------------------------------------------------
echo "[4/6] pip install -e '.[inference]'  (kaolin + gsplat + gradio)"
export PIP_FIND_LINKS="$KAOLIN_FIND_LINKS"
pip install -e "$SAM3D_OBJ_DIR[inference]"

# -----------------------------------------------------------------------------
# 5. hydra patch + the 3 stage wrapper packages.
#    patching/hydra swaps in a hydra core/utils.py fix not yet on PyPI that
#    instantiate(config) relies on. It asserts hydra==1.3.2 and downloads the
#    file over the network (so this step needs network access). Then install the
#    stage wrappers (generate_mask / surface_voxel / texture) editable — their
#    deps (numpy/torch/hydra/omegaconf/trimesh/pillow) are already satisfied.
# -----------------------------------------------------------------------------
echo "[5/7] hydra patch + stage wrapper packages"
python "$HYDRA_PATCH"
pip install -e "$MASK_PKG_DIR"
pip install -e "$SURFACE_VOXEL_PKG_DIR"
pip install -e "$TEXTURE_PKG_DIR"

# -----------------------------------------------------------------------------
# 6. UNIFIED VENV: make TRELLIS-arts (part flow / assemble / platform) run in
#    THIS SAME venv — no second conda/arts-gen env, no cross-env subprocess.
#    TRELLIS-arts has no setup.py; it is used via PYTHONPATH. The heavy deps
#    part flow needs (torch 2.5.1+cu121, spconv-cu121, flash_attn, xformers,
#    einops, safetensors, easydict) are ALREADY installed above by sam3d's
#    requirements — so we only top up a couple of small pure-python ones
#    defensively. The trellis renderer deps (nvdiffrast/diffoctreerast/vox2seq)
#    are NOT needed for inference: mesh/GS come from sam3d's decode_slat, and
#    part flow only runs decode_ss (SS-VAE threshold -> voxel coords).
# -----------------------------------------------------------------------------
echo "[6/7] unified venv: trellis-arts top-up deps"
pip install "safetensors" "easydict" "trimesh"   # harmless no-ops if already present

# -----------------------------------------------------------------------------
# 7. Smoke imports — sam3d wrappers AND trellis part flow, all in this venv.
#    LIDRA_SKIP_INIT=true short-circuits the missing sam3d_objects.init module.
# -----------------------------------------------------------------------------
echo "[7/7] verifying imports (sam3d + trellis, one venv)"
LIDRA_SKIP_INIT=true python -c "import sam3d_objects; from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap; print('sam3d_objects OK')"
LIDRA_SKIP_INIT=true python -c "from surface_voxel.pipeline import SurfaceVoxelPipeline; from texture.pipeline import TexturePipeline; print('stage wrappers OK')"
PYTHONPATH="$REPO_ROOT/TRELLIS-arts" python -c "from trellis.models.part_flow import PartSSLatentFlowModel; from trellis.models.sparse_structure_vae import SparseStructureDecoder; print('trellis part-flow OK')"

echo "[done] UNIFIED venv at $VENV_DIR  (runs ALL 4 infer stages: ss/part/slat/assemble + the platform)"
echo "       activate : source \"$VENV_DIR/bin/activate\""
echo "       run anything trellis with:  PYTHONPATH=$REPO_ROOT/TRELLIS-arts python ..."
echo "       weights  : /robot/data-lab/jzh/art-gen/weights  (= checkpoints/hf)"
echo "       config   : <weights>/pipeline.yaml"
echo "       NOTE: infer_stage.py runs the sam3d SS/SLat glue with sys.executable"
echo "             (this same venv) — no separate environment is opened."
