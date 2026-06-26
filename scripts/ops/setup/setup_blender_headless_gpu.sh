#!/usr/bin/env bash
# ============================================================
# setup_blender_headless_gpu.sh
#
# One-shot setup for Blender EEVEE_NEXT GPU rendering inside a headless
# VolcEngine ML container (4× RTX 4090 / driver 550.144.03).
#
# Why this is needed:
#   Container has CUDA compute but no GPU graphics path — /dev/dri missing,
#   nvidia-drm.modeset=N, cgroup blocks major 226. EGL surfaceless is dead.
#   Workaround: Xvfb virtual display + NVIDIA Vulkan driver. Renders go GPU
#   via Vulkan, never through DRM.
#
# What this installs:
#   1. apt packages: xvfb, vulkan-tools, mesa-utils
#   2. libnvidia-gpucomp.so.550.144.03 (download from internal TOS bucket).
#      NVIDIA Container Toolkit ships a CSV that doesn't list this lib —
#      missing it makes NVIDIA Vulkan driver fail with EGL_NOT_INITIALIZED.
#   3. /etc/vulkan/icd.d/nvidia_icd.json — Vulkan loader registration.
#   4. Xvfb on display :99 (background process).
#
# Idempotent. Re-run after container restart (apt + lib persist, Xvfb dies).
#
# Usage:
#   bash scripts/ops/setup/setup_blender_headless_gpu.sh
#
# After this, run_physx_mobility_cloud.sh sets env vars itself. For ad-hoc
# Blender calls export manually:
#   export __GLX_VENDOR_LIBRARY_NAME=nvidia
#   export __VK_LAYER_NV_optimus=NVIDIA_only
#   export DISPLAY=:99
#
# Verification:
#   DISPLAY=:99 vulkaninfo 2>&1 | grep -i NVIDIA   # should list 4090 device
#   blender --gpu-backend vulkan -b --python-expr "..."   # should run EEVEE
# ============================================================

set -euo pipefail

GPUCOMP_LIB_URL="${GPUCOMP_LIB_URL:-https://ml-platform.tos-vpc.cloud.vnet.com/rclone_tmp_dir/libnvidia-gpucomp.so.550.144.03}"
GPUCOMP_LIB_NAME="libnvidia-gpucomp.so.550.144.03"
GPUCOMP_LIB_DIR="/usr/lib/x86_64-linux-gnu"
ICD_PATH="/etc/vulkan/icd.d/nvidia_icd.json"

if [ "$(id -u)" -ne 0 ]; then
  echo "[error] must run as root (writes to /usr/lib and /etc)" >&2
  exit 1
fi

banner() {
  echo ""
  echo "============================================================"
  echo "  $1"
  echo "============================================================"
}

banner "[1/5] apt packages (xvfb, vulkan-tools, mesa-utils)"
if ! dpkg -l xvfb vulkan-tools mesa-utils >/dev/null 2>&1; then
  apt-get update
  apt-get install -y xvfb vulkan-tools mesa-utils
else
  echo "  already installed"
fi

banner "[2/5] libnvidia-gpucomp.so install"
if [ ! -f "$GPUCOMP_LIB_DIR/$GPUCOMP_LIB_NAME" ]; then
  echo "  downloading from $GPUCOMP_LIB_URL"
  TMP="$(mktemp -d)"
  ( cd "$TMP" && wget -q "$GPUCOMP_LIB_URL" -O "$GPUCOMP_LIB_NAME" )
  cp "$TMP/$GPUCOMP_LIB_NAME" "$GPUCOMP_LIB_DIR/"
  ( cd "$GPUCOMP_LIB_DIR" \
    && ln -sf "$GPUCOMP_LIB_NAME" libnvidia-gpucomp.so.1 \
    && ln -sf "$GPUCOMP_LIB_NAME" libnvidia-gpucomp.so )
  ldconfig
  rm -rf "$TMP"
  echo "  installed"
else
  echo "  already present at $GPUCOMP_LIB_DIR/$GPUCOMP_LIB_NAME"
fi

banner "[3/5] Vulkan ICD registration"
mkdir -p "$(dirname "$ICD_PATH")"
cat > "$ICD_PATH" <<'EOF'
{
 "file_format_version" : "1.0.0",
 "ICD": {
  "library_path": "libGLX_nvidia.so.0",
  "api_version" : "1.3.242"
 }
}
EOF
echo "  wrote $ICD_PATH"

banner "[4/5] Xvfb on :99"
if pgrep -x Xvfb >/dev/null 2>&1; then
  echo "  already running (pid $(pgrep -x Xvfb | head -1))"
else
  Xvfb :99 -screen 0 1024x768x24 &
  sleep 1
  echo "  started (pid $(pgrep -x Xvfb | head -1))"
fi

banner "[5/5] verification"
echo -n "  gpucomp lib:       "
[ -f "$GPUCOMP_LIB_DIR/$GPUCOMP_LIB_NAME" ] && echo "OK" || { echo "MISSING"; exit 1; }
echo -n "  ICD JSON:          "
[ -f "$ICD_PATH" ] && echo "OK" || { echo "MISSING"; exit 1; }
echo -n "  Xvfb process:      "
pgrep -x Xvfb >/dev/null 2>&1 && echo "OK" || { echo "MISSING"; exit 1; }
echo -n "  Vulkan sees NVIDIA: "
if DISPLAY=:99 vulkaninfo 2>/dev/null | grep -q "NVIDIA"; then
  GPUS=$(DISPLAY=:99 vulkaninfo 2>/dev/null | grep -c "NVIDIA")
  echo "OK ($GPUS GPU(s) listed)"
else
  echo "FAIL — vulkaninfo doesn't list NVIDIA device"
  echo ""
  echo "Diagnostic:"
  DISPLAY=:99 vulkaninfo 2>&1 | head -10
  exit 1
fi

echo ""
echo "============================================================"
echo "  SETUP DONE"
echo "============================================================"
echo ""
echo "To run Blender EEVEE manually (e.g. for one-off tests):"
echo "  export __GLX_VENDOR_LIBRARY_NAME=nvidia"
echo "  export __VK_LAYER_NV_optimus=NVIDIA_only"
echo "  export DISPLAY=:99"
echo "  blender --gpu-backend vulkan -b -P <script.py>"
echo ""
echo "Pipeline scripts (run_physx_mobility_cloud.sh) export these themselves."
echo ""
echo "Container restart? Just re-run this script. Idempotent; only Xvfb"
echo "needs to be respawned, the rest persists in /usr/lib and /etc."
