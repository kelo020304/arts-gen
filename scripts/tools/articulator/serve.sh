#!/usr/bin/env bash
# Serve the web editor on http://localhost:8000/.
# Run from inside the arts-gen conda env so /api/preprocess can find
# numpy/scipy/Blender. If you forget, the warning below tells you.
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "arts-gen" ]]; then
  echo "[warning] not in arts-gen env — /api/preprocess will likely fail" >&2
  echo "[warning] activate first:  conda activate arts-gen" >&2
fi
echo "[articulator] serving on http://localhost:${PORT}/"
exec python serve.py
