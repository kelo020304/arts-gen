#!/usr/bin/env python3
"""Compatibility wrapper for the renumbered pipeline.

Canonical entry point: `pipeline/04_build_valid_parts_manifest.py`.
This file is kept so older commands still execute the new implementation.
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "pipeline/04_build_valid_parts_manifest.py"
    print("[compat] utils/build_manifest.py has moved to pipeline/04_build_valid_parts_manifest.py", flush=True)
    runpy.run_path(str(target), run_name="__main__")
