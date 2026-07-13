#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    server_root = repo_root / "workbenches/fridge_3dgs/server"
    if str(server_root) not in sys.path:
        sys.path.insert(0, str(server_root))
    os.environ.setdefault("FRIDGE_3DGS_PORT", "7865")
    os.environ.setdefault("ARTS_GEN_DATA_ROOT", "/robot/data-lab/jzh/art-gen")
    from app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
