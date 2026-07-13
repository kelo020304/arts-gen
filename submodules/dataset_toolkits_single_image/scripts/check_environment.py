#!/usr/bin/env python3
"""Strict environment preflight for dataset_toolkits.

This script intentionally fails fast instead of applying fallbacks. It verifies
that the current Python process is the single official conda environment and
that all local model/code/data paths referenced by the config are present.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "utils"))

from config_loader import PipelineConfig, load_config  # noqa: E402


EXPECTED_CONDA_ENV = "dataset_toolkits"
DINO_CHECKPOINT_BY_MODEL = {
    "dinov2_vitl14_reg": "dinov2_vitl14_reg4_pretrain.pth",
}
REQUIRED_IMPORTS = (
    "numpy",
    "yaml",
    "PIL",
    "matplotlib",
    "trimesh",
    "open3d",
    "torch",
    "torchvision",
    "tqdm",
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly verify the official dataset_toolkits runtime environment."
    )
    parser.add_argument("--config", required=True, help="Dataset YAML config to validate.")
    return parser.parse_args(argv)


def result(name: str, ok: bool, detail: str) -> CheckResult:
    return CheckResult(name=name, ok=ok, detail=detail)


def require_absolute_dir(raw_path: str, label: str) -> CheckResult:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return result(label, False, f"not absolute: {raw_path}")
    if not path.is_dir():
        return result(label, False, f"directory missing: {path}")
    return result(label, True, str(path))


def require_absolute_file(raw_path: str, label: str) -> CheckResult:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return result(label, False, f"not absolute: {raw_path}")
    if not path.is_file():
        return result(label, False, f"file missing: {path}")
    return result(label, True, str(path))


def require_trellis_checkpoint_prefix(raw_path: str, label: str) -> CheckResult:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return result(label, False, f"not absolute: {raw_path}")
    if path.exists():
        return result(label, True, str(path))
    json_path = path.with_suffix(".json")
    weights_path = path.with_suffix(".safetensors")
    if json_path.is_file() and weights_path.is_file():
        return result(label, True, f"{json_path} + {weights_path}")
    return result(label, False, f"missing checkpoint pair: {json_path}, {weights_path}")


def check_conda_env(expected: str) -> CheckResult:
    current = os.environ.get("CONDA_DEFAULT_ENV")
    if current != expected:
        return result("conda env", False, f"expected {expected!r}, got {current!r}")
    return result("conda env", True, current)


def check_imports() -> list[CheckResult]:
    checks: list[CheckResult] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            checks.append(result(f"import {module_name}", False, repr(exc)))
            continue
        version = getattr(module, "__version__", "unknown-version")
        checks.append(result(f"import {module_name}", True, str(version)))
    return checks


def check_torch_cuda() -> CheckResult:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return result("torch cuda", False, f"torch import failed: {exc!r}")
    if not torch.cuda.is_available():
        return result("torch cuda", False, "torch.cuda.is_available() is false")
    return result("torch cuda", True, f"torch={torch.__version__} cuda={torch.version.cuda}")


def check_blender(cfg: PipelineConfig) -> CheckResult:
    blender = cfg.render.blender
    path = Path(blender)
    if not path.is_absolute():
        return result("blender", False, f"not absolute: {blender}")
    if not path.is_file():
        return result("blender", False, f"file missing: {path}")
    if not os.access(path, os.X_OK):
        return result("blender", False, f"not executable: {path}")
    return result("blender", True, str(path))


def check_data_contract(cfg: PipelineConfig) -> list[CheckResult]:
    data_root = Path(cfg.data_root)
    return [
        require_absolute_dir(str(data_root), "data_root"),
        require_absolute_dir(cfg.finaljson_dir, "raw/finaljson"),
        require_absolute_dir(cfg.partseg_dir, "raw/partseg"),
    ]


def check_dinov2(cfg: PipelineConfig) -> list[CheckResult]:
    checks = [
        require_absolute_dir(cfg.feature.dinov2_repo, "feature.dinov2_repo"),
        require_absolute_dir(cfg.feature.torch_hub_dir, "feature.torch_hub_dir"),
    ]
    checkpoint_name = DINO_CHECKPOINT_BY_MODEL.get(cfg.feature.model)
    if checkpoint_name is None:
        checks.append(
            result(
                "feature.model",
                False,
                f"unsupported model {cfg.feature.model!r}; supported={sorted(DINO_CHECKPOINT_BY_MODEL)}",
            )
        )
        return checks
    checkpoint = Path(cfg.feature.torch_hub_dir) / "checkpoints" / checkpoint_name
    checks.append(require_absolute_file(str(checkpoint), "DINOv2 checkpoint"))
    return checks


def check_trellis_import(cfg: PipelineConfig) -> CheckResult:
    trellis_root = Path(cfg.trellis.root)
    if not trellis_root.is_absolute():
        return result("TRELLIS import", False, f"trellis.root not absolute: {trellis_root}")
    if not trellis_root.is_dir():
        return result("TRELLIS import", False, f"trellis.root missing: {trellis_root}")
    sys.path.insert(0, str(trellis_root))
    try:
        trellis_pkg_dir = trellis_root / "trellis"
        if trellis_pkg_dir.is_dir() and "trellis" not in sys.modules:
            pkg = types.ModuleType("trellis")
            pkg.__path__ = [str(trellis_pkg_dir)]  # type: ignore[attr-defined]
            pkg.__file__ = str(trellis_pkg_dir / "__init__.py")
            sys.modules["trellis"] = pkg
        import trellis.models  # noqa: F401,PLC0415
        import trellis.modules.sparse  # noqa: F401,PLC0415
        import utils3d.torch  # noqa: F401,PLC0415
    except Exception as exc:  # noqa: BLE001
        return result("TRELLIS import", False, repr(exc))
    return result("TRELLIS import", True, str(trellis_root))


def check_trellis_assets(cfg: PipelineConfig) -> list[CheckResult]:
    return [
        require_absolute_dir(cfg.trellis.root, "trellis.root"),
        require_trellis_checkpoint_prefix(cfg.trellis.ss_encoder, "trellis.ss_encoder"),
        require_trellis_checkpoint_prefix(cfg.trellis.ss_decoder, "trellis.ss_decoder"),
        require_trellis_checkpoint_prefix(cfg.trellis.slat_encoder, "trellis.slat_encoder"),
        check_trellis_import(cfg),
    ]


def load_config_check(config_path: str) -> tuple[PipelineConfig | None, CheckResult]:
    try:
        cfg = load_config(config_path)
    except Exception as exc:  # noqa: BLE001
        return None, result("config", False, repr(exc))
    return cfg, result("config", True, config_path)


def print_results(checks: list[CheckResult]) -> None:
    width = max(len(check.name) for check in checks) if checks else 10
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"[{status}] {check.name:<{width}} {check.detail}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks: list[CheckResult] = [check_conda_env(EXPECTED_CONDA_ENV)]
    checks.extend(check_imports())
    checks.append(check_torch_cuda())

    cfg, config_result = load_config_check(args.config)
    checks.append(config_result)
    if cfg is not None:
        checks.extend(check_data_contract(cfg))
        checks.append(check_blender(cfg))
        checks.extend(check_dinov2(cfg))
        checks.extend(check_trellis_assets(cfg))

    print_results(checks)
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
