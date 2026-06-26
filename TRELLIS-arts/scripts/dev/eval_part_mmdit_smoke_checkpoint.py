#!/usr/bin/env python3
"""Evaluate PartMMDiT smoke metrics from a checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_PATH = PROJECT_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))

_pkg = types.ModuleType("trellis")
_pkg.__path__ = [str(TRELLIS_PATH / "trellis")]
_pkg.__package__ = "trellis"
sys.modules.setdefault("trellis", _pkg)
for sp in ("models", "modules", "trainers", "utils", "datasets"):
    module = types.ModuleType(f"trellis.{sp}")
    module.__path__ = [str(TRELLIS_PATH / "trellis" / sp)]
    module.__package__ = f"trellis.{sp}"
    sys.modules.setdefault(f"trellis.{sp}", module)
pipelines = types.ModuleType("trellis.pipelines")
pipelines.__path__ = [str(TRELLIS_PATH / "trellis" / "pipelines")]
pipelines.__package__ = "trellis.pipelines"
sys.modules.setdefault("trellis.pipelines", pipelines)

from trellis.datasets.arts.part_mmdit import PartMMDiTDataset  # noqa: E402
from trellis.models.part_flow import PartMMDiTModel  # noqa: E402
from trellis.trainers.arts.part_mmdit import _eval_smoke_curve  # noqa: E402
from trellis.utils.arts.config_utils import config_to_dict, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    cfg_dict = config_to_dict(cfg)
    dataset = PartMMDiTDataset(cfg_dict["data"])
    model = PartMMDiTModel(**cfg_dict["model"]).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f"{args.ckpt} missing key 'model'")
    model.load_state_dict(ckpt["model"])
    metrics = _eval_smoke_curve(
        model,
        dataset,
        device,
        cfg_dict["flow"],
        cfg_dict["eval"],
    )
    out = {
        "ckpt": str(args.ckpt),
        "cos_single": metrics["cos_single"],
        "cos_multi": metrics["cos_multi"],
        "cos_buttons": metrics["cos_buttons"],
        "vel_err_button_0_t0.02": metrics["vel_err_button_0_t0.02"],
        "vel_err_lid_0_t0.02": metrics["vel_err_lid_0_t0.02"],
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
