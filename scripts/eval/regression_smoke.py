#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts" / "dev") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "dev"))
PYTHON = Path(os.environ.get("ARTS_GEN_PYTHON", "/opt/venvs/arts-gen/bin/python"))
SMOKE_ROOT = Path(os.environ.get("ARTS_GEN_REGRESSION_ROOT", "/mnt/robot-data-lab/jzh/art-gen/regression-smoke/0626"))
BASELINE_JSON = Path("docs/runbooks/regression/baseline_0626.json")
BASELINE_MD = Path("docs/runbooks/regression/baseline_0626.md")
DEFAULT_GPU = os.environ.get("REGRESSION_SMOKE_GPU", "5")

SMOKE_OBJECT_ID = "0023687e90394c3e97ab19b0160cafb3"
SMOKE_DATA_ROOT = Path("/mnt/robot-data-lab/jzh/art-gen/data/phyx-verse")
SMOKE_IMAGES = [
    SMOKE_DATA_ROOT / "renders" / SMOKE_OBJECT_ID / "angle_0" / "rgb" / f"view_{idx}.png"
    for idx in range(4)
]
SMOKE_MASKS = [
    SMOKE_DATA_ROOT / "renders" / SMOKE_OBJECT_ID / "angle_0" / "mask" / f"mask_{idx}.npy"
    for idx in range(4)
]
SMOKE_PART_INFO = SMOKE_DATA_ROOT / "reconstruction" / "part_info" / SMOKE_OBJECT_ID / "part_info.json"

SS_FLOW_CKPT = Path(
    os.environ.get(
        "REGRESSION_SS_FLOW_CKPT",
        "/mnt/robot-data-lab/jzh/art-gen/ckpt/tre-ss-flow/tre-ss-concat-0616-1/ckpts/denoiser_ema0.999_step0012500.pt",
    )
)
PART_SEG_CKPT = Path(
    os.environ.get(
        "REGRESSION_PART_SEG_CKPT",
        "/mnt/robot-data-lab/jzh/art-gen/ckpt/part-prompt-seg/part_promptable_seg_full_S_0616-1/ckpts/step_50000.pt",
    )
)
PART_SEG_WARM = Path(
    os.environ.get(
        "REGRESSION_PART_SEG_WARM",
        "/mnt/robot-data-lab/jzh/art-gen/ckpt/part-prompt-seg/part_promptable_seg_full_S_0616-1/ckpts/step_50000.pt",
    )
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env={**os.environ, **(env or {})},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    proc.elapsed_seconds = time.time() - started  # type: ignore[attr-defined]
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed rc={proc.returncode}; log={log_path}\n"
            + "\n".join(proc.stdout.splitlines()[-40:])
        )
    return proc


def _coords_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.int64).reshape(-1, 3)
    gt = np.asarray(gt, dtype=np.int64).reshape(-1, 3)
    pred_set = set(map(tuple, pred.tolist()))
    gt_set = set(map(tuple, gt.tolist()))
    union = pred_set | gt_set
    if not union:
        return 1.0
    return float(len(pred_set & gt_set) / len(union))


def _load_gt_part_coords() -> dict[int, np.ndarray]:
    part_info = _read_json(SMOKE_PART_INFO)
    parts = part_info.get("parts")
    if not isinstance(parts, dict):
        raise ValueError(f"{SMOKE_PART_INFO} missing object 'parts'")
    out: dict[int, np.ndarray] = {}
    root = SMOKE_DATA_ROOT / "reconstruction" / "voxel_expanded" / SMOKE_OBJECT_ID / "angle_0" / "64"
    for part_key, part in parts.items():
        if not isinstance(part, dict) or "label" not in part:
            continue
        label = int(part["label"])
        path = root / f"ind_{part_key}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"GT part voxel file not found: {path}")
        out[label] = np.load(path).astype(np.int32)
    if not out:
        raise RuntimeError(f"no GT part coords loaded from {root}")
    return out


def _load_gt_surface_coords() -> np.ndarray:
    path = SMOKE_DATA_ROOT / "reconstruction" / "voxel_expanded" / SMOKE_OBJECT_ID / "angle_0" / "64" / "surface.npy"
    if not path.is_file():
        raise FileNotFoundError(f"GT surface voxel file not found: {path}")
    return np.load(path).astype(np.int32)


def chain_part_seg(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = SMOKE_ROOT / "part_seg_quick"
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        str(PYTHON),
        "scripts/inference/reconstruct.py",
        "--images",
        *[str(path) for path in SMOKE_IMAGES],
        "--masks",
        *[str(path) for path in SMOKE_MASKS],
        "--part-info",
        str(SMOKE_PART_INFO),
        "--out-dir",
        str(out_dir),
        "--ss-flow-ckpt",
        str(SS_FLOW_CKPT),
        "--part-seg-ckpt",
        str(PART_SEG_CKPT),
        "--quick-steps",
    ]
    _run(
        cmd,
        env={"CUDA_VISIBLE_DEVICES": str(args.gpu), "SS_FLOW_FUSION_MODE": "concat"},
        timeout=1800,
        log_path=out_dir / "command.log",
    )
    summary = _read_json(out_dir / "summary.json")
    gt = _load_gt_part_coords()
    ious = []
    part_rows = []
    for part in summary["parts"]:
        part_id = int(part["part_id"])
        pred_path = out_dir / f"part_{part_id:02d}_{str(part['label']).replace('/', '_')}" / "voxel.npz"
        with np.load(pred_path, allow_pickle=False) as data:
            pred = np.asarray(data["coords"], dtype=np.int32)
        iou = _coords_iou(pred, gt[part_id])
        ious.append(iou)
        part_rows.append({
            "part_id": part_id,
            "label": str(part["label"]),
            "iou": iou,
            "pred_voxels": int(pred.shape[0]),
            "gt_voxels": int(gt[part_id].shape[0]),
        })
    result = {
        "chain": "part_seg",
        "status": "passed",
        "command": cmd,
        "out_dir": str(out_dir),
        "mIoU": float(np.mean(ious)) if ious else float("nan"),
        "part_rows": part_rows,
        "key_fields": {
            "mIoU": float(np.mean(ious)) if ious else float("nan"),
            "parts": len(part_rows),
        },
    }
    _write_json(out_dir / "regression_result.json", result)
    return result


def chain_ss_flow(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = SMOKE_ROOT / "ss_flow_quick"
    out_dir.mkdir(parents=True, exist_ok=True)
    script = r"""
from pathlib import Path
import json, os, sys
import numpy as np
import torch
from PIL import Image
repo = Path.cwd()
sys.path.insert(0, str(repo / "TRELLIS-arts"))
import inference
from scripts.inference.reconstruct import _rgba_with_mask_alpha, _load_image, _load_mask
images = [Path(p) for p in os.environ["SMOKE_IMAGES"].split(":")]
masks = [Path(p) for p in os.environ["SMOKE_MASKS"].split(":")]
rgba = [_rgba_with_mask_alpha(_load_image(i), _load_mask(m)) for i, m in zip(images, masks)]
tokens = inference._images_to_tokens(rgba).detach().float().cpu()
z = inference.run_ss_flow_from_tokens(
    tokens,
    os.environ["SS_FLOW_CKPT"],
    num_steps=int(os.environ.get("SS_STEPS", "2")),
    cfg_strength=7.5,
    fusion_mode="concat",
)
coords = inference.decode_ss(z, "pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors", threshold=0.0)
out = Path(os.environ["OUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "whole_pred.npz", coords=coords.numpy().astype(np.int32), resolution=np.int32(64))
print(json.dumps({"coords": int(coords.shape[0]), "latent_shape": list(z.shape)}))
"""
    proc = _run(
        [str(PYTHON), "-c", script],
        env={
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "SMOKE_IMAGES": ":".join(str(path) for path in SMOKE_IMAGES),
            "SMOKE_MASKS": ":".join(str(path) for path in SMOKE_MASKS),
            "SS_FLOW_CKPT": str(SS_FLOW_CKPT),
            "SS_STEPS": "2" if args.quick else "20",
            "OUT_DIR": str(out_dir),
            "SS_FLOW_FUSION_MODE": "concat",
        },
        timeout=1200,
        log_path=out_dir / "command.log",
    )
    last_json = json.loads(proc.stdout.strip().splitlines()[-1])
    pred = np.load(out_dir / "whole_pred.npz")["coords"]
    gt = _load_gt_surface_coords()
    voxel_iou = _coords_iou(pred, gt)
    result = {
        "chain": "ss_flow",
        "status": "passed",
        "command": [str(PYTHON), "-c", "<ss_flow_quick_script>"],
        "out_dir": str(out_dir),
        "Voxel IoU": voxel_iou,
        "coords": int(pred.shape[0]),
        "key_fields": {
            "Voxel IoU": voxel_iou,
            "coords": int(pred.shape[0]),
            "latent_shape": last_json["latent_shape"],
        },
    }
    _write_json(out_dir / "regression_result.json", result)
    return result


def _parse_losses(text: str) -> list[float]:
    losses: list[float] = []
    for line in text.splitlines():
        for pattern in (r"loss=([0-9.eE+-]+)", r"total ([0-9.eE+-]+)"):
            match = re.search(pattern, line)
            if match:
                try:
                    losses.append(float(match.group(1)))
                except ValueError:
                    pass
    return losses


def chain_cotrain(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = SMOKE_ROOT / "cotrain_quick"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "chain": "cotrain",
        "status": "deprecated",
        "command": [],
        "out_dir": str(out_dir),
        "reason": "cotrain/joint/stage experiment training was removed from the open-source active tree",
        "key_fields": {
            "cotrain_active": False,
        },
    }
    _write_json(out_dir / "regression_result.json", result)
    return result


def chain_e2e_eval(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = SMOKE_ROOT / "e2e_eval_quick"
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        str(PYTHON),
        "scripts/inference/reconstruct.py",
        "--smoke-from-dataset",
        "--quick-steps",
        "--ss-flow-ckpt",
        str(SS_FLOW_CKPT),
        "--part-seg-ckpt",
        str(PART_SEG_CKPT),
        "--out-dir",
        str(out_dir),
    ]
    _run(
        cmd,
        env={"CUDA_VISIBLE_DEVICES": str(args.gpu), "SS_FLOW_FUSION_MODE": "concat"},
        timeout=1800,
        log_path=out_dir / "command.log",
    )
    summary = _read_json(out_dir / "summary.json")
    required = [out_dir / "labeled_voxel.npy", out_dir / "whole_voxel.npz", out_dir / "overall" / "overall.ply"]
    missing = [str(path) for path in required if not path.exists()]
    part_meshes = [Path(part["mesh_path"]) for part in summary["parts"] if part.get("mesh_path")]
    part_gaussians = [Path(part["gaussian_path"]) for part in summary["parts"] if part.get("gaussian_path")]
    missing.extend(str(path) for path in [*part_meshes, *part_gaussians] if not path.exists())
    if missing:
        raise RuntimeError("e2e_eval missing artifacts: " + ", ".join(missing))
    result = {
        "chain": "e2e_eval",
        "status": "passed",
        "command": cmd,
        "out_dir": str(out_dir),
        "artifact_dir": str(out_dir),
        "key_fields": {
            "whole_voxels": int(summary["whole_voxel_coords"]["shape"][0]),
            "parts": len(summary["parts"]),
            "part_meshes": len(part_meshes),
            "part_gaussians": len(part_gaussians),
            "artifact_dir": str(out_dir),
        },
    }
    _write_json(out_dir / "regression_result.json", result)
    return result


CHAIN_FUNCS = {
    "part_seg": chain_part_seg,
    "cotrain": chain_cotrain,
    "ss_flow": chain_ss_flow,
    "e2e_eval": chain_e2e_eval,
}


def _load_baseline() -> dict[str, Any]:
    if not BASELINE_JSON.is_file():
        return {}
    payload = _read_json(BASELINE_JSON)
    if not isinstance(payload, dict):
        return {}
    return payload.get("chains", {}) if isinstance(payload.get("chains"), dict) else {}


def _compare(chain: str, result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    base = baseline.get(chain)
    if not isinstance(base, dict):
        return {"status": "no_baseline"}
    base_fields = dict(base.get("key_fields") or {})
    now_fields = dict(result.get("key_fields") or {})
    comparisons: dict[str, Any] = {}
    status = "passed"
    for key in ("mIoU", "Voxel IoU"):
        if key in base_fields and key in now_fields:
            before = float(base_fields[key])
            after = float(now_fields[key])
            drift = abs(after - before) / max(abs(before), 1.0e-12)
            comparisons[key] = {"baseline": before, "current": after, "relative_drift": drift}
            if drift > 0.005:
                status = "failed"
    if chain == "cotrain" and "loss_last" in now_fields:
        loss = float(now_fields["loss_last"])
        if not math.isfinite(loss) or loss <= 0 or loss > 1.0e5:
            status = "failed"
        comparisons["loss_last"] = {
            "baseline": base_fields.get("loss_last"),
            "current": loss,
            "finite": math.isfinite(loss),
        }
    if chain == "e2e_eval":
        artifact_dir = Path(str(now_fields.get("artifact_dir", "")))
        ok = artifact_dir.is_dir() and (artifact_dir / "summary.json").is_file()
        comparisons["artifacts"] = {"artifact_dir": str(artifact_dir), "complete": ok}
        if not ok:
            status = "failed"
    return {"status": status, "comparisons": comparisons}


def _write_baseline_md(chains: dict[str, Any]) -> None:
    lines = [
        "# Regression Baseline 0626",
        "",
        "Generated by `scripts/eval/regression_smoke.sh --write-baseline`.",
        "",
        "Authority: e2e path follows `docs/runbooks/0617_128ee_correct_slat_flow.md`; quick smoke uses the fixed dataset sample `phyx-verse::0023687e90394c3e97ab19b0160cafb3 angle=0` with 4 views.",
        "",
        "| Chain | Status | Key fields | Output |",
        "|---|---|---|---|",
    ]
    for chain in ("part_seg", "cotrain", "ss_flow", "e2e_eval"):
        item = chains.get(chain, {})
        key_fields = json.dumps(item.get("key_fields", {}), ensure_ascii=False, sort_keys=True)
        lines.append(f"| `{chain}` | `{item.get('status', 'missing')}` | `{key_fields}` | `{item.get('out_dir', '')}` |")
    lines.extend(
        [
            "",
            "## Commands",
            "",
        ]
    )
    for chain in ("part_seg", "cotrain", "ss_flow", "e2e_eval"):
        item = chains.get(chain, {})
        lines.append(f"### {chain}")
        lines.append("")
        command = item.get("command", [])
        if command:
            lines.append("```bash")
            lines.append(" ".join(str(part) for part in command))
            lines.append("```")
        else:
            lines.append("No command recorded.")
        lines.append("")
    BASELINE_MD.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_chain(args: argparse.Namespace) -> int:
    result = CHAIN_FUNCS[args.chain](args)
    baseline = _load_baseline()
    gate = _compare(args.chain, result, baseline)
    result["gate"] = gate
    out_dir = Path(result["out_dir"])
    _write_json(out_dir / "regression_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if gate["status"] == "failed":
        return 2
    return 0


def write_baseline(args: argparse.Namespace) -> int:
    chains: dict[str, Any] = {}
    for chain in ("part_seg", "cotrain", "ss_flow", "e2e_eval"):
        local = argparse.Namespace(**vars(args))
        local.chain = chain
        chains[chain] = CHAIN_FUNCS[chain](local)
        chains[chain]["gate"] = {"status": "baseline"}
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke_root": str(SMOKE_ROOT),
        "chains": chains,
    }
    _write_json(BASELINE_JSON, payload)
    _write_baseline_md(chains)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified regression smoke gate for arts-gen refactor.")
    parser.add_argument("--chain", choices=sorted(CHAIN_FUNCS), default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-baseline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_baseline:
        return write_baseline(args)
    if not args.chain:
        raise SystemExit("--chain is required unless --write-baseline is set")
    return run_chain(args)


if __name__ == "__main__":
    raise SystemExit(main())
