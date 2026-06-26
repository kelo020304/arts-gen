#!/usr/bin/env python3
"""Pipeline step 03b: SLat (.pt) -> mesh.obj + gaussians.ply.

Thin orchestrator (D-23). Source: scripts/train/stage4/render_utils.py:35-150 (D-30).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "TRELLIS-arts"))

import torch  # noqa: E402
from inference import decode_slat  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="SLat decode -> mesh + gaussians")
    ap.add_argument("--slat", required=True, help="path to slat.pt")
    ap.add_argument("--ckpt", required=True,
                    help="SLat Gaussian decoder basename (no extension) or .safetensors")
    ap.add_argument("--formats", default="mesh,gaussian",
                    help="comma-separated subset of {mesh, gaussian}")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    slat = torch.load(args.slat, weights_only=False)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    out_dict = decode_slat(slat, args.ckpt, formats=formats)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    if "gaussian" in formats and out_dict.get("gaussian") is not None:
        gs = out_dict["gaussian"]
        ply_path = out / "gaussians.ply"
        # Trellis Gaussian objects expose ``save_ply``; fall back to torch.save.
        save_ply = getattr(gs, "save_ply", None)
        if callable(save_ply):
            save_ply(str(ply_path))
        else:
            torch.save(gs, ply_path)
        print(f"[03_final_decode] saved {ply_path.name}")

    if "mesh" in formats and out_dict.get("mesh") is not None:
        mesh = out_dict["mesh"]
        obj_path = out / "mesh.obj"
        export = getattr(mesh, "export", None)
        if callable(export):
            export(str(obj_path))
        else:
            # Best-effort fallback: write a marker so downstream knows the
            # current decoder ckpt does not emit mesh data.
            obj_path.write_text("# mesh format unsupported by loaded decoder\n")
        print(f"[03_final_decode] saved {obj_path.name}")
    elif "mesh" in formats:
        # decoder did not produce a mesh; leave a stub for traceability.
        (out / "mesh.obj").write_text("# mesh not produced by current decoder\n")
        print("[03_final_decode] mesh.obj stub written (decoder has no mesh head)")

    print(f"[03_final_decode] DONE -> {out}")


if __name__ == "__main__":
    main()
