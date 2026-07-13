from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def _default_ckpt() -> Path:
    env = os.environ.get("SAM3_CKPT_PATH")
    if env:
        return Path(env)
    # arts-recon convention: submodules/sam3/ckpt/sam3.pt under the repo root.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "submodules" / "sam3" / "ckpt" / "sam3.pt"
        if candidate.exists():
            return candidate
    return Path("submodules/sam3/ckpt/sam3.pt")


def _parse_box(s: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--box must be 'cx,cy,w,h', got {s!r}"
        )
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--box has non-numeric value: {e}")
    for v in vals:
        if not (0.0 <= v <= 1.0):
            raise argparse.ArgumentTypeError(
                f"--box values must lie in [0, 1], got {vals}"
            )
    return vals  # type: ignore[return-value]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mask",
        description="Box-prompted segmentation via SAM 3.",
    )
    p.add_argument("--image", required=True, type=Path,
                   help="Path to RGB image")
    p.add_argument("--box", required=True, type=_parse_box,
                   help='Normalized box "cx,cy,w,h" in [0, 1]')
    p.add_argument("-o", "--output-dir", required=True, type=Path,
                   help="Where to write mask.png + prompt.json")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Path to sam3.pt (env SAM3_CKPT_PATH overrides default)")
    p.add_argument("--device", default="cuda", help="cuda | cpu (default cuda)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    image_path: Path = args.image
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt: Path = args.ckpt if args.ckpt is not None else _default_ckpt()
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    from PIL import Image
    from .types import BoxPrompt
    from .pipeline import MaskPipeline

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1024, 1024))

    cx, cy, w, h = args.box
    box = BoxPrompt(cx=cx, cy=cy, w=w, h=h)
    W, H = image.size
    x0, y0, x1, y1 = box.to_xyxy_pixels(W, H)

    pipeline = MaskPipeline(ckpt_path=ckpt, device=args.device)
    state = pipeline.embed(image)
    out = pipeline.predict_box(state, x0, y0, x1, y1)

    out.save_png(output_dir / "mask.png")
    (output_dir / "prompt.json").write_text(
        json.dumps({"box": box.as_list(), "score": out.score}, indent=2)
    )
    print(f"wrote {output_dir / 'mask.png'} (score={out.score:.3f})")
    return 0
