#!/usr/bin/env python3
"""Official TRELLIS image-to-3D color control rendered with the QC save path."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = PROJECT_ROOT / "TRELLIS-arts"
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import diff_gaussian_rasterization  # noqa: E402,F401
from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: E402

sys.modules.pop("trellis.renderers", None)
gaussian_render_mod = importlib.import_module("trellis.renderers.gaussian_render")
GaussianRenderer = gaussian_render_mod.GaussianRenderer


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.detach().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def channel_stats(image: Image.Image) -> dict[str, Any]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mask = arr.max(axis=2) > 0.03
    vals = arr[mask] if mask.any() else arr.reshape(-1, 3)
    return {
        "pixels": int(vals.shape[0]),
        "mean_rgb": [float(x) for x in vals.mean(axis=0).tolist()],
        "min_rgb": [float(x) for x in vals.min(axis=0).tolist()],
        "max_rgb": [float(x) for x in vals.max(axis=0).tolist()],
    }


def make_renderer(resolution: int = 512) -> Any:
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = int(resolution)
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.ssaa = 1
    renderer.pipe.kernel_size = 0.1
    return renderer


def canonical_camera(device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    import utils3d

    c2w = torch.eye(4, dtype=torch.float32, device=device)
    c2w[2, 3] = 1.25
    c2w[:3, 1:3] *= -1
    extrinsic = torch.inverse(c2w)
    fov = torch.tensor(np.deg2rad(40.0), dtype=torch.float32, device=device)
    intrinsic = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    return extrinsic, intrinsic


def save_panel(path: Path, cells: list[tuple[str, Image.Image]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = cells[0][1].size
    label_h = 28
    canvas = Image.new("RGB", (len(cells) * w, h + label_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(cells):
        x = idx * w
        draw.text((x + 6, 7), label, fill=(255, 255, 255))
        canvas.paste(image.convert("RGB"), (x, label_h))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--slat-steps", type=int, default=12)
    parser.add_argument("--ss-cfg", type=float, default=7.5)
    parser.add_argument("--slat-cfg", type=float, default=3.0)
    args = parser.parse_args()

    input_path = require_file(args.input, "input image")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_img = Image.open(input_path).convert("RGBA")
    print(f"[python] {sys.executable}", flush=True)
    print(f"[torch] {torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"[pipeline] loading {args.model}", flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    print("[pipeline] loaded and moved to cuda", flush=True)
    outputs = pipeline.run(
        input_img,
        seed=int(args.seed),
        preprocess_image=True,
        formats=["gaussian"],
        sparse_structure_sampler_params={"steps": int(args.ss_steps), "cfg_strength": float(args.ss_cfg)},
        slat_sampler_params={"steps": int(args.slat_steps), "cfg_strength": float(args.slat_cfg)},
    )
    gaussian = outputs["gaussian"][0] if isinstance(outputs["gaussian"], list) else outputs["gaussian"]
    renderer = make_renderer(512)
    extr, intr = canonical_camera()
    rendered = renderer.render(gaussian, extr, intr, colors_overwrite=None)["color"].detach().float().cpu().clamp(0, 1)
    rendered_img = tensor_to_image(rendered)
    render_path = out_dir / "official_trellis_render_same_save_fn.png"
    rendered_img.save(render_path)
    panel_path = out_dir / "official_trellis_color_control_panel.png"
    save_panel(panel_path, [("input", input_img.convert("RGB").resize(rendered_img.size)), ("official TRELLIS render", rendered_img)])
    features = gaussian._features_dc.detach().float().cpu()
    report = {
        "python": sys.executable,
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
        "official_renderer": {
            "gaussian_renderer_module": gaussian_render_mod.__file__,
            "diff_gaussian_module": diff_gaussian_rasterization.__file__,
            "used_gsplat": False,
        },
        "pipeline_model": args.model,
        "input": str(input_path.resolve()),
        "render_path": str(render_path.resolve()),
        "panel_path": str(panel_path.resolve()),
        "render_stats": channel_stats(rendered_img),
        "feature_stats": {
            "_features_dc_shape": list(features.shape),
            "_features_dc_mean": [float(x) for x in features.reshape(-1, 3).mean(dim=0).tolist()],
            "_features_dc_min": [float(x) for x in features.reshape(-1, 3).min(dim=0).values.tolist()],
            "_features_dc_max": [float(x) for x in features.reshape(-1, 3).max(dim=0).values.tolist()],
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[renderer] {gaussian_render_mod.__file__}", flush=True)
    print(f"[diff_gaussian] {diff_gaussian_rasterization.__file__}", flush=True)
    print(f"[render-stats] mean_rgb={report['render_stats']['mean_rgb']}", flush=True)
    print(f"[features-dc] mean={report['feature_stats']['_features_dc_mean']}", flush=True)
    print(f"[done] render={render_path}", flush=True)
    print(f"[done] panel={panel_path}", flush=True)
    print(f"[done] report={report_path}", flush=True)


if __name__ == "__main__":
    main()
