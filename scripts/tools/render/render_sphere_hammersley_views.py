#!/usr/bin/env python3
"""Render one articulated object from sphere-hammersley views in Blender."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import bpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLKIT_PIPELINE = PROJECT_ROOT / "submodules" / "dataset_toolkits" / "pipeline"
if str(TOOLKIT_PIPELINE) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_PIPELINE))

import importlib.util

STEP02_PATH = TOOLKIT_PIPELINE / "02_blender_render.py"
spec = importlib.util.spec_from_file_location("dataset_toolkits_step02_render", STEP02_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load Step02 render module spec: {STEP02_PATH}")
step02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step02)


def radical_inverse(base: int, n: int) -> float:
    val = 0.0
    inv_base = 1.0 / float(base)
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def sphere_hammersley_sequence(
    n: int,
    num_samples: int,
    offset: tuple[float, float],
) -> tuple[float, float]:
    """Match TRELLIS-arts/dataset_toolkits/utils.py sphere sampling exactly."""
    u = n / float(num_samples)
    v = radical_inverse(2, n)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2.0 * u if u < 0.25 else 2.0 / 3.0 * u + 1.0 / 3.0
    theta = math.acos(1.0 - 2.0 * u) - math.pi / 2.0
    phi = v * 2.0 * math.pi
    return phi, theta


def set_camera_pose_sphere(camera_obj, yaw_rad: float, pitch_rad: float, radius: float) -> None:
    camera_obj.location = (
        radius * math.cos(yaw_rad) * math.cos(pitch_rad),
        radius * math.sin(yaw_rad) * math.cos(pitch_rad),
        radius * math.sin(pitch_rad),
    )
    bpy.context.view_layer.update()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objs-dir", required=True)
    parser.add_argument("--finaljson", required=True)
    parser.add_argument("--transforms-json", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--num-views", type=int, default=150)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--offset-u", type=float, default=None)
    parser.add_argument("--offset-v", type=float, default=None)
    parser.add_argument("--rgb-engine", choices=("BLENDER_EEVEE_NEXT", "CYCLES"), default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--cycles-device", choices=("CPU", "CUDA", "OPTIX"), default="CPU")
    parser.add_argument("--rgb-material-mode", choices=step02.RGB_MATERIAL_MODES, default="imported")
    parser.add_argument("--obj-up-axis", choices=("Y", "Z"), default="Y")
    if "--" not in argv:
        raise ValueError("Blender script arguments must be passed after '--'")
    return parser.parse_args(argv[argv.index("--") + 1 :])


def main() -> None:
    args = parse_args(sys.argv)
    if args.num_views < 1:
        raise ValueError(f"--num-views must be >= 1, got {args.num_views}")
    if args.radius <= 0:
        raise ValueError(f"--radius must be positive, got {args.radius}")
    if (args.offset_u is None) != (args.offset_v is None):
        raise ValueError("--offset-u and --offset-v must be provided together")
    if args.seed is not None and args.offset_u is not None:
        raise ValueError("--seed is mutually exclusive with explicit --offset-u/--offset-v")
    if args.offset_u is None:
        import random

        rng = random.Random(args.seed)
        hammersley_offset = (rng.random(), rng.random())
    else:
        hammersley_offset = (float(args.offset_u), float(args.offset_v))

    output_folder = Path(args.output_folder).resolve()
    rgb_dir = output_folder / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    step02.init_render(
        engine=args.rgb_engine,
        resolution=args.resolution,
        cycles_rgb_samples=step02.DEFAULT_CYCLES_RGB_SAMPLES,
        cycles_rgb_denoise=True,
        cycles_device=args.cycles_device,
    )
    step02.init_scene()
    step02.load_parts_with_transforms(args.objs_dir, args.finaljson, args.transforms_json, args.obj_up_axis)
    step02.apply_rgb_material_mode(args.rgb_material_mode)
    scale, scene_offset = step02.normalize_scene()

    camera_obj = step02.init_camera()
    step02.init_lighting()
    camera_obj.data.lens = 16.0 / math.tan(math.radians(step02.FOV_DEG) / 2.0)

    for obj in bpy.data.objects:
        if obj.type not in ("MESH", "CAMERA", "LIGHT"):
            obj.hide_render = True
            obj.hide_viewport = True

    start = time.perf_counter()
    frames = []
    for view_idx in range(args.num_views):
        yaw_rad, pitch_rad = sphere_hammersley_sequence(view_idx, args.num_views, hammersley_offset)
        set_camera_pose_sphere(camera_obj, yaw_rad, pitch_rad, args.radius)
        bpy.context.scene.frame_set(view_idx + 1)
        rgb_path = rgb_dir / f"view_{view_idx}.png"
        bpy.context.scene.render.engine = args.rgb_engine
        bpy.context.scene.render.filepath = str(rgb_path)
        bpy.ops.render.render(write_still=True)
        frames.append(
            {
                "file_path": f"rgb/view_{view_idx}.png",
                "view_index": view_idx,
                "yaw_rad": yaw_rad,
                "pitch_rad": pitch_rad,
                "camera_angle_x": camera_obj.data.angle_x,
                "transform_matrix": step02.get_transform_matrix(camera_obj),
            }
        )
        print(f"[sphere-render] {view_idx + 1}/{args.num_views} {rgb_path}", flush=True)

    elapsed = time.perf_counter() - start
    camera_transforms = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [scene_offset.x, scene_offset.y, scene_offset.z],
        "resolution": args.resolution,
        "fov_deg": step02.FOV_DEG,
        "view_sampler": "trellis_official_sphere_hammersley",
        "hammersley_offset": [float(hammersley_offset[0]), float(hammersley_offset[1])],
        "radius": float(args.radius),
        "total_views": args.num_views,
        "render_engine": args.rgb_engine,
        "cycles_device": args.cycles_device if args.rgb_engine == "CYCLES" else None,
        "elapsed_seconds": elapsed,
        "frames": frames,
    }
    with (output_folder / "camera_transforms.json").open("w", encoding="utf-8") as handle:
        json.dump(camera_transforms, handle, indent=2)
        handle.write("\n")
    print(f"[sphere-render] completed {args.num_views} views in {elapsed:.3f}s -> {output_folder}", flush=True)


if __name__ == "__main__":
    main()
