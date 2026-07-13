#!/usr/bin/env python3
"""Build real-data PNG assets for the PartSeg method figure."""

from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import binary_dilation


ROOT = Path("/root/code/arts-gen")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_ROOT = Path("/robot/data-lab/jzh/art-gen/data/phyx-verse")
EVAL_ROOT = Path("/robot/data-lab/jzh/art-gen/ee-eval/0706-old-S0618-step50000/_platform_runs/held")
OUT_ROOT = ROOT / "figure_assets"

WHITE = (255, 255, 255)
PART_COLORS = [
    (73, 139, 205),   # part1 blue
    (91, 168, 111),   # part2 green
    (226, 143, 65),   # part3 orange
]
BODY_COLOR = (205, 210, 216)
WHOLE_COLOR = (202, 218, 232)
BOUNDARY_RED = (220, 64, 56)


@dataclass(frozen=True)
class PartSpec:
    name: str
    mask_label: int
    voxel_file: str
    mesh_obj_ids: tuple[int, ...]


@dataclass(frozen=True)
class ObjectSpec:
    obj_id: str
    alias: str
    view_indices: tuple[int, int, int, int]
    mask_view: int
    parts: tuple[PartSpec, PartSpec, PartSpec]
    body_mesh_obj_ids: tuple[int, ...]


OBJECTS = [
    ObjectSpec(
        obj_id="0a46621504c24197b5653608f474f73b",
        alias="bedside_cabinet",
        view_indices=(1, 4, 8, 11),
        mask_view=1,
        parts=(
            PartSpec("middle drawer", 1, "part_00_voxel.npz", (0,)),
            PartSpec("bottom drawer", 2, "part_01_voxel.npz", (1,)),
            PartSpec("top drawer", 9, "part_02_voxel.npz", (8,)),
        ),
        body_mesh_obj_ids=(2, 3, 4, 5, 6, 7),
    ),
    ObjectSpec(
        obj_id="01816801a27444cbb5cfb934de39d483",
        alias="faucet",
        view_indices=(0, 5, 6, 11),
        mask_view=0,
        parts=(
            PartSpec("left handle", 2, "part_00_voxel.npz", (1,)),
            PartSpec("right handle", 3, "part_01_voxel.npz", (2,)),
            PartSpec("spout", 4, "part_02_voxel.npz", (3,)),
        ),
        body_mesh_obj_ids=(0,),
    ),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT_20 = font(20)
FONT_24 = font(24)
FONT_28 = font(28, bold=True)
FONT_36 = font(36, bold=True)
FONT_44 = font(44, bold=True)


def alpha_on_white(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, WHITE + (255,))
    bg.alpha_composite(im)
    return bg.convert("RGB")


def crop_content(im: Image.Image, pad: int = 24) -> Image.Image:
    rgba = im.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return im.convert("RGB")
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(im.width, x1 + pad)
    y1 = min(im.height, y1 + pad)
    return im.crop((x0, y0, x1, y1)).convert("RGB")


def paste_fit(canvas: Image.Image, im: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    max_w, max_h = x1 - x0, y1 - y0
    scale = min(max_w / im.width, max_h / im.height)
    new_size = (max(1, int(im.width * scale)), max(1, int(im.height * scale)))
    resized = im.resize(new_size, Image.Resampling.LANCZOS)
    px = x0 + (max_w - new_size[0]) // 2
    py = y0 + (max_h - new_size[1]) // 2
    canvas.paste(resized, (px, py))


def rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int = 22) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=(255, 255, 255), outline=(223, 229, 237), width=2)


def make_input_views(spec: ObjectSpec, out: Path) -> None:
    canvas = Image.new("RGB", (1400, 1400), WHITE)
    draw = ImageDraw.Draw(canvas)
    boxes = [(90, 90, 675, 675), (725, 90, 1310, 675), (90, 725, 675, 1310), (725, 725, 1310, 1310)]
    for view, box in zip(spec.view_indices, boxes):
        im = alpha_on_white(DATA_ROOT / "renders" / spec.obj_id / "angle_0" / "rgb" / f"view_{view}.png")
        rounded_panel(draw, box, radius=28)
        paste_fit(canvas, im, (box[0] + 28, box[1] + 28, box[2] - 28, box[3] - 28))
    canvas.save(out)


def make_vlm_masks(spec: ObjectSpec, out: Path) -> None:
    base_path = DATA_ROOT / "renders" / spec.obj_id / "angle_0" / "rgb" / f"view_{spec.mask_view}.png"
    mask_path = DATA_ROOT / "renders" / spec.obj_id / "angle_0" / "mask" / f"mask_{spec.mask_view}.npy"
    base = alpha_on_white(base_path).convert("RGBA")
    mask = np.load(mask_path)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    arr = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    for idx, part in enumerate(spec.parts):
        region = mask == int(part.mask_label)
        arr[region, :3] = PART_COLORS[idx]
        arr[region, 3] = 120
    overlay = Image.fromarray(arr, "RGBA")
    composed = Image.alpha_composite(base, overlay)

    canvas = Image.new("RGB", (1400, 1400), WHITE)
    large = composed.convert("RGB")
    paste_fit(canvas, large, (95, 65, 1305, 1190))
    draw = ImageDraw.Draw(canvas)
    for idx, part in enumerate(spec.parts):
        region = mask == int(part.mask_label)
        if not bool(region.any()):
            continue
        ys, xs = np.nonzero(region)
        cx = int(xs.mean() / mask.shape[1] * 1210 + 95)
        cy = int(ys.mean() / mask.shape[0] * 1125 + 65)
        label = f"part{idx + 1}: {part.name}"
        tw = int(draw.textlength(label, font=FONT_28))
        lx = min(max(cx - tw // 2, 56), 1400 - tw - 56)
        ly = min(max(cy + 24, 50), 1320)
        draw.rounded_rectangle((lx - 18, ly - 12, lx + tw + 18, ly + 34), radius=16, fill=WHITE, outline=PART_COLORS[idx], width=3)
        draw.text((lx, ly - 4), label, fill=PART_COLORS[idx], font=FONT_28)
    canvas.save(out)


def dense_from_coords(coords: np.ndarray, resolution: int = 64) -> np.ndarray:
    dense = np.zeros((resolution, resolution, resolution), dtype=bool)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords.size:
        valid = np.all((coords >= 0) & (coords < resolution), axis=1)
        coords = coords[valid]
        dense[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return dense


def coords_from_dense(dense: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(dense).astype(bool)).astype(np.int32)


def load_npz_coords(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    return np.asarray(data["coords"], dtype=np.int32).reshape(-1, 3)


def color_shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    factor = max(0.0, min(1.5, factor))
    return tuple(max(0, min(255, int(c * factor + 255 * (1.0 - factor) * 0.08))) for c in color)


def project_iso(points: np.ndarray, scale: float, offset: tuple[float, float], center: np.ndarray) -> np.ndarray:
    p = points - center[None, :]
    x = (p[:, 0] - p[:, 1]) * 0.866
    y = (p[:, 0] + p[:, 1]) * 0.42 - p[:, 2] * 0.76
    return np.stack([x * scale + offset[0], y * scale + offset[1]], axis=1)


def render_voxels(
    labeled_coords: list[tuple[np.ndarray, tuple[int, int, int]]],
    out: Path | None = None,
    *,
    size: tuple[int, int] = (1400, 1400),
    pad: int = 120,
    title: str | None = None,
    crop_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> Image.Image:
    canvas = Image.new("RGB", size, WHITE)
    draw = ImageDraw.Draw(canvas)
    coords_all = []
    label_items = []
    occ_set: set[tuple[int, int, int]] = set()
    for coords, color in labeled_coords:
        c = np.asarray(coords, dtype=np.int32).reshape(-1, 3)
        if c.size == 0:
            continue
        coords_all.append(c)
        for row in c:
            tup = tuple(int(v) for v in row)
            occ_set.add(tup)
            label_items.append((tup, color))
    if not coords_all:
        if out is not None:
            canvas.save(out)
        return canvas
    all_coords = np.concatenate(coords_all, axis=0)
    if crop_bounds is None:
        mn = all_coords.min(axis=0).astype(float)
        mx = all_coords.max(axis=0).astype(float) + 1.0
    else:
        mn, mx = crop_bounds
        mn = mn.astype(float)
        mx = mx.astype(float)
    corners = np.array(
        [[mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]], [mn[0], mx[1], mn[2]], [mn[0], mn[1], mx[2]],
         [mx[0], mx[1], mn[2]], [mx[0], mn[1], mx[2]], [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]]],
        dtype=float,
    )
    center = (mn + mx) / 2.0
    proj0 = project_iso(corners, 1.0, (0, 0), center)
    span = proj0.max(axis=0) - proj0.min(axis=0)
    scale = min((size[0] - 2 * pad) / max(span[0], 1.0), (size[1] - 2 * pad) / max(span[1], 1.0))
    offset = (size[0] / 2.0, size[1] / 2.0 + 35)

    faces = []
    face_defs = [
        ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], 1.08),
        ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], 0.88),
        ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)], 0.74),
    ]
    for (x, y, z), color in label_items:
        depth = x + y + z
        for neighbor_delta, verts_delta, shade in face_defs:
            nx, ny, nz = x + neighbor_delta[0], y + neighbor_delta[1], z + neighbor_delta[2]
            if (nx, ny, nz) in occ_set:
                continue
            verts = np.array([[x + dx, y + dy, z + dz] for dx, dy, dz in verts_delta], dtype=float)
            poly = project_iso(verts, scale, offset, center)
            faces.append((depth + sum(neighbor_delta) * 0.01, poly, color_shade(color, shade)))
    faces.sort(key=lambda item: item[0])
    for _depth, poly, color in faces:
        draw.polygon([tuple(p) for p in poly], fill=color, outline=(255, 255, 255))
    if title:
        draw.text((size[0] // 2, size[1] - 88), title, anchor="mm", fill=(92, 101, 116), font=FONT_28)
    if out is not None:
        canvas.save(out)
    return canvas


def object_run_dir(spec: ObjectSpec) -> Path:
    return EVAL_ROOT / f"{spec.obj_id}-0" / "real-B"


def load_pipeline_voxels(spec: ObjectSpec) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    run = object_run_dir(spec)
    whole = load_npz_coords(run / "voxel.npz")
    parts = [load_npz_coords(run / "parts" / part.voxel_file) for part in spec.parts]
    whole_dense = dense_from_coords(whole, 64)
    part_dense = np.zeros_like(whole_dense)
    for coords in parts:
        part_dense |= dense_from_coords(coords, 64)
    body = coords_from_dense(whole_dense & ~part_dense)
    return whole, parts, body


def make_voxel_whole(spec: ObjectSpec, out: Path) -> None:
    whole, _parts, _body = load_pipeline_voxels(spec)
    render_voxels([(whole, WHOLE_COLOR)], out, size=(1400, 1400), pad=150)


def make_voxel_parts(spec: ObjectSpec, out: Path) -> None:
    _whole, parts, body = load_pipeline_voxels(spec)
    items: list[tuple[np.ndarray, tuple[int, int, int]]] = [(body, BODY_COLOR)]
    for idx, coords in enumerate(parts):
        items.append((coords, PART_COLORS[idx]))
    render_voxels(items, out, size=(1400, 1400), pad=135)


def make_latent_grid(out: Path) -> None:
    coords = []
    for x in range(16):
        for y in range(16):
            for z in range(16):
                if x in (0, 15) or y in (0, 15) or z in (0, 15):
                    coords.append((x, y, z))
    render_voxels([(np.asarray(coords, dtype=np.int32), (178, 214, 240))], out, size=(1400, 1400), pad=170)


def make_boundary_band(spec: ObjectSpec, out: Path) -> None:
    whole, parts, body = load_pipeline_voxels(spec)
    whole_d = dense_from_coords(whole, 64)
    parts_d = np.zeros_like(whole_d)
    for coords in parts:
        parts_d |= dense_from_coords(coords, 64)
    body_d = dense_from_coords(body, 64)
    band = binary_dilation(parts_d, iterations=2) & binary_dilation(body_d, iterations=2) & whole_d
    band_coords = coords_from_dense(band)
    if band_coords.size == 0:
        band_coords = parts[0][: min(200, len(parts[0]))]
    center = np.median(band_coords, axis=0).astype(int)
    mn = np.maximum(center - 11, 0)
    mx = np.minimum(center + 12, 64)
    in_crop = lambda c: c[np.all((c >= mn) & (c < mx), axis=1)]  # noqa: E731
    body_c = in_crop(body)
    part_c = in_crop(np.concatenate(parts, axis=0))
    band_c = in_crop(band_coords)
    items = [(body_c, BODY_COLOR), (part_c, (122, 166, 210)), (band_c, BOUNDARY_RED)]
    render_voxels(items, out, size=(1400, 1400), pad=180, crop_bounds=(mn, mx))


def corrupt_full_occ(whole_coords: np.ndarray) -> np.ndarray:
    from scripts.train.part_promptable_seg.train_part_promptable_seg import corrupt_voxel_occ

    dense = dense_from_coords(whole_coords, 64).astype(np.float32)
    ten = torch.from_numpy(dense[None, None])
    torch.manual_seed(706)
    corrupted, _stats = corrupt_voxel_occ(
        ten,
        enabled=True,
        drop_prob=0.03,
        shell_prob=0.08,
        speckle_prob=0.0003,
    )
    return coords_from_dense(corrupted[0, 0].detach().cpu().numpy() > 0.5)


def make_corruption_pair(spec: ObjectSpec, out: Path) -> None:
    whole, _parts, _body = load_pipeline_voxels(spec)
    corrupted = corrupt_full_occ(whole)
    clean_img = render_voxels([(whole, WHOLE_COLOR)], None, size=(900, 1020), pad=125)
    dirty_img = render_voxels([(corrupted, (220, 185, 152))], None, size=(900, 1020), pad=125)
    canvas = Image.new("RGB", (1800, 1200), WHITE)
    canvas.paste(clean_img, (0, 90))
    canvas.paste(dirty_img, (900, 90))
    draw = ImageDraw.Draw(canvas)
    draw.text((450, 80), "clean", anchor="mm", fill=(92, 101, 116), font=FONT_36)
    draw.text((1350, 80), "corrupted", anchor="mm", fill=(92, 101, 116), font=FONT_36)
    draw.line((900, 150, 900, 1080), fill=(226, 231, 238), width=2)
    canvas.save(out)


def rotation_matrix() -> np.ndarray:
    az = math.radians(-38)
    el = math.radians(24)
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]], dtype=float)
    rx = np.array([[1, 0, 0], [0, math.cos(el), -math.sin(el)], [0, math.sin(el), math.cos(el)]], dtype=float)
    return rx @ rz


def render_meshes(mesh_items: list[tuple[trimesh.Trimesh, tuple[int, int, int], np.ndarray]], out: Path) -> None:
    size = (1400, 1400)
    canvas = Image.new("RGB", size, WHITE)
    draw = ImageDraw.Draw(canvas)
    if not mesh_items:
        canvas.save(out)
        return
    all_vertices = np.concatenate([mesh.vertices for mesh, _color, _off in mesh_items], axis=0)
    center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) / 2.0
    span = float(np.max(all_vertices.max(axis=0) - all_vertices.min(axis=0)))
    span = max(span, 1.0e-6)
    R = rotation_matrix()
    light = np.array([-0.3, -0.5, 0.85])
    light = light / np.linalg.norm(light)
    for mesh, color, offset in mesh_items:
        polys = []
        verts = (mesh.vertices - center[None, :]) / span + offset[None, :]
        rv = verts @ R.T
        scale = 760.0
        screen = np.stack([rv[:, 0] * scale + size[0] / 2, -rv[:, 2] * scale + size[1] / 2 + 30], axis=1)
        tri = mesh.faces
        v0 = rv[tri[:, 0]]
        v1 = rv[tri[:, 1]]
        v2 = rv[tri[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        nlen = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(nlen, 1.0e-8)
        shade = np.clip(0.62 + 0.38 * np.maximum(normals @ light, 0.0), 0.45, 1.05)
        depth = rv[tri].mean(axis=(1, 2))
        for face_idx, face in enumerate(tri):
            poly = screen[face]
            if np.any(poly[:, 0] < -1000) or np.any(poly[:, 0] > size[0] + 1000):
                continue
            polys.append((depth[face_idx], poly, color_shade(color, float(shade[face_idx]))))
        polys.sort(key=lambda item: item[0])
        for _depth, poly, face_color in polys:
            draw.polygon([tuple(p) for p in poly], fill=face_color, outline=None)
    canvas = canvas.filter(ImageFilter.SMOOTH_MORE)
    canvas.save(out)


def make_parts_mesh(spec: ObjectSpec, out: Path) -> None:
    mesh_root = DATA_ROOT / "raw" / "partseg" / spec.obj_id / "objs"
    mesh_items: list[tuple[trimesh.Trimesh, tuple[int, int, int], np.ndarray]] = []
    all_meshes = []
    for part in spec.parts:
        for obj_id in part.mesh_obj_ids:
            mesh = trimesh.load(mesh_root / f"{obj_id}.obj", force="mesh", process=False)
            all_meshes.append(mesh)
    for obj_id in spec.body_mesh_obj_ids:
        mesh = trimesh.load(mesh_root / f"{obj_id}.obj", force="mesh", process=False)
        all_meshes.append(mesh)
    global_vertices = np.concatenate([m.vertices for m in all_meshes], axis=0)
    global_center = (global_vertices.min(axis=0) + global_vertices.max(axis=0)) / 2.0
    global_span = max(float(np.max(global_vertices.max(axis=0) - global_vertices.min(axis=0))), 1.0e-6)
    for obj_id in spec.body_mesh_obj_ids:
        mesh = trimesh.load(mesh_root / f"{obj_id}.obj", force="mesh", process=False)
        mesh_items.append((mesh, (218, 222, 227), np.zeros(3, dtype=float)))
    for idx, part in enumerate(spec.parts):
        for obj_id in part.mesh_obj_ids:
            mesh = trimesh.load(mesh_root / f"{obj_id}.obj", force="mesh", process=False)
            local_center = mesh.vertices.mean(axis=0)
            direction = local_center - global_center
            if np.linalg.norm(direction) > 1.0e-8:
                direction = direction / np.linalg.norm(direction)
            offset = direction * 0.24
            mesh_items.append((mesh, PART_COLORS[idx], offset))
    render_meshes(mesh_items, out)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *, dashed: bool = False) -> None:
    color = (118, 126, 138)
    if dashed:
        x0, y0 = start
        x1, y1 = end
        segments = 14
        for i in range(segments):
            if i % 2 == 0:
                xa = x0 + (x1 - x0) * i / segments
                ya = y0 + (y1 - y0) * i / segments
                xb = x0 + (x1 - x0) * (i + 1) / segments
                yb = y0 + (y1 - y0) * (i + 1) / segments
                draw.line((xa, ya, xb, yb), fill=color, width=4)
    else:
        draw.line((*start, *end), fill=color, width=4)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 18
    left = (end[0] - length * math.cos(angle - 0.45), end[1] - length * math.sin(angle - 0.45))
    right = (end[0] - length * math.cos(angle + 0.45), end[1] - length * math.sin(angle + 0.45))
    draw.polygon([end, left, right], fill=color)


def box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], label: str, *, dash: bool = False) -> None:
    if dash:
        x0, y0, x1, y1 = xy
        draw.rounded_rectangle(xy, radius=24, fill=(250, 250, 250), outline=(150, 156, 166), width=3)
        for x in range(x0, x1, 26):
            draw.line((x, y0, min(x + 13, x1), y0), fill=WHITE, width=5)
            draw.line((x, y1, min(x + 13, x1), y1), fill=WHITE, width=5)
        for y in range(y0, y1, 26):
            draw.line((x0, y, x0, min(y + 13, y1)), fill=WHITE, width=5)
            draw.line((x1, y, x1, min(y + 13, y1)), fill=WHITE, width=5)
    else:
        draw.rounded_rectangle(xy, radius=22, fill=(247, 248, 250), outline=(153, 160, 170), width=3)
    draw.text(((xy[0] + xy[2]) // 2, (xy[1] + xy[3]) // 2), label, anchor="mm", fill=(91, 98, 110), font=FONT_28)


def make_layout_wireframe(out: Path) -> None:
    canvas = Image.new("RGB", (2800, 1200), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((50, 80, 1110, 1120), radius=36, outline=(190, 196, 205), width=4)
    draw.rounded_rectangle((1160, 80, 2750, 1120), radius=36, outline=(190, 196, 205), width=4)
    draw.text((95, 135), "(a) Pipeline Overview", fill=(80, 86, 96), font=FONT_36)
    draw.text((1205, 135), "(b) Promptable Part Segmentation", fill=(80, 86, 96), font=FONT_36)

    box(draw, (100, 260, 330, 520), "1")
    box(draw, (430, 270, 620, 505), "V")
    box(draw, (725, 200, 855, 285), "E1")
    box(draw, (710, 335, 910, 435), "F")
    box(draw, (760, 490, 990, 690), "3")
    box(draw, (470, 760, 780, 920), "2")
    draw_arrow(draw, (330, 390), (430, 390))
    draw_arrow(draw, (620, 325), (725, 245))
    draw_arrow(draw, (790, 285), (805, 335))
    draw_arrow(draw, (810, 435), (860, 490))
    draw_arrow(draw, (620, 390), (555, 760))
    draw_arrow(draw, (780, 840), (1160, 840))

    box(draw, (1220, 255, 1430, 405), "3")
    box(draw, (1490, 250, 1640, 405), "E1")
    box(draw, (1710, 245, 1890, 405), "6")
    box(draw, (1980, 330, 2240, 570), "T")
    box(draw, (2320, 340, 2465, 545), "E2")
    box(draw, (2540, 285, 2710, 460), "4")
    box(draw, (2540, 535, 2710, 710), "5")
    box(draw, (1535, 705, 1730, 835), "P")
    box(draw, (1280, 740, 1480, 835), "N")
    box(draw, (1770, 720, 2240, 1015), "training-only\n7        8", dash=True)
    box(draw, (2290, 805, 2475, 905), "residual")
    box(draw, (2520, 805, 2700, 905), "argmax")
    draw_arrow(draw, (1430, 330), (1490, 330))
    draw_arrow(draw, (1640, 330), (1710, 330))
    draw_arrow(draw, (1890, 330), (1980, 450))
    draw_arrow(draw, (1730, 770), (1980, 520))
    draw_arrow(draw, (1480, 790), (1535, 770))
    draw_arrow(draw, (2240, 450), (2320, 445))
    draw_arrow(draw, (2465, 445), (2540, 372))
    draw_arrow(draw, (2625, 460), (2625, 535))
    draw_arrow(draw, (2540, 650), (2475, 850))
    draw_arrow(draw, (2630, 710), (2630, 805))
    draw_arrow(draw, (2100, 570), (1930, 720), dashed=True)
    draw_arrow(draw, (2100, 570), (2230, 720), dashed=True)
    canvas.save(out)


def build_object(spec: ObjectSpec) -> list[str]:
    out_dir = OUT_ROOT / spec.obj_id
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    makers = [
        ("input_views.png", lambda p: make_input_views(spec, p), "4 selected input render views, 2x2 on white."),
        ("vlm_masks.png", lambda p: make_vlm_masks(spec, p), "One real render with transparent part masks and part-name labels."),
        ("voxel_whole.png", lambda p: make_voxel_whole(spec, p), "64^3 whole occupancy from ee-eval voxel.npz, rendered as shaded cubes."),
        ("voxel_parts.png", lambda p: make_voxel_parts(spec, p), "Promptable-seg part voxels from ee-eval part_XX_voxel.npz, colored part1/part2/part3/body."),
        ("parts_mesh.png", lambda p: make_parts_mesh(spec, p), "Exploded colored part mesh render from phyx-verse part OBJ assets."),
        ("latent_grid.png", make_latent_grid, "Program-generated 16^3 latent grid cube array."),
        ("boundary_band.png", lambda p: make_boundary_band(spec, p), "Local part-body boundary close-up with a 2-voxel red band."),
        ("corruption_pair.png", lambda p: make_corruption_pair(spec, p), "Clean vs corrupted full occupancy using the training corrupt_voxel_occ implementation."),
    ]
    for name, maker, desc in makers:
        path = out_dir / name
        maker(path)
        outputs.append(f"{spec.obj_id}/{name}: {desc}")
    return outputs


def main() -> int:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    descriptions: list[str] = []
    for spec in OBJECTS:
        descriptions.append(f"{spec.obj_id}/ ({spec.alias})")
        descriptions.extend(build_object(spec))
        descriptions.append("")
    make_layout_wireframe(OUT_ROOT / "layout_wireframe.png")
    descriptions.append("layout_wireframe.png: Gray layout guide for image2 composition, with numbered asset placeholders and module nodes.")
    readme = [
        "# PartSeg Figure Assets",
        "",
        "All PNGs are generated by code from local renders, masks, voxel outputs, or mesh assets.",
        "Backgrounds are white, with no axes or coordinate grids. Part colors are fixed: part1=blue, part2=green, part3=orange, body=light gray.",
        "",
        "## Files",
        "",
    ]
    readme.extend(f"- {line}" for line in descriptions if line)
    (OUT_ROOT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"wrote {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
