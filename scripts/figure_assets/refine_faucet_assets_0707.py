#!/usr/bin/env python3
"""Refine selected faucet figure assets without touching other objects."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path("/root/code/arts-gen")
OBJ_ID = "01816801a27444cbb5cfb934de39d483"
DATA_ROOT = Path("/robot/data-lab/jzh/art-gen/data/phyx-verse")
RENDER_ROOT = DATA_ROOT / "renders" / OBJ_ID / "angle_0"
MESH_ROOT = DATA_ROOT / "raw" / "partseg" / OBJ_ID / "objs"
OUT_DIR = ROOT / "figure_assets" / OBJ_ID
PREVIEW_DIR = ROOT / "figure_out" / "faucet_refine_0707"

WHITE = (255, 255, 255)
PANEL = (226, 232, 240)
TEXT = (70, 78, 92)
PART_COLORS = [
    (73, 139, 205),   # part1 blue: left handle
    (91, 168, 111),   # part2 green: right handle
    (226, 143, 65),   # part3 orange: spout
]
BODY_COLOR = (206, 211, 217)
WHOLE_COLOR = (200, 207, 214)
VIEW_INDICES = (0, 5, 6, 11)
MASK_LABELS = (2, 3, 4)
PART_NAMES = ("part1 left handle", "part2 right handle", "part3 spout")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


F18 = font(18)
F20 = font(20)
F24 = font(24, bold=True)
F28 = font(28, bold=True)
F36 = font(36, bold=True)


def alpha_on_white(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, WHITE + (255,))
    bg.alpha_composite(im)
    return bg.convert("RGB")


def paste_fit(canvas: Image.Image, im: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    max_w, max_h = x1 - x0, y1 - y0
    scale = min(max_w / im.width, max_h / im.height)
    new_size = (max(1, int(im.width * scale)), max(1, int(im.height * scale)))
    resized = im.resize(new_size, Image.Resampling.LANCZOS)
    px = x0 + (max_w - new_size[0]) // 2
    py = y0 + (max_h - new_size[1]) // 2
    canvas.paste(resized, (px, py))


def overlay_mask_view(view_idx: int) -> Image.Image:
    base = alpha_on_white(RENDER_ROOT / "rgb" / f"view_{view_idx}.png").convert("RGBA")
    mask = np.load(RENDER_ROOT / "mask" / f"mask_{view_idx}.npy")
    arr = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    for label, color in zip(MASK_LABELS, PART_COLORS):
        region = mask == int(label)
        arr[region, :3] = color
        arr[region, 3] = 128
    composed = Image.alpha_composite(base, Image.fromarray(arr, "RGBA"))
    return composed.convert("RGB")


def make_four_view_part_masks() -> Image.Image:
    canvas = Image.new("RGB", (1400, 1400), WHITE)
    draw = ImageDraw.Draw(canvas)
    boxes = [
        (90, 90, 675, 675),
        (725, 90, 1310, 675),
        (90, 725, 675, 1310),
        (725, 725, 1310, 1310),
    ]
    for view_idx, box in zip(VIEW_INDICES, boxes):
        draw.rounded_rectangle(box, radius=28, fill=WHITE, outline=PANEL, width=2)
        im = overlay_mask_view(view_idx)
        paste_fit(canvas, im, (box[0] + 28, box[1] + 28, box[2] - 28, box[3] - 28))
        draw.rounded_rectangle((box[0] + 18, box[1] + 18, box[0] + 116, box[1] + 54), radius=12, fill=WHITE, outline=(199, 207, 219), width=2)
        draw.text((box[0] + 35, box[1] + 23), f"view {view_idx}", fill=TEXT, font=F20)
    legend_x, legend_y = 384, 1334
    for idx, (name, color) in enumerate(zip(PART_NAMES, PART_COLORS)):
        x = legend_x + idx * 235
        draw.rounded_rectangle((x, legend_y - 12, x + 26, legend_y + 14), radius=6, fill=color)
        draw.text((x + 36, legend_y - 16), name, fill=color, font=F20)
    return canvas


def color_lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(a[i] * (1.0 - t) + b[i] * t)) for i in range(3))


def token_color(i: int, total: int) -> tuple[int, int, int]:
    anchors = [
        (114, 169, 218),
        (124, 198, 160),
        (240, 190, 105),
        (183, 154, 219),
        (102, 183, 205),
    ]
    phase = (i / max(1, total - 1)) * (len(anchors) - 1)
    lo = int(math.floor(phase))
    hi = min(lo + 1, len(anchors) - 1)
    return color_lerp(anchors[lo], anchors[hi], phase - lo)


def make_latent_strip() -> Image.Image:
    canvas = Image.new("RGB", (1400, 1400), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((700, 410), "Latent feature tokens", anchor="mm", fill=TEXT, font=F36)
    n = 24
    gap = 8
    token_w = 38
    token_h = 132
    total_w = n * token_w + (n - 1) * gap
    x0 = (1400 - total_w) // 2
    y0 = 622
    for i in range(n):
        x = x0 + i * (token_w + gap)
        color = token_color(i, n)
        draw.rounded_rectangle((x, y0, x + token_w, y0 + token_h), radius=8, fill=color, outline=(238, 242, 247), width=2)
        shine = color_lerp(color, WHITE, 0.28)
        draw.rounded_rectangle((x + 5, y0 + 7, x + token_w - 5, y0 + 28), radius=5, fill=shine)
    draw.text((700, 826), "compact prompt-conditioned feature strip", anchor="mm", fill=(112, 120, 135), font=F24)
    return canvas


def make_latent_matrix() -> Image.Image:
    canvas = Image.new("RGB", (1400, 1400), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((700, 260), "Latent feature token matrix", anchor="mm", fill=TEXT, font=F36)
    rows, cols = 8, 16
    cell = 48
    gap = 8
    total_w = cols * cell + (cols - 1) * gap
    total_h = rows * cell + (rows - 1) * gap
    x0 = (1400 - total_w) // 2
    y0 = 430
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            color = token_color(idx, rows * cols)
            # Add a mild deterministic value variation so it reads as features, not a grid in space.
            wave = 0.16 * math.sin(idx * 0.73) + 0.10 * math.cos((r - c) * 0.91)
            color = color_lerp(color, WHITE if wave > 0 else (80, 92, 110), abs(wave))
            x = x0 + c * (cell + gap)
            y = y0 + r * (cell + gap)
            draw.rounded_rectangle((x, y, x + cell, y + cell), radius=8, fill=color, outline=(241, 244, 249), width=2)
    draw.text((700, y0 + total_h + 95), "128 learned feature tokens; color indicates feature values", anchor="mm", fill=(112, 120, 135), font=F24)
    return canvas


def shade_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * factor + 255 * (1 - factor) * 0.06))) for c in color)


def camera_project(vertices: np.ndarray, center: np.ndarray, span: float, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # OBJ faucet assets use Y as the upright axis; this projection matches the ee-eval upright convention visually.
    p = (vertices - center[None, :]) / max(span, 1.0e-6) + offset[None, :]
    az = math.radians(-38.0)
    ca, sa = math.cos(az), math.sin(az)
    x = p[:, 0] * ca - p[:, 2] * sa
    z = p[:, 0] * sa + p[:, 2] * ca
    y = p[:, 1]
    sx = x * 740.0 + 700.0
    sy = (-y * 760.0 + z * 155.0) + 735.0
    depth = z * 1.1 + x * 0.25 - y * 0.08
    return np.stack([sx, sy], axis=1), depth, np.stack([x, y, z], axis=1)


def render_mesh_items(
    mesh_items: list[tuple[trimesh.Trimesh, tuple[int, int, int], np.ndarray]],
    out: Path,
    *,
    title: str | None = None,
) -> Image.Image:
    canvas = Image.new("RGB", (1400, 1400), WHITE)
    draw = ImageDraw.Draw(canvas)
    if not mesh_items:
        canvas.save(out)
        return canvas
    all_vertices = np.concatenate([mesh.vertices for mesh, _color, _off in mesh_items], axis=0)
    center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) / 2.0
    span = float(np.max(all_vertices.max(axis=0) - all_vertices.min(axis=0)))
    light = np.asarray([-0.35, 0.85, 0.45], dtype=np.float64)
    light = light / np.linalg.norm(light)
    polys = []
    for mesh, color, offset in mesh_items:
        screen, depth, cam = camera_project(mesh.vertices, center, span, offset)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        v0, v1, v2 = cam[faces[:, 0]], cam[faces[:, 1]], cam[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        nlen = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(nlen, 1.0e-8)
        shade = np.clip(0.62 + 0.38 * np.maximum(normals @ light, 0.0), 0.42, 1.08)
        face_depth = depth[faces].mean(axis=1)
        for fidx, face in enumerate(faces):
            poly = screen[face]
            if np.any(poly[:, 0] < -300) or np.any(poly[:, 0] > 1700) or np.any(poly[:, 1] < -300) or np.any(poly[:, 1] > 1700):
                continue
            polys.append((float(face_depth[fidx]), poly, shade_color(color, float(shade[fidx]))))
    polys.sort(key=lambda item: item[0])
    for _depth, poly, color in polys:
        draw.polygon([tuple(p) for p in poly], fill=color)
    if title:
        draw.text((700, 74), title, anchor="mm", fill=TEXT, font=F28)
    canvas = canvas.filter(ImageFilter.SMOOTH_MORE)
    canvas.save(out)
    return canvas


def make_mesh_renders() -> tuple[Image.Image, Image.Image]:
    body_ids = (0,)
    part_ids = (1, 2, 3)
    all_meshes: list[trimesh.Trimesh] = []
    for idx in body_ids + part_ids:
        all_meshes.append(trimesh.load(MESH_ROOT / f"{idx}.obj", force="mesh", process=False))
    global_vertices = np.concatenate([m.vertices for m in all_meshes], axis=0)
    global_center = (global_vertices.min(axis=0) + global_vertices.max(axis=0)) / 2.0

    overall_items = [(mesh, WHOLE_COLOR, np.zeros(3, dtype=float)) for mesh in all_meshes]
    overall = render_mesh_items(overall_items, PREVIEW_DIR / "07_parts_mesh_overall_orientation_check.png", title="overall orientation check")

    part_items: list[tuple[trimesh.Trimesh, tuple[int, int, int], np.ndarray]] = []
    for idx in body_ids:
        mesh = trimesh.load(MESH_ROOT / f"{idx}.obj", force="mesh", process=False)
        part_items.append((mesh, BODY_COLOR, np.zeros(3, dtype=float)))
    for color_idx, idx in enumerate(part_ids):
        mesh = trimesh.load(MESH_ROOT / f"{idx}.obj", force="mesh", process=False)
        direction = mesh.vertices.mean(axis=0) - global_center
        # Keep explosion in the horizontal object plane; do not rotate or lay parts down.
        direction[1] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm > 1.0e-8:
            direction = direction / norm
        offset = direction * 0.12
        part_items.append((mesh, PART_COLORS[color_idx], offset.astype(float)))
    parts = render_mesh_items(part_items, OUT_DIR / "parts_mesh.png")
    return overall, parts


def make_preview(paths: list[Path], out: Path) -> None:
    thumbs = []
    for path in paths:
        im = Image.open(path).convert("RGB")
        im.thumbnail((610, 430), Image.Resampling.LANCZOS)
        thumbs.append((path.name, im.copy()))
    w, h = 2 * 720, 3 * 520
    canvas = Image.new("RGB", (w, h), WHITE)
    draw = ImageDraw.Draw(canvas)
    for idx, (name, im) in enumerate(thumbs):
        col, row = idx % 2, idx // 2
        x0 = col * 720
        y0 = row * 520
        label = name
        if len(label) > 42:
            label = label[:39] + "..."
        draw.text((x0 + 26, y0 + 20), label, fill=TEXT, font=F24)
        px = x0 + (720 - im.width) // 2
        py = y0 + 74 + (430 - im.height) // 2
        canvas.paste(im, (px, py))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    part_masks = make_four_view_part_masks()
    for name in ("vlm_masks.png", "04_part_masks.png"):
        part_masks.save(OUT_DIR / name)

    strip = make_latent_strip()
    matrix = make_latent_matrix()
    strip.save(OUT_DIR / "06_latent_grid_strip.png")
    matrix.save(OUT_DIR / "06_latent_grid_matrix.png")
    matrix.save(OUT_DIR / "06_latent_grid.png")
    matrix.save(OUT_DIR / "latent_grid.png")

    _overall, parts = make_mesh_renders()
    parts.save(OUT_DIR / "07_parts_mesh.png")

    make_preview(
        [
            OUT_DIR / "input_views.png",
            OUT_DIR / "04_part_masks.png",
            OUT_DIR / "06_latent_grid_strip.png",
            OUT_DIR / "06_latent_grid_matrix.png",
            PREVIEW_DIR / "07_parts_mesh_overall_orientation_check.png",
            OUT_DIR / "07_parts_mesh.png",
        ],
        PREVIEW_DIR / "refined_faucet_assets_preview.png",
    )
    print(f"wrote {OUT_DIR}")
    print(f"preview {PREVIEW_DIR / 'refined_faucet_assets_preview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
