"""Build an academic-style Part SS Latent Flow schematic as SVG + PNG/PDF.

This script deliberately avoids the previous matplotlib-card look.  It uses
SVG for crisp paper-style geometry and embeds only the real smoke-test assets
that should stay faithful: RGB views, masks, DINO PCA maps, and W heatmaps.

Run:
    /home/mi/anaconda3/envs/arts-gen/bin/python scripts/tools/make_part_ss_latent_flow_academic_svg.py
"""

from __future__ import annotations

import base64
import html
import subprocess
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from make_part_ss_latent_flow_ppt_figure import (
    BLUE,
    GRAY,
    GREEN,
    INK,
    ORANGE,
    PART_COLORS,
    PURPLE,
    RED,
    dino_pca_maps,
    latent_thumbnail,
    load_assets,
    mask_rgb,
    overlap_heatmap,
)


ROOT = Path("/home/mi/jzh/AAAI2027/arts-reconstruction")
OUT_DIR = ROOT / "docs/figures"
OUT_SVG = OUT_DIR / "part_ss_latent_flow_mask_overlap_pooling_v7.svg"
OUT_HTML = OUT_DIR / "part_ss_latent_flow_mask_overlap_pooling_v7.html"
OUT_PNG = OUT_DIR / "part_ss_latent_flow_mask_overlap_pooling_v7.png"
OUT_PDF = OUT_DIR / "part_ss_latent_flow_mask_overlap_pooling_v7.pdf"

W, H = 1600, 900


def data_uri(img: np.ndarray | Image.Image) -> str:
    if isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
    else:
        image = img
    buf = BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def heat_uri(arr: np.ndarray) -> str:
    rgb = plt.cm.YlOrBr(0.08 + 0.92 * arr)[..., :3]
    return data_uri(rgb)


def svg_rect(x, y, w, h, stroke, fill="white", sw=2, rx=10, dash=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
    )


def svg_text(x, y, text, size=16, color=INK, weight=500, anchor="middle", family="DejaVu Sans"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{color}">{html.escape(text)}</text>'
    )


def svg_image(uri, x, y, w, h, stroke="#d8d8d8"):
    return (
        f'<image href="{uri}" x="{x}" y="{y}" width="{w}" height="{h}" '
        f'preserveAspectRatio="xMidYMid meet"/>'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" '
        f'stroke="{stroke}" stroke-width="1"/>'
    )


def arrow(x1, y1, x2, y2, color, sw=2.3):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{sw}" marker-end="url(#{color[1:]}Arrow)"/>'
    )


def polyline(points, color, sw=2.3):
    pts = " ".join(f"{x},{y}" for x, y in points)
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}" '
        f'stroke-linejoin="round" marker-end="url(#{color[1:]}Arrow)"/>'
    )


def token_grid(x, y, rows, cols, color, cell=8, gap=3, alpha=0.78):
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(
                f'<rect x="{x + c * (cell + gap):.1f}" y="{y + r * (cell + gap):.1f}" '
                f'width="{cell}" height="{cell}" fill="{color}" fill-opacity="{alpha}" '
                f'stroke="{color}" stroke-width="0.7"/>'
            )
    return "\n".join(out)


def memory_block(x, y, w, h, color, title, shape, rows, cols):
    title_size = 14 if w < 120 else 17
    shape_size = 11 if w < 120 else 13
    return "\n".join([
        svg_rect(x, y, w, h, color, "white", 2, 8),
        svg_text(x + 12, y + 23, title, title_size, color, 700, "start"),
        svg_text(x + w - 10, y + 23, shape, shape_size, GRAY, 500, "end"),
        token_grid(x + 18, y + 43, rows, cols, color, cell=9, gap=4, alpha=0.72),
    ])


def mini_cube(x, y, s, color, label):
    shade = color
    return "\n".join([
        f'<polygon points="{x},{y+s} {x+s},{y+s} {x+s},{y+2*s} {x},{y+2*s}" '
        f'fill="{shade}" fill-opacity="0.45" stroke="{color}" stroke-width="2"/>',
        f'<polygon points="{x},{y+s} {x+s*0.35},{y+s*0.68} {x+s*1.35},{y+s*0.68} {x+s},{y+s}" '
        f'fill="{shade}" fill-opacity="0.25" stroke="{color}" stroke-width="2"/>',
        f'<polygon points="{x+s},{y+s} {x+s*1.35},{y+s*0.68} {x+s*1.35},{y+s*1.68} {x+s},{y+2*s}" '
        f'fill="{shade}" fill-opacity="0.32" stroke="{color}" stroke-width="2"/>',
        svg_text(x + s * 0.65, y + 2 * s + 22, label, 13, color, 700),
    ])


def build_svg() -> str:
    part_info, view_ids, rgbs, masks, tokens, latent, voxel_parts = load_assets()
    dino_maps = dino_pca_maps(tokens, view_ids)
    heat = overlap_heatmap(masks, part_info["num_parts"])

    rgb_uris = [data_uri(im) for im in rgbs[:4]]
    mask_uris = [data_uri(mask_rgb(m, part_info["num_parts"])) for m in masks[:4]]
    dino_uris = [data_uri(dm) for dm in dino_maps[:4]]
    heatmap_uri = heat_uri(heat)
    latent_uri = data_uri(latent_thumbnail(latent))

    defs = f"""
    <defs>
      <marker id="{BLUE[1:]}Arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{BLUE}"/></marker>
      <marker id="{GREEN[1:]}Arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{GREEN}"/></marker>
      <marker id="{ORANGE[1:]}Arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{ORANGE}"/></marker>
      <marker id="{RED[1:]}Arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{RED}"/></marker>
      <marker id="{PURPLE[1:]}Arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{PURPLE}"/></marker>
      <marker id="{GRAY[1:]}Arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{GRAY}"/></marker>
      <style>
        .small {{ font-family: DejaVu Sans, Arial, sans-serif; font-size: 12px; fill: {GRAY}; }}
        .label {{ font-family: DejaVu Sans, Arial, sans-serif; font-size: 14px; fill: {INK}; }}
      </style>
    </defs>
    """

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        defs,
        '<rect width="1600" height="900" fill="white"/>',
        svg_rect(30, 42, 1540, 810, "#6f6b80", "none", 3, 34, "4 7"),
        svg_text(800, 45, "Part SS Latent Flow Predictor", 30, "#5c5877", 800, family="DejaVu Serif"),
        '<line x1="70" y1="45" x2="610" y2="45" stroke="#6f6b80" stroke-width="2" stroke-dasharray="4 7"/>',
        '<line x1="990" y1="45" x2="1530" y2="45" stroke="#6f6b80" stroke-width="2" stroke-dasharray="4 7"/>',
    ]

    # Left evidence panels.
    svg += [
        svg_rect(70, 115, 180, 190, GREEN, "#fbfffb", 2, 14),
        svg_text(160, 140, "4-view RGB", 14, GREEN, 700),
    ]
    for i, uri in enumerate(rgb_uris):
        x = 92 + (i % 2) * 76
        y = 156 + (i // 2) * 66
        svg.append(svg_image(uri, x, y, 58, 58, "#c9d7c6"))

    svg += [
        svg_rect(70, 345, 180, 190, "#2b7c91", "#fbfdff", 2, 14),
        svg_text(160, 370, "4-view masks", 14, "#2b7c91", 700),
    ]
    for i, uri in enumerate(mask_uris):
        x = 92 + (i % 2) * 76
        y = 386 + (i // 2) * 62
        svg.append(svg_image(uri, x, y, 58, 52, "#b9d5de"))

    svg += [
        svg_rect(70, 575, 180, 170, RED, "#fffafa", 2, 14),
        svg_text(160, 600, "global SS latent", 14, RED, 700),
        mini_cube(96, 625, 38, RED, "z_global"),
        svg_image(latent_uri, 166, 625, 54, 54, "#f0bab2"),
    ]

    # DINO encoder and token branch.
    svg += [
        polyline([(250, 210), (285, 210)], GREEN),
        svg_rect(285, 135, 90, 340, "#6a6480", "#fbfbff", 2, 14),
        svg_text(330, 285, "DINOv2", 15, "#5c5877", 700),
        svg_text(330, 307, "Encoder", 12, "#5c5877", 500),
    ]
    for k in range(4):
        svg.append(f'<polygon points="{307+k*8},{185+k*4} {329+k*8},{195+k*4} {329+k*8},{280+k*4} {307+k*8},{270+k*4}" fill="#d9d4e8" stroke="#7b7590" stroke-width="1"/>')
    for i, uri in enumerate(dino_uris):
        svg.append(svg_image(uri, 400 + i * 54, 145, 44, 44, "#c9d0ef"))
    svg.append(memory_block(398, 210, 198, 92, BLUE, "X_img", "[V*T,1024]", 3, 14))
    svg.append(polyline([(375, 275), (398, 275)], BLUE))

    # Weight branch.
    svg += [
        polyline([(250, 440), (285, 440), (285, 410), (398, 410)], "#2b7c91"),
        svg_rect(398, 345, 198, 132, ORANGE, "#fffdf6", 2, 14),
        svg_text(497, 372, "mask-overlap weights", 14, ORANGE, 700),
        svg_text(497, 396, "W = overlap(mask, patch)", 12, ORANGE, 700),
        svg_image(heatmap_uri, 430, 412, 132, 42, ORANGE),
        svg_text(497, 468, "W [K,V*T], weights only", 12, GRAY, 500),
    ]

    # Global branch.
    svg += [
        polyline([(250, 655), (285, 655), (285, 625), (398, 625)], RED),
        svg_rect(398, 575, 198, 132, RED, "#fffafa", 2, 14),
        svg_text(497, 602, "global tokenization", 14, RED, 700),
        svg_text(497, 628, "patchify P=2 + 3D APE", 12, RED, 500),
        memory_block(430, 645, 132, 48, RED, "Z_global", "[512,D]", 2, 10),
    ]

    # Condition builder in the reference-paper style.
    svg += [
        svg_rect(645, 104, 520, 620, PURPLE, "#fffefe", 2, 18),
        svg_text(905, 130, "Condition Builder", 18, PURPLE, 800),
        svg_rect(675, 160, 220, 130, ORANGE, "#fffdf6", 2, 14),
        svg_text(785, 184, "Image Memory Projection", 14, ORANGE, 700),
        memory_block(690, 202, 88, 62, BLUE, "X_img", "[V*T,1024]", 2, 6),
        svg_rect(803, 214, 56, 38, BLUE, "#f7faff", 1.6, 7),
        svg_text(831, 238, "Linear", 11, BLUE, 700),
        memory_block(878, 202, 92, 62, BLUE, "X_proj", "[V*T,D]", 2, 6),
        arrow(778, 233, 803, 233, BLUE, 1.8),
        arrow(859, 233, 878, 233, BLUE, 1.8),
        svg_rect(675, 330, 300, 165, ORANGE, "#fffdf6", 2, 14),
        svg_text(825, 354, "Mask-Overlap Weighted Pooling", 14, ORANGE, 700),
        svg_image(heatmap_uri, 700, 382, 122, 50, ORANGE),
        svg_text(761, 448, "W [K,V*T]", 12, ORANGE, 700),
        svg_rect(850, 386, 78, 38, ORANGE, "#fff9eb", 1.6, 7),
        svg_text(889, 410, "W @ X_proj", 11, ORANGE, 700),
        memory_block(945, 376, 92, 62, GREEN, "Q_local", "[K,D]", 3, 5),
        arrow(822, 407, 850, 407, ORANGE, 1.8),
        arrow(928, 407, 945, 407, GREEN, 1.8),
        svg_rect(675, 540, 300, 92, GREEN, "#fbfffb", 2, 14),
        svg_text(825, 565, "Trainable Query Refinement", 14, GREEN, 700),
        svg_rect(708, 586, 78, 30, GREEN, "#f6fff7", 1.5, 6),
        svg_text(747, 606, "+ slot/type", 10, GREEN, 700),
        svg_rect(810, 586, 74, 30, GREEN, "#f6fff7", 1.5, 6),
        svg_text(847, 606, "Encoder", 10, GREEN, 700),
        memory_block(910, 576, 92, 50, GREEN, "Q_part", "[K,D]", 2, 6),
        arrow(786, 601, 810, 601, GREEN, 1.8),
        arrow(884, 601, 910, 601, GREEN, 1.8),
        svg_rect(1010, 205, 120, 380, "#8a8496", "#fbfbfb", 2, 14),
        svg_text(1070, 228, "Cross-Attention", 13, "#6b657a", 700),
        svg_text(1070, 247, "Memory", 13, "#6b657a", 700),
        memory_block(1030, 282, 80, 70, GREEN, "Q_part", "[K,D]", 2, 5),
        memory_block(1030, 382, 80, 82, BLUE, "X_proj", "[V*T,D]", 3, 5),
        memory_block(1030, 496, 80, 62, RED, "Z_global", "[512,D]", 2, 5),
        svg_text(1070, 615, "M = concat(..., D)", 12, PURPLE, 700),
    ]
    svg += [
        polyline([(596, 256), (645, 256)], BLUE, 2),
        polyline([(596, 420), (645, 420)], ORANGE, 2),
        polyline([(596, 660), (630, 660), (630, 585), (675, 585)], RED, 2),
        polyline([(970, 233), (990, 233), (990, 318), (1010, 318)], BLUE, 1.8),
        arrow(1037, 407, 1010, 407, GREEN, 1.8),
        polyline([(1002, 601), (990, 601), (990, 318), (1010, 318)], GREEN, 1.8),
    ]

    # Flow + outputs.
    svg += [
        arrow(1130, 410, 1190, 410, PURPLE, 2.4),
        svg_rect(1190, 210, 150, 420, PURPLE, "#fbfaff", 2, 16),
        svg_text(1265, 238, "Flow DiT", 16, PURPLE, 800),
    ]
    for i in range(5):
        y = 285 + i * 56
        svg.append(svg_rect(1214, y, 102, 34, PURPLE, "white", 1.5, 7))
        svg.append(svg_text(1265, y + 22, "Transformer", 11, PURPLE, 700))
    svg.append(svg_text(1265, 596, "cross-attend to M", 12, GRAY, 500))
    svg.append(mini_cube(1218, 676, 48, PURPLE, "x_t part latent"))
    svg.append(arrow(1265, 676, 1265, 630, PURPLE, 2))

    svg += [
        arrow(1340, 410, 1392, 410, PURPLE, 2.4),
        mini_cube(1392, 365, 56, PURPLE, "pred part latent"),
        arrow(1450, 485, 1450, 560, GRAY, 2.2),
        svg_rect(1370, 560, 165, 95, GRAY, "#fbfbfb", 2, 12),
        svg_text(1452, 595, "Frozen SS Decoder", 15, GRAY, 800),
        svg_text(1452, 625, "latent -> voxel", 12, GRAY, 500),
        arrow(1450, 365, 1450, 300, GRAY, 2.2),
        svg_rect(1370, 125, 165, 175, GRAY, "#fbfbfb", 2, 12),
        svg_text(1452, 154, "64^3 part voxels", 15, GRAY, 800),
    ]
    for i, color in enumerate(PART_COLORS[:4]):
        x = 1400 + (i % 2) * 68
        y = 180 + (i // 2) * 56
        svg.append(mini_cube(x, y, 18, color, ""))

    # Legend.
    legend = [
        (GREEN, "part/query trainable"),
        ("#2b7c91", "mask/source"),
        ("#d9d4e8", "shared/fusion"),
        (ORANGE, "computed weights"),
        (GRAY, "prediction/output"),
        (BLUE, "frozen DINO"),
        (RED, "trainable latent"),
    ]
    x = 80
    for color, label in legend:
        svg.append(svg_rect(x, 795, 28, 18, color, "#fbfbfb", 1.5, 5))
        svg.append(svg_text(x + 38, 810, label, 11, INK, 500, "start"))
        x += 195 if len(label) > 15 else 150

    svg.append("</svg>")
    return "\n".join(svg)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    OUT_SVG.write_text(svg, encoding="utf-8")
    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>html,body{margin:0;width:1600px;height:900px;overflow:hidden}"
        "@page{size:1600px 900px;margin:0}</style></head><body>"
        + svg
        + "</body></html>"
    )
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    html_url = OUT_HTML.resolve().as_uri()
    chrome = "google-chrome"
    subprocess.run([
        chrome,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        "--window-size=1600,900",
        f"--screenshot={OUT_PNG}",
        html_url,
    ], check=True)
    subprocess.run([
        chrome,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={OUT_PDF}",
        html_url,
    ], check=True)
    print(f"saved {OUT_SVG}")
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
