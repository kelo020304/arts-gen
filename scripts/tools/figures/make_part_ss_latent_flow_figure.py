"""Generate publication-quality figure for Mask-Overlap Weighted Part SS Latent Flow.

Style: Demo-JEPA / Dreamer Predictor (CVPR / NeurIPS / SIGGRAPH Asia).
Aspect: 16:8 (2:1) — fits a one-column-wide slide or 2-column paper figure.

Inputs: real samples from data/smoke_test/1.
Output: docs/figures/part_ss_latent_flow_module.png  +  .pdf

Run:
    /home/mi/anaconda3/envs/arts-gen/bin/python scripts/tools/make_part_ss_latent_flow_figure.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

# ─────────────────────────── Config ───────────────────────────
ROOT = Path("/home/mi/jzh/AAAI2027/arts-reconstruction")
DATA = ROOT / "data/smoke_test/1"
OBJ_ID = "100154"  # Container with 4 rotation_lids + 1 base_body
ANGLE = "angle_0"

RGB_DIR = DATA / f"renders/{OBJ_ID}/{ANGLE}/rgb"
MASK_DIR = DATA / f"renders/{OBJ_ID}/{ANGLE}/mask"
PART_INFO = DATA / f"reconstruction/part_info/{OBJ_ID}/part_info.json"
VOXEL_DIR = DATA / f"reconstruction/voxel_expanded/{OBJ_ID}/{ANGLE}/64"

OUT_DIR = ROOT / "docs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "part_ss_latent_flow_module.png"
OUT_PDF = OUT_DIR / "part_ss_latent_flow_module.pdf"

# ─────────────────────────── Palette ───────────────────────────
# Demo-JEPA inspired (soft, desaturated, paper-friendly)
COL_SRC = "#4A6FA5"   # blue   — source / image input
COL_MSK = "#5A9F7C"   # green  — mask / part-aware
COL_ATT = "#E8B458"   # orange — attention block
COL_FUS = "#8A7BB3"   # purple — fusion / shared
COL_OUT = "#9A9A9A"   # gray   — prediction / output
COL_BORDER = "#2C2C2C"
COL_BG = "#FCFBF7"     # cream
COL_TENSOR_BG = "#F2F0E8"  # very light gray-cream for tensor blocks

# Part colors (for masks + voxels). Use a clean qualitative scheme.
PART_COLORS = ["#E07A5F", "#F2CC8F", "#81B29A", "#3D405B", "#9E8FB2"]

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.6,
})


# ─────────────────────────── Helpers ───────────────────────────

def load_part_info():
    with open(PART_INFO) as f:
        return json.load(f)


def load_views():
    rgb_files = sorted(RGB_DIR.glob("view_*.png"))[:4]
    view_ids = [int(p.stem.split("_")[1]) for p in rgb_files]
    rgbs = [np.asarray(Image.open(p).convert("RGB")) for p in rgb_files]
    masks = [np.load(MASK_DIR / f"mask_{vid}.npy") for vid in view_ids]
    return view_ids, rgbs, masks


def mask_to_color(mask: np.ndarray, num_parts: int) -> np.ndarray:
    """Convert int label mask to RGB image using PART_COLORS, background white."""
    h, w = mask.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)  # white bg
    for label in range(1, num_parts + 1):
        if (mask == label).any():
            c = mcolors.to_rgb(PART_COLORS[(label - 1) % len(PART_COLORS)])
            for ch in range(3):
                rgb[..., ch] = np.where(mask == label, c[ch], rgb[..., ch])
    return rgb


def add_box(ax, x, y, w, h, color, label=None, *,
            fill_alpha=0.10, border_lw=1.1, radius=0.6, ls="-",
            label_color=None, label_size=9.5, label_weight="bold",
            label_pos="top", icon=None):
    """Add a rounded rectangle box with optional centered label and icon.

    icon: 'frozen' (cyan ❄) | 'trainable' (orange ★) | None.
    """
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=border_lw, edgecolor=color,
        facecolor=mcolors.to_rgba(color, alpha=fill_alpha),
        linestyle=ls,
        zorder=2,
    )
    ax.add_patch(rect)
    if label:
        lc = label_color or color
        if label_pos == "top":
            ax.text(x + w / 2, y + h - 1.2, label,
                    ha="center", va="top", fontsize=label_size,
                    fontweight=label_weight, color=lc, zorder=3)
        elif label_pos == "center":
            ax.text(x + w / 2, y + h / 2, label,
                    ha="center", va="center", fontsize=label_size,
                    fontweight=label_weight, color=lc, zorder=3)
        elif label_pos == "bottom":
            ax.text(x + w / 2, y + 0.8, label,
                    ha="center", va="bottom", fontsize=label_size,
                    fontweight=label_weight, color=lc, zorder=3)
    if icon == "frozen":
        ax.text(x + w - 0.6, y + h - 0.6, "❄", ha="right", va="top",
                fontsize=11, color="#3D8BCD", zorder=4)
    elif icon == "trainable":
        ax.text(x + w - 0.6, y + h - 0.6, "★", ha="right", va="top",
                fontsize=11, color="#E04E2B", zorder=4)


def add_tensor(ax, cx, cy, w, h, text, color=COL_BORDER, fontsize=8):
    """Small tensor shape pill."""
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.3",
        linewidth=0.6, edgecolor=color,
        facecolor=COL_TENSOR_BG,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            style="italic", color=COL_BORDER, zorder=3)


def add_arrow(ax, x1, y1, x2, y2, color=COL_BORDER, lw=1.0,
              style="->", connection="arc3,rad=0"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style + ",head_width=2.5,head_length=3",
        connectionstyle=connection,
        linewidth=lw, color=color, zorder=4, shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arrow)


def imshow_inset(ax_main, img, x, y, w, h, border_color=None, border_lw=0.6):
    """Embed an image at (x,y) with size (w,h) in main axes coordinates."""
    # Convert main-axes data coords to figure coords for add_axes.
    fig = ax_main.figure
    trans = ax_main.transData
    inv = fig.transFigure.inverted()
    (fx0, fy0) = inv.transform(trans.transform((x, y)))
    (fx1, fy1) = inv.transform(trans.transform((x + w, y + h)))
    ax_in = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0], zorder=5)
    ax_in.imshow(img)
    ax_in.set_xticks([]); ax_in.set_yticks([])
    if border_color is not None:
        for s in ax_in.spines.values():
            s.set_edgecolor(border_color)
            s.set_linewidth(border_lw)
    else:
        for s in ax_in.spines.values():
            s.set_visible(False)
    return ax_in


def voxel_inset(ax_main, voxel_dict, part_keys, x, y, w, h,
                single_part_idx=None, downsample=2, edge_alpha=0.25):
    """Render voxels as filled 3D cubes (publication-quality look).

    voxel_dict: {part_key: (N,3) int coords in 64^3 grid}
    single_part_idx: if not None, only show that one part.
    downsample: downsample factor (2 → 32^3 grid, faster + cleaner look).
    """
    fig = ax_main.figure
    trans = ax_main.transData
    inv = fig.transFigure.inverted()
    (fx0, fy0) = inv.transform(trans.transform((x, y)))
    (fx1, fy1) = inv.transform(trans.transform((x + w, y + h)))
    ax3 = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0], projection="3d", zorder=5)
    ax3.set_axis_off()

    res = 64 // downsample

    # Build a combined occupancy grid + facecolor grid
    occ = np.zeros((res, res, res), dtype=bool)
    fc = np.empty((res, res, res, 4), dtype=np.float32)
    fc[..., 3] = 0.0  # transparent default

    keys = [part_keys[single_part_idx]] if single_part_idx is not None else part_keys
    for i, k in enumerate(keys):
        coords = voxel_dict.get(k)
        if coords is None or coords.size == 0:
            continue
        ci = single_part_idx if single_part_idx is not None else i
        # Body (last big static part) gets lower alpha so movable parts show through
        is_body = (single_part_idx is None) and (i == len(keys) - 1) and coords.shape[0] > 5000
        alpha = 0.45 if is_body else 0.95
        c = mcolors.to_rgba(PART_COLORS[ci % len(PART_COLORS)], alpha=alpha)
        ds = coords // downsample
        ds = np.unique(ds, axis=0)
        valid = (ds >= 0).all(1) & (ds < res).all(1)
        ds = ds[valid]
        occ[ds[:, 0], ds[:, 1], ds[:, 2]] = True
        for ch in range(4):
            fc[ds[:, 0], ds[:, 1], ds[:, 2], ch] = c[ch]

    # Use voxels() to render
    if occ.any():
        ec = (0.15, 0.15, 0.15, edge_alpha)
        ax3.voxels(occ, facecolors=fc, edgecolors=ec, linewidth=0.25)

    ax3.set_xlim(0, res); ax3.set_ylim(0, res); ax3.set_zlim(0, res)
    ax3.set_box_aspect((1, 1, 1))
    ax3.view_init(elev=22, azim=-60)
    return ax3


# ─────────────────────────── Build figure ───────────────────────────

def build_figure():
    pi = load_part_info()
    num_parts = pi["num_parts"]
    part_keys_all = list(pi["parts"].keys())

    view_ids, rgbs, masks = load_views()

    # Load per-part voxel index arrays (N,3 in 64^3)
    voxel_dict = {}
    if VOXEL_DIR.exists():
        for k in part_keys_all:
            p = VOXEL_DIR / f"ind_{k}.npy"
            if p.exists():
                voxel_dict[k] = np.load(p)
        # base body fallback (some samples don't ind_<base>)
        if part_keys_all[-1] not in voxel_dict:
            surf = VOXEL_DIR / "surface.npy"
            if surf.exists():
                voxel_dict[part_keys_all[-1]] = np.load(surf)
    part_keys = [k for k in part_keys_all if k in voxel_dict]

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 80)
    ax.set_axis_off()

    # ───── Title ─────
    ax.text(80, 75.5, "Mask-Overlap Weighted Part SS Latent Flow",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=COL_BORDER)

    # ═══════════════════════════════════════════════════════════════
    # ZONE 1 — Inputs (x: 2-22)
    # ═══════════════════════════════════════════════════════════════
    # RGB panel
    add_box(ax, 2, 48, 20, 22, COL_SRC,
            label="Multi-View RGB", label_pos="top",
            label_size=10, fill_alpha=0.08)
    # 2x2 grid of RGB views
    for i, img in enumerate(rgbs[:4]):
        r, c = i // 2, i % 2
        x0 = 3.5 + c * 8.5
        y0 = 50 + (1 - r) * 8.5
        imshow_inset(ax, img, x0, y0, 7.5, 7.5, border_color=COL_SRC, border_lw=0.6)

    # Mask panel
    add_box(ax, 2, 17, 20, 22, COL_MSK,
            label="Multi-View Part Masks", label_pos="top",
            label_size=10, fill_alpha=0.08)
    for i, m in enumerate(masks[:4]):
        r, c = i // 2, i % 2
        x0 = 3.5 + c * 8.5
        y0 = 19 + (1 - r) * 8.5
        mc = mask_to_color(m, num_parts)
        imshow_inset(ax, mc, x0, y0, 7.5, 7.5, border_color=COL_MSK, border_lw=0.6)

    # ═══════════════════════════════════════════════════════════════
    # ZONE 2 — Encoders (x: 26-40)
    # ═══════════════════════════════════════════════════════════════
    # DINOv2 (frozen)
    add_box(ax, 26, 53, 12, 12, COL_SRC,
            label="DINOv2", label_pos="center",
            label_size=11, icon="frozen", fill_alpha=0.18)
    ax.text(32, 56.5, "frozen", ha="center", va="center",
            fontsize=7.5, style="italic", color=COL_SRC)

    # Connect RGB → DINOv2
    add_arrow(ax, 22, 59, 26, 59, lw=1.4, color=COL_SRC)

    # Mask-Overlap Pooling (parameter-free)
    add_box(ax, 26, 22, 12, 12, COL_MSK,
            label="Mask-Overlap\nPooling", label_pos="center",
            label_size=9.5, fill_alpha=0.18)
    ax.text(32, 25.5, "parameter-free", ha="center", va="center",
            fontsize=7.5, style="italic", color=COL_MSK)
    add_arrow(ax, 22, 28, 26, 28, lw=1.4, color=COL_MSK)

    # DINOv2 output tensor
    add_tensor(ax, 44, 59, 12, 3.2, r"$F_{img}\ [V, T, D]$",
               color=COL_SRC, fontsize=9)
    add_arrow(ax, 38, 59, 38.2, 59, lw=1.4, color=COL_SRC)

    # Mask-overlap output tensor
    add_tensor(ax, 44, 28, 12, 3.2, r"$W\ [K, V, T]$",
               color=COL_MSK, fontsize=9)
    add_arrow(ax, 38, 28, 38.2, 28, lw=1.4, color=COL_MSK)

    # ═══════════════════════════════════════════════════════════════
    # ZONE 3 — Part Query Builder (expanded, center, the KEY module)
    # ═══════════════════════════════════════════════════════════════
    # Big box containing the part query builder logic
    add_box(ax, 52, 11, 38, 56, COL_FUS,
            label="Part Query Builder", label_pos="top",
            label_size=11, fill_alpha=0.05, border_lw=1.4)

    # Inside: image tokens row (top)
    imshow_inset(ax, np.zeros((1, 1, 3)) + 1, 0, 0, 0.001, 0.001)  # dummy

    # Image-tokens visual (a strip representing V*T tokens)
    img_strip = np.linspace(0, 1, 60).reshape(1, -1)
    img_strip_rgb = plt.cm.Blues(0.3 + 0.6 * img_strip)[..., :3].squeeze()[None]
    imshow_inset(ax, img_strip_rgb, 56, 56.5, 30, 3, border_color=COL_SRC)
    ax.text(71, 60.5, "Image Tokens (V·T)", ha="center", va="bottom",
            fontsize=8.5, color=COL_SRC, fontweight="bold")

    # Mask-overlap weights visual (K rows × V·T cols heatmap)
    K = num_parts
    rng = np.random.default_rng(7)
    W_vis = rng.random((K, 60)) ** 2  # sparse-ish weights
    # mask out so each row is dominated near different columns
    for k in range(K):
        center = int(60 * (k + 0.5) / K)
        decay = np.exp(-((np.arange(60) - center) ** 2) / 60)
        W_vis[k] = W_vis[k] * decay + 0.1 * decay
    W_vis = W_vis / W_vis.max()
    cmap_w = plt.cm.Greens
    W_rgb = cmap_w(0.2 + 0.7 * W_vis)[..., :3]
    imshow_inset(ax, W_rgb, 56, 41, 30, 9, border_color=COL_MSK)
    ax.text(71, 50.5, "Mask-Overlap Weights  $W \\in \\mathbb{R}^{K \\times V T}$",
            ha="center", va="bottom", fontsize=8.5, color=COL_MSK, fontweight="bold")

    # Operation: weighted pooling — drawn as ⊗ ⊕
    ax.text(71, 38.5, r"$Q^{part}_k = \sum_{v,t} W_{k,v,t} \cdot F^{img}_{v,t}$",
            ha="center", va="center", fontsize=11, color=COL_FUS,
            fontweight="bold")

    # Output: K part queries — small colored boxes
    qy = 23
    qx0 = 56
    qw = 5
    qgap = 1
    qtotw = K * qw + (K - 1) * qgap
    qx_start = 71 - qtotw / 2
    for k in range(K):
        cx = qx_start + k * (qw + qgap)
        rect = FancyBboxPatch(
            (cx, qy), qw, 6.5,
            boxstyle="round,pad=0.0,rounding_size=0.4",
            linewidth=0.8, edgecolor=PART_COLORS[k % len(PART_COLORS)],
            facecolor=mcolors.to_rgba(PART_COLORS[k % len(PART_COLORS)], alpha=0.3),
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx + qw / 2, qy + 3.2, f"$q_{k+1}$",
                ha="center", va="center", fontsize=9.5, color=COL_BORDER)
    ax.text(71, 21, r"Per-Part Queries  $Q^{part}\ [K, D]$",
            ha="center", va="top", fontsize=9, color=COL_FUS,
            fontweight="bold")

    # Inputs into builder — arrows from tensors at zone 2
    add_arrow(ax, 50, 59, 56, 58, lw=1.3, color=COL_SRC)
    add_arrow(ax, 50, 28, 56, 45, lw=1.3, color=COL_MSK,
              connection="arc3,rad=0.2")

    # ═══════════════════════════════════════════════════════════════
    # ZONE 4 — Flow DiT (x: 96-130)
    # ═══════════════════════════════════════════════════════════════
    add_box(ax, 96, 13, 28, 54, COL_ATT,
            label="Flow DiT", label_pos="top",
            label_size=12, fill_alpha=0.07, border_lw=1.5, icon="trainable")

    # Internal: stacked transformer layers (thin bars)
    n_layers = 6
    for i in range(n_layers):
        ly = 22 + i * 5.5
        bar = FancyBboxPatch(
            (100, ly), 20, 4,
            boxstyle="round,pad=0.0,rounding_size=0.3",
            linewidth=0.5, edgecolor=COL_ATT,
            facecolor=mcolors.to_rgba(COL_ATT, alpha=0.2 + 0.05 * i),
            zorder=3,
        )
        ax.add_patch(bar)
        if i == n_layers - 1:
            ax.text(110, ly + 2, "DiT Block × N", ha="center", va="center",
                    fontsize=8.5, color=COL_BORDER, fontweight="bold")
        else:
            ax.text(110, ly + 2, "self-attn  +  cross-attn  +  MLP",
                    ha="center", va="center", fontsize=7.2, color="#555")

    # Inputs into Flow DiT
    # (a) Q_part from below-left
    add_arrow(ax, 90, 26, 96, 32, lw=1.6, color=COL_FUS)
    ax.text(93, 23, r"$Q^{part}$", ha="center", va="center",
            fontsize=9.5, color=COL_FUS, fontweight="bold")

    # (b) Image Tokens (cross-attention K/V) from above (re-tap from DINOv2 strip)
    # Draw a routed arrow from top of part query builder image strip down to DiT
    add_arrow(ax, 86, 58, 96, 55, lw=1.3, color=COL_SRC,
              connection="arc3,rad=-0.15")
    ax.text(94, 60, "K, V", ha="center", va="center",
            fontsize=8.5, color=COL_SRC, fontweight="bold")

    # (c) Global SS Latent z_t (from bottom)
    add_tensor(ax, 110, 9.5, 16, 3.2, r"$z_t\ [N, C]$  +  $t$",
               color=COL_FUS, fontsize=9)
    add_arrow(ax, 110, 11.2, 110, 13, lw=1.4, color=COL_FUS)
    ax.text(110, 6.5, "Global SS Latents  +  timestep",
            ha="center", va="center", fontsize=8, color="#555",
            style="italic")

    # ═══════════════════════════════════════════════════════════════
    # ZONE 5 — Output (x: 130-158)
    # ═══════════════════════════════════════════════════════════════
    add_arrow(ax, 124, 40, 130, 40, lw=1.6, color=COL_OUT)
    add_tensor(ax, 134, 40, 8, 3.2, r"$\hat z_0\ [K, \cdot]$",
               color=COL_OUT, fontsize=9)
    add_arrow(ax, 138, 40, 142, 40, lw=1.6, color=COL_OUT)

    # Per-Part Voxels — stacked 3D scatter previews
    add_box(ax, 130, 13, 28, 22, COL_OUT,
            label="Per-Part Voxels  $\\,\\hat V\\ [K, 64^3]$",
            label_pos="bottom", label_size=10, fill_alpha=0.06, border_lw=1.2,
            label_color=COL_OUT)

    # Place K mini voxel views in 2 rows
    if part_keys:
        K_show = min(len(part_keys), 4)  # show up to 4 parts
        cols = 2
        rows = 2
        vw, vh = 12, 8
        x0 = 132
        y0 = 17
        for idx in range(K_show):
            r, c = idx // cols, idx % cols
            xv = x0 + c * (vw + 1)
            yv = y0 + (rows - 1 - r) * (vh + 0.5)
            voxel_inset(ax, voxel_dict, part_keys, xv, yv, vw, vh,
                        single_part_idx=idx)

    # Final big output — combined voxel
    add_box(ax, 130, 39, 28, 24, COL_OUT,
            label="Reconstructed Object", label_pos="top",
            label_size=10, fill_alpha=0.04, border_lw=1.2,
            label_color=COL_OUT)
    voxel_inset(ax, voxel_dict, part_keys, 132, 42, 24, 17)
    # also add part-color tinted scatter (all parts overlaid)

    # ═══════════════════════════════════════════════════════════════
    # Legend at bottom
    # ═══════════════════════════════════════════════════════════════
    legend_y = 4
    legend_items = [
        (COL_SRC, "Image Branch"),
        (COL_MSK, "Mask Branch"),
        (COL_FUS, "Fusion / Part Query"),
        (COL_ATT, "Flow DiT"),
        (COL_OUT, "Output"),
    ]
    total_w = 0
    item_w = 22
    start_x = 80 - (len(legend_items) * item_w) / 2
    for i, (c, label) in enumerate(legend_items):
        x = start_x + i * item_w
        rect = FancyBboxPatch(
            (x, legend_y - 1.2), 3, 2.4,
            boxstyle="round,pad=0.0,rounding_size=0.3",
            linewidth=0.8, edgecolor=c,
            facecolor=mcolors.to_rgba(c, alpha=0.18),
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x + 4, legend_y, label, ha="left", va="center",
                fontsize=9, color=COL_BORDER)

    # Add ❄ / ★ legend items at the right
    legend_right_x = start_x + len(legend_items) * item_w + 2
    ax.text(legend_right_x, legend_y, "❄", ha="left", va="center",
            fontsize=11, color="#3D8BCD")
    ax.text(legend_right_x + 2.2, legend_y, "Frozen", ha="left", va="center",
            fontsize=9, color=COL_BORDER)
    ax.text(legend_right_x + 11, legend_y, "★", ha="left", va="center",
            fontsize=11, color="#E04E2B")
    ax.text(legend_right_x + 13, legend_y, "Trainable", ha="left", va="center",
            fontsize=9, color=COL_BORDER)

    # ───── Save ─────
    fig.savefig(OUT_PNG, dpi=200, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)
    print(f"[OK] saved → {OUT_PNG}")
    print(f"[OK] saved → {OUT_PDF}")


if __name__ == "__main__":
    build_figure()
