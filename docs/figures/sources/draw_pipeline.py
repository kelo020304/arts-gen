#!/usr/bin/env python3
"""
Reconstruction pipeline figure — 16:9 horizontal, paper-ready.
Run:  python docs/draw_pipeline.py
Output: docs/images/pipeline.{pdf,png}
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ---------- Output ----------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "images")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Palette ----------
C_PANEL_BG   = "#F9FAFB"   # panel background
C_PANEL_HEAD = "#DBEAFE"   # panel title bar
C_PANEL_EDGE = "#93C5FD"   # panel border

C_FROZEN = "#E5E7EB"       # frozen VAE
C_MV_FT  = "#BFDBFE"       # fine-tuned for MV
C_NEW    = "#FDBA74"       # new module (Part Predictor)
C_DECODE = "#D1D5DB"       # frozen decoder / evaluator
C_DATA   = "#FFFFFF"       # data tensor box
C_OP     = "#F3F4F6"       # operation box

EDGE = "#1F2937"
COND = "#9CA3AF"

# ---------- Figure (16:9) ----------
FIG_W, FIG_H = 16.0, 9.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")


# ---------- Helpers ----------
def panel(x, y, w, h, title):
    """Panel with a header title bar (like reference figure)."""
    head_h = 0.38
    # body
    body = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.00,rounding_size=0.08",
        facecolor=C_PANEL_BG, edgecolor=C_PANEL_EDGE, linewidth=1.2,
    )
    ax.add_patch(body)
    # header bar (rectangle at top)
    header = Rectangle(
        (x, y + h - head_h), w, head_h,
        facecolor=C_PANEL_HEAD, edgecolor=C_PANEL_EDGE, linewidth=1.2,
    )
    ax.add_patch(header)
    ax.text(
        x + w / 2, y + h - head_h / 2, title,
        ha="center", va="center", fontsize=10.5,
        fontfamily="serif", fontweight="bold", color="#1E3A8A",
    )


def box(cx, cy, w, h, text, color, fs=9, bold=False, italic=False, lw=1.2):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=color, edgecolor=EDGE, linewidth=lw,
    )
    ax.add_patch(p)
    ax.text(
        cx, cy, text,
        ha="center", va="center", fontsize=fs, fontfamily="serif",
        fontweight="bold" if bold else "normal",
        fontstyle="italic" if italic else "normal",
    )


def arrow(x1, y1, x2, y2, lw=1.4, color=EDGE):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", mutation_scale=15,
        linewidth=lw, color=color,
    ))


def dashed(x1, y1, x2, y2, lw=1.1, color=COND):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", mutation_scale=12,
        linewidth=lw, color=color, linestyle=(0, (4, 2.5)),
    ))


# ============================================================
# Title
# ============================================================
ax.text(
    FIG_W / 2, 8.65,
    "Multi-view Articulated Reconstruction with Decoupled Part Prediction",
    ha="center", fontsize=13.5, fontfamily="serif", fontweight="bold",
)

# ============================================================
# Panel layout
#   P1 (input & preprocess)  |  P2 (parallel reasoning)  |  P3 (decode & assemble)
# ============================================================
Y_PAN = 0.45
H_PAN = 7.85

P1_X, P1_W = 0.30, 4.20
P2_X, P2_W = 4.75, 6.50
P3_X, P3_W = 11.50, 4.20

panel(P1_X, Y_PAN, P1_W, H_PAN, "1. Multi-view Preprocessing")
panel(P2_X, Y_PAN, P2_W, H_PAN, "2. Parallel Part & Latent Reasoning")
panel(P3_X, Y_PAN, P3_W, H_PAN, "3. Per-part Decoding  &  Assembly")

# usable vertical range inside panels (below title bar)
PAN_TOP = Y_PAN + H_PAN - 0.55
PAN_BOT = Y_PAN + 0.35

# ============================================================
# Panel 1: Multi-view Preprocessing
# ============================================================
p1_cx = P1_X + P1_W / 2

# input images
box(p1_cx, 7.30, 3.3, 0.55,
    r"Multi-view Images  $x_{mv}$  ($V$ views)",
    C_DATA, fs=9.5, bold=True)

arrow(p1_cx, 7.00, p1_cx, 6.68)

# DINOv2
box(p1_cx, 6.40, 2.8, 0.52, "DINOv2 Encoder", C_FROZEN, fs=9)

arrow(p1_cx, 6.12, p1_cx, 5.80)

# mv tokens (data)
box(p1_cx, 5.52, 3.3, 0.50,
    r"mv tokens  $\in \mathbb{R}^{V \times T \times D}$",
    C_DATA, fs=9, italic=True)

arrow(p1_cx, 5.25, p1_cx, 4.92)

# SS Flow
box(p1_cx, 4.60, 3.3, 0.62,
    "Stage 2:  SS Flow\n(fine-tuned for MV)",
    C_MV_FT, fs=9)

arrow(p1_cx, 4.27, p1_cx, 3.97)

# SS latent
box(p1_cx, 3.68, 3.3, 0.48,
    r"SS latent  $\in \mathbb{R}^{8 \times 16^3}$",
    C_DATA, fs=9, italic=True)

arrow(p1_cx, 3.42, p1_cx, 3.10)

# SS-VAE decoder
box(p1_cx, 2.80, 3.3, 0.56,
    "Stage 1:  SS-VAE Decoder\n(frozen)",
    C_FROZEN, fs=9)

arrow(p1_cx, 2.50, p1_cx, 2.18)

# active voxel coords (output of panel 1)
box(p1_cx, 1.88, 3.5, 0.54,
    r"active voxels  $\in \mathbb{R}^{N \times 3}$",
    C_DATA, fs=9.5, bold=True, italic=True)

# ============================================================
# Panel 2: Parallel Part & Latent Reasoning
# ============================================================
p2_cx = P2_X + P2_W / 2

# incoming arrow from P1 → P2 (carrying active voxels)
arrow(P1_X + P1_W - 0.1, 1.88, P2_X + 0.45, 1.88, lw=1.6)

# entry hub in P2 (active voxels replicated)
HUB_X = P2_X + 0.85
HUB_Y = 1.88
box(HUB_X, HUB_Y, 1.35, 0.50, "split", C_OP, fs=9, italic=True)

# branch upward to Part Predictor, downward to SLat Flow
UPPER_Y = 5.70
LOWER_Y = 3.30

# vertical connectors from hub
arrow(HUB_X, HUB_Y + 0.25, HUB_X, UPPER_Y - 0.75)
arrow(HUB_X, HUB_Y + 0.25, HUB_X, LOWER_Y - 0.35)

# then horizontal to each module
arrow(HUB_X, UPPER_Y - 0.75, HUB_X + 1.05, UPPER_Y - 0.75)
arrow(HUB_X, LOWER_Y - 0.35, HUB_X + 1.05, LOWER_Y - 0.35)

# ---- Upper branch: Part Predictor (NEW) ----
PP_CX = HUB_X + 2.55
PP_CY = UPPER_Y
box(PP_CX, PP_CY, 2.95, 1.50,
    "Part Predictor  (NEW)\n"
    r"Query Transformer" + "\n"
    r"$K$ queries, VLM class init" + "\n"
    "(Hungarian matching)",
    C_NEW, fs=9, bold=True)

# arrow from left edge of Part Predictor in
arrow(HUB_X + 1.05, UPPER_Y - 0.75, PP_CX - 1.47, UPPER_Y - 0.35)

# output of Part Predictor
arrow(PP_CX + 1.48, PP_CY, PP_CX + 2.05, PP_CY)
box(PP_CX + 2.70, PP_CY, 1.55, 0.50,
    r"labels  $\ell \in \{1..K\}^N$",
    C_DATA, fs=8.5, italic=True)

# ---- Lower branch: SLat Flow ----
SL_CX = HUB_X + 2.55
SL_CY = LOWER_Y
box(SL_CX, SL_CY, 2.95, 1.35,
    "Stage 4:  SLat Flow\n(fine-tuned for MV)\n"
    r"all $N$ voxels, part-unaware",
    C_MV_FT, fs=9)

arrow(HUB_X + 1.05, LOWER_Y - 0.35, SL_CX - 1.47, LOWER_Y - 0.10)

# output of SLat Flow
arrow(SL_CX + 1.48, SL_CY, SL_CX + 2.05, SL_CY)
box(SL_CX + 2.70, SL_CY, 1.55, 0.50,
    r"$z_{slat} \in \mathbb{R}^{N \times C}$",
    C_DATA, fs=8.5, italic=True)

# ---- Cross-attention from mv tokens (dashed) ----
# Source: mv tokens box in P1 (right edge ≈ (p1_cx + 1.65, 5.52))
MV_SRC_X = p1_cx + 1.65
MV_SRC_Y = 5.52

# dashed into Part Predictor (top-left)
dashed(MV_SRC_X, MV_SRC_Y, PP_CX - 1.47, PP_CY + 0.55)
# dashed into SLat Flow (bottom-left)
dashed(MV_SRC_X, MV_SRC_Y - 0.30, SL_CX - 1.47, SL_CY + 0.50)
# dashed into SS Flow (small, inside P1) — show conditioning
dashed(MV_SRC_X - 0.30, MV_SRC_Y - 0.30, p1_cx + 1.65, 4.60)

ax.text(PP_CX - 1.55, PP_CY + 0.83, "cross-attn", fontsize=7.2,
        color="#4B5563", style="italic", fontfamily="serif")

# ============================================================
# Panel 3: Per-part Decoding & Assembly
# ============================================================
p3_cx = P3_X + P3_W / 2

# Merge arrow from P2 outputs into P3
# the two data boxes at (PP_CX+2.70, PP_CY) and (SL_CX+2.70, SL_CY)
DA_X = PP_CX + 3.50  # data box right edge
arrow(PP_CX + 3.48, PP_CY, P3_X + 0.40, PAN_TOP - 1.00, lw=1.4)
arrow(SL_CX + 3.48, SL_CY, P3_X + 0.40, PAN_TOP - 1.00, lw=1.4)

# Split op
SP_Y = PAN_TOP - 1.00
box(p3_cx, SP_Y, 3.60, 0.60,
    r"Split $z_{slat}$ by $\ell$  $\rightarrow$  $K$ subsets",
    C_OP, fs=9, bold=True)

arrow(p3_cx, SP_Y - 0.32, p3_cx, SP_Y - 0.80)

# Stage 3 Decoder (frozen)
DEC_Y = SP_Y - 1.20
box(p3_cx, DEC_Y, 3.80, 0.80,
    "Stage 3:  SLat VAE Decoder\n(frozen, GS head)\n"
    r"per-part decode",
    C_DECODE, fs=9)

arrow(p3_cx, DEC_Y - 0.42, p3_cx, DEC_Y - 0.80)

# per-part meshes row
MESH_Y = DEC_Y - 1.10
for i, dx in enumerate([-1.40, -0.45, 0.50, 1.40]):
    label = "mesh$_K$" if i == 3 else f"mesh$_{i+1}$"
    box(p3_cx + dx, MESH_Y, 0.80, 0.48, label, C_DATA, fs=8.5, italic=True)
ax.text(p3_cx + 1.00, MESH_Y, "$\\cdots$", ha="center", va="center",
        fontsize=11, fontfamily="serif")

arrow(p3_cx, MESH_Y - 0.26, p3_cx, MESH_Y - 0.65)

# Assembly → URDF
ASM_Y = MESH_Y - 0.95
box(p3_cx, ASM_Y, 3.60, 0.54,
    r"Assembly  $\rightarrow$  URDF / MJCF",
    C_OP, fs=9.5, bold=True)

arrow(p3_cx, ASM_Y - 0.30, p3_cx, ASM_Y - 0.70)

# Simulation
SIM_Y = ASM_Y - 1.00
box(p3_cx, SIM_Y, 3.60, 0.54,
    "Simulation  (Isaac Sim / MuJoCo)",
    C_FROZEN, fs=9.5, bold=True)


# ============================================================
# Legend (bottom, horizontal)
# ============================================================
leg_y = 0.15
leg_items = [
    (C_FROZEN, "Frozen (VAE / non-trainable)"),
    (C_MV_FT,  "Fine-tuned for Multi-view"),
    (C_NEW,    "New module (ours)"),
    (C_DECODE, "Frozen decoder / evaluator"),
    (C_OP,     "Operation"),
    (C_DATA,   "Data tensor"),
]

# compute starts so it's centered
lx = 0.55
sw, sh = 0.28, 0.20
gap = 2.30

for i, (c, lbl) in enumerate(leg_items):
    x0 = lx + i * gap
    p = FancyBboxPatch(
        (x0, leg_y - 0.05), sw, sh,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=c, edgecolor=EDGE, linewidth=1.0,
    )
    ax.add_patch(p)
    ax.text(x0 + sw + 0.10, leg_y + sh / 2 - 0.05, lbl,
            va="center", fontsize=7.8, fontfamily="serif")

# dashed legend entry (far right)
dx0 = lx + len(leg_items) * gap
ax.plot([dx0, dx0 + sw + 0.05], [leg_y + sh / 2 - 0.05] * 2,
        linestyle=(0, (4, 2.5)), color=COND, linewidth=1.1)
ax.add_patch(FancyArrowPatch(
    (dx0 + sw - 0.03, leg_y + sh / 2 - 0.05),
    (dx0 + sw + 0.06, leg_y + sh / 2 - 0.05),
    arrowstyle="->", mutation_scale=10, color=COND, linewidth=1.0,
))
ax.text(dx0 + sw + 0.16, leg_y + sh / 2 - 0.05,
        "cross-attn condition (mv tokens)",
        va="center", fontsize=7.8, fontfamily="serif",
        color="#4B5563", style="italic")


# ============================================================
# Save
# ============================================================
plt.tight_layout()
out_pdf = os.path.join(OUT_DIR, "pipeline.pdf")
out_png = os.path.join(OUT_DIR, "pipeline.png")
plt.savefig(out_pdf, bbox_inches="tight", dpi=200)
plt.savefig(out_png, bbox_inches="tight", dpi=300)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
