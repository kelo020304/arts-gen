"""Generate PPT figures for arts-reconstruction data inventory + pipeline.

输出 → docs/figures/:
  fig1_dataset_inventory_template.png  — 数据集 inventory 表格模板（数字留空）
  fig2_realappliance_categories.png    — RealAppliance 17 类 bar
  fig3_joint_type_pie.png              — revolute / prismatic 占比 pie
  fig4_joint_count_histogram.png       — joint-count per object 直方图
  fig5_pipeline_coverage.png           — 数据集 × pipeline 步骤覆盖矩阵
  fig6_pipeline_4view.png              — dataset_toolkits（4-view 多视角）流程图
  fig7_pipeline_1view.png              — dataset_toolkits_single_image（单图）流程图

Run:
    /home/mi/anaconda3/envs/arts-gen/bin/python -m scripts.tools.make_dataset_ppt_figures
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})


# ───────────────────────── Data ingest ─────────────────────────
RA_FINAL = Path("data/RealAppliance-4view-0515-baked/raw/finaljson")
RA_MANUALS = Path("data/RealAppliance/source/manuals_pdfs")
MI_FINAL = Path("data/Mi-PhysX/raw/finaljson")
PM_FINAL = Path("data/PhysX-Mobility/raw/finaljson")


def _ra_category_map() -> dict[str, str]:
    m = {}
    for d in sorted(RA_MANUALS.iterdir()):
        match = re.match(r"(\d{3})_([a-z]+)_", d.name)
        if match:
            m[match.group(1)] = match.group(2)
    return m


def _ra_stats():
    ra_cat = _ra_category_map()
    n_joints, n_parts = [], []
    joint_types = Counter()
    cat_counter = Counter()
    for f in sorted(RA_FINAL.glob("*.json")):
        src_id = f.stem.removeprefix("ra_")
        d = json.loads(f.read_text())
        movable = [p for p in d["parts"] if p["label"] != 0]
        n_joints.append(len(movable))
        n_parts.append(len(d["parts"]))
        cat_counter[ra_cat.get(src_id, "unknown")] += 1
        for info in d["group_info"].values():
            if isinstance(info, list) and len(info) == 4:
                joint_types[info[3]] += 1
    return {
        "n_models": len(n_joints),
        "categories": cat_counter,
        "n_joints": n_joints,
        "n_parts": n_parts,
        "joint_types": joint_types,
    }


def _count(p: Path) -> int:
    return len(list(p.glob("*.json"))) if p.is_dir() else 0


# ───────────────────────── Figure 1: Inventory table template ─────────────────────────

def fig1_inventory_template() -> None:
    """干净的表格模板；数字 / "TBD" / "—" 字段留给用户在 PPT 里填。"""
    cols = ["Dataset", "Source", "# Objects", "Categories",
            "Articulated?", "Role in this work", "Data path"]
    rows = [
        ["RealAppliance",
         "arxiv 2512.00287",
         "100", "17",
         "Yes (revolute + prismatic)",
         "V1 KinematicSolver eval",
         "data/RealAppliance-4view-0515-baked/"],
        ["Mi-PhysX",
         "Xiaomi internal",
         "TBD", "TBD",
         "Yes",
         "Factory part target",
         "data/Mi-PhysX/raw/"],
        ["PhysX-Mobility",
         "PartNet-Mobility derived",
         "TBD", "TBD",
         "Yes",
         "TRELLIS-arts training base",
         "(external storage)"],
        ["PhysX-Anything",
         "NVIDIA reference",
         "—", "—",
         "—",
         "Code reference only",
         "submodules/PhysX-Anything/"],
    ]

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=cols, loc="center", cellLoc="left",
        colWidths=[0.10, 0.13, 0.07, 0.08, 0.16, 0.20, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    # Header style
    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor("#2E86AB")
        cell.set_text_props(color="white", weight="bold")

    # Per-row dataset-color tint on first column
    tints = ["#D6EAF8", "#F5CBA7", "#FAD7A0", "#E5E7E9"]
    for i, t in enumerate(tints, start=1):
        for j in range(len(cols)):
            cell = table[i, j]
            cell.set_facecolor(t if j == 0 else "white")
        table[i, 0].set_text_props(weight="bold")

    ax.set_title(
        "Figure 1 — Dataset inventory (template)\n"
        "Fill TBD cells in PPT with current counts",
        loc="center", pad=14,
    )
    plt.tight_layout()
    out = OUT / "fig1_dataset_inventory_template.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ───────────────────── Figure 1 (SAPIEN-style) ─────────────────────

def _stats_from_finaljson(json_dir: Path) -> dict:
    """Compute per-dataset stats from raw/finaljson/*.json files.

    Schema: {'category', 'parts'[list], 'group_info': {gid: [parts, parent, params, type_letter]}}
    Letter encoding (confirmed from smoke_test): C=revolute, B=prismatic, E=fixed.
    """
    if not json_dir.exists():
        return {}
    parts_per_obj = []
    rev = pris = 0
    cats: set[str] = set()
    n_obj = 0
    for f in sorted(json_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        n_obj += 1
        cats.add(str(d.get("category", "")).strip())
        parts_per_obj.append(len(d.get("parts", [])))
        for gid, val in d.get("group_info", {}).items():
            if not isinstance(val, list) or len(val) < 4:
                continue
            letter = val[-1]
            if letter == "C":
                rev += 1
            elif letter == "B":
                pris += 1
    return {
        "n_obj": n_obj,
        "n_cat": len(cats),
        "parts_mean": float(np.mean(parts_per_obj)) if parts_per_obj else 0.0,
        "parts_max": int(np.max(parts_per_obj)) if parts_per_obj else 0,
        "revolute": rev,
        "prismatic": pris,
    }


def _stats_from_partinfo(root: Path) -> dict:
    """Stats from reconstruction/part_info/<id>/part_info.json (PhysX-Mobility format)."""
    if not root.exists():
        return {}
    parts_per_obj, cats, rev, pris, n_obj = [], set(), 0, 0, 0
    for d in sorted(root.iterdir()):
        f = d / "part_info.json"
        if not f.exists():
            continue
        try:
            pi = json.loads(f.read_text())
        except Exception:
            continue
        n_obj += 1
        cats.add(str(pi.get("category", "")).strip())
        parts_per_obj.append(int(pi.get("num_parts", 0)))
        for k, p in pi.get("parts", {}).items():
            j = p.get("joint")
            if j == "revolute":
                rev += 1
            elif j == "prismatic":
                pris += 1
    return {
        "n_obj": n_obj,
        "n_cat": len(cats),
        "parts_mean": float(np.mean(parts_per_obj)) if parts_per_obj else 0.0,
        "parts_max": int(np.max(parts_per_obj)) if parts_per_obj else 0,
        "revolute": rev,
        "prismatic": pris,
    }


def fig1_inventory_sapien_style() -> None:
    """SAPIEN/GAPartNet-style cross-dataset comparison table.

    Columns: Dataset | # Obj | # Cat | Parts (mean / max) | Joints (Rev / Pris) | Textured | Domain | Split
    """
    ra = _stats_from_finaljson(Path("data/RealAppliance-4view-0515-baked/raw/finaljson"))
    # RA "category" field in finaljson is just "appliance" — true 17 sub-categories
    # come from manuals_pdfs/ directory names (see _ra_category_map).
    ra_real_cats = _ra_stats()["categories"]
    if ra and ra_real_cats:
        ra["n_cat"] = len(ra_real_cats)

    # Stats computed on dev machine (full sets on external storage):
    #   PhysX-Mobility:  /mnt/robot-data-lab/arts-gen-data/data/PhysX-Mobility-full-4view-0511/raw/finaljson
    #   Mi-PhysX:        /mnt/robot-data-lab/arts-gen-data/data/PhysX-Mobility-MI-4view-0514/raw/finaljson
    PM_FULL = {
        "n_obj": 2019, "n_cat": 132, "parts_mean": 6.98, "parts_max": 116,
        "revolute": 2629, "prismatic": 7250,
    }
    MI_FULL = {
        "n_obj": 542, "n_cat": 44, "parts_mean": 4.79, "parts_max": 47,
        "revolute": 1069, "prismatic": 814,
    }

    def fmt_parts(s):
        return f"{s['parts_mean']:.1f} / {s['parts_max']}" if s else "—"

    def fmt_joints(s):
        return f"{s['revolute']:,} / {s['prismatic']:,}" if s else "—"

    cols = ["Dataset", "# Obj", "# Cat", "Parts (mean / max)",
            "Joints (Rev / Pris)", "Textured", "Domain", "Split"]
    rows = [
        ["PhysX-Mobility",
         f"{PM_FULL['n_obj']:,}",
         str(PM_FULL["n_cat"]),
         fmt_parts(PM_FULL),
         fmt_joints(PM_FULL),
         "Yes", "Synthetic (CAD)", "Train"],
        ["Mi-PhysX",
         f"{MI_FULL['n_obj']:,}",
         str(MI_FULL["n_cat"]),
         fmt_parts(MI_FULL),
         fmt_joints(MI_FULL),
         "Yes", "Synthetic (CAD, Xiaomi)", "Train"],
        ["RealAppliance",
         str(ra.get("n_obj", "—")),
         str(ra.get("n_cat", "—")),
         fmt_parts(ra),
         fmt_joints(ra),
         "Yes", "Real photo + annotation", "Real-world test"],
    ]

    # Style: clean academic table (thin gray rules, no heavy fills)
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=cols, loc="center", cellLoc="center",
        colWidths=[0.14, 0.07, 0.06, 0.13, 0.12, 0.08, 0.18, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Header
    for j in range(len(cols)):
        c = table[0, j]
        c.set_facecolor("#2C3E50")
        c.set_text_props(color="white", weight="bold")
        c.set_height(0.16)

    # Per-row dataset name color tint (first col only) + bold name
    tints = ["#D6EAF8", "#FAD7A0", "#D5F5E3"]
    for i, t in enumerate(tints, start=1):
        for j in range(len(cols)):
            cell = table[i, j]
            cell.set_facecolor(t if j == 0 else "white")
            cell.set_edgecolor("#BDBDBD")
            cell.set_linewidth(0.6)
            if j == 0:
                cell.set_text_props(weight="bold")
            else:
                cell.set_text_props(family="monospace") if j in (1, 2, 3, 4) else None

    ax.set_title(
        "Table 1 — Dataset inventory used in this work.",
        loc="center", pad=10, fontsize=12, weight="bold",
    )

    # Caption / footnote
    fig.text(
        0.5, -0.02,
        "PhysX-Mobility is derived from PartNet-Mobility.  "
        "Mi-PhysX is an in-house articulated-CAD set curated by Xiaomi.  "
        "All datasets provide textured meshes with revolute and prismatic joints.",
        ha="center", va="top", fontsize=8.5, style="italic", color="#555",
    )

    plt.tight_layout()
    out = OUT / "fig1_dataset_inventory_sapien.png"
    plt.savefig(out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[OK] {out}")


# ───────────────────────── Figure 2 ─────────────────────────

def fig2_categories(cat_counter: Counter) -> None:
    items = cat_counter.most_common()
    names = [k for k, _ in items]
    counts = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), counts, color="#2E86AB",
                   edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("# models")
    ax.set_title(f"Figure 2 — RealAppliance category distribution "
                 f"(n = {sum(counts)} models · {len(names)} categories)")
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.15, str(v), ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    out = OUT / "fig2_realappliance_categories.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ───────────────────────── Figure 3 ─────────────────────────

def fig3_joint_type_pie(joint_types: Counter, ra_n: int) -> None:
    rev = joint_types.get("C", 0)
    pri = joint_types.get("B", 0)
    total = rev + pri
    fig, ax = plt.subplots(figsize=(7, 6))
    _wedges, _texts, autotexts = ax.pie(
        [rev, pri], labels=[f"Revolute (C)\n{rev}", f"Prismatic (B)\n{pri}"],
        colors=["#A23B72", "#F18F01"],
        autopct=lambda p: f"{p:.1f}%",
        startangle=90, counterclock=False,
        wedgeprops=dict(edgecolor="black", linewidth=0.8),
        textprops=dict(fontsize=12),
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontweight("bold")
    ax.set_title(
        f"Figure 3 — Joint type distribution across RealAppliance\n"
        f"(n = {total} joints across {ra_n} models)"
    )
    plt.tight_layout()
    out = OUT / "fig3_joint_type_pie.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ───────────────────────── Figure 4 ─────────────────────────

def fig4_joint_count_hist(n_joints: list[int]) -> None:
    hist = Counter(n_joints)
    xs = sorted(hist.keys())
    ys = [hist[x] for x in xs]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(xs, ys, color="#2E86AB",
                   edgecolor="black", linewidth=0.5, width=0.7)
    ax.set_xlabel("# articulated joints per model")
    ax.set_ylabel("# models")
    ax.set_xticks(xs)
    ax.set_title(
        f"Figure 4 — Joint-count distribution per RealAppliance model\n"
        f"(n = {len(n_joints)} models · mean = {np.mean(n_joints):.1f} · max = {max(n_joints)})"
    )
    for bar, v in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.3, str(v), ha="center", va="bottom", fontsize=10)

    v1_ids = {"007", "017", "027", "037", "047", "057", "067", "077", "087", "097"}
    v1_joint_counts = []
    for f in sorted(RA_FINAL.glob("*.json")):
        sid = f.stem.removeprefix("ra_")
        if sid in v1_ids:
            d = json.loads(f.read_text())
            v1_joint_counts.append(sum(1 for p in d["parts"] if p["label"] != 0))
    v1_hist = Counter(v1_joint_counts)
    if v1_hist:
        ax.bar(list(v1_hist.keys()), list(v1_hist.values()),
               color="#C73E1D", edgecolor="black", linewidth=0.5,
               width=0.35, label="V1 eval subset (10 IDs ending in 7)")
        ax.legend(loc="upper right")

    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, max(ys) * 1.18)
    plt.tight_layout()
    out = OUT / "fig4_joint_count_histogram.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ───────────────────────── Figure 5: Pipeline coverage matrix (rewritten) ─────────────────────────

def fig5_pipeline_coverage() -> None:
    """重设计：一句话说清楚"哪个 dataset 跑过哪些 pipeline 步骤"。

    行 = 4 个 dataset；列 = 11 个 pipeline 步骤（dataset_toolkits 4-view 主线）；
    单元 ● = full coverage（产物在）；◐ = partial（仅 smoke / 子集）；空 = 未跑。
    """
    rows = ["RealAppliance\n(100 models)", "Mi-PhysX\n(2 models)",
            "PhysX-Mobility\n(training base)", "PhysX-Anything\n(reference)"]
    cols = [
        "1\njoint\nxform",
        "2\nrender\n12-view",
        "3\nbbox\nGT",
        "4\nvoxelize",
        "5\nDINOv2\nfeat",
        "6\nmanifest",
        "7\nVLM\njsonl",
        "8\nSS\nlatent",
        "9\npart-compl\nmanifest",
        "10\ndecode\nSS",
        "11\nweb\npreview",
    ]
    # 2 = full, 1 = partial, 0 = not run
    M = np.array([
        # 1  2  3  4  5  6  7  8  9 10 11
        [2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 1],   # RealAppliance
        [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1],   # Mi-PhysX
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],   # PhysX-Mobility
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # PhysX-Anything
    ])

    fig, ax = plt.subplots(figsize=(13, 4.0))
    cmap = matplotlib.colors.ListedColormap(["#F4F4F4", "#FCE5B7", "#5DADE2"])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    glyph = {0: "", 1: "◐", 2: "●"}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, glyph[M[i, j]], ha="center", va="center",
                    fontsize=20,
                    color=("white" if M[i, j] == 2 else "#7D6608"),
                    fontweight="bold")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_title(
        "Figure 5 — Which dataset has been processed by which pipeline step?\n"
        "● done (all artifacts present) · ◐ partial (smoke / subset) · blank = not run",
        pad=14,
    )
    # Gridlines for readability
    for k in range(M.shape[0] + 1):
        ax.axhline(k - 0.5, color="white", linewidth=2)
    for k in range(M.shape[1] + 1):
        ax.axvline(k - 0.5, color="white", linewidth=2)

    legend_patches = [
        mpatches.Patch(color="#5DADE2", label="● full coverage"),
        mpatches.Patch(color="#FCE5B7", label="◐ partial / smoke"),
        mpatches.Patch(color="#F4F4F4", label="(blank) not run"),
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              bbox_to_anchor=(1.18, 1.0), fontsize=9, frameon=False)
    plt.tight_layout()
    out = OUT / "fig5_pipeline_coverage.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ───────────────────────── Figure 6 & 7: pipeline flow diagrams ─────────────────────────

def _draw_pipeline(ax, steps, title, color_input, color_step, color_output):
    """通用流水线图：左侧 input → 中间一排 step boxes（带阶段分组色）→ 右侧 output。"""
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title(title, pad=10)

    # Input box (left)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, 3.0), 2.2, 2.0,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor=color_input, edgecolor="black", linewidth=1.2,
    ))
    ax.text(1.3, 4.0,
            "raw\nURDF / USD\n+ part meshes",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # Output box (right)
    ax.add_patch(mpatches.FancyBboxPatch(
        (11.4, 3.0), 2.4, 2.0,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor=color_output, edgecolor="black", linewidth=1.2,
    ))
    ax.text(12.6, 4.0,
            "VLM JSONL\n+ Part-completion\nmanifest\n+ Web preview",
            ha="center", va="center", fontsize=9, fontweight="bold")

    # Steps: 2 rows × 6 cols arrangement
    cols_per_row = 6
    n = len(steps)
    rows = [steps[i:i + cols_per_row] for i in range(0, n, cols_per_row)]
    # x positions in the middle band [2.8 .. 11.0]
    band_lo, band_hi = 2.8, 11.0
    for ri, row in enumerate(rows):
        y = 5.5 - ri * 2.6   # row 0 high, row 1 low
        xs = np.linspace(band_lo, band_hi - 1.2, max(len(row), 1))
        for (sid, label, color), x in zip(row, xs):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y - 0.55), 1.3, 1.1,
                boxstyle="round,pad=0.04,rounding_size=0.10",
                facecolor=color, edgecolor="black", linewidth=0.8,
            ))
            ax.text(x + 0.65, y + 0.18,
                    f"{sid}", ha="center", va="center",
                    fontsize=11, fontweight="bold")
            ax.text(x + 0.65, y - 0.20,
                    label, ha="center", va="center", fontsize=8)
        # Row connector arrows
        for i in range(len(row) - 1):
            x_from = xs[i] + 1.3
            x_to = xs[i + 1]
            ax.annotate("", xy=(x_to, y), xytext=(x_from, y),
                         arrowprops=dict(arrowstyle="->", lw=1.0, color="#555"))
    # Input → first step + last step → output
    first_x = np.linspace(band_lo, band_hi - 1.2, cols_per_row)[0]
    last_row = rows[-1]
    last_x = np.linspace(band_lo, band_hi - 1.2, cols_per_row)[len(last_row) - 1] + 1.3
    ax.annotate("", xy=(first_x, 5.5), xytext=(2.4, 4.0),
                 arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    last_y = 5.5 - (len(rows) - 1) * 2.6
    ax.annotate("", xy=(11.4, 4.0), xytext=(last_x, last_y),
                 arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))

    # v2: 取消跨行 arc（容易让图变乱）；改成 step 编号 + 在右端 / 左端各加
    # 一个 "↓ continue" / "→" 小提示，靠数字阅读顺序保证流向。
    if len(rows) > 1:
        # row 0 末尾向下提示
        x_end_r0 = np.linspace(band_lo, band_hi - 1.2, cols_per_row)[len(rows[0]) - 1] + 1.3
        y_r0 = 5.5
        ax.annotate("↓", xy=(x_end_r0 + 0.15, y_r0 - 0.7),
                     xytext=(x_end_r0 + 0.15, y_r0 + 0.0),
                     ha="center", fontsize=14, color="#555")
        # row 1 起点向右提示
        x_start_r1 = np.linspace(band_lo, band_hi - 1.2, cols_per_row)[0]
        y_r1 = 5.5 - 2.6
        ax.text(x_start_r1 - 0.4, y_r1, "↳", fontsize=14, color="#555",
                ha="center", va="center")


def fig6_pipeline_4view() -> None:
    """dataset_toolkits（4-view 多视角）默认 11 步 profile。"""
    GROUP_DATA = "#D6EAF8"      # 几何 / mesh
    GROUP_RENDER = "#FAD7A0"    # 视觉渲染
    GROUP_FEAT = "#E8DAEF"      # 特征 / latent
    GROUP_MANIFEST = "#D5F5E3"  # manifest / 输出

    steps = [
        ("01", "joint\ntransform",   GROUP_DATA),
        ("02", "4 quadrants ×\n3 views = 12v\nRGB + mask",  GROUP_RENDER),
        ("03", "bbox GT\nfrom mask",  GROUP_DATA),
        ("04", "per-part 64³\nvoxel + label",   GROUP_DATA),
        ("05", "DINOv2-L/14\nfeature",          GROUP_FEAT),
        ("06", "PhysX\nmanifest",     GROUP_MANIFEST),
        ("07", "VLM JSONL\n(4-view groups)",   GROUP_MANIFEST),
        ("08", "Per-part\nSS latent",  GROUP_FEAT),
        ("09", "Part-compl\nmanifest", GROUP_MANIFEST),
        ("10", "Decode SS\n→ voxel QC", GROUP_FEAT),
        ("11", "HTML\npreview",       GROUP_MANIFEST),
    ]
    fig, ax = plt.subplots(figsize=(15, 7))
    _draw_pipeline(
        ax, steps,
        title="Figure 6 — dataset_toolkits (4-view multi-view pipeline)\n"
              "11 default steps · input URDF/USD → output VLM JSONL + Part-completion manifest + Web preview",
        color_input="#D6EAF8", color_step="#F5CBA7", color_output="#D5F5E3",
    )
    # Group legend
    legend_patches = [
        mpatches.Patch(color=GROUP_DATA,    label="Geometry / data prep"),
        mpatches.Patch(color=GROUP_RENDER,  label="Render"),
        mpatches.Patch(color=GROUP_FEAT,    label="Feature / latent"),
        mpatches.Patch(color=GROUP_MANIFEST,label="Manifest / preview"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=10, frameon=False)
    plt.tight_layout()
    out = OUT / "fig6_pipeline_4view.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def fig7_pipeline_1view() -> None:
    """dataset_toolkits_single_image（1-view 单图）默认 11 步 profile。"""
    GROUP_DATA = "#D6EAF8"
    GROUP_RENDER = "#FAD7A0"
    GROUP_FEAT = "#E8DAEF"
    GROUP_MANIFEST = "#D5F5E3"

    steps = [
        ("01", "joint\ntransform",       GROUP_DATA),
        ("02", "canonical\ntransform",   GROUP_DATA),
        ("03", "voxelize\n(overall + per-part)", GROUP_DATA),
        ("04", "valid-parts\nmanifest",   GROUP_MANIFEST),
        ("05", "render\npart_complete\n16-view",  GROUP_RENDER),
        ("06", "DINOv2 feat\n(16-view)", GROUP_FEAT),
        ("07", "Per-part\nSS latent",    GROUP_FEAT),
        ("08", "Decode SS\n→ voxel QC",  GROUP_FEAT),
        ("09", "1-image VLM\nJSONL",      GROUP_MANIFEST),
        ("10", "1-image Part-\ncompl manifest", GROUP_MANIFEST),
        ("11", "Web preview\n(VLM + PC)", GROUP_MANIFEST),
    ]
    fig, ax = plt.subplots(figsize=(15, 7))
    _draw_pipeline(
        ax, steps,
        title="Figure 7 — dataset_toolkits_single_image (1-view single-image pipeline)\n"
              "11 default steps · input URDF/USD → output 1-image VLM JSONL + Part-completion manifest + Web preview",
        color_input="#D6EAF8", color_step="#F5CBA7", color_output="#D5F5E3",
    )
    legend_patches = [
        mpatches.Patch(color=GROUP_DATA,    label="Geometry / data prep"),
        mpatches.Patch(color=GROUP_RENDER,  label="Render"),
        mpatches.Patch(color=GROUP_FEAT,    label="Feature / latent"),
        mpatches.Patch(color=GROUP_MANIFEST,label="Manifest / preview"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=10, frameon=False)
    plt.tight_layout()
    out = OUT / "fig7_pipeline_1view.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def main() -> None:
    ra = _ra_stats()
    fig1_inventory_template()
    fig1_inventory_sapien_style()
    fig2_categories(ra["categories"])
    fig3_joint_type_pie(ra["joint_types"], ra["n_models"])
    fig4_joint_count_hist(ra["n_joints"])
    fig5_pipeline_coverage()
    fig6_pipeline_4view()
    fig7_pipeline_1view()
    print(f"\nAll 7 figures saved to {OUT}/")


if __name__ == "__main__":
    main()
