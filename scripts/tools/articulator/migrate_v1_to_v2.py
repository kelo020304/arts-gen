"""One-shot migration: ``labels.json`` v1 (earbud-case-specific) -> v2 (generic schema).

v1 layout:  flat dict with hard-coded keys (labels, hinge, lid_offset, part_scales,
external_earbuds_v2). Cluster IDs map to label strings ("body" / "lid" / "earbud_L" /
"earbud_R" / "ignore" / "unlabeled"). One revolute hinge between body and lid.

v2 layout:  parts[] + joints[] + external_meshes[] (see scripts/tools/articulator/schema.py).

Run::

    python scripts/tools/articulator/migrate_v1_to_v2.py outputs/xiaomi_buds6_seed3d/labels.json

Original is backed up to ``<path>.v1.bak`` before overwriting.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from schema import SCHEMA_VERSION, validate


# Drive parameters chosen empirically for the earbud case (see commit
# history of build_usd.py — strong hold + hard angular limit). Migration
# preserves these so v2 output is functionally identical to v1.
DEFAULT_REVOLUTE_DRIVE = {
    "target": None,        # set per-joint = bake_angle
    "stiffness": 5.0e4,
    "damping": 1.0e3,
    "max_force": 1.0e5,
}

# Collision approx + resolution per role (matches build_usd.py v1 hard-coded values)
DEFAULT_COLLISION = {
    "body": {"approx": "sdf", "resolution": 256},
    "lid": {"approx": "sdf", "resolution": 192},
    "earbud_L": {"approx": "sdf", "resolution": 128},
    "earbud_R": {"approx": "sdf", "resolution": 128},
}

DEFAULT_PHYSICS = {
    "body": "kinematic",
    "lid": "dynamic",
    "earbud_L": "dynamic",
    "earbud_R": "dynamic",
}

# Per-part mass (kg). Matches the hard-coded constants in v1 build_usd.py.
DEFAULT_MASS = {
    "body": 0.025,
    "lid": 0.010,
    "earbud_L": 0.0044,
    "earbud_R": 0.0044,
}

# Filtered pairs (collision exclusions) — replicates build_usd.py v1 logic
DEFAULT_FILTERED_PAIRS = {
    "lid": ["body", "Earbud_L", "Earbud_R"],   # lid never self-collides with case-internals
    "Earbud_L": ["lid"],                        # earbuds don't collide with lid
    "Earbud_R": ["lid"],
}

# Map v1 internal label -> v2 prim id. v1 used lowercase ("earbud_L") in labels
# but the USDA file used Capitalized prim names ("Earbud_L"). Preserve those
# capitalized prim names so existing Isaac Sim references keep working.
V1_LABEL_TO_PRIM_ID = {
    "body": "body",
    "lid": "lid",
    "earbud_L": "Earbud_L",
    "earbud_R": "Earbud_R",
}

EARBUD_CASE_DEVICE_NAME = "Buds6Case"
EARBUD_CASE_JOINT_ID = "lid_joint"


def _migrate(v1: dict, device_name: str = "device") -> dict:
    """Build v2 dict from v1 dict. Pure function (no I/O).
    ``device_name`` is propagated to the v2 ``device`` field; callers
    typically pass the parent directory name of labels.json."""
    labels: dict[str, str] = v1["labels"]

    # Group cluster ids by label
    clusters_by_label: dict[str, list[str]] = {}
    for cid, lab in labels.items():
        if lab in ("ignore", "unlabeled"):
            continue
        clusters_by_label.setdefault(lab, []).append(cid)

    # Build parts[]. Key v2 part ids by the v1 *prim* name (capitalized for
    # earbuds) so generated USDA prim paths match v1 (e.g. /World/Earbud_L).
    parts: list[dict] = []
    part_scales: dict[str, list[float]] = v1.get("part_scales", {})
    for role in ("body", "lid", "earbud_L", "earbud_R"):
        if role not in clusters_by_label:
            continue
        prim_id = V1_LABEL_TO_PRIM_ID[role]
        parts.append({
            "id": prim_id,
            "clusters": sorted(clusters_by_label[role]),
            "physics": DEFAULT_PHYSICS[role],
            "collision": dict(DEFAULT_COLLISION[role]),
            "scale_xyz": list(part_scales.get(role, [1, 1, 1])),
            "mass": DEFAULT_MASS[role],
        })

    # Build joints[] — single revolute body↔lid (if both exist).
    # Joint id matches v1 USDA name ("lid_joint") to keep prim paths stable.
    joints: list[dict] = []
    hinge = v1.get("hinge")
    has_body = any(p["id"] == "body" for p in parts)
    has_lid = any(p["id"] == "lid" for p in parts)
    if hinge is not None and has_body and has_lid:
        bake = float(hinge.get("lid_open_deg", 0))
        drive = dict(DEFAULT_REVOLUTE_DRIVE)
        drive["target"] = bake
        joints.append({
            "id": EARBUD_CASE_JOINT_ID,
            "parent": "body",
            "child": "lid",
            "type": "revolute",
            "axis_p0": list(hinge["p0"]),
            "axis_p1": list(hinge["p1"]),
            "lower": bake,
            "upper": 120.0,
            "bake_angle": bake,
            "offset": list(v1.get("lid_offset", [0.0, 0.0, 0.0])),
            "drive": drive,
            "limit_hard": True,
            "filtered_pairs": list(DEFAULT_FILTERED_PAIRS["lid"]),
        })

    # Build external_meshes[] from v1.external_earbuds_v2.group_transforms.
    # In v1 the cluster GLB is one bundled file with per-cluster labels; the
    # actual mesh visible in IsaacSim is the *bundled* clusters.glb subset to
    # the requested label, transformed by group_transforms[label].
    external_meshes: list[dict] = []
    ee = v1.get("external_earbuds_v2", {})
    glb_path = ee.get("clusters_glb")
    cluster_labels = ee.get("cluster_labels", {})
    group_transforms = ee.get("group_transforms", {})
    if glb_path and cluster_labels and group_transforms:
        for v1_side in ("earbud_L", "earbud_R"):
            tx = group_transforms.get(v1_side)
            if tx is None:
                continue
            external_meshes.append({
                "attach_to": V1_LABEL_TO_PRIM_ID[v1_side],
                "glb": glb_path,
                "cluster_filter": [cid for cid, lab in cluster_labels.items() if lab == v1_side],
                "transform": {
                    "t": list(tx.get("translate", [0, 0, 0])),
                    "q_wxyz": list(tx.get("rotate_quat_wxyz", [1, 0, 0, 0])),
                    "s": list(tx.get("scale", [1, 1, 1])),
                },
            })
        # Also stash filter pairs on those parts (collision exclusions for the
        # earbud parts vs lid).
        for p in parts:
            if p["id"] in DEFAULT_FILTERED_PAIRS:
                p["filtered_pairs"] = list(DEFAULT_FILTERED_PAIRS[p["id"]])

    # Detect "this is the earbud case" by part-id signature; use the legacy
    # device name so generated USDA prim paths match v1.
    part_ids = {p["id"] for p in parts}
    if part_ids == {"body", "lid", "Earbud_L", "Earbud_R"}:
        device_field = EARBUD_CASE_DEVICE_NAME
    else:
        device_field = device_name

    return {
        "version": SCHEMA_VERSION,
        "device": device_field,
        "source_glb": v1.get("glb"),
        "physical_dims_mm": v1.get("physical_dims_mm", {"x": 50, "y": 50, "z": 25}),
        "parts": parts,
        "joints": joints,
        "external_meshes": external_meshes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_json", help="path to v1 labels.json (will be overwritten)")
    ap.add_argument("--dry_run", action="store_true",
                    help="print v2 output to stdout, do NOT write the file")
    args = ap.parse_args()

    src = Path(args.labels_json).resolve()
    v1 = json.loads(src.read_text())

    if v1.get("version") == SCHEMA_VERSION:
        raise SystemExit(f"[migrate] {src} is already v{SCHEMA_VERSION}; nothing to do")

    # Derive a sensible device name from the parent dir (e.g. ``xiaomi_buds6_seed3d``)
    device_name = src.parent.name or "device"
    v2 = _migrate(v1, device_name=device_name)
    validate(v2)  # fail-fast: do not write if migration produces an invalid schema

    if args.dry_run:
        print(json.dumps(v2, indent=2))
        return

    backup = src.with_suffix(src.suffix + ".v1.bak")
    shutil.copy2(src, backup)
    src.write_text(json.dumps(v2, indent=2))
    print(f"[migrate] backed up -> {backup}")
    print(f"[migrate] wrote v{SCHEMA_VERSION} schema -> {src}")
    print(f"[migrate]   parts:  {[p['id'] for p in v2['parts']]}")
    print(f"[migrate]   joints: {[(j['id'], j['parent'] + '->' + j['child'], j['type']) for j in v2['joints']]}")
    print(f"[migrate]   external_meshes: {len(v2['external_meshes'])}")


if __name__ == "__main__":
    main()
