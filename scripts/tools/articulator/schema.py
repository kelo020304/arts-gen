"""Schema validator for v2 articulator ``labels.json``.

The v2 schema describes an articulated device as ``parts[]`` (rigid bodies)
plus ``joints[]`` (parent/child connections), which makes the labelling
tool generic across earbud cases, folding phones, drawers, etc.

This module is intentionally dependency-free (stdlib only) so it can be
imported by both the migration script and ``build_usd.py`` without
dragging in numpy / trimesh.

Validation philosophy: NO silent fallbacks. Any schema violation raises
``SystemExit`` with a precise message — surfacing bugs early rather than
letting bad data flow into the USDA emitter.
"""
from __future__ import annotations

from typing import Any, Iterable

SCHEMA_VERSION = 3
# v2 files remain readable: ``sites`` is optional, so a v2 doc parses cleanly
# under the v3 validator. The migrator (``migrate_v2_to_v3.py``) just bumps
# the version field; it never has to invent data.
SUPPORTED_VERSIONS = {2, 3}

JOINT_TYPES = {"revolute", "prismatic", "fixed", "free"}
PHYSICS_KINDS = {"kinematic", "dynamic"}
COLLISION_APPROXES = {"sdf", "convexHull", "convexDecomposition", "none"}
# Site "kind" hints downstream consumers what a region means semantically.
# ``screen`` is special-cased by build_usd.py (PR-C) to optionally place a
# material-override overlay there. Other kinds are emitted as plain invisible
# Cubes — the kind is just a label for robot policies.
SITE_KINDS = {"screen", "button", "camera", "handle", "custom"}


class SchemaError(SystemExit):
    """Schema violation. Inherits SystemExit so it propagates as a hard
    failure when raised from CLI scripts (no silent fallback)."""

    def __init__(self, msg: str):
        super().__init__(f"[schema] {msg}")


# ---------------------------------------------------------------------------
# Per-part / per-joint validators
# ---------------------------------------------------------------------------


def _require(d: dict, key: str, ctx: str) -> Any:
    if key not in d:
        raise SchemaError(f"{ctx}: missing required field '{key}'")
    return d[key]


def _require_vec3(v: Any, ctx: str) -> tuple[float, float, float]:
    if not (isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(c, (int, float)) for c in v)):
        raise SchemaError(f"{ctx}: expected 3-element numeric vector, got {v!r}")
    return float(v[0]), float(v[1]), float(v[2])


def _validate_part(p: dict, idx: int) -> None:
    ctx = f"parts[{idx}]"
    pid = _require(p, "id", ctx)
    if not isinstance(pid, str) or not pid:
        raise SchemaError(f"{ctx}.id: must be a non-empty string")
    clusters = _require(p, "clusters", ctx)
    if not isinstance(clusters, list) or not all(isinstance(c, str) for c in clusters):
        raise SchemaError(f"{ctx}.clusters: must be a list[str]")
    physics = _require(p, "physics", ctx)
    if physics not in PHYSICS_KINDS:
        raise SchemaError(f"{ctx}.physics: must be one of {sorted(PHYSICS_KINDS)}, got {physics!r}")
    coll = _require(p, "collision", ctx)
    approx = _require(coll, "approx", f"{ctx}.collision")
    if approx not in COLLISION_APPROXES:
        raise SchemaError(f"{ctx}.collision.approx: must be one of {sorted(COLLISION_APPROXES)}, got {approx!r}")
    if approx == "sdf":
        res = _require(coll, "resolution", f"{ctx}.collision")
        if not (isinstance(res, int) and res > 0):
            raise SchemaError(f"{ctx}.collision.resolution: positive int required for sdf, got {res!r}")
    _require_vec3(_require(p, "scale_xyz", ctx), f"{ctx}.scale_xyz")
    # mass is optional; if present must be positive
    if "mass" in p:
        m = p["mass"]
        if not (isinstance(m, (int, float)) and m > 0):
            raise SchemaError(f"{ctx}.mass: positive numeric required, got {m!r}")
    # collision_group is optional; parts with the same non-empty string
    # don't collide with each other (like Isaac Sim's collision filter groups).
    if "collision_group" in p and p["collision_group"] is not None:
        if not isinstance(p["collision_group"], str):
            raise SchemaError(f"{ctx}.collision_group: string or null required")
    sites = p.get("sites", [])
    if not isinstance(sites, list):
        raise SchemaError(f"{ctx}.sites: must be a list (may be empty)")
    site_ids: list[str] = []
    for si, s in enumerate(sites):
        _validate_site(s, si, ctx)
        site_ids.append(s["id"])
    if len(site_ids) != len(set(site_ids)):
        dups = [x for x in set(site_ids) if site_ids.count(x) > 1]
        raise SchemaError(f"{ctx}.sites: duplicate site ids: {dups}")


def _validate_site(s: dict, idx: int, ctx_part: str) -> None:
    """A site is an invisible AABB attached to a rigid body, used to label
    semantic regions ('screen', 'handle', etc.) that robot policies can
    address by prim path. See scripts/tools/articulator/usd_layout.svg."""
    ctx = f"{ctx_part}.sites[{idx}]"
    sid = _require(s, "id", ctx)
    if not isinstance(sid, str) or not sid:
        raise SchemaError(f"{ctx}.id: must be a non-empty string")
    kind = _require(s, "kind", ctx)
    if kind not in SITE_KINDS:
        raise SchemaError(f"{ctx}.kind: must be one of {sorted(SITE_KINDS)}, got {kind!r}")
    amin = _require_vec3(_require(s, "aabb_min", ctx), f"{ctx}.aabb_min")
    amax = _require_vec3(_require(s, "aabb_max", ctx), f"{ctx}.aabb_max")
    if any(amin[i] > amax[i] for i in range(3)):
        raise SchemaError(f"{ctx}: aabb_min must be component-wise <= aabb_max ({amin} vs {amax})")

    mo = s.get("material_override")
    if mo is not None:
        # Any site kind may carry a material override — paint a button red,
        # paint a handle rubber-black, etc. The ``kind`` is just a semantic
        # label; the override is geometric (an AABB-bound visible slab).
        if not isinstance(mo, dict):
            raise SchemaError(f"{ctx}.material_override: must be an object")
        col = _require(mo, "diffuseColor", f"{ctx}.material_override")
        if not (isinstance(col, (list, tuple)) and len(col) == 3
                and all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in col)):
            raise SchemaError(f"{ctx}.material_override.diffuseColor: 3 numbers in [0,1] required")
        for k in ("roughness", "metallic"):
            if k in mo:
                v = mo[k]
                if not (isinstance(v, (int, float)) and 0 <= v <= 1):
                    raise SchemaError(f"{ctx}.material_override.{k}: number in [0,1] required")


def _validate_joint(j: dict, idx: int, part_ids: set[str]) -> None:
    ctx = f"joints[{idx}]"
    jid = _require(j, "id", ctx)
    if not isinstance(jid, str) or not jid:
        raise SchemaError(f"{ctx}.id: must be a non-empty string")
    parent = _require(j, "parent", ctx)
    child = _require(j, "child", ctx)
    if parent not in part_ids:
        raise SchemaError(f"{ctx}.parent={parent!r}: not in parts[]")
    if child not in part_ids:
        raise SchemaError(f"{ctx}.child={child!r}: not in parts[]")
    if parent == child:
        raise SchemaError(f"{ctx}: parent and child are the same ({parent!r})")
    jt = _require(j, "type", ctx)
    if jt not in JOINT_TYPES:
        raise SchemaError(f"{ctx}.type: must be one of {sorted(JOINT_TYPES)}, got {jt!r}")
    if jt == "revolute":
        _require_vec3(_require(j, "axis_p0", ctx), f"{ctx}.axis_p0")
        _require_vec3(_require(j, "axis_p1", ctx), f"{ctx}.axis_p1")
        for k in ("lower", "upper"):
            v = _require(j, k, ctx)
            if not isinstance(v, (int, float)):
                raise SchemaError(f"{ctx}.{k}: numeric required, got {v!r}")
    elif jt == "prismatic":
        _require_vec3(_require(j, "axis_dir", ctx), f"{ctx}.axis_dir")
        _require_vec3(_require(j, "axis_origin", ctx), f"{ctx}.axis_origin")
        for k in ("lower", "upper"):
            v = _require(j, k, ctx)
            if not isinstance(v, (int, float)):
                raise SchemaError(f"{ctx}.{k}: numeric required, got {v!r}")
    # fixed / free: no extra required fields


def _validate_external_mesh(em: dict, idx: int, part_ids: set[str]) -> None:
    ctx = f"external_meshes[{idx}]"
    attach = _require(em, "attach_to", ctx)
    if attach not in part_ids:
        raise SchemaError(f"{ctx}.attach_to={attach!r}: not in parts[]")
    _require(em, "glb", ctx)
    t = _require(em, "transform", ctx)
    _require_vec3(_require(t, "t", f"{ctx}.transform"), f"{ctx}.transform.t")
    s = _require(t, "s", f"{ctx}.transform")
    _require_vec3(s, f"{ctx}.transform.s")
    q = _require(t, "q_wxyz", f"{ctx}.transform")
    if not (isinstance(q, (list, tuple)) and len(q) == 4):
        raise SchemaError(f"{ctx}.transform.q_wxyz: 4-element quat required, got {q!r}")


# ---------------------------------------------------------------------------
# Topology checks
# ---------------------------------------------------------------------------


def _check_tree_topology(parts: list[dict], joints: list[dict]) -> None:
    """Each part must have at most one incoming joint (so the kinematic
    structure is a forest of trees, not a DAG). Detect cycles."""
    incoming: dict[str, str] = {}  # child -> joint id
    for j in joints:
        if j["type"] == "free":
            continue  # free joints don't form a parent edge
        c = j["child"]
        if c in incoming:
            raise SchemaError(
                f"part {c!r} has multiple incoming joints "
                f"({incoming[c]!r} and {j['id']!r}) — kinematic structure must be a tree"
            )
        incoming[c] = j["id"]

    # DFS for cycles
    parent_of: dict[str, str] = {}
    for j in joints:
        if j["type"] != "free":
            parent_of[j["child"]] = j["parent"]

    for start in {p["id"] for p in parts}:
        seen, node = [], start
        while node in parent_of:
            seen.append(node)
            node = parent_of[node]
            if node in seen:
                raise SchemaError(f"cycle detected in joints: {' -> '.join(seen + [node])}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate(labels: dict) -> None:
    """Validate a v2 ``labels.json`` dict in-place. Raises SchemaError on
    any violation; returns silently on success."""
    v = labels.get("version")
    if v not in SUPPORTED_VERSIONS:
        raise SchemaError(
            f"unsupported schema version: {v!r} (supported {sorted(SUPPORTED_VERSIONS)}). "
            f"Run scripts/tools/articulator/migrate_v1_to_v2.py for v1, or migrate_v2_to_v3.py for v2."
        )

    parts = labels.get("parts")
    if not isinstance(parts, list) or not parts:
        raise SchemaError("'parts' must be a non-empty list")
    for i, p in enumerate(parts):
        _validate_part(p, i)

    ids = [p["id"] for p in parts]
    if len(ids) != len(set(ids)):
        dups = [x for x in set(ids) if ids.count(x) > 1]
        raise SchemaError(f"duplicate part ids: {dups}")
    part_ids = set(ids)

    joints = labels.get("joints", [])
    if not isinstance(joints, list):
        raise SchemaError("'joints' must be a list (may be empty)")
    for i, j in enumerate(joints):
        _validate_joint(j, i, part_ids)

    jids = [j["id"] for j in joints]
    if len(jids) != len(set(jids)):
        dups = [x for x in set(jids) if jids.count(x) > 1]
        raise SchemaError(f"duplicate joint ids: {dups}")

    _check_tree_topology(parts, joints)

    em = labels.get("external_meshes", [])
    if not isinstance(em, list):
        raise SchemaError("'external_meshes' must be a list (may be empty)")
    for i, e in enumerate(em):
        _validate_external_mesh(e, i, part_ids)

    splits = labels.get("split_clusters", {})
    if not isinstance(splits, dict):
        raise SchemaError("'split_clusters' must be an object/dict (may be empty)")
    for name, info in splits.items():
        ctx = f"split_clusters[{name!r}]"
        if not isinstance(info, dict):
            raise SchemaError(f"{ctx}: must be an object")
        _require(info, "parent", ctx)
        _require_vec3(_require(info, "aabb_min", ctx), f"{ctx}.aabb_min")
        _require_vec3(_require(info, "aabb_max", ctx), f"{ctx}.aabb_max")

    dims = labels.get("physical_dims_mm")
    if dims is not None:
        for k in ("x", "y", "z"):
            if k not in dims or not isinstance(dims[k], (int, float)) or dims[k] <= 0:
                raise SchemaError(f"physical_dims_mm.{k}: positive numeric required")


def topo_sort(labels: dict) -> list[str]:
    """Return part ids in parents-before-children order. Roots first.
    Caller must have validated the schema first (no cycle / missing-ref check here)."""
    parent_of = {j["child"]: j["parent"] for j in labels.get("joints", []) if j["type"] != "free"}
    order: list[str] = []
    seen: set[str] = set()

    def visit(pid: str):
        if pid in seen:
            return
        if pid in parent_of:
            visit(parent_of[pid])
        seen.add(pid)
        order.append(pid)

    for p in labels["parts"]:
        visit(p["id"])
    return order


def joints_by_child(labels: dict) -> dict[str, dict]:
    """Map child part id -> the joint whose ``child`` is that part. Useful
    for the USDA emitter: per-part joint lookup. Excludes free joints (they
    don't have a meaningful parent edge in the kinematic tree)."""
    return {j["child"]: j for j in labels.get("joints", []) if j["type"] != "free"}
