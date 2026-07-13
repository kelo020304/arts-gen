"""Standalone joint sampling and transform utilities.

This module extracts the joint parsing, sampling, and transform math from
``arts-reconstruction/scripts/joint_angle_expand.py`` without depending on
``trimesh`` or Blender-specific runtime modules.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import deque
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

ALLOWED_JOINT_TYPES = {"A", "B", "C", "CB", "D", "E"}


def _is_int_like(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _normalize_group_info(jsondata: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(jsondata, Mapping):
        raise TypeError(f"jsondata must be a mapping, got {type(jsondata).__name__}")
    if "group_info" not in jsondata:
        raise KeyError("jsondata must contain 'group_info'")

    raw_group_info = jsondata["group_info"]
    if not isinstance(raw_group_info, Mapping):
        raise TypeError(
            f"jsondata['group_info'] must be a mapping, got {type(raw_group_info).__name__}"
        )

    group_info: Dict[str, Any] = {}
    for raw_gid, gval in raw_group_info.items():
        gid = str(raw_gid)
        if gid in group_info:
            raise ValueError(f"duplicate group id after normalization: {gid}")
        group_info[gid] = gval

    if "0" not in group_info:
        raise KeyError("group_info must contain base group '0'")
    return group_info


def _normalize_part_indices(value: Any, field_name: str) -> List[int]:
    if _is_int_like(value):
        return [int(value)]
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be an int or list[int], got {type(value).__name__}")

    indices: List[int] = []
    for idx, item in enumerate(value):
        if not _is_int_like(item):
            raise TypeError(
                f"{field_name}[{idx}] must be an int, got {type(item).__name__}"
            )
        indices.append(int(item))
    return indices


def _normalize_parent_group(value: Any, gid: str) -> str:
    if not isinstance(value, (str, int, np.integer)):
        raise TypeError(
            f"group {gid} parent_group must be a string or int, got {type(value).__name__}"
        )
    return str(value)


def _normalize_params(value: Any, gid: str) -> List[Any]:
    if not isinstance(value, list):
        raise TypeError(f"group {gid} params must be a list, got {type(value).__name__}")
    return value


def _normalize_joint_type(value: Any, gid: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    if value not in ALLOWED_JOINT_TYPES:
        raise ValueError(f"group {gid} has unsupported joint type: {value}")
    return value


def _parse_group_entry(gid: str, gval: Any) -> Tuple[List[int], str, List[Any], str]:
    if not isinstance(gval, list):
        raise TypeError(f"group {gid} entry must be a list, got {type(gval).__name__}")
    if len(gval) != 4:
        raise ValueError(f"group {gid} entry must have length 4, got {len(gval)}")

    part_indices = _normalize_part_indices(gval[0], f"group {gid} part_indices")
    parent_group = _normalize_parent_group(gval[1], gid)
    params = _normalize_params(gval[2], gid)
    joint_type = _normalize_joint_type(gval[3], gid)
    return part_indices, parent_group, params, joint_type


def _validate_group_connectivity(group_info: Mapping[str, Any], children: Mapping[str, List[str]]) -> None:
    queue: deque[str] = deque(["0"])
    visited = set()

    while queue:
        gid = queue.popleft()
        if gid in visited:
            continue
        visited.add(gid)
        queue.extend(children[gid])

    missing = set(group_info) - visited
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"group tree contains unreachable groups: {missing_list}")


def _as_float_array(values: Sequence[Any], expected_len: int, name: str) -> np.ndarray:
    if len(values) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(values)}")
    for idx, value in enumerate(values):
        if not _is_number(value):
            raise TypeError(f"{name}[{idx}] must be numeric, got {type(value).__name__}")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_float_state_mapping(
    joint_states: Mapping[str, Any], expected_group_ids: Sequence[str], group_joint: Mapping[str, Dict[str, Any]]
) -> Dict[str, float]:
    if not isinstance(joint_states, Mapping):
        raise TypeError(f"joint_states must be a mapping, got {type(joint_states).__name__}")

    normalized: Dict[str, float] = {}
    for raw_gid, raw_state in joint_states.items():
        gid = str(raw_gid)
        if gid in normalized:
            raise ValueError(f"duplicate joint state after normalization: {gid}")
        if gid not in group_joint:
            raise KeyError(f"joint_states contains unknown group id: {gid}")
        if not _is_number(raw_state):
            raise TypeError(f"joint state for group {gid} must be numeric, got {type(raw_state).__name__}")
        state = float(raw_state)
        if not math.isfinite(state):
            raise ValueError(f"joint state for group {gid} must be finite")
        if state < 0.0 or state > 1.0:
            raise ValueError(f"joint state for group {gid} must be in [0, 1], got {state}")
        normalized[gid] = state

    expected = set(expected_group_ids)
    missing = expected - set(normalized)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"joint_states is missing groups: {missing_list}")

    if normalized["0"] != 0.0:
        raise ValueError("base group '0' must have joint state 0.0")

    for gid, jinfo in group_joint.items():
        if jinfo["joint_type"] == "E" and normalized[gid] != 0.0:
            raise ValueError(f"fixed group {gid} must have joint state 0.0")

    return normalized


def _unit_vector(data: Sequence[Any]) -> np.ndarray:
    vector = np.asarray(data, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"direction must be 1D, got shape {vector.shape}")
    norm = math.sqrt(float(np.dot(vector, vector)))
    if norm <= 0.0:
        raise ValueError("direction vector must have non-zero length")
    return vector / norm


def rotation_matrix(angle: float, direction: Sequence[Any], point: Sequence[Any] | None = None) -> np.ndarray:
    """Return a 4x4 homogeneous matrix rotating around ``direction`` and ``point``.

    This matches the numeric behavior of ``trimesh.transformations.rotation_matrix``.
    """
    sina = np.sin(angle)
    cosa = np.cos(angle)

    direction_vec = _unit_vector(direction[:3])

    matrix = np.diag([cosa, cosa, cosa, 1.0])
    matrix[:3, :3] += np.outer(direction_vec, direction_vec) * (1.0 - cosa)

    direction_vec = direction_vec * sina
    matrix[:3, :3] += np.array(
        [
            [0.0, -direction_vec[2], direction_vec[1]],
            [direction_vec[2], 0.0, -direction_vec[0]],
            [-direction_vec[1], direction_vec[0], 0.0],
        ],
        dtype=np.float64,
    )

    if point is not None:
        point_vec = np.asarray(point[:3], dtype=np.float64)
        if point_vec.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {point_vec.shape}")
        if not np.all(np.isfinite(point_vec)):
            raise ValueError("point must contain only finite values")
        matrix[:3, 3] = point_vec - np.dot(matrix[:3, :3], point_vec)

    return matrix


def translation_matrix(direction: Sequence[Any]) -> np.ndarray:
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"translation direction must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("translation direction must contain only finite values")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = vector
    return matrix


def _transform_direction(parent_world: np.ndarray, vec: Sequence[Any]) -> List[float]:
    direction = _as_float_array(vec[:3], 3, "joint direction")
    rotated = parent_world[:3, :3] @ direction
    return _unit_vector(rotated).tolist()


def _transform_point(parent_world: np.ndarray, pt: Sequence[Any]) -> List[float]:
    point = _as_float_array(pt[:3], 3, "joint point")
    point_h = np.ones(4, dtype=np.float64)
    point_h[:3] = point
    return (parent_world @ point_h)[:3].tolist()


def _lift_joint_params_to_world(joint_type: str, params: List[Any], parent_world: np.ndarray) -> List[Any]:
    lifted = list(params)
    if joint_type == "B":
        lifted[:3] = _transform_direction(parent_world, params[:3])
        return lifted
    if joint_type in {"C", "A", "D"}:
        lifted[:3] = _transform_direction(parent_world, params[:3])
        lifted[3:6] = _transform_point(parent_world, params[3:6])
        return lifted
    if joint_type == "CB":
        lifted[:3] = _transform_direction(parent_world, params[:3])
        lifted[3:6] = _transform_point(parent_world, params[3:6])
        lifted[8:11] = _transform_direction(parent_world, params[8:11])
        return lifted
    if joint_type in {"E"}:
        return lifted
    raise ValueError(f"unsupported joint type: {joint_type}")


def _detect_pivot_mode(
    child_origin: Sequence[Any], parent_origin: Sequence[Any], mesh_centroid: Sequence[Any]
) -> str:
    """Choose between global and parent-offset nested pivot annotations."""
    child_o = np.array(child_origin[:3], dtype=np.float64)
    parent_o = np.array(parent_origin[:3], dtype=np.float64)
    centroid = np.array(mesh_centroid[:3], dtype=np.float64)

    global_pivot = child_o
    offset_pivot = parent_o + child_o

    dist_global = np.linalg.norm(centroid - global_pivot)
    dist_offset = np.linalg.norm(centroid - offset_pivot)
    return "parent_offset" if dist_offset < dist_global else "global"


def _resolve_obj_path(obj_dir: str, obj_ref: int | str) -> str:
    if isinstance(obj_ref, str):
        obj_path = os.path.join(obj_dir, f"{obj_ref}.obj")
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")
        return obj_path

    if _is_int_like(obj_ref):
        obj_idx = int(obj_ref)
        candidate_paths = [
            os.path.join(obj_dir, f"original-{obj_idx}.obj"),
            os.path.join(obj_dir, f"part_{obj_idx}.obj"),
        ]
        for candidate_path in candidate_paths:
            if os.path.exists(candidate_path):
                return candidate_path
        raise FileNotFoundError(f"OBJ file not found for part index {obj_idx}: {candidate_paths}")

    raise TypeError(f"obj_ref must be a string or int-like value, got {type(obj_ref).__name__}")


def _get_mesh_centroid(obj_dir: str, part_idx: int | str | Sequence[int | str]) -> np.ndarray:
    """Compute the centroid from one or more OBJ files inside ``obj_dir``."""
    if isinstance(part_idx, str):
        obj_refs: List[int | str] = [part_idx]
    elif _is_int_like(part_idx):
        obj_refs = [int(part_idx)]
    else:
        if not isinstance(part_idx, Sequence):
            raise TypeError(
                f"part_idx must be an int, str, or sequence[int | str], got {type(part_idx).__name__}"
            )
        obj_refs = []
        for idx, obj_idx in enumerate(part_idx):
            if not _is_int_like(obj_idx) and not isinstance(obj_idx, str):
                raise TypeError(
                    f"part_idx[{idx}] must be an int or str, got {type(obj_idx).__name__}"
                )
            obj_refs.append(obj_idx if isinstance(obj_idx, str) else int(obj_idx))

    vertices: List[List[float]] = []
    for obj_ref in obj_refs:
        obj_path = _resolve_obj_path(obj_dir, obj_ref)

        with open(obj_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("v "):
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not vertices:
        raise ValueError(f"No vertices found in OBJ set for {obj_refs}")
    return np.mean(vertices, axis=0)


def _resolve_nested_params(
    joint_type: str, child_params: Sequence[Any], parent_params: Sequence[Any], pivot_mode: str
) -> List[Any]:
    """Resolve nested rest-space params before world lifting."""
    resolved = list(child_params)
    if pivot_mode == "global":
        return resolved
    if pivot_mode != "parent_offset":
        raise ValueError(f"unsupported pivot mode: {pivot_mode}")

    if joint_type in {"C", "A", "D", "CB"}:
        for idx in range(3, 6):
            resolved[idx] = float(parent_params[idx]) + float(child_params[idx])
    return resolved


def parse_joint_params(jsondata) -> List[Dict[str, Any]]:
    """Parse ``group_info`` into structured joint descriptions."""
    group_info = _normalize_group_info(jsondata)
    joints: List[Dict[str, Any]] = []

    for gid, gval in group_info.items():
        if gid == "0":
            continue
        part_indices, parent_group, params, joint_type = _parse_group_entry(gid, gval)
        joints.append(
            {
                "group_id": gid,
                "part_indices": part_indices,
                "parent_group": parent_group,
                "joint_type": joint_type,
                "params": params,
            }
        )

    return joints


def build_group_tree(jsondata) -> Tuple[Dict[str, List[str]], Dict[str, List[int]], Dict[str, Dict[str, Any]]]:
    """Build parent-child adjacency and group metadata for traversal."""
    group_info = _normalize_group_info(jsondata)

    children: Dict[str, List[str]] = {gid: [] for gid in group_info}
    group_parts: Dict[str, List[int]] = {}
    group_joint: Dict[str, Dict[str, Any]] = {}

    for gid, gval in group_info.items():
        if gid == "0":
            group_parts[gid] = _normalize_part_indices(gval, "group 0 part_indices")
            group_joint[gid] = {"joint_type": "E", "params": []}
            continue

        part_indices, parent_group, params, joint_type = _parse_group_entry(gid, gval)
        if parent_group not in group_info:
            raise KeyError(f"group {gid} references missing parent group {parent_group}")

        group_parts[gid] = part_indices
        group_joint[gid] = {"joint_type": joint_type, "params": params}
        children[parent_group].append(gid)

    _validate_group_connectivity(group_info, children)
    return children, group_parts, group_joint


def get_all_descendant_parts(gid, children, group_parts) -> List[int]:
    """Collect part indices for ``gid`` and all descendants using BFS."""
    if gid not in group_parts:
        raise KeyError(f"unknown group id: {gid}")

    parts: List[int] = []
    queue: deque[str] = deque([gid])

    while queue:
        current_gid = queue.popleft()
        parts.extend(group_parts[current_gid])
        queue.extend(children[current_gid])

    return parts


def sample_joint_states(jsondata, seed) -> Dict[str, float]:
    """Sample deterministic joint states in ``[0, 1]`` for articulated groups."""
    group_info = _normalize_group_info(jsondata)
    np.random.seed(int(seed))

    joint_states: Dict[str, float] = {}
    for gid, gval in group_info.items():
        if gid == "0":
            joint_states[gid] = 0.0
            continue

        _, _, _, joint_type = _parse_group_entry(gid, gval)
        if joint_type in {"E"}:
            joint_states[gid] = 0.0
        else:
            joint_states[gid] = float(np.random.uniform(0.0, 1.0))

    return joint_states


def _compute_local_transform(joint_type: str, params: List[Any], state: float) -> np.ndarray:
    if joint_type == "B":
        direction = _as_float_array(params[:3], 3, "prismatic direction")
        slide_range = _as_float_array(params[6:8], 2, "prismatic slide_range")
        displacement = slide_range[0] + (slide_range[1] - slide_range[0]) * state
        return translation_matrix(direction * displacement)

    if joint_type == "C":
        axis_dir = _as_float_array(params[:3], 3, "revolute axis_dir")
        axis_pos = _as_float_array(params[3:6], 3, "revolute axis_pos")
        angle_range = _as_float_array(params[6:8], 2, "revolute angle_range")
        angle_lo = angle_range[0] * math.pi
        angle_hi = angle_range[1] * math.pi
        angle = angle_lo + (angle_hi - angle_lo) * state
        return rotation_matrix(angle, axis_dir, axis_pos)

    if joint_type == "A":
        axis_dir = _as_float_array(params[:3], 3, "free axis_dir")
        axis_pos = _as_float_array(params[3:6], 3, "free axis_pos")
        return rotation_matrix(2.0 * math.pi * state, axis_dir, axis_pos)

    if joint_type == "D":
        axis_dir = _as_float_array(params[:3], 3, "pivot axis_dir")
        axis_pos = _as_float_array(params[3:6], 3, "pivot axis_pos")
        return rotation_matrix(math.pi * state, axis_dir, axis_pos)

    if joint_type == "CB":
        slide_range = _as_float_array(params[6:8], 2, "compound slide_range")
        if state < 0.5:
            slide_dir = _as_float_array(params[8:11], 3, "compound slide_dir")
            sub_state = state * 2.0
            displacement = slide_range[0] + (slide_range[1] - slide_range[0]) * sub_state
            return translation_matrix(slide_dir * displacement)

        axis_dir = _as_float_array(params[:3], 3, "compound axis_dir")
        axis_pos = _as_float_array(params[3:6], 3, "compound axis_pos")
        sub_state = (state - 0.5) * 2.0
        return rotation_matrix(math.pi * sub_state, axis_dir, axis_pos)

    if joint_type in {"E"}:
        return np.eye(4, dtype=np.float64)

    raise ValueError(f"unsupported joint type: {joint_type}")


def compute_joint_transforms(jsondata, joint_states, num_parts, obj_dir=None) -> Dict[int, List[List[float]]]:
    """Compute per-part homogeneous transforms without mutating any meshes."""
    if not _is_int_like(num_parts):
        raise TypeError(f"num_parts must be an int, got {type(num_parts).__name__}")
    num_parts_int = int(num_parts)
    if num_parts_int < 0:
        raise ValueError(f"num_parts must be non-negative, got {num_parts_int}")
    obj_dir_path = os.fspath(obj_dir) if obj_dir is not None else None
    parts_data = None
    if obj_dir_path is not None:
        parts_data = jsondata["parts"]
        if not isinstance(parts_data, list):
            raise TypeError(f"jsondata['parts'] must be a list, got {type(parts_data).__name__}")

    children, group_parts, group_joint = build_group_tree(jsondata)
    state_map = _as_float_state_mapping(joint_states, list(group_parts), group_joint)

    part_transforms = {part_idx: np.eye(4, dtype=np.float64) for part_idx in range(num_parts_int)}
    group_world_transform = {"0": np.eye(4, dtype=np.float64)}
    queue: deque[Tuple[str, str | None]] = deque([("0", None)])
    visited = set()

    while queue:
        gid, parent_gid = queue.popleft()
        if gid in visited:
            continue
        visited.add(gid)

        for child_gid in children[gid]:
            queue.append((child_gid, gid))

        if gid == "0":
            continue

        if parent_gid is None or parent_gid not in group_world_transform:
            raise KeyError(f"missing parent world transform for group {gid}: {parent_gid}")

        parent_world = group_world_transform[parent_gid]
        jinfo = group_joint[gid]
        joint_type = jinfo["joint_type"]
        state = state_map[gid]

        if joint_type == "E" or (joint_type == "A" and (parent_gid == "0" or obj_dir_path is None)):
            local_mat = np.eye(4, dtype=np.float64)
        else:
            raw_params = jinfo["params"]
            if obj_dir_path is not None and parent_gid != "0" and joint_type in {"C", "A", "D", "CB"}:
                parent_jinfo = group_joint[parent_gid]
                child_obj_refs: List[str] = []
                for part_idx in group_parts[gid]:
                    if part_idx < 0 or part_idx >= len(parts_data):
                        raise ValueError(
                            f"group {gid} references part index {part_idx}, outside [0, {len(parts_data) - 1}]"
                        )
                    part_data = parts_data[part_idx]
                    if not isinstance(part_data, Mapping):
                        raise TypeError(
                            f"jsondata['parts'][{part_idx}] must be a mapping, "
                            f"got {type(part_data).__name__}"
                        )
                    obj_names = part_data.get("obj")
                    if not isinstance(obj_names, list):
                        raise TypeError(f"part {part_idx} obj field must be a list")
                    if not obj_names:
                        raise ValueError(f"part {part_idx} must reference at least one OBJ file")

                    for obj_name in obj_names:
                        if not isinstance(obj_name, str):
                            raise TypeError(
                                f"part {part_idx} obj entries must be strings, got {type(obj_name).__name__}"
                            )
                        child_obj_refs.append(obj_name)

                centroid = _get_mesh_centroid(obj_dir_path, child_obj_refs)
                pivot_mode = _detect_pivot_mode(
                    raw_params[3:6], parent_jinfo["params"][3:6], centroid
                )
                raw_params = _resolve_nested_params(
                    joint_type, raw_params, parent_jinfo["params"], pivot_mode
                )

            world_params = _lift_joint_params_to_world(joint_type, raw_params, parent_world)
            local_mat = _compute_local_transform(joint_type, world_params, state)
            local_mat = np.linalg.inv(parent_world) @ local_mat @ parent_world

        group_world_transform[gid] = parent_world @ local_mat
        affected_parts = get_all_descendant_parts(gid, children, group_parts)

        for part_idx in affected_parts:
            if part_idx < 0 or part_idx >= num_parts_int:
                raise ValueError(
                    f"group {gid} references part index {part_idx}, outside [0, {num_parts_int - 1}]"
                )
            part_transforms[part_idx] = group_world_transform[gid].copy()

    return {part_idx: transform.tolist() for part_idx, transform in part_transforms.items()}


def _zero_joint_states(jsondata) -> Dict[str, float]:
    group_info = _normalize_group_info(jsondata)
    return {gid: 0.0 for gid in group_info}


def generate_transforms_json(object_id, angle_idx, jsondata, num_parts, obj_dir=None) -> Dict[str, Any]:
    """Generate JSON-serializable joint states and per-part transforms."""
    if angle_idx == 0:
        joint_states = _zero_joint_states(jsondata)
        part_transforms = {
            part_idx: np.eye(4, dtype=np.float64).tolist() for part_idx in range(int(num_parts))
        }
    else:
        object_id_text = str(object_id)
        try:
            object_seed = int(object_id_text)
        except ValueError:
            try:
                object_seed = int(object_id_text, 16)
            except ValueError:
                digest = hashlib.sha256(object_id_text.encode("utf-8")).hexdigest()
                object_seed = int(digest[:16], 16)
        seed = (object_seed * 100 + int(angle_idx)) % (2**32 - 1)
        joint_states = sample_joint_states(jsondata, seed)
        part_transforms = compute_joint_transforms(jsondata, joint_states, num_parts, obj_dir=obj_dir)

    return {
        "object_id": object_id,
        "angle_idx": angle_idx,
        "joint_states": joint_states,
        "part_transforms": part_transforms,
    }


__all__ = [
    "build_group_tree",
    "compute_joint_transforms",
    "generate_transforms_json",
    "get_all_descendant_parts",
    "parse_joint_params",
    "rotation_matrix",
    "sample_joint_states",
    "translation_matrix",
]
