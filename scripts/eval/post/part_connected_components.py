from __future__ import annotations

from typing import Any

import numpy as np


def unique_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    return np.unique(arr, axis=0).astype(np.int32, copy=False)


def connected_components(coords: np.ndarray, *, resolution: int = 64) -> list[np.ndarray]:
    arr = unique_coords(coords).astype(np.int64, copy=False)
    if arr.size == 0:
        return []
    coord_set = {tuple(map(int, row)) for row in arr.tolist()}
    seen: set[tuple[int, int, int]] = set()
    components: list[np.ndarray] = []
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for start in coord_set:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[tuple[int, int, int]] = []
        while stack:
            x, y, z = stack.pop()
            component.append((x, y, z))
            for dx, dy, dz in neighbors:
                neighbor = (x + dx, y + dy, z + dz)
                if (
                    0 <= neighbor[0] < int(resolution)
                    and 0 <= neighbor[1] < int(resolution)
                    and 0 <= neighbor[2] < int(resolution)
                    and neighbor in coord_set
                    and neighbor not in seen
                ):
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(component, dtype=np.int32))
    components.sort(key=lambda item: int(item.shape[0]), reverse=True)
    return components


def bbox_gap(a: np.ndarray, b: np.ndarray) -> int:
    if a.size == 0 or b.size == 0:
        return 10**9
    a64 = np.asarray(a, dtype=np.int64).reshape(-1, 3)
    b64 = np.asarray(b, dtype=np.int64).reshape(-1, 3)
    alo, ahi = a64.min(axis=0), a64.max(axis=0)
    blo, bhi = b64.min(axis=0), b64.max(axis=0)
    gaps = np.maximum(0, np.maximum(alo - bhi - 1, blo - ahi - 1))
    return int(gaps.max(initial=0))


def filter_part_connected_components(
    coords: np.ndarray,
    *,
    part_index: int,
    part_name: str,
    min_component_voxels: int,
    min_component_fraction: float,
    max_component_distance: int,
    max_large_component_distance: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    original = unique_coords(coords)
    components = connected_components(original)
    thresholds = {
        "min_component_voxels": int(min_component_voxels),
        "min_component_fraction": float(min_component_fraction),
        "max_component_distance": int(max_component_distance),
        "max_large_component_distance": (
            None if max_large_component_distance is None else int(max_large_component_distance)
        ),
    }
    if not components:
        return original, {
            "enabled": True,
            "part_index": int(part_index),
            "part_name": str(part_name),
            "component_count": 0,
            "largest_component_voxels": 0,
            "largest_component_fraction": 0.0,
            "kept_component_count": 0,
            "removed_component_count": 0,
            "original_voxels": 0,
            "kept_voxels": 0,
            "reassigned_to_body_voxels": 0,
            "thresholds": thresholds,
            "components": [],
        }

    main = components[0]
    main_count = max(1, int(main.shape[0]))
    kept: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    removed_voxels = 0
    for component_index, component in enumerate(components):
        count = int(component.shape[0])
        fraction = float(count / main_count)
        distance = 0 if component_index == 0 else bbox_gap(component, main)
        large = count >= int(min_component_voxels) or fraction >= float(min_component_fraction)
        near = distance <= int(max_component_distance)
        large_within_limit = large and (
            max_large_component_distance is None or distance <= int(max_large_component_distance)
        )
        keep = component_index == 0 or near or large_within_limit
        if keep:
            kept.append(component)
        else:
            removed_voxels += count
        if component_index == 0:
            reason = "largest"
        elif near:
            reason = "near"
        elif large_within_limit:
            reason = "large_within_distance_limit"
        elif large:
            reason = "large_but_remote_reassigned_to_body"
        else:
            reason = "small_remote_island_reassigned_to_body"
        details.append(
            {
                "component_index": int(component_index),
                "voxels": count,
                "fraction_of_largest": fraction,
                "bbox_gap_to_largest": int(distance),
                "kept": bool(keep),
                "reason": reason,
            }
        )

    filtered = unique_coords(np.concatenate(kept, axis=0) if kept else np.empty((0, 3), dtype=np.int32))
    return filtered, {
        "enabled": True,
        "part_index": int(part_index),
        "part_name": str(part_name),
        "component_count": int(len(components)),
        "largest_component_voxels": int(main_count),
        "largest_component_fraction": float(main_count / max(1, int(original.shape[0]))),
        "kept_component_count": int(sum(1 for item in details if item["kept"])),
        "removed_component_count": int(sum(1 for item in details if not item["kept"])),
        "original_voxels": int(original.shape[0]),
        "kept_voxels": int(filtered.shape[0]),
        "reassigned_to_body_voxels": int(removed_voxels),
        "thresholds": thresholds,
        "components": details,
    }
