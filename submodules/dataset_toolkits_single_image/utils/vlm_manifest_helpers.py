from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ACCEPTED_MOTION_TYPES = {"A", "B", "C"}
MIN_PART_VOXELS = 5
VOXEL_FILTER_REASON = "below_min_part_voxels_5"
SYSTEM_PROMPT = ""
FOUR_IMAGE_GROUP_PROMPT = (
    "<image>\n<image>\n<image>\n<image>\n"
    "These are 4 views of an articulated object. "
    "Image 1 (group_0), Image 2 (group_1), Image 3 (group_2), Image 4 (group_3). "
    "Identify its physical structure: name, category, dimensions, "
    "and all components with their parent-child relationships, "
    "motion types (rotate/prismatic), motion parameters, "
    "and 2D bounding boxes in each view. Output as JSON."
)
SINGLE_IMAGE_PROMPT = (
    "<image>\n"
    "This is a view of an articulated object. "
    "Identify its physical structure: name, category, dimensions, "
    "and all components with their parent-child relationships, "
    "motion types (rotate/prismatic), motion parameters, "
    "and 2D bounding boxes in each view. Output as JSON."
)


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list, got {type(value).__name__}")
    return value


def require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def round_list(values: list[float], decimals: int = 2) -> list[float]:
    return [round(value, decimals) for value in values]


def label_sort_key(label: int | str) -> tuple[int, int | str]:
    if isinstance(label, int):
        return (0, label)
    if isinstance(label, str) and label.isdigit():
        return (0, int(label))
    return (1, str(label))


def get_manifest_angle_parts(
    manifest: dict[str, Any],
    object_id: str,
    angle_idx: int,
) -> dict[str, Any]:
    objects = require_mapping(manifest.get("objects"), "manifest['objects']")
    object_record = require_mapping(
        objects.get(object_id),
        f"manifest['objects']['{object_id}']",
    )
    angles = require_mapping(
        object_record.get("angles"),
        f"manifest['objects']['{object_id}']['angles']",
    )
    angle_record = require_mapping(
        angles.get(str(angle_idx)),
        f"manifest['objects']['{object_id}']['angles']['{angle_idx}']",
    )
    return require_mapping(
        angle_record.get("parts"),
        f"manifest['objects']['{object_id}']['angles']['{angle_idx}']['parts']",
    )


def is_voxel_kept_part(part_record: dict[str, Any], component_key: str, context: str) -> bool:
    has_voxel_ind = part_record.get("has_voxel_ind")
    if not isinstance(has_voxel_ind, bool):
        raise TypeError(f"{context}['{component_key}']['has_voxel_ind'] must be bool")

    num_voxels = part_record.get("num_voxels")
    if isinstance(num_voxels, bool) or not isinstance(num_voxels, int):
        raise TypeError(f"{context}['{component_key}']['num_voxels'] must be int")

    voxel_ind_path = part_record.get("voxel_ind_path")
    if has_voxel_ind:
        if not isinstance(voxel_ind_path, str) or not voxel_ind_path:
            raise ValueError(
                f"{context}['{component_key}'] has_voxel_ind=true but voxel_ind_path is empty"
            )
        if num_voxels <= MIN_PART_VOXELS:
            raise ValueError(
                f"{context}['{component_key}'] has_voxel_ind=true but num_voxels={num_voxels} "
                f"<= MIN_PART_VOXELS={MIN_PART_VOXELS}"
            )
        return True

    filter_reason = part_record.get("filter_reason")
    if filter_reason is not None and filter_reason != VOXEL_FILTER_REASON:
        raise ValueError(
            f"{context}['{component_key}'] unexpected voxel filter_reason: {filter_reason!r}"
        )
    return False


def validate_component_tree(
    components: dict[str, Any],
    context: str,
    expected_root_key: str | None = None,
) -> None:
    if not components:
        raise ValueError(f"{context}: components must not be empty")

    roots: list[str] = []
    for comp_key, comp_payload in components.items():
        if not isinstance(comp_key, str) or not comp_key:
            raise ValueError(f"{context}: component keys must be non-empty strings")
        comp = require_mapping(comp_payload, f"{context}.components['{comp_key}']")
        parent = comp.get("parent")
        if parent is None:
            roots.append(comp_key)
        else:
            if not isinstance(parent, str) or not parent:
                raise TypeError(
                    f"{context}.components['{comp_key}']['parent'] must be null or a non-empty string"
                )
            if parent == comp_key:
                raise ValueError(f"{context}: component '{comp_key}' cannot parent itself")

    if len(roots) != 1:
        raise ValueError(
            f"{context}: expected exactly one root component, found {len(roots)}: {roots}"
        )

    root_key = roots[0]
    if expected_root_key is not None and root_key != expected_root_key:
        raise ValueError(f"{context}: expected root key '{expected_root_key}', got '{root_key}'")

    for comp_key, comp_payload in components.items():
        comp = require_mapping(comp_payload, f"{context}.components['{comp_key}']")
        children = require_list(
            comp.get("children"),
            f"{context}.components['{comp_key}']['children']",
        )
        seen_children: set[str] = set()
        for child_idx, child_key_raw in enumerate(children):
            child_key = require_string(
                child_key_raw,
                f"{context}.components['{comp_key}']['children'][{child_idx}]",
            )
            if child_key in seen_children:
                raise ValueError(
                    f"{context}: duplicate child '{child_key}' under component '{comp_key}'"
                )
            seen_children.add(child_key)
            if child_key not in components:
                raise ValueError(
                    f"{context}: component '{comp_key}' references missing child '{child_key}'"
                )
            child_parent = require_mapping(
                components[child_key],
                f"{context}.components['{child_key}']",
            ).get("parent")
            if child_parent != comp_key:
                raise ValueError(
                    f"{context}: child '{child_key}' points to parent '{child_parent}', expected '{comp_key}'"
                )

    for comp_key, comp_payload in components.items():
        comp = require_mapping(comp_payload, f"{context}.components['{comp_key}']")
        parent = comp.get("parent")
        if parent is None:
            continue
        if parent not in components:
            raise ValueError(f"{context}: component '{comp_key}' references missing parent '{parent}'")
        parent_children = require_list(
            require_mapping(components[parent], f"{context}.components['{parent}']").get("children"),
            f"{context}.components['{parent}']['children']",
        )
        if comp_key not in parent_children:
            raise ValueError(f"{context}: parent '{parent}' does not list child '{comp_key}'")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(comp_key: str) -> None:
        if comp_key in visiting:
            raise ValueError(f"{context}: cycle detected at component '{comp_key}'")
        if comp_key in visited:
            return
        visiting.add(comp_key)
        children = require_list(
            require_mapping(components[comp_key], f"{context}.components['{comp_key}']").get("children"),
            f"{context}.components['{comp_key}']['children']",
        )
        for child_key in children:
            visit(str(child_key))
        visiting.remove(comp_key)
        visited.add(comp_key)

    visit(root_key)
    unreachable = sorted(set(components) - visited)
    if unreachable:
        raise ValueError(f"{context}: unreachable components from root '{root_key}': {unreachable}")


def filter_components_tree_for_voxel_kept_parts(
    components_tree: dict[str, Any],
    manifest_angle_parts: dict[str, Any],
    *,
    object_id: str,
    angle_idx: int,
) -> tuple[dict[str, Any] | None, int]:
    components = require_mapping(components_tree.get("components"), "components_tree['components']")
    root_key = require_string(components_tree.get("name"), "components_tree['name']")
    if root_key not in components:
        raise ValueError(f"Object {object_id}: root key '{root_key}' missing from components")

    kept_target_keys: set[str] = set()
    filtered_out_count = 0
    context = f"manifest object {object_id} angle_{angle_idx} parts"
    for comp_key, comp in components.items():
        comp_payload = require_mapping(comp, f"components_tree['components']['{comp_key}']")
        if comp_payload.get("parent") is None:
            continue
        part_record = manifest_angle_parts.get(comp_key)
        if part_record is None:
            raise KeyError(
                f"{context} missing component '{comp_key}'. Cannot decide voxel-kept VLM target set."
            )
        part_payload = require_mapping(part_record, f"{context}['{comp_key}']")
        if is_voxel_kept_part(part_payload, comp_key, context):
            kept_target_keys.add(comp_key)
        else:
            filtered_out_count += 1

    if not kept_target_keys:
        return None, filtered_out_count

    filtered_components: dict[str, dict[str, Any]] = {}
    root_component = dict(require_mapping(components[root_key], f"components['{root_key}']"))
    root_component["children"] = []
    filtered_components[root_key] = root_component

    for comp_key in components:
        if comp_key not in kept_target_keys:
            continue
        comp_copy = dict(require_mapping(components[comp_key], f"components['{comp_key}']"))
        comp_copy["children"] = []
        filtered_components[comp_key] = comp_copy

    def nearest_kept_parent(comp_key: str) -> str:
        parent = require_mapping(components[comp_key], f"components['{comp_key}']").get("parent")
        seen: set[str] = set()
        while isinstance(parent, str) and parent:
            if parent in seen:
                raise ValueError(f"Object {object_id}: cycle while repairing parent for {comp_key}")
            seen.add(parent)
            if parent == root_key or parent in kept_target_keys:
                return parent
            parent_payload = require_mapping(components[parent], f"components['{parent}']")
            parent = parent_payload.get("parent")
        return root_key

    for comp_key in kept_target_keys:
        parent_key = nearest_kept_parent(comp_key)
        filtered_components[comp_key]["parent"] = parent_key
        filtered_components[parent_key]["children"].append(comp_key)

    for comp in filtered_components.values():
        comp["children"].sort()

    filtered_tree = {key: value for key, value in components_tree.items() if key != "components"}
    filtered_tree["components"] = {
        key: filtered_components[key]
        for key in components
        if key in filtered_components
    }
    validate_component_tree(
        filtered_tree["components"],
        context=f"Object {object_id} angle_{angle_idx} voxel-kept components tree",
        expected_root_key=root_key,
    )
    return filtered_tree, filtered_out_count


def build_components_tree(
    obj_data: dict[str, Any],
    object_id: str,
    part_info_data: dict[str, Any],
) -> dict[str, Any] | None:
    missing_fields = [field for field in ("object_name", "category", "dimension") if field not in obj_data]
    if missing_fields:
        raise ValueError(f"Object {object_id} missing required fields: {', '.join(missing_fields)}")
    if "parts" not in obj_data:
        raise ValueError(f"Object {object_id} missing required field: parts")
    if "group_info" not in obj_data:
        raise ValueError(f"Object {object_id} missing required field: group_info")

    name = obj_data["object_name"]
    category = obj_data["category"]
    dimension = obj_data["dimension"]
    parts = obj_data["parts"]
    group_info = obj_data["group_info"]

    parts_by_label = {part["label"]: part for part in parts}
    fixed_group_ids: set[str] = set()
    articulated_groups: dict[str, tuple[list[int], str, list[Any], str]] = {}

    for group_id, group_val in group_info.items():
        if not isinstance(group_val, list) or len(group_val) == 0:
            raise ValueError(
                f"Object {object_id} group {group_id}: expected non-empty list, got {type(group_val).__name__}"
            )

        is_articulated = len(group_val) == 4 and isinstance(group_val[3], str)
        if not is_articulated:
            fixed_group_ids.add(str(group_id))
            continue

        links, parent_group_str, params, motion_type = group_val
        motion_type_str = str(motion_type)
        if motion_type_str not in ACCEPTED_MOTION_TYPES:
            continue
        if not isinstance(params, list) or len(params) != 8:
            raise ValueError(f"Object {object_id} group {group_id} has invalid motion params: {params}")
        if isinstance(links, int):
            link_ids = [links]
        elif isinstance(links, list):
            link_ids = links
        else:
            raise ValueError(f"Object {object_id} group {group_id} has invalid links payload: {links}")
        articulated_groups[str(group_id)] = (link_ids, str(parent_group_str), params, motion_type_str)

    if not articulated_groups:
        return None

    base_key = name
    components: dict[str, dict[str, Any]] = {}
    component_key_by_label: dict[int, str] = {}
    bbox_key_by_label: dict[int, str] = {}
    part_name_by_label: dict[int, str] = {}
    name_to_labels: dict[str, list[int]] = {}
    group_link_ids: dict[str, list[int]] = {}
    label_to_key = require_mapping(part_info_data.get("label_to_key"), "part_info['label_to_key']")

    for group_id, (link_ids, _parent_group_str, _params, _motion_type_str) in articulated_groups.items():
        group_link_ids[group_id] = link_ids
        for link_id in link_ids:
            part_info = parts_by_label.get(link_id)
            if part_info is None:
                raise ValueError(f"Object {object_id} group {group_id} references missing part label: {link_id}")
            if "name" not in part_info:
                raise ValueError(f"Object {object_id} part label {link_id} missing required field: name")
            part_name = part_info["name"]
            part_name_by_label[link_id] = part_name
            name_to_labels.setdefault(part_name, []).append(link_id)

    for part_name, labels in name_to_labels.items():
        labels.sort(key=label_sort_key)
        for index, label in enumerate(labels, start=1):
            bbox_key_by_label[label] = part_name if len(labels) == 1 else f"{part_name}_{index}"
            canonical_key = label_to_key.get(str(label))
            if not isinstance(canonical_key, str) or not canonical_key:
                raise ValueError(
                    f"Object {object_id} part label {label} ('{part_name}') is missing a non-empty canonical key in part_info['label_to_key']"
                )
            if canonical_key == base_key:
                raise ValueError(
                    f"Object {object_id} physical component key '{canonical_key}' conflicts with root key '{base_key}'"
                )
            if canonical_key in component_key_by_label.values():
                raise ValueError(f"Object {object_id} duplicate canonical component key: '{canonical_key}'")
            component_key_by_label[label] = canonical_key

    for group_id, (link_ids, parent_group_str, params, motion_type_str) in articulated_groups.items():
        for link_id in link_ids:
            part_name = part_name_by_label[link_id]
            component_key = component_key_by_label[link_id]

            if parent_group_str in fixed_group_ids:
                parent = base_key
            elif parent_group_str in group_link_ids:
                parent_first_label = group_link_ids[parent_group_str][0]
                parent = component_key_by_label[parent_first_label] if parent_first_label in component_key_by_label else None
            else:
                raise ValueError(f"Object {object_id} group {group_id} references unknown parent group: {parent_group_str}")

            pos = round_list(params[3:6])
            if motion_type_str == "C":
                comp = {
                    "item_name": part_name,
                    "parent": parent,
                    "children": [],
                    "pos": pos,
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "is_rotate": True,
                    "rotate_axis": round_list(params[0:3]),
                    "rotate_range": [round(params[6] * math.pi, 2), round(params[7] * math.pi, 2)],
                    "rotate_damp": 10,
                    "is_prismatic": False,
                    "prismatic_axis": None,
                    "prismatic_range": None,
                    "prismatic_damp": None,
                }
            elif motion_type_str == "A":
                comp = {
                    "item_name": part_name,
                    "parent": parent,
                    "children": [],
                    "pos": pos,
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "is_rotate": True,
                    "rotate_axis": round_list(params[0:3]),
                    "rotate_range": [-3.14, 3.14],
                    "rotate_damp": 10,
                    "is_prismatic": False,
                    "prismatic_axis": None,
                    "prismatic_range": None,
                    "prismatic_damp": None,
                }
            elif motion_type_str == "B":
                comp = {
                    "item_name": part_name,
                    "parent": parent,
                    "children": [],
                    "pos": pos,
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "is_rotate": False,
                    "rotate_axis": None,
                    "rotate_range": None,
                    "rotate_damp": None,
                    "is_prismatic": True,
                    "prismatic_axis": round_list(params[0:3]),
                    "prismatic_range": [round(params[6], 2), round(params[7], 2)],
                    "prismatic_damp": 10,
                }
            else:
                raise ValueError(f"Unexpected motion type: {motion_type_str}")

            comp["_link_id"] = link_id
            comp["_group_id"] = group_id
            comp["_bbox_key"] = bbox_key_by_label[link_id]
            if component_key in components:
                raise ValueError(f"Object {object_id} duplicate component key while building tree: '{component_key}'")
            components[component_key] = comp

    for comp_key, comp in components.items():
        if comp["parent"] is not None:
            continue
        group_id = comp["_group_id"]
        parent_group_str = articulated_groups[group_id][1]
        if parent_group_str not in group_link_ids:
            raise ValueError(f"Object {object_id} group {group_id} parent group not found: {parent_group_str}")
        parent_first_label = group_link_ids[parent_group_str][0]
        if parent_first_label not in component_key_by_label:
            raise ValueError(f"Object {object_id} group {group_id} parent label not resolved: {parent_first_label}")
        comp["parent"] = component_key_by_label[parent_first_label]

    base_children: list[str] = []
    for comp_key, comp in components.items():
        if comp["parent"] == base_key:
            base_children.append(comp_key)
    for comp_key, comp in components.items():
        parent_key = comp["parent"]
        if parent_key != base_key and parent_key in components:
            components[parent_key]["children"].append(comp_key)

    base_children.sort()
    for comp in components.values():
        comp["children"].sort()

    base_component = {
        "item_name": name,
        "parent": None,
        "children": base_children,
        "pos": [0.0, 0.0, 0.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "is_rotate": False,
        "rotate_axis": None,
        "rotate_range": None,
        "rotate_damp": None,
        "is_prismatic": False,
        "prismatic_axis": None,
        "prismatic_range": None,
        "prismatic_damp": None,
    }

    ordered_components = {base_key: base_component}
    bbox_key_by_component: dict[str, str] = {}
    sorted_articulated_keys = sorted(components.keys(), key=lambda key: components[key]["_link_id"])
    for comp_key in sorted_articulated_keys:
        if comp_key == base_key:
            raise ValueError(f"Object {object_id} physical component key '{comp_key}' conflicts with root key")
        comp = dict(components[comp_key])
        bbox_key_by_component[comp_key] = str(comp.pop("_bbox_key"))
        del comp["_link_id"]
        del comp["_group_id"]
        ordered_components[comp_key] = comp

    result = {
        "name": name,
        "category": category,
        "dimension": dimension,
        "components": ordered_components,
        "_bbox_key_by_component": bbox_key_by_component,
    }
    validate_component_tree(result["components"], context=f"Object {object_id} components tree", expected_root_key=base_key)
    return result


def format_bbox(bbox: list[int]) -> str:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox payload: {bbox}")
    x1, y1, x2, y2 = bbox
    for coord in (x1, y1, x2, y2):
        if not isinstance(coord, int):
            raise TypeError(f"Bbox coordinates must be integers, got: {bbox}")
        if coord < 0 or coord > 1000:
            raise ValueError(f"Bbox coordinates must be within 0..1000, got: {bbox}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Bbox must have positive area, got: {bbox}")
    return f"<|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"


def build_prompt_text(image_count: int) -> str:
    if image_count == 1:
        return SINGLE_IMAGE_PROMPT
    if image_count == 4:
        return FOUR_IMAGE_GROUP_PROMPT
    raise ValueError(f"Unsupported image count for prompt construction: {image_count}")


def resolve_remote_renders_root(data_root: str, image_prefix: str) -> str:
    if not image_prefix:
        raise ValueError("config.vlm.image_prefix must be a non-empty string")

    data_root_parts = Path(data_root).resolve().parts
    if "data" not in data_root_parts:
        raise ValueError(f"data_root must contain a 'data' path segment to resolve image paths: {data_root}")

    data_index = data_root_parts.index("data")
    relative_data_root = "/".join(data_root_parts[data_index:])
    return f"{image_prefix.rstrip('/')}/{relative_data_root}/renders"


def resolve_bbox_source_key(
    components_tree: dict[str, Any],
    parts_payload: dict[str, Any],
    comp_key: str,
    comp: dict[str, Any],
) -> str:
    if comp_key in parts_payload:
        return comp_key

    bbox_key_by_component = components_tree.get("_bbox_key_by_component", {})
    fallback_bbox_key = None
    if isinstance(bbox_key_by_component, dict):
        raw_fallback_bbox_key = bbox_key_by_component.get(comp_key)
        if isinstance(raw_fallback_bbox_key, str):
            fallback_bbox_key = raw_fallback_bbox_key

    item_name = comp.get("item_name")
    matching_keys = [key for key in parts_payload if isinstance(item_name, str) and key == item_name]
    if fallback_bbox_key in parts_payload:
        return fallback_bbox_key
    if len(matching_keys) == 1:
        return matching_keys[0]

    available_keys = sorted(parts_payload.keys())
    raise KeyError(f"Missing bbox payload for component '{comp_key}'. Available keys: {available_keys}")


def bbox_area(bbox: list[int]) -> int:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return 0
    x1, y1, x2, y2 = bbox
    if not all(isinstance(coord, int) for coord in (x1, y1, x2, y2)):
        return 0
    return max(0, x2 - x1) * max(0, y2 - y1)


def has_positive_bbox_area(bbox: list[int]) -> bool:
    return bbox_area(bbox) > 0


def build_answer_json(
    components_tree: dict[str, Any],
    selected_views: list[dict[str, Any]],
    bbox_gt: dict[str, Any],
) -> str:
    parts_payload = require_mapping(bbox_gt.get("parts"), "bbox_gt['parts']")

    result = {
        "name": components_tree["name"],
        "category": components_tree["category"],
        "dimension": components_tree["dimension"],
        "components": {},
    }

    for comp_key, comp in components_tree["components"].items():
        comp_copy = dict(comp)
        if comp.get("parent") is None:
            result["components"][comp_key] = comp_copy
            continue

        bbox_source_key = resolve_bbox_source_key(components_tree, parts_payload, comp_key, comp_copy)
        part_views = require_mapping(
            parts_payload[bbox_source_key].get("views"),
            f"bbox_gt['parts']['{bbox_source_key}']['views']",
        )

        bbox_dict: dict[str, str] = {}
        for view_slot, view in enumerate(selected_views, start=1):
            view_idx_str = str(view["view_index"])
            view_entry = require_mapping(
                part_views.get(view_idx_str),
                f"bbox_gt['parts']['{bbox_source_key}']['views']['{view_idx_str}']",
            )
            visible = view_entry.get("visible")
            if not isinstance(visible, bool):
                raise TypeError(
                    f"bbox_gt['parts']['{bbox_source_key}']['views']['{view_idx_str}']['visible'] must be a bool"
                )
            bbox = view_entry.get("bbox")

            if visible:
                if bbox is None:
                    raise ValueError(f"Visible bbox is missing for component {comp_key}, view {view_idx_str}")
                if not has_positive_bbox_area(bbox):
                    bbox_dict[f"image_{view_slot}"] = "not visible"
                    continue
                bbox_dict[f"image_{view_slot}"] = format_bbox(bbox)
            else:
                if bbox is not None:
                    raise ValueError(
                        f"Invisible view has non-null bbox for component {comp_key}, view {view_idx_str}"
                    )
                bbox_dict[f"image_{view_slot}"] = "not visible"

        comp_copy["bbox"] = bbox_dict
        result["components"][comp_key] = comp_copy

    validate_component_tree(
        result["components"],
        context=f"answer JSON for {components_tree['name']}",
        expected_root_key=components_tree["name"],
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def build_sample(sample_id: str, images: list[str], answer_json: str) -> dict[str, Any]:
    return {
        "id": sample_id,
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "user", "value": build_prompt_text(len(images))},
            {"from": "assistant", "value": answer_json},
        ],
        "images": images,
    }
