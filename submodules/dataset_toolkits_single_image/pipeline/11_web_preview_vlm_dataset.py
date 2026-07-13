#!/usr/bin/env python3
"""JSONL-first web preview for VLM training samples.

This preview is intentionally centered on the *actual* JSONL records used
for VLM training.  It does not silently fall back to all rendered views or all raw
parts: selected RGB views, target components, bbox fields, and masks are
derived from the JSONL sample plus the strict asset paths implied by that sample.
This VLM preview intentionally does not load voxel payloads.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import PipelineConfig, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_IMAGE_PATTERN = re.compile(
    r"/renders/(?P<object_id>[^/]+)/(?P<angle>angle_(?P<angle_idx>\d+))/(?P<subdir>.+)/(?P<filename>view_(?P<view_idx>\d+)\.png)$"
)
BBOX_PATTERN = re.compile(
    r"^<\|box_start\|>\((?P<x1>\d+),(?P<y1>\d+)\),\((?P<x2>\d+),(?P<y2>\d+)\)<\|box_end\|>$"
)
GROUP_ORDER = ("group_0", "group_1", "group_2", "group_3")
LEGACY_QUADRANT_VIEWS_PER_GROUP = 3
PART_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff",
]
THREE_VERSION = "0.160.0"
VENDOR_ROOT = REPO_ROOT / "vendor" / "three" / THREE_VERSION
VENDOR_ASSETS = {
    "three.module.js": VENDOR_ROOT / "three.module.js",
    "three/addons/controls/OrbitControls.js": VENDOR_ROOT / "three" / "addons" / "controls" / "OrbitControls.js",
}
CLASSIC_VENDOR_ASSETS = {
    "three.min.js": VENDOR_ROOT / "classic" / "three.min.js",
    "OrbitControls.js": VENDOR_ROOT / "classic" / "OrbitControls.js",
}


@dataclass(frozen=True)
class ImageRef:
    path: Path
    object_id: str
    angle_name: str
    angle_idx: int
    view_idx: int
    slot: int


@dataclass(frozen=True)
class SampleRecord:
    index: int
    source_name: str
    line_number: int
    sample_id: str
    object_id: str
    angle_name: str
    angle_idx: int
    view_indices: tuple[int, ...]
    image_paths: tuple[Path, ...]
    target_keys: tuple[str, ...]
    root_key: str
    assistant: dict[str, Any]
    raw_record: dict[str, Any]


@dataclass
class PreviewState:
    config: PipelineConfig
    jsonl_path: Path
    samples: list[SampleRecord]
    host: str
    port: int

    @property
    def data_root(self) -> Path:
        return Path(self.config.data_root)

    @property
    def renders_dir(self) -> Path:
        return Path(self.config.renders_dir)

    @property
    def reconstruction_dir(self) -> Path:
        return Path(self.config.reconstruction_dir)

    @property
    def voxel_resolution(self) -> str:
        return str(self.config.voxel.resolution)


class PreviewError(Exception):
    """User-visible preview/data error."""


# ---------------------------------------------------------------------------
# Argument / path helpers
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a JSONL-first web preview for VLM training samples."
    )
    parser.add_argument("--config", required=True, help="Path to the pipeline YAML config.")
    parser.add_argument(
        "--jsonl",
        help=(
            "VLM JSONL to preview. Defaults to the single-image "
            "vlm/training_json/arts_mllm_<dataset>_1img.jsonl."
        ),
    )
    parser.add_argument(
        "--object-ids",
        help="Optional comma-separated object IDs. Only matching JSONL samples are indexed.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host, default 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port, default 8765.")
    parser.add_argument(
        "--output-dir",
        help="Static preview output directory. Defaults to <data_root>/preview/vlm_training.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Debug mode: run the dynamic HTTP preview instead of generating static files.",
    )
    parser.add_argument(
        "--regen-voxels-only",
        action="store_true",
        help=(
            "Static mode: only regenerate decoded voxel payload JS files and refresh index.html. "
            "This skips RGB preview PNGs, sample detail JS, and index.js."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker count for --regen-voxels-only. Default 1.",
    )
    return parser.parse_args(argv)


def dataset_slug(dataset_name: str) -> str:
    return dataset_name.lower().replace(" ", "_")


def default_jsonl_path(config: PipelineConfig) -> Path:
    return Path(config.vlm_dir) / "training_json" / f"arts_mllm_{dataset_slug(config.dataset_name)}_1img.jsonl"


def parse_object_ids(raw_value: str) -> set[str]:
    object_ids = {item.strip() for item in raw_value.split(",") if item.strip()}
    if not object_ids:
        raise ValueError("--object-ids must contain at least one non-empty id")
    return object_ids


def normalize_training_image_path(image_path: str, data_root: Path) -> ImageRef:
    normalized = image_path.replace("\\", "/")
    match = TRAINING_IMAGE_PATTERN.search(normalized)
    if match is None:
        raise PreviewError(f"Training image path does not match renders/rgb/view_N.png: {image_path}")
    if not match.group("subdir").endswith("rgb"):
        raise PreviewError(f"Training image is not an RGB render: {image_path}")

    object_id = match.group("object_id")
    angle_name = match.group("angle")
    angle_idx = int(match.group("angle_idx"))
    view_idx = int(match.group("view_idx"))
    render_suffix = normalized.split("/renders/", 1)[1]
    local_path = data_root / "renders" / Path(render_suffix)
    return ImageRef(
        path=local_path,
        object_id=object_id,
        angle_name=angle_name,
        angle_idx=angle_idx,
        view_idx=view_idx,
        slot=-1,
    )


def validate_component_tree(components: dict[str, Any], expected_root_key: str | None = None) -> str:
    roots = [key for key, comp in components.items() if isinstance(comp, dict) and comp.get("parent") is None]
    if len(roots) != 1:
        raise PreviewError(f"assistant JSON must contain exactly one root, found {len(roots)}: {roots}")
    root_key = roots[0]
    if expected_root_key is not None and root_key != expected_root_key:
        raise PreviewError(f"assistant JSON root key '{root_key}' != name '{expected_root_key}'")

    for key, comp in components.items():
        if not isinstance(comp, dict):
            raise PreviewError(f"component '{key}' must be an object")
        parent = comp.get("parent")
        if parent == key:
            raise PreviewError(f"component '{key}' cannot parent itself")
        if parent is not None and parent not in components:
            raise PreviewError(f"component '{key}' references missing parent '{parent}'")
        children = comp.get("children")
        if not isinstance(children, list):
            raise PreviewError(f"component '{key}' children must be a list")
        for child in children:
            if child not in components:
                raise PreviewError(f"component '{key}' references missing child '{child}'")
            child_parent = components[child].get("parent")
            if child_parent != key:
                raise PreviewError(
                    f"component '{key}' lists child '{child}', but child parent is '{child_parent}'"
                )
    return root_key


def extract_assistant_payload(record: dict[str, Any]) -> dict[str, Any]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise PreviewError("record.conversations must be a list")
    if len(conversations) != 3:
        raise PreviewError(f"record.conversations must contain exactly 3 turns, found {len(conversations)}")
    assistant_turn = conversations[2]
    if not isinstance(assistant_turn, dict) or assistant_turn.get("from") != "assistant":
        raise PreviewError("record.conversations[2] must be the assistant JSON turn")
    value = assistant_turn.get("value")
    if not isinstance(value, str) or not value.strip():
        raise PreviewError("assistant turn value must be non-empty JSON text")
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise PreviewError("assistant JSON must be an object")
    return payload


def parse_bbox_value(value: Any) -> dict[str, Any]:
    if value == "not visible":
        return {"visible": False, "box": None}
    if not isinstance(value, str):
        raise PreviewError(f"bbox value must be a box string or 'not visible', got {value!r}")
    match = BBOX_PATTERN.match(value)
    if match is None:
        raise PreviewError(f"bbox string has unexpected format: {value}")
    x1, y1, x2, y2 = (int(match.group(name)) for name in ("x1", "y1", "x2", "y2"))
    if not all(0 <= value <= 1000 for value in (x1, y1, x2, y2)):
        raise PreviewError(f"bbox coordinates must be in Qwen/VLM 0-1000 space: {value}")
    if x2 <= x1 or y2 <= y1:
        raise PreviewError(f"bbox must have positive area: {value}")
    return {"visible": True, "box": [x1, y1, x2, y2]}


def validate_target_bbox(component_key: str, component: dict[str, Any], image_count: int) -> None:
    bbox = component.get("bbox")
    if not isinstance(bbox, dict):
        raise PreviewError(f"target component '{component_key}' missing bbox object")
    for slot in range(1, image_count + 1):
        key = f"image_{slot}"
        if key not in bbox:
            raise PreviewError(f"target component '{component_key}' missing bbox.{key}")
        parse_bbox_value(bbox[key])
    if not any(parse_bbox_value(bbox[f"image_{slot}"])["visible"] for slot in range(1, image_count + 1)):
        raise PreviewError(f"target component '{component_key}' is not visible in any selected image")


def parse_sample_record(
    payload: dict[str, Any],
    *,
    index: int,
    source_name: str,
    line_number: int,
    data_root: Path,
) -> SampleRecord:
    sample_id = payload.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise PreviewError(f"line {line_number}: record.id must be a non-empty string")
    images = payload.get("images")
    if not isinstance(images, list) or len(images) not in {1, 4}:
        raise PreviewError(
            f"sample {sample_id}: expected 1 or 4 images, got "
            f"{len(images) if isinstance(images, list) else type(images).__name__}"
        )
    image_count = len(images)

    image_refs = [normalize_training_image_path(image, data_root) for image in images]
    object_ids = {ref.object_id for ref in image_refs}
    angle_names = {ref.angle_name for ref in image_refs}
    if len(object_ids) != 1 or len(angle_names) != 1:
        raise PreviewError(f"sample {sample_id}: all images must belong to one object/angle")
    for slot, ref in enumerate(image_refs):
        if image_count == 4:
            expected_quadrant = slot
            actual_quadrant = ref.view_idx // LEGACY_QUADRANT_VIEWS_PER_GROUP
            if actual_quadrant != expected_quadrant:
                raise PreviewError(
                    f"sample {sample_id}: image_{slot + 1} uses view_{ref.view_idx}, "
                    f"expected quadrant {expected_quadrant}"
                )
        image_refs[slot] = ImageRef(
            path=ref.path,
            object_id=ref.object_id,
            angle_name=ref.angle_name,
            angle_idx=ref.angle_idx,
            view_idx=ref.view_idx,
            slot=slot,
        )

    assistant = extract_assistant_payload(payload)
    components = assistant.get("components")
    if not isinstance(components, dict):
        raise PreviewError(f"sample {sample_id}: assistant JSON missing components object")
    answer_name = assistant.get("name")
    root_key = validate_component_tree(components, answer_name if isinstance(answer_name, str) else None)
    target_keys = tuple(key for key, comp in components.items() if isinstance(comp, dict) and comp.get("parent") is not None)
    if not target_keys:
        raise PreviewError(f"sample {sample_id}: assistant JSON has no non-root target components")
    expected_id_suffix = f"_{image_refs[0].object_id}_{image_refs[0].angle_name}"
    expected_view_suffix = f"{expected_id_suffix}_view_{image_refs[0].view_idx}"
    if not (sample_id.endswith(expected_id_suffix) or sample_id.endswith(expected_view_suffix)):
        raise PreviewError(
            f"sample {sample_id}: id must end with '{expected_id_suffix}' or "
            f"'{expected_view_suffix}' to match image paths"
        )
    for key in target_keys:
        validate_target_bbox(key, components[key], image_count)

    return SampleRecord(
        index=index,
        source_name=source_name,
        line_number=line_number,
        sample_id=sample_id,
        object_id=image_refs[0].object_id,
        angle_name=image_refs[0].angle_name,
        angle_idx=image_refs[0].angle_idx,
        view_indices=tuple(ref.view_idx for ref in image_refs),
        image_paths=tuple(ref.path for ref in image_refs),
        target_keys=target_keys,
        root_key=root_key,
        assistant=assistant,
        raw_record=payload,
    )


def build_sample_index(
    jsonl_path: Path,
    data_root: Path,
    object_filter: set[str] | None = None,
) -> list[SampleRecord]:
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"VLM JSONL not found: {jsonl_path}")

    samples: list[SampleRecord] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreviewError(f"{jsonl_path}:{line_number}: invalid JSONL record: {exc}") from exc
            if not isinstance(payload, dict):
                raise PreviewError(f"{jsonl_path}:{line_number}: JSONL record must be an object")
            sample = parse_sample_record(
                payload,
                index=len(samples),
                source_name=jsonl_path.name,
                line_number=line_number,
                data_root=data_root,
            )
            if object_filter is not None and sample.object_id not in object_filter:
                continue
            samples.append(sample)
    if not samples:
        raise PreviewError(f"No training samples indexed from {jsonl_path}")
    return samples


# ---------------------------------------------------------------------------
# Data loading for detail view
# ---------------------------------------------------------------------------


def part_info_path(state: PreviewState, object_id: str) -> Path:
    return state.data_root / "part_info" / object_id / "part_info.json"


def mask_path(state: PreviewState, sample: SampleRecord, view_idx: int) -> Path:
    for image_path in sample.image_paths:
        if image_path.name != f"view_{view_idx}.png":
            continue
        if image_path.parent.name == "rgb" and image_path.parent.parent.name == "part_complete":
            return image_path.parent.parent / "mask" / "label" / f"mask_{view_idx}.npy"
    return state.renders_dir / sample.object_id / sample.angle_name / "mask" / f"mask_{view_idx}.npy"


def rgb_path_for_sample(sample: SampleRecord, slot: int) -> Path:
    if slot < 0 or slot >= len(sample.image_paths):
        raise PreviewError(f"RGB slot out of range: {slot}")
    return sample.image_paths[slot]


def surface_path(state: PreviewState, sample: SampleRecord) -> Path:
    return state.reconstruction_dir / "voxel_expanded" / sample.object_id / sample.angle_name / state.voxel_resolution / "surface.npy"


def part_voxel_path(state: PreviewState, sample: SampleRecord, component_key: str) -> Path:
    return state.reconstruction_dir / "voxel_expanded" / sample.object_id / sample.angle_name / state.voxel_resolution / f"ind_{component_key}.npy"


def decoded_latent_dir(state: PreviewState, sample: SampleRecord) -> Path:
    return state.reconstruction_dir / "ss_latent_decoded" / sample.object_id / sample.angle_name / state.voxel_resolution


def decoded_overall_path(state: PreviewState, sample: SampleRecord) -> Path:
    return decoded_latent_dir(state, sample) / "overall.npy"


def decoded_part_voxel_path(state: PreviewState, sample: SampleRecord, component_key: str) -> Path:
    return decoded_latent_dir(state, sample) / "parts" / f"{component_key}.npy"


def decoded_metrics_path(state: PreviewState, sample: SampleRecord) -> Path:
    return decoded_latent_dir(state, sample) / "metrics.json"


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PreviewError(f"Required JSON file missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise PreviewError(f"JSON file must contain an object: {path}")
    return data


def load_coords(path: Path) -> list[list[int]]:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise PreviewError(f"Voxel array must have shape (N, 3): {path} got {arr.shape}")
    return arr.astype(int).tolist()


def color_for_index(index: int) -> str:
    if index < len(PART_COLORS):
        return PART_COLORS[index]
    digest = hashlib.sha256(str(index).encode("utf-8")).hexdigest()
    return f"#{digest[:6]}"


def component_motion_summary(component: dict[str, Any]) -> str:
    if component.get("is_rotate"):
        return f"rotate axis={component.get('rotate_axis')} range={component.get('rotate_range')}"
    if component.get("is_prismatic"):
        return f"prismatic axis={component.get('prismatic_axis')} range={component.get('prismatic_range')}"
    return "static/unknown"


def validate_part_info(
    part_info: dict[str, Any],
    *,
    sample: SampleRecord,
    parts_payload: dict[str, Any],
) -> dict[str, int]:
    """Validate the mask-label contract and return component_key -> mask label.

    The renderer masks use positive labels, while part_info.label_to_key is keyed
    by raw zero-based labels.  We require the round trip:

        parts[key].label == raw_label + 1
        label_to_key[str(raw_label)] == key

    This prevents silently painting a target component with another part's mask.
    """
    object_id = part_info.get("object_id")
    if object_id != sample.object_id:
        raise PreviewError(
            f"part_info.object_id '{object_id}' does not match sample object '{sample.object_id}'"
        )

    num_parts = part_info.get("num_parts")
    if isinstance(num_parts, bool) or not isinstance(num_parts, int) or num_parts <= 0:
        raise PreviewError(f"part_info.num_parts must be a positive integer for {sample.object_id}")
    if num_parts != len(parts_payload):
        raise PreviewError(
            f"part_info.num_parts={num_parts} does not match parts count={len(parts_payload)} "
            f"for {sample.object_id}"
        )

    label_to_key_payload = part_info.get("label_to_key")
    if not isinstance(label_to_key_payload, dict):
        raise PreviewError(f"part_info.label_to_key missing or invalid for {sample.object_id}")

    label_to_key: dict[int, str] = {}
    for raw_label_text, key in label_to_key_payload.items():
        if not isinstance(raw_label_text, str) or not raw_label_text.isdigit():
            raise PreviewError(f"part_info.label_to_key key must be a numeric string: {raw_label_text!r}")
        if not isinstance(key, str) or not key:
            raise PreviewError(f"part_info.label_to_key[{raw_label_text!r}] must be a component key")
        raw_label = int(raw_label_text)
        if raw_label in label_to_key:
            raise PreviewError(f"duplicate raw label in part_info.label_to_key: {raw_label}")
        if key not in parts_payload:
            raise PreviewError(f"part_info.label_to_key[{raw_label}] references missing part '{key}'")
        label_to_key[raw_label] = key

    key_by_label: dict[int, str] = {}
    label_by_key: dict[str, int] = {}
    for key, payload in parts_payload.items():
        if not isinstance(payload, dict):
            raise PreviewError(f"part_info.parts['{key}'] must be an object")
        label = payload.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label <= 0 or label > num_parts:
            raise PreviewError(f"part_info label for '{key}' must be an integer in [1, {num_parts}]")
        if label in key_by_label:
            raise PreviewError(
                f"duplicate positive mask label {label}: '{key_by_label[label]}' and '{key}'"
            )

        raw_label = payload.get("raw_label", label - 1)
        if isinstance(raw_label, bool) or not isinstance(raw_label, int):
            raise PreviewError(f"part_info raw_label for '{key}' must be an integer")
        if raw_label != label - 1:
            raise PreviewError(
                f"part_info label/raw_label mismatch for '{key}': label={label}, raw_label={raw_label}"
            )
        mapped_key = label_to_key.get(raw_label)
        if mapped_key != key:
            raise PreviewError(
                f"part_info.label_to_key[{raw_label!r}]='{mapped_key}' does not map back to '{key}'"
            )
        key_by_label[label] = key
        label_by_key[key] = label

    if len(label_to_key) != len(parts_payload):
        raise PreviewError(
            f"part_info.label_to_key count={len(label_to_key)} does not match parts count={len(parts_payload)}"
        )
    return label_by_key


def build_sample_detail(state: PreviewState, sample_index: int) -> dict[str, Any]:
    if sample_index < 0 or sample_index >= len(state.samples):
        raise PreviewError(f"sample index out of range: {sample_index}")
    sample = state.samples[sample_index]
    assistant_components = sample.assistant["components"]
    missing_rgbs = [str(path) for path in sample.image_paths if not path.is_file()]
    if missing_rgbs:
        raise PreviewError(
            f"sample {sample.sample_id}: missing selected RGB image(s); no fallback allowed: {missing_rgbs}"
        )

    part_info = load_json_file(part_info_path(state, sample.object_id))
    parts_payload = part_info.get("parts")
    if not isinstance(parts_payload, dict):
        raise PreviewError(f"part_info.parts missing or invalid for {sample.object_id}")
    label_by_key = validate_part_info(part_info, sample=sample, parts_payload=parts_payload)

    validation_errors: list[str] = []
    masks: dict[str, Any] = {}
    labels_by_view: dict[int, set[int]] = {}
    label_counts_by_view: dict[int, dict[int, int]] = {}
    for view_idx in sample.view_indices:
        path = mask_path(state, sample, view_idx)
        if not path.is_file():
            validation_errors.append(f"missing mask npy for view_{view_idx}: {path}")
            continue
        mask = np.load(path)
        if mask.ndim != 2:
            raise PreviewError(f"mask must be 2D: {path} got {mask.shape}")
        unique_labels, unique_counts = np.unique(mask, return_counts=True)
        count_map = {
            int(label): int(count)
            for label, count in zip(unique_labels.tolist(), unique_counts.tolist())
            if int(label) != 0
        }
        labels = set(count_map)
        labels.discard(0)
        labels_by_view[view_idx] = labels
        label_counts_by_view[view_idx] = count_map
        mask_int = mask.astype(int)
        masks[str(view_idx)] = {
            "h": int(mask_int.shape[0]),
            "w": int(mask_int.shape[1]),
            "data": mask_int.ravel().tolist(),
            "path": str(path),
            "target_pixel_counts": count_map,
        }

    target_parts: list[dict[str, Any]] = []
    voxels: dict[str, Any] = {}
    surface = surface_path(state, sample)
    if surface.is_file():
        voxels["surface"] = load_coords(surface)
    else:
        validation_errors.append(f"missing surface voxel: {surface}")

    for part_index, component_key in enumerate(sample.target_keys):
        if component_key not in parts_payload:
            raise PreviewError(
                f"target component '{component_key}' from JSONL is missing in part_info.parts "
                f"for object {sample.object_id}; no fallback allowed"
            )
        part_payload = parts_payload[component_key]
        if not isinstance(part_payload, dict):
            raise PreviewError(f"part_info.parts['{component_key}'] must be an object")
        label = label_by_key[component_key]

        component = assistant_components[component_key]
        bbox_payload = component["bbox"]
        bboxes: dict[str, Any] = {}
        visible_slots: list[int] = []
        visible_views_from_bbox: list[int] = []
        visible_views_from_mask: list[int] = []
        mask_pixels_by_slot: dict[str, int] = {}
        mask_pixels_by_view: dict[str, int] = {}
        for slot, view_idx in enumerate(sample.view_indices, start=1):
            parsed_bbox = parse_bbox_value(bbox_payload[f"image_{slot}"])
            bboxes[str(slot)] = parsed_bbox
            pixel_count = label_counts_by_view.get(view_idx, {}).get(label, 0)
            mask_pixels_by_slot[str(slot)] = pixel_count
            mask_pixels_by_view[str(view_idx)] = pixel_count
            mask_has_label = pixel_count > 0
            if mask_has_label:
                visible_views_from_mask.append(view_idx)
            if parsed_bbox["visible"]:
                visible_slots.append(slot)
                visible_views_from_bbox.append(view_idx)
                if not mask_has_label:
                    validation_errors.append(
                        f"{component_key}: bbox says image_{slot}/view_{view_idx} visible, "
                        "but mask does not contain its label"
                    )
            elif mask_has_label:
                validation_errors.append(
                    f"{component_key}: mask contains label in image_{slot}/view_{view_idx}, "
                    "but JSONL bbox says not visible"
                )

        voxel_path = part_voxel_path(state, sample, component_key)
        has_voxel = voxel_path.is_file()
        voxel_count = 0
        if has_voxel:
            coords = load_coords(voxel_path)
            voxels[component_key] = coords
            voxel_count = len(coords)
            if voxel_count == 0:
                raise PreviewError(
                    f"empty target part voxel for {component_key}: {voxel_path}; no fallback allowed"
                )
        else:
            raise PreviewError(
                f"missing target part voxel for {component_key}: {voxel_path}; no fallback allowed"
            )

        target_parts.append({
            "key": component_key,
            "item_name": component.get("item_name", component_key),
            "label": label,
            "type": part_payload.get("type", ""),
            "joint": part_payload.get("joint", ""),
            "joint_type": part_payload.get("joint_type", ""),
            "parent": component.get("parent"),
            "children": component.get("children", []),
            "motion": component_motion_summary(component),
            "color": color_for_index(part_index),
            "bboxes": bboxes,
            "visible_slots": visible_slots,
            "visible_views_from_bbox": visible_views_from_bbox,
            "visible_views_from_mask": visible_views_from_mask,
            "mask_pixels_by_slot": mask_pixels_by_slot,
            "mask_pixels_by_view": mask_pixels_by_view,
            "has_voxel": has_voxel,
            "voxel_count": voxel_count,
            "voxel_path": str(voxel_path),
        })

    decoded_root = decoded_latent_dir(state, sample)
    decoded_metrics_file = decoded_metrics_path(state, sample)
    decoded_metrics: dict[str, Any] = {}
    if decoded_metrics_file.is_file():
        try:
            decoded_metrics = load_json_file(decoded_metrics_file)
        except Exception as exc:  # noqa: BLE001 - optional QC data should not block preview
            validation_errors.append(f"decoded latent metrics unreadable: {decoded_metrics_file}: {exc!r}")

    decoded_voxels: dict[str, list[list[int]]] = {}
    decoded_voxel_paths: dict[str, str] = {}
    decoded_overall_file = decoded_overall_path(state, sample)
    if decoded_overall_file.is_file():
        try:
            decoded_voxels["overall"] = load_coords(decoded_overall_file)
            decoded_voxel_paths["overall"] = str(decoded_overall_file)
        except Exception as exc:  # noqa: BLE001
            validation_errors.append(f"decoded overall voxel unreadable: {decoded_overall_file}: {exc!r}")

    for component_key in sample.target_keys:
        decoded_part_file = decoded_part_voxel_path(state, sample, component_key)
        if not decoded_part_file.is_file():
            continue
        try:
            decoded_voxels[component_key] = load_coords(decoded_part_file)
            decoded_voxel_paths[component_key] = str(decoded_part_file)
        except Exception as exc:  # noqa: BLE001
            validation_errors.append(f"decoded target voxel unreadable for {component_key}: {decoded_part_file}: {exc!r}")

    raw_record = json.loads(json.dumps(sample.raw_record, ensure_ascii=False))
    raw_record["images"] = [path.as_uri() for path in sample.image_paths]

    return {
        "sample": sample_summary(sample),
        "root_key": sample.root_key,
        "target_parts": target_parts,
        "masks": masks,
        "voxels": voxels,
        "decoded_latent_root": str(decoded_root),
        "decoded_voxels": decoded_voxels,
        "decoded_voxel_paths": decoded_voxel_paths,
        "decoded_metrics_path": str(decoded_metrics_file),
        "decoded_metrics": decoded_metrics,
        "view_summaries": [
            {
                "slot": slot + 1,
                "view": view_idx,
                "quadrant": slot,
                "quadrant_name": GROUP_ORDER[slot],
                "rgb_path": str(sample.image_paths[slot]),
                "visible_labels": sorted(labels_by_view.get(view_idx, set())),
            }
            for slot, view_idx in enumerate(sample.view_indices)
        ],
        "assistant": sample.assistant,
        "raw_record": raw_record,
        "voxel_resolution": state.voxel_resolution,
        "surface_voxel_path": str(surface),
        "validation_errors": validation_errors,
        "prev_index": sample.index - 1 if sample.index > 0 else None,
        "next_index": sample.index + 1 if sample.index + 1 < len(state.samples) else None,
    }


def sample_summary(sample: SampleRecord) -> dict[str, Any]:
    return {
        "index": sample.index,
        "source_name": sample.source_name,
        "line_number": sample.line_number,
        "sample_id": sample.sample_id,
        "object_id": sample.object_id,
        "angle": sample.angle_idx,
        "angle_name": sample.angle_name,
        "view_indices": list(sample.view_indices),
        "target_keys": list(sample.target_keys),
        "target_count": len(sample.target_keys),
    }


# ---------------------------------------------------------------------------
# Vendor assets
# ---------------------------------------------------------------------------


def load_vendor_asset(asset: str) -> bytes:
    path = VENDOR_ASSETS.get(asset)
    if path is None:
        raise FileNotFoundError(asset)
    if not path.is_file():
        raise FileNotFoundError(f"{asset}: expected vendored file at {path}")
    return path.read_bytes()


def validate_vendor_assets() -> None:
    missing: list[str] = []
    for asset in VENDOR_ASSETS:
        try:
            load_vendor_asset(asset)
        except Exception as exc:
            missing.append(f"{asset}: {exc}")
    if missing:
        raise PreviewError(
            f"Three.js vendor assets are unavailable under {VENDOR_ROOT}; "
            + " | ".join(missing)
        )


# ---------------------------------------------------------------------------
# Static site generation
# ---------------------------------------------------------------------------


def default_static_output_dir(config: PipelineConfig) -> Path:
    return Path(config.preview_dir) / "vlm_training"


def js_safe_json(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("</", "<\\/")


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_js_assignment(path: Path, prefix: str, payload: Any) -> None:
    write_text_atomic(path, f"{prefix}{js_safe_json(payload)};\n")


def sample_file_stem(sample: SampleRecord) -> str:
    return f"sample_{sample.index:06d}"


def static_voxel_assignment_prefix(sample: SampleRecord) -> str:
    return f"window.__VOXEL_PAYLOADS=window.__VOXEL_PAYLOADS||{{}};window.__VOXEL_PAYLOADS[{sample.index}]="


def static_decoded_voxel_assignment_prefix(sample: SampleRecord) -> str:
    return (
        "window.__DECODED_VOXEL_PAYLOADS=window.__DECODED_VOXEL_PAYLOADS||{};"
        f"window.__DECODED_VOXEL_PAYLOADS[{sample.index}]="
    )


def write_static_voxel_payload(output_dir: Path, sample: SampleRecord, voxel_payload: dict[str, Any]) -> None:
    stem = sample_file_stem(sample)
    write_js_assignment(
        output_dir / "voxels" / f"{stem}.js",
        static_voxel_assignment_prefix(sample),
        voxel_payload,
    )


def static_voxel_payload_path(output_dir: Path, sample: SampleRecord) -> Path:
    return output_dir / "voxels" / f"{sample_file_stem(sample)}.js"


def static_decoded_voxel_payload_path(output_dir: Path, sample: SampleRecord) -> Path:
    return output_dir / "decoded_voxels" / f"{sample_file_stem(sample)}.js"


def write_static_decoded_voxel_payload(
    output_dir: Path,
    sample: SampleRecord,
    decoded_payload: dict[str, Any],
) -> None:
    write_js_assignment(
        static_decoded_voxel_payload_path(output_dir, sample),
        static_decoded_voxel_assignment_prefix(sample),
        decoded_payload,
    )


def static_decoded_voxel_payload_is_current(path: Path, sample: SampleRecord) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    prefix = static_decoded_voxel_assignment_prefix(sample)
    stripped = text.strip()
    return (
        stripped.startswith(prefix)
        and stripped.endswith(";")
        and '"decoded_overall"' in stripped
        and '"decoded_target_voxels"' in stripped
        and '"decoded_metrics"' in stripped
    )


def read_js_assignment_payload(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    marker = "]="
    marker_index = text.find(marker)
    if marker_index < 0:
        raise PreviewError(f"JS assignment marker not found in {path}")
    payload_text = text[marker_index + len(marker):].strip()
    if payload_text.endswith(";"):
        payload_text = payload_text[:-1]
    return json.loads(payload_text)


def load_existing_static_voxel_payload(path: Path, sample: SampleRecord) -> dict[str, Any]:
    payload = read_js_assignment_payload(path)
    if not isinstance(payload, dict):
        raise PreviewError(f"existing voxel payload is not an object: {path}")
    if payload.get("sample_index") != sample.index:
        raise PreviewError(
            f"existing voxel payload sample_index mismatch for {path}: "
            f"{payload.get('sample_index')!r} != {sample.index}"
        )
    for key in ("sample_id", "voxel_resolution", "surface_path", "fatal_errors", "surface", "target_voxels"):
        if key not in payload:
            raise PreviewError(f"existing voxel payload missing {key}: {path}")
    return payload


def refresh_static_voxel_payload(
    *,
    state: PreviewState,
    sample: SampleRecord,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, str | None, list[str], str, str | None]:
    payload_path = static_voxel_payload_path(output_dir, sample)
    if payload_path.is_file():
        try:
            existing_text = payload_path.read_text(encoding="utf-8")
            prefix = static_voxel_assignment_prefix(sample)
            stripped_text = existing_text.strip()
            if not stripped_text.startswith(prefix):
                raise PreviewError(f"existing voxel payload has unexpected assignment prefix: {payload_path}")
            if not stripped_text.endswith(";"):
                raise PreviewError(f"existing voxel payload is not terminated by semicolon: {payload_path}")
            body = stripped_text[len(prefix):-1].strip()
            if not body.endswith("}"):
                raise PreviewError(f"existing voxel payload body is not a JSON object: {payload_path}")
            if '"decoded_overall":' in body and '"decoded_target_voxels":' in body and '"decoded_metrics":' in body:
                return None, None, [], "skipped_existing_decoded", None
            decoded_payload, validation_errors = build_static_decoded_voxel_payload(
                state=state,
                sample=sample,
            )
            decoded_json = js_safe_json(decoded_payload)
            patched_body = body[:-1] + "," + decoded_json[1:-1] + "}"
            return None, f"{prefix}{patched_body};\n", validation_errors, "patched_existing_base", None
        except Exception as exc:  # noqa: BLE001 - corrupt or stale static payload falls back to source voxels
            fallback_reason = f"{payload_path}: {exc!r}"
    else:
        fallback_reason = f"{payload_path}: missing existing voxel payload"

    payload, validation_errors = build_static_voxel_payload(state=state, sample=sample)
    return payload, None, validation_errors, "rebuilt_full", fallback_reason


def copy_classic_vendor_assets(output_dir: Path) -> None:
    vendor_dir = output_dir / "vendor"
    vendor_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for asset_name, src in CLASSIC_VENDOR_ASSETS.items():
        if not src.is_file():
            missing.append(f"{asset_name}: expected vendored file at {src}")
            continue
        shutil.copy2(src, vendor_dir / asset_name)
    if missing:
        raise PreviewError(f"Missing classic Three.js vendor assets for static preview: {missing}")


def rgba_from_hex(color: str, alpha: int) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha)


def scale_bbox(box: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    sx1 = max(0, min(width - 1, round(x1 / 1000 * width)))
    sy1 = max(0, min(height - 1, round(y1 / 1000 * height)))
    sx2 = max(0, min(width - 1, round(x2 / 1000 * width)))
    sy2 = max(0, min(height - 1, round(y2 / 1000 * height)))
    if sx2 <= sx1:
        sx2 = min(width - 1, sx1 + 1)
    if sy2 <= sy1:
        sy2 = min(height - 1, sy1 + 1)
    return (sx1, sy1, sx2, sy2)


def generate_overlay_png(
    *,
    rgb_path: Path,
    mask: np.ndarray,
    target_parts: list[dict[str, Any]],
    slot: int,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(rgb_path).convert("RGBA")
    width, height = image.size
    if mask.shape[0] != height or mask.shape[1] != width:
        raise PreviewError(
            f"mask shape {mask.shape} does not match RGB size {(width, height)} for {rgb_path}"
        )

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay_arr = np.array(overlay)
    for part in target_parts:
        color = rgba_from_hex(part["color"], 92)
        overlay_arr[mask == part["label"]] = color
    overlay = Image.fromarray(overlay_arr, "RGBA")
    image = Image.alpha_composite(image, overlay)

    draw = ImageDraw.Draw(image)
    for part in target_parts:
        bbox = part["bboxes"][str(slot)]
        if not bbox["visible"]:
            continue
        x1, y1, x2, y2 = scale_bbox(bbox["box"], width, height)
        color = part["color"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label_y = max(0, y1 - 14)
        draw.rectangle((x1, label_y, min(width, x1 + 8 + len(part["key"]) * 7), label_y + 13), fill=color)
        draw.text((x1 + 3, label_y + 1), part["key"], fill=(0, 0, 0, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def generate_bbox_preview_png(
    *,
    rgb_path: Path,
    target_parts: list[dict[str, Any]],
    slot: int,
    output_path: Path,
) -> None:
    """Draw JSONL bbox targets onto the RGB for preview only.

    The training image path remains the original RGB in the JSONL; this derived
    PNG is only for human inspection of the bbox annotations.
    """
    from PIL import Image, ImageDraw

    image = Image.open(rgb_path).convert("RGBA")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for part in target_parts:
        bbox = part["bboxes"][str(slot)]
        if not bbox["visible"]:
            continue
        x1, y1, x2, y2 = scale_bbox(bbox["box"], width, height)
        color = part["color"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label = f'{part["key"]}  L{part["label"]}'
        label_width = min(width, x1 + 10 + len(label) * 8)
        label_y = max(0, y1 - 18)
        draw.rectangle((x1, label_y, label_width, label_y + 17), fill=color)
        draw.text((x1 + 4, label_y + 2), label, fill=(0, 0, 0, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_static_voxel_payload(
    *,
    state: PreviewState,
    sample: SampleRecord,
    target_voxels: dict[str, list[list[int]]] | None = None,
    target_voxel_paths: dict[str, str] | None = None,
    fatal_voxel_errors: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    validation_errors: list[str] = []

    if target_voxels is None or target_voxel_paths is None or fatal_voxel_errors is None:
        target_voxels = {}
        target_voxel_paths = {}
        fatal_voxel_errors = []
        for component_key in sample.target_keys:
            voxel_path = part_voxel_path(state, sample, component_key)
            if not voxel_path.is_file():
                fatal_voxel_errors.append(
                    f"FATAL VOXEL: missing target part voxel for {component_key}: {voxel_path}; no fallback allowed"
                )
                continue
            coords = load_coords(voxel_path)
            if not coords:
                fatal_voxel_errors.append(
                    f"FATAL VOXEL: empty target part voxel for {component_key}: {voxel_path}; no fallback allowed"
                )
                continue
            target_voxels[component_key] = coords
            target_voxel_paths[component_key] = str(voxel_path)
    else:
        target_voxels = dict(target_voxels)
        target_voxel_paths = dict(target_voxel_paths)
        fatal_voxel_errors = list(fatal_voxel_errors)

    surface = surface_path(state, sample)
    surface_coords: list[list[int]] = []
    if fatal_voxel_errors:
        validation_errors.extend(fatal_voxel_errors)
        target_voxels = {}
        target_voxel_paths = {}
    elif surface.is_file():
        surface_coords = load_coords(surface)
    else:
        validation_errors.append(f"missing surface voxel: {surface}")

    decoded_payload, decoded_validation_errors = build_static_decoded_voxel_payload(state=state, sample=sample)
    validation_errors.extend(decoded_validation_errors)

    voxel_payload = {
        "sample_index": sample.index,
        "sample_id": sample.sample_id,
        "voxel_resolution": state.voxel_resolution,
        "surface_path": str(surface),
        "fatal_errors": fatal_voxel_errors,
        "surface": surface_coords,
        "target_voxel_paths": target_voxel_paths,
        "target_voxels": target_voxels,
    }
    voxel_payload.update(decoded_payload)
    return voxel_payload, validation_errors


def build_static_decoded_voxel_payload(
    *,
    state: PreviewState,
    sample: SampleRecord,
) -> tuple[dict[str, Any], list[str]]:
    validation_errors: list[str] = []
    decoded_root = decoded_latent_dir(state, sample)
    decoded_metrics_file = decoded_metrics_path(state, sample)
    decoded_metrics: dict[str, Any] = {}
    if decoded_metrics_file.is_file():
        try:
            decoded_metrics = load_json_file(decoded_metrics_file)
        except Exception as exc:  # noqa: BLE001 - preview should report optional QC data problems
            validation_errors.append(f"decoded latent metrics unreadable: {decoded_metrics_file}: {exc!r}")

    decoded_overall_file = decoded_overall_path(state, sample)
    decoded_overall_coords: list[list[int]] = []
    if decoded_overall_file.is_file():
        try:
            decoded_overall_coords = load_coords(decoded_overall_file)
        except Exception as exc:  # noqa: BLE001
            validation_errors.append(f"decoded overall voxel unreadable: {decoded_overall_file}: {exc!r}")

    decoded_target_voxels: dict[str, list[list[int]]] = {}
    decoded_target_voxel_paths: dict[str, str] = {}
    for component_key in sample.target_keys:
        decoded_part_file = decoded_part_voxel_path(state, sample, component_key)
        if not decoded_part_file.is_file():
            continue
        try:
            decoded_target_voxels[component_key] = load_coords(decoded_part_file)
            decoded_target_voxel_paths[component_key] = str(decoded_part_file)
        except Exception as exc:  # noqa: BLE001
            validation_errors.append(f"decoded target voxel unreadable for {component_key}: {decoded_part_file}: {exc!r}")

    return {
        "decoded_latent_root": str(decoded_root),
        "decoded_overall_path": str(decoded_overall_file),
        "decoded_overall": decoded_overall_coords,
        "decoded_target_voxel_paths": decoded_target_voxel_paths,
        "decoded_target_voxels": decoded_target_voxels,
        "decoded_metrics_path": str(decoded_metrics_file),
        "decoded_metrics": decoded_metrics,
    }, validation_errors


def build_static_sample_assets(
    *,
    state: PreviewState,
    sample: SampleRecord,
    output_dir: Path,
) -> dict[str, Any]:
    assistant_components = sample.assistant["components"]
    missing_rgbs = [str(path) for path in sample.image_paths if not path.is_file()]
    if missing_rgbs:
        raise PreviewError(
            f"sample {sample.sample_id}: missing selected RGB image(s); no fallback allowed: {missing_rgbs}"
        )

    part_info = load_json_file(part_info_path(state, sample.object_id))
    parts_payload = part_info.get("parts")
    if not isinstance(parts_payload, dict):
        raise PreviewError(f"part_info.parts missing or invalid for {sample.object_id}")
    label_by_key = validate_part_info(part_info, sample=sample, parts_payload=parts_payload)

    validation_errors: list[str] = []
    masks_by_view: dict[int, np.ndarray] = {}
    labels_by_view: dict[int, set[int]] = {}
    label_counts_by_view: dict[int, dict[int, int]] = {}
    for view_idx in sample.view_indices:
        path = mask_path(state, sample, view_idx)
        if not path.is_file():
            raise PreviewError(f"missing mask npy for view_{view_idx}: {path}; no fallback allowed")
        mask = np.load(path)
        if mask.ndim != 2:
            raise PreviewError(f"mask must be 2D: {path} got {mask.shape}")
        unique_labels, unique_counts = np.unique(mask, return_counts=True)
        count_map = {
            int(label): int(count)
            for label, count in zip(unique_labels.tolist(), unique_counts.tolist())
            if int(label) != 0
        }
        labels_by_view[view_idx] = set(count_map)
        label_counts_by_view[view_idx] = count_map
        masks_by_view[view_idx] = mask.astype(int)

    target_parts: list[dict[str, Any]] = []
    for part_index, component_key in enumerate(sample.target_keys):
        if component_key not in parts_payload:
            raise PreviewError(
                f"target component '{component_key}' from JSONL is missing in part_info.parts "
                f"for object {sample.object_id}; no fallback allowed"
            )
        part_payload = parts_payload[component_key]
        if not isinstance(part_payload, dict):
            raise PreviewError(f"part_info.parts['{component_key}'] must be an object")
        label = label_by_key[component_key]
        component = assistant_components[component_key]

        bboxes: dict[str, Any] = {}
        visible_slots: list[int] = []
        visible_views_from_bbox: list[int] = []
        visible_views_from_mask: list[int] = []
        mask_pixels_by_slot: dict[str, int] = {}
        mask_pixels_by_view: dict[str, int] = {}
        bbox_payload = component["bbox"]
        for slot, view_idx in enumerate(sample.view_indices, start=1):
            parsed_bbox = parse_bbox_value(bbox_payload[f"image_{slot}"])
            bboxes[str(slot)] = parsed_bbox
            pixel_count = label_counts_by_view.get(view_idx, {}).get(label, 0)
            mask_pixels_by_slot[str(slot)] = pixel_count
            mask_pixels_by_view[str(view_idx)] = pixel_count
            mask_has_label = pixel_count > 0
            if mask_has_label:
                visible_views_from_mask.append(view_idx)
            if parsed_bbox["visible"]:
                visible_slots.append(slot)
                visible_views_from_bbox.append(view_idx)
                if not mask_has_label:
                    validation_errors.append(
                        f"{component_key}: bbox says image_{slot}/view_{view_idx} visible, "
                        "but mask does not contain its label"
                    )
            elif mask_has_label:
                validation_errors.append(
                    f"{component_key}: mask contains label in image_{slot}/view_{view_idx}, "
                    "but JSONL bbox says not visible"
                )

        target_parts.append({
            "key": component_key,
            "item_name": component.get("item_name", component_key),
            "label": label,
            "type": part_payload.get("type", ""),
            "joint": part_payload.get("joint", ""),
            "joint_type": part_payload.get("joint_type", ""),
            "parent": component.get("parent"),
            "children": component.get("children", []),
            "motion": component_motion_summary(component),
            "color": color_for_index(part_index),
            "bboxes": bboxes,
            "visible_slots": visible_slots,
            "visible_views_from_bbox": visible_views_from_bbox,
            "visible_views_from_mask": visible_views_from_mask,
            "mask_pixels_by_slot": mask_pixels_by_slot,
            "mask_pixels_by_view": mask_pixels_by_view,
        })

    preview_images: list[str] = []
    for slot, view_idx in enumerate(sample.view_indices, start=1):
        rel_path = Path("previews") / f"{sample_file_stem(sample)}_image_{slot}_view_{view_idx}_bbox.png"
        generate_bbox_preview_png(
            rgb_path=sample.image_paths[slot - 1],
            target_parts=target_parts,
            slot=slot,
            output_path=output_dir / rel_path,
        )
        preview_images.append(rel_path.as_posix())

    raw_record = json.loads(json.dumps(sample.raw_record, ensure_ascii=False))
    detail_payload = {
        "sample": sample_summary(sample),
        "root_key": sample.root_key,
        "target_parts": target_parts,
        "view_summaries": [
            {
                "slot": slot + 1,
                "view": view_idx,
                "quadrant": slot,
                "quadrant_name": GROUP_ORDER[slot],
                "rgb_path": str(sample.image_paths[slot]),
                "preview_src": preview_images[slot],
                "visible_labels": sorted(labels_by_view.get(view_idx, set())),
            }
            for slot, view_idx in enumerate(sample.view_indices)
        ],
        "preview_images": preview_images,
        "assistant": sample.assistant,
        "raw_record": raw_record,
        "validation_errors": validation_errors,
        "prev_index": sample.index - 1 if sample.index > 0 else None,
        "next_index": sample.index + 1 if sample.index + 1 < len(state.samples) else None,
    }
    return detail_payload


STATIC_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARTS-Qwen VLM Single-view Preview</title>
<style>
:root{--bg:#07090d;--ink:#eef5ff;--muted:#8fa0b7;--panel:#101821;--panel2:#151f2a;--line:#263447;--line2:#3b526d;--cyan:#39d2c0;--amber:#f4c95d;--danger:#ff6b83;--radius:16px;--shadow:0 18px 48px rgba(0,0,0,.36)}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;overflow:hidden;color:var(--ink);font-family:"Noto Sans SC","Avenir Next","Segoe UI",system-ui,sans-serif;background:radial-gradient(circle at 18% 8%,rgba(57,210,192,.14),transparent 30%),radial-gradient(circle at 86% 18%,rgba(244,201,93,.12),transparent 28%),linear-gradient(135deg,#06080c,#0a1017 48%,#07090d);font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}button,input{font:inherit}code,pre{font-family:"JetBrains Mono","SFMono-Regular",Menlo,Consolas,monospace}.top{height:68px;display:flex;align-items:center;justify-content:space-between;gap:18px;padding:10px 18px;border-bottom:1px solid rgba(255,255,255,.08);background:rgba(8,13,19,.86);backdrop-filter:blur(18px);box-shadow:0 10px 28px rgba(0,0,0,.26);position:relative;z-index:5}.brand{min-width:0}.brand h1{margin:0;font-size:20px;letter-spacing:.02em;line-height:1.1}.brand p{margin:5px 0 0;color:var(--muted);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.stats{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}.stat{padding:6px 10px;border:1px solid rgba(57,210,192,.36);border-radius:999px;background:rgba(18,29,40,.88);color:var(--muted);font-size:12px;white-space:nowrap}.stat b{color:var(--amber);font-size:14px}.path{position:absolute;left:18px;right:18px;bottom:-18px;color:#617188;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;opacity:.78;pointer-events:none}.dashboard{height:calc(100vh - 68px);display:grid;grid-template-columns:minmax(300px,19vw) minmax(720px,1fr) minmax(430px,.46fr);gap:14px;padding:18px;min-height:0}.pane{min-width:0;min-height:0;overflow:hidden;display:flex;flex-direction:column;background:linear-gradient(180deg,rgba(17,26,36,.96),rgba(10,16,24,.96));border:1px solid rgba(255,255,255,.09);border-radius:var(--radius);box-shadow:var(--shadow)}.pane-title{height:46px;flex:0 0 auto;display:flex;align-items:center;justify-content:space-between;gap:10px;padding:0 14px;border-bottom:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.025);color:var(--amber);font-weight:850;font-size:14px}.pane-title span:last-child{color:var(--muted);font-size:11px;font-weight:600}.controls{padding:12px;border-bottom:1px solid rgba(255,255,255,.08);background:rgba(5,9,14,.24)}.search{display:flex;gap:8px}.search input{min-width:0;flex:1;height:38px;padding:0 11px;border-radius:11px;border:1px solid var(--line2);background:#111c27;color:var(--ink);outline:none}.search input:focus{border-color:var(--cyan);box-shadow:0 0 0 3px rgba(57,210,192,.14)}.btn,.mini-btn,.pager button,.tabs button{border:1px solid rgba(57,210,192,.66);background:var(--cyan);color:#061014;border-radius:10px;min-height:36px;padding:0 11px;font-weight:850;cursor:pointer;transition:transform .13s ease,filter .13s ease,border-color .13s ease,color .13s ease,background .13s ease}.btn:hover,.mini-btn:hover,.pager button:hover,.tabs button:hover{filter:brightness(1.06)}.btn:active,.mini-btn:active,.pager button:active,.tabs button:active{transform:translateY(1px) scale(.98)}.sample-list{min-height:0;overflow:auto;padding:12px;scrollbar-color:#3f5773 #0c141e}.pager{position:sticky;top:0;z-index:3;margin-bottom:10px;padding:10px;border:1px solid rgba(255,255,255,.1);border-radius:14px;background:rgba(13,20,29,.94);backdrop-filter:blur(12px);box-shadow:0 10px 24px rgba(0,0,0,.24)}.pager-row{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}.pager-row:last-child{margin-bottom:0}.pager .range{color:var(--amber);font-weight:850}.pager small{color:var(--muted);font-size:11px}.pager input{width:72px;height:32px;border:1px solid var(--line2);background:#111c27;color:var(--ink);border-radius:8px;padding:0 8px}.pager button[disabled]{opacity:.44;cursor:not-allowed;filter:none}.card{width:100%;text-align:left;border:1px solid rgba(255,255,255,.09);border-radius:14px;background:linear-gradient(180deg,#141f2b,#101822);color:var(--ink);padding:9px;margin-bottom:9px;cursor:pointer;display:grid;grid-template-columns:74px minmax(0,1fr);gap:10px;align-items:center;box-shadow:0 8px 20px rgba(0,0,0,.18);transition:border-color .14s ease,background .14s ease,transform .14s ease}.card:hover,.card.active{border-color:var(--cyan);background:#172638}.card:active{transform:scale(.992)}.thumb{aspect-ratio:1/1;border-radius:10px;border:1px solid #31445b;overflow:hidden;background-color:#efe6d2;background-image:linear-gradient(45deg,#d8d0bf 25%,transparent 25%),linear-gradient(-45deg,#d8d0bf 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#d8d0bf 75%),linear-gradient(-45deg,transparent 75%,#d8d0bf 75%);background-position:0 0,0 11px,11px -11px,-11px 0;background-size:22px 22px}.thumb img{width:100%;height:100%;object-fit:contain;display:block}.thumb.missing-thumb{display:grid;place-items:center;color:#64748b;font-size:11px;background:#09111a}.card-main{min-width:0}.card-main b{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#f5d987;font-size:14px}.card-main small{display:block;color:var(--muted);font-size:11.5px;line-height:1.35;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.image-pane{background:linear-gradient(180deg,rgba(10,14,20,.92),rgba(5,8,12,.94))}.image-stage{min-height:0;flex:1;padding:16px;display:grid;grid-template-rows:auto minmax(0,1fr);gap:12px}.view-meta{display:flex;align-items:center;justify-content:space-between;gap:12px;color:var(--muted);font-size:13px}.view-meta b{color:var(--amber)}.view-meta a{color:var(--cyan);text-decoration:none;font-weight:800}.hero-frame{min-height:0;display:flex;align-items:center;justify-content:center;border:1px solid rgba(255,255,255,.1);border-radius:18px;overflow:hidden;background-color:#f7f0dc;background-image:linear-gradient(45deg,#e3dac8 25%,transparent 25%),linear-gradient(-45deg,#e3dac8 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#e3dac8 75%),linear-gradient(-45deg,transparent 75%,#e3dac8 75%);background-position:0 0,0 16px,16px -16px,-16px 0;background-size:32px 32px;box-shadow:inset 0 0 0 1px rgba(0,0,0,.16),0 20px 50px rgba(0,0,0,.28)}.hero-frame img{display:block;width:100%;height:100%;max-width:100%;max-height:100%;object-fit:contain}.empty{padding:28px;color:#697a91;text-align:center;line-height:1.55}.json-pane{background:linear-gradient(180deg,rgba(12,19,28,.96),rgba(7,12,18,.98))}.pane-actions{display:flex;align-items:center;gap:8px}.mini-btn{background:#162130;color:var(--ink);border-color:#40546d;min-height:30px;border-radius:8px}.mini-btn:hover{color:var(--cyan);border-color:var(--cyan);filter:none}.json-body{min-height:0;overflow:auto;padding:14px;scrollbar-color:#3f5773 #0b111a}.sample-title{margin-bottom:12px}.sample-title h2{margin:0;font-size:20px;line-height:1.2;text-wrap:balance}.sample-title .meta{color:var(--muted);font-size:13px;margin-top:6px;line-height:1.45}.tabs{display:flex;gap:8px;position:sticky;top:0;z-index:2;padding-bottom:10px;background:linear-gradient(180deg,#0b121b 74%,rgba(11,18,27,0))}.tabs button{background:#172231;color:var(--muted);border-color:#42566e}.tabs button.active{background:var(--cyan);color:#061014;border-color:var(--cyan)}.parts{display:grid;grid-template-columns:1fr;gap:9px;margin:8px 0 10px}.part{border:1px solid #33475c;background:#14202d;border-radius:11px;padding:10px;font-size:13px;line-height:1.45}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;vertical-align:middle}.muted{color:var(--muted)}.error{border:1px solid rgba(255,107,131,.58);background:rgba(255,107,131,.12);color:#ffc6cf;padding:10px;border-radius:12px;margin-bottom:12px;font-size:13px;line-height:1.45}pre{margin:0;white-space:pre;overflow:auto;max-width:100%;background:#060b12;border:1px solid #273a51;border-radius:13px;padding:13px;font-size:13px;line-height:1.55;color:#dbe8f7;box-shadow:inset 0 1px 0 rgba(255,255,255,.035)}body.json-focus .dashboard{grid-template-columns:minmax(460px,.72fr) minmax(760px,1fr);grid-template-rows:minmax(0,1fr)}body.json-focus .sample-pane{display:none}body.json-focus .image-pane{grid-column:1;grid-row:1}body.json-focus .json-pane{grid-column:2;grid-row:1}body.json-focus pre{font-size:17px;line-height:1.62}@media(max-width:1420px){.dashboard{grid-template-columns:minmax(280px,22vw) minmax(520px,1fr) minmax(380px,.56fr);gap:12px;padding:14px}.card{grid-template-columns:62px minmax(0,1fr)}.sample-title h2{font-size:18px}pre{font-size:12.5px}}@media(max-width:960px){body{overflow:auto}.top{position:sticky;top:0}.dashboard{display:block;height:auto;padding:12px}.pane{height:auto;min-height:420px;margin-bottom:12px}.sample-pane{height:520px}.image-pane{height:760px}.json-pane{height:680px}.hero-frame{min-height:620px}.stats{justify-content:flex-start}.top{height:auto;align-items:flex-start;flex-direction:column}.path{position:static;width:100%;margin-top:4px}.brand p{white-space:normal}.dashboard{height:auto}}
</style>
</head>
<body>
<header class="top">
  <div class="brand"><h1>VLM 单视角 RGB + bbox 预览</h1><p>每条训练样本只展示 1 张 part_complete RGB，并在预览层叠加 JSONL bbox；训练图片路径不改。</p></div>
  <div class="stats" id="stats"></div>
  <div class="path" id="path"></div>
</header>
<main class="dashboard">
  <section class="pane sample-pane">
    <div class="pane-title"><span>样本 / 物体</span><span>单图样本列表</span></div>
    <div class="controls"><div class="search"><input id="q" placeholder="搜索 sample_id / object_id / component"><button class="btn" id="open-first">打开首个</button></div></div>
    <aside class="sample-list" id="list"></aside>
  </section>
  <section class="pane image-pane">
    <div class="pane-title"><span>单视角 RGB + bbox</span><span>part_complete 前 8 固定视角，1 view = 1 sample</span></div>
    <section class="image-stage" id="views"><div class="empty">选择一个训练样本</div></section>
  </section>
  <section class="pane json-pane">
    <div class="pane-title"><span>JSONL / 标注</span><span class="pane-actions"><button class="mini-btn" id="json-focus-btn" type="button">放大 JSON</button><span>assistant / full / parts</span></span></div>
    <div class="json-body" id="detail"><div class="empty">选择样本后显示 assistant JSON、完整 record 和目标部件可见性。</div></div>
  </section>
</main>
<script src="data/index.js"></script>
<script>
window.__SAMPLE_DETAILS=window.__SAMPLE_DETAILS||{};
const samples=window.PREVIEW_INDEX.samples||[];let filtered=samples;let current=null;let jsonTab='assistant';let activeScript=new Set();let page=0;const pageSize=60;
function esc(s){return String(s??'').replace(/[&<>'\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function norm(s){return String(s||'').toLowerCase().trim();}
function scriptOnce(src){return new Promise((res,rej)=>{if(activeScript.has(src))return res();const el=document.createElement('script');el.src=src;el.onload=()=>{activeScript.add(src);res();};el.onerror=()=>rej(new Error('加载失败 '+src));document.body.appendChild(el);});}
function init(){document.getElementById('path').innerHTML='JSONL = <code>'+esc(window.PREVIEW_INDEX.jsonl_path)+'</code>';document.getElementById('stats').innerHTML=`<span class=stat><b>${samples.length}</b> samples</span><span class=stat><b>${window.PREVIEW_INDEX.object_count}</b> objects</span><span class=stat><b>${window.PREVIEW_INDEX.min_target_count}</b>-${window.PREVIEW_INDEX.max_target_count}</b> targets/sample</span>`;document.getElementById('q').oninput=filter;document.getElementById('open-first').onclick=()=>{if(filtered[0])openSample(filtered[0].index)};renderList();}
function filter(){const q=norm(document.getElementById('q').value);filtered=samples.filter(s=>!q||norm(s.sample_id).includes(q)||norm(s.object_id).includes(q)||s.target_keys.some(k=>norm(k).includes(q)));page=0;renderList();}
function gotoPage(next){const total=Math.max(1,Math.ceil(filtered.length/pageSize));page=Math.min(Math.max(0,next),total-1);renderList();}
function sampleThumb(s){return s.preview_thumb?`<span class="thumb"><img loading="lazy" src="${esc(s.preview_thumb)}" alt=""></span>`:`<span class="thumb missing-thumb">no image</span>`;}
function renderList(){const list=document.getElementById('list');const total=Math.max(1,Math.ceil(filtered.length/pageSize));page=Math.min(Math.max(0,page),total-1);const start=page*pageSize;const end=Math.min(start+pageSize,filtered.length);const rows=filtered.slice(start,end).map(s=>`<button class="card ${current&&current.sample.index===s.index?'active':''}" data-i="${s.index}">${sampleThumb(s)}<span class="card-main"><b>#${s.index} ${esc(s.object_id)}</b><small>${esc(s.sample_id)}</small><small>${esc(s.angle_name)} · view [${s.view_indices.join(', ')}] · ${s.target_count} targets</small><small>${esc(s.target_keys.slice(0,5).join(', '))}${s.target_keys.length>5?' …':''}</small></span></button>`).join('');list.innerHTML=`<div class="pager"><div class="pager-row"><button data-page="prev" ${page===0?'disabled':''}>上一页</button><span class="range">${filtered.length?start+1:0}-${end} / ${filtered.length}</span><button data-page="next" ${page>=total-1?'disabled':''}>下一页</button></div><div class="pager-row"><small>第 ${page+1} / ${total} 页，每页 ${pageSize} 条</small><span><input id="page-input" type="number" min="1" max="${total}" value="${page+1}"><button data-page="jump">跳页</button></span></div></div>${rows||'<div class=empty>没有匹配样本</div>'}`;list.querySelector('[data-page="prev"]')?.addEventListener('click',()=>gotoPage(page-1));list.querySelector('[data-page="next"]')?.addEventListener('click',()=>gotoPage(page+1));list.querySelector('[data-page="jump"]')?.addEventListener('click',()=>gotoPage(Number(document.getElementById('page-input').value||1)-1));list.querySelector('#page-input')?.addEventListener('keydown',e=>{if(e.key==='Enter')gotoPage(Number(e.currentTarget.value||1)-1);});list.querySelectorAll('[data-i]').forEach(b=>b.onclick=()=>openSample(Number(b.dataset.i)));}
async function openSample(index){document.getElementById('detail').innerHTML='<div class=empty>加载 sample metadata...</div>';document.getElementById('views').innerHTML='<div class=empty>加载 RGB...</div>';await scriptOnce(`data/sample_${String(index).padStart(6,'0')}.js`);current=window.__SAMPLE_DETAILS[index];renderList();renderSample();}
function renderSample(){const d=current;const v=(d.view_summaries||[])[0];const multi=(d.view_summaries||[]).length>1?` · 原记录含 ${(d.view_summaries||[]).length} 张，本页按单图样本只展示第一张`:'';document.getElementById('views').innerHTML=v?`<div class="view-meta"><span><b>image_${v.slot}</b> / view_${v.view} / ${esc(v.quadrant_name)}${multi}</span><a href="${esc(v.preview_src)}" target="_blank" rel="noreferrer">打开 PNG</a></div><div class="hero-frame"><img src="${esc(v.preview_src)}" alt="single RGB bbox preview for sample ${esc(d.sample.sample_id)}"></div>`:'<div class=empty>这个样本没有可展示图片</div>';const errors=d.validation_errors.length?`<div class=error><b>DATA WARNINGS (${d.validation_errors.length})</b><br>${d.validation_errors.map(esc).join('<br>')}</div>`:'';document.getElementById('detail').innerHTML=`${errors}<div class=sample-title><h2>#${d.sample.index} ${esc(d.sample.sample_id)}</h2><div class=meta>root=${esc(d.root_key)} · object=${esc(d.sample.object_id)} · ${esc(d.sample.angle_name)} · view=[${d.sample.view_indices.join(', ')}] · targets=${d.sample.target_count}</div></div><div class=tabs><button onclick="jsonTab='assistant';renderJson()" id=tab-assistant>assistant JSON</button><button onclick="jsonTab='record';renderJson()" id=tab-record>full JSONL record</button><button onclick="jsonTab='parts';renderJson()" id=tab-parts>target parts</button></div><div id=partsbox style="display:none"><div class=parts>${d.target_parts.map(p=>`<div class=part><span class=dot style="background:${p.color}"></span><b>${esc(p.key)}</b> · label ${p.label}<br><span class=muted>${esc(p.motion)} · visible views ${esc(JSON.stringify(p.visible_views_from_mask))} · mask px ${esc(JSON.stringify(p.mask_pixels_by_slot))}</span></div>`).join('')}</div></div><pre id=jsonbox></pre>`;renderJson();}
function renderJson(){if(!current)return;document.getElementById('tab-assistant')?.classList.toggle('active',jsonTab==='assistant');document.getElementById('tab-record')?.classList.toggle('active',jsonTab==='record');document.getElementById('tab-parts')?.classList.toggle('active',jsonTab==='parts');const parts=document.getElementById('partsbox');const box=document.getElementById('jsonbox');if(jsonTab==='parts'){parts.style.display='block';box.style.display='none';return;}parts.style.display='none';box.style.display='block';box.textContent=JSON.stringify(jsonTab==='assistant'?current.assistant:current.raw_record,null,2);}
document.getElementById('views').addEventListener('error',e=>{const img=e.target;if(!(img instanceof HTMLImageElement))return;const note=document.createElement('div');note.className='error';note.textContent='图片加载失败：'+(img.getAttribute('src')||'');img.replaceWith(note);},true);
document.getElementById('list').addEventListener('error',e=>{const img=e.target;if(!(img instanceof HTMLImageElement))return;const thumb=img.closest('.thumb');if(!thumb)return;thumb.classList.add('missing-thumb');thumb.textContent='no image';},true);
document.getElementById('json-focus-btn').onclick=()=>{document.body.classList.toggle('json-focus');document.getElementById('json-focus-btn').textContent=document.body.classList.contains('json-focus')?'退出放大':'放大 JSON';};
init();
</script>
</body>
</html>"""

def generate_static_site(state: PreviewState, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("data", "previews", "vendor"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    copy_classic_vendor_assets(output_dir)

    sample_summaries: list[dict[str, Any]] = []
    target_counts: list[int] = []
    for sample_number, sample in enumerate(state.samples, start=1):
        detail_payload = build_static_sample_assets(
            state=state,
            sample=sample,
            output_dir=output_dir,
        )
        summary = sample_summary(sample)
        summary["preview_thumb"] = detail_payload["preview_images"][0]
        sample_summaries.append(summary)
        target_counts.append(summary["target_count"])

        stem = sample_file_stem(sample)
        write_js_assignment(
            output_dir / "data" / f"{stem}.js",
            f"window.__SAMPLE_DETAILS=window.__SAMPLE_DETAILS||{{}};window.__SAMPLE_DETAILS[{sample.index}]=",
            detail_payload,
        )
        if sample_number % 100 == 0 or sample_number == len(state.samples):
            print(f"Generated {sample_number}/{len(state.samples)} samples", flush=True)

    object_ids = sorted({sample.object_id for sample in state.samples})
    index_payload = {
        "dataset_name": state.config.dataset_name,
        "data_root": state.config.data_root,
        "jsonl_path": str(state.jsonl_path),
        "sample_count": len(state.samples),
        "object_count": len(object_ids),
        "min_target_count": min(target_counts) if target_counts else 0,
        "max_target_count": max(target_counts) if target_counts else 0,
        "objects": object_ids,
        "samples": sample_summaries,
    }
    write_js_assignment(output_dir / "data" / "index.js", "window.PREVIEW_INDEX=", index_payload)
    (output_dir / "index.html").write_text(STATIC_HTML_PAGE, encoding="utf-8")
    print(f"Static preview generated: {output_dir / 'index.html'}", flush=True)


def regenerate_static_decoded_voxel_payload_for_sample(
    state: PreviewState,
    output_dir: Path,
    sample: SampleRecord,
) -> dict[str, Any]:
    decoded_payload_path = static_decoded_voxel_payload_path(output_dir, sample)
    if static_decoded_voxel_payload_is_current(decoded_payload_path, sample):
        return {
            "skipped": True,
            "has_data": False,
            "warning_count": 0,
            "has_warnings": False,
        }

    decoded_payload, validation_errors = build_static_decoded_voxel_payload(
        state=state,
        sample=sample,
    )
    has_data = bool(decoded_payload.get("decoded_overall") or decoded_payload.get("decoded_target_voxels"))
    write_static_decoded_voxel_payload(output_dir, sample, decoded_payload)
    return {
        "skipped": False,
        "has_data": has_data,
        "warning_count": len(validation_errors),
        "has_warnings": bool(validation_errors),
    }


def regenerate_static_voxel_payloads(state: PreviewState, output_dir: Path, jobs: int = 1) -> None:
    decoded_voxel_dir = output_dir / "decoded_voxels"
    decoded_voxel_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(output_dir / "index.html", STATIC_HTML_PAGE)

    jobs = max(1, int(jobs))
    samples_with_warnings = 0
    warning_count = 0
    payloads_with_any_decoded = 0
    skipped_existing_count = 0
    completed_count = 0

    def record_result(result: dict[str, Any]) -> None:
        nonlocal completed_count, samples_with_warnings, warning_count
        nonlocal payloads_with_any_decoded, skipped_existing_count
        completed_count += 1
        if result["skipped"]:
            skipped_existing_count += 1
        if result["has_data"]:
            payloads_with_any_decoded += 1
        if result["has_warnings"]:
            samples_with_warnings += 1
            warning_count += int(result["warning_count"])
        if completed_count % 100 == 0 or completed_count == len(state.samples):
            print(f"Regenerated decoded voxel payloads {completed_count}/{len(state.samples)} samples", flush=True)

    if jobs == 1:
        for sample in state.samples:
            record_result(regenerate_static_decoded_voxel_payload_for_sample(state, output_dir, sample))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(regenerate_static_decoded_voxel_payload_for_sample, state, output_dir, sample)
                for sample in state.samples
            ]
            for future in concurrent.futures.as_completed(futures):
                record_result(future.result())

    print(f"Static decoded voxel payloads regenerated: {decoded_voxel_dir}", flush=True)
    print(f"Static preview index refreshed: {output_dir / 'index.html'}", flush=True)
    print(
        f"Generated decoded voxel payloads with data: {payloads_with_any_decoded}",
        flush=True,
    )
    if skipped_existing_count:
        print(f"Skipped existing decoded voxel payloads: {skipped_existing_count}", flush=True)
    if warning_count:
        print(
            f"Voxel payload warnings: {warning_count} warning(s) across {samples_with_warnings} sample(s)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class PreviewHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: PreviewState):
        super().__init__(server_address, PreviewRequestHandler)
        self.state = state


class PreviewRequestHandler(BaseHTTPRequestHandler):
    server: PreviewHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}", file=sys.stderr)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == "/":
                self.send_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/index":
                self.handle_api_index()
                return
            if path.startswith("/api/sample/"):
                sample_index = int(path.rsplit("/", 1)[1])
                self.send_json(build_sample_detail(self.server.state, sample_index))
                return
            if path.startswith("/api/rgb/"):
                _, _, _, sample_text, slot_text = path.split("/", 4)
                self.handle_rgb(int(sample_text), int(slot_text))
                return
            if path.startswith("/vendor/"):
                self.handle_vendor(path[len("/vendor/"):])
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
        except (PreviewError, ValueError, KeyError, IndexError, FileNotFoundError) as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # explicit server-side error, not a silent fallback
            self.send_json({"error": f"internal preview error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_api_index(self) -> None:
        state = self.server.state
        object_ids = sorted({sample.object_id for sample in state.samples})
        target_counts = [len(sample.target_keys) for sample in state.samples]
        samples = [sample_summary(sample) for sample in state.samples]
        self.send_json({
            "dataset_name": state.config.dataset_name,
            "data_root": state.config.data_root,
            "jsonl_path": str(state.jsonl_path),
            "sample_count": len(state.samples),
            "object_count": len(object_ids),
            "max_target_count": max(target_counts) if target_counts else 0,
            "min_target_count": min(target_counts) if target_counts else 0,
            "objects": object_ids,
            "samples": samples,
        })

    def handle_rgb(self, sample_index: int, slot: int) -> None:
        sample = self.server.state.samples[sample_index]
        path = rgb_path_for_sample(sample, slot)
        if not path.is_file():
            raise FileNotFoundError(f"RGB image missing: {path}")
        self.send_file(path, "image/png")

    def handle_vendor(self, asset: str) -> None:
        try:
            data = load_vendor_asset(asset)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND, "vendor asset not found")
            return
        except OSError as exc:
            self.send_json(
                {"error": f"failed to load vendor asset {asset}: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self.send_bytes(data, "text/javascript; charset=utf-8", cache=True)

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def send_file(self, path: Path, content_type: str) -> None:
        self.send_bytes(path.read_bytes(), content_type)

    def send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        cache: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=86400")
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# HTML / JS
# ---------------------------------------------------------------------------


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARTS-Qwen VLM Training Preview</title>
<style>
* { box-sizing: border-box; }
:root {
  --bg: #090d13; --panel: #111923; --panel2: #172231; --panel3: #202c3a;
  --line: #314256; --text: #edf2f7; --muted: #91a0b3; --quiet: #637388;
  --accent: #f3c74f; --cyan: #39d2c0; --danger: #ff5a70; --ok: #72e6ac;
}
body { margin: 0; background: var(--bg); color: var(--text); font-family: "Noto Sans SC", "IBM Plex Sans", "Segoe UI", sans-serif; }
button, input { font: inherit; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.index-view { min-height: 100vh; padding: 16px; }
.detail-view { height: 100vh; overflow: hidden; }
[hidden] { display: none !important; }
.hero { display: grid; grid-template-columns: minmax(0, 1fr) minmax(520px, 48%); gap: 18px; padding: 18px 20px; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }
.hero h1 { margin: 0 0 8px; font-size: 31px; }
.hero p { margin: 6px 0 0; color: var(--muted); line-height: 1.55; }
.stats-grid { display: grid; grid-template-columns: repeat(4, minmax(96px, 1fr)); gap: 10px; align-self: start; }
.stat { border: 1px solid var(--line); border-radius: 8px; background: var(--panel2); padding: 11px 12px; }
.stat strong { display: block; color: var(--accent); font-size: 22px; }
.stat span { display: block; margin-top: 3px; color: var(--muted); font-size: 12px; }
.search-row { margin-top: 14px; display: grid; grid-template-columns: minmax(280px, 620px) auto; gap: 10px; }
.search-row input, .top-bar input { height: 36px; border: 1px solid #46566b; border-radius: 5px; padding: 0 11px; background: #172233; color: var(--text); outline: none; }
.search-row input:focus, .top-bar input:focus, select:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(243,199,79,.18); }
.search-row button, .tool-btn { height: 36px; border: 1px solid var(--cyan); border-radius: 5px; padding: 0 14px; color: #071014; background: var(--cyan); font-weight: 750; cursor: pointer; }
.hint { margin-top: 9px; color: var(--muted); min-height: 1.4em; }
.sample-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; margin-top: 16px; }
.sample-card { border: 1px solid var(--line); border-radius: 9px; overflow: hidden; background: var(--panel); color: var(--text); text-align: left; cursor: pointer; transition: transform .14s, border-color .14s; }
.sample-card:hover { transform: translateY(-2px); border-color: var(--cyan); }
.sample-thumb { aspect-ratio: 1; background: #070b11; display: flex; align-items: center; justify-content: center; border-bottom: 1px solid var(--line); }
.sample-thumb img { width: 92%; height: 92%; object-fit: contain; }
.sample-body { padding: 11px 13px 13px; }
.sample-id { display: block; color: var(--accent); font-weight: 800; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sample-meta { display: block; margin-top: 5px; color: var(--muted); font-size: 12px; line-height: 1.45; }
.top-bar { height: 48px; padding: 8px 12px; display: flex; align-items: center; gap: 9px; background: var(--panel); border-bottom: 1px solid var(--line); white-space: nowrap; }
.top-bar .tool-btn { height: 30px; padding: 0 10px; }
.top-bar .secondary { background: #1b2938; border-color: #3f5368; color: var(--text); }
.top-bar .status { margin-left: auto; overflow: hidden; text-overflow: ellipsis; color: var(--muted); font-size: 12px; }
.main { height: calc(100vh - 48px); display: grid; grid-template-columns: 320px minmax(480px, 47%) minmax(420px, 1fr); overflow: hidden; }
.left-panel { background: var(--panel); border-right: 1px solid var(--line); overflow: auto; padding: 10px; }
.section-title { color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-size: 11px; margin: 4px 0 9px; }
.part-btn { display: grid; grid-template-columns: 13px minmax(0, 1fr) auto; gap: 8px; width: 100%; padding: 8px; margin-bottom: 7px; border: 1px solid transparent; border-radius: 7px; background: var(--panel2); color: var(--text); cursor: pointer; text-align: left; }
.part-btn:hover { background: #1c2938; }
.part-btn.active { border-color: var(--part-color); }
.part-btn.in-focus { box-shadow: inset 3px 0 0 var(--accent); }
.dot { width: 12px; height: 12px; border-radius: 50%; margin-top: 3px; background: var(--part-color); }
.part-name { font-weight: 750; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.part-meta { color: var(--quiet); font-size: 10px; margin-top: 2px; line-height: 1.35; }
.part-side { display: grid; gap: 4px; justify-items: end; }
.badge { font-size: 10px; border-radius: 3px; padding: 1px 5px; }
.badge-ok { color: #06110d; background: var(--ok); }
.badge-warn { color: #16080a; background: var(--danger); }
.vis-grid { display: grid; grid-template-columns: repeat(4, 15px); gap: 3px; }
.vis-cell { height: 14px; border: 1px solid #39495a; border-radius: 3px; color: var(--quiet); font-size: 9px; line-height: 12px; text-align: center; }
.vis-cell.visible { background: var(--part-color); border-color: var(--part-color); color: #071014; }
.rgb-section { background: #0d1118; overflow: auto; padding: 8px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; align-content: start; }
.rgb-card { border: 1px solid #283747; border-radius: 7px; background: #111822; overflow: hidden; cursor: pointer; }
.rgb-card.active { border-color: var(--accent); box-shadow: 0 0 0 1px rgba(243,199,79,.35); }
.rgb-head { display: flex; justify-content: space-between; gap: 8px; padding: 7px 8px; background: #151f2a; border-bottom: 1px solid #243242; }
.rgb-head strong { color: var(--accent); font-size: 12px; }
.rgb-head span { color: var(--muted); font-size: 11px; }
.rgb-card canvas { width: 100%; aspect-ratio: 1; display: block; background: #05080d; }
.right-panel { display: grid; grid-template-rows: minmax(260px, 58%) minmax(180px, 42%); min-width: 0; border-left: 1px solid var(--line); }
.voxel-section { position: relative; background: #080d14; min-height: 240px; border-bottom: 1px solid var(--line); }
.voxel-section canvas { width: 100% !important; height: 100% !important; }
.voxel-controls { position: absolute; right: 8px; top: 8px; z-index: 3; display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px; max-width: calc(100% - 16px); pointer-events: auto; }
.voxel-layer-btn { min-height: 26px; border: 1px solid #3e5066; background: rgba(22,33,48,.9); color: var(--muted); border-radius: 4px; padding: 0 8px; font-size: 11px; font-weight: 700; cursor: pointer; }
.voxel-layer-btn.active { border-color: var(--cyan); background: rgba(47,216,209,.9); color: #071014; }
.voxel-layer-btn:not(.active) { opacity: .72; text-decoration: line-through; }
.voxel-layer-btn:focus-visible { outline: 2px solid var(--cyan); outline-offset: 2px; }
.voxel-info { position: absolute; left: 8px; bottom: 8px; padding: 4px 9px; border-radius: 4px; background: rgba(0,0,0,.55); color: var(--muted); font-size: 12px; pointer-events: none; }
.json-section { overflow: auto; background: #0c121b; padding: 10px; }
.error-panel { border: 1px solid rgba(255,90,112,.45); border-radius: 7px; padding: 8px 10px; margin-bottom: 10px; background: rgba(255,90,112,.09); color: #ffc6cf; }
.error-panel strong { color: var(--danger); }
.json-tabs { display: flex; gap: 6px; margin-bottom: 8px; }
.json-tabs button { border: 1px solid #3e5066; background: #162130; color: var(--muted); border-radius: 4px; padding: 5px 9px; cursor: pointer; }
.json-tabs button.active { background: var(--cyan); border-color: var(--cyan); color: #071014; }
pre { margin: 0; padding: 10px; border: 1px solid #233144; border-radius: 6px; background: #07101a; color: #cbd5e1; font-size: 11px; line-height: 1.45; overflow: auto; }
.loading { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--quiet); }
@media (max-width: 1220px) { .main { grid-template-columns: 300px 1fr; grid-template-rows: 54% 46%; } .right-panel { grid-column: 1 / -1; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr; } .voxel-section { border-bottom: 0; border-right: 1px solid var(--line); } .hero { grid-template-columns: 1fr; } }
@media (min-width: 1221px) { .main { grid-template-columns: minmax(380px, 22vw) minmax(720px, 1fr) minmax(560px, .8fr); } .left-panel { padding: 12px; } .part-btn { padding: 10px; min-height: 76px; } .part-name { font-size: 13px; } .part-meta { font-size: 11px; } .rgb-section { height: 100%; padding: 12px; gap: 12px; grid-template-rows: repeat(2, minmax(0, 1fr)); align-content: stretch; } .rgb-card { min-height: 0; display: flex; flex-direction: column; border-radius: 9px; } .rgb-head { flex: 0 0 auto; min-height: 36px; align-items: center; } .rgb-head strong { font-size: 13px; } .rgb-card canvas { flex: 1 1 auto; min-height: 0; height: 100%; aspect-ratio: auto; object-fit: contain; } .right-panel { grid-template-rows: minmax(0, 56%) minmax(280px, 44%); } .json-section { padding: 14px; } pre { white-space: pre; font-size: 13px; line-height: 1.55; } }
.error-panel { max-height: 160px; overflow: auto; }
</style>
</head>
<body>
<section class="index-view" id="index-view">
  <div class="hero">
    <div>
      <h1>VLM 训练样本预览</h1>
      <p>JSONL-first：只展示实际进入训练的 JSONL 样本；单图样本就只显示 1 张图。不会用全 12 视角或 raw parts 静默兜底。</p>
      <p id="dataset-path"></p>
      <div class="search-row">
        <input id="search" type="search" placeholder="搜索 sample_id / object_id / component，例如 100866 或 switch_0">
        <button id="open-first" type="button">打开首个匹配</button>
      </div>
      <div class="hint" id="hint"></div>
    </div>
    <div class="stats-grid" id="index-stats"></div>
  </div>
  <div class="sample-grid" id="sample-grid"></div>
</section>

<section class="detail-view" id="detail-view" hidden>
  <div class="top-bar">
    <button class="tool-btn secondary" id="back" type="button">返回列表</button>
    <button class="tool-btn secondary" id="prev" type="button">上一条</button>
    <button class="tool-btn secondary" id="next" type="button">下一条</button>
    <input id="jump" type="search" placeholder="sample index / object_id">
    <button class="tool-btn" id="jump-btn" type="button">打开</button>
    <div class="status" id="status"></div>
  </div>
  <div class="main">
    <div class="left-panel" id="part-list"><div class="loading">No sample</div></div>
    <div class="rgb-section" id="rgb-section"><div class="loading">No sample</div></div>
    <div class="right-panel">
      <div class="voxel-section" id="voxel-section">
        <div class="voxel-controls" id="voxel-layer-controls">
          <button class="voxel-layer-btn active" data-voxel-layer="surface" type="button" title="surface voxel">surface</button>
          <button class="voxel-layer-btn active" data-voxel-layer="target" type="button" title="GT target movable component voxel">GT target</button>
          <button class="voxel-layer-btn active" data-voxel-layer="decodedOverall" type="button" title="decoded overall SS latent">decoded overall</button>
          <button class="voxel-layer-btn active" data-voxel-layer="decodedTarget" type="button" title="decoded target part SS latent">decoded target</button>
        </div>
        <div class="loading" id="voxel-loading">No sample</div><div class="voxel-info" id="voxel-info"></div>
      </div>
      <div class="json-section" id="json-section"><div class="loading">No sample</div></div>
    </div>
  </div>
</section>

<script type="importmap">
{
  "imports": {
    "three": "/vendor/three.module.js",
    "three/addons/": "/vendor/three/addons/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let indexData = null;
let samples = [];
let filteredSamples = [];
let currentData = null;
let activeParts = new Set();
let activeView = null;
let rgbCanvases = {};
let currentJsonTab = 'assistant';
const voxelLayerVisible = { surface: true, target: true, decodedOverall: true, decodedTarget: true };

const voxelContainer = document.getElementById('voxel-section');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
renderer.setClearColor(0x080d14);
voxelContainer.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 500);
camera.position.set(128, 96, 128);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(32, 32, 32);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.62));
const dl = new THREE.DirectionalLight(0xffffff, 0.85);
dl.position.set(60, 90, 60);
scene.add(dl);
function makeAxisLabel(text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 128; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = '700 34px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineWidth = 8;
  ctx.strokeStyle = 'rgba(0,0,0,.85)'; ctx.fillStyle = color;
  ctx.strokeText(text, 64, 32); ctx.fillText(text, 64, 32);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, depthWrite: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(10, 5, 1);
  return sprite;
}
function addVoxelAxes() {
  const axisLen = 72;
  const origin = new THREE.Vector3(0, 0, 0);
  const group = new THREE.Group();
  group.name = 'voxel-coordinate-axes';
  [
    { name: 'X', dir: new THREE.Vector3(1, 0, 0), color: 0xff4d4d, css: '#ff4d4d' },
    { name: 'Y', dir: new THREE.Vector3(0, 1, 0), color: 0x42e66f, css: '#42e66f' },
    { name: 'Z', dir: new THREE.Vector3(0, 0, 1), color: 0x4da3ff, css: '#4da3ff' },
  ].forEach(axis => {
    const arrow = new THREE.ArrowHelper(axis.dir, origin, axisLen, axis.color, 5, 2.6);
    arrow.line.material.depthTest = false;
    arrow.cone.material.depthTest = false;
    group.add(arrow);
    const label = makeAxisLabel(axis.name, axis.css);
    label.position.copy(axis.dir.clone().multiplyScalar(axisLen + 7));
    group.add(label);
  });
  const boxGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(63, 63, 63));
  const boxMat = new THREE.LineBasicMaterial({ color: 0x33485f, transparent: true, opacity: 0.68 });
  const box = new THREE.LineSegments(boxGeo, boxMat);
  box.position.set(31.5, 31.5, 31.5);
  group.add(box);
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(1.7, 16, 8),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  );
  marker.name = 'voxel-origin-marker';
  group.add(marker);
  scene.add(group);
}
addVoxelAxes();
let voxelMeshes = {};

function bindVoxelLayerControls() {
  document.querySelectorAll('[data-voxel-layer]').forEach(btn => {
    btn.onclick = () => {
      const key = btn.dataset.voxelLayer;
      voxelLayerVisible[key] = !voxelLayerVisible[key];
      updateVoxelLayerControls();
      rebuildVoxels();
    };
  });
  updateVoxelLayerControls();
}
function updateVoxelLayerControls() {
  document.querySelectorAll('[data-voxel-layer]').forEach(btn => {
    const visible = voxelLayerVisible[btn.dataset.voxelLayer];
    btn.classList.toggle('active', visible);
    btn.setAttribute('aria-pressed', String(visible));
  });
}

function resizeRenderer() {
  const w = voxelContainer.clientWidth;
  const h = voxelContainer.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resizeRenderer);
new ResizeObserver(resizeRenderer).observe(voxelContainer);
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
animate();

function escapeHtml(s) {
  return String(s).replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
}
function normalize(s) { return String(s || '').trim().toLowerCase(); }

async function init() {
  const resp = await fetch('/api/index');
  if (!resp.ok) throw new Error(await resp.text());
  indexData = await resp.json();
  samples = indexData.samples;
  filteredSamples = samples;
  renderIndexHeader();
  renderGrid();
}

function renderIndexHeader() {
  document.getElementById('dataset-path').innerHTML = `JSONL = <code>${escapeHtml(indexData.jsonl_path)}</code>`;
  document.getElementById('index-stats').innerHTML = `
    <div class="stat"><strong>${indexData.sample_count}</strong><span>训练样本</span></div>
    <div class="stat"><strong>${indexData.object_count}</strong><span>物体数</span></div>
    <div class="stat"><strong>${indexData.max_target_count}</strong><span>最大 target 数</span></div>
    <div class="stat"><strong>${indexData.min_target_count}</strong><span>最小 target 数</span></div>
  `;
  setHint(`显示 ${Math.min(filteredSamples.length, 500)} / ${filteredSamples.length} 条。输入可筛选；列表最多渲染 500 条避免浏览器卡顿。`);
}
function setHint(text) { document.getElementById('hint').textContent = text; }

function sampleMatches(sample, query) {
  if (!query) return true;
  return normalize(sample.sample_id).includes(query)
    || normalize(sample.object_id).includes(query)
    || sample.target_keys.some(k => normalize(k).includes(query));
}
function filterSamples() {
  const query = normalize(document.getElementById('search').value);
  filteredSamples = samples.filter(s => sampleMatches(s, query));
  renderGrid();
  setHint(`匹配 ${filteredSamples.length} 条，显示前 ${Math.min(filteredSamples.length, 500)} 条。`);
}
function renderGrid() {
  const grid = document.getElementById('sample-grid');
  const visible = filteredSamples.slice(0, 500);
  grid.innerHTML = visible.map(sample => `
    <button class="sample-card" type="button" data-index="${sample.index}">
      <span class="sample-thumb"><img loading="lazy" src="/api/rgb/${sample.index}/0" alt="${escapeHtml(sample.sample_id)}"></span>
      <span class="sample-body">
        <span class="sample-id">#${sample.index} ${escapeHtml(sample.object_id)}</span>
        <span class="sample-meta">${escapeHtml(sample.angle_name)} · views [${sample.view_indices.join(', ')}]</span>
        <span class="sample-meta">targets ${sample.target_count}: ${escapeHtml(sample.target_keys.slice(0, 4).join(', '))}${sample.target_keys.length > 4 ? ' …' : ''}</span>
      </span>
    </button>
  `).join('');
  grid.querySelectorAll('[data-index]').forEach(card => card.onclick = () => loadSample(Number(card.dataset.index)));
}

async function loadSample(index) {
  document.getElementById('index-view').hidden = true;
  document.getElementById('detail-view').hidden = false;
  document.getElementById('part-list').innerHTML = '<div class="loading">Loading sample...</div>';
  document.getElementById('rgb-section').innerHTML = '<div class="loading">Loading RGB...</div>';
  document.getElementById('json-section').innerHTML = '<div class="loading">Loading JSON...</div>';
  document.getElementById('voxel-info').textContent = '';
  try {
    const resp = await fetch(`/api/sample/${index}`);
    if (!resp.ok) throw new Error(await resp.text());
    currentData = await resp.json();
  } catch (e) {
    document.getElementById('part-list').innerHTML = `<div class="error-panel"><strong>Load failed</strong><br>${escapeHtml(e.message)}</div>`;
    return;
  }
  activeParts = new Set(currentData.target_parts.map(p => p.key));
  activeView = currentData.view_summaries[0].view;
  document.getElementById('jump').value = currentData.sample.index;
  renderEverything();
}

function renderEverything() {
  const s = currentData.sample;
  document.getElementById('status').textContent = `#${s.index} ${s.object_id} ${s.angle_name} · views [${s.view_indices.join(', ')}] · targets ${s.target_count}`;
  document.getElementById('prev').disabled = currentData.prev_index === null;
  document.getElementById('next').disabled = currentData.next_index === null;
  renderPartList();
  renderRGBViews();
  renderJSON();
  rebuildVoxels();
  resizeRenderer();
}

function renderPartList() {
  const container = document.getElementById('part-list');
  const errorHtml = currentData.validation_errors.length ? `
    <div class="error-panel"><strong>DATA ERRORS (${currentData.validation_errors.length})</strong><br>${currentData.validation_errors.map(escapeHtml).join('<br>')}</div>
  ` : '';
  container.innerHTML = `${errorHtml}<div class="section-title">Training target components</div>`;
  currentData.target_parts.forEach((p, i) => {
    const btn = document.createElement('button');
    btn.className = 'part-btn';
    btn.style.setProperty('--part-color', p.color);
    if (activeParts.has(p.key)) btn.classList.add('active');
    if (p.visible_views_from_bbox.includes(activeView)) btn.classList.add('in-focus');
    const badgeClass = p.has_voxel ? 'badge-ok' : 'badge-warn';
    const badgeText = p.has_voxel ? 'TARGET' : 'NO VOX';
    const cells = currentData.view_summaries.map((summary, idx) => {
      const visible = p.visible_views_from_bbox.includes(summary.view);
      return `<span class="vis-cell ${visible ? 'visible' : ''}">${idx + 1}</span>`;
    }).join('');
    const maskPixels = currentData.view_summaries.map((summary, idx) => `i${idx + 1}:${p.mask_pixels_by_slot[String(idx + 1)] || 0}`).join(' ');
    btn.innerHTML = `
      <span class="dot"></span>
      <span>
        <div class="part-name" title="${escapeHtml(p.key)}">${escapeHtml(p.key)}</div>
        <div class="part-meta">label ${p.label} | ${escapeHtml(p.joint_type || p.joint || '')}</div>
        <div class="part-meta">${escapeHtml(p.motion)}</div>
        <div class="part-meta">mask px ${escapeHtml(maskPixels)}</div>
        <div class="part-meta" title="${escapeHtml(p.voxel_path)}">voxel ${p.voxel_count} @ ${escapeHtml(currentData.voxel_resolution)}</div>
      </span>
      <span class="part-side"><span class="badge ${badgeClass}">${badgeText}</span><span class="vis-grid">${cells}</span></span>
    `;
    btn.onclick = () => {
      if (activeParts.has(p.key)) activeParts.delete(p.key); else activeParts.add(p.key);
      renderPartList(); redrawAllCanvases(); rebuildVoxels();
    };
    container.appendChild(btn);
  });
}

function renderRGBViews() {
  const section = document.getElementById('rgb-section');
  section.innerHTML = '';
  rgbCanvases = {};
  currentData.view_summaries.forEach((summary, slotIdx) => {
    const card = document.createElement('div');
    card.className = 'rgb-card';
    if (summary.view === activeView) card.classList.add('active');
    const visibleParts = currentData.target_parts.filter(p => p.visible_views_from_bbox.includes(summary.view));
    card.innerHTML = `
      <div class="rgb-head"><strong>image_${summary.slot} / ${summary.quadrant_name}</strong><span>view_${summary.view} · ${visibleParts.length}/${currentData.target_parts.length} visible</span></div>
      <canvas width="512" height="512"></canvas>
    `;
    card.onclick = () => { activeView = summary.view; renderEverything(); };
    section.appendChild(card);
    const canvas = card.querySelector('canvas');
    rgbCanvases[summary.view] = canvas;
    const img = new Image();
    img.onload = () => { canvas._rgbImage = img; drawCanvas(summary.view); };
    img.src = `/api/rgb/${currentData.sample.index}/${slotIdx}`;
  });
}
function redrawAllCanvases() { currentData.view_summaries.forEach(s => drawCanvas(s.view)); }
function hexToRGB(hex) { return [parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16)]; }
function drawCanvas(viewIdx) {
  const canvas = rgbCanvases[viewIdx];
  if (!canvas || !canvas._rgbImage) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 512, 512);
  ctx.drawImage(canvas._rgbImage, 0, 0, 512, 512);
  const maskInfo = currentData.masks[String(viewIdx)];
  if (maskInfo && activeParts.size) {
    const overlay = ctx.createImageData(maskInfo.w, maskInfo.h);
    const partByLabel = new Map(currentData.target_parts.map(p => [p.label, p]));
    for (let i = 0; i < maskInfo.data.length; i++) {
      const label = maskInfo.data[i];
      if (!label) continue;
      const part = partByLabel.get(label);
      if (!part || !activeParts.has(part.key)) continue;
      const [r,g,b] = hexToRGB(part.color);
      const off = i * 4;
      overlay.data[off] = r; overlay.data[off + 1] = g; overlay.data[off + 2] = b; overlay.data[off + 3] = 105;
    }
    const tmp = document.createElement('canvas');
    tmp.width = maskInfo.w; tmp.height = maskInfo.h;
    tmp.getContext('2d').putImageData(overlay, 0, 0);
    ctx.drawImage(tmp, 0, 0, 512, 512);
  }
  const slotIndex = currentData.view_summaries.findIndex(s => s.view === viewIdx);
  const slot = String(slotIndex + 1);
  currentData.target_parts.forEach(part => {
    if (!activeParts.has(part.key)) return;
    const bbox = part.bboxes[slot];
    if (!bbox || !bbox.visible) return;
    const [x1,y1,x2,y2] = bbox.box.map(v => v / 1000 * 512);
    ctx.strokeStyle = part.color; ctx.lineWidth = 3; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = part.color; ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(part.key, x1 + 4, Math.max(14, y1 + 14));
  });
}

function createVoxelMesh(coords, color, opacity, size = 0.9) {
  const geo = new THREE.BoxGeometry(size, size, size);
  const mat = new THREE.MeshLambertMaterial({ color: new THREE.Color(color), transparent: opacity < 1, opacity, depthWrite: opacity >= 1 });
  const mesh = new THREE.InstancedMesh(geo, mat, coords.length);
  const dummy = new THREE.Object3D();
  for (let i = 0; i < coords.length; i++) {
    dummy.position.set(coords[i][0], coords[i][1], coords[i][2]); dummy.updateMatrix(); mesh.setMatrixAt(i, dummy.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
  return mesh;
}
function clearVoxels() {
  for (const mesh of Object.values(voxelMeshes)) { scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); }
  voxelMeshes = {};
}
function rebuildVoxels() {
  if (!currentData) return;
  clearVoxels();
  if (voxelLayerVisible.surface && currentData.voxels.surface) {
    const m = createVoxelMesh(currentData.voxels.surface, '#888888', 0.13); scene.add(m); voxelMeshes.surface = m;
  }
  currentData.target_parts.forEach(part => {
    if (!voxelLayerVisible.target || !activeParts.has(part.key) || !currentData.voxels[part.key]) return;
    const m = createVoxelMesh(currentData.voxels[part.key], part.color, 1.0); scene.add(m); voxelMeshes[part.key] = m;
  });
  if (voxelLayerVisible.decodedOverall && currentData.decoded_voxels?.overall?.length) {
    const m = createVoxelMesh(currentData.decoded_voxels.overall, '#00e5ff', 0.32, 0.58); scene.add(m); voxelMeshes.decoded_overall = m;
  }
  currentData.target_parts.forEach(part => {
    if (!voxelLayerVisible.decodedTarget || !activeParts.has(part.key) || !currentData.decoded_voxels?.[part.key]) return;
    const m = createVoxelMesh(currentData.decoded_voxels[part.key], '#ff4de3', 0.36, 1.08); scene.add(m); voxelMeshes[`decoded_${part.key}`] = m;
  });
  updateVoxelInfo();
}
function fmtMetric(m) {
  if (!m) return '';
  const f = x => Number.isFinite(Number(x)) ? Number(x).toFixed(3) : 'n/a';
  return `IoU ${f(m.iou)} P ${f(m.precision)} R ${f(m.recall)}`;
}
function avgPartMetric(metrics, keys) {
  if (!metrics || !keys.length) return '';
  const vals = [];
  keys.forEach(k => { const m = metrics[k]; if (m && Number.isFinite(Number(m.iou))) vals.push(m); });
  if (!vals.length) return '';
  const avg = name => vals.reduce((sum, m) => sum + Number(m[name] || 0), 0) / vals.length;
  return `decoded target avg IoU ${avg('iou').toFixed(3)} P ${avg('precision').toFixed(3)} R ${avg('recall').toFixed(3)}`;
}
function updateVoxelInfo() {
  const surf = currentData?.voxels.surface?.length || 0;
  let activeVoxelCount = 0;
  activeParts.forEach(k => { if (currentData.voxels[k]) activeVoxelCount += currentData.voxels[k].length; });
  const decodedOverall = currentData?.decoded_voxels?.overall?.length || 0;
  let decodedActiveCount = 0;
  activeParts.forEach(k => { if (currentData.decoded_voxels?.[k]) decodedActiveCount += currentData.decoded_voxels[k].length; });
  const metricBits = [];
  const metrics = currentData?.decoded_metrics || {};
  if (metrics.overall) metricBits.push('overall ' + fmtMetric(metrics.overall));
  const partLine = avgPartMetric(metrics.parts, Array.from(activeParts));
  if (partLine) metricBits.push(partLine);
  const countText = (label, visible, count) => `${label} ${visible ? count : 'off'}`;
  const countBits = [
    countText('GT surface', voxelLayerVisible.surface, surf),
    countText('target', voxelLayerVisible.target, activeVoxelCount),
    countText('decoded overall', voxelLayerVisible.decodedOverall, decodedOverall),
    countText('decoded target', voxelLayerVisible.decodedTarget, decodedActiveCount),
  ];
  document.getElementById('voxel-info').textContent = `res ${currentData.voxel_resolution} | ${countBits.join(' | ')}${metricBits.length ? ' | ' + metricBits.join(' | ') : ''} | 坐标轴 X红 Y绿 Z蓝`;
  document.getElementById('voxel-info').title = [currentData.surface_voxel_path || '', currentData.decoded_metrics_path || ''].filter(Boolean).join('\n');
}

function renderJSON() {
  const container = document.getElementById('json-section');
  const data = currentJsonTab === 'assistant' ? currentData.assistant : currentData.raw_record;
  container.innerHTML = `
    <div class="json-tabs">
      <button data-tab="assistant" class="${currentJsonTab === 'assistant' ? 'active' : ''}">assistant JSON</button>
      <button data-tab="record" class="${currentJsonTab === 'record' ? 'active' : ''}">full JSONL record</button>
    </div>
    <pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>
  `;
  container.querySelectorAll('[data-tab]').forEach(btn => btn.onclick = () => { currentJsonTab = btn.dataset.tab; renderJSON(); });
}

function openFirstMatch() { if (filteredSamples.length) loadSample(filteredSamples[0].index); }
function jumpOpen() {
  const q = normalize(document.getElementById('jump').value);
  if (!q) return;
  if (/^\d+$/.test(q)) { const idx = Number(q); if (idx >= 0 && idx < samples.length) { loadSample(idx); return; } }
  const hit = samples.find(s => normalize(s.sample_id).includes(q) || normalize(s.object_id).includes(q));
  if (hit) loadSample(hit.index);
}

document.getElementById('search').addEventListener('input', filterSamples);
document.getElementById('search').addEventListener('keydown', e => { if (e.key === 'Enter') openFirstMatch(); });
document.getElementById('open-first').onclick = openFirstMatch;
document.getElementById('back').onclick = () => { document.getElementById('detail-view').hidden = true; document.getElementById('index-view').hidden = false; };
document.getElementById('prev').onclick = () => { if (currentData?.prev_index !== null) loadSample(currentData.prev_index); };
document.getElementById('next').onclick = () => { if (currentData?.next_index !== null) loadSample(currentData.next_index); };
document.getElementById('jump-btn').onclick = jumpOpen;
document.getElementById('jump').addEventListener('keydown', e => { if (e.key === 'Enter') jumpOpen(); });

bindVoxelLayerControls();
init().catch(e => { document.body.innerHTML = `<div class="error-panel"><strong>Preview init failed</strong><br>${escapeHtml(e.message)}</div>`; });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_server(state: PreviewState) -> None:
    validate_vendor_assets()
    server = PreviewHTTPServer((state.host, state.port), state)
    url = f"http://{state.host}:{state.port}/"
    print(f"Preview URL: {url}", flush=True)
    print(f"JSONL: {state.jsonl_path}", flush=True)
    print(f"Indexed samples: {len(state.samples)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down preview server.", flush=True)
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.serve and args.regen_voxels_only:
            raise PreviewError("--regen-voxels-only cannot be combined with --serve")
        config = load_config(args.config)
        jsonl_path = Path(args.jsonl) if args.jsonl else default_jsonl_path(config)
        object_filter = parse_object_ids(args.object_ids) if args.object_ids else None
        samples = build_sample_index(
            jsonl_path=jsonl_path,
            data_root=Path(config.data_root),
            object_filter=object_filter,
        )
        state = PreviewState(
            config=config,
            jsonl_path=jsonl_path,
            samples=samples,
            host=args.host,
            port=args.port,
        )
        if args.serve:
            run_server(state)
        else:
            output_dir = Path(args.output_dir) if args.output_dir else default_static_output_dir(config)
            if args.regen_voxels_only:
                regenerate_static_voxel_payloads(state, output_dir, jobs=args.jobs)
            else:
                generate_static_site(state, output_dir)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
