"""Pool cached DINO patch tokens inside a static part observation box."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class StaticDinoPartFeature:
    feature: tuple[float, ...]
    view_indices: tuple[int, ...]
    input_files: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def pool_static_part_dino_feature(
    render_root: Path,
    label: str,
    *,
    view_indices: list[int] | tuple[int, ...],
    state_index: int = 0,
) -> StaticDinoPartFeature | None:
    """Return normalized part-minus-object DINO features from selected views."""
    render_root = Path(render_root)
    normalized = str(render_root.resolve()).replace("\\", "/").lower()
    if "/renders/" not in normalized:
        raise ValueError("static DINO features require a renders/<object> root")
    state_dir = render_root / f"angle_{int(state_index)}"
    boxes_path = state_dir / "bbox_gt.json"
    dataset_root = render_root.parent.parent
    tokens_path = (
        dataset_root / "reconstruction" / "dinov2_tokens"
        / render_root.name / f"angle_{int(state_index)}" / "tokens.npz"
    )
    if not boxes_path.is_file() or not tokens_path.is_file():
        return None
    boxes = json.loads(boxes_path.read_text(encoding="utf-8"))
    part_key = _match_part_key(boxes.get("parts") or {}, label)
    if part_key is None:
        return None
    with np.load(tokens_path, allow_pickle=False) as payload:
        if "patchtokens" not in payload:
            return None
        patch_tokens = np.asarray(payload["patchtokens"], dtype=np.float32)
    pooled = []
    input_files = [str(boxes_path.resolve()), str(tokens_path.resolve())]
    used_indices = []
    for view_index in dict.fromkeys(int(value) for value in view_indices):
        if not 0 <= view_index < len(patch_tokens):
            continue
        box = ((boxes["parts"][part_key].get("views") or {}).get(str(view_index)) or {}).get("bbox")
        image_path = state_dir / "rgb" / f"view_{view_index}.png"
        if not box or not image_path.is_file():
            continue
        mask = _part_patch_mask(image_path, box, patch_tokens.shape[-2:])
        if mask is None or not mask.any():
            continue
        tokens = patch_tokens[view_index].reshape((patch_tokens.shape[1], -1))
        part_feature = _unit(tokens[:, mask.reshape(-1)].mean(axis=1))
        global_feature = _unit(tokens.mean(axis=1))
        pooled.append(part_feature - global_feature)
        used_indices.append(view_index)
        input_files.append(str(image_path.resolve()))
    if not pooled:
        return None
    feature = _unit(np.mean(np.asarray(pooled, dtype=np.float64), axis=0))
    return StaticDinoPartFeature(
        feature=tuple(float(value) for value in feature),
        view_indices=tuple(used_indices),
        input_files=tuple(dict.fromkeys(input_files)),
    )


def _part_patch_mask(image_path: Path, raw_box, grid_shape: tuple[int, int]) -> np.ndarray | None:
    image = Image.open(image_path).convert("RGBA")
    alpha = np.asarray(image.getchannel("A"))
    y_values, x_values = np.where(alpha > 204)
    if not len(x_values):
        return None
    center_x = (float(x_values.min()) + float(x_values.max())) * 0.5
    center_y = (float(y_values.min()) + float(y_values.max())) * 0.5
    crop_size = max(float(x_values.max() - x_values.min()), float(y_values.max() - y_values.min())) * 1.2
    if crop_size <= 1e-6:
        return None
    crop_x0 = center_x - crop_size * 0.5
    crop_y0 = center_y - crop_size * 0.5
    scale_x = float(image.width) / 1000.0
    scale_y = float(image.height) / 1000.0
    box_x0, box_y0, box_x1, box_y1 = (float(value) for value in raw_box)
    grid_h, grid_w = grid_shape
    grid_x0 = (box_x0 * scale_x - crop_x0) / crop_size * grid_w
    grid_x1 = (box_x1 * scale_x - crop_x0) / crop_size * grid_w
    grid_y0 = (box_y0 * scale_y - crop_y0) / crop_size * grid_h
    grid_y1 = (box_y1 * scale_y - crop_y0) / crop_size * grid_h
    x_centers = np.arange(grid_w, dtype=np.float64) + 0.5
    y_centers = np.arange(grid_h, dtype=np.float64) + 0.5
    return (
        (y_centers[:, None] >= grid_y0) & (y_centers[:, None] <= grid_y1)
        & (x_centers[None, :] >= grid_x0) & (x_centers[None, :] <= grid_x1)
    )


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / max(float(np.linalg.norm(values)), 1e-12)


def _match_part_key(parts: dict, label: str) -> str | None:
    wanted = _normalize_label(label)
    matches = [key for key in parts if _normalize_label(key) == wanted]
    return matches[0] if len(matches) == 1 else None


def _normalize_label(value: str) -> str:
    value = Path(str(value)).stem.lower()
    value = re.sub(r"^part_\d+_", "", value)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)
