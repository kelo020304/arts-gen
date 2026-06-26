"""Semantic direction priors for final joint limits.

This module may use part-name semantics, but it must not read authored USD or
GT numeric joint limits. Geometry scan traces remain the source of travel
length; semantics only chooses which side of the rest pose is the usable range.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping


_PULL_OUT_PHRASES = (
    "pull-out",
    "pull out",
    "pullout",
    "slide-out",
    "slide out",
)
_PULL_OUT_WORDS = (
    "drawer",
    "basket",
    "pan",
    "tray",
    "slider",
)
_PULL_OUT_CJK = (
    "炸篮",
    "炸桶",
)


def load_part_semantics(source_root: Path, object_id: str) -> dict[str, list[str]]:
    """Load source part labels as part_name -> semantic labels."""
    source_id = object_id.removeprefix("ra_")
    path = source_root / f"source/model/{source_id}/gt_part.json"
    if not path.is_file():
        return {}
    return part_semantics_from_gt_part(json.loads(path.read_text()))


def part_semantics_from_gt_part(gt_part: Mapping[str, object]) -> dict[str, list[str]]:
    """Invert gt_part's semantic-label -> part-name mapping."""
    semantics: defaultdict[str, list[str]] = defaultdict(list)
    for label, part_name in gt_part.items():
        if isinstance(part_name, str) and part_name:
            semantics[part_name].append(str(label))
    return dict(semantics)


def apply_motion_direction_prior(
    prediction: Mapping[str, object],
    *,
    joint: Mapping[str, object],
    part_semantics: Mapping[str, list[str]],
) -> dict:
    """Clamp final prismatic limits when part semantics imply a one-sided rest pose."""
    out = dict(prediction)
    if out.get("type") != "prismatic" or out.get("status") != "ok":
        return out

    policy = _semantic_policy_for_joint(joint, part_semantics)
    if policy is None:
        return out

    raw_lower = out.get("predicted_lower")
    raw_upper = out.get("predicted_upper")
    if raw_lower is None or raw_upper is None:
        return out

    lower = float(raw_lower)
    upper = float(raw_upper)
    if policy == "positive_only" and lower < 0.0 <= upper:
        out["predicted_lower"] = 0.0
        out["motion_direction_prior"] = {
            "policy": policy,
            "reason": "pull_out_rest_pose",
            "raw_predicted_lower": lower,
            "raw_predicted_upper": upper,
            "labels": _labels_for_joint(joint, part_semantics),
        }
    elif policy == "negative_only" and lower <= 0.0 < upper:
        out["predicted_upper"] = 0.0
        out["motion_direction_prior"] = {
            "policy": policy,
            "reason": "pull_out_rest_pose",
            "raw_predicted_lower": lower,
            "raw_predicted_upper": upper,
            "labels": _labels_for_joint(joint, part_semantics),
        }
    return out


def _semantic_policy_for_joint(
    joint: Mapping[str, object],
    part_semantics: Mapping[str, list[str]],
) -> str | None:
    labels = _labels_for_joint(joint, part_semantics)
    text = " ".join(labels).lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    if (
        any(phrase in text for phrase in _PULL_OUT_PHRASES)
        or any(word in words for word in _PULL_OUT_WORDS)
        or any(phrase in text for phrase in _PULL_OUT_CJK)
    ):
        return "positive_only"
    return None


def _labels_for_joint(
    joint: Mapping[str, object],
    part_semantics: Mapping[str, list[str]],
) -> list[str]:
    labels: list[str] = []
    for part in joint.get("moving_parts", ()):
        labels.extend(part_semantics.get(str(part), ()))
    return labels
