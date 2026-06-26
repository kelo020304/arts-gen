"""Per-direction GT comparison for predicted joint ranges."""

from __future__ import annotations

from .config import ComparisonConfig


def _direction_metrics(
    predicted_value,
    gt_value: float,
    gt_range: float,
    status: str,
    config: ComparisonConfig,
) -> dict:
    if status != "ok" or predicted_value is None:
        return {"abs_err": None, "rel_err": None, "success": None}
    abs_err = abs(float(predicted_value) - float(gt_value))
    rel_err = abs_err / max(abs(gt_range), 1e-6)
    return {
        "abs_err": abs_err,
        "rel_err": rel_err,
        "success": rel_err < config.success_rel_err_threshold,
    }


def compare(
    predicted: dict,
    gt: dict,
    config: ComparisonConfig | None = None,
) -> dict:
    cfg = config or ComparisonConfig()
    gt_lower = float(gt["lower"])
    gt_upper = float(gt["upper"])
    gt_range = gt_upper - gt_lower

    lower = _direction_metrics(
        predicted.get("predicted_lower"),
        gt_lower,
        gt_range,
        predicted.get("status_lower"),
        cfg,
    )
    upper = _direction_metrics(
        predicted.get("predicted_upper"),
        gt_upper,
        gt_range,
        predicted.get("status_upper"),
        cfg,
    )
    result = {
        "object_id": predicted["object_id"],
        "joint_name": predicted["joint_name"],
        "type": predicted["type"],
        "canonical_unit": predicted["canonical_unit"],
        "predicted_lower": predicted.get("predicted_lower"),
        "predicted_upper": predicted.get("predicted_upper"),
        "gt_lower": gt_lower,
        "gt_upper": gt_upper,
        "status": predicted["status"],
        "status_lower": predicted.get("status_lower"),
        "status_upper": predicted.get("status_upper"),
        "abs_err_lower": lower["abs_err"],
        "rel_err_lower": lower["rel_err"],
        "success_lower": lower["success"],
        "abs_err_upper": upper["abs_err"],
        "rel_err_upper": upper["rel_err"],
        "success_upper": upper["success"],
    }

    if predicted.get("status_lower") == "ok" and predicted.get("status_upper") == "ok":
        pred_lower = float(predicted["predicted_lower"])
        pred_upper = float(predicted["predicted_upper"])
        inter = max(0.0, min(pred_upper, gt_upper) - max(pred_lower, gt_lower))
        union = max(pred_upper, gt_upper) - min(pred_lower, gt_lower)
        result["iou_range"] = inter / max(union, 1e-6)
        result["success"] = bool(lower["success"] and upper["success"])
    else:
        result["iou_range"] = None
        result["success"] = None
    return result
