from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


SMALL_BOUNDARY = 500
MEDIUM_BOUNDARY = 3000
SIZE_GAP_THRESHOLD = 10.0
MULTI_PART_THRESHOLD = 6
EVAL_REQUIRED_FILES = ("summary.json", "part_metrics.jsonl", "object_metrics.jsonl")
EVAL_OPTIONAL_FILES = ("selected_examples.json", "plots", "report.md")
TEST_REQUIRED_FILES = ("index.json",)


METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "target_iou": {
        "label": "Target IoU",
        "formula": "union(pred target part voxels) ∩ union(gt target part voxels) / union(pred target part voxels, gt target part voxels)",
        "meaning": "衡量目标部件整体区域是否对齐，越高越好。",
    },
    "part_iou": {
        "label": "Part IoU",
        "formula": "mean_i IoU(pred part_i voxels, gt part_i voxels)",
        "meaning": "逐部件计算 IoU 后平均，能反映小部件被平均后的表现。",
    },
    "recall": {
        "label": "Recall",
        "formula": "overlap_voxels / gt_voxels",
        "meaning": "GT 部件有多少被预测覆盖，低说明漏检。",
    },
    "precision": {
        "label": "Precision",
        "formula": "overlap_voxels / pred_voxels",
        "meaning": "预测体素有多少落在 GT 上，低说明多预测或串到别的区域。",
    },
    "f1": {
        "label": "F1",
        "formula": "2 * precision * recall / (precision + recall)",
        "meaning": "综合 Precision 和 Recall，任一很低都会拉低。",
    },
    "count_error": {
        "label": "数量误差",
        "formula": "mean(abs(pred_count - raw_count) / max(raw_count, 1))",
        "meaning": "预测体素数量和 GT 体素数量的相对偏差，越低越好。",
    },
    "empty_rate": {
        "label": "空预测率",
        "formula": "count(parts with pred_count == 0) / count(parts)",
        "meaning": "多少部件完全没预测出来，越低越好。",
    },
    "confusion": {
        "label": "部件混淆",
        "formula": "mean(assignment_offdiag_max)",
        "meaning": "预测部件与非对应 GT 部件的最大重叠，越低表示串部件越少。",
    },
    "binding_diag": {
        "label": "绑定 diag IoU",
        "formula": "mean(assignment_diag_iou)",
        "meaning": "预测部件与其【对应】GT 部件的 IoU（分配矩阵对角），越高越好——衡量每个部件是否认领了自己的区域（身份绑定正确）。与 confusion（off-diag）配对看：diag↑ / off-diag↓ 才是绑定真的好。",
    },
    "small_binding_diag": {
        "label": "小部件绑定 diag",
        "formula": "mean(assignment_diag_iou for parts with raw_count < 500)",
        "meaning": "只看小部件所在物体的对角 IoU。小件最容易因身份被大件尺寸淹没而绑错，越高越好。",
    },
    "small_confusion": {
        "label": "小部件混淆",
        "formula": "mean(assignment_offdiag_max for parts with raw_count < 500)",
        "meaning": "只看小部件所在物体的 off-diag。越低越好；若整体 confusion 低但这个高，说明身份在小件上被尺寸掩盖（diagnose 的核心信号）。",
    },
    "small_recall": {
        "label": "小部件 Recall",
        "formula": "mean(recall for parts with raw_count < 500)",
        "meaning": "只看小部件的召回，衡量按钮、把手、锁等小区域漏检情况。",
    },
    "small_empty_rate": {
        "label": "小部件空预测率",
        "formula": "count(small parts with pred_count == 0) / count(small parts)",
        "meaning": "小部件完全空预测的比例，越低越好。",
    },
    "multi_target_iou": {
        "label": "多部件 Target",
        "formula": "mean(Target IoU for objects with object_part_count >= 6)",
        "meaning": "只看多部件物体的整体目标区域表现。",
    },
    "multi_worst_part_iou": {
        "label": "多部件最差 Part",
        "formula": "mean(min part_iou per object) for objects with object_part_count >= 6",
        "meaning": "多部件物体里最差部件的平均表现，能暴露被平均值掩盖的失败部件。",
    },
    "size_gap_target_iou": {
        "label": "大小差距 Target",
        "formula": f"mean(Target IoU for objects with size_mix_ratio >= {SIZE_GAP_THRESHOLD:g})",
        "meaning": "只看同一物体内大小部件差距很大的样本。",
    },
    "scale_rel_error": {
        "label": "相对尺度误差",
        "formula": "mean_obj Σ_i |Vᵢ^pred/ΣV^pred − Vᵢ^gt/ΣV^gt|",
        "meaning": "各部件体素体积归一化后的分布 L1 距离，直接衡量部件之间的相对大小是否保持（越低越好，0=相对尺寸完美；漏掉某个部件也会被罚）。",
    },
    "size_ratio_error": {
        "label": "大小比误差",
        "formula": "mean_obj |log(V_max/V_min)^pred − log(V_max/V_min)^gt|",
        "meaning": "物体内最大/最小部件体积比的对数误差，专测大小差距大时模型是否偏向大部件、把小部件做小（越低越好）。",
    },
    "interior_recall": {
        "label": "内部 recall",
        "formula": "mean_part |pred ∩ 内部GT| / |内部GT|；内部GT = 6 邻居全被占据的体素",
        "meaning": "只在被自身完全包住的内部体素上算 recall，直接衡量模型对看不见的内部 / 遮挡区域的补全完整度（越高越好）。",
    },
}


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def inspect_eval_report(report_dir: Path) -> dict[str, Any]:
    return _inspect_files(
        report_dir,
        required=EVAL_REQUIRED_FILES,
        optional=EVAL_OPTIONAL_FILES,
        root_label="report_dir",
    )


def inspect_test_export(export_dir: Path) -> dict[str, Any]:
    return _inspect_files(
        export_dir,
        required=TEST_REQUIRED_FILES,
        optional=(),
        root_label="export_dir",
    )


def _inspect_files(
    root: Path,
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    root_label: str,
) -> dict[str, Any]:
    missing = [rel for rel in required if not (root / rel).is_file()]
    optional_missing = [rel for rel in optional if not (root / rel).exists()]
    diagnostics = {
        "status": "incomplete" if missing else "ok",
        "message": "ok",
        "missing": missing,
        "optional_missing": optional_missing,
        "errors": [],
        root_label: str(root),
        "expected": list(required),
    }
    return _finish_diagnostics(diagnostics)


def _finish_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    if diagnostics.get("missing"):
        diagnostics["status"] = "incomplete"
        diagnostics["message"] = "缺少必需指标文件，当前只显示可解析内容。"
    elif diagnostics.get("errors"):
        diagnostics["status"] = "incomplete"
        diagnostics["message"] = "部分指标文件解析失败，当前只显示可解析内容。"
    else:
        diagnostics["message"] = "ok"
    return diagnostics


def _record_parse_error(diagnostics: dict[str, Any], path: Path, message: str) -> None:
    diagnostics.setdefault("errors", []).append({"path": path.name, "message": message})
    diagnostics["status"] = "incomplete"


def _read_json_tolerant(path: Path, default: Any, diagnostics: dict[str, Any]) -> Any:
    if not path.is_file():
        return default
    try:
        return read_json(path, default)
    except Exception as exc:
        _record_parse_error(diagnostics, path, str(exc))
        return default


def _read_jsonl_tolerant(path: Path, diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        _record_parse_error(diagnostics, path, str(exc))
        return rows
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            diagnostics.setdefault("errors", []).append({
                "path": f"{path.name}:{lineno}",
                "message": str(exc),
            })
            diagnostics["status"] = "incomplete"
    return rows


def _selected_examples_tolerant(report_dir: Path, diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return _selected_examples(report_dir)
    except Exception as exc:
        _record_parse_error(diagnostics, report_dir / "selected_examples.json", str(exc))
        return []


def _num(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _group_part_rows_by_object(part_rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in part_rows:
        key = (str(row.get("obj_id", "")), int(row.get("dataset_index", -1)))
        groups.setdefault(key, []).append(row)
    return list(groups.values())


def _pred_count(row: dict[str, Any]) -> int:
    return max(0, int(row.get("pred_count", 0) or 0))


def _rel_volume_l1(rows: list[dict[str, Any]]) -> float:
    """L1 distance between normalized pred/gt part-volume distributions for one object."""
    pred = [_pred_count(row) for row in rows]
    gt = [max(0, part_raw_count(row)) for row in rows]
    total_gt = sum(gt)
    if total_gt <= 0:
        return float("nan")
    total_pred = sum(pred)
    ratio_gt = [g / total_gt for g in gt]
    ratio_pred = [p / total_pred for p in pred] if total_pred > 0 else [0.0] * len(pred)
    return float(sum(abs(a - b) for a, b in zip(ratio_pred, ratio_gt)))


def _size_ratio_logerr(rows: list[dict[str, Any]]) -> float:
    """|log(Vmax/Vmin)_pred - log(Vmax/Vmin)_gt| over parts that exist in GT."""
    gt = [max(0, part_raw_count(row)) for row in rows]
    keep = [i for i, g in enumerate(gt) if g > 0]
    if len(keep) < 2:
        return float("nan")
    gt_pos = [gt[i] for i in keep]
    pred_pos = [max(1, _pred_count(rows[i])) for i in keep]  # floor 1: empty pred -> large ratio error
    lr_gt = math.log(max(gt_pos) / min(gt_pos))
    lr_pred = math.log(max(pred_pos) / min(pred_pos))
    return abs(lr_pred - lr_gt)


def _metric(label_key: str, value: float, *, count: int | None = None) -> dict[str, Any]:
    definition = METRIC_DEFINITIONS[label_key]
    out = {
        "key": label_key,
        "label": definition["label"],
        "value": value,
        "formula": definition["formula"],
        "meaning": definition["meaning"],
    }
    if count is not None:
        out["count"] = count
    return out


def part_raw_count(row: dict[str, Any]) -> int:
    return int(row.get("part_raw_voxel_count", row.get("raw_ind_count", 0)) or 0)


def part_iou(row: dict[str, Any]) -> float:
    return _num(row.get("part_iou", row.get("decode_iou_pred_vs_raw_ind")))


def part_recall(row: dict[str, Any]) -> float:
    return _num(row.get("part_recall", row.get("decode_recall_pred_vs_raw_ind")))


def part_precision(row: dict[str, Any]) -> float:
    return _num(row.get("part_precision", row.get("decode_precision_pred_vs_raw_ind")))


def size_bucket(raw_count: int) -> str:
    if raw_count < SMALL_BOUNDARY:
        return "small"
    if raw_count < MEDIUM_BOUNDARY:
        return "medium"
    return "large"


def _object_part_count(row: dict[str, Any]) -> int:
    return int(row.get("object_part_count", row.get("parts", 0)) or 0)


def _target_iou(row: dict[str, Any]) -> float:
    return _num(row.get("target_parts_iou_pred_vs_gt", row.get("target_iou")))


def _target_recall(row: dict[str, Any]) -> float:
    return _num(row.get("target_parts_recall_pred_vs_gt", row.get("target_recall")))


def _target_precision(row: dict[str, Any]) -> float:
    return _num(row.get("target_parts_precision_pred_vs_gt", row.get("target_precision")))


def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    if math.isnan(precision) or math.isnan(recall) or denom <= 0:
        return 0.0
    return 2.0 * precision * recall / denom


def _selected_examples(report_dir: Path) -> list[dict[str, Any]]:
    examples = read_json(report_dir / "selected_examples.json", [])
    return examples if isinstance(examples, list) else []


def _plots(report_dir: Path) -> list[dict[str, str]]:
    plots_root = report_dir / "plots"
    if not plots_root.is_dir():
        return []
    return [
        {"name": path.name, "path": str(path.relative_to(report_dir))}
        for path in sorted(plots_root.glob("*.png"))
    ]


def load_eval_metrics(report_dir: Path, *, tolerant: bool = False) -> dict[str, Any]:
    diagnostics = inspect_eval_report(report_dir) if tolerant else {}
    if tolerant:
        summary = _read_json_tolerant(report_dir / "summary.json", {}, diagnostics) or {}
        part_rows = _read_jsonl_tolerant(report_dir / "part_metrics.jsonl", diagnostics)
        object_rows = _read_jsonl_tolerant(report_dir / "object_metrics.jsonl", diagnostics)
    else:
        summary = read_json(report_dir / "summary.json", {}) or {}
        part_rows = read_jsonl(report_dir / "part_metrics.jsonl")
        object_rows = read_jsonl(report_dir / "object_metrics.jsonl")

    part_ious = [part_iou(row) for row in part_rows]
    recalls = [part_recall(row) for row in part_rows]
    precisions = [part_precision(row) for row in part_rows]
    recall = _safe_mean(recalls)
    precision = _safe_mean(precisions)
    count_errors = []
    empty_count = 0
    confusion = []
    binding_diag = []
    by_size: dict[str, list[dict[str, Any]]] = {"small": [], "medium": [], "large": []}
    for row in part_rows:
        raw_count = max(part_raw_count(row), 0)
        pred_count = int(row.get("pred_count", 0) or 0)
        count_errors.append(abs(pred_count - raw_count) / max(raw_count, 1))
        if pred_count == 0:
            empty_count += 1
        confusion.append(_num(row.get("assignment_offdiag_max")))
        binding_diag.append(_num(row.get("assignment_diag_iou")))
        by_size[size_bucket(raw_count)].append(row)

    # Relative-scale metrics — computed per object from existing per-part voxel
    # counts (no eval re-run needed). Directly measures cross-part size fidelity.
    object_part_groups = _group_part_rows_by_object(part_rows)
    scale_rel_error = _safe_mean(_rel_volume_l1(group) for group in object_part_groups)
    size_ratio_error = _safe_mean(_size_ratio_logerr(group) for group in object_part_groups)
    # Interior recall (amodal completion of hidden voxels) — computed per part at
    # eval time; older runs without it simply skip (NaN -> excluded from mean).
    interior_recall = _safe_mean(_num(row.get("interior_recall")) for row in part_rows)

    target_values = [_target_iou(row) for row in object_rows]
    overall_summary = summary.get("overall", {}) if isinstance(summary, dict) else {}
    target_iou = _num(overall_summary.get("target_parts_iou_mean"), _safe_mean(target_values))
    mean_part_iou = _num(overall_summary.get("part_iou_mean"), _safe_mean(part_ious))

    multi_objects = [row for row in object_rows if _object_part_count(row) >= MULTI_PART_THRESHOLD]
    gap_objects = [
        row for row in object_rows
        if _num(row.get("size_mix_ratio")) >= SIZE_GAP_THRESHOLD
    ]
    small_parts = by_size["small"]

    size_buckets = {
        "small": {
            "label": "小部件",
            "definition": f"raw_count < {SMALL_BOUNDARY}",
            "parts": len(by_size["small"]),
            "part_iou": _safe_mean(part_iou(row) for row in by_size["small"]),
            "recall": _safe_mean(part_recall(row) for row in by_size["small"]),
            "precision": _safe_mean(part_precision(row) for row in by_size["small"]),
        },
        "medium": {
            "label": "中部件",
            "definition": f"{SMALL_BOUNDARY} <= raw_count < {MEDIUM_BOUNDARY}",
            "parts": len(by_size["medium"]),
            "part_iou": _safe_mean(part_iou(row) for row in by_size["medium"]),
            "recall": _safe_mean(part_recall(row) for row in by_size["medium"]),
            "precision": _safe_mean(part_precision(row) for row in by_size["medium"]),
        },
        "large": {
            "label": "大部件",
            "definition": f"raw_count >= {MEDIUM_BOUNDARY}",
            "parts": len(by_size["large"]),
            "part_iou": _safe_mean(part_iou(row) for row in by_size["large"]),
            "recall": _safe_mean(part_recall(row) for row in by_size["large"]),
            "precision": _safe_mean(part_precision(row) for row in by_size["large"]),
        },
    }

    result = {
        "task_kind": "eval",
        "summary": summary,
        "overall": {
            "target_iou": _metric("target_iou", target_iou, count=len(object_rows)),
            "part_iou": _metric("part_iou", mean_part_iou, count=len(part_rows)),
            "recall": _metric("recall", recall, count=len(part_rows)),
            "precision": _metric("precision", precision, count=len(part_rows)),
            "f1": _metric("f1", _f1(precision, recall), count=len(part_rows)),
            "count_error": _metric("count_error", _safe_mean(count_errors), count=len(part_rows)),
            "empty_rate": _metric("empty_rate", empty_count / len(part_rows) if part_rows else float("nan"), count=len(part_rows)),
            "confusion": _metric("confusion", _safe_mean(confusion), count=len(part_rows)),
            "binding_diag": _metric("binding_diag", _safe_mean(binding_diag), count=len(part_rows)),
            "scale_rel_error": _metric("scale_rel_error", scale_rel_error, count=len(object_part_groups)),
            "size_ratio_error": _metric("size_ratio_error", size_ratio_error, count=len(object_part_groups)),
            "interior_recall": _metric("interior_recall", interior_recall, count=len(part_rows)),
        },
        "focused": {
            "small_recall": _metric("small_recall", _safe_mean(part_recall(row) for row in small_parts), count=len(small_parts)),
            "small_empty_rate": _metric(
                "small_empty_rate",
                sum(1 for row in small_parts if int(row.get("pred_count", 0) or 0) == 0) / len(small_parts)
                if small_parts else float("nan"),
                count=len(small_parts),
            ),
            "small_binding_diag": _metric(
                "small_binding_diag",
                _safe_mean(_num(row.get("assignment_diag_iou")) for row in small_parts),
                count=len(small_parts),
            ),
            "small_confusion": _metric(
                "small_confusion",
                _safe_mean(_num(row.get("assignment_offdiag_max")) for row in small_parts),
                count=len(small_parts),
            ),
            "multi_target_iou": _metric("multi_target_iou", _safe_mean(_target_iou(row) for row in multi_objects), count=len(multi_objects)),
            "multi_worst_part_iou": _metric(
                "multi_worst_part_iou",
                _safe_mean(_num(row.get("part_iou_min")) for row in multi_objects),
                count=len(multi_objects),
            ),
            "size_gap_target_iou": _metric("size_gap_target_iou", _safe_mean(_target_iou(row) for row in gap_objects), count=len(gap_objects)),
        },
        "size_buckets": size_buckets,
        "metric_definitions": METRIC_DEFINITIONS,
        "examples": _selected_examples_tolerant(report_dir, diagnostics) if tolerant else _selected_examples(report_dir),
        "plots": _plots(report_dir),
    }
    if tolerant:
        diagnostics = _finish_diagnostics(diagnostics)
        if diagnostics["status"] != "ok":
            result["diagnostics"] = diagnostics
    return result


def load_test_metrics(export_dir: Path, *, tolerant: bool = False) -> dict[str, Any]:
    diagnostics = inspect_test_export(export_dir) if tolerant else {}
    if tolerant:
        index = _read_json_tolerant(export_dir / "index.json", {}, diagnostics) or {}
    else:
        index = read_json(export_dir / "index.json", {}) or {}
    examples = index.get("examples", []) if isinstance(index, dict) else []
    parts = 0
    for example in examples:
        if isinstance(example, dict):
            parts += len(example.get("parts", []) or [])
    result = {
        "task_kind": "test",
        "overall": {
            "examples": {
                "key": "examples",
                "label": "导出样本",
                "value": len(examples),
                "formula": "len(index.json.examples)",
                "meaning": "Test/export 产出的样本数量。",
            },
            "parts": {
                "key": "parts",
                "label": "导出部件",
                "value": parts,
                "formula": "sum(len(example.parts))",
                "meaning": "已打包给下游 SLAT 的部件数量。",
            },
        },
        "index": index,
    }
    if tolerant:
        diagnostics = _finish_diagnostics(diagnostics)
        if diagnostics["status"] != "ok":
            result["diagnostics"] = diagnostics
    return result
