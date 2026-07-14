"""Run GT-free kinematic self-refinement for every decoded moving component."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .run_kin_agent import _semantic_part_category, _semantic_type_hint
from .sdk import (
    KinematicAgentConfig,
    DEFAULT_AXIS_FAMILY_MODEL,
    DEFAULT_RANGE_PRIOR,
    DEFAULT_STATIC_AXIS_FAMILY_MODEL,
    DEFAULT_STATIC_DINO_AXIS_MODEL,
    apply_axis_family_reranker,
    apply_phyx_door_contact_axis_critic,
    apply_phyx_knob_thin_axis_critic,
    apply_static_axis_family_reranker,
    apply_static_dino_door_axis_reranker,
    audit_decoded_bundle_collisions,
    calibrate_range_candidate,
    delivery_joint_payload,
    export_decoded_mesh_obj,
    estimate_motion_hypotheses_from_render_states,
    estimate_static_part_observation,
    infer_kinematics,
    load_mesh_points,
    load_range_prior,
    pool_static_part_dino_feature,
    propose_collision_clear_interval,
    write_kinematic_bundle_mjcf,
    write_kinematic_bundle_usda,
)


LID_TYPE_SECONDARY_RATIO_THRESHOLD = 0.10
LID_TYPE_UNCERTAINTY_MARGIN = 0.03
FORBIDDEN_DELIVERY_INPUT_TOKENS = (
    "/raw/partseg/",
    "/reconstruction/part_info/",
    "/joint_transforms/",
    "/source/model/",
)


def _assert_decoded_delivery_input(path: Path) -> None:
    normalized = str(Path(path).resolve()).replace("\\", "/").lower()
    if any(token in normalized for token in FORBIDDEN_DELIVERY_INPUT_TOKENS):
        raise ValueError(
            "Kin Agent delivery accepts decoded SLat assets only; "
            f"forbidden GT/source input path: {path}"
        )


def _safe_name(value: str, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    if not name:
        name = fallback
    if name[0].isdigit():
        name = f"{fallback}_{name}"
    return name[:72]


def run_bundle(
    summary_path: Path,
    out_dir: Path,
    *,
    max_iterations: int = 7,
    dataset_id: str | None = None,
    motion_observation_root: Path | None = None,
    static_observation_root: Path | None = None,
    static_view_indices: tuple[int, ...] | list[int] = (0, 3, 8, 11),
) -> dict:
    if not 1 <= int(max_iterations) < 10:
        raise ValueError("max_iterations must be in [1, 9]")
    geometry_iterations = max(1, int(max_iterations) - 2)
    evidence_trace_limit = geometry_iterations + 4
    summary_path = Path(summary_path).resolve()
    _assert_decoded_delivery_input(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = list(summary.get("parts") or [])
    body = next((item for item in records if item.get("kind") == "body" and item.get("mesh_path")), None)
    moving = [item for item in records if item.get("kind") != "body" and item.get("mesh_path")]
    if body is None or not moving:
        raise ValueError("SLat summary must contain one decoded body mesh and at least one decoded moving part mesh")
    body_source = Path(str(body["mesh_path"])).resolve()
    _assert_decoded_delivery_input(body_source)
    for item in moving:
        _assert_decoded_delivery_input(Path(str(item["mesh_path"])).resolve())
    object_name = _resolve_object_name(summary, summary_path, dataset_id)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    _write_json(status_path, {"state": "running", "progress": 2, "message": "Loading decoded SLat meshes"})
    assets_dir = out_dir / "assets"
    body_obj = export_decoded_mesh_obj(body_source, assets_dir / "body.obj")
    static_points = load_mesh_points(body_source)
    parts: list[dict] = []
    result_rows: list[dict] = []
    force_prismatic_local_z = str(dataset_id or "").lower() == "realappliance"
    observation_input_files: list[Path] = []
    for index, item in enumerate(moving, start=1):
        label = str(item.get("label") or f"part_{index:02d}")
        source_mesh = Path(str(item["mesh_path"])).resolve()
        part_name = _safe_name(label, f"part_{index:02d}")
        part_obj = export_decoded_mesh_obj(source_mesh, assets_dir / f"{part_name}.obj")
        type_hint = _semantic_type_hint(label)
        part_category = _semantic_part_category(label)
        moving_points = load_mesh_points(source_mesh)
        result = infer_kinematics(
            moving_points,
            static_points,
            config=KinematicAgentConfig(max_iterations=geometry_iterations),
            joint_type_hint=type_hint,
            part_category=part_category,
            dataset_profile=dataset_id,
        )
        geometry_interval = [result.candidate.lower, result.candidate.upper]
        geometry_result = result
        motion_observation = None
        static_observation = None
        motion_hypotheses = {}
        if (
            motion_observation_root is not None
            and Path(motion_observation_root).is_dir()
        ):
            motion_hypotheses = estimate_motion_hypotheses_from_render_states(
                Path(motion_observation_root), label,
            )
            motion_observation = _select_motion_observation(
                result, motion_hypotheses, part_category=part_category,
            )
        if (
            motion_observation is None
            and static_observation_root is not None
            and Path(static_observation_root).is_dir()
        ):
            static_observation = estimate_static_part_observation(
                Path(static_observation_root), label,
            )
            if static_observation is not None:
                observation_input_files.extend(Path(path) for path in static_observation.input_files)
        if motion_observation is not None and part_category != "knob":
            observation_input_files.extend(Path(path) for path in motion_observation.input_files)
            result = _apply_motion_observation(
                result, motion_observation, max_iterations=evidence_trace_limit,
                part_category=part_category,
            )
        elif motion_observation is not None:
            observation_input_files.extend(Path(path) for path in motion_observation.input_files)
        axis_family_model = None
        static_axis_family_model = None
        static_dino_door_axis_model = None
        phyx_thin_axis_critic = None
        phyx_door_contact_axis_critic = None
        if str(dataset_id or "").lower() == "realappliance" and part_category == "knob":
            result, axis_family_model = apply_axis_family_reranker(
                result,
                label=label,
                body_points=static_points,
                moving_points=moving_points,
                observation=motion_observation or static_observation,
                max_iterations=evidence_trace_limit,
            )
            if axis_family_model is None:
                result, phyx_thin_axis_critic = apply_phyx_knob_thin_axis_critic(
                    result,
                    dataset_id=dataset_id,
                    part_category=part_category,
                    moving_points=moving_points,
                    max_iterations=evidence_trace_limit,
                    min_confidence=0.95,
                    max_score_drop=0.15,
                    allowed_dataset_ids=("realappliance",),
                )
        elif (
            str(dataset_id or "").lower() == "realappliance"
            and part_category == "door"
            and motion_observation is None
            and static_observation_root is not None
        ):
            dino_feature = pool_static_part_dino_feature(
                Path(static_observation_root), label,
                view_indices=static_view_indices,
            )
            if dino_feature is not None:
                observation_input_files.extend(Path(path) for path in dino_feature.input_files)
            result, static_dino_door_axis_model = apply_static_dino_door_axis_reranker(
                result,
                dino_feature=dino_feature,
                max_iterations=evidence_trace_limit,
            )
        elif (
            str(dataset_id or "").lower() == "realappliance"
            and part_category == "lid"
            and motion_observation is None
        ):
            result, static_axis_family_model = apply_static_axis_family_reranker(
                result,
                label=label,
                category=part_category,
                body_points=static_points,
                moving_points=moving_points,
                static_observation=static_observation,
                max_iterations=evidence_trace_limit,
            )
        elif str(dataset_id or "").lower() == "phyx-verse" and part_category == "knob":
            result, phyx_thin_axis_critic = apply_phyx_knob_thin_axis_critic(
                result,
                dataset_id=dataset_id,
                part_category=part_category,
                moving_points=moving_points,
                max_iterations=evidence_trace_limit,
            )
        elif (
            str(dataset_id or "").lower() == "phyx-verse"
            and part_category == "door"
            and motion_observation is None
        ):
            result, phyx_door_contact_axis_critic = apply_phyx_door_contact_axis_critic(
                result,
                dataset_id=dataset_id,
                part_category=part_category,
                body_points=static_points,
                moving_points=moving_points,
                max_iterations=evidence_trace_limit,
            )
        range_prior = load_range_prior(
            dataset_id, part_category, result.candidate.joint_type,
        )
        range_calibration = None
        if range_prior is not None:
            result = _apply_range_prior(
                result, range_prior, max_iterations=evidence_trace_limit,
                object_diagonal=_object_diagonal(static_points, moving_points),
                observation_diagnostics=(
                    motion_observation.diagnostics if motion_observation is not None else None
                ),
            )
        result = _consolidate_evidence_trace(
            result,
            geometry_result=geometry_result,
            max_iterations=max_iterations,
        )
        range_calibration = _last_range_calibration(result)
        parts.append({
            "part_id": item.get("part_id", index), "label": label,
            "body_name": part_name, "joint_name": f"{part_name}_joint",
            "mesh": part_obj, "source_mesh": source_mesh, "candidate": result.candidate,
        })
        candidate_payload = asdict(result.candidate)
        signals = result.candidate.signals
        range_censored = float(signals.get("range_censored", 0.0)) >= 0.5
        observed_state_span = float(signals.get("motion_observed_span", 0.0)) > 0.0
        motion_observation_noisy = (
            observed_state_span and float(signals.get("motion_observation_confidence", 0.0)) < 0.8
        )
        review_reasons = []
        if float(signals.get("type_confidence", 0.0)) < 0.8:
            review_reasons.append("type ambiguity")
        if float(signals.get("axis_confidence", 0.0)) < 0.8:
            review_reasons.append("axis ambiguity")
        if float(signals.get("motion_type_uncertain", 0.0)) >= 0.5:
            review_reasons.append("motion type near line/arc boundary")
        if phyx_thin_axis_critic and phyx_thin_axis_critic.get("review_required"):
            review_reasons.append(str(phyx_thin_axis_critic["review_reason"]))
        if phyx_door_contact_axis_critic and phyx_door_contact_axis_critic.get("review_required"):
            review_reasons.append(str(phyx_door_contact_axis_critic["review_reason"]))
        if range_censored:
            review_reasons.append(
                "mechanical stop not confirmed" if observed_state_span else "mechanical stop not observed"
            )
        delivery_payload = delivery_joint_payload(result.candidate, force_prismatic_local_z)
        result_rows.append({
            "part_id": item.get("part_id", index), "label": label,
            "source_mesh": str(source_mesh), "decoded_obj": str(part_obj),
            "iterations": result.iterations, "candidate": candidate_payload,
            "delivery_candidate": delivery_payload,
            "range_estimate": {
                "status": (
                    "observed_state_span_noisy+learned_calibration" if motion_observation_noisy and range_calibration
                    else "observed_state_span+learned_calibration" if observed_state_span and range_calibration
                    else "learned_interval_censored" if range_calibration
                    else "observed_state_span_noisy" if motion_observation_noisy
                    else "observed_state_span" if observed_state_span
                    else "censored" if range_censored else "geometry_collision_bound"
                ),
                "tested_collision_free_interval": geometry_interval,
                "observed_state_interval": (
                    range_calibration.get("observed_state_interval") if range_calibration
                    else [
                        float(signals.get("motion_observed_lower", 0.0)),
                        float(signals.get("motion_observed_upper", signals["motion_observed_span"])),
                    ] if observed_state_span else None
                ),
                "estimated_usable_interval": (
                    range_calibration.get("estimated_usable_interval") if range_calibration
                    else [float(result.candidate.lower), float(result.candidate.upper)]
                ),
                "prediction_interval": (
                    range_calibration.get("prediction_interval") if range_calibration else None
                ),
                "mechanical_stop_confirmed": bool(
                    range_calibration.get("mechanical_stop_confirmed", False)
                    if range_calibration else signals.get("mechanical_stop_confirmed", 0.0)
                ),
                "learned_prior": range_prior.to_dict() if range_prior is not None else None,
                "prior_applied": bool(range_calibration),
                "export_fallback_interval": [delivery_payload["lower"], delivery_payload["upper"]],
                "confidence": float(signals.get("range_confidence", 0.0)),
            },
            "requires_review": bool(review_reasons),
            "review_reasons": review_reasons,
            "motion_observation": motion_observation.to_dict() if motion_observation is not None else None,
            "static_observation": static_observation.to_dict() if static_observation is not None else None,
            "motion_observation_hypotheses": {
                key: value.to_dict() for key, value in motion_hypotheses.items()
            },
            "axis_family_model": axis_family_model,
            "static_axis_family_model": static_axis_family_model,
            "static_dino_door_axis_model": static_dino_door_axis_model,
            "phyx_knob_thin_axis_critic": phyx_thin_axis_critic,
            "phyx_door_contact_axis_critic": phyx_door_contact_axis_critic,
            "trace": result.trace,
        })
        _write_json(status_path, {
            "state": "running", "progress": 10 + int(75 * index / len(moving)),
            "message": f"Solved {label}", "completed_parts": index, "total_parts": len(moving),
        })
    _write_json(status_path, {
        "state": "running", "progress": 88,
        "message": "Auditing decoded mesh motion and part interference",
        "completed_parts": len(moving), "total_parts": len(moving),
    })
    collision_audit = audit_decoded_bundle_collisions(body_obj, parts)
    audit_by_label = {item["label"]: item for item in collision_audit["per_joint"]}
    accepted_collision_revision = False
    for feedback_index, (part, row) in enumerate(zip(parts, result_rows, strict=True), start=1):
        _write_json(status_path, {
            "state": "running",
            "progress": 88 + int(6 * (feedback_index - 1) / max(1, len(parts))),
            "message": f"Exact collision feedback for {row['label']}",
            "completed_parts": len(moving),
            "total_parts": len(moving),
            "collision_feedback_part": feedback_index,
        })
        audit = audit_by_label[str(row["label"])]
        feedback = _apply_decoded_collision_feedback(
            body_obj,
            part,
            row,
            audit,
            max_iterations=max_iterations,
            force_prismatic_local_z=force_prismatic_local_z,
        )
        accepted_collision_revision = accepted_collision_revision or bool(
            feedback.get("accept_gate", {}).get("accepted")
        )
    if accepted_collision_revision:
        collision_audit = audit_decoded_bundle_collisions(body_obj, parts)
        audit_by_label = {item["label"]: item for item in collision_audit["per_joint"]}
    collision_audit_path = out_dir / "decoded_collision_audit.json"
    _write_json(collision_audit_path, collision_audit)
    for row in result_rows:
        audit = audit_by_label[str(row["label"])]
        row["collision_audit"] = audit
        if audit["requires_review"]:
            if audit["collision_detected"]:
                reason = "decoded mesh collision across predicted range"
            else:
                reason = "decoded collision audit unverified"
            if reason not in row["review_reasons"]:
                row["review_reasons"].append(reason)
            row["requires_review"] = True
    apply_root_correction = str(dataset_id or "").lower() == "realappliance"
    xml_path = write_kinematic_bundle_mjcf(
        out_dir / "object.xml", object_name=object_name, body_mesh=body_obj, parts=parts,
        force_prismatic_local_z=force_prismatic_local_z,
        apply_root_correction=apply_root_correction,
    )
    usd_path = write_kinematic_bundle_usda(
        out_dir / "object.usda", object_name=object_name, body_mesh=body_source, parts=parts,
        force_prismatic_local_z=force_prismatic_local_z,
        apply_root_correction=apply_root_correction,
    )
    validation = _run_mujoco_validation(xml_path, out_dir, expected_joints=len(parts))
    payload = {
        "format": "arts_gen_kin_agent_v17",
        "input_contract": (
            "decoded SLat meshes plus optional calibrated multi-state 2D boxes/cameras; "
            "no GT mesh, joint annotations, joint transforms, or source USD joint fields"
        ),
        "summary_path": str(summary_path), "object_name": object_name,
        "max_iterations": max_iterations, "dataset_id": dataset_id,
        "force_prismatic_local_z": force_prismatic_local_z,
        "apply_root_correction": apply_root_correction,
        "input_files": [_file_stamp(summary_path), _file_stamp(body_source)]
        + [_file_stamp(Path(str(item["mesh_path"])).resolve()) for item in moving]
        + [_file_stamp(path) for path in dict.fromkeys(observation_input_files)]
        + ([_file_stamp(DEFAULT_RANGE_PRIOR)] if DEFAULT_RANGE_PRIOR.is_file() else [])
        + ([_file_stamp(DEFAULT_AXIS_FAMILY_MODEL)] if DEFAULT_AXIS_FAMILY_MODEL.is_file() else [])
        + ([_file_stamp(DEFAULT_STATIC_AXIS_FAMILY_MODEL)] if DEFAULT_STATIC_AXIS_FAMILY_MODEL.is_file() else [])
        + ([_file_stamp(DEFAULT_STATIC_DINO_AXIS_MODEL)] if DEFAULT_STATIC_DINO_AXIS_MODEL.is_file() else []),
        "prior_files": [_file_stamp(DEFAULT_RANGE_PRIOR)] if DEFAULT_RANGE_PRIOR.is_file() else [],
        "axis_family_model_file": (
            _file_stamp(DEFAULT_AXIS_FAMILY_MODEL) if DEFAULT_AXIS_FAMILY_MODEL.is_file() else None
        ),
        "static_axis_family_model_file": (
            _file_stamp(DEFAULT_STATIC_AXIS_FAMILY_MODEL)
            if DEFAULT_STATIC_AXIS_FAMILY_MODEL.is_file() else None
        ),
        "static_dino_axis_model_file": (
            _file_stamp(DEFAULT_STATIC_DINO_AXIS_MODEL)
            if DEFAULT_STATIC_DINO_AXIS_MODEL.is_file() else None
        ),
        "motion_observation_root": str(Path(motion_observation_root).resolve()) if motion_observation_root else None,
        "static_observation_root": str(Path(static_observation_root).resolve()) if static_observation_root else None,
        "static_view_indices": [int(value) for value in static_view_indices],
        "evidence_mode": "dataset_motion_states" if motion_observation_root else "static_decoded_geometry",
        "body_source_mesh": str(body_source),
        "body_decoded_obj": str(body_obj), "parts": result_rows,
        "xml_path": str(xml_path), "usd_path": str(usd_path),
        "validation": validation,
        "collision_audit": collision_audit,
        "collision_audit_path": str(collision_audit_path),
    }
    _write_json(out_dir / "kinematic_result.json", payload)
    validation_ok = bool(validation.get("ok"))
    delivery_needs_review = bool(collision_audit.get("requires_review"))
    _write_json(status_path, {
        "state": "complete" if validation_ok and not delivery_needs_review else "needs_review",
        "progress": 100,
        "message": (
            "Combined XML and USD written; physical validation passed"
            if validation_ok and not delivery_needs_review
            else "Combined XML and USD written; decoded collision or physical validation needs review"
        ),
        "completed_parts": len(moving), "total_parts": len(moving),
    })
    return payload


def _apply_decoded_collision_feedback(
    body_mesh,
    part: dict,
    row: dict,
    audit: dict,
    *,
    max_iterations: int,
    force_prismatic_local_z: bool,
) -> dict:
    candidate = part["candidate"]
    if audit.get("collision_detected"):
        feedback = propose_collision_clear_interval(
            body_mesh,
            part["mesh"],
            candidate,
            audit,
        )
        feedback = _gate_collision_revision_by_axis_evidence(feedback, candidate)
    else:
        feedback = {
            "version": "decoded_collision_feedback_v1",
            "status": "clear" if not audit.get("requires_review") else "review",
            "candidate_interval": [float(candidate.lower), float(candidate.upper)],
            "proposal": None,
            "retained_fraction": 1.0,
            "accept_gate": {
                "accepted": False,
                "retained_fraction_ok": True,
                "observed_motion_preserved": True,
                "reasons": [],
            },
            "requires_review": bool(audit.get("requires_review")),
            "review_reason": (
                "decoded_collision_audit_unverified" if audit.get("requires_review") else None
            ),
            "exact_evaluations": [],
        }
    accepted = bool(feedback.get("accept_gate", {}).get("accepted"))
    if accepted:
        proposal = feedback["proposal"]
        candidate = replace(
            candidate,
            lower=float(proposal["lower"]),
            upper=float(proposal["upper"]),
            signals={
                **candidate.signals,
                "decoded_collision_feedback_used": 1.0,
                "decoded_collision_retained_fraction": float(feedback["retained_fraction"]),
            },
            reason=candidate.reason + "; range revised by exact decoded-collision feedback",
        )
        part["candidate"] = candidate
        row["candidate"] = asdict(candidate)
        row["delivery_candidate"] = delivery_joint_payload(candidate, force_prismatic_local_z)
        row["range_estimate"]["estimated_usable_interval"] = [
            float(candidate.lower), float(candidate.upper),
        ]
        row["range_estimate"]["collision_revised_interval"] = [
            float(candidate.lower), float(candidate.upper),
        ]
        row["range_estimate"]["export_fallback_interval"] = [
            float(row["delivery_candidate"]["lower"]),
            float(row["delivery_candidate"]["upper"]),
        ]
    row["collision_feedback"] = feedback
    trace = list(row.get("trace") or [])
    if len(trace) < int(max_iterations):
        trace.append({
            "iteration": len(trace) + 1,
            "stage": "decoded_collision_feedback",
            "selected": asdict(candidate),
            "collision_feedback": feedback,
            "decision": {
                "action": "accept_range_revision" if accepted else "keep_incumbent",
                "stop_reason": (
                    "exact_collision_clear_interval_accepted"
                    if accepted else feedback.get("review_reason") or feedback.get("status")
                ),
            },
            "stop_reason": (
                "exact_collision_clear_interval_accepted"
                if accepted else feedback.get("review_reason") or feedback.get("status")
            ),
        })
    row["trace"] = trace
    row["iterations"] = len(trace)
    if feedback.get("requires_review"):
        reason = "decoded collision feedback rejected: " + str(
            feedback.get("review_reason") or feedback.get("status")
        )
        if reason not in row["review_reasons"]:
            row["review_reasons"].append(reason)
        row["requires_review"] = True
    return feedback


def _gate_collision_revision_by_axis_evidence(feedback: dict, candidate) -> dict:
    if not feedback.get("accept_gate", {}).get("accepted"):
        return feedback
    signals = candidate.signals
    independent = any(float(signals.get(key, 0.0)) >= 0.5 for key in (
        "motion_observation_used",
        "motion_axis_family_used",
        "motion_type_axis_used",
        "phyx_thin_axis_used",
        "decoded_knob_thin_axis_used",
        "phyx_door_contact_axis_used",
    ))
    model_only = any(float(signals.get(key, 0.0)) >= 0.5 for key in (
        "axis_family_model_used",
        "static_visual_axis_model_used",
        "static_dino_door_axis_used",
    )) and not independent
    weak_axis = float(signals.get("axis_confidence", 0.0)) < 0.8
    if not model_only and not weak_axis:
        return feedback
    reasons = list(feedback["accept_gate"].get("reasons") or [])
    if model_only:
        reasons.append("axis_family_model_not_independently_observed")
    if weak_axis:
        reasons.append("axis_confidence_below_auto_revision_threshold")
    feedback = dict(feedback)
    feedback["status"] = "rejected"
    feedback["requires_review"] = True
    feedback["review_reason"] = ";".join(reasons)
    feedback["accept_gate"] = {
        **feedback["accept_gate"],
        "accepted": False,
        "independent_axis_evidence": independent,
        "reasons": reasons,
    }
    return feedback


def _consolidate_evidence_trace(result, *, geometry_result, max_iterations: int):
    """Represent motion/model/range critics as one bounded evidence-fusion round."""
    geometry_trace = list(geometry_result.trace)
    post_rows = list(result.trace[len(geometry_trace):])
    if not post_rows:
        return geometry_result
    if len(geometry_trace) >= int(max_iterations):
        return geometry_result
    substeps = []
    for row in post_rows:
        substeps.append({
            key: row[key]
            for key in (
                "stage", "observation", "axis_family_model", "thin_axis_critic",
                "door_contact_axis_critic", "static_axis_family_model",
                "static_dino_door_axis_model",
                "range_calibration", "selected",
            )
            if key in row
        })
    fused = {
        "iteration": len(geometry_trace) + 1,
        "stage": "evidence_fusion_critic",
        "selected": asdict(result.candidate),
        "substeps": substeps,
        "alternatives": [row.get("selected") for row in post_rows if row.get("selected")][:6],
        "stop_reason": "evidence_fused",
    }
    range_calibration = next((
        row.get("range_calibration")
        for row in reversed(post_rows)
        if isinstance(row.get("range_calibration"), dict)
    ), None)
    if range_calibration is not None:
        fused["range_calibration"] = range_calibration
    trace = [*geometry_trace, fused]
    return replace(result, iterations=len(trace), trace=trace)


def _select_motion_observation(result, hypotheses, *, part_category: str | None):
    """Select a line/circle hypothesis without consulting joint annotations."""
    current = hypotheses.get(result.candidate.joint_type)
    if str(part_category or "").lower() != "lid":
        return current
    linear = hypotheses.get("prismatic")
    circular = hypotheses.get("revolute")
    if linear is None or circular is None:
        return current
    ratio = float(linear.diagnostics.get("trajectory_secondary_ratio", float("inf")))
    selected = linear if ratio < LID_TYPE_SECONDARY_RATIO_THRESHOLD else circular
    margin = abs(ratio - LID_TYPE_SECONDARY_RATIO_THRESHOLD)
    type_confidence = float(np.clip(0.5 + 4.0 * margin, 0.5, 0.98))
    return replace(selected, diagnostics={
        **selected.diagnostics,
        "motion_type_classifier_used": 1.0,
        "motion_type_secondary_ratio": ratio,
        "motion_type_secondary_ratio_threshold": LID_TYPE_SECONDARY_RATIO_THRESHOLD,
        "motion_type_margin": margin,
        "motion_type_confidence": type_confidence,
        "motion_type_uncertain": 1.0 if margin < LID_TYPE_UNCERTAINTY_MARGIN else 0.0,
        "motion_type_selected_prismatic": 1.0 if selected.joint_type == "prismatic" else 0.0,
    })


def _cardinal_family(axis) -> int:
    return int(np.argmax(np.abs(np.asarray(axis, dtype=np.float64))))


def _existing_cardinal_axis(result, family: int, joint_type: str):
    ranked = []
    for row in result.trace:
        for raw in [row.get("selected"), *(row.get("alternatives") or [])]:
            if not raw or raw.get("joint_type") != joint_type or not raw.get("axis_world"):
                continue
            axis = np.asarray(raw["axis_world"], dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12 or _cardinal_family(axis) != family:
                continue
            axis /= norm
            ranked.append((abs(float(axis[family])), float(raw.get("score", 0.0)), axis))
    if not ranked:
        return None
    return max(ranked, key=lambda item: (item[0], item[1]))[2]


def _apply_motion_observation(
    result, observation, *, max_iterations: int, part_category: str | None = None,
):
    """Use motion evidence as a bounded critic over the geometry proposal."""
    if observation.observed_span <= 1e-4:
        return result
    candidate = result.candidate
    observed_axis = np.asarray(observation.axis_world, dtype=np.float64)
    geometry_axis = np.asarray(candidate.axis_world, dtype=np.float64)
    category = str(part_category or "").lower()
    type_classifier_used = float(
        observation.diagnostics.get("motion_type_classifier_used", 0.0)
    ) >= 0.5
    type_changed = type_classifier_used and observation.joint_type != candidate.joint_type
    type_confidence = float(observation.diagnostics.get(
        "motion_type_confidence", candidate.signals.get("type_confidence", 0.0),
    ))
    geometry_family = _cardinal_family(geometry_axis)
    observed_family = _cardinal_family(observed_axis)
    family_disagrees = geometry_family != observed_family
    base_axis_confidence = float(candidate.signals.get("axis_confidence", 0.0))
    family_axis = (
        _existing_cardinal_axis(result, observed_family, candidate.joint_type)
        if category == "door" and family_disagrees else None
    )
    family_gate = family_axis is not None and (
        observation.confidence >= 0.35 or base_axis_confidence < 0.5
    )
    full_motion_update = observation.confidence >= 0.45 or type_changed
    type_axis_gate = type_classifier_used and (
        observation.confidence >= 0.35 or type_changed
    )
    if not full_motion_update and not family_gate and not type_classifier_used:
        return result
    lower = float(observation.diagnostics.get("observed_lower", 0.0))
    upper = float(observation.diagnostics.get("observed_upper", observation.observed_span))
    if family_gate:
        # The ordered trajectory supplies the range sign.  Do not align it to
        # the rejected geometry family; only orient the existing cardinal
        # candidate to the observed motion axis.
        if float(family_axis @ observed_axis) < 0.0:
            family_axis = -family_axis
    elif float(observed_axis @ geometry_axis) < 0.0:
        observed_axis = -observed_axis
        lower, upper = -upper, -lower
    span = float(upper - lower)
    selected_axis = (
        family_axis if family_gate
        else observed_axis if full_motion_update or type_axis_gate
        else geometry_axis
    )
    selected_type = observation.joint_type if type_classifier_used else candidate.joint_type
    interval_update = full_motion_update or family_gate
    origin = (
        observation.origin_world
        if full_motion_update and selected_type == "revolute"
        else candidate.origin_world
    )
    selected_lower = lower if interval_update else candidate.lower
    selected_upper = upper if interval_update else candidate.upper
    selected_score = (
        max(candidate.score, min(0.99, 0.75 + 0.20 * observation.confidence))
        if full_motion_update else candidate.score
    )
    axis_confidence = max(
        float(candidate.signals.get("axis_confidence", 0.0)),
        float(observation.confidence),
    )
    if family_gate and not full_motion_update:
        axis_confidence = min(axis_confidence, max(0.45, float(observation.confidence)))
    refined = replace(
        candidate,
        joint_type=selected_type,
        axis_world=tuple(float(value) for value in selected_axis),
        origin_world=tuple(float(value) for value in origin),
        lower=selected_lower,
        upper=selected_upper,
        score=selected_score,
        signals={
            **candidate.signals,
            "motion_observation_used": 1.0 if full_motion_update else 0.0,
            "motion_observation_confidence": float(observation.confidence),
            "motion_observation_states": float(observation.state_count),
            "motion_observed_span": span if interval_update else 0.0,
            "motion_observed_lower": lower if interval_update else 0.0,
            "motion_observed_upper": upper if interval_update else 0.0,
            "motion_range_orientation_used": 1.0 if family_gate else 0.0,
            "motion_axis_family_disagreement": 1.0 if family_disagrees else 0.0,
            "motion_axis_family_used": 1.0 if family_gate else 0.0,
            "motion_axis_family": float(observed_family),
            "axis_confidence": axis_confidence,
            "type_confidence": type_confidence if type_classifier_used else float(
                candidate.signals.get("type_confidence", 0.0)
            ),
            "motion_type_classifier_used": 1.0 if type_classifier_used else 0.0,
            "motion_type_secondary_ratio": float(observation.diagnostics.get(
                "motion_type_secondary_ratio", 0.0,
            )),
            "motion_type_threshold": float(observation.diagnostics.get(
                "motion_type_secondary_ratio_threshold", 0.0,
            )),
            "motion_type_margin": float(observation.diagnostics.get("motion_type_margin", 0.0)),
            "motion_type_uncertain": float(observation.diagnostics.get("motion_type_uncertain", 0.0)),
            "motion_type_changed": 1.0 if type_changed else 0.0,
            "motion_type_axis_used": 1.0 if type_axis_gate else 0.0,
            "range_confidence": max(
                float(candidate.signals.get("range_confidence", 0.0)),
                min(0.75, 0.45 + 0.30 * observation.confidence),
            ) if full_motion_update else float(candidate.signals.get("range_confidence", 0.0)),
            # Multiple states show a covered interval, not necessarily a hard stop.
            "range_censored": 1.0 if interval_update else float(
                candidate.signals.get("range_censored", 0.0)
            ),
        },
        reason=candidate.reason + (
            "; reranked to an existing cardinal family from calibrated motion observations"
            if family_gate and not full_motion_update
            else "; refined from calibrated multi-state 2D motion observations"
        ),
    )
    trace = list(result.trace)
    row = {
        "iteration": min(max_iterations, len(trace) + 1),
        "stage": "motion_axis_family_critic" if family_gate and not full_motion_update else "motion_observation_critic",
        "selected": asdict(refined),
        "observation": observation.to_dict(),
        "alternatives": [asdict(candidate)],
    }
    if len(trace) < max_iterations:
        trace.append(row)
    else:
        trace[-1] = row
    return replace(result, candidate=refined, iterations=len(trace), trace=trace)


def _apply_range_prior(
    result,
    prior,
    *,
    max_iterations: int,
    object_diagonal: float | None = None,
    observation_diagnostics: dict | None = None,
):
    """Apply an evidence-gated frozen interval strategy."""
    candidate = result.candidate
    refined, calibration = calibrate_range_candidate(
        candidate, prior,
        object_diagonal=object_diagonal,
        observation_diagnostics=observation_diagnostics,
    )
    if calibration.get("applied"):
        trace = list(result.trace)
        row = {
            "iteration": min(max_iterations, len(trace) + 1),
            "stage": "range_calibration_critic",
            "selected": asdict(refined),
            "range_prior": prior.to_dict(),
            "range_calibration": calibration,
            "alternatives": [asdict(candidate)],
        }
        if len(trace) < max_iterations:
            trace.append(row)
        else:
            trace[-1] = row
        return replace(result, candidate=refined, iterations=len(trace), trace=trace)
    if prior.strategy != "legacy_span":
        return result

    # Compatibility for explicitly loaded v1 artifacts only.
    observed_span = float(candidate.signals.get("motion_observed_span", 0.0))
    # State coverage is a lower-bound observation, but noisy circle fits can
    # overestimate it.  Keep the train-only median as the center estimate and
    # truncate observation evidence at the frozen q90 uncertainty bound.
    fallback_span = max(float(prior.median), min(observed_span, float(prior.q90)))
    if observed_span > 1e-8:
        observed_lower = float(candidate.signals.get("motion_observed_lower", candidate.lower))
        observed_upper = float(candidate.signals.get("motion_observed_upper", candidate.upper))
        if observed_lower >= -1e-6:
            lower, upper = 0.0, fallback_span
        elif observed_upper <= 1e-6:
            lower, upper = -fallback_span, 0.0
        else:
            scale = fallback_span / observed_span
            lower, upper = observed_lower * scale, observed_upper * scale
    elif candidate.upper > 0.0:
        lower, upper = 0.0, fallback_span
    else:
        lower, upper = -fallback_span, 0.0
    refined = replace(
        candidate,
        lower=lower,
        upper=upper,
        signals={
            **candidate.signals,
            "range_prior_used": 1.0,
            "range_prior_q10": float(prior.q10),
            "range_prior_q50": float(prior.median),
            "range_prior_q90": float(prior.q90),
            "range_prior_objects": float(prior.n_objects),
            "range_confidence": max(
                float(candidate.signals.get("range_confidence", 0.0)),
                float(prior.confidence),
            ),
            "range_censored": 1.0,
        },
        reason=candidate.reason + "; calibrated with frozen train-object travel prior",
    )
    trace = list(result.trace)
    row = {
        "iteration": min(max_iterations, len(trace) + 1),
        "stage": "range_prior_critic",
        "selected": asdict(refined),
        "range_prior": prior.to_dict(),
        "alternatives": [asdict(candidate)],
    }
    if len(trace) < max_iterations:
        trace.append(row)
    else:
        trace[-1] = row
    return replace(result, candidate=refined, iterations=len(trace), trace=trace)


def _last_range_calibration(result) -> dict | None:
    for row in reversed(result.trace):
        calibration = row.get("range_calibration")
        if isinstance(calibration, dict) and calibration.get("applied"):
            return calibration
    return None


def _object_diagonal(static_points: np.ndarray, moving_points: np.ndarray) -> float:
    minimum = np.minimum(np.min(static_points, axis=0), np.min(moving_points, axis=0))
    maximum = np.maximum(np.max(static_points, axis=0), np.max(moving_points, axis=0))
    return float(np.linalg.norm(maximum - minimum))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**payload, "updated_unix": time.time()}, indent=2), encoding="utf-8")


def _file_stamp(path: Path) -> dict:
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _resolve_object_name(summary: dict, summary_path: Path, dataset_id: str | None) -> str:
    explicit = str(summary.get("object_name") or "").strip()
    if explicit:
        return _safe_name(explicit, "decoded_object")
    session_root = summary_path.parents[3]
    for metadata_name in ("dataset.json", "manifest.json"):
        metadata_path = session_root / metadata_name
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dataset = metadata.get("dataset") if isinstance(metadata.get("dataset"), dict) else metadata
        object_id = str(dataset.get("object_id") or dataset.get("object") or "").strip()
        if object_id:
            prefix = str(dataset_id or dataset.get("dataset_id") or "object").strip()
            return _safe_name(f"{prefix}_{object_id}", "decoded_object")
    return _safe_name(session_root.name, "decoded_object")


def _run_mujoco_validation(xml_path: Path, out_dir: Path, *, expected_joints: int) -> dict:
    validator = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "post" / "validate_mujoco_rd.py"
    python = Path(sys.executable)
    if importlib.util.find_spec("mujoco") is None:
        candidate = Path("/opt/venvs/arts-gen/bin/python")
        if candidate.is_file():
            python = candidate
    report_path = out_dir / "mujoco_validation.json"
    image_path = out_dir / "mujoco_qpos_compare.png"
    command = [
        str(python), str(validator), "--xml", str(xml_path),
        "--out-json", str(report_path), "--out-png", str(image_path),
        "--expect-nq", str(expected_joints), "--expect-nv", str(expected_joints),
        "--expect-nu", "0", "--expect-njnt", str(expected_joints),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    try:
        process = subprocess.run(
            command, cwd=str(Path(__file__).resolve().parents[2]), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "available": False, "error": f"{type(exc).__name__}: {exc}"}
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    return {
        "ok": bool(report.get("ok")), "available": "error" not in (report.get("mujoco") or {}),
        "python": str(python), "return_code": process.returncode,
        "report_path": str(report_path) if report_path.is_file() else None,
        "image_path": str(image_path) if image_path.is_file() else None,
        "stderr_tail": process.stderr[-1000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=7)
    parser.add_argument("--dataset-id")
    parser.add_argument("--motion-observation-root", type=Path)
    parser.add_argument("--static-observation-root", type=Path)
    parser.add_argument("--static-view-indices", default="0,3,8,11")
    args = parser.parse_args()
    try:
        payload = run_bundle(
            args.summary_json, args.out_dir,
            max_iterations=args.max_iterations, dataset_id=args.dataset_id,
            motion_observation_root=args.motion_observation_root,
            static_observation_root=args.static_observation_root,
            static_view_indices=tuple(
                int(value) for value in str(args.static_view_indices).split(",") if value.strip()
            ),
        )
    except Exception as exc:
        _write_json(args.out_dir / "status.json", {
            "state": "failed", "progress": 0, "message": str(exc),
        })
        raise
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
