"""Compile and validate the agent-maintained estimate_limit.py artifact."""

from __future__ import annotations

import ast
import runpy
from pathlib import Path
from typing import Any

from .candidate_file import extract_editable_code
from .compile_signals import signal_bundle_from_messages
from .schemas import CandidateReport, EstimateContext, LimitEstimate


class CandidateCompileError(RuntimeError):
    """Raised when the estimate_limit.py editable artifact is invalid."""


def compile_candidate_report(
    candidate_path: Path,
    ctx: EstimateContext,
) -> CandidateReport:
    candidate_path = candidate_path.resolve()
    full_code = candidate_path.read_text()
    _assert_required_function(full_code)
    try:
        compile(full_code, str(candidate_path), "exec")
    except SyntaxError as exc:
        raise CandidateCompileError(f"syntax error line {exc.lineno}: {exc.msg}") from exc

    try:
        globals_dict = runpy.run_path(str(candidate_path), run_name="__ks_candidate__")
    except Exception as exc:
        raise CandidateCompileError(f"candidate import failed: {type(exc).__name__}: {exc}") from exc

    estimate_fn = globals_dict.get("estimate_limits")
    if not callable(estimate_fn):
        raise CandidateCompileError("estimate_limit.py must define callable estimate_limits(ctx)")

    try:
        raw = estimate_fn(ctx)
    except Exception as exc:
        raise CandidateCompileError(f"estimate_limits failed: {type(exc).__name__}: {exc}") from exc

    estimates = _coerce_estimates(raw)
    errors = _validate_estimates(estimates, ctx)
    return CandidateReport(
        passed=not errors,
        estimates=estimates,
        errors=errors,
        details={
            "compile_signals": signal_bundle_from_messages(
                errors=errors,
                source="compiler",
            ).to_dict(),
        },
    )


def _assert_required_function(full_code: str) -> None:
    editable = extract_editable_code(full_code)
    try:
        tree = ast.parse(editable)
    except SyntaxError:
        return
    has_estimate = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "estimate_limits"
        for node in tree.body
    )
    if not has_estimate:
        raise CandidateCompileError(
            "estimate_limit.py editable section must define estimate_limits(ctx)"
        )


def _coerce_estimates(raw: Any) -> list[LimitEstimate]:
    if not isinstance(raw, list):
        raise CandidateCompileError("estimate_limits(ctx) must return list[LimitEstimate]")
    estimates = []
    for item in raw:
        if isinstance(item, LimitEstimate):
            estimates.append(item)
        elif isinstance(item, dict):
            try:
                estimates.append(LimitEstimate(**item))
            except Exception as exc:
                raise CandidateCompileError(f"invalid estimate dict: {exc}") from exc
        else:
            raise CandidateCompileError(
                "estimate_limits(ctx) items must be LimitEstimate or dict"
            )
    return estimates


def _validate_estimates(estimates: list[LimitEstimate], ctx: EstimateContext) -> list[str]:
    errors = []
    seen = set()
    for estimate in estimates:
        if estimate.joint_name in seen:
            errors.append(f"{estimate.joint_name}: duplicate estimate")
        seen.add(estimate.joint_name)
        if estimate.joint_name not in ctx.joints:
            errors.append(f"{estimate.joint_name}: joint not present in context")
            continue
        errors.extend(_validate_axis_override(estimate, ctx))
    missing = set(ctx.joints) - seen
    if missing:
        errors.append(f"missing estimates for joints: {sorted(missing)}")
    return errors


def _validate_axis_override(estimate: LimitEstimate, ctx: EstimateContext) -> list[str]:
    if estimate.axis_world is None:
        return []
    if _unit_axis(estimate.axis_world) is None:
        return [f"{estimate.joint_name}: axis_world must be a non-zero 3-vector"]
    return []


def _unit_axis(raw: Any) -> tuple[float, float, float] | None:
    if raw is None or len(raw) != 3:
        return None
    axis = tuple(float(value) for value in raw)
    norm = sum(value * value for value in axis) ** 0.5
    if norm <= 1e-12:
        return None
    return tuple(value / norm for value in axis)
