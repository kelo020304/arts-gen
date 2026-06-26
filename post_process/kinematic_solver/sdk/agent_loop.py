"""Controlled API loop that may edit only estimate_limit.py's user region."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import unified_diff
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .candidate_file import extract_editable_code, replace_editable_code
from .compile_signals import (
    render_compile_signal_bundle,
    signal_bundle_from_messages,
)
from .compiler import CandidateCompileError, compile_candidate_report
from .live_viewer import LiveViewer, editable_region_sha256
from .schemas import CandidateReport, EstimateContext, LimitEstimate

AgentRequestFn = Callable[[list[dict[str, str]]], str]
ReportCallback = Callable[[int, CandidateReport], dict[str, Any] | None]
ReportValidator = Callable[[CandidateReport], CandidateReport]
DEFAULT_API_TIMEOUT_SECONDS = 600.0
MAX_PROMPT_CODE_CHARS = 5000
PRISMATIC_STEP_MM = (10.0, 5.0, 2.5, 1.0, 0.5)
REVOLUTE_STEP_DEGREES = (10.0, 5.0, 2.5, 1.0, 0.5)
COMPACT_EDITABLE_TEMPLATE = '''def estimate_limits(ctx):
    """Return one bounded-action LimitEstimate per joint in ctx.joints."""
    import math

    evidence = getattr(ctx, "evidence", {}) or {}
    joints = getattr(ctx, "joints", {}) or {}
    estimates = []

    def norm(raw):
        vals = [float(v) for v in (raw or [1.0, 0.0, 0.0])]
        n = math.sqrt(sum(v * v for v in vals))
        return [v / n for v in vals] if n > 1e-12 else [1.0, 0.0, 0.0]

    for name in sorted(joints):
        joint = joints[name]
        ev = evidence.get(name, {}) or {}
        init = ev.get("initial_estimate") or {}
        axis = norm(init.get("axis_world", joint.get("axis_world", [1.0, 0.0, 0.0])))
        lower = float(init.get("lower", joint.get("lower", 0.0)) or 0.0)
        upper = float(init.get("upper", joint.get("upper", 0.0)) or 0.0)
        if not init and str(joint.get("type", "")).lower() == "prismatic" and abs(upper) <= 1e-12:
            upper = 0.1
        estimates.append(LimitEstimate(
            joint_name=name,
            lower=lower,
            upper=upper,
            axis_world=axis,
            confidence=0.35,
            reason="compact seed: start from VLM/context estimate; update one bounded action per iteration from feedback",
        ))
    return estimates
'''


class AgentLoopError(RuntimeError):
    """Raised when the controlled agent loop cannot call or parse the provider."""


@dataclass(frozen=True)
class AgentLoopConfig:
    max_iterations: int = 3
    provider: str = "openrouter"
    model: str = "gpt-5.5"
    thinking_level: str = "high"
    base_url: str = "https://api-router.evad.mioffice.cn/v1"
    api_key: str | None = None
    timeout_seconds: float = DEFAULT_API_TIMEOUT_SECONDS
    heartbeat_interval_seconds: float = 2.0


@dataclass(frozen=True)
class AgentLoopResult:
    report: CandidateReport
    iterations: int


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat completions client."""

    def __init__(self, config: AgentLoopConfig):
        self.config = config

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.config.api_key:
            raise AgentLoopError(
                "missing API key: set OPENROUTER_API_KEY or OPENAI_API_KEY"
            )
        started = time.monotonic()
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            elapsed = time.monotonic() - started
            response_body = _safe_http_error_body(exc)
            message = (
                f"API request failed: HTTP {exc.code} after {elapsed:.1f}s "
                f"(provider={self.config.provider}, model={self.config.model}, "
                f"base_url={_safe_base_url(url)}, request_chars={_message_chars(messages)})"
            )
            if exc.code in {502, 503, 504}:
                message += (
                    ". This is an upstream/proxy timeout or gateway failure; "
                    "try a faster model/lower thinking level or rerun when the router is healthy"
                )
            if response_body:
                message += f". Response body: {response_body}"
            raise AgentLoopError(message) from exc
        except TimeoutError as exc:
            raise AgentLoopError(
                f"API request timed out after {self.config.timeout_seconds:g}s while waiting for model response"
            ) from exc
        except urllib.error.URLError as exc:
            raise AgentLoopError(f"API request failed: {exc.reason}") from exc
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentLoopError("API response missing choices[0].message.content") from exc


def resolve_agent_loop_config(
    *,
    max_iterations: int,
    timeout_seconds: float = DEFAULT_API_TIMEOUT_SECONDS,
    heartbeat_interval_seconds: float = 2.0,
) -> AgentLoopConfig:
    openrouter_key = _first_env_key("OPENROUTER_API_KEY", "OPENROUTER_API_KEYS")
    openai_key = _first_env_key("OPENAI_API_KEY", "OPENAI_API_KEYS")
    if openrouter_key:
        return AgentLoopConfig(
            max_iterations=max_iterations,
            provider="openrouter",
            model=os.environ.get("ARTICRAFT_MODEL") or "gpt-5.5",
            thinking_level=os.environ.get("ARTICRAFT_THINKING_LEVEL") or "high",
            base_url=os.environ.get("OPENROUTER_BASE_URL")
            or "https://api-router.evad.mioffice.cn/v1",
            api_key=openrouter_key,
            timeout_seconds=timeout_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
    return AgentLoopConfig(
        max_iterations=max_iterations,
        provider="openai",
        model=os.environ.get("ARTICRAFT_MODEL") or "gpt-5.5",
        thinking_level=os.environ.get("ARTICRAFT_THINKING_LEVEL") or "high",
        base_url=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        api_key=openai_key,
        timeout_seconds=timeout_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )


def extract_editable_response_code(response_text: str) -> str:
    if "# >>> USER_CODE_START" in response_text and "# >>> USER_CODE_END" in response_text:
        return extract_editable_code(response_text).strip()
    fenced = re.search(
        r"```(?:python|py)?\s*\n(?P<code>.*?)```",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        return fenced.group("code").strip()
    return response_text.strip()


def extract_action_response_json(response_text: str) -> dict[str, Any]:
    """Extract the Articraft-style structured action JSON from a model response."""
    text = response_text.strip()
    fenced = re.search(
        r"```(?:json)?\s*\n(?P<json>.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        text = fenced.group("json").strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentLoopError(f"agent response must be JSON action only: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AgentLoopError("agent response JSON must be an object")
    return parsed


def apply_action_update(
    previous: CandidateReport,
    action: dict[str, Any],
    ctx: EstimateContext,
) -> CandidateReport:
    """Apply one bounded structured action per joint to the current estimates."""
    updates = action.get("updates", action)
    if not isinstance(updates, dict):
        return CandidateReport(
            passed=False,
            estimates=list(previous.estimates),
            errors=["agent action JSON must contain object field 'updates'"],
        )
    previous_by_joint = {estimate.joint_name: estimate for estimate in previous.estimates}
    errors: list[str] = []
    updated: list[LimitEstimate] = []
    for joint_name in sorted(previous_by_joint):
        before = previous_by_joint[joint_name]
        raw_update = updates.get(joint_name, {})
        if raw_update in (None, False):
            raw_update = {}
        if not isinstance(raw_update, dict):
            errors.append(f"{joint_name}: update must be an object")
            raw_update = {}
        try:
            updated.append(_apply_joint_action(before, raw_update, ctx))
        except (TypeError, ValueError) as exc:
            errors.append(f"{joint_name}: {exc}")
            updated.append(before)

    unknown = sorted(set(updates) - set(previous_by_joint))
    for joint_name in unknown:
        errors.append(f"{joint_name}: action references unknown joint")

    candidate = CandidateReport(
        passed=not errors,
        estimates=updated,
        errors=errors,
        warnings=_action_warnings(action),
        details={"agent_action": action},
    )
    if not errors:
        budget_errors = incremental_action_errors(previous, candidate, ctx=ctx)
        if budget_errors:
            candidate = CandidateReport(
                passed=False,
                estimates=updated,
                errors=budget_errors,
                warnings=list(candidate.warnings),
                details=dict(candidate.details),
            )
    return candidate


def render_estimate_limits_from_report(report: CandidateReport) -> str:
    """Render deterministic editable Python from the current structured estimates."""
    rows: list[dict[str, Any]] = []
    for estimate in sorted(report.estimates, key=lambda item: item.joint_name):
        rows.append(
            {
                "joint_name": estimate.joint_name,
                "lower": float(estimate.lower),
                "upper": float(estimate.upper),
                "axis_world": (
                    [float(value) for value in estimate.axis_world]
                    if estimate.axis_world is not None
                    else None
                ),
                "axis_label": estimate.axis_label,
                "confidence": estimate.confidence,
                "reason": estimate.reason,
            }
        )
    return (
        "def estimate_limits(ctx):\n"
        "    \"\"\"Deterministic artifact generated from structured agent actions.\"\"\"\n"
        f"    rows = {rows!r}\n"
        "    return [LimitEstimate(**row) for row in rows]\n"
    )


def _apply_joint_action(
    before: LimitEstimate,
    update: dict[str, Any],
    ctx: EstimateContext,
) -> LimitEstimate:
    joint_type = _joint_type(ctx, before.joint_name)
    axis = list(_unit_axis(before.axis_world or [1.0, 0.0, 0.0]) or (1.0, 0.0, 0.0))
    axis_label = before.axis_label
    lower = float(before.lower)
    upper = float(before.upper)

    axis_update = update.get("axis", update.get("axis_action"))
    if axis_update is not None:
        axis, axis_label = _apply_axis_action(
            before.joint_name,
            axis,
            axis_label,
            axis_update,
            ctx,
        )
    lower += _limit_delta_to_native(
        update.get("lower_delta", update.get("lower")),
        joint_type=joint_type,
        field="lower_delta",
    )
    upper += _limit_delta_to_native(
        update.get("upper_delta", update.get("upper")),
        joint_type=joint_type,
        field="upper_delta",
    )
    state = str(update.get("state", "") or "").strip()
    reason = str(update.get("reason", before.reason or "") or "")
    reason_bits = []
    if state:
        reason_bits.append(f"state={state}")
    if reason:
        reason_bits.append(reason)
    return LimitEstimate(
        joint_name=before.joint_name,
        lower=lower,
        upper=upper,
        axis_world=axis,
        axis_label=axis_label,
        confidence=before.confidence,
        reason="; ".join(reason_bits),
    )


def _apply_axis_action(
    joint_name: str,
    current_axis: list[float],
    current_label: str | None,
    raw_action: Any,
    ctx: EstimateContext,
) -> tuple[list[float], str | None]:
    if isinstance(raw_action, str):
        raw_action = {"op": raw_action}
    if not isinstance(raw_action, dict):
        raise ValueError("axis action must be an object")
    op = str(raw_action.get("op", "keep") or "keep").lower()
    if op in {"keep", "none", "noop"}:
        return current_axis, current_label
    if op in {"set_candidate", "candidate", "switch_candidate", "set", "set_axis"}:
        raise ValueError("axis action may only keep or rotate by at most 5 degrees from the SDK geometry baseline")
    if op == "rotate":
        axis = _unit_axis(raw_action.get("axis_world"))
        if axis is None:
            raise ValueError("axis rotate requires target axis_world")
        return list(axis), raw_action.get("axis_label") or current_label
    raise ValueError(f"unsupported axis op {op!r}")


def _axis_candidates_for_joint(
    ctx: EstimateContext | None,
    joint_name: str,
) -> list[tuple[tuple[float, float, float], str | None]]:
    if ctx is None:
        return []
    evidence = ctx.evidence.get(joint_name, {}) or {}
    candidates: list[tuple[tuple[float, float, float], str | None]] = []
    recommended = _unit_axis(evidence.get("recommended_axis_world"))
    if recommended is not None:
        candidates.append((recommended, "recommended"))
    raw_candidates = evidence.get("axis_candidates") or []
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            axis = _unit_axis(item.get("axis_world"))
            if axis is not None:
                label = item.get("axis_label") or item.get("label")
                candidates.append((axis, str(label) if label is not None else None))
    return candidates


def _limit_delta_to_native(raw_delta: Any, *, joint_type: str | None, field: str) -> float:
    if raw_delta in (None, "", False):
        return 0.0
    if isinstance(raw_delta, (int, float)):
        value = float(raw_delta)
        unit = "degree" if joint_type == "revolute" else "mm"
    elif isinstance(raw_delta, dict):
        value = float(raw_delta.get("value", 0.0) or 0.0)
        unit = str(raw_delta.get("unit") or ("degree" if joint_type == "revolute" else "mm")).lower()
    else:
        raise ValueError(f"{field} must be a number or object")
    if abs(value) <= 1e-12:
        return 0.0
    allowed = REVOLUTE_STEP_DEGREES if joint_type == "revolute" else PRISMATIC_STEP_MM
    if not any(abs(abs(value) - step) <= 1e-9 for step in allowed):
        allowed_text = ", ".join(str(step).rstrip("0").rstrip(".") for step in allowed)
        raise ValueError(f"{field} value {value:g} must be one of +/-{allowed_text}")
    if joint_type == "revolute":
        if unit not in {"degree", "degrees", "deg"}:
            raise ValueError(f"{field} for revolute joints must use degrees")
        return math.radians(value)
    if unit not in {"mm", "millimeter", "millimeters"}:
        raise ValueError(f"{field} for prismatic joints must use mm")
    return value / 1000.0


def _action_warnings(action: dict[str, Any]) -> list[str]:
    note = action.get("note") or action.get("warning")
    return [str(note)] if note else []


def _axis_close(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    *,
    dot_threshold: float,
) -> bool:
    return sum(a * b for a, b in zip(left, right, strict=True)) >= dot_threshold


def run_agent_loop(
    candidate_path: Path,
    ctx: EstimateContext,
    *,
    config: AgentLoopConfig,
    request_fn: AgentRequestFn | None = None,
    viewer: LiveViewer | None = None,
    on_report: ReportCallback | None = None,
    validate_report: ReportValidator | None = None,
) -> AgentLoopResult:
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    candidate_path = Path(candidate_path)
    request = request_fn or OpenAICompatibleClient(config).complete
    last_feedback = "No previous compile or validation feedback."
    last_report = _seed_report_from_context(ctx)
    last_feedback = (
        "VLM initial joint estimates are rough seed values. "
        "Do not mark all joints correct in the first action without changing "
        "at least one axis or limit and observing validation feedback."
    )
    acted_joints: set[str] = set()

    for iteration in range(1, config.max_iterations + 1):
        current_code = extract_editable_code(candidate_path.read_text())
        previous_report = last_report if last_report.estimates else _seed_report_from_context(ctx)
        details = {
            "candidate_path": str(candidate_path),
            "editable_sha256": editable_region_sha256(candidate_path),
            "provider": config.provider,
            "model": config.model,
            "loop_mode": "structured_action",
        }
        if viewer is not None:
            viewer.emit(
                "agent_iteration_started",
                iteration=iteration,
                status="running",
                phase="agent",
                details=details,
            )
        messages = build_action_messages(
            ctx,
            previous_report=previous_report,
            last_feedback=last_feedback,
            config=config,
        )
        if viewer is not None:
            viewer.emit(
                "api_call_started",
                iteration=iteration,
                status="running",
                phase="api",
                details={
                    "provider": config.provider,
                    "model": config.model,
                    "thinking_level": config.thinking_level,
                },
            )
        try:
            response_text = _request_with_heartbeat(
                request,
                messages,
                viewer=viewer,
                iteration=iteration,
                config=config,
            )
        except Exception as exc:
            if viewer is not None:
                viewer.emit(
                    "api_call_finished",
                    iteration=iteration,
                    status="failed",
                    phase="api",
                    errors=[str(exc)],
                )
            setattr(exc, "agent_iteration", iteration)
            raise
        if viewer is not None:
            viewer.emit(
                "api_call_finished",
                iteration=iteration,
                status="passed",
                phase="api",
                details={"response_chars": len(response_text)},
            )

        try:
            action = extract_action_response_json(response_text)
        except AgentLoopError as exc:
            last_feedback = str(exc)
            last_report = CandidateReport(
                passed=False,
                estimates=list(previous_report.estimates),
                errors=[last_feedback],
                details={
                    "compile_signals": signal_bundle_from_messages(
                        errors=[last_feedback],
                        source="agent",
                    ).to_dict(),
                },
            )
            if viewer is not None:
                viewer.emit_report(
                    "validation_finished",
                    last_report,
                    iteration=iteration,
                    phase="validation",
                    details={"raw_response": response_text[:2000]},
                )
            continue

        action_report = apply_action_update(previous_report, action, ctx)
        if viewer is not None:
            viewer.emit(
                "agent_action_applied",
                iteration=iteration,
                status="passed" if action_report.passed else "failed",
                phase="action",
                details={
                    "action": action,
                    "action_summary": _summarize_action(action),
                },
                errors=list(action_report.errors),
                warnings=list(action_report.warnings),
            )
        if not action_report.passed:
            last_report = CandidateReport(
                passed=False,
                estimates=action_report.estimates,
                errors=list(action_report.errors),
                warnings=list(action_report.warnings),
                details={
                    **action_report.details,
                    "compile_signals": signal_bundle_from_messages(
                        errors=list(action_report.errors),
                        warnings=list(action_report.warnings),
                        source="harness",
                    ).to_dict(),
                },
            )
            last_feedback = _feedback_from_report(last_report)
            if viewer is not None:
                viewer.emit_report(
                    "validation_finished",
                    last_report,
                    iteration=iteration,
                    phase="validation",
                    details={"required_estimates": sorted(ctx.joints)},
                )
            continue

        acted_joints.update(_changed_joints(previous_report, action_report))
        if _no_effective_estimate_change(previous_report, action_report):
            error = (
                "agent action made no effective change from the current VLM/candidate "
                "state; at least one bounded axis/lower/upper action is required "
                "before the run can pass"
            )
            last_report = CandidateReport(
                passed=False,
                estimates=action_report.estimates,
                errors=[error],
                warnings=list(action_report.warnings),
                details={
                    **action_report.details,
                    "compile_signals": signal_bundle_from_messages(
                        errors=[error],
                        warnings=list(action_report.warnings),
                        source="harness",
                    ).to_dict(),
                },
            )
            last_feedback = _feedback_from_report(last_report)
            if viewer is not None:
                viewer.emit_report(
                    "validation_finished",
                    last_report,
                    iteration=iteration,
                    phase="validation",
                    details={"required_estimates": sorted(ctx.joints)},
                )
            continue

        new_code = render_estimate_limits_from_report(action_report)
        editable_diff = "\n".join(unified_diff(
            current_code.strip().splitlines(),
            new_code.strip().splitlines(),
            fromfile="before/estimate_limit.py:editable",
            tofile="after/estimate_limit.py:editable",
            lineterm="",
        ))
        candidate_path.write_text(
            replace_editable_code(candidate_path.read_text(), new_code)
        )
        details = {
            "candidate_path": str(candidate_path),
            "editable_sha256": editable_region_sha256(candidate_path),
            "editable_code": new_code.strip(),
            "editable_diff": editable_diff,
            "loop_mode": "structured_action",
        }
        if viewer is not None:
            viewer.emit(
                "agent_edit_applied",
                iteration=iteration,
                status="passed",
                phase="edit",
                details=details,
            )
            viewer.emit(
                "compile_started",
                iteration=iteration,
                status="running",
                phase="compile",
                details=details,
            )

        try:
            report = compile_candidate_report(
                candidate_path,
                _context_with_last_feedback(ctx, last_feedback),
            )
            report = _with_context_defaults(report, ctx)
        except CandidateCompileError as exc:
            last_feedback = f"Compile failed: {exc}"
            last_report = CandidateReport(
                passed=False,
                estimates=[],
                errors=[last_feedback],
                details={
                    "compile_signals": signal_bundle_from_messages(
                        errors=[last_feedback],
                        source="compiler",
                    ).to_dict(),
                },
            )
            if viewer is not None:
                viewer.emit(
                    "compile_finished",
                    iteration=iteration,
                    status="failed",
                    phase="compile",
                    details=details,
                    errors=[str(exc)],
                )
            continue

        action_errors = incremental_action_errors(previous_report, report, ctx=ctx)
        if action_errors:
            last_feedback = "Agent action rejected: " + "; ".join(action_errors)
            last_report = CandidateReport(
                passed=False,
                estimates=report.estimates,
                errors=action_errors,
                details={
                    **report.details,
                    "compile_signals": signal_bundle_from_messages(
                        errors=action_errors,
                        warnings=list(report.warnings),
                        source="harness",
                    ).to_dict(),
                },
            )
            if viewer is not None:
                viewer.emit(
                    "compile_finished",
                    iteration=iteration,
                    status="passed",
                    phase="compile",
                    details=details,
                )
                viewer.emit_report(
                    "validation_finished",
                    last_report,
                    iteration=iteration,
                    phase="validation",
                    details={"required_estimates": sorted(ctx.joints)},
                )
            if on_report is not None:
                visualization_details = on_report(iteration, report)
                if viewer is not None and visualization_details:
                    viewer.emit(
                        "object_visualization_written",
                        iteration=iteration,
                        status="passed" if last_report.passed else "failed",
                        phase="visualization",
                        details=visualization_details,
                        errors=list(last_report.errors),
                        warnings=list(last_report.warnings),
                    )
            continue

        if validate_report is not None:
            report = validate_report(report)
        if report.passed:
            unexercised_errors = _unexercised_joint_errors(ctx, acted_joints)
            if unexercised_errors:
                report = CandidateReport(
                    passed=False,
                    estimates=report.estimates,
                    errors=[*report.errors, *unexercised_errors],
                    warnings=list(report.warnings),
                    details={
                        **report.details,
                        "compile_signals": signal_bundle_from_messages(
                            errors=[*report.errors, *unexercised_errors],
                            warnings=list(report.warnings),
                            source="harness",
                        ).to_dict(),
                    },
                )
        last_report = report
        if viewer is not None:
            viewer.emit(
                "compile_finished",
                iteration=iteration,
                status="passed",
                phase="compile",
                details=details,
            )
            viewer.emit_report(
                "validation_finished",
                report,
                iteration=iteration,
                phase="validation",
                details={"required_estimates": sorted(ctx.joints)},
            )
        if on_report is not None:
            visualization_details = on_report(iteration, report)
            if viewer is not None and visualization_details:
                viewer.emit(
                    "object_visualization_written",
                    iteration=iteration,
                    status="passed" if report.passed else "failed",
                    phase="visualization",
                    details=visualization_details,
                    errors=list(report.errors),
                    warnings=list(report.warnings),
                )
        if report.passed:
            if viewer is not None:
                viewer.emit(
                    "agent_loop_finished",
                    iteration=iteration,
                    status="passed",
                    phase="agent",
                )
            return AgentLoopResult(report=report, iterations=iteration)
        last_feedback = _feedback_from_report(report)

    if viewer is not None:
        viewer.emit(
            "agent_loop_finished",
            iteration=config.max_iterations,
            status="failed",
            phase="agent",
            errors=list(last_report.errors),
        )
    return AgentLoopResult(report=last_report, iterations=config.max_iterations)


def _compile_current_candidate(
    candidate_path: Path,
    ctx: EstimateContext,
) -> CandidateReport | None:
    try:
        return compile_candidate_report(candidate_path, ctx)
    except CandidateCompileError as exc:
        return CandidateReport(
            passed=False,
            estimates=[],
            errors=[f"Compile failed: {exc}"],
            details={
                "compile_signals": signal_bundle_from_messages(
                    errors=[f"Compile failed: {exc}"],
                    source="compiler",
                ).to_dict(),
            },
        )


def _seed_report_from_context(ctx: EstimateContext) -> CandidateReport:
    estimates: list[LimitEstimate] = []
    for joint_name in sorted(ctx.joints):
        joint = ctx.joints[joint_name]
        evidence = ctx.evidence.get(joint_name, {}) or {}
        initial = evidence.get("initial_estimate") or {}
        axis = _initial_axis_for_joint(ctx, joint_name, joint, initial)
        lower = float(initial.get("lower", joint.get("lower", 0.0)) or 0.0)
        upper_default = joint.get("upper", 0.0)
        if str(joint.get("type", "")).lower() == "prismatic" and abs(float(upper_default or 0.0)) <= 1e-12:
            upper_default = 0.1
        upper = float(initial.get("upper", upper_default) or 0.0)
        estimates.append(
            LimitEstimate(
                joint_name=joint_name,
                lower=lower,
                upper=upper,
                axis_world=list(_unit_axis(axis) or (1.0, 0.0, 0.0)),
                axis_label=initial.get("axis_label") or joint.get("axis_label"),
                confidence=0.35,
                reason="seeded from VLM/context before structured action loop",
            )
        )
    return CandidateReport(
        passed=bool(estimates),
        estimates=estimates,
        details={
            "compile_signals": signal_bundle_from_messages(
                errors=[],
                source="harness",
            ).to_dict(),
        },
    )


def _with_context_defaults(report: CandidateReport, ctx: EstimateContext) -> CandidateReport:
    updated: list[LimitEstimate] = []
    for estimate in report.estimates:
        joint = ctx.joints.get(estimate.joint_name, {})
        evidence = ctx.evidence.get(estimate.joint_name, {}) or {}
        initial = evidence.get("initial_estimate") or {}
        raw_axis = estimate.axis_world
        if raw_axis is None:
            raw_axis = _initial_axis_for_joint(ctx, estimate.joint_name, joint, initial)
        updated.append(
            LimitEstimate(
                joint_name=estimate.joint_name,
                lower=estimate.lower,
                upper=estimate.upper,
                axis_world=list(_unit_axis(raw_axis) or (1.0, 0.0, 0.0)),
                axis_label=estimate.axis_label or initial.get("axis_label") or joint.get("axis_label"),
                confidence=estimate.confidence,
                reason=estimate.reason,
            )
        )
    return CandidateReport(
        passed=report.passed,
        estimates=updated,
        errors=list(report.errors),
        warnings=list(report.warnings),
        details=dict(report.details),
    )


def _initial_axis_for_joint(
    ctx: EstimateContext,
    joint_name: str,
    joint: dict,
    initial: dict,
) -> Any:
    evidence = ctx.evidence.get(joint_name, {}) or {}
    return (
        evidence.get("recommended_axis_world")
        or initial.get("axis_world")
        or joint.get("axis_world")
        or [1.0, 0.0, 0.0]
    )


def _context_with_last_feedback(ctx: EstimateContext, last_feedback: str) -> EstimateContext:
    evidence = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in ctx.evidence.items()
    }
    evidence["__last_feedback__"] = str(last_feedback)
    evidence["last_feedback"] = str(last_feedback)
    return EstimateContext(
        object_id=ctx.object_id,
        joints=ctx.joints,
        evidence=evidence,
    )


def _feedback_from_report(report: CandidateReport) -> str:
    bundle = report.details.get("compile_signals") if report.details else None
    if bundle:
        return render_compile_signal_bundle(bundle)
    return render_compile_signal_bundle(
        signal_bundle_from_messages(
            errors=list(report.errors),
            warnings=list(report.warnings),
            source="harness",
        )
    )


def _summarize_action(action: dict[str, Any]) -> list[dict[str, Any]]:
    updates = action.get("updates", action)
    if not isinstance(updates, dict):
        return []
    summary: list[dict[str, Any]] = []
    for joint_name, update in sorted(updates.items()):
        if not isinstance(update, dict):
            summary.append({"joint_name": joint_name, "error": "update is not object"})
            continue
        summary.append(
            {
                "joint_name": joint_name,
                "state": update.get("state"),
                "axis": update.get("axis", update.get("axis_action")),
                "lower_delta": update.get("lower_delta", update.get("lower")),
                "upper_delta": update.get("upper_delta", update.get("upper")),
                "reason": update.get("reason"),
            }
        )
    return summary


def _no_effective_estimate_change(previous: CandidateReport, current: CandidateReport) -> bool:
    if not previous.estimates or not current.estimates:
        return False
    previous_by_joint = {estimate.joint_name: estimate for estimate in previous.estimates}
    current_by_joint = {estimate.joint_name: estimate for estimate in current.estimates}
    if set(previous_by_joint) != set(current_by_joint):
        return False
    for joint_name, before in previous_by_joint.items():
        after = current_by_joint[joint_name]
        if _changed_estimate_fields(before, after):
            return False
    return True


def _changed_joints(previous: CandidateReport, current: CandidateReport) -> set[str]:
    previous_by_joint = {estimate.joint_name: estimate for estimate in previous.estimates}
    changed: set[str] = set()
    for after in current.estimates:
        before = previous_by_joint.get(after.joint_name)
        if before is not None and _changed_estimate_fields(before, after):
            changed.add(after.joint_name)
    return changed


def _unexercised_joint_errors(ctx: EstimateContext, acted_joints: set[str]) -> list[str]:
    missing = sorted(set(ctx.joints) - set(acted_joints))
    if not missing:
        return []
    return [
        "agent cannot pass yet: "
        + ", ".join(missing)
        + " has not had an effective bounded action in this run; "
        "Articraft-style validation requires every rough VLM joint to be exercised "
        "by at least one axis/lower/upper action before marking the run correct"
    ]


def incremental_action_errors(
    previous: CandidateReport | None,
    current: CandidateReport,
    *,
    ctx: EstimateContext | None = None,
) -> list[str]:
    """Enforce one effective axis/lower/upper adjustment per agent iteration."""
    if previous is None or not previous.estimates:
        return []
    if not current.estimates:
        return ["agent action produced no estimates"]
    previous_by_joint = {estimate.joint_name: estimate for estimate in previous.estimates}
    current_by_joint = {estimate.joint_name: estimate for estimate in current.estimates}
    if set(previous_by_joint) != set(current_by_joint):
        return ["agent action must keep the same joint estimate set and adjust one existing joint"]

    changed: dict[str, list[str]] = {}
    for joint_name, before in previous_by_joint.items():
        after = current_by_joint[joint_name]
        fields = _changed_estimate_fields(before, after)
        if fields:
            changed[joint_name] = fields

    errors: list[str] = []
    for joint_name, fields in changed.items():
        before = previous_by_joint[joint_name]
        after = current_by_joint[joint_name]
        joint_type = _joint_type(ctx, joint_name)
        if fields.count("axis") > 1 or fields.count("lower") > 1 or fields.count("upper") > 1:
            errors.append(f"{joint_name}: duplicate action field in one iteration")
        if "axis" in fields:
            error = _axis_step_error(
                joint_name,
                before.axis_world,
                after.axis_world,
                ctx=ctx,
            )
            if error:
                errors.append(error)
        if "lower" in fields:
            error = _limit_step_error(
                joint_name,
                "lower",
                float(after.lower) - float(before.lower),
                joint_type=joint_type,
            )
            if error:
                errors.append(error)
        if "upper" in fields:
            error = _limit_step_error(
                joint_name,
                "upper",
                float(after.upper) - float(before.upper),
                joint_type=joint_type,
            )
            if error:
                errors.append(error)
    return errors


def _changed_estimate_fields(before: LimitEstimate, after: LimitEstimate) -> list[str]:
    fields: list[str] = []
    if not _same_axis(before.axis_world, after.axis_world) or before.axis_label != after.axis_label:
        fields.append("axis")
    if abs(float(before.lower) - float(after.lower)) > 1e-9:
        fields.append("lower")
    if abs(float(before.upper) - float(after.upper)) > 1e-9:
        fields.append("upper")
    return fields


def _same_axis(left, right) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None or len(left) != len(right):
        return False
    return all(abs(float(a) - float(b)) <= 1e-9 for a, b in zip(left, right, strict=True))


def _joint_type(ctx: EstimateContext | None, joint_name: str) -> str | None:
    if ctx is None:
        return None
    joint = ctx.joints.get(joint_name, {})
    raw = joint.get("type")
    return str(raw) if raw is not None else None


def _axis_step_error(
    joint_name: str,
    before_axis,
    after_axis,
    *,
    ctx: EstimateContext | None,
) -> str | None:
    before = _unit_axis(before_axis)
    after = _unit_axis(after_axis)
    if before is None or after is None:
        return f"{joint_name}: axis action requires non-zero before/after axis vectors"
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(before, after, strict=True))))
    angle_degrees = math.degrees(math.acos(dot))
    if angle_degrees <= 5.05:
        return None
    return (
        f"{joint_name}: axis changed by {angle_degrees:.6g} degrees; "
        "axis actions must only micro-rotate the current SDK geometry unit vector by at most 5 degrees"
    )


def _limit_step_error(
    joint_name: str,
    field: str,
    delta: float,
    *,
    joint_type: str | None,
) -> str | None:
    allowed = _allowed_limit_steps(joint_type)
    if any(abs(abs(delta) - step) <= 1e-9 for step in allowed):
        return None
    if joint_type == "revolute":
        return (
            f"{joint_name}: {field} changed by {delta:.6g} rad; "
            "allowed revolute limit steps are +/-10, +/-5, +/-2.5, +/-1, +/-0.5 degrees"
        )
    return (
        f"{joint_name}: {field} changed by {delta:.6g} m; "
        "allowed prismatic limit steps are +/-10, +/-5, +/-2.5, +/-1, +/-0.5 mm"
    )


def _allowed_limit_steps(joint_type: str | None) -> tuple[float, ...]:
    if joint_type == "revolute":
        return tuple(math.radians(value) for value in (10.0, 5.0, 2.5, 1.0, 0.5))
    return (0.010, 0.005, 0.0025, 0.001, 0.0005)


def _unit_axis(raw) -> tuple[float, float, float] | None:
    if raw is None or len(raw) != 3:
        return None
    axis = tuple(float(value) for value in raw)
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 1e-12:
        return None
    return tuple(value / norm for value in axis)


def _request_with_heartbeat(
    request: AgentRequestFn,
    messages: list[dict[str, str]],
    *,
    viewer: LiveViewer | None,
    iteration: int,
    config: AgentLoopConfig,
) -> str:
    result: dict[str, str] = {}
    error: dict[str, BaseException] = {}

    def target() -> None:
        try:
            result["text"] = request(messages)
        except BaseException as exc:  # propagate after joining the worker thread
            error["exc"] = exc

    thread = threading.Thread(target=target, daemon=True)
    started = time.monotonic()
    thread.start()
    interval = max(float(config.heartbeat_interval_seconds), 0.001)

    def emit_waiting(elapsed_seconds: float) -> None:
        if viewer is None:
            return
        viewer.emit(
            "api_waiting",
            iteration=iteration,
            status="running",
            phase="api",
            details={
                "elapsed_seconds": round(elapsed_seconds, 3),
                "timeout_seconds": config.timeout_seconds,
                "provider": config.provider,
                "model": config.model,
                "request_chars": _message_chars(messages),
            },
        )

    if thread.is_alive():
        emit_waiting(0.0)
    while thread.is_alive():
        thread.join(timeout=interval)
        if thread.is_alive():
            emit_waiting(time.monotonic() - started)
    if "exc" in error:
        raise error["exc"]
    return result.get("text", "")


def build_action_messages(
    ctx: EstimateContext,
    *,
    previous_report: CandidateReport,
    last_feedback: str,
    config: AgentLoopConfig,
) -> list[dict[str, str]]:
    system = (
        "You are an Articraft-style kinematic joint-limit agent. "
        "Do not write Python. Return only one compact JSON object. "
        "The local SDK will apply your action, regenerate estimate_limit.py, "
        "compile, drive the full range, validate, and visualize. "
        "Each iteration is one bounded search step. The SDK supplies a geometry-relation "
        "joint axis when the mesh pose supports one. Your main job is to search "
        "lower/upper limits. For each joint, choose state 'need_fix' or 'correct'. "
        "If a joint is correct, keep it unchanged. For a joint that needs fixing, "
        "you may perform at most one micro axis rotation, one lower_delta, and one "
        "upper_delta in this iteration. Axis action op is keep or rotate only. "
        "rotate must provide a target axis_world within 5 degrees of the current "
        "SDK geometry axis; do not switch to a different axis family. Prismatic limit deltas are in mm and must be one of "
        "+/-10, +/-5, +/-2.5, +/-1, +/-0.5. Revolute limit deltas are in degrees "
        "with the same ladder. Blocking compile_signals from the last iteration "
        "are authoritative; fix those joints before marking them correct. "
        f"Thinking level requested by the harness: {config.thinking_level}."
    )
    payload = {
        "object_id": ctx.object_id,
        "current_estimates": _compact_estimates(previous_report),
        "joint_context": _compact_joint_context(ctx),
        "last_feedback": last_feedback,
        "allowed_action_schema": {
            "updates": {
                "<joint_name>": {
                    "state": "need_fix|correct",
                    "axis": {
                        "op": "keep|rotate",
                        "axis_world": "<required target for rotate>",
                    },
                    "lower_delta": {"unit": "mm|degree", "value": 0},
                    "upper_delta": {"unit": "mm|degree", "value": 0},
                    "reason": "short reason tied to validation signals",
                }
            }
        },
        "allowed_steps": {
            "prismatic_mm": list(PRISMATIC_STEP_MM),
            "revolute_degrees": list(REVOLUTE_STEP_DEGREES),
            "axis_rotate_max_degrees": 5.0,
            "axis_source": "SDK geometry-relation baseline; agent may only micro-rotate",
        },
        "response_contract": "Return JSON only. No markdown unless fenced as json. No Python code.",
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
    ]


def _compact_estimates(report: CandidateReport) -> list[dict[str, Any]]:
    return [
        {
            "joint_name": estimate.joint_name,
            "lower": float(estimate.lower),
            "upper": float(estimate.upper),
            "axis_world": (
                [float(value) for value in estimate.axis_world]
                if estimate.axis_world is not None
                else None
            ),
            "axis_label": estimate.axis_label,
            "state_hint": "correct" if report.passed else "need_fix",
            "reason": estimate.reason,
        }
        for estimate in report.estimates
    ]


def _compact_joint_context(ctx: EstimateContext) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for joint_name in sorted(ctx.joints):
        joint = ctx.joints[joint_name]
        evidence = ctx.evidence.get(joint_name, {}) or {}
        compact[joint_name] = {
            "type": joint.get("type"),
            "parent": joint.get("parent"),
            "sdk_axis_world": evidence.get("recommended_axis_world"),
            "axis_candidates": [
                {"axis_label": label, "axis_world": list(axis)}
                for axis, label in _axis_candidates_for_joint(ctx, joint_name)
            ],
            "initial_estimate": evidence.get("initial_estimate"),
            "labels": evidence.get("labels"),
        }
    return compact


def build_agent_messages(
    ctx: EstimateContext,
    *,
    current_code: str,
    last_feedback: str,
    config: AgentLoopConfig,
) -> list[dict[str, str]]:
    system = (
        "You are maintaining only the editable region of estimate_limit.py. "
        "Return Python code for that region only. Define estimate_limits(ctx). "
        "Use ctx.joints and ctx.evidence. Do not read files, source USD limits, "
        "GT limit JSON, network resources, or mutate sdk/docs/harness code. "
        "This is an iterative action loop, not a one-shot final answer. "
        "For each joint independently, keep a state in your reasoning: need_fix or correct. "
        "In one iteration, each joint may either stay unchanged or take a bounded action: "
        "the SDK provides a geometry-relation axis baseline when available, and you may only micro-rotate "
        "the current unit axis by no more than 5 degrees; do not switch to a "
        "different axis family. Change lower at most once, "
        "and change upper at most once. Prismatic limit actions are exactly +/-10mm, +/-5mm, "
        "+/-2.5mm, +/-1mm, or +/-0.5mm, expressed in meters in LimitEstimate. "
        "Revolute limit actions use the same numeric ladder in degrees, expressed in radians. "
        "Joints that are already correct should stay unchanged. "
        "Use the previous Articraft-style <compile_signals> feedback and visual result "
        "to decide the next action. Blocking failure signals are authoritative: fix "
        "those joints before marking them correct. "
        "Return one LimitEstimate per joint. Keep reasoning in reason fields. "
        f"Thinking level requested by the harness: {config.thinking_level}."
    )
    prompt_code = _current_code_for_prompt(current_code)
    user = {
        "object_id": ctx.object_id,
        "joints": ctx.joints,
        "evidence": ctx.evidence,
        "last_feedback": last_feedback,
        "current_editable_code": prompt_code,
        "current_editable_code_compacted": prompt_code != current_code,
        "prompt_contract": (
            "Return a complete replacement for the editable region. "
            "Prefer concise code under 140 lines; do not preserve old helper code unless necessary."
        ),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, indent=2, sort_keys=True)},
    ]


def _current_code_for_prompt(current_code: str) -> str:
    if len(current_code) <= MAX_PROMPT_CODE_CHARS:
        return current_code
    return (
        "# The current editable region is intentionally compacted for the API prompt because "
        f"the on-disk region is {len(current_code)} characters and has caused gateway timeouts. "
        "You may replace the whole editable region with concise code.\n"
        + COMPACT_EDITABLE_TEMPLATE
    )


def _message_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages)


def _safe_base_url(raw_url: str) -> str:
    parsed = urllib.parse.urlparse(raw_url)
    if not parsed.netloc:
        return raw_url
    return f"{parsed.scheme}://{parsed.netloc}"


def _safe_http_error_body(exc: urllib.error.HTTPError, limit: int = 600) -> str:
    try:
        raw = exc.read()
    except Exception:
        return ""
    if not raw:
        return ""
    text = raw.decode("utf-8", errors="replace").strip()
    text = re.sub(r"sk-[A-Za-z0-9_\\-]+", "sk-<redacted>", text)
    return text[:limit]


def _first_env_key(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if not value:
            continue
        first = value.split(",")[0].strip()
        if first:
            return first
    return None
