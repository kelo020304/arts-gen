"""Articraft-style structured feedback for candidate compile/QC results."""

from __future__ import annotations

from .schemas import CompileSignal, CompileSignalBundle


def compile_signal_from_error(error: str, *, source: str = "harness") -> CompileSignal:
    text = str(error).strip()
    joint_name = _joint_prefix(text)
    lower = text.lower()
    if "intersects" in lower or "collides" in lower or "overlap" in lower:
        kind = "sampled_motion_overlap"
        code = "QC_SAMPLED_MOTION_OVERLAP"
        summary = "Sampled articulation motion intersects geometry."
    elif "axis" in lower:
        kind = "joint_axis"
        code = "QC_JOINT_AXIS"
        summary = "Joint axis validation failed."
    elif "allowed" in lower and "step" in lower:
        kind = "agent_action_space"
        code = "QC_AGENT_ACTION_SPACE"
        summary = "Agent action violated the bounded action space."
    elif "missing" in lower:
        kind = "candidate_contract"
        code = "QC_CANDIDATE_CONTRACT"
        summary = "Candidate output contract is incomplete."
    else:
        kind = "candidate_qc"
        code = "QC_CANDIDATE"
        summary = "Candidate validation failed."
    return CompileSignal(
        severity="failure",
        kind=kind,
        code=code,
        summary=summary,
        detail=text,
        source=source,
        group="qc",
        blocking=True,
        joint_name=joint_name,
    )


def compile_signal_from_warning(warning: str, *, source: str = "harness") -> CompileSignal:
    text = str(warning).strip()
    return CompileSignal(
        severity="warning",
        kind="candidate_warning",
        code="WARN_CANDIDATE",
        summary="Candidate validation emitted a warning.",
        detail=text,
        source=source,
        group="qc",
        blocking=False,
        joint_name=_joint_prefix(text),
    )


def signal_bundle_from_messages(
    *,
    errors: list[str],
    warnings: list[str] | None = None,
    source: str = "harness",
) -> CompileSignalBundle:
    signals = [
        compile_signal_from_error(error, source=source)
        for error in errors
    ]
    signals.extend(
        compile_signal_from_warning(warning, source=source)
        for warning in (warnings or [])
    )
    if errors:
        return CompileSignalBundle(
            status="failure",
            summary=(
                f"Candidate failed compile/QC with {len(errors)} blocking "
                f"failure{'s' if len(errors) != 1 else ''}."
            ),
            signals=signals,
        )
    if warnings:
        return CompileSignalBundle(
            status="success",
            summary=(
                f"Candidate passed with {len(warnings)} warning"
                f"{'s' if len(warnings) != 1 else ''}."
            ),
            signals=signals,
        )
    return CompileSignalBundle(
        status="success",
        summary="Candidate passed compile/QC.",
        signals=[],
    )


def render_compile_signal_bundle(bundle: CompileSignalBundle | dict | None) -> str:
    if bundle is None:
        return "<compile_signals>\nstatus: unknown\n</compile_signals>"
    payload = bundle.to_dict() if isinstance(bundle, CompileSignalBundle) else dict(bundle)
    lines = [
        "<compile_signals>",
        f"status: {payload.get('status', 'unknown')}",
        f"summary: {payload.get('summary', '')}",
    ]
    signals = payload.get("signals") or []
    for index, signal in enumerate(signals, start=1):
        if not isinstance(signal, dict):
            continue
        lines.append(f"[{index}] {signal.get('severity', 'note')} {signal.get('code', '')} {signal.get('kind', '')}")
        if signal.get("joint_name"):
            lines.append(f"joint: {signal['joint_name']}")
        if signal.get("summary"):
            lines.append(f"summary: {signal['summary']}")
        if signal.get("detail"):
            lines.append(f"detail: {signal['detail']}")
    lines.append("</compile_signals>")
    return "\n".join(lines)


def _joint_prefix(text: str) -> str | None:
    prefix, sep, _rest = text.partition(":")
    if sep and prefix.strip():
        return prefix.strip()
    return None
