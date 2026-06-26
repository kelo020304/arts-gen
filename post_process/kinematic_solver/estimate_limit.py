"""Agent-maintained joint-limit estimator.

Only the region between USER_CODE_START and USER_CODE_END is editable by the
agent. The surrounding harness is stable repo code: it builds context, compiles
this file, validates the returned estimates, and writes predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from post_process.kinematic_solver.sdk import (
    CandidateCompileError,
    CandidateReport,
    EstimateContext,
    LimitEstimate,
    build_context_from_roots,
    compile_candidate_report,
    context_details,
    editable_region_sha256,
    extract_editable_code,
    load_vlm_initial_context,
    LiveViewer,
    resolve_agent_loop_config,
    run_agent_loop,
    start_live_viewer_server,
    validate_motion_search_from_roots,
    with_axis_candidate_evidence,
    write_iteration_mjcf_preview,
    write_rest_mjcf_preview,
)
from post_process.kinematic_solver.sdk.compile_signals import signal_bundle_from_messages


# >>> USER_CODE_START
def estimate_limits(ctx):
    """Generic seed estimator for non-agent runs.

    Agent-loop runs copy this file into the run directory before writing a
    deterministic artifact, so this maintained file stays context-driven.
    """
    import ast
    import math
    import re

    evidence = getattr(ctx, "evidence", {}) or {}
    joints = getattr(ctx, "joints", {}) or {}
    feedback = str(evidence.get("__last_feedback__", "") or evidence.get("last_feedback", "") or "")

    def norm(raw):
        vals = [float(v) for v in (raw or [1.0, 0.0, 0.0])]
        n = math.sqrt(sum(v * v for v in vals))
        return [v / n for v in vals] if n > 1e-12 else [1.0, 0.0, 0.0]

    def rotate_toward(current, target, max_degrees=5.0):
        cur = norm(current)
        tgt = norm(target)
        dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(cur, tgt))))
        angle = math.acos(dot)
        if angle <= 1e-12:
            return cur
        step = min(angle, math.radians(max_degrees))
        sin_angle = math.sin(angle)
        a = math.sin(angle - step) / sin_angle
        b = math.sin(step) / sin_angle
        return norm([a * cur[i] + b * tgt[i] for i in range(3)])

    def parse_vector(label):
        match = re.search(label + r"=\[([^\]]+)\]", feedback)
        if not match:
            return None
        try:
            return [float(v) for v in ast.literal_eval("[" + match.group(1) + "]")]
        except Exception:
            return None

    target_axis = parse_vector("target_axis_world")
    previous_axis = parse_vector("candidate_axis_world")

    estimates = []
    for name in sorted(joints):
        joint = joints[name]
        ev = evidence.get(name, {}) or {}
        init = ev.get("initial_estimate") or {}
        axis = norm(ev.get("recommended_axis_world", init.get("axis_world", joint.get("axis_world", [1.0, 0.0, 0.0]))))
        reason = "seeded from SDK geometry axis and VLM/context limit estimate"
        if target_axis is not None and previous_axis is not None and name in feedback:
            axis = rotate_toward(previous_axis, target_axis, 5.0)
            reason = "rotated 5 degrees from previous candidate_axis_world toward target_axis_world"
        lower = float(init.get("lower", joint.get("lower", 0.0)) or 0.0)
        upper_default = joint.get("upper", 0.0)
        if str(joint.get("type", "")).lower() == "prismatic" and abs(float(upper_default or 0.0)) <= 1e-12:
            upper_default = 0.1
        upper = float(init.get("upper", upper_default) or 0.0)
        estimates.append(LimitEstimate(
            joint_name=name,
            lower=lower,
            upper=upper,
            axis_world=axis,
            confidence=0.35,
            reason=reason,
        ))
    return estimates
# >>> USER_CODE_END


def resolve_api_settings() -> dict[str, str | None]:
    """Use Articraft-compatible environment variable names without logging secrets."""
    has_openrouter_key = bool(
        os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEYS")
    )
    return {
        "provider": "openrouter" if has_openrouter_key else "openai",
        "model": os.environ.get("ARTICRAFT_MODEL") or "gpt-5.5",
        "thinking_level": os.environ.get("ARTICRAFT_THINKING_LEVEL") or "high",
        "openrouter_base_url": os.environ.get("OPENROUTER_BASE_URL"),
        "openai_base_url": os.environ.get("OPENAI_BASE_URL"),
    }


def _load_context(path: Path) -> EstimateContext:
    payload = json.loads(path.read_text())
    return EstimateContext(
        object_id=payload["object_id"],
        joints=dict(payload["joints"]),
        evidence=dict(payload.get("evidence", {})),
    )


def _write_predictions(path: Path, ctx: EstimateContext, report) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for estimate in report.estimates:
            joint = ctx.joints[estimate.joint_name]
            out.write(json.dumps({
                "object_id": ctx.object_id,
                "joint_name": estimate.joint_name,
                "type": joint.get("type"),
                "canonical_unit": joint.get("canonical_unit"),
                "predicted_lower": estimate.lower,
                "predicted_upper": estimate.upper,
                "predicted_axis_world": (
                    [float(value) for value in estimate.axis_world]
                    if estimate.axis_world is not None
                    else None
                ),
                "predicted_axis_label": estimate.axis_label,
                "status": "ok" if report.passed else "candidate_failed",
                "reason": estimate.reason,
                "confidence": estimate.confidence,
            }) + "\n")


def _write_object_visualization(args, ctx: EstimateContext, report) -> dict | None:
    if args.converter_output_root is None or not report.estimates:
        return None
    return write_iteration_mjcf_preview(
        ctx,
        list(report.estimates),
        converter_output_root=args.converter_output_root,
        run_dir=args.out_dir,
        iteration=getattr(args, "_current_iteration", 0),
        motion_search=report.details.get("motion_search"),
        joint_states=report.details.get("joint_states"),
        manual_sliders=bool(report.passed),
    )


def _with_recommended_axes(
    ctx: EstimateContext,
    estimates: list[LimitEstimate],
) -> list[LimitEstimate]:
    updated: list[LimitEstimate] = []
    for estimate in estimates:
        evidence = ctx.evidence.get(estimate.joint_name, {}) or {}
        axis = evidence.get("recommended_axis_world") or estimate.axis_world
        updated.append(
            LimitEstimate(
                joint_name=estimate.joint_name,
                lower=estimate.lower,
                upper=estimate.upper,
                axis_world=axis,
                axis_label=evidence.get("recommended_axis_label") or estimate.axis_label,
                confidence=estimate.confidence,
                reason=(
                    "seeded with SDK geometry axis and VLM rough range"
                    if evidence.get("recommended_axis_world") is not None
                    else estimate.reason
                ),
            )
        )
    return updated


def _write_rest_visualization(args, ctx: EstimateContext) -> dict | None:
    if args.converter_output_root is None:
        return None
    return write_rest_mjcf_preview(
        ctx,
        converter_output_root=args.converter_output_root,
        run_dir=args.out_dir,
    )


def _with_motion_validation(args, ctx: EstimateContext, report: CandidateReport) -> CandidateReport:
    if not report.passed or args.converter_output_root is None or args.skip_motion_validation:
        return report
    try:
        validation = validate_motion_search_from_roots(
            ctx,
            report.estimates,
            converter_output_root=args.converter_output_root,
        )
        errors = validation.errors
        details = {
            **report.details,
            "motion_search": validation.traces,
            "joint_states": _joint_states_from_errors(ctx, errors),
            "compile_signals": signal_bundle_from_messages(
                errors=[*report.errors, *errors],
                warnings=list(report.warnings),
                source="tests",
            ).to_dict(),
        }
    except Exception as exc:
        errors = [f"motion validation failed: {type(exc).__name__}: {exc}"]
        details = {
            **report.details,
            "joint_states": _joint_states_from_errors(ctx, errors),
            "compile_signals": signal_bundle_from_messages(
                errors=[*report.errors, *errors],
                warnings=list(report.warnings),
                source="tests",
            ).to_dict(),
        }
    if not errors:
        details = {
            **details,
            "compile_signals": signal_bundle_from_messages(
                errors=list(report.errors),
                warnings=list(report.warnings),
                source="tests",
            ).to_dict(),
        }
        return CandidateReport(
            passed=True,
            estimates=report.estimates,
            errors=list(report.errors),
            warnings=list(report.warnings),
            details=details,
        )
    return CandidateReport(
        passed=False,
        estimates=report.estimates,
        errors=[*report.errors, *errors],
        warnings=list(report.warnings),
        details=details,
    )


def _joint_states_from_errors(ctx: EstimateContext, errors: list[str]) -> dict[str, dict]:
    by_joint = {joint_name: [] for joint_name in ctx.joints}
    global_errors = []
    for error in errors:
        prefix, sep, rest = str(error).partition(":")
        if sep and prefix in by_joint:
            by_joint[prefix].append(rest.strip())
        else:
            global_errors.append(str(error))
    states = {}
    for joint_name in sorted(ctx.joints):
        joint_errors = [*by_joint[joint_name], *global_errors]
        states[joint_name] = {
            "status": "need_fix" if joint_errors else "correct",
            "errors": joint_errors,
        }
    return states


def _clear_run_artifacts(
    out_dir: Path,
    *,
    object_id: str | None = None,
    clear_frontend_state: bool = True,
) -> list[str]:
    """Remove only files this entrypoint generates for a previous run."""
    out_dir = Path(out_dir)
    removed: list[str] = []
    names = ["candidate_report.json", "predictions.jsonl"]
    if clear_frontend_state:
        names.append("frontend_state.json")
    for name in names:
        path = out_dir / name
        if path.exists():
            path.unlink()
            removed.append(name)
    object_assets = out_dir / "object_assets"
    if object_assets.exists():
        shutil.rmtree(object_assets)
        removed.append("object_assets")
    if object_id:
        viz_dir = out_dir / object_id / "agent_viz"
        if viz_dir.exists():
            shutil.rmtree(viz_dir)
            removed.append(f"{object_id}/agent_viz")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile post_process/kinematic_solver/estimate_limit.py and write predictions."
    )
    parser.add_argument(
        "--candidate-path",
        type=Path,
        default=Path(__file__),
        help="Internal/testing override. Agent workflow should leave this as estimate_limit.py.",
    )
    parser.add_argument("--context-json", type=Path)
    parser.add_argument("--object-id")
    parser.add_argument("--converter-output-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--initial-joints-json",
        type=Path,
        help="Rough VLM initial joint guesses. Prismatic limits are mm; revolute limits are degrees.",
    )
    parser.add_argument(
        "--skip-motion-validation",
        action="store_true",
        help="Compile/write previews without geometry motion validation. Intended for UI smoke tests only.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--print-api-settings",
        action="store_true",
        help="Print provider/model/base-url fields only; never prints API keys.",
    )
    parser.add_argument(
        "--live-viewer",
        action="store_true",
        help="Write agent_events.jsonl/frontend_state.json and serve the post_process MJCF viewer.",
    )
    parser.add_argument(
        "--no-live-server",
        action="store_true",
        help="Only write live-viewer files; do not start the local polling server.",
    )
    parser.add_argument(
        "--live-host",
        default="127.0.0.1",
        help="Host for the live viewer server when --live-viewer is enabled.",
    )
    parser.add_argument(
        "--live-port",
        type=int,
        default=0,
        help="Port for the live viewer server. Use 0 to choose a free port.",
    )
    parser.add_argument(
        "--open-live-viewer",
        action="store_true",
        help="Open the live viewer URL in the default browser.",
    )
    parser.add_argument(
        "--live-hold-seconds",
        type=float,
        default=0.0,
        help="Keep the live server alive after completion for this many seconds.",
    )
    parser.add_argument(
        "--agent-loop",
        action="store_true",
        help="Call the configured API and iteratively replace only the editable region.",
    )
    parser.add_argument(
        "--max-agent-iterations",
        type=int,
        default=3,
        help="Maximum API edit/compile/validate attempts for --agent-loop.",
    )
    parser.add_argument(
        "--api-timeout-seconds",
        type=float,
        default=600.0,
        help="Timeout per OpenAI-compatible API request.",
    )
    parser.add_argument(
        "--api-heartbeat-seconds",
        type=float,
        default=2.0,
        help="Emit live api_waiting events at this interval while the API call blocks.",
    )
    args = parser.parse_args()

    if args.print_api_settings:
        print(json.dumps(resolve_api_settings(), indent=2))
        return

    run_mode = "agent-loop" if args.agent_loop else "single-check"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    initial_removed_artifacts = _clear_run_artifacts(
        args.out_dir,
        object_id=args.object_id,
    )
    if args.agent_loop and args.candidate_path.resolve() == Path(__file__).resolve():
        run_candidate_path = args.out_dir / "estimate_limit.py"
        run_candidate_path.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
        args.candidate_path = run_candidate_path
    viewer = LiveViewer(args.out_dir) if args.live_viewer else None
    server = None
    if viewer is not None:
        viewer.prepare()
        if not args.no_live_server:
            server = start_live_viewer_server(
                args.out_dir,
                host=args.live_host,
                port=args.live_port,
                run_id=viewer.run_id,
            )
            print(f"[LIVE] {server.url}")
            if args.open_live_viewer:
                import webbrowser

                webbrowser.open(server.url)
    if args.live_viewer:
        print(f"[MODE] {run_mode}")

    try:
        if viewer is not None:
            viewer.emit(
                "run_started",
                iteration=0,
                status="running",
                phase="start",
                details={
                    "candidate_path": str(args.candidate_path),
                    "out_dir": str(args.out_dir),
                    "cleared_artifacts": initial_removed_artifacts,
                    "mode": run_mode,
                    "max_agent_iterations": args.max_agent_iterations if args.agent_loop else None,
                },
            )

        if viewer is not None:
            viewer.emit(
                "context_started",
                iteration=0,
                status="running",
                phase="context",
                details={
                    "context_json": str(args.context_json) if args.context_json else None,
                    "object_id": args.object_id,
                },
            )
        initial_estimates = []
        try:
            if args.context_json is not None:
                ctx = _load_context(args.context_json)
            else:
                if not (args.object_id and args.converter_output_root and args.source_root):
                    raise SystemExit(
                        "provide --context-json or --object-id with --converter-output-root and --source-root"
                    )
                ctx = build_context_from_roots(
                    object_id=args.object_id,
                    converter_output_root=args.converter_output_root,
                    source_root=args.source_root,
                )
                ctx = with_axis_candidate_evidence(
                    ctx,
                    converter_output_root=args.converter_output_root,
                )
            if args.initial_joints_json is not None:
                ctx, initial_estimates = load_vlm_initial_context(ctx, args.initial_joints_json)
                if args.converter_output_root is not None:
                    ctx = with_axis_candidate_evidence(
                        ctx,
                        converter_output_root=args.converter_output_root,
                    )
                    initial_estimates = _with_recommended_axes(ctx, initial_estimates)
        except Exception as exc:
            if viewer is not None:
                viewer.emit(
                    "context_finished",
                    iteration=0,
                    status="failed",
                    phase="context",
                    errors=[str(exc)],
                )
                viewer.emit(
                    "run_finished",
                    iteration=0,
                    status="failed",
                    phase="context",
                    errors=[str(exc)],
                )
            raise

        if viewer is not None:
            viewer.emit(
                "context_finished",
                iteration=0,
                status="passed",
                phase="context",
                details=context_details(ctx),
            )
        context_removed_artifacts = _clear_run_artifacts(
            args.out_dir,
            object_id=ctx.object_id,
            clear_frontend_state=False,
        )
        if viewer is not None and context_removed_artifacts:
            viewer.emit(
                "run_artifacts_cleared",
                iteration=0,
                status="passed",
                phase="start",
                details={"cleared_artifacts": context_removed_artifacts},
            )
        if viewer is not None:
            rest_details = _write_rest_visualization(args, ctx)
            if rest_details:
                viewer.emit(
                    "rest_visualization_written",
                    iteration=0,
                    status="passed",
                    phase="visualization",
                    details=rest_details,
                    notes=["initial asset loaded before limits are known"],
                )
            if initial_estimates and args.converter_output_root is not None:
                initial_details = write_iteration_mjcf_preview(
                    ctx,
                    initial_estimates,
                    converter_output_root=args.converter_output_root,
                    run_dir=args.out_dir,
                    iteration=0,
                    preview_kind="vlm_initial",
                    joint_states={
                        estimate.joint_name: {
                            "status": "need_fix",
                            "errors": ["rough VLM initial guess"],
                        }
                        for estimate in initial_estimates
                    },
                )
                viewer.emit(
                    "initial_visualization_written",
                    iteration=0,
                    status="passed",
                    phase="visualization",
                    details=initial_details,
                    notes=["rough VLM initial axis/range visualized before agent edits"],
                )

        if args.agent_loop:
            try:
                loop_result = run_agent_loop(
                    args.candidate_path,
                    ctx,
                    config=resolve_agent_loop_config(
                        max_iterations=args.max_agent_iterations,
                        timeout_seconds=args.api_timeout_seconds,
                        heartbeat_interval_seconds=args.api_heartbeat_seconds,
                    ),
                    viewer=viewer,
                    on_report=(
                        (lambda iteration, report: (
                            setattr(args, "_current_iteration", iteration)
                            or _write_object_visualization(args, ctx, report)
                        ))
                        if viewer is not None
                        else None
                    ),
                    validate_report=lambda report: _with_motion_validation(args, ctx, report),
                )
                report = loop_result.report
            except Exception as exc:
                failed_iteration = int(
                    getattr(
                        exc,
                        "agent_iteration",
                        getattr(args, "_current_iteration", 0),
                    )
                )
                (args.out_dir / "candidate_report.json").write_text(json.dumps({
                    "passed": False,
                    "estimates": [],
                    "errors": [str(exc)],
                    "warnings": [],
                }, indent=2))
                if viewer is not None:
                    viewer.emit(
                        "run_finished",
                        iteration=failed_iteration,
                        status="failed",
                        phase="agent",
                        errors=[str(exc)],
                    )
                raise SystemExit(f"agent loop failed: {exc}") from exc

            (args.out_dir / "candidate_report.json").write_text(report.to_json())
            final_iteration = loop_result.iterations
            if report.passed:
                predictions_path = args.out_dir / "predictions.jsonl"
                _write_predictions(predictions_path, ctx, report)
                if viewer is not None:
                    viewer.emit(
                        "predictions_written",
                        iteration=final_iteration,
                        status="passed",
                        phase="write_output",
                        details={"predictions_path": str(predictions_path)},
                    )
                    viewer.emit(
                        "run_finished",
                        iteration=final_iteration,
                        status="passed",
                        phase="done",
                    )
                print(f"[OK] estimate_limit agent-loop {args.candidate_path}")
                return
            if viewer is not None:
                viewer.emit(
                    "run_finished",
                    iteration=final_iteration,
                    status="failed",
                    phase="validation",
                    errors=list(report.errors),
                )
            raise SystemExit("candidate validation failed: " + "; ".join(report.errors))

        iteration = 1
        code_digest = editable_region_sha256(args.candidate_path)
        compile_details = {
            "candidate_path": str(args.candidate_path),
            "editable_sha256": code_digest,
            "editable_code": extract_editable_code(args.candidate_path.read_text()).strip(),
        }
        if viewer is not None:
            viewer.emit(
                "agent_iteration_started",
                iteration=iteration,
                status="running",
                phase="agent",
                details=compile_details,
            )
            viewer.emit(
                "compile_started",
                iteration=iteration,
                status="running",
                phase="compile",
                details=compile_details,
            )
        try:
            report = compile_candidate_report(args.candidate_path, ctx)
        except CandidateCompileError as exc:
            report_path = args.out_dir / "candidate_report.json"
            report_path.write_text(json.dumps({
                "passed": False,
                "estimates": [],
                "errors": [str(exc)],
                "warnings": [],
            }, indent=2))
            if viewer is not None:
                viewer.emit(
                    "compile_finished",
                    iteration=iteration,
                    status="failed",
                    phase="compile",
                    details=compile_details,
                    errors=[str(exc)],
                )
                viewer.emit(
                    "run_finished",
                    iteration=iteration,
                    status="failed",
                    phase="compile",
                    errors=[str(exc)],
                )
            raise SystemExit(f"candidate compile failed: {exc}") from exc
        report = _with_motion_validation(args, ctx, report)

        if viewer is not None:
            viewer.emit(
                "compile_finished",
                iteration=iteration,
                status="passed",
                phase="compile",
                details=compile_details,
            )
            viewer.emit_report(
                "validation_finished",
                report,
                iteration=iteration,
                phase="validation",
                details={"required_estimates": sorted(ctx.joints)},
            )
            setattr(args, "_current_iteration", iteration)
            visualization_details = _write_object_visualization(args, ctx, report)
            if visualization_details:
                viewer.emit(
                    "object_visualization_written",
                    iteration=iteration,
                    status="passed",
                    phase="visualization",
                    details=visualization_details,
                )

        (args.out_dir / "candidate_report.json").write_text(report.to_json())
        if report.passed:
            predictions_path = args.out_dir / "predictions.jsonl"
            _write_predictions(predictions_path, ctx, report)
            if viewer is not None:
                viewer.emit(
                    "predictions_written",
                    iteration=iteration,
                    status="passed",
                    phase="write_output",
                    details={"predictions_path": str(predictions_path)},
                )
                viewer.emit(
                    "run_finished",
                    iteration=iteration,
                    status="passed",
                    phase="done",
                )
        else:
            if viewer is not None:
                viewer.emit(
                    "run_finished",
                    iteration=iteration,
                    status="failed",
                    phase="validation",
                    errors=list(report.errors),
                )
            raise SystemExit("candidate validation failed: " + "; ".join(report.errors))
        print(f"[OK] estimate_limit {args.candidate_path}")
    finally:
        if server is not None:
            if args.live_hold_seconds > 0:
                import time

                time.sleep(args.live_hold_seconds)
            server.stop()


if __name__ == "__main__":
    main()
