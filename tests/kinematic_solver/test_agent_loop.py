import json
import time
import urllib.error
from io import BytesIO

import pytest

from post_process.kinematic_solver.sdk.agent_loop import (
    AgentLoopConfig,
    AgentLoopError,
    OpenAICompatibleClient,
    apply_action_update,
    build_action_messages,
    build_agent_messages,
    extract_action_response_json,
    incremental_action_errors,
    extract_editable_response_code,
    render_estimate_limits_from_report,
    resolve_agent_loop_config,
    run_agent_loop,
)
from post_process.kinematic_solver.sdk.compile_signals import signal_bundle_from_messages
from post_process.kinematic_solver.sdk.live_viewer import LiveViewer
from post_process.kinematic_solver.sdk.schemas import CandidateReport
from post_process.kinematic_solver.sdk.schemas import EstimateContext
from post_process.kinematic_solver.sdk.schemas import LimitEstimate


def _scaffold(body: str) -> str:
    return (
        "from post_process.kinematic_solver.sdk import LimitEstimate\n\n"
        "OUTSIDE_VALUE = 7\n"
        "# >>> USER_CODE_START\n"
        f"{body}"
        "# >>> USER_CODE_END\n"
    )


def _ctx_for_action_space(joint_type: str = "prismatic") -> EstimateContext:
    return EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": joint_type,
            }
        },
        evidence={},
    )


def _ctx_with_axis_candidates(joint_type: str = "prismatic") -> EstimateContext:
    return EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": joint_type,
            }
        },
        evidence={
            "part_02": {
                "recommended_axis_world": [0.0, -1.0, 0.0],
                "axis_candidates": [
                    {"axis_label": "pca", "axis_world": [0.0, -1.0, 0.0]},
                ],
            }
        },
    )


def _report(estimate: LimitEstimate) -> CandidateReport:
    return CandidateReport(passed=True, estimates=[estimate])


def test_extract_editable_response_code_prefers_python_fence():
    response = (
        "Here is the update.\n"
        "```python\n"
        "def estimate_limits(ctx):\n"
        "    return []\n"
        "```\n"
    )

    code = extract_editable_response_code(response)

    assert code == "def estimate_limits(ctx):\n    return []"


def test_extract_action_response_json_accepts_fenced_json():
    action = extract_action_response_json(
        "```json\n"
        "{\"updates\":{\"part_02\":{\"state\":\"need_fix\",\"upper_delta\":{\"unit\":\"mm\",\"value\":5}}}}\n"
        "```"
    )

    assert action["updates"]["part_02"]["upper_delta"]["value"] == 5


def test_build_action_messages_is_small_and_does_not_send_editable_code():
    ctx = _ctx_with_axis_candidates("prismatic")
    report = _report(LimitEstimate("part_02", lower=0.0, upper=0.03, axis_world=[1.0, 0.0, 0.0]))

    messages = build_action_messages(
        ctx,
        previous_report=report,
        last_feedback="<compile_signals>\nstatus: failure\n</compile_signals>",
        config=AgentLoopConfig(max_iterations=1, api_key="fake"),
    )
    payload = json.loads(messages[1]["content"])

    assert "current_editable_code" not in payload
    assert payload["current_estimates"][0]["joint_name"] == "part_02"
    assert payload["allowed_steps"]["prismatic_mm"] == [10.0, 5.0, 2.5, 1.0, 0.5]
    assert payload["allowed_action_schema"]["updates"]["<joint_name>"]["axis"]["op"] == "keep|rotate"
    assert len(messages[1]["content"]) < 5000


def test_apply_action_update_prismatic_upper_delta_mm():
    previous = _report(LimitEstimate("part_02", lower=0.0, upper=0.03, axis_world=[1.0, 0.0, 0.0]))

    report = apply_action_update(
        previous,
        {"updates": {"part_02": {"state": "need_fix", "upper_delta": {"unit": "mm", "value": 5}}}},
        _ctx_for_action_space("prismatic"),
    )

    assert report.passed is True
    assert report.estimates[0].upper == pytest.approx(0.035)


def test_apply_action_update_revolute_delta_deg():
    previous = _report(LimitEstimate("part_02", lower=0.0, upper=0.0, axis_world=[0.0, 0.0, 1.0]))

    report = apply_action_update(
        previous,
        {"updates": {"part_02": {"state": "need_fix", "upper_delta": {"unit": "degree", "value": 5}}}},
        _ctx_for_action_space("revolute"),
    )

    assert report.passed is True
    assert report.estimates[0].upper == pytest.approx(0.08726646259971647)


def test_apply_action_update_rejects_outside_ladder():
    previous = _report(LimitEstimate("part_02", lower=0.0, upper=0.03, axis_world=[1.0, 0.0, 0.0]))

    report = apply_action_update(
        previous,
        {"updates": {"part_02": {"upper_delta": {"unit": "mm", "value": 3}}}},
        _ctx_for_action_space("prismatic"),
    )

    assert report.passed is False
    assert "must be one of" in report.errors[0]


def test_render_estimate_limits_from_report_is_deterministic_python():
    report = _report(
        LimitEstimate(
            "part_02",
            lower=0.0,
            upper=0.035,
            axis_world=[0.0, -1.0, 0.0],
            axis_label="-Y",
            reason="drawer axis fixed",
        )
    )

    code = render_estimate_limits_from_report(report)

    assert "Deterministic artifact generated from structured agent actions" in code
    assert "part_02" in code
    assert "axis_world': [0.0, -1.0, 0.0]" in code


def test_agent_loop_rejects_noop_first_action_from_vlm_seed(tmp_path):
    candidate = tmp_path / "estimate_limit.py"
    candidate.write_text(_scaffold(
        "def estimate_limits(ctx):\n"
        "    return [LimitEstimate(joint_name='part_02', lower=0.0, upper=0.30, axis_world=[0.0, 1.0, 0.0])]\n"
    ))
    ctx = EstimateContext(
        object_id="ra_063",
        joints={"part_02": {"type": "prismatic"}},
        evidence={
            "part_02": {
                "initial_estimate": {
                    "axis_world": [0.0, 0.0, 1.0],
                    "lower": 0.0,
                    "upper": 0.04,
                }
            }
        },
    )

    result = run_agent_loop(
        candidate,
        ctx,
        config=AgentLoopConfig(
            max_iterations=1,
            provider="openrouter",
            model="fake-model",
            thinking_level="medium",
            base_url="https://example.invalid/v1",
            api_key="not-used-by-fake",
        ),
        request_fn=lambda _messages: (
            "```json\n"
            "{\"updates\":{\"part_02\":{\"state\":\"correct\",\"axis\":{\"op\":\"keep\"},"
            "\"lower_delta\":{\"unit\":\"mm\",\"value\":0},"
            "\"upper_delta\":{\"unit\":\"mm\",\"value\":0},"
            "\"reason\":\"looks fine\"}}}\n"
            "```"
        ),
    )

    assert result.report.passed is False
    assert "made no effective change" in result.report.errors[0]


def test_agent_loop_starts_from_vlm_seed_not_stale_candidate_file(tmp_path):
    candidate = tmp_path / "estimate_limit.py"
    candidate.write_text(_scaffold(
        "def estimate_limits(ctx):\n"
        "    return [LimitEstimate(joint_name='part_02', lower=0.0, upper=0.30, axis_world=[0.0, 1.0, 0.0])]\n"
    ))
    ctx = EstimateContext(
        object_id="ra_063",
        joints={"part_02": {"type": "prismatic"}},
        evidence={
            "part_02": {
                "initial_estimate": {
                    "axis_world": [0.0, 0.0, 1.0],
                    "lower": 0.0,
                    "upper": 0.04,
                }
            }
        },
    )
    prompts = []

    result = run_agent_loop(
        candidate,
        ctx,
        config=AgentLoopConfig(
            max_iterations=1,
            provider="openrouter",
            model="fake-model",
            thinking_level="medium",
            base_url="https://example.invalid/v1",
            api_key="not-used-by-fake",
        ),
        request_fn=lambda messages: prompts.append(messages) or (
            "```json\n"
            "{\"updates\":{\"part_02\":{\"state\":\"need_fix\","
            "\"upper_delta\":{\"unit\":\"mm\",\"value\":10},"
            "\"reason\":\"expand from VLM seed\"}}}\n"
            "```"
        ),
    )

    payload = json.loads(prompts[0][1]["content"])
    assert payload["current_estimates"][0]["upper"] == 0.04
    assert payload["current_estimates"][0]["axis_world"] == [0.0, 0.0, 1.0]
    assert result.report.estimates[0].upper == pytest.approx(0.05)


def test_agent_loop_cannot_pass_until_every_joint_has_effective_action(tmp_path):
    candidate = tmp_path / "estimate_limit.py"
    candidate.write_text(_scaffold("def estimate_limits(ctx):\n    return []\n"))
    ctx = EstimateContext(
        object_id="ra_036",
        joints={
            "part_03": {"type": "revolute"},
            "part_07": {"type": "prismatic"},
        },
        evidence={
            "part_03": {
                "initial_estimate": {
                    "axis_world": [0.0, 0.0, 1.0],
                    "lower": -3.141592653589793,
                    "upper": 3.141592653589793,
                }
            },
            "part_07": {
                "initial_estimate": {
                    "axis_world": [0.0, 0.0, 1.0],
                    "lower": 0.0,
                    "upper": 0.03,
                }
            },
        },
    )

    result = run_agent_loop(
        candidate,
        ctx,
        config=AgentLoopConfig(
            max_iterations=1,
            provider="openrouter",
            model="fake-model",
            thinking_level="medium",
            base_url="https://example.invalid/v1",
            api_key="not-used-by-fake",
        ),
        request_fn=lambda _messages: (
            "```json\n"
            "{\"updates\":{"
            "\"part_03\":{\"state\":\"correct\",\"axis\":{\"op\":\"keep\"},\"lower_delta\":{\"unit\":\"degree\",\"value\":0},\"upper_delta\":{\"unit\":\"degree\",\"value\":0}},"
            "\"part_07\":{\"state\":\"need_fix\",\"upper_delta\":{\"unit\":\"mm\",\"value\":10}}"
            "}}\n"
            "```"
        ),
    )

    assert result.report.passed is False
    assert "part_03" in result.report.errors[0]
    assert "has not had an effective bounded action" in result.report.errors[0]


def test_compile_signal_classifies_sampled_motion_overlap_before_axis_hint():
    bundle = signal_bundle_from_messages(
        errors=[
            "part_02: candidate motion still intersects parent/static geometry at upper endpoint q=0.07; fix axis or extend limit"
        ],
        source="tests",
    )

    assert bundle.signals[0].code == "QC_SAMPLED_MOTION_OVERLAP"
    assert bundle.signals[0].joint_name == "part_02"


def test_incremental_action_allows_prismatic_mm_step_ladder():
    previous = _report(LimitEstimate("part_02", lower=0.0, upper=0.03, axis_world=[1.0, 0.0, 0.0]))
    current = _report(LimitEstimate("part_02", lower=0.0, upper=0.035, axis_world=[1.0, 0.0, 0.0]))

    assert incremental_action_errors(previous, current, ctx=_ctx_for_action_space("prismatic")) == []


def test_incremental_action_rejects_prismatic_limit_step_outside_ladder():
    previous = _report(LimitEstimate("part_02", lower=0.0, upper=0.03, axis_world=[1.0, 0.0, 0.0]))
    current = _report(LimitEstimate("part_02", lower=0.0, upper=0.033, axis_world=[1.0, 0.0, 0.0]))

    errors = incremental_action_errors(previous, current, ctx=_ctx_for_action_space("prismatic"))

    assert errors
    assert "allowed prismatic limit steps" in errors[0]


def test_incremental_action_allows_revolute_degree_step_ladder():
    previous = _report(LimitEstimate("part_02", lower=0.0, upper=0.0, axis_world=[0.0, 0.0, 1.0]))
    current = _report(LimitEstimate("part_02", lower=0.0, upper=0.08726646259971647, axis_world=[0.0, 0.0, 1.0]))

    assert incremental_action_errors(previous, current, ctx=_ctx_for_action_space("revolute")) == []


def test_incremental_action_allows_axis_rotation_up_to_five_degrees():
    previous = _report(LimitEstimate("part_02", lower=0.0, upper=0.03, axis_world=[0.0, 0.0, 1.0]))
    current = _report(LimitEstimate(
        "part_02",
        lower=0.0,
        upper=0.03,
        axis_world=[0.0, -0.08715574274765817, 0.9961946980917455],
    ))

    assert incremental_action_errors(previous, current, ctx=_ctx_for_action_space("revolute")) == []


def test_incremental_action_rejects_axis_rotation_over_five_degrees():
    previous = _report(LimitEstimate("part_02", lower=0.0, upper=0.03, axis_world=[0.0, 0.0, 1.0]))
    current = _report(LimitEstimate(
        "part_02",
        lower=0.0,
        upper=0.03,
        axis_world=[0.0, -0.10452846326765347, 0.9945218953682733],
    ))

    errors = incremental_action_errors(previous, current, ctx=_ctx_for_action_space("revolute"))

    assert errors
    assert "axis changed by 6" in errors[0]


def test_incremental_action_allows_tiny_axis_rotation_rounding_over_five_degrees():
    previous = _report(LimitEstimate("part_02", lower=0.0, upper=0.03, axis_world=[0.0, 0.0, 1.0]))
    current = _report(LimitEstimate(
        "part_02",
        lower=0.0,
        upper=0.03,
        axis_world=[0.0, -0.087177805, 0.996192768],
    ))

    assert incremental_action_errors(previous, current, ctx=_ctx_for_action_space("revolute")) == []


def test_incremental_action_rejects_axis_switch_to_sdk_candidate_axis():
    previous = _report(LimitEstimate("part_02", lower=0.0, upper=0.03, axis_world=[1.0, 0.0, 0.0]))
    current = _report(LimitEstimate("part_02", lower=0.0, upper=0.03, axis_world=[0.0, -1.0, 0.0]))

    errors = incremental_action_errors(previous, current, ctx=_ctx_with_axis_candidates("prismatic"))

    assert errors
    assert "axis changed by 90" in errors[0]


def test_incremental_action_rejects_axis_switch_to_non_candidate_axis():
    previous = _report(LimitEstimate("part_02", lower=0.0, upper=0.03, axis_world=[1.0, 0.0, 0.0]))
    current = _report(LimitEstimate("part_02", lower=0.0, upper=0.03, axis_world=[0.0, 0.0, 1.0]))

    errors = incremental_action_errors(previous, current, ctx=_ctx_with_axis_candidates("prismatic"))

    assert errors
    assert "axis changed by 90" in errors[0]


def test_agent_loop_default_timeout_allows_long_high_thinking_calls(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)

    config = resolve_agent_loop_config(max_iterations=1)

    assert config.timeout_seconds == 600.0


def test_openai_compatible_client_reports_read_timeout_budget(monkeypatch):
    def fake_urlopen(_request, *, timeout):
        assert timeout == 12.5
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        AgentLoopConfig(
            max_iterations=1,
            provider="openrouter",
            model="fake-model",
            thinking_level="high",
            base_url="https://example.invalid/v1",
            api_key="fake-key",
            timeout_seconds=12.5,
        )
    )

    try:
        client.complete([{"role": "user", "content": "hello"}])
    except AgentLoopError as exc:
        assert str(exc) == "API request timed out after 12.5s while waiting for model response"
    else:
        raise AssertionError("expected AgentLoopError")


def test_openai_compatible_client_reports_gateway_timeout_diagnostics(monkeypatch):
    def fake_urlopen(_request, *, timeout):
        raise urllib.error.HTTPError(
            url="https://router.example/v1/chat/completions",
            code=504,
            msg="Gateway Timeout",
            hdrs={},
            fp=BytesIO(b'{"error":"upstream timed out"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        AgentLoopConfig(
            max_iterations=1,
            provider="openrouter",
            model="fake-model",
            thinking_level="medium",
            base_url="https://router.example/v1",
            api_key="fake-key",
            timeout_seconds=12.5,
        )
    )

    with pytest.raises(AgentLoopError) as exc_info:
        client.complete([{"role": "user", "content": "hello"}])

    message = str(exc_info.value)
    assert "HTTP 504" in message
    assert "provider=openrouter" in message
    assert "model=fake-model" in message
    assert "base_url=https://router.example" in message
    assert "request_chars=5" in message
    assert "upstream/proxy timeout" in message
    assert "upstream timed out" in message


def test_build_agent_messages_compacts_large_current_editable_code():
    ctx = EstimateContext(
        object_id="ra_036",
        joints={"part_07": {"type": "prismatic"}},
        evidence={"part_07": {"initial_estimate": {"axis_world": [1, 0, 0], "lower": 0.0, "upper": 0.03}}},
    )

    messages = build_agent_messages(
        ctx,
        current_code="def estimate_limits(ctx):\n    pass\n" + ("# old helper\n" * 800),
        last_feedback="No previous feedback",
        config=AgentLoopConfig(max_iterations=1, api_key="fake"),
    )
    payload = json.loads(messages[1]["content"])

    assert payload["current_editable_code_compacted"] is True
    assert len(payload["current_editable_code"]) < 2500
    assert "compact seed" in payload["current_editable_code"]
    assert "Return a complete replacement" in payload["prompt_contract"]


def test_agent_loop_retries_until_candidate_report_passes(tmp_path):
    candidate = tmp_path / "estimate_limit.py"
    candidate.write_text(_scaffold("def estimate_limits(ctx):\n    return []\n"))
    ctx = EstimateContext(
        object_id="ra_063",
        joints={"part_02": {"type": "prismatic"}},
        evidence={
            "part_02": {
                "labels": ["pull-out drawer pan"],
                "initial_estimate": {"axis_world": [1.0, 0.0, 0.0], "lower": 0.0, "upper": 0.03},
            }
        },
    )
    viewer = LiveViewer(tmp_path / "live")
    viewer.prepare()
    responses = iter([
        "```json\n{\"updates\":{\"part_02\":{\"state\":\"need_fix\",\"upper_delta\":{\"unit\":\"mm\",\"value\":3}}}}\n```",
        "```json\n{\"updates\":{\"part_02\":{\"state\":\"need_fix\",\"upper_delta\":{\"unit\":\"mm\",\"value\":10},\"reason\":\"extend drawer search\"}}}\n```",
    ])
    prompts = []
    callbacks = []

    def fake_request(messages):
        prompts.append(messages)
        return next(responses)

    def fake_on_report(iteration, report):
        callbacks.append((iteration, report.passed))
        return {
            "object_viewers": [
                {
                    "joint_name": "part_02",
                    "href": "ra_063/agent_viz/part_02/step_viewer.html",
                }
            ]
        }

    result = run_agent_loop(
        candidate,
        ctx,
        config=AgentLoopConfig(
            max_iterations=2,
            provider="openrouter",
            model="fake-model",
            thinking_level="high",
            base_url="https://example.invalid/v1",
            api_key="not-used-by-fake",
            heartbeat_interval_seconds=0.01,
        ),
        request_fn=fake_request,
        viewer=viewer,
        on_report=fake_on_report,
    )
    report = result.report

    assert report.passed is True
    assert result.iterations == 2
    assert report.estimates[0].joint_name == "part_02"
    updated = candidate.read_text()
    assert "OUTSIDE_VALUE = 7" in updated
    assert "Deterministic artifact generated from structured agent actions" in updated
    assert "'upper': 0.04" in updated
    assert len(prompts) == 2
    assert "must be one of" in json.dumps(prompts[1])
    assert callbacks == [(2, True)]

    events = [
        json.loads(line)
        for line in (viewer.events_path).read_text().splitlines()
    ]
    validation_events = [
        event for event in events if event["event"] == "validation_finished"
    ]
    assert [event["iteration"] for event in validation_events] == [1, 2]
    assert [event["status"] for event in validation_events] == ["failed", "passed"]
    baseline_events = [
        event for event in events if event["event"] == "baseline_validation_finished"
    ]
    assert baseline_events == []
    edit_events = [event for event in events if event["event"] == "agent_edit_applied"]
    assert "editable_code" in edit_events[-1]["details"]
    assert "editable_diff" in edit_events[-1]["details"]
    action_events = [event for event in events if event["event"] == "agent_action_applied"]
    assert [event["status"] for event in action_events] == ["failed", "passed"]
    viz_events = [event for event in events if event["event"] == "object_visualization_written"]
    assert [event["iteration"] for event in viz_events] == [2]
    assert viz_events[-1]["details"]["object_viewers"][0]["joint_name"] == "part_02"
    assert events[-1]["event"] == "agent_loop_finished"
    assert events[-1]["iteration"] == 2


def test_agent_loop_exposes_last_feedback_to_candidate_runtime_context(tmp_path):
    candidate = tmp_path / "estimate_limit.py"
    candidate.write_text(_scaffold(
        "def estimate_limits(ctx):\n"
        "    return [LimitEstimate(joint_name='part_02', lower=0.0, upper=0.03)]\n"
    ))
    ctx = EstimateContext(
        object_id="ra_063",
        joints={"part_02": {"type": "prismatic"}},
        evidence={
            "part_02": {
                "labels": ["pull-out drawer pan"],
                "initial_estimate": {"axis_world": [1.0, 0.0, 0.0], "lower": 0.0, "upper": 0.03},
            }
        },
    )
    viewer = LiveViewer(tmp_path / "live")
    viewer.prepare()

    def fake_request(_messages):
        return (
            "```json\n"
            "{\"updates\":{\"part_02\":{\"state\":\"need_fix\","
            "\"upper_delta\":{\"unit\":\"mm\",\"value\":10},"
            "\"reason\":\"upper endpoint q=0.03 intersects; extend one step\"}}}\n"
            "```"
        )

    def fake_validate(report):
        upper = report.estimates[0].upper if report.estimates else 0.0
        if upper >= 0.04:
            return CandidateReport(passed=True, estimates=report.estimates)
        return CandidateReport(
            passed=False,
            estimates=report.estimates,
            errors=[
                "part_02: candidate motion still intersects parent/static geometry "
                "at upper endpoint q=0.03"
            ],
        )

    result = run_agent_loop(
        candidate,
        ctx,
        config=AgentLoopConfig(
            max_iterations=1,
            provider="openrouter",
            model="fake-model",
            thinking_level="high",
            base_url="https://example.invalid/v1",
            api_key="not-used-by-fake",
            heartbeat_interval_seconds=0.01,
        ),
        request_fn=fake_request,
        viewer=viewer,
        validate_report=fake_validate,
    )

    assert result.report.passed is True
    assert result.report.estimates[0].upper == 0.04


def test_agent_loop_emits_api_waiting_heartbeat_while_request_blocks(tmp_path):
    candidate = tmp_path / "estimate_limit.py"
    candidate.write_text(_scaffold(
        "def estimate_limits(ctx):\n"
        "    return [LimitEstimate(joint_name='part_02', lower=0.0, upper=0.15)]\n"
    ))
    ctx = EstimateContext(
        object_id="ra_063",
        joints={"part_02": {"type": "prismatic"}},
        evidence={},
    )
    viewer = LiveViewer(tmp_path / "live")
    viewer.prepare()

    def slow_request(_messages):
        time.sleep(0.04)
        return (
            "```python\n"
            "def estimate_limits(ctx):\n"
            "    return [LimitEstimate(joint_name='part_02', lower=0.0, upper=0.15)]\n"
            "```"
        )

    run_agent_loop(
        candidate,
        ctx,
        config=AgentLoopConfig(
            max_iterations=1,
            provider="openrouter",
            model="fake-model",
            thinking_level="high",
            base_url="https://example.invalid/v1",
            api_key="not-used-by-fake",
            heartbeat_interval_seconds=0.01,
        ),
        request_fn=slow_request,
        viewer=viewer,
    )

    events = [
        json.loads(line)
        for line in viewer.events_path.read_text().splitlines()
    ]
    waiting_events = [event for event in events if event["event"] == "api_waiting"]
    assert waiting_events
    assert waiting_events[-1]["status"] == "running"
    assert waiting_events[0]["details"]["elapsed_seconds"] == 0.0
    assert waiting_events[-1]["details"]["elapsed_seconds"] > 0


def test_agent_loop_attaches_iteration_to_api_failure(tmp_path):
    candidate = tmp_path / "estimate_limit.py"
    candidate.write_text(_scaffold(
        "def estimate_limits(ctx):\n"
        "    return [LimitEstimate(joint_name='part_02', lower=0.0, upper=0.15)]\n"
    ))
    ctx = EstimateContext(
        object_id="ra_063",
        joints={"part_02": {"type": "prismatic"}},
        evidence={},
    )
    viewer = LiveViewer(tmp_path / "live")
    viewer.prepare()

    def failing_request(_messages):
        raise AgentLoopError("API request timed out after 12.5s while waiting for model response")

    with pytest.raises(AgentLoopError) as exc_info:
        run_agent_loop(
            candidate,
            ctx,
            config=AgentLoopConfig(
                max_iterations=1,
                provider="openrouter",
                model="fake-model",
                thinking_level="high",
                base_url="https://example.invalid/v1",
                api_key="not-used-by-fake",
                heartbeat_interval_seconds=0.01,
            ),
            request_fn=failing_request,
            viewer=viewer,
        )

    assert getattr(exc_info.value, "agent_iteration") == 1
    events = [
        json.loads(line)
        for line in viewer.events_path.read_text().splitlines()
    ]
    failed_api = [event for event in events if event["event"] == "api_call_finished"][-1]
    assert failed_api["iteration"] == 1
    assert failed_api["status"] == "failed"


def test_agent_loop_retries_when_report_validator_rejects_motion(tmp_path):
    candidate = tmp_path / "estimate_limit.py"
    candidate.write_text(_scaffold(
        "def estimate_limits(ctx):\n"
        "    return [LimitEstimate(joint_name='part_00', lower=0.0, upper=1.0,\n"
        "                          axis_world=[0.0, 1.0, 0.0])]\n"
    ))
    ctx = EstimateContext(
        object_id="ra_063",
        joints={"part_00": {"type": "revolute", "axis_world": [0, 1, 0]}},
        evidence={"part_00": {"labels": ["temperature knob"]}},
    )
    responses = iter([
        (
            "```json\n"
            "{\"updates\":{\"part_00\":{\"state\":\"need_fix\","
            "\"axis\":{\"op\":\"rotate\",\"axis_world\":[0.08715574274765817,0.9961946980917455,0.0]},"
            "\"reason\":\"try nearby knob normal\"}}}\n"
            "```"
        ),
        (
            "```json\n"
            "{\"updates\":{\"part_00\":{\"state\":\"need_fix\","
            "\"axis\":{\"op\":\"rotate\",\"axis_world\":[0.0,1.0,0.0]},"
            "\"reason\":\"previous rotated axis collided; return one step\"}}}\n"
            "```"
        ),
    ])
    feedback_prompts = []

    def fake_request(messages):
        feedback_prompts.append(json.dumps(messages))
        return next(responses)

    def fake_validator(report):
        estimate = report.estimates[0] if report.estimates else None
        if estimate and estimate.axis_world and estimate.axis_world[0] > 0:
            return CandidateReport(
                passed=False,
                estimates=report.estimates,
                errors=["part_00: candidate motion collides/interferes at q=1"],
            )
        return report

    result = run_agent_loop(
        candidate,
        ctx,
        config=AgentLoopConfig(
            max_iterations=2,
            provider="openrouter",
            model="fake-model",
            thinking_level="high",
            base_url="https://example.invalid/v1",
            api_key="not-used-by-fake",
        ),
        request_fn=fake_request,
        validate_report=fake_validator,
    )

    assert result.report.passed is True
    assert result.iterations == 2
    assert "collides/interferes" in feedback_prompts[1]


def test_incremental_action_errors_allow_one_action_budget_per_joint_in_one_iteration():
    previous = CandidateReport(
        passed=True,
        estimates=[
            LimitEstimate(joint_name="part_00", lower=0.0, upper=1.0, axis_world=[0, 1, 0], axis_label="+Y"),
            LimitEstimate(joint_name="part_02", lower=0.0, upper=0.10, axis_world=[0, 1, 0], axis_label="+Y"),
        ],
    )
    current = CandidateReport(
        passed=True,
        estimates=[
            LimitEstimate(joint_name="part_00", lower=-0.001, upper=1.001, axis_world=[0.08715574274765817, 0.9961946980917455, 0], axis_label=None),
            LimitEstimate(joint_name="part_02", lower=0.0, upper=0.101, axis_world=[0.08715574274765817, 0.9961946980917455, 0], axis_label=None),
        ],
    )

    assert incremental_action_errors(previous, current) == []


def test_incremental_action_errors_reject_limit_step_larger_than_one_mm():
    previous = CandidateReport(
        passed=True,
        estimates=[
            LimitEstimate(joint_name="part_02", lower=0.0, upper=0.10, axis_world=[0, -1, 0], axis_label="-Y"),
        ],
    )
    current = CandidateReport(
        passed=True,
        estimates=[
            LimitEstimate(joint_name="part_02", lower=0.0, upper=0.103, axis_world=[0, -1, 0], axis_label="-Y"),
        ],
    )

    errors = incremental_action_errors(previous, current)

    assert errors == [
        "part_02: upper changed by 0.003 m; allowed prismatic limit steps are +/-10, +/-5, +/-2.5, +/-1, +/-0.5 mm"
    ]
