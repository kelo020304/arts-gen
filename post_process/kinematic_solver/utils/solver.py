"""Forward-scan joint range estimator."""

from __future__ import annotations

from collections.abc import Callable

from .config import SearchConfig


def find_max_valid_q_directed(
    *,
    evaluator: Callable[[float], bool],
    direction: int,
    step: float,
    initial_high: float,
) -> dict:
    """Scan away from zero and return the last valid grid point."""
    sign = 1 if direction >= 0 else -1
    if not evaluator(0.0):
        return {"status": "initial_collision", "q": None, "samples": [{"q": 0.0, "valid": False}]}

    q_abs = 0.0
    last_valid = 0.0
    samples = [{"q": 0.0, "valid": True}]
    while q_abs + step <= initial_high + 1e-12:
        q_abs = round(q_abs + step, 12)
        q_signed = sign * q_abs
        valid = bool(evaluator(q_signed))
        samples.append({"q": q_signed, "valid": valid})
        if not valid:
            break
        last_valid = q_signed

    return {"status": "ok", "q": last_valid, "samples": samples}


def estimate_range(joint: dict, evaluator: Callable[[float], bool],
                   config: SearchConfig | None = None) -> dict:
    cfg = config or SearchConfig()
    if hasattr(evaluator, "calibrate_at_zero"):
        evaluator.calibrate_at_zero()
    if joint["type"] == "prismatic":
        step = cfg.prismatic_step_m
        high = cfg.initial_high_prismatic_m
    elif joint["type"] == "revolute":
        step = cfg.revolute_step_rad
        high = cfg.initial_high_revolute_rad
    else:
        raise ValueError(f"unsupported joint type: {joint['type']!r}")

    upper = find_max_valid_q_directed(
        evaluator=evaluator, direction=1, step=step, initial_high=high
    )
    lower = find_max_valid_q_directed(
        evaluator=evaluator, direction=-1, step=step, initial_high=high
    )
    status_upper = upper["status"]
    status_lower = lower["status"]
    status = "ok" if status_upper == "ok" and status_lower == "ok" else "partial"
    return {
        "object_id": joint.get("object_id"),
        "joint_name": joint.get("joint_name"),
        "type": joint["type"],
        "canonical_unit": joint.get(
            "canonical_unit",
            "meters" if joint["type"] == "prismatic" else "radians",
        ),
        "predicted_lower": lower["q"],
        "predicted_upper": upper["q"],
        "status": status,
        "status_lower": status_lower,
        "status_upper": status_upper,
        "trace_lower": lower["samples"],
        "trace_upper": upper["samples"],
    }
