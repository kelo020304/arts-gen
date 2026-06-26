"""Small signed-axis action search helpers for estimate_limit.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


Evaluator = Callable[[float], bool]


@dataclass(frozen=True)
class AxisAction:
    label: str
    axis_world: tuple[float, float, float]


@dataclass(frozen=True)
class RangeSearchResult:
    status: str
    limit: float
    samples: list[dict]


@dataclass(frozen=True)
class AxisSearchResult:
    axis_label: str
    axis_world: tuple[float, float, float]
    status: str
    limit: float
    samples: list[dict]


AXIS_ACTIONS: tuple[AxisAction, ...] = (
    AxisAction("+X", (1.0, 0.0, 0.0)),
    AxisAction("-X", (-1.0, 0.0, 0.0)),
    AxisAction("+Y", (0.0, 1.0, 0.0)),
    AxisAction("-Y", (0.0, -1.0, 0.0)),
    AxisAction("+Z", (0.0, 0.0, 1.0)),
    AxisAction("-Z", (0.0, 0.0, -1.0)),
)


def refine_positive_limit(
    evaluator: Evaluator,
    *,
    initial_step: float,
    max_limit: float,
    min_step: float,
) -> RangeSearchResult:
    """Scan outward, then binary-refine between last valid and first invalid."""
    if initial_step <= 0.0 or max_limit <= 0.0 or min_step <= 0.0:
        raise ValueError("initial_step, max_limit, and min_step must be positive")
    samples: list[dict] = []
    if not evaluator(0.0):
        return RangeSearchResult(
            status="initial_invalid",
            limit=0.0,
            samples=[{"q": 0.0, "valid": False}],
        )
    samples.append({"q": 0.0, "valid": True})

    low = 0.0
    high = None
    q = 0.0
    while q + initial_step <= max_limit + 1e-12:
        q = round(q + initial_step, 12)
        valid = bool(evaluator(q))
        samples.append({"q": q, "valid": valid})
        if valid:
            low = q
            continue
        high = q
        break
    if high is None:
        return RangeSearchResult(status="ok", limit=low, samples=samples)

    while high - low > min_step + 1e-12:
        mid = (low + high) * 0.5
        valid = bool(evaluator(mid))
        samples.append({"q": mid, "valid": valid})
        if valid:
            low = mid
        else:
            high = mid
    return RangeSearchResult(status="ok", limit=low, samples=samples)


def search_axis_actions(
    evaluator_for_axis: Callable[[AxisAction], Evaluator],
    *,
    initial_step: float,
    max_limit: float,
    min_step: float,
    actions: tuple[AxisAction, ...] = AXIS_ACTIONS,
) -> list[AxisSearchResult]:
    results = []
    for action in actions:
        result = refine_positive_limit(
            evaluator_for_axis(action),
            initial_step=initial_step,
            max_limit=max_limit,
            min_step=min_step,
        )
        results.append(AxisSearchResult(
            axis_label=action.label,
            axis_world=action.axis_world,
            status=result.status,
            limit=result.limit,
            samples=result.samples,
        ))
    return sorted(results, key=lambda item: item.limit, reverse=True)
