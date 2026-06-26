"""Schemas exposed to estimate_limit.py and the harness."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LimitEstimate:
    joint_name: str
    lower: float
    upper: float
    axis_world: list[float] | tuple[float, float, float] | None = None
    axis_label: str | None = None
    confidence: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        if lower > upper:
            raise ValueError(
                f"{self.joint_name}: lower must be <= upper, got {lower} > {upper}"
            )
        if self.confidence is not None:
            confidence = float(self.confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"{self.joint_name}: confidence must be in [0, 1], got {confidence}"
                )
        if self.axis_world is not None:
            if len(self.axis_world) != 3:
                raise ValueError(
                    f"{self.joint_name}: axis_world must have exactly 3 values"
                )
            axis = [float(value) for value in self.axis_world]
            if sum(value * value for value in axis) <= 1e-12:
                raise ValueError(f"{self.joint_name}: axis_world must be non-zero")


@dataclass(frozen=True)
class EstimateContext:
    object_id: str
    joints: dict[str, dict[str, Any]]
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompileSignal:
    severity: str
    kind: str
    code: str
    summary: str
    detail: str = ""
    source: str = "harness"
    group: str = "qc"
    blocking: bool = False
    joint_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value not in (None, "")}


@dataclass(frozen=True)
class CompileSignalBundle:
    status: str
    summary: str
    signals: list[CompileSignal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "signals": [signal.to_dict() for signal in self.signals],
        }


@dataclass(frozen=True)
class CandidateReport:
    passed: bool
    estimates: list[LimitEstimate]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "passed": self.passed,
                "estimates": [asdict(estimate) for estimate in self.estimates],
                "errors": list(self.errors),
                "warnings": list(self.warnings),
                "details": dict(self.details),
            },
            indent=2,
        )
