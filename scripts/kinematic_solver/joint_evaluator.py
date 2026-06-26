"""Apply a joint q and evaluate geometric constraints."""

from __future__ import annotations

import numpy as np

from .backend import CollisionBackend
from .manual_transform import apply_joint_transform_world_baked


class JointEvaluator:
    def __init__(self, *, joint: dict, constraints: list, backend: CollisionBackend) -> None:
        self.joint = joint
        self.constraints = constraints
        self.backend = backend

    def reset_pose_to_zero(self) -> None:
        self.backend.reset_to_identity()

    def __call__(self, q_signed: float) -> bool:
        self.reset_pose_to_zero()
        direction = 1 if q_signed >= 0 else -1
        rotation, translation = apply_joint_transform_world_baked(
            joint_type=self.joint["type"],
            direction=direction,
            q_abs=abs(float(q_signed)),
            axis_world=np.asarray(self.joint["axis_world"], dtype=float),
            origin_world=np.asarray(self.joint["origin_world"], dtype=float),
        )
        for part in self.joint["moving_parts"]:
            self.backend.set_pose(part, rotation, translation)
        for constraint in self.constraints:
            if hasattr(constraint, "set_current_q"):
                constraint.set_current_q(float(q_signed))
        return all(constraint() for constraint in self.constraints)
