from post_process.kinematic_solver.utils.config import CollisionConstraintConfig
from post_process.kinematic_solver.utils.constraints import CollisionConstraint


class PairBackend:
    def __init__(self, pairs):
        self.pairs = set(pairs)

    def overlapping_pairs(self, moving_parts, static_parts):
        allowed = {
            (moving, static)
            for moving in moving_parts
            for static in static_parts
            if moving != static
        }
        return sorted(self.pairs & allowed)

    def overlap(self, moving_parts, static_parts):
        return bool(self.overlapping_pairs(moving_parts, static_parts))


def test_collision_constraint_ignores_baseline_pairs_but_blocks_new_pairs():
    backend = PairBackend({("part_00", "body")})
    constraint = CollisionConstraint(
        ["part_00"],
        ["body", "part_01"],
        backend=backend,
        config=CollisionConstraintConfig(allow_initial_penetration=True),
    )

    assert constraint.calibrate_at_zero() is True
    assert constraint.check() is True

    backend.pairs = {("part_00", "body"), ("part_00", "part_01")}

    assert constraint.check() is False
