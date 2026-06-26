from post_process.kinematic_solver.utils.config import SearchConfig
from post_process.kinematic_solver.utils.solver import estimate_range, find_max_valid_q_directed


def test_find_max_valid_q_directed_returns_last_valid_grid_point():
    def evaluator(q_signed):
        return q_signed <= 0.25

    out = find_max_valid_q_directed(
        evaluator=evaluator,
        direction=1,
        step=0.1,
        initial_high=0.5,
    )

    assert out["status"] == "ok"
    assert abs(out["q"] - 0.2) < 1e-9


def test_estimate_range_emits_per_direction_statuses():
    joint = {
        "object_id": "ra_test",
        "joint_name": "j0",
        "type": "prismatic",
        "canonical_unit": "meters",
    }

    def evaluator(q_signed):
        return -0.15 <= q_signed <= 0.25

    out = estimate_range(joint, evaluator, SearchConfig(prismatic_step_m=0.1))

    assert out["status"] == "ok"
    assert out["status_lower"] == "ok"
    assert out["status_upper"] == "ok"
    assert abs(out["predicted_lower"] + 0.1) < 1e-9
    assert abs(out["predicted_upper"] - 0.2) < 1e-9


def test_estimate_range_calibrates_evaluator_before_initial_collision_check():
    joint = {
        "object_id": "ra_test",
        "joint_name": "j0",
        "type": "prismatic",
        "canonical_unit": "meters",
    }

    class BaselineOverlapEvaluator:
        def __init__(self):
            self.calibrated = False

        def calibrate_at_zero(self):
            self.calibrated = True
            return True

        def __call__(self, q_signed):
            if q_signed == 0.0 and not self.calibrated:
                return False
            return -0.15 <= q_signed <= 0.25

    out = estimate_range(
        joint,
        BaselineOverlapEvaluator(),
        SearchConfig(prismatic_step_m=0.1),
    )

    assert out["status"] == "ok"
    assert out["trace_upper"][0] == {"q": 0.0, "valid": True}
    assert len(out["trace_upper"]) > 1
