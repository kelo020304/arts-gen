import pytest

from post_process.kinematic_solver.sdk.motion_search import (
    AXIS_ACTIONS,
    refine_positive_limit,
    search_axis_actions,
)


def test_axis_actions_cover_signed_xyz_for_translation_and_rotation():
    labels = [action.label for action in AXIS_ACTIONS]

    assert labels == ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    assert AXIS_ACTIONS[0].axis_world == (1.0, 0.0, 0.0)
    assert AXIS_ACTIONS[3].axis_world == (0.0, -1.0, 0.0)


def test_refine_positive_limit_scans_coarse_then_halves_resolution():
    calls = []

    def evaluator(q):
        calls.append(round(q, 6))
        return q <= 0.23

    result = refine_positive_limit(
        evaluator,
        initial_step=0.10,
        max_limit=0.50,
        min_step=0.0125,
    )

    assert result.status == "ok"
    assert result.limit == pytest.approx(0.225)
    assert calls[:4] == [0.0, 0.1, 0.2, 0.3]
    assert any(sample["valid"] is False for sample in result.samples)


def test_search_axis_actions_ranks_longest_valid_motion():
    def evaluator_for_axis(action):
        threshold = 0.35 if action.label == "-Y" else 0.12

        def evaluator(q):
            return q <= threshold

        return evaluator

    results = search_axis_actions(
        evaluator_for_axis,
        initial_step=0.10,
        max_limit=0.50,
        min_step=0.025,
    )

    assert results[0].axis_label == "-Y"
    assert results[0].axis_world == (0.0, -1.0, 0.0)
    assert results[0].limit == pytest.approx(0.35, abs=0.025)
