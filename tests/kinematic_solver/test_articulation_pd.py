import sys
import types
from pathlib import Path

import numpy as np
import pytest

from post_process.kinematic_solver.utils.errors import SchemaMismatchError
from post_process.kinematic_solver.utils.validate import ValidationContext, _articulation_pd_validate


def _install_fake_isaac(monkeypatch, calls):
    modules = {
        "omni": types.ModuleType("omni"),
        "omni.isaac": types.ModuleType("omni.isaac"),
        "omni.isaac.kit": types.ModuleType("omni.isaac.kit"),
        "omni.isaac.core": types.ModuleType("omni.isaac.core"),
        "omni.isaac.core.articulations": types.ModuleType("omni.isaac.core.articulations"),
        "omni.isaac.core.utils": types.ModuleType("omni.isaac.core.utils"),
        "omni.isaac.core.utils.types": types.ModuleType("omni.isaac.core.utils.types"),
        "omni.isaac.core.utils.stage": types.ModuleType("omni.isaac.core.utils.stage"),
    }

    class SimulationApp:
        def __init__(self, config):
            assert config == {"headless": True}

        def close(self):
            calls["closed"] = True

    class Scene:
        def add_default_ground_plane(self):
            calls["ground"] = True

        def add(self, obj):
            calls["added"] = obj

    class World:
        def __init__(self, stage_units_in_meters):
            assert stage_units_in_meters == 1.0
            self.scene = Scene()

        def reset(self):
            calls["reset"] = True

        def step(self, render=False):
            assert render is False
            calls["steps"] = calls.get("steps", 0) + 1

    class Articulation:
        def __init__(self, prim_path):
            assert prim_path == "/World/ra_test"
            self.dof_names = calls.get("dof_names", ["drawer_joint"])
            self.q = 0.0

        def initialize(self):
            calls["initialized"] = True

        def apply_action(self, action):
            assert len(action.joint_indices) == 1
            assert int(action.joint_indices[0]) == 0
            self.q = float(action.joint_positions[0])

        def get_applied_joint_efforts(self):
            return np.array([1.0])

        def get_joint_positions(self):
            return np.array([self.q])

    class ArticulationAction:
        def __init__(self, joint_positions, joint_indices):
            self.joint_positions = joint_positions
            self.joint_indices = joint_indices

    def add_reference_to_stage(usd_path, prim_path):
        calls["reference"] = (usd_path, prim_path)

    modules["omni.isaac.kit"].SimulationApp = SimulationApp
    modules["omni.isaac.core"].World = World
    modules["omni.isaac.core.articulations"].Articulation = Articulation
    modules["omni.isaac.core.utils.types"].ArticulationAction = ArticulationAction
    modules["omni.isaac.core.utils.stage"].add_reference_to_stage = add_reference_to_stage
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_articulation_pd_validate_runs_to_completion_and_closes_sim(tmp_path, monkeypatch):
    calls = {}
    _install_fake_isaac(monkeypatch, calls)
    predicted = tmp_path / "predicted.usd"
    predicted.write_text("#usda 1.0\n")
    ctx = ValidationContext(
        prediction={"status": "ok", "predicted_lower": 0.0, "predicted_upper": 1.0},
        vlm_oracle_model={},
        joint_name="joint0",
        object_id="ra_test",
        usd_path=predicted,
        predicted_usd_path=predicted,
        part_to_obj_path={},
        vhacd_cache_root=tmp_path,
        coacd_run_params={},
        vhacd_cache_metadata={},
        stage_metadata={"joint_prim_paths": {"joint0": "/World/ra_test/drawer_joint"}},
    )

    result = _articulation_pd_validate(ctx)

    assert result["validation_status"] == "passed"
    assert result["max_torque_Nm"] == pytest.approx(1.0)
    assert result["reach_error_rel"] == pytest.approx(0.0)
    assert calls["reference"] == (str(predicted), "/World/ra_test")
    assert calls["closed"] is True


def test_articulation_pd_validate_raises_when_joint_dof_is_missing(tmp_path, monkeypatch):
    calls = {"dof_names": ["unrelated_joint"]}
    _install_fake_isaac(monkeypatch, calls)
    predicted = tmp_path / "predicted.usd"
    predicted.write_text("#usda 1.0\n")
    ctx = ValidationContext(
        prediction={"status": "ok", "predicted_lower": 0.0, "predicted_upper": 1.0},
        vlm_oracle_model={},
        joint_name="joint0",
        object_id="ra_test",
        usd_path=predicted,
        predicted_usd_path=predicted,
        part_to_obj_path={},
        vhacd_cache_root=tmp_path,
        coacd_run_params={},
        vhacd_cache_metadata={},
        stage_metadata={"joint_prim_paths": {"joint0": "/World/ra_test/drawer_joint"}},
    )

    with pytest.raises(SchemaMismatchError, match="joint0"):
        _articulation_pd_validate(ctx)

    assert calls["closed"] is True
