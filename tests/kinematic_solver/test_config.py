from pathlib import Path

from post_process.kinematic_solver.utils.config import (
    ComparisonConfig,
    CollisionConstraintConfig,
    SearchConfig,
    V1_COACD_RUN_PARAMS,
    V1_CONDA_PYTHON,
    V1_PINNED_COACD_VERSION,
    V1_TEN_IDS,
    V1_VHACD_CACHE_METADATA,
    V1DatasetRoots,
)


def test_v1_ten_ids_is_ten_zero_padded_ra_NNN_ending_in_7():
    assert V1_TEN_IDS == [
        "ra_007", "ra_017", "ra_027", "ra_037", "ra_047",
        "ra_057", "ra_067", "ra_077", "ra_087", "ra_097",
    ]


def test_v1_coacd_run_params_has_17_fields_with_real_metric_true():
    assert set(V1_COACD_RUN_PARAMS) == {
        "threshold", "preprocess_mode", "preprocess_resolution",
        "resolution", "mcts_iterations", "mcts_max_depth", "mcts_nodes",
        "pca", "merge", "decimate", "max_ch_vertex", "extrude",
        "extrude_margin", "apx_mode", "seed", "max_convex_hull",
        "real_metric",
    }
    assert V1_COACD_RUN_PARAMS["real_metric"] is True
    assert V1_COACD_RUN_PARAMS["seed"] == 0


def test_v1_vhacd_cache_metadata_pins_coacd_1_0_9():
    assert V1_VHACD_CACHE_METADATA == {"backend": "coacd", "version": "1.0.9"}
    assert V1_PINNED_COACD_VERSION == "1.0.9"


def test_v1_conda_python_is_local_env_isaacsim():
    assert V1_CONDA_PYTHON == Path("/home/mi/anaconda3/envs/env-isaacsim/bin/python")


def test_v1_dataset_roots_default_to_baked_and_realappliance():
    roots = V1DatasetRoots()
    assert roots.converter_output_root == Path("data/RealAppliance-4view-0515-baked")
    assert roots.source_root == Path("data/RealAppliance")


def test_v1_dataset_roots_builds_aligned_usd_path_from_object_id():
    roots = V1DatasetRoots(source_root=Path("/data/RealAppliance"))

    assert roots.aligned_usd_for("ra_007") == Path(
        "/data/RealAppliance/source/model/007/Aligned.usd"
    )
    assert roots.aligned_usd_for("custom_001") == Path(
        "/data/RealAppliance/source/model/custom_001/Aligned.usd"
    )


def test_search_config_v1_defaults():
    cfg = SearchConfig()
    assert cfg.prismatic_step_m == 0.01
    assert abs(cfg.revolute_step_rad - 0.034906585) < 1e-6
    assert cfg.initial_high_prismatic_m == 0.5
    assert abs(cfg.initial_high_revolute_rad - 3.141592653) < 1e-6
    assert cfg.allow_initial_penetration is False
    assert cfg.viz_stride == 5


def test_collision_constraint_config_v1_strict_default():
    cfg = CollisionConstraintConfig()
    assert cfg.allow_initial_penetration is False


def test_comparison_config_default_threshold():
    cfg = ComparisonConfig()
    assert cfg.success_rel_err_threshold == 0.10
