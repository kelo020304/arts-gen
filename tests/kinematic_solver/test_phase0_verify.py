import hashlib
import importlib.metadata
import json
import types

import pytest

from post_process.kinematic_solver.utils.config import (
    V1_COACD_RUN_PARAMS,
    V1_VHACD_CACHE_METADATA,
    V1DatasetRoots,
)
from post_process.kinematic_solver.utils.errors import (
    CoacdParamsMissingError,
    DatasetFingerprintDriftError,
    DependencyMissingError,
    FingerprintAlreadyWrittenError,
    MissingModelError,
    SchemaMismatchError,
    VhacdCacheMissingError,
    VhacdParamsMismatchError,
)
from post_process.kinematic_solver.utils.phase0_verify import (
    assert_dataset_fingerprint_matches,
    preflight_dependencies,
    preflight_inputs,
    verify_cache,
    write_dataset_fingerprint,
)


def _make_fake_roots(tmp_path, ids=("ra_007",)):
    co = tmp_path / "converter_output"
    src_root = tmp_path / "source"
    for object_id in ids:
        src_id = object_id.removeprefix("ra_")
        (co / "raw/finaljson").mkdir(parents=True, exist_ok=True)
        (co / f"raw/finaljson/{object_id}.json").write_text("{}")
        obj_dir = co / f"raw/partseg/{object_id}/objs"
        obj_dir.mkdir(parents=True, exist_ok=True)
        (obj_dir / "body.obj").write_text("v 0 0 0\n")
        (obj_dir / "part_00.obj").write_text("v 0 0 0\n")
        (src_root / f"source/model/{src_id}").mkdir(parents=True, exist_ok=True)
        (src_root / f"source/model/{src_id}/Aligned.usd").write_text("#usda 1.0\n")
    return V1DatasetRoots(converter_output_root=co, source_root=src_root)


def _mock_dependencies(monkeypatch):
    def fake_import(name):
        if name == "coacd":
            return types.SimpleNamespace(run_coacd=object())
        return types.SimpleNamespace()

    def fake_version(pkg):
        if pkg == "coacd":
            return "1.0.9"
        return importlib.metadata.version(pkg)

    def fake_signature(_fn):
        return types.SimpleNamespace(
            parameters={k: object() for k in V1_COACD_RUN_PARAMS}
        )

    monkeypatch.setattr("importlib.import_module", fake_import)
    monkeypatch.setattr("importlib.metadata.version", fake_version)
    monkeypatch.setattr("inspect.signature", fake_signature)


def _write_valid_cache(roots, object_id="ra_007"):
    co = roots.converter_output_root
    (co / f"raw/vlm_oracle").mkdir(parents=True, exist_ok=True)
    (co / f"raw/vlm_oracle/{object_id}.json").write_text(
        json.dumps({"joints": {"joint0": {}}})
    )
    (co / f"raw/stage_metadata").mkdir(parents=True, exist_ok=True)
    (co / f"raw/stage_metadata/{object_id}.json").write_text(
        json.dumps({
            "object_id": object_id,
            "source_id": object_id.removeprefix("ra_"),
            "meters_per_unit": 1.0,
            "joint_prim_paths": {"joint0": "/World/joint0"},
            "stage_up_axis": "Z",
        })
    )
    cache_dir = co / f"raw/vhacd/{object_id}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    obj_dir = co / f"raw/partseg/{object_id}/objs"
    for obj in sorted(obj_dir.glob("*.obj")):
        (cache_dir / f"{obj.stem}.json").write_text(
            json.dumps({
                "object_id": object_id,
                "part_name": obj.stem,
                "source_obj": str(obj),
                "source_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
                "vhacd_cache_metadata": dict(V1_VHACD_CACHE_METADATA),
                "coacd_run_params": dict(V1_COACD_RUN_PARAMS),
                "frame": "world_baked",
                "hulls": [{"hull_index": 0, "vertices": [], "faces": []}],
                "n_hulls": 1,
            })
        )


def test_preflight_dependencies_passes_when_all_present(monkeypatch):
    _mock_dependencies(monkeypatch)
    preflight_dependencies()


def test_preflight_dependencies_raises_when_coacd_missing(monkeypatch):
    def fake_import(name):
        if name == "coacd":
            raise ImportError("simulated missing")
        return types.SimpleNamespace()

    monkeypatch.setattr("importlib.import_module", fake_import)
    with pytest.raises(DependencyMissingError) as exc:
        preflight_dependencies()
    assert "coacd" in str(exc.value)
    assert "pip install coacd==1.0.9" in str(exc.value)


def test_preflight_dependencies_raises_on_coacd_version_drift(monkeypatch):
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: types.SimpleNamespace(run_coacd=object()),
    )

    def fake_version(pkg):
        return "1.0.6" if pkg == "coacd" else importlib.metadata.version(pkg)

    monkeypatch.setattr("importlib.metadata.version", fake_version)
    with pytest.raises(DependencyMissingError) as exc:
        preflight_dependencies()
    assert "1.0.6" in str(exc.value)
    assert "1.0.9" in str(exc.value)


def test_preflight_dependencies_raises_when_coacd_params_missing(monkeypatch):
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: types.SimpleNamespace(run_coacd=object()),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda pkg: "1.0.9")
    monkeypatch.setattr(
        "inspect.signature",
        lambda _fn: types.SimpleNamespace(parameters={"threshold": object()}),
    )
    with pytest.raises(CoacdParamsMissingError) as exc:
        preflight_dependencies()
    assert "real_metric" in str(exc.value)


def test_preflight_inputs_passes_when_files_present(tmp_path, monkeypatch):
    _mock_dependencies(monkeypatch)
    roots = _make_fake_roots(tmp_path)
    preflight_inputs(roots, ids=["ra_007"])


def test_preflight_inputs_raises_missing_model_when_finaljson_absent(tmp_path, monkeypatch):
    _mock_dependencies(monkeypatch)
    roots = _make_fake_roots(tmp_path)
    (roots.converter_output_root / "raw/finaljson/ra_007.json").unlink()
    with pytest.raises(MissingModelError):
        preflight_inputs(roots, ids=["ra_007"])


def test_preflight_inputs_raises_missing_model_when_aligned_usd_absent(tmp_path, monkeypatch):
    _mock_dependencies(monkeypatch)
    roots = _make_fake_roots(tmp_path)
    (roots.source_root / "source/model/007/Aligned.usd").unlink()
    with pytest.raises(MissingModelError):
        preflight_inputs(roots, ids=["ra_007"])


def test_write_dataset_fingerprint_emits_json(tmp_path):
    roots = _make_fake_roots(tmp_path)
    run_dir = tmp_path / "run"
    out = write_dataset_fingerprint(roots, run_dir, ids=["ra_007"])
    fp = json.loads(out.read_text())
    assert out == run_dir / "dataset_fingerprint.json"
    assert fp["converter_output_root"].endswith("converter_output")
    assert "ra_007/part_00.obj" in fp["objs"]
    assert len(fp["objs"]["ra_007/part_00.obj"]) == 64


def test_write_dataset_fingerprint_refuses_overwrite(tmp_path):
    roots = _make_fake_roots(tmp_path)
    run_dir = tmp_path / "run"
    write_dataset_fingerprint(roots, run_dir, ids=["ra_007"])
    with pytest.raises(FingerprintAlreadyWrittenError):
        write_dataset_fingerprint(roots, run_dir, ids=["ra_007"])


def test_assert_dataset_fingerprint_matches_passes_after_write(tmp_path):
    roots = _make_fake_roots(tmp_path)
    run_dir = tmp_path / "run"
    write_dataset_fingerprint(roots, run_dir, ids=["ra_007"])
    assert_dataset_fingerprint_matches(roots, run_dir, ids=["ra_007"])


def test_assert_dataset_fingerprint_matches_drifts_when_obj_changes(tmp_path):
    roots = _make_fake_roots(tmp_path)
    run_dir = tmp_path / "run"
    write_dataset_fingerprint(roots, run_dir, ids=["ra_007"])
    obj = roots.converter_output_root / "raw/partseg/ra_007/objs/part_00.obj"
    obj.write_text("v 1 1 1\n")
    with pytest.raises(DatasetFingerprintDriftError) as exc:
        assert_dataset_fingerprint_matches(roots, run_dir, ids=["ra_007"])
    assert "OBJ hash drift" in str(exc.value)


def test_assert_dataset_fingerprint_matches_drifts_on_root_swap(tmp_path):
    roots = _make_fake_roots(tmp_path)
    run_dir = tmp_path / "run"
    write_dataset_fingerprint(roots, run_dir, ids=["ra_007"])
    other = _make_fake_roots(tmp_path / "swap")
    with pytest.raises(DatasetFingerprintDriftError) as exc:
        assert_dataset_fingerprint_matches(other, run_dir, ids=["ra_007"])
    assert "converter_output_root drift" in str(exc.value)


def test_verify_cache_passes_on_fresh_artifacts(tmp_path, monkeypatch):
    _mock_dependencies(monkeypatch)
    roots = _make_fake_roots(tmp_path)
    _write_valid_cache(roots)
    run_dir = tmp_path / "run"
    write_dataset_fingerprint(roots, run_dir, ids=["ra_007"])
    verify_cache(
        roots,
        run_dir,
        ids=["ra_007"],
        expected_coacd_run_params=dict(V1_COACD_RUN_PARAMS),
        expected_vhacd_cache_metadata=dict(V1_VHACD_CACHE_METADATA),
    )


def test_verify_cache_raises_on_stale_obj_hash(tmp_path, monkeypatch):
    _mock_dependencies(monkeypatch)
    roots = _make_fake_roots(tmp_path)
    _write_valid_cache(roots)
    run_dir = tmp_path / "run"
    write_dataset_fingerprint(roots, run_dir, ids=["ra_007"])
    obj = roots.converter_output_root / "raw/partseg/ra_007/objs/part_00.obj"
    obj.write_text("v 9 9 9\n")
    write_dataset_fingerprint(roots, tmp_path / "run2", ids=["ra_007"])
    with pytest.raises(VhacdParamsMismatchError):
        verify_cache(roots, tmp_path / "run2", ids=["ra_007"])


def test_verify_cache_raises_on_extra_cache_file(tmp_path, monkeypatch):
    _mock_dependencies(monkeypatch)
    roots = _make_fake_roots(tmp_path)
    _write_valid_cache(roots)
    (roots.converter_output_root / "raw/vhacd/ra_007/stale.json").write_text("{}")
    run_dir = tmp_path / "run"
    write_dataset_fingerprint(roots, run_dir, ids=["ra_007"])
    with pytest.raises(VhacdCacheMissingError):
        verify_cache(roots, run_dir, ids=["ra_007"])


def test_verify_cache_raises_when_stage_metadata_object_id_wrong(tmp_path, monkeypatch):
    _mock_dependencies(monkeypatch)
    roots = _make_fake_roots(tmp_path)
    _write_valid_cache(roots)
    meta = roots.converter_output_root / "raw/stage_metadata/ra_007.json"
    data = json.loads(meta.read_text())
    data["object_id"] = "wrong"
    meta.write_text(json.dumps(data))
    run_dir = tmp_path / "run"
    write_dataset_fingerprint(roots, run_dir, ids=["ra_007"])
    with pytest.raises(SchemaMismatchError):
        verify_cache(roots, run_dir, ids=["ra_007"])
