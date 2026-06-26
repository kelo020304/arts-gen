"""V1 Phase 0 verifier."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path

from .config import (
    V1_COACD_RUN_PARAMS,
    V1_CONDA_PYTHON,
    V1_PINNED_COACD_VERSION,
    V1_TEN_IDS,
    V1_VHACD_CACHE_METADATA,
    V1DatasetRoots,
)
from .errors import (
    CoacdParamsMissingError,
    DatasetFingerprintDriftError,
    DependencyMissingError,
    FingerprintAlreadyWrittenError,
    MissingModelError,
    SchemaMismatchError,
    VhacdCacheMissingError,
    VhacdParamsMismatchError,
)

_DEPS = [
    (
        "coacd",
        f"{V1_CONDA_PYTHON} -m pip install coacd=={V1_PINNED_COACD_VERSION}",
    ),
    ("fcl", f"{V1_CONDA_PYTHON} -m pip install python-fcl"),
    ("trimesh", "already in env-isaacsim"),
    ("matplotlib", "already in env-isaacsim"),
]


def preflight_dependencies() -> None:
    """Check runtime dependencies, pinned coacd version, and run_coacd kwargs."""
    modules = {}
    for pkg, hint in _DEPS:
        try:
            modules[pkg] = importlib.import_module(pkg)
        except ImportError as exc:
            raise DependencyMissingError(
                f"{pkg} not importable in current Python: {exc}. Install via: {hint}"
            ) from exc

    try:
        installed = importlib.metadata.version("coacd")
    except importlib.metadata.PackageNotFoundError as exc:
        raise DependencyMissingError(
            f"coacd installed but PyPI metadata missing: {exc}. Reinstall via: "
            f"pip install coacd=={V1_PINNED_COACD_VERSION}"
        ) from exc
    if installed != V1_PINNED_COACD_VERSION:
        raise DependencyMissingError(
            f"coacd version drift: got {installed!r}, "
            f"V1 pinned {V1_PINNED_COACD_VERSION!r}. Install via: "
            f"pip install coacd=={V1_PINNED_COACD_VERSION}"
        )

    sig = inspect.signature(modules["coacd"].run_coacd)
    missing = [k for k in V1_COACD_RUN_PARAMS if k not in sig.parameters]
    if missing:
        raise CoacdParamsMissingError(
            f"coacd=={V1_PINNED_COACD_VERSION}.run_coacd missing kwargs {missing}; "
            "V1 spec assumes these are present."
        )


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise MissingModelError(f"required file missing: {path}")


def _require_dir(path: Path) -> None:
    if not path.is_dir():
        raise MissingModelError(f"required dir missing: {path}")


def preflight_inputs(roots: V1DatasetRoots, ids: list[str] = V1_TEN_IDS) -> None:
    """Check dependencies and required converter/source input paths."""
    preflight_dependencies()
    for object_id in ids:
        _require_file(roots.converter_output_root / f"raw/finaljson/{object_id}.json")
        _require_dir(roots.converter_output_root / f"raw/partseg/{object_id}/objs")
        _require_file(roots.aligned_usd_for(object_id))


def _compute_fingerprint(roots: V1DatasetRoots, ids: list[str]) -> dict:
    obj_hashes: dict[str, str] = {}
    for object_id in ids:
        obj_dir = roots.converter_output_root / f"raw/partseg/{object_id}/objs"
        for obj in sorted(obj_dir.glob("*.obj")):
            rel = f"{object_id}/{obj.name}"
            obj_hashes[rel] = hashlib.sha256(obj.read_bytes()).hexdigest()
    convert_report = roots.converter_output_root / "raw/convert_report.json"
    return {
        "converter_output_root": str(roots.converter_output_root.resolve()),
        "convert_report_sha256": (
            hashlib.sha256(convert_report.read_bytes()).hexdigest()
            if convert_report.is_file()
            else None
        ),
        "objs": obj_hashes,
    }


def write_dataset_fingerprint(
    roots: V1DatasetRoots,
    run_output_dir: Path,
    ids: list[str] = V1_TEN_IDS,
) -> Path:
    """Write dataset_fingerprint.json once; refuse to overwrite it."""
    run_output_dir.mkdir(parents=True, exist_ok=True)
    out = run_output_dir / "dataset_fingerprint.json"
    if out.is_file():
        raise FingerprintAlreadyWrittenError(
            f"{out} already exists; refusing to overwrite. Use a fresh run_output_dir."
        )
    out.write_text(json.dumps(_compute_fingerprint(roots, ids), indent=2))
    return out


def assert_dataset_fingerprint_matches(
    roots: V1DatasetRoots,
    run_output_dir: Path,
    ids: list[str] = V1_TEN_IDS,
) -> None:
    """Compare current dataset hashes against the run's baseline fingerprint."""
    fp_file = run_output_dir / "dataset_fingerprint.json"
    _require_file(fp_file)
    expected = json.loads(fp_file.read_text())
    current = _compute_fingerprint(roots, ids)

    if expected["converter_output_root"] != current["converter_output_root"]:
        raise DatasetFingerprintDriftError(
            "converter_output_root drift: expected "
            f"{expected['converter_output_root']!r}, got {current['converter_output_root']!r}"
        )
    if expected["convert_report_sha256"] != current["convert_report_sha256"]:
        raise DatasetFingerprintDriftError(
            "convert_report.json sha256 drift; converter re-ran since 0a"
        )

    diffs = {
        key: (expected["objs"].get(key), current["objs"].get(key))
        for key in set(expected["objs"]) | set(current["objs"])
        if expected["objs"].get(key) != current["objs"].get(key)
    }
    if diffs:
        raise DatasetFingerprintDriftError(
            "OBJ hash drift since 0a (showing first 5): "
            f"{dict(list(diffs.items())[:5])}"
        )


def verify_cache(
    roots: V1DatasetRoots,
    run_output_dir: Path,
    ids: list[str] = V1_TEN_IDS,
    expected_coacd_run_params: dict | None = None,
    expected_vhacd_cache_metadata: dict | None = None,
) -> None:
    """Strict Phase 0d cache gate for stage metadata and VHACD cache files."""
    preflight_inputs(roots, ids)
    assert_dataset_fingerprint_matches(roots, run_output_dir, ids)

    from .data_prep import list_obj_groups

    for object_id in ids:
        meta_file = roots.converter_output_root / f"raw/stage_metadata/{object_id}.json"
        _require_file(meta_file)
        meta = json.loads(meta_file.read_text())
        required_meta = {
            "object_id",
            "source_id",
            "meters_per_unit",
            "joint_prim_paths",
            "stage_up_axis",
        }
        missing_meta = required_meta - set(meta)
        if missing_meta:
            raise SchemaMismatchError(
                f"{meta_file}: stage_metadata missing required keys {sorted(missing_meta)}"
            )
        if meta["object_id"] != object_id:
            raise SchemaMismatchError(
                f"{meta_file}: object_id mismatch {meta['object_id']!r} vs {object_id!r}"
            )
        expected_source = object_id.removeprefix("ra_")
        if meta["source_id"] != expected_source:
            raise SchemaMismatchError(
                f"{meta_file}: source_id mismatch {meta['source_id']!r} vs {expected_source!r}"
            )

        oracle_file = roots.converter_output_root / f"raw/vlm_oracle/{object_id}.json"
        _require_file(oracle_file)
        oracle_joints = set(json.loads(oracle_file.read_text()).get("joints", {}))
        meta_joints = set(meta["joint_prim_paths"])
        if oracle_joints != meta_joints:
            raise SchemaMismatchError(
                f"{meta_file}: joint_prim_paths coverage mismatch; "
                f"missing {oracle_joints - meta_joints}, extra {meta_joints - oracle_joints}"
            )

        obj_dir = roots.converter_output_root / f"raw/partseg/{object_id}/objs"
        vhacd_dir = roots.converter_output_root / f"raw/vhacd/{object_id}"
        if not vhacd_dir.is_dir():
            raise VhacdCacheMissingError(f"{object_id}: vhacd cache dir missing: {vhacd_dir}")

        expected_stems = list_obj_groups(obj_dir)
        cached_stems = {p.stem for p in vhacd_dir.glob("*.json")}
        if expected_stems != cached_stems:
            raise VhacdCacheMissingError(
                f"{vhacd_dir}: cache file set mismatch; "
                f"missing {sorted(expected_stems - cached_stems)}, "
                f"extra {sorted(cached_stems - expected_stems)}"
            )

        for obj in sorted(obj_dir.glob("*.obj")):
            if obj.stem not in expected_stems:
                continue
            cache_file = vhacd_dir / f"{obj.stem}.json"
            _require_file(cache_file)
            cached = json.loads(cache_file.read_text())
            required_cache = {
                "object_id",
                "part_name",
                "source_obj",
                "source_sha256",
                "vhacd_cache_metadata",
                "coacd_run_params",
                "frame",
                "hulls",
                "n_hulls",
            }
            missing_cache = required_cache - set(cached)
            if missing_cache:
                raise SchemaMismatchError(
                    f"{cache_file}: missing required keys {sorted(missing_cache)}"
                )
            if cached["object_id"] != object_id:
                raise SchemaMismatchError(
                    f"{cache_file}: object_id mismatch {cached['object_id']!r} vs {object_id!r}"
                )
            if cached["part_name"] != obj.stem:
                raise SchemaMismatchError(
                    f"{cache_file}: part_name mismatch {cached['part_name']!r} vs {obj.stem!r}"
                )
            if cached["frame"] != "world_baked":
                raise SchemaMismatchError(
                    f"{cache_file}: frame must be 'world_baked', got {cached['frame']!r}"
                )
            if not isinstance(cached["hulls"], list) or cached["n_hulls"] != len(cached["hulls"]):
                raise SchemaMismatchError(f"{cache_file}: n_hulls does not match hulls length")
            for i, hull in enumerate(cached["hulls"]):
                if not {"hull_index", "vertices", "faces"} <= set(hull):
                    raise SchemaMismatchError(
                        f"{cache_file}: hull[{i}] missing vertices/faces/hull_index"
                    )

            current_sha = hashlib.sha256(obj.read_bytes()).hexdigest()
            if cached["source_sha256"] != current_sha:
                raise VhacdParamsMismatchError(
                    f"{cache_file}: source_sha256 drift; re-run --stage vhacd"
                )
            if (
                expected_coacd_run_params is not None
                and cached["coacd_run_params"] != expected_coacd_run_params
            ):
                raise VhacdParamsMismatchError(
                    f"{cache_file}: cached coacd_run_params drift; re-run --stage vhacd"
                )
            if (
                expected_vhacd_cache_metadata is not None
                and cached["vhacd_cache_metadata"] != expected_vhacd_cache_metadata
            ):
                raise VhacdParamsMismatchError(
                    f"{cache_file}: vhacd_cache_metadata drift; re-run --stage vhacd"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 KinematicSolver Phase 0 verifier")
    parser.add_argument(
        "--stage",
        choices=["preflight", "fingerprint", "verify_cache"],
        required=True,
    )
    parser.add_argument("--converter-output-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--run-output-dir", type=Path)
    parser.add_argument("--object-ids", default=",".join(V1_TEN_IDS))
    args = parser.parse_args()

    roots = V1DatasetRoots(
        converter_output_root=args.converter_output_root or V1DatasetRoots().converter_output_root,
        source_root=args.source_root or V1DatasetRoots().source_root,
    )
    ids = [s.strip() for s in args.object_ids.split(",") if s.strip()]

    if args.stage == "preflight":
        preflight_inputs(roots, ids)
        print("[OK] phase 0a preflight passed.")
    elif args.stage == "fingerprint":
        if args.run_output_dir is None:
            parser.error("--run-output-dir required for --stage fingerprint")
        out = write_dataset_fingerprint(roots, args.run_output_dir, ids)
        print(f"[OK] dataset fingerprint written to {out}.")
    elif args.stage == "verify_cache":
        if args.run_output_dir is None:
            parser.error("--run-output-dir required for --stage verify_cache")
        verify_cache(
            roots,
            args.run_output_dir,
            ids,
            expected_coacd_run_params=dict(V1_COACD_RUN_PARAMS),
            expected_vhacd_cache_metadata=dict(V1_VHACD_CACHE_METADATA),
        )
        print("[OK] phase 0d verify_cache passed.")


if __name__ == "__main__":
    main()
