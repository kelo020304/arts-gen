#!/usr/bin/env python3
"""Create the fixed official object split for promptable part segmentation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.part_promptable_seg.part_promptable_seg_utils import (  # noqa: E402
    OFFICIAL_SPLIT_PATH,
    enumerate_part_rows_multi,
    format_table,
    make_base_datasets,
    object_key,
    parse_dataset_specs_from_env,
    DatasetSpec,
)


DEFAULT_GATE2_META = Path("/mnt/robot-data-lab/jzh/art-gen-output/debug/part_promptable_seg_gate2_S_dim256_depth6/run_metadata.json")
DEFAULT_LEGACY_HELDOUT_SPLITS = (
    Path("/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg/manifests/split_official_v1.json"),
    Path("/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg/manifests/split_official_multi_v1.json"),
)
DEFAULT_0511_ROOT = Path(
    "/mnt/robot-data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511/"
    "PhysX-Mobility-full-4view-0511"
)
DEFAULT_0511_MANIFEST = "manifests/part_completion/arts_mllm_physx-mobility.train.jsonl"
DRAWER_TERMS = ("drawer", "抽屉")
DOOR_TERMS = ("door", "门")
BUTTON_TERMS = ("button", "按键", "按钮", "键")


def obj_features(rows) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[object_key(row)].append(row)
    feats = {}
    for obj_key, obj_rows in grouped.items():
        part_names = [row.part_name.lower() for row in obj_rows]
        part_texts = [
            " ".join(
                str(getattr(row, name, "") or "")
                for name in ("part_name", "semantic_type", "part_item_name", "sample_part_names")
            ).lower()
            for row in obj_rows
        ]
        raw_counts = [int(row.raw_count) for row in obj_rows]
        drawer_count = sum(1 for text in part_texts if any(term in text for term in DRAWER_TERMS))
        strict_door_count = sum(
            1
            for text in part_texts
            if any(term in text for term in DOOR_TERMS) and not any(term in text for term in BUTTON_TERMS)
        )
        feats[obj_key] = {
            "dataset_id": obj_rows[0].dataset_id,
            "obj_id": obj_rows[0].obj_id,
            "rows": len(obj_rows),
            "angles": len({int(row.angle_idx) for row in obj_rows}),
            "parts": len({row.part_name for row in obj_rows}),
            "multi_button": sum(1 for name in part_names if "button" in name) >= 2,
            "door_lid": any(("door" in name) or ("lid" in name) for name in part_names),
            "drawer_or_strict_door": drawer_count > 0 or strict_door_count > 0,
            "drawer_count": drawer_count,
            "strict_door_count": strict_door_count,
            "tiny": any(count < 50 for count in raw_counts),
            "small": any(50 <= count < 500 for count in raw_counts),
            "medium": any(500 <= count < 3000 for count in raw_counts),
            "large": any(count >= 3000 for count in raw_counts),
            "single_part": len({row.part_name for row in obj_rows}) == 1,
            "button": any("button" in name for name in part_names),
            "min_raw": min(raw_counts),
            "max_raw": max(raw_counts),
        }
    return feats


def coverage(ids: list[str], feats: dict[str, dict[str, Any]]) -> dict[str, int]:
    keys = ("multi_button", "door_lid", "drawer_or_strict_door", "tiny", "small", "medium", "large", "single_part", "button")
    return {key: int(sum(1 for obj_id in ids if feats[obj_id][key])) for key in keys}


def _spec_manifest_abs(spec: DatasetSpec, manifest: str | Path) -> Path:
    path = Path(manifest)
    return path if path.is_absolute() else spec.data_root / path


def _filter_spec_manifests_for_obj_ids(spec: DatasetSpec, obj_ids: set[str]) -> DatasetSpec:
    if not obj_ids:
        return spec
    kept: list[str] = []
    for manifest in spec.manifest_paths:
        path = _spec_manifest_abs(spec, manifest)
        found = False
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if str(rec.get("object_id")) in obj_ids:
                    found = True
                    break
        if found:
            kept.append(str(manifest))
    if not kept:
        raise RuntimeError(f"no manifests in {spec.dataset_id} contain requested object ids")
    return replace(spec, manifest_paths=tuple(kept))


def select_extra_ids(
    *,
    all_ids: list[str],
    heldout: list[str],
    feats: dict[str, dict[str, Any]],
    target_count: int,
    seed: int,
) -> list[str]:
    rng = random.Random(int(seed))
    selected = list(dict.fromkeys(heldout))
    required_predicates = [
        ("multi_button", lambda f: bool(f["multi_button"])),
        ("door_lid", lambda f: bool(f["door_lid"])),
        ("tiny", lambda f: bool(f["tiny"])),
        ("single_part", lambda f: bool(f["single_part"])),
        ("large", lambda f: bool(f["large"])),
    ]
    for _name, pred in required_predicates:
        if any(pred(feats[obj_id]) for obj_id in selected):
            continue
        candidates = [obj_id for obj_id in all_ids if obj_id not in selected and pred(feats[obj_id])]
        if not candidates:
            raise RuntimeError(f"cannot satisfy heldout coverage predicate {_name}")
        candidates.sort(key=lambda oid: (feats[oid]["min_raw"], oid))
        selected.append(candidates[0])
    remaining = [obj_id for obj_id in all_ids if obj_id not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, int(target_count) - len(selected))])
    if len(selected) != int(target_count):
        raise RuntimeError(f"selected heldout count {len(selected)} != target {target_count}")
    return selected


def pick_proxy_ids(ids: list[str], feats: dict[str, dict[str, Any]], *, count: int, seed: int) -> list[str]:
    rng = random.Random(int(seed))
    selected: list[str] = []
    for pred in (
        lambda f: bool(f["multi_button"]),
        lambda f: bool(f["door_lid"]),
        lambda f: bool(f["tiny"]),
        lambda f: bool(f["single_part"]),
    ):
        candidates = [obj_id for obj_id in ids if obj_id not in selected and pred(feats[obj_id])]
        if candidates:
            candidates.sort(key=lambda oid: (feats[oid]["min_raw"], oid))
            selected.append(candidates[0])
    remaining = [obj_id for obj_id in ids if obj_id not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, int(count) - len(selected))])
    return selected[: int(count)]


def ensure_proxy_contains(seed_ids: list[str], required_ids: list[str], *, count: int) -> list[str]:
    selected = []
    for obj_id in required_ids:
        if obj_id not in selected:
            selected.append(obj_id)
    for obj_id in seed_ids:
        if obj_id not in selected:
            selected.append(obj_id)
    return selected[: int(count)]


def pick_realappliance_heldout_ids(
    ids: list[str],
    feats: dict[str, dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[str]:
    ra_ids = [obj_id for obj_id in ids if bool(feats[obj_id].get("realappliance"))]
    rng = random.Random(int(seed) + 301)
    shuffled = sorted(ra_ids)
    rng.shuffle(shuffled)
    return sorted(shuffled[: min(int(count), len(shuffled))])


def resolve_legacy_heldout_keys(
    paths: list[Path],
    feats: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    by_obj_id: dict[str, list[str]] = defaultdict(list)
    for key, feat in feats.items():
        by_obj_id[str(feat.get("obj_id", ""))].append(key)
    selected: list[str] = []
    used_paths: list[str] = []

    def add_key(key: str) -> None:
        if key in feats and key not in selected:
            selected.append(key)

    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        refs = data.get("heldout_keys", data.get("heldout_ids", []))
        used_paths.append(str(path))
        for ref in refs:
            if isinstance(ref, dict):
                if "object_key" in ref:
                    add_key(str(ref["object_key"]))
                elif "dataset_id" in ref and "obj_id" in ref:
                    add_key(f"{ref['dataset_id']}::{ref['obj_id']}")
                elif "obj_id" in ref:
                    for key in by_obj_id.get(str(ref["obj_id"]), []):
                        add_key(key)
                continue
            text = str(ref)
            if "::" in text:
                add_key(text)
            else:
                for key in by_obj_id.get(text, []):
                    add_key(key)
    return selected, used_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OFFICIAL_SPLIT_PATH)
    parser.add_argument("--proxy-out", type=Path, default=None)
    parser.add_argument("--heldout-fraction", type=float, default=0.10)
    parser.add_argument("--proxy-objects", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--gate2-metadata", type=Path, default=DEFAULT_GATE2_META)
    parser.add_argument("--realappliance-heldout-objects", type=int, default=20)
    parser.add_argument("--legacy-heldout-split", type=Path, action="append", default=list(DEFAULT_LEGACY_HELDOUT_SPLITS))
    parser.add_argument("--no-legacy-heldout", action="store_true")
    parser.add_argument("--include-0511-drawer-door", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--physx0511-root", type=Path, default=DEFAULT_0511_ROOT)
    parser.add_argument("--physx0511-manifest", default=DEFAULT_0511_MANIFEST)
    parser.add_argument("--physx0511-dataset-id", default="physx-0511-drawer-door")
    parser.add_argument("--physx0511-heldout-fraction", type=float, default=None)
    args = parser.parse_args()

    specs = parse_dataset_specs_from_env()
    physx0511_candidate_keys: list[str] = []
    physx0511_train_keys: list[str] = []
    physx0511_heldout_keys: list[str] = []
    physx0511_stats: dict[str, Any] = {}
    if bool(args.include_0511_drawer_door):
        raw_spec = DatasetSpec(
            dataset_id=str(args.physx0511_dataset_id),
            data_root=Path(args.physx0511_root),
            manifest_paths=(str(args.physx0511_manifest),),
        )
        raw_base = make_base_datasets([raw_spec])
        raw_rows = enumerate_part_rows_multi(raw_base)
        raw_feats = obj_features(raw_rows)
        physx0511_candidate_keys = sorted(
            obj_id for obj_id, feat in raw_feats.items() if bool(feat.get("drawer_or_strict_door"))
        )
        if not physx0511_candidate_keys:
            raise RuntimeError("0511 drawer/strict-door subset is empty")
        held_frac = float(args.physx0511_heldout_fraction) if args.physx0511_heldout_fraction is not None else float(args.heldout_fraction)
        held_count = max(1, int(round(len(physx0511_candidate_keys) * held_frac)))
        rng = random.Random(int(args.seed) + 511)
        shuffled = list(physx0511_candidate_keys)
        rng.shuffle(shuffled)
        physx0511_heldout_keys = sorted(shuffled[:held_count])
        held_set = set(physx0511_heldout_keys)
        physx0511_train_keys = [obj_id for obj_id in physx0511_candidate_keys if obj_id not in held_set]
        wanted_obj_ids = {key.split("::", 1)[1] if "::" in key else key for key in physx0511_candidate_keys}
        specs.append(_filter_spec_manifests_for_obj_ids(raw_spec, wanted_obj_ids))
        physx0511_stats = {
            "dataset_id": str(args.physx0511_dataset_id),
            "data_root": str(args.physx0511_root),
            "manifest_path": str(args.physx0511_manifest),
            "candidate_object_count": len(physx0511_candidate_keys),
            "train_object_count": len(physx0511_train_keys),
            "heldout_object_count": len(physx0511_heldout_keys),
            "heldout_fraction": float(held_frac),
            "objects_ge1_drawer": int(sum(1 for key in physx0511_candidate_keys if raw_feats[key]["drawer_count"] >= 1)),
            "objects_ge1_strict_door": int(sum(1 for key in physx0511_candidate_keys if raw_feats[key]["strict_door_count"] >= 1)),
            "objects_ge3_drawer": int(sum(1 for key in physx0511_candidate_keys if raw_feats[key]["drawer_count"] >= 3)),
            "objects_ge2_strict_door": int(sum(1 for key in physx0511_candidate_keys if raw_feats[key]["strict_door_count"] >= 2)),
        }
    bases = make_base_datasets(specs)
    rows = enumerate_part_rows_multi(bases)
    if physx0511_candidate_keys:
        physx0511_key_set = set(physx0511_candidate_keys)
        rows = [
            row
            for row in rows
            if row.dataset_id != str(args.physx0511_dataset_id) or object_key(row) in physx0511_key_set
        ]
    feats = obj_features(rows)
    all_ids = sorted(feats)
    if not all_ids:
        raise RuntimeError("no object ids found")

    gate2_ids: list[str] = []
    if args.gate2_metadata.is_file():
        meta = json.loads(args.gate2_metadata.read_text(encoding="utf-8"))
        gate2_ids = [str(x) for x in meta.get("split", {}).get("heldout_obj_ids", [])]
    if not gate2_ids:
        print(f"[split] Gate2 heldout ids not found in {args.gate2_metadata}; using seeded heldout only")
    missing_gate2 = sorted(
        gate2_id
        for gate2_id in gate2_ids
        if gate2_id not in set(all_ids) and not any(feat["obj_id"] == gate2_id for feat in feats.values())
    )
    if missing_gate2:
        print(f"[split] Gate2 heldout ids missing from configured datasets; ignoring first={missing_gate2[:10]}")
    gate2_keys = [
        key
        for key, feat in feats.items()
        if key in set(gate2_ids) or feat["obj_id"] in set(gate2_ids)
    ]
    for obj_key, feat in feats.items():
        feat["realappliance"] = (
            str(feat.get("dataset_id", "")).lower() == "realappliance"
            or "realappliance" in str(feat.get("dataset_id", "")).lower()
        )
    realappliance_keys = sorted(obj_id for obj_id, feat in feats.items() if bool(feat.get("realappliance")))
    realappliance_heldout_keys = pick_realappliance_heldout_ids(
        all_ids,
        feats,
        count=int(args.realappliance_heldout_objects),
        seed=int(args.seed),
    )
    if realappliance_keys and len(realappliance_heldout_keys) != min(int(args.realappliance_heldout_objects), len(realappliance_keys)):
        raise RuntimeError("failed to select requested RealAppliance heldout object count")
    legacy_heldout_keys, legacy_heldout_paths = resolve_legacy_heldout_keys(
        [] if bool(args.no_legacy_heldout) else list(args.legacy_heldout_split),
        feats,
    )
    required_heldout = list(dict.fromkeys([*gate2_keys, *legacy_heldout_keys, *realappliance_heldout_keys]))

    required_heldout = list(dict.fromkeys([*required_heldout, *physx0511_heldout_keys]))

    non_0511_ids = [obj_id for obj_id in all_ids if obj_id not in set(physx0511_candidate_keys)]
    target_heldout = max(
        len(required_heldout),
        int(round(len(non_0511_ids) * float(args.heldout_fraction))) + len(physx0511_heldout_keys),
    )
    extra_candidate_ids = [
        obj_id
        for obj_id in all_ids
        if obj_id in required_heldout
        or obj_id in physx0511_candidate_keys
        or not bool(feats[obj_id].get("realappliance"))
    ]
    heldout_ids = select_extra_ids(
        all_ids=extra_candidate_ids,
        heldout=required_heldout,
        feats=feats,
        target_count=target_heldout,
        seed=int(args.seed),
    )
    heldout_set = set(heldout_ids)
    train_ids = [obj_id for obj_id in all_ids if obj_id not in heldout_set]
    if set(train_ids) & heldout_set:
        raise RuntimeError("official split overlap detected")
    if not set(gate2_keys).issubset(heldout_set):
        raise RuntimeError("official split does not contain all Gate2 heldout ids")
    if not set(legacy_heldout_keys).issubset(heldout_set):
        raise RuntimeError("official split does not contain all legacy official heldout keys")
    if not set(realappliance_heldout_keys).issubset(heldout_set):
        raise RuntimeError("official split does not contain all RealAppliance heldout keys")
    if physx0511_heldout_keys and not set(physx0511_heldout_keys).issubset(heldout_set):
        raise RuntimeError("official split does not contain all 0511 drawer/door heldout keys")
    realappliance_train_keys = sorted(obj_id for obj_id in realappliance_keys if obj_id not in heldout_set)
    physx0511_train_keys = sorted(obj_id for obj_id in physx0511_candidate_keys if obj_id not in heldout_set)

    payload = {
        "version": "split_official_v4_multi_0511_drawer_door" if bool(args.include_0511_drawer_door) else ("split_official_v3_multi" if len(specs) > 1 else "split_official_v1"),
        "seed": int(args.seed),
        "datasets": [
            {
                "dataset_id": spec.dataset_id,
                "data_root": str(spec.data_root),
                "manifest_paths": [str(path) for path in spec.manifest_paths],
            }
            for spec in specs
        ],
        "data_root": str(specs[0].data_root),
        "manifest_path": str(specs[0].manifest_paths[0]),
        "object_count": len(all_ids),
        "train_keys": train_ids,
        "heldout_keys": heldout_ids,
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "gate2_heldout_keys": gate2_keys,
        "gate2_heldout_ids": gate2_keys,
        "legacy_heldout_split_paths": legacy_heldout_paths,
        "legacy_heldout_keys": legacy_heldout_keys,
        "legacy_heldout_object_count": len(legacy_heldout_keys),
        "realappliance_heldout_keys": realappliance_heldout_keys,
        "realappliance_train_keys": realappliance_train_keys,
        "realappliance_object_count": len(realappliance_keys),
        "realappliance_heldout_object_count": len(realappliance_heldout_keys),
        "physx0511_drawer_door": physx0511_stats,
        "physx0511_drawer_door_keys": physx0511_candidate_keys,
        "physx0511_drawer_door_train_keys": physx0511_train_keys,
        "physx0511_drawer_door_heldout_keys": physx0511_heldout_keys,
        "coverage": {
            "train": coverage(train_ids, feats),
            "heldout": coverage(heldout_ids, feats),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    proxy_out = args.proxy_out or args.out.with_name("split_official_v1_proxy32.json")
    proxy_train_ids = pick_proxy_ids(train_ids, feats, count=int(args.proxy_objects), seed=int(args.seed) + 1)
    proxy_heldout_ids = pick_proxy_ids(heldout_ids, feats, count=int(args.proxy_objects), seed=int(args.seed) + 2)
    proxy_heldout_ids = ensure_proxy_contains(
        proxy_heldout_ids,
        realappliance_heldout_keys[: min(4, len(realappliance_heldout_keys))],
        count=int(args.proxy_objects),
    )
    proxy = {
        "version": "split_official_v1_proxy32",
        "split_json": str(args.out),
        "train": [
            {"object_key": obj_id, "dataset_id": feats[obj_id]["dataset_id"], "obj_id": feats[obj_id]["obj_id"]}
            for obj_id in proxy_train_ids
        ],
        "heldout": [
            {"object_key": obj_id, "dataset_id": feats[obj_id]["dataset_id"], "obj_id": feats[obj_id]["obj_id"]}
            for obj_id in proxy_heldout_ids
        ],
        "realappliance_heldout_keys": realappliance_heldout_keys,
        "physx0511_drawer_door_heldout_keys": physx0511_heldout_keys,
    }
    proxy_out.parent.mkdir(parents=True, exist_ok=True)
    proxy_out.write_text(json.dumps(proxy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    table = format_table(
        [
            {"split": "train", "objects": len(train_ids), **payload["coverage"]["train"]},
            {"split": "heldout", "objects": len(heldout_ids), **payload["coverage"]["heldout"]},
        ],
        ["split", "objects", "multi_button", "door_lid", "drawer_or_strict_door", "tiny", "small", "medium", "large", "single_part", "button"],
    )
    print(args.out)
    print(proxy_out)
    print(table)
    if legacy_heldout_keys:
        print(f"Legacy official heldout preserved: {len(legacy_heldout_keys)} objects from {legacy_heldout_paths}")
    if realappliance_heldout_keys:
        print("RealAppliance heldout:", ", ".join(realappliance_heldout_keys))
    if physx0511_stats:
        print(
            "0511 drawer/door subset: "
            f"objects={physx0511_stats['candidate_object_count']} "
            f"train={physx0511_stats['train_object_count']} "
            f"heldout={physx0511_stats['heldout_object_count']} "
            f">=3drawer={physx0511_stats['objects_ge3_drawer']} "
            f">=2strictdoor={physx0511_stats['objects_ge2_strict_door']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
