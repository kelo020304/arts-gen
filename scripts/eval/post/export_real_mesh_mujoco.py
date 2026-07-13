#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


DATA_ROOTS = {
    "phyx-verse": Path("/robot/data-lab/jzh/art-gen/data/phyx-verse"),
    "realappliance": Path("/robot/data-lab/jzh/art-gen/data/realappliance"),
}


def _safe_name(value: str, max_len: int = 80) -> str:
    chars: list[str] = []
    for ch in str(value):
        if ch.isascii() and (ch.isalnum() or ch in ("-", "_", ".")):
            chars.append(ch)
        elif ch.isspace():
            chars.append("_")
        else:
            chars.append(f"_u{ord(ch):04x}_")
    out = "".join(chars).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return (out or "item")[:max_len]


def _xml_name(value: str, fallback: str = "item", max_len: int = 80) -> str:
    out = _safe_name(value, max_len=max_len)
    if out and not (out[0].isalpha() or out[0] == "_"):
        out = f"{fallback}_{out}"
    return out[:max_len]


def _prefix(dataset_id: str, object_id: str, angle: int) -> str:
    return f"{dataset_id}__{_safe_name(object_id)}__angle_{int(angle):02d}"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        child = None
        for child in elem:
            _indent_xml(child, level + 1)
        if child is not None and (not child.tail or not child.tail.strip()):
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def _read_mtl(mtl_path: Path) -> tuple[list[str], list[str], tuple[float, float, float] | None]:
    maps: list[str] = []
    rewritten: list[str] = []
    kd: tuple[float, float, float] | None = None
    for raw_line in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            rewritten.append(raw_line)
            continue
        parts = stripped.split(maxsplit=1)
        key = parts[0].lower()
        if key == "kd" and len(parts) == 2:
            nums = parts[1].split()
            if len(nums) >= 3:
                try:
                    kd = (float(nums[0]), float(nums[1]), float(nums[2]))
                except ValueError:
                    pass
            rewritten.append(raw_line)
            continue
        if len(parts) == 2 and key in {"map_kd", "map_ka", "map_ks", "map_bump", "bump"}:
            maps.append(parts[1])
            rewritten.append(raw_line)
            continue
        rewritten.append(raw_line)
    return maps, rewritten, kd


def _copy_texture(src_mtl: Path, raw_map: str, assets_dir: Path) -> str:
    src = (src_mtl.parent / raw_map).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"texture referenced by {src_mtl} not found: {src}")
    from PIL import Image  # noqa: PLC0415

    parent_name = Path(raw_map).parent.as_posix()
    if parent_name in {"", "."}:
        rel_dir = "textures"
    elif parent_name == "..":
        rel_dir = "textures"
    else:
        rel_dir = parent_name.lstrip("../")
    dest_rel = Path(rel_dir) / f"{_safe_name(src.stem, 96)}.png"
    dest = assets_dir / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src)
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGBA" if "A" in image.mode else "RGB")
    image.save(dest)
    return dest_rel.as_posix()


def _copy_obj_with_assets(
    *,
    src_obj: Path,
    src_mtl: Path,
    dest_obj: Path,
    dest_mtl: Path,
    assets_dir: Path,
) -> dict[str, Any]:
    maps, mtl_lines, kd = _read_mtl(src_mtl)
    texture_rewrites: dict[str, str] = {}
    for raw_map in maps:
        texture_rewrites[raw_map] = _copy_texture(src_mtl, raw_map, assets_dir)

    rewritten_mtl: list[str] = []
    for raw_line in mtl_lines:
        stripped = raw_line.strip()
        parts = stripped.split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() in {"map_kd", "map_ka", "map_ks", "map_bump", "bump"}:
            rewritten_mtl.append(f"{parts[0]} {texture_rewrites[parts[1]]}")
        else:
            rewritten_mtl.append(raw_line)
    dest_mtl.write_text("\n".join(rewritten_mtl) + "\n", encoding="utf-8")

    rewritten_obj: list[str] = []
    for raw_line in src_obj.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw_line.strip().lower().startswith("mtllib "):
            rewritten_obj.append(f"mtllib {dest_mtl.name}")
        else:
            rewritten_obj.append(raw_line)
    dest_obj.write_text("\n".join(rewritten_obj) + "\n", encoding="utf-8")

    first_texture = next(iter(texture_rewrites.values()), None)
    return {
        "source_obj": str(src_obj),
        "source_mtl": str(src_mtl),
        "obj_file": f"assets/{dest_obj.name}",
        "mtl_file": f"assets/{dest_mtl.name}",
        "texture_files": [f"assets/{value}" for value in texture_rewrites.values()],
        "primary_texture": None if first_texture is None else f"assets/{first_texture}",
        "kd": kd,
    }


def _rgba(kd: tuple[float, float, float] | None) -> str:
    if kd is None:
        kd = (0.72, 0.76, 0.80)
    return " ".join(f"{max(0.0, min(1.0, float(v))):.6g}" for v in (*kd, 1.0))


def _part_role(part_name: str, part: dict[str, Any]) -> str:
    text = f"{part_name} {part.get('type', '')}".lower()
    if str(part.get("parent_group", "")) in {"", "None", "none"} and str(part.get("joint", "")) == "fixed":
        return "body"
    if "body" in text or "cabinet_body" in text:
        return "body"
    return "part"


def _source_mesh_items(dataset_id: str, object_id: str, assets_dir: Path) -> list[dict[str, Any]]:
    data_root = DATA_ROOTS[dataset_id]
    part_info_path = data_root / "reconstruction" / "part_info" / object_id / "part_info.json"
    if not part_info_path.is_file():
        raise FileNotFoundError(f"part_info not found: {part_info_path}")
    part_info = _load_json(part_info_path)
    obj_dir = data_root / "raw" / "partseg" / object_id / "objs"
    if not obj_dir.is_dir():
        raise FileNotFoundError(f"raw obj dir not found: {obj_dir}")

    items: list[dict[str, Any]] = []
    seen_dest: set[str] = set()
    for part_idx, (part_name, part) in enumerate((part_info.get("parts") or {}).items()):
        if not isinstance(part, dict):
            continue
        role = _part_role(str(part_name), part)
        stems = [str(item) for item in part.get("obj_files", []) if str(item)]
        for stem_idx, stem in enumerate(stems):
            src_obj = obj_dir / f"{stem}.obj"
            src_mtl = obj_dir / f"{stem}.mtl"
            if not src_obj.is_file():
                raise FileNotFoundError(f"source obj not found: {src_obj}")
            if not src_mtl.is_file():
                raise FileNotFoundError(f"source mtl not found: {src_mtl}")
            base = _safe_name(stem, 64)
            if base in seen_dest:
                base = f"{base}_{part_idx}_{stem_idx}"
            seen_dest.add(base)
            copied = _copy_obj_with_assets(
                src_obj=src_obj,
                src_mtl=src_mtl,
                dest_obj=assets_dir / f"{base}.obj",
                dest_mtl=assets_dir / f"{base}.mtl",
                assets_dir=assets_dir,
            )
            items.append(
                {
                    **copied,
                    "role": role,
                    "part_name": str(part_name),
                    "part_type": str(part.get("type", "")),
                    "source_stem": stem,
                    "part_index": int(part.get("part_index", part_idx)),
                    "joint": str(part.get("joint", "")),
                }
            )
    if not items:
        raise ValueError(f"no source mesh items for {dataset_id}::{object_id}")
    return items


def _write_xml(out_xml: Path, model_name: str, mesh_items: list[dict[str, Any]]) -> None:
    root = ET.Element("mujoco", {"model": _safe_name(model_name, 120)})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": ".", "texturedir": ".", "balanceinertia": "true"})
    ET.SubElement(root, "option", {"gravity": "0 0 -9.81"})
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", {"type": "mesh", "group": "2", "contype": "0", "conaffinity": "0"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "floor_checker",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.18 0.18 0.18",
            "rgb2": "0.62 0.62 0.62",
            "width": "256",
            "height": "256",
        },
    )
    ET.SubElement(asset, "material", {"name": "floor_checker", "texture": "floor_checker", "texrepeat": "8 8"})

    for idx, item in enumerate(mesh_items):
        mesh_name = f"mesh_{idx:02d}_{_safe_name(item['source_stem'], 40)}"
        item["mesh_name"] = mesh_name
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(item["obj_file"]), "inertia": "shell"})
        mat_name = f"mat_{idx:02d}_{_safe_name(item['source_stem'], 40)}"
        item["material_name"] = mat_name
        if item.get("primary_texture"):
            tex_name = f"tex_{idx:02d}_{_safe_name(item['source_stem'], 40)}"
            item["texture_name"] = tex_name
            ET.SubElement(asset, "texture", {"name": tex_name, "type": "2d", "file": str(item["primary_texture"])})
            ET.SubElement(asset, "material", {"name": mat_name, "texture": tex_name})
        else:
            ET.SubElement(asset, "material", {"name": mat_name, "rgba": _rgba(item.get("kd"))})

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "top", "pos": "0 0 3", "dir": "0 0 -1"})
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "5 5 0.05",
            "pos": "0 0 -0.02",
            "material": "floor_checker",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    object_body = ET.SubElement(worldbody, "body", {"name": "object", "pos": "0 0 0"})
    for idx, item in enumerate(mesh_items):
        label = _xml_name(item["part_name"], fallback="part", max_len=64) or f"part_{idx:02d}"
        parent = object_body
        if item.get("role") != "body":
            parent = ET.SubElement(object_body, "body", {"name": f"{label}_{idx:02d}", "pos": "0 0 0"})
        ET.SubElement(
            parent,
            "geom",
            {
                "name": f"{label}_{idx:02d}_visual",
                "mesh": item["mesh_name"],
                "material": item["material_name"],
            },
        )

    _indent_xml(root)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_xml.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def _sample_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "held"):
        for item in selection.get("samples", {}).get(split, []):
            rows.append(
                {
                    "split": split,
                    "dataset_id": str(item["dataset_id"]),
                    "object_id": str(item.get("object_id") or item.get("obj_id")),
                    "angle": int(item.get("angle", item.get("angle_idx", 0))),
                }
            )
    return rows


def _update_summary(summary_path: Path, xml_path: Path, assets_dir: Path, mesh_items: list[dict[str, Any]]) -> None:
    if not summary_path.is_file():
        return
    summary = _load_json(summary_path)
    summary["mujoco_xml"] = str(xml_path.resolve())
    summary["mujoco_assets_dir"] = str(assets_dir.resolve())
    summary["real_mujoco"] = {
        "enabled": True,
        "source": "dataset raw/partseg OBJ+MTL+texture assets",
        "xml": str(xml_path.resolve()),
        "assets_dir": str(assets_dir.resolve()),
        "mesh_count": len(mesh_items),
        "textured_mesh_count": sum(1 for item in mesh_items if item.get("primary_texture")),
        "source_meshes": mesh_items,
    }
    summary["mujoco_export_source"] = "dataset_raw_obj_mtl_texture"
    summary["mujoco_textured_assets"] = {
        "enabled": True,
        "source": "dataset raw/partseg OBJ+MTL+texture assets",
        "appearance_source": "dataset-raw-obj-mtl-texture",
        "mesh_count": len(mesh_items),
        "textured_mesh_count": sum(1 for item in mesh_items if item.get("primary_texture")),
    }
    _write_json(summary_path, summary)


def export_one(out_dir: Path, sample: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    dataset_id = sample["dataset_id"]
    object_id = sample["object_id"]
    angle = int(sample["angle"])
    prefix = _prefix(dataset_id, object_id, angle)
    mujoco_dir = out_dir / f"{prefix}__mujoco"
    assets_dir = mujoco_dir / "assets"
    xml_path = mujoco_dir / f"{prefix}.xml"
    if xml_path.is_file() and not overwrite:
        return {"sample": sample, "status": "skipped", "xml": str(xml_path)}
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    mesh_items = _source_mesh_items(dataset_id, object_id, assets_dir)
    _write_xml(xml_path, prefix, mesh_items)
    _write_json(mujoco_dir / "source_mesh_manifest.json", {"sample": sample, "mesh_items": mesh_items})
    _update_summary(out_dir / f"{prefix}__summary.json", xml_path, assets_dir, mesh_items)
    return {
        "sample": sample,
        "status": "done",
        "xml": str(xml_path),
        "mesh_count": len(mesh_items),
        "textured_mesh_count": sum(1 for item in mesh_items if item.get("primary_texture")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite EE MuJoCo XMLs to use real source OBJ/MTL/textures.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--angle", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    selection_json = args.selection_json or out_dir / "selection.json"
    if args.dataset_id and args.object_id:
        samples = [
            {
                "split": "single",
                "dataset_id": str(args.dataset_id),
                "object_id": str(args.object_id),
                "angle": int(args.angle),
            }
        ]
        selection_source = "cli"
    else:
        if not selection_json.is_file():
            raise FileNotFoundError(
                f"selection json not found: {selection_json}; pass --dataset-id/--object-id/--angle for a single sample"
            )
        selection = _load_json(selection_json)
        samples = _sample_rows(selection)
        selection_source = str(selection_json)
    records = [export_one(out_dir, sample, bool(args.overwrite)) for sample in samples]
    _write_json(out_dir / "real_mesh_mujoco_export.json", {"selection_source": selection_source, "records": records})
    done = sum(1 for item in records if item["status"] == "done")
    skipped = sum(1 for item in records if item["status"] == "skipped")
    print(f"[export_real_mesh_mujoco] done={done} skipped={skipped} total={len(records)} out_dir={out_dir}")
    for item in records:
        sample = item["sample"]
        print(
            f"{item['status']} {sample['dataset_id']}::{sample['object_id']} angle={sample['angle']} "
            f"xml={item['xml']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
