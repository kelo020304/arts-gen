from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
HSSD_SINGLE_Y_JOINT_Z_POLICY = "hssd_single_y_joint_z"
VALID_OBJ_UP_AXES = {"Y", "Z"}
VALID_OBJ_UP_AXIS_POLICIES = {HSSD_SINGLE_Y_JOINT_Z_POLICY}
OBJECT_FILTER_MODE_ALL = "all"
OBJECT_FILTER_MODE_ARTICULATED_ONLY = "articulated_only"
OBJECT_FILTER_MODE_MULTI_PART_ONLY = "multi_part_only"
VALID_OBJECT_FILTER_MODES = {
    OBJECT_FILTER_MODE_ALL,
    OBJECT_FILTER_MODE_ARTICULATED_ONLY,
    OBJECT_FILTER_MODE_MULTI_PART_ONLY,
}
VALID_FINALJSON_JOINT_TYPES = {"A", "B", "C", "CB", "D", "E"}


@dataclass
class JointTransformConfig:
    num_angles: int
    articulated_objects: str | list[str]
    static_objects: list[str]


@dataclass
class RenderConfig:
    resolution: int
    blender: str
    obj_up_axis: str = "Y"


@dataclass
class VoxelConfig:
    resolution: int


@dataclass
class GeometryConfig:
    obj_up_axis_policy: str | None = None


@dataclass
class FeatureConfig:
    model: str
    dinov2_repo: str
    torch_hub_dir: str


@dataclass
class TrellisConfig:
    root: str
    ss_encoder: str
    ss_decoder: str
    slat_encoder: str


@dataclass
class VLMConfig:
    image_prefix: str


@dataclass
class ObjectFilterConfig:
    mode: str = OBJECT_FILTER_MODE_ALL


@dataclass
class PipelineConfig:
    dataset_name: str
    data_root: str
    joint_transform: JointTransformConfig
    render: RenderConfig
    voxel: VoxelConfig
    geometry: GeometryConfig
    feature: FeatureConfig
    trellis: TrellisConfig
    vlm: VLMConfig
    object_filter: ObjectFilterConfig

    @property
    def _data_root_path(self) -> Path:
        return Path(self.data_root)

    @property
    def raw_dir(self) -> str:
        return str(self._data_root_path / "raw")

    @property
    def joint_transforms_dir(self) -> str:
        return str(self._data_root_path / "joint_transforms")

    @property
    def part_info_dir(self) -> str:
        return str(self._data_root_path / "part_info")

    @property
    def renders_dir(self) -> str:
        return str(self._data_root_path / "renders")

    @property
    def reconstruction_dir(self) -> str:
        return str(self._data_root_path / "reconstruction")

    @property
    def vlm_dir(self) -> str:
        return str(self._data_root_path / "vlm")

    @property
    def preview_dir(self) -> str:
        return str(self._data_root_path / "preview")

    @property
    def finaljson_dir(self) -> str:
        return str(Path(self.raw_dir) / "finaljson")

    @property
    def partseg_dir(self) -> str:
        return str(Path(self.raw_dir) / "partseg")

    def list_object_ids(self) -> list[str]:
        finaljson_path = Path(self.finaljson_dir)
        if not finaljson_path.is_dir():
            raise FileNotFoundError(f"finaljson_dir does not exist: {finaljson_path}")

        finaljson_files = sorted(
            path for path in finaljson_path.glob("*.json") if path.is_file()
        )
        if self.object_filter.mode == OBJECT_FILTER_MODE_ALL:
            return [path.stem for path in finaljson_files]
        object_ids: list[str] = []
        for path in finaljson_files:
            with path.open("r", encoding="utf-8") as handle:
                finaljson_data = _require_mapping(json.load(handle), f"finaljson[{path}]")
            if self.object_filter.mode == OBJECT_FILTER_MODE_ARTICULATED_ONLY:
                keep_object = _finaljson_has_bc_joint(finaljson_data, path)
            elif self.object_filter.mode == OBJECT_FILTER_MODE_MULTI_PART_ONLY:
                keep_object = _finaljson_has_multiple_parts(finaljson_data, path)
            else:
                raise ValueError(f"Unsupported object_filter.mode: {self.object_filter.mode!r}")
            if keep_object:
                object_ids.append(path.stem)
        return object_ids

    def is_articulated(self, object_id: str) -> bool:
        object_id = str(object_id)
        if object_id in self.joint_transform.static_objects:
            return False
        articulated = self.joint_transform.articulated_objects
        if articulated == "all":
            return True
        return object_id in articulated

    def get_num_angles(self, object_id: str) -> int:
        return self.joint_transform.num_angles if self.is_articulated(object_id) else 1


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return value


def _require_key(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required field '{section}.{key}'")
    return mapping[key]


def _require_string(mapping: dict[str, Any], key: str, section: str) -> str:
    value = _require_key(mapping, key, section)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Field '{section}.{key}' must be a non-empty string")
    return value


def _require_int(mapping: dict[str, Any], key: str, section: str) -> int:
    value = _require_key(mapping, key, section)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Field '{section}.{key}' must be an integer")
    return value


def _require_object_id_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"Field '{field_name}' must be a list")
    object_ids: list[str] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise TypeError(
                f"Field '{field_name}' must contain only string or integer IDs"
            )
        object_ids.append(str(item))
    if len(object_ids) != len(set(object_ids)):
        raise ValueError(f"Field '{field_name}' contains duplicate object IDs")
    return object_ids


def _validate_positive(value: int, field_name: str) -> int:
    if value < 1:
        raise ValueError(f"Field '{field_name}' must be >= 1")
    return value


def _require_part_index_list(value: Any, field_name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer or a list of integers")
    if isinstance(value, int):
        return
    part_indices = _require_list(value, field_name)
    for index, item in enumerate(part_indices):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{field_name}[{index}] must be an integer")


def _finaljson_has_bc_joint(finaljson_data: dict[str, Any], finaljson_path: Path) -> bool:
    group_info = _require_mapping(
        _require_key(finaljson_data, "group_info", f"finaljson[{finaljson_path}]"),
        f"finaljson[{finaljson_path}].group_info",
    )

    has_bc_joint = False
    for raw_group_id, raw_entry in group_info.items():
        group_id = str(raw_group_id)
        field_name = f"finaljson[{finaljson_path}].group_info[{group_id!r}]"
        if group_id == "0":
            _require_part_index_list(raw_entry, field_name)
            continue

        group_entry = _require_list(raw_entry, field_name)
        if len(group_entry) != 4:
            raise ValueError(f"{field_name} must have length 4, got {len(group_entry)}")

        _require_part_index_list(group_entry[0], f"{field_name}[0]")

        parent_group = group_entry[1]
        if isinstance(parent_group, bool) or not isinstance(parent_group, (int, str)):
            raise TypeError(f"{field_name}[1] must be a string or integer parent group")

        _require_list(group_entry[2], f"{field_name}[2]")

        joint_type = group_entry[3]
        if not isinstance(joint_type, str) or not joint_type:
            raise TypeError(f"{field_name}[3] must be a non-empty joint type string")
        if joint_type not in VALID_FINALJSON_JOINT_TYPES:
            raise ValueError(
                f"{field_name}[3] has unsupported joint type {joint_type!r}; "
                f"expected one of {sorted(VALID_FINALJSON_JOINT_TYPES)}"
            )
        if joint_type in {"B", "C"}:
            has_bc_joint = True

    return has_bc_joint


def _finaljson_has_multiple_parts(finaljson_data: dict[str, Any], finaljson_path: Path) -> bool:
    parts = _require_list(
        _require_key(finaljson_data, "parts", f"finaljson[{finaljson_path}]"),
        f"finaljson[{finaljson_path}].parts",
    )
    return len(parts) > 1


def _validate_data_root(data_root: str) -> None:
    if not os.path.isabs(data_root):
        raise ValueError(f"data_root must be an absolute path: {data_root}")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root does not exist: {data_root}")


def _validate_blender(blender: str) -> None:
    if not os.path.isabs(blender):
        raise ValueError(f"render.blender must be an absolute path: {blender}")
    if not os.path.isfile(blender):
        raise FileNotFoundError(f"blender executable does not exist: {blender}")
    if not os.access(blender, os.X_OK):
        raise PermissionError(f"blender is not executable: {blender}")


def resolve_repo_path(raw_path: str | Path) -> str:
    """Resolve an absolute or repository-relative local path.

    Dataset roots and external executables stay absolute, but model code and
    checkpoint paths may be written relative to the repository root so ignored
    local assets can live under ``pretrained/``.
    """
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def _parse_joint_transform(section: dict[str, Any]) -> JointTransformConfig:
    section_name = "joint_transform"
    num_angles = _validate_positive(
        _require_int(section, "num_angles", section_name),
        f"{section_name}.num_angles",
    )
    articulated_raw = _require_key(section, "articulated_objects", section_name)
    if isinstance(articulated_raw, str):
        if articulated_raw != "all":
            raise ValueError(
                "Field 'joint_transform.articulated_objects' must be 'all' or a list"
            )
        articulated_objects: str | list[str] = articulated_raw
    else:
        articulated_objects = _require_object_id_list(
            articulated_raw, "joint_transform.articulated_objects"
        )
    static_objects = _require_object_id_list(
        _require_key(section, "static_objects", section_name),
        "joint_transform.static_objects",
    )
    if articulated_objects != "all":
        overlap = sorted(set(articulated_objects) & set(static_objects))
        if overlap:
            raise ValueError(
                "joint_transform.articulated_objects and joint_transform.static_objects "
                f"overlap: {overlap}"
            )
    return JointTransformConfig(
        num_angles=num_angles,
        articulated_objects=articulated_objects,
        static_objects=static_objects,
    )


def _parse_render(section: dict[str, Any]) -> RenderConfig:
    section_name = "render"
    resolution = _validate_positive(
        _require_int(section, "resolution", section_name),
        f"{section_name}.resolution",
    )
    blender = _require_string(section, "blender", section_name)
    obj_up_axis_raw = section.get("obj_up_axis", "Y")
    if not isinstance(obj_up_axis_raw, str):
        raise TypeError("Field 'render.obj_up_axis' must be a string when provided")
    obj_up_axis = obj_up_axis_raw.upper()
    if obj_up_axis not in VALID_OBJ_UP_AXES:
        raise ValueError("Field 'render.obj_up_axis' must be 'Y' or 'Z'")
    _validate_blender(blender)
    return RenderConfig(
        resolution=resolution,
        blender=blender,
        obj_up_axis=obj_up_axis,
    )


def _parse_voxel(section: dict[str, Any]) -> VoxelConfig:
    section_name = "voxel"
    resolution = _validate_positive(
        _require_int(section, "resolution", section_name),
        f"{section_name}.resolution",
    )
    return VoxelConfig(resolution=resolution)


def _parse_geometry(section: dict[str, Any]) -> GeometryConfig:
    section_name = "geometry"
    obj_up_axis_policy_raw = section.get("obj_up_axis_policy")
    if obj_up_axis_policy_raw is None:
        return GeometryConfig()
    if not isinstance(obj_up_axis_policy_raw, str) or not obj_up_axis_policy_raw:
        raise TypeError(
            "Field 'geometry.obj_up_axis_policy' must be a non-empty string when provided"
        )
    obj_up_axis_policy = obj_up_axis_policy_raw.strip()
    if obj_up_axis_policy not in VALID_OBJ_UP_AXIS_POLICIES:
        raise ValueError(
            "Field 'geometry.obj_up_axis_policy' must be "
            f"one of {sorted(VALID_OBJ_UP_AXIS_POLICIES)}, got {obj_up_axis_policy!r}"
        )
    return GeometryConfig(obj_up_axis_policy=obj_up_axis_policy)


def _contains_bc_joint(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_bc_joint(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_bc_joint(item) for item in value)
    return value in {"B", "C"}


def _resolve_hssd_single_y_joint_z(finaljson_data: dict[str, Any]) -> str:
    parts = _require_key(finaljson_data, "parts", "finaljson")
    if not isinstance(parts, list):
        raise TypeError("Field 'finaljson.parts' must be a list")
    if len(parts) <= 1:
        return "Y"

    group_info = finaljson_data.get("group_info", {})
    return "Z" if _contains_bc_joint(group_info) else "Y"


def resolve_obj_up_axis(
    config: "PipelineConfig",
    finaljson_path: str | Path,
    finaljson_data: dict[str, Any] | None = None,
) -> str:
    """Resolve the Blender OBJ up-axis (Y/Z) for one object finaljson.

    With no geometry.obj_up_axis_policy configured, preserve legacy behavior by
    returning render.obj_up_axis (default Y). The HSSD-only policy uses
    finaljson structure: single-part or no B/C joint stays Y; any B/C joint is Z.
    """
    policy = config.geometry.obj_up_axis_policy
    if policy is None:
        return config.render.obj_up_axis

    if policy != HSSD_SINGLE_Y_JOINT_Z_POLICY:
        raise ValueError(f"Unsupported OBJ up-axis policy: {policy!r}")
    if config.dataset_name != "HSSD":
        raise ValueError(
            f"OBJ up-axis policy {policy!r} is HSSD-only, "
            f"got dataset_name={config.dataset_name!r}"
        )

    if finaljson_data is None:
        path = Path(finaljson_path)
        with path.open("r", encoding="utf-8") as handle:
            finaljson_data = _require_mapping(json.load(handle), f"finaljson[{path}]")
    return _resolve_hssd_single_y_joint_z(finaljson_data)


def _parse_object_filter(config: dict[str, Any]) -> ObjectFilterConfig:
    if "object_filter" not in config:
        return ObjectFilterConfig()

    section = _require_mapping(config["object_filter"], "object_filter")
    unknown_keys = sorted(set(section) - {"mode"})
    if unknown_keys:
        raise ValueError(f"Unknown fields in object_filter: {unknown_keys}")
    mode = _require_string(section, "mode", "object_filter")
    if mode not in VALID_OBJECT_FILTER_MODES:
        raise ValueError(
            "Field 'object_filter.mode' must be one of "
            f"{sorted(VALID_OBJECT_FILTER_MODES)}, got {mode!r}"
        )
    return ObjectFilterConfig(mode=mode)


def _parse_feature(section: dict[str, Any]) -> FeatureConfig:
    return FeatureConfig(
        model=_require_string(section, "model", "feature"),
        dinov2_repo=resolve_repo_path(_require_string(section, "dinov2_repo", "feature")),
        torch_hub_dir=resolve_repo_path(_require_string(section, "torch_hub_dir", "feature")),
    )


def _parse_trellis(section: dict[str, Any]) -> TrellisConfig:
    section_name = "trellis"
    return TrellisConfig(
        root=resolve_repo_path(_require_string(section, "root", section_name)),
        ss_encoder=resolve_repo_path(_require_string(section, "ss_encoder", section_name)),
        ss_decoder=resolve_repo_path(_require_string(section, "ss_decoder", section_name)),
        slat_encoder=resolve_repo_path(_require_string(section, "slat_encoder", section_name)),
    )


def _parse_vlm(section: dict[str, Any]) -> VLMConfig:
    return VLMConfig(image_prefix=_require_string(section, "image_prefix", "vlm"))


def load_config(yaml_path: str) -> PipelineConfig:
    yaml_file = Path(yaml_path)
    if not yaml_file.is_file():
        raise FileNotFoundError(f"YAML config file does not exist: {yaml_path}")

    with yaml_file.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    config = _require_mapping(raw_config, "root")
    dataset_name = _require_string(config, "dataset_name", "root")
    data_root = _require_string(config, "data_root", "root")
    _validate_data_root(data_root)

    joint_transform = _parse_joint_transform(
        _require_mapping(_require_key(config, "joint_transform", "root"), "joint_transform")
    )
    render = _parse_render(_require_mapping(_require_key(config, "render", "root"), "render"))
    voxel = _parse_voxel(_require_mapping(_require_key(config, "voxel", "root"), "voxel"))
    geometry = _parse_geometry(
        _require_mapping(config.get("geometry", {}), "geometry")
    )
    if (
        geometry.obj_up_axis_policy == HSSD_SINGLE_Y_JOINT_Z_POLICY
        and dataset_name != "HSSD"
    ):
        raise ValueError(
            f"geometry.obj_up_axis_policy={HSSD_SINGLE_Y_JOINT_Z_POLICY!r} "
            f"is HSSD-only, got dataset_name={dataset_name!r}"
        )
    object_filter = _parse_object_filter(config)
    feature = _parse_feature(
        _require_mapping(_require_key(config, "feature", "root"), "feature")
    )
    trellis = _parse_trellis(
        _require_mapping(_require_key(config, "trellis", "root"), "trellis")
    )
    vlm = _parse_vlm(_require_mapping(_require_key(config, "vlm", "root"), "vlm"))

    return PipelineConfig(
        dataset_name=dataset_name,
        data_root=data_root,
        joint_transform=joint_transform,
        render=render,
        voxel=voxel,
        geometry=geometry,
        feature=feature,
        trellis=trellis,
        vlm=vlm,
        object_filter=object_filter,
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load and validate a pipeline YAML config.")
    parser.add_argument("yaml_path", help="Path to the YAML config file")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.yaml_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
