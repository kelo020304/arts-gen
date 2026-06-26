"""Token-conditioned whole-object SS latent flow dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


__all__ = ["SSFlowGlobalZDataset"]


class SSFlowGlobalZDataset(Dataset):
    """Load pre-encoded DINOv2 tokens and target global SS latents.

    Multi-view mode emits ``x_0``, flattened multi-view ``cond`` tokens, and
    ``cam_pose`` aligned to the same view order as the token blocks. Single-view
    mode expands each manifest sample into one item per manifest view and emits
    ``cond`` as one official TRELLIS DINO token sequence. Multiflow-view mode
    keeps one item per manifest sample and emits ``cond`` as [V,T,D] for
    train-time per-view velocity averaging.
    """

    value_range = (0, 1)

    def __init__(self, data_config: dict):
        super().__init__()
        self.data_root = Path(data_config["data_root"])
        self.recon_subdir = str(data_config.get("recon_subdir", "reconstruction"))
        self.manifest_path = data_config.get("manifest_path")
        self.num_views = int(data_config.get("num_views", 4))
        self.default_view_indices = [
            int(v) for v in data_config.get("view_indices", list(range(self.num_views)))
        ]
        self.view_aug = dict(data_config.get("view_aug", {}) or {})
        if bool(self.view_aug.get("enabled", False)):
            raise NotImplementedError("SSFlowGlobalZDataset view_aug.enabled is reserved but not implemented")
        self.condition_mode = str(data_config.get("condition_mode", "multi_view"))
        if self.condition_mode not in {"multi_view", "single_view", "multiflow_view"}:
            raise ValueError(f"unsupported condition_mode={self.condition_mode!r}")
        self.ignore_manifest_dinov2_tokens_path = bool(
            data_config.get("ignore_manifest_dinov2_tokens_path", False)
        )
        self.cache_in_memory = bool(data_config.get("cache_in_memory", False))
        self.latent_subdir = str(data_config.get("latent_subdir", "ss_latents_expanded"))
        self.tokens_subdir = str(data_config.get("tokens_subdir", "dinov2_tokens"))
        self.renders_subdir = str(data_config.get("renders_subdir", "renders"))
        self.expected_token_count = int(data_config.get("token_count", 1374))
        self.expected_token_dim = int(data_config.get("token_dim", 1024))
        self.test_obj_ids = self._load_obj_ids(
            data_config.get("test_obj_ids"),
            data_config.get("test_obj_ids_file"),
        )
        self.test_samples = data_config.get("test_samples", data_config.get("test_object_angles"))
        self.exclude_obj_ids = self._load_obj_ids(
            data_config.get("exclude_obj_ids"),
            data_config.get("exclude_obj_ids_file"),
        )
        self.one_sample_per_object = bool(data_config.get("one_sample_per_object", False))
        self.max_samples = data_config.get("max_samples")

        if len(self.default_view_indices) != self.num_views:
            raise ValueError(
                f"view_indices length must equal num_views={self.num_views}, got {self.default_view_indices}"
            )
        if len(set(self.default_view_indices)) != len(self.default_view_indices):
            raise ValueError(f"view_indices must be unique, got {self.default_view_indices}")
        if min(self.default_view_indices) < 0:
            raise ValueError(f"view_indices must be non-negative physical view ids, got {self.default_view_indices}")

        self.recon_root = self.data_root / self.recon_subdir
        self.latent_root = self.recon_root / self.latent_subdir
        self.token_root = self.recon_root / self.tokens_subdir
        self._latent_cache: dict[str, torch.Tensor] = {}
        self._tokens_cache: dict[str, torch.Tensor] = {}
        self.samples = self._load_samples()
        self.samples = self._filter_samples(self.samples)
        if self.condition_mode == "single_view":
            self.samples = self._expand_single_view_samples(self.samples)
        if not self.samples:
            raise RuntimeError("SSFlowGlobalZDataset produced 0 samples")
        self.loads = [1] * len(self.samples)
        print(
            f"[SSFlowGlobalZDataset] {len(self.samples)} samples "
            f"(mode={self.condition_mode}, views={self.num_views}, recon_root={self.recon_root})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _manifest_abs(self) -> Path | None:
        if self.manifest_path in (None, "", "null"):
            return None
        path = Path(str(self.manifest_path))
        return path if path.is_absolute() else self.data_root / path

    def _rooted(self, rel_or_abs: str | Path) -> Path:
        path = Path(rel_or_abs)
        return path if path.is_absolute() else self.data_root / path

    def _aux_path(self, rel_or_abs: str | Path) -> Path:
        path = Path(rel_or_abs)
        if path.is_absolute():
            return path
        cwd_path = Path.cwd() / path
        return cwd_path if cwd_path.exists() else self.data_root / path

    def _load_obj_ids(self, inline_ids: Any, file_path: Any) -> List[str]:
        ids: List[str] = []
        if inline_ids:
            if isinstance(inline_ids, (str, int)):
                ids.append(str(inline_ids))
            else:
                ids.extend(str(obj_id) for obj_id in inline_ids)
        if file_path not in (None, "", "null"):
            path = self._aux_path(file_path)
            if not path.is_file():
                raise FileNotFoundError(f"object id list not found: {path}")
            text = path.read_text(encoding="utf-8").strip()
            if path.suffix == ".json":
                payload = json.loads(text)
                if isinstance(payload, dict):
                    payload = payload.get("object_ids", payload.get("obj_ids"))
                if not isinstance(payload, list):
                    raise ValueError(f"{path} must contain a JSON list or object_ids list")
                ids.extend(str(obj_id) for obj_id in payload)
            else:
                for line in text.splitlines():
                    line = line.split("#", 1)[0].strip()
                    if line:
                        ids.append(line)
        out: List[str] = []
        seen: set[str] = set()
        for obj_id in ids:
            obj_id = str(obj_id)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            out.append(obj_id)
        return out

    def _default_latent_rel(self, obj_id: str, angle_idx: int) -> str:
        return f"{self.recon_subdir}/{self.latent_subdir}/{obj_id}/angle_{angle_idx}/latent.npz"

    def _default_tokens_rel(self, obj_id: str, angle_idx: int) -> str:
        return f"{self.recon_subdir}/{self.tokens_subdir}/{obj_id}/angle_{angle_idx}/tokens.npz"

    def _default_camera_rel(self, obj_id: str, angle_idx: int) -> str:
        return f"{self.renders_subdir}/{obj_id}/angle_{angle_idx}/camera_transforms.json"

    def _sample_from_record(self, rec: Dict[str, Any]) -> Dict[str, Any] | None:
        complete = rec.get("complete", True)
        if complete is False:
            return None
        obj_id = str(rec.get("object_id", rec.get("obj_id", rec.get("id", ""))))
        if not obj_id:
            raise KeyError(f"manifest record missing object_id/obj_id/id: {rec}")
        if "angle_idx" not in rec and "angle" not in rec:
            raise KeyError(f"manifest record missing angle_idx/angle for object_id={obj_id}")
        angle_idx = int(rec.get("angle_idx", rec.get("angle")))
        view_indices = rec.get("view_indices", self.default_view_indices)
        view_indices = [int(view_idx) for view_idx in view_indices]
        if len(view_indices) != self.num_views:
            raise ValueError(
                f"object_id={obj_id} angle_idx={angle_idx} view_indices length must "
                f"equal num_views={self.num_views}, got {view_indices}"
            )
        if len(set(view_indices)) != len(view_indices):
            raise ValueError(
                f"object_id={obj_id} angle_idx={angle_idx} view_indices must be unique, got {view_indices}"
            )
        if min(view_indices) < 0:
            raise ValueError(
                f"object_id={obj_id} angle_idx={angle_idx} view_indices must be non-negative physical view ids, "
                f"got {view_indices}"
            )
        paths = dict(rec.get("paths", {}))
        tokens_rel = self._default_tokens_rel(obj_id, angle_idx)
        if not self.ignore_manifest_dinov2_tokens_path:
            tokens_rel = paths.get("dinov2_tokens", tokens_rel)
        return {
            "obj_id": obj_id,
            "angle_idx": angle_idx,
            "sample_id": str(rec.get("sample_id", f"{obj_id}_angle_{angle_idx}")),
            "view_indices": view_indices,
            "z_global_rel": paths.get("overall_latent", self._default_latent_rel(obj_id, angle_idx)),
            "tokens_rel": tokens_rel,
            "camera_rel": paths.get("camera_transforms", self._default_camera_rel(obj_id, angle_idx)),
        }

    def _samples_from_manifest(self, manifest_abs: Path) -> List[Dict[str, Any]]:
        if not manifest_abs.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest_abs}")

        text = manifest_abs.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"manifest is empty: {manifest_abs}")

        rows: List[Dict[str, Any]] = []
        if manifest_abs.suffix == ".jsonl":
            for line_no, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                sample = self._sample_from_record(rec)
                if sample is not None:
                    rows.append(sample)
        else:
            payload = json.loads(text)
            if isinstance(payload, dict) and "samples" in payload:
                for rec in payload["samples"]:
                    sample = self._sample_from_record(rec)
                    if sample is not None:
                        rows.append(sample)
            elif isinstance(payload, list):
                for rec in payload:
                    obj_id = str(rec.get("id", rec.get("object_id", rec.get("obj_id", ""))))
                    if not obj_id:
                        raise KeyError(f"manifest record missing object id: {rec}")
                    angles = rec.get("angles")
                    if angles is None:
                        sample = self._sample_from_record(rec)
                        if sample is not None:
                            rows.append(sample)
                    else:
                        for angle_idx in angles:
                            angle = int(angle_idx)
                            rows.append({
                                "obj_id": obj_id,
                                "angle_idx": angle,
                                "sample_id": f"{obj_id}_angle_{angle}",
                                "view_indices": list(self.default_view_indices),
                                "z_global_rel": self._default_latent_rel(obj_id, angle),
                                "tokens_rel": self._default_tokens_rel(obj_id, angle),
                                "camera_rel": self._default_camera_rel(obj_id, angle),
                            })
            else:
                raise ValueError(f"unsupported manifest format at {manifest_abs}: {type(payload)}")

        return self._dedupe_samples(rows)

    def _samples_from_dirs(self) -> List[Dict[str, Any]]:
        if not self.latent_root.is_dir():
            raise FileNotFoundError(f"latent root not found: {self.latent_root}")
        if not self.token_root.is_dir():
            raise FileNotFoundError(f"token root not found: {self.token_root}")

        rows: List[Dict[str, Any]] = []
        for obj_dir in sorted(path for path in self.latent_root.iterdir() if path.is_dir()):
            obj_id = obj_dir.name
            for angle_dir in sorted(path for path in obj_dir.iterdir() if path.is_dir() and path.name.startswith("angle_")):
                try:
                    angle_idx = int(angle_dir.name.split("_", 1)[1])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"invalid angle directory name: {angle_dir}") from exc
                token_path = self.token_root / obj_id / angle_dir.name / "tokens.npz"
                latent_path = angle_dir / "latent.npz"
                if token_path.is_file() and latent_path.is_file():
                    rows.append({
                        "obj_id": obj_id,
                        "angle_idx": angle_idx,
                        "sample_id": f"{obj_id}_{angle_dir.name}",
                        "view_indices": list(self.default_view_indices),
                        "z_global_rel": self._default_latent_rel(obj_id, angle_idx),
                        "tokens_rel": self._default_tokens_rel(obj_id, angle_idx),
                        "camera_rel": self._default_camera_rel(obj_id, angle_idx),
                    })
        return rows

    @staticmethod
    def _dedupe_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for sample in samples:
            key = (str(sample["obj_id"]), int(sample["angle_idx"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(sample)
        return out

    def _load_samples(self) -> List[Dict[str, Any]]:
        manifest_abs = self._manifest_abs()
        if manifest_abs is not None:
            return self._samples_from_manifest(manifest_abs)
        return self._samples_from_dirs()

    def _filter_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = list(samples)
        if self.test_samples:
            wanted = self._parse_sample_keys(self.test_samples)
            by_key = {
                (str(sample["obj_id"]), int(sample["angle_idx"])): sample
                for sample in out
            }
            missing = [key for key in wanted if key not in by_key]
            if missing:
                missing_str = ", ".join(f"{obj_id}:angle_{angle_idx}" for obj_id, angle_idx in missing)
                raise RuntimeError(f"requested test_samples not found: {missing_str}")
            out = [by_key[key] for key in wanted]
        if self.test_obj_ids:
            wanted = {str(obj_id) for obj_id in self.test_obj_ids}
            out = [sample for sample in out if str(sample["obj_id"]) in wanted]
        if self.exclude_obj_ids:
            excluded = {str(obj_id) for obj_id in self.exclude_obj_ids}
            out = [sample for sample in out if str(sample["obj_id"]) not in excluded]
        if self.one_sample_per_object:
            selected: List[Dict[str, Any]] = []
            seen_objects: set[str] = set()
            for sample in out:
                obj_id = str(sample["obj_id"])
                if obj_id in seen_objects:
                    continue
                seen_objects.add(obj_id)
                selected.append(sample)
            out = selected
        if self.max_samples is not None:
            max_samples = int(self.max_samples)
            if max_samples <= 0:
                raise ValueError(f"max_samples must be positive when set, got {max_samples}")
            out = out[:max_samples]
        return out

    @staticmethod
    def _expand_single_view_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sample in samples:
            view_indices = [int(view_idx) for view_idx in sample["view_indices"]]
            for view_pos, view_idx in enumerate(view_indices):
                item = dict(sample)
                item["view_idx"] = int(view_idx)
                item["view_pos"] = int(view_pos)
                item["source_view_indices"] = list(view_indices)
                item["sample_id"] = f"{sample['sample_id']}_view_{view_idx}"
                out.append(item)
        return out

    @staticmethod
    def _parse_sample_keys(value: Any) -> List[tuple[str, int]]:
        keys: List[tuple[str, int]] = []
        for item in value:
            if isinstance(item, str):
                if ":angle_" in item:
                    obj_id, angle_text = item.split(":angle_", 1)
                elif ":" in item:
                    obj_id, angle_text = item.split(":", 1)
                else:
                    raise ValueError(f"test_samples string must be '<obj_id>:angle_<idx>', got {item!r}")
                keys.append((str(obj_id), int(angle_text)))
                continue
            if not isinstance(item, dict):
                raise TypeError(f"test_samples entries must be dicts or strings, got {type(item)}")
            obj_id = str(item.get("obj_id", item.get("object_id", "")))
            if not obj_id:
                raise KeyError(f"test_samples entry missing obj_id/object_id: {item}")
            if "angle_idx" not in item and "angle" not in item:
                raise KeyError(f"test_samples entry missing angle_idx/angle: {item}")
            keys.append((obj_id, int(item.get("angle_idx", item.get("angle")))))
        return keys

    @staticmethod
    def _load_latent(path: Path) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(f"target z_global latent not found: {path}")
        data = np.load(path)
        if "mean" not in data.files:
            raise KeyError(f"{path} expected key 'mean', found keys {data.files}")
        latent = torch.from_numpy(np.asarray(data["mean"])).float()
        if tuple(latent.shape) != (8, 16, 16, 16):
            raise ValueError(f"{path} expected latent shape (8,16,16,16), got {tuple(latent.shape)}")
        return latent

    def _load_latent_cached(self, path: Path) -> torch.Tensor:
        if not self.cache_in_memory:
            return self._load_latent(path)
        key = str(path)
        if key not in self._latent_cache:
            self._latent_cache[key] = self._load_latent(path)
        return self._latent_cache[key].clone()

    def _load_all_tokens_cached(self, path: Path) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(f"DINOv2 tokens not found: {path}")
        key = str(path)
        if self.cache_in_memory and key in self._tokens_cache:
            return self._tokens_cache[key]
        data = np.load(path)
        if "tokens" not in data.files:
            raise KeyError(f"{path} expected key 'tokens', found keys {data.files}")
        tokens = torch.from_numpy(np.asarray(data["tokens"])).float()
        if tokens.dim() != 3:
            raise ValueError(f"{path} expected tokens shape [V,T,D], got {tuple(tokens.shape)}")
        if self.cache_in_memory:
            self._tokens_cache[key] = tokens
        return tokens

    def _load_tokens(self, path: Path, view_indices: List[int]) -> torch.Tensor:
        tokens = self._load_all_tokens_cached(path)
        if min(view_indices) < 0:
            raise ValueError(f"{path} cannot select negative physical view ids: {view_indices}")
        max_view = max(view_indices)
        if tokens.shape[0] <= max_view:
            raise ValueError(
                f"{path} has {tokens.shape[0]} views, cannot select view_indices={view_indices}"
            )
        selected = tokens[view_indices]
        if selected.shape[0] != self.num_views:
            raise ValueError(
                f"{path} selected {selected.shape[0]} views, expected num_views={self.num_views}"
            )
        return selected.reshape(-1, selected.shape[-1]).contiguous()

    def _load_view_tokens(self, path: Path, view_indices: List[int]) -> torch.Tensor:
        tokens = self._load_all_tokens_cached(path)
        if min(view_indices) < 0:
            raise ValueError(f"{path} cannot select negative physical view ids: {view_indices}")
        max_view = max(view_indices)
        if tokens.shape[0] <= max_view:
            raise ValueError(
                f"{path} has {tokens.shape[0]} views, cannot select view_indices={view_indices}"
            )
        selected = tokens[view_indices].contiguous()
        expected = (self.num_views, self.expected_token_count, self.expected_token_dim)
        if tuple(selected.shape) != expected:
            raise ValueError(f"{path} selected token shape {tuple(selected.shape)} != {expected}")
        return selected

    def _load_single_view_token(self, path: Path, view_idx: int) -> torch.Tensor:
        tokens = self._load_all_tokens_cached(path)
        if view_idx < 0 or tokens.shape[0] <= int(view_idx):
            raise ValueError(f"{path} has {tokens.shape[0]} views, cannot select view_idx={view_idx}")
        selected = tokens[int(view_idx)]
        expected = (self.expected_token_count, self.expected_token_dim)
        if tuple(selected.shape) != expected:
            raise ValueError(f"{path} view_idx={view_idx} expected token shape {expected}, got {tuple(selected.shape)}")
        return selected.contiguous()

    def _load_cam_pose(self, path: Path, view_indices: List[int]) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(f"camera transforms not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = payload.get("frames") if isinstance(payload, dict) else payload
        if not isinstance(frames, list):
            raise ValueError(f"{path} expected a frames list, got {type(frames)}")
        by_view: Dict[int, Dict[str, Any]] = {}
        for pos, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise ValueError(f"{path} frame {pos} is not a mapping: {type(frame)}")
            view_idx = int(frame.get("view_index", pos))
            by_view[view_idx] = frame
        missing = [view_idx for view_idx in view_indices if view_idx not in by_view]
        if missing:
            raise ValueError(f"{path} missing camera frames for view_indices={missing}")
        ref_frame = by_view[int(view_indices[0])]
        if "azimuth_deg" not in ref_frame:
            raise KeyError(f"{path} frame view_index={view_indices[0]} missing azimuth_deg")
        az_ref = math.radians(float(ref_frame["azimuth_deg"]))
        rows = []
        for view_idx in view_indices:
            frame = by_view[int(view_idx)]
            if "azimuth_deg" not in frame or "elevation_deg" not in frame:
                raise KeyError(f"{path} frame view_index={view_idx} missing azimuth_deg/elevation_deg")
            d_az = math.radians(float(frame["azimuth_deg"])) - az_ref
            el = math.radians(float(frame["elevation_deg"]))
            rows.append([math.sin(d_az), math.cos(d_az), math.sin(el), math.cos(el)])
        return torch.tensor(rows, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        latent = self._load_latent_cached(self._rooted(sample["z_global_rel"]))
        token_path = self._rooted(sample["tokens_rel"])
        if self.condition_mode == "single_view":
            cond = self._load_single_view_token(token_path, int(sample["view_idx"]))
            return {
                "x_0": latent,
                "cond": cond,
            }
        if self.condition_mode == "multiflow_view":
            cond = self._load_view_tokens(token_path, sample["view_indices"])
            return {
                "x_0": latent,
                "cond": cond,
            }
        cond = self._load_tokens(token_path, sample["view_indices"])
        cam_pose = self._load_cam_pose(self._rooted(sample["camera_rel"]), sample["view_indices"])
        return {
            "x_0": latent,
            "cond": cond,
            "cam_pose": cam_pose,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]], split_size=None):
        if split_size is not None:
            groups = [
                batch[i:i + int(split_size)]
                for i in range(0, len(batch), int(split_size))
            ]
            return [SSFlowGlobalZDataset.collate_fn(group) for group in groups]
        out = {
            "x_0": torch.stack([sample["x_0"] for sample in batch], dim=0),
            "cond": torch.stack([sample["cond"] for sample in batch], dim=0),
        }
        if "cam_pose" in batch[0]:
            out["cam_pose"] = torch.stack([sample["cam_pose"] for sample in batch], dim=0)
        return out
