#!/usr/bin/env python3
"""Run Hunyuan3D-Part/X-Part as an ee-eval post-processing step.

The upstream demo assumes online Hugging Face access and newer torch runtime
features.  This wrapper keeps those compatibility fixes local to the X-Part
process: staged weights, no online downloads, Sonata flash attention disabled,
and small shims for optional sampling packages.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import types
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh


THIRD_PARTY_ROOT = Path("/robot/data-lab/jzh/art-gen/third-party-weights/post_smooth_eval")
DEFAULT_XPART_ROOT = THIRD_PARTY_ROOT / "hunyuan3d-part" / "XPart"
DEFAULT_XPART_WEIGHTS = THIRD_PARTY_ROOT / "hunyuan3d-part" / "pretrained_weights" / "hunyuan3d-part"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value)).strip("_")
    return out or "component"


def _install_torch_compat() -> None:
    class _XPU:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def device_count() -> int:
            return 0

        @staticmethod
        def manual_seed(seed: int) -> None:
            return None

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def get_device_capability(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {}

        @staticmethod
        def get_device_properties(*_args: Any, **_kwargs: Any) -> Any:
            return types.SimpleNamespace(name="xpu-unavailable")

    if not hasattr(torch, "xpu"):
        torch.xpu = _XPU()  # type: ignore[attr-defined]

    if not hasattr(torch.nn, "RMSNorm"):

        class RMSNorm(torch.nn.Module):
            def __init__(
                self,
                normalized_shape: int | tuple[int, ...],
                eps: float = 1.0e-5,
                elementwise_affine: bool = True,
                device: torch.device | str | None = None,
                dtype: torch.dtype | None = None,
            ) -> None:
                super().__init__()
                if isinstance(normalized_shape, int):
                    normalized_shape = (normalized_shape,)
                self.normalized_shape = tuple(normalized_shape)
                self.eps = float(eps)
                if elementwise_affine:
                    self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
                else:
                    self.register_parameter("weight", None)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                return out if self.weight is None else out * self.weight

        torch.nn.RMSNorm = RMSNorm  # type: ignore[attr-defined]


def _install_sampling_shims() -> None:
    if "fpsample" not in sys.modules:
        fpsample = types.ModuleType("fpsample")

        def fps_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
            pts = np.asarray(points, dtype=np.float32)
            total = int(len(pts))
            count = min(int(n_samples), total)
            if count <= 0:
                return np.empty((0,), dtype=np.int64)
            selected = np.empty(count, dtype=np.int64)
            selected[0] = 0
            min_dist = np.full(total, np.inf, dtype=np.float32)
            for idx in range(1, count):
                dist = np.sum((pts - pts[selected[idx - 1]]) ** 2, axis=1)
                min_dist = np.minimum(min_dist, dist)
                selected[idx] = int(np.argmax(min_dist))
            return selected

        fpsample.fps_sampling = fps_sampling  # type: ignore[attr-defined]
        sys.modules["fpsample"] = fpsample

    if "torch_cluster" not in sys.modules:
        torch_cluster = types.ModuleType("torch_cluster")

        def fps(
            src: torch.Tensor,
            batch: torch.Tensor | None = None,
            ratio: float | None = None,
            random_start: bool = True,
            batch_size: int | None = None,
            ptr: torch.Tensor | list[int] | None = None,
        ) -> torch.Tensor:
            del batch_size, ptr
            points = src.float()
            if batch is None:
                batch = torch.zeros((points.shape[0],), dtype=torch.long, device=points.device)
            out: list[torch.Tensor] = []
            use_ratio = 0.5 if ratio is None else float(ratio)
            for batch_id in torch.unique(batch).detach().cpu().tolist():
                local = torch.nonzero(batch == int(batch_id), as_tuple=False).flatten()
                if local.numel() == 0:
                    continue
                count = max(1, int(np.ceil(float(local.numel()) * use_ratio)))
                count = min(count, int(local.numel()))
                local_points = points[local]
                start = 0 if not random_start else int((int(batch_id) * 9973) % int(local.numel()))
                selected = [start]
                min_dist = torch.full((local_points.shape[0],), float("inf"), device=points.device)
                for _ in range(1, count):
                    dist = torch.sum((local_points - local_points[selected[-1]].unsqueeze(0)) ** 2, dim=1)
                    min_dist = torch.minimum(min_dist, dist)
                    selected.append(int(torch.argmax(min_dist).detach().cpu().item()))
                out.append(local[torch.as_tensor(selected, dtype=torch.long, device=local.device)])
            if not out:
                return torch.empty((0,), dtype=torch.long, device=points.device)
            return torch.cat(out, dim=0)

        torch_cluster.fps = fps  # type: ignore[attr-defined]
        sys.modules["torch_cluster"] = torch_cluster


def _patch_sonata(xpart_root: Path) -> None:
    sonata_config = xpart_root / "partgen" / "config" / "sonata.json"

    def make_sonata_model(config_path: str | Path = sonata_config, *, module: Any | None = None) -> torch.nn.Module:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        cfg["enable_flash"] = False
        if module is not None:
            point_transformer = module.model.PointTransformerV3
        else:
            from partgen.models.sonata.model import PointTransformerV3 as point_transformer

        model = point_transformer(**cfg)
        params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"Model params: {params / 1e6:.2f}M {params} (enable_flash=False)", flush=True)
        return model

    import partgen.models.sonata as part_sonata

    part_sonata.load_by_config = lambda config_path: make_sonata_model(config_path, module=part_sonata)
    part_sonata.load = lambda *_args, **_kwargs: make_sonata_model(sonata_config, module=part_sonata)

    sys.path.insert(0, str(xpart_root / "partgen"))
    try:
        import models.sonata as top_sonata

        top_sonata.load_by_config = lambda config_path: make_sonata_model(config_path, module=top_sonata)
        top_sonata.load = lambda *_args, **_kwargs: make_sonata_model(sonata_config, module=top_sonata)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[xpart] top-level sonata patch skipped: {exc!r}", flush=True)


def _prepare_imports(xpart_root: Path) -> None:
    sys.path.insert(0, str(xpart_root))
    sys.path.insert(0, str(xpart_root / "partgen"))
    sys.path.insert(0, str(xpart_root.parent / "P3-SAM"))
    _install_torch_compat()
    _install_sampling_shims()
    _patch_sonata(xpart_root)


def _load_pipeline(xpart_root: Path, weights: Path, *, verbose: bool, device: str) -> Any:
    _prepare_imports(xpart_root)
    from partgen.partformer_pipeline import PartFormerPipeline

    pipeline = PartFormerPipeline.from_pretrained(model_path=str(weights), verbose=bool(verbose))
    _patch_p3sam_limits()
    bbox_predictor = getattr(pipeline, "bbox_predictor", None)
    if bbox_predictor is not None:
        bbox_predictor.point_num = int(os.environ.get("XPART_P3SAM_POINT_NUM", str(getattr(bbox_predictor, "point_num", 100000))))
        bbox_predictor.prompt_num = int(os.environ.get("XPART_P3SAM_PROMPT_NUM", str(getattr(bbox_predictor, "prompt_num", 400))))
        if os.environ.get("XPART_DISABLE_DATAPARALLEL", "1") == "1":
            bbox_predictor.model_parallel = bbox_predictor.model
    pipeline.to(device=device, dtype=torch.float32)
    return pipeline


def _patch_p3sam_limits() -> None:
    """Make P3-SAM's hardcoded 100k/400 sampling constants configurable.

    Upstream ``mesh_sam`` accepts ``point_num``/``prompt_num`` arguments but then
    overwrites them inside the function body.  Re-defining the function from
    source keeps the third-party tree untouched while letting this post runner
    use smaller smoke settings instead of OOMing.
    """

    try:
        import partgen.bbox_estimator.auto_mask_api as auto_mask_api
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[xpart] P3-SAM limit patch skipped: {exc!r}", flush=True)
        return
    source = textwrap.dedent(inspect.getsource(auto_mask_api.mesh_sam))
    original = "    point_num = 100000\n    prompt_num = 400\n"
    replacement = (
        "    point_num = int(os.environ.get('XPART_P3SAM_POINT_NUM', str(point_num)))\n"
        "    prompt_num = int(os.environ.get('XPART_P3SAM_PROMPT_NUM', str(prompt_num)))\n"
    )
    if original not in source:
        print("[xpart] P3-SAM limit patch skipped: expected constants not found", flush=True)
        return
    source = source.replace(original, replacement)
    source = source.replace("        bs = 64\n", "        bs = int(os.environ.get('XPART_P3SAM_BATCH_SIZE', '64'))\n")
    namespace = auto_mask_api.__dict__
    exec(compile(source, str(Path(auto_mask_api.__file__)), "exec"), namespace)
    print(
        "[xpart] patched P3-SAM mesh_sam limits: "
        f"point_num={os.environ.get('XPART_P3SAM_POINT_NUM')} "
        f"prompt_num={os.environ.get('XPART_P3SAM_PROMPT_NUM')} "
        f"batch_size={os.environ.get('XPART_P3SAM_BATCH_SIZE', '64')}",
        flush=True,
    )


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"empty mesh: {path}")
    return mesh


def _normalize_mesh_like_xpart(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray, float]:
    """Match PartFormerPipeline.normalize_mesh without calling mesh_path path."""

    out = mesh.copy()
    vertices = np.asarray(out.vertices, dtype=np.float32)
    min_xyz = np.min(vertices, axis=0)
    max_xyz = np.max(vertices, axis=0)
    center = ((min_xyz + max_xyz) / 2.0).astype(np.float32)
    scale = float(np.max(max_xyz - min_xyz) / 2.0 / 0.8)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid mesh normalization scale: {scale}")
    out.vertices = (vertices - center) / scale
    return out, center, scale


def _normalize_mesh_with_transform(mesh: trimesh.Trimesh, center: np.ndarray, scale: float) -> trimesh.Trimesh:
    out = mesh.copy()
    out.vertices = (np.asarray(out.vertices, dtype=np.float32) - center) / float(scale)
    return out


def _denormalize_scene(scene: trimesh.Scene, center: np.ndarray, scale: float) -> trimesh.Scene:
    for geom in scene.geometry.values():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        geom.vertices = np.asarray(geom.vertices, dtype=np.float32) * float(scale) + center
    return scene


def _sample_mesh_surface(mesh: trimesh.Trimesh, num_points: int, seed: int) -> torch.Tensor:
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("cannot sample empty mesh")
    points, face_idx = trimesh.sample.sample_surface(mesh, int(num_points), seed=int(seed))
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_norm, 1.0e-8)
    sharpedge = np.zeros((int(num_points), 1), dtype=np.float32)
    surface = np.concatenate(
        [np.asarray(points, dtype=np.float32), normals.astype(np.float32), sharpedge],
        axis=1,
    )
    return torch.from_numpy(surface)


def _sample_object_surface_with_upstream_utils(mesh: trimesh.Trimesh, num_points: int, seed: int) -> torch.Tensor:
    """Use X-Part's own mesh sampler for the global object condition."""

    from partgen.utils.mesh_utils import SampleMesh, load_surface_points

    raw = SampleMesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), -1, seed=int(seed))
    rng = np.random.default_rng(seed=int(seed))
    surface, _ = load_surface_points(
        rng,
        raw["random_surface"],
        raw["sharp_surface"],
        pc_size=int(num_points),
        pc_sharpedge_size=0,
        return_sharpedge_label=True,
        return_normal=True,
    )
    return surface.float()


def _load_component_bboxes(labels_path: Path, mesh_paths: list[Path] | None = None) -> tuple[list[str], np.ndarray]:
    labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    if isinstance(labels_payload, dict):
        items = labels_payload.get("components", [])
    else:
        items = labels_payload
    labels: list[str] = []
    bboxes: list[np.ndarray] = []
    for idx, item in enumerate(items):
        label = str(item.get("label") or item.get("component") or f"component_{idx:02d}")
        mesh_path = None
        if mesh_paths is not None and idx < len(mesh_paths):
            mesh_path = mesh_paths[idx]
        elif item.get("mesh_path"):
            mesh_path = Path(item["mesh_path"])
        elif item.get("before_mesh"):
            mesh_path = Path(item["before_mesh"])
        if mesh_path is None:
            continue
        mesh = _load_mesh(Path(mesh_path))
        labels.append(label)
        bboxes.append(np.asarray(mesh.bounds, dtype=np.float32))
    if not bboxes:
        raise ValueError(f"no component bboxes found in {labels_path}")
    return labels, np.stack(bboxes, axis=0)


def _load_component_meshes(labels_path: Path, mesh_paths: list[Path] | None = None) -> list[tuple[str, trimesh.Trimesh]]:
    labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    items = labels_payload.get("components", []) if isinstance(labels_payload, dict) else labels_payload
    components: list[tuple[str, trimesh.Trimesh]] = []
    for idx, item in enumerate(items):
        label = str(item.get("label") or item.get("component") or f"component_{idx:02d}")
        mesh_path = None
        if mesh_paths is not None and idx < len(mesh_paths):
            mesh_path = mesh_paths[idx]
        elif item.get("mesh_path"):
            mesh_path = Path(item["mesh_path"])
        elif item.get("before_mesh"):
            mesh_path = Path(item["before_mesh"])
        if mesh_path is None:
            continue
        mesh = _load_mesh(Path(mesh_path))
        components.append((label, mesh))
    if not components:
        raise ValueError(f"no component meshes found in {labels_path}")
    return components


def _build_seg_conditioning(
    overall_mesh: trimesh.Trimesh,
    component_meshes: list[tuple[str, trimesh.Trimesh]],
    *,
    surface_points: int,
    seed: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, float]:
    normalized_overall, center, scale = _normalize_mesh_like_xpart(overall_mesh)
    labels: list[str] = []
    aabbs: list[np.ndarray] = []
    part_surfaces: list[torch.Tensor] = []
    for idx, (label, component_mesh) in enumerate(component_meshes):
        normalized_component = _normalize_mesh_with_transform(component_mesh, center, scale)
        if len(normalized_component.faces) == 0:
            raise ValueError(f"component has zero faces and cannot condition X-Part: {label}")
        labels.append(label)
        aabbs.append(np.asarray(normalized_component.bounds, dtype=np.float32))
        part_surfaces.append(
            _sample_mesh_surface(
                normalized_component,
                num_points=int(surface_points),
                seed=int(seed) + idx + 1,
            )
        )
    object_surface = _sample_object_surface_with_upstream_utils(
        normalized_overall,
        num_points=int(surface_points),
        seed=int(seed),
    ).unsqueeze(0)
    aabb = torch.from_numpy(np.stack(aabbs, axis=0)).float().unsqueeze(0)
    part_surface_inbbox = torch.stack(part_surfaces, dim=0).float().unsqueeze(0)
    return labels, object_surface, aabb, part_surface_inbbox, center, scale


def _export_scene_components(scene: trimesh.Scene, labels: list[str] | None, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, str] = {}
    geometries = list(scene.geometry.items())
    for idx, (name, geom) in enumerate(geometries):
        mesh = geom if isinstance(geom, trimesh.Trimesh) else trimesh.util.concatenate(tuple(geom.dump()))
        label = labels[idx] if labels is not None and idx < len(labels) else str(mesh.metadata.get("name") or name or f"component_{idx:02d}")
        path = out_dir / f"{_safe_name(label)}.obj"
        mesh.export(path)
        exported[label] = str(path)
    return exported


def run(args: argparse.Namespace) -> dict[str, Any]:
    xpart_root = Path(args.xpart_root).resolve()
    weights = Path(args.weights).resolve()
    mesh_input = Path(args.mesh_input).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    pipeline = _load_pipeline(xpart_root, weights, verbose=bool(args.verbose), device=str(args.device))
    load_seconds = time.time() - started

    mesh = _load_mesh(mesh_input)
    labels: list[str] | None = None
    component_meshes: list[tuple[str, trimesh.Trimesh]] | None = None
    aabb = None
    denormalize_center: np.ndarray | None = None
    denormalize_scale: float | None = None
    if args.component_labels:
        component_paths = [Path(path).resolve() for path in args.component_mesh] if args.component_mesh else None
        component_meshes = _load_component_meshes(Path(args.component_labels).resolve(), component_paths)
        labels = [label for label, _mesh in component_meshes]
        if args.conditioning_mode == "legacy_bbox":
            _labels, aabb_np = _load_component_bboxes(Path(args.component_labels).resolve(), component_paths)
            labels = _labels
            aabb = torch.from_numpy(aabb_np).float()

    inference_started = time.time()
    call_kwargs = {
        "num_inference_steps": int(args.steps),
        "octree_resolution": int(args.octree_resolution),
        "num_chunks": int(args.num_chunks),
        "mc_algo": str(args.mc_algo),
        "output_type": "trimesh",
        "seed": int(args.seed),
        "enable_pbar": bool(args.progress),
    }
    if aabb is None:
        if component_meshes is None:
            scene, aux = pipeline(mesh_path=str(mesh_input), **call_kwargs)
            mode = "p3sam"
        else:
            (
                labels,
                obj_surface,
                aabb,
                part_surface_inbbox,
                denormalize_center,
                denormalize_scale,
            ) = _build_seg_conditioning(
                mesh,
                component_meshes,
                surface_points=int(args.surface_points),
                seed=int(args.seed),
            )
            scene, aux = pipeline(
                obj_surface=obj_surface,
                aabb=aabb,
                part_surface_inbbox=part_surface_inbbox,
                **call_kwargs,
            )
            scene = _denormalize_scene(scene, denormalize_center, denormalize_scale)
            mode = "seg_surface_conditioned"
    else:
        scene, aux = pipeline(mesh=mesh, aabb=aabb, part_surface_inbbox=None, **call_kwargs)
        mode = "legacy_bbox_injected"
    inference_seconds = time.time() - inference_started

    output_glb = out_dir / "output.glb"
    scene.export(output_glb)
    component_paths = _export_scene_components(scene, labels, out_dir / "components")

    aux_paths: dict[str, str] = {}
    if aux is not None:
        for name, aux_scene in zip(("out_bbox", "input_bbox", "explode"), aux, strict=False):
            path = out_dir / f"{name}.glb"
            aux_scene.export(path)
            aux_paths[name] = str(path)

    report = {
        "status": "done",
        "mode": mode,
        "mesh_input": str(mesh_input),
        "out_dir": str(out_dir),
        "output_glb": str(output_glb),
        "component_paths": component_paths,
        "aux_paths": aux_paths,
        "component_labels": labels,
        "conditioning_mode": str(args.conditioning_mode),
        "surface_points": int(args.surface_points),
        "normalization": {
            "applied": denormalize_center is not None,
            "center": denormalize_center.tolist() if denormalize_center is not None else None,
            "scale": float(denormalize_scale) if denormalize_scale is not None else None,
        },
        "weights": str(weights),
        "xpart_root": str(xpart_root),
        "steps": int(args.steps),
        "octree_resolution": int(args.octree_resolution),
        "num_chunks": int(args.num_chunks),
        "mc_algo": str(args.mc_algo),
        "load_seconds": float(load_seconds),
        "inference_seconds": float(inference_seconds),
        "total_seconds": float(time.time() - started),
    }
    _write_json(out_dir / "report.json", report)
    print(f"[xpart] report -> {out_dir / 'report.json'}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--xpart-root", type=Path, default=DEFAULT_XPART_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_XPART_WEIGHTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--octree-resolution", type=int, default=512)
    parser.add_argument("--num-chunks", type=int, default=400000)
    parser.add_argument("--mc-algo", default="mc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--p3sam-point-num", type=int, default=40000)
    parser.add_argument("--p3sam-prompt-num", type=int, default=96)
    parser.add_argument("--p3sam-batch-size", type=int, default=64)
    parser.add_argument("--disable-dataparallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--component-labels", type=Path, default=None)
    parser.add_argument("--component-mesh", type=Path, action="append", default=[])
    parser.add_argument(
        "--conditioning-mode",
        choices=("seg_surface", "legacy_bbox"),
        default="seg_surface",
        help=(
            "seg_surface samples conditioning from true decoded component meshes; "
            "legacy_bbox reproduces the old bbox-only path that samples all faces inside each box."
        ),
    )
    parser.add_argument("--surface-points", type=int, default=81920)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["XPART_P3SAM_POINT_NUM"] = str(int(args.p3sam_point_num))
    os.environ["XPART_P3SAM_PROMPT_NUM"] = str(int(args.p3sam_prompt_num))
    os.environ["XPART_P3SAM_BATCH_SIZE"] = str(int(args.p3sam_batch_size))
    os.environ["XPART_DISABLE_DATAPARALLEL"] = "1" if bool(args.disable_dataparallel) else "0"
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
