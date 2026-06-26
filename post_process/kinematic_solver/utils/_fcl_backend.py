"""python-fcl backend over precomputed world-baked VHACD JSON caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from post_process.kinematic_solver.sdk.coordinate_frame import (
    SOURCE_TO_CANONICAL_TRANSFORM,
    source_to_canonical_points,
)

from .errors import SchemaMismatchError, VhacdCacheMissingError, VhacdParamsMismatchError
from .render import RenderHull


class FclBackend:
    def __init__(self) -> None:
        self._objects: dict[str, list[object]] = {}
        self._hull_geom: dict[str, list[dict[str, np.ndarray]]] = {}
        self._exact_objects: dict[str, list[object]] = {}
        self._poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def load_model(
        self,
        *,
        object_id: str,
        part_to_obj_path: dict[str, Path],
        vhacd_cache_root: Path,
        coacd_run_params: dict,
        vhacd_cache_metadata: dict,
        coordinate_transform: str | None = None,
    ) -> None:
        import fcl

        self.clear()
        for part_name in sorted(part_to_obj_path):
            cache_file = vhacd_cache_root / f"{part_name}.json"
            if not cache_file.is_file():
                raise VhacdCacheMissingError(f"missing VHACD cache: {cache_file}")
            payload = json.loads(cache_file.read_text())
            if payload.get("object_id") != object_id or payload.get("part_name") != part_name:
                raise SchemaMismatchError(f"{cache_file}: object_id/part_name mismatch")
            if payload.get("coacd_run_params") != coacd_run_params:
                raise VhacdParamsMismatchError(f"{cache_file}: coacd_run_params mismatch")
            if payload.get("vhacd_cache_metadata") != vhacd_cache_metadata:
                raise VhacdParamsMismatchError(f"{cache_file}: vhacd_cache_metadata mismatch")
            if payload.get("frame") != "world_baked":
                raise SchemaMismatchError(f"{cache_file}: frame must be world_baked")

            objects = []
            hull_geoms = []
            for hull in payload.get("hulls", []):
                vertices = np.asarray(hull["vertices"], dtype=np.float64)
                if coordinate_transform == SOURCE_TO_CANONICAL_TRANSFORM:
                    vertices = source_to_canonical_points(vertices)
                faces = np.asarray(hull["faces"], dtype=np.int32)
                model = fcl.BVHModel()
                model.beginModel(len(vertices), len(faces))
                model.addSubModel(vertices, faces)
                model.endModel()
                objects.append(fcl.CollisionObject(model, fcl.Transform()))
                hull_geoms.append({"vertices": vertices, "faces": faces})
            self._objects[part_name] = objects
            self._hull_geom[part_name] = hull_geoms
            self._poses[part_name] = (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    def load_exact_meshes(
        self,
        *,
        part_to_obj_path: dict[str, Path],
        coordinate_transform: str | None = None,
    ) -> None:
        """Load raw triangle meshes as parallel BVHs for Articraft-style exact checks.

        VHACD hull overlap can false-positive at near-touching surfaces; the
        exact-mesh path is used by CollisionConstraint(use_exact_mesh=True) to
        match Articraft's SAT triangle-pair semantics.
        """
        import fcl
        import trimesh

        self._exact_objects.clear()
        for part_name in sorted(part_to_obj_path):
            obj_path = part_to_obj_path[part_name]
            if not obj_path.is_file():
                continue
            mesh = trimesh.load(str(obj_path), force="mesh", process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            if vertices.shape[0] < 3 or faces.shape[0] < 1:
                continue
            if coordinate_transform == SOURCE_TO_CANONICAL_TRANSFORM:
                vertices = source_to_canonical_points(vertices)
            model = fcl.BVHModel()
            model.beginModel(len(vertices), len(faces))
            model.addSubModel(vertices, faces)
            model.endModel()
            self._exact_objects[part_name] = [fcl.CollisionObject(model, fcl.Transform())]

    def set_pose(self, part_name: str, rotation: np.ndarray, translation: np.ndarray) -> None:
        import fcl

        transform = fcl.Transform(
            np.asarray(rotation, dtype=np.float64),
            np.asarray(translation, dtype=np.float64),
        )
        for obj in self._objects.get(part_name, []):
            obj.setTransform(transform)
        for obj in self._exact_objects.get(part_name, []):
            obj.setTransform(transform)
        self._poses[part_name] = (
            np.asarray(rotation, dtype=np.float64),
            np.asarray(translation, dtype=np.float64),
        )

    def reset_to_identity(self) -> None:
        import fcl

        identity = fcl.Transform()
        for objects in self._objects.values():
            for obj in objects:
                obj.setTransform(identity)
        for objects in self._exact_objects.values():
            for obj in objects:
                obj.setTransform(identity)
        for part_name in self._objects:
            self._poses[part_name] = (
                np.eye(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )

    def overlap(self, moving_parts: list[str], static_parts: list[str]) -> bool:
        return bool(self.overlapping_pairs(moving_parts, static_parts))

    def overlapping_pairs(
        self,
        moving_parts: list[str],
        static_parts: list[str],
    ) -> list[tuple[str, str]]:
        return self._overlapping_pairs(moving_parts, static_parts, self._objects)

    def exact_overlapping_pairs(
        self,
        moving_parts: list[str],
        static_parts: list[str],
    ) -> list[tuple[str, str]]:
        if not self._exact_objects:
            raise RuntimeError(
                "exact mesh BVHs not loaded; call load_exact_meshes() first"
            )
        return self._overlapping_pairs(moving_parts, static_parts, self._exact_objects)

    def _overlapping_pairs(
        self,
        moving_parts: list[str],
        static_parts: list[str],
        store: dict[str, list[object]],
    ) -> list[tuple[str, str]]:
        import fcl

        request = fcl.CollisionRequest()
        pairs: list[tuple[str, str]] = []
        for moving in moving_parts:
            for static in static_parts:
                if moving == static:
                    continue
                pair_overlaps = False
                for a in store.get(moving, []):
                    for b in store.get(static, []):
                        result = fcl.CollisionResult()
                        if fcl.collide(a, b, request, result) > 0 or result.is_collision:
                            pair_overlaps = True
                            break
                    if pair_overlaps:
                        break
                if pair_overlaps:
                    pairs.append((moving, static))
        return pairs

    def clear(self) -> None:
        self._objects.clear()
        self._hull_geom.clear()
        self._exact_objects.clear()
        self._poses.clear()

    def iter_render_hulls(self):
        for part_name in sorted(self._hull_geom):
            rotation, translation = self._poses.get(
                part_name,
                (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)),
            )
            for geom in self._hull_geom[part_name]:
                yield RenderHull(
                    part_name=part_name,
                    vertices=np.asarray(geom["vertices"], dtype=np.float64),
                    faces=np.asarray(geom["faces"], dtype=np.int32),
                    rotation=rotation,
                    translation=translation,
                )
