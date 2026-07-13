#!/usr/bin/env python3
"""Preprocess component-scene GLBs for HoloPart OOD diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def _process_mesh(mesh: trimesh.Trimesh, *, target_faces: int, smooth_iterations: int) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    before_faces = int(len(mesh.faces))
    before_vertices = int(len(mesh.vertices))
    out = mesh.copy()
    used_pymeshlab = False
    error = ""
    try:
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(np.asarray(out.vertices), np.asarray(out.faces)))
        ms.meshing_merge_close_vertices()
        if int(smooth_iterations) > 0:
            try:
                ms.apply_coord_taubin_smoothing(stepsmoothnum=int(smooth_iterations))
            except Exception:
                ms.apply_filter("apply_coord_taubin_smoothing", stepsmoothnum=int(smooth_iterations))
        if int(target_faces) > 0 and before_faces > int(target_faces):
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target_faces), preservenormal=True)
        current = ms.current_mesh()
        out = trimesh.Trimesh(vertices=current.vertex_matrix(), faces=current.face_matrix(), process=False)
        used_pymeshlab = True
    except Exception as exc:
        error = repr(exc)
        if int(smooth_iterations) > 0:
            from trimesh.smoothing import filter_taubin

            filter_taubin(out, lamb=0.5, nu=-0.53, iterations=int(smooth_iterations))
    out.remove_unreferenced_vertices()
    return out, {
        "before_vertices": before_vertices,
        "before_faces": before_faces,
        "after_vertices": int(len(out.vertices)),
        "after_faces": int(len(out.faces)),
        "target_faces": int(target_faces),
        "smooth_iterations": int(smooth_iterations),
        "used_pymeshlab": bool(used_pymeshlab),
        "fallback_error": error,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    scene = trimesh.load(input_path, force="scene", process=False)
    out_scene = trimesh.Scene()
    records: list[dict[str, Any]] = []
    for idx, (name, geom) in enumerate(scene.geometry.items()):
        mesh = geom if isinstance(geom, trimesh.Trimesh) else trimesh.util.concatenate(tuple(geom.dump()))
        processed, stats = _process_mesh(mesh, target_faces=int(args.target_faces), smooth_iterations=int(args.smooth_iterations))
        label = str(mesh.metadata.get("name") or name or f"geometry_{idx}")
        processed.metadata["name"] = label
        out_scene.add_geometry(processed, geom_name=label)
        records.append({"name": label, **stats})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_scene.export(output_path)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "components": records,
        "total_before_faces": int(sum(row["before_faces"] for row in records)),
        "total_after_faces": int(sum(row["after_faces"] for row in records)),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=30000)
    parser.add_argument("--smooth-iterations", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(f"[preprocess-scene] {report['output']} faces {report['total_before_faces']} -> {report['total_after_faces']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
