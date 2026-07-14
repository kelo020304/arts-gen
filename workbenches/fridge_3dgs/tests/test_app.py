from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbenches.fridge_3dgs.server import app as app_module


def image_data_url(color: tuple[int, int, int], size: tuple[int, int] = (12, 8)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class FridgeMultiviewImportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_work_root = app_module.WORK_ROOT
        self.original_session_id = app_module.SESSION_ID
        self.original_dataset_config = app_module.DATASET_CONFIG
        app_module.WORK_ROOT = Path(self.temp_dir.name) / "workbench"
        app_module.SESSION_ID = "direct_upload_test"
        app_module.JOBS.clear()
        app_module.JOB_META.clear()
        app_module.KIN_JOBS.clear()
        app_module.KIN_JOB_META.clear()

    def tearDown(self) -> None:
        app_module.WORK_ROOT = self.original_work_root
        app_module.SESSION_ID = self.original_session_id
        app_module.DATASET_CONFIG = self.original_dataset_config
        self.temp_dir.cleanup()

    def import_request(self) -> app_module.ImportViewsRequest:
        source_ids = [0, 3, 8, 11]
        return app_module.ImportViewsRequest(
            views=[
                app_module.ImportViewSpec(
                    view_index=index,
                    image_data_url=image_data_url((index * 50, 20, 200 - index * 40)),
                    name=f"view_{index}",
                    original_name=f"view_{source_ids[index]}.png",
                    source_view_id=source_ids[index],
                )
                for index in range(4)
            ]
        )

    def seed_stale_derivatives(self) -> Path:
        root = app_module.session_dir()
        for index in range(4):
            np.save(root / "mask" / f"mask_{index}.npy", np.ones((8, 12), dtype=np.int32))
            Image.new("L", (12, 8), 1).save(root / "mask" / f"mask_{index}.png")
            Image.new("RGBA", (12, 8), (0, 255, 0, 128)).save(root / "mask_preview" / f"mask_{index}.png")
            Image.new("RGB", (12, 8), (0, 0, 0)).save(root / "dino_input" / f"view_{index}.png")
            (root / "sam3" / f"text_view_{index}.json").write_text("{}\n", encoding="utf-8")
        np.savez(root / "dino_tokens" / "tokens.npz", tokens=np.zeros((4, 1, 1), dtype=np.float32))
        (root / "manifest.json").write_text("{}\n", encoding="utf-8")
        return root

    def test_import_four_views_invalidates_stale_inputs_and_tracks_source(self) -> None:
        root = self.seed_stale_derivatives()

        payload = app_module.import_views(self.import_request())

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["input_source"]["type"], "direct_upload")
        self.assertEqual(
            [item["source_view_id"] for item in payload["input_source"]["views"]],
            [0, 3, 8, 11],
        )
        for index in range(4):
            image_path = root / "rgb" / f"view_{index}.png"
            self.assertTrue(image_path.is_file())
            with Image.open(image_path) as image:
                self.assertEqual(image.size, (12, 8))
            self.assertFalse((root / "mask" / f"mask_{index}.npy").exists())
            self.assertFalse((root / "mask_preview" / f"mask_{index}.png").exists())
            self.assertFalse((root / "dino_input" / f"view_{index}.png").exists())
            self.assertFalse((root / "sam3" / f"text_view_{index}.json").exists())
            camera = json.loads((root / "camera" / f"view_{index}.json").read_text(encoding="utf-8"))
            self.assertEqual(camera["source"], "direct_upload")
        self.assertFalse((root / "dino_tokens" / "tokens.npz").exists())
        self.assertFalse((root / "manifest.json").exists())

        for index in range(4):
            np.save(root / "mask" / f"mask_{index}.npy", np.ones((8, 12), dtype=np.int32))
        finalized = app_module.finalize(
            app_module.FinalizeRequest(labels=[app_module.LabelSpec(id=1, name="body", color="#146c94")])
        )
        self.assertTrue(finalized["ok"])
        self.assertIsNone(finalized["manifest"]["source_3dgs"])
        self.assertEqual(finalized["manifest"]["input_source"]["type"], "direct_upload")

    def test_import_accepts_one_to_four_views_and_requires_contiguous_indices(self) -> None:
        request = self.import_request()
        request.views.pop()
        imported = app_module.import_views(request)
        self.assertTrue(imported["ok"])
        self.assertEqual(len([view for view in imported["views"] if view["image_url"]]), 3)

        duplicate = self.import_request()
        duplicate.views[2].view_index = 1
        with self.assertRaises(HTTPException) as error:
            app_module.import_views(duplicate)
        self.assertEqual(error.exception.status_code, 400)

        with self.assertRaises(HTTPException) as error:
            app_module.import_views(app_module.ImportViewsRequest(views=[]))
        self.assertEqual(error.exception.status_code, 400)

    def test_finalize_normalizes_physical_views_to_four_model_slots(self) -> None:
        request = self.import_request()
        request.views = request.views[:3]
        app_module.import_views(request)
        root = app_module.session_dir()
        for index in range(3):
            np.save(root / "mask" / f"mask_{index}.npy", np.full((8, 12), index + 1, dtype=np.int32))
        labels = [app_module.LabelSpec(id=index + 1, name=f"part_{index + 1}") for index in range(3)]

        finalized = app_module.finalize(app_module.FinalizeRequest(labels=labels))

        mapping = finalized["manifest"]["model_inputs"]
        self.assertEqual(finalized["manifest"]["view_count"], 3)
        self.assertEqual(mapping["model_slot_to_physical_view"], [0, 1, 2, 0])
        self.assertFalse((root / "rgb" / "view_3.png").exists())
        for slot, source in enumerate([0, 1, 2, 0]):
            with Image.open(root / "model_input" / "rgb" / f"view_{slot}.png") as image:
                with Image.open(root / "rgb" / f"view_{source}.png") as expected:
                    self.assertEqual(image.getpixel((0, 0)), expected.getpixel((0, 0)))
            np.testing.assert_array_equal(
                np.load(root / "model_input" / "mask" / f"mask_{slot}.npy"),
                np.load(root / "mask" / f"mask_{source}.npy"),
            )

    def test_dataset_catalog_and_load_copy_integer_masks_without_sam3(self) -> None:
        dataset_root = Path(self.temp_dir.name) / "dataset"
        render_root = dataset_root / "renders" / "obj-1" / "angle_2"
        (render_root / "rgb").mkdir(parents=True)
        (render_root / "mask").mkdir(parents=True)
        for view_index in (4, 7):
            Image.new("RGB", (12, 8), (view_index, 20, 30)).save(render_root / "rgb" / f"view_{view_index}.png")
            raw_mask = np.full((8, 12), 9, dtype=np.int16)
            raw_mask[:, :2] = 5
            np.save(render_root / "mask" / f"mask_{view_index}.npy", raw_mask)
        manifest = dataset_root / "manifest.jsonl"
        manifest.write_text(json.dumps({
            "object_id": "obj-1",
            "angle_idx": 2,
            "sample_id": "obj-1-2",
            "view_indices": [4, 7],
            "target_part_names": ["door"],
            "target_parts": [{"name": "door", "original_label": 9}],
        }) + "\n", encoding="utf-8")
        config = dataset_root / "config.json"
        config.write_text(json.dumps({
            "datasets": [{
                "dataset_id": "fixture",
                "data_root": str(dataset_root),
                "manifest_paths": [str(manifest)],
            }]
        }), encoding="utf-8")
        app_module.DATASET_CONFIG = config

        catalog = app_module.dataset_objects(limit=10)
        self.assertEqual(catalog["objects"][0]["object_id"], "obj-1")
        loaded = app_module.dataset_load(app_module.DatasetLoadRequest(
            object_id="obj-1", angle_idx=2, view_count=2
        ))

        self.assertTrue(loaded["ok"])
        self.assertEqual(loaded["dataset"]["source_view_indices"], [4, 7])
        self.assertEqual(loaded["labels"], [{"id": 1, "name": "door", "color": None}])
        self.assertEqual(loaded["input_source"]["type"], "dataset")
        root = app_module.session_dir()
        loaded_mask = np.load(root / "mask" / "mask_0.npy")
        self.assertEqual(loaded_mask.dtype, np.int32)
        self.assertEqual(sorted(np.unique(loaded_mask).tolist()), [0, 1])
        object_mask = np.load(root / "object_mask" / "mask_0.npy")
        self.assertEqual(object_mask.dtype, np.uint8)
        self.assertTrue(bool(np.all(object_mask == 1)))
        self.assertEqual(loaded["dataset"]["object_mask_source"], "raw_mask_positive")
        part_info = json.loads((root / "part_info.json").read_text(encoding="utf-8"))
        self.assertEqual([part["label"] for part in part_info["parts"].values()], [1])
        self.assertFalse((root / "mask" / "mask_2.npy").exists())
        self.assertEqual(app_module.session()["dataset"]["object_id"], "obj-1")

    def test_reconstruct_repairs_missing_part_info_from_labels(self) -> None:
        request = self.import_request()
        request.views = request.views[:1]
        app_module.import_views(request)
        root = app_module.session_dir()
        np.save(root / "mask" / "mask_0.npy", np.ones((8, 12), dtype=np.int32))
        (root / "labels.json").write_text(
            json.dumps([{"id": 1, "name": "door", "color": None}]),
            encoding="utf-8",
        )
        self.assertFalse((root / "part_info.json").exists())

        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with patch.object(
            app_module,
            "_ckpt_status",
            return_value={"fixture": {"exists": True, "path": "/fixture"}},
        ), patch.object(app_module.subprocess, "Popen", return_value=FakeProcess()):
            job = app_module._reconstruct_start(app_module.ReconstructRequest(), root)

        self.assertTrue((root / "part_info.json").is_file())
        self.assertEqual(job["cmd"][job["cmd"].index("--part-info") + 1], str(root / "part_info.json"))

    def test_reconstruct_uses_normalized_four_slot_paths(self) -> None:
        request = self.import_request()
        request.views = request.views[:2]
        app_module.import_views(request)
        root = app_module.session_dir()
        for index in range(2):
            np.save(root / "mask" / f"mask_{index}.npy", np.ones((8, 12), dtype=np.int32))
        app_module.finalize(app_module.FinalizeRequest(labels=[app_module.LabelSpec(id=1, name="door")]))

        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with patch.object(
            app_module,
            "_ckpt_status",
            return_value={"fixture": {"exists": True, "path": "/fixture"}},
        ), patch.object(app_module.subprocess, "Popen", return_value=FakeProcess()):
            job = app_module._reconstruct_start(app_module.ReconstructRequest(), root)

        self.assertEqual(job["model_inputs"]["model_slot_to_physical_view"], [0, 1, 0, 1])
        self.assertEqual(job["stage"], "dino_ss_flow")
        self.assertTrue(job["cmd"][1].endswith("scripts/inference/reconstruct_stages.py"))
        self.assertEqual(job["cmd"][job["cmd"].index("--stage") + 1], "dino_ss_flow")
        image_arg = job["cmd"].index("--images")
        mask_arg = job["cmd"].index("--masks")
        object_mask_arg = job["cmd"].index("--object-masks")
        self.assertEqual(job["cmd"][image_arg + 1 : image_arg + 5], job["model_inputs"]["images"])
        self.assertEqual(job["cmd"][mask_arg + 1 : mask_arg + 5], job["model_inputs"]["masks"])
        self.assertEqual(
            job["cmd"][object_mask_arg + 1 : object_mask_arg + 5],
            job["model_inputs"]["object_masks"],
        )
        config = json.loads((root / "reconstruct" / "pipeline" / "dino_ss_flow" / "ckpt_config.json").read_text())
        self.assertEqual(config["ss_steps"], 20)
        self.assertEqual(config["ss_cfg_strength"], 7.5)
        self.assertEqual(config["ss_fusion_mode"], "concat")
        self.assertEqual(config["ss_seed"], 20260713)
        self.assertFalse(job["quick_steps"])

    def test_reconstruct_pipeline_reports_independent_stage_progress(self) -> None:
        root = app_module.session_dir()
        stage_root = root / "reconstruct" / "pipeline" / "dino_ss_flow"
        stage_root.mkdir(parents=True)
        Image.new("RGB", (4, 4), (1, 2, 3)).save(stage_root / "token_pca.png")
        (stage_root / "status.json").write_text(json.dumps({
            "stage": "dino_ss_flow",
            "state": "running",
            "progress": 25,
            "artifacts": {"pca": "token_pca.png"},
        }), encoding="utf-8")

        payload = app_module.reconstruct_pipeline()

        self.assertEqual(payload["stages"]["dino_ss_flow"]["progress"], 25)
        self.assertTrue(payload["stages"]["dino_ss_flow"]["artifact_urls"]["pca"].endswith("token_pca.png"))
        self.assertEqual(payload["stages"]["ss_decode"]["state"], "not_started")
        self.assertEqual(payload["active_jobs"], [])

    def test_reconstruct_ss_decode_voxel_returns_interactive_payload(self) -> None:
        root = app_module.session_dir()
        stage_root = root / "reconstruct" / "pipeline" / "ss_decode"
        stage_root.mkdir(parents=True)
        coords = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        np.save(stage_root / "whole_coords.npy", coords)
        (stage_root / "status.json").write_text(
            json.dumps({"stage": "ss_decode", "state": "complete", "progress": 100}),
            encoding="utf-8",
        )

        payload = app_module.reconstruct_ss_decode_voxel()

        self.assertEqual(payload["resolution"], 64)
        self.assertEqual(payload["voxel_count"], 3)
        self.assertEqual(payload["display_count"], 3)
        self.assertEqual(payload["bounds"], {"min": [1, 2, 3], "max": [7, 8, 9]})
        self.assertEqual(payload["coords"], coords.tolist())

    def test_reconstruct_ss_decode_voxel_rejects_stale_artifact(self) -> None:
        root = app_module.session_dir()
        stage_root = root / "reconstruct" / "pipeline" / "ss_decode"
        stage_root.mkdir(parents=True)
        np.save(stage_root / "whole_coords.npy", np.asarray([[1, 2, 3]], dtype=np.int32))

        with self.assertRaises(HTTPException) as error:
            app_module.reconstruct_ss_decode_voxel()

        self.assertEqual(error.exception.status_code, 409)

    def test_reconstruct_part_seg_voxel_includes_residual_body_and_label_layers(self) -> None:
        root = app_module.session_dir()
        ss_root = root / "reconstruct" / "pipeline" / "ss_decode"
        stage_root = root / "reconstruct" / "pipeline" / "part_prompt_seg"
        ss_root.mkdir(parents=True)
        stage_root.mkdir(parents=True)
        whole = np.asarray([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.int32)
        np.save(ss_root / "whole_coords.npy", whole)
        np.savez_compressed(
            stage_root / "part_coords.npz",
            **{"3": np.asarray([[2, 2, 2], [3, 3, 3], [30, 30, 30]], dtype=np.int32)},
        )
        (stage_root / "metadata.json").write_text(json.dumps({
            "part_ids": [3],
            "part_names": {"3": "drawer_0"},
        }), encoding="utf-8")
        (stage_root / "status.json").write_text(
            json.dumps({"stage": "part_prompt_seg", "state": "complete", "progress": 100}),
            encoding="utf-8",
        )
        (root / "labels.json").write_text(json.dumps([
            {"id": 3, "name": "main drawer", "color": "#123456"},
        ]), encoding="utf-8")

        payload = app_module.reconstruct_part_prompt_seg_voxel()

        self.assertEqual(payload["whole_voxel_count"], 4)
        self.assertEqual(payload["voxel_count"], 4)
        self.assertEqual([layer["id"] for layer in payload["layers"]], ["body", "part-3"])
        self.assertEqual(payload["layers"][0]["coords"], [[1, 1, 1], [4, 4, 4]])
        self.assertEqual(payload["layers"][0]["voxel_count"], 2)
        self.assertEqual(payload["layers"][1]["coords"], [[2, 2, 2], [3, 3, 3]])
        self.assertEqual(payload["layers"][1]["label"], "main drawer")
        self.assertEqual(payload["layers"][1]["color"], "#123456")

    def test_reconstruct_part_seg_voxel_rejects_incomplete_stage(self) -> None:
        with self.assertRaises(HTTPException) as error:
            app_module.reconstruct_part_prompt_seg_voxel()

        self.assertEqual(error.exception.status_code, 409)

    def test_reconstruct_part_seg_voxel_exposes_cross_label_conflicts(self) -> None:
        root = app_module.session_dir()
        ss_root = root / "reconstruct" / "pipeline" / "ss_decode"
        stage_root = root / "reconstruct" / "pipeline" / "part_prompt_seg"
        ss_root.mkdir(parents=True)
        stage_root.mkdir(parents=True)
        np.save(ss_root / "whole_coords.npy", np.asarray([
            [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4],
        ], dtype=np.int32))
        np.savez_compressed(
            stage_root / "part_coords.npz",
            **{
                "1": np.asarray([[2, 2, 2], [3, 3, 3]], dtype=np.int32),
                "2": np.asarray([[3, 3, 3], [4, 4, 4]], dtype=np.int32),
            },
        )
        (stage_root / "metadata.json").write_text(json.dumps({
            "part_ids": [1, 2], "part_names": {"1": "door", "2": "drawer"},
        }), encoding="utf-8")
        (stage_root / "status.json").write_text(json.dumps({
            "stage": "part_prompt_seg", "state": "complete", "progress": 100,
        }), encoding="utf-8")

        payload = app_module.reconstruct_part_prompt_seg_voxel()

        self.assertEqual(payload["voxel_count"], 4)
        self.assertEqual(payload["whole_voxel_count"], 4)
        self.assertEqual(payload["overlap_voxel_count"], 1)
        self.assertEqual([layer["id"] for layer in payload["layers"]], ["body", "part-1", "part-2", "conflicts"])
        self.assertEqual(payload["layers"][-1]["coords"], [[3, 3, 3]])
        self.assertEqual([layer["voxel_count"] for layer in payload["layers"]], [1, 1, 1, 1])

    def test_part_prompt_seg_checkpoint_catalog_and_safe_resolution(self) -> None:
        ckpt_root = Path(self.temp_dir.name) / "part-prompt-seg"
        recommended = ckpt_root / "joint-run" / "ckpts" / "latest.pt"
        latest = ckpt_root / "joint-run" / "ckpts" / "latest.pt"
        missing_latest = ckpt_root / "fallback-run" / "ckpts" / "step_120000.pt"
        recommended.parent.mkdir(parents=True)
        missing_latest.parent.mkdir(parents=True)
        recommended_target = recommended.parent / "step_130000.pt"
        recommended_target.write_bytes(b"recommended")
        recommended.symlink_to(recommended_target.name)
        missing_latest.write_bytes(b"fallback")
        (missing_latest.parent / "step_90000.pt").write_bytes(b"older")
        (ckpt_root / "joint-run" / "ckpts" / "notes.txt").write_text("ignored", encoding="utf-8")

        with (
            patch.object(app_module, "PART_SEG_CKPT_ROOT", ckpt_root),
            patch.object(app_module, "DEFAULT_PART_SEG_RUN", "joint-run"),
            patch.object(app_module, "DEFAULT_PART_SEG_CKPT", recommended),
        ):
            payload = app_module.part_prompt_seg_checkpoints()
            config = app_module._ckpt_config(
                app_module.ReconstructRequest(
                    part_seg_run_id="joint-run",
                ),
                Path(self.temp_dir.name) / "out",
            )
            with self.assertRaises(HTTPException) as missing_error:
                app_module._resolve_part_seg_run_id("fallback-run")
            with self.assertRaises(HTTPException):
                app_module._resolve_part_seg_run_id("../outside")

        self.assertEqual(payload["default_id"], "joint-run")
        self.assertEqual([item["id"] for item in payload["runs"]], ["joint-run"])
        self.assertEqual(config["part_seg_ckpt"], str(latest))
        self.assertTrue(payload["runs"][0]["uses_latest"])
        self.assertEqual(missing_error.exception.status_code, 404)
        self.assertNotIn("step", payload["runs"][0])

        with self.assertRaises(ValueError):
            app_module.ReconstructRequest(part_seg_checkpoint_id="joint-run/ckpts/step_130000.pt")

    def test_reconstruct_viewer_manifest_reports_complete_and_part_assets(self) -> None:
        root = app_module.session_dir()
        stage_root = root / "reconstruct" / "pipeline" / "slat_decode"
        part_root = stage_root / "part_01_door"
        (stage_root / "overall").mkdir(parents=True)
        part_root.mkdir(parents=True)
        (stage_root / "overall" / "complete.glb").write_bytes(b"glb")
        (stage_root / "overall" / "complete.ply").write_bytes(b"ply")
        (part_root / "mesh.glb").write_bytes(b"glb")
        (root / "dataset.json").write_text(json.dumps({"object_id": "090"}), encoding="utf-8")
        (stage_root / "summary.json").write_text(json.dumps({
            "overall_assets": {"mesh": "complete.glb", "gaussian": "complete.ply"},
            "parts": [{
                "part_id": 1,
                "label": "door_0",
                "kind": "body",
                "voxel_count": 42,
                "mesh_path": str(part_root / "mesh.glb"),
                "gaussian_path": str(part_root / "missing.ply"),
            }],
        }), encoding="utf-8")

        payload = app_module.reconstruct_viewer_manifest()

        viewer = payload["viewer"]
        self.assertEqual(viewer["title"], "090 decoded components")
        self.assertEqual(viewer["overall"]["label"], "Complete")
        self.assertTrue(viewer["overall"]["visible"])
        self.assertTrue(viewer["overall"]["mesh_url"].endswith("/overall/complete.glb"))
        self.assertTrue(viewer["overall"]["gaussian_url"].endswith("/overall/complete.ply"))
        self.assertEqual(viewer["components"][0]["label"], "door_0")
        self.assertEqual(viewer["components"][0]["kind"], "body")
        self.assertEqual(viewer["components"][0]["voxel_count"], 42)
        self.assertFalse(viewer["components"][0]["visible"])
        self.assertTrue(viewer["components"][0]["mesh_url"].endswith("/part_01_door/mesh.glb"))
        self.assertIsNone(viewer["components"][0]["gaussian_url"])

    def test_kin_agent_start_uses_slat_summary_and_sub_ten_budget(self) -> None:
        root = app_module.session_dir()
        stage_root = root / "reconstruct" / "pipeline" / "slat_decode"
        body = stage_root / "body" / "mesh.glb"
        moving = stage_root / "part" / "mesh.glb"
        body.parent.mkdir(parents=True)
        moving.parent.mkdir(parents=True)
        body.write_bytes(b"body")
        moving.write_bytes(b"part")
        (stage_root / "summary.json").write_text(json.dumps({
            "parts": [
                {"part_id": -1, "label": "body", "kind": "body", "mesh_path": str(body)},
                {"part_id": 1, "label": "drawer", "kind": "part", "mesh_path": str(moving)},
            ],
        }), encoding="utf-8")
        (stage_root / "status.json").write_text(json.dumps({
            "stage": "slat_decode", "state": "complete", "progress": 100,
        }), encoding="utf-8")
        (root / "dataset.json").write_text(json.dumps({"dataset_id": "realappliance"}), encoding="utf-8")

        class FakeProcess:
            pid = 12345

            def poll(self):
                return None

        with patch.object(app_module.subprocess, "Popen", return_value=FakeProcess()):
            payload = app_module.kin_agent_start(app_module.KinAgentRequest(max_iterations=9))

        self.assertEqual(payload["max_iterations"], 9)
        self.assertIn("post_process.kinematic_solver.run_kin_agent_bundle", payload["cmd"])
        self.assertEqual(payload["cmd"][payload["cmd"].index("--dataset-id") + 1], "realappliance")
        config = app_module.kin_agent_config()
        self.assertTrue(config["ready"])
        self.assertEqual(len(config["parts"]), 2)

    def test_kin_agent_result_is_hidden_when_slat_summary_is_newer(self) -> None:
        root = app_module.session_dir()
        stage_root = root / "reconstruct" / "pipeline" / "slat_decode"
        body = stage_root / "body" / "mesh.glb"
        moving = stage_root / "part" / "mesh.glb"
        body.parent.mkdir(parents=True)
        moving.parent.mkdir(parents=True)
        body.write_bytes(b"body")
        moving.write_bytes(b"part")
        summary_path = stage_root / "summary.json"
        summary_path.write_text(json.dumps({
            "parts": [
                {"part_id": -1, "label": "body", "kind": "body", "mesh_path": str(body)},
                {"part_id": 1, "label": "drawer", "kind": "part", "mesh_path": str(moving)},
            ],
        }), encoding="utf-8")
        (stage_root / "status.json").write_text(json.dumps({
            "stage": "slat_decode", "state": "complete", "progress": 100,
        }), encoding="utf-8")
        kin_root = root / "kin_agent"
        kin_root.mkdir()
        result_path = kin_root / "kinematic_result.json"
        result_path.write_text(json.dumps({
            "summary_path": str(summary_path),
            "body_source_mesh": str(body),
            "parts": [{"source_mesh": str(moving)}],
        }), encoding="utf-8")

        self.assertIsNotNone(app_module._kin_result_payload(root))
        newer = result_path.stat().st_mtime_ns + 1_000_000_000
        os.utime(summary_path, ns=(newer, newer))

        self.assertIsNone(app_module._kin_result_payload(root))
        self.assertIsNone(app_module.kin_agent_config()["result"])
        with self.assertRaises(HTTPException) as error:
            app_module.kin_agent_result()
        self.assertEqual(error.exception.status_code, 404)

    def test_kin_agent_start_reuses_matching_cached_result(self) -> None:
        root = app_module.session_dir()
        stage_root = root / "reconstruct" / "pipeline" / "slat_decode"
        body = stage_root / "body" / "mesh.glb"
        moving = stage_root / "part" / "mesh.glb"
        body.parent.mkdir(parents=True)
        moving.parent.mkdir(parents=True)
        body.write_bytes(b"body")
        moving.write_bytes(b"part")
        summary_path = stage_root / "summary.json"
        summary_path.write_text(json.dumps({
            "parts": [
                {"part_id": -1, "label": "body", "kind": "body", "mesh_path": str(body)},
                {"part_id": 1, "label": "drawer", "kind": "part", "mesh_path": str(moving)},
            ],
        }), encoding="utf-8")
        (stage_root / "status.json").write_text(json.dumps({
            "stage": "slat_decode", "state": "complete", "progress": 100,
        }), encoding="utf-8")
        (root / "dataset.json").write_text(json.dumps({"dataset_id": "realappliance"}), encoding="utf-8")
        kin_root = root / "kin_agent"
        kin_root.mkdir()
        xml_path = kin_root / "object.xml"
        usd_path = kin_root / "object.usda"
        collision_audit_path = kin_root / "decoded_collision_audit.json"
        xml_path.write_text("<mujoco/>", encoding="utf-8")
        usd_path.write_text("#usda 1.0\n", encoding="utf-8")
        collision_audit_path.write_text('{"requires_review": false}\n', encoding="utf-8")

        def stamp(path: Path) -> dict:
            stat = path.stat()
            return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

        (kin_root / "kinematic_result.json").write_text(json.dumps({
            "format": "arts_gen_kin_agent_v17",
            "summary_path": str(summary_path), "max_iterations": 7, "dataset_id": "realappliance",
            "body_source_mesh": str(body), "parts": [{"source_mesh": str(moving)}],
            "xml_path": str(xml_path), "usd_path": str(usd_path),
            "collision_audit_path": str(collision_audit_path),
            "collision_audit": {"version": "decoded_collision_audit_v2", "requires_review": False},
            "input_files": [stamp(summary_path), stamp(body), stamp(moving)],
        }), encoding="utf-8")

        with patch.object(app_module.subprocess, "Popen") as popen:
            payload = app_module.kin_agent_start(app_module.KinAgentRequest(max_iterations=7))

        self.assertTrue(payload["cached"])
        self.assertIsNone(payload["job_id"])
        self.assertTrue(xml_path.is_file())
        popen.assert_not_called()

    def test_kin_agent_requires_decoded_body_and_moving_mesh(self) -> None:
        root = app_module.session_dir()
        stage_root = root / "reconstruct" / "pipeline" / "slat_decode"
        stage_root.mkdir(parents=True)
        (stage_root / "summary.json").write_text(json.dumps({
            "parts": [{"part_id": -1, "label": "body", "kind": "body", "mesh_path": str(stage_root / "missing.glb")}],
        }), encoding="utf-8")
        (stage_root / "status.json").write_text(json.dumps({
            "stage": "slat_decode", "state": "complete", "progress": 100,
        }), encoding="utf-8")

        self.assertFalse(app_module.kin_agent_config()["ready"])

    def test_import_requires_explicit_confirmation_before_replacing_rgb(self) -> None:
        request = self.import_request()
        app_module.import_views(request)

        with self.assertRaises(HTTPException) as error:
            app_module.import_views(request)
        self.assertEqual(error.exception.status_code, 409)

        request.replace_existing = True
        replaced = app_module.import_views(request)
        self.assertTrue(replaced["ok"])

    def test_recapture_invalidates_only_the_changed_view_mask(self) -> None:
        root = app_module.session_dir()
        for index in (0, 1):
            np.save(root / "mask" / f"mask_{index}.npy", np.ones((8, 12), dtype=np.int32))

        app_module.save_view(
            app_module.SaveViewRequest(
                view_index=0,
                image_data_url=image_data_url((10, 20, 30)),
                camera={"position": [0, 0, 1]},
                name="front",
            )
        )

        self.assertFalse((root / "mask" / "mask_0.npy").exists())
        self.assertTrue((root / "mask" / "mask_1.npy").exists())
        self.assertEqual(app_module._input_source_state(root)["type"], "3dgs_capture")

    def test_saving_mask_invalidates_dino_and_manifest_but_keeps_mask(self) -> None:
        root = app_module.session_dir()
        app_module.save_view(
            app_module.SaveViewRequest(
                view_index=0,
                image_data_url=image_data_url((10, 20, 30)),
                camera={},
                name="front",
            )
        )
        Image.new("RGB", (12, 8), (0, 0, 0)).save(root / "dino_input" / "view_0.png")
        np.savez(root / "dino_tokens" / "tokens.npz", tokens=np.zeros((4, 1, 1), dtype=np.float32))
        (root / "manifest.json").write_text("{}\n", encoding="utf-8")

        saved = app_module.save_mask(
            app_module.SaveMaskRequest(
                view_index=0,
                mask_data_url=image_data_url((1, 1, 1)),
                labels=[app_module.LabelSpec(id=1, name="body", color="#146c94")],
            )
        )

        self.assertTrue(saved["ok"])
        self.assertTrue((root / "mask" / "mask_0.npy").is_file())
        self.assertTrue((root / "mask_preview" / "mask_0.png").is_file())
        self.assertFalse((root / "dino_input" / "view_0.png").exists())
        self.assertFalse((root / "dino_tokens" / "tokens.npz").exists())
        self.assertFalse((root / "manifest.json").exists())

    def test_transaction_restores_old_files_when_publish_fails(self) -> None:
        root = app_module.session_dir()
        target_a = root / "rgb" / "view_0.png"
        target_b = root / "camera" / "view_0.json"
        removal = root / "manifest.json"
        target_a.write_bytes(b"old-image")
        target_b.write_bytes(b"old-camera")
        removal.write_bytes(b"old-manifest")
        staged_a = root / "staged-image"
        staged_b = root / "staged-camera"
        staged_a.write_bytes(b"new-image")
        staged_b.write_bytes(b"new-camera")

        real_replace = app_module.os.replace
        call_count = 0

        def flaky_replace(source, target):
            nonlocal call_count
            call_count += 1
            if call_count == 5:
                raise OSError("injected publish failure")
            return real_replace(source, target)

        with patch.object(app_module.os, "replace", side_effect=flaky_replace):
            with self.assertRaises(OSError):
                app_module._transactional_replace(
                    root,
                    [(staged_a, target_a), (staged_b, target_b)],
                    [removal],
                )

        self.assertEqual(target_a.read_bytes(), b"old-image")
        self.assertEqual(target_b.read_bytes(), b"old-camera")
        self.assertEqual(removal.read_bytes(), b"old-manifest")

    def test_mask_labels_stay_consistent_across_views_and_previews(self) -> None:
        root = app_module.session_dir()
        for index in (0, 1):
            app_module.save_view(
                app_module.SaveViewRequest(
                    view_index=index,
                    image_data_url=image_data_url((10, 20, 30)),
                    camera={},
                    name=f"view_{index}",
                )
            )
        labels = [
            app_module.LabelSpec(id=1, name="body", color="#ff0000"),
            app_module.LabelSpec(id=2, name="door", color="#00ff00"),
        ]
        app_module.save_mask(app_module.SaveMaskRequest(view_index=0, mask_data_url=image_data_url((1, 1, 1)), labels=labels))
        app_module.save_mask(app_module.SaveMaskRequest(view_index=1, mask_data_url=image_data_url((2, 2, 2)), labels=labels))
        with Image.open(root / "mask_preview" / "mask_1.png") as preview:
            before = preview.getpixel((0, 0))

        recolored = [
            app_module.LabelSpec(id=1, name="body", color="#0000ff"),
            app_module.LabelSpec(id=2, name="door", color="#ff00ff"),
        ]
        app_module.save_mask(app_module.SaveMaskRequest(view_index=0, mask_data_url=image_data_url((1, 1, 1)), labels=recolored))
        with Image.open(root / "mask_preview" / "mask_1.png") as preview:
            after = preview.getpixel((0, 0))
        self.assertNotEqual(before, after)

        with self.assertRaises(HTTPException) as error:
            app_module.save_mask(
                app_module.SaveMaskRequest(
                    view_index=0,
                    mask_data_url=image_data_url((1, 1, 1)),
                    labels=[app_module.LabelSpec(id=1, name="body", color="#0000ff")],
                )
            )
        self.assertEqual(error.exception.status_code, 400)
        self.assertIn("2", str(error.exception.detail))

    def test_finalize_rejects_missing_labels_and_refreshes_previews_atomically(self) -> None:
        root = app_module.session_dir()
        for index in (0, 1):
            app_module.save_view(
                app_module.SaveViewRequest(
                    view_index=index,
                    image_data_url=image_data_url((10, 20, 30)),
                    camera={},
                    name=f"view_{index}",
                )
            )
        labels = [
            app_module.LabelSpec(id=1, name="body", color="#ff0000"),
            app_module.LabelSpec(id=2, name="door", color="#00ff00"),
        ]
        app_module.save_mask(app_module.SaveMaskRequest(view_index=0, mask_data_url=image_data_url((1, 1, 1)), labels=labels))
        app_module.save_mask(app_module.SaveMaskRequest(view_index=1, mask_data_url=image_data_url((2, 2, 2)), labels=labels))
        app_module.finalize(app_module.FinalizeRequest(labels=labels))
        labels_before = (root / "labels.json").read_bytes()
        manifest_before = (root / "manifest.json").read_bytes()

        with self.assertRaises(HTTPException) as error:
            app_module.finalize(
                app_module.FinalizeRequest(labels=[app_module.LabelSpec(id=1, name="body", color="#0000ff")])
            )
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual((root / "labels.json").read_bytes(), labels_before)
        self.assertEqual((root / "manifest.json").read_bytes(), manifest_before)

        with Image.open(root / "mask_preview" / "mask_1.png") as preview:
            before = preview.getpixel((0, 0))
        recolored = [
            app_module.LabelSpec(id=1, name="body", color="#0000ff"),
            app_module.LabelSpec(id=2, name="door", color="#ff00ff"),
        ]
        app_module.finalize(app_module.FinalizeRequest(labels=recolored))
        with Image.open(root / "mask_preview" / "mask_1.png") as preview:
            after = preview.getpixel((0, 0))
        self.assertNotEqual(before, after)

    def test_run_workspace_create_switch_and_list_are_persistent(self) -> None:
        created = app_module.run_select(app_module.RunSelectRequest(run_id="eval-090", create=True))

        self.assertEqual(created["active_run"], "eval-090")
        self.assertEqual(app_module.session_dir(), app_module.WORK_ROOT / "ee-eval" / "eval-090")
        marker = json.loads(
            (app_module.WORK_ROOT / "ee-eval" / app_module.ACTIVE_RUN_FILE).read_text(encoding="utf-8")
        )
        self.assertEqual(marker["run_id"], "eval-090")
        (app_module.session_dir() / "dataset.json").write_text(
            json.dumps({"object_id": "090"}), encoding="utf-8"
        )

        listing = app_module.runs()

        self.assertEqual(listing["active_run"], "eval-090")
        self.assertEqual(listing["runs"][0]["id"], "eval-090")
        self.assertEqual(listing["runs"][0]["object_id"], "090")
        self.assertTrue(listing["runs"][0]["active"])

    def test_run_workspace_rejects_unsafe_ids_and_missing_runs(self) -> None:
        for run_id in ("../escape", "/absolute", "has space", "", ".hidden"):
            with self.subTest(run_id=run_id), self.assertRaises(HTTPException) as error:
                app_module.run_select(app_module.RunSelectRequest(run_id=run_id, create=True))
            self.assertEqual(error.exception.status_code, 400)

        with self.assertRaises(HTTPException) as error:
            app_module.run_select(app_module.RunSelectRequest(run_id="not-created"))
        self.assertEqual(error.exception.status_code, 404)

    def test_run_workspace_switch_is_blocked_while_reconstruct_runs(self) -> None:
        app_module.run_select(app_module.RunSelectRequest(run_id="first", create=True))

        class RunningProcess:
            def poll(self):
                return None

        app_module.JOBS["active-job"] = RunningProcess()
        with self.assertRaises(HTTPException) as error:
            app_module.run_select(app_module.RunSelectRequest(run_id="second", create=True))

        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(app_module.SESSION_ID, "first")
        self.assertEqual(
            json.loads((app_module.WORK_ROOT / "ee-eval" / app_module.ACTIVE_RUN_FILE).read_text(encoding="utf-8"))["run_id"],
            "first",
        )


if __name__ == "__main__":
    unittest.main()
