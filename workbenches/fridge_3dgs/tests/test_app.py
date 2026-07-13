from __future__ import annotations

import base64
import io
import json
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
            np.save(render_root / "mask" / f"mask_{view_index}.npy", np.full((8, 12), 9, dtype=np.int16))
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
        self.assertEqual(loaded["labels"], [{"id": 9, "name": "door", "color": None}])
        self.assertEqual(loaded["input_source"]["type"], "dataset")
        root = app_module.session_dir()
        self.assertEqual(np.load(root / "mask" / "mask_0.npy").dtype, np.int32)
        self.assertFalse((root / "mask" / "mask_2.npy").exists())
        self.assertEqual(app_module.session()["dataset"]["object_id"], "obj-1")

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
        image_arg = job["cmd"].index("--images")
        mask_arg = job["cmd"].index("--masks")
        self.assertEqual(job["cmd"][image_arg + 1 : image_arg + 5], job["model_inputs"]["images"])
        self.assertEqual(job["cmd"][mask_arg + 1 : mask_arg + 5], job["model_inputs"]["masks"])

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


if __name__ == "__main__":
    unittest.main()
