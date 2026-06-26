import json
import inspect
from pathlib import Path

import numpy as np
import torch

import eval_part_ss_latent_flow_full as full_eval
from eval_part_ss_latent_flow_full import (
    _decode_batch_samples,
    _decode_one_sample,
    _merge_shard_outputs,
    _metric_outputs_complete,
    _persist_metric_outputs,
    _render_selected_examples,
    _set_eval_sample_seed,
    _shard_report_root,
    _shard_sample_indices,
)
from part_ss_full_eval_report import (
    enrich_part_rows,
    build_object_rows,
    select_visualization_examples,
    summarize_tables,
    write_markdown_report,
)


def _part(obj_id, part_name, count, k, iou, recall, *, cos=0.9, sample_id=None):
    return {
        "obj_id": obj_id,
        "sample_id": sample_id or f"{obj_id}_sample",
        "dataset_index": int(obj_id),
        "target_part_name": part_name,
        "part_raw_voxel_count": count,
        "object_part_count": k,
        "decode_iou_pred_vs_raw_ind": iou,
        "decode_recall_pred_vs_raw_ind": recall,
        "decode_precision_pred_vs_raw_ind": min(1.0, recall + 0.1),
        "latent_cos": cos,
        "latent_mse": 0.01 + iou,
        "latent_l1": 0.02 + iou,
        "mse_vs_zero": 1.0 - iou,
        "assignment_diag_iou": iou,
        "assignment_offdiag_max": 0.5 - iou / 2.0,
        "pred_count": int(count // 2),
        "raw_ind_count": count,
    }


def test_full_eval_tables_include_size_k_and_cross_bucket_markdown(tmp_path):
    rows = enrich_part_rows(
        [
            _part("1", "tiny", 20, 2, 0.10, 0.20),
            _part("2", "mid", 800, 4, 0.40, 0.50),
            _part("3", "big", 4000, 17, 0.70, 0.80),
        ],
        size_boundaries=(500.0, 3000.0),
    )
    object_rows = build_object_rows(rows, [])
    summary = summarize_tables(rows, object_rows)

    assert summary["by_size"]["small"]["parts"] == 1
    assert summary["by_size"]["medium"]["iou_mean"] == 0.40
    assert summary["by_size"]["medium"]["cos_mean"] == 0.9
    assert summary["by_k"]["k_16_plus"]["recall_mean"] == 0.80
    assert summary["by_size_x_k"]["large"]["k_16_plus"]["parts"] == 1

    report_path = write_markdown_report(
        tmp_path / "report.md",
        summary=summary,
        part_rows=rows,
        object_rows=object_rows,
        selected_examples=[],
        plots=[],
        step=123,
        stage="part_ss_latent_flow_single_view",
    )
    text = report_path.read_text(encoding="utf-8")
    assert "## Size Buckets" in text
    assert "| small | 1 | 0.100000 | 0.200000 | 0.900000 |" in text
    assert "## K Buckets" in text
    assert "| k_16_plus | 1 | 0.700000 | 0.800000 | 0.900000 |" in text
    assert "## Size x K IoU" in text
    assert "| large | nan | nan | nan | nan | 0.700000 |" in text


def test_object_rows_capture_size_mix_and_target_part_iou():
    rows = enrich_part_rows(
        [
            _part("10", "small_handle", 30, 3, 0.10, 0.20, sample_id="s10"),
            _part("10", "large_body", 5000, 3, 0.80, 0.90, sample_id="s10"),
            _part("10", "medium_knob", 900, 3, 0.20, 0.30, sample_id="s10"),
        ],
        size_boundaries=(500.0, 3000.0),
    )
    object_rows = build_object_rows(
        rows,
        [{
            "obj_id": "10",
            "sample_id": "s10",
            "dataset_index": 10,
            "part_count": 3,
            "target_parts_iou_pred_vs_gt": 0.55,
            "target_parts_precision_pred_vs_gt": 0.60,
            "target_parts_recall_pred_vs_gt": 0.70,
        }],
    )

    assert len(object_rows) == 1
    row = object_rows[0]
    assert row["small_count"] == 1
    assert row["medium_count"] == 1
    assert row["large_count"] == 1
    assert row["has_small_large_mix"] is True
    assert row["size_mix_ratio"] == 5000 / 30
    assert row["target_parts_iou_pred_vs_gt"] == 0.55


def test_visualization_selection_picks_worst_small_high_k_and_mixed_objects():
    part_rows = enrich_part_rows(
        [
            _part("1", "small_bad", 20, 2, 0.05, 0.10),
            _part("2", "small_ok", 25, 8, 0.50, 0.50),
            _part("3", "large_bad_high_k", 5000, 17, 0.20, 0.30),
        ],
        size_boundaries=(500.0, 3000.0),
    )
    object_rows = [
        {
            "obj_id": "3",
            "sample_id": "3_sample",
            "dataset_index": 3,
            "object_part_count": 17,
            "target_parts_iou_pred_vs_gt": 0.15,
            "has_small_large_mix": True,
            "size_mix_ratio": 100.0,
        },
        {
            "obj_id": "2",
            "sample_id": "2_sample",
            "dataset_index": 2,
            "object_part_count": 8,
            "target_parts_iou_pred_vs_gt": 0.60,
            "has_small_large_mix": False,
            "size_mix_ratio": 1.5,
        },
    ]

    selected = select_visualization_examples(part_rows, object_rows, limit_per_group=1)
    groups = {item["group"] for item in selected}
    assert {"worst_small_parts", "worst_high_k_objects", "worst_mixed_size_objects"} <= groups
    assert any(item["obj_id"] == "1" and item["group"] == "worst_small_parts" for item in selected)
    assert any(item["obj_id"] == "3" and item["group"] == "worst_mixed_size_objects" for item in selected)


def test_markdown_report_links_plots_and_voxel_examples(tmp_path):
    rows = enrich_part_rows([_part("1", "small_bad", 20, 2, 0.05, 0.10)], size_boundaries=(500.0, 3000.0))
    summary = summarize_tables(rows, build_object_rows(rows, []))
    selected = [{
        "group": "worst_small_parts",
        "obj_id": "1",
        "sample_id": "1_sample",
        "dataset_index": 1,
        "label": "small_bad",
        "metric": 0.05,
        "png": "voxel_examples/worst_small_parts/001.png",
    }]
    report_path = write_markdown_report(
        tmp_path / "report.md",
        summary=summary,
        part_rows=rows,
        object_rows=[],
        selected_examples=selected,
        plots=[Path("plots/iou_by_size.png")],
        step=50000,
        stage="part_ss_latent_flow",
    )
    text = report_path.read_text(encoding="utf-8")
    assert "[iou_by_size.png](plots/iou_by_size.png)" in text
    assert "## Voxel Examples" in text
    assert "[png](voxel_examples/worst_small_parts/001.png)" in text
    # Ensure JSON-safe summaries are not polluted by Path objects.
    json.dumps(summary)


def test_metric_outputs_persist_before_optional_render_or_plot_steps(tmp_path):
    rows = enrich_part_rows([_part("1", "small_bad", 20, 2, 0.05, 0.10)], size_boundaries=(500.0, 3000.0))
    object_rows = build_object_rows(rows, [])
    summary = summarize_tables(rows, object_rows)

    _persist_metric_outputs(tmp_path, part_rows=rows, object_rows=object_rows, summary=summary)

    assert (tmp_path / "part_metrics.jsonl").is_file()
    assert (tmp_path / "object_metrics.jsonl").is_file()
    assert (tmp_path / "summary.json").is_file()
    persisted = json.loads((tmp_path / "part_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert persisted["target_part_name"] == "small_bad"


def test_metric_outputs_complete_requires_all_primary_metric_files(tmp_path):
    rows = enrich_part_rows([_part("1", "small_bad", 20, 2, 0.05, 0.10)], size_boundaries=(500.0, 3000.0))
    object_rows = build_object_rows(rows, [])
    summary = summarize_tables(rows, object_rows)

    assert _metric_outputs_complete(tmp_path) is False
    _persist_metric_outputs(tmp_path, part_rows=rows, object_rows=object_rows, summary=summary)
    assert _metric_outputs_complete(tmp_path) is True


def test_shard_sample_indices_use_stable_stride_partition():
    indices = list(range(10))

    assert _shard_sample_indices(indices, num_shards=3, shard_index=0) == [0, 3, 6, 9]
    assert _shard_sample_indices(indices, num_shards=3, shard_index=1) == [1, 4, 7]
    assert _shard_sample_indices(indices, num_shards=3, shard_index=2) == [2, 5, 8]
    assert _shard_report_root(Path("/tmp/report/step_000123"), num_shards=3, shard_index=1) == Path(
        "/tmp/report/step_000123/shards/shard_00001_of_00003"
    )


def test_merge_shard_outputs_rebuilds_root_tables_and_prefixes_voxel_links(tmp_path):
    step_root = tmp_path / "step_000123"
    shard0 = _shard_report_root(step_root, num_shards=2, shard_index=0)
    shard1 = _shard_report_root(step_root, num_shards=2, shard_index=1)
    rows0 = enrich_part_rows([_part("1", "tiny", 20, 2, 0.10, 0.20)], size_boundaries=(500.0, 3000.0))
    rows1 = enrich_part_rows([_part("2", "big", 4000, 8, 0.60, 0.70)], size_boundaries=(500.0, 3000.0))
    objects0 = build_object_rows(rows0, [])
    objects1 = build_object_rows(rows1, [])
    _persist_metric_outputs(shard0, part_rows=rows0, object_rows=objects0, summary=summarize_tables(rows0, objects0))
    _persist_metric_outputs(shard1, part_rows=rows1, object_rows=objects1, summary=summarize_tables(rows1, objects1))
    (shard0 / "selected_examples.json").write_text(
        json.dumps([{"group": "worst_small_parts", "dataset_index": 1, "png": "voxel_examples/shared/001.png"}]),
        encoding="utf-8",
    )

    report_path = _merge_shard_outputs(
        report_root=tmp_path,
        step=123,
        num_shards=2,
        stage="part_ss_latent_flow",
        size_boundaries=(500.0, 3000.0),
        write_voxel_examples=1,
    )

    assert report_path == step_root / "report.md"
    part_lines = (step_root / "part_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(part_lines) == 2
    summary = json.loads((step_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["parts"] == 2
    selected = json.loads((step_root / "selected_examples.json").read_text(encoding="utf-8"))
    assert selected[0]["png"] == "shards/shard_00000_of_00002/voxel_examples/shared/001.png"


def test_merge_shard_outputs_reselects_global_examples_limit(tmp_path):
    step_root = tmp_path / "step_000123"
    shard0 = _shard_report_root(step_root, num_shards=2, shard_index=0)
    shard1 = _shard_report_root(step_root, num_shards=2, shard_index=1)
    rows0 = enrich_part_rows(
        [
            _part("1", "small_bad", 20, 2, 0.10, 0.20, sample_id="s1"),
            _part("2", "small_worse", 30, 2, 0.01, 0.10, sample_id="s2"),
        ],
        size_boundaries=(500.0, 3000.0),
    )
    rows1 = enrich_part_rows(
        [_part("3", "small_ok", 40, 2, 0.50, 0.50, sample_id="s3")],
        size_boundaries=(500.0, 3000.0),
    )
    objects0 = build_object_rows(rows0, [])
    objects1 = build_object_rows(rows1, [])
    _persist_metric_outputs(shard0, part_rows=rows0, object_rows=objects0, summary=summarize_tables(rows0, objects0))
    _persist_metric_outputs(shard1, part_rows=rows1, object_rows=objects1, summary=summarize_tables(rows1, objects1))
    for shard in (shard0, shard1):
        (shard / "selected_examples.json").write_text(
            json.dumps([
                {
                    "group": "worst_small_parts",
                    "dataset_index": 1,
                    "obj_id": "1",
                    "png": "voxel_examples/shared/001.png",
                },
                {
                    "group": "worst_small_parts",
                    "dataset_index": 2,
                    "obj_id": "2",
                    "png": "voxel_examples/shared/002.png",
                },
            ]),
            encoding="utf-8",
        )

    _merge_shard_outputs(
        report_root=tmp_path,
        step=123,
        num_shards=2,
        stage="part_ss_latent_flow",
        size_boundaries=(500.0, 3000.0),
        write_voxel_examples=1,
    )

    selected = json.loads((step_root / "selected_examples.json").read_text(encoding="utf-8"))
    small = [item for item in selected if item["group"] == "worst_small_parts"]
    assert len(small) == 1
    assert small[0]["dataset_index"] == 2


def test_finalize_persists_metric_outputs_before_optional_artifacts(tmp_path, monkeypatch):
    rows = enrich_part_rows([_part("1", "small_bad", 20, 2, 0.05, 0.10)], size_boundaries=(500.0, 3000.0))
    object_rows = build_object_rows(rows, [])
    summary = summarize_tables(rows, object_rows)
    calls = []
    real_persist = full_eval._persist_metric_outputs

    def wrapped_persist(*args, **kwargs):
        calls.append("persist")
        return real_persist(*args, **kwargs)

    def fake_render_selected_examples(**kwargs):
        calls.append("render")
        return kwargs["selected"]

    def fake_write_plots(*args, **kwargs):
        calls.append("plots")
        raise RuntimeError("plot failed after metrics")

    monkeypatch.setattr(full_eval, "_persist_metric_outputs", wrapped_persist)
    monkeypatch.setattr(full_eval, "_render_selected_examples", fake_render_selected_examples)
    monkeypatch.setattr(full_eval, "_write_plots", fake_write_plots)

    try:
        full_eval._finalize_full_eval_outputs(
            report_root=tmp_path,
            part_rows=rows,
            object_rows=object_rows,
            summary=summary,
            model=None,
            dataset=None,
            decoder=None,
            device=torch.device("cpu"),
            flow_cfg={},
            threshold=0.0,
            debug_thresholds=(0.0,),
            base_seed=42,
            write_voxel_examples=0,
            ckpt_step=123,
            stage="part_ss_latent_flow",
        )
    except RuntimeError as exc:
        assert str(exc) == "plot failed after metrics"
    else:
        raise AssertionError("expected plot failure")

    assert calls == ["persist", "render", "plots"]
    assert (tmp_path / "part_metrics.jsonl").is_file()
    assert (tmp_path / "object_metrics.jsonl").is_file()
    assert (tmp_path / "summary.json").is_file()


def test_render_selected_examples_shares_sanitized_png_for_duplicate_groups(tmp_path, monkeypatch):
    class DummyDataset:
        samples = [{"sample": idx} for idx in range(8)]

        def __getitem__(self, index):
            return {"index": index}

        @staticmethod
        def collate_fn(batch):
            return {"index": [batch[0]["index"]]}

    def fake_decode_one_sample(**kwargs):
        return [], {}, {"obj_id": "cat/102068", "parts": []}

    def fake_make_panel(_panel, path):
        path.write_text("png", encoding="utf-8")

    monkeypatch.setattr(full_eval, "_decode_one_sample", fake_decode_one_sample)
    monkeypatch.setattr(full_eval, "_make_object_panel_png", fake_make_panel)
    selected = [
        {"group": "worst_small_parts", "dataset_index": 7, "obj_id": "x"},
        {"group": "worst_high_k_objects", "dataset_index": 7, "obj_id": "x"},
    ]

    rendered = _render_selected_examples(
        selected=selected,
        model=None,
        dataset=DummyDataset(),
        decoder=None,
        device=torch.device("cpu"),
        flow_cfg={},
        report_root=tmp_path,
        threshold=0.0,
        debug_thresholds=(0.0,),
        base_seed=42,
    )

    pngs = {item["png"] for item in rendered}
    assert pngs == {"voxel_examples/shared/000007_cat_102068_target_parts.png"}
    assert (tmp_path / next(iter(pngs))).read_text(encoding="utf-8") == "png"


def test_merge_shard_outputs_has_no_dead_skip_existing_argument():
    signature = inspect.signature(_merge_shard_outputs)
    assert "skip_existing" not in signature.parameters


def test_render_selected_examples_does_not_reseed_global_rng(tmp_path, monkeypatch):
    class DummyDataset:
        samples = [{"sample": 0}]

        def __getitem__(self, index):
            return {"index": index}

        @staticmethod
        def collate_fn(batch):
            return {"index": [batch[0]["index"]]}

    def fail_reseed(*args, **kwargs):
        raise AssertionError("_render_selected_examples should pass base_seed to decode, not reseed globally")

    def fake_decode_one_sample(**kwargs):
        assert kwargs["base_seed"] == 42
        return [], {}, {"obj_id": "1", "parts": []}

    def fake_make_panel(_panel, path):
        path.write_text("png", encoding="utf-8")

    monkeypatch.setattr(full_eval, "_set_eval_sample_seed", fail_reseed)
    monkeypatch.setattr(full_eval, "_decode_one_sample", fake_decode_one_sample)
    monkeypatch.setattr(full_eval, "_make_object_panel_png", fake_make_panel)

    rendered = _render_selected_examples(
        selected=[{"group": "worst_small_parts", "dataset_index": 0, "obj_id": "1"}],
        model=None,
        dataset=DummyDataset(),
        decoder=None,
        device=torch.device("cpu"),
        flow_cfg={},
        report_root=tmp_path,
        threshold=0.0,
        debug_thresholds=(0.0,),
        base_seed=42,
    )

    assert rendered[0]["png"] == "voxel_examples/shared/000000_1_target_parts.png"


def test_eval_sample_seed_is_deterministic_per_dataset_index():
    _set_eval_sample_seed(42, 17)
    first = torch.randn(4)
    _set_eval_sample_seed(999, 17)
    different_base = torch.randn(4)
    _set_eval_sample_seed(42, 17)
    second = torch.randn(4)
    _set_eval_sample_seed(42, 18)
    different_index = torch.randn(4)

    assert torch.equal(first, second)
    assert not torch.equal(first, different_base)
    assert not torch.equal(first, different_index)


def test_decode_one_sample_is_repeatable_with_eval_sample_seed(monkeypatch):
    class DummyDataset:
        @staticmethod
        def _iter_rgb_paths(_sample_meta):
            return []

        @staticmethod
        def _iter_mask_paths(_sample_meta):
            return []

    def fake_sample_part_ss_latent(*args, **kwargs):
        return torch.randn(1, 1, 2, 2)

    def fake_decode_ss_latent_to_coords(_decoder, latent, threshold=0.0):
        value = int(float(latent.reshape(-1)[0]) > 0.0)
        return torch.tensor([[value, 0, 0]], dtype=torch.long)

    def fake_decode_ss_latent_to_coords_with_stats(_decoder, latent, threshold=0.0, debug_thresholds=(0.0,)):
        coords = fake_decode_ss_latent_to_coords(_decoder, latent, threshold=threshold)
        return coords, {
            "logit_max": float(latent.max().item()),
            "logit_mean": float(latent.mean().item()),
            "logit_p99": float(latent.max().item()),
            "counts": {str(threshold): int(coords.shape[0])},
        }

    monkeypatch.setattr(full_eval, "sample_part_ss_latent", fake_sample_part_ss_latent)
    monkeypatch.setattr(full_eval, "decode_ss_latent_to_coords", fake_decode_ss_latent_to_coords)
    monkeypatch.setattr(full_eval, "decode_ss_latent_to_coords_with_stats", fake_decode_ss_latent_to_coords_with_stats)
    batch = {
        "obj_id": ["1"],
        "sample_id": ["sample_1"],
        "part_valid": torch.tensor([[True]]),
        "z_global": torch.zeros(1, 1, 2, 2),
        "cond": torch.zeros(1, 1, 2),
        "mask_token_labels": torch.zeros(1, 1, dtype=torch.long),
        "target_slots": torch.tensor([[1]]),
        "x_1_parts": torch.zeros(1, 1, 2, 2),
        "target_part_names": [["handle"]],
        "raw_ind_coords": [[torch.tensor([[0, 0, 0]], dtype=torch.long)]],
        "raw_surface_coords": [torch.tensor([[0, 0, 0]], dtype=torch.long)],
    }

    _set_eval_sample_seed(123, 5)
    rows_a, object_a, panel_a = _decode_one_sample(
        model=None,
        dataset=DummyDataset(),
        decoder=None,
        sample_meta={},
        batch=batch,
        dataset_index=5,
        device=torch.device("cpu"),
        flow_cfg={"num_steps": 2, "noise_scale": 1.0, "latent_scale": 1.0},
        threshold=0.0,
        debug_thresholds=(0.0,),
        include_panel_arrays=True,
    )
    _set_eval_sample_seed(123, 5)
    rows_b, object_b, panel_b = _decode_one_sample(
        model=None,
        dataset=DummyDataset(),
        decoder=None,
        sample_meta={},
        batch=batch,
        dataset_index=5,
        device=torch.device("cpu"),
        flow_cfg={"num_steps": 2, "noise_scale": 1.0, "latent_scale": 1.0},
        threshold=0.0,
        debug_thresholds=(0.0,),
        include_panel_arrays=True,
    )

    assert rows_a[0]["pred_count_at_thresholds"] == rows_b[0]["pred_count_at_thresholds"]
    assert object_a == object_b
    assert np.array_equal(panel_a["parts"][0]["pred_coords"], panel_b["parts"][0]["pred_coords"])


def test_decode_batch_samples_calls_sampler_once_for_full_batch(monkeypatch):
    class DummyDataset:
        @staticmethod
        def _iter_rgb_paths(_sample_meta):
            return []

        @staticmethod
        def _iter_mask_paths(_sample_meta):
            return []

    calls = []

    def fake_sample_part_ss_latent(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["z_global"].shape[0] == 2
        assert kwargs["initial_noise"].shape[:2] == (2, 1)
        return kwargs["initial_noise"]

    def fake_decode_ss_latent_to_coords(_decoder, latent, threshold=0.0):
        value = int(float(latent.reshape(-1)[0]) > 0.0)
        return torch.tensor([[value, 0, 0]], dtype=torch.long)

    def fake_decode_ss_latent_to_coords_with_stats(_decoder, latent, threshold=0.0, debug_thresholds=(0.0,)):
        coords = fake_decode_ss_latent_to_coords(_decoder, latent, threshold=threshold)
        return coords, {
            "logit_max": float(latent.max().item()),
            "logit_mean": float(latent.mean().item()),
            "logit_p99": float(latent.max().item()),
            "counts": {str(threshold): int(coords.shape[0])},
        }

    monkeypatch.setattr(full_eval, "sample_part_ss_latent", fake_sample_part_ss_latent)
    monkeypatch.setattr(full_eval, "decode_ss_latent_to_coords", fake_decode_ss_latent_to_coords)
    monkeypatch.setattr(full_eval, "decode_ss_latent_to_coords_with_stats", fake_decode_ss_latent_to_coords_with_stats)
    batch = {
        "obj_id": ["1", "2"],
        "sample_id": ["sample_1", "sample_2"],
        "part_valid": torch.tensor([[True], [True]]),
        "z_global": torch.zeros(2, 1, 2, 2),
        "cond": torch.zeros(2, 1, 2),
        "mask_token_labels": torch.zeros(2, 1, dtype=torch.long),
        "target_slots": torch.tensor([[1], [1]]),
        "x_1_parts": torch.zeros(2, 1, 1, 2, 2),
        "target_part_names": [["handle"], ["door"]],
        "raw_ind_coords": [
            [torch.tensor([[0, 0, 0]], dtype=torch.long)],
            [torch.tensor([[1, 0, 0]], dtype=torch.long)],
        ],
        "raw_surface_coords": [
            torch.tensor([[0, 0, 0]], dtype=torch.long),
            torch.tensor([[1, 0, 0]], dtype=torch.long),
        ],
    }

    decoded = _decode_batch_samples(
        model=None,
        dataset=DummyDataset(),
        decoder=None,
        sample_metas=[{}, {}],
        batch=batch,
        dataset_indices=[5, 6],
        device=torch.device("cpu"),
        flow_cfg={"num_steps": 2, "noise_scale": 1.0, "latent_scale": 1.0},
        threshold=0.0,
        debug_thresholds=(0.0,),
        include_panel_arrays=False,
        base_seed=123,
    )

    assert len(calls) == 1
    assert len(decoded) == 2
    assert decoded[0][0][0]["obj_id"] == "1"
    assert decoded[1][0][0]["obj_id"] == "2"
