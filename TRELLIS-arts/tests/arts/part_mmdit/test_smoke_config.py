from trellis.utils.arts.config_utils import load_config


def _part_groups_for_obj_ids(cfg, obj_ids):
    import json
    from pathlib import Path

    manifest_path = Path(cfg.data.data_root) / cfg.data.manifest_path
    obj_ids = {str(obj_id) for obj_id in obj_ids}
    groups = {"single": set(), "multi": set(), "buttons": set()}
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if not line.strip():
                continue
            record = json.loads(line)
            obj_id = str(record["object_id"])
            if obj_id not in obj_ids:
                continue
            names = [str(name) for name in record["target_part_names"]]
            if len(names) == 1:
                groups["single"].add(obj_id)
            if len(names) > 1:
                groups["multi"].add(obj_id)
            if any(name.startswith("button") for name in names):
                groups["buttons"].add(obj_id)
    return groups


def test_smoke_config_loads_part_mmdit_stage():
    cfg = load_config("configs/arts/part_mmdit/smoke_test.yaml")

    assert cfg.stage == "part_mmdit"
    assert cfg.model.patch_size == 2
    assert list(cfg.model.cross_part_layers) == [3, 6, 9]
    assert cfg.data.num_views == 4
    assert cfg.data.max_samples == 20
    assert cfg.flow.t_schedule == "uniform"
    assert cfg.flow.timestep_shift == 0.0
    assert cfg.eval.fixed_every == 250
    assert cfg.eval.smoke_metrics is True
    assert cfg.loss.cfg_dropout_name == 0.1
    assert cfg.loss.cfg_dropout_anchor == 0.1
    assert cfg.training.max_steps == 2000
    assert cfg.training.checkpoint_every == 500


def test_overnight_config_uses_held_out_val_split():
    cfg = load_config("configs/arts/part_mmdit/train_overnight.yaml")

    train_val_ids = set(str(obj_id) for obj_id in cfg.data.exclude_obj_ids)
    eval_val_ids = set(str(obj_id) for obj_id in cfg.eval.data.include_obj_ids)

    assert cfg.stage == "part_mmdit"
    assert "max_samples" not in cfg.data
    assert len(train_val_ids) == 32
    assert train_val_ids == eval_val_ids
    assert cfg.flow.t_schedule == "uniform"
    assert cfg.flow.timestep_shift == 0.0
    assert cfg.eval.fixed_every == 2000
    assert cfg.training.checkpoint_every == 2000
    assert cfg.training.batch_size == 24
    assert cfg.training.max_steps == 2400
    assert cfg.training.lr == 2.0e-4
    assert cfg.training.warmup_steps == 500
    assert cfg.training.fp16 is True
    assert cfg.training.output_dir == "/robot/data-lab/jzh/art-gen/outputs/part_mmdit_train_v1"


def test_queue_uniform_shift0_config_uses_disjoint_val_and_queue_metrics():
    cfg = load_config("configs/arts/part_mmdit/train_queue_uniform_shift0.yaml")

    train_val_ids = set(str(obj_id) for obj_id in cfg.data.exclude_obj_ids)
    eval_val_ids = set(str(obj_id) for obj_id in cfg.eval.data.include_obj_ids)

    assert cfg.stage == "part_mmdit"
    assert "max_samples" not in cfg.data
    assert len(train_val_ids) == 32
    assert train_val_ids == eval_val_ids
    assert cfg.flow.t_schedule == "uniform"
    assert cfg.flow.timestep_shift == 0.0
    assert cfg.flow.dynamic_timestep_shift is False
    assert cfg.flow.s_name == 1.0
    assert cfg.flow.s_anchor == 1.0
    assert cfg.eval.queue_metrics is True
    assert cfg.eval.cond_only is True
    assert cfg.eval.decode is True
    assert cfg.eval.fixed_every == 2000
    assert cfg.eval.eval_every == 2000
    assert cfg.eval.velocity_t_values == [0.02, 0.1, 0.3, 0.5, 0.9]
    assert cfg.training.fp16 is True
    assert cfg.training.checkpoint_every == 2000
    assert cfg.training.output_dir == "/robot/data-lab/jzh/art-gen/outputs/part_mmdit_queue_uniform_shift0"


def test_full_uniform_config_is_full_manifest_with_grouped_val_split():
    cfg = load_config("configs/arts/part_mmdit/train_full_uniform.yaml")

    train_val_ids = set(str(obj_id) for obj_id in cfg.data.exclude_obj_ids)
    eval_val_ids = set(str(obj_id) for obj_id in cfg.eval.data.include_obj_ids)
    val_groups = _part_groups_for_obj_ids(cfg, eval_val_ids)
    train_prefer_ids = set(str(obj_id) for obj_id in cfg.eval.train_prefer_obj_ids)

    assert cfg.stage == "part_mmdit"
    assert cfg.model.patch_size == 2
    assert cfg.model.num_blocks == 12
    assert cfg.model.max_parts == 20
    assert list(cfg.model.cross_part_layers) == [3, 6, 9]
    assert "include_obj_ids" not in cfg.data
    assert train_val_ids == eval_val_ids
    assert train_prefer_ids.isdisjoint(eval_val_ids)
    assert all(val_groups[group] for group in ("single", "multi", "buttons"))
    assert set(str(obj_id) for obj_id in cfg.eval.prefer_obj_ids) <= eval_val_ids
    assert cfg.flow.t_schedule == "uniform"
    assert cfg.flow.timestep_shift == 0.0
    assert cfg.flow.latent_scale == 8.0
    assert cfg.loss.part_weight_mode == "raw_voxel_count"
    assert cfg.loss.object_balanced is True
    assert cfg.loss.cfg_dropout_name == 0.1
    assert cfg.loss.cfg_dropout_anchor == 0.1
    assert cfg.training.batch_size == 24
    assert cfg.training.max_steps == 20000
    assert cfg.training.warmup_steps == 1000
    assert cfg.eval.queue_metrics is True
    assert cfg.eval.cond_only is True


def test_medium_logit_normal_config_has_disjoint_rich_split():
    cfg = load_config("configs/arts/part_mmdit/medium_128_logit_normal.yaml")

    train_ids = {str(obj_id) for obj_id in cfg.data.include_obj_ids}
    val_ids = {str(obj_id) for obj_id in cfg.eval.data.include_obj_ids}

    assert cfg.stage == "part_mmdit"
    assert cfg.flow.t_schedule == "logit_normal"
    assert cfg.flow.t_logit_normal_mean == 0.0
    assert cfg.flow.t_logit_normal_std == 1.0
    assert len(train_ids) == 128
    assert len(val_ids) == 32
    assert train_ids.isdisjoint(val_ids)
    assert cfg.eval.fixed_every == 100
    assert cfg.eval.max_samples == 16
    assert cfg.eval.train_max_samples == 16
    assert cfg.eval.decode is False
    assert cfg.eval.metrics_markdown_path == "code_update/code_update_part_mmdit.md"


def test_overfit_102276_logit_normal_config_is_cos_only():
    cfg = load_config("configs/arts/part_mmdit/overfit_102276_logit_normal.yaml")

    assert cfg.stage == "part_mmdit"
    assert list(cfg.data.include_obj_ids) == ["102276"]
    assert "data" not in cfg.eval
    assert cfg.flow.t_schedule == "logit_normal"
    assert cfg.eval.decode is False
    assert cfg.eval.max_samples == 1
    assert cfg.eval.train_max_samples == 1
    assert cfg.training.batch_size == 1
