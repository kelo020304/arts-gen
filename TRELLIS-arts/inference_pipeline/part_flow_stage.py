# TRELLIS-arts/inference_pipeline/part_flow_stage.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def save_part_latents(run_dir, part_latents: dict, *, target_part_names: list) -> list:
    """part_latents = {part_name: np.float32[8,16,16,16]}（run_part_ss_latent_flow decode=False 返回）。
    按 target_part_names 原序写 parts/part_NN_latent.npy + parts/part_NN_meta.json，供 sam3d 解码。"""
    parts_dir = Path(run_dir)/"parts"; parts_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for part_index, name in enumerate(target_part_names):
        if name not in part_latents:
            raise KeyError(f"part_latents 缺 part：{name}")
        latent = np.asarray(part_latents[name])
        if latent.shape != (8, 16, 16, 16):
            raise ValueError(f"part {name} latent 形状异常 {latent.shape}（期望 (8,16,16,16)）")
        latent_fname = f"part_{part_index:02d}_latent.npy"
        np.save(parts_dir/latent_fname, latent.astype(np.float32))
        meta_fname = f"part_{part_index:02d}_meta.json"
        (parts_dir/meta_fname).write_text(json.dumps(
            {"part_index": part_index, "target_part_name": str(name)}))
        written.append(latent_fname)
    if not written: raise ValueError("无 part 可写")
    return written

def save_part_voxels(run_dir, part_coords: dict, *, target_part_names: list, resolution=64) -> list:
    """part_coords = {part_name: np.int64[N,3]}（run_part_ss_latent_flow 真实返回）。
    按 target_part_names 原序写 parts/part_NN_voxel.npz（保序，不 sorted）。"""
    parts_dir = Path(run_dir)/"parts"; parts_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for part_index, name in enumerate(target_part_names):
        if name not in part_coords:
            raise KeyError(f"part_coords 缺 part：{name}")
        xyz = np.asarray(part_coords[name])
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"part {name} coords 形状异常 {xyz.shape}（期望 [N,3]）")
        fname = f"part_{part_index:02d}_voxel.npz"
        np.savez_compressed(parts_dir/fname, coords=xyz.astype(np.int32),
                            resolution=np.int32(resolution), coord_frame="canonical_grid",
                            source="pred", part_index=np.int32(part_index),
                            target_part_name=str(name))
        written.append(fname)
    if not written: raise ValueError("无 part 可写")
    return written

def run(run_dir, data_config, *, object_id, angle_idx, view_mode, part_flow_ckpt, ss_decoder_ckpt,
        num_steps=None, decode_backend: str = "trellis") -> list:
    """顶层 part 阶段：ss_latent.npy + object_inputs → run_part_ss_latent_flow。

    decode_backend="trellis"（默认）：用 TRELLIS SS decoder 解码后存 voxel
      （parts/part_NN_voxel.npz）。0526 4-view 等用 TRELLIS SS VAE latent 空间训练的
      ckpt 必须走这条。
    decode_backend="sam3d"：跳过 TRELLIS 解码，存 per-part latent（parts/part_NN_latent.npy
      + part_NN_meta.json）交给 sam3d decode glue。只用于 latent 本身属于 sam3d SS VAE
      空间的 ckpt；不是 TRELLIS 0526 ckpt 的等价 decoder 替换。"""
    import torch
    from inference_pipeline.object_inputs import load_object_inputs
    from inference import run_part_ss_latent_flow
    if decode_backend not in ("sam3d", "trellis"):
        raise ValueError(f"decode_backend 必须是 'sam3d' 或 'trellis'，得到 {decode_backend!r}")
    run_dir = Path(run_dir)
    z_global = torch.from_numpy(np.load(run_dir/"ss_latent.npy")).float()       # [8,16,16,16]
    item = load_object_inputs(data_config, object_id=object_id, angle_idx=angle_idx, view_mode=view_mode)
    target_part_names = list(item["target_part_names"])
    # part_token_weights：软 mask-重叠池化权重，ckpt 训练时用了（eval/trainer 都转发），
    # 漏传会退化到硬 min_fg=3 投票分支 → part 预测变糙。item.get(...) 对不用池化的
    # 数据集/ckpt 返回 None，保持旧硬投票路径。
    part_token_weights = item.get("part_token_weights")
    if decode_backend == "sam3d":
        result = run_part_ss_latent_flow(
            z_global, item["cond"], part_flow_ckpt,
            target_slots=item["target_slots"].tolist(),
            mask_token_labels=item["mask_token_labels"],
            target_part_names=target_part_names,
            part_token_weights=part_token_weights,
            ss_decoder_ckpt="", num_steps=num_steps, decode=False)
        return save_part_latents(run_dir, result["part_latents"],
                                 target_part_names=target_part_names)
    result = run_part_ss_latent_flow(
        z_global, item["cond"], part_flow_ckpt,
        target_slots=item["target_slots"].tolist(),
        mask_token_labels=item["mask_token_labels"],
        target_part_names=target_part_names,
        part_token_weights=part_token_weights,
        ss_decoder_ckpt=ss_decoder_ckpt, num_steps=num_steps, decode=True)
    return save_part_voxels(run_dir, result["part_coords"],
                            target_part_names=target_part_names)
