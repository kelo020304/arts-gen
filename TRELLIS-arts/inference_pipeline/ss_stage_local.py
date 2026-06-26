from __future__ import annotations
from pathlib import Path
import numpy as np
from .object_inputs import load_object_inputs   # 顶层导入，便于 monkeypatch
from .voxel_io import save_voxel


def _to_np(x): return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def run_mode_a(data_config, *, object_id, angle_idx, view_mode, out_dir, resolution=64, ss_decoder_ckpt: str = "") -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    item = load_object_inputs(data_config, object_id=object_id, angle_idx=angle_idx, view_mode=view_mode)
    z = _to_np(item["z_global"]).astype(np.float32)
    if z.shape != (8, resolution // 4, resolution // 4, resolution // 4):
        raise ValueError(f"z_global 形状异常 {z.shape}")
    np.save(out_dir / "ss_latent.npy", np.ascontiguousarray(z, np.float32))
    if ss_decoder_ckpt:
        import torch
        from inference import decode_ss

        coords = _to_np(decode_ss(torch.from_numpy(z).float(), ss_decoder_ckpt, threshold=0.0)).astype(np.int32)
        source = "gt_z_global_decoded"
    else:
        coords = _to_np(item["raw_surface_coords"]).astype(np.int32)
        source = "gt"
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError(f"raw_surface_coords 形状异常 {coords.shape}")
    save_voxel(out_dir, coords, resolution=resolution, source=source)
    return {"num_voxels": int(coords.shape[0]), "latent_shape": tuple(z.shape)}
