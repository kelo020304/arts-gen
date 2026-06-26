from __future__ import annotations
from pathlib import Path
import numpy as np


def run(run_dir, *, encoder_ckpt: str, resolution: int = 64) -> dict:
    """mode B：把 sam3d 出的 surface voxel（``voxel.npz``）用 TRELLIS SS encoder 重编码成
    z_global，覆盖 ``ss_latent.npy``。

    sam3d SS glue 写出的 ``ss_latent.npy`` 是 **sam3d SS 空间**的 latent，而 TRELLIS 训练
    的 part flow（如 0526）吃的 z_global 是 **TRELLIS SS 空间**（pipeline step 8 用同一个
    ``ss_enc_conv3d`` encoder 编的）。直接喂 sam3d latent 会空间不匹配 → part 预测错。
    这里只取 sam3d 的 **voxel**（图→surface voxel），再用 TRELLIS encoder 编成正确的
    z_global，让两端空间一致。失败暴露：缺 voxel.npz 抛 FileNotFoundError。
    """
    from inference import encode_ss

    run_dir = Path(run_dir)
    voxel_npz = run_dir / "voxel.npz"
    if not voxel_npz.is_file():
        raise FileNotFoundError(f"mode B 缺 voxel.npz（先跑 SS sam3d）：{voxel_npz}")
    with np.load(voxel_npz) as data:
        coords = np.asarray(data["coords"]).astype(np.int32).reshape(-1, 3)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError(f"voxel.npz coords 形状异常 {coords.shape}")
    z_global = encode_ss(coords, encoder_ckpt, resolution=resolution).numpy().astype(np.float32)
    np.save(run_dir / "ss_latent.npy", np.ascontiguousarray(z_global))
    return {"num_voxels": int(coords.shape[0]), "latent_shape": tuple(z_global.shape)}
