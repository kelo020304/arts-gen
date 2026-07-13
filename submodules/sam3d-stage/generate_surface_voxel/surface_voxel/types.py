from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class VoxelOutput:
    coords: torch.IntTensor
    downsample_factor: int
    rotation: torch.Tensor
    translation: torch.Tensor
    scale: torch.Tensor
    intrinsics: torch.Tensor
    pointmap_unnorm: torch.Tensor | None
    seed: int

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        coords_cpu = self.coords.detach().cpu()
        surface = coords_cpu[:, 1:].to(torch.int64).numpy()

        if surface.shape[0] == 0:
            raise ValueError("surface voxel coords are empty (N=0); inference produced no voxels")

        lo, hi = int(surface.min()), int(surface.max())
        if lo < 0 or hi > 63:
            raise ValueError(
                f"surface voxel coords out of [0, 63] range: min={lo}, max={hi}"
            )

        np.save(output_dir / "surface.npy", surface)

        pose = {
            "rotation": self.rotation.detach().cpu().tolist(),
            "translation": self.translation.detach().cpu().tolist(),
            "scale": self.scale.detach().cpu().tolist(),
            "intrinsics": self.intrinsics.detach().cpu().tolist(),
            "downsample_factor": int(self.downsample_factor),
            "seed": int(self.seed),
        }
        with (output_dir / "pose.json").open("w") as f:
            json.dump(pose, f, indent=2)

        if self.pointmap_unnorm is not None:
            np.save(
                output_dir / "pointmap_unnorm.npy",
                self.pointmap_unnorm.detach().cpu().to(torch.float32).numpy(),
            )

    @classmethod
    def load(cls, input_dir: Path) -> "VoxelOutput":
        input_dir = Path(input_dir)

        surface = np.load(input_dir / "surface.npy")
        n = surface.shape[0]
        # Re-add batch column (all zeros) and restore int32 dtype expected downstream.
        batch_col = np.zeros((n, 1), dtype=np.int64)
        full = np.concatenate([batch_col, surface], axis=1).astype(np.int32)
        coords = torch.from_numpy(full)

        with (input_dir / "pose.json").open() as f:
            pose = json.load(f)

        rotation = torch.tensor(pose["rotation"], dtype=torch.float32)
        translation = torch.tensor(pose["translation"], dtype=torch.float32)
        scale = torch.tensor(pose["scale"], dtype=torch.float32)
        intrinsics = torch.tensor(pose["intrinsics"], dtype=torch.float32)

        pointmap_path = input_dir / "pointmap_unnorm.npy"
        pointmap_unnorm = None
        if pointmap_path.exists():
            pointmap_unnorm = torch.from_numpy(np.load(pointmap_path)).to(torch.float32)

        return cls(
            coords=coords,
            downsample_factor=int(pose["downsample_factor"]),
            rotation=rotation,
            translation=translation,
            scale=scale,
            intrinsics=intrinsics,
            pointmap_unnorm=pointmap_unnorm,
            seed=int(pose["seed"]),
        )
