from __future__ import annotations
from pathlib import Path
import numpy as np

VOXEL_NPZ = "voxel.npz"
VOXEL_BIN = "voxel.bin"


def save_voxel(run_dir, coords, *, resolution: int = 64, source: str, basename: str = "voxel") -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    c = np.asarray(coords).astype(np.int32).reshape(-1, 3)
    if c.size and (int(c.min()) < 0 or int(c.max()) >= int(resolution)):
        raise ValueError(f"voxel coords 越界 [0,{resolution}): min={int(c.min())} max={int(c.max())}")
    np.savez_compressed(
        run_dir / f"{basename}.npz",
        coords=c,
        resolution=np.int32(resolution),
        coord_frame="canonical_grid",
        source=str(source),
    )
    (run_dir / f"{basename}.bin").write_bytes(voxel_bin_bytes(c))


def voxel_bin_bytes(coords) -> bytes:
    """(N,3) int coords -> the exact little-endian uint16 byte layout of voxel.bin
    (flat x,y,z,...), the only format the web VoxelRenderer can parse."""
    return np.asarray(coords).astype(np.int32).reshape(-1, 3).astype("<u2").tobytes()


def npz_to_bin_bytes(npz_path) -> bytes:
    """Load a voxel.npz (``coords`` key) and return its voxel.bin byte content.
    Lets the server serve voxel.bin on the fly for runs that only wrote voxel.npz."""
    with np.load(Path(npz_path)) as data:
        coords = data["coords"]
    return voxel_bin_bytes(coords)


def load_voxel(npz_path) -> dict:
    """Load a voxel ``.npz`` written by save_voxel or part stage helpers."""
    with np.load(Path(npz_path), allow_pickle=False) as data:
        if "coords" not in data.files:
            raise KeyError(f"{npz_path} expected key 'coords', found {data.files}")
        coords = np.asarray(data["coords"]).astype(np.int32).reshape(-1, 3)
        resolution = int(data["resolution"]) if "resolution" in data.files else 64
        source = str(data["source"].item() if "source" in data.files and getattr(data["source"], "shape", None) == () else data["source"]) if "source" in data.files else ""
    if coords.size and (int(coords.min()) < 0 or int(coords.max()) >= resolution):
        raise ValueError(
            f"voxel coords 越界 [0,{resolution}): min={int(coords.min())} max={int(coords.max())}"
        )
    return {"coords": coords, "resolution": resolution, "source": source}


def load_voxel_bin(path) -> np.ndarray:
    raw = Path(path).read_bytes()
    return np.frombuffer(raw, dtype="<u2").reshape(-1, 3)
