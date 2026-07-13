# surface_voxel

A small Python library that wraps the **Stage A** (sparse-structure) portion of
[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) into a clean
importable API plus a CLI. Given an RGB image and a binary mask, it returns the
predicted surface voxel coordinates (in a `64^3` grid) and the object's
camera-frame pose (rotation, translation, scale) and camera intrinsics.

It stops after sparse-structure generation, so it skips the heavy SLAT / mesh /
Gaussian-splat decoding stages.

## Install

```bash
pip install -e .
```

This assumes you already have `sam3d_objects` installed in the same Python env
(`pip install -e .` from the sam-3d-objects repo) and that its model checkpoints
are available locally — see `checkpoints/hf/pipeline.yaml` in that repo.

## CLI

```bash
python -m surface_voxel \
    --image  path/to/image.png \
    --mask   path/to/mask.png \
    --output-dir path/to/out \
    --config checkpoints/hf/pipeline.yaml \
    --seed 42
```

Or via the installed entry point: `surface-voxel --image ... --mask ... -o ...`.

Pass `--no-layout-aux` to skip writing `pointmap_unnorm.npy` (smaller output).

## Python API

```python
import numpy as np
from PIL import Image
from surface_voxel import SurfaceVoxelPipeline, VoxelOutput

image = np.array(Image.open("image.png").convert("RGB"))
mask  = np.array(Image.open("mask.png"))  > 127

pipe = SurfaceVoxelPipeline(config_path="checkpoints/hf/pipeline.yaml")
out: VoxelOutput = pipe(image, mask, seed=42)
out.save("out/")
pipe.unload()

# later:
reloaded = VoxelOutput.load("out/")
```

## Output format

After `out.save(dir)` you get:

| File                  | Shape / Type           | Notes |
| --------------------- | ---------------------- | ----- |
| `surface.npy`         | `(N, 3)` `int64`       | Voxel coordinates `[x, y, z]` in `[0, 63]`. Batch column dropped. |
| `pose.json`           | JSON                   | Keys: `rotation` `(1,1,4)`, `translation` `(1,3)`, `scale` `(1,3)`, `intrinsics` `(3,3)`, `downsample_factor` int, `seed` int. Stored as nested lists. |
| `pointmap_unnorm.npy` | `(518, 518, 3)` `f32`  | Optional. The model's internal un-normalized pointmap; useful for layout post-optimization. |

`VoxelOutput.load(dir)` re-adds the zero batch column and restores the
`coords: torch.IntTensor (N, 4)` form expected downstream.
