# texture

Stage B wrapper around SAM 3D Objects: takes voxel coords (Stage A output) plus the original RGB image + mask, and produces a Gaussian splat (`.ply`) and optional mesh (`.glb`).

## Install

```bash
pip install -e .
```

Requires the `sam3d_objects` package and its checkpoints (`pipeline.yaml`) on `PYTHONPATH` / disk.

## CLI

```bash
python -m texture \
    --voxel-dir path/to/stage_a_output \
    --image scene.png \
    --mask object_mask.png \
    -o stage_b_output/ \
    --config ./checkpoints/hf/pipeline.yaml
```

`--voxel-dir` must contain `surface.npy` ((N, 3) int64, values in [0, 63]) and `pose.json`. These are the artifacts written by the sister `surface_voxel` library, but they can also be hand-crafted - this library reads them directly and does not import `surface_voxel`.

Outputs:

- `splat.ply` - Gaussian splat (always written when `gaussian` is in `--formats`)
- `mesh.glb` - mesh export via trimesh (when `mesh` is in `--formats`)

Optional flags: `--seed` (override; pose.json seed is used otherwise), `--formats gaussian mesh gaussian_4`, `--with-layout-postprocess`.

## Library

```python
import numpy as np
from PIL import Image
from texture import TexturePipeline

pipeline = TexturePipeline(config_path="./checkpoints/hf/pipeline.yaml")
image = np.array(Image.open("scene.png").convert("RGB"))
mask = np.array(Image.open("object_mask.png")) > 127

out = pipeline("stage_a_output/", image, mask)
out.save("stage_b_output/")
print(out.num_gaussians, out.ply_path, out.glb_path)

pipeline.unload()
```

## Notes

- **Seed coupling**: when this stage runs split from Stage A, it applies `torch.manual_seed(seed + 1)` before `sample_slat`. This intentionally diverges from a monolithic in-process run (which uses one seed call across both stages) to give Stage B an independent RNG starting state.
- **VRAM**: `instantiate(config)` still loads the full pipeline (including Stage A models). Pruning is a TODO. `load_gs4_decoder=False` (default) skips the 4x decoder (~163MB).
- **Layout post-optimization**: `--with-layout-postprocess` requires a pointmap, which the `slat_preprocessor` does not emit. Run post-optim from Stage A unless you extend this pipeline to plumb `pointmap_unnorm.npy` through.
