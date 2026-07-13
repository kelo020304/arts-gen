# sam3d-stages

Staged **single-image → 3D** reconstruction. A thin, decomposed wrapper around
Meta's **SAM 3** (segmentation) and **SAM 3D Objects** (3D reconstruction):
the monolithic image→3D inference is split into independent CLI stages, each
with an optional web viewer, so the intermediate **surface voxel** can be
produced, inspected and decoded separately.

## Pipeline

```
image + box ──[mask]──▶ mask
image + mask ──[surface-voxel : Stage A]──▶ surface voxels (surface.npy + pose.json)
voxels + image + mask ──[texture : Stage B]──▶ Gaussian splat (.ply) + mesh (.obj/.glb)
```

## Modules

| Dir | CLI | Function | Backend |
|-----|-----|----------|---------|
| `generate_mask/`          | `mask`          | box-prompted segmentation | SAM 3 |
| `generate_surface_voxel/` | `surface-voxel` | Stage A: image+mask → surface voxel coords | SAM 3D Objects |
| `generate_texture/`       | `texture`       | Stage B: voxels+image+mask → Gaussian splat + mesh | SAM 3D Objects |
| `web/web_generate_mask/`  | —               | web UI for mask | |
| `web/web_surface_voxel/`  | —               | web UI for Stage A | |
| `web/web_texture/`        | —               | web UI for Stage B | |
| `web/shared/`             | —               | shared web utilities | |

## Setup

The model repos and their checkpoints are **not** tracked here (large; upstream
has its own git). Recreate them under `submodules/`:

```bash
mkdir -p submodules && cd submodules
git clone git@github.com:facebookresearch/sam3.git              # tested @ 11dec29
git clone git@github.com:facebookresearch/sam-3d-objects.git    # tested @ 81a8237
cd ..
# SAM 3D Objects conda env is renamed so it does not clash:
sed -i 's/^name: sam3d-objects$/name: sam3d-objects-process/' submodules/sam-3d-objects/environments/default.yml
```

Conda envs (per each upstream repo's setup docs):
- `sam3` — segmentation (generate_mask)
- `sam3d-objects-process` — Stage A + Stage B (surface_voxel, texture)

Download SAM 3D Objects checkpoints into
`submodules/sam-3d-objects/checkpoints/hf/` per its README.

Install the stages (editable):
```bash
pip install -e generate_mask generate_surface_voxel generate_texture
```

## Usage

```bash
# Stage 0 — mask (SAM 3)
mask --help

# Stage A — image + mask -> surface voxels
surface-voxel --image img.png --mask mask.png -o voxel_dir/

# Stage B — voxels + image + mask -> gaussian splat + mesh
texture --voxel-dir voxel_dir/ --image img.png --mask mask.png \
        -o out/ --formats gaussian mesh
```

## Notes

- `submodules/`, checkpoints/weights and `web/*/data/` (per-session runtime
  outputs) are git-ignored; only the stage source + web front/back ends are tracked.
