# web_texture

FastAPI web app for **Stage B** of the arts-recon pipeline: takes the voxel
artifacts from Stage A (`surface.npy` + `pose.json`, optionally
`pointmap_unnorm.npy`) plus the original RGB image and object mask, and
produces a Gaussian splat (`splat.ply`) and a textured mesh (`mesh.glb`) via
the `texture.TexturePipeline`. The frontend is plain ES6 + Three.js served
from the sister `arts_recon_web_shared` package; the viewer toggles between
the splat preview and the mesh GLB.

## Install

```bash
pip install -e web/shared
pip install -e generate_texture
pip install -e web/web_texture
```

## Run

```bash
./scripts/run_server.sh
```

Then open <http://localhost:8002/>.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SAM3D_CONFIG_PATH` | *(required)* | Path to `pipeline.yaml` checkpoint config. |
| `WEB_DATA_DIR` | `./data` | Where SHA-keyed job dirs (`<sha>/input/`, `<sha>/output/`) are written. |
| `PORT` | `8002` | Uvicorn bind port. |
| `http_proxy`/`https_proxy`/`no_proxy` | optional | Forwarded to the model process for any outbound HTTP needs. |

## Endpoints

- `GET /` — frontend
- `GET /api/health` — status + GPU info (from `shared.backend`)
- `POST /api/run` — multipart upload, returns `{sha, files, num_gaussians, formats}`
- `GET /api/jobs/{sha}/{filename}` — serves cached outputs
