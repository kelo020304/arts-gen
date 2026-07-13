# web_surface_voxel

FastAPI web app for **Stage A** of the arts-recon pipeline: upload an RGB image
and a binary mask, run the SAM 3D Objects sparse-structure stage, and view the
resulting 64³ surface voxel grid in the browser. Outputs (`surface.npy`,
`pose.json`, optional `pointmap_unnorm.npy`) are cached on disk by a SHA-256 of
the inputs so re-running the same image/mask/seed is instant.

## Run

```bash
pip install -e ../shared
pip install -e ../../generate_surface_voxel
pip install -e .
./scripts/run_server.sh
```

Then open <http://localhost:8001/>.

## Environment variables

| Name | Required | Default | Purpose |
|---|---|---|---|
| `SAM3D_CONFIG_PATH` | yes | — | Path to `pipeline.yaml` for SAM 3D Objects checkpoints. |
| `WEB_DATA_DIR` | no | `./data` | Per-job cache root: `<sha>/{input,output}/`. |
| `PORT` | no | `8001` | uvicorn port. |
| `http_proxy` / `https_proxy` | no | `http://127.0.0.1:10808` | For first-time HF downloads (DINO etc.). |

## Endpoints

- `GET /` — frontend SPA.
- `GET /api/health` — status + GPU info + `{stage, model_loaded, config_path}`.
- `POST /api/run` — multipart: `image`, `mask`, `seed` (int), `keep_layout_aux` (bool). Returns `{sha, files, coords_count, coords_preview_url, pose}`.
- `GET /api/jobs/{sha}/{filename}` — serves cached output (`surface.npy`, `pose.json`, `pointmap_unnorm.npy`, `coords_preview.json`).
