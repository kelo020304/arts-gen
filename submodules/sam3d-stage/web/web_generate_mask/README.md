# web_generate_mask

FastAPI web app for **interactive box-prompted segmentation** via SAM 3.
The user uploads an image, the server runs the SAM 3 image backbone once and
caches the embedding in an in-memory session, then each drag-released box on
the frontend canvas hits `/api/sessions/{sid}/predict` for near-instant mask
refinement. Clicking "Save mask" writes `mask.png` + `prompt.json` under a
SHA-keyed job directory. The frontend is a single-file SPA using three stacked
canvases (image / mask / box) and the shared `setupUpload` + `setupStatus` ESM
helpers.

## Install

```bash
pip install -e web/shared
pip install -e generate_mask
pip install -e web/web_generate_mask
```

## Run

```bash
./scripts/run_server.sh
```

Then open <http://localhost:8003/>.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SAM3_CKPT_PATH` | *(required)* | Path to `sam3.pt`. |
| `WEB_DATA_DIR` | `./data` | Where SHA-keyed job dirs (`<sha>/input/`, `<sha>/output/`) are written. |
| `PORT` | `8003` | Uvicorn bind port. |
| `http_proxy`/`https_proxy`/`no_proxy` | optional | Forwarded for outbound HTTP. |

## Endpoints

- `GET /` — frontend SPA
- `GET /api/health` — status + GPU info + active session count
- `POST /api/upload` — multipart `image`; embeds image, returns `{session_id, image_url, width, height}`
- `GET /api/sessions/{sid}/image` — the resized PNG fed to SAM 3
- `POST /api/sessions/{sid}/predict` — JSON `{cx, cy, w, h}`; returns `{mask_png_base64, score}` (data-URL for direct `<img>.src`)
- `POST /api/sessions/{sid}/save` — JSON `{cx, cy, w, h}`; persists to disk, returns `{sha, files, score}`
- `DELETE /api/sessions/{sid}` — drop a session
- `GET /api/jobs/{sha}/{filename}` — serves cached outputs

## Box format

Boxes are **normalized** to the resized image: `cx`, `cy`, `w`, `h` are all in
`[0, 1]`, center-and-size (not corner-corner). The frontend computes them from
pixel drag coordinates after dividing by the canvas's `imgW`/`imgH`.
