# arts_recon_web_shared

Reusable building blocks for the two independent FastAPI web apps
(`web_surface_voxel`, `web_texture`). They share **code** through this package
but keep their own data dirs, ports, and runtime state. Stack: **FastAPI +
vanilla ES6 + Three.js** (no React/Vue/Gradio, no transpilation). Frontend JS
modules are served by the backend as static assets and imported via
`<script type="module">`.

Install (editable, from the repo root):

```
pip install -e web/shared
```

Then mount the shared statics from your app and add your own `/api/run` route:

```python
from pathlib import Path
from shared.backend import create_app

app = create_app(
    title="surface-voxel demo",
    data_dir=Path("data/jobs"),
    app_frontend_dir=Path(__file__).parent / "frontend",
    health_extras=lambda: {"models_loaded": MODELS_READY},
)
```

## Public API (locked — Wave 2 consumes this)

### `shared.backend.cache`

| Symbol | Signature | Behavior |
|---|---|---|
| `compute_sha` | `(parts: dict[str, bytes \| str \| int \| float]) -> str` | SHA-256 over canonicalized dict (sorted keys; `bytes` raw, scalars `repr().utf-8`). Returns hex digest truncated to 16 chars. |
| `JobDirs` | `JobDirs(base_dir: Path, sha: str)` | Cache layout `<base>/<sha>/{input,output}/`. Properties: `root`, `input_dir`, `output_dir`. Methods: `exists()` (output non-empty), `ensure_input()`, `ensure_output()`. |

### `shared.backend.upload`

| Symbol | Signature | Behavior |
|---|---|---|
| `save_upload` | `async (file: UploadFile, dest: Path) -> Path` | Streams the upload to `dest` in 1 MB chunks. Raises HTTP 413 if total > 50 MB. |
| `read_image_rgb` | `(path: Path) -> np.ndarray` | PNG/JPG/WebP -> `(H, W, 3)` uint8 RGB; alpha dropped. |
| `read_mask_bool` | `(path: Path) -> np.ndarray` | PNG -> `(H, W)` bool. Single-channel: threshold `> 127`. RGB(A): any non-zero channel. |
| `read_npy` | `(path: Path) -> np.ndarray` | `np.load` passthrough. Raises HTTP 400 with a clear message on failure. |

### `shared.backend.server_base`

```python
create_app(
    title: str,
    *,
    data_dir: Path,
    health_extras: Callable[[], dict] | None = None,
    cors_origins: list[str] | None = None,
    app_frontend_dir: Path | None = None,
) -> FastAPI
```

Pre-wired:

- CORS (default `allow_origins=["*"]`; override via `cors_origins`).
- `GET /api/health` -> `{status, device, vram_used_mb, vram_total_mb, **health_extras()}`.
  GPU info via lazy `torch.cuda.mem_get_info(0)`; reports `device="cpu"` if
  torch missing or CUDA unavailable.
- `GET /api/jobs/{sha}/{filename}` -> serves `data_dir/<sha>/output/<filename>`.
  Filename validated against `..`, `/`, `\`, and leading `.`.
- Static mounts:
  - `/static`     -> this package's `shared/frontend/` (located via `importlib.resources`).
  - `/static_app` -> caller's `app_frontend_dir` (only mounted if it exists).

Consumers add their own `POST /api/run` route.

### Frontend ES6 modules (served at `/static/<file>`)

All modules are pure ES6, no build step. Three.js comes from
`https://cdn.jsdelivr.net/npm/three@0.160.0/` (locked version) with addons:
`OrbitControls`, `GLTFLoader`, `PLYLoader`.

#### `upload.js`

```js
import { setupUpload } from "/static/upload.js";

const ui = setupUpload(container, [
  { name: "image", accept: "image/*", label: "RGB image" },
  { name: "mask",  accept: "image/*", label: "Mask PNG"  },
]);
ui.getFormData();   // FormData with all slot files
ui.isComplete();    // true iff every slot has a file
ui.onChange(cb);    // fires on every set/clear
ui.reset();
```

Each slot renders as a drag-drop + click card. When a file is dropped, the
card collapses into a row with thumbnail (for `image/*`), filename, byte size,
and a `clear` button.

#### `progress.js`

```js
import { setupStatus } from "/static/progress.js";
const status = setupStatus(container);
status.set("running", "encoding...");   // level in 'info'|'ok'|'warn'|'error'|'running'
status.reset();
```

Level maps to CSS class `.status-<level>`; `running` shows a pulsing dot.

#### `voxel_viewer.js`

```js
import { createVoxelViewer } from "/static/voxel_viewer.js";
const viewer = createVoxelViewer(canvas);
viewer.load(coords, { gridSize: 64, color: "#88ccff", voxelSize: null });
viewer.fit();
viewer.dispose();
```

- `coords`: `number[][]` of shape `[N][3]`, ints in `[0, gridSize)`.
- `voxelSize` defaults to `1/gridSize` (whole grid spans ~1 unit cube,
  centered at the origin).
- Uses `THREE.InstancedMesh`. Caps at 30000 voxels (random downsample with
  `console.warn`) for performance.
- OrbitControls, ambient + directional lights, dark background.

#### `gs_viewer.js`

```js
import { createGsViewer } from "/static/gs_viewer.js";
const v = createGsViewer(canvas);
await v.loadGlb("/api/jobs/<sha>/mesh.glb");
await v.loadPly("/api/jobs/<sha>/splat.ply");
v.clear();
v.dispose();
```

- `loadGlb`: `GLTFLoader`, scene centered on bounding box.
- `loadPly`: `PLYLoader`, rendered as `THREE.Points`. If the PLY has
  `f_dc_0/1/2` (Gaussian-splat SH DC), each point gets a per-vertex color via
  `0.5 + SH_C0 * f_dc_i`; otherwise uniform light gray. Preview only; full-
  fidelity splatting is out of scope.

#### `styles.css`

Two-column desktop layout (`.layout` -> 380px sidebar + viewer). Upload-slot
drag/dragover/populated states, status-line colors, primary button, dark
viewer pane. Import once per page:

```html
<link rel="stylesheet" href="/static/styles.css">
```

## Notes

- Shared/ is **library code only** — no example apps, no tests. Wave 2 apps
  own their own entry points and `POST /api/run` handlers.
- `torch` is **not** a hard dep. `/api/health` imports it lazily and falls
  back to `device="cpu"` if missing.
- No global singletons: each consuming app calls `create_app(...)` with its
  own `data_dir` and `health_extras`.
