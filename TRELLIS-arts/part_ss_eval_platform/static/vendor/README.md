# Vendored third-party libraries

Offline, no build system. Served as static files by the platform's stdlib
http.server (`_send_file`, `Cache-Control: no-store`).

## Three.js r160 (0.160.0) — MIT
- `three.module.js`     — source: https://unpkg.com/three@0.160.0/build/three.module.js
- `GLTFLoader.js`       — source: https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js
- `OrbitControls.js`    — source: https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js
- License: MIT — see `LICENSE-three` (copied from https://unpkg.com/three@0.160.0/LICENSE).
- Local modification: in `GLTFLoader.js` and `OrbitControls.js` the bare
  `from 'three'` import was rewritten to `from './three.module.js'` so they
  resolve without an importmap/bundler.

## GaussianRenderer — first-party (this repo)
`../viewer3d.js` renders gaussian `.ply` files as a Three.js `Points` cloud
(gaussian centers + colors), NOT anisotropic splatting. It sits behind the
isolated `{load,clear,dispose}` renderer interface, so it can later be replaced
with a full WebGL splat shader (e.g. MIT-licensed antimatter15/splat) without
touching the voxel/mesh renderers. No third-party GS code is vendored here.
