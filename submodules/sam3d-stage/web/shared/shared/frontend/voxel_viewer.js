import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MAX_VOXELS = 30000;

/** Three.js viewer for sparse voxel coords.
 *
 * `load(coords, opts)` — single-color view (original API).
 * `loadMulti(layers, opts)` — multiple colored layers stacked in one scene;
 *   used for IoU comparison overlays (TP/FP/FN). Each layer is
 *   `{coords: number[][], color: string, opacity?: number}`.
 *
 * @param {HTMLCanvasElement} canvas
 * @returns {{
 *   load(coords, opts): void,
 *   loadMulti(layers, opts): void,
 *   dispose(): void,
 *   fit(): void,
 *   getCameraQuat(): THREE.Quaternion
 * }}
 */
export function createVoxelViewer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111418);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.position.set(1.4, 1.2, 1.6);
  camera.lookAt(0, 0, 0);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const dir = new THREE.DirectionalLight(0xffffff, 0.85);
  dir.position.set(3, 4, 2);
  scene.add(dir);

  let meshes = [];
  let raf = 0;
  let running = true;

  function resize() {
    const w = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
    const h = canvas.clientHeight || canvas.parentElement.clientHeight || 400;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function tick() {
    if (!running) return;
    raf = requestAnimationFrame(tick);
    resize();
    controls.update();
    renderer.render(scene, camera);
  }
  tick();

  const ro = new ResizeObserver(resize);
  ro.observe(canvas);

  function clearMeshes() {
    for (const m of meshes) {
      scene.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
    meshes = [];
  }

  function downsampleCoords(pts) {
    if (pts.length <= MAX_VOXELS) return pts;
    console.warn(`voxel_viewer: downsampling ${pts.length} -> ${MAX_VOXELS}`);
    const out = [];
    const seen = new Set();
    while (out.length < MAX_VOXELS) {
      const idx = Math.floor(Math.random() * pts.length);
      if (seen.has(idx)) continue;
      seen.add(idx);
      out.push(pts[idx]);
    }
    return out;
  }

  function makeInstancedMesh(coords, color, voxelSize, gridSize, opacity) {
    const pts = downsampleCoords(coords);
    const geom = new THREE.BoxGeometry(voxelSize, voxelSize, voxelSize);
    const mat = new THREE.MeshLambertMaterial({
      color: new THREE.Color(color),
      transparent: opacity != null && opacity < 1,
      opacity: opacity ?? 1.0,
    });
    const im = new THREE.InstancedMesh(geom, mat, pts.length);
    const m = new THREE.Matrix4();
    const offset = 0.5;
    for (let i = 0; i < pts.length; i++) {
      const [x, y, z] = pts[i];
      const px = (x + offset) / gridSize - 0.5;
      const py = (y + offset) / gridSize - 0.5;
      const pz = (z + offset) / gridSize - 0.5;
      m.makeTranslation(px, py, pz);
      im.setMatrixAt(i, m);
    }
    im.instanceMatrix.needsUpdate = true;
    return im;
  }

  function load(coords, opts = {}) {
    clearMeshes();
    if (!coords || !coords.length) return;
    const gridSize = opts.gridSize || 64;
    const voxelSize = opts.voxelSize != null ? opts.voxelSize : 1 / gridSize;
    const im = makeInstancedMesh(coords, opts.color || "#88ccff", voxelSize, gridSize, 1.0);
    scene.add(im);
    meshes.push(im);
    fit();
  }

  /** Render multiple colored voxel layers in one scene.
   * @param {Array<{coords: number[][], color: string, opacity?: number}>} layers
   * @param {{gridSize?: number, voxelSize?: number}} opts
   */
  function loadMulti(layers, opts = {}) {
    clearMeshes();
    if (!layers || !layers.length) return;
    const gridSize = opts.gridSize || 64;
    const voxelSize = opts.voxelSize != null ? opts.voxelSize : 1 / gridSize;
    for (const layer of layers) {
      if (!layer.coords || !layer.coords.length) continue;
      const im = makeInstancedMesh(layer.coords, layer.color, voxelSize, gridSize, layer.opacity ?? 1.0);
      scene.add(im);
      meshes.push(im);
    }
    fit();
  }

  function fit() {
    camera.position.set(1.4, 1.2, 1.6);
    controls.target.set(0, 0, 0);
    controls.update();
  }

  function dispose() {
    running = false;
    cancelAnimationFrame(raf);
    ro.disconnect();
    clearMeshes();
    controls.dispose();
    renderer.dispose();
  }

  function getCameraQuat() {
    return camera.quaternion;
  }

  return { load, loadMulti, dispose, fit, getCameraQuat };
}
