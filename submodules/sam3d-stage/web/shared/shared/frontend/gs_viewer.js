import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader }    from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js";
import { PLYLoader }     from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/PLYLoader.js";

/** Three.js viewer that loads either .ply (GS point cloud preview) or .glb (mesh).
 * @param {HTMLCanvasElement} canvas
 * @returns {{ loadGlb(url): Promise<void>, loadPly(url): Promise<void>,
 *            clear(): void, dispose(): void }}
 */
export function createGsViewer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111418);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1000);
  camera.position.set(1.4, 1.2, 1.6);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const dir = new THREE.DirectionalLight(0xffffff, 0.85);
  dir.position.set(3, 4, 2);
  scene.add(dir);

  let content = null;
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

  function clear() {
    if (!content) return;
    scene.remove(content);
    content.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const m of mats) m.dispose();
      }
    });
    content = null;
  }

  function frame(object) {
    const box = new THREE.Box3().setFromObject(object);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
    object.position.sub(center);

    const dist = maxDim * 2.0;
    camera.position.set(dist, dist * 0.85, dist);
    camera.near = Math.max(maxDim / 100, 0.001);
    camera.far  = dist * 100;
    camera.updateProjectionMatrix();
    controls.target.set(0, 0, 0);
    controls.update();
  }

  function loadGlb(url) {
    clear();
    const loader = new GLTFLoader();
    return new Promise((resolve, reject) => {
      loader.load(
        url,
        (gltf) => {
          content = gltf.scene;
          scene.add(content);
          frame(content);
          resolve();
        },
        undefined,
        (err) => reject(err),
      );
    });
  }

  function loadPly(url) {
    clear();
    const loader = new PLYLoader();
    return new Promise((resolve, reject) => {
      loader.load(
        url,
        (geometry) => {
          const attrs = geometry.attributes;
          let mat;
          if (attrs.f_dc_0 && attrs.f_dc_1 && attrs.f_dc_2) {
            const n = attrs.position.count;
            const colors = new Float32Array(n * 3);
            const r = attrs.f_dc_0.array;
            const g = attrs.f_dc_1.array;
            const b = attrs.f_dc_2.array;
            const SH_C0 = 0.28209479177387814;
            for (let i = 0; i < n; i++) {
              colors[i * 3 + 0] = clamp01(0.5 + SH_C0 * r[i]);
              colors[i * 3 + 1] = clamp01(0.5 + SH_C0 * g[i]);
              colors[i * 3 + 2] = clamp01(0.5 + SH_C0 * b[i]);
            }
            geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
            mat = new THREE.PointsMaterial({ size: 0.005, vertexColors: true, sizeAttenuation: true });
          } else {
            mat = new THREE.PointsMaterial({ size: 0.005, color: 0xcccccc, sizeAttenuation: true });
          }
          content = new THREE.Points(geometry, mat);
          scene.add(content);
          frame(content);
          resolve();
        },
        undefined,
        (err) => reject(err),
      );
    });
  }

  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

  function dispose() {
    running = false;
    cancelAnimationFrame(raf);
    ro.disconnect();
    clear();
    controls.dispose();
    renderer.dispose();
  }

  function getCameraQuat() {
    return camera.quaternion;
  }

  return { loadGlb, loadPly, clear, dispose, getCameraQuat };
}
