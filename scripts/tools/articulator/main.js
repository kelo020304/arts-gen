// Generic articulated-device annotation tool — v2 schema (parts[] + joints[]
// + external_meshes[]). Replaces the earbud-specific v1 editor. See
// docs/superpowers/specs/2026-05-08-generic-articulator-design.md for the
// full design.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const SCHEMA_VERSION = 3;
// We emit v3, but accept v2 on load (sites are optional, so v2 docs parse
// cleanly under the v3 reader). Matches schema.py SUPPORTED_VERSIONS.
const SUPPORTED_LABEL_VERSIONS = new Set([2, 3]);
const SITE_KINDS = ['screen', 'button', 'camera', 'handle', 'custom'];

// Per-part colors (assigned in order parts are added). Last entry is the
// fallback for the 9th+ part. ``null`` color = unlabelled cluster (gray).
const PART_PALETTE = [
  0x55aa55, 0x4488ff, 0xff5555, 0xff8800,
  0x9966cc, 0xddcc44, 0x66ccaa, 0xcc66aa,
  0xaaaaaa,
];
const UNLABELLED_COLOR = 0x666666;

const JOINT_TYPES = ['revolute', 'prismatic', 'fixed', 'free'];
const PHYSICS_KINDS = ['kinematic', 'dynamic'];
const COLLISION_APPROXES = ['sdf', 'convexHull', 'convexDecomposition', 'none'];

// ---- State (mirrors the v2 labels.json structure exactly) ------------------

const state = {
  glbName: null,
  device: 'Device',
  source_glb: '',
  physical_dims_mm: { x: 50, y: 50, z: 25 },

  // Schema: parts/joints/external_meshes are the fields we serialize.
  parts: [],            // [{id, clusters[], physics, collision, scale_xyz, mass, filtered_pairs?}, ...]
  joints: [],           // [{id, parent, child, type, ...}, ...]
  external_meshes: [],  // [{attach_to, glb, transform, cluster_filter?}, ...]
  split_clusters: {},   // {<split_name>: {parent, aabb_min[3], aabb_max[3]}}  — see box-select


  // UI-only state (not serialized)
  clusters: [],            // [{id, name, mesh, verts}, ...]  loaded GLB clusters
  selectedClusterId: null,
  selectedJointId: null,   // joint whose preview slider is active
  baseLidPose: new Map(),  // partId -> original geom for slider preview
  hingeMarkers: { p0: null, p1: null },
};
// Expose state on window so F12 console can inspect (e.g. _state.split_clusters).
window._state = state;

// ---- Three.js scene --------------------------------------------------------

const wrap = document.getElementById('canvas-wrap');
const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'low-power' });
renderer.setPixelRatio(1);
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222);

const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 100);
camera.position.set(2.5, 2.5, 2.5);
camera.lookAt(0, 0, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dir = new THREE.DirectionalLight(0xffffff, 0.8); dir.position.set(2, 3, 2); scene.add(dir);
const dir2 = new THREE.DirectionalLight(0xffffff, 0.4); dir2.position.set(-2, -1, -2); scene.add(dir2);

const grid = new THREE.GridHelper(2, 20, 0x444444, 0x333333);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
scene.add(new THREE.AxesHelper(0.5));

const meshGroup = new THREE.Group();
scene.add(meshGroup);

const hingeLine = new THREE.Line(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0xffff00 })
);
hingeLine.visible = false;
scene.add(hingeLine);

const transformCtrl = new TransformControls(camera, renderer.domElement);
transformCtrl.setSize(0.8);
scene.add(transformCtrl);
transformCtrl.addEventListener('dragging-changed', (e) => {
  controls.enabled = !e.value;
  // On drag END, persist the joint update + rebuild the joint card so other
  // computed UI (tree summary, axis line) stays consistent.
  if (!e.value) {
    const obj = transformCtrl.object;
    if (obj && obj.userData.markerKind) {
      saveToLocalStorage();
      rebuildTreeSummary();
    }
  }
});
transformCtrl.addEventListener('objectChange', () => {
  const obj = transformCtrl.object;
  if (!obj) return;
  if (obj.userData.markerKind) {
    // Hinge p0/p1 sphere being dragged in 3D. Marker position is in WORLD
    // (= LOCAL × _displayScale), but j.axis_p0/p1 are stored in LOCAL so
    // they match the GLB intrinsic frame and Blender's vertex.co.
    const j = state.selectedJointId ? state.joints.find((x) => x.id === state.selectedJointId) : null;
    if (!j) return;
    const fieldKey = (j.type === 'prismatic')
      ? (obj.userData.markerKind === 'p0' ? 'axis_origin' : 'axis_dir')
      : (obj.userData.markerKind === 'p0' ? 'axis_p0' : 'axis_p1');
    const s = _displayScale();
    j[fieldKey] = [obj.position.x / s, obj.position.y / s, obj.position.z / s];
    syncJointCardInputs(j);
    updateHingeLine();
    return;
  }
  if (obj.userData.extMeshIdx != null) pullExtMeshTransformFromGroup(obj);
});

window.addEventListener('keydown', (e) => {
  if (!transformCtrl.object) return;
  if (e.key === 't' || e.key === 'T') transformCtrl.setMode('translate');
  else if (e.key === 'r' || e.key === 'R') transformCtrl.setMode('rotate');
  else if (e.key === 's' || e.key === 'S') transformCtrl.setMode('scale');
  else if (e.key === 'Escape') transformCtrl.detach();
});

state.hingeMarkers.p0 = makeMarker(0xff5555, 'p0');
state.hingeMarkers.p1 = makeMarker(0x55ff55, 'p1');
state.hingeMarkers.p0.visible = false;
state.hingeMarkers.p1.visible = false;
scene.add(state.hingeMarkers.p0);
scene.add(state.hingeMarkers.p1);

function makeMarker(color, kind) {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(0.015, 16, 12),
    new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false })
  );
  m.renderOrder = 999;            // draw on top so it's always grabbable
  m.userData.markerKind = kind;   // 'p0' | 'p1' — used by the gizmo handler
  return m;
}

// ---- dim visualization (wireframe of physical_dims_mm + R/G/B axis labels) ----
//
// User edits dim X/Y/Z in mm but they need a 3-D reference to know "X
// corresponds to which mesh edge?". This draws a wireframe box around the
// loaded mesh's bbox + colored sprites at the three edges with the current
// dim values. Updates on GLB load and on every dim input change.
const dimGroup = new THREE.Group();
const dimWireframe = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
  new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.45 }),
);
dimGroup.add(dimWireframe);

function _makeDimLabelSprite() {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 64;
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, depthWrite: false });
  const s = new THREE.Sprite(mat);
  s.scale.set(0.18, 0.045, 1);
  s.userData.canvas = canvas;
  s.userData.tex = tex;
  return s;
}
function _setSpriteText(sprite, text, color) {
  const c = sprite.userData.canvas;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = 'rgba(20,20,20,0.85)';
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.font = 'bold 30px monospace';
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, c.width / 2, c.height / 2);
  sprite.userData.tex.needsUpdate = true;
}
const dimLabelX = _makeDimLabelSprite();
const dimLabelY = _makeDimLabelSprite();
const dimLabelZ = _makeDimLabelSprite();
dimGroup.add(dimLabelX, dimLabelY, dimLabelZ);
dimGroup.visible = false;
scene.add(dimGroup);

function updateDimViz() {
  if (state.clusters.length === 0) { dimGroup.visible = false; return; }
  const box = new THREE.Box3().setFromObject(meshGroup);
  if (box.isEmpty()) { dimGroup.visible = false; return; }
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  // Wireframe: 1×1×1 unit box scaled to bbox size, positioned at center.
  dimWireframe.position.copy(center);
  dimWireframe.scale.copy(size);
  // Labels along the three principal edges of the bbox (slightly outside so
  // they don't z-fight with the wireframe).
  const off = Math.max(size.x, size.y, size.z) * 0.06;
  dimLabelX.position.set(center.x, box.min.y - off, box.max.z + off);
  dimLabelY.position.set(box.max.x + off, center.y, box.max.z + off);
  dimLabelZ.position.set(box.max.x + off, box.min.y - off, center.z);
  _setSpriteText(dimLabelX, `X ${state.physical_dims_mm.x.toFixed(1)} mm`, '#ff8888');
  _setSpriteText(dimLabelY, `Y ${state.physical_dims_mm.y.toFixed(1)} mm`, '#88ff88');
  _setSpriteText(dimLabelZ, `Z ${state.physical_dims_mm.z.toFixed(1)} mm`, '#88aaff');
  dimGroup.visible = true;
}

function resize() {
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// ---- Render loop -----------------------------------------------------------

const angleSlider = document.getElementById('lid-angle');
const angleNum = document.getElementById('lid-angle-num');
angleSlider.addEventListener('input', () => { angleNum.value = angleSlider.value; });
angleNum.addEventListener('input', () => { angleSlider.value = angleNum.value; });

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  applyPreviewAngle();
  renderer.render(scene, camera);
}
animate();

// ---- Logging ---------------------------------------------------------------

function log(msg) {
  const el = document.getElementById('log');
  el.innerText = msg + '\n' + el.innerText;
  console.log(msg);
}

// ---- File loading ----------------------------------------------------------

document.getElementById('file-input').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  state.glbName = f.name;
  state.source_glb = f.name;
  const reader = new FileReader();
  reader.onload = (ev) => loadGLB(ev.target.result);
  reader.readAsArrayBuffer(f);
});

const DEFAULT_GLB_CANDIDATES = [
  './data/clean.glb',
  '../../outputs/xiaomi_buds6_seed3d/clean.glb',
  '/outputs/xiaomi_buds6_seed3d/clean.glb',
];

async function loadFromUrl(url, displayName) {
  log(`fetching ${url}`);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
  const buf = await resp.arrayBuffer();
  state.glbName = displayName || url.split('/').pop();
  state.source_glb = url;
  loadGLB(buf);
}

// ---- Tab switcher ----------------------------------------------------------

document.querySelectorAll('.tab').forEach((t) => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((x) => x.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach((x) => x.classList.remove('active'));
    t.classList.add('active');
    const panel = document.querySelector(`.tab-panel[data-tab-panel="${t.dataset.tab}"]`);
    if (panel) panel.classList.add('active');
    // Some panels need re-render on show (cluster table picks up new parts)
    if (t.dataset.tab === 'clusters') rebuildClusterTable();
    if (t.dataset.tab === 'parts') rebuildPartsList();
    if (t.dataset.tab === 'sites') rebuildSitesList();
  });
});

// ---- DBSCAN preprocess (server-side via /api/preprocess) ------------------

function setBanner(state, text) {
  const banner = document.getElementById('preprocess-banner');
  const t = document.getElementById('preprocess-banner-text');
  banner.classList.remove('active', 'error');
  if (state === 'hidden') return;
  if (state === 'error') banner.classList.add('error');
  banner.classList.add('active');
  t.textContent = text;
}

document.getElementById('btn-preprocess').addEventListener('click', async () => {
  const fileEl = document.getElementById('preprocess-input');
  const f = fileEl.files[0];
  const btn = document.getElementById('btn-preprocess');
  const status = document.getElementById('preprocess-status');
  const logEl = document.getElementById('preprocess-log');

  if (!f) {
    setBanner('error', '请先在 ① DBSCAN 预处理 这一列选一个原始 GLB 文件');
    status.textContent = '↑ pick a raw GLB above';
    status.style.color = '#f88';
    return;
  }

  const eps = +document.getElementById('preprocess-eps').value;
  const minVerts = +document.getElementById('preprocess-minverts').value;

  // Big visible banner above the panel content + animated dot status next
  // to the button. Either is enough to tell the user "it's running".
  setBanner('active', `DBSCAN 预处理运行中 (eps=${eps}, min_verts=${minVerts})…  Blender + 点云聚类通常 30–90 s`);
  status.style.color = '#aaa';
  status.textContent = `running (eps=${eps}, min_verts=${minVerts})`;
  btn.disabled = true;
  logEl.style.display = 'none';
  logEl.textContent = '';
  let dots = 0;
  const tick = setInterval(() => {
    dots = (dots + 1) % 4;
    status.textContent = `running (eps=${eps}, min_verts=${minVerts})${'.'.repeat(dots)}`;
  }, 400);
  const t0 = Date.now();

  const url = `/api/preprocess?eps=${encodeURIComponent(eps)}&min_verts=${encodeURIComponent(minVerts)}`;
  let resp, payload;
  try {
    resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: await f.arrayBuffer(),
    });
    const ct = resp.headers.get('content-type') || '';
    if (!resp.ok || !ct.includes('application/json')) {
      const text = (await resp.text()).slice(0, 200);
      throw new Error(
        `HTTP ${resp.status} (${ct || 'no content-type'}). ` +
        `服务器没有 /api/preprocess 端点 — 重启 serve.sh 让它加载新代码. Body: ${text}`
      );
    }
    payload = await resp.json();
  } catch (e) {
    clearInterval(tick); btn.disabled = false;
    setBanner('error', `DBSCAN 失败: ${e.message}`);
    status.textContent = 'failed';
    status.style.color = '#f88';
    log('[preprocess] error: ' + e.message);
    return;
  }
  clearInterval(tick);
  btn.disabled = false;
  const dt = ((Date.now() - t0) / 1000).toFixed(1);

  logEl.style.display = 'block';
  logEl.textContent = payload.log || '';
  if (payload.ok && payload.preprocessed_path) {
    status.textContent = `OK — ${dt}s, auto-loading…`;
    status.style.color = '#8f8';
    log(`[preprocess] ${dt}s — eps=${payload.eps} min_verts=${payload.min_verts}`);
    await loadFromUrl(payload.preprocessed_path + '?t=' + Date.now(), 'preprocessed.glb');
    status.textContent = `OK — ${dt}s, ${state.clusters.length} clusters loaded`;
    setBanner('active', `DBSCAN 完成 (${dt}s) — ${state.clusters.length} clusters 已加载到 viewport`);
    setTimeout(() => setBanner('hidden'), 4000);
  } else {
    status.textContent = `failed after ${dt}s (see log)`;
    status.style.color = '#f88';
    setBanner('error', `DBSCAN 失败 (${dt}s) — 看下方 log`);
    log('[preprocess] failed — see log');
  }
});

// Load a v2 labels.json (file dialog OR server fetch). Replaces in-memory
// state, then triggers full UI rebuild.
function loadLabelsFromObject(data, sourceLabel) {
  if (!SUPPORTED_LABEL_VERSIONS.has(data.version)) {
    log(`[error] labels.json version ${data.version} not supported (need 2 or 3; run migrate_v*.py to upgrade)`);
    return;
  }
  if (data.device) state.device = data.device;
  if (data.physical_dims_mm) state.physical_dims_mm = data.physical_dims_mm;
  state.parts = Array.isArray(data.parts) ? data.parts : [];
  state.joints = Array.isArray(data.joints) ? data.joints : [];
  state.external_meshes = Array.isArray(data.external_meshes) ? data.external_meshes : [];
  state.split_clusters = (data.split_clusters && typeof data.split_clusters === 'object') ? data.split_clusters : {};
  log(`loaded ${state.parts.length} parts / ${state.joints.length} joints / ${Object.keys(state.split_clusters).length} splits from ${sourceLabel}`);
  rebuildAll();
  saveToLocalStorage();
}

document.getElementById('labels-input').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    try { loadLabelsFromObject(JSON.parse(ev.target.result), f.name); }
    catch (err) { log('parse error: ' + err.message); }
  };
  reader.readAsText(f);
});

const DEFAULT_LABELS_CANDIDATES = [
  './data/labels.json',
  '../../outputs/xiaomi_buds6_seed3d/labels.json',
  '/outputs/xiaomi_buds6_seed3d/labels.json',
];

document.getElementById('btn-load-labels-server').addEventListener('click', async () => {
  for (const url of DEFAULT_LABELS_CANDIDATES) {
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      loadLabelsFromObject(data, url);
      return;
    } catch (e) { log('  - ' + url + ' ' + e.message); }
  }
  log('failed to fetch labels.json from default paths');
});

document.getElementById('btn-load-default').addEventListener('click', async () => {
  for (const url of DEFAULT_GLB_CANDIDATES) {
    try { await loadFromUrl(url, 'clean.glb'); return; }
    catch (e) { log('  - ' + url + ' ' + e.message); }
  }
  log('failed to fetch clean.glb from default paths');
});

(async () => {
  const p = new URLSearchParams(location.search).get('glb');
  if (p) {
    try { await loadFromUrl(p, p.split('/').pop()); }
    catch (e) { log('?glb load failed: ' + e.message); }
    return;
  }
  if (localStorage.getItem('articulator:clean.glb')) {
    log('found saved state for clean.glb — auto-loading');
    for (const url of DEFAULT_GLB_CANDIDATES) {
      try { await loadFromUrl(url, 'clean.glb'); return; }
      catch (e) { /* try next */ }
    }
    log('auto-load failed; click "从服务器加载 clean.glb" to retry');
  }
})();

function loadGLB(arrayBuffer) {
  const loader = new GLTFLoader();
  loader.parse(arrayBuffer, '', (gltf) => {
    while (meshGroup.children.length) meshGroup.remove(meshGroup.children[0]);
    state.clusters = [];

    let id = 0;
    gltf.scene.traverse((obj) => {
      if (!obj.isMesh) return;
      const mat = new THREE.MeshLambertMaterial({
        color: UNLABELLED_COLOR, transparent: true, opacity: 0.95, side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(obj.geometry.clone(), mat);
      mesh.name = obj.name || `cluster_${String(id).padStart(2, '0')}`;
      mesh.userData.clusterId = id;
      obj.updateMatrixWorld(true);
      mesh.applyMatrix4(obj.matrixWorld);
      meshGroup.add(mesh);
      state.clusters.push({ id, name: mesh.name, mesh, verts: mesh.geometry.attributes.position.count });
      id += 1;
    });

    log(`loaded ${state.clusters.length} clusters from ${state.glbName}`);
    fitCamera();
    restoreFromLocalStorage();
    pruneOrphansAgainstClusters();
    reapplyPersistedSplits();
    rebuildAll();
    wireAutosave();
  }, (err) => log('parse error: ' + err));
}

function fitCamera() {
  const box = new THREE.Box3().setFromObject(meshGroup);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const r = Math.max(size.x, size.y, size.z) * 1.5;
  camera.position.set(center.x + r, center.y + r, center.z + r);
  controls.target.copy(center);
  controls.update();
}

// ---- localStorage persistence ---------------------------------------------

function storageKey() { return 'articulator:' + (state.glbName || 'unknown'); }

function saveToLocalStorage() {
  if (!state.glbName) return;
  const payload = {
    device: state.device,
    physical_dims_mm: state.physical_dims_mm,
    parts: state.parts,
    joints: state.joints,
    external_meshes: state.external_meshes,
    split_clusters: state.split_clusters,
  };
  try { localStorage.setItem(storageKey(), JSON.stringify(payload)); }
  catch (e) { log('localStorage save failed: ' + e.message); }
}

function restoreFromLocalStorage() {
  const raw = localStorage.getItem(storageKey());
  if (!raw) return;
  try {
    const data = JSON.parse(raw);
    if (data.device) state.device = data.device;
    if (data.physical_dims_mm) state.physical_dims_mm = data.physical_dims_mm;
    if (Array.isArray(data.parts)) state.parts = data.parts;
    if (Array.isArray(data.joints)) state.joints = data.joints;
    if (Array.isArray(data.external_meshes)) state.external_meshes = data.external_meshes;
    if (data.split_clusters && typeof data.split_clusters === 'object') state.split_clusters = data.split_clusters;
    log(`restored ${state.parts.length} parts / ${state.joints.length} joints / ${Object.keys(state.split_clusters).length} splits from localStorage`);
  } catch (e) { log('localStorage restore failed: ' + e.message); }
}

let _autosaveWired = false;
function wireAutosave() {
  if (_autosaveWired) return;
  _autosaveWired = true;
  setInterval(saveToLocalStorage, 1500);
}

// ---- Cluster <-> part assignment ------------------------------------------

function partOfCluster(clusterName) {
  for (const p of state.parts) {
    if (p.clusters.includes(clusterName)) return p;
  }
  return null;
}

function clusterColor(clusterName) {
  const p = partOfCluster(clusterName);
  if (!p) return UNLABELLED_COLOR;
  const idx = state.parts.indexOf(p);
  return PART_PALETTE[Math.min(idx, PART_PALETTE.length - 1)];
}

function assignClusterToPart(clusterName, partId) {
  // Remove from any other part first
  for (const p of state.parts) {
    p.clusters = p.clusters.filter((c) => c !== clusterName);
  }
  if (partId) {
    const p = state.parts.find((x) => x.id === partId);
    if (p && !p.clusters.includes(clusterName)) p.clusters.push(clusterName);
  }
  refreshClusterColors();
  rebuildClusterTable();
  rebuildPartsList();   // cluster-count chips on each part
  saveToLocalStorage();
}

function refreshClusterColors() {
  const labelled = new Set();
  state.parts.forEach((p) => p.clusters.forEach((c) => labelled.add(c)));
  for (const c of state.clusters) {
    const isSelected = state.selectedClusterId === c.id;
    c.mesh.material.color.set(clusterColor(c.name));
    c.mesh.material.opacity = labelled.has(c.name) ? 0.95 : 0.8;
    if (c.mesh.material.emissive) {
      // bright-yellow glow on the selected cluster so it pops out from the
      // surrounding similarly-colored unlabelled mesh.
      c.mesh.material.emissive.set(isSelected ? 0xffee44 : 0x000000);
      c.mesh.material.emissiveIntensity = isSelected ? 0.7 : 0;
    }
    c.mesh.visible = true;
  }
}

function selectCluster(id) {
  state.selectedClusterId = id;
  refreshClusterColors();
  rebuildClusterTable();
}

// ---- Heuristic auto-label (best-effort: by world-X / world-Y position) ----
// Lightweight: for the EARBUD CASE shape, assigns body / lid / earbud_L / earbud_R
// by clustering on (yMax, xCenter). Users can override afterwards.

function clusterCentroid(c) {
  const pos = c.mesh.geometry.attributes.position;
  let sx = 0, sy = 0, sz = 0;
  for (let i = 0; i < pos.count; i++) {
    sx += pos.getX(i); sy += pos.getY(i); sz += pos.getZ(i);
  }
  const w = c.mesh.matrixWorld;
  const v = new THREE.Vector3(sx / pos.count, sy / pos.count, sz / pos.count).applyMatrix4(w);
  return v;
}

function autoLabelClusters() {
  if (state.parts.length === 0) {
    // Synthesise a default earbud-case parts list (works for the existing asset
    // and gives users a known-good starting point).
    state.parts = [
      mkDefaultPart('body', 'kinematic'),
      mkDefaultPart('lid', 'dynamic'),
      mkDefaultPart('earbud_L', 'dynamic'),
      mkDefaultPart('earbud_R', 'dynamic'),
    ];
    state.parts[0].mass = 0.025;
    state.parts[1].mass = 0.010;
    state.parts[2].mass = 0.0044;
    state.parts[3].mass = 0.0044;
  }
  const centroids = state.clusters.map((c) => ({ c, p: clusterCentroid(c) }));
  // Compute scene bbox
  const ys = centroids.map((x) => x.p.y);
  const yMid = (Math.min(...ys) + Math.max(...ys)) / 2;
  for (const { c, p } of centroids) {
    let label;
    if (p.y > yMid) {
      label = 'lid';   // upper half
    } else {
      // lower half: earbuds are wider on x, body is centered
      const xs = centroids.map((x) => x.p.x);
      const xMid = (Math.min(...xs) + Math.max(...xs)) / 2;
      if (Math.abs(p.x - xMid) > (Math.max(...xs) - Math.min(...xs)) * 0.35) {
        label = p.x < xMid ? 'earbud_L' : 'earbud_R';
      } else {
        label = 'body';
      }
    }
    if (state.parts.find((x) => x.id === label)) {
      assignClusterToPart(c.name, label);
    }
  }
  log('heuristic auto-label complete');
}

// ---- Parts UI -------------------------------------------------------------

function mkDefaultPart(id, physics = 'dynamic') {
  return {
    id,
    clusters: [],
    physics,
    collision: { approx: 'sdf', resolution: 128 },
    scale_xyz: [1, 1, 1],
    mass: 0.05,
  };
}

function uniquePartId(base) {
  let n = 1;
  let id = base;
  while (state.parts.find((p) => p.id === id)) { n++; id = `${base}${n}`; }
  return id;
}

function rebuildPartsList() {
  const root = document.getElementById('parts-list');
  root.innerHTML = '';
  state.parts.forEach((part, idx) => {
    const card = document.createElement('div');
    card.className = 'card';

    const swatch = `<span class="swatch" style="background:#${PART_PALETTE[Math.min(idx, PART_PALETTE.length - 1)].toString(16).padStart(6,'0')};display:inline-block;"></span>`;
    card.innerHTML = `
      <div class="card-title">
        ${swatch}
        <input class="id-input" data-part-idx="${idx}" type="text" value="${escapeAttr(part.id)}">
        <span style="color:#666;font-size:10px;">${part.clusters.length} clusters</span>
        <button class="tiny danger" data-rm-part="${idx}" title="delete part">×</button>
      </div>
      <div class="field-grid">
        <label>physics</label>
        <select data-part-physics="${idx}">
          ${PHYSICS_KINDS.map((k) => `<option value="${k}"${part.physics === k ? ' selected' : ''}>${k}</option>`).join('')}
        </select>
        <label>collision</label>
        <div class="row">
          <select data-part-coll-approx="${idx}">
            ${COLLISION_APPROXES.map((k) => `<option value="${k}"${part.collision.approx === k ? ' selected' : ''}>${k}</option>`).join('')}
          </select>
          <span style="font-size:10px;color:#888;">res</span>
          <input type="number" data-part-coll-res="${idx}" value="${part.collision.resolution || 128}" style="width:55px;" ${part.collision.approx === 'sdf' ? '' : 'disabled'}>
        </div>
        <label>scale xyz</label>
        <div class="row">
          <input type="number" data-part-scale="${idx}" data-axis="0" value="${part.scale_xyz[0]}" step="0.01" style="width:55px;">
          <input type="number" data-part-scale="${idx}" data-axis="1" value="${part.scale_xyz[1]}" step="0.01" style="width:55px;">
          <input type="number" data-part-scale="${idx}" data-axis="2" value="${part.scale_xyz[2]}" step="0.01" style="width:55px;">
        </div>
        <label>mass (kg)</label>
        <input type="number" data-part-mass="${idx}" value="${part.mass ?? 0.05}" step="0.001" style="width:80px;">
        <label>碰撞组</label>
        <input type="text" data-part-cgroup="${idx}" value="${escapeAttr(part.collision_group || '')}" placeholder="留空 = 跟所有人碰撞" style="width:140px;" title="同名组的 part 之间不碰撞 (e.g. 'phone' 给 main+fold 双屏)">
      </div>
    `;
    root.appendChild(card);
  });

  // Wire events
  root.querySelectorAll('input.id-input').forEach((el) => {
    el.addEventListener('change', (e) => {
      const i = +e.target.dataset.partIdx;
      const oldId = state.parts[i].id;
      const newId = e.target.value.trim();
      if (!newId || (newId !== oldId && state.parts.find((p) => p.id === newId))) {
        e.target.value = oldId;
        return;
      }
      // Cascade: update any joint / external_mesh / filtered_pairs referencing this id
      state.parts[i].id = newId;
      state.joints.forEach((j) => {
        if (j.parent === oldId) j.parent = newId;
        if (j.child === oldId) j.child = newId;
        if (j.filtered_pairs) j.filtered_pairs = j.filtered_pairs.map((x) => x === oldId ? newId : x);
      });
      state.external_meshes.forEach((em) => { if (em.attach_to === oldId) em.attach_to = newId; });
      state.parts.forEach((p) => {
        if (p.filtered_pairs) p.filtered_pairs = p.filtered_pairs.map((x) => x === oldId ? newId : x);
      });
      rebuildAll();
      saveToLocalStorage();
    });
  });
  root.querySelectorAll('button[data-rm-part]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const i = +btn.dataset.rmPart;
      const removed = state.parts[i].id;
      state.parts.splice(i, 1);
      state.joints = state.joints.filter((j) => j.parent !== removed && j.child !== removed);
      state.external_meshes = state.external_meshes.filter((em) => em.attach_to !== removed);
      rebuildAll();
      saveToLocalStorage();
    });
  });
  root.querySelectorAll('select[data-part-physics]').forEach((el) => {
    el.addEventListener('change', (e) => {
      state.parts[+e.target.dataset.partPhysics].physics = e.target.value;
      saveToLocalStorage();
    });
  });
  root.querySelectorAll('select[data-part-coll-approx]').forEach((el) => {
    el.addEventListener('change', (e) => {
      const i = +e.target.dataset.partCollApprox;
      state.parts[i].collision.approx = e.target.value;
      rebuildPartsList();
      saveToLocalStorage();
    });
  });
  root.querySelectorAll('input[data-part-coll-res]').forEach((el) => {
    el.addEventListener('change', (e) => {
      state.parts[+e.target.dataset.partCollRes].collision.resolution = +e.target.value;
      saveToLocalStorage();
    });
  });
  root.querySelectorAll('input[data-part-scale]').forEach((el) => {
    el.addEventListener('change', (e) => {
      const i = +e.target.dataset.partScale;
      const a = +e.target.dataset.axis;
      state.parts[i].scale_xyz[a] = +e.target.value;
      saveToLocalStorage();
    });
  });
  root.querySelectorAll('input[data-part-mass]').forEach((el) => {
    el.addEventListener('change', (e) => {
      state.parts[+e.target.dataset.partMass].mass = +e.target.value;
      saveToLocalStorage();
    });
  });
  root.querySelectorAll('input[data-part-cgroup]').forEach((el) => {
    el.addEventListener('change', (e) => {
      const v = e.target.value.trim();
      state.parts[+e.target.dataset.partCgroup].collision_group = v || null;
      saveToLocalStorage();
    });
  });
}

document.getElementById('btn-add-part').addEventListener('click', () => {
  state.parts.push(mkDefaultPart(uniquePartId('part')));
  rebuildAll();
  saveToLocalStorage();
});

document.getElementById('btn-auto-label').addEventListener('click', autoLabelClusters);

// ---- Joints UI ------------------------------------------------------------

function mkDefaultJoint(parent, child) {
  return {
    id: `${parent}_${child}_joint`,
    parent,
    child,
    type: 'revolute',
    axis_p0: [0, 0, 0],
    axis_p1: [1, 0, 0],
    lower: -120,
    upper: 120,
    bake_angle: 0,
    drive: { target: 0, stiffness: 50000, damping: 1000, max_force: 100000 },
    limit_hard: true,
    filtered_pairs: [],
    offset: [0, 0, 0],
  };
}

function uniqueJointId(base) {
  let n = 1;
  let id = base;
  while (state.joints.find((j) => j.id === id)) { n++; id = `${base}_${n}`; }
  return id;
}

function rebuildJointsList() {
  const root = document.getElementById('joints-list');
  root.innerHTML = '';
  state.joints.forEach((j, idx) => {
    const card = document.createElement('div');
    card.className = 'card';
    const isSelected = state.selectedJointId === j.id;
    card.style.borderColor = isSelected ? '#4a8' : '#333';

    const partOptions = state.parts.map((p) => `<option value="${escapeAttr(p.id)}">${escapeAttr(p.id)}</option>`).join('');

    card.innerHTML = `
      <div class="card-title">
        <input class="id-input" data-joint-idx="${idx}" type="text" value="${escapeAttr(j.id)}">
        <button class="tiny" data-select-joint="${idx}">${isSelected ? '★ active' : 'select'}</button>
        <button class="tiny danger" data-rm-joint="${idx}">×</button>
      </div>
      <div class="field-grid">
        <label>type</label>
        <select data-joint-type="${idx}">
          ${JOINT_TYPES.map((t) => `<option value="${t}"${j.type === t ? ' selected' : ''}>${t}</option>`).join('')}
        </select>
        <label>parent</label>
        <select data-joint-parent="${idx}">${partOptions}</select>
        <label>child</label>
        <select data-joint-child="${idx}">${partOptions}</select>
      </div>
      <div data-joint-typespecific="${idx}" style="margin-top:6px;"></div>
    `;
    // Set the parent / child select values (innerHTML setting doesn't preserve selectedness reliably)
    card.querySelector(`select[data-joint-parent="${idx}"]`).value = j.parent;
    card.querySelector(`select[data-joint-child="${idx}"]`).value = j.child;
    root.appendChild(card);

    renderJointTypeSpecific(card.querySelector(`[data-joint-typespecific="${idx}"]`), idx);
  });

  root.querySelectorAll('input.id-input').forEach((el) => {
    el.addEventListener('change', (e) => {
      const i = +e.target.dataset.jointIdx;
      const oldId = state.joints[i].id;
      const newId = e.target.value.trim();
      if (!newId || (newId !== oldId && state.joints.find((j) => j.id === newId))) {
        e.target.value = oldId;
        return;
      }
      state.joints[i].id = newId;
      saveToLocalStorage();
      rebuildJointsList();
    });
  });
  root.querySelectorAll('button[data-select-joint]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const i = +btn.dataset.selectJoint;
      state.selectedJointId = state.joints[i].id;
      restoreLidBaseGeometry();   // wipe any prior preview
      rebuildJointsList();
      updateAngleSliderEnabled();
      updateHingeMarkers();
    });
  });
  root.querySelectorAll('button[data-rm-joint]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const i = +btn.dataset.rmJoint;
      if (state.selectedJointId === state.joints[i].id) state.selectedJointId = null;
      state.joints.splice(i, 1);
      rebuildJointsList();
      updateAngleSliderEnabled();
      updateHingeMarkers();
      saveToLocalStorage();
    });
  });
  root.querySelectorAll('select[data-joint-type]').forEach((el) => {
    el.addEventListener('change', (e) => {
      const i = +e.target.dataset.jointType;
      state.joints[i].type = e.target.value;
      rebuildJointsList();
      saveToLocalStorage();
    });
  });
  root.querySelectorAll('select[data-joint-parent]').forEach((el) => {
    el.addEventListener('change', (e) => {
      const i = +e.target.dataset.jointParent;
      state.joints[i].parent = e.target.value;
      saveToLocalStorage();
    });
  });
  root.querySelectorAll('select[data-joint-child]').forEach((el) => {
    el.addEventListener('change', (e) => {
      const i = +e.target.dataset.jointChild;
      state.joints[i].child = e.target.value;
      saveToLocalStorage();
    });
  });
}

function renderJointTypeSpecific(host, idx) {
  const j = state.joints[idx];
  if (j.type === 'free') {
    host.innerHTML = `<div style="font-size:11px;color:#888;">free → child 是自由刚体（USDA 不会发 joint）</div>`;
    return;
  }
  if (j.type === 'fixed') {
    host.innerHTML = `<div style="font-size:11px;color:#888;">fixed → 两 part 刚性焊接</div>`;
    return;
  }
  // revolute or prismatic
  const isPrismatic = j.type === 'prismatic';
  host.innerHTML = `
    <div class="field-grid" style="grid-template-columns: 60px 1fr;">
      <label>${isPrismatic ? 'origin' : 'p0'}</label>
      <div class="row">
        <input type="number" data-axis="${idx}-${isPrismatic ? 'origin' : 'p0'}-0" value="${(isPrismatic ? j.axis_origin || [0,0,0] : j.axis_p0)[0]}" step="0.01" style="width:55px;">
        <input type="number" data-axis="${idx}-${isPrismatic ? 'origin' : 'p0'}-1" value="${(isPrismatic ? j.axis_origin || [0,0,0] : j.axis_p0)[1]}" step="0.01" style="width:55px;">
        <input type="number" data-axis="${idx}-${isPrismatic ? 'origin' : 'p0'}-2" value="${(isPrismatic ? j.axis_origin || [0,0,0] : j.axis_p0)[2]}" step="0.01" style="width:55px;">
      </div>
      <label>${isPrismatic ? 'dir' : 'p1'}</label>
      <div class="row">
        <input type="number" data-axis="${idx}-${isPrismatic ? 'dir' : 'p1'}-0" value="${(isPrismatic ? j.axis_dir || [1,0,0] : j.axis_p1)[0]}" step="0.01" style="width:55px;">
        <input type="number" data-axis="${idx}-${isPrismatic ? 'dir' : 'p1'}-1" value="${(isPrismatic ? j.axis_dir || [1,0,0] : j.axis_p1)[1]}" step="0.01" style="width:55px;">
        <input type="number" data-axis="${idx}-${isPrismatic ? 'dir' : 'p1'}-2" value="${(isPrismatic ? j.axis_dir || [1,0,0] : j.axis_p1)[2]}" step="0.01" style="width:55px;">
      </div>
      <label>limits</label>
      <div class="row">
        <span style="font-size:10px;color:#888;">lo</span>
        <input type="number" data-joint-lower="${idx}" value="${j.lower}" step="${isPrismatic ? '0.001' : '1'}" style="width:75px;">
        <span style="font-size:10px;color:#888;">hi</span>
        <input type="number" data-joint-upper="${idx}" value="${j.upper}" step="${isPrismatic ? '0.001' : '1'}" style="width:75px;">
      </div>
      <label>bake</label>
      <input type="number" data-joint-bake="${idx}" value="${isPrismatic ? (j.bake_distance ?? 0) : (j.bake_angle ?? 0)}" step="${isPrismatic ? '0.001' : '1'}" style="width:80px;">
      <label>offset</label>
      <div class="row">
        <input type="number" data-joint-offset="${idx}-0" value="${j.offset[0]}" step="0.01" style="width:55px;">
        <input type="number" data-joint-offset="${idx}-1" value="${j.offset[1]}" step="0.01" style="width:55px;">
        <input type="number" data-joint-offset="${idx}-2" value="${j.offset[2]}" step="0.01" style="width:55px;">
      </div>
      <label>drive target</label>
      <input type="number" data-joint-drive="${idx}-target" value="${j.drive.target}" step="${isPrismatic ? '0.001' : '1'}" style="width:100px;" title="${isPrismatic ? '目标位移 (m)' : '目标角度 (deg)'}">
      <label>stiffness K</label>
      <input type="number" data-joint-drive="${idx}-stiffness" value="${j.drive.stiffness}" min="0" step="500" style="width:100px;" title="弹簧刚度 — 越大越硬，把 part 拉向 target 更快">
      <label>damping B</label>
      <input type="number" data-joint-drive="${idx}-damping" value="${j.drive.damping}" min="0" step="50" style="width:100px;" title="阻尼 — 越大震荡越少；过小→过冲，过大→反应迟钝">
      <label>max force</label>
      <input type="number" data-joint-drive="${idx}-max_force" value="${j.drive.max_force ?? 0}" min="0" step="500" style="width:100px;" title="${isPrismatic ? '驱动最大牛 (N)' : '驱动最大力矩 (N·m)'} — 上限钳制 stiffness/damping 算出的力">
      <label>limit_hard</label>
      <input type="checkbox" data-joint-limit-hard="${idx}"${j.limit_hard ? ' checked' : ''} style="width:auto;">
    </div>
  `;
  // Axis numeric inputs
  host.querySelectorAll('input[data-axis]').forEach((el) => {
    el.addEventListener('change', (e) => {
      const [iStr, fieldName, axStr] = e.target.dataset.axis.split('-');
      const i = +iStr;
      const ax = +axStr;
      const jj = state.joints[i];
      const fieldKey = (jj.type === 'prismatic')
        ? (fieldName === 'origin' ? 'axis_origin' : 'axis_dir')
        : (fieldName === 'p0' ? 'axis_p0' : 'axis_p1');
      jj[fieldKey] = jj[fieldKey] || [0, 0, 0];
      jj[fieldKey][ax] = +e.target.value;
      updateHingeMarkers();
      saveToLocalStorage();
    });
  });
  // (no pick button — drag the red/green spheres in the viewport instead)
  host.querySelectorAll('input[data-joint-lower]').forEach((el) => {
    el.addEventListener('change', (e) => {
      state.joints[+e.target.dataset.jointLower].lower = +e.target.value;
      saveToLocalStorage();
    });
  });
  host.querySelectorAll('input[data-joint-upper]').forEach((el) => {
    el.addEventListener('change', (e) => {
      state.joints[+e.target.dataset.jointUpper].upper = +e.target.value;
      saveToLocalStorage();
    });
  });
  host.querySelectorAll('input[data-joint-bake]').forEach((el) => {
    el.addEventListener('change', (e) => {
      const i = +e.target.dataset.jointBake;
      const jj = state.joints[i];
      if (jj.type === 'prismatic') jj.bake_distance = +e.target.value;
      else jj.bake_angle = +e.target.value;
      saveToLocalStorage();
    });
  });
  host.querySelectorAll('input[data-joint-offset]').forEach((el) => {
    el.addEventListener('change', (e) => {
      const [iStr, axStr] = e.target.dataset.jointOffset.split('-');
      state.joints[+iStr].offset[+axStr] = +e.target.value;
      saveToLocalStorage();
    });
  });
  host.querySelectorAll('input[data-joint-drive]').forEach((el) => {
    el.addEventListener('change', (e) => {
      const [iStr, key] = e.target.dataset.jointDrive.split('-');
      state.joints[+iStr].drive[key] = +e.target.value;
      saveToLocalStorage();
    });
  });
  host.querySelectorAll('input[data-joint-limit-hard]').forEach((el) => {
    el.addEventListener('change', (e) => {
      state.joints[+e.target.dataset.jointLimitHard].limit_hard = e.target.checked;
      saveToLocalStorage();
    });
  });
}

document.getElementById('btn-add-joint').addEventListener('click', () => {
  if (state.parts.length < 2) {
    alert('需要至少 2 个 part 才能加 joint');
    return;
  }
  const j = mkDefaultJoint(state.parts[0].id, state.parts[1].id);
  j.id = uniqueJointId(j.id);
  state.joints.push(j);
  rebuildAll();
  saveToLocalStorage();
});

document.getElementById('btn-snap-hinge').addEventListener('click', () => {
  if (!state.selectedJointId) {
    log('[snap] select a joint first');
    return;
  }
  const j = state.joints.find((x) => x.id === state.selectedJointId);
  if (!j || j.type !== 'revolute') {
    log('[snap] only revolute joints can snap to split boundary');
    return;
  }
  // Find a split whose parent/child matches this joint
  const splitName = Object.keys(state.split_clusters).find((n) => {
    const info = state.split_clusters[n];
    return j.parent === info.parent || j.child === info.parent
        || j.parent === n || j.child === n;
  });
  if (!splitName) {
    log(`[snap] no box-split found for parts ${j.parent} / ${j.child}`);
    return;
  }
  const n = _snapRevoluteJointsToSplit(splitName);
  if (n) {
    log(`[snap] aligned ${j.id} to ${splitName} boundary plane`);
    rebuildJointsList();
    saveToLocalStorage();
  }
});

document.getElementById('btn-bake-angle').addEventListener('click', () => {
  if (!state.selectedJointId) return;
  const j = state.joints.find((x) => x.id === state.selectedJointId);
  if (!j || j.type !== 'revolute') return;
  j.bake_angle = +angleSlider.value;
  log(`baked ${j.id}.bake_angle = ${j.bake_angle}°`);
  rebuildJointsList();
  saveToLocalStorage();
});

function updateAngleSliderEnabled() {
  const j = state.selectedJointId
    ? state.joints.find((x) => x.id === state.selectedJointId)
    : null;
  const isRevolute = j && j.type === 'revolute';
  angleSlider.disabled = !isRevolute;
  angleNum.disabled = !isRevolute;
  document.getElementById('btn-bake-angle').disabled = !isRevolute;
  if (isRevolute) {
    angleSlider.value = j.bake_angle ?? 0;
    angleNum.value = angleSlider.value;
  } else {
    angleSlider.value = 0;
    angleNum.value = 0;
  }
}

// ---- Hinge markers + line --------------------------------------------------

// Schema-side coords (j.axis_p0/p1, AABBs) live in the GLB's intrinsic
// local frame. Cluster meshes render at LOCAL × cluster.mesh.scale. The
// viewport gizmo / hinge line need to display at the SAME world position
// as the rendered mesh, so multiply by this factor whenever we go local
// -> world for visualization, and divide when going back (gizmo drag).
function _displayScale() {
  // All clusters from the same GLB share scale (set on the gltf node's
  // matrix). Pull it from any visible cluster.
  const c = state.clusters.find((x) => x.mesh.visible) || state.clusters[0];
  return c ? c.mesh.scale.x : 1.0;
}

function updateHingeMarkers() {
  const j = state.selectedJointId ? state.joints.find((x) => x.id === state.selectedJointId) : null;
  if (!j || (j.type !== 'revolute' && j.type !== 'prismatic')) {
    state.hingeMarkers.p0.visible = false;
    state.hingeMarkers.p1.visible = false;
    hingeLine.visible = false;
    if (transformCtrl.object && transformCtrl.object.userData.markerKind) transformCtrl.detach();
    return;
  }
  const p0 = j.type === 'revolute' ? j.axis_p0 : j.axis_origin || [0, 0, 0];
  const p1 = j.type === 'revolute' ? j.axis_p1 : (() => {
    const o = j.axis_origin || [0, 0, 0];
    const d = j.axis_dir || [1, 0, 0];
    return [o[0] + d[0] * 0.1, o[1] + d[1] * 0.1, o[2] + d[2] * 0.1];
  })();
  const s = _displayScale();
  state.hingeMarkers.p0.position.set(p0[0] * s, p0[1] * s, p0[2] * s);
  state.hingeMarkers.p1.position.set(p1[0] * s, p1[1] * s, p1[2] * s);
  state.hingeMarkers.p0.visible = true;
  state.hingeMarkers.p1.visible = true;
  updateHingeLine();
}

function updateHingeLine() {
  hingeLine.geometry.setFromPoints([
    state.hingeMarkers.p0.position.clone(),
    state.hingeMarkers.p1.position.clone(),
  ]);
  hingeLine.visible = true;
}

// Live-update the joint card's p0/p1 number inputs without re-rendering the
// whole card (which would steal focus from the active gizmo).
function syncJointCardInputs(j) {
  const idx = state.joints.indexOf(j);
  if (idx < 0) return;
  const isPrismatic = j.type === 'prismatic';
  const f0 = isPrismatic ? 'origin' : 'p0';
  const f1 = isPrismatic ? 'dir' : 'p1';
  const v0 = isPrismatic ? (j.axis_origin || [0, 0, 0]) : j.axis_p0;
  const v1 = isPrismatic ? (j.axis_dir || [1, 0, 0]) : j.axis_p1;
  for (let i = 0; i < 3; i++) {
    const a = document.querySelector(`input[data-axis="${idx}-${f0}-${i}"]`);
    const b = document.querySelector(`input[data-axis="${idx}-${f1}-${i}"]`);
    if (a && document.activeElement !== a) a.value = v0[i];
    if (b && document.activeElement !== b) b.value = v1[i];
  }
}

// ---- Live preview: rotate child verts by slider angle ---------------------

function applyPreviewAngle() {
  if (!state.selectedJointId) return;
  const j = state.joints.find((x) => x.id === state.selectedJointId);
  if (!j || j.type !== 'revolute') return;
  const partId = j.child;
  const part = state.parts.find((p) => p.id === partId);
  if (!part) return;
  const angle = (+angleSlider.value) * Math.PI / 180;

  const p0 = new THREE.Vector3(...j.axis_p0);
  const p1 = new THREE.Vector3(...j.axis_p1);
  const axis = new THREE.Vector3().subVectors(p1, p0).normalize();
  const q = new THREE.Quaternion().setFromAxisAngle(axis, angle);

  for (const cname of part.clusters) {
    const c = state.clusters.find((x) => x.name === cname);
    if (!c) continue;
    if (!state.baseLidPose.has(c.name)) {
      state.baseLidPose.set(c.name, c.mesh.geometry.attributes.position.array.slice());
    }
    const base = state.baseLidPose.get(c.name);
    const pos = c.mesh.geometry.attributes.position.array;
    const tmp = new THREE.Vector3();
    for (let i = 0; i < base.length; i += 3) {
      tmp.set(base[i] - p0.x, base[i + 1] - p0.y, base[i + 2] - p0.z);
      tmp.applyQuaternion(q);
      pos[i] = tmp.x + p0.x;
      pos[i + 1] = tmp.y + p0.y;
      pos[i + 2] = tmp.z + p0.z;
    }
    c.mesh.geometry.attributes.position.needsUpdate = true;
    c.mesh.geometry.computeBoundingBox();
    c.mesh.geometry.computeBoundingSphere();
  }
}

function restoreLidBaseGeometry() {
  for (const [name, base] of state.baseLidPose) {
    const c = state.clusters.find((x) => x.name === name);
    if (!c) continue;
    c.mesh.geometry.attributes.position.array.set(base);
    c.mesh.geometry.attributes.position.needsUpdate = true;
    c.mesh.geometry.computeBoundingBox();
    c.mesh.geometry.computeBoundingSphere();
  }
  state.baseLidPose.clear();
}

// ---- Cluster table --------------------------------------------------------

function rebuildClusterTable() {
  const root = document.getElementById('cluster-table');
  root.innerHTML = '';
  state.clusters.forEach((c) => {
    const row = document.createElement('div');
    row.className = 'cluster-row';
    if (state.selectedClusterId === c.id) row.classList.add('selected');
    const part = partOfCluster(c.name);
    const color = clusterColor(c.name);
    row.innerHTML = `
      <span class="swatch" style="background:#${color.toString(16).padStart(6, '0')};"></span>
      <span class="name" data-cluster-id="${c.id}">${escapeAttr(c.name)}</span>
      <select data-cluster-assign="${c.id}" data-cluster-name="${escapeAttr(c.name)}">
        <option value="">— 未分配 —</option>
        ${state.parts.map((p) => `<option value="${escapeAttr(p.id)}"${part && part.id === p.id ? ' selected' : ''}>${escapeAttr(p.id)}</option>`).join('')}
      </select>
    `;
    root.appendChild(row);
  });
  root.querySelectorAll('[data-cluster-id]').forEach((el) => {
    el.addEventListener('click', (e) => {
      const id = +e.target.dataset.clusterId;
      selectCluster(state.selectedClusterId === id ? null : id);
    });
  });
  root.querySelectorAll('select[data-cluster-assign]').forEach((el) => {
    el.addEventListener('change', (e) => {
      assignClusterToPart(e.target.dataset.clusterName, e.target.value || null);
    });
  });
}

// ---- Box-select part splitter --------------------------------------------
//
// DBSCAN can't split a fully-connected mesh (e.g. a Seed3D folding phone
// where the two screens + hinge are welded into one component). The user
// drags a 2-D rectangle on the canvas; everything inside the rectangle is
// gathered into a freshly-created part. Whole selected clusters move as-is;
// partially selected clusters are split first, and the inside cluster joins
// the new part. UVs travel with their triangles.
const BOX_SELECT_MIN_TRI_RATIO = 0.01;
const BOX_SELECT_WHOLE_CLUSTER_RATIO = 0.95;

const boxSelect = {
  active: false,
  dragging: false,
  startX: 0, startY: 0,   // CSS pixels relative to canvas
  curX: 0, curY: 0,
  // When non-null, the next dragged rectangle defines a *site AABB* for
  // ``targetPartId`` instead of cutting a cluster. See setSiteDrawMode().
  targetPartId: null,
};

function setBoxSelectMode(active) {
  boxSelect.active = active;
  if (!active) boxSelect.targetPartId = null;  // exiting clears any site target
  document.getElementById('box-select-banner').style.display = active ? 'block' : 'none';
  document.getElementById('box-overlay').style.display = 'none';
  controls.enabled = !active;
  renderer.domElement.style.cursor = active ? 'crosshair' : '';
  document.getElementById('btn-box-select').classList.toggle('active', active);
}

document.getElementById('btn-box-select').addEventListener('click', () => {
  setBoxSelectMode(!boxSelect.active);
});
window.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (boxSelect.active) { setBoxSelectMode(false); return; }
  if (transformCtrl.object) { transformCtrl.detach(); return; }
  if (state.selectedClusterId !== null) { selectCluster(null); return; }
});

function _canvasXY(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}
function _toNDC(x, y) {
  const rect = renderer.domElement.getBoundingClientRect();
  return { x: (x / rect.width) * 2 - 1, y: -(y / rect.height) * 2 + 1 };
}

renderer.domElement.addEventListener('pointerdown', (e) => {
  if (!boxSelect.active || e.button !== 0) return;
  e.preventDefault();
  const p = _canvasXY(e);
  boxSelect.dragging = true;
  boxSelect.startX = p.x; boxSelect.startY = p.y;
  boxSelect.curX = p.x; boxSelect.curY = p.y;
  const overlay = document.getElementById('box-overlay');
  overlay.style.display = 'block';
  overlay.style.left = `${p.x}px`;
  overlay.style.top = `${p.y}px`;
  overlay.style.width = '0px';
  overlay.style.height = '0px';
});
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!boxSelect.dragging) return;
  const p = _canvasXY(e);
  boxSelect.curX = p.x; boxSelect.curY = p.y;
  const overlay = document.getElementById('box-overlay');
  const x0 = Math.min(boxSelect.startX, boxSelect.curX);
  const y0 = Math.min(boxSelect.startY, boxSelect.curY);
  const x1 = Math.max(boxSelect.startX, boxSelect.curX);
  const y1 = Math.max(boxSelect.startY, boxSelect.curY);
  overlay.style.left = `${x0}px`; overlay.style.top = `${y0}px`;
  overlay.style.width = `${x1 - x0}px`; overlay.style.height = `${y1 - y0}px`;
});
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!boxSelect.dragging) return;
  boxSelect.dragging = false;
  document.getElementById('box-overlay').style.display = 'none';
  const x0 = Math.min(boxSelect.startX, boxSelect.curX);
  const y0 = Math.min(boxSelect.startY, boxSelect.curY);
  const x1 = Math.max(boxSelect.startX, boxSelect.curX);
  const y1 = Math.max(boxSelect.startY, boxSelect.curY);
  if (x1 - x0 < 6 || y1 - y0 < 6) {
    log('[box-select] rectangle too small, ignored');
    return;
  }
  const ndcMin = _toNDC(x0, y1);   // bottom-left in NDC (Y flipped)
  const ndcMax = _toNDC(x1, y0);   // top-right in NDC
  if (boxSelect.targetPartId) {
    performSiteDraw(ndcMin.x, ndcMin.y, ndcMax.x, ndcMax.y, boxSelect.targetPartId);
  } else {
    performBoxSplit(ndcMin.x, ndcMin.y, ndcMax.x, ndcMax.y);
  }
});

function setSiteDrawMode(partId) {
  // Site-draw rides on top of box-select: same drag/overlay UX, just a
  // different pointer-up handler. Setting partId enables the mode; passing
  // null exits.
  boxSelect.targetPartId = partId || null;
  setBoxSelectMode(!!partId);
  const banner = document.getElementById('box-select-banner');
  banner.textContent = partId
    ? `🟦 Site 框选模式 (part=${partId}) — 拖动画框, Esc 退出`
    : `📦 框选模式：拖动画框 — Esc 退出`;
}

function performSiteDraw(nx0, ny0, nx1, ny1, partId) {
  // Iterate vertices belonging to ``partId``'s clusters; collect those whose
  // projected NDC falls inside the rect; AABB-bound them in mesh-local frame
  // (= GLB intrinsic, same coord system as split_clusters). No mesh edit.
  const part = state.parts.find((p) => p.id === partId);
  if (!part) { log(`[site] part ${partId} disappeared`); return; }

  const aabbMin = [Infinity, Infinity, Infinity];
  const aabbMax = [-Infinity, -Infinity, -Infinity];
  let nIn = 0;
  const m = new THREE.Vector3();
  for (const c of state.clusters) {
    if (!part.clusters.includes(c.name)) continue;
    if (!c.mesh.visible) continue;
    const pos = c.mesh.geometry.attributes.position.array;
    const n = pos.length / 3;
    for (let i = 0; i < n; i++) {
      m.set(pos[i*3], pos[i*3+1], pos[i*3+2]).applyMatrix4(c.mesh.matrixWorld).project(camera);
      if (m.x < nx0 || m.x > nx1 || m.y < ny0 || m.y > ny1) continue;
      const lx = pos[i*3], ly = pos[i*3+1], lz = pos[i*3+2];
      if (lx < aabbMin[0]) aabbMin[0] = lx; if (lx > aabbMax[0]) aabbMax[0] = lx;
      if (ly < aabbMin[1]) aabbMin[1] = ly; if (ly > aabbMax[1]) aabbMax[1] = ly;
      if (lz < aabbMin[2]) aabbMin[2] = lz; if (lz > aabbMax[2]) aabbMax[2] = lz;
      nIn++;
    }
  }
  if (nIn === 0) {
    setBanner('error', `Site 框选完成 — part "${partId}" 没有顶点落在框内`);
    setTimeout(() => setBanner('hidden'), 3500);
    setSiteDrawMode(null);
    return;
  }
  // Auto-name the site by current count + kind=custom (user can rename).
  if (!Array.isArray(part.sites)) part.sites = [];
  const baseId = `site_${part.sites.length}`;
  let sid = baseId, k = 0;
  while (part.sites.find((s) => s.id === sid)) { k++; sid = `${baseId}_${k}`; }
  part.sites.push({ id: sid, kind: 'custom', aabb_min: aabbMin, aabb_max: aabbMax });
  setSiteDrawMode(null);
  log(`[site] ${partId}.${sid}: ${nIn} verts in box, AABB min=${aabbMin.map((v)=>v.toFixed(3))} max=${aabbMax.map((v)=>v.toFixed(3))}`);
  rebuildSitesList();
  saveToLocalStorage();
}

// Render the per-part site cards under the Sites tab.
function rebuildSitesList() {
  const wrap = document.getElementById('sites-list');
  if (!wrap) return;
  wrap.innerHTML = '';
  if (state.parts.length === 0) {
    wrap.innerHTML = '<div style="font-size:11px;color:#888;">先在 ② Parts 里加 part, 才能给它标 site</div>';
    return;
  }
  for (const part of state.parts) {
    if (!Array.isArray(part.sites)) part.sites = [];
    const card = document.createElement('div');
    card.className = 'card';
    card.style.flex = '1 1 360px';
    card.innerHTML = `
      <div class="card-title">
        <b>${escapeAttr(part.id)}</b>
        <span style="color:#888;font-size:11px;">(${part.sites.length} sites)</span>
        <button class="add tiny" data-act="draw" data-pid="${escapeAttr(part.id)}">+ 框选 site</button>
      </div>
      <div data-role="sites-of-${escapeAttr(part.id)}" style="display:flex;flex-direction:column;gap:6px;"></div>
    `;
    wrap.appendChild(card);
    const inner = card.querySelector(`[data-role="sites-of-${cssEscape(part.id)}"]`);
    if (part.sites.length === 0) {
      inner.innerHTML = '<div style="font-size:11px;color:#777;font-style:italic;">尚未标 site</div>';
    } else {
      for (let i = 0; i < part.sites.length; i++) {
        inner.appendChild(_siteRow(part, i));
      }
    }
    card.querySelector('button[data-act="draw"]').addEventListener('click', () => {
      setSiteDrawMode(part.id);
    });
  }
}

function _siteRow(part, idx) {
  const s = part.sites[idx];
  const row = document.createElement('div');
  row.style.cssText = 'display:grid;grid-template-columns:1fr 90px 18px;gap:4px;align-items:center;padding:4px;background:#151515;border-radius:3px;';
  const fmtV = (v) => `[${v.map((x)=>(+x).toFixed(3)).join(',')}]`;
  const kindOptions = SITE_KINDS.map((k) => `<option value="${k}"${k===s.kind?' selected':''}>${k}</option>`).join('');
  row.innerHTML = `
    <input type="text" value="${escapeAttr(s.id)}" data-act="id" style="width:100%;">
    <select data-act="kind">${kindOptions}</select>
    <button class="danger tiny" data-act="del" title="删除 site">✕</button>
    <div style="grid-column:1/-1;font-size:10px;color:#888;font-family:monospace;">
      min ${fmtV(s.aabb_min)} max ${fmtV(s.aabb_max)}
    </div>
    <div data-role="override" style="grid-column:1/-1;"></div>
  `;
  row.querySelector('input[data-act="id"]').addEventListener('change', (e) => {
    const newId = e.target.value.trim();
    if (!newId) { e.target.value = s.id; return; }
    if (part.sites.find((x, j) => j !== idx && x.id === newId)) {
      log(`[site] duplicate id "${newId}" within ${part.id}`);
      e.target.value = s.id; return;
    }
    s.id = newId;
    saveToLocalStorage();
  });
  row.querySelector('select[data-act="kind"]').addEventListener('change', (e) => {
    s.kind = e.target.value;
    rebuildSitesList();
    saveToLocalStorage();
  });
  row.querySelector('button[data-act="del"]').addEventListener('click', () => {
    part.sites.splice(idx, 1);
    rebuildSitesList();
    saveToLocalStorage();
  });
  // material_override is generic — any site kind can carry one (paint a
  // screen black, paint a button red, paint a handle rubber-dark).
  row.querySelector('[data-role="override"]').appendChild(_overrideControls(part, idx));
  return row;
}

function _overrideControls(part, idx) {
  const s = part.sites[idx];
  const wrap = document.createElement('div');
  wrap.style.cssText = 'margin-top:4px;padding:5px;background:#1a1a1a;border:1px solid #333;border-radius:3px;font-size:11px;';
  const enabled = !!s.material_override;
  const mo = s.material_override || {};
  const col = mo.diffuseColor || [0.02, 0.02, 0.02];
  const hex = '#' + col.map((c) => Math.round(Math.max(0, Math.min(1, c)) * 255).toString(16).padStart(2, '0')).join('');
  const rough = mo.roughness != null ? mo.roughness : 0.15;
  const metal = mo.metallic != null ? mo.metallic : 0.0;
  wrap.innerHTML = `
    <label style="display:flex;align-items:center;gap:6px;color:#aac;">
      <input type="checkbox" data-act="enable" ${enabled?'checked':''}>
      启用 material override (在 site AABB 上盖一片自定义 PBR — 屏幕黑 / 按键红 / 把手深灰，都行)
    </label>
    <div data-role="fields" style="display:${enabled?'grid':'none'};grid-template-columns:auto 1fr;gap:4px 8px;margin-top:5px;align-items:center;">
      <label style="color:#888;">diffuseColor</label>
      <input type="color" data-act="color" value="${hex}">
      <label style="color:#888;">roughness</label>
      <input type="number" data-act="rough" value="${rough}" min="0" max="1" step="0.05" style="width:80px;">
      <label style="color:#888;">metallic</label>
      <input type="number" data-act="metal" value="${metal}" min="0" max="1" step="0.05" style="width:80px;">
    </div>
  `;
  const fields = wrap.querySelector('[data-role="fields"]');
  function _writeOverride() {
    const c = wrap.querySelector('input[data-act="color"]').value;
    const rgb = [c.slice(1,3), c.slice(3,5), c.slice(5,7)].map((h) => parseInt(h,16) / 255);
    const r = +wrap.querySelector('input[data-act="rough"]').value;
    const m = +wrap.querySelector('input[data-act="metal"]').value;
    s.material_override = { diffuseColor: rgb, roughness: r, metallic: m };
    saveToLocalStorage();
  }
  wrap.querySelector('input[data-act="enable"]').addEventListener('change', (e) => {
    if (e.target.checked) {
      _writeOverride();
      fields.style.display = 'grid';
    } else {
      delete s.material_override;
      fields.style.display = 'none';
      saveToLocalStorage();
    }
  });
  for (const sel of ['color', 'rough', 'metal']) {
    wrap.querySelector(`input[data-act="${sel}"]`).addEventListener('change', _writeOverride);
  }
  return wrap;
}

// Best-effort CSS-attribute-safe escape for our part ids (which may contain
// underscores / digits but typically are valid CSS idents already).
function cssEscape(s) {
  return (window.CSS && CSS.escape) ? CSS.escape(s) : String(s).replace(/[^\w-]/g, '_');
}

async function performBoxSplit(nx0, ny0, nx1, ny1) {
  const sourceClusters = state.clusters.filter((c) => c.mesh.visible);
  setBanner('active', `框选拆分中…  扫描 ${sourceClusters.length} 个 cluster 的三角形`);
  await new Promise((r) => requestAnimationFrame(r));
  const t0 = performance.now();

  const selectedClusterNames = [];
  const splitClusterNames = [];
  let wholeClusters = 0;
  let splitTriangles = 0;
  let tinyHits = 0;
  for (const c of sourceClusters) {
    const counts = _countTrisInRect(c, nx0, ny0, nx1, ny1);
    const ratio = counts.tri ? counts.inside / counts.tri : 0;
    if (!counts.inside) continue;
    if (ratio < BOX_SELECT_MIN_TRI_RATIO) {
      tinyHits++;
      continue;
    }
    if (ratio >= BOX_SELECT_WHOLE_CLUSTER_RATIO) {
      selectedClusterNames.push(c.name);
      wholeClusters++;
      continue;
    }
    const split = trySplitClusterByRect(c, nx0, ny0, nx1, ny1);
    if (!split) continue;
    selectedClusterNames.push(split.newCluster.name);
    splitClusterNames.push(split.newCluster.name);
    splitTriangles += split.numTriangles;
  }
  if (selectedClusterNames.length === 0) {
    const hint = tinyHits ? '（命中过小，已忽略 <1% 的擦边 cluster）' : '';
    setBanner('error', `框选完成 — 框内没有可用三角形${hint}`);
    setTimeout(() => setBanner('hidden'), 3500);
    log(`[box-select] no selected geometry${tinyHits ? ` (${tinyHits} tiny hits ignored)` : ''}`);
    return;
  }

  const part = createBoxSelectionPart(selectedClusterNames);
  const dt = ((performance.now() - t0) / 1000).toFixed(2);
  log(
    `[box-select] ${dt}s — created part ${part.id} from ${selectedClusterNames.length} cluster(s) `
    + `(${wholeClusters} whole, ${splitClusterNames.length} split, ${splitTriangles} split triangles)`
  );
  setBanner('active', `框选完成 (${dt}s) — ${part.id} 包含 ${selectedClusterNames.length} 个 cluster`);
  setTimeout(() => setBanner('hidden'), 3500);
  // After every split, try snapping any revolute joint that points at this
  // split's parent/child to the AABB cut plane. Whole-cluster moves do not
  // create split metadata, so they do not need snapping.
  const snapped = splitClusterNames.reduce(
    (acc, name) => acc + _snapRevoluteJointsToSplit(name),
    0,
  );
  if (snapped) log(`[box-select] auto-snapped ${snapped} hinge axis to split boundary`);
  rebuildAll();
  saveToLocalStorage();
}

function createBoxSelectionPart(clusterNames) {
  const selected = Array.from(new Set(clusterNames));
  const selectedSet = new Set(selected);
  for (const p of state.parts) {
    p.clusters = p.clusters.filter((name) => !selectedSet.has(name));
  }
  const part = mkDefaultPart(uniquePartId('box_part'));
  part.clusters = selected;
  state.parts.push(part);
  return part;
}

// Locate the unique "narrow" axis of a split's AABB — the axis where
// AABB extent is much smaller than the parent extent (i.e. the axis along
// which the user actually sliced). Used to compute the cut plane.
function _splitNarrowAxis(splitName) {
  const info = state.split_clusters[splitName];
  if (!info) return null;
  const ext = [
    info.aabb_max[0] - info.aabb_min[0],
    info.aabb_max[1] - info.aabb_min[1],
    info.aabb_max[2] - info.aabb_min[2],
  ];
  // Whichever axis has the SMALLEST extent is the "thin" cut direction.
  // Heuristic: pick the axis whose AABB extent is < 60 % of the largest.
  const maxE = Math.max(...ext);
  const small = ext.findIndex((e) => e < 0.6 * maxE);
  return small >= 0 ? small : null;
}

// Snap revolute joint axes that match the given split (parent or child)
// to the split's AABB boundary plane: 2 endpoints at min/max corners on
// the 2 wider axes, both at the cut value on the narrow axis.
function _snapRevoluteJointsToSplit(splitName) {
  const info = state.split_clusters[splitName];
  if (!info) return 0;
  const narrow = _splitNarrowAxis(splitName);
  if (narrow === null) return 0;
  const cutVal = (info.aabb_min[narrow] + info.aabb_max[narrow]) / 2;
  const wide = [0, 1, 2].filter((a) => a !== narrow);
  // Build p0 / p1 along the wider axis, at the cut plane on the narrow axis.
  // Use the AABB's min corner of the OTHER wide axis to keep the axis a line
  // (not a degenerate point).
  const p0 = [0, 0, 0], p1 = [0, 0, 0];
  p0[narrow] = cutVal; p1[narrow] = cutVal;
  p0[wide[0]] = info.aabb_min[wide[0]]; p1[wide[0]] = info.aabb_max[wide[0]];
  p0[wide[1]] = (info.aabb_min[wide[1]] + info.aabb_max[wide[1]]) / 2;
  p1[wide[1]] = p0[wide[1]];

  let n = 0;
  for (const j of state.joints) {
    if (j.type !== 'revolute') continue;
    const involves = (j.parent === info.parent || j.child === info.parent
                      || j.parent === splitName || j.child === splitName);
    if (!involves) continue;
    j.axis_p0 = p0.slice();
    j.axis_p1 = p1.slice();
    n++;
  }
  if (n) updateHingeMarkers();
  return n;
}

// Lightweight count-only helper that mirrors trySplitClusterByRect's
// inside-test, used to pick the "best" cluster to split before doing the
// heavier geometry mutation.
function _countTrisInRect(cluster, nx0, ny0, nx1, ny1) {
  const geom = cluster.mesh.geometry;
  const pos = geom.attributes.position.array;
  const idxAttr = geom.index;
  const triCount = idxAttr ? idxAttr.count / 3 : pos.length / 9;
  const ndc = _projectMeshNDC(cluster.mesh);
  let inside = 0;
  for (let t = 0; t < triCount; t++) {
    let i0, i1, i2;
    if (idxAttr) {
      i0 = idxAttr.array[t * 3]; i1 = idxAttr.array[t * 3 + 1]; i2 = idxAttr.array[t * 3 + 2];
    } else {
      i0 = t * 3; i1 = t * 3 + 1; i2 = t * 3 + 2;
    }
    const cx = (ndc[i0 * 2] + ndc[i1 * 2] + ndc[i2 * 2]) / 3;
    const cy = (ndc[i0 * 2 + 1] + ndc[i1 * 2 + 1] + ndc[i2 * 2 + 1]) / 3;
    if (cx >= nx0 && cx <= nx1 && cy >= ny0 && cy <= ny1) inside++;
  }
  return { inside, tri: triCount };
}

// Build a NDC-space xy buffer for every vertex of the given mesh.
function _projectMeshNDC(mesh) {
  const pos = mesh.geometry.attributes.position.array;
  const n = pos.length / 3;
  const out = new Float32Array(n * 2);
  const tmp = new THREE.Vector3();
  mesh.updateMatrixWorld(true);
  for (let i = 0; i < n; i++) {
    tmp.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).applyMatrix4(mesh.matrixWorld).project(camera);
    out[i * 2] = tmp.x;
    out[i * 2 + 1] = tmp.y;
  }
  return out;
}

// Drop part-cluster refs and split_clusters whose names don't exist in the
// freshly-loaded GLB. Splits whose parent is itself an unknown name (i.e.
// the chain leads back to nothing real) also get dropped. Run after
// restoreFromLocalStorage on every loadGLB so old session bloat doesn't
// leak into the next export.
function pruneOrphansAgainstClusters() {
  const realClusters = new Set(state.clusters.map((c) => c.name));
  const beforeSplits = Object.keys(state.split_clusters || {}).length;
  // Iterate to fixed point — split-of-split chains.
  let changed = true;
  while (changed) {
    changed = false;
    for (const name of Object.keys(state.split_clusters)) {
      const info = state.split_clusters[name];
      const parentReal = realClusters.has(info.parent);
      const parentIsKnownSplit = info.parent in state.split_clusters;
      if (!parentReal && !parentIsKnownSplit) {
        delete state.split_clusters[name];
        changed = true;
      }
    }
  }
  // Names of clusters that WILL exist after splits are reapplied:
  const afterSplit = new Set([...realClusters, ...Object.keys(state.split_clusters)]);
  let pruned = 0;
  for (const p of state.parts) {
    const before = p.clusters.length;
    p.clusters = p.clusters.filter((c) => afterSplit.has(c));
    pruned += before - p.clusters.length;
  }
  const droppedSplits = beforeSplits - Object.keys(state.split_clusters).length;
  if (pruned + droppedSplits > 0) {
    log(`[prune] removed ${pruned} stale cluster refs + ${droppedSplits} orphan splits`);
  }
}

// Re-create the in-memory mesh for each persisted split by applying its AABB
// to the parent cluster (or a previously-replayed split). After this runs,
// state.clusters contains the split-derived meshes again, exactly like
// what the user would see right after doing a fresh box-select.
function reapplyPersistedSplits() {
  const splits = state.split_clusters || {};
  // Topo sort: parent before child
  const order = [];
  const seen = new Set();
  function visit(name) {
    if (seen.has(name)) return;
    if (splits[name]) visit(splits[name].parent);
    seen.add(name);
    if (splits[name]) order.push(name);
  }
  for (const n of Object.keys(splits)) visit(n);

  let applied = 0;
  for (const splitName of order) {
    const info = splits[splitName];
    const parent = state.clusters.find((c) => c.name === info.parent);
    if (!parent) continue;
    const result = applyAABBSplit(parent, info.aabb_min, info.aabb_max, splitName);
    if (result) applied++;
  }
  if (applied) log(`[splits] reapplied ${applied} persisted box-split(s)`);
}

// Triangle-mask split shared between box-select (screen-space rect) and
// AABB replay. Mutates ``cluster``'s geometry to keep only the OUTSIDE
// triangles, and creates a new cluster (named ``newName``) for the inside.
function _splitClusterFromMask(cluster, insideTri, nIn, triCount, newName) {
  if (nIn === 0 || nIn === triCount) return null;
  const geom = cluster.mesh.geometry;
  const pos = geom.attributes.position.array;
  const uv = geom.attributes.uv ? geom.attributes.uv.array : null;
  const idxAttr = geom.index;

  const inPos = []; const outPos = [];
  const inUV = uv ? [] : null;
  const outUV = uv ? [] : null;
  for (let t = 0; t < triCount; t++) {
    let i0, i1, i2;
    if (idxAttr) {
      i0 = idxAttr.array[t * 3]; i1 = idxAttr.array[t * 3 + 1]; i2 = idxAttr.array[t * 3 + 2];
    } else {
      i0 = t * 3; i1 = t * 3 + 1; i2 = t * 3 + 2;
    }
    const dst = insideTri[t] ? inPos : outPos;
    const dstUV = insideTri[t] ? inUV : outUV;
    for (const i of [i0, i1, i2]) {
      dst.push(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
      if (uv) dstUV.push(uv[i * 2], uv[i * 2 + 1]);
    }
  }

  function _mkGeom(arr, uvArr) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3));
    if (uvArr) g.setAttribute('uv', new THREE.Float32BufferAttribute(uvArr, 2));
    g.computeBoundingBox();
    g.computeBoundingSphere();
    return g;
  }

  const insideGeom = _mkGeom(inPos, inUV);
  const outsideGeom = _mkGeom(outPos, outUV);

  cluster.mesh.geometry.dispose();
  cluster.mesh.geometry = outsideGeom;

  const newMat = cluster.mesh.material.clone();
  const newMesh = new THREE.Mesh(insideGeom, newMat);
  newMesh.name = newName;
  newMesh.position.copy(cluster.mesh.position);
  newMesh.quaternion.copy(cluster.mesh.quaternion);
  newMesh.scale.copy(cluster.mesh.scale);
  meshGroup.add(newMesh);

  const newId = state.clusters.length === 0 ? 0 : Math.max(...state.clusters.map((c) => c.id)) + 1;
  const newCluster = {
    id: newId,
    name: newName,
    mesh: newMesh,
    verts: insideGeom.attributes.position.count,
  };
  state.clusters.push(newCluster);
  return { newCluster, numTriangles: nIn };
}

// Apply a previously-recorded AABB split to a cluster — same algorithm as
// box-select but the inside-test uses a precomputed world-space AABB.
function applyAABBSplit(cluster, aabbMin, aabbMax, newName) {
  const geom = cluster.mesh.geometry;
  const pos = geom.attributes.position.array;
  const idxAttr = geom.index;
  const triCount = idxAttr ? idxAttr.count / 3 : pos.length / 9;
  const inside = new Uint8Array(triCount);
  let nIn = 0;
  for (let t = 0; t < triCount; t++) {
    let i0, i1, i2;
    if (idxAttr) {
      i0 = idxAttr.array[t * 3]; i1 = idxAttr.array[t * 3 + 1]; i2 = idxAttr.array[t * 3 + 2];
    } else {
      i0 = t * 3; i1 = t * 3 + 1; i2 = t * 3 + 2;
    }
    const cx = (pos[i0 * 3] + pos[i1 * 3] + pos[i2 * 3]) / 3;
    const cy = (pos[i0 * 3 + 1] + pos[i1 * 3 + 1] + pos[i2 * 3 + 1]) / 3;
    const cz = (pos[i0 * 3 + 2] + pos[i1 * 3 + 2] + pos[i2 * 3 + 2]) / 3;
    if (cx >= aabbMin[0] && cx <= aabbMax[0]
        && cy >= aabbMin[1] && cy <= aabbMax[1]
        && cz >= aabbMin[2] && cz <= aabbMax[2]) {
      inside[t] = 1;
      nIn++;
    }
  }
  return _splitClusterFromMask(cluster, inside, nIn, triCount, newName);
}

function trySplitClusterByRect(cluster, nx0, ny0, nx1, ny1) {
  const geom = cluster.mesh.geometry;
  const pos = geom.attributes.position.array;
  const uv = geom.attributes.uv ? geom.attributes.uv.array : null;
  const idxAttr = geom.index;
  const ndc = _projectMeshNDC(cluster.mesh);
  const triCount = idxAttr ? idxAttr.count / 3 : pos.length / 9;

  const insideTri = new Uint8Array(triCount);
  let nIn = 0;
  for (let t = 0; t < triCount; t++) {
    let i0, i1, i2;
    if (idxAttr) {
      i0 = idxAttr.array[t * 3]; i1 = idxAttr.array[t * 3 + 1]; i2 = idxAttr.array[t * 3 + 2];
    } else {
      i0 = t * 3; i1 = t * 3 + 1; i2 = t * 3 + 2;
    }
    const cx = (ndc[i0 * 2] + ndc[i1 * 2] + ndc[i2 * 2]) / 3;
    const cy = (ndc[i0 * 2 + 1] + ndc[i1 * 2 + 1] + ndc[i2 * 2 + 1]) / 3;
    if (cx >= nx0 && cx <= nx1 && cy >= ny0 && cy <= ny1) {
      insideTri[t] = 1;
      nIn++;
    }
  }
  if (nIn === 0 || nIn === triCount) return null;   // nothing or everything → no useful split

  // Compute the inside subset's AABB in cluster-local coords. This is what
  // build_usd.py uses to replay the split on the full-resolution mesh
  // server-side (where the decimated triangle indices don't correspond to
  // anything). Coords are mesh-local, but since loadGLB() already baked the
  // GLB-node world transform into the vertex buffer, "local" here == "the
  // GLB's intrinsic frame", which matches what build_usd.py sees.
  const aabbMin = [Infinity, Infinity, Infinity];
  const aabbMax = [-Infinity, -Infinity, -Infinity];
  for (let t = 0; t < triCount; t++) {
    if (!insideTri[t]) continue;
    let i0, i1, i2;
    if (idxAttr) {
      i0 = idxAttr.array[t * 3]; i1 = idxAttr.array[t * 3 + 1]; i2 = idxAttr.array[t * 3 + 2];
    } else {
      i0 = t * 3; i1 = t * 3 + 1; i2 = t * 3 + 2;
    }
    for (const i of [i0, i1, i2]) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      if (x < aabbMin[0]) aabbMin[0] = x; if (x > aabbMax[0]) aabbMax[0] = x;
      if (y < aabbMin[1]) aabbMin[1] = y; if (y > aabbMax[1]) aabbMax[1] = y;
      if (z < aabbMin[2]) aabbMin[2] = z; if (z > aabbMax[2]) aabbMax[2] = z;
    }
  }

  // Build flat (non-indexed) inside / outside geoms — simpler than rebuilding
  // a shared vertex pool with separate index buffers, and avoids dangling
  // verts. UVs (if present) are duplicated alongside.
  const inPos = []; const outPos = [];
  const inUV = uv ? [] : null;
  const outUV = uv ? [] : null;
  for (let t = 0; t < triCount; t++) {
    let i0, i1, i2;
    if (idxAttr) {
      i0 = idxAttr.array[t * 3]; i1 = idxAttr.array[t * 3 + 1]; i2 = idxAttr.array[t * 3 + 2];
    } else {
      i0 = t * 3; i1 = t * 3 + 1; i2 = t * 3 + 2;
    }
    const dst = insideTri[t] ? inPos : outPos;
    const dstUV = insideTri[t] ? inUV : outUV;
    for (const i of [i0, i1, i2]) {
      dst.push(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
      if (uv) dstUV.push(uv[i * 2], uv[i * 2 + 1]);
    }
  }

  function _mkGeom(arr, uvArr) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3));
    if (uvArr) g.setAttribute('uv', new THREE.Float32BufferAttribute(uvArr, 2));
    g.computeBoundingBox();
    g.computeBoundingSphere();
    return g;
  }

  const insideGeom = _mkGeom(inPos, inUV);
  const outsideGeom = _mkGeom(outPos, outUV);

  // Replace the original cluster mesh's geometry with the "outside" subset.
  // Apply the same world transform as the original, by setting the new mesh's
  // matrix to identity (we already projected in world space via matrixWorld,
  // but the BufferGeometry's vertex coords are in the original local frame
  // — we need to keep cluster.mesh.matrix unchanged).
  cluster.mesh.geometry.dispose();
  cluster.mesh.geometry = outsideGeom;

  // Create new cluster from the inside subset (as a sibling mesh in the same
  // local frame as the parent — so we copy parent's matrix).
  const baseName = cluster.name;
  let n = 1;
  let newName = `${baseName}_split${n}`;
  while (state.clusters.find((c) => c.name === newName)) { n++; newName = `${baseName}_split${n}`; }

  const newMat = cluster.mesh.material.clone();
  const newMesh = new THREE.Mesh(insideGeom, newMat);
  newMesh.name = newName;
  newMesh.matrix.copy(cluster.mesh.matrix);
  newMesh.matrixAutoUpdate = cluster.mesh.matrixAutoUpdate;
  // The original mesh's vertex coords were already in world frame after the
  // initial loadGLB (we baked applyMatrix4 there), so the new mesh stays at
  // identity transform — same as cluster.mesh.
  newMesh.position.copy(cluster.mesh.position);
  newMesh.quaternion.copy(cluster.mesh.quaternion);
  newMesh.scale.copy(cluster.mesh.scale);
  meshGroup.add(newMesh);

  const newId = state.clusters.length === 0 ? 0 : Math.max(...state.clusters.map((c) => c.id)) + 1;
  const newCluster = {
    id: newId,
    name: newName,
    mesh: newMesh,
    verts: insideGeom.attributes.position.count,
  };
  state.clusters.push(newCluster);

  // Persist split for server-side replay on the full-res mesh.
  state.split_clusters[newName] = {
    parent: cluster.name,
    aabb_min: aabbMin,
    aabb_max: aabbMax,
  };

  return { newCluster, numTriangles: nIn };
}


// ---- Pointer / shift+click axis picking ----------------------------------

const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();

renderer.domElement.addEventListener('pointerdown', (e) => {
  // Box-select mode owns clicks; orbit-camera owns plain drags.
  if (boxSelect.active) return;

  const rect = renderer.domElement.getBoundingClientRect();
  ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(ndc, camera);

  // Hinge markers take priority — clicking p0 or p1 attaches the gizmo so
  // the user can drag the endpoint in 3D. Skip this if the markers aren't
  // currently visible (no joint selected or joint is fixed/free).
  if (state.hingeMarkers.p0.visible || state.hingeMarkers.p1.visible) {
    const markerHits = raycaster.intersectObjects(
      [state.hingeMarkers.p0, state.hingeMarkers.p1].filter((m) => m.visible),
      false,
    );
    if (markerHits.length > 0) {
      transformCtrl.attach(markerHits[0].object);
      transformCtrl.setMode('translate');
      e.preventDefault();
      e.stopPropagation();
      return;
    }
  }

  const hits = raycaster.intersectObjects(meshGroup.children, false);

  // Plain (no-shift) click on canvas: if it landed on a cluster, select it;
  // otherwise OrbitControls is doing its drag — don't fight with that.
  // Discriminator: only treat as a click when the user didn't drag.
  // We arm a small mousemove guard so a click that becomes a drag is ignored.
  if (e.shiftKey) return;
  const downX = e.clientX, downY = e.clientY;
  const upHandler = (ue) => {
    renderer.domElement.removeEventListener('pointerup', upHandler);
    if (Math.hypot(ue.clientX - downX, ue.clientY - downY) > 4) return;   // a drag, not a click
    const r2 = renderer.domElement.getBoundingClientRect();
    ndc.x = ((ue.clientX - r2.left) / r2.width) * 2 - 1;
    ndc.y = -((ue.clientY - r2.top) / r2.height) * 2 + 1;
    raycaster.setFromCamera(ndc, camera);
    const hits2 = raycaster.intersectObjects(meshGroup.children, false);
    if (hits2.length === 0) {
      selectCluster(null);
      return;
    }
    const c = state.clusters.find((x) => x.mesh === hits2[0].object);
    if (c) selectCluster(state.selectedClusterId === c.id ? null : c.id);
  };
  renderer.domElement.addEventListener('pointerup', upHandler);
});

// ---- External meshes UI ---------------------------------------------------

function rebuildExtMeshesList() {
  const root = document.getElementById('ext-meshes-list');
  root.innerHTML = '';
  state.external_meshes.forEach((em, idx) => {
    const card = document.createElement('div');
    card.className = 'card';
    const partOptions = state.parts.map((p) => `<option value="${escapeAttr(p.id)}">${escapeAttr(p.id)}</option>`).join('');
    card.innerHTML = `
      <div class="card-title">
        <span style="flex:1;font-size:11px;color:#bbb;">${escapeAttr(em.glb || '(无 glb 路径)')}</span>
        <button class="tiny" data-ext-gizmo="${idx}">gizmo</button>
        <button class="tiny danger" data-ext-rm="${idx}">×</button>
      </div>
      <div class="field-grid">
        <label>attach_to</label>
        <select data-ext-attach="${idx}">${partOptions}</select>
        <label>glb 路径</label>
        <input type="text" data-ext-glb="${idx}" value="${escapeAttr(em.glb || '')}" placeholder="path/to/file.glb">
        <label>translate</label>
        <div class="row">
          <input type="number" data-ext-t="${idx}-0" value="${em.transform.t[0]}" step="0.01" style="width:55px;">
          <input type="number" data-ext-t="${idx}-1" value="${em.transform.t[1]}" step="0.01" style="width:55px;">
          <input type="number" data-ext-t="${idx}-2" value="${em.transform.t[2]}" step="0.01" style="width:55px;">
        </div>
        <label>scale</label>
        <div class="row">
          <input type="number" data-ext-s="${idx}-0" value="${em.transform.s[0]}" step="0.01" style="width:55px;">
          <input type="number" data-ext-s="${idx}-1" value="${em.transform.s[1]}" step="0.01" style="width:55px;">
          <input type="number" data-ext-s="${idx}-2" value="${em.transform.s[2]}" step="0.01" style="width:55px;">
        </div>
        <label>quat wxyz</label>
        <div class="row">
          <input type="number" data-ext-q="${idx}-0" value="${em.transform.q_wxyz[0]}" step="0.01" style="width:55px;">
          <input type="number" data-ext-q="${idx}-1" value="${em.transform.q_wxyz[1]}" step="0.01" style="width:55px;">
          <input type="number" data-ext-q="${idx}-2" value="${em.transform.q_wxyz[2]}" step="0.01" style="width:55px;">
          <input type="number" data-ext-q="${idx}-3" value="${em.transform.q_wxyz[3]}" step="0.01" style="width:55px;">
        </div>
      </div>
    `;
    card.querySelector(`select[data-ext-attach="${idx}"]`).value = em.attach_to;
    root.appendChild(card);
  });

  root.querySelectorAll('button[data-ext-rm]').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.external_meshes.splice(+btn.dataset.extRm, 1);
      rebuildExtMeshesList();
      saveToLocalStorage();
    });
  });
  root.querySelectorAll('select[data-ext-attach]').forEach((el) => {
    el.addEventListener('change', (e) => {
      state.external_meshes[+e.target.dataset.extAttach].attach_to = e.target.value;
      saveToLocalStorage();
    });
  });
  root.querySelectorAll('input[data-ext-glb]').forEach((el) => {
    el.addEventListener('change', (e) => {
      state.external_meshes[+e.target.dataset.extGlb].glb = e.target.value;
      saveToLocalStorage();
    });
  });
  for (const fk of ['t', 's', 'q']) {
    root.querySelectorAll(`input[data-ext-${fk}]`).forEach((el) => {
      el.addEventListener('change', (e) => {
        const [iStr, axStr] = e.target.dataset[`ext${fk.toUpperCase()}`].split('-');
        const arr = state.external_meshes[+iStr].transform[fk === 'q' ? 'q_wxyz' : fk];
        arr[+axStr] = +e.target.value;
        saveToLocalStorage();
      });
    });
  }
  root.querySelectorAll('button[data-ext-gizmo]').forEach((btn) => {
    btn.addEventListener('click', () => {
      log('TODO: per-external-mesh 3D gizmo (load mesh into scene + attach TransformControls)');
    });
  });
}

document.getElementById('btn-add-ext-mesh').addEventListener('click', () => {
  if (state.parts.length === 0) { alert('先加 part'); return; }
  state.external_meshes.push({
    attach_to: state.parts[0].id,
    glb: '',
    transform: { t: [0, 0, 0], q_wxyz: [1, 0, 0, 0], s: [1, 1, 1] },
  });
  rebuildExtMeshesList();
  saveToLocalStorage();
});

function pullExtMeshTransformFromGroup(/* obj */) { /* TODO when gizmo is wired */ }

// ---- Tree summary ---------------------------------------------------------

function rebuildTreeSummary() {
  const el = document.getElementById('tree-summary');
  el.classList.remove('error');
  if (state.parts.length === 0) {
    el.textContent = '未加载';
    return;
  }
  const childrenOf = {};
  const incoming = {};
  for (const j of state.joints) {
    if (j.type === 'free') continue;
    childrenOf[j.parent] = childrenOf[j.parent] || [];
    childrenOf[j.parent].push({ child: j.child, joint: j });
    if (incoming[j.child]) {
      el.classList.add('error');
      el.textContent = `[error] part "${j.child}" has multiple incoming joints (${incoming[j.child]} + ${j.id})`;
      return;
    }
    incoming[j.child] = j.id;
  }
  const roots = state.parts.filter((p) => !incoming[p.id]);
  // Cycle detection (DFS)
  for (const start of state.parts) {
    let node = start.id;
    const seen = new Set();
    while (incoming[node]) {
      if (seen.has(node)) {
        el.classList.add('error');
        el.textContent = `[error] cycle through ${[...seen, node].join(' -> ')}`;
        return;
      }
      seen.add(node);
      node = state.joints.find((j) => j.id === incoming[node]).parent;
    }
  }
  const lines = [];
  function visit(pid, prefix) {
    const p = state.parts.find((x) => x.id === pid);
    const physics = p ? p.physics : '?';
    const j = state.joints.find((x) => x.child === pid);
    const tag = j ? `${j.type} → ${j.parent}` : (incoming[pid] ? '?' : 'root');
    lines.push(`${prefix}${pid}  (${physics}, ${tag})`);
    const kids = (childrenOf[pid] || []).map((x) => x.child);
    kids.forEach((k, i) => visit(k, prefix + '  '));
  }
  roots.forEach((r) => visit(r.id, ''));
  // Free parts
  state.parts.forEach((p) => {
    if (!incoming[p.id] && !roots.includes(p)) {
      lines.push(`${p.id}  (${p.physics}, free)`);
    }
  });
  el.textContent = lines.join('\n');
}

// ---- Aggregate rebuild ----------------------------------------------------

function rebuildAll() {
  rebuildPartsList();
  rebuildJointsList();
  rebuildClusterTable();
  rebuildExtMeshesList();
  rebuildTreeSummary();
  refreshClusterColors();
  updateAngleSliderEnabled();
  updateHingeMarkers();
  updateDimViz();
  document.getElementById('device-name').value = state.device;
  document.getElementById('dim-x').value = state.physical_dims_mm.x;
  document.getElementById('dim-y').value = state.physical_dims_mm.y;
  document.getElementById('dim-z').value = state.physical_dims_mm.z;
}

// ---- Device + dims wiring -------------------------------------------------

document.getElementById('device-name').addEventListener('change', (e) => {
  state.device = e.target.value.trim() || 'Device';
  saveToLocalStorage();
});
for (const k of ['x', 'y', 'z']) {
  const el = document.getElementById('dim-' + k);
  // Use 'input' (not just 'change') so the wireframe label updates while
  // the user is dragging / typing, not only after blur.
  el.addEventListener('input', (e) => {
    const v = +e.target.value;
    if (!isFinite(v) || v <= 0) return;
    state.physical_dims_mm[k] = v;
    updateDimViz();
  });
  el.addEventListener('change', () => saveToLocalStorage());
}

// ---- Export ---------------------------------------------------------------

function buildLabelsJSON() {
  return {
    version: SCHEMA_VERSION,
    device: state.device,
    source_glb: state.source_glb,
    physical_dims_mm: state.physical_dims_mm,
    parts: state.parts,
    joints: state.joints,
    external_meshes: state.external_meshes,
    split_clusters: state.split_clusters || {},
  };
}

document.getElementById('btn-export').addEventListener('click', () => {
  const data = buildLabelsJSON();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'labels.json';
  a.click();
  log(`exported labels.json (v${SCHEMA_VERSION}, ${state.parts.length} parts / ${state.joints.length} joints)`);
});

// ---- Direct USDA build via /api/build_usd --------------------------------

function _defaultExportPaths() {
  // Best-effort guesses. User can edit before clicking.
  const dev = state.device || 'device';
  // clean.glb: prefer the URL we last loaded, else preprocess output.
  let cleanGlb = state.source_glb || `scripts/tools/articulator/data/preprocessed.glb`;
  // Strip a leading "./" or absolute "/" so the server resolves it against
  // the repo root.
  cleanGlb = cleanGlb.replace(/^\.?\//, '').replace(/\?.*$/, '');
  // If the path looks like our own preprocess output, expose it directly.
  if (cleanGlb.endsWith('preprocessed.glb')) cleanGlb = `scripts/tools/articulator/data/preprocessed.glb`;
  return {
    clean_glb_path: cleanGlb,
    out_path: `outputs/${dev}/${dev}.usda`,
  };
}

function _ensureExportPathsFilled() {
  const cgEl = document.getElementById('export-clean-glb');
  const opEl = document.getElementById('export-out-path');
  if (!cgEl.value) cgEl.value = _defaultExportPaths().clean_glb_path;
  if (!opEl.value) opEl.value = _defaultExportPaths().out_path;
}

function _updateExportSummary() {
  const el = document.getElementById('export-summary-text');
  if (!el) return;
  const splits = Object.keys(state.split_clusters || {});
  const partsLine = state.parts.map((p) => `${p.id}(${p.clusters.length})`).join(', ') || '(no parts)';
  el.innerHTML = `${state.parts.length} parts: ${escapeAttr(partsLine)}` +
    `<br>${state.joints.length} joints, ${splits.length} box-splits${splits.length ? ': ' + splits.join(', ') : ''}` +
    `<br>${state.external_meshes.length} ext-meshes`;
}

document.querySelector('.tab[data-tab="export"]').addEventListener('click', () => {
  _ensureExportPathsFilled();
  _updateExportSummary();
});

document.getElementById('btn-clear-splits').addEventListener('click', () => {
  const n = Object.keys(state.split_clusters || {}).length;
  if (!n) { log('[splits] nothing to clear'); return; }
  if (!confirm(`清空 ${n} 个 box-split 记录？必须重新加载 GLB（重 preprocess 或重 load clean.glb）才能生效。`)) return;
  state.split_clusters = {};
  // Also drop any in-memory split-derived clusters from state.clusters and
  // remove their meshes from the scene — they're stale now.
  const splitNames = state.clusters.filter((c) => c.name.includes('_split')).map((c) => c.name);
  for (const name of splitNames) {
    const c = state.clusters.find((x) => x.name === name);
    if (c) { meshGroup.remove(c.mesh); c.mesh.geometry.dispose(); }
  }
  state.clusters = state.clusters.filter((c) => !c.name.includes('_split'));
  // Drop part-cluster refs that pointed at split-derived clusters.
  for (const p of state.parts) {
    p.clusters = p.clusters.filter((cn) => !cn.includes('_split'));
  }
  log(`[splits] cleared ${n} box-splits + dropped ${splitNames.length} split-derived clusters from state`);
  rebuildAll();
  saveToLocalStorage();
});

document.getElementById('btn-build-usd').addEventListener('click', async () => {
  _ensureExportPathsFilled();
  _updateExportSummary();
  const cleanGlb = document.getElementById('export-clean-glb').value.trim();
  const textureGlb = document.getElementById('export-texture-glb').value.trim();
  const outPath = document.getElementById('export-out-path').value.trim();
  const splitNames = Object.keys(state.split_clusters || {});
  log(`[build_usd] sending ${state.parts.length} parts, ${state.joints.length} joints, ${splitNames.length} splits ${splitNames.length ? '['+splitNames.join(',')+']' : ''}`);
  const btn = document.getElementById('btn-build-usd');
  const status = document.getElementById('build-usd-status');
  const logEl = document.getElementById('build-usd-log');
  if (!cleanGlb || !outPath) {
    status.textContent = '请填两个路径';
    status.style.color = '#f88';
    return;
  }
  status.style.color = '#aaa';
  status.textContent = 'running…';
  btn.disabled = true;
  logEl.style.display = 'none';
  logEl.textContent = '';
  setBanner('active', `导出 USDA → ${outPath} … Blender + USD 生成约 30–60 s`);

  let dots = 0;
  const tick = setInterval(() => {
    dots = (dots + 1) % 4;
    status.textContent = `running${'.'.repeat(dots)}`;
  }, 400);
  const t0 = Date.now();

  let resp, payload;
  try {
    resp = await fetch('/api/build_usd', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        labels: buildLabelsJSON(),
        clean_glb_path: cleanGlb,
        texture_glb_path: textureGlb || null,
        out_path: outPath,
      }),
    });
    const ct = resp.headers.get('content-type') || '';
    if (!resp.ok || !ct.includes('application/json')) {
      // Most common cause: serve.py is the old version without the
      // /api/build_usd handler. Fall through to a useful diagnostic.
      const text = (await resp.text()).slice(0, 200);
      throw new Error(
        `HTTP ${resp.status} (${ct || 'no content-type'}). ` +
        `服务器可能在跑旧版 serve.py — 重启它让它加载 /api/build_usd 端点 ` +
        `(Ctrl+C, 然后 \`bash scripts/tools/articulator/serve.sh\` 重新启动). ` +
        `Body: ${text}`
      );
    }
    payload = await resp.json();
  } catch (e) {
    clearInterval(tick); btn.disabled = false;
    status.textContent = 'failed';
    status.style.color = '#f88';
    setBanner('error', `导出失败: ${e.message}`);
    log('[build_usd] error: ' + e.message);
    return;
  }
  clearInterval(tick);
  btn.disabled = false;
  const dt = ((Date.now() - t0) / 1000).toFixed(1);
  logEl.style.display = 'block';
  logEl.textContent = payload.log || '';
  if (payload.ok) {
    status.textContent = `OK — ${dt}s, ${payload.size_kb} KB → ${payload.out_path}`;
    status.style.color = '#8f8';
    setBanner('active', `USDA 已导出 (${dt}s, ${payload.size_kb} KB) — ${payload.out_path}`);
    setTimeout(() => setBanner('hidden'), 5000);
    log(`[build_usd] ${dt}s — ${payload.size_kb} KB → ${payload.out_path}`);
  } else {
    status.textContent = `failed after ${dt}s (see log)`;
    status.style.color = '#f88';
    setBanner('error', `USDA 导出失败 (${dt}s) — 看下方 log`);
    log('[build_usd] failed — see log');
  }
});

document.getElementById('btn-show-cmd').addEventListener('click', () => {
  const cmd = `# 1) save labels.json next to your input GLB
# 2) (need arts-gen activated)
python scripts/tools/articulator/build_usd.py \\
    --clean_glb path/to/clean.glb \\
    --labels    path/to/labels.json \\
    --out       path/to/output.usda`;
  const box = document.getElementById('cmd-box');
  box.textContent = cmd;
  box.style.display = 'block';
});

// ---- Helpers --------------------------------------------------------------

function escapeAttr(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
