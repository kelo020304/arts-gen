const params = new URLSearchParams(window.location.search);
const resultUrl = params.get("result");
const stateEl = document.getElementById("state");
const selectEl = document.getElementById("partSelect");
const playButton = document.getElementById("playButton");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101210);
const objectRoot = new THREE.Group();
scene.add(objectRoot);
const camera = new THREE.PerspectiveCamera(42, 1, 0.001, 100);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.outputEncoding = THREE.sRGBEncoding;
document.body.appendChild(renderer.domElement);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x424942, 1.25));
const key = new THREE.DirectionalLight(0xffffff, 1.2);
key.position.set(2, -3, 4);
scene.add(key);
const loader = new THREE.GLTFLoader();
const records = [];
let activeIndex = 0;
let playing = true;

function resourceUrl(value) {
  if (!value) return null;
  if (/^(data:|blob:|https?:)/.test(value)) return value;
  let base;
  if (resultUrl.startsWith("/")) {
    const staticIndex = window.location.pathname.indexOf("/static/");
    const appPath = staticIndex >= 0 ? window.location.pathname.slice(0, staticIndex + 1) : "/";
    base = new URL(`${appPath}${resultUrl.slice(1)}`, window.location.origin);
  } else {
    base = new URL(resultUrl, window.location.href);
  }
  if (value.startsWith("/")) {
    const apiIndex = base.pathname.indexOf("/api/");
    const appPath = apiIndex >= 0 ? base.pathname.slice(0, apiIndex + 1) : "/";
    return new URL(`${appPath}${value.slice(1)}`, base.origin).toString();
  }
  return new URL(value, base).toString();
}

function loadGlb(url) {
  return new Promise((resolve, reject) => loader.load(url, (gltf) => resolve(gltf.scene), undefined, reject));
}

function vector(values, fallback = [1, 0, 0]) {
  const raw = Array.isArray(values) && values.length === 3 ? values : fallback;
  const out = new THREE.Vector3(...raw.map(Number));
  return out.lengthSq() > 1e-12 ? out.normalize() : new THREE.Vector3(...fallback);
}

async function addPart(item, index) {
  const model = await loadGlb(resourceUrl(item.mesh_url));
  const candidate = item.delivery_candidate || item.candidate || {};
  const group = new THREE.Group();
  const axis = vector(candidate.axis_world);
  const origin = new THREE.Vector3(...(candidate.origin_world || [0, 0, 0]).map(Number));
  if (candidate.joint_type === "revolute") {
    group.position.copy(origin);
    model.position.sub(origin);
  }
  group.add(model);
  objectRoot.add(group);
  const axisConfidence = Number(item.candidate?.signals?.axis_confidence || 0);
  const guideColor = axisConfidence >= 0.7 ? 0x39b86b : axisConfidence >= 0.4 ? 0xf0b43c : 0xe05a47;
  const guide = new THREE.Group();
  const arrow = new THREE.ArrowHelper(axis, origin, 1, guideColor, 0.12, 0.07);
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.02, 16, 10),
    new THREE.MeshBasicMaterial({ color: guideColor }),
  );
  marker.position.copy(origin);
  guide.add(arrow, marker);
  objectRoot.add(guide);
  records.push({ item, index, group, guide, arrow, marker, axis, origin, lower: Number(candidate.lower || 0), upper: Number(candidate.upper || 0) });
}

function fitCamera() {
  records.forEach((record) => { record.guide.visible = false; });
  const box = new THREE.Box3().setFromObject(scene);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  records.forEach((record, index) => {
    record.arrow.setLength(size * 0.32, size * 0.045, size * 0.025);
    record.marker.scale.setScalar(size * 0.8);
    record.guide.visible = index === activeIndex;
  });
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(size * 0.9, -size * 1.1, size * 0.75));
  camera.near = Math.max(size / 1000, 0.001);
  camera.far = Math.max(size * 10, 10);
  camera.updateProjectionMatrix();
  controls.update();
}

function resetMotion() {
  records.forEach((record) => {
    const candidate = record.item.delivery_candidate || record.item.candidate || {};
    record.group.position.copy(candidate.joint_type === "revolute" ? record.origin : new THREE.Vector3());
    record.group.quaternion.identity();
  });
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
}

function animate(time) {
  requestAnimationFrame(animate);
  resetMotion();
  const record = records[activeIndex];
  records.forEach((item, index) => { item.guide.visible = index === activeIndex; });
  if (record && playing) {
    const phase = (Math.sin(time * 0.0012) + 1) * 0.5;
    const q = record.lower + (record.upper - record.lower) * phase;
    const candidate = record.item.delivery_candidate || record.item.candidate || {};
    if (candidate.joint_type === "prismatic") {
      record.group.position.copy(record.axis).multiplyScalar(q);
    } else {
      record.group.quaternion.setFromAxisAngle(record.axis, q);
    }
    const axisConfidence = Number(record.item.candidate?.signals?.axis_confidence || 0);
    stateEl.textContent = `${record.item.label} · ${candidate.joint_type} · q ${q.toFixed(4)} · axis confidence ${axisConfidence.toFixed(2)}`;
  }
  controls.update();
  renderer.render(scene, camera);
}

async function init() {
  if (!resultUrl) throw new Error("Missing result URL");
  const response = await fetch(resultUrl);
  if (!response.ok) throw new Error(`Result request failed (${response.status})`);
  const result = await response.json();
  if (!result.body_mesh_url) throw new Error("Decoded body mesh is unavailable");
  if (result.apply_root_correction) objectRoot.rotation.x = Math.PI * 0.5;
  objectRoot.add(await loadGlb(resourceUrl(result.body_mesh_url)));
  const parts = (result.parts || []).filter((item) => item.mesh_url);
  await Promise.all(parts.map(addPart));
  records.sort((a, b) => a.index - b.index);
  selectEl.replaceChildren(...records.map((record, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = record.item.label;
    return option;
  }));
  selectEl.addEventListener("change", () => { activeIndex = Number(selectEl.value || 0); });
  fitCamera();
  stateEl.textContent = `${records.length} predicted joints`;
}

playButton.addEventListener("click", () => {
  playing = !playing;
  playButton.textContent = playing ? "Ⅱ" : "▶";
  playButton.title = playing ? "Pause motion" : "Play motion";
});
window.addEventListener("resize", resize);
resize();
animate(0);
init().catch((error) => { stateEl.textContent = error.message || String(error); });
