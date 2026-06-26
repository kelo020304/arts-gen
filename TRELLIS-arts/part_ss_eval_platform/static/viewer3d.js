// static/viewer3d.js  (ES module)
// Shared 3D viewer for the inference platform.
// Exposes Viewer3D + 3 isolated renderers (VoxelRenderer / MeshRenderer / GaussianRenderer),
// all sharing a uniform { load(url), clear(), dispose() } interface.
// Also attaches Viewer3D + parsePlyXyzRgb to window so the non-module app.js can use them.
import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { GLTFLoader } from './vendor/GLTFLoader.js';

const RES = 64;  // voxel grid resolution (aligned with meta.voxel_resolution; default 64)

// --- VoxelRenderer: read voxel.bin (LE uint16, flat x,y,z) -> InstancedMesh of cubes ---
// Byte format (backend inference_pipeline/voxel_io.py): c.astype("<u2").tobytes() of an (N,3)
// int32 coord array -> flat little-endian uint16 [x,y,z, x,y,z, ...]. N = array.length / 3.
export class VoxelRenderer {
  constructor(scene) { this.scene = scene; this.mesh = null; this.bodyMesh = null; }
  async load(url) {
    this.clear();
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('voxel.bin 请求失败: HTTP ' + resp.status);
    const buf = await resp.arrayBuffer();
    const a = new Uint16Array(buf);                 // LE uint16, length = N*3
    const n = Math.floor(a.length / 3);
    if (!n) throw new Error('voxel.bin 为空');
    // Cubes smaller than the 1-unit cell so neighbours DON'T share faces: leaves
    // a thin gap (background shows through) -> the real "一格一格" voxel-grid look
    // instead of a merged solid surface. flatShading keeps each face crisp.
    const CELL = 0.86;
    const geo = new THREE.BoxGeometry(CELL, CELL, CELL);
    // Grey + semi-transparent: these are SURFACE voxels (a hollow shell); the
    // translucency lets you see through the front cells to the back ones, so it
    // reads as a surface-only occupancy rather than a solid filled block.
    const mat = new THREE.MeshLambertMaterial({
      color: 0xb9bdc8, flatShading: true, transparent: true, opacity: 0.5,
      depthWrite: false,
    });
    const mesh = new THREE.InstancedMesh(geo, mat, n);
    const m = new THREE.Matrix4();
    for (let i = 0; i < n; i++) {
      // center the 0..63 grid around the origin by subtracting RES/2
      m.setPosition(a[i * 3] - RES / 2, a[i * 3 + 1] - RES / 2, a[i * 3 + 2] - RES / 2);
      mesh.setMatrixAt(i, m);
    }
    mesh.instanceMatrix.needsUpdate = true;
    // Voxel data is TRELLIS Z-up; three.js is Y-up. Rotate -90° about X so the
    // object STANDS UP (was lying face-down) — Z-up -> Y-up. Only the voxel needs
    // this; part mesh/GS arrive as Y-up glb/ply.
    mesh.rotation.x = -Math.PI / 2;
    this.mesh = mesh; this.scene.add(mesh);
    return { count: n };
  }
  // Labeled voxels: LE uint16 [x,y,z,label, ...] (4 per voxel).
  //  label === BODY_LABEL (65535) -> grey translucent BODY context (the whole SS
  //    structure minus the target parts), like the eval's "GT body context".
  //  label 0,1,2,...           -> one PALETTE colour per target part (opaque), so
  //    opener_0 / opener_1 / ... read as distinct solids on top of the grey body.
  async loadLabeled(url) {
    this.clear();
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('part voxel 请求失败: HTTP ' + resp.status);
    const a = new Uint16Array(await resp.arrayBuffer());   // [x,y,z,label, ...]
    const n = Math.floor(a.length / 4);
    if (!n) throw new Error('part voxel 为空');
    const BODY = 65535;
    const PALETTE = [
      0x4f8cff, 0xff6b6b, 0x4ec9a8, 0xffb454,
      0xb084ff, 0xff8ad1, 0x9acd32, 0x00bcd4,
    ];
    const bodyIdx = [], partIdx = [];
    for (let i = 0; i < n; i++) (a[i * 4 + 3] === BODY ? bodyIdx : partIdx).push(i);
    // 一组体素 -> 一个 InstancedMesh；colorFn(label) 给每实例上色。
    const build = (idxs, colorFn, opacity, depthWrite) => {
      if (!idxs.length) return null;
      const geo = new THREE.BoxGeometry(0.86, 0.86, 0.86);
      const mat = new THREE.MeshLambertMaterial({ flatShading: true, transparent: true, opacity, depthWrite });
      const mesh = new THREE.InstancedMesh(geo, mat, idxs.length);
      const m = new THREE.Matrix4(), col = new THREE.Color();
      idxs.forEach((s, j) => {
        m.setPosition(a[s * 4] - RES / 2, a[s * 4 + 1] - RES / 2, a[s * 4 + 2] - RES / 2);
        mesh.setMatrixAt(j, m);
        col.setHex(colorFn(a[s * 4 + 3])); mesh.setColorAt(j, col);
      });
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      mesh.rotation.x = -Math.PI / 2;   // Z-up -> Y-up, same as the grey voxel
      return mesh;
    };
    // body 先加（灰、半透明、不写深度，让彩色 part 透出来）；part 后加（彩、较实）。
    this.bodyMesh = build(bodyIdx, () => 0x9aa3b5, 0.34, false);
    this.mesh = build(partIdx, (lbl) => PALETTE[lbl % PALETTE.length], 0.92, true);
    if (this.bodyMesh) this.scene.add(this.bodyMesh);
    if (this.mesh) this.scene.add(this.mesh);
    return { count: partIdx.length + bodyIdx.length, partCount: partIdx.length, bodyCount: bodyIdx.length };
  }
  clear() {
    for (const key of ['mesh', 'bodyMesh']) {
      const m = this[key];
      if (m) {
        this.scene.remove(m);
        m.geometry.dispose();
        m.material.dispose();
        this[key] = null;
      }
    }
  }
  dispose() { this.clear(); }
}

// --- MeshRenderer: GLTFLoader loads *.glb ---
export class MeshRenderer {
  constructor(scene) {
    this.scene = scene;
    this.root = null;
    this.loader = new GLTFLoader();
    this.parts = [];
  }
  load(url) {
    this.clear();
    const isBody = isBodyMeshUrl(url);
    return this._loadOne(url, 0, { isBody }).then(({ scene, vertices, triangles, color }) => {
      this.root = scene;
      this.parts = [{
        index: 0,
        body: isBody,
        stem: meshStemFromUrl(url),
        object: scene,
        color,
        vertices,
        triangles,
      }];
      this.scene.add(scene);
      return {
        ok: true,
        count: vertices,
        triangles,
        parts: 1,
        partStats: this.parts.map(part => meshPartStat(part)),
      };
    });
  }
  async loadMany(urls) {
    this.clear();
    if (!Array.isArray(urls) || urls.length === 0) throw new Error('mesh URL 列表为空');
    const group = new THREE.Group();
    let vertices = 0, triangles = 0;
    const parts = [];
    for (let i = 0; i < urls.length; i++) {
      const isBody = isBodyMeshUrl(urls[i]);
      const loaded = await this._loadOne(urls[i], i, { isBody });
      vertices += loaded.vertices;
      triangles += loaded.triangles;
      parts.push({
        index: i,
        body: isBody,
        stem: meshStemFromUrl(urls[i]),
        object: loaded.scene,
        color: loaded.color,
        vertices: loaded.vertices,
        triangles: loaded.triangles,
      });
      group.add(loaded.scene);
    }
    this.root = group;
    this.parts = parts;
    this.scene.add(group);
    return { ok: true, count: vertices, triangles, parts: urls.length, partStats: parts.map(part => meshPartStat(part)) };
  }
  _loadOne(url, partIndex, opts = {}) {
    return new Promise((res, rej) => this.loader.load(url, g => {
      const scene = g.scene;
      let vertices = 0;
      let triangles = 0;
      const colorAcc = { r: 0, g: 0, b: 0, n: 0 };
      scene.traverse(obj => {
        const geo = obj.geometry;
        if (!geo) return;
        const pos = geo.getAttribute('position');
        vertices += pos ? pos.count : 0;
        triangles += geo.index ? Math.floor(geo.index.count / 3) : (pos ? Math.floor(pos.count / 3) : 0);
        applyMeshMaterialDefaults(obj);
        accumulateMeshColor(obj, colorAcc);
        obj.userData.meshPartIndex = partIndex;
        obj.userData.meshPartBody = Boolean(opts.isBody);
      });
      scene.userData.meshPartIndex = partIndex;
      scene.userData.meshPartBody = Boolean(opts.isBody);
      res({ scene, vertices, triangles, color: colorFromAccum(colorAcc, opts.isBody) });
    }, undefined, err => rej(err)));
  }
  highlight(partIndex) {
    const selected = Number.isInteger(partIndex) ? partIndex : null;
    this.parts.forEach(part => {
      const isSelected = selected === null || part.index === selected;
      part.object.visible = true;
      part.object.scale.setScalar(selected === null ? 1 : (isSelected ? 1.025 : 0.998));
      part.object.traverse(obj => {
        if (!obj.material) return;
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        mats.forEach(mtl => {
          if (!mtl.userData.__meshLegendBase) {
            mtl.userData.__meshLegendBase = {
              opacity: mtl.opacity,
              transparent: mtl.transparent,
              emissive: mtl.emissive ? mtl.emissive.clone() : null,
              color: mtl.color ? mtl.color.clone() : null,
            };
          }
          const base = mtl.userData.__meshLegendBase;
          if (selected === null) {
            mtl.opacity = base.opacity;
            mtl.transparent = base.transparent;
            if (mtl.emissive && base.emissive) mtl.emissive.copy(base.emissive);
            if (mtl.color && base.color && !obj.geometry?.getAttribute('color')) mtl.color.copy(base.color);
          } else if (isSelected) {
            mtl.opacity = 1;
            mtl.transparent = base.transparent;
            if (mtl.emissive) mtl.emissive.setHex(0x353020);
          } else {
            mtl.opacity = part.body ? 0.18 : 0.28;
            mtl.transparent = true;
            if (mtl.emissive) mtl.emissive.setHex(0x000000);
          }
          mtl.needsUpdate = true;
        });
      });
    });
  }
  clear() {
    if (this.root) {
      this.scene.remove(this.root);
      this.root.traverse(obj => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
          mats.forEach(mtl => { if (mtl.map) mtl.map.dispose(); mtl.dispose(); });
        }
      });
      this.root = null;
    }
    this.parts = [];
  }
  dispose() { this.clear(); }
}

function meshPartStat(part) {
  return {
    index: part.index,
    body: part.body,
    stem: part.stem,
    color: part.color,
    vertices: part.vertices,
    triangles: part.triangles,
  };
}

function isBodyMeshUrl(url) {
  const value = String(url || "");
  try {
    const parsed = new URL(value, window.location.href);
    const rel = parsed.searchParams.get("rel") || parsed.pathname;
    return /(^|\/)body\.glb$/i.test(rel);
  } catch (err) {
    return /(^|\/)body\.glb($|\?)/i.test(value);
  }
}

function meshStemFromUrl(url) {
  const value = String(url || "");
  try {
    const parsed = new URL(value, window.location.href);
    const rel = parsed.searchParams.get("rel") || parsed.pathname;
    return (rel.split("/").pop() || "mesh").replace(/\.[^.]+$/, "");
  } catch (err) {
    return (value.split("?")[0].split("/").pop() || "mesh").replace(/\.[^.]+$/, "");
  }
}

function accumulateMeshColor(obj, acc) {
  const geo = obj.geometry;
  const colors = geo?.getAttribute?.('color');
  if (colors && colors.count) {
    const step = Math.max(1, Math.floor(colors.count / 1200));
    for (let i = 0; i < colors.count; i += step) {
      acc.r += colors.getX(i);
      acc.g += colors.getY(i);
      acc.b += colors.getZ(i);
      acc.n += 1;
    }
    return;
  }
  const mat = Array.isArray(obj.material) ? obj.material[0] : obj.material;
  if (mat?.color) {
    acc.r += mat.color.r;
    acc.g += mat.color.g;
    acc.b += mat.color.b;
    acc.n += 1;
  }
}

function colorFromAccum(acc, isBody) {
  if (!acc.n) return isBody ? "#8f98aa" : "#9aa8ff";
  const c = new THREE.Color(acc.r / acc.n, acc.g / acc.n, acc.b / acc.n);
  return "#" + c.getHexString();
}

function applyMeshMaterialDefaults(obj) {
  if (!obj.material) {
    const hasVertexColor = Boolean(obj.geometry?.getAttribute('color'));
    obj.material = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      vertexColors: hasVertexColor,
      roughness: 0.82,
      metalness: 0.04,
      side: THREE.DoubleSide,
      flatShading: false,
    });
    return;
  }
  const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
  mats.forEach(mtl => {
    mtl.side = THREE.DoubleSide;
    if ('roughness' in mtl && mtl.roughness === undefined) mtl.roughness = 0.82;
    if ('metalness' in mtl && mtl.metalness === undefined) mtl.metalness = 0.04;
    if (obj.geometry?.getAttribute('color') && 'vertexColors' in mtl) {
      mtl.vertexColors = true;
    }
    mtl.needsUpdate = true;
  });
}

// --- GaussianRenderer: read gaussian .ply centers + colors -> THREE.Points
// (isolated behind the interface so it can later be swapped for a splat shader) ---
export class GaussianRenderer {
  constructor(scene) { this.scene = scene; this.points = null; }
  async load(url) {
    this.clear();
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('ply 请求失败: HTTP ' + resp.status);
    const buf = await resp.arrayBuffer();
    const { positions, colors } = parsePlyXyzRgb(buf);   // parse ply vertices x,y,z + (f_dc/rgb)
    if (!positions.length) throw new Error('ply 无顶点');
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    if (colors.length) geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({ size: 0.02, vertexColors: colors.length > 0 });
    this.points = new THREE.Points(geo, mat); this.scene.add(this.points);
    return { count: positions.length / 3 };
  }
  clear() {
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      this.points.material.dispose();
      this.points = null;
    }
  }
  dispose() { this.clear(); }
}

// Minimal PLY parser (ascii + binary_little_endian), extracts x,y,z + (red,green,blue | f_dc_0..2).
// SH band-0 -> RGB via 0.5 + 0.2820948 * f_dc, clamped to [0,1]. No color props -> colors=[].
// Errors surface (no silent fallback). Returns { positions:[], colors:[] } as flat Number arrays.
const SH_C0 = 0.2820948;  // 0.5 / sqrt(pi) -> SH band-0 basis constant

// byte size per PLY scalar type
const PLY_TYPE_BYTES = {
  char: 1, uchar: 1, int8: 1, uint8: 1,
  short: 2, ushort: 2, int16: 2, uint16: 2,
  int: 4, uint: 4, int32: 4, uint32: 4,
  float: 4, float32: 4,
  double: 8, float64: 8,
};

function clamp01(v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

export function parsePlyXyzRgb(buf) {
  const bytes = new Uint8Array(buf);

  // --- locate the end of the ASCII header ("end_header\n") ---
  const headerMarker = 'end_header';
  let headerEnd = -1;
  // scan as latin1 text up to a reasonable header cap
  const scanLen = Math.min(bytes.length, 1 << 20);
  let headerText = '';
  for (let i = 0; i < scanLen; i++) headerText += String.fromCharCode(bytes[i]);
  const markerIdx = headerText.indexOf(headerMarker);
  if (markerIdx < 0) throw new Error('PLY 缺少 end_header');
  // header body ends after the newline that follows the marker
  let nlIdx = headerText.indexOf('\n', markerIdx);
  if (nlIdx < 0) throw new Error('PLY end_header 后缺少换行');
  headerEnd = nlIdx + 1;
  const header = headerText.slice(0, headerEnd);

  // --- parse header: format + vertex element + its property list (in order) ---
  const lines = header.split('\n');
  let format = null;          // 'ascii' | 'binary_little_endian' | 'binary_big_endian'
  let vertexCount = 0;
  let inVertex = false;
  let sawVertexElement = false;
  const props = [];           // [{ name, type }] for the vertex element, in declared order

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    const tok = line.split(/\s+/);
    if (tok[0] === 'format') {
      format = tok[1];
    } else if (tok[0] === 'element') {
      // entering a new element block; only collect properties of the 'vertex' element
      inVertex = (tok[1] === 'vertex');
      if (inVertex) { sawVertexElement = true; vertexCount = parseInt(tok[2], 10); }
    } else if (tok[0] === 'property' && inVertex) {
      // 'property <type> <name>'  (vertex elements have no list properties)
      props.push({ type: tok[1], name: tok[tok.length - 1] });
    }
  }

  if (!format) throw new Error('PLY 缺少 format 行');
  if (!sawVertexElement) throw new Error('PLY 缺少 vertex element');
  if (!vertexCount) return { positions: [], colors: [] };

  // index the properties we care about
  const idx = {};
  props.forEach((p, i) => { idx[p.name] = i; });
  const has = name => Object.prototype.hasOwnProperty.call(idx, name);
  if (!has('x') || !has('y') || !has('z')) throw new Error('PLY vertex 缺少 x/y/z 属性');

  const hasRGB = has('red') && has('green') && has('blue');
  const hasFdc = has('f_dc_0') && has('f_dc_1') && has('f_dc_2');

  const positions = [];
  const colors = [];

  if (format === 'ascii') {
    // body = whitespace-separated numeric tokens, vertexCount rows of props.length values
    const body = headerText.length >= bytes.length
      ? headerText.slice(headerEnd)
      : decodeAsciiBody(bytes, headerEnd);
    const nums = body.split(/\s+/).filter(s => s.length > 0);
    const stride = props.length;
    if (nums.length < vertexCount * stride) {
      throw new Error('PLY ascii 数据不足: 期望 ' + (vertexCount * stride) + ' 个值，实得 ' + nums.length);
    }
    for (let v = 0; v < vertexCount; v++) {
      const base = v * stride;
      positions.push(
        parseFloat(nums[base + idx.x]),
        parseFloat(nums[base + idx.y]),
        parseFloat(nums[base + idx.z]),
      );
      if (hasRGB) {
        colors.push(
          parseFloat(nums[base + idx.red]) / 255,
          parseFloat(nums[base + idx.green]) / 255,
          parseFloat(nums[base + idx.blue]) / 255,
        );
      } else if (hasFdc) {
        colors.push(
          clamp01(0.5 + SH_C0 * parseFloat(nums[base + idx.f_dc_0])),
          clamp01(0.5 + SH_C0 * parseFloat(nums[base + idx.f_dc_1])),
          clamp01(0.5 + SH_C0 * parseFloat(nums[base + idx.f_dc_2])),
        );
      }
    }
    return { positions, colors };
  }

  if (format === 'binary_little_endian') {
    // compute byte stride + per-property offsets within one vertex record
    let stride = 0;
    const offsets = new Array(props.length);
    for (let i = 0; i < props.length; i++) {
      const sz = PLY_TYPE_BYTES[props[i].type];
      if (sz === undefined) throw new Error('PLY 不支持的属性类型: ' + props[i].type);
      offsets[i] = stride;
      stride += sz;
    }
    const dv = new DataView(buf, headerEnd);
    const needed = vertexCount * stride;
    if (dv.byteLength < needed) {
      throw new Error('PLY binary 数据不足: 期望 ' + needed + ' 字节，实得 ' + dv.byteLength);
    }
    const readScalar = (recBase, pIdx) => readLE(dv, recBase + offsets[pIdx], props[pIdx].type);
    for (let v = 0; v < vertexCount; v++) {
      const recBase = v * stride;
      positions.push(
        readScalar(recBase, idx.x),
        readScalar(recBase, idx.y),
        readScalar(recBase, idx.z),
      );
      if (hasRGB) {
        colors.push(
          readScalar(recBase, idx.red) / 255,
          readScalar(recBase, idx.green) / 255,
          readScalar(recBase, idx.blue) / 255,
        );
      } else if (hasFdc) {
        colors.push(
          clamp01(0.5 + SH_C0 * readScalar(recBase, idx.f_dc_0)),
          clamp01(0.5 + SH_C0 * readScalar(recBase, idx.f_dc_1)),
          clamp01(0.5 + SH_C0 * readScalar(recBase, idx.f_dc_2)),
        );
      }
    }
    return { positions, colors };
  }

  throw new Error('PLY 不支持的 format: ' + format);
}

// decode the binary body region as latin1 text (used only when the ascii scan was capped)
function decodeAsciiBody(bytes, start) {
  let s = '';
  for (let i = start; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return s;
}

// read one little-endian scalar of the given PLY type from a DataView at byteOffset
function readLE(dv, off, type) {
  switch (type) {
    case 'char': case 'int8': return dv.getInt8(off);
    case 'uchar': case 'uint8': return dv.getUint8(off);
    case 'short': case 'int16': return dv.getInt16(off, true);
    case 'ushort': case 'uint16': return dv.getUint16(off, true);
    case 'int': case 'int32': return dv.getInt32(off, true);
    case 'uint': case 'uint32': return dv.getUint32(off, true);
    case 'float': case 'float32': return dv.getFloat32(off, true);
    case 'double': case 'float64': return dv.getFloat64(off, true);
    default: throw new Error('PLY 不支持的属性类型: ' + type);
  }
}

export class Viewer3D {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f1220);
    const w = container.clientWidth || 480, h = container.clientHeight || 360;
    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
    // Default front view: camera on +X so the model's +X axis faces the user.
    this.camera.position.set(96, 26, 0);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.setSize(w, h);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.45;
    container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.65));
    const key = new THREE.DirectionalLight(0xffffff, 1.35); key.position.set(3, 2, 1); this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.9); fill.position.set(-2, 1, -2); this.scene.add(fill);
    this.voxel = new VoxelRenderer(this.scene);
    this.mesh = new MeshRenderer(this.scene);
    this.gauss = new GaussianRenderer(this.scene);
    this._raf = null;
    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);
    this._loop();
  }
  _loop() {
    this._raf = requestAnimationFrame(() => this._loop());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
  resize() {
    const w = this.container.clientWidth || 480, h = this.container.clientHeight || 360;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }
  async show(kind, url) {  // kind: 'voxel' | 'mesh' | 'gaussian'
    this.voxel.clear(); this.mesh.clear(); this.gauss.clear();
    if (kind === 'voxel') return this.voxel.load(url);
    if (kind === 'partvoxel') return this.voxel.loadLabeled(url);
    if (kind === 'mesh') {
      const result = Array.isArray(url) ? await this.mesh.loadMany(url) : await this.mesh.load(url);
      this.frameObject(this.mesh.root);
      return result;
    }
    if (kind === 'gaussian') return this.gauss.load(url);
    throw new Error('未知渲染类型：' + kind);
  }
  highlightMeshPart(partIndex) {
    this.mesh.highlight(Number.isInteger(partIndex) ? partIndex : null);
  }
  frameObject(object) {
    if (!object) return;
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z, 0.1);
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(radius * 2.45, radius * 0.65, 0));
    this.camera.near = Math.max(radius / 100, 0.001);
    this.camera.far = Math.max(radius * 20, 10);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }
  clearAll() { this.voxel.clear(); this.mesh.clear(); this.gauss.clear(); }
  dispose() {
    if (this._raf !== null) cancelAnimationFrame(this._raf);
    this._raf = null;
    window.removeEventListener('resize', this._onResize);
    this.voxel.dispose(); this.mesh.dispose(); this.gauss.dispose();
    this.controls.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement && this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}

// Attach to window so the non-module app.js can use them (INTEGRATION STYLE, low risk).
window.Viewer3D = Viewer3D;
window.parsePlyXyzRgb = parsePlyXyzRgb;
