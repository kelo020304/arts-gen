const RESOLUTION = 64;
const DEFAULT_AZIMUTH = -45 * Math.PI / 180;
const DEFAULT_ELEVATION = 30 * Math.PI / 180;
const DEFAULT_ZOOM = 0.92;

const FACE_DEFINITIONS = [
  { delta: [-1, 0, 0], normal: [-1, 0, 0], corners: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]] },
  { delta: [1, 0, 0], normal: [1, 0, 0], corners: [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]] },
  { delta: [0, -1, 0], normal: [0, -1, 0], corners: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]] },
  { delta: [0, 1, 0], normal: [0, 1, 0], corners: [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]] },
  { delta: [0, 0, -1], normal: [0, 0, -1], corners: [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]] },
  { delta: [0, 0, 1], normal: [0, 0, 1], corners: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]] },
];

const VERTEX_SHADER = `#version 300 es
precision highp float;
in vec3 aPosition;
in vec3 aNormal;
uniform vec3 uRight;
uniform vec3 uUp;
uniform vec3 uView;
uniform vec2 uViewport;
uniform float uScale;
out float vLight;
void main() {
  vec3 relative = aPosition - vec3(32.0);
  vec2 screen = vec2(dot(relative, uRight), dot(relative, uUp)) * uScale;
  float depth = -dot(relative, uView) / 64.0;
  gl_Position = vec4(2.0 * screen.x / uViewport.x, 2.0 * screen.y / uViewport.y, depth, 1.0);
  vLight = 0.72 + 0.20 * max(0.0, dot(aNormal, normalize(vec3(-0.35, -0.25, 0.90))));
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
in float vLight;
uniform vec4 uColor;
uniform bool uUseLight;
out vec4 outColor;
void main() {
  float light = uUseLight ? vLight : 1.0;
  outColor = vec4(uColor.rgb * light, uColor.a);
}`;

function coordKey(x, y, z) {
  return x * RESOLUTION * RESOLUTION + y * RESOLUTION + z;
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function colorRgb(value) {
  const match = String(value || "").match(/^#([0-9a-f]{6})$/i);
  if (!match) return [168 / 255, 168 / 255, 168 / 255];
  const packed = Number.parseInt(match[1], 16);
  return [((packed >> 16) & 255) / 255, ((packed >> 8) & 255) / 255, (packed & 255) / 255];
}

function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Voxel viewer shader failed: ${message}`);
  }
  return shader;
}

function createProgram(gl) {
  const program = gl.createProgram();
  const vertex = createShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
  const fragment = createShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`Voxel viewer program failed: ${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

export class VoxelViewer {
  constructor(canvas, stats, overlay = null) {
    this.canvas = canvas;
    this.stats = stats;
    this.overlay = overlay || canvas.parentElement.querySelector(".ssVoxelOverlay");
    this.overlayCtx = this.overlay?.getContext("2d") || null;
    this.gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: true,
      depth: true,
      powerPreference: "high-performance",
    });
    if (!this.gl) throw new Error("WebGL2 is required for the voxel viewer");
    this.program = createProgram(this.gl);
    this.locations = this.getLocations();
    this.positionBuffer = this.gl.createBuffer();
    this.normalBuffer = this.gl.createBuffer();
    this.fillIndexBuffer = this.gl.createBuffer();
    this.edgeIndexBuffer = this.gl.createBuffer();
    this.fillIndexCount = 0;
    this.edgeIndexCount = 0;
    this.faceCount = 0;
    this.layers = [];
    this.totalVoxelCount = 0;
    this.azimuth = DEFAULT_AZIMUTH;
    this.elevation = DEFAULT_ELEVATION;
    this.zoom = DEFAULT_ZOOM;
    this.dragging = false;
    this.lastPointer = [0, 0];
    this.framePending = false;
    this.configureGl();
    this.bindInteractions();
    this.resizeObserver = new ResizeObserver(() => this.requestDraw());
    this.resizeObserver.observe(canvas);
  }

  getLocations() {
    const gl = this.gl;
    return {
      position: gl.getAttribLocation(this.program, "aPosition"),
      normal: gl.getAttribLocation(this.program, "aNormal"),
      right: gl.getUniformLocation(this.program, "uRight"),
      up: gl.getUniformLocation(this.program, "uUp"),
      view: gl.getUniformLocation(this.program, "uView"),
      viewport: gl.getUniformLocation(this.program, "uViewport"),
      scale: gl.getUniformLocation(this.program, "uScale"),
      color: gl.getUniformLocation(this.program, "uColor"),
      useLight: gl.getUniformLocation(this.program, "uUseLight"),
    };
  }

  configureGl() {
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.clearColor(245 / 255, 246 / 255, 245 / 255, 1);
  }

  setData(payload) {
    const sourceLayers = Array.isArray(payload.layers) && payload.layers.length
      ? payload.layers
      : [{ id: "whole", label: "whole object", color: "#a8a8a8", visible: true, coords: payload.coords || [] }];
    const seen = new Set();
    const layers = sourceLayers.map((layer, index) => {
      const coords = [];
      for (const coord of Array.isArray(layer.coords) ? layer.coords : []) {
        if (!Array.isArray(coord) || coord.length < 3
          || coord[0] < 0 || coord[0] >= RESOLUTION
          || coord[1] < 0 || coord[1] >= RESOLUTION
          || coord[2] < 0 || coord[2] >= RESOLUTION) continue;
        const key = coordKey(coord[0], coord[1], coord[2]);
        if (seen.has(key)) continue;
        seen.add(key);
        coords.push(coord);
      }
      return {
        id: String(layer.id ?? index),
        label: String(layer.label || layer.id || `layer ${index + 1}`),
        color: colorRgb(layer.color),
        visible: layer.visible !== false,
        voxelCount: Number(layer.voxel_count ?? coords.length),
        coords,
      };
    });
    this.uploadSurface(layers);
    this.totalVoxelCount = Number(payload.voxel_count || layers.reduce((sum, layer) => sum + layer.voxelCount, 0));
    this.updateStats();
    this.resetView();
  }

  setLayerVisible(layerId, visible) {
    const layer = this.layers.find((item) => item.id === String(layerId));
    if (!layer || layer.visible === !!visible) return;
    layer.visible = !!visible;
    this.updateStats();
    this.requestDraw();
  }

  updateStats() {
    const visible = this.layers.filter((layer) => layer.visible);
    const visibleVoxels = visible.reduce((sum, layer) => sum + layer.voxelCount, 0);
    const layerSuffix = this.layers.length > 1
      ? ` · ${visible.length}/${this.layers.length} labels visible · ${visibleVoxels.toLocaleString()} visible voxels`
      : "";
    this.stats.textContent = `${this.totalVoxelCount.toLocaleString()} occupied voxels${layerSuffix} · ${this.faceCount.toLocaleString()} label-surface faces · drag to orbit · wheel to zoom · double-click to reset`;
  }

  uploadSurface(layerInputs) {
    const labelByCoord = new Map();
    layerInputs.forEach((layer, index) => {
      for (const coord of layer.coords) labelByCoord.set(coordKey(coord[0], coord[1], coord[2]), index);
    });
    const positions = [];
    const normals = [];
    const fillIndices = [];
    const edgeIndices = [];
    let vertexOffset = 0;
    let faceCount = 0;
    const layerRecords = [];
    layerInputs.forEach((layer, layerIndex) => {
      const fillStart = fillIndices.length;
      const edgeStart = edgeIndices.length;
      let layerFaces = 0;
      for (const [x, y, z] of layer.coords) {
        for (const definition of FACE_DEFINITIONS) {
          const nx = x + definition.delta[0];
          const ny = y + definition.delta[1];
          const nz = z + definition.delta[2];
          const neighborInside = nx >= 0 && nx < RESOLUTION
            && ny >= 0 && ny < RESOLUTION
            && nz >= 0 && nz < RESOLUTION;
          if (neighborInside && labelByCoord.get(coordKey(nx, ny, nz)) === layerIndex) continue;
          for (const corner of definition.corners) {
            positions.push(x + corner[0], y + corner[1], z + corner[2]);
            normals.push(...definition.normal);
          }
          fillIndices.push(vertexOffset, vertexOffset + 1, vertexOffset + 2, vertexOffset, vertexOffset + 2, vertexOffset + 3);
          edgeIndices.push(
            vertexOffset, vertexOffset + 1,
            vertexOffset + 1, vertexOffset + 2,
            vertexOffset + 2, vertexOffset + 3,
            vertexOffset + 3, vertexOffset,
          );
          vertexOffset += 4;
          faceCount += 1;
          layerFaces += 1;
        }
      }
      layerRecords.push({
        ...layer,
        faceCount: layerFaces,
        fillOffset: fillStart * Uint32Array.BYTES_PER_ELEMENT,
        fillCount: fillIndices.length - fillStart,
        edgeOffset: edgeStart * Uint32Array.BYTES_PER_ELEMENT,
        edgeCount: edgeIndices.length - edgeStart,
      });
    });
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.fillIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(fillIndices), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.edgeIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(edgeIndices), gl.STATIC_DRAW);
    this.fillIndexCount = fillIndices.length;
    this.edgeIndexCount = edgeIndices.length;
    this.faceCount = faceCount;
    this.layers = layerRecords;
  }

  resetView() {
    this.azimuth = DEFAULT_AZIMUTH;
    this.elevation = DEFAULT_ELEVATION;
    this.zoom = DEFAULT_ZOOM;
    this.requestDraw();
  }

  bindInteractions() {
    this.canvas.addEventListener("pointerdown", (event) => {
      this.dragging = true;
      this.lastPointer = [event.clientX, event.clientY];
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (!this.dragging) return;
      const dx = event.clientX - this.lastPointer[0];
      const dy = event.clientY - this.lastPointer[1];
      this.lastPointer = [event.clientX, event.clientY];
      this.azimuth += dx * 0.009;
      this.elevation = Math.max(-1.35, Math.min(1.35, this.elevation + dy * 0.009));
      this.requestDraw();
    });
    const stopDrag = () => { this.dragging = false; };
    this.canvas.addEventListener("pointerup", stopDrag);
    this.canvas.addEventListener("pointercancel", stopDrag);
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.zoom = Math.max(0.35, Math.min(3.5, this.zoom * Math.exp(-event.deltaY * 0.0012)));
      this.requestDraw();
    }, { passive: false });
    this.canvas.addEventListener("dblclick", () => this.resetView());
  }

  requestDraw() {
    if (this.framePending) return;
    this.framePending = true;
    requestAnimationFrame(() => {
      this.framePending = false;
      this.draw();
    });
  }

  camera() {
    const ca = Math.cos(this.azimuth);
    const sa = Math.sin(this.azimuth);
    const ce = Math.cos(this.elevation);
    const se = Math.sin(this.elevation);
    return {
      right: [-sa, ca, 0],
      up: [-se * ca, -se * sa, ce],
      view: [ce * ca, ce * sa, se],
    };
  }

  projectionScale(width, height, camera) {
    const corners = [];
    for (const x of [-32, 32]) {
      for (const y of [-32, 32]) {
        for (const z of [-32, 32]) corners.push([x, y, z]);
      }
    }
    const xs = corners.map((point) => dot(point, camera.right));
    const ys = corners.map((point) => dot(point, camera.up));
    const spanX = Math.max(...xs) - Math.min(...xs);
    const spanY = Math.max(...ys) - Math.min(...ys);
    return Math.min(width * 0.72 / spanX, height * 0.72 / spanY) * this.zoom;
  }

  bindVertexBuffers() {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(this.locations.position);
    gl.vertexAttribPointer(this.locations.position, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.normalBuffer);
    gl.enableVertexAttribArray(this.locations.normal);
    gl.vertexAttribPointer(this.locations.normal, 3, gl.FLOAT, false, 0, 0);
  }

  drawOverlay(width, height, dpr, camera, scale) {
    if (!this.overlayCtx || !this.overlay) return;
    if (this.overlay.width !== width || this.overlay.height !== height) {
      this.overlay.width = width;
      this.overlay.height = height;
    }
    const ctx = this.overlayCtx;
    ctx.clearRect(0, 0, width, height);
    const project = (point) => {
      const relative = point.map((value) => value - 32);
      return [width / 2 + dot(relative, camera.right) * scale, height / 2 - dot(relative, camera.up) * scale];
    };
    const corners = [
      [0, 0, 0], [64, 0, 0], [0, 64, 0], [64, 64, 0],
      [0, 0, 64], [64, 0, 64], [0, 64, 64], [64, 64, 64],
    ].map(project);
    const edges = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
    ctx.save();
    ctx.strokeStyle = "rgba(78, 84, 80, 0.20)";
    ctx.lineWidth = Math.max(0.5, 0.7 * dpr);
    ctx.setLineDash([3 * dpr, 3 * dpr]);
    ctx.beginPath();
    for (const [start, end] of edges) {
      ctx.moveTo(...corners[start]);
      ctx.lineTo(...corners[end]);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    const origin = project([0, 0, 0]);
    const axes = [
      { end: [64, 0, 0], label: "X", color: "#b43b35" },
      { end: [0, 64, 0], label: "Y", color: "#39844c" },
      { end: [0, 0, 64], label: "Z", color: "#356fa8" },
    ];
    ctx.lineWidth = 1.5 * dpr;
    ctx.font = `${12 * dpr}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const axis of axes) {
      const end = project(axis.end);
      ctx.strokeStyle = axis.color;
      ctx.beginPath();
      ctx.moveTo(...origin);
      ctx.lineTo(...end);
      ctx.stroke();
      const length = Math.hypot(end[0] - origin[0], end[1] - origin[1]) || 1;
      ctx.fillStyle = axis.color;
      ctx.fillText(
        axis.label,
        end[0] + (end[0] - origin[0]) / length * 12 * dpr,
        end[1] + (end[1] - origin[1]) / length * 12 * dpr,
      );
    }
    ctx.fillStyle = "#555a57";
    ctx.font = `${10 * dpr}px system-ui, sans-serif`;
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    const elev = Math.round(this.elevation * 180 / Math.PI);
    const azim = Math.round(this.azimuth * 180 / Math.PI);
    ctx.fillText(`64³ · elev ${elev}° · azim ${azim}°`, width - 10 * dpr, 10 * dpr);
    ctx.restore();
  }

  draw() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    const gl = this.gl;
    const camera = this.camera();
    const scale = this.projectionScale(width, height, camera);
    gl.viewport(0, 0, width, height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.uniform3fv(this.locations.right, camera.right);
    gl.uniform3fv(this.locations.up, camera.up);
    gl.uniform3fv(this.locations.view, camera.view);
    gl.uniform2f(this.locations.viewport, width, height);
    gl.uniform1f(this.locations.scale, scale);
    this.bindVertexBuffers();

    gl.enable(gl.POLYGON_OFFSET_FILL);
    gl.polygonOffset(1, 1);
    gl.uniform1i(this.locations.useLight, 1);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.fillIndexBuffer);
    for (const layer of this.layers) {
      if (!layer.visible || !layer.fillCount) continue;
      gl.uniform4f(this.locations.color, ...layer.color, 1);
      gl.drawElements(gl.TRIANGLES, layer.fillCount, gl.UNSIGNED_INT, layer.fillOffset);
    }
    gl.disable(gl.POLYGON_OFFSET_FILL);

    gl.uniform4f(this.locations.color, 35 / 255, 38 / 255, 36 / 255, 0.22);
    gl.uniform1i(this.locations.useLight, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.edgeIndexBuffer);
    for (const layer of this.layers) {
      if (!layer.visible || !layer.edgeCount) continue;
      gl.drawElements(gl.LINES, layer.edgeCount, gl.UNSIGNED_INT, layer.edgeOffset);
    }
    gl.disable(gl.BLEND);

    this.drawOverlay(width, height, dpr, camera, scale);
  }
}
