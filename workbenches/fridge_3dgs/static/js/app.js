import { VoxelViewer } from "./voxel-viewer.js?v=20260713-kin-agent-v8";

const VIEW_NAMES = ["front", "front_left", "front_right", "side"];
const SUPPORTED_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);
const SUPPORTED_IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "webp"]);
const DEFAULT_LABELS = [
  { id: 1, name: "body", color: "#146c94" },
  { id: 2, name: "door", color: "#b45f06" },
  { id: 3, name: "drawer", color: "#2a7f62" },
];

const APP_ROOT = new URL("../../", import.meta.url);

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function appUrl(path) {
  return new URL(String(path).replace(/^\/+/, ""), APP_ROOT).toString();
}

function resourceUrl(path) {
  if (!path) return "";
  if (/^(data:|blob:|https?:)/.test(path)) return path;
  return appUrl(path);
}

function versionedResourceUrl(path, version) {
  const url = resourceUrl(path);
  if (!url || !version) return url;
  const out = new URL(url);
  out.searchParams.set("v", String(version));
  return out.toString();
}

function escapeHtml(value) {
  return String(value === null || value === undefined ? "" : value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

const state = {
  config: null,
  session: null,
  activeStage: "views",
  activeSlot: 0,
  activeView: 0,
  activeLabel: 1,
  labels: cloneJson(DEFAULT_LABELS),
  tool: "sam3_points",
  pointMode: 1,
  points: [],
  mask: null,
  maskWidth: 0,
  maskHeight: 0,
  samCandidates: [],
  samPreview: null,
  jobId: null,
  pollTimer: null,
  jobIds: {},
  pollTimers: {},
  pipeline: {},
  runCatalog: [],
  activeRun: null,
  partSegRuns: [],
  selectedPartSegRun: null,
  ssVoxelViewer: null,
  ssVoxelLoadedUrl: null,
  ssVoxelLoading: false,
  partVoxelViewer: null,
  partVoxelLoadedUrl: null,
  partVoxelLoading: false,
  kinAgentConfig: null,
  kinAgentJobId: null,
  kinAgentPollTimer: null,
  orientation: { x: 0, y: 225, z: 180 },
  inputMode: "upload",
  datasetObjects: [],
  datasetBusy: false,
  uploadFiles: [null, null, null, null],
  uploadPreviewUrls: [null, null, null, null],
  pendingUploadSlot: null,
  uploadBusy: false,
};

const el = (id) => document.getElementById(id);

function setStatus(text, level = "info") {
  const node = el("appStatus");
  node.textContent = text;
  node.style.color = level === "error" ? "var(--red)" : level === "warn" ? "var(--amber)" : "var(--muted)";
}

async function api(path, options = {}) {
  const res = await fetch(appUrl(path), {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await res.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = { ok: false, detail: text };
  }
  if (!res.ok) {
    const detail = payload.detail || payload.error || text || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail, null, 2));
  }
  return payload;
}

function setBadge(id, text, cls) {
  const node = el(id);
  node.textContent = text;
  node.className = `badge ${cls || ""}`.trim();
}

function outputRelUrl(path) {
  if (!state.config || !path) return null;
  const root = state.config.work_root;
  return path.startsWith(root) ? resourceUrl(`/outputs/${path.slice(root.length + 1)}`) : null;
}

async function loadConfig() {
  state.config = await api("/api/config");
  await loadRunCatalog();
  state.session = await api("/api/session");
  await loadPartSegCheckpoints();
  const storedSource = state.session && state.session.input_source ? state.session.input_source.type : "none";
  const pointCloudOk = !!(state.config && state.config.point_cloud && state.config.point_cloud.exists);
  state.inputMode = storedSource === "dataset" || storedSource === "dataset_import"
    ? "dataset"
    : pointCloudOk && ["3dgs_capture", "mixed"].includes(storedSource)
      ? "3dgs"
      : "upload";
  if (Array.isArray(state.session.labels) && state.session.labels.length) {
    state.labels = state.session.labels.map((item) => ({
      id: Number(item.id),
      name: item.name || `part_${item.id}`,
      color:
        item.color ||
        (DEFAULT_LABELS[(Number(item.id) - 1) % DEFAULT_LABELS.length]
          ? DEFAULT_LABELS[(Number(item.id) - 1) % DEFAULT_LABELS.length].color
          : "#146c94"),
    }));
    state.activeLabel = state.labels[0] ? state.labels[0].id : 1;
  }
  renderAll();
  await loadPipeline();
  if (state.inputMode === "3dgs") ensureViewerLoaded();
  if (state.inputMode === "dataset") loadDatasetCatalog().catch((error) => setStatus(error.message, "error"));
}

async function loadRunCatalog() {
  const payload = await api("/api/runs");
  state.runCatalog = payload.runs || [];
  state.activeRun = payload.active_run || null;
  renderRunCatalog();
}

function renderRunCatalog() {
  const select = el("runWorkspaceSelect");
  if (!select) return;
  select.innerHTML = state.runCatalog.map((run) => {
    const object = run.object_id ? ` · ${run.object_id}` : " · empty";
    return `<option value="${escapeHtml(run.id)}"${run.id === state.activeRun ? " selected" : ""}>${escapeHtml(run.id + object)}</option>`;
  }).join("");
  const active = state.runCatalog.find((run) => run.id === state.activeRun);
  el("runWorkspaceSummary").textContent = active
    ? `${active.path}${active.object_id ? ` · object ${active.object_id}` : " · no object loaded"}`
    : "No active run folder";
}

async function selectRunWorkspace(runId, create = false) {
  const clean = String(runId || "").trim();
  if (!clean) throw new Error("Enter or select an EE-Eval run folder");
  await api("/api/runs/select", {
    method: "POST",
    body: JSON.stringify({ run_id: clean, create }),
  });
  Object.values(state.pollTimers).forEach((timer) => window.clearTimeout(timer));
  state.jobIds = {};
  state.pollTimers = {};
  state.pipeline = {};
  state.ssVoxelLoadedUrl = null;
  state.partVoxelLoadedUrl = null;
  state.kinAgentConfig = null;
  state.kinAgentJobId = null;
  window.clearTimeout(state.kinAgentPollTimer);
  state.session = null;
  unloadViewer();
  await loadConfig();
  setStage("views");
  setStatus(`${create ? "Created" : "Opened"} EE-Eval run ${clean}`);
}

async function loadPartSegCheckpoints() {
  const payload = await api("/api/checkpoints/part-prompt-seg");
  state.partSegRuns = payload.runs || [];
  const storageKey = `eeEvalPartSegRun:${state.activeRun || "default"}`;
  const stored = window.localStorage.getItem(storageKey);
  const available = new Set(state.partSegRuns.map((item) => item.id));
  state.selectedPartSegRun = available.has(stored)
    ? stored
    : available.has(payload.selected_id)
      ? payload.selected_id
      : payload.default_id;
  renderPartSegCheckpointSelect();
}

function renderPartSegCheckpointSelect() {
  const select = el("partSegCkptSelect");
  if (!select) return;
  select.innerHTML = "";
  for (const item of state.partSegRuns) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.run_name;
    option.selected = item.id === state.selectedPartSegRun;
    select.appendChild(option);
  }
}

async function refreshSession() {
  state.session = await api("/api/session");
  if (Array.isArray(state.session.labels) && state.session.labels.length) {
    state.labels = state.session.labels.map((item, index) => ({
      id: Number(item.id),
      name: item.name || `part_${item.id}`,
      color: item.color || DEFAULT_LABELS[index % DEFAULT_LABELS.length]?.color || "#146c94",
    }));
    if (!state.labels.some((item) => Number(item.id) === Number(state.activeLabel))) {
      state.activeLabel = state.labels[0].id;
    }
  }
  renderAll();
  await loadPipeline();
  await loadKinAgentConfig();
}

function contractText(contract) {
  if (!contract || !contract.ok) {
    if (contract && contract.missing) return `Missing ${contract.missing.length}`;
    return (contract && contract.error) || "Not ready";
  }
  return `${contract.image_count} RGB / ${contract.mask_count} masks / labels ${contract.positive_labels.join(",")}`;
}

function renderAll() {
  const cfg = state.config;
  const session = state.session;
  const pointCloudOk = !!(cfg && cfg.point_cloud && cfg.point_cloud.exists);
  const sam3Running = !!(cfg && cfg.sam3 && cfg.sam3.health && cfg.sam3.health.running);
  const contractOk = !!(session && session.contract && session.contract.ok);
  const savedViewCount = session && Array.isArray(session.views) ? session.views.filter((view) => view.image_url).length : 0;
  el("sessionPath").textContent = (session && session.session_dir) || "";
  if (state.inputMode !== "3dgs") {
    setBadge("pointCloudBadge", `${savedViewCount}/4 RGB`, savedViewCount >= 1 ? "ok" : "warn");
  } else {
    setBadge("pointCloudBadge", pointCloudOk ? "PLY ok" : "PLY missing", pointCloudOk ? "ok" : "err");
  }
  setBadge("sam3Badge", sam3Running ? "SAM3 running" : "SAM3 idle", sam3Running ? "ok" : "warn");
  setBadge("contractBadge", contractOk ? `${savedViewCount} view ready` : "masks pending", contractOk ? "ok" : "warn");
  renderInputMode();
  renderUploadSlots();
  renderSlots();
  renderViewAngleButtons();
  renderRenders();
  renderLabels();
  renderMaskViewList();
  renderExport();
  renderCkpts();
  renderSsWorkspace();
  renderPartSegWorkspace();
  renderDatasetSummary();
  const maskHint = el("maskModeHint");
  if (maskHint) {
    maskHint.textContent = state.inputMode === "dataset"
      ? "Dataset label maps are loaded directly. You can inspect or correct them before running."
      : "Use positive and negative points with SAM3, then apply the selected mask to a stable part label.";
  }
  setStatus(contractText(session ? session.contract : null));
}

function slotImage(view) {
  return versionedResourceUrl(view && view.image_url, view && view.image_mtime);
}

function persistedViewName(index, view = null) {
  const current = view || (state.session && state.session.views ? state.session.views[index] : null);
  return (current && current.camera && current.camera.name) || VIEW_NAMES[index] || `view_${index}`;
}

function viewSourceText(view) {
  const source = view && view.camera ? view.camera.source : null;
  if (source === "direct_upload") return "uploaded";
  if (source === "dataset") return "dataset";
  if (source === "3dgs_capture") return "3DGS";
  if (view && view.camera) return "3DGS";
  return "saved";
}

function renderInputMode() {
  const pointCloudOk = !!(state.config && state.config.point_cloud && state.config.point_cloud.exists);
  const source = state.session && state.session.input_source ? state.session.input_source.type : "none";
  const sourceLabels = {
    direct_upload: "Current session: wild image upload",
    dataset: "Current session: dataset RGB + masks",
    dataset_import: "Current session: dataset RGB + masks",
    "3dgs_capture": "Current session: 3DGS capture",
    mixed: "Current session: mixed view sources",
    none: "Current session: no saved views",
  };
  el("inputSourceSummary").textContent = sourceLabels[source] || `Current session: ${source}`;
  el("threeDgsWorkspace").hidden = state.inputMode !== "3dgs";
  el("uploadWorkspace").hidden = state.inputMode !== "upload";
  el("datasetWorkspace").hidden = state.inputMode !== "dataset";
  document.querySelectorAll("[data-input-mode]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.inputMode === state.inputMode);
  });
  el("sourceUploadBtn").disabled = state.uploadBusy;
  el("sourceDatasetBtn").disabled = state.uploadBusy || state.datasetBusy;
  el("source3dgsBtn").disabled = state.uploadBusy || !pointCloudOk;
  el("source3dgsBtn").title = pointCloudOk ? "Use 3DGS capture" : "Point cloud is unavailable";
}

function setInputMode(mode) {
  if (state.uploadBusy) return;
  if (mode === "3dgs" && !(state.config && state.config.point_cloud && state.config.point_cloud.exists)) {
    setStatus("Point cloud is unavailable; use four-image input", "warn");
    return;
  }
  state.inputMode = ["upload", "dataset", "3dgs"].includes(mode) ? mode : "upload";
  renderInputMode();
  if (state.inputMode === "3dgs") ensureViewerLoaded();
  else unloadViewer();
  if (state.inputMode === "dataset") {
    loadDatasetCatalog().catch((error) => setStatus(error.message, "error"));
  }
  setStatus(
    state.inputMode === "dataset"
      ? "Dataset input ready"
      : state.inputMode === "upload"
        ? "Wild 1-4 image input ready"
        : "3DGS capture ready",
  );
}

function renderDatasetSummary() {
  const node = el("datasetSummary");
  if (!node) return;
  const source = state.session && state.session.input_source;
  const dataset = state.session && (state.session.dataset || state.session.dataset_sample);
  if (!dataset && !["dataset", "dataset_import"].includes(source && source.type)) {
    node.innerHTML = '<div class="emptyDataset">Select an object and angle to materialize its RGB and 2D masks.</div>';
    return;
  }
  const objectId = dataset?.object_id || dataset?.obj_id || source?.object_id || "-";
  const angle = dataset?.angle_idx ?? dataset?.angle ?? source?.angle_idx ?? 0;
  const views = (state.session?.views || []).filter((view) => view.image_url).length;
  const masks = (state.session?.views || []).filter((view) => view.mask_exists).length;
  node.innerHTML = `<div class="datasetLoaded"><strong>${escapeHtml(objectId)}</strong><span>angle ${escapeHtml(angle)}</span><span>${views} RGB</span><span>${masks} masks</span></div>`;
}

function renderDatasetAngles() {
  const objectSelect = el("datasetObjectSelect");
  const angleSelect = el("datasetAngleSelect");
  if (!objectSelect || !angleSelect) return;
  const [datasetId, objectId] = objectSelect.value.split("::", 2);
  const selected = state.datasetObjects.find((item) =>
    String(item.dataset_id || "default") === datasetId && String(item.object_id || item.obj_id) === objectId
  );
  const angles = selected?.angles?.length ? selected.angles : [0];
  angleSelect.innerHTML = angles.map((angle) => `<option value="${escapeHtml(angle)}">${escapeHtml(angle)}</option>`).join("");
}

async function loadDatasetCatalog() {
  if (state.datasetObjects.length || state.datasetBusy) return;
  state.datasetBusy = true;
  renderInputMode();
  try {
    const payload = await api("/api/dataset/objects?limit=20000");
    state.datasetObjects = payload.objects || payload.samples || [];
    const select = el("datasetObjectSelect");
    select.innerHTML = state.datasetObjects.map((item) => {
      const id = item.object_id || item.obj_id;
      const datasetId = item.dataset_id || "default";
      const name = item.name || item.category || "";
      return `<option value="${escapeHtml(datasetId)}::${escapeHtml(id)}">${escapeHtml(datasetId)} · ${escapeHtml(id)}${name ? ` · ${escapeHtml(name)}` : ""}</option>`;
    }).join("");
    const currentDataset = state.session?.dataset;
    if (currentDataset) {
      const currentValue = `${currentDataset.dataset_id || "default"}::${currentDataset.object_id}`;
      if (Array.from(select.options).some((option) => option.value === currentValue)) select.value = currentValue;
    }
    renderDatasetAngles();
    if (currentDataset) {
      el("datasetAngleSelect").value = String(currentDataset.angle_idx || 0);
      el("datasetViewCount").value = String(currentDataset.physical_view_count || 4);
    }
  } finally {
    state.datasetBusy = false;
    renderInputMode();
  }
}

async function loadDatasetSample() {
  const selectedValue = el("datasetObjectSelect").value;
  if (!selectedValue) throw new Error("Select a dataset object first");
  const [datasetId, objectId] = selectedValue.split("::", 2);
  const angleIdx = Number(el("datasetAngleSelect").value || 0);
  const viewCount = Number(el("datasetViewCount").value || 4);
  const currentDataset = state.session?.dataset;
  if (
    currentDataset &&
    String(currentDataset.dataset_id) === String(datasetId) &&
    String(currentDataset.object_id) === String(objectId) &&
    Number(currentDataset.angle_idx) === angleIdx &&
    Number(currentDataset.physical_view_count) === viewCount
  ) {
    await loadPipeline();
    state.activeView = 0;
    setStage("masks");
    await loadMaskForActiveView();
    setStatus(`Restored cached ${objectId} from ${state.activeRun}`);
    return;
  }
  const existing = state.session && Array.isArray(state.session.views) && state.session.views.some((view) => view.image_url);
  if (existing && !window.confirm("Replace the current RGB views and masks with this dataset sample?")) return;
  state.datasetBusy = true;
  renderInputMode();
  setStatus(`Loading dataset sample ${objectId}`);
  try {
    await api("/api/dataset/load", {
      method: "POST",
      body: JSON.stringify({
        dataset_id: datasetId,
        object_id: objectId,
        angle_idx: angleIdx,
        view_count: viewCount,
        replace_existing: Boolean(existing),
      }),
    });
    state.activeView = 0;
    await refreshSession();
    setStage("masks");
    await loadMaskForActiveView();
    setStatus(`Dataset sample ${objectId} loaded`);
  } finally {
    state.datasetBusy = false;
    renderInputMode();
  }
}

function uploadSlotLabel(index) {
  return `view_${index}`;
}

function renderUploadSlots() {
  const grid = el("uploadPreviewGrid");
  if (!grid) return;
  grid.innerHTML = "";
  const selectedCount = state.uploadFiles.filter(Boolean).length;
  el("uploadSelectionSummary").textContent = `${selectedCount}/4 selected`;
  el("importFourViewsBtn").disabled = state.uploadBusy || selectedCount < 1;
  el("chooseFourViewsBtn").disabled = state.uploadBusy;
  el("clearFourViewsBtn").disabled = state.uploadBusy || selectedCount === 0;
  const pointCloudOk = !!(state.config && state.config.point_cloud && state.config.point_cloud.exists);
  el("sourceUploadBtn").disabled = state.uploadBusy;
  el("sourceDatasetBtn").disabled = state.uploadBusy || state.datasetBusy;
  el("source3dgsBtn").disabled = state.uploadBusy || !pointCloudOk;
  document.querySelectorAll(".stageTab").forEach((btn) => {
    btn.disabled = state.uploadBusy;
  });
  for (const id of ["finalizeBtn", "checkDinoBtn", "startSsFlowBtn", "startSsDecodeBtn", "startPartSegBtn", "startSlatDecodeBtn"]) {
    el(id).disabled = state.uploadBusy;
  }
  if (Object.keys(STAGE_RUNNERS).length) renderPipeline();
  for (let i = 0; i < 4; i++) {
    const file = state.uploadFiles[i];
    const card = document.createElement("div");
    card.className = "uploadCard";
    card.dataset.uploadCard = String(i);
    const preview = file
      ? `<img class="uploadPreview" alt="" src="${state.uploadPreviewUrls[i]}">`
      : `<div class="uploadEmpty">${uploadSlotLabel(i)}</div>`;
    card.innerHTML = `
      ${preview}
      <div class="uploadCardMeta">
        <div class="slotTitle">${uploadSlotLabel(i)}</div>
        <div class="uploadFilename" title="${escapeHtml(file ? file.name : "")}">${escapeHtml(file ? file.name : "Not selected")}</div>
        <div class="uploadCardActions">
          <button class="iconButton" data-upload-move="-1" data-index="${i}" title="Move left" aria-label="Move left" ${state.uploadBusy || i === 0 ? "disabled" : ""}>←</button>
          <button class="iconButton" data-upload-move="1" data-index="${i}" title="Move right" aria-label="Move right" ${state.uploadBusy || i === 3 ? "disabled" : ""}>→</button>
          <button data-upload-slot="${i}" ${state.uploadBusy ? "disabled" : ""}>${file ? "Replace" : "Choose"}</button>
        </div>
      </div>`;
    grid.appendChild(card);
  }
}

function renderSlots() {
  const list = el("viewSlotList");
  list.innerHTML = "";
  for (let i = 0; i < 4; i++) {
    const view = state.session && state.session.views ? state.session.views[i] : null;
    const card = document.createElement("div");
    card.className = `slotCard ${state.activeSlot === i ? "active" : ""}`;
    const image = slotImage(view);
    card.innerHTML = `
      ${image ? `<img class="slotThumb" alt="" src="${image}">` : `<span class="slotThumb emptyPreview"></span>`}
      <div class="slotMeta">
        <div class="slotTitle">${i + 1}. ${escapeHtml(persistedViewName(i, view))}</div>
        <div class="muted">${view && view.image_url ? `RGB ${viewSourceText(view)}` : "empty"}</div>
        <div class="slotActions">
          <button data-action="select" data-index="${i}">Select</button>
          <button data-action="capture" data-index="${i}">Capture</button>
        </div>
      </div>`;
    list.appendChild(card);
  }
}

function renderViewAngleButtons() {
  document.querySelectorAll("[data-view-slot]").forEach((btn) => {
    btn.classList.toggle("active", Number(btn.dataset.viewSlot) === Number(state.activeSlot));
  });
}

function renderRenders() {
  const grid = el("renderGrid");
  grid.innerHTML = "";
  let count = 0;
  for (let i = 0; i < 4; i++) {
    const view = state.session && state.session.views ? state.session.views[i] : null;
    if (view && view.image_url) count += 1;
    const fig = document.createElement("figure");
    fig.className = "renderItem";
    const image = versionedResourceUrl(view && view.image_url, view && view.image_mtime);
    fig.innerHTML = `
      ${image ? `<img class="renderThumb" alt="" src="${image}">` : `<div class="renderThumb emptyPreview"></div>`}
      <figcaption>${escapeHtml(persistedViewName(i, view))} ${view && view.image_url ? viewSourceText(view) : "empty"}</figcaption>`;
    grid.appendChild(fig);
  }
  el("renderSummary").textContent = `${count}/4 saved`;
}

function renderMaskViewList() {
  const list = el("maskViewList");
  list.innerHTML = "";
  for (let i = 0; i < 4; i++) {
    const view = state.session && state.session.views ? state.session.views[i] : null;
    if (!view || !view.image_url) continue;
    const btn = document.createElement("button");
    btn.className = state.activeView === i ? "active" : "";
    btn.dataset.index = String(i);
    btn.textContent = `${i + 1}. ${persistedViewName(i, view)} ${view && view.mask_exists ? "mask" : "rgb"}`;
    list.appendChild(btn);
  }
}

function labelColor(labelId) {
  const label = state.labels.find((item) => Number(item.id) === Number(labelId));
  return (label && label.color) || "#146c94";
}

function hexToRgb(hex) {
  const value = String(hex || "#146c94").replace("#", "").padEnd(6, "0").slice(0, 6);
  return [parseInt(value.slice(0, 2), 16), parseInt(value.slice(2, 4), 16), parseInt(value.slice(4, 6), 16)];
}

function updateActiveLabelColorVars() {
  const list = el("samCandidates");
  if (list) list.style.setProperty("--active-label-color", labelColor(state.activeLabel));
}

function renderLabels() {
  const list = el("labelList");
  list.innerHTML = "";
  for (const label of state.labels) {
    const row = document.createElement("div");
    row.className = `labelRow ${Number(label.id) === Number(state.activeLabel) ? "active" : ""}`;
    row.innerHTML = `
      <button class="labelPick" style="background:${label.color}" data-id="${label.id}" title="${label.name}"></button>
      <input type="text" value="${label.name}" data-name-id="${label.id}">
      <input type="color" value="${label.color}" data-color-id="${label.id}">`;
    list.appendChild(row);
  }
  updateActiveLabelColorVars();
  renderMask();
}

function renderExport() {
  const contract = (state.session && state.session.contract) || {};
  const views = (state.session && state.session.views) || [];
  const rows = views
    .map((view, idx) => {
      const image = view.image_url ? "RGB" : "RGB missing";
      const mask = view.mask_exists ? "mask" : "mask missing";
      return `<div>${idx + 1}. ${escapeHtml(persistedViewName(idx, view))}: ${image}, ${mask}</div>`;
    })
    .join("");
  el("exportSummary").textContent = contractText(contract);
  el("inputContract").innerHTML = `
    <div class="sectionTitle">Contract</div>
    <div>${contract.ok ? "ok" : "pending"}</div>
    <div class="muted">${contract.error || ""}</div>
    <hr>
    ${rows}
    <hr>
    <div class="pathText">${(state.session && state.session.session_dir) || ""}</div>`;
  el("manifestBox").textContent = state.session && state.session.manifest ? JSON.stringify(state.session.manifest, null, 2) : "";
}

function renderCkptList(id, keys) {
  const list = el(id);
  if (!list) return;
  list.innerHTML = "";
  const ckpts = (state.config && state.config.ckpts) || {};
  for (const key of keys) {
    const item = ckpts[key];
    if (!item) continue;
    const div = document.createElement("div");
    div.className = `ckptItem ${item.exists ? "ok" : "err"}`;
    div.innerHTML = `<strong>${key}</strong><span>${item.exists ? "exists" : "missing"}</span><div class="pathText">${item.path}</div>`;
    list.appendChild(div);
  }
}

function renderCkpts() {
  renderCkptList("ssCkptList", ["ss_flow_ckpt"]);
  renderCkptList("ssDecodeCkptList", ["ss_decoder_ckpt"]);
  const selected = state.partSegRuns.find((item) => item.id === state.selectedPartSegRun);
  const partList = el("partCkptList");
  if (partList) {
    partList.innerHTML = selected
      ? `<div class="ckptItem ok"><strong>${escapeHtml(selected.run_name)}</strong><span>latest</span><div class="pathText">${escapeHtml(selected.checkpoint_path)}</div></div>`
      : "";
  }
  renderCkptList("slatCkptList", ["slat_flow_ckpt", "slat_mesh_decoder_ckpt", "slat_gaussian_decoder_ckpt"]);
}

function renderSsWorkspace() {
  const box = el("ssInputContract");
  if (!box) return;
  const contract = (state.session && state.session.contract) || {};
  const views = (state.session && state.session.views) || [];
  const dino = (state.session && state.session.ssflow_inputs) || {};
  const rows = views
    .map((view, idx) => {
      const rgb = view.image_url ? "RGB" : "RGB missing";
      const mask = view.mask_exists ? "mask" : "mask missing";
      return `<div>${idx + 1}. ${escapeHtml(persistedViewName(idx, view))}: ${rgb}, ${mask}</div>`;
    })
    .join("");
  box.innerHTML = `
    <div>${contract.ok ? "ready" : "pending"}</div>
    <div class="muted">${contractText(contract)}</div>
    <div class="muted">DINO ${dino.ok ? "ready" : "pending"} / expected ${(dino.expected_token_shape || []).join(" x ")}</div>
    <hr>
    ${rows}
    <hr>
    <div class="pathText">${dino.preprocess || ""}</div>
    <hr>
    <div class="pathText">${(state.session && state.session.session_dir) || ""}</div>`;
}

function renderPartSegWorkspace() {
  const box = el("partMaskSummary");
  if (!box) return;
  const views = (state.session && state.session.views) || [];
  const contract = (state.session && state.session.contract) || {};
  const labelRows = state.labels
    .map((label) => `<div><span class="labelDot" style="background:${label.color}"></span>${label.id}: ${label.name}</div>`)
    .join("");
  const maskRows = views
    .map((view, idx) => {
      const preview = view.mask_preview_url
        ? `<img class="maskMini" alt="" src="${versionedResourceUrl(view.mask_preview_url, view.mask_preview_mtime)}">`
        : `<div class="maskMini empty"></div>`;
      return `<div class="maskSummaryRow">${preview}<span>${idx + 1}. ${escapeHtml(persistedViewName(idx, view))}</span><strong>${view.mask_exists ? "mask" : "missing"}</strong></div>`;
    })
    .join("");
  box.innerHTML = `
    <div>${contract.ok ? "ready" : "pending"}</div>
    <div class="muted">${contractText(contract)}</div>
    <hr>
    <div class="maskSummaryGrid">${maskRows}</div>
    <hr>
    <div class="labelSummary">${labelRows}</div>`;
}

function setStage(stage) {
  state.activeStage = stage;
  document.querySelectorAll(".stage").forEach((node) => node.classList.toggle("active", node.id === `stage-${stage}`));
  document.querySelectorAll(".stageTab").forEach((node) => node.classList.toggle("active", node.dataset.stage === stage));
}

function ensureViewerLoaded() {
  const viewerFrame = el("viewerFrame");
  if (viewerFrame && !viewerFrame.getAttribute("src")) viewerFrame.src = appUrl("viewer");
}

function unloadViewer() {
  const viewerFrame = el("viewerFrame");
  if (viewerFrame && viewerFrame.getAttribute("src")) viewerFrame.removeAttribute("src");
}

function viewerPost(type, payload = {}) {
  const target = el("viewerFrame").contentWindow;
  if (target) target.postMessage({ type: `fridge3dgs.${type}`, ...payload }, window.location.origin);
}

function readOrientationInputs() {
  state.orientation = {
    x: Number(el("orientX").value || 0),
    y: Number(el("orientY").value || 0),
    z: Number(el("orientZ").value || 0),
  };
  return state.orientation;
}

function setOrientationInputs(next) {
  state.orientation = {
    x: Number(next.x || 0),
    y: Number(next.y || 0),
    z: Number(next.z || 0),
  };
  el("orientX").value = String(state.orientation.x);
  el("orientY").value = String(state.orientation.y);
  el("orientZ").value = String(state.orientation.z);
}

function applyOrientation(next = readOrientationInputs()) {
  setOrientationInputs(next);
  viewerPost("orientation", { orientation: state.orientation });
  setStatus(`Object rotation X=${state.orientation.x}, Y=${state.orientation.y}, Z=${state.orientation.z}`);
}

function revokeUploadPreview(index) {
  if (state.uploadPreviewUrls[index]) URL.revokeObjectURL(state.uploadPreviewUrls[index]);
  state.uploadPreviewUrls[index] = null;
}

function setUploadFile(index, file, render = true) {
  if (state.uploadBusy) return;
  if (!file) return;
  const extension = String(file.name || "").split(".").pop().toLowerCase();
  if (!SUPPORTED_IMAGE_TYPES.has(file.type) && !SUPPORTED_IMAGE_EXTENSIONS.has(extension)) {
    throw new Error(`${file.name} must be PNG, JPEG, or WebP`);
  }
  revokeUploadPreview(index);
  state.uploadFiles[index] = file;
  state.uploadPreviewUrls[index] = URL.createObjectURL(file);
  if (render) renderUploadSlots();
}

function clearUploadSelection(render = true) {
  for (let i = 0; i < 4; i++) {
    revokeUploadPreview(i);
    state.uploadFiles[i] = null;
  }
  state.pendingUploadSlot = null;
  el("fourViewInput").value = "";
  el("singleViewInput").value = "";
  if (render) renderUploadSlots();
}

function selectUploadBatch(fileList) {
  if (state.uploadBusy) return;
  const files = Array.from(fileList || []);
  if (files.length < 1 || files.length > 4) throw new Error(`Select 1 to 4 images, got ${files.length}`);
  const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
  files.sort((left, right) => collator.compare(left.name, right.name));
  clearUploadSelection(false);
  files.forEach((file, index) => setUploadFile(index, file, false));
  renderUploadSlots();
  setStatus(`${files.length} image${files.length === 1 ? "" : "s"} selected`);
}

function moveUploadFile(index, delta) {
  if (state.uploadBusy) return;
  const target = index + delta;
  if (target < 0 || target > 3) return;
  [state.uploadFiles[index], state.uploadFiles[target]] = [state.uploadFiles[target], state.uploadFiles[index]];
  [state.uploadPreviewUrls[index], state.uploadPreviewUrls[target]] = [state.uploadPreviewUrls[target], state.uploadPreviewUrls[index]];
  renderUploadSlots();
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(String(reader.result || "")), { once: true });
    reader.addEventListener("error", () => reject(reader.error || new Error(`Failed to read ${file.name}`)), { once: true });
    reader.readAsDataURL(file);
  });
}

function inferSourceViewId(file) {
  const match = String(file && file.name).match(/view[_-]?(\d+)/i);
  return match ? Number(match[1]) : null;
}

async function importFourViews() {
  const files = state.uploadFiles.filter(Boolean);
  if (files.length < 1 || files.length > 4) throw new Error("Select 1 to 4 images first");
  const existing = state.session && Array.isArray(state.session.views) && state.session.views.some((view) => view.image_url);
  if (
    existing &&
    !window.confirm(
      `Replace RGB views in ${state.session.session_dir}? Existing masks and DINO derivatives will be cleared.`,
    )
  ) {
    return;
  }

  state.uploadBusy = true;
  renderUploadSlots();
  setStatus(`Importing ${files.length} image${files.length === 1 ? "" : "s"}`);
  try {
    const encoded = await Promise.all(files.map(fileToDataUrl));
    await api("/api/views/import", {
      method: "POST",
      body: JSON.stringify({
        views: files.map((file, index) => ({
          view_index: index,
          image_data_url: encoded[index],
          name: uploadSlotLabel(index),
          original_name: file.name,
          source_view_id: inferSourceViewId(file),
        })),
        replace_existing: Boolean(existing),
      }),
    });
    clearUploadSelection(false);
    state.activeView = 0;
    await refreshSession();
    setStage("renders");
    setStatus(`Imported ${files.length} RGB view${files.length === 1 ? "" : "s"}; add masks with SAM3 points`);
  } finally {
    state.uploadBusy = false;
    renderUploadSlots();
  }
}

function captureFromViewer() {
  const requestId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      window.removeEventListener("message", onMessage);
      reject(new Error("viewer capture timed out"));
    }, 15000);
    function onMessage(event) {
      const msg = event.data || {};
      if (msg.type !== "fridge3dgs.captureResult" || msg.requestId !== requestId) return;
      window.clearTimeout(timer);
      window.removeEventListener("message", onMessage);
      if (msg.ok) resolve(msg);
      else reject(new Error(msg.error || "viewer capture failed"));
    }
    window.addEventListener("message", onMessage);
    viewerPost("capture", { requestId });
  });
}

async function captureSlot(index = state.activeSlot) {
  const current = state.session && state.session.views ? state.session.views[index] : null;
  if (current && current.mask_exists && !window.confirm(`Recapturing ${VIEW_NAMES[index]} will clear its saved mask. Continue?`)) {
    return;
  }
  state.activeSlot = index;
  renderSlots();
  renderViewAngleButtons();
  setStatus(`Capturing ${VIEW_NAMES[index]}`);
  const shot = await captureFromViewer();
  await api("/api/views", {
    method: "POST",
    body: JSON.stringify({
      view_index: index,
      image_data_url: shot.image_data_url,
      camera: shot.camera,
      name: VIEW_NAMES[index],
    }),
  });
  await refreshSession();
  setStatus(`Saved ${VIEW_NAMES[index]}`);
}

async function loadMaskForActiveView() {
  state.samCandidates = [];
  state.samPreview = null;
  const samList = el("samCandidates");
  if (samList) samList.innerHTML = "";
  const view = state.session && state.session.views ? state.session.views[state.activeView] : null;
  if (!view || !view.image_url) {
    state.mask = null;
    renderMask();
    setStatus(`Save RGB for ${persistedViewName(state.activeView, view)} first`, "warn");
    return;
  }
  const img = el("editImage");
  img.src = versionedResourceUrl(view.image_url, view.image_mtime || Date.now());
  await img.decode();
  state.maskWidth = img.naturalWidth;
  state.maskHeight = img.naturalHeight;
  const viewport = el("editorViewport");
  viewport.style.aspectRatio = `${state.maskWidth} / ${state.maskHeight}`;
  const canvas = el("maskCanvas");
  canvas.width = state.maskWidth;
  canvas.height = state.maskHeight;
  state.mask = new Uint16Array(state.maskWidth * state.maskHeight);
  if (view.mask_exists) {
    try {
      const payload = await api(`/api/masks/${state.activeView}`);
      await mergeMaskDataUrl(payload.mask_data_url, 1, { rawLabels: true, replaceAll: true });
    } catch (error) {
      setStatus(error.message, "warn");
    }
  }
  state.points = [];
  renderMask();
}

function renderMask() {
  const canvas = el("maskCanvas");
  if (!state.mask || !canvas.width || !canvas.height) return;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const image = ctx.createImageData(canvas.width, canvas.height);
  for (let i = 0; i < state.mask.length; i++) {
    const label = state.mask[i];
    if (label > 0) {
      const [r, g, b] = hexToRgb(labelColor(label));
      image.data[i * 4] = r;
      image.data[i * 4 + 1] = g;
      image.data[i * 4 + 2] = b;
      image.data[i * 4 + 3] = 142;
    }
  }
  if (
    state.samPreview &&
    state.samPreview.pixels &&
    state.samPreview.width === canvas.width &&
    state.samPreview.height === canvas.height
  ) {
    const [r, g, b] = hexToRgb(labelColor(state.activeLabel));
    for (let i = 0; i < state.samPreview.pixels.length; i++) {
      if (!state.samPreview.pixels[i]) continue;
      image.data[i * 4] = r;
      image.data[i * 4 + 1] = g;
      image.data[i * 4 + 2] = b;
      image.data[i * 4 + 3] = 210;
    }
  }
  ctx.putImageData(image, 0, 0);
  for (const point of state.points) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = point.label ? "#2a7f62" : "#b42318";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
  }
}

function canvasPoint(event) {
  const canvas = el("maskCanvas");
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(canvas.width - 1, ((event.clientX - rect.left) / rect.width) * canvas.width)),
    y: Math.max(0, Math.min(canvas.height - 1, ((event.clientY - rect.top) / rect.height) * canvas.height)),
  };
}

function paintAt(x, y) {
  if (!state.mask) return;
  const radius = Number(el("brushSize").value);
  const label = state.tool === "erase" ? 0 : Number(state.activeLabel);
  const minX = Math.max(0, Math.floor(x - radius));
  const maxX = Math.min(state.maskWidth - 1, Math.ceil(x + radius));
  const minY = Math.max(0, Math.floor(y - radius));
  const maxY = Math.min(state.maskHeight - 1, Math.ceil(y + radius));
  const r2 = radius * radius;
  for (let yy = minY; yy <= maxY; yy++) {
    const dy = yy - y;
    for (let xx = minX; xx <= maxX; xx++) {
      const dx = xx - x;
      if (dx * dx + dy * dy <= r2) {
        state.mask[yy * state.maskWidth + xx] = label;
      }
    }
  }
  renderMask();
}

async function saveActiveMask() {
  if (!state.mask) throw new Error("no mask loaded");
  const out = document.createElement("canvas");
  out.width = state.maskWidth;
  out.height = state.maskHeight;
  const ctx = out.getContext("2d");
  const image = ctx.createImageData(out.width, out.height);
  for (let i = 0; i < state.mask.length; i++) {
    image.data[i * 4] = Math.min(255, state.mask[i]);
    image.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  await api("/api/masks", {
    method: "POST",
    body: JSON.stringify({
      view_index: state.activeView,
      mask_data_url: out.toDataURL("image/png"),
      labels: state.labels,
    }),
  });
  await refreshSession();
  setStatus(`Saved mask ${persistedViewName(state.activeView)}`);
}

async function mergeMaskDataUrl(dataUrl, labelId, options = {}) {
  const img = new Image();
  img.src = dataUrl;
  await img.decode();
  const tmp = document.createElement("canvas");
  tmp.width = state.maskWidth || img.naturalWidth;
  tmp.height = state.maskHeight || img.naturalHeight;
  const ctx = tmp.getContext("2d");
  ctx.drawImage(img, 0, 0, tmp.width, tmp.height);
  const data = ctx.getImageData(0, 0, tmp.width, tmp.height).data;
  if (!state.mask || options.replaceAll) {
    state.maskWidth = tmp.width;
    state.maskHeight = tmp.height;
    state.mask = new Uint16Array(tmp.width * tmp.height);
  }
  for (let i = 0; i < state.mask.length; i++) {
    const value = data[i * 4];
    if (options.rawLabels) {
      state.mask[i] = value;
    } else if (value > 0 || data[i * 4 + 3] > 0) {
      state.mask[i] = Number(labelId);
    }
  }
  renderMask();
}

async function decodeBinaryMaskDataUrl(dataUrl) {
  const img = new Image();
  img.src = dataUrl;
  await img.decode();
  const tmp = document.createElement("canvas");
  tmp.width = state.maskWidth || img.naturalWidth;
  tmp.height = state.maskHeight || img.naturalHeight;
  const ctx = tmp.getContext("2d");
  ctx.drawImage(img, 0, 0, tmp.width, tmp.height);
  const data = ctx.getImageData(0, 0, tmp.width, tmp.height).data;
  const pixels = new Uint8Array(tmp.width * tmp.height);
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] = data[i * 4] > 0 || data[i * 4 + 3] > 0 ? 1 : 0;
  }
  return { width: tmp.width, height: tmp.height, pixels };
}

function renderCandidateSelection() {
  document.querySelectorAll(".candidateItem").forEach((item) => {
    item.classList.toggle(
      "active",
      !!state.samPreview && Number(item.dataset.rank) === Number(state.samPreview.rank),
    );
  });
}

async function setSamPreview(candidate) {
  if (!candidate || !candidate.mask_data_url) return;
  if (!candidate.previewMask) {
    candidate.previewMask = await decodeBinaryMaskDataUrl(candidate.mask_data_url);
  }
  state.samPreview = {
    rank: Number(candidate.rank),
    score: candidate.score,
    area: candidate.area,
    width: candidate.previewMask.width,
    height: candidate.previewMask.height,
    pixels: candidate.previewMask.pixels,
  };
  renderMask();
  renderCandidateSelection();
  setStatus(`Preview SAM3 candidate #${Number(candidate.rank) + 1} as label ${state.activeLabel}`);
}

function clearSamPreview() {
  state.samPreview = null;
  renderMask();
  renderCandidateSelection();
}

function clearLabel(labelId = state.activeLabel) {
  if (!state.mask) return;
  for (let i = 0; i < state.mask.length; i++) {
    if (Number(state.mask[i]) === Number(labelId)) state.mask[i] = 0;
  }
  renderMask();
}

function setPointMode(mode) {
  state.pointMode = mode;
  if (mode !== null) state.tool = "sam3_points";
  const pointPos = el("pointPosBtn");
  const pointNeg = el("pointNegBtn");
  const paint = el("paintToolBtn");
  const erase = el("eraseToolBtn");
  if (pointPos) pointPos.classList.toggle("active", mode === 1);
  if (pointNeg) pointNeg.classList.toggle("active", mode === 0);
  if (paint) paint.classList.remove("active");
  if (erase) erase.classList.remove("active");
  if (mode === 1) setStatus("SAM3 positive point mode");
  if (mode === 0) setStatus("SAM3 negative point mode");
}

function setManualTool(tool) {
  state.tool = tool;
  state.pointMode = null;
  el("paintToolBtn").classList.toggle("active", tool === "paint");
  el("eraseToolBtn").classList.toggle("active", tool === "erase");
  el("pointPosBtn").classList.remove("active");
  el("pointNegBtn").classList.remove("active");
  setStatus(tool === "erase" ? "Manual erase mode" : "Manual paint fix mode");
}

async function renderSamCandidates(payload) {
  const list = el("samCandidates");
  updateActiveLabelColorVars();
  list.innerHTML = "";
  if (!payload || !Array.isArray(payload.candidates) || payload.candidates.length === 0) {
    state.samCandidates = [];
    clearSamPreview();
    list.innerHTML = `<div class="emptyState">No SAM3 candidates</div>`;
    return;
  }
  state.samCandidates = payload.candidates || [];
  for (const candidate of state.samCandidates) {
    const item = document.createElement("div");
    item.className = "candidateItem";
    item.dataset.rank = String(candidate.rank);
    item.innerHTML = `
      <div class="candidateThumb coloredMask" style="--mask-url:url('${candidate.mask_data_url}')">
        <span>#${candidate.rank + 1}</span>
      </div>
      <div class="candidateMeta">
        <strong>Candidate #${candidate.rank + 1}</strong>
        <span class="muted">score ${Number(candidate.score || 0).toFixed(3)} / area ${candidate.area}</span>
        <div class="toolRow">
          <button data-preview-candidate="${candidate.rank}">Preview</button>
          <button data-save-candidate="${candidate.rank}" class="primary">Save to Label</button>
          <button data-use-candidate="${candidate.rank}">Apply</button>
        </div>
      </div>`;
    item.addEventListener("mouseenter", () => runSamAction(() => setSamPreview(candidate)));
    item.addEventListener("click", (event) => {
      if (!event.target.closest("button")) runSamAction(() => setSamPreview(candidate));
    });
    item.querySelector("[data-preview-candidate]").addEventListener("click", () => runSamAction(() => setSamPreview(candidate)));
    item.querySelector("[data-use-candidate]").addEventListener("click", () => runSamAction(() => applySamCandidate(candidate)));
    item.querySelector("[data-save-candidate]").addEventListener("click", () =>
      runSamAction(async () => {
        await applySamCandidate(candidate);
        await saveActiveMask();
      }),
    );
    list.appendChild(item);
  }
  await setSamPreview(state.samCandidates[0]);
}

async function runSamAction(action) {
  try {
    await action();
  } catch (error) {
    const message = error.message || String(error);
    el("sam3Status").textContent = message;
    setStatus(message, "error");
  }
}

async function applySamCandidate(candidate, options = {}) {
  if (options.replaceLabel !== false) clearLabel(state.activeLabel);
  await mergeMaskDataUrl(candidate.mask_data_url, state.activeLabel);
  state.samPreview = null;
  renderCandidateSelection();
  renderMask();
  setStatus(`Applied SAM3 candidate to label ${state.activeLabel}`);
}

async function startSam3() {
  el("sam3Status").textContent = "starting";
  const payload = await api("/api/sam3/start", {
    method: "POST",
    body: JSON.stringify({ port: state.config.sam3.port, device: "cuda", confidence_threshold: Number(el("samThreshold").value) }),
  });
  el("sam3Status").textContent = JSON.stringify(payload.health || payload, null, 2);
  state.config = await api("/api/config");
  renderAll();
}

async function ensureSam3Running() {
  const health = state.config && state.config.sam3 && state.config.sam3.health;
  const running = !!(health && health.running);
  if (!running || health.import_ok === false) await startSam3();
}

async function runSamText() {
  await ensureSam3Running();
  el("sam3Status").textContent = `SAM3 text: ${el("samPrompt").value}`;
  const payload = await api("/api/sam3/text", {
    method: "POST",
    body: JSON.stringify({
      view_index: state.activeView,
      prompt: el("samPrompt").value,
      confidence_threshold: Number(el("samThreshold").value),
    }),
  });
  el("sam3Status").textContent = JSON.stringify({ mode: payload.mode, count: payload.count, label: state.activeLabel }, null, 2);
  await renderSamCandidates(payload);
}

async function runSamPoints() {
  await ensureSam3Running();
  if (!state.points.length) {
    setStatus("Add SAM3 positive/negative points first", "warn");
    return;
  }
  el("sam3Status").textContent = `SAM3 points: ${state.points.length}`;
  const payload = await api("/api/sam3/points", {
    method: "POST",
    body: JSON.stringify({
      view_index: state.activeView,
      points: state.points.map((p) => [p.x, p.y]),
      point_labels: state.points.map((p) => p.label),
      multimask_output: true,
    }),
  });
  el("sam3Status").textContent = JSON.stringify({ mode: payload.mode, count: payload.count, points: state.points.length, label: state.activeLabel }, null, 2);
  await renderSamCandidates(payload);
}

async function writeExport() {
  const payload = await api("/api/export/finalize", {
    method: "POST",
    body: JSON.stringify({ labels: state.labels }),
  });
  await refreshSession();
  el("manifestBox").textContent = JSON.stringify(payload.manifest, null, 2);
  return payload;
}

async function checkDinoTokens() {
  el("dinoCheckStatus").textContent = "checking DINO tokens";
  try {
    const payload = await api("/api/ssflow/dino/check", { method: "POST", body: "{}" });
    el("dinoCheckStatus").textContent = JSON.stringify(payload, null, 2);
    renderDinoTokenVisualizations({
      artifact_urls: {
        pca_views: payload.visualization?.pca_view_urls || [],
        rgb_views: payload.visualization?.input_view_urls || [],
      },
    });
    setStatus(`DINO ok ${payload.shape.join(" x ")}`);
  } catch (error) {
    el("dinoCheckStatus").textContent = error.message;
    setStatus("DINO check failed", "warn");
  }
}

async function finalizeExport() {
  const payload = await writeExport();
  setStatus(payload.ok ? "Export finalized" : "Export contract pending", payload.ok ? "info" : "warn");
}

const STAGE_RUNNERS = {
  dino_ss_flow: {
    dependency: null,
    statusId: "ssStatus",
    outputId: "ssOutputs",
    progressBarId: "ssProgressBar",
    progressTextId: "ssProgressText",
    quickId: "ssQuickSteps",
    buttonId: "startSsFlowBtn",
    label: "DINO + SS Flow",
  },
  ss_decode: {
    dependency: "dino_ss_flow",
    statusId: "ssDecodeStatus",
    outputId: "ssDecodeOutputs",
    progressBarId: "ssDecodeProgressBar",
    progressTextId: "ssDecodeProgressText",
    buttonId: "startSsDecodeBtn",
    label: "SS Decoder",
  },
  part_prompt_seg: {
    dependency: "ss_decode",
    statusId: "partStatus",
    outputId: "partOutputs",
    progressBarId: "partProgressBar",
    progressTextId: "partProgressText",
    buttonId: "startPartSegBtn",
    label: "Part Prompt Seg",
  },
  slat_decode: {
    dependency: "part_prompt_seg",
    statusId: "slatStatus",
    outputId: "slatOutputs",
    progressBarId: "slatProgressBar",
    progressTextId: "slatProgressText",
    quickId: "slatQuickSteps",
    buttonId: "startSlatDecodeBtn",
    label: "Mesh + GS Decode",
  },
};

function stageComplete(status) {
  return ["complete", "cached"].includes(String(status?.state || ""));
}

function updateStageProgress(stage, status = {}) {
  const runner = STAGE_RUNNERS[stage];
  if (!runner) return;
  const progress = Math.max(0, Math.min(100, Number(status.progress || 0)));
  el(runner.progressBarId).style.width = `${progress}%`;
  const stateText = String(status.state || "not_started").replaceAll("_", " ");
  el(runner.progressTextId).textContent = `${stateText} · ${progress}%${status.message ? ` · ${status.message}` : ""}`;
  const dependencyReady = !runner.dependency || stageComplete(state.pipeline[runner.dependency]);
  const contractReady = !!state.session?.contract?.ok;
  el(runner.buttonId).disabled = state.uploadBusy || !contractReady || !dependencyReady || Object.values(state.jobIds).some(Boolean);
}

function renderPipeline() {
  Object.entries(STAGE_RUNNERS).forEach(([stage, runner]) => {
    const status = state.pipeline[stage] || { state: "not_started", progress: 0 };
    updateStageProgress(stage, status);
    renderReconOutputs(status.files || [], runner.outputId);
  });
  renderDinoTokenVisualizations(state.pipeline.dino_ss_flow || {});
  renderSsVoxelVisualization(state.pipeline.ss_decode || {});
  renderPartVoxelVisualization(state.pipeline.part_prompt_seg || {});
  if (stageComplete(state.pipeline.slat_decode)) {
    const manifest = appUrl("api/reconstruct/viewer-manifest");
    const viewer = el("componentViewerFrame");
    const next = `${appUrl("static/component-viewer.html")}?manifest=${encodeURIComponent(manifest)}&mode=mesh`;
    if (viewer.src !== next) viewer.src = next;
  } else {
    el("componentViewerFrame").removeAttribute("src");
  }
}

function renderPartVoxelLegend(layers) {
  const legend = el("partVoxelLegend");
  if (!legend) return;
  legend.innerHTML = "";
  for (const layer of layers || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `partVoxelLegendItem${layer.visible === false ? " off" : ""}`;
    button.dataset.layerId = String(layer.id);
    button.innerHTML = `<span class="partVoxelLegendSwatch" style="--swatch:${escapeHtml(layer.color)}"></span><span>${escapeHtml(layer.label)}</span><span class="partVoxelLegendCount">${Number(layer.voxel_count || 0).toLocaleString()}</span>`;
    button.addEventListener("click", () => {
      const visible = button.classList.contains("off");
      button.classList.toggle("off", !visible);
      state.partVoxelViewer?.setLayerVisible(String(layer.id), visible);
    });
    legend.appendChild(button);
  }
}

async function renderPartVoxelVisualization(status) {
  const panel = el("partVoxelPanel");
  const source = (status.files || []).find((file) => file.rel === "part_coords.npz");
  if (!panel) return;
  panel.hidden = !source;
  const sourceKey = source ? `${source.url}:${source.mtime_ns || source.size || ""}` : null;
  if (!source || state.partVoxelLoadedUrl === sourceKey || state.partVoxelLoading) return;
  state.partVoxelLoading = true;
  try {
    const payload = await api("/api/reconstruct/voxel/part_prompt_seg");
    state.partVoxelViewer ||= new VoxelViewer(el("partVoxelCanvas"), el("partVoxelStats"), el("partVoxelOverlay"));
    state.partVoxelViewer.setData(payload);
    renderPartVoxelLegend(payload.layers);
    state.partVoxelLoadedUrl = sourceKey;
  } catch (error) {
    el("partVoxelStats").textContent = error.message || String(error);
  } finally {
    state.partVoxelLoading = false;
  }
}

async function renderSsVoxelVisualization(status) {
  const panel = el("ssVoxelPanel");
  const source = (status.files || []).find((file) => file.rel === "whole_coords.npy");
  if (!panel) return;
  panel.hidden = !source;
  const sourceKey = source ? `${source.url}:${source.mtime_ns || source.size || ""}` : null;
  if (!source || state.ssVoxelLoadedUrl === sourceKey || state.ssVoxelLoading) return;
  state.ssVoxelLoading = true;
  try {
    const payload = await api("/api/reconstruct/voxel/ss_decode");
    state.ssVoxelViewer ||= new VoxelViewer(el("ssVoxelCanvas"), el("ssVoxelStats"), el("ssVoxelOverlay"));
    state.ssVoxelViewer.setData(payload);
    state.ssVoxelLoadedUrl = sourceKey;
  } catch (error) {
    el("ssVoxelStats").textContent = error.message || String(error);
  } finally {
    state.ssVoxelLoading = false;
  }
}

async function loadPipeline() {
  try {
    const payload = await api("/api/reconstruct/pipeline");
    state.pipeline = payload.stages || {};
    for (const job of payload.active_jobs || []) {
      const stage = String(job.stage || "");
      if (STAGE_RUNNERS[stage] && !state.jobIds[stage]) {
        state.jobIds[stage] = job.job_id;
        pollReconstruct(stage);
      }
    }
    renderPipeline();
  } catch (error) {
    setStatus(`Pipeline status unavailable: ${error.message}`, "warn");
  }
}

async function startReconstruct(stage) {
  const runner = STAGE_RUNNERS[stage];
  if (!runner) throw new Error(`unknown stage ${stage}`);
  el(runner.statusId).textContent = "finalizing inputs";
  try {
    await writeExport();
    const payload = await api("/api/reconstruct/start", {
      method: "POST",
      body: JSON.stringify({
        stage,
        quick_steps: runner.quickId ? el(runner.quickId).checked : false,
        part_seg_run_id: stage === "part_prompt_seg" ? state.selectedPartSegRun : null,
      }),
    });
    state.jobIds[stage] = payload.job_id;
    el(runner.statusId).textContent = JSON.stringify(payload, null, 2);
    renderPipeline();
    pollReconstruct(stage);
  } catch (error) {
    el(runner.statusId).textContent = error.message;
    setStatus(`${runner.label} blocked`, "warn");
  }
}

async function pollReconstruct(stage) {
  const runner = STAGE_RUNNERS[stage];
  const jobId = state.jobIds[stage];
  if (!runner || !jobId) return;
  window.clearTimeout(state.pollTimers[stage]);
  try {
    const payload = await api(`/api/reconstruct/status/${jobId}`);
    state.pipeline = payload.pipeline || state.pipeline;
    el(runner.statusId).textContent = payload.log_tail || JSON.stringify(payload.progress || payload, null, 2);
    renderPipeline();
    if (payload.running) {
      state.pollTimers[stage] = window.setTimeout(() => pollReconstruct(stage), 1500);
    } else {
      state.jobIds[stage] = null;
      renderPipeline();
      if (stage === "slat_decode" && payload.return_code === 0) loadKinAgentConfig();
      setStatus(payload.return_code === 0 ? `${runner.label} complete` : `${runner.label} failed`, payload.return_code === 0 ? "info" : "error");
    }
  } catch (error) {
    state.jobIds[stage] = null;
    window.clearTimeout(state.pollTimers[stage]);
    el(runner.statusId).textContent = error.message || String(error);
    renderPipeline();
    setStatus(`${runner.label} status polling failed`, "error");
  }
}

async function loadKinAgentConfig() {
  try {
    state.kinAgentConfig = await api("/api/kin-agent/config");
    renderKinAgent(state.kinAgentConfig);
  } catch (error) {
    el("kinAgentStatus").textContent = error.message || String(error);
  }
}

function renderKinAgentTrace(trace = []) {
  if (!trace.length) return "";
  const rows = trace.map((row, index) => {
    const selected = row.selected || {};
    const axis = (selected.axis_world || []).map((value) => Number(value).toFixed(3)).join(", ");
    const critic = row.critic_feedback || {};
    const issueCodes = (critic.issues || []).map((issue) => issue.code).filter(Boolean);
    const feedback = [critic.verdict, ...issueCodes, ...(critic.recommended_actions || []), row.stop_reason]
      .filter(Boolean);
    const stage = String(row.stage || "proposal_validation").replaceAll("_", " ");
    return `<li><span>${String(index + 1).padStart(2, "0")}</span><div><strong>${escapeHtml(stage)}</strong><small>${escapeHtml(selected.joint_type || "unknown")} · [${escapeHtml(axis)}] · ${Number(selected.lower || 0).toFixed(3)} → ${Number(selected.upper || 0).toFixed(3)} · score ${Number(selected.score || 0).toFixed(3)}</small>${feedback.length ? `<small class="kinAgentTraceFeedback">${escapeHtml(feedback.join(" · "))}</small>` : ""}</div></li>`;
  }).join("");
  return `<details class="kinAgentTrace"><summary>Refinement trace · ${trace.length} rounds</summary><ol>${rows}</ol></details>`;
}

function renderKinAgent(payload = {}) {
  const ready = !!payload.ready;
  const parts = (payload.parts || []).filter((item) => item.kind !== "body");
  const body = (payload.parts || []).find((item) => item.kind === "body");
  const motionToggle = el("kinAgentMotionStates");
  if (motionToggle) {
    motionToggle.disabled = !payload.dataset_motion_states_available;
    const activeRunKey = state.activeRun || "default";
    if (motionToggle.dataset.runId !== activeRunKey) {
      motionToggle.checked = !!payload.dataset_motion_states_available;
      motionToggle.dataset.runId = activeRunKey;
    }
    if (!payload.dataset_motion_states_available) motionToggle.checked = false;
  }
  const requestedEvidence = motionToggle?.checked ? "dataset motion states" : "static decoded geometry";
  const resultMode = payload.result?.evidence_mode?.replaceAll("_", " ");
  el("kinAgentInputSummary").innerHTML = ready
    ? `<strong>${escapeHtml(body?.label || "body")}</strong><div>${parts.length} moving components · ${escapeHtml(requestedEvidence)}${resultMode ? ` · result ${escapeHtml(resultMode)}` : ""}</div>`
    : "Waiting for decoded body and parts";
  const progress = payload.status || { state: "not_started", progress: 0 };
  el("startKinAgentBtn").disabled = !ready || progress.state === "running" || !!state.kinAgentJobId || Object.values(state.jobIds).some(Boolean);
  el("kinAgentProgressBar").style.width = `${Math.max(0, Math.min(100, Number(progress.progress || 0)))}%`;
  el("kinAgentProgressText").textContent = `${String(progress.state || "not_started").replaceAll("_", " ")} · ${Number(progress.progress || 0)}%${progress.message ? ` · ${progress.message}` : ""}`;
  const result = payload.result;
  const downloads = el("kinAgentDownloads");
  downloads.innerHTML = result
    ? [
        result.xml_url ? `<a href="${resourceUrl(result.xml_url)}" target="_blank" rel="noreferrer">MJCF XML</a>` : "",
        result.usd_url ? `<a href="${resourceUrl(result.usd_url)}" target="_blank" rel="noreferrer">USD</a>` : "",
        result.validation?.report_url ? `<a href="${resourceUrl(result.validation.report_url)}" target="_blank" rel="noreferrer">Validation JSON</a>` : "",
        result.collision_audit_url ? `<a href="${resourceUrl(result.collision_audit_url)}" target="_blank" rel="noreferrer">Collision Audit JSON</a>` : "",
        `<span>${escapeHtml(result.input_contract || "decoded meshes")}</span>`,
      ].filter(Boolean).join("")
    : "No XML or USD yet";
  const results = el("kinAgentResults");
  const validation = el("kinAgentValidation");
  validation.innerHTML = result?.validation?.image_url
    ? `<figure><img src="${resourceUrl(result.validation.image_url)}" alt="MuJoCo qpos validation" /><figcaption>MuJoCo qpos validation · ${result.validation.ok ? "passed" : "needs review"} · decoded collision ${result.collision_audit?.requires_review ? "needs review" : "clear"}</figcaption></figure>`
    : "";
  results.innerHTML = result ? (result.parts || []).map((item) => {
    const candidate = item.candidate || {};
    const delivered = item.delivery_candidate || candidate;
    const signals = candidate.signals || {};
    const axis = (delivered.axis_world || []).map((value) => Number(value).toFixed(4)).join(", ");
    const origin = (delivered.origin_world || []).map((value) => Number(value).toFixed(4)).join(", ");
    const canonicalAxis = (candidate.axis_world || []).map((value) => Number(value).toFixed(4)).join(", ");
    const confidence = (key) => Number(signals[key] || 0).toFixed(2);
    const rangeStatus = item.range_estimate?.status || (Number(signals.range_censored || 0) >= 0.5 ? "censored" : "observed_stop");
    const rangeState = ` · ${rangeStatus}`;
    const predictionOuter = item.range_estimate?.prediction_interval?.outer_q90;
    const predictionText = Array.isArray(predictionOuter)
      ? `${Number(predictionOuter[0]).toFixed(4)} → ${Number(predictionOuter[1]).toFixed(4)}`
      : null;
    const motionStates = Number(signals.motion_observation_states || 0);
    const evidenceParts = [];
    const collision = item.collision_audit || {};
    if (motionStates > 0) {
      const qualifier = Number(signals.motion_observation_confidence || 0) < 0.8 ? "noisy " : "";
      evidenceParts.push(`${motionStates} ${qualifier}calibrated motion states`);
    }
    if (Number(signals.axis_family_model_used || 0) >= 0.5) evidenceParts.push("train-only axis-family model");
    if (Number(signals.phyx_thin_axis_used || 0) >= 0.5) evidenceParts.push("decoded thin-axis critic");
    if (Number(signals.motion_type_classifier_used || 0) >= 0.5) evidenceParts.push("dual-trajectory type critic");
    if (Number(signals.motion_axis_family_used || 0) >= 0.5) evidenceParts.push("motion-family axis critic");
    if (Number(signals.range_prior_used || 0) >= 0.5) evidenceParts.push("train-only range prior");
    if (collision.method) evidenceParts.push("decoded mesh collision audit");
    if (!evidenceParts.length) evidenceParts.push("static decoded geometry");
    const evidence = evidenceParts.join(" + ");
    const reviewState = item.requires_review ? " · review" : "";
    return `<article class="kinAgentResult">
      <div class="kinAgentResultHead"><strong>${escapeHtml(item.label)}</strong><span>${escapeHtml(delivered.joint_type || "unknown")}${reviewState}</span></div>
      <dl><dt>Axis</dt><dd>[${axis}]</dd><dt>Origin</dt><dd>[${origin}]</dd>
      ${delivered.source !== "canonical prediction" ? `<dt>Canonical axis</dt><dd>[${canonicalAxis}]</dd>` : ""}
      <dt>Range</dt><dd>${Number(delivered.lower || 0).toFixed(4)} → ${Number(delivered.upper || 0).toFixed(4)}${rangeState}</dd>
      ${predictionText ? `<dt>Range q90</dt><dd>${predictionText}</dd>` : ""}
      <dt>Stop</dt><dd>${item.range_estimate?.mechanical_stop_confirmed ? "confirmed" : "not confirmed"}</dd>
      ${collision.status ? `<dt>Collision</dt><dd>${escapeHtml(collision.status)}${collision.first_invalid_q == null ? "" : ` · first q ${Number(collision.first_invalid_q).toFixed(4)}`}${(collision.recommended_actions || []).length ? ` · ${escapeHtml(collision.recommended_actions.join(" · "))}` : ""}</dd>` : ""}
      <dt>Evidence</dt><dd>${escapeHtml(evidence)}</dd>
      <dt>Confidence</dt><dd>type ${confidence("type_confidence")} · axis ${confidence("axis_confidence")} · range ${confidence("range_confidence")}</dd>
      ${item.requires_review ? `<dt>Review</dt><dd>${escapeHtml((item.review_reasons || []).join(" · "))}</dd>` : ""}
      <dt>Score</dt><dd>${Number(candidate.score || 0).toFixed(3)}</dd><dt>Rounds</dt><dd>${Number(item.iterations || 0)} / ${Number(result.max_iterations || 0)}</dd></dl>
      ${renderKinAgentTrace(item.trace || [])}
    </article>`;
  }).join("") : "";
  const viewer = el("kinAgentViewerFrame");
  if (result) {
    const resultUrl = appUrl("api/kin-agent/result");
    const next = `${appUrl("static/kin-agent-viewer.html")}?result=${encodeURIComponent(resultUrl)}&v=${encodeURIComponent(progress.updated_unix || Date.now())}`;
    if (viewer.src !== next) viewer.src = next;
  } else {
    viewer.removeAttribute("src");
  }
}

async function startKinAgent() {
  const maxIterations = Math.max(1, Math.min(9, Number(el("kinAgentIterations").value || 7)));
  el("kinAgentIterations").value = String(maxIterations);
  try {
    const payload = await api("/api/kin-agent/start", {
      method: "POST",
      body: JSON.stringify({
        max_iterations: maxIterations,
        use_dataset_motion_states: !!el("kinAgentMotionStates")?.checked,
      }),
    });
    if (payload.cached) {
      state.kinAgentConfig = { ...(state.kinAgentConfig || {}), status: { state: "cached", progress: 100, message: "Inputs unchanged" }, result: payload.result };
      renderKinAgent(state.kinAgentConfig);
      el("kinAgentStatus").textContent = "Cached result reused";
      setStatus("Kin Agent cached result reused");
      return;
    }
    state.kinAgentJobId = payload.job_id;
    el("kinAgentStatus").textContent = JSON.stringify(payload, null, 2);
    renderKinAgent({ ...(state.kinAgentConfig || {}), status: { state: "starting", progress: 1 } });
    pollKinAgent();
  } catch (error) {
    el("kinAgentStatus").textContent = error.message || String(error);
    setStatus("Kin Agent blocked", "warn");
  }
}

async function pollKinAgent() {
  if (!state.kinAgentJobId) return;
  window.clearTimeout(state.kinAgentPollTimer);
  try {
    const payload = await api(`/api/kin-agent/status/${state.kinAgentJobId}`);
    el("kinAgentStatus").textContent = payload.log_tail || JSON.stringify(payload.progress, null, 2);
    state.kinAgentConfig = { ...(state.kinAgentConfig || {}), status: payload.progress, result: payload.result };
    renderKinAgent(state.kinAgentConfig);
    if (payload.running) {
      state.kinAgentPollTimer = window.setTimeout(pollKinAgent, 1200);
    } else {
      state.kinAgentJobId = null;
      await loadKinAgentConfig();
      const validationOk = payload.result?.validation?.ok !== false;
      const message = payload.return_code !== 0 ? "Kin Agent failed" : validationOk ? "Kin Agent complete" : "Kin Agent needs review";
      setStatus(message, payload.return_code !== 0 ? "error" : validationOk ? "info" : "warn");
    }
  } catch (error) {
    state.kinAgentJobId = null;
    el("kinAgentStatus").textContent = error.message || String(error);
  }
}

function renderDinoTokenVisualizations(status) {
  const grid = el("dinoTokenGrid");
  if (!grid) return;
  grid.innerHTML = "";
  const pca = status.artifact_urls?.pca_views || [];
  const sessionRgb = (state.session?.ssflow_inputs?.views || []).map((view) => view.preview_url).filter(Boolean);
  const rgb = status.artifact_urls?.rgb_views || sessionRgb;
  if (!pca.length) {
    grid.innerHTML = '<div class="emptyDataset">Run DINO + SS Flow to visualize 37×37 spatial patch tokens.</div>';
    return;
  }
  for (const [label, items] of [["DINO RGB input", rgb], ["Token PCA", pca]]) {
    const row = document.createElement("section");
    row.className = "tokenVizRow";
    const cells = items.map((url, index) => `
      <figure class="tokenVizCell">
        <img src="${resourceUrl(url)}" alt="">
        <figcaption>Slot ${index + 1}</figcaption>
      </figure>`).join("");
    row.innerHTML = `<strong>${label}</strong><div class="tokenVizItems">${cells}</div>`;
    grid.appendChild(row);
  }
}

function renderReconOutputs(files, outputId) {
  const grid = el(outputId);
  if (!grid) return;
  grid.innerHTML = "";
  for (const file of files) {
    if (!file.url) continue;
    const item = document.createElement("div");
    item.className = "outputItem";
    const isImage = /\.(png|jpg|jpeg|webp)$/i.test(file.rel || "");
    item.innerHTML = `
      ${isImage ? `<img alt="" src="${resourceUrl(file.url)}">` : ""}
      <a href="${resourceUrl(file.url)}" target="_blank" rel="noreferrer">${escapeHtml(file.rel)}</a>
      <div class="muted">${file.size} bytes</div>`;
    grid.appendChild(item);
  }
}

function bindEvents() {
  el("openRunWorkspaceBtn").addEventListener("click", () => {
    selectRunWorkspace(el("runWorkspaceSelect").value, false).catch((error) => setStatus(error.message, "error"));
  });
  el("createRunWorkspaceBtn").addEventListener("click", () => {
    selectRunWorkspace(el("newRunWorkspaceName").value, true).catch((error) => setStatus(error.message, "error"));
  });
  el("partSegCkptSelect").addEventListener("change", (event) => {
    state.selectedPartSegRun = event.target.value;
    window.localStorage.setItem(`eeEvalPartSegRun:${state.activeRun || "default"}`, state.selectedPartSegRun);
    renderCkpts();
    setStatus("Part Prompt Seg training folder selected; latest checkpoint will be used");
  });
  document.querySelectorAll(".stageTab").forEach((btn) => {
    btn.addEventListener("click", () => {
      setStage(btn.dataset.stage);
      if (btn.dataset.stage === "masks") {
        loadMaskForActiveView().catch((error) => setStatus(error.message, "error"));
      }
      if (["ssflow", "ssdecode", "partseg", "slatdecode"].includes(btn.dataset.stage)) {
        loadPipeline();
      }
      if (btn.dataset.stage === "kinagent") loadKinAgentConfig();
    });
  });
  document.querySelectorAll("[data-input-mode]").forEach((btn) => {
    btn.addEventListener("click", () => setInputMode(btn.dataset.inputMode));
  });
  el("datasetObjectSelect").addEventListener("change", renderDatasetAngles);
  el("datasetLoadBtn").addEventListener("click", () => {
    loadDatasetSample().catch((error) => setStatus(error.message, "error"));
  });
  el("chooseFourViewsBtn").addEventListener("click", () => {
    if (state.uploadBusy) return;
    el("fourViewInput").value = "";
    el("fourViewInput").click();
  });
  el("fourViewInput").addEventListener("change", (event) => {
    try {
      selectUploadBatch(event.target.files);
    } catch (error) {
      event.target.value = "";
      setStatus(error.message, "error");
    }
  });
  el("singleViewInput").addEventListener("change", (event) => {
    try {
      const file = event.target.files && event.target.files[0];
      if (file && state.pendingUploadSlot !== null) setUploadFile(state.pendingUploadSlot, file);
    } catch (error) {
      setStatus(error.message, "error");
    } finally {
      event.target.value = "";
      state.pendingUploadSlot = null;
    }
  });
  el("uploadPreviewGrid").addEventListener("click", (event) => {
    const move = event.target.closest("[data-upload-move]");
    if (move) {
      moveUploadFile(Number(move.dataset.index), Number(move.dataset.uploadMove));
      return;
    }
    const choose = event.target.closest("[data-upload-slot]");
    if (!choose) return;
    state.pendingUploadSlot = Number(choose.dataset.uploadSlot);
    el("singleViewInput").value = "";
    el("singleViewInput").click();
  });
  el("uploadPreviewGrid").addEventListener("dragover", (event) => {
    event.preventDefault();
    if (state.uploadBusy) return;
    const card = event.target.closest("[data-upload-card]");
    document.querySelectorAll(".uploadCard.dragover").forEach((node) => node.classList.remove("dragover"));
    if (card) card.classList.add("dragover");
  });
  el("uploadPreviewGrid").addEventListener("dragleave", (event) => {
    const card = event.target.closest("[data-upload-card]");
    if (card && !card.contains(event.relatedTarget)) card.classList.remove("dragover");
  });
  el("uploadPreviewGrid").addEventListener("drop", (event) => {
    event.preventDefault();
    if (state.uploadBusy) return;
    document.querySelectorAll(".uploadCard.dragover").forEach((node) => node.classList.remove("dragover"));
    const files = event.dataTransfer && event.dataTransfer.files;
    const card = event.target.closest("[data-upload-card]");
    try {
      if (card && files && files.length === 1) setUploadFile(Number(card.dataset.uploadCard), files[0]);
      else selectUploadBatch(files);
    } catch (error) {
      setStatus(error.message, "error");
    }
  });
  el("clearFourViewsBtn").addEventListener("click", () => clearUploadSelection());
  el("importFourViewsBtn").addEventListener("click", () => {
    importFourViews().catch((error) => setStatus(error.message, "error"));
  });
  el("viewSlotList").addEventListener("click", async (event) => {
    const btn = event.target.closest("button");
    if (!btn) return;
    const idx = Number(btn.dataset.index);
    state.activeSlot = idx;
    if (btn.dataset.action === "capture") await captureSlot(idx);
    renderSlots();
    renderViewAngleButtons();
    if (btn.dataset.action === "select") setStatus(`Active slot: ${persistedViewName(idx)}`);
  });
  document.querySelectorAll("[data-view-slot]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.activeSlot = Number(btn.dataset.viewSlot);
      renderSlots();
      renderViewAngleButtons();
      setStatus(`Active slot: ${persistedViewName(state.activeSlot)}`);
    });
  });
  el("viewerCaptureBtn").addEventListener("click", () => captureSlot());
  el("viewerFrameBtn").addEventListener("click", () => viewerPost("frame"));
  el("viewerResetBtn").addEventListener("click", () => viewerPost("reset"));
  el("viewerNearBtn").addEventListener("click", () => viewerPost("zoom", { delta: -0.18 }));
  el("viewerFarBtn").addEventListener("click", () => viewerPost("zoom", { delta: 0.22 }));
  el("orientApplyBtn").addEventListener("click", () => applyOrientation());
  el("orientXPlusBtn").addEventListener("click", () => {
    const next = readOrientationInputs();
    next.x += 90;
    applyOrientation(next);
  });
  el("orientXMinusBtn").addEventListener("click", () => {
    const next = readOrientationInputs();
    next.x -= 90;
    applyOrientation(next);
  });
  el("orientDefaultBtn").addEventListener("click", () => applyOrientation({ x: 0, y: 225, z: 180 }));
  el("orientSuperSplatBtn").addEventListener("click", () => applyOrientation({ x: 0, y: 45, z: 180 }));
  el("orientZeroBtn").addEventListener("click", () => applyOrientation({ x: 0, y: 0, z: 0 }));
  el("refreshSessionBtn").addEventListener("click", refreshSession);
  el("maskViewList").addEventListener("click", async (event) => {
    const btn = event.target.closest("button");
    if (!btn) return;
    state.activeView = Number(btn.dataset.index);
    renderMaskViewList();
    await loadMaskForActiveView();
  });
  el("labelList").addEventListener("click", (event) => {
    const pick = event.target.closest("[data-id]");
    if (pick) {
      state.activeLabel = Number(pick.dataset.id);
      renderLabels();
      renderCandidateSelection();
    }
  });
  el("labelList").addEventListener("input", (event) => {
    const nameId = event.target.dataset.nameId;
    const colorId = event.target.dataset.colorId;
    if (nameId) state.labels.find((item) => Number(item.id) === Number(nameId)).name = event.target.value;
    if (colorId) state.labels.find((item) => Number(item.id) === Number(colorId)).color = event.target.value;
    updateActiveLabelColorVars();
    renderMask();
  });
  el("addLabelBtn").addEventListener("click", () => {
    const next = Math.max(0, ...state.labels.map((item) => Number(item.id))) + 1;
    state.labels.push({ id: next, name: `part_${next}`, color: ["#7a4fb3", "#007a78", "#a53f2b"][next % 3] });
    state.activeLabel = next;
    renderLabels();
  });
  el("paintToolBtn").addEventListener("click", () => {
    setManualTool("paint");
  });
  el("eraseToolBtn").addEventListener("click", () => {
    setManualTool("erase");
  });
  el("brushSize").addEventListener("input", () => {
    el("brushSizeOut").textContent = el("brushSize").value;
  });
  el("clearLabelBtn").addEventListener("click", () => clearLabel());
  el("clearMaskBtn").addEventListener("click", () => {
    if (state.mask) state.mask.fill(0);
    clearSamPreview();
    renderMask();
  });
  el("saveMaskBtn").addEventListener("click", saveActiveMask);

  let painting = false;
  el("maskCanvas").addEventListener("pointerdown", (event) => {
    if (!state.mask) return;
    const point = canvasPoint(event);
    if (state.pointMode !== null) {
      state.points.push({ ...point, label: state.pointMode });
      renderMask();
      return;
    }
    painting = true;
    el("maskCanvas").setPointerCapture(event.pointerId);
    paintAt(point.x, point.y);
  });
  el("maskCanvas").addEventListener("pointermove", (event) => {
    if (!painting) return;
    const point = canvasPoint(event);
    paintAt(point.x, point.y);
  });
  el("maskCanvas").addEventListener("pointerup", () => {
    painting = false;
  });
  el("maskCanvas").addEventListener("pointercancel", () => {
    painting = false;
  });

  el("sam3StartBtn").addEventListener("click", () => runSamAction(startSam3));
  el("samTextBtn").addEventListener("click", () => runSamAction(runSamText));
  el("samPointBtn").addEventListener("click", () => runSamAction(runSamPoints));
  el("pointPosBtn").addEventListener("click", () => {
    setPointMode(1);
  });
  el("pointNegBtn").addEventListener("click", () => {
    setPointMode(0);
  });
  el("pointClearBtn").addEventListener("click", () => {
    state.points = [];
    setPointMode(1);
    renderMask();
  });
  document.querySelectorAll("[data-sam-prompt]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const labelId = Number(btn.dataset.labelId);
      if (labelId > 0 && state.labels.some((item) => Number(item.id) === labelId)) {
        state.activeLabel = labelId;
        renderLabels();
      }
      el("samPrompt").value = btn.dataset.samPrompt;
      runSamAction(runSamText);
    });
  });
  setPointMode(1);
  el("finalizeBtn").addEventListener("click", finalizeExport);
  el("checkDinoBtn").addEventListener("click", checkDinoTokens);
  el("startSsFlowBtn").addEventListener("click", () => startReconstruct("dino_ss_flow"));
  el("startSsDecodeBtn").addEventListener("click", () => startReconstruct("ss_decode"));
  el("startPartSegBtn").addEventListener("click", () => startReconstruct("part_prompt_seg"));
  el("startSlatDecodeBtn").addEventListener("click", () => startReconstruct("slat_decode"));
  el("openKinAgentBtn").addEventListener("click", () => {
    setStage("kinagent");
    loadKinAgentConfig();
  });
  el("startKinAgentBtn").addEventListener("click", startKinAgent);
  el("kinAgentMotionStates").addEventListener("change", () => {
    renderKinAgent(state.kinAgentConfig || {});
  });
  window.addEventListener("message", (event) => {
    if (state.inputMode === "3dgs" && event.data && event.data.type === "fridge3dgs.viewerReady") {
      applyOrientation(state.orientation);
      setStatus("3DGS viewer ready");
    }
    if (state.inputMode === "3dgs" && event.data && event.data.type === "fridge3dgs.viewerError") {
      setStatus(event.data.error, "error");
    }
  });
}

bindEvents();
loadConfig().then(loadMaskForActiveView).catch((error) => setStatus(error.message, "error"));
