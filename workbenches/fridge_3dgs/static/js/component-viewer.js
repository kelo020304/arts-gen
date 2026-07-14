const params = new URLSearchParams(window.location.search);
const manifestUrl = params.get("manifest");
const layersEl = document.getElementById("layers");
const legendEl = document.getElementById("legendItems");
const stateEl = document.getElementById("viewerState");
const titleEl = document.getElementById("viewerTitle");
const palette = ["#d8d4ca", "#42c6ab", "#df8b47", "#75a7e8", "#d477aa", "#b0d063", "#9a82dd"];
let manifest = null;
let mode = params.get("mode") === "gs" ? "gs" : "mesh";
let records = [];
let activeId = null;
let relayingCamera = false;
let lastCamera = null;

function absoluteUrl(value, base) {
  if (!value) return null;
  if (/^(data:|blob:|https?:)/.test(value)) return value;
  let resolvedBase;
  if (String(base).startsWith("/")) {
    const staticIndex = window.location.pathname.indexOf("/static/");
    const appPath = staticIndex >= 0 ? window.location.pathname.slice(0, staticIndex + 1) : "/";
    resolvedBase = new URL(`${appPath}${String(base).slice(1)}`, window.location.origin);
  } else {
    resolvedBase = new URL(base, window.location.href);
  }
  if (value.startsWith("/")) {
    const apiIndex = resolvedBase.pathname.indexOf("/api/");
    const appPath = apiIndex >= 0 ? resolvedBase.pathname.slice(0, apiIndex + 1) : "/";
    return new URL(`${appPath}${value.slice(1)}`, resolvedBase.origin).toString();
  }
  return new URL(value, resolvedBase).toString();
}

function normalize(raw, base) {
  const source = raw.viewer || raw;
  const items = [];
  const add = (entry, fallbackId, fallbackLabel, kind) => {
    if (!entry) return;
    const mesh = entry.mesh_url || entry.mesh || (entry.mesh_path && !entry.mesh_path.startsWith("/") ? entry.mesh_path : null);
    const gs = entry.gaussian_url || entry.gs_url || entry.gaussian || (entry.gaussian_path && !entry.gaussian_path.startsWith("/") ? entry.gaussian_path : null);
    if (!mesh && !gs) return;
    items.push({
      id: String(entry.id ?? entry.part_id ?? fallbackId), label: entry.label || entry.name || fallbackLabel,
      kind: entry.kind || kind, mesh: absoluteUrl(mesh, base), gs: absoluteUrl(gs, base),
      fallbackGs: absoluteUrl(entry.fallback_gaussian_url || gs || source.fallback_gaussian_url, base),
      color: entry.color || palette[items.length % palette.length], visible: entry.visible === true,
    });
  };
  add(source.overall, "overall", "Complete", "overall");
  add(source.body, "body", "Body", "body");
  (source.components || source.parts || []).forEach((entry, index) => add(entry, `part-${index + 1}`, `Part ${index + 1}`, "part"));
  const overallGs = items.find((item) => item.kind === "overall")?.gs || null;
  items.forEach((item) => { item.fallbackGs ||= item.gs || overallGs; });
  if (!items.some((item) => item.visible) && items[0]) items[0].visible = true;
  return { title: source.title || raw.title || "Decoded components", components: items };
}

function assetFor(record) { return mode === "mesh" ? record.mesh : record.gs; }

function childUrl(record) {
  const url = new URL("../viewer.html", import.meta.url);
  url.searchParams.set("embedded", "1");
  if (mode === "mesh") {
    if (!record.fallbackGs) throw new Error(`Mesh layer ${record.label} needs fallback_gaussian_url or an overall GS`);
    url.searchParams.set("mesh", record.mesh);
    url.searchParams.set("content", record.fallbackGs || record.gs);
  } else {
    url.searchParams.set("content", record.gs);
  }
  return url.toString();
}

function setInteractiveLayer(preferred = activeId) {
  const visible = records.filter((record) => record.visible && assetFor(record));
  const selected = visible.find((record) => record.id === preferred) || visible.at(-1) || null;
  activeId = selected ? selected.id : null;
  records.forEach((record) => {
    record.frame?.classList.toggle("interactive", record.id === activeId);
    record.button?.classList.toggle("activeLayer", record.id === activeId);
  });
}

function renderLegend() {
  legendEl.innerHTML = "";
  records.forEach((record) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `legendItem${record.visible ? "" : " off"}`;
    button.style.setProperty("--swatch", record.color);
    const swatch = document.createElement("span");
    swatch.className = "legendSwatch";
    const label = document.createElement("span");
    label.textContent = record.label;
    const kind = document.createElement("span");
    kind.className = "legendKind";
    kind.textContent = record.kind;
    button.append(swatch, label, kind);
    button.disabled = !assetFor(record);
    button.title = assetFor(record) ? `${record.visible ? "Hide" : "Show"} ${record.label}` : `${mode.toUpperCase()} output unavailable`;
    button.addEventListener("click", () => {
      record.visible = !record.visible;
      button.classList.toggle("off", !record.visible);
      if (record.frame) record.frame.hidden = !record.visible;
      if (record.visible) activeId = record.id;
      setInteractiveLayer();
      if (record.visible && record.frame && lastCamera) {
        record.frame.contentWindow.postMessage({ type: "fridge3dgs.cameraState", camera: lastCamera }, "*");
      }
    });
    record.button = button;
    legendEl.appendChild(button);
  });
  setInteractiveLayer();
}

function renderLayers() {
  layersEl.innerHTML = "";
  records.forEach((record) => {
    record.frame = null;
    if (!assetFor(record)) return;
    const frame = document.createElement("iframe");
    frame.className = "componentLayer";
    frame.title = `${record.label} ${mode}`;
    frame.src = childUrl(record);
    frame.hidden = !record.visible;
    frame.dataset.componentId = record.id;
    layersEl.appendChild(frame);
    record.frame = frame;
  });
  renderLegend();
  stateEl.textContent = `${mode === "mesh" ? "Mesh" : "Gaussian splats"} · ${records.filter((item) => assetFor(item)).length} available`;
}

function setMode(next) {
  mode = next;
  document.querySelectorAll("[data-mode]").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  renderLayers();
}

window.addEventListener("message", (event) => {
  const msg = event.data || {};
  if (msg.type === "fridge3dgs.viewerReady") {
    const readyRecord = records.find((record) => record.frame && record.frame.contentWindow === event.source);
    if (readyRecord?.visible && lastCamera) {
      readyRecord.frame.contentWindow.postMessage({ type: "fridge3dgs.cameraState", camera: lastCamera }, "*");
    }
    return;
  }
  if (msg.type !== "fridge3dgs.cameraChanged" || relayingCamera) return;
  const sourceRecord = records.find((record) => record.frame && record.frame.contentWindow === event.source);
  if (!sourceRecord || sourceRecord.id !== activeId) return;
  lastCamera = msg.camera || lastCamera;
  relayingCamera = true;
  records.forEach((record) => {
    if (record.visible && record.frame && record !== sourceRecord) {
      record.frame.contentWindow.postMessage({ type: "fridge3dgs.cameraState", camera: msg.camera }, "*");
    }
  });
  window.setTimeout(() => { relayingCamera = false; }, 0);
});

document.querySelectorAll("[data-mode]").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
document.getElementById("showAll").addEventListener("click", () => {
  const available = records.filter((record) => assetFor(record));
  const shouldShow = available.some((record) => !record.visible);
  available.forEach((record) => { record.visible = shouldShow; if (record.frame) record.frame.hidden = !shouldShow; });
  renderLegend();
});

async function init() {
  if (!manifestUrl) throw new Error("Missing ?manifest=<json-url>");
  const resolved = new URL(manifestUrl, window.location.href);
  const response = await fetch(resolved);
  if (!response.ok) throw new Error(`Manifest request failed (${response.status})`);
  manifest = normalize(await response.json(), resolved);
  records = manifest.components;
  if (!records.length) throw new Error("Manifest contains no viewable components");
  titleEl.textContent = manifest.title;
  setMode(mode);
}

init().catch((error) => {
  stateEl.textContent = "Unable to load viewer";
  const message = document.createElement("div");
  message.className = "componentError";
  message.textContent = String(error.message || error);
  layersEl.replaceChildren(message);
});
