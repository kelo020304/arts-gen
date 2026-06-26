// Visible build marker. If the header shows this tag and the console logs it,
// the browser is running THIS app.js (not a stale cached copy).
const BUILD_TAG = "infer-frontend-2026-06-04-stage-reuse";

// Valid 采样方式 per task — MUST match the python argparse `choices`:
//   eval (eval_part_ss_latent_flow_full.py):       first | spread | all
//   test (export_part_ss_latent_flow_examples.py): first | spread
// "random" was never a valid backend value (it raises unknown sample_mode).
const SAMPLE_MODES = {
  eval: ["all", "first", "spread"],
  test: ["first", "spread"],
};

const NEW_JOB_DEFAULTS = {
  task_type: "eval",
  view_mode: "four",
  experiment_name: "",
  output_root: "/robot/data-lab/jzh/art-gen-output",
  checkpoint: "",
  load_dir: "",
  step: "",
  gpu_ids: "0",
  max_samples: "-1",
  sample_mode: "all",
  object_ids: "",
  overrides: "",
};

const state = {
  view: "home",
  summary: { running: 0, completed: 0 },
  experiments: [],
  jobs: [],
  selected: new Set(),
  details: new Map(),
  activeLogJob: null,
  metricTab: "core",
  gpus: [],
  // Checkpoint dropdown options for the eval job form (/api/eval/options).
  evalCheckpoints: [],
  // 指标说明 panel starts COLLAPSED; toggled via [data-toggle-explain].
  explainOpen: false,
  // Controlled form state for the "new job" view. Persisted here so any
  // re-render (5s poll, opening a log modal, ...) restores typed values
  // instead of wiping them.
  form: { ...NEW_JOB_DEFAULTS },
  kin: {
    roots: [],
    runs: [],
    results: [],
    selectedRoot: "",
    objectId: "",
    sourceRunId: "",
    angleIdx: "0",
    outputRoot: "/root/code/arts-gen/kin_test/eval_platform_runs",
    testDataRoot: "/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-single-image-0512",
    gpuIds: "0",
    agentLoop: false,
    skipMotionValidation: true,
    overwrite: false,
    activeRunId: "",
    log: "",
    error: "",
  },
};

function route(path) {
  const base = new URL(".", window.location.href);
  return new URL(path.replace(/^\/+/, ""), base).toString();
}

const metricOrder = [
  "target_iou",
  "part_iou",
  "recall",
  "precision",
  "f1",
  "count_error",
  "empty_rate",
  "confusion",
  "binding_diag",
  "small_confusion",
  "small_binding_diag",
  "small_recall",
  "small_empty_rate",
  "multi_target_iou",
  "multi_worst_part_iou",
  "size_gap_target_iou",
];

const primaryMetrics = ["target_iou", "part_iou", "recall", "f1", "empty_rate", "count_error"];

// v11 metric tabs. Each key is read from detail.metrics.overall[key] ?? focused[key].
const METRIC_TABS = {
  core: {
    label: "基础质量",
    keys: ["target_iou", "part_iou", "recall", "precision", "f1", "count_error", "interior_recall"],
  },
  hard: {
    label: "复杂物体",
    keys: ["multi_target_iou", "multi_worst_part_iou", "size_gap_target_iou", "scale_rel_error", "size_ratio_error", "small_recall"],
  },
  failure: {
    label: "失败诊断",
    keys: ["confusion", "binding_diag", "small_confusion", "small_binding_diag", "empty_rate", "small_empty_rate", "count_error"],
  },
};
const METRIC_TAB_ORDER = ["core", "hard", "failure"];

// Keys rendered as Math.round(v*100)+'%'. All others use fmt() (3 decimals).
const PERCENT_KEYS = new Set(["count_error", "empty_rate", "small_empty_rate"]);
// Lower-is-better keys flip the color thresholds. All other keys are higher-better.
const LOWER_BETTER_KEYS = new Set([
  "count_error", "empty_rate", "confusion", "small_confusion", "small_empty_rate", "scale_rel_error", "size_ratio_error",
]);

// Read a metric value (overall first, then focused) for one detail. null when absent/NaN.
function metricValueFor(detail, key) {
  const metrics = detail.metrics || {};
  const entry = metrics.overall?.[key] ?? metrics.focused?.[key];
  if (!entry) return null;
  const v = Number(entry.value);
  return Number.isFinite(v) ? v : null;
}

// Find the {label} for a metric key, preferring metric entries then metric_definitions.
function metricLabelFor(details, key) {
  for (const detail of details) {
    const entry = detail.metrics?.overall?.[key] ?? detail.metrics?.focused?.[key];
    if (entry?.label) return entry.label;
  }
  const def = details[0]?.metrics?.metric_definitions?.[key];
  if (def?.label) return def.label;
  return key;
}

// Color class per contract. null -> '' (muted via .num default).
function metricColorClass(key, value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "";
  const v = Number(value);
  if (LOWER_BETTER_KEYS.has(key)) {
    if (v <= 0.10) return "green-text";
    if (v <= 0.30) return "amber-text";
    return "red-text";
  }
  if (v >= 0.65) return "green-text";
  if (v >= 0.40) return "amber-text";
  return "red-text";
}

function fillColorClass(key, value) {
  return metricColorClass(key, value).replace("-text", "");
}

// Direction indicator like in papers: ↑ higher-is-better, ↓ lower-is-better.
function metricArrow(key) {
  return LOWER_BETTER_KEYS.has(key) ? "↓" : "↑";
}

// Display text: percent keys -> rounded %, others -> 3-decimal fmt(). null -> '-'.
function metricNumberText(key, value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "-";
  if (PERCENT_KEYS.has(key)) return `${Math.round(Number(value) * 100)}%`;
  return fmt(value);
}

// Bar width %: clamp(round(value*100), 2, 100) for any numeric value, 0 when null.
function metricBarWidth(key, value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return 0;
  return Math.max(2, Math.min(100, Math.round(Number(value) * 100)));
}

function qs(sel, root = document) {
  return root.querySelector(sel);
}

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const n = Number(value);
  if (Math.abs(n) >= 100) return n.toFixed(1);
  if (Math.abs(n) >= 10) return n.toFixed(2);
  return n.toFixed(3);
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function api(path, options = {}) {
  const resp = await fetch(route(path), options);
  const data = await resp.json();
  if (!resp.ok) {
    const err = new Error(data?.error?.message || `HTTP ${resp.status}`);
    err.code = data?.error?.code;
    err.status = resp.status;
    throw err;
  }
  return data;
}

async function refreshBase() {
  const [summary, experiments, jobs, gpus] = await Promise.all([
    api("/api/summary"),
    api("/api/experiments"),
    api("/api/jobs"),
    api("/api/gpus"),
  ]);
  state.summary = summary;
  state.experiments = experiments.experiments || [];
  state.jobs = jobs.jobs || [];
  state.gpus = gpus.gpus || [];
  // Checkpoint dropdown options. Tolerant: an older server without the endpoint
  // must not break the whole base refresh — the field stays a free-text input.
  try {
    const opts = await api("/api/eval/options");
    state.evalCheckpoints = opts.checkpoints || [];
  } catch (err) {
    state.evalCheckpoints = state.evalCheckpoints || [];
  }
}

function setView(view) {
  state.view = view;
  render();
  if (view === "runs") {
    hydrateSelectedExperiments();
  } else if (view === "kinematic") {
    refreshKinematic();
  }
}

function shell() {
  return `
    <header class="topbar">
      <button class="brand" data-action="home">
        <span>Part SS</span>
        <b>Eval Console</b>
      </button>
      <div class="status-pills">
        <button class="pill-link" data-action="running" title="查看运行中实验 + GPU 显存">运行中 <b>${state.summary.running || 0}</b></button>
        <span>已完成 <b>${state.summary.completed || 0}</b></span>
        <span class="build-tag" title="前端构建版本">build ${BUILD_TAG}</span>
      </div>
    </header>
    <main id="main"></main>
  `;
}

function renderHome() {
  return `
    <section class="entry">
      <div class="entry-grid">
        <button class="entry-card" data-action="runs">
          <span>01</span>
          <b>查看已跑实验</b>
          <small>比较 full eval / test export 指标</small>
        </button>
        <button class="entry-card accent" data-action="new">
          <span>02</span>
          <b>启动新的推理</b>
          <small>从 checkpoint 发起 eval 或样例导出</small>
        </button>
        <button class="entry-card" data-action="infer">
          <span>03</span>
          <b>全流程推理</b>
          <small>多视角图 → part flow → 逐 part mesh/GS 重建</small>
        </button>
        <button class="entry-card" data-action="kinematic">
          <span>04</span>
          <b>Kinematic 评测</b>
          <small>stage3 part → GT-as-VLM → joint limit solver</small>
        </button>
      </div>
    </section>
  `;
}

function jobStatusMeta(status) {
  if (status === "running") return { cls: "running", label: "运行中" };
  if (status === "completed") return { cls: "completed", label: "已完成" };
  if (status === "failed") return { cls: "failed", label: "失败" };
  if (status === "terminated") return { cls: "terminated", label: "已终止" };
  return { cls: "pending", label: status || "等待中" };
}

function renderRunChips() {
  const chips = state.jobs.map(job => {
    const meta = jobStatusMeta(job.status);
    const prog = job.progress;
    // Real progress (parsed from the eval log) only meaningful while running.
    const known = meta.cls === "running" && prog && Number(prog.total) > 0;
    const pct = known ? Math.round(Number(prog.fraction) * 100) : null;
    const label = known ? `运行中 ${pct}% · ${prog.done}/${prog.total}` : meta.label;
    const bar = known
      ? `<i class="is-determinate" style="width:${pct}%"></i>`
      : "<i></i>";
    const resultBtn = meta.cls === "completed"
      ? `<button class="chip-link" data-result-job="${esc(job.id)}">查看结果 →</button>`
      : meta.cls === "running"
        ? `<button class="chip-link danger" data-term-job="${esc(job.id)}">终止</button>`
        : "";
    // A div (not <button>) so it can hold the inner "查看结果" button.
    return `
    <div class="run-chip status-${meta.cls}" data-log-job="${esc(job.id)}" title="点击查看日志">
      <div class="run-chip-head">
        <span class="run-chip-name">${esc(job.name)}</span>
        <div class="run-chip-actions">
          ${resultBtn}
          <b class="run-chip-status">${esc(label)}</b>
        </div>
      </div>
      <div class="run-chip-bar">${bar}</div>
    </div>`;
  }).join("");
  return chips || "<span class='muted'>暂无运行任务</span>";
}

function gpuById(index) {
  return state.gpus.find(g => Number(g.index) === Number(index));
}

function gb(mb) {
  return (Number(mb) / 1024).toFixed(1);
}

// Per-job GPU + free/total VRAM text, e.g. "GPU 0 · 空闲 12.3/24.0 GB · 占用 35%".
function jobGpuText(gpuIds) {
  const ids = String(gpuIds || "").split(",").map(s => s.trim()).filter(Boolean);
  if (!ids.length) return "—";
  return ids.map(id => {
    const g = gpuById(id);
    if (!g) return `GPU ${esc(id)}`;
    return `GPU ${id} · 空闲 ${gb(g.free_mb)}/${gb(g.total_mb)} GB · 占用 ${g.util}%`;
  }).join("；");
}

function renderGpuCards() {
  if (!state.gpus.length) return "<span class='muted'>读不到 GPU（nvidia-smi 不可用）</span>";
  return state.gpus.map(g => {
    const pct = g.total_mb ? Math.round((g.used_mb / g.total_mb) * 100) : 0;
    const fill = pct > 85 ? "red" : pct > 60 ? "amber" : "green";
    return `
      <div class="gpu-card">
        <div class="gpu-card-head"><b>GPU ${g.index}</b><span>占用 ${g.util}%</span></div>
        <div class="gpu-mem">空闲 <b>${gb(g.free_mb)}</b> / 总 ${gb(g.total_mb)} GB<span class="muted">（已用 ${gb(g.used_mb)}）</span></div>
        <div class="bar"><span class="fill ${fill}" style="width:${pct}%"></span></div>
      </div>`;
  }).join("");
}

function renderRunning() {
  const running = state.jobs.filter(job => job.status === "running");
  const rows = running.map(job => {
    const prog = job.progress;
    const progText = prog && Number(prog.total) > 0
      ? `${Math.round(Number(prog.fraction) * 100)}% · ${prog.done}/${prog.total}`
      : "初始化中";
    return `
      <div class="running-row">
        <div class="running-name"><b>${esc(job.name)}</b><span class="muted">${esc(job.task_type)} / ${esc(job.view_mode)}</span></div>
        <div class="running-gpu">${esc(jobGpuText(job.gpu_ids))}</div>
        <div class="running-prog">${esc(progText)}</div>
        <div class="running-actions">
          <button class="chip-link" data-log-job="${esc(job.id)}">日志</button>
          <button class="chip-link danger" data-term-job="${esc(job.id)}">终止</button>
        </div>
      </div>`;
  }).join("");
  return `
    <section class="running-view">
      <div class="panel">
        <div class="panel-head">
          <h2>运行中的实验</h2>
          <button class="ghost" data-action="home">返回</button>
        </div>
        <div class="panel-body">
          <div class="section-label">GPU 显存</div>
          <div class="gpu-grid">${renderGpuCards()}</div>
          <div class="section-label">运行中（${running.length}）</div>
          <div class="running-list">${rows || "<div class='no-metrics'><b>当前没有运行中的实验</b></div>"}</div>
        </div>
      </div>
    </section>
    ${state.activeLogJob ? renderLogModal() : ""}
  `;
}

async function refreshKinematic({ renderAfter = true } = {}) {
  try {
    const rootsData = await api("/api/infer/roots");
    state.kin.roots = rootsData.roots || [];
    if (!state.kin.selectedRoot && state.kin.roots.length) {
      const preferred = state.kin.roots.find(entry =>
        String(entry.path || entry.root || "").includes("/full-stage")
      );
      const selected = preferred || state.kin.roots[0];
      state.kin.selectedRoot = selected.path || selected.root || "";
    }
    if (state.kin.selectedRoot) {
      const runsData = await api(`/api/infer/runs?root=${encodeURIComponent(state.kin.selectedRoot)}`);
      state.kin.runs = runsData.runs || [];
    } else {
      state.kin.runs = [];
    }
    const resultsData = await api(`/api/kin/runs?root=${encodeURIComponent(state.kin.outputRoot || "")}`);
    state.kin.results = resultsData.runs || [];
    state.kin.error = "";
  } catch (err) {
    state.kin.error = err.message || String(err);
  }
  if (renderAfter && state.view === "kinematic") render();
}

function rootPathOf(entry) {
  return entry?.path || entry?.root || "";
}

function kinRootOptions() {
  return (state.kin.roots || []).map(entry => {
    const path = rootPathOf(entry);
    const label = entry.name ? `${entry.name} (${path})` : path;
    const selected = state.kin.selectedRoot === path ? " selected" : "";
    return `<option value="${esc(path)}"${selected}>${esc(label)}</option>`;
  }).join("");
}

function kinFilteredRuns() {
  const objectId = state.kin.objectId.trim();
  const runs = state.kin.runs || [];
  return runs
    .filter(run => !objectId || String(run.object_id || "").includes(objectId))
    .filter(run => {
      const ss = run.stage_status || {};
      return ss.slat === "done" || ss.assemble === "done";
    })
    .slice(0, 400);
}

function kinRunOptions() {
  return kinFilteredRuns().map(run => {
    const selected = state.kin.sourceRunId === run.run_id ? " selected" : "";
    const label = `${run.object_id}-${run.angle_idx ?? 0} · ${run.run_id} · ${run.mode || ""}/${run.view || ""}`;
    return `<option value="${esc(run.run_id)}" data-object="${esc(run.object_id)}" data-angle="${esc(run.angle_idx ?? 0)}"${selected}>${esc(label)}</option>`;
  }).join("");
}

async function startKinematicJob() {
  const k = state.kin;
  if (!k.selectedRoot || !k.objectId.trim() || !k.sourceRunId.trim()) {
    window.alert("请先选择 stage3 root、object_id 和 source run");
    return;
  }
  const payload = {
    source_root: k.selectedRoot,
    object_id: k.objectId.trim(),
    source_run_id: k.sourceRunId.trim(),
    angle_idx: Number.parseInt(k.angleIdx || "0", 10) || 0,
    output_root: k.outputRoot.trim(),
    test_data_root: k.testDataRoot.trim(),
    gpu_ids: k.gpuIds.trim() || "0",
    agent_loop: Boolean(k.agentLoop),
    skip_motion_validation: Boolean(k.skipMotionValidation),
    overwrite: Boolean(k.overwrite),
  };
  try {
    const data = await api("/api/kin/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    k.activeRunId = data.run_id || "";
    await refreshBase();
    await refreshKinematic({ renderAfter: false });
    render();
  } catch (err) {
    window.alert(`启动 kinematic 失败：${err.message || err}`);
  }
}

async function loadKinematicLog(runId) {
  const k = state.kin;
  k.activeRunId = runId || k.activeRunId;
  if (!k.activeRunId) return;
  try {
    const data = await api(
      `/api/kin/log?root=${encodeURIComponent(k.outputRoot || "")}&run_id=${encodeURIComponent(k.activeRunId)}`
    );
    k.log = data.log || "暂无日志";
  } catch (err) {
    k.log = err.message || String(err);
  }
  render();
}

function kinArtifactUrl(runId, rel) {
  return route(
    `/api/kin/artifact?root=${encodeURIComponent(state.kin.outputRoot || "")}` +
      `&run_id=${encodeURIComponent(runId)}` +
      `&rel=${encodeURIComponent(rel)}`
  );
}

function renderKinematicResults() {
  const rows = (state.kin.results || []).map(run => {
    const active = state.kin.activeRunId === run.run_id ? " is-selected" : "";
    const report = run.has_report
      ? `<a class="kin-link" href="${esc(kinArtifactUrl(run.run_id, "candidate_report.json"))}" target="_blank">report</a>`
      : "<span class='muted'>report</span>";
    const pred = run.has_predictions
      ? `<a class="kin-link" href="${esc(kinArtifactUrl(run.run_id, "predictions.jsonl"))}" target="_blank">predictions</a>`
      : "<span class='muted'>predictions</span>";
    const stateLink = run.has_frontend_state
      ? `<a class="kin-link" href="${esc(kinArtifactUrl(run.run_id, "frontend_state.json"))}" target="_blank">frontend_state</a>`
      : "<span class='muted'>frontend_state</span>";
    return `
      <div class="kin-result${active}">
        <div>
          <b>${esc(run.run_id)}</b>
          <span class="muted">${esc(run.object_id || "")} · angle ${esc(run.angle_idx ?? 0)} · joints ${esc(run.joint_count ?? 0)}</span>
          <span class="badge-row">
            <span class="badge ${run.status === "done" ? "done" : "eval"}">${esc(run.status || "prepared")}</span>
          </span>
        </div>
        <div class="kin-actions">
          <button class="ghost" data-kin-log="${esc(run.run_id)}">日志</button>
          ${report}
          ${pred}
          ${stateLink}
        </div>
      </div>
    `;
  }).join("");
  return rows || "<div class='no-metrics'><b>还没有 kinematic run</b></div>";
}

function renderKinematic() {
  const k = state.kin;
  return `
    <section class="kinematic-view">
      <div class="panel">
        <div class="panel-head">
          <h2>Kinematic 评测</h2>
          <button class="ghost" data-action="home">返回</button>
        </div>
        <div class="panel-body">
          ${k.error ? `<div class="diagnostic-card"><b>加载失败</b><span>${esc(k.error)}</span></div>` : ""}
          <div class="kin-grid">
            <label>stage3 root
              <select name="kin_root">${kinRootOptions()}</select>
            </label>
            <label>object_id
              <input name="kin_object_id" value="${esc(k.objectId)}" placeholder="101940">
            </label>
            <label>source run
              <select name="kin_source_run"><option value="">选择 slat done 的 run</option>${kinRunOptions()}</select>
            </label>
            <label>angle_idx
              <input name="kin_angle_idx" type="number" min="0" step="1" value="${esc(k.angleIdx)}">
            </label>
            <label>输出 root
              <input name="kin_output_root" value="${esc(k.outputRoot)}">
            </label>
            <label>测试数据 root
              <input name="kin_test_data_root" value="${esc(k.testDataRoot)}">
            </label>
            <label>GPU
              <input name="kin_gpu_ids" value="${esc(k.gpuIds)}">
            </label>
          </div>
          <div class="kin-toggles">
            <label class="check-row"><input type="checkbox" name="kin_skip_motion"${k.skipMotionValidation ? " checked" : ""}> 跳过 motion validation</label>
            <label class="check-row"><input type="checkbox" name="kin_agent_loop"${k.agentLoop ? " checked" : ""}> agent loop</label>
            <label class="check-row"><input type="checkbox" name="kin_overwrite"${k.overwrite ? " checked" : ""}> 覆盖同名 run</label>
          </div>
          <div class="kin-submit">
            <button class="primary" data-kin-start>启动 kinematic</button>
            <button class="ghost" data-kin-refresh>刷新</button>
          </div>
          <div class="stage-card-meta">输入来自 stage3 的 <code>parts/body.glb</code> 和 <code>parts/part_*.glb</code>；VLM 初值暂用测试集 <code>part_info</code> 真值。</div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h2>结果</h2>
          <span class="pill">${esc((k.results || []).length)} runs</span>
        </div>
        <div class="panel-body">
          <div class="kin-results">${renderKinematicResults()}</div>
          ${k.activeRunId ? `<pre class="kin-log">${esc(k.log || "点“日志”查看输出")}</pre>` : ""}
        </div>
      </div>
    </section>
    ${state.activeLogJob ? renderLogModal() : ""}
  `;
}

function renderNew() {
  const f = state.form;
  const sel = (field, value) => (f[field] === value ? " selected" : "");
  const sampleModes = SAMPLE_MODES[f.task_type] || SAMPLE_MODES.eval;
  const sampleModeOptions = sampleModes
    .map(m => `<option value="${m}"${sel("sample_mode", m)}>${m}</option>`)
    .join("");
  // Checkpoint dropdown (datalist) — pick a scanned ckpt OR type any path.
  const ckptOptions = (state.evalCheckpoints || [])
    .map(c => `<option value="${esc(c.path)}">${esc(c.label)}</option>`)
    .join("");
  return `
    <section class="workbench">
      <div class="panel config-panel">
        <div class="panel-head">
          <h1>新推理任务</h1>
          <button class="ghost" data-action="home">返回</button>
        </div>
        <form id="jobForm" class="form-grid">
          <label>任务类型
            <select name="task_type">
              <option value="eval"${sel("task_type", "eval")}>Eval</option>
              <option value="test"${sel("task_type", "test")}>Test</option>
            </select>
          </label>
          <label>视角
            <select name="view_mode">
              <option value="four"${sel("view_mode", "four")}>4-view</option>
              <option value="single"${sel("view_mode", "single")}>1-view</option>
            </select>
          </label>
          <label>实验名称
            <input name="experiment_name" value="${esc(f.experiment_name)}" placeholder="part-ss-eval-0529" required>
          </label>
          <label>输出位置
            <input name="output_root" value="${esc(f.output_root)}" placeholder="/robot/data-lab/jzh/art-gen-output" required>
          </label>
          <label>Checkpoint
            <input name="checkpoint" value="${esc(f.checkpoint)}" placeholder="选择或输入 /path/to/step_200000.pt" list="evalCkptOptions" autocomplete="off">
            <datalist id="evalCkptOptions">${ckptOptions}</datalist>
          </label>
          <label>LOAD_DIR
            <input name="load_dir" value="${esc(f.load_dir)}" placeholder="/path/to/run">
          </label>
          <label>STEP
            <input name="step" type="number" value="${esc(f.step)}" placeholder="200000">
          </label>
          <label>GPU
            <input name="gpu_ids" value="${esc(f.gpu_ids)}" placeholder="0">
          </label>
          <label>样本数量
            <input name="max_samples" type="number" value="${esc(f.max_samples)}" placeholder="-1">
          </label>
          <label>采样方式
            <select name="sample_mode">${sampleModeOptions}</select>
          </label>
          <label class="wide">Object IDs
            <input name="object_ids" value="${esc(f.object_ids)}" placeholder="100013,100214">
          </label>
          <label class="wide">额外 Overrides
            <textarea name="overrides" placeholder="training.foo=bar&#10;loss.decode_aware_weight=0.02">${esc(f.overrides)}</textarea>
          </label>
          <button class="primary wide" type="submit">启动任务</button>
        </form>
      </div>
      <aside class="panel summary-panel">
        <h2>任务摘要</h2>
        <dl id="jobPreview"></dl>
      </aside>
    </section>
    <section class="bottom-tray">
      <div class="tray-title">运行中的实验</div>
      <div class="chip-row">${renderRunChips()}</div>
    </section>
    ${state.activeLogJob ? renderLogModal() : ""}
  `;
}

function experimentById(id) {
  return state.experiments.find(exp => exp.id === id);
}

function placeholderDetail(exp, status = "loading", message = "正在载入指标") {
  if (!exp) return null;
  const diagnostics = {
    status,
    message,
    missing: [],
    optional_missing: [],
    errors: [],
    report_dir: exp.report_dir || "",
    export_dir: exp.export_dir || "",
  };
  return {
    ...exp,
    metrics: {
      task_kind: exp.kind,
      overall: {},
      focused: {},
      diagnostics,
    },
    examples: [],
    artifacts: {},
  };
}

function selectedRecords() {
  return [...state.selected]
    .map(id => state.details.get(id) || placeholderDetail(experimentById(id)))
    .filter(Boolean);
}

function metricValue(detail, key) {
  const metrics = detail.metrics || {};
  return metrics.overall?.[key]?.value ?? metrics.focused?.[key]?.value;
}

function metricMeta(details, key) {
  for (const detail of details) {
    const metric = detail.metrics?.overall?.[key] || detail.metrics?.focused?.[key];
    if (metric && Number.isFinite(Number(metric.value))) return metric;
  }
  return null;
}

async function loadExperimentDetail(id) {
  if (state.details.has(id)) return state.details.get(id);
  try {
    const detail = await api(`/api/experiments/${id}`);
    state.details.set(id, detail.experiment);
    return detail.experiment;
  } catch (err) {
    console.error(err);
    const fallback = placeholderDetail(
      experimentById(id),
      "error",
      `指标载入失败: ${err.message || err}`,
    );
    if (fallback) {
      fallback.metrics.diagnostics.errors = [{ path: "", message: String(err.message || err) }];
      state.details.set(id, fallback);
      return fallback;
    }
    throw err;
  }
}

async function hydrateSelectedExperiments({ rerender = true } = {}) {
  if (!state.experiments.length) return;
  const validIds = new Set(state.experiments.map(exp => exp.id));
  let changed = false;
  for (const id of [...state.selected]) {
    if (!validIds.has(id)) {
      state.selected.delete(id);
      state.details.delete(id);
      changed = true;
    }
  }
  if (!state.selected.size) {
    state.selected.add(state.experiments[0].id);
    changed = true;
  }
  for (const id of [...state.selected]) {
    if (!state.details.has(id)) {
      await loadExperimentDetail(id);
      changed = true;
    }
  }
  if (rerender && changed && state.view === "runs") render();
}

function viewLabel(exp) {
  if (exp.view_mode === "single" || exp.view === "single" || exp.views === 1) return "1-view";
  if (exp.view_mode === "four" || exp.view === "four" || exp.views === 4) return "4-view";
  return exp.view_mode || exp.view || "";
}

function expMeta(exp) {
  const kindLabel = exp.kind === "eval" ? "Eval" : "Test";
  const loc = exp.kind === "eval" ? (exp.report_dir || "") : (exp.export_dir || "");
  return loc ? `${kindLabel} · ${loc}` : kindLabel;
}

function renderRuns() {
  const runItems = state.experiments.map(exp => {
    const selected = state.selected.has(exp.id);
    return `
      <div class="run-item${selected ? " is-selected" : ""}" data-toggle-exp="${esc(exp.id)}">
        <button class="run-del" data-del-exp="${esc(exp.id)}" title="删除实验">✕</button>
        <span class="box">${selected ? "✓" : ""}</span>
        <span>
          <span class="run-name">${esc(exp.name)}</span>
          <span class="run-meta">${esc(expMeta(exp))}</span>
          <span class="badge-row">
            <span class="badge eval">${exp.kind === "eval" ? "Eval" : "Test"}</span>
            <span class="badge done">完成</span>
          </span>
        </span>
      </div>
    `;
  }).join("");

  const details = selectedRecords();
  const workspace = details.length ? renderWorkspace(details) : renderEmptyWorkspace();

  return `
    <section class="layout">
      <aside class="panel">
        <div class="panel-head">
          <h2>选择对比组</h2>
          <button class="ghost" data-action="home">返回</button>
        </div>
        <div class="panel-body">
          <div class="filters">
            <input class="input" placeholder="搜索实验名 / step / object">
            <select class="select"><option>全部类型</option><option>Eval</option><option>Test</option></select>
          </div>
          <div class="run-list">${runItems || "<span class='muted'>未扫描到实验</span>"}</div>
        </div>
      </aside>
      <section class="workspace">${workspace}</section>
    </section>
  `;
}

function renderEmptyWorkspace() {
  return `
    <div class="panel">
      <div class="panel-head">
        <h2>实验对比</h2>
        <span class="pill">已选择 0 个实验</span>
      </div>
      <div class="panel-body">
        <div class="no-metrics">
          <b>还没有选中的实验</b>
          <span>点左侧任意实验行即可载入指标；支持多选对比。</span>
        </div>
      </div>
    </div>
  `;
}

function renderWorkspace(details) {
  // One single-detail + examples block PER selected experiment, so the cards
  // and 标志性样例 correspond to what is selected on the left (not just the first).
  const perExp = details
    .map(detail => `${renderSingleDetailPanel(detail)}${renderCasesPanel(detail)}`)
    .join("");
  return `${renderComparePanel(details)}${perExp}`;
}

function renderComparePanel(details) {
  const tab = METRIC_TABS[state.metricTab] || METRIC_TABS.core;
  const keys = tab.keys;
  const gridStyle = `grid-template-columns:144px repeat(${keys.length},minmax(0,1fr));`;

  const chips = METRIC_TAB_ORDER.map(id => `
    <div class="metric-chip${id === state.metricTab ? " is-active" : ""}" data-metric-tab="${id}">${esc(METRIC_TABS[id].label)}</div>
  `).join("");

  const head = `
    <div class="matrix-head" style="${gridStyle}">
      <div>实验</div>
      ${keys.map(key => `<div>${esc(metricLabelFor(details, key))} <span class="dir-arrow" title="${LOWER_BETTER_KEYS.has(key) ? "越低越好" : "越高越好"}">${metricArrow(key)}</span></div>`).join("")}
    </div>
  `;

  const rows = details.map(detail => {
    const cells = keys.map(key => {
      const value = metricValueFor(detail, key);
      const numCls = metricColorClass(key, value);
      const fillCls = fillColorClass(key, value);
      const width = metricBarWidth(key, value);
      return `
        <div class="cell">
          <span class="num ${numCls}">${esc(metricNumberText(key, value))}</span>
          <div class="bar"><span class="fill ${fillCls}" style="width:${width}%"></span></div>
        </div>
      `;
    }).join("");
    return `
      <div class="matrix-row" style="${gridStyle}">
        <div>
          <div class="run-title">${esc(detail.name)}</div>
          <div class="run-view">${esc(detail.kind === "eval" ? "Eval" : "Test")} · ${esc(viewLabel(detail))}</div>
        </div>
        ${cells}
      </div>
    `;
  }).join("");

  return `
    <div class="panel">
      <div class="panel-head">
        <h2>实验对比</h2>
        <span class="pill">已选择 ${details.length} 个实验</span>
      </div>
      <div class="panel-body">
        <div class="compare-toolbar">
          <div class="metric-switch">${chips}</div>
          <span class="pill">指标说明见下方</span>
        </div>
        ${renderDiagnostics(details)}
        <div class="matrix">${head}${rows}</div>
        ${renderMetricExplain(details[0])}
      </div>
    </div>
  `;
}

function renderMetricExplain(detail) {
  const defs = detail?.metrics?.metric_definitions;
  if (!defs || typeof defs !== "object") return "";
  const rows = Object.entries(defs).map(([key, def]) => `
    <div class="explain-row">
      <strong>${esc(def.label)} <span class="dir-arrow" title="${LOWER_BETTER_KEYS.has(key) ? "越低越好" : "越高越好"}">${metricArrow(key)}</span></strong>
      <span>${esc(def.formula)}</span>
      <span>${esc(def.meaning)}</span>
    </div>
  `).join("");
  if (!rows) return "";
  return `
    <div class="metric-explain">
      <button class="explain-toggle" data-toggle-explain>
        <span>指标说明</span><span class="explain-arrow">${state.explainOpen ? "▾" : "▸"}</span>
      </button>
      ${state.explainOpen ? '<div class="explain-rows">' + rows + '</div>' : ''}
    </div>
  `;
}

// Heuristic "主要问题" card: pick the weakest signal across a few key metrics.
function primaryIssue(detail) {
  const recall = metricValueFor(detail, "recall");
  const smallRecall = metricValueFor(detail, "small_recall");
  const emptyRate = metricValueFor(detail, "empty_rate");
  const confusion = metricValueFor(detail, "confusion");
  if (smallRecall !== null && smallRecall < 0.4) {
    return { label: "小部件漏检", cls: "red-text", note: `small recall ${fmt(smallRecall)}` };
  }
  if (emptyRate !== null && emptyRate > 0.1) {
    return { label: "空预测偏高", cls: "red-text", note: `empty rate ${metricNumberText("empty_rate", emptyRate)}` };
  }
  if (confusion !== null && confusion > 0.3) {
    return { label: "部件混淆", cls: "amber-text", note: `confusion ${fmt(confusion)}` };
  }
  if (recall !== null && recall < 0.4) {
    return { label: "整体召回低", cls: "red-text", note: `recall ${fmt(recall)}` };
  }
  return { label: "表现稳定", cls: "green-text", note: "无显著短板" };
}

function detailCard(title, value, valueCls, note) {
  return `
    <div class="detail-card">
      <h3>${esc(title)}</h3>
      <strong class="${valueCls}">${esc(value)}</strong>
      <p>${esc(note)}</p>
    </div>
  `;
}

function renderSingleDetailPanel(detail) {
  if (!detail) return "";
  const issue = primaryIssue(detail);
  const multi = metricValueFor(detail, "multi_target_iou");
  const countError = metricValueFor(detail, "count_error");
  const confusion = metricValueFor(detail, "confusion");
  return `
    <div class="panel">
      <div class="panel-head">
        <h2>${esc(detail.name)}</h2>
        <span class="pill">单实验指标</span>
      </div>
      <div class="panel-body">
        <div class="single-detail">
          ${detailCard("主要问题", issue.label, issue.cls, issue.note)}
          ${detailCard("复杂物体", metricNumberText("multi_target_iou", multi), metricColorClass("multi_target_iou", multi), "6+ part target IoU")}
          ${detailCard("数量误差", metricNumberText("count_error", countError), metricColorClass("count_error", countError), "pred/raw voxel count")}
          ${detailCard("部件混淆", metricNumberText("confusion", confusion), metricColorClass("confusion", confusion), "offdiag assignment max")}
        </div>
      </div>
    </div>
  `;
}

function renderDiagnostics(details) {
  const items = details
    .map(detail => ({ detail, diagnostics: detail.metrics?.diagnostics || detail.diagnostics }))
    .filter(item => item.diagnostics && item.diagnostics.status !== "ok");
  if (!items.length) return "";
  return `
    <section class="diagnostic-list">
      ${items.map(({ detail, diagnostics }) => {
        const missing = diagnostics.missing || [];
        const optionalMissing = diagnostics.optional_missing || [];
        const errors = diagnostics.errors || [];
        const location = diagnostics.report_dir || diagnostics.export_dir || detail.root || "";
        return `
          <article class="diagnostic-card ${diagnostics.status === "loading" ? "loading" : ""}">
            <div>
              <b>${esc(detail.name)}</b>
              <span>${esc(diagnostics.message || diagnostics.status || "指标不完整")}</span>
              ${location ? `<code>${esc(location)}</code>` : ""}
            </div>
            ${missing.length ? `<p><strong>缺少</strong>${missing.map(item => `<em>${esc(item)}</em>`).join("")}</p>` : ""}
            ${optionalMissing.length ? `<p><strong>可选缺少</strong>${optionalMissing.map(item => `<em>${esc(item)}</em>`).join("")}</p>` : ""}
            ${errors.length ? `<ul>${errors.map(err => `<li>${esc(err.path || "parse")} ${esc(err.message || err)}</li>`).join("")}</ul>` : ""}
          </article>
        `;
      }).join("")}
    </section>
  `;
}

function artifactUrl(detail, rel) {
  const prefix = detail.kind === "eval" && detail.report_dir ? `${detail.report_dir}/` : "";
  return route(`/artifacts/${detail.id}/${prefix}${rel}`);
}

// Normalize an example for legacy data without kind/metrics fields.
function exampleKind(ex) {
  if (ex.kind === "good" || ex.kind === "bad") return ex.kind;
  return String(ex.group || "").startsWith("best_") ? "good" : "bad";
}

function caseMetricInline(label, key, value) {
  const cls = metricColorClass(key, value);
  return `<span>${esc(label)} <b class="${cls}">${esc(metricNumberText(key, value))}</b></span>`;
}

// Pick up to `n` good + `n` bad examples, de-duplicated by object so the same
// object never shows twice (special buckets + general fallback can overlap).
function pickCaseExamples(detail, perKind = 3) {
  const all = detail?.examples || [];
  const seen = new Set();
  const pick = kind => {
    const out = [];
    for (const ex of all) {
      if (exampleKind(ex) !== kind) continue;
      const key = String(ex.obj_id ?? ex.dataset_index ?? "");
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(ex);
      if (out.length >= perKind) break;
    }
    return out;
  };
  const good = pick("good");
  const bad = pick("bad");
  return [...good, ...bad];
}

function renderCasesPanel(detail) {
  const examples = pickCaseExamples(detail);
  if (!examples.length) {
    return `
      <div class="panel">
        <div class="panel-head">
          <h2>标志性样例 · ${esc(detail?.name || "")}</h2>
          <span class="pill">输入图与 voxel 对比</span>
        </div>
        <div class="panel-body">
          <div class="no-metrics">
            <b>本次评估没有样例 voxel 图</b>
            <span>通常是因为：样本数太少没凑出好/坏样例，或这个 run 是旧代码跑的（没有样例富化）。用最新代码、把"样本数量"调大些（如 ≥10）重跑即可生成解码 voxel 图。</span>
          </div>
        </div>
      </div>
    `;
  }
  const rows = examples.map(ex => {
    const kind = exampleKind(ex);
    const tagLabel = kind === "good" ? "好样例" : "坏样例";
    const m = ex.metrics || {};
    const visual = ex.png
      ? `<img class="case-panel" src="${artifactUrl(detail, ex.png)}" alt="${esc(ex.label || "voxel example")}">`
      : `<div class="case-panel-empty">无 voxel 图</div>`;
    return `
      <div class="case-row">
        <div class="case-head">
          <div class="case-meta">
            <span class="case-tag ${kind}">${tagLabel}</span>
            <span class="case-obj">obj ${esc(ex.obj_id)}</span>
            ${ex.label ? `<span class="case-sub">${esc(ex.label)}</span>` : ""}
          </div>
          <div class="case-metrics-inline">
            ${caseMetricInline("Recall", "recall", m.recall ?? null)}
            ${caseMetricInline("Precision", "precision", m.precision ?? null)}
            ${caseMetricInline("数量误差", "count_error", m.count_error ?? null)}
            ${caseMetricInline("混淆", "confusion", m.confusion ?? null)}
            ${caseMetricInline("最差部件", "part_iou", m.worst_part ?? null)}
          </div>
        </div>
        <div class="case-visual">
          <div class="visual-head">输入图 + GT / Pred / Overlap（解码 voxel）</div>
          ${visual}
        </div>
      </div>
    `;
  }).join("");
  return `
    <div class="panel">
      <div class="panel-head">
        <h2>标志性样例 · ${esc(detail.name)}</h2>
        <span class="pill">输入图与 voxel 对比</span>
      </div>
      <div class="panel-body">${rows}</div>
    </div>
  `;
}

function renderLogModal() {
  const job = state.jobs.find(item => item.id === state.activeLogJob);
  return `
    <div class="modal">
      <div class="terminal-panel">
        <div class="terminal-head">
          <b>${job?.name || "日志"}</b>
          <button class="ghost" data-close-log>关闭</button>
        </div>
        <pre id="terminalLog">loading...</pre>
      </div>
    </div>
  `;
}

function render() {
  document.body.innerHTML = shell();
  const main = qs("#main");
  if (state.view === "home") main.innerHTML = renderHome();
  if (state.view === "new") main.innerHTML = renderNew();
  if (state.view === "runs") main.innerHTML = renderRuns();
  if (state.view === "running") main.innerHTML = renderRunning();
  if (state.view === "infer") {
    main.innerHTML = '<section class="infer-view" id="infer-root"></section>';
    if (window.mountInfer) window.mountInfer(document.getElementById("infer-root"), api);
  }
  if (state.view === "kinematic") main.innerHTML = renderKinematic();
  bindEvents();
  updatePreview();
  if (state.activeLogJob) loadLog(state.activeLogJob);
}

// Match a finished job to its scanned experiment (by run dir, then by name).
function experimentForJob(job) {
  if (!job) return null;
  return state.experiments.find(exp =>
    (job.run_dir && exp.root === job.run_dir) ||
    String(exp.name || "").trim() === String(job.name || "").trim(),
  );
}

// "查看结果" — jump to the runs view with this job's experiment selected.
async function openJobResult(jobId) {
  const job = state.jobs.find(item => item.id === jobId);
  const exp = experimentForJob(job);
  if (exp) {
    state.selected = new Set([exp.id]);
    setView("runs");
    return;
  }
  // Not scanned yet (just finished) — rescan once, then retry.
  await refreshBase();
  const exp2 = experimentForJob(job);
  if (exp2) state.selected = new Set([exp2.id]);
  setView("runs");
  if (!exp2) window.alert("结果还没扫描到，稍等几秒刷新后再点。");
}

function bindResultButtons(root) {
  root.querySelectorAll("[data-result-job]").forEach(el => {
    el.addEventListener("click", event => {
      event.stopPropagation();  // don't also open the log modal
      openJobResult(el.dataset.resultJob);
    });
  });
}

async function terminateJob(jobId) {
  const job = state.jobs.find(item => item.id === jobId);
  if (!window.confirm(`确定终止运行中的实验 "${job?.name || jobId}"？\n会杀掉它的整个进程树并释放 GPU，不可恢复。`)) return;
  try {
    await api(`/api/jobs/${jobId}/terminate`, { method: "POST" });
  } catch (err) {
    window.alert(`终止失败：${err.message || err}`);
    return;
  }
  await refreshBase();
  render();
}

function bindTermButtons(root) {
  root.querySelectorAll("[data-term-job]").forEach(el => {
    el.addEventListener("click", event => {
      event.stopPropagation();  // don't also open the log modal
      terminateJob(el.dataset.termJob);
    });
  });
}

// Wire the per-run delete buttons. Bound in bindEvents and re-bindable anywhere
// run-items are produced.
function bindDeleteButtons(root) {
  root.querySelectorAll("[data-del-exp]").forEach(el => {
    el.addEventListener("click", async event => {
      event.stopPropagation();  // don't toggle selection
      const id = el.dataset.delExp;
      const exp = experimentById(id);
      const name = exp?.name || id;
      if (!window.confirm(`确定删除实验 "${name}"？这会删除磁盘上的 run 目录，不可恢复。`)) return;
      try {
        await api("/api/experiments/" + id, { method: "DELETE" });  // api() throws on !ok
        state.selected.delete(id);
        state.details.delete(id);
        await refreshBase();
        render();
      } catch (err) {
        window.alert("删除失败：" + (err.message || err));
      }
    });
  });
}

function bindEvents() {
  document.querySelectorAll("[data-action]").forEach(el => {
    el.addEventListener("click", () => setView(el.dataset.action === "home" ? "home" : el.dataset.action));
  });
  const kinPanel = qs(".kinematic-view");
  if (kinPanel) {
    const syncKin = () => {
      const val = name => qs(`[name="${name}"]`, kinPanel)?.value ?? "";
      state.kin.selectedRoot = val("kin_root");
      state.kin.objectId = val("kin_object_id");
      state.kin.sourceRunId = val("kin_source_run");
      state.kin.angleIdx = val("kin_angle_idx");
      state.kin.outputRoot = val("kin_output_root");
      state.kin.testDataRoot = val("kin_test_data_root");
      state.kin.gpuIds = val("kin_gpu_ids");
      state.kin.skipMotionValidation = Boolean(qs('[name="kin_skip_motion"]', kinPanel)?.checked);
      state.kin.agentLoop = Boolean(qs('[name="kin_agent_loop"]', kinPanel)?.checked);
      state.kin.overwrite = Boolean(qs('[name="kin_overwrite"]', kinPanel)?.checked);
    };
    kinPanel.addEventListener("input", syncKin);
    kinPanel.addEventListener("change", event => {
      syncKin();
      if (event.target?.name === "kin_source_run") {
        const opt = event.target.selectedOptions?.[0];
        if (opt?.dataset.object) state.kin.objectId = opt.dataset.object;
        if (opt?.dataset.angle) state.kin.angleIdx = opt.dataset.angle;
        render();
      } else if (event.target?.name === "kin_root") {
        refreshKinematic();
      } else {
        render();
      }
    });
    qs("[data-kin-start]", kinPanel)?.addEventListener("click", startKinematicJob);
    qs("[data-kin-refresh]", kinPanel)?.addEventListener("click", () => refreshKinematic());
    kinPanel.querySelectorAll("[data-kin-log]").forEach(el => {
      el.addEventListener("click", () => loadKinematicLog(el.dataset.kinLog));
    });
  }
  document.querySelectorAll("[data-toggle-exp]").forEach(el => {
    el.addEventListener("click", async () => {
      const id = el.dataset.toggleExp;
      if (state.selected.has(id)) {
        if (!state.details.has(id)) {
          render();
          await loadExperimentDetail(id);
        } else if (state.selected.size > 1) {
          state.selected.delete(id);
          render();
          return;
        }
      } else {
        state.selected.add(id);
        render();
        await loadExperimentDetail(id);
      }
      render();
    });
  });
  document.querySelectorAll("[data-metric-tab]").forEach(el => {
    el.addEventListener("click", () => {
      const tab = el.dataset.metricTab;
      if (METRIC_TABS[tab]) {
        state.metricTab = tab;
        render();
      }
    });
  });
  bindDeleteButtons(document);
  const explainToggle = qs("[data-toggle-explain]");
  if (explainToggle) explainToggle.addEventListener("click", () => {
    state.explainOpen = !state.explainOpen;
    render();
  });
  document.querySelectorAll("[data-log-job]").forEach(el => {
    el.addEventListener("click", () => {
      state.activeLogJob = el.dataset.logJob;
      render();
    });
  });
  bindResultButtons(document);
  bindTermButtons(document);
  const close = qs("[data-close-log]");
  if (close) close.addEventListener("click", () => {
    state.activeLogJob = null;
    render();
  });
  const form = qs("#jobForm");
  if (form) {
    form.addEventListener("input", () => {
      captureForm();
      updatePreview();
    });
    // Task type changes the set of valid 采样方式 — refresh options and drop
    // any value the new task's backend wouldn't accept (e.g. test + "all").
    form.addEventListener("change", event => {
      if (event.target.name !== "task_type") return;
      captureForm();
      const valid = SAMPLE_MODES[state.form.task_type] || SAMPLE_MODES.eval;
      if (!valid.includes(state.form.sample_mode)) state.form.sample_mode = valid[0];
      render();
    });
    form.addEventListener("submit", submitJob);
  }
}

// Persist the current form field values into state.form so they survive the
// next render() (5s poll, log modal, view switch). Only known fields are
// copied, keeping state.form's shape stable.
function captureForm() {
  const form = qs("#jobForm");
  if (!form) return;
  const fd = new FormData(form);
  for (const key of Object.keys(state.form)) {
    if (fd.has(key)) state.form[key] = String(fd.get(key) ?? "");
  }
}

function formPayload(form) {
  const fd = new FormData(form);
  // Trim text fields so a stray space (e.g. in experiment_name) never reaches
  // the backend path. The server trims again as defense-in-depth.
  const t = key => String(fd.get(key) ?? "").trim();
  return {
    task_type: fd.get("task_type"),
    view_mode: fd.get("view_mode"),
    experiment_name: t("experiment_name"),
    output_root: t("output_root"),
    checkpoint: t("checkpoint"),
    load_dir: t("load_dir"),
    step: t("step"),
    gpu_ids: t("gpu_ids"),
    max_samples: Number(fd.get("max_samples")),
    sample_mode: fd.get("sample_mode"),
    object_ids: t("object_ids"),
    overrides: String(fd.get("overrides") || "").split(/\n+/).map(s => s.trim()).filter(Boolean),
  };
}

function updatePreview() {
  const form = qs("#jobForm");
  const preview = qs("#jobPreview");
  if (!form || !preview) return;
  const p = formPayload(form);
  preview.innerHTML = `
    <dt>任务</dt><dd>${p.task_type} / ${p.view_mode}</dd>
    <dt>实验</dt><dd>${p.experiment_name || "-"}</dd>
    <dt>输出</dt><dd>${p.output_root || "-"}</dd>
    <dt>权重</dt><dd>${p.checkpoint || `${p.load_dir || "-"} @ ${p.step || "-"}`}</dd>
    <dt>GPU</dt><dd>${p.gpu_ids || "-"}</dd>
    <dt>样本</dt><dd>${p.sample_mode} / ${p.max_samples}</dd>
  `;
}

function experimentNameExists(name) {
  const target = String(name || "").trim();
  if (!target) return false;
  const inRuns = state.experiments.some(exp => String(exp.name || "").trim() === target);
  const inJobs = state.jobs.some(job => String(job.name || "").trim() === target);
  return inRuns || inJobs;
}

function overwriteMessage(name) {
  return `实验名 "${name}" 已存在。\n点“确定”覆盖重跑，点“取消”返回改名字。`;
}

async function submitJob(event) {
  event.preventDefault();
  const payload = formPayload(event.currentTarget);
  if (!payload.experiment_name) {
    window.alert("请填写实验名称");
    return;
  }
  // Optimistic pre-check against known experiments/jobs (saves a round-trip).
  if (experimentNameExists(payload.experiment_name)) {
    if (!window.confirm(overwriteMessage(payload.experiment_name))) return;
    payload.overwrite = true;
  }
  await postJob(payload);
}

async function postJob(payload) {
  try {
    await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    // Backend found an existing run dir the pre-check didn't know about (e.g. a
    // failed run with no report): offer the overwrite choice, don't hard-fail.
    if (err.code === "experiment_exists" && !payload.overwrite) {
      if (!window.confirm(overwriteMessage(payload.experiment_name))) return;
      return postJob({ ...payload, overwrite: true });
    }
    window.alert(`启动失败：${err.message || err}`);
    return;
  }
  await refreshBase();
  render();
}

async function loadLog(jobId) {
  const terminal = qs("#terminalLog");
  if (!terminal) return;
  try {
    const payload = await api(`/api/jobs/${jobId}/log`);
    terminal.textContent = payload.log || "暂无日志";
  } catch (err) {
    terminal.textContent = String(err.message || err);
  }
}

// Patch only the live regions (header pills + running-jobs tray) in place,
// without rebuilding the whole body. Used by the poll on the "new"/"home"
// views so an in-progress form (focus, cursor, typed text) is never disturbed.
function updateLiveRegions() {
  const pills = qs(".status-pills");
  if (pills) {
    pills.innerHTML = `
      <button class="pill-link" data-action="running" title="查看运行中实验 + GPU 显存">运行中 <b>${state.summary.running || 0}</b></button>
      <span>已完成 <b>${state.summary.completed || 0}</b></span>
      <span class="build-tag" title="前端构建版本">build ${BUILD_TAG}</span>
    `;
    pills.querySelectorAll("[data-action]").forEach(el => {
      el.addEventListener("click", () => setView(el.dataset.action));
    });
  }
  const chipRow = qs(".chip-row");
  if (chipRow) {
    chipRow.innerHTML = renderRunChips();
    chipRow.querySelectorAll("[data-log-job]").forEach(el => {
      el.addEventListener("click", () => {
        state.activeLogJob = el.dataset.logJob;
        render();
      });
    });
    bindResultButtons(chipRow);
    bindTermButtons(chipRow);
  }
}

async function boot() {
  console.log(`[part_ss_eval_platform] app.js build = ${BUILD_TAG}`);
  await refreshBase();
  render();
  setInterval(async () => {
    await refreshBase();
    if (state.view === "runs") {
      await hydrateSelectedExperiments({ rerender: false });
      render();
      return;
    }
    if (state.view === "running") {
      // GPU/VRAM + progress change over time — full re-render is fine (no form).
      render();
      return;
    }
    if (state.view === "kinematic") {
      await refreshKinematic({ renderAfter: false });
      render();
      return;
    }
    // home / new: never full-render here — it would wipe the form's focus and
    // typed-but-uncaptured keystrokes. Patch live regions only.
    updateLiveRegions();
    if (state.activeLogJob) loadLog(state.activeLogJob);
  }, 5000);
}

boot().catch(err => {
  document.body.innerHTML = `<pre class="boot-error">${err.message || err}</pre>`;
});
