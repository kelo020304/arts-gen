// static/infer.js  (ES module)
//
// 推理视图逻辑：表单 / 输入预览 / 4 阶段卡（轮询 manifest）/ 共享 3D 查看器加载。
// 由 app.js（普通脚本）通过 window.mountInfer(container, api) 调用；api 是 app.js 的
// fetch helper（throw on !ok，返回已解析 JSON）。
//
// 设计要点：
//   - 表单值持久在模块级 formState，re-render（轮询触发）不丢用户输入。
//   - manifest 每 5s 轮询一次，状态来自 meta.stage_status；门控：上一阶段 done 才能 [运行]。
//   - [看产物] 复用同一个 window.Viewer3D 实例（懒建），缺失则文本兜底，绝不静默崩溃。
//   - 后端 query 参数用 object_id / run_id（与 server.py 一致），不是 object / run。

const STAGES = ["ss", "part", "slat", "assemble"];

const STAGE_LABELS = {
  ss: "SS（TRELLIS SS Flow / SAM3D → surface voxel）",
  part: "TRELLIS Encode + Part Flow / Promptable Seg",
  slat: "SAM3D SLat（part voxel → mesh/GS）",
  assemble: "组装（合并 part 产物）",
};

const STATUS_LABELS = {
  pending: "未开始",
  running: "运行中",
  done: "已完成",
  failed: "失败",
};

// 默认表单值。output_root 留空 → 用 roots 下拉第一个；
// data_config / ckpt 由 /api/infer/options 填下拉；ss_flow_ckpt 留空表示继续走默认 SAM3D SS。
// 选择 denoiser/ss_flow ckpt 后，mode-B 的 SS 阶段切到 TRELLIS SS Flow。
//
// 下拉字段（data_config / ckpt）的契约：
//   - <field>          下拉选中的路径（"__custom__" 表示走自定义）
//   - <field>_custom   选「+ 自定义路径」时手填的路径
//   - use_custom_<field> 下拉是否切到「+ 自定义路径」
// effectiveOption() 据此算出最终生效路径（自定义优先），提交时一律用生效路径。
const DEFAULT_FORM = {
  mode: "B",
  view: "four",
  angle_idx: "0",
  object_prefix1: "",
  object_prefix2: "",
  object_prefix3: "",
  object_prefix4: "",
  object_id: "",
  output_root: "",      // 选中的 root 路径（来自 /api/infer/roots 或新建）
  new_root: "/robot/data-lab/jzh/art-gen-output/full-stage", // 「+ 新建」时的手填路径
  use_new_root: false,  // root 下拉是否切到「新建」
  data_config: "",
  data_config_custom: "",
  use_custom_data_config: false,
  part_backend: "promptable_seg",
  part_flow_ckpt: "",
  part_flow_ckpt_custom: "",
  use_custom_part_flow_ckpt: false,
  part_seg_ckpt: "",
  part_seg_ckpt_custom: "",
  use_custom_part_seg_ckpt: false,
  ss_flow_ckpt: "",
  ss_flow_ckpt_custom: "",
  use_custom_ss_flow_ckpt: false,
  gpu_ids: "0",
};

// 下拉字段名 → options 来源（用于渲染 <select> 与默认选中）。
const OPTION_FIELDS = ["data_config", "part_flow_ckpt", "part_seg_ckpt", "ss_flow_ckpt"];
const OPTIONAL_OPTION_FIELDS = new Set(["ss_flow_ckpt"]);

// 模块级状态：跨 re-render 持久。
const formState = { ...DEFAULT_FORM };
const moduleState = {
  api: null,
  container: null,
  subview: "config",    // 'config'（表单 + 输入预览 + [进入推理]）| 'run'（阶段卡）
  roots: [],            // [{root|path, name}]
  configs: [],          // /api/infer/options 的 configs：[{path, label}]
  checkpoints: [],      // /api/infer/options 的 checkpoints：[{path, label}]
  objects: [],          // /api/infer/objects: [{object_id, angles, name, category, target_part_names}]
  inputs: null,         // /api/infer/inputs 返回（含 rgb_paths / has_gt_voxel）
  inputsError: "",
  manifest: null,       // /api/infer/manifest 返回（含 meta.stage_status）
  manifestError: "",
  stageOutputs: {},     // /api/infer/stage_outputs 返回，供覆盖确认/按钮提示
  runId: "",            // 浏览器生成，提交时下发；同一 object/run 复用
  runReuseInfo: "",      // 进入运行页时本次打开的是复用 run 还是新 run
  stageJobs: {},        // stage -> 最近一次提交的 job id（用于 [停止]）
  viewer: null,         // 复用的 window.Viewer3D 实例
  viewerHostStage: null,// 当前 viewer 对应的 stage（主可视化面板）
  activeStage: "ss",    // 主可视化面板当前聚焦的阶段
  activeKind: "voxel",  // 主可视化面板当前聚焦的产物类型
  activeArtifact: null, // {stage, kind}：render 后用于恢复主 viewer
  meshLegend: [],       // 当前 mesh 的 part legend：[{index, stem, label, color, ...}]
  selectedMeshPart: null,
  partLabelCache: {},   // cacheKey -> {stem -> label}
  stageLogs: {},        // stage -> 最近一次拉取的日志文本
  pollTimer: null,
  loadingInputs: false,
  autoShown: new Set(),  // 已自动加载过产物的 stage（如 SS done → 自动 showArtifact voxel）
  autoLogged: new Set(), // 已自动拉过失败日志的 stage（failed → 自动 fetchStageLog）
};

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

// run_<YYYYMMDDHHMMSS>：14 位零填充时间戳（本地时钟分量；月份 +1）。
// 同一秒内连续生成会撞 id，故对重复时间戳追加一个短计数后缀。
let _lastRunStamp = "";
let _runStampSeq = 0;
function makeRunId() {
  const d = new Date();
  const pad = (n, w = 2) => String(n).padStart(w, "0");
  const stamp =
    pad(d.getFullYear(), 4) +
    pad(d.getMonth() + 1) +
    pad(d.getDate()) +
    pad(d.getHours()) +
    pad(d.getMinutes()) +
    pad(d.getSeconds());
  if (stamp === _lastRunStamp) {
    _runStampSeq += 1;
    return "run_" + stamp + "_" + _runStampSeq;
  }
  _lastRunStamp = stamp;
  _runStampSeq = 0;
  return "run_" + stamp;
}

// roots 条目兼容 {root,name} 与 {path,name}（后端 list_roots 当前给 path）。
function rootPathOf(entry) {
  return entry?.root ?? entry?.path ?? "";
}

// 当前生效的输出 root 路径（新建优先）。
function effectiveRoot() {
  return formState.use_new_root ? formState.new_root.trim() : formState.output_root.trim();
}

function runRootPath() {
  const root = effectiveRoot().replace(/\/+$/, "");
  const objectId = formState.object_id.trim();
  const runId = moduleState.runId;
  if (!root || !objectId || !runId) return "";
  return `${root}/${objectId}-${effectiveAngleIdx()}/${runId}`;
}

// 下拉字段（data_config / ckpt）当前生效的路径：自定义优先，否则取下拉选中值。
function effectiveOption(field) {
  if (formState[`use_custom_${field}`]) return String(formState[`${field}_custom`] || "").trim();
  return String(formState[field] || "").trim();
}

function checkpointOptionsFor(field) {
  if (field === "part_seg_ckpt") {
    return (moduleState.checkpoints || []).filter(item => {
      const text = `${item.label || ""} ${item.path || ""}`;
      return /part[-_]promptable[-_]seg/i.test(text) || /promptable.*seg/i.test(text);
    });
  }
  if (field !== "ss_flow_ckpt") return moduleState.checkpoints;
  return (moduleState.checkpoints || []).filter(item => {
    const text = `${item.label || ""} ${item.path || ""}`;
    return (
      /(^|\/)denoiser(?:_ema[0-9.]+)?_step\d+\.pt$/i.test(item.path || item.label || "") ||
      /ss[-_]flow/i.test(text)
    );
  });
}

function effectiveAngleIdx() {
  const n = Number.parseInt(String(formState.angle_idx ?? "0").trim() || "0", 10);
  return Number.isFinite(n) && n >= 0 ? n : 0;
}

// ---- 数据加载 ----

async function loadRoots() {
  try {
    const data = await moduleState.api("/api/infer/roots");
    moduleState.roots = data.roots || [];
    // 默认选中第一个（若用户还没选过）。
    if (!formState.output_root && !formState.use_new_root && moduleState.roots.length) {
      const preferred = moduleState.roots.find(
        entry => rootPathOf(entry).replace(/\/+$/, "") === DEFAULT_FORM.new_root,
      );
      formState.output_root = rootPathOf(preferred || moduleState.roots[0]);
    }
    if (!moduleState.roots.length && !formState.output_root && !formState.use_new_root) {
      formState.use_new_root = true;
      formState.new_root = formState.new_root || DEFAULT_FORM.new_root;
    }
  } catch (err) {
    moduleState.roots = [];
    if (!formState.output_root && !formState.use_new_root) {
      formState.use_new_root = true;
      formState.new_root = formState.new_root || DEFAULT_FORM.new_root;
    }
  }
  render();
}

async function loadOptions() {
  try {
    const data = await moduleState.api("/api/infer/options");
    moduleState.configs = data.configs || [];
    moduleState.checkpoints = data.checkpoints || [];
  } catch (err) {
    moduleState.configs = [];
    moduleState.checkpoints = [];
  }
  // 默认选中（仅当用户还没选过、且没切到自定义时）。
  if (
    !formState.data_config &&
    !formState.use_custom_data_config &&
    moduleState.configs.length
  ) {
    const preferred = moduleState.configs.find(
      c => c.label === "part_ss_latent_flow/part_ss_latent_flow.yaml",
    );
    formState.data_config = (preferred || moduleState.configs[0]).path;
  }
  if (
    !formState.part_flow_ckpt &&
    !formState.use_custom_part_flow_ckpt &&
    moduleState.checkpoints.length
  ) {
    // part flow ckpt is a TRELLIS training output (step_*.pt under a
    // part-ss-latent-flow run) — NEVER a .safetensors. Prefer that explicitly so
    // a hard refresh doesn't leave the dropdown on the first entry, which (since
    // sam3d weights joined the scan) can be a .safetensors that torch.load can't
    // read -> "invalid load key, 'x'". Fall back to any step_*.pt.
    const pf =
      moduleState.checkpoints.find(
        c => /part[-_]ss[-_]latent[-_]flow/i.test(String(c.label)) &&
             String(c.label).endsWith(".pt")) ||
      moduleState.checkpoints.find(c => /(^|\/)step_\d+\.pt$/i.test(String(c.label)));
    if (pf) formState.part_flow_ckpt = pf.path;
  }
  if (
    !formState.part_seg_ckpt &&
    !formState.use_custom_part_seg_ckpt &&
    moduleState.checkpoints.length
  ) {
    const ps =
      moduleState.checkpoints.find(c =>
        /part[-_]promptable[-_]seg[-_]full[-_]M[-_]0612-3/i.test(String(c.path || c.label)) &&
        /\/latest\.pt$/i.test(String(c.path || c.label))) ||
      checkpointOptionsFor("part_seg_ckpt").find(c => /\/latest\.pt$/i.test(String(c.path || c.label))) ||
      checkpointOptionsFor("part_seg_ckpt").find(c => /(^|\/)step_\d+\.pt$/i.test(String(c.path || c.label)));
    if (ps) formState.part_seg_ckpt = ps.path;
  }
  if (
    !formState.ss_flow_ckpt &&
    !formState.use_custom_ss_flow_ckpt &&
    moduleState.checkpoints.length
  ) {
    const sf =
      moduleState.checkpoints.find(c => /tre_mf_4view_multiflow_0611.*denoiser_ema0\.9999_step0020000\.pt/i.test(String(c.path || c.label))) ||
      checkpointOptionsFor("ss_flow_ckpt").find(c => /denoiser_ema0\.9999_step0020000\.pt$/i.test(String(c.path || c.label))) ||
      checkpointOptionsFor("ss_flow_ckpt").find(c => /denoiser_step0020000\.pt$/i.test(String(c.path || c.label)));
    if (sf) formState.ss_flow_ckpt = sf.path;
  }
  render();
}

async function loadObjects() {
  try {
    const data = await moduleState.api("/api/infer/objects?limit=200000");
    moduleState.objects = data.objects || [];
  } catch (err) {
    moduleState.objects = [];
  }
  render();
}

function selectedObjectInfo() {
  const objectId = formState.object_id.trim();
  if (!objectId) return null;
  return moduleState.objects.find(item => String(item.object_id) === objectId) || null;
}

function objectOptions() {
  return filteredObjects()
    .map(item => {
      const parts = (item.target_part_names || []).slice(0, 3).join(",");
      const angles = (item.angles || []).join(",");
      const label = [
        item.name || item.category || "",
        angles ? `angles:${angles}` : "",
        parts ? `parts:${parts}` : "",
      ].filter(Boolean).join(" · ");
      return `<option value="${esc(item.object_id)}" label="${esc(label)}"></option>`;
    })
    .join("");
}

function objectPrefixes(width, basePrefix = "") {
  const counts = new Map();
  moduleState.objects.forEach(item => {
    const objectId = String(item.object_id || "");
    if (basePrefix && !objectId.startsWith(basePrefix)) return;
    const prefix = objectId.slice(0, width);
    if (!prefix) return;
    counts.set(prefix, (counts.get(prefix) || 0) + 1);
  });
  return Array.from(counts.entries())
    .sort((a, b) => a[0].localeCompare(b[0], undefined, { numeric: true }))
    .map(([prefix, count]) => ({ prefix, count }));
}

function objectPrefixOptions(width, value, basePrefix = "", allLabel = "全部") {
  const prefixes = objectPrefixes(width, basePrefix);
  const opts = prefixes
    .map(({ prefix, count }) => {
      const selected = value === prefix ? " selected" : "";
      return `<option value="${esc(prefix)}"${selected}>${esc(prefix)} (${count})</option>`;
    })
    .join("");
  const total = prefixes.reduce((sum, item) => sum + item.count, 0);
  const allSelected = value ? "" : " selected";
  return `<option value=""${allSelected}>${esc(allLabel)} (${total})</option>` + opts;
}

function filteredObjects() {
  const prefix = String(
    formState.object_prefix4 ||
      formState.object_prefix3 ||
      formState.object_prefix2 ||
      formState.object_prefix1 ||
      "",
  ).trim();
  if (!prefix) return moduleState.objects;
  return moduleState.objects.filter(item => String(item.object_id || "").startsWith(prefix));
}

function selectedObjectVisible() {
  const objectId = formState.object_id.trim();
  if (!objectId) return true;
  return filteredObjects().some(item => String(item.object_id) === objectId);
}

function syncAngleForSelectedObject() {
  const info = selectedObjectInfo();
  const angles = info?.angles || [];
  if (!angles.length) return false;
  const current = effectiveAngleIdx();
  if (angles.includes(current)) return false;
  formState.angle_idx = String(angles[0]);
  return true;
}

async function loadInputs() {
  const objectId = formState.object_id.trim();
  if (!objectId) {
    moduleState.inputs = null;
    moduleState.inputsError = "";
    render();
    return;
  }
  moduleState.loadingInputs = true;
  moduleState.inputsError = "";
  render();
  try {
    const data = await moduleState.api(
      `/api/infer/inputs?object_id=${encodeURIComponent(objectId)}` +
        `&angle_idx=${encodeURIComponent(effectiveAngleIdx())}` +
        `&view_mode=${encodeURIComponent(formState.view)}`,
    );
    moduleState.inputs = data;
    moduleState.inputsError = "";
  } catch (err) {
    moduleState.inputs = null;
    moduleState.inputsError = err?.message || String(err);
  } finally {
    moduleState.loadingInputs = false;
    render();
  }
}

async function loadManifest() {
  const objectId = formState.object_id.trim();
  const root = effectiveRoot();
  const runId = moduleState.runId;
  if (!objectId || !root || !runId) {
    moduleState.manifest = null;
    moduleState.manifestError = "";
    return;
  }
  try {
    const data = await moduleState.api(
      `/api/infer/manifest?root=${encodeURIComponent(root)}` +
        `&object_id=${encodeURIComponent(objectId)}` +
        `&run_id=${encodeURIComponent(runId)}` +
        `&angle_idx=${encodeURIComponent(effectiveAngleIdx())}`,
    );
    moduleState.manifest = data;
    moduleState.manifestError = "";
  } catch (err) {
    // run 尚未创建（404）属正常态：此时所有 stage 视为 pending，不当错误报。
    moduleState.manifest = null;
    moduleState.manifestError = err?.status === 404 ? "" : (err?.message || String(err));
  }
}

// stage_status 的稳定签名，用于判断是否需要重绘（避免每 5s 无脑重绘打断用户输入）。
function stageStatusSignature() {
  const ss = moduleState.manifest?.meta?.stage_status || {};
  return STAGES.map(s => `${s}:${ss[s] || "pending"}`).join("|");
}

function stageOutputSignature() {
  return STAGES.map(stage => {
    const out = moduleState.stageOutputs?.[stage];
    return `${stage}:${out?.exists ? (out.artifacts || []).join(",") : ""}`;
  }).join("|");
}

function startPolling() {
  stopPolling();
  moduleState.pollTimer = window.setInterval(async () => {
    // 容器已从文档移除（用户切走了视图）→ 停止轮询，省资源。
    if (!moduleState.container || !document.body.contains(moduleState.container)) {
      stopPolling();
      return;
    }
    // 仅在运行子视图轮询 manifest：配置子视图没有阶段卡，且重绘会打断表单输入。
    if (moduleState.subview !== "run") return;
    const before = `${stageStatusSignature()}::${stageOutputSignature()}`;
    await loadManifest();
    await loadStageOutputs();
    // 仅当阶段状态变化时重绘，避免无谓地清掉输入焦点 / viewer 画布。
    if (`${stageStatusSignature()}::${stageOutputSignature()}` !== before || moduleState.manifestError) render();
  }, 5000);
}

function stopPolling() {
  if (moduleState.pollTimer) {
    window.clearInterval(moduleState.pollTimer);
    moduleState.pollTimer = null;
  }
}

// ---- 阶段状态推断 ----

function stageStatus(stage) {
  const ss = moduleState.manifest?.meta?.stage_status || {};
  const raw = ss[stage];
  if (raw === "done" || raw === "running" || raw === "failed" || raw === "pending") return raw;
  return "pending";
}

// 门控：第一个 stage 总可运行；其余仅当前一个 stage done 才可运行。
function stageRunnable(stage) {
  const idx = STAGES.indexOf(stage);
  if (idx <= 0) return true;
  return stageStatus(STAGES[idx - 1]) === "done";
}

function focusedStage() {
  return STAGES.includes(moduleState.activeStage) ? moduleState.activeStage : "ss";
}

async function loadStageOutputs() {
  const objectId = formState.object_id.trim();
  const root = effectiveRoot();
  const runId = moduleState.runId;
  if (!objectId || !root || !runId) {
    moduleState.stageOutputs = {};
    return;
  }
  try {
    const data = await moduleState.api(
      `/api/infer/stage_outputs?root=${encodeURIComponent(root)}` +
        `&object_id=${encodeURIComponent(objectId)}` +
        `&run_id=${encodeURIComponent(runId)}` +
        `&angle_idx=${encodeURIComponent(effectiveAngleIdx())}`,
    );
    moduleState.stageOutputs = data.outputs || {};
  } catch (err) {
    moduleState.stageOutputs = {};
  }
}

function stageHasOutput(stage) {
  return Boolean(moduleState.stageOutputs?.[stage]?.exists);
}

function stageOutputList(stage) {
  return moduleState.stageOutputs?.[stage]?.artifacts || [];
}

function overwriteStageMessage(stage) {
  const artifacts = stageOutputList(stage).slice(0, 8).join("\n");
  return (
    `${STAGE_LABELS[stage] || stage} 已有产物或状态已完成。\n` +
    `run_id: ${moduleState.runId}\n` +
    `run root: ${runRootPath()}\n` +
    (artifacts ? `已有文件:\n${artifacts}\n` : "") +
    "点“确定”覆盖重跑该阶段，点“取消”保留并直接查看已有产物。"
  );
}

function defaultArtifactKind(stage) {
  const artifacts = stageOutputList(stage);
  if (artifacts.some(name => name.endsWith(".glb"))) return "mesh";
  if (artifacts.some(name => name.endsWith(".ply"))) return "gaussian";
  if (stage === "slat" || stage === "assemble") return "mesh";
  return "voxel";
}

// ---- 提交 / 终止 ----

// 纯字段校验（不含阶段门控）。[进入推理] 与 validateForRun 共用：
// 进入推理时只校验字段；运行某阶段时再叠加 stageRunnable 门控。
function validateFields() {
  const objectId = formState.object_id.trim();
  if (!objectId) return "请先填写 object_id";
  if (!effectiveRoot()) return "请先选择或填写输出 root";
  if (!effectiveOption("data_config")) return "请先选择或填写 data_config 路径";
  if (formState.part_backend === "promptable_seg") {
    if (!effectiveOption("part_seg_ckpt")) return "请先选择或填写 part promptable seg ckpt";
  } else if (!effectiveOption("part_flow_ckpt")) {
    return "请先选择或填写 part flow ckpt";
  }
  return "";
}

function validateForRun(stage) {
  const fieldProblem = validateFields();
  if (fieldProblem) return fieldProblem;
  if (!stageRunnable(stage)) return "上一阶段尚未完成，无法运行该阶段";
  return "";
}

async function findReusableRun() {
  const root = effectiveRoot();
  const objectId = formState.object_id.trim();
  const wantedMode = formState.mode;
  const wantedView = formState.view;
  const wantedAngle = effectiveAngleIdx();
  if (!root || !objectId) return null;
  try {
    const data = await moduleState.api(
      `/api/infer/latest_run?root=${encodeURIComponent(root)}` +
        `&object_id=${encodeURIComponent(objectId)}` +
        `&mode=${encodeURIComponent(wantedMode)}` +
        `&view=${encodeURIComponent(wantedView)}` +
        `&angle_idx=${encodeURIComponent(wantedAngle)}`,
    );
    if (data.run) return data.run;
  } catch (err) {
    /* Older server fallback below. */
  }
  try {
    const data = await moduleState.api(`/api/infer/runs?root=${encodeURIComponent(root)}`);
    const runs = (data.runs || []).filter(run =>
      String(run.object_id || "") === objectId &&
      run.mode === wantedMode &&
      run.view === wantedView &&
      Number(run.angle_idx ?? -1) === wantedAngle
    );
    if (!runs.length) return null;
    runs.sort((a, b) => String(b.run_id || "").localeCompare(String(a.run_id || "")));
    return runs[0] || null;
  } catch (err) {
    return null;
  }
}

function applyRunOverview(run) {
  if (!run) return;
  if (run.mode === "A" || run.mode === "B") formState.mode = run.mode;
  if (run.view === "single" || run.view === "four") formState.view = run.view;
  if (run.angle_idx !== undefined && run.angle_idx !== null) formState.angle_idx = String(run.angle_idx);
}

function applyManifestMeta() {
  const meta = moduleState.manifest?.meta || {};
  if (meta.mode === "A" || meta.mode === "B") formState.mode = meta.mode;
  if (meta.view === "single" || meta.view === "four") formState.view = meta.view;
  if (meta.angle_idx !== undefined && meta.angle_idx !== null) formState.angle_idx = String(meta.angle_idx);
}

async function runStage(stage, { overwrite = false } = {}) {
  const fieldProblem = validateFields();
  if (fieldProblem) {
    window.alert(fieldProblem);
    return;
  }
  if (!moduleState.runId) moduleState.runId = makeRunId();
  await loadManifest();
  applyManifestMeta();
  await loadStageOutputs();
  const problem = validateForRun(stage);
  if (problem) {
    window.alert(problem);
    return;
  }
  moduleState.activeStage = stage;
  moduleState.activeKind = defaultArtifactKind(stage);
  moduleState.activeArtifact = null;
  if (!overwrite && (stageHasOutput(stage) || stageStatus(stage) === "done")) {
    if (!window.confirm(overwriteStageMessage(stage))) {
      if (stageHasOutput(stage) || stageStatus(stage) === "done") {
        showArtifact(stage, defaultArtifactKind(stage));
      }
      return;
    }
    overwrite = true;
  }
  const body = {
    stage,
    object_id: formState.object_id.trim(),
    root: effectiveRoot(),
    run_id: moduleState.runId,
    mode: formState.mode,
    view: formState.view,
    angle_idx: effectiveAngleIdx(),
    data_config: effectiveOption("data_config"),
    part_backend: formState.part_backend,
    part_flow_ckpt: effectiveOption("part_flow_ckpt"),
    part_seg_ckpt: effectiveOption("part_seg_ckpt"),
    ss_flow_ckpt: effectiveOption("ss_flow_ckpt"),
    decode_backend: "trellis",
    gpu_ids: formState.gpu_ids.trim() || "0",
    overwrite,
  };
  try {
    const data = await moduleState.api("/api/infer/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const jobId = data?.job?.id;
    if (jobId) moduleState.stageJobs[stage] = jobId;
    startPolling();
    await loadManifest();
    await loadStageOutputs();
    render();
  } catch (err) {
    if (err?.code === "infer_stage_exists" && !overwrite) {
      if (!window.confirm(`${err.message || err}\n\n确认覆盖重跑？`)) return;
      return runStage(stage, { overwrite: true });
    }
    window.alert(`运行 ${stage} 失败：${err?.message || err}`);
  }
}

async function stopStage(stage) {
  const jobId = moduleState.stageJobs[stage];
  if (!jobId) {
    window.alert("没有可终止的 job（本会话未从这里启动该阶段）");
    return;
  }
  if (!window.confirm(`确定终止阶段 "${stage}" 的 job？会杀掉进程树并释放 GPU。`)) return;
  try {
    await moduleState.api(`/api/jobs/${jobId}/terminate`, { method: "POST" });
    await loadManifest();
    render();
  } catch (err) {
    window.alert(`终止失败：${err?.message || err}`);
  }
}

// ---- 子视图切换：配置 → 运行 ----

// [进入推理]：先校验字段（不含阶段门控）；不通过则 alert 并停留在配置页。
// 通过后优先复用同 root/object_id 的最近 run_id；没有历史 run 才生成全新 run_id。
// 模式 A 的新 run 会自动跑一次 SS；复用 run 则直接加载已有 manifest/产物。
async function enterInfer() {
  const problem = validateFields();
  if (problem) {
    window.alert(problem);
    return;
  }
  const selectedMode = formState.mode;
  const selectedView = formState.view;
  const selectedAngle = String(effectiveAngleIdx());
  const reusedRun = await findReusableRun();
  if (reusedRun) applyRunOverview(reusedRun);
  formState.mode = selectedMode;
  formState.view = selectedView;
  formState.angle_idx = selectedAngle;
  moduleState.runId = reusedRun?.run_id || makeRunId();
  moduleState.runReuseInfo = reusedRun
    ? `复用 ${reusedRun.object_id}/${reusedRun.run_id}`
    : "新建 run";
  moduleState.stageJobs = {};
  moduleState.stageLogs = {};
  moduleState.manifest = null;
  moduleState.manifestError = "";
  moduleState.stageOutputs = {};
  moduleState.activeStage = "ss";
  moduleState.activeKind = "voxel";
  moduleState.activeArtifact = null;
  moduleState.autoShown = new Set();
  moduleState.autoLogged = new Set();
  moduleState.subview = "run";
  render();
  await loadManifest();
  applyManifestMeta();
  await loadStageOutputs();
  render();
  if (reusedRun) {
    const firstDone = STAGES.find(stage => stageStatus(stage) === "done" || stageHasOutput(stage));
    if (firstDone) showArtifact(firstDone, defaultArtifactKind(firstDone));
    return;
  }
  if (formState.mode === "A") {
    // 模式 A 的 SS 是快速数据拷贝 → 进入即自动运行，立即 done 并解锁后续阶段。
    runStage("ss");
  }
}

// [← 返回配置]：切回配置子视图。下次 [进入推理] 会重新按 root/object/mode/view
// 选择可复用 run；不存在才新建。
function backToConfig() {
  moduleState.subview = "config";
  render();
}

// ---- 产物查看 ----

// route()：把以 "/" 开头的 API 路径解析成「相对当前页面」的绝对 URL，与 app.js 的
// route()/api() 完全一致。关键：平台常经 code-server 的 `/proxy/7861/` 反向代理访问，
// 此时页面在 http://host/proxy/7861/ —— 直接用绝对路径 `/api/...` 会丢掉 `/proxy/7861/`
// 前缀打到 80 端口 → 404（<img src> 与 viewer 的 artifactUrl 不走 api()，必须自己 route）。
// 直连 host:7861 时 base 即根，行为不变。
function route(path) {
  const base = new URL(".", window.location.href);
  return new URL(String(path).replace(/^\/+/, ""), base).toString();
}

// rel 与 kind 的映射：voxel → voxel.bin；mesh → parts/part_00.glb；gaussian → parts/part_00.ply。
const ARTIFACT_RELS = {
  voxel: "voxel.bin",
  mesh: "parts/part_00.glb",
  gaussian: "parts/part_00.ply",
};

function artifactUrl(rel) {
  const root = effectiveRoot();
  const objectId = formState.object_id.trim();
  const runId = moduleState.runId;
  return route(
    "/api/infer/artifact" +
    `?root=${encodeURIComponent(root)}` +
    `&object_id=${encodeURIComponent(objectId)}` +
    `&run_id=${encodeURIComponent(runId)}` +
    `&angle_idx=${encodeURIComponent(effectiveAngleIdx())}` +
    `&rel=${encodeURIComponent(rel)}`,
  );
}

function artifactUrls(rels) {
  return rels.map(rel => artifactUrl(rel));
}

function stageMeshRels(stage) {
  return stageOutputList(stage)
    .filter(name => name.endsWith(".glb"))
    .sort(componentArtifactCompare);
}

function componentArtifactCompare(a, b) {
  const ka = componentArtifactKey(a);
  const kb = componentArtifactKey(b);
  return (
    ka.group - kb.group ||
    ka.index - kb.index ||
    String(a).localeCompare(String(b))
  );
}

function componentArtifactKey(rel) {
  const name = String(rel || "").split("/").pop() || "";
  const stem = name.replace(/\.[^.]+$/, "");
  if (stem === "overall") return { group: -1, index: -1 };
  if (stem === "body") return { group: 0, index: -1 };
  const m = /^part_(\d+)$/.exec(stem);
  if (m) return { group: 1, index: Number(m[1]) };
  return { group: 2, index: Number.MAX_SAFE_INTEGER };
}

// part 阶段的逐 part voxel：合并端点（每体素带 part 标签，前端按 part 上色）。
function partVoxelsUrl() {
  const root = effectiveRoot();
  const objectId = formState.object_id.trim();
  const runId = moduleState.runId;
  return route(
    "/api/infer/part_voxels" +
    `?root=${encodeURIComponent(root)}` +
    `&object_id=${encodeURIComponent(objectId)}` +
    `&run_id=${encodeURIComponent(runId)}` +
    `&angle_idx=${encodeURIComponent(effectiveAngleIdx())}`,
  );
}

function partLabelsUrl() {
  const root = effectiveRoot();
  const objectId = formState.object_id.trim();
  const runId = moduleState.runId;
  return "/api/infer/part_labels" +
    `?root=${encodeURIComponent(root)}` +
    `&object_id=${encodeURIComponent(objectId)}` +
    `&run_id=${encodeURIComponent(runId)}` +
    `&angle_idx=${encodeURIComponent(effectiveAngleIdx())}`;
}

function partLabelCacheKey() {
  return [
    effectiveRoot(),
    formState.object_id.trim(),
    moduleState.runId,
    effectiveAngleIdx(),
  ].join("::");
}

async function loadPartLabelMap() {
  const key = partLabelCacheKey();
  if (moduleState.partLabelCache[key]) return moduleState.partLabelCache[key];
  try {
    const data = await moduleState.api(partLabelsUrl());
    const map = {};
    (data.components || []).forEach(item => {
      if (item.stem) map[item.stem] = item.label || item.stem;
    });
    moduleState.partLabelCache[key] = map;
    return map;
  } catch (err) {
    moduleState.partLabelCache[key] = {};
    return {};
  }
}

function renderMeshLegend(items) {
  moduleState.meshLegend = items || [];
  if (!moduleState.meshLegend.length) moduleState.selectedMeshPart = null;
  const host = moduleState.container?.querySelector("[data-mesh-legend]");
  if (!host) return;
  if (!moduleState.meshLegend.length) {
    host.innerHTML = '<div class="mesh-legend-empty">mesh legend</div>';
    return;
  }
  const selected = moduleState.selectedMeshPart;
  const buttons = moduleState.meshLegend
    .map(item => {
      const active = selected === item.index ? " is-active" : "";
      const color = item.legendColor || item.color || (item.body ? "#8f98aa" : "#9aa8ff");
      const meta = item.body ? "body" : item.stem;
      return `
        <button class="mesh-legend-item${active}" data-mesh-part="${esc(item.index)}" title="${esc(item.label)}">
          <span class="mesh-legend-swatch" style="background:${esc(color)}"></span>
          <span class="mesh-legend-text">
            <span class="mesh-legend-name">${esc(item.label)}</span>
            <span class="mesh-legend-meta">${esc(meta)}</span>
          </span>
        </button>
      `;
    })
    .join("");
  const allActive = selected === null ? " is-active" : "";
  host.innerHTML = `
    <button class="mesh-legend-all${allActive}" data-mesh-part="all">全部</button>
    <div class="mesh-legend-items">${buttons}</div>
  `;
  bindMeshLegendEvents();
}

function meshLegendColor(item) {
  if (item.body) return "#8f98aa";
  const palette = [
    "#2f80ed",
    "#eb5757",
    "#27ae60",
    "#f2994a",
    "#9b51e0",
    "#00a6a6",
    "#d94695",
    "#6b8e23",
  ];
  const stemIndex = /^part_(\d+)$/.exec(String(item.stem || ""));
  const index = stemIndex ? Number.parseInt(stemIndex[1], 10) : item.index;
  return palette[Math.max(0, index) % palette.length];
}

function bindMeshLegendEvents() {
  moduleState.container?.querySelectorAll("[data-mesh-part]").forEach(el => {
    el.addEventListener("click", event => {
      event.stopPropagation();
      const value = el.dataset.meshPart;
      selectMeshLegendPart(value === "all" ? null : Number.parseInt(value, 10));
    });
  });
}

function selectMeshLegendPart(index) {
  moduleState.selectedMeshPart = Number.isInteger(index) ? index : null;
  if (moduleState.viewer?.highlightMeshPart) {
    moduleState.viewer.highlightMeshPart(moduleState.selectedMeshPart);
  }
  renderMeshLegend(moduleState.meshLegend);
}

// 确保主可视化区有一个 .viewer3d 容器并复用单个 Viewer3D 实例。
// window.Viewer3D 缺失时返回 null（调用方走文本兜底）。
function ensureViewer(stage) {
  const host = moduleState.container.querySelector(".infer-main-viewer .viewer3d");
  if (!host) return null;
  if (typeof window.Viewer3D !== "function") return null;
  // 复用条件：实例存在且其 canvas 仍挂在当前主 viewer 内（未被 re-render 替换）。
  const stillMounted =
    moduleState.viewer && host.contains(moduleState.viewer.renderer?.domElement);
  if (!stillMounted) {
    // 旧实例已失效（re-render 后 canvas 脱离 DOM）→ 销毁，释放 WebGL 上下文。
    if (moduleState.viewer) {
      try {
        moduleState.viewer.dispose();
      } catch (err) {
        /* dispose 失败不致命 */
      }
    }
    host.innerHTML = "";
    moduleState.viewer = new window.Viewer3D(host);
    moduleState.viewerHostStage = stage;
    moduleState.meshLegend = [];
    moduleState.selectedMeshPart = null;
  }
  return moduleState.viewer;
}

function setViewerHint(_stage, text) {
  const host = moduleState.container.querySelector(".infer-main-viewer .viewer3d");
  if (!host) return;
  let hint = host.querySelector(".viewer3d-hint");
  if (!hint) {
    hint = document.createElement("div");
    hint.className = "viewer3d-hint";
    host.appendChild(hint);
  }
  hint.textContent = text;
}

function setViewerStatus(text) {
  const el = moduleState.container.querySelector("[data-viewer-status]");
  if (el) el.textContent = text || "";
}

async function showArtifact(stage, kind) {
  // part 阶段的「看 voxel」显示逐 part voxel（每 part 一种颜色），而不是 SS 的整体
  // voxel.bin —— part flow 的产物是 parts/part_NN_voxel.npz，合并端点带 part 标签。
  const isPartVoxel = stage === "part" && kind === "voxel";
  const rel = isPartVoxel ? "parts/" : ARTIFACT_RELS[kind];
  if (!rel) return;
  if (!effectiveRoot() || !formState.object_id.trim() || !moduleState.runId) {
    window.alert("缺少 root / object_id / run_id，无法定位产物");
    return;
  }
  const needsStageRender = focusedStage() !== stage;
  moduleState.activeStage = stage;
  moduleState.activeKind = kind;
  moduleState.activeArtifact = { stage, kind };
  if (needsStageRender) {
    render();
    return;
  }
  const viewer = ensureViewer(stage);
  if (!viewer) {
    setViewerHint(stage, "3D 查看器不可用（window.Viewer3D 未加载）。产物路径：" + rel);
    return;
  }
  const viewerKind = isPartVoxel ? "partvoxel" : kind;
  const meshRels = kind === "mesh" ? stageMeshRels(stage) : [];
  const url = isPartVoxel
    ? partVoxelsUrl()
    : (kind === "mesh" && meshRels.length > 1 ? artifactUrls(meshRels) : artifactUrl(meshRels[0] || rel));
  if (viewerKind !== "mesh") {
    moduleState.selectedMeshPart = null;
    renderMeshLegend([]);
  }
  setViewerHint(stage, `加载 ${kind} 中…`);
  setViewerStatus(`${STAGE_LABELS[stage] || stage} · ${kind}`);
  try {
    const result = await viewer.show(viewerKind, url);
    const count = result?.count;
    const detail =
      viewerKind === "partvoxel"
        ? `${kind}：body ${result?.bodyCount ?? 0} / part ${result?.partCount ?? 0}`
        : (viewerKind === "mesh"
          ? meshStatusText(kind, result, count)
          : (count ? `${kind}：${count} 个元素` : ""));
    setViewerStatus(detail);
    // 成功返回即移除提示，露出画布；mesh renderer 原本没有 count，会一直显示“加载中”。
    const host = moduleState.container.querySelector(".infer-main-viewer .viewer3d");
    const hint = host?.querySelector(".viewer3d-hint");
    if (hint) hint.textContent = "";
    if (viewerKind === "mesh") {
      const labels = await loadPartLabelMap();
      const legend = (result?.partStats || []).map(item => ({
        ...item,
        label: labels[item.stem] || (item.body ? "body" : item.stem || `part_${item.index}`),
        legendColor: meshLegendColor(item),
      }));
      moduleState.selectedMeshPart = null;
      renderMeshLegend(legend);
      if (viewer.highlightMeshPart) viewer.highlightMeshPart(null);
    }
  } catch (err) {
    setViewerHint(stage, `加载失败：${err?.message || err}`);
    setViewerStatus(`加载失败：${err?.message || err}`);
  }
}

function meshStatusText(kind, result, count) {
  const stats = Array.isArray(result?.partStats) ? result.partStats : [];
  const bodyCount = stats.filter(item => item.body).length;
  const partCount = stats.length ? stats.filter(item => !item.body).length : (result?.parts ?? 1);
  const prefix = bodyCount ? `body ${bodyCount} + part ${partCount}` : `${partCount} part`;
  return `${kind}：${prefix} · ${count ?? 0} 顶点 / ${result?.triangles ?? 0} 三角面`;
}

async function focusStage(stage) {
  if (!STAGES.includes(stage)) return;
  moduleState.activeStage = stage;
  moduleState.activeKind = defaultArtifactKind(stage);
  if (stageStatus(stage) === "done") {
    moduleState.activeArtifact = { stage, kind: defaultArtifactKind(stage) };
    render();
    return;
  }
  moduleState.activeArtifact = null;
  render();
}

// 拉取某 stage 的运行日志并填入其卡内 <pre class="stage-log">（textContent，可选可复制），
// 然后展开（is-open）。缺 root/object/run 时无声返回（按钮在有 run 时才出现）。
async function fetchStageLog(stage) {
  const root = effectiveRoot();
  const objectId = formState.object_id.trim();
  const runId = moduleState.runId;
  if (!root || !objectId || !runId) return;
  if (moduleState.activeStage !== stage) {
    moduleState.activeStage = stage;
    moduleState.activeArtifact = null;
    render();
  }
  const pre = moduleState.container?.querySelector(`.stage-log[data-log="${stage}"]`);
  if (!pre) return;
  try {
    const data = await moduleState.api(
      "/api/infer/log" +
        `?root=${encodeURIComponent(root)}` +
        `&object_id=${encodeURIComponent(objectId)}` +
        `&run_id=${encodeURIComponent(runId)}` +
        `&angle_idx=${encodeURIComponent(effectiveAngleIdx())}` +
        `&stage=${encodeURIComponent(stage)}`,
    );
    moduleState.stageLogs[stage] = data?.log ?? "";
    pre.textContent = moduleState.stageLogs[stage];
  } catch (err) {
    moduleState.stageLogs[stage] = `拉取日志失败：${err?.message || err}`;
    pre.textContent = moduleState.stageLogs[stage];
  }
  pre.classList.add("is-open");
}

// render() 收尾的副作用：阶段状态已知且对应 DOM 已就位后触发。
//   1) SS done → 自动加载 voxel（模式 A 用户无需点「看 voxel」即可看到 GT voxel）。
//   2) 任一阶段 failed → 自动拉日志并展开（错误立即可见）。
// 用 autoShown / autoLogged 去重，避免每 5s 轮询重绘时反复触发。
function applyStageSideEffects() {
  if (moduleState.subview !== "run") return;
  if (stageStatus("ss") === "done" && !moduleState.autoShown.has("ss")) {
    moduleState.autoShown.add("ss");
    showArtifact("ss", "voxel");
  }
  // part done → 自动显示彩色逐 part voxel（每 part 一色，看出补全的各部件）。
  if (stageStatus("part") === "done" && !moduleState.autoShown.has("part")) {
    moduleState.autoShown.add("part");
    showArtifact("part", "voxel");
  }
  // SLat done → 自动切到真正 decode 出来的 mesh（body + each part），避免停在
  // 上一阶段的彩色 voxel 造成误判。
  if (stageStatus("slat") === "done" && stageHasOutput("slat") && !moduleState.autoShown.has("slat")) {
    moduleState.autoShown.add("slat");
    showArtifact("slat", "mesh");
  }
  STAGES.forEach(stage => {
    if (stageStatus(stage) === "failed" && !moduleState.autoLogged.has(stage)) {
      moduleState.autoLogged.add(stage);
      fetchStageLog(stage);
    }
  });
}

// ---- 渲染 ----

function rootOptions() {
  const opts = moduleState.roots
    .map(entry => {
      const path = rootPathOf(entry);
      const selected = !formState.use_new_root && formState.output_root === path ? " selected" : "";
      const label = entry.name ? `${entry.name} (${path})` : path;
      return `<option value="${esc(path)}"${selected}>${esc(label)}</option>`;
    })
    .join("");
  const newSelected = formState.use_new_root ? " selected" : "";
  return opts + `<option value="__new__"${newSelected}>+ 新建</option>`;
}

// 下拉字段（data_config / ckpt）的 <option> 列表：
// items（[{path,label}]）+ 末尾「+ 自定义路径」逃生口。空 items 时下拉仍有自定义项可用。
function optionSelectOptions(field, items) {
  const useCustom = formState[`use_custom_${field}`];
  const emptyOption = OPTIONAL_OPTION_FIELDS.has(field)
    ? `<option value=""${!useCustom && !formState[field] ? " selected" : ""}>（不使用）</option>`
    : "";
  const opts = items
    .map(item => {
      const selected = !useCustom && formState[field] === item.path ? " selected" : "";
      return `<option value="${esc(item.path)}"${selected}>${esc(item.label)}</option>`;
    })
    .join("");
  const customSelected = useCustom ? " selected" : "";
  return emptyOption + opts + `<option value="__custom__"${customSelected}>+ 自定义路径</option>`;
}

// 渲染一个下拉字段（label + <select> + 选「自定义」时的手填框）。
function renderOptionField(labelText, field, items, customPlaceholder) {
  const useCustom = formState[`use_custom_${field}`];
  const customInput = useCustom
    ? `<label>${esc(labelText)} · 自定义路径
         <input name="${field}_custom" value="${esc(formState[`${field}_custom`])}" placeholder="${esc(customPlaceholder)}">
       </label>`
    : "";
  return `
    <label>${esc(labelText)}
      <select name="${field}">${optionSelectOptions(field, items)}</select>
    </label>
    ${customInput}
  `;
}

function renderInputPreview() {
  if (moduleState.loadingInputs) {
    return '<div class="stage-card-meta">加载输入预览中…</div>';
  }
  if (moduleState.inputsError) {
    return `<div class="stage-card-meta">输入预览失败：${esc(moduleState.inputsError)}</div>`;
  }
  const inputs = moduleState.inputs;
  if (!inputs) {
    return '<div class="stage-card-meta">填写 object_id 后失焦以加载输入预览。</div>';
  }
  const gtBadge = inputs.has_gt_voxel
    ? '<span class="stage-card-status" style="color:var(--green)">有 GT voxel</span>'
    : '<span class="stage-card-status" style="color:var(--muted)">无 GT voxel</span>';
  const parts = (inputs.target_part_names || []).map(esc).join("、") || "（无）";
  const info = selectedObjectInfo();
  const anglesText = info?.angles?.length ? info.angles.join(", ") : String(inputs.angle_idx ?? 0);
  const rgbPaths = inputs.rgb_paths || [];
  // rgb_paths 是服务端文件系统路径，不能直接作为 URL；改用 /api/infer/rgb 端点按
  // object_id + angle_idx + view 取真实图片（server 内部解析回 FS 路径）。
  const obj = encodeURIComponent(inputs.object_id);
  const angle = inputs.angle_idx ?? 0;
  let previewBody;
  if (rgbPaths.length) {
    previewBody =
      '<div class="input-preview">' +
      rgbPaths
        .map((p, i) => {
          const view = (inputs.view_indices || [])[i];
          // route()：经 /proxy/7861/ 反代时给绝对路径补上代理前缀，否则 <img> 打到 80 → 404。
          const src = route(
            `/api/infer/rgb?object_id=${obj}` +
              `&angle_idx=${encodeURIComponent(angle)}` +
              `&view=${encodeURIComponent(view)}`,
          );
          return (
            `<figure><img src="${esc(src)}" loading="lazy"` +
            ` onerror="this.closest('figure').classList.add('img-failed')">` +
            `<figcaption>v${esc(view)} · view_${esc(view)}.png</figcaption></figure>`
          );
        })
        .join("") +
      "</div>";
  } else {
    previewBody = '<div class="stage-card-meta">该物体没有可预览的 rgb 视图。</div>';
  }
  return `
    <div class="stage-card-head">
      <span class="stage-card-title">输入预览 · ${esc(inputs.object_id)}</span>
      ${gtBadge}
    </div>
    <div class="stage-card-meta">angle：${esc(inputs.angle_idx ?? 0)} · 可用 angles：${esc(anglesText)}</div>
    <div class="stage-card-meta">目标 part：${parts}</div>
    ${previewBody}
  `;
}

function renderStageButton(stage) {
  const status = stageStatus(stage);
  const hasOutput = stageHasOutput(stage);
  const runnable = stageRunnable(stage);
  const runDisabled = !runnable || status === "running" ? " disabled" : "";
  const hasJob = Boolean(moduleState.stageJobs[stage]);
  const stopDisabled = hasJob && status === "running" ? "" : " disabled";
  // 各阶段产物不同，按钮分开避免混淆：
  //  ss   → 整体稀疏结构 voxel（灰、半透明）。无 mesh/GS。
  //  part → 逐 part voxel（每 part 一种颜色）。mesh/GS 要等 slat 解码出来。
  //  slat/assemble → 逐 part / 组装后的 mesh、GS。
  let showButtons = "";
  if (status === "done" || hasOutput) {
    if (stage === "ss") {
      showButtons =
        `<button class="ghost" data-infer-show="ss" data-kind="voxel">看 voxel（整体）</button>`;
    } else if (stage === "part") {
      showButtons =
        `<button class="ghost" data-infer-show="part" data-kind="voxel">看 part voxel</button>`;
    } else {
      showButtons =
        `<button class="ghost" data-infer-show="${stage}" data-kind="voxel">看 voxel</button>` +
        `<button class="ghost" data-infer-show="${stage}" data-kind="mesh">看 mesh</button>` +
        `<button class="ghost" data-infer-show="${stage}" data-kind="gaussian">看 GS</button>`;
    }
  }
  // 门控未通过 → 追加 is-locked，CSS 据此变暗，配合 meta 的「等待上一阶段完成」。
  const lockedClass = runnable ? "" : " is-locked";
  // 运行中 → 不定式进度条（CSS 动画驱动），仅运行时渲染。
  const progressBar =
    status === "running"
      ? '<div class="stage-progress"><div class="stage-progress-bar"></div></div>'
      : "";
  // 阶段已有日志（运行中 / 已完成 / 失败）→ 给个 [日志] 按钮按需拉取查看。
  const hasLog = status === "running" || status === "done" || status === "failed";
  const logButton = hasLog
    ? `<button class="ghost" data-infer-log="${esc(stage)}">日志</button>`
    : "";
  // 完成后把醒目的 primary「运行」降级为 ghost「重跑」，让卡片视觉重心落到下方产物
  // viewer（用户：运行完应跳到可视化，不应仍停在「运行」按钮）。其余状态保持「运行」。
  const runButton =
    status === "done" || hasOutput
      ? `<button class="ghost" data-infer-run="${esc(stage)}"${runDisabled}>重跑</button>`
      : `<button class="primary" data-infer-run="${esc(stage)}"${runDisabled}>运行</button>`;
  const activeClass = focusedStage() === stage ? " is-active" : "";
  return `
    <div class="stage-card infer-stage-button is-${esc(status)}${lockedClass}${activeClass}" data-stage="${esc(stage)}">
      <div class="stage-card-head">
        <span class="stage-card-title">${esc(STAGE_LABELS[stage] || stage)}</span>
        <span class="stage-card-status">${esc(STATUS_LABELS[status] || status)}</span>
      </div>
      <div class="stage-card-meta">阶段 <b>${esc(stage)}</b>${runnable ? "" : " · 等待上一阶段完成"}${hasOutput && status !== "done" ? " · 已有产物" : ""}</div>
      ${progressBar}
      <div class="stage-card-actions">
        ${runButton}
        <button class="ghost" data-infer-stop="${esc(stage)}"${stopDisabled}>停止</button>
        ${showButtons}
        ${logButton}
      </div>
    </div>
  `;
}

function renderMainViewer() {
  const stage = focusedStage();
  const logText = moduleState.stageLogs[stage] || "";
  const logOpen = logText ? " is-open" : "";
  const title = STAGE_LABELS[stage] || stage;
  return `
    <section class="infer-main-viewer" data-active-stage="${esc(stage)}">
      <div class="infer-viewer-head">
        <div>
          <div class="infer-viewer-title">${esc(title)}</div>
          <div class="infer-viewer-status" data-viewer-status>
            ${moduleState.activeArtifact ? esc(stage === moduleState.activeArtifact.stage ? moduleState.activeKind : "") : ""}
          </div>
        </div>
      </div>
      <div class="infer-viewer-body">
        <aside class="mesh-legend-panel" data-mesh-legend>
          <div class="mesh-legend-empty">mesh legend</div>
        </aside>
        <div class="viewer3d"><div class="viewer3d-hint">运行或选择一个已完成阶段查看产物</div></div>
      </div>
      <pre class="stage-log${logOpen}" data-log="${esc(stage)}">${esc(logText)}</pre>
    </section>
  `;
}

function renderForm() {
  const f = formState;
  const radio = (name, value, label, current) =>
    `<label class="infer-radio"><input type="radio" name="${name}" value="${value}"${
      current === value ? " checked" : ""
    }> ${esc(label)}</label>`;
  const pipelineText =
    f.mode === "B"
      ? effectiveOption("ss_flow_ckpt")
        ? `B：4-view 输入 · TRELLIS SS Flow 出 z_global + voxel · ${f.part_backend === "promptable_seg" ? "Part Promptable Seg" : "0526 Part Flow"} · SAM3D SLat`
        : `B：4-view 输入 · SAM3D 默认 SS 出 surface voxel · TRELLIS SS encoder 生成 z_global · ${f.part_backend === "promptable_seg" ? "Part Promptable Seg" : "0526 Part Flow"} · SAM3D SLat`
      : `A：使用数据集中已有 GT/TRELLIS SS latent 与 voxel · ${f.part_backend === "promptable_seg" ? "Part Promptable Seg" : "0526 Part Flow"} · SAM3D SLat`;
  return `
    <form class="infer-form" id="inferForm">
      <h1>全流程推理</h1>
      <div class="stage-card-meta">run_id：${esc(moduleState.runId || "（运行时生成）")}</div>

      <fieldset>
        <legend>模式</legend>
        ${radio("mode", "A", "A（GT voxel / SS 已知）", f.mode)}
        ${radio("mode", "B", "B（4-view SS；可选 TRELLIS SS Flow）", f.mode)}
      </fieldset>

      <fieldset>
        <legend>视角</legend>
        ${radio("view", "single", "single（1 视角）", f.view)}
        ${radio("view", "four", "four（4 视角）", f.view)}
      </fieldset>

      <div class="pipeline-note">${esc(pipelineText)}</div>

      <div class="infer-object-picker">
        <label>首位
          <select name="object_prefix1">${objectPrefixOptions(1, f.object_prefix1, "", "全部")}</select>
        </label>
        <label>前两位
          <select name="object_prefix2">${objectPrefixOptions(2, f.object_prefix2, f.object_prefix1, "全部")}</select>
        </label>
        <label>前三位
          <select name="object_prefix3">${objectPrefixOptions(3, f.object_prefix3, f.object_prefix2 || f.object_prefix1, "全部")}</select>
        </label>
        <label>前四位
          <select name="object_prefix4">${objectPrefixOptions(4, f.object_prefix4, f.object_prefix3 || f.object_prefix2 || f.object_prefix1, "全部")}</select>
        </label>
        <label>object_id
          <input name="object_id" list="inferObjectOptions" value="${esc(f.object_id)}" placeholder="100013" autocomplete="off">
          <datalist id="inferObjectOptions">${objectOptions()}</datalist>
        </label>
      </div>
      ${
        selectedObjectVisible()
          ? ""
          : `<div class="stage-card-meta">当前 object_id 不在所选前缀下；切到“全部”或对应前缀可在下拉中看到。</div>`
      }

      <label>angle_idx
        <input name="angle_idx" type="number" min="0" step="1" value="${esc(f.angle_idx)}" placeholder="0">
      </label>

      <label>输出 root
        <select name="output_root">${rootOptions()}</select>
      </label>
      ${
        f.use_new_root
          ? `<label>新建 root 路径
               <input name="new_root" value="${esc(f.new_root)}" placeholder="/robot/data-lab/jzh/art-gen-output/full-stage">
             </label>`
          : ""
      }

      ${renderOptionField("data_config", "data_config", moduleState.configs, "/path/to/data_config.yaml")}
      <fieldset>
        <legend>Stage2</legend>
        ${radio("part_backend", "promptable_seg", "part-promptable-seg（推荐）", f.part_backend)}
        ${radio("part_backend", "part_flow", "part flow（旧后端）", f.part_backend)}
      </fieldset>
      ${
        f.part_backend === "promptable_seg"
          ? renderOptionField("part promptable seg ckpt", "part_seg_ckpt", checkpointOptionsFor("part_seg_ckpt"), "/path/to/part_promptable_seg/latest.pt")
          : renderOptionField("part flow ckpt", "part_flow_ckpt", checkpointOptionsFor("part_flow_ckpt"), "/path/to/part_flow.pt")
      }
      ${renderOptionField("TRELLIS ss flow ckpt", "ss_flow_ckpt", checkpointOptionsFor("ss_flow_ckpt"), "/path/to/denoiser_ema0.9999_step0020000.pt")}
      <label>GPU
        <input name="gpu_ids" value="${esc(f.gpu_ids)}" placeholder="0">
      </label>
    </form>
  `;
}

// 配置子视图：居中单列 = 表单 + 输入预览 + 底部大号主按钮 [进入推理]。
function renderConfig() {
  return `
    <div class="infer-config">
      ${renderForm()}
      <div class="infer-input-preview">${renderInputPreview()}</div>
      <button class="enter-infer-btn" data-action="enter-infer">进入推理</button>
    </div>
  `;
}

// 运行子视图：居中单列 = [← 返回配置] 头部 + 4 阶段卡 + manifest 错误行。
function renderRun() {
  const rootPath = runRootPath();
  return `
    <div class="infer-run">
      <div class="infer-run-header">
        <button class="infer-back-btn" data-action="back-config">← 返回配置</button>
        <div class="infer-run-context">
          <span class="stage-card-meta">run_id：${esc(moduleState.runId || "（运行时生成）")} · mode=${esc(formState.mode)} · view=${esc(formState.view)} · angle=${esc(effectiveAngleIdx())} · ${esc(moduleState.runReuseInfo)}</span>
          ${rootPath ? `<span class="stage-card-meta">root：${esc(rootPath)}</span>` : ""}
        </div>
      </div>
      <div class="stage-list">
        ${STAGES.map(renderStageButton).join("")}
      </div>
      ${renderMainViewer()}
      ${moduleState.manifestError ? `<div class="stage-card-meta">manifest 错误：${esc(moduleState.manifestError)}</div>` : ""}
    </div>
  `;
}

function render() {
  const c = moduleState.container;
  if (!c || !document.body.contains(c)) return;
  const artifactToRestore = moduleState.activeArtifact;
  c.innerHTML = moduleState.subview === "run" ? renderRun() : renderConfig();
  bindEvents();
  if (moduleState.subview === "run") renderMeshLegend(moduleState.meshLegend);
  // 阶段卡及其 viewer/log 容器此刻已在 DOM 中：触发 SS done 自动加载 voxel、
  // failed 自动拉日志等副作用（autoShown / autoLogged 去重，不会反复触发）。
  if (moduleState.subview === "run") applyStageSideEffects();
  if (
    moduleState.subview === "run" &&
    artifactToRestore &&
    moduleState.activeArtifact === artifactToRestore
  ) {
    const { stage, kind } = artifactToRestore;
    if (stageStatus(stage) === "done") showArtifact(stage, kind);
  }
}

// 把表单输入同步进 formState（受控）。注意 select / radio 的特殊处理。
function syncFormFromDom() {
  const form = moduleState.container.querySelector("#inferForm");
  if (!form) return;
  const get = name => form.querySelector(`[name="${name}"]`);
  const checkedRadio = name => {
    const el = form.querySelector(`[name="${name}"]:checked`);
    return el ? el.value : undefined;
  };
  const mode = checkedRadio("mode");
  if (mode) formState.mode = mode;
  const view = checkedRadio("view");
  if (view) formState.view = view;
  const partBackend = checkedRadio("part_backend");
  if (partBackend) formState.part_backend = partBackend;
  // 自由文本字段：含每个下拉字段的 _custom 手填框（仅在切到自定义时存在于 DOM）。
  const textFields = [
    "object_id",
    "object_prefix1",
    "object_prefix2",
    "object_prefix3",
    "object_prefix4",
    "angle_idx",
    "gpu_ids",
    "new_root",
  ].concat(
    OPTION_FIELDS.map(field => `${field}_custom`),
  );
  textFields.forEach(name => {
    const el = get(name);
    if (el) formState[name] = el.value;
  });
  const rootSel = get("output_root");
    if (rootSel) {
    if (rootSel.value === "__new__") {
      formState.use_new_root = true;
    } else {
      formState.use_new_root = false;
      formState.output_root = rootSel.value;
    }
  }
  const prefix1Sel = get("object_prefix1");
  if (prefix1Sel) formState.object_prefix1 = prefix1Sel.value;
  const prefix2Sel = get("object_prefix2");
  if (prefix2Sel) formState.object_prefix2 = prefix2Sel.value;
  const prefix3Sel = get("object_prefix3");
  if (prefix3Sel) formState.object_prefix3 = prefix3Sel.value;
  const prefix4Sel = get("object_prefix4");
  if (prefix4Sel) formState.object_prefix4 = prefix4Sel.value;
  // 下拉字段：__custom__ 切到自定义（露出手填框），否则下拉值即生效路径。
  OPTION_FIELDS.forEach(field => {
    const sel = get(field);
    if (!sel) return;
    if (sel.value === "__custom__") {
      formState[`use_custom_${field}`] = true;
    } else {
      formState[`use_custom_${field}`] = false;
      formState[field] = sel.value;
    }
  });
}

function bindEvents() {
  const form = moduleState.container.querySelector("#inferForm");
  if (form) {
    // 所有输入即时同步进 formState（不 re-render，避免打断输入）。
    form.addEventListener("input", syncFormFromDom);
    // root 下拉 / 模式 / 视角 / ckpt 下拉变化需 re-render（新建框、自定义框显隐）。
    form.addEventListener("change", event => {
      syncFormFromDom();
      const name = event.target?.name;
      // 下拉/单选切换需 re-render：新建框、自定义框显隐、pipeline note。
      if (
        name === "output_root" ||
        name === "mode" ||
        name === "view" ||
        name === "part_backend" ||
        name === "object_prefix1" ||
        name === "object_prefix2" ||
        name === "object_prefix3" ||
        name === "object_prefix4" ||
        name === "angle_idx" ||
        OPTION_FIELDS.includes(name)
      ) {
        if (
          name === "object_prefix1" &&
          formState.object_prefix2 &&
          (!formState.object_prefix1 || !formState.object_prefix2.startsWith(formState.object_prefix1))
        ) {
          formState.object_prefix2 = "";
          formState.object_prefix3 = "";
          formState.object_prefix4 = "";
        }
        if (
          name === "object_prefix2" &&
          formState.object_prefix3 &&
          (!formState.object_prefix2 || !formState.object_prefix3.startsWith(formState.object_prefix2))
        ) {
          formState.object_prefix3 = "";
          formState.object_prefix4 = "";
        }
        if (
          name === "object_prefix3" &&
          formState.object_prefix4 &&
          (!formState.object_prefix3 || !formState.object_prefix4.startsWith(formState.object_prefix3))
        ) {
          formState.object_prefix4 = "";
        }
        render();
        if (name === "angle_idx" || name === "view") loadInputs();
      }
    });
    form.addEventListener("submit", e => e.preventDefault());
  const objInput = form.querySelector('[name="object_id"]');
  if (objInput) {
    objInput.addEventListener("blur", () => {
      syncFormFromDom();
      const changed = syncAngleForSelectedObject();
      if (changed) render();
      loadInputs();
    });
    objInput.addEventListener("change", () => {
      syncFormFromDom();
      const changed = syncAngleForSelectedObject();
      if (changed) render();
      loadInputs();
    });
  }
    const angleInput = form.querySelector('[name="angle_idx"]');
    if (angleInput) {
      angleInput.addEventListener("blur", () => {
        syncFormFromDom();
        loadInputs();
      });
    }
  }
  // 子视图切换按钮（[进入推理] 仅在 config 子视图、[← 返回配置] 仅在 run 子视图）。
  const enterBtn = moduleState.container.querySelector('[data-action="enter-infer"]');
  if (enterBtn) {
    enterBtn.addEventListener("click", () => {
      syncFormFromDom();
      enterInfer();
    });
  }
  const backBtn = moduleState.container.querySelector('[data-action="back-config"]');
  if (backBtn) {
    backBtn.addEventListener("click", () => backToConfig());
  }
  moduleState.container.querySelectorAll("[data-stage]").forEach(el => {
    el.addEventListener("click", event => {
      if (event.target instanceof Element && event.target.closest("button")) return;
      focusStage(el.dataset.stage);
    });
  });
  moduleState.container.querySelectorAll("[data-infer-run]").forEach(el => {
    el.addEventListener("click", event => {
      event.stopPropagation();
      syncFormFromDom();
      runStage(el.dataset.inferRun);
    });
  });
  moduleState.container.querySelectorAll("[data-infer-stop]").forEach(el => {
    el.addEventListener("click", event => {
      event.stopPropagation();
      stopStage(el.dataset.inferStop);
    });
  });
  moduleState.container.querySelectorAll("[data-infer-show]").forEach(el => {
    el.addEventListener("click", event => {
      event.stopPropagation();
      showArtifact(el.dataset.inferShow, el.dataset.kind);
    });
  });
  moduleState.container.querySelectorAll("[data-infer-log]").forEach(el => {
    el.addEventListener("click", event => {
      event.stopPropagation();
      fetchStageLog(el.dataset.inferLog);
    });
  });
}

// ---- 入口 ----

export function mountInfer(container, api) {
  moduleState.container = container;
  moduleState.api = api;
  // 每次挂载从配置子视图开始：用户先填表单 → [进入推理] 才复用/生成 run_id 并进运行视图。
  moduleState.subview = "config";
  render();
  loadRoots();
  loadOptions();
  loadObjects();
  // 已填过 object_id（re-mount）时自动刷新输入预览（仍停在配置子视图，不预取 manifest）。
  if (formState.object_id.trim()) {
    loadInputs();
  }
  startPolling();
}

window.mountInfer = mountInfer;
