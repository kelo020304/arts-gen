/**
 * Wire up a drag-drop + click uploader to a container.
 *
 * @param {HTMLElement} container - element to attach UI inside
 * @param {Array<{name: string, accept: string, label: string}>} slots
 *        e.g. [{name:'image', accept:'image/*', label:'RGB image'},
 *              {name:'mask',  accept:'image/*', label:'Mask PNG'}]
 * @returns {{getFormData: () => FormData,
 *            isComplete: () => boolean,
 *            onChange: (cb) => void,
 *            reset: () => void}}
 */
export function setupUpload(container, slots) {
  container.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "upload-slots";
  container.appendChild(wrap);

  const state = new Map();
  const listeners = [];

  for (const slot of slots) {
    state.set(slot.name, { file: null, slot });
    wrap.appendChild(buildSlot(slot, state, listeners));
  }

  function emitChange() {
    for (const cb of listeners) cb();
  }

  function buildSlot(slot, state, listeners) {
    const root = document.createElement("div");
    root.className = "upload-slot";
    root.dataset.slot = slot.name;

    const input = document.createElement("input");
    input.type = "file";
    input.accept = slot.accept || "*/*";
    root.appendChild(input);

    renderEmpty(root, slot, input);

    input.addEventListener("change", () => {
      const file = input.files && input.files[0];
      if (!file) return;
      setFile(slot.name, file, root, input);
    });

    root.addEventListener("dragover", (e) => {
      e.preventDefault();
      root.classList.add("dragover");
    });
    root.addEventListener("dragleave", () => root.classList.remove("dragover"));
    root.addEventListener("drop", (e) => {
      e.preventDefault();
      root.classList.remove("dragover");
      const file = e.dataTransfer.files && e.dataTransfer.files[0];
      if (!file) return;
      setFile(slot.name, file, root, input);
    });

    return root;
  }

  function setFile(name, file, root, input) {
    state.get(name).file = file;
    renderFilled(root, state.get(name).slot, file, () => clearFile(name, root, input));
    emitChange();
  }

  function clearFile(name, root, input) {
    state.get(name).file = null;
    input.value = "";
    renderEmpty(root, state.get(name).slot, input);
    emitChange();
  }

  function renderEmpty(root, slot, input) {
    root.classList.remove("populated");
    root.innerHTML = "";
    root.appendChild(input);
    const empty = document.createElement("div");
    empty.className = "upload-slot-empty";
    empty.innerHTML = `<div class="label">${slot.label}</div>
      <div class="hint">click or drop a file</div>`;
    empty.addEventListener("click", () => input.click());
    root.appendChild(empty);
  }

  function renderFilled(root, slot, file, onClear) {
    root.classList.add("populated");
    root.innerHTML = "";
    const row = document.createElement("div");
    row.className = "upload-slot-filled";

    const thumb = document.createElement("div");
    thumb.className = "upload-slot-thumb";
    if (file.type && file.type.startsWith("image/")) {
      const url = URL.createObjectURL(file);
      thumb.style.backgroundImage = `url(${url})`;
    } else {
      thumb.textContent = ext(file.name).toUpperCase();
      thumb.style.display = "flex";
      thumb.style.alignItems = "center";
      thumb.style.justifyContent = "center";
      thumb.style.fontSize = "10px";
      thumb.style.color = "#666";
    }
    row.appendChild(thumb);

    const meta = document.createElement("div");
    meta.className = "upload-slot-meta";
    meta.innerHTML = `<div class="name">${slot.label}</div>
      <div class="sub">${file.name} (${humanSize(file.size)})</div>`;
    row.appendChild(meta);

    const clear = document.createElement("button");
    clear.className = "upload-slot-clear";
    clear.type = "button";
    clear.textContent = "clear";
    clear.addEventListener("click", onClear);
    row.appendChild(clear);

    root.appendChild(row);
  }

  function ext(name) {
    const i = name.lastIndexOf(".");
    return i >= 0 ? name.slice(i + 1) : "?";
  }

  function humanSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  return {
    getFormData() {
      const fd = new FormData();
      for (const [name, { file }] of state.entries()) {
        if (file) fd.append(name, file, file.name);
      }
      return fd;
    },
    isComplete() {
      for (const { file } of state.values()) {
        if (!file) return false;
      }
      return true;
    },
    onChange(cb) {
      listeners.push(cb);
    },
    reset() {
      for (const name of state.keys()) {
        const root = wrap.querySelector(`[data-slot="${name}"]`);
        const input = root.querySelector("input[type=file]");
        state.get(name).file = null;
        input.value = "";
        renderEmpty(root, state.get(name).slot, input);
      }
      emitChange();
    },
  };
}
