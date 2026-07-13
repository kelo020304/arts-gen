/** Attach a single status line to a container.
 * @param {HTMLElement} container
 * @returns {{ set(level, msg): void, reset(): void }}
 *   level in 'info' | 'ok' | 'warn' | 'error' | 'running'
 */
export function setupStatus(container) {
  const LEVELS = new Set(["info", "ok", "warn", "error", "running"]);
  const line = document.createElement("div");
  line.className = "status-line";
  line.textContent = "";
  container.appendChild(line);

  function set(level, msg) {
    if (!LEVELS.has(level)) level = "info";
    line.className = `status-line status-${level}`;
    line.textContent = msg;
  }

  function reset() {
    line.className = "status-line";
    line.textContent = "";
  }

  return { set, reset };
}
