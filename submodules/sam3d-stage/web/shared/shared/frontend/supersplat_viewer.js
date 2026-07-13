/**
 * Mount the bundled SuperSplat (PlayCanvas) viewer into a DOM container,
 * loading a Gaussian-splat PLY from a URL.
 *
 * Each call to loadPly() tears down and rebuilds the viewer because SuperSplat's
 * main() takes contentUrl only at init time and exposes no swap-content API.
 *
 * @param {HTMLElement} container - the parent element to mount into
 * @returns {{ loadPly(url): Promise<void>, clear(): void }}
 */
export function createSuperSplatViewer(container) {
  let bundlePromise = null;
  let settingsPromise = null;
  let mountedCamEntity = null;   // PlayCanvas Entity holding the camera, set after loadPly

  function loadBundle() {
    if (!bundlePromise) {
      bundlePromise = import("/static/supersplat/index.js");
    }
    return bundlePromise;
  }
  function loadSettings() {
    if (!settingsPromise) {
      settingsPromise = fetch("/static/supersplat/settings.json", { cache: "no-store" })
        .then(r => r.ok ? r.json() : Promise.reject(new Error("settings.json " + r.status)));
    }
    return settingsPromise;
  }

  function clear() {
    container.innerHTML = "";
    mountedCamEntity = null;
  }

  function getCameraQuat() {
    if (!mountedCamEntity || typeof mountedCamEntity.getRotation !== "function") return null;
    return mountedCamEntity.getRotation();   // PlayCanvas Quat has .x .y .z .w
  }

  async function loadPly(plyUrl) {
    clear();
    console.log("[supersplat_viewer] loadPly", plyUrl);

    container.style.position = container.style.position || "relative";
    const appEl = document.createElement("pc-app");
    appEl.style.cssText = "display:block; width:100%; height:100%;";
    const camEl = document.createElement("pc-entity");
    camEl.setAttribute("name", "viewer-camera");
    appEl.appendChild(camEl);
    container.appendChild(appEl);

    let bundle, settings;
    try {
      [bundle, settings] = await Promise.all([loadBundle(), loadSettings()]);
      console.log("[supersplat_viewer] bundle + settings loaded");
    } catch (e) {
      console.error("[supersplat_viewer] failed to fetch bundle/settings", e);
      container.innerHTML = `<div style="color:#f87171; padding:20px; font-family:monospace;">
        SuperSplat assets failed to load: ${e.message}<br>
        Check Network tab for /static/supersplat/*</div>`;
      throw e;
    }

    if (typeof appEl.ready !== "function" || typeof camEl.ready !== "function") {
      const msg = "pc-app / pc-entity custom elements did not register — SuperSplat bundle didn't run its top-level code.";
      console.error("[supersplat_viewer]", msg);
      container.innerHTML = `<div style="color:#f87171; padding:20px; font-family:monospace;">${msg}</div>`;
      throw new Error(msg);
    }
    await Promise.all([appEl.ready(), camEl.ready()]);
    console.log("[supersplat_viewer] pc-app ready, starting main()");

    bundle.main(appEl.app, camEl.entity, settings, {
      contentUrl: plyUrl,
      aa: true,
      noui: true,
      noanim: true,
      noXr: true,
      noAudio: true,
    });

    mountedCamEntity = camEl.entity;

    // SuperSplat loads a default HDR skybox + env atlas for IBL. For a clean
    // product-style "object on white" view, kill the skybox/env and set the
    // camera to clear with solid white. main() does its setup async, so we
    // also patch on every frame for the first ~2s in case it gets re-assigned
    // by the bundle's deferred init.
    //
    // We also align the splat from sam3d's Z-up convention to Y-up — sam3d
    // rotates the mesh GLB at export but NOT the gsplat PLY, so the splat
    // arrives 90° off. Apply -90° around X to every non-camera entity in
    // app.root until the splat loads in (gsplat entity appears async).
    const fixScene = () => {
      const app = appEl.app;
      const cam = camEl.entity && camEl.entity.camera;
      if (app && app.scene) {
        try { app.scene.skybox = null; } catch (_) {}
        try { app.scene.envAtlas = null; } catch (_) {}
        try { app.scene.skyboxIntensity = 0; } catch (_) {}
        try { app.scene.exposure = 1; } catch (_) {}
      }
      if (cam) {
        try { cam.clearColor = { r: 1, g: 1, b: 1, a: 1 }; } catch (_) {}
        try { cam.clearColorBuffer = true; } catch (_) {}
      }
      // Z-up → Y-up: rotate every non-camera entity. setLocalEulerAngles is
      // absolute, so reapplying every tick is idempotent.
      if (app && app.root && app.root.children) {
        for (const ent of app.root.children) {
          if (!ent || ent === camEl.entity) continue;
          if (typeof ent.setLocalEulerAngles !== "function") continue;
          try { ent.setLocalEulerAngles(-90, 0, 0); } catch (_) {}
        }
      }
    };
    fixScene();
    let n = 0;
    const id = setInterval(() => { fixScene(); if (++n > 60) clearInterval(id); }, 33);
  }

  return { loadPly, clear, getCameraQuat };
}
