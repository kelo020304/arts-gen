import { main } from "../vendor/supersplat/index.js";

const statusEl = document.getElementById("viewerStatus");
const appEl = document.getElementById("pcapp");
const cameraEl = document.getElementById("viewerCamera");
const APP_ROOT = new URL("../../", import.meta.url);

function appUrl(path) {
  return new URL(String(path).replace(/^\/+/, ""), APP_ROOT).toString();
}

let viewer = null;
let firstFrameReady = false;
let pendingOrientation = { x: 0, y: 225, z: 180 };
let orientationAnchor = null;
let orientationCenter = null;
let embedded = false;
let applyingRemoteCamera = false;
let meshEntity = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function vecToArray(vec) {
  if (!vec) return null;
  return [Number(vec.x || 0), Number(vec.y || 0), Number(vec.z || 0)];
}

function cameraPayload() {
  const camera = cameraEl.entity;
  const managerCamera = viewer && viewer.cameraManager ? viewer.cameraManager.camera : null;
  let euler = null;
  try {
    euler = vecToArray(camera.getEulerAngles());
  } catch {
    euler = null;
  }
  return {
    position: vecToArray(camera.getPosition()),
    euler,
    fov: camera.camera ? camera.camera.fov : null,
    manager: managerCamera
      ? {
          position: vecToArray(managerCamera.position),
          angles: vecToArray(managerCamera.angles),
          distance: Number(managerCamera.distance || 0),
          fov: Number(managerCamera.fov || 0),
        }
      : null,
    first_frame_ready: firstFrameReady,
    object_orientation: pendingOrientation,
    captured_unix: Date.now() / 1000,
  };
}

function applyCameraPayload(payload) {
  if (!payload || !viewer || !cameraEl.entity) return;
  applyingRemoteCamera = true;
  const managerCamera = viewer.cameraManager && viewer.cameraManager.camera;
  if (managerCamera && payload.manager) {
    const source = payload.manager;
    if (Array.isArray(source.position)) managerCamera.position.set(...source.position);
    if (Array.isArray(source.angles)) managerCamera.angles.set(...source.angles);
    if (Number.isFinite(source.distance)) managerCamera.distance = source.distance;
    if (Number.isFinite(source.fov)) managerCamera.fov = source.fov;
  } else {
    if (Array.isArray(payload.position)) cameraEl.entity.setPosition(...payload.position);
    if (Array.isArray(payload.euler)) cameraEl.entity.setEulerAngles(...payload.euler);
    if (Number.isFinite(payload.fov) && cameraEl.entity.camera) cameraEl.entity.camera.fov = payload.fov;
  }
  appEl.app.renderNextFrame = true;
  window.setTimeout(() => { applyingRemoteCamera = false; }, 0);
}

function makeEmbeddedCanvasTransparent() {
  if (!embedded || !cameraEl.entity.camera || !appEl.app) return;
  const clear = cameraEl.entity.camera.clearColor;
  if (clear) clear.a = 0;
  cameraEl.entity.camera.clearColorBuffer = true;
  appEl.app.scene.skybox = null;
}

function hideGsplat() {
  const entity = findGsplatEntity();
  if (entity) entity.enabled = false;
}

async function loadMesh(meshUrl) {
  const asset = await new Promise((resolve, reject) => {
    appEl.app.assets.loadFromUrl(meshUrl, "container", (error, loaded) => {
      if (error || !loaded) reject(new Error(error || `Failed to load ${meshUrl}`));
      else resolve(loaded);
    });
  });
  meshEntity = asset.resource.instantiateRenderEntity?.() || asset.resource.instantiateModelEntity?.();
  if (!meshEntity) throw new Error("GLB container cannot be instantiated");
  meshEntity.name = "component-mesh";
  meshEntity.setLocalEulerAngles(pendingOrientation.x, pendingOrientation.y, pendingOrientation.z);
  appEl.app.root.addChild(meshEntity);
  hideGsplat();
  appEl.app.renderNextFrame = true;
  if (viewer && viewer.global && viewer.global.events) viewer.global.events.fire("inputEvent", "frame");
}

function findGsplatEntity() {
  if (!appEl.app) return null;
  return appEl.app.root.findByName("gsplat");
}

function applyObjectOrientation(orientation = pendingOrientation) {
  pendingOrientation = {
    x: Number(orientation.x || 0),
    y: Number(orientation.y || 0),
    z: Number(orientation.z || 0),
  };
  const entity = findGsplatEntity();
  if (!entity) return false;
  if (!orientationAnchor && entity.gsplat && entity.gsplat.customAabb && entity.gsplat.customAabb.center) {
    orientationCenter = entity.gsplat.customAabb.center.clone();
    orientationAnchor = entity.getWorldTransform().transformPoint(orientationCenter.clone());
  }
  entity.setLocalPosition(0, 0, 0);
  entity.setLocalEulerAngles(pendingOrientation.x, pendingOrientation.y, pendingOrientation.z);
  if (orientationAnchor && orientationCenter) {
    const moved = entity.getWorldTransform().transformPoint(orientationCenter.clone());
    entity.setLocalPosition(
      orientationAnchor.x - moved.x,
      orientationAnchor.y - moved.y,
      orientationAnchor.z - moved.z,
    );
  }
  appEl.app.renderNextFrame = true;
  if (viewer && viewer.global && viewer.global.events) {
    viewer.global.events.fire("inputEvent", "frame");
  }
  setStatus(`Rotation X${pendingOrientation.x} Y${pendingOrientation.y} Z${pendingOrientation.z}`);
  return true;
}

function applyObjectOrientationWhenReady(retries = 80) {
  if (applyObjectOrientation(pendingOrientation)) return;
  if (retries > 0) {
    window.setTimeout(() => applyObjectOrientationWhenReady(retries - 1), 100);
  }
}

function zoomCamera(delta) {
  const amount = Number(delta || 0);
  if (!viewer || !viewer.inputController || !viewer.inputController.frame || !appEl.app) return false;
  viewer.inputController.frame.deltas.move.append([0, 0, amount]);
  appEl.app.renderNextFrame = true;
  setStatus(amount < 0 ? "Camera nearer" : "Camera farther");
  return true;
}

async function nextFrame() {
  const app = appEl.app;
  app.renderNextFrame = true;
  await new Promise((resolve) => app.once("frameend", resolve));
}

async function capture() {
  if (!viewer || !appEl.app) {
    throw new Error("viewer is not ready");
  }
  await nextFrame();
  const canvas = appEl.querySelector("canvas");
  if (!canvas) {
    throw new Error("viewer canvas not found");
  }
  return {
    image_data_url: canvas.toDataURL("image/png"),
    camera: cameraPayload(),
    width: canvas.width,
    height: canvas.height,
  };
}

async function init() {
  try {
    const params = new URLSearchParams(window.location.search);
    const requestedContent = params.get("content");
    const requestedMesh = params.get("mesh");
    embedded = params.get("embedded") === "1";
    if (embedded) pendingOrientation = { x: 0, y: 0, z: 0 };
    const contentUrl = requestedContent
      ? new URL(requestedContent, window.location.href).toString()
      : appUrl("assets/point_cloud.ply");
    window.firstFrame = () => {
      firstFrameReady = true;
      applyObjectOrientation(pendingOrientation);
      if (requestedMesh && meshEntity) hideGsplat();
      makeEmbeddedCanvasTransparent();
      if (!requestedMesh) setStatus("Ready");
      window.parent.postMessage({ type: "fridge3dgs.viewerReady" }, "*");
    };
    await customElements.whenDefined("pc-app");
    await appEl.ready();
    const settings = await fetch(new URL("../vendor/supersplat/settings.json", import.meta.url)).then((res) => res.json());
    viewer = main(appEl.app, cameraEl.entity, settings, {
      contentUrl,
      aa: true,
      noui: true,
      noanim: true,
      ministats: false,
    });
    window.fridge3dgsViewer = { viewer, app: appEl.app, camera: cameraEl.entity };
    if (embedded) {
      document.body.classList.add("viewerEmbedded");
      appEl.app.on("frameend", () => {
        if (!applyingRemoteCamera) {
          window.parent.postMessage({ type: "fridge3dgs.cameraChanged", camera: cameraPayload() }, "*");
        }
      });
      makeEmbeddedCanvasTransparent();
    }
    applyObjectOrientationWhenReady();
    if (requestedMesh) {
      setStatus("Loading mesh");
      await loadMesh(new URL(requestedMesh, window.location.href).toString());
      setStatus("Ready");
    } else {
      setStatus("Loading splats");
    }
  } catch (error) {
    console.error(error);
    setStatus(`Error: ${error.message || error}`);
    window.parent.postMessage(
      { type: "fridge3dgs.viewerError", error: String(error.message || error) },
      window.location.origin,
    );
  }
}

window.addEventListener("message", async (event) => {
  const msg = event.data || {};
  if (!String(msg.type || "").startsWith("fridge3dgs.")) return;
  try {
    if (msg.type === "fridge3dgs.capture") {
      const result = await capture();
      if (event.source) event.source.postMessage(
        { type: "fridge3dgs.captureResult", requestId: msg.requestId, ok: true, ...result },
        event.origin,
      );
    } else if (msg.type === "fridge3dgs.frame") {
      if (viewer && viewer.global && viewer.global.events) viewer.global.events.fire("inputEvent", "frame");
    } else if (msg.type === "fridge3dgs.reset") {
      if (viewer && viewer.global && viewer.global.events) viewer.global.events.fire("inputEvent", "reset");
    } else if (msg.type === "fridge3dgs.zoom") {
      zoomCamera(msg.delta);
    } else if (msg.type === "fridge3dgs.orientation") {
      applyObjectOrientation(msg.orientation || pendingOrientation);
    } else if (msg.type === "fridge3dgs.cameraState") {
      applyCameraPayload(msg.camera);
    } else if (msg.type === "fridge3dgs.visibility") {
      const target = meshEntity || findGsplatEntity();
      if (target) target.enabled = msg.visible !== false;
      if (appEl.app) appEl.app.renderNextFrame = true;
    }
  } catch (error) {
    if (event.source) event.source.postMessage(
      { type: "fridge3dgs.captureResult", requestId: msg.requestId, ok: false, error: String(error.message || error) },
      event.origin,
    );
  }
});

init();
