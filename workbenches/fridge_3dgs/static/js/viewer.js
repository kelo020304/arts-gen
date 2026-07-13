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
    const requestedContent = new URLSearchParams(window.location.search).get("content");
    const contentUrl = requestedContent
      ? new URL(requestedContent, window.location.href).toString()
      : appUrl("assets/point_cloud.ply");
    window.firstFrame = () => {
      firstFrameReady = true;
      applyObjectOrientation(pendingOrientation);
      setStatus("Ready");
      window.parent.postMessage({ type: "fridge3dgs.viewerReady" }, window.location.origin);
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
    applyObjectOrientationWhenReady();
    setStatus("Loading splats");
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
    }
  } catch (error) {
    if (event.source) event.source.postMessage(
      { type: "fridge3dgs.captureResult", requestId: msg.requestId, ok: false, error: String(error.message || error) },
      event.origin,
    );
  }
});

init();
