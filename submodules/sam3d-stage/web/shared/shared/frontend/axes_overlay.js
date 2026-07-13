/**
 * RGB world-axes gizmo in the bottom-right corner. Tiny Three.js scene with
 * three RGB axes (cylinder meshes, not lines, so they have actual thickness)
 * that rotates as the active viewer's camera moves, so the user always knows
 * which direction is +X / +Y / +Z in world space.
 *
 * Reusable across viewers — the viewer just has to provide a way to read its
 * current camera quaternion.
 */
import * as THREE from "three";

/**
 * @param {HTMLElement} host - the container the overlay attaches to (will be
 *   forced to position:relative so the absolute overlay child lands correctly)
 * @param {() => {x:number,y:number,z:number,w:number} | null} getCameraQuat -
 *   returns the active viewer's world-space camera quaternion each frame.
 *   Return null to hide the overlay.
 * @param {object} [opts]
 * @param {number} [opts.size=120]   pixel size of the overlay
 * @param {number} [opts.right=14]   px offset from right edge of host
 * @param {number} [opts.bottom=14]  px offset from bottom edge of host
 * @returns {{ dispose(): void }}
 */
export function createAxesOverlay(host, getCameraQuat, opts = {}) {
  const size   = opts.size   ?? 120;
  const right  = opts.right  ?? 14;
  const bottom = opts.bottom ?? 14;

  // We use position:fixed so the overlay stays in the viewport corner
  // regardless of how tall the viewer's host element gets (SuperSplat's
  // pc-app frequently overflows its parent, pushing absolute-positioned
  // siblings off-screen). The `host` parameter is no longer used for
  // positioning, only to track ownership / lifecycle.
  void host;

  const canvas = document.createElement("canvas");
  canvas.style.cssText = `
    position: fixed;
    right: ${right}px;
    bottom: ${bottom}px;
    width: ${size}px;
    height: ${size}px;
    display: block;
    pointer-events: none;
    z-index: 99999;
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(0, 0, 0, 0.35);
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.35);
  `;
  document.body.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(size, size, false);
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 10);
  camera.position.set(0, 0, 3);
  camera.lookAt(0, 0, 0);

  // Subtle ambient so the cones aren't totally flat
  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
  dirLight.position.set(2, 3, 4);
  scene.add(dirLight);

  // Group holds the three axes; we rotate this by the inverse of the viewer
  // camera quaternion so the axes appear to spin as seen from that camera.
  const group = new THREE.Group();
  scene.add(group);

  /**
   * Build an axis: cylinder shaft + cone tip, both colored, pointing along +axis.
   * Default cylinder is along +Y, so we apply rotation matrices.
   */
  function axisMesh(color, axis) {
    const shaft = new THREE.CylinderGeometry(0.045, 0.045, 0.85, 14);
    shaft.translate(0, 0.425, 0);     // shaft from 0 to 0.85 along +Y
    const tip = new THREE.ConeGeometry(0.12, 0.30, 16);
    tip.translate(0, 0.85 + 0.15, 0); // tip starts where shaft ends

    // Merge by adding both as children (cheaper than BufferGeometryUtils)
    const mat = new THREE.MeshLambertMaterial({ color });
    const shaftMesh = new THREE.Mesh(shaft, mat);
    const tipMesh   = new THREE.Mesh(tip, mat);
    const axisGroup = new THREE.Group();
    axisGroup.add(shaftMesh);
    axisGroup.add(tipMesh);

    // Default orientation is +Y; rotate to align with requested axis
    if (axis === "x") axisGroup.rotation.z = -Math.PI / 2;       // +Y -> +X
    else if (axis === "z") axisGroup.rotation.x = Math.PI / 2;   // +Y -> +Z
    return axisGroup;
  }

  group.add(axisMesh(0xff2828, "x"));   // red    +X
  group.add(axisMesh(0x28c828, "y"));   // green  +Y
  group.add(axisMesh(0x3060ff, "z"));   // blue   +Z

  // Small central sphere for visual anchor
  group.add(new THREE.Mesh(
    new THREE.SphereGeometry(0.07, 12, 12),
    new THREE.MeshLambertMaterial({ color: 0x303030 }),
  ));

  // Scale the whole gizmo to fit comfortably inside the overlay box
  // (axes are 1.15 units long; at cam dist 3 + FOV 35° they exceed the
  // visible ~0.95 half-width without scaling).
  group.scale.setScalar(0.6);

  const q = new THREE.Quaternion();
  let raf = 0, running = true;
  function tick() {
    if (!running) return;
    raf = requestAnimationFrame(tick);
    const view = getCameraQuat();
    if (view) {
      q.set(view.x, view.y, view.z, view.w).invert();
      group.quaternion.copy(q);
      canvas.style.display = "block";
    } else {
      canvas.style.display = "none";
    }
    renderer.render(scene, camera);
  }
  tick();

  return {
    dispose() {
      running = false;
      cancelAnimationFrame(raf);
      renderer.dispose();
      canvas.remove();
    },
  };
}
