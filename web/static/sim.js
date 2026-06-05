// sim.js — Three.js 3D robot arm visualiser + SocketIO client

// ── SocketIO ──────────────────────────────────────────────────────────────────
const socket = io();
let simRunning = false;

socket.on("connect",    () => addLog("Connected to server", "ok"));
socket.on("disconnect", () => addLog("Disconnected", "err"));
socket.on("state",      (data) => updateFromState(data));

// ── Three.js setup ────────────────────────────────────────────────────────────
const canvas = document.getElementById("three-canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.setClearColor(0x0d0d0f);

const scene = new THREE.Scene();
const camera3d = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera3d.position.set(1.2, 1.0, 1.2);
camera3d.lookAt(0, 0.3, 0);

// Lighting
scene.add(new THREE.AmbientLight(0x888899, 1.2));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
dirLight.position.set(2, 4, 2);
dirLight.castShadow = true;
scene.add(dirLight);

// Grid
const grid = new THREE.GridHelper(1.2, 12, 0x2a2a40, 0x1e1e2e);
scene.add(grid);

// Robot arm geometry
const linkMat = new THREE.MeshPhongMaterial({ color: 0x4f8cff, shininess: 60 });
const jointMat = new THREE.MeshPhongMaterial({ color: 0x00e5a0, shininess: 80 });
const eeMat   = new THREE.MeshPhongMaterial({ color: 0xff4f6a, shininess: 80 });

const LINK_RADIUS = 0.025;
const links  = [];
const joints = [];

for (let i = 0; i < 6; i++) {
  const joint = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 12),
    i === 5 ? eeMat : jointMat);
  joint.castShadow = true;
  scene.add(joint);
  joints.push(joint);

  const link = new THREE.Mesh(new THREE.CylinderGeometry(LINK_RADIUS, LINK_RADIUS, 1, 8),
    linkMat);
  link.castShadow = true;
  scene.add(link);
  links.push(link);
}

// Target sphere
const targetMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.025, 10, 10),
  new THREE.MeshPhongMaterial({ color: 0xffaa00, wireframe: false })
);
targetMesh.visible = false;
scene.add(targetMesh);

// Workspace objects (cylinders for detected targets)
const objMeshes = [];
const OBJ_COLORS = { red: 0xff3344, green: 0x22cc66, blue: 0x4488ff, yellow: 0xffdd00 };
for (let i = 0; i < 5; i++) {
  const m = new THREE.Mesh(
    new THREE.CylinderGeometry(0.04, 0.04, 0.05, 16),
    new THREE.MeshPhongMaterial({ color: 0xffffff })
  );
  m.visible = false;
  scene.add(m);
  objMeshes.push(m);
}

function updateRobotGeometry(linkPositions) {
  if (!linkPositions || linkPositions.length < 6) return;
  const base = new THREE.Vector3(0, 0, 0);
  const pts = [base, ...linkPositions.map(p => new THREE.Vector3(...p))];

  for (let i = 0; i < 6; i++) {
    const a = pts[i], b = pts[i + 1];
    joints[i].position.copy(b);

    const mid = a.clone().add(b).multiplyScalar(0.5);
    const len = a.distanceTo(b);
    link = links[i];
    link.position.copy(mid);
    link.scale.y = Math.max(len, 0.001);

    const dir = b.clone().sub(a).normalize();
    const up  = new THREE.Vector3(0, 1, 0);
    const q   = new THREE.Quaternion().setFromUnitVectors(up, dir);
    link.setRotationFromQuaternion(q);
  }

  const ee = pts[6] || pts[pts.length - 1];
  document.getElementById("ee-readout").textContent =
    `EE: x=${ee.x.toFixed(3)}  y=${ee.y.toFixed(3)}  z=${ee.z.toFixed(3)}`;
}

function updateScene(sceneData) {
  if (!sceneData) return;
  sceneData.forEach((obj, i) => {
    if (i >= objMeshes.length) return;
    const m = objMeshes[i];
    const [x, y] = obj.pos;
    m.position.set(x, 0.025, y);
    m.material.color.setHex(OBJ_COLORS[obj.color] || 0xffffff);
    m.visible = true;
  });
  for (let i = sceneData.length; i < objMeshes.length; i++) objMeshes[i].visible = false;
}

function updateFromState(data) {
  if (data.arm && data.arm.link_positions) updateRobotGeometry(data.arm.link_positions);
  if (data.scene) updateScene(data.scene);
  if (data.log) data.log.slice(-5).forEach(l => addLog(l));
  updateSliders(data.arm);
}

// ── Resize ────────────────────────────────────────────────────────────────────
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera3d.aspect = w / h;
  camera3d.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas);
resize();

// Animate
(function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera3d);
})();

// ── Joint sliders ─────────────────────────────────────────────────────────────
const sliderContainer = document.getElementById("joint-sliders");
const LIMITS_DEG = [[-166,166],[-101,101],[-166,166],[-176,-4],[-166,166],[-1,215]];

for (let i = 0; i < 6; i++) {
  const div = document.createElement("div");
  div.className = "joint-ctrl";
  div.innerHTML = `
    <label>J${i+1}</label>
    <input type="range" id="j${i+1}" min="${LIMITS_DEG[i][0]}" max="${LIMITS_DEG[i][1]}" value="0"
      oninput="onSlider(${i},this.value)">
    <span class="val" id="v${i+1}">0°</span>`;
  sliderContainer.appendChild(div);
}

function onSlider(idx, val) {
  document.getElementById(`v${idx+1}`).textContent = Math.round(val) + "°";
  const q = {};
  for (let i = 0; i < 6; i++) {
    q[`joint_${i+1}`] = parseFloat(document.getElementById(`j${i+1}`).value);
  }
  fetch("/api/joints", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(q),
  })
  .then(r => r.json())
  .then(s => updateRobotGeometry(s.link_positions));
}

function updateSliders(arm) {
  if (!arm || !arm.joints_deg) return;
  for (let i = 0; i < 6; i++) {
    const v = arm.joints_deg[`joint_${i+1}`] || 0;
    const el = document.getElementById(`j${i+1}`);
    if (el) { el.value = v; document.getElementById(`v${i+1}`).textContent = Math.round(v) + "°"; }
  }
}

// ── Controls ──────────────────────────────────────────────────────────────────
function detect() {
  const color = document.getElementById("color-select").value;
  document.getElementById("status-badge").className = "badge badge-detect";
  document.getElementById("status-badge").textContent = "Detecting…";
  fetch("/api/detect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ color }),
  })
  .then(r => r.json())
  .then(d => {
    if (d.detections && d.detections.length) {
      const p = d.pick_target;
      targetMesh.position.set(p[0], p[2], p[1]);
      targetMesh.visible = true;
      addLog(`Detected ${color} → [${p.map(v=>v.toFixed(2)).join(", ")}]`, "ok");
    } else {
      addLog(`No ${color} object found`, "err");
    }
    document.getElementById("status-badge").className = "badge badge-idle";
    document.getElementById("status-badge").textContent = "Idle";
  });
  renderCamera();
}

function randomScene() {
  fetch("/api/scene/randomize", { method: "POST" })
    .then(r => r.json()).then(d => { updateScene(d.scene); renderCamera(); });
}

function goHome() {
  fetch("/api/home", { method: "POST" });
  addLog("Homing…");
}

function toggleSim() {
  const btn = document.getElementById("sim-btn");
  if (!simRunning) {
    simRunning = true;
    btn.textContent = "■ Stop";
    btn.classList.add("running");
    document.getElementById("status-badge").className = "badge badge-running";
    document.getElementById("status-badge").textContent = "Running";
    socket.emit("start_sim", { color: document.getElementById("color-select").value });
    addLog("Simulation started", "ok");
  } else {
    simRunning = false;
    btn.textContent = "▶ Run Sim";
    btn.classList.remove("running");
    document.getElementById("status-badge").className = "badge badge-idle";
    document.getElementById("status-badge").textContent = "Idle";
    socket.emit("stop_sim");
  }
}

// ── Camera canvas placeholder ─────────────────────────────────────────────────
function renderCamera() {
  const cvs = document.getElementById("cam-canvas");
  const ctx = cvs.getContext("2d");
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, cvs.width, cvs.height);
  ctx.strokeStyle = "#2a2a40";
  for (let x = 0; x < cvs.width; x += 32)  { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cvs.height); ctx.stroke(); }
  for (let y = 0; y < cvs.height; y += 32) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cvs.width, y); ctx.stroke(); }
  ctx.fillStyle = "#00e5a0"; ctx.font = "11px monospace";
  ctx.fillText("Workspace Camera Feed", 8, 18);
  ctx.fillStyle = "#888"; ctx.fillText("(connect to backend to stream live)", 8, 34);
}

// ── Log ───────────────────────────────────────────────────────────────────────
const logEl = document.getElementById("log");
const seen = new Set();
function addLog(msg, cls = "") {
  if (seen.has(msg)) return;
  seen.add(msg);
  const d = document.createElement("div");
  d.className = "entry " + cls;
  d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.prepend(d);
  if (logEl.children.length > 40) logEl.lastChild.remove();
  setTimeout(() => seen.delete(msg), 5000);
}

renderCamera();
