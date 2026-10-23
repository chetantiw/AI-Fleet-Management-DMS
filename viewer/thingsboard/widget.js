// Cabin twin widget: renders a generic driver from the `twin` telemetry key (one JSON object per frame).
let twinInst = null;

self.onInit = function () {
  const root = self.ctx.$container[0];
  twinInst = makeCabinTwin(root);
  self.onDataUpdated();
};

self.onDataUpdated = function () {
  if (!twinInst) return;
  for (const d of self.ctx.data || []) {
    if (!d.dataKey || d.dataKey.name !== "twin" || !d.data || !d.data.length) continue;
    const [ts, v] = d.data[d.data.length - 1];
    let f = v;
    if (typeof v === "string") { try { f = JSON.parse(v); } catch (e) { continue; } }
    if (f && typeof f === "object") twinInst.push(f, ts);
  }
};

self.onResize = function () { if (twinInst) twinInst.resize(); };
self.onDestroy = function () { if (twinInst) twinInst.destroy(); twinInst = null; };

self.typeParameters = function () {
  return { maxDatasources: 1, maxDataKeys: 1, dataKeysOptional: false, singleEntity: true };
};

function makeCabinTwin(root) {
  const viewEl = root.querySelector(".tw-view");
  const NEUTRAL = { yaw: 0, pitch: 0, roll: 0, eyeL: 1, eyeR: 1, jaw: 0, lean: [0, 0], lE: [0.75, -0.95, 0.45], lW: [0.45, -1.15, 1.1],
    rE: [-0.75, -0.95, 0.45], rW: [-0.45, -1.15, 1.1], armOk: 0, face: 1, facing: 1, closed: 0, hidden: 0, phone: 0, mouth: 0, ear: 0,
    belt: -1, perclos: 0, hand: "", alert: "", ev: [], driver: "" };
  let cur = Object.assign({}, NEUTRAL), lastTs = 0, lastRecv = 0, raf = 0, alive = true;

  // ---------- three.js scene ----------
  const view = viewEl;
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  view.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(50, 1, 0.05, 30);
  const controls = new THREE.OrbitControls(cam, renderer.domElement);
  controls.enableDamping = true; controls.enabled = false;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x445055, 0.85));
  const sun = new THREE.DirectionalLight(0xffffff, 0.75);
  sun.position.set(1.5, 3, 2.5); sun.castShadow = true; sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -1.5, right: 1.5, top: 1.5, bottom: -1.5 });
  scene.add(sun);

  const M = (c, o = {}) => new THREE.MeshStandardMaterial(Object.assign({ color: c, roughness: 0.75, metalness: 0.05 }, o));
  const mat = {
    cab: M(0x3b4448), cabDark: M(0x262c2f), seat: M(0x4a5a63), seatB: M(0x5c6e78), skin: M(0xb07a55), hair: M(0x1e1a18),
    shirt: M(0xd9dde0), trouser: M(0x2d3440), wheel: M(0x8a4b2a, { roughness: 0.5 }), wheelHub: M(0x23292c),
    eyeW: M(0xf2efe9, { roughness: 0.3 }), iris: M(0x2a1d16), lip: M(0x6e3b2c), belt: M(0x2b2f33, { roughness: 0.9 }),
    beltOff: M(0xc4362c, { transparent: true, opacity: 0.85 }), beltNA: M(0x8a9396, { transparent: true, opacity: 0.35 }),
    phone: M(0x111418, { roughness: 0.3, metalness: 0.4 }), screen: M(0x5ab0ff, { emissive: 0x2a7fd4, emissiveIntensity: 0.9 }),
    bottle: M(0x8fd0e8, { transparent: true, opacity: 0.6, roughness: 0.1 }), cap: M(0x2468c8), glass: M(0x9fc4d4, { transparent: true, opacity: 0.18 }),
    floor: M(0x2f383b),
  };
  const box = (w, h, d, m) => { const o = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m); o.castShadow = o.receiveShadow = true; return o; };
  const cyl = (r1, r2, h, m, seg = 20) => { const o = new THREE.Mesh(new THREE.CylinderGeometry(r1, r2, h, seg), m); o.castShadow = true; return o; };
  const sph = (r, m) => { const o = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 18), m); o.castShadow = true; return o; };
  const at = (o, x, y, z) => { o.position.set(x, y, z); scene.add(o); return o; };

  // Coordinates: metres, y up, the driver faces +z (windscreen), +x is the driver's left.
  // Indian right-hand-drive bus: door pillar on the driver's right (-x), gear lever on the left (+x).
  at(box(2.2, 0.02, 2.4, mat.floor), 0, 0, 0.2).receiveShadow = true;
  at(box(1.9, 0.62, 0.55, mat.cab), 0.05, 0.62, 1.15);                // dashboard
  const dashTop = at(box(1.9, 0.06, 0.75, mat.cabDark), 0.05, 0.95, 1.05); dashTop.rotation.x = -0.12;
  at(box(0.08, 1.9, 0.12, mat.cab), -0.62, 0.95, 0.75);               // A-pillar / door frame
  at(box(0.02, 1.2, 1.1, mat.glass), -0.66, 1.25, 0.2);               // side window
  at(box(0.25, 0.5, 0.35, mat.cabDark), 0.55, 0.25, 0.55);            // gearbox housing
  const lever = at(cyl(0.015, 0.015, 0.35, mat.wheelHub), 0.55, 0.62, 0.55); lever.rotation.x = -0.25;
  at(sph(0.045, mat.wheelHub), 0.55, 0.79, 0.6);
  // seat
  at(box(0.55, 0.12, 0.5, mat.seat), 0, 0.46, -0.02);
  at(box(0.32, 0.4, 0.32, mat.cabDark), 0, 0.2, -0.02);
  const back = at(box(0.55, 0.85, 0.12, mat.seatB), 0, 0.92, -0.3); back.rotation.x = -0.12;
  // steering wheel: a big, fairly flat bus wheel
  const wheel = new THREE.Group(); wheel.position.set(0, 0.9, 0.5); wheel.rotation.x = -1.05; scene.add(wheel);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.25, 0.022, 14, 48), mat.wheel); rim.castShadow = true; wheel.add(rim);
  const hub = cyl(0.07, 0.07, 0.05, mat.wheelHub); hub.rotation.x = Math.PI / 2; wheel.add(hub);
  for (const a of [0, 2.1, 4.2]) { const sp = box(0.24, 0.025, 0.02, mat.wheelHub); sp.position.set(Math.cos(a) * 0.12, Math.sin(a) * 0.12, 0); sp.rotation.z = a; wheel.add(sp); }
  const column = at(cyl(0.04, 0.05, 0.5, mat.wheelHub), 0, 0.72, 0.68); column.rotation.x = 0.55;

  // ---------- generic driver ----------
  const S = 0.40;                                   // shoulder width (m) = 1 telemetry unit
  const legs = new THREE.Group(); scene.add(legs);
  for (const sx of [1, -1]) {
    const th = box(0.15, 0.15, 0.46, mat.trouser); th.position.set(sx * 0.1, 0.58, 0.18); legs.add(th);
    const sh = box(0.12, 0.48, 0.13, mat.trouser); sh.position.set(sx * 0.1, 0.3, 0.42); legs.add(sh);
    const ft = box(0.11, 0.07, 0.24, mat.cabDark); ft.position.set(sx * 0.1, 0.05, 0.5); legs.add(ft);
  }
  const hips = new THREE.Group(); hips.position.set(0, 0.55, -0.05); scene.add(hips);
  const torso = new THREE.Group(); hips.add(torso);
  const chest = new THREE.Mesh(new THREE.CapsuleGeometry(0.15, 0.28, 8, 20), mat.shirt); chest.castShadow = true;
  chest.scale.set(1.25, 1, 0.72); chest.position.set(0, 0.27, 0); torso.add(chest);
  const belly = new THREE.Mesh(new THREE.CapsuleGeometry(0.14, 0.1, 8, 20), mat.shirt); belly.scale.set(1.15, 1, 0.8); belly.position.set(0, 0.08, 0.01); torso.add(belly);
  const shoulderBar = cyl(0.075, 0.075, 0.42, mat.shirt); shoulderBar.rotation.z = Math.PI / 2; shoulderBar.position.set(0, 0.47, 0); torso.add(shoulderBar);
  const SH = { L: new THREE.Vector3(S / 2, 0.47, 0), R: new THREE.Vector3(-S / 2, 0.47, 0) };   // in torso space
  const neck = cyl(0.05, 0.055, 0.1, mat.skin); neck.position.set(0, 0.55, 0); torso.add(neck);
  const headPivot = new THREE.Group(); headPivot.position.set(0, 0.58, 0.01); headPivot.rotation.order = "YXZ"; torso.add(headPivot);
  const skull = sph(0.1, mat.skin); skull.scale.set(0.95, 1.18, 1.05); skull.position.set(0, 0.12, 0); headPivot.add(skull);
  const hair = sph(0.103, mat.hair); hair.scale.set(0.97, 0.72, 1.07); hair.position.set(0, 0.18, -0.012); headPivot.add(hair);
  const nose = box(0.022, 0.04, 0.03, mat.skin); nose.position.set(0, 0.115, 0.105); headPivot.add(nose);
  for (const sx of [1, -1]) { const ear = sph(0.022, mat.skin); ear.scale.set(0.5, 1, 0.8); ear.position.set(sx * 0.096, 0.12, 0); headPivot.add(ear); }
  const eyes = {};
  for (const [k, sx] of [["L", 1], ["R", -1]]) {
    const g = new THREE.Group(); g.position.set(sx * 0.037, 0.145, 0.088); headPivot.add(g);
    const w = sph(0.016, mat.eyeW); w.scale.set(1.2, 0.8, 0.6); g.add(w);
    const ir = sph(0.007, mat.iris); ir.position.set(0, 0, 0.009); g.add(ir);
    const lid = sph(0.019, mat.skin); lid.scale.set(1.2, 1, 0.75); lid.position.set(0, 0.016, 0.001); g.add(lid);   // upper lid slides down
    const brow = box(0.034, 0.006, 0.01, mat.hair); brow.position.set(0, 0.024, 0.006); g.add(brow);
    eyes[k] = { lid };
  }
  const mouthG = new THREE.Group(); mouthG.position.set(0, 0.065, 0.095); headPivot.add(mouthG);
  const lipU = box(0.05, 0.008, 0.012, mat.lip); mouthG.add(lipU);
  const mouthIn = box(0.042, 0.02, 0.006, mat.hair); mouthIn.position.z = -0.003; mouthG.add(mouthIn);
  const lipL = box(0.05, 0.008, 0.012, mat.lip); mouthG.add(lipL);
  const chin = sph(0.062, mat.skin); chin.scale.set(1.1, 0.8, 1); chin.position.set(0, 0.055, 0.035); headPivot.add(chin);

  // arms: simple limbs re-placed every frame from a two-bone IK solve
  const limb = (r, m) => { const o = cyl(r, r * 0.9, 1, m, 14); scene.add(o); return o; };
  const arms = {};
  for (const k of ["L", "R"]) {
    arms[k] = { up: limb(0.048, mat.shirt), fore: limb(0.038, mat.skin), elbow: at(sph(0.043, mat.shirt), 0, 0, 0),
      hand: at(sph(0.045, mat.skin), 0, 0, 0) };
    arms[k].hand.scale.set(0.85, 1.15, 0.65);
  }
  const L1 = 0.29, L2 = 0.28;
  const Y = new THREE.Vector3(0, 1, 0);
  function placeLimb(o, a, b) {
    const d = new THREE.Vector3().subVectors(b, a); const len = d.length();
    o.position.copy(a).addScaledVector(d, 0.5); o.scale.set(1, len, 1);
    o.quaternion.setFromUnitVectors(Y, d.normalize());
  }

  // props
  const phone = new THREE.Group();
  phone.add(box(0.07, 0.145, 0.012, mat.phone));
  const scr = box(0.06, 0.125, 0.002, mat.screen); scr.position.z = 0.007; phone.add(scr);
  scene.add(phone);
  const bottle = new THREE.Group();
  bottle.add(cyl(0.032, 0.032, 0.2, mat.bottle)); const cp = cyl(0.016, 0.016, 0.03, mat.cap); cp.position.y = 0.115; bottle.add(cp);
  scene.add(bottle);

  // seat belt: anchor above the driver's right shoulder on the pillar, across the chest to the left hip buckle
  const beltOn = new THREE.Mesh(new THREE.BufferGeometry(), mat.belt); torso.add(beltOn);
  (function () {
    const c = new THREE.CatmullRomCurve3([new THREE.Vector3(-0.2, 0.56, -0.1), new THREE.Vector3(-0.15, 0.47, 0.115),
      new THREE.Vector3(0, 0.27, 0.125), new THREE.Vector3(0.17, 0.04, 0.12), new THREE.Vector3(0.21, -0.02, 0.05)]);
    beltOn.geometry = new THREE.TubeGeometry(c, 40, 0.013, 6, false); beltOn.scale.set(1, 1, 1);
  })();
  const beltOff = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.55, 1.55, -0.15), new THREE.Vector3(-0.56, 1.0, -0.12), new THREE.Vector3(-0.54, 0.62, -0.1)]), 20, 0.013, 6, false), mat.beltOff);
  scene.add(beltOff);
  const buckle = at(box(0.05, 0.08, 0.03, mat.cabDark), 0.25, 0.53, 0.0);

  // ---------- camera presets ----------
  const CAMS = {
    dms: { pos: [0.06, 1.62, 1.45], look: [0, 0.98, 0.05], fov: 46 },     // high on the dash, looking down like the bus camera
    side: { pos: [1.9, 1.3, 0.35], look: [0, 0.95, 0.15], fov: 42 },
    free: { pos: [1.5, 1.7, 1.9], look: [0, 0.9, 0.1], fov: 42 },
  };
  let camMode = "dms";
  function setCam(m) {
    camMode = m; const c = CAMS[m];
    cam.position.set(...c.pos); cam.fov = c.fov; cam.updateProjectionMatrix();
    controls.target.set(...c.look); cam.lookAt(controls.target); controls.enabled = m === "free"; controls.update();
    root.querySelectorAll(".camset button").forEach((b) => b.classList.toggle("on", b.dataset.cam === m));
  }
  root.querySelectorAll(".camset button").forEach((b) => b.addEventListener("click", () => setCam(b.dataset.cam)));
  setCam("dms");

  function resize() {
    const r = view.getBoundingClientRect();
    renderer.setSize(r.width, r.height, false); cam.aspect = r.width / Math.max(r.height, 1); cam.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize); ro.observe(view);

  scene.background = new THREE.Color(0xdfe3e1);

  // ---------- pose application ----------
  const V = (a) => new THREE.Vector3(a[0], a[1], a[2]);
  // telemetry arm points are in the camera's view of the shoulders: x left, y up (image), z towards camera.
  // The camera looks down at the driver, so "lower in the image" is partly "further forward"; fold some of it into z.
  function toBody(p) {
    const down = Math.max(0, -p[1]);
    return new THREE.Vector3(p[0] * S, p[1] * S * 0.8, (Math.max(p[2], -0.3) * 0.45 + down * 0.55 + 0.15) * S);
  }
  const POSE = { yaw: 0, pitch: 0, roll: 0, eyeL: 1, eyeR: 1, jaw: 0, lean: [0, 0], lE: [0.75, -0.95, 0.45], lW: [0.45, -1.15, 1.1],
    rE: [-0.75, -0.95, 0.45], rW: [-0.45, -1.15, 1.1] };
  const over = { L: 0, R: 0, kindL: "", kindR: "" };   // smoothed weight of "hand at ear / mouth" overrides
  function solveArm(k, sh, elbowT, wristT) {
    const d = new THREE.Vector3().subVectors(wristT, sh);
    let dist = d.length(); const n = d.clone().normalize();
    dist = Math.min(Math.max(dist, 0.12), L1 + L2 - 0.004);
    const cosA = (L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist);
    const sinA = Math.sqrt(Math.max(0, 1 - cosA * cosA));
    let pole = new THREE.Vector3().subVectors(elbowT, sh);
    pole.addScaledVector(n, -pole.dot(n));
    if (pole.lengthSq() < 1e-6) pole.set(k === "L" ? 1 : -1, -1, 0).addScaledVector(n, -n.dot(new THREE.Vector3(k === "L" ? 1 : -1, -1, 0)));
    pole.normalize();
    const elbow = sh.clone().addScaledVector(n, L1 * cosA).addScaledVector(pole, L1 * sinA);
    const wrist = sh.clone().addScaledVector(n, dist);
    const A = arms[k];
    placeLimb(A.up, sh, elbow); placeLimb(A.fore, elbow, wrist); A.elbow.position.copy(elbow);
    const fd = new THREE.Vector3().subVectors(wrist, elbow).normalize();
    A.hand.position.copy(wrist).addScaledVector(fd, 0.05);
    A.hand.quaternion.setFromUnitVectors(Y, fd);
    return { wrist: A.hand.position.clone(), fd };
  }

  function apply(f, dt) {
    const k = Math.min(1, dt * 12);              // ease towards the frame (frames are 10 Hz, display is 60 Hz)
    const lerp = (a, b) => a + (b - a) * k;
    for (const n of ["yaw", "pitch", "roll", "eyeL", "eyeR", "jaw"]) POSE[n] = lerp(POSE[n], f[n]);
    POSE.lean = POSE.lean.map((v, i) => lerp(v, f.lean[i]));
    for (const n of ["lE", "lW", "rE", "rW"]) POSE[n] = POSE[n].map((v, i) => lerp(v, f[n][i]));
    const present = f.face || f.armOk;
    hips.visible = legs.visible = arms.L.up.visible = true;

    // torso
    hips.position.x = -POSE.lean[0] * S * 0.35;
    torso.rotation.z = -POSE.lean[0] * 0.25;
    torso.rotation.x = Math.max(-0.2, Math.min(0.45, POSE.lean[1] * 0.35));
    // head (degrees -> radians; clamp to anatomical range)
    const c = (v, m) => Math.max(-m, Math.min(m, v)) * Math.PI / 180;
    headPivot.rotation.set(c(POSE.pitch, 60), c(POSE.yaw, 80), c(-POSE.roll, 40));
    for (const [e, v] of [["L", POSE.eyeL], ["R", POSE.eyeR]]) {
      const open = Math.max(0, Math.min(1, v));
      eyes[e].lid.position.y = 0.006 + 0.012 * open;     // lid covers the eye when shut
      eyes[e].lid.scale.y = 1 - 0.35 * open;
    }
    const j = Math.max(0, Math.min(1, POSE.jaw));
    lipL.position.y = -0.004 - j * 0.03; mouthIn.scale.y = 0.4 + j * 1.6; mouthIn.position.y = -j * 0.015; chin.position.y = 0.055 - j * 0.02;
    hips.updateMatrixWorld(true);

    // hand overrides from the alert engine
    const side = f.hand || "L";
    for (const s of ["L", "R"]) {
      const want = (f.ear || f.mouth) && side === s ? 1 : 0;
      over[s] += (want - over[s]) * Math.min(1, dt * 6);
      if (want) over["kind" + s] = f.ear ? "ear" : "mouth";
    }
    const headW = new THREE.Vector3(); headPivot.getWorldPosition(headW);
    const hand = {};
    for (const s of ["L", "R"]) {
      const sx = s === "L" ? 1 : -1;
      const shW = torso.localToWorld(SH[s].clone());
      const mid = torso.localToWorld(new THREE.Vector3(0, 0.47, 0));
      let e = mid.clone().add(toBody(POSE[s === "L" ? "lE" : "rE"]));
      let w = mid.clone().add(toBody(POSE[s === "L" ? "lW" : "rW"]));
      if (over[s] > 0.01) {
        const kind = over["kind" + s];
        const wT = kind === "ear" ? headW.clone().add(new THREE.Vector3(sx * 0.11, 0.12, 0.0))
                                  : headW.clone().add(new THREE.Vector3(sx * 0.03, 0.04, 0.17));
        const eT = kind === "ear" ? shW.clone().add(new THREE.Vector3(sx * 0.22, -0.18, 0.12))
                                  : shW.clone().add(new THREE.Vector3(sx * 0.12, -0.25, 0.25));
        w.lerp(wT, over[s]); e.lerp(eT, over[s]);
      }
      hand[s] = solveArm(s, shW, e, w);
    }

    // props
    const hs = hand[side];
    const showPhone = f.phone || f.ear;
    const showBottle = f.mouth && !f.phone && !f.ear;
    phone.visible = !!showPhone; bottle.visible = !!showBottle;
    if (showPhone) {
      phone.position.copy(hs.wrist).addScaledVector(hs.fd, 0.06);
      if (f.ear) { phone.lookAt(headW); phone.rotateY(Math.PI / 2); }
      else { phone.quaternion.setFromUnitVectors(Y, hs.fd); phone.rotateX(-1.1); }
    }
    if (showBottle) {
      bottle.position.copy(hs.wrist).addScaledVector(hs.fd, 0.03);
      bottle.quaternion.setFromUnitVectors(Y, new THREE.Vector3(0, 1, -0.9).normalize());
    }
    beltOn.visible = f.belt === 1; beltOff.visible = f.belt === 0;
    beltOn.material = mat.belt;
    if (f.belt === -1) { beltOn.visible = true; beltOn.material = mat.beltNA; }
    // driver absent: fade the person away
    const vis = !(f.face === 0 && f.armOk === 0 && f.ev.length === 0 && f.alert.startsWith("DRIVER ABSENT"));
    for (const o of [hips, legs, arms.L.up, arms.L.fore, arms.L.elbow, arms.L.hand, arms.R.up, arms.R.fore, arms.R.elbow, arms.R.hand]) o.visible = vis;
  }



  const chips = root.querySelector(".tw-chips"), drv = root.querySelector(".tw-driver"),
        alertEl = root.querySelector(".tw-alert"), stale = root.querySelector(".tw-stale");
  function chip(text, cls) { return `<span class="tw-chip ${cls}">${text}</span>`; }
  function panel(f) {
    drv.textContent = f.driver || (f.face ? "Identifying driver…" : "No driver");
    const eyes = !f.face ? chip("Eyes n/a", "na") : f.hidden ? chip("Eyes hidden", "na") : !f.facing ? chip("Head turned", "warn")
               : f.closed ? chip("Eyes closed", "bad") : chip("Eyes open", "ok");
    chips.innerHTML = eyes
      + (f.ear ? chip("Hand at ear", "warn") : f.mouth ? chip("Hand at mouth", "warn") : "")
      + (f.phone ? chip("Phone", "bad") : "")
      + (f.belt === 1 ? chip("Belt on", "ok") : f.belt === 0 ? chip("Belt off", "bad") : chip("Belt checking", "na"))
      + chip("PERCLOS " + Math.round((f.perclos || 0) * 100) + "%", f.perclos > 0.25 ? "bad" : f.perclos > 0.15 ? "warn" : "ok");
    alertEl.hidden = !f.alert; alertEl.textContent = f.alert || "";
  }
  panel(cur);
  let prev = performance.now();
  function loop(now) {
    if (!alive) return;
    const dt = Math.min(0.1, (now - prev) / 1000); prev = now;
    apply(cur, dt);
    const age = lastRecv ? (Date.now() - lastRecv) / 1000 : 0;
    stale.hidden = !lastRecv || age < 3; if (!stale.hidden) stale.querySelector("b").textContent = Math.round(age);
    controls.update(); renderer.render(scene, cam);
    raf = requestAnimationFrame(loop);
  }
  raf = requestAnimationFrame(loop);
  return {
    push(f, ts) {
      if (ts && ts < lastTs) return;               // ignore out-of-order frames
      lastTs = ts || lastTs; lastRecv = Date.now();
      cur = Object.assign({}, NEUTRAL, f);
      for (const k of ["lean", "lE", "lW", "rE", "rW"]) if (typeof cur[k] === "string") { try { cur[k] = JSON.parse(cur[k]); } catch (e) { cur[k] = NEUTRAL[k]; } }
      if (!Array.isArray(cur.ev)) cur.ev = [];
      panel(cur);
    },
    resize,
    destroy() { alive = false; cancelAnimationFrame(raf); ro.disconnect(); controls.dispose(); renderer.dispose(); renderer.domElement.remove(); },
  };
}
