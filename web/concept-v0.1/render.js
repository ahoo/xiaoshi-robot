(function () {
  'use strict';
  const C = SimCore, W = C.W, H = C.H, GX = C.GX, GY = C.GY, HOME = C.HOME, L1 = C.L1, L2 = C.L2, ik = C.ik, fk = C.fk, clamp = C.clamp;
  const $ = id => document.getElementById(id);
  const stage = $('stage'), glc = $('gl'), ovc = $('ov'), ctx = ovc.getContext('2d'), pipEl = $('pip');
  if (!window.THREE) { stage.insertAdjacentHTML('beforeend', '<p class="fail">3D 库没有加载成功，请刷新页面重试。</p>'); return; }
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const DT = 1 / 60, PI = Math.PI;
  const sim = new C.Sim(); for (let i = 0; i < 270; i++) sim.step(DT);

  let mode = 'sim', playing = !reduce, speed = 1, tool = 'box', view = 'follow', autoRot = true, armAuto = true, hiPart = -1;
  const show = { lidar: true, fov: true, path: true };
  let sw = 800, sh = 500, dpr = 1; const pip = { x: 12, y: 0, w: 200, h: 150 };

  /* ---------- three.js 场景 ---------- */
  const renderer = new THREE.WebGLRenderer({ canvas: glc, antialias: true });
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.setClearColor(0x0c221e, 1);
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(36, 1.6, 0.05, 40); cam.layers.enable(1);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x2c5a4e, 0.56));
  const sun = new THREE.DirectionalLight(0xfff1dc, 0.5); sun.position.set(0.4, 5.5, 3.8); sun.target.position.set(2, 0, 1.5); sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048); { const s = sun.shadow.camera; s.left = -3.3; s.right = 3.3; s.top = 3.3; s.bottom = -3.3; s.near = 0.5; s.far = 12; } sun.shadow.bias = -0.0006;
  scene.add(sun, sun.target);

  const matCache = {};
  function mat(c) { return matCache[c] || (matCache[c] = new THREE.MeshLambertMaterial({ color: c })); }
  function box(w, h, d, m) { const me = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m); me.castShadow = true; me.receiveShadow = true; return me; }
  function cyl(r1, r2, h, m, seg) { const me = new THREE.Mesh(new THREE.CylinderGeometry(r1, r2, h, seg || 24), m); me.castShadow = true; me.receiveShadow = true; return me; }
  function zcyl(r, h, m, seg) { const g = new THREE.CylinderGeometry(r, r, h, seg || 20); g.rotateX(PI / 2); const me = new THREE.Mesh(g, m); me.castShadow = true; return me; }

  // 地面：一张切割垫，10 cm 细格、50 cm 粗格
  (function () {
    const c = document.createElement('canvas'); c.width = 2048; c.height = 1536; const x = c.getContext('2d');
    x.fillStyle = '#1f5045'; x.fillRect(0, 0, 2048, 1536);
    x.strokeStyle = 'rgba(255,255,255,.085)'; x.lineWidth = 2; x.beginPath();
    for (let i = 0; i <= 40; i++) { x.moveTo(i * 51.2, 0); x.lineTo(i * 51.2, 1536); } for (let j = 0; j <= 30; j++) { x.moveTo(0, j * 51.2); x.lineTo(2048, j * 51.2); } x.stroke();
    x.strokeStyle = 'rgba(255,255,255,.24)'; x.lineWidth = 3; x.beginPath();
    for (let i = 0; i <= 8; i++) { x.moveTo(i * 256, 0); x.lineTo(i * 256, 1536); } for (let j = 0; j <= 6; j++) { x.moveTo(0, j * 256); x.lineTo(2048, j * 256); } x.stroke();
    x.strokeStyle = 'rgba(255,209,102,.2)'; x.lineWidth = 3; x.beginPath();
    [30, 45, 60].forEach(a => { const r = a * PI / 180; x.moveTo(0, 1536); x.lineTo(Math.cos(r) * 1500, 1536 - Math.sin(r) * 1500); }); x.stroke();
    x.fillStyle = 'rgba(255,255,255,.5)'; x.font = '500 30px ui-monospace,Menlo,monospace';
    for (let i = 1; i < 8; i++) x.fillText(String(i * 50), i * 256 + 8, 1522); for (let j = 1; j < 6; j++) x.fillText(String(j * 50), 10, j * 256 - 8);
    x.strokeStyle = 'rgba(255,255,255,.45)'; x.lineWidth = 6; x.strokeRect(3, 3, 2042, 1530);
    const tex = new THREE.CanvasTexture(c); tex.anisotropy = 8;
    const fl = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.MeshLambertMaterial({ map: tex })); fl.rotation.x = -PI / 2; fl.position.set(W / 2, 0, H / 2); fl.receiveShadow = true; scene.add(fl);
    const base = box(W + 0.5, 0.04, H + 0.5, mat(0x173b33)); base.position.set(W / 2, -0.021, H / 2); base.castShadow = false; scene.add(base);
    const wm = mat(0xe9e5d6);
    [[W / 2, -0.015, W + 0.06, 0.03], [W / 2, H + 0.015, W + 0.06, 0.03], [-0.015, H / 2, 0.03, H], [W + 0.015, H / 2, 0.03, H]].forEach(a => { const w = box(a[2], 0.14, a[3], wm); w.position.set(a[0], 0.07, a[1]); scene.add(w); });
  })();

  function buildObs(o) {
    const g = new THREE.Group(), K = 0xc9a374, KD = 0xa9824f, P = 0xf1eee2; let b;
    switch (o.kind) {
      case 'sofa': {
        b = box(o.w, 0.2, o.h, mat(0xbf9667)); b.position.y = 0.1;
        const bk = box(o.w, 0.24, 0.16, mat(KD)); bk.position.set(0, 0.32, -o.h / 2 + 0.08);
        const a1 = box(0.14, 0.12, o.h - 0.16, mat(KD)); a1.position.set(-o.w / 2 + 0.07, 0.26, 0.08); const a2 = a1.clone(); a2.position.x = o.w / 2 - 0.07;
        g.add(b, bk, a1, a2); break;
      }
      case 'table': { b = box(o.w, 0.24, o.h, mat(0xd9bd92)); b.position.y = 0.12; const t = box(o.w + 0.04, 0.02, o.h + 0.04, mat(0x9c774a)); t.position.y = 0.25; g.add(b, t); break; }
      case 'shelf': { b = box(o.w, 0.62, o.h, mat(0xb48c5c)); b.position.y = 0.31; g.add(b); for (let k = 1; k <= 3; k++) { const s = box(o.w - 0.06, 0.012, 0.01, mat(P)); s.position.set(0, k * 0.155, o.h / 2 + 0.003); s.castShadow = false; g.add(s); } break; }
      case 'plant': {
        b = cyl(o.r, o.r * 0.78, 0.2, mat(0xd98a5a), 24); b.position.y = 0.1; g.add(b);
        const lm = new THREE.MeshLambertMaterial({ color: 0x79c08a, flatShading: true });
        [[0, 0.36, 0, 0.17], [0.08, 0.5, 0.04, 0.12], [-0.07, 0.48, -0.05, 0.11]].forEach(a => { const s = new THREE.Mesh(new THREE.IcosahedronGeometry(a[3], 1), lm); s.position.set(a[0], a[1], a[2]); s.castShadow = true; g.add(s); });
        break;
      }
      case 'chair': { b = box(o.w, 0.22, o.h, mat(K)); b.position.y = 0.11; const bk = box(o.w, 0.3, 0.05, mat(KD)); bk.position.set(0, 0.37, -o.h / 2 + 0.025); g.add(b, bk); break; }
      case 'box': { b = box(o.w, 0.22, o.h, mat(K)); b.position.y = 0.11; const t = box(o.w + 0.002, 0.004, 0.05, mat(0xe7d2a8)); t.position.y = 0.222; t.castShadow = false; g.add(b, t); if (o.user) g.userData.drop = 1; break; }
      case 'dock': { b = box(o.w, 0.07, o.h, mat(P)); b.position.y = 0.035; const s = new THREE.Mesh(new THREE.BoxGeometry(o.w * 0.7, 0.014, 0.01), new THREE.MeshBasicMaterial({ color: 0xff6b2c })); s.position.set(0, 0.045, -o.h / 2 - 0.004); g.add(b, s); break; }
      case 'basket': {
        const t = 0.008, hh = 0.09, pm = mat(P), om = mat(0xff6b2c);
        const f = box(o.w, 0.01, o.h, pm); f.position.y = 0.005; g.add(f);
        [[0, -(o.h / 2 - t / 2), o.w, t], [0, o.h / 2 - t / 2, o.w, t], [-(o.w / 2 - t / 2), 0, t, o.h], [o.w / 2 - t / 2, 0, t, o.h]].forEach(a => {
          const w = box(a[2], hh, a[3], pm); w.position.set(a[0], hh / 2, a[1]); const r = box(a[2] + 0.004, 0.008, a[3] + 0.004, om); r.position.set(a[0], hh + 0.003, a[1]); g.add(w, r);
        });
        break;
      }
    }
    g.position.set(o.x, 0, o.y); return g;
  }
  function buildObj(o) {
    const g = new THREE.Group(), k = o.cls.key;
    if (k === 'sock') { const a = box(0.11, 0.028, 0.042, mat(0xe2483d)), c = box(0.03, 0.03, 0.044, mat(0xf6e9d8)); c.position.x = -0.042; g.add(a, c); }
    else if (k === 'block') g.add(box(0.045, 0.045, 0.045, mat(0x4f9de0)));
    else if (k === 'ball') { const s = new THREE.Mesh(new THREE.SphereGeometry(0.033, 20, 14), mat(0xd6e84a)); s.castShadow = true; g.add(s); }
    else if (k === 'remote') { const a = box(0.13, 0.02, 0.042, mat(0xdcd6ea)), c = box(0.02, 0.004, 0.02, mat(0xff6b2c)); c.position.set(0.04, 0.011, 0); g.add(a, c); }
    else { const a = box(0.09, 0.02, 0.044, mat(0xf2b02e)); a.position.y = 0.002; const c = box(0.045, 0.018, 0.036, mat(0xfbe3a0)); c.position.set(-0.005, 0.019, 0); g.add(a, c); [[0.03, 0.022], [0.03, -0.022], [-0.03, 0.022], [-0.03, -0.022]].forEach(p => { const w = zcyl(0.01, 0.008, mat(0x22262a), 12); w.position.set(p[0], -0.01, p[1]); g.add(w); }); }
    return g;
  }

  // 机器人：局部坐标 +x 朝前，y 朝上
  const bot = (function () {
    const g = new THREE.Group(), mP = mat(0xf1eee2), mD = mat(0x262b2e), mO = mat(0xff6b2c), mR = mat(0x15171a);
    const low = cyl(0.1, 0.096, 0.04, mD, 40); low.position.y = 0.035; g.add(low);
    const up = cyl(0.1, 0.1, 0.035, mP, 40); up.position.y = 0.1025; g.add(up);
    const trim = cyl(0.1014, 0.1014, 0.006, mO, 40); trim.position.y = 0.0865; g.add(trim);
    for (let k = 0; k < 3; k++) { const a = k * 2 * PI / 3 + PI, p = cyl(0.005, 0.005, 0.032, mD, 8); p.position.set(Math.cos(a) * 0.088, 0.07, Math.sin(a) * 0.088); g.add(p); }
    const lidar = new THREE.Group(); lidar.position.y = 0.069; lidar.add(cyl(0.034, 0.034, 0.022, mR, 24)); const lr = cyl(0.0348, 0.0348, 0.005, mO, 24); lr.position.y = 0.006; lidar.add(lr);
    const lw = box(0.004, 0.012, 0.016, mat(0x7fd6ff)); lw.position.set(0.034, -0.002, 0); lidar.add(lw); g.add(lidar);
    function wheel(z) { const w = new THREE.Group(); w.position.set(0, 0.034, z); w.add(zcyl(0.034, 0.02, mR, 28)); const hub = box(0.05, 0.009, 0.023, mO); w.add(hub); const hub2 = box(0.009, 0.05, 0.023, mO); w.add(hub2); g.add(w); return w; }
    const wl = wheel(-0.094), wr = wheel(0.094);
    [0.076, -0.076].forEach(x => { const s = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 8), mR); s.position.set(x, 0.012, 0); g.add(s); });
    const visor = new THREE.Mesh(new THREE.CylinderGeometry(0.1022, 0.1022, 0.027, 24, 1, true, PI / 2 - 0.75, 1.5), new THREE.MeshPhongMaterial({ color: 0x0b0e10, shininess: 90, side: THREE.DoubleSide })); visor.position.y = 0.104; g.add(visor);
    const eyeM = new THREE.MeshBasicMaterial({ color: 0xfff3c4 }), eyes = [];
    [-1, 1].forEach(s => { const a = s * 0.33, e = new THREE.Mesh(new THREE.BoxGeometry(0.003, 0.013, 0.017), eyeM); e.position.set(Math.cos(a) * 0.104, 0.105, Math.sin(a) * 0.104); e.rotation.y = -a; g.add(e); eyes.push(e); });
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.0065, 0.0065, 0.004, 16), new THREE.MeshPhongMaterial({ color: 0x24507a, shininess: 120 })); lens.rotation.z = PI / 2; lens.position.set(0.104, 0.105, 0); g.add(lens);
    const ledM = new THREE.MeshBasicMaterial({ color: 0x8fe3a1 }), led = new THREE.Mesh(new THREE.SphereGeometry(0.007, 10, 8), ledM); led.position.set(-0.085, 0.124, 0); g.add(led);
    const tur = cyl(0.03, 0.034, 0.024, mO, 24); tur.position.set(C.S.x, 0.132, 0); g.add(tur);
    const sh = new THREE.Group(); sh.position.set(C.S.x, C.S.y, 0); g.add(sh);
    sh.add(zcyl(0.017, 0.054, mD)); const l1 = box(L1, 0.02, 0.028, mP); l1.position.x = L1 / 2; sh.add(l1);
    const el = new THREE.Group(); el.position.x = L1; sh.add(el);
    el.add(zcyl(0.015, 0.048, mO)); const l2 = box(L2, 0.018, 0.024, mP); l2.position.x = L2 / 2; el.add(l2);
    const wrg = new THREE.Group(); wrg.position.x = L2; el.add(wrg);
    wrg.add(zcyl(0.012, 0.042, mD)); const palm = box(0.02, 0.024, 0.094, mD); palm.position.x = 0.018; wrg.add(palm);
    const fL = box(0.036, 0.016, 0.006, mO), fR = box(0.036, 0.016, 0.006, mO); fL.position.x = fR.position.x = 0.044; wrg.add(fL, fR);
    const wc = box(0.012, 0.01, 0.014, mat(0x24507a)); wc.position.set(0.02, 0.017, 0); wrg.add(wc);
    scene.add(g);
    const eye = new THREE.PerspectiveCamera(80, 4 / 3, 0.02, 12); eye.rotation.order = 'YXZ'; eye.position.set(0.118, 0.105, 0); eye.rotation.y = -PI / 2; eye.rotation.x = -0.2; g.add(eye);
    return { g, lidar, wl, wr, eyes, ledM, sh, el, wrg, fL, fR, eye };
  })();

  // 辅助图层（机器人自己的相机看不到）
  const NR = sim.lidar.n;
  const rayPos = new Float32Array(NR * 6), rayGeo = new THREE.BufferGeometry(); rayGeo.setAttribute('position', new THREE.BufferAttribute(rayPos, 3));
  const rays = new THREE.LineSegments(rayGeo, new THREE.LineBasicMaterial({ color: 0xff6b2c, transparent: true, opacity: 0.2, depthWrite: false })); rays.layers.set(1); rays.frustumCulled = false; scene.add(rays);
  const hitPos = new Float32Array(NR * 3), hitGeo = new THREE.BufferGeometry(); hitGeo.setAttribute('position', new THREE.BufferAttribute(hitPos, 3));
  const hits = new THREE.Points(hitGeo, new THREE.PointsMaterial({ color: 0xffd166, size: 3.5, sizeAttenuation: false })); hits.layers.set(1); hits.frustumCulled = false; scene.add(hits);
  const FN = 24, fovPos = new Float32Array((FN + 2) * 3), fovGeo = new THREE.BufferGeometry(); fovGeo.setAttribute('position', new THREE.BufferAttribute(fovPos, 3));
  { const idx = []; for (let i = 1; i <= FN; i++) idx.push(0, i + 1, i); fovGeo.setIndex(idx); }
  const fov = new THREE.Mesh(fovGeo, new THREE.MeshBasicMaterial({ color: 0xfff3c4, transparent: true, opacity: 0.13, depthWrite: false, side: THREE.DoubleSide })); fov.layers.set(1); fov.frustumCulled = false; scene.add(fov);

  const obsMesh = new Map(), objMesh = new Map();
  function syncMeshes() {
    for (const o of sim.obstacles) if (!obsMesh.has(o)) { const m = buildObs(o); obsMesh.set(o, m); scene.add(m); }
    for (const [o, m] of obsMesh) if (sim.obstacles.indexOf(o) < 0) { scene.remove(m); obsMesh.delete(o); }
    for (const o of sim.objects) if (!objMesh.has(o)) { const m = buildObj(o); objMesh.set(o, m); scene.add(m); }
    for (const [o, m] of objMesh) if (sim.objects.indexOf(o) < 0) { scene.remove(m); objMesh.delete(o); }
  }

  /* ---------- 机械臂显示姿态 ---------- */
  const dj = sim.arm.j.slice(); let dgrip = sim.arm.grip, armT = 0;
  const slR = $('slR'), slH = $('slH'), slG = $('slG');
  function anatTarget(dt) {
    if (armAuto && !reduce) { armT += dt; slR.value = (21 + 6 * Math.sin(armT * 0.9)).toFixed(1); slH.value = (15 + 13 * Math.sin(armT * 0.6 + 1)).toFixed(1); slG.value = Math.round(50 + 50 * Math.sin(armT * 1.3)); }
    const r = +slR.value / 100, h = +slH.value / 100, g = +slG.value / 100, phi = -PI / 2 + clamp((h - 0.1) / 0.2, 0, 1) * (PI / 2 - 0.5);
    $('oR').textContent = (+slR.value).toFixed(1) + ' cm'; $('oH').textContent = (+slH.value).toFixed(1) + ' cm'; $('oG').textContent = (g * 7.5).toFixed(1) + ' cm';
    return { j: ik(r, h, phi), g };
  }
  const deg = a => Math.round(a * 180 / PI) + '°';

  /* ---------- 相机 ---------- */
  const orbit = { az: -0.55, pol: 0.92, zoom: 1, tx: sim.robot.x, ty: 0.08, tz: sim.robot.y, d: 2.2 };
  function camUpdate(dt) {
    const rb = sim.robot, aspect = sw / sh, nar = Math.min(1, aspect / 1.3); let tx, ty, tz, d;
    if (mode === 'anatomy') { tx = rb.x; ty = 0.14; tz = rb.y; d = 0.82 / nar; if (autoRot && !reduce) orbit.az += dt * 0.22; }
    else if (view === 'follow') { tx = rb.x; ty = 0.08; tz = rb.y; d = (['ALIGN', 'GRASP', 'ALIGNP', 'PLACE'].indexOf(sim.state) >= 0 ? 1.3 : 2.1) / nar; }
    else { tx = 2; ty = 0; tz = 1.5; d = Math.max(2.5 / (Math.tan(18 * PI / 180) * aspect), 5); }
    d *= orbit.zoom; const k = 1 - Math.exp(-dt * 3.2);
    orbit.tx += (tx - orbit.tx) * k; orbit.ty += (ty - orbit.ty) * k; orbit.tz += (tz - orbit.tz) * k; orbit.d += (d - orbit.d) * k;
    const sp = Math.sin(orbit.pol);
    cam.position.set(orbit.tx + orbit.d * sp * Math.sin(orbit.az), orbit.ty + orbit.d * Math.cos(orbit.pol), orbit.tz + orbit.d * sp * Math.cos(orbit.az));
    cam.lookAt(orbit.tx, orbit.ty, orbit.tz);
  }

  /* ---------- 每帧更新场景 ---------- */
  let clock = 0;
  function updateScene(dt) {
    clock += dt; syncMeshes();
    const rb = sim.robot;
    bot.g.position.set(rb.x, 0, rb.y); bot.g.rotation.y = -rb.th;
    let tj = sim.arm.j, tg = sim.arm.grip;
    if (mode === 'anatomy') { const a = anatTarget(dt); tj = a.j; tg = a.g; }
    const k = mode === 'anatomy' ? 1 - Math.exp(-dt * 9) : 1;
    for (let i = 0; i < 3; i++) dj[i] += (tj[i] - dj[i]) * k; dgrip += (tg - dgrip) * k;
    bot.sh.rotation.z = dj[0]; bot.el.rotation.z = dj[1] - dj[0]; bot.wrg.rotation.z = dj[2] - dj[1];
    bot.fL.position.z = -(0.007 + 0.032 * dgrip); bot.fR.position.z = 0.007 + 0.032 * dgrip;
    bot.wl.rotation.z = -rb.wl; bot.wr.rotation.z = -rb.wr; if (!reduce) bot.lidar.rotation.y -= dt * 14;
    const st = sim.state, blink = (clock % 3.6) < 0.12 ? 0.15 : 1, ey = st === 'DETECT' ? 1.4 : (st === 'GRASP' || st === 'PLACE' || st === 'ALIGN' || st === 'ALIGNP') ? 0.5 : 1;
    bot.eyes.forEach(e => { e.scale.y = blink * ey; });
    bot.ledM.color.setHex(st === 'IDLE' ? 0x8fe3a1 : st === 'WAIT' ? 0xffd166 : 0xff6b2c);
    if (mode === 'anatomy') { $('jA').textContent = deg(dj[0]); $('jB').textContent = deg(dj[1] - dj[0]); $('jC').textContent = deg(dj[2] - dj[1]); }

    for (const [o, m] of obsMesh) { m.position.x = o.x; m.position.z = o.y; if (m.userData.drop > 0) { m.userData.drop = Math.max(0, m.userData.drop - dt * 3.2); m.position.y = m.userData.drop * m.userData.drop * 0.55; } }
    for (const [o, m] of objMesh) { const p = sim.objPos(o, dj); m.position.set(p.x, p.z, p.y); m.rotation.y = o.held ? -rb.th : o.rot; }

    const L = sim.lidar; rays.visible = hits.visible = show.lidar; fov.visible = show.fov && mode === 'sim';
    if (show.lidar) {
      for (let i = 0; i < NR; i++) {
        const a = L.a0 + i * 2 * PI / NR, x = L.ox + Math.cos(a) * L.t[i], y = L.oy + Math.sin(a) * L.t[i], q = i * 6;
        rayPos[q] = L.ox; rayPos[q + 1] = 0.069; rayPos[q + 2] = L.oy; rayPos[q + 3] = x; rayPos[q + 4] = 0.069; rayPos[q + 5] = y;
        hitPos[i * 3] = x; hitPos[i * 3 + 1] = L.hit[i] ? 0.069 : -5; hitPos[i * 3 + 2] = y;
      }
      rayGeo.attributes.position.needsUpdate = true; hitGeo.attributes.position.needsUpdate = true;
    }
    if (fov.visible) {
      const pts = sim.fovPoly(FN); fovPos[0] = rb.x; fovPos[1] = 0.006; fovPos[2] = rb.y;
      for (let i = 0; i <= FN; i++) { fovPos[(i + 1) * 3] = pts[i].x; fovPos[(i + 1) * 3 + 1] = 0.006; fovPos[(i + 1) * 3 + 2] = pts[i].y; }
      fovGeo.attributes.position.needsUpdate = true;
    }
  }

  /* ---------- 叠加层：检测框、路径、标注 ---------- */
  const v3 = new THREE.Vector3(), FONT = '500 12px "Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif';
  function proj(x, up, y, camera, r) { v3.set(x, up, y).project(camera); return { x: r.x + (v3.x * 0.5 + 0.5) * r.w, y: r.y + (0.5 - v3.y * 0.5) * r.h, z: v3.z }; }
  function bbox(o, camera, r) {
    const p = sim.objPos(o, dj), e = Math.max(o.cls.r, 0.03) + 0.01, hh = o.h / 2 + 0.008; let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (let a = -1; a <= 1; a += 2) for (let b = -1; b <= 1; b += 2) for (let c = -1; c <= 1; c += 2) {
      const q = proj(p.x + a * e, p.z + c * hh, p.y + b * e, camera, r); if (q.z > 1 || q.z < -1) return null;
      if (q.x < x0) x0 = q.x; if (q.x > x1) x1 = q.x; if (q.y < y0) y0 = q.y; if (q.y > y1) y1 = q.y;
    }
    return { x0: x0 - 3, y0: y0 - 3, x1: x1 + 3, y1: y1 + 3 };
  }
  function bracket(b, col, lw) {
    const l = Math.max(4, Math.min(10, (b.x1 - b.x0) / 3, (b.y1 - b.y0) / 3)); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath();
    ctx.moveTo(b.x0, b.y0 + l); ctx.lineTo(b.x0, b.y0); ctx.lineTo(b.x0 + l, b.y0); ctx.moveTo(b.x1 - l, b.y0); ctx.lineTo(b.x1, b.y0); ctx.lineTo(b.x1, b.y0 + l);
    ctx.moveTo(b.x1, b.y1 - l); ctx.lineTo(b.x1, b.y1); ctx.lineTo(b.x1 - l, b.y1); ctx.moveTo(b.x0 + l, b.y1); ctx.lineTo(b.x0, b.y1); ctx.lineTo(b.x0, b.y1 - l); ctx.stroke();
  }
  function tag(x, y, text, bg, fg) { ctx.font = FONT; const w = ctx.measureText(text).width + 10; ctx.fillStyle = bg; ctx.fillRect(x, y - 17, w, 17); ctx.fillStyle = fg; ctx.textBaseline = 'middle'; ctx.fillText(text, x + 5, y - 8); }
  const PARTS = [
    ['机械臂', '肩、肘、腕三个俯仰关节，加腕部旋转。臂展 32 cm', () => bot.el.getWorldPosition(v3)],
    ['夹爪和腕部相机', '平行夹爪开口 7.5 cm。相机负责最后 10 cm 的对准', () => bot.wrg.localToWorld(v3.set(0.05, 0, 0))],
    ['主相机和表情灯', '前向 RGB-D，视场约 96°。两格灯会眨眼、会眯眼', () => bot.g.localToWorld(v3.set(0.106, 0.105, 0))],
    ['激光雷达', '夹在上下层板之间，360° 扫描，离地 7 cm', () => bot.g.localToWorld(v3.set(-0.03, 0.069, 0.07))],
    ['差速驱动轮', '两个带编码器的减速电机，前后各一个万向球', () => bot.g.localToWorld(v3.set(0, 0.034, 0.107))],
    ['电池和主控', '3S 锂电约 29 Wh，放在后部当配重', () => bot.g.localToWorld(v3.set(-0.085, 0.035, -0.04))],
    ['防跌落和碰撞环', '三个下视 ToF 防掉台阶，碰撞环兜底', () => bot.g.localToWorld(v3.set(0.098, 0.03, -0.03))],
    ['麦克风和状态灯', '听指令，也听东西掉地上的声音', () => bot.g.localToWorld(v3.set(-0.085, 0.127, 0))]
  ];
  $('parts').innerHTML = PARTS.map((p, i) => `<li data-i="${i}"><span class="n">${i + 1}</span><span><b>${p[0]}</b><small>${p[1]}</small></span></li>`).join('');
  $('parts').addEventListener('pointerover', e => { const li = e.target.closest('li'); hiPart = li ? +li.dataset.i : -1; });
  $('parts').addEventListener('pointerleave', () => { hiPart = -1; });

  function drawOverlay() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, sw, sh);
    const full = { x: 0, y: 0, w: sw, h: sh }, rb = sim.robot;
    if (mode === 'anatomy') {
      PARTS.forEach((p, i) => {
        p[2](); const q = proj(v3.x, v3.y, v3.z, cam, full), hi = i === hiPart, r = hi ? 13 : 10;
        if (hi) { ctx.strokeStyle = '#fff3c4'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(q.x, q.y, r + 4, 0, 2 * PI); ctx.stroke(); }
        ctx.fillStyle = '#ff6b2c'; ctx.beginPath(); ctx.arc(q.x, q.y, r, 0, 2 * PI); ctx.fill();
        ctx.fillStyle = '#1c0c03'; ctx.font = '500 12px ui-monospace,Menlo,monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(i + 1), q.x, q.y + 0.5); ctx.textAlign = 'left';
      });
      return;
    }
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, sw, sh); ctx.rect(pip.x, pip.y - 22, pip.w, pip.h + 22); ctx.clip('evenodd');
    const P = sim.path;
    if (show.path && P && P.length > 1 && ['PLAN', 'APPROACH', 'CARRY', 'EXPLORE', 'DOCK'].indexOf(sim.state) >= 0) {
      ctx.strokeStyle = '#ff6b2c'; ctx.lineWidth = 2.5; ctx.setLineDash([7, 5]); ctx.lineJoin = 'round'; ctx.beginPath();
      let q = proj(rb.x, 0.01, rb.y, cam, full); ctx.moveTo(q.x, q.y);
      for (let i = Math.min(sim.wp, P.length - 1); i < P.length; i++) { q = proj(P[i].x, 0.01, P[i].y, cam, full); ctx.lineTo(q.x, q.y); }
      ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = '#ff6b2c'; ctx.beginPath(); ctx.arc(q.x, q.y, 4, 0, 2 * PI); ctx.fill();
    }
    const b = sim.basket; let q = proj(b.x, 0.16, b.y, cam, full); tag(q.x - 22, q.y, '收纳筐', 'rgba(12,34,30,.78)', '#eef0e4');
    q = proj(HOME.x, 0.12, 2.93, cam, full); tag(q.x - 22, q.y, '充电座', 'rgba(12,34,30,.78)', '#eef0e4');
    for (const o of sim.objects) {
      if (o.delivered) continue; const known = sim.memory.get(o.id), conf = sim.live.get(o.id), isT = o === sim.target || o.held;
      if (!known && !conf) continue;
      const bb = bbox(o, cam, full); if (!bb) continue;
      if (isT) { bracket(bb, '#ff6b2c', 2.5); tag(bb.x0, bb.y0 - 3, (o.held ? '已夹住 ' : '目标 ') + o.cls.name, '#ff6b2c', '#1c0c03'); }
      else if (conf) { bracket(bb, '#fff3c4', 2); tag(bb.x0, bb.y0 - 3, o.cls.name + ' ' + conf.toFixed(2), '#fff3c4', '#1b2b27'); }
      else { bracket(bb, 'rgba(255,243,196,.5)', 1.5); }
    }
    ctx.restore();
    // 画中画：主相机里的检测框
    ctx.save(); ctx.beginPath(); ctx.rect(pip.x, pip.y, pip.w, pip.h); ctx.clip();
    for (const o of sim.objects) {
      const conf = sim.live.get(o.id); if (!conf) continue; const bb = bbox(o, bot.eye, pip); if (!bb) continue;
      ctx.strokeStyle = o === sim.target ? '#ff6b2c' : '#fff3c4'; ctx.lineWidth = 1.5; ctx.strokeRect(bb.x0, bb.y0, bb.x1 - bb.x0, bb.y1 - bb.y0);
      ctx.font = '500 10px "Noto Sans SC",sans-serif'; ctx.fillStyle = ctx.strokeStyle; ctx.textBaseline = 'bottom'; ctx.fillText(o.cls.name + ' ' + conf.toFixed(2), bb.x0, bb.y0 - 2);
    }
    ctx.restore();
  }

  /* ---------- 小地图 ---------- */
  const mapc = $('map'), mctx = mapc.getContext('2d'), gimg = document.createElement('canvas'); gimg.width = GX; gimg.height = GY;
  const gctx = gimg.getContext('2d'), gdata = gctx.createImageData(GX, GY);
  const OCOL = { sock: '#e2483d', block: '#4f9de0', ball: '#d6e84a', remote: '#dcd6ea', car: '#f2b02e' };
  function drawMap() {
    const cw = mapc.clientWidth; if (!cw) return; const ch = cw * 0.75, pw = Math.round(cw * dpr), ph = Math.round(ch * dpr);
    if (mapc.width !== pw || mapc.height !== ph) { mapc.width = pw; mapc.height = ph; }
    const k = cw / W, lo = sim.lo, d = gdata.data, rb = sim.robot;
    for (let i = 0; i < GX * GY; i++) { const v = lo[i], q = i * 4; if (v > 0) { d[q] = 241; d[q + 1] = 238; d[q + 2] = 226; } else if (v < 0) { d[q] = 38; d[q + 1] = 98; d[q + 2] = 79; } else { d[q] = 13; d[q + 1] = 36; d[q + 2] = 32; } d[q + 3] = 255; }
    gctx.putImageData(gdata, 0, 0);
    mctx.setTransform(dpr, 0, 0, dpr, 0, 0); mctx.imageSmoothingEnabled = false; mctx.drawImage(gimg, 0, 0, cw, ch);
    const f = sim.fovPoly(14); mctx.fillStyle = 'rgba(255,243,196,.2)'; mctx.beginPath(); mctx.moveTo(rb.x * k, rb.y * k); f.forEach(p => mctx.lineTo(p.x * k, p.y * k)); mctx.closePath(); mctx.fill();
    const L = sim.lidar; mctx.fillStyle = '#ffd166';
    for (let i = 0; i < NR; i++) if (L.hit[i]) { const a = L.a0 + i * 2 * PI / NR; mctx.fillRect((L.ox + Math.cos(a) * L.t[i]) * k - 1, (L.oy + Math.sin(a) * L.t[i]) * k - 1, 2, 2); }
    const b = sim.basket; mctx.strokeStyle = '#ff6b2c'; mctx.lineWidth = 1.5; mctx.strokeRect((b.x - b.w / 2) * k, (b.y - b.h / 2) * k, b.w * k, b.h * k);
    const P = sim.path;
    if (P && P.length > 1 && ['PLAN', 'APPROACH', 'CARRY', 'EXPLORE', 'DOCK'].indexOf(sim.state) >= 0) { mctx.strokeStyle = '#ff6b2c'; mctx.lineWidth = 2; mctx.beginPath(); mctx.moveTo(rb.x * k, rb.y * k); for (let i = Math.min(sim.wp, P.length - 1); i < P.length; i++) mctx.lineTo(P[i].x * k, P[i].y * k); mctx.stroke(); }
    for (const o of sim.objects) {
      if (o.delivered || o.held || !sim.memory.has(o.id)) continue;
      mctx.fillStyle = OCOL[o.cls.key]; mctx.beginPath(); mctx.arc(o.x * k, o.y * k, 4, 0, 2 * PI); mctx.fill(); mctx.strokeStyle = '#0d2420'; mctx.lineWidth = 1; mctx.stroke();
      if (o === sim.target) { mctx.strokeStyle = '#ff6b2c'; mctx.lineWidth = 1.5; mctx.beginPath(); mctx.arc(o.x * k, o.y * k, 8, 0, 2 * PI); mctx.stroke(); }
    }
    mctx.fillStyle = '#f1eee2'; mctx.beginPath(); mctx.arc(rb.x * k, rb.y * k, 0.1 * k, 0, 2 * PI); mctx.fill();
    mctx.strokeStyle = '#ff6b2c'; mctx.lineWidth = 2; mctx.beginPath(); mctx.moveTo(rb.x * k, rb.y * k); mctx.lineTo((rb.x + Math.cos(rb.th) * 0.14) * k, (rb.y + Math.sin(rb.th) * 0.14) * k); mctx.stroke();
  }

  /* ---------- 文字状态 ---------- */
  const ST = {
    BOOT: ['巡视', '开机环顾，雷达建图'], SCAN: ['巡视', '原地环顾，相机搜索'], EXPLORE: ['巡视', '前往还没看过的区域'], DETECT: ['识别', '发现新物体'],
    PLAN: ['规划', 'A* 规划路径'], APPROACH: ['导航', '导航接近目标'], ALIGN: ['抓取', '转向对准目标'], GRASP: ['抓取', '机械臂下探、夹取、抬起'],
    CARRY: ['搬运', '带着物体去收纳筐'], ALIGNP: ['放置', '对准收纳筐'], PLACE: ['放置', '伸臂、松爪、收臂'], DOCK: ['回充', '返回充电座'], IDLE: ['回充', '在充电座待命，等新任务'], WAIT: ['导航', '路被堵住，等待重试']
  };
  const chipEls = Array.prototype.slice.call($('chips').children), stMain = $('stMain'), stSub = $('stSub'), statsEl = $('stats'), logEl = $('log');
  let lastLogSeq = -1, lastHud = '';
  function fmtT(t) { const m = t / 60 | 0, s = Math.floor(t % 60); return (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s; }
  function updateHud() {
    if (mode === 'anatomy') { if (lastHud !== 'A') { lastHud = 'A'; stMain.textContent = '结构拆解'; stSub.textContent = '8 个部件，7 个电机'; stSub.className = ''; statsEl.innerHTML = ''; } return; }
    const s = ST[sim.state], rb = sim.robot, o = sim.held || sim.target; let sub = s[1];
    if (o && (sim.state === 'APPROACH' || sim.state === 'PLAN' || sim.state === 'ALIGN')) { const m = sim.memory.get(o.id); sub += '：' + o.cls.name + ' · ' + Math.hypot(o.x - rb.x, o.y - rb.y).toFixed(2) + ' m' + (m && m.conf ? ' · 置信度 ' + m.conf.toFixed(2) : ' · App 指定'); }
    else if (o && sim.held) sub += '：' + o.cls.name + ' ' + o.cls.mass + ' g';
    const alert = sim.avoidFlash > 0; if (alert) sub = '路径上发现障碍，重新规划';
    const key = s[0] + sub + alert; if (key !== lastHud) { lastHud = key; stMain.textContent = s[0]; stSub.textContent = sub; stSub.className = alert ? 'alert' : ''; chipEls.forEach(c => { c.className = c.textContent === s[0] ? 'on' : (alert && c.textContent === '导航' ? 'flash' : ''); }); }
    statsEl.innerHTML = '已归位 <b>' + sim.delivered + '/' + sim.objects.length + '</b><br>里程 <b>' + rb.odo.toFixed(1) + ' m</b><br>用时 <b>' + fmtT(sim.t) + '</b>';
    if (sim.logSeq !== lastLogSeq) { lastLogSeq = sim.logSeq; logEl.innerHTML = sim.logs.slice(-7).reverse().map(l => '<li class="' + l.kind + '"><time>' + fmtT(l.t) + '</time><span>' + l.msg + '</span></li>').join(''); }
  }

  /* ---------- 交互 ---------- */
  const toastEl = $('toast'); let toastT = 0;
  function toast(msg) { toastEl.textContent = msg; toastEl.classList.add('show'); clearTimeout(toastT); toastT = setTimeout(() => toastEl.classList.remove('show'), 2400); }
  function seg(id, attr, fn) { const el = $(id); el.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; Array.prototype.forEach.call(el.children, c => c.classList.toggle('on', c === b)); fn(b.dataset[attr]); }); }
  seg('modeSeg', 'mode', v => { mode = v; document.body.dataset.mode = v; lastHud = ''; orbit.zoom = 1; if (v === 'anatomy') { autoRot = true; armAuto = true; } resize(); });
  seg('toolSeg', 'tool', v => { tool = v; }); seg('viewSeg', 'view', v => { view = v; orbit.zoom = 1; }); seg('speedSeg', 'speed', v => { speed = +v; });
  const btnPlay = $('btnPlay'); function setPlay(p) { playing = p; btnPlay.textContent = p ? '暂停' : '继续'; } setPlay(playing);
  btnPlay.addEventListener('click', () => setPlay(!playing));
  $('btnThrow').addEventListener('click', () => { const r = sim.throwObject(); toast(r.msg); if (!playing) setPlay(true); });
  $('btnReset').addEventListener('click', () => { sim.reset(); for (let i = 0; i < 60; i++) sim.step(DT); lastLogSeq = -1; toast('房间已重置'); });
  Array.prototype.forEach.call(document.querySelectorAll('[data-show]'), b => b.addEventListener('click', () => { const k = b.dataset.show; show[k] = !show[k]; b.classList.toggle('on', show[k]); }));
  [slR, slH, slG].forEach(s => s.addEventListener('input', () => { armAuto = false; }));
  $('btnDemo').addEventListener('click', () => { armAuto = true; });
  $('zin').addEventListener('click', () => { orbit.zoom = clamp(orbit.zoom * 0.8, 0.4, 2.4); }); $('zout').addEventListener('click', () => { orbit.zoom = clamp(orbit.zoom * 1.25, 0.4, 2.4); });

  const ptrs = new Map(); let moved = 0, pinch = 0;
  function pdist() { const a = Array.from(ptrs.values()); return Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y); }
  glc.addEventListener('pointerdown', e => { try { glc.setPointerCapture(e.pointerId); } catch (err) { } ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY }); if (ptrs.size === 1) moved = 0; else { moved = 99; pinch = pdist(); } });
  glc.addEventListener('pointermove', e => {
    const p = ptrs.get(e.pointerId); if (!p) return; const dx = e.clientX - p.x, dy = e.clientY - p.y; p.x = e.clientX; p.y = e.clientY;
    if (ptrs.size === 1) { moved += Math.abs(dx) + Math.abs(dy); if (moved > 5) { orbit.az -= dx * 0.006; orbit.pol = clamp(orbit.pol - dy * 0.005, 0.2, 1.45); autoRot = false; } }
    else if (ptrs.size === 2) { const d = pdist(); if (pinch > 0 && d > 0) orbit.zoom = clamp(orbit.zoom * pinch / d, 0.4, 2.4); pinch = d; }
  });
  function pup(e) { if (!ptrs.has(e.pointerId)) return; ptrs.delete(e.pointerId); if (e.type === 'pointerup' && moved <= 5 && ptrs.size === 0) click(e); }
  glc.addEventListener('pointerup', pup); glc.addEventListener('pointercancel', pup);
  glc.addEventListener('wheel', e => { if (!(e.ctrlKey || e.metaKey)) return; e.preventDefault(); orbit.zoom = clamp(orbit.zoom * Math.exp(e.deltaY * 0.01), 0.4, 2.4); }, { passive: false });
  function click(e) {
    if (mode !== 'sim') return; const r = glc.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, full = { x: 0, y: 0, w: sw, h: sh };
    let best = null, bd = 26;
    for (const o of sim.objects) { if (o.delivered || o.held) continue; const q = proj(o.x, o.h / 2, o.y, cam, full), d = Math.hypot(q.x - mx, q.y - my); if (d < bd) { bd = d; best = o; } }
    if (best) { toast(sim.prioritize(best.id).msg); return; }
    v3.set(mx / sw * 2 - 1, 1 - my / sh * 2, 0.5).unproject(cam).sub(cam.position); if (v3.y >= -1e-6) return;
    const t = -cam.position.y / v3.y, x = cam.position.x + v3.x * t, y = cam.position.z + v3.z * t;
    if (x < 0 || y < 0 || x > W || y > H) return;
    toast((tool === 'box' ? sim.placeBox(x, y) : sim.moveBasket(x, y)).msg);
  }

  /* ---------- 尺寸与主循环 ---------- */
  function resize() {
    sw = stage.clientWidth || 800; sh = stage.clientHeight || 500; dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(dpr); renderer.setSize(sw, sh, false); cam.aspect = sw / sh; cam.updateProjectionMatrix();
    ovc.width = Math.round(sw * dpr); ovc.height = Math.round(sh * dpr);
    pip.w = Math.round(clamp(sw * 0.25, 112, 230)); pip.h = Math.round(pip.w * 0.75); pip.x = 12; pip.y = sh - 12 - pip.h;
    pipEl.style.width = pip.w + 'px'; pipEl.style.height = pip.h + 'px';
  }
  if (window.ResizeObserver) new ResizeObserver(resize).observe(stage); else window.addEventListener('resize', resize);
  resize();
  let last = performance.now(), acc = 0, fr = 0;
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now; fr++;
    if (mode === 'sim' && playing) { acc += dt * speed; let n = 0; while (acc >= DT && n < 14) { sim.step(DT); acc -= DT; n++; } if (n === 14) acc = 0; }
    camUpdate(dt); updateScene(dt);
    renderer.setScissorTest(false); renderer.setViewport(0, 0, sw, sh); renderer.render(scene, cam);
    if (mode === 'sim') { renderer.setScissorTest(true); renderer.setScissor(pip.x, 12, pip.w, pip.h); renderer.setViewport(pip.x, 12, pip.w, pip.h); renderer.render(scene, bot.eye); renderer.setScissorTest(false); }
    drawOverlay(); if (fr % 2 === 0 && mode === 'sim') drawMap(); if (fr % 4 === 0) updateHud();
    requestAnimationFrame(frame);
  }
  camUpdate(1); updateHud(); requestAnimationFrame(frame);
  window.__xiaoshi = { sim, orbit };
})();
