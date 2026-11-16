/* 小拾 仿真内核：射线雷达 → 占据栅格 → A* → 差速跟踪 → 逆运动学抓取。不依赖渲染。 */
var SimCore = (function () {
  'use strict';
  const W = 4, H = 3, CS = 0.05, GX = 80, GY = 60, CCS = 0.25, CGX = 16, CGY = 12;
  const PASS = 0.149, RANGE = 3.5, CAM_R = 2.3, CAM_HALF = 48 * Math.PI / 180, NRAY = 144;
  const HOME = { x: 0.3, y: 2.64 };
  const S = { x: 0.03, y: 0.145 }, L1 = 0.13, L2 = 0.13, L3 = 0.06;
  const PI = Math.PI;
  const CLASSES = {
    sock: { key: 'sock', name: '袜子', h: 0.03, r: 0.055, gw: 0.034, mass: 30 },
    block: { key: 'block', name: '积木', h: 0.045, r: 0.03, gw: 0.045, mass: 20 },
    ball: { key: 'ball', name: '网球', h: 0.066, r: 0.033, gw: 0.064, mass: 58 },
    remote: { key: 'remote', name: '遥控器', h: 0.02, r: 0.065, gw: 0.042, mass: 90 },
    car: { key: 'car', name: '玩具车', h: 0.04, r: 0.05, gw: 0.046, mass: 60 }
  };
  const CKEYS = Object.keys(CLASSES);
  const OFF = [];
  for (let dy = -7; dy <= 7; dy++) for (let dx = -7; dx <= 7; dx++) { const d = Math.hypot(dx, dy) * CS; if (d <= 0.351) OFF.push([dx, dy, d]); }
  const DX = [1, -1, 0, 0, 1, 1, -1, -1], DY = [0, 0, 1, -1, 1, -1, 1, -1];

  function wrap(a) { while (a > PI) a -= 2 * PI; while (a < -PI) a += 2 * PI; return a; }
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function dRect(px, py, o) { const dx = Math.max(Math.abs(px - o.x) - o.w / 2, 0), dy = Math.max(Math.abs(py - o.y) - o.h / 2, 0); return Math.hypot(dx, dy); }
  function dObs(px, py, o) { return o.type === 'rect' ? dRect(px, py, o) : Math.max(Math.hypot(px - o.x, py - o.y) - o.r, 0); }
  function rayRect(ox, oy, dx, dy, o) {
    const x0 = o.x - o.w / 2, x1 = o.x + o.w / 2, y0 = o.y - o.h / 2, y1 = o.y + o.h / 2;
    let tmin = -Infinity, tmax = Infinity;
    if (Math.abs(dx) < 1e-9) { if (ox < x0 || ox > x1) return Infinity; }
    else { let a = (x0 - ox) / dx, b = (x1 - ox) / dx; if (a > b) { const t = a; a = b; b = t; } if (a > tmin) tmin = a; if (b < tmax) tmax = b; }
    if (Math.abs(dy) < 1e-9) { if (oy < y0 || oy > y1) return Infinity; }
    else { let a = (y0 - oy) / dy, b = (y1 - oy) / dy; if (a > b) { const t = a; a = b; b = t; } if (a > tmin) tmin = a; if (b < tmax) tmax = b; }
    if (tmax < tmin || tmax < 0) return Infinity;
    return tmin > 0 ? tmin : 0;
  }
  function rayCirc(ox, oy, dx, dy, o) {
    const fx = ox - o.x, fy = oy - o.y, b = fx * dx + fy * dy, c = fx * fx + fy * fy - o.r * o.r, disc = b * b - c;
    if (disc < 0) return Infinity; const s = Math.sqrt(disc), t = -b - s;
    if (t > 0) return t; if (-b + s > 0) return 0; return Infinity;
  }
  // 逆运动学：给定指尖到机器人中心的水平距离 r、高度 h、夹爪俯仰 phi，解出三个关节的绝对角
  function ik(r, h, phi) {
    let wx = r - L3 * Math.cos(phi) - S.x, wy = h - L3 * Math.sin(phi) - S.y;
    let d = Math.hypot(wx, wy); const dmax = L1 + L2 - 0.004;
    if (d > dmax) { wx *= dmax / d; wy *= dmax / d; d = dmax; }
    if (d < 0.03) { wx = 0.03; wy = 0; d = 0.03; }
    const a = Math.atan2(wy, wx) + Math.acos(clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1));
    const ex = L1 * Math.cos(a), ey = L1 * Math.sin(a);
    return [a, Math.atan2(wy - ey, wx - ex), phi];
  }
  function fk(j) {
    const ex = S.x + L1 * Math.cos(j[0]), ey = S.y + L1 * Math.sin(j[0]);
    const wx = ex + L2 * Math.cos(j[1]), wy = ey + L2 * Math.sin(j[1]);
    return { ex, ey, wx, wy, tx: wx + L3 * Math.cos(j[2]), ty: wy + L3 * Math.sin(j[2]) };
  }
  const J_STOW = ik(0.085, 0.15, -1.4), J_CARRY = ik(0.11, 0.2, -PI / 2);

  function furniture() {
    return [
      { type: 'rect', kind: 'sofa', x: 0.95, y: 0.40, w: 1.5, h: 0.6 },
      { type: 'rect', kind: 'table', x: 1.0, y: 1.45, w: 0.8, h: 0.45 },
      { type: 'rect', kind: 'shelf', x: 2.6, y: 0.2, w: 0.9, h: 0.3 },
      { type: 'circ', kind: 'plant', x: 3.55, y: 0.45, r: 0.17 },
      { type: 'rect', kind: 'chair', x: 3.0, y: 1.9, w: 0.42, h: 0.42 },
      { type: 'rect', kind: 'box', x: 1.9, y: 2.3, w: 0.36, h: 0.3 },
      { type: 'rect', kind: 'dock', x: 0.3, y: 2.93, w: 0.24, h: 0.12 },
      { type: 'rect', kind: 'basket', x: 3.45, y: 2.55, w: 0.26, h: 0.18 }
    ];
  }

  class Sim {
    constructor() {
      const n = GX * GY;
      this._g = new Float32Array(n); this._came = new Int32Array(n); this._st = new Uint32Array(n); this._cl = new Uint32Array(n); this._gen = 0;
      this._hf = new Float64Array(60000); this._hi = new Int32Array(60000);
      this.reset();
    }
    reset() {
      this.t = 0; this.obstacles = furniture(); this.basket = this.obstacles.find(o => o.kind === 'basket');
      this.objects = []; this.nextId = 1; this.userBoxes = 0;
      this.robot = { x: HOME.x, y: HOME.y, th: -PI / 2, v: 0, w: 0, wl: 0, wr: 0, odo: 0 };
      this.lo = new Int8Array(GX * GY); this.cov = new Uint8Array(CGX * CGY); this.dist = new Float32Array(GX * GY).fill(0.5);
      this.memory = new Map(); this.live = new Map();
      this.lidar = { n: NRAY, t: new Float32Array(NRAY), hit: new Uint8Array(NRAY), ox: HOME.x, oy: HOME.y, a0: 0 };
      this.path = null; this.wp = 1; this.replanT = 0; this.fail = 0; this.lastAvoid = -9; this.avoidFlash = 0;
      this.state = 'BOOT'; this.stateT = 0; this.phase = 0; this.next = null; this.target = null; this.held = null; this.priority = null;
      this.pendingDetect = null; this.spin = 0; this.exT = -1; this.exC = null; this.senseT = 0; this.idleT = 0; this.backup = 0; this.bump = 0;
      this.vt = 0; this.wt = 0; this.gr = 0.2; this.ght = 0.02; this.pr = 0.28;
      this.arm = { j: J_STOW.slice(), from: J_STOW.slice(), to: J_STOW.slice(), grip: 0.35, gf: 0.35, gt: 0.35, p: 1, dur: 1 };
      this.logs = []; this.logSeq = 0; this.delivered = 0;
      const init = [['sock', 0.35, 1.02], ['block', 1.78, 1.0], ['ball', 2.55, 1.25], ['remote', 2.45, 2.72], ['car', 3.6, 1.25]];
      for (const [k, x, y] of init) this.addObject(k, x, y);
      this.log('开机：原地环顾，雷达开始建图', 'idle');
      this.sense();
    }
    addObject(k, x, y) {
      const c = CLASSES[k];
      const o = { id: this.nextId++, cls: c, x, y, h: c.h, rot: Math.random() * PI, held: false, delivered: false, skipUntil: 0, skipN: 0, slot: null, anim: null };
      this.objects.push(o); return o;
    }
    log(msg, kind) { const l = this.logs[this.logs.length - 1]; if (l && l.msg === msg && this.t - l.t < 90) return; this.logs.push({ t: this.t, msg, kind: kind || '' }); if (this.logs.length > 40) this.logs.shift(); this.logSeq++; }
    set(s) { this.state = s; this.stateT = 0; this.phase = 0; this.replanT = 0; this.fail = 0; }

    /* ---------- 感知 ---------- */
    castRay(ox, oy, ang, maxR) {
      const dx = Math.cos(ang), dy = Math.sin(ang);
      let t = maxR, kind = 0;
      const tx = dx > 1e-9 ? (W - ox) / dx : dx < -1e-9 ? -ox / dx : Infinity, ty = dy > 1e-9 ? (H - oy) / dy : dy < -1e-9 ? -oy / dy : Infinity;
      const tw = Math.min(tx, ty); if (tw < t) { t = tw; kind = 2; }
      const O = this.obstacles;
      for (let i = 0; i < O.length; i++) { const o = O[i]; const tt = o.type === 'rect' ? rayRect(ox, oy, dx, dy, o) : rayCirc(ox, oy, dx, dy, o); if (tt < t) { t = tt; kind = 1; } }
      this._rk = kind; return t;
    }
    los(ax, ay, bx, by) { const d = Math.hypot(bx - ax, by - ay); if (d < 1e-6) return true; return this.castRay(ax, ay, Math.atan2(by - ay, bx - ax), d + 0.01) >= d - 0.012; }
    sense() {
      const rb = this.robot, L = this.lidar, lo = this.lo;
      L.ox = rb.x; L.oy = rb.y; L.a0 = rb.th;
      for (let i = 0; i < NRAY; i++) {
        const ang = rb.th + i * 2 * PI / NRAY, dx = Math.cos(ang), dy = Math.sin(ang);
        const t = this.castRay(rb.x, rb.y, ang, RANGE), k = this._rk;
        L.t[i] = t; L.hit[i] = k;
        let last = -1;
        for (let s = 0.02; s < t - 0.02; s += 0.03) {
          const cx = (rb.x + dx * s) / CS | 0, cy = (rb.y + dy * s) / CS | 0;
          if (cx < 0 || cy < 0 || cx >= GX || cy >= GY) break;
          const idx = cy * GX + cx; if (idx !== last) { last = idx; if (lo[idx] > -3) lo[idx]--; }
        }
        if (k === 1) {
          const cx = Math.floor((rb.x + dx * (t + 0.012)) / CS), cy = Math.floor((rb.y + dy * (t + 0.012)) / CS);
          if (cx >= 0 && cy >= 0 && cx < GX && cy < GY) { const idx = cy * GX + cx; lo[idx] = Math.min(12, lo[idx] + 6); }
        }
      }
      // 相机：覆盖记录 + 物体识别
      const cov = this.cov;
      for (let j = 0; j < CGY; j++) for (let i = 0; i < CGX; i++) {
        const q = j * CGX + i; if (cov[q]) continue;
        const cx = (i + 0.5) * CCS, cy = (j + 0.5) * CCS, d = Math.hypot(cx - rb.x, cy - rb.y);
        if (d < 0.3 || (d < CAM_R && Math.abs(wrap(Math.atan2(cy - rb.y, cx - rb.x) - rb.th)) < CAM_HALF && this.los(rb.x, rb.y, cx, cy))) cov[q] = 1;
      }
      this.live.clear();
      for (const o of this.objects) {
        if (o.held || o.delivered) continue;
        const d = Math.hypot(o.x - rb.x, o.y - rb.y);
        if (d > CAM_R || d < 0.11) continue;
        if (Math.abs(wrap(Math.atan2(o.y - rb.y, o.x - rb.x) - rb.th)) > CAM_HALF) continue;
        if (!this.los(rb.x, rb.y, o.x, o.y)) continue;
        const conf = clamp(0.985 - 0.33 * Math.pow(d / CAM_R, 1.3) + (Math.random() - 0.5) * 0.02, 0.5, 0.99);
        this.live.set(o.id, conf);
        const m = this.memory.get(o.id);
        if (!m || m.conf === null) {
          this.memory.set(o.id, { conf, src: 'cam' });
          if (!m) {
            this.log(`识别到 ${o.cls.name}（置信度 ${conf.toFixed(2)}）`, 'detect');
            if (this.state === 'BOOT' || this.state === 'SCAN' || this.state === 'EXPLORE' || this.state === 'DOCK' || this.state === 'IDLE') this.pendingDetect = o;
          }
        } else if (conf > m.conf) m.conf = conf;
      }
    }
    fovPoly(n) {
      const rb = this.robot, pts = [];
      for (let i = 0; i <= n; i++) { const a = rb.th - CAM_HALF + 2 * CAM_HALF * i / n; const t = this.castRay(rb.x, rb.y, a, CAM_R); pts.push({ x: rb.x + Math.cos(a) * t, y: rb.y + Math.sin(a) * t }); }
      return pts;
    }

    /* ---------- 代价地图 + A* ---------- */
    computeDist(ign) {
      const D = this.dist, lo = this.lo; D.fill(0.5);
      const stamp = (cx, cy) => { for (let k = 0; k < OFF.length; k++) { const o = OFF[k], x = cx + o[0], y = cy + o[1]; if (x < 0 || y < 0 || x >= GX || y >= GY) continue; const q = y * GX + x; if (o[2] < D[q]) D[q] = o[2]; } };
      for (let cy = 0; cy < GY; cy++) for (let cx = 0; cx < GX; cx++) if (lo[cy * GX + cx] > 0) stamp(cx, cy);
      for (const o of this.objects) { if (o.held || o.delivered || o.id === ign || !this.memory.has(o.id)) continue; stamp(clamp(o.x / CS | 0, 0, GX - 1), clamp(o.y / CS | 0, 0, GY - 1)); }
      for (let cy = 0; cy < GY; cy++) for (let cx = 0; cx < GX; cx++) { const x = (cx + 0.5) * CS, y = (cy + 0.5) * CS, wd = Math.min(x, W - x, y, H - y), q = cy * GX + cx; if (wd < D[q]) D[q] = wd; }
    }
    dAt(x, y) { const cx = Math.floor(x / CS), cy = Math.floor(y / CS); if (cx < 0 || cy < 0 || cx >= GX || cy >= GY) return 0; return this.dist[cy * GX + cx]; }
    _enter(a, b, near) { const D = this.dist; if (D[b] >= PASS) return true; return near && D[a] < PASS && D[b] >= D[a] - 1e-6 && D[b] > 0.06; }
    astar(gx, gy, R) {
      const D = this.dist, rb = this.robot;
      const sx = clamp(Math.floor(rb.x / CS), 0, GX - 1), sy = clamp(Math.floor(rb.y / CS), 0, GY - 1), s = sy * GX + sx;
      const gen = ++this._gen, G = this._g, C = this._came, ST = this._st, CL = this._cl, HF = this._hf, HI = this._hi;
      let hn = 0;
      const push = (f, i) => { let k = hn++; HF[k] = f; HI[k] = i; while (k > 0) { const p = (k - 1) >> 1; if (HF[p] <= HF[k]) break; const tf = HF[p], ti = HI[p]; HF[p] = HF[k]; HI[p] = HI[k]; HF[k] = tf; HI[k] = ti; k = p; } };
      const pop = () => { const top = HI[0]; hn--; if (hn > 0) { HF[0] = HF[hn]; HI[0] = HI[hn]; let k = 0; for (; ;) { const l = 2 * k + 1, r = l + 1; let m = k; if (l < hn && HF[l] < HF[m]) m = l; if (r < hn && HF[r] < HF[m]) m = r; if (m === k) break; const tf = HF[m], ti = HI[m]; HF[m] = HF[k]; HI[m] = HI[k]; HF[k] = tf; HI[k] = ti; k = m; } } return top; };
      const hf = i => Math.max(0, Math.hypot((i % GX + 0.5) * CS - gx, ((i / GX | 0) + 0.5) * CS - gy) - R);
      G[s] = 0; ST[s] = gen; C[s] = -1; push(hf(s), s);
      let found = -1;
      while (hn > 0 && hn < 59000) {
        const c = pop(); if (CL[c] === gen) continue; CL[c] = gen;
        const cx = c % GX, cy = c / GX | 0, wx = (cx + 0.5) * CS, wy = (cy + 0.5) * CS;
        if ((wx - gx) * (wx - gx) + (wy - gy) * (wy - gy) <= R * R && D[c] >= PASS) { found = c; break; }
        const near = Math.abs(cx - sx) <= 7 && Math.abs(cy - sy) <= 7;
        for (let k = 0; k < 8; k++) {
          const nx = cx + DX[k], ny = cy + DY[k]; if (nx < 0 || ny < 0 || nx >= GX || ny >= GY) continue;
          const n = ny * GX + nx; if (CL[n] === gen) continue;
          if (!this._enter(c, n, near)) continue;
          if (k >= 4 && (!this._enter(c, cy * GX + nx, near) || !this._enter(c, ny * GX + cx, near))) continue;
          const dn = D[n];
          const ng = G[c] + (k < 4 ? CS : CS * 1.4142) * (1 + (dn < 0.3 ? (0.3 - dn) / 0.15 * 2.5 : 0) + (dn < PASS ? 6 : 0));
          if (ST[n] !== gen || ng < G[n]) { G[n] = ng; ST[n] = gen; C[n] = c; push(ng + hf(n), n); }
        }
      }
      if (found < 0) return null;
      const pts = [];
      for (let i = found; i !== -1; i = C[i]) pts.push({ x: (i % GX + 0.5) * CS, y: ((i / GX | 0) + 0.5) * CS });
      pts.reverse(); pts[0] = { x: rb.x, y: rb.y };
      return this.smooth(pts);
    }
    clearSeg(a, b) {
      const da = this.dAt(a.x, a.y), db = this.dAt(b.x, b.y); if (da < PASS || db < PASS) return false;
      const need = Math.min(0.2, da, db) - 1e-4, L = Math.hypot(b.x - a.x, b.y - a.y), n = Math.ceil(L / 0.025);
      for (let i = 1; i < n; i++) { const u = i / n; if (this.dAt(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u) < need) return false; }
      return true;
    }
    smooth(pts) {
      const out = [pts[0]]; let i = 0;
      while (i < pts.length - 1) { let j = Math.min(pts.length - 1, i + 60); while (j > i + 1 && !this.clearSeg(pts[i], pts[j])) j--; out.push(pts[j]); i = j; }
      return out;
    }
    pathBlocked() {
      const P = this.path, rb = this.robot; let ax = rb.x, ay = rb.y;
      for (let i = this.wp; i < P.length; i++) {
        const bx = P[i].x, by = P[i].y, L = Math.hypot(bx - ax, by - ay), n = Math.max(1, Math.ceil(L / 0.04));
        for (let s = 1; s <= n; s++) { const x = ax + (bx - ax) * s / n, y = ay + (by - ay) * s / n; if (Math.hypot(x - rb.x, y - rb.y) > 0.14 && this.dAt(x, y) < PASS - 0.005) return true; }
        ax = bx; ay = by;
      }
      return false;
    }

    /* ---------- 导航 ---------- */
    nav(dt, gx, gy, R, ign, vmax) {
      this.replanT -= dt;
      if (this.replanT <= 0) {
        this.replanT = 0.3; this.computeDist(ign);
        const P = this.path, rb = this.robot; let dEnd = 9, blocked = false;
        if (P) { const e = P[P.length - 1]; dEnd = Math.hypot(e.x - rb.x, e.y - rb.y); if (Math.hypot(e.x - gx, e.y - gy) > R + 0.04) dEnd = 9; blocked = this.pathBlocked(); }
        if (!P || dEnd > 0.25 || blocked) {
          const p = this.astar(gx, gy, R);
          if (!p) { if (blocked) this.path = null; if (++this.fail >= 3) { this.fail = 0; return 'fail'; } }
          else {
            this.fail = 0;
            if (blocked && this.t - this.lastAvoid > 1.2) { this.lastAvoid = this.t; this.avoidFlash = 1.4; this.log('路径上发现障碍 → 重新规划', 'avoid'); }
            this.path = p; this.wp = 1;
          }
        }
      }
      if (!this.path) return 'moving';
      return this.follow(vmax);
    }
    follow(vmax) {
      const rb = this.robot, P = this.path, end = P[P.length - 1], dEnd = Math.hypot(end.x - rb.x, end.y - rb.y);
      if (dEnd < 0.04) return 'arrived';
      while (this.wp < P.length - 1 && Math.hypot(P[this.wp].x - rb.x, P[this.wp].y - rb.y) < 0.09) this.wp++;
      const T = P[Math.min(this.wp, P.length - 1)], e = wrap(Math.atan2(T.y - rb.y, T.x - rb.x) - rb.th), ae = Math.abs(e);
      let v = ae > 0.8 ? 0 : vmax * Math.pow(1 - ae / 0.8, 0.8);
      v = Math.min(v, 0.06 + dEnd * 1.6); if (this.dAt(rb.x, rb.y) < 0.2) v *= 0.7;
      this.vt = v; this.wt = clamp(4.5 * e, -2.6, 2.6);
      return 'moving';
    }
    truthFree(x, y) {
      if (x < 0.095 || y < 0.095 || x > W - 0.095 || y > H - 0.095) return false;
      const O = this.obstacles; for (let i = 0; i < O.length; i++) if (dObs(x, y, O[i]) < 0.088) return false;
      return true;
    }
    onBump() {
      const rb = this.robot; let best = null, bd = 9;
      for (const o of this.obstacles) { const d = dObs(rb.x, rb.y, o); if (d < bd) { bd = d; best = o; } }
      if (best && bd < 0.2) {
        let qx, qy;
        if (best.type === 'rect') { qx = clamp(rb.x, best.x - best.w / 2, best.x + best.w / 2); qy = clamp(rb.y, best.y - best.h / 2, best.y + best.h / 2); }
        else { const a = Math.atan2(rb.y - best.y, rb.x - best.x); qx = best.x + Math.cos(a) * (best.r - 0.01); qy = best.y + Math.sin(a) * (best.r - 0.01); }
        const cx = Math.floor(qx / CS), cy = Math.floor(qy / CS);
        for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) { const x = cx + i, y = cy + j; if (x >= 0 && y >= 0 && x < GX && y < GY) this.lo[y * GX + x] = 12; }
      }
      this.backup = 0.45; this.bump = 0; this.replanT = 0; this.path = null;
      this.log('碰撞环触发 → 后退并把障碍写进地图', 'avoid');
    }

    /* ---------- 机械臂 ---------- */
    armTo(j, grip, dur) { const a = this.arm; a.from = a.j.slice(); a.gf = a.grip; a.to = j.slice(); a.gt = grip; a.p = 0; a.dur = dur; }
    updArm(dt) { const a = this.arm; if (a.p >= 1) return; a.p = Math.min(1, a.p + dt / a.dur); const e = a.p * a.p * (3 - 2 * a.p); for (let k = 0; k < 3; k++) a.j[k] = a.from[k] + (a.to[k] - a.from[k]) * e; a.grip = a.gf + (a.gt - a.gf) * e; }
    tipWorld(j) { const f = fk(j || this.arm.j), rb = this.robot; return { x: rb.x + Math.cos(rb.th) * f.tx, y: rb.y + Math.sin(rb.th) * f.tx, z: f.ty }; }
    objPos(o, j) {
      if (o.held) { const t = this.tipWorld(j); return { x: t.x, y: t.y, z: t.z + o.h / 2 - this.ght }; }
      if (o.delivered) {
        const b = this.basket, tx = b.x + o.slot.dx, ty = b.y + o.slot.dy, tz = o.slot.z, a = o.anim;
        if (a && a.p < 1) { const e = a.p * a.p; return { x: a.fx + (tx - a.fx) * a.p, y: a.fy + (ty - a.fy) * a.p, z: a.fz + (tz - a.fz) * e }; }
        return { x: tx, y: ty, z: tz };
      }
      return { x: o.x, y: o.y, z: o.h / 2 };
    }

    /* ---------- 任务决策 ---------- */
    skip(o) { o.skipN++; o.skipUntil = this.t + 4 * Math.pow(2, Math.min(o.skipN, 4)); if (this.priority === o.id) this.priority = null; this.log(`${o.cls.name} 周围太挤，够不到，稍后再试`, 'warn'); }
    pending() { const r = []; for (const o of this.objects) if (!o.delivered && !o.held && this.memory.has(o.id) && o.skipUntil <= this.t) r.push(o); return r; }
    hasWork() { if (this.pendingDetect) return true; if (this.pending().length) return true; for (let i = 0; i < this.cov.length; i++) if (!this.cov[i]) return true; return false; }
    goCarry() {
      const b = this.basket; this.computeDist(this.held.id);
      const p = this.astar(b.x, b.y, 0.3);
      if (p) { this.path = p; this.wp = 1; this.next = 'CARRY'; this.set('PLAN'); this.log(`规划路径 → 收纳筐`, 'plan'); }
      else { if (this.state !== 'WAIT') this.log('去收纳筐的路被堵住了，原地等待', 'warn'); this.set('WAIT'); }
    }
    choose() {
      const rb = this.robot; this.path = null; this.target = null; this.pendingDetect = null;
      if (this.held) return this.goCarry();
      const c = this.pending();
      const rank = o => o.id === this.priority ? -1e3 : Math.hypot(o.x - rb.x, o.y - rb.y);
      c.sort((a, b) => rank(a) - rank(b));
      for (const o of c.slice(0, 4)) {
        this.computeDist(o.id); const p = this.astar(o.x, o.y, 0.24);
        if (p) { this.target = o; this.path = p; this.wp = 1; this.next = 'APPROACH'; this.set('PLAN'); this.log(`规划路径 → ${o.cls.name}`, 'plan'); return; }
        this.skip(o);
      }
      this.computeDist(-1);
      const cells = [];
      for (let j = 0; j < CGY; j++) for (let i = 0; i < CGX; i++) {
        const q = j * CGX + i; if (this.cov[q]) continue;
        const cx = (i + 0.5) * CCS, cy = (j + 0.5) * CCS;
        if (this.dAt(cx, cy) < PASS) { this.cov[q] = 1; continue; }
        cells.push({ q, x: cx, y: cy, s: Math.hypot(cx - rb.x, cy - rb.y) + 0.25 * Math.abs(wrap(Math.atan2(cy - rb.y, cx - rb.x) - rb.th)) });
      }
      cells.sort((a, b) => a.s - b.s);
      for (const k of cells.slice(0, 5)) {
        const p = this.astar(k.x, k.y, 0.5);
        if (p) { this.exT = k.q; this.exC = k; this.path = p; this.wp = 1; if (this.state !== 'EXPLORE') this.log('去还没看过的区域找一找', 'idle'); this.set('EXPLORE'); return; }
        this.cov[k.q] = 1;
      }
      if (Math.hypot(HOME.x - rb.x, HOME.y - rb.y) > 0.08) {
        const p = this.astar(HOME.x, HOME.y, 0.05);
        if (p) { this.path = p; this.wp = 1; if (this.state !== 'DOCK') this.log('没有待办了，回充电座', 'idle'); this.set('DOCK'); return; }
      }
      this.set('IDLE');
    }
    tick(dt) {
      const rb = this.robot, st = this.state;
      switch (st) {
        case 'BOOT': case 'SCAN':
          if (this.pendingDetect) { this.set('DETECT'); break; }
          this.wt = 2.2; this.spin += Math.abs(rb.w) * dt;
          if (this.spin >= 2 * PI) { if (st === 'SCAN' && this.exT >= 0) this.cov[this.exT] = 1; this.spin = 0; this.choose(); }
          break;
        case 'DETECT': if (this.stateT > 0.7) this.choose(); break;
        case 'EXPLORE': {
          if (this.pendingDetect) { this.set('DETECT'); break; }
          if (this.cov[this.exT]) { this.choose(); break; }
          const r = this.nav(dt, this.exC.x, this.exC.y, 0.5, -1, 0.34);
          if (r === 'arrived') { this.spin = 0; this.set('SCAN'); } else if (r === 'fail') { this.cov[this.exT] = 1; this.choose(); }
          break;
        }
        case 'PLAN': if (this.stateT > 0.45) this.set(this.next); break;
        case 'APPROACH': {
          const o = this.target; if (!o || o.delivered) { this.choose(); break; }
          const r = this.nav(dt, o.x, o.y, 0.24, o.id, 0.34);
          if (r === 'arrived') this.set('ALIGN'); else if (r === 'fail') { this.skip(o); this.choose(); }
          break;
        }
        case 'ALIGN': {
          const o = this.target, d = Math.hypot(o.x - rb.x, o.y - rb.y), e = wrap(Math.atan2(o.y - rb.y, o.x - rb.x) - rb.th);
          this.wt = clamp(5 * e, -2, 2);
          if (d < 0.15) this.vt = -0.06; else if (d > 0.275 && Math.abs(e) < 0.3) this.vt = 0.06;
          if (Math.abs(e) < 0.025 && Math.abs(rb.w) < 0.25 && d >= 0.15 && d <= 0.275) {
            this.gr = clamp(d, 0.15, 0.27); this.ght = Math.max(0.012, o.h * 0.35);
            this.set('GRASP'); this.armTo(ik(this.gr, 0.11, -PI / 2), 1, 0.6);
          } else if (this.stateT > 4) { this.skip(o); this.choose(); }
          break;
        }
        case 'GRASP': {
          const o = this.target; if (this.arm.p < 1) break;
          if (this.phase === 0) { this.armTo(ik(this.gr, this.ght, -PI / 2), 1, 0.45); this.phase = 1; }
          else if (this.phase === 1) { this.armTo(this.arm.j, o.cls.gw / 0.076, 0.35); this.phase = 2; }
          else if (this.phase === 2) { o.held = true; this.held = o; this.log(`夹住 ${o.cls.name}（${o.cls.mass} g）`, 'grasp'); this.armTo(J_CARRY, this.arm.grip, 0.7); this.phase = 3; }
          else this.goCarry();
          break;
        }
        case 'CARRY': {
          const b = this.basket, r = this.nav(dt, b.x, b.y, 0.3, this.held.id, 0.28);
          if (r === 'arrived') this.set('ALIGNP'); else if (r === 'fail') { this.log('去收纳筐的路被堵住了，原地等待', 'warn'); this.set('WAIT'); }
          break;
        }
        case 'ALIGNP': {
          const b = this.basket, d = Math.hypot(b.x - rb.x, b.y - rb.y), e = wrap(Math.atan2(b.y - rb.y, b.x - rb.x) - rb.th);
          if (d > 0.345) { this.path = null; this.set('CARRY'); break; }
          this.wt = clamp(5 * e, -2, 2);
          if (Math.abs(e) < 0.03 && Math.abs(rb.w) < 0.25) { this.pr = clamp(d, 0.2, 0.3); this.set('PLACE'); this.armTo(ik(this.pr, 0.19, -0.9), this.arm.grip, 0.65); }
          break;
        }
        case 'PLACE': {
          if (this.arm.p < 1) break; const o = this.held;
          if (this.phase === 0) { this.armTo(ik(this.pr, 0.145, -0.9), this.arm.grip, 0.4); this.phase = 1; }
          else if (this.phase === 1) { this.armTo(this.arm.j, 1, 0.3); this.phase = 2; }
          else if (this.phase === 2) {
            const f = this.objPos(o), n = this.delivered;
            o.slot = { dx: [-0.075, 0, 0.075][n % 3], dy: (n / 3 | 0) % 2 ? 0.035 : -0.035, z: 0.012 + o.h / 2 + (n / 6 | 0) * 0.035 };
            o.anim = { p: 0, fx: f.x, fy: f.y, fz: f.z }; o.held = false; o.delivered = true; this.held = null; this.delivered++;
            this.memory.delete(o.id); if (this.priority === o.id) this.priority = null;
            this.log(`${o.cls.name} 已放进收纳筐（${this.delivered}/${this.objects.length}）`, 'place');
            this.armTo(J_STOW, 0.35, 0.7); this.phase = 3;
          } else this.choose();
          break;
        }
        case 'DOCK': {
          if (this.hasWork()) { this.choose(); break; }
          if (this.phase === 0) { const r = this.nav(dt, HOME.x, HOME.y, 0.05, -1, 0.3); if (r === 'arrived') this.phase = 1; else if (r === 'fail') this.set('IDLE'); }
          else { const e = wrap(-PI / 2 - rb.th); this.wt = clamp(4 * e, -2, 2); if (Math.abs(e) < 0.03) { this.set('IDLE'); this.log('回到充电座，待命', 'idle'); } }
          break;
        }
        case 'IDLE': this.idleT -= dt; if (this.idleT <= 0) { this.idleT = 0.5; if (this.hasWork()) this.choose(); } break;
        case 'WAIT': if (this.stateT > 1) this.choose(); break;
      }
    }
    step(dt) {
      const rb = this.robot; this.t += dt; this.stateT += dt; this.vt = 0; this.wt = 0;
      if (this.avoidFlash > 0) this.avoidFlash -= dt;
      this.senseT -= dt; if (this.senseT <= 0) { this.senseT = 0.1; this.sense(); }
      this.updArm(dt);
      for (const o of this.objects) if (o.anim && o.anim.p < 1) o.anim.p = Math.min(1, o.anim.p + dt / 0.3);
      if (this.backup > 0) { this.backup -= dt; this.vt = -0.1; } else this.tick(dt);
      rb.v += clamp(this.vt - rb.v, -1.5 * dt, 0.8 * dt); rb.w += clamp(this.wt - rb.w, -10 * dt, 10 * dt);
      const nx = rb.x + Math.cos(rb.th) * rb.v * dt, ny = rb.y + Math.sin(rb.th) * rb.v * dt;
      if (this.truthFree(nx, ny)) { rb.x = nx; rb.y = ny; rb.odo += Math.abs(rb.v) * dt; if (rb.v > 0.01) this.bump = 0; }
      else { if (rb.v > 0) { this.bump += dt; if (this.bump > 0.35) this.onBump(); } rb.v = 0; }
      rb.th = wrap(rb.th + rb.w * dt);
      rb.wl += (rb.v - rb.w * 0.094) / 0.034 * dt; rb.wr += (rb.v + rb.w * 0.094) / 0.034 * dt;
    }

    /* ---------- 用户操作 ---------- */
    prioritize(id) {
      const o = this.objects.find(o => o.id === id);
      if (!o || o.delivered) return { ok: false, msg: '这个已经在收纳筐里了' };
      if (o.held || this.target === o) return { ok: false, msg: `正在处理${o.cls.name}` };
      if (!this.memory.has(id)) this.memory.set(id, { conf: null, src: 'app' });
      o.skipUntil = 0; this.priority = id; this.log(`收到指令：先拿 ${o.cls.name}`, 'cmd');
      if (!this.held && ['BOOT', 'SCAN', 'EXPLORE', 'DETECT', 'DOCK', 'IDLE', 'WAIT', 'PLAN', 'APPROACH', 'ALIGN'].indexOf(this.state) >= 0) this.choose();
      return { ok: true, msg: `好，先去拿${o.cls.name}` };
    }
    throwObject() {
      const rb = this.robot;
      for (let n = 0; n < 80; n++) {
        const x = 0.22 + Math.random() * (W - 0.44), y = 0.22 + Math.random() * (H - 0.44);
        if (Math.hypot(x - rb.x, y - rb.y) < 0.3) continue;
        let ok = true;
        for (const o of this.obstacles) if (dObs(x, y, o) < (o.kind === 'basket' ? 0.3 : 0.17)) { ok = false; break; }
        if (ok) for (const o of this.objects) if (!o.delivered && !o.held && Math.hypot(o.x - x, o.y - y) < 0.2) { ok = false; break; }
        if (!ok) continue;
        const o = this.addObject(CKEYS[Math.random() * CKEYS.length | 0], x, y);
        this.cov.fill(0); this.log('听到落地声 → 重新巡视找新物体', 'cmd');
        return { ok: true, msg: `扔了一个${o.cls.name}，看它多久能找到`, obj: o };
      }
      return { ok: false, msg: '地上太满了，放不下新物体' };
    }
    _overlap(r, m) {
      for (const o of this.obstacles) {
        if (o === r) continue;
        if (o.type === 'rect') { if (Math.abs(o.x - r.x) < (o.w + r.w) / 2 + m && Math.abs(o.y - r.y) < (o.h + r.h) / 2 + m) return true; }
        else if (dRect(o.x, o.y, r) < o.r + m) return true;
      }
      return false;
    }
    placeBox(x, y) {
      const r = { type: 'rect', kind: 'box', user: true, x, y, w: 0.3, h: 0.24 }, rb = this.robot;
      if (this.userBoxes >= 10) return { ok: false, msg: '纸箱已经放了 10 个，先重置房间' };
      if (x - 0.15 < 0.04 || x + 0.15 > W - 0.04 || y - 0.12 < 0.04 || y + 0.12 > H - 0.04) return { ok: false, msg: '太靠墙了，放不下' };
      if (dRect(rb.x, rb.y, r) < 0.3) return { ok: false, msg: '离机器人太近，换个位置' };
      if (dRect(HOME.x, HOME.y, r) < 0.28) return { ok: false, msg: '别堵住充电座' };
      if (this._overlap(r, 0.02)) return { ok: false, msg: '这里已经有东西了' };
      for (const o of this.objects) if (!o.delivered && !o.held && dRect(o.x, o.y, r) < 0.1) return { ok: false, msg: '会压到地上的物体' };
      this.obstacles.push(r); this.userBoxes++;
      return { ok: true, msg: '放下一个纸箱，机器人要靠雷达自己发现它' };
    }
    moveBasket(x, y) {
      const b = this.basket, rb = this.robot;
      if (this.state === 'ALIGNP' || this.state === 'PLACE') return { ok: false, msg: '正在放东西，等一下再移' };
      const r = { x, y, w: b.w, h: b.h };
      if (x - b.w / 2 < 0.06 || x + b.w / 2 > W - 0.06 || y - b.h / 2 < 0.06 || y + b.h / 2 > H - 0.06) return { ok: false, msg: '太靠墙了，放不下' };
      if (dRect(rb.x, rb.y, r) < 0.3) return { ok: false, msg: '离机器人太近，换个位置' };
      for (const o of this.obstacles) {
        if (o === b) continue;
        if (o.type === 'rect' ? (Math.abs(o.x - x) < (o.w + b.w) / 2 + 0.04 && Math.abs(o.y - y) < (o.h + b.h) / 2 + 0.04) : dRect(o.x, o.y, r) < o.r + 0.04) return { ok: false, msg: '这里已经有东西了' };
      }
      for (const o of this.objects) if (!o.delivered && !o.held && dRect(o.x, o.y, r) < 0.12) return { ok: false, msg: '会压到地上的物体' };
      b.x = x; b.y = y; this.log('收纳筐被挪走了，目的地更新', 'cmd');
      return { ok: true, msg: '收纳筐换了位置' };
    }
  }
  return { Sim, W, H, CS, GX, GY, CCS, CGX, CGY, PASS, CAM_R, CAM_HALF, HOME, S, L1, L2, L3, ik, fk, J_STOW, J_CARRY, CLASSES, wrap, clamp };
})();
if (typeof module !== 'undefined') module.exports = SimCore;
