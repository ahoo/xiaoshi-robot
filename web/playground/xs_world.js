/* xs_world.js: a small 2-D world for the rules of xs_logic.js: the room, the people (people.Script), what the robot sees of them, and the mode supervisor (modes.Supervisor) with its
 * modes written as generators the way the Python ones are. Movement is a point-mass robot on a path from A*, grasping is a timer with the success rate that was measured in the physics
 * simulator (results/grasp_bench.json), falling and getting up are timers copied from a recorded run. This is a behaviour model, not the physics. */
(function (G) {
'use strict';
const XS = G.XS;
const { clamp, Battery, Guard, Voice } = XS;
const hypot = Math.hypot, PI = Math.PI;
const wrap = a => { while (a > PI) a -= 2 * PI; while (a < -PI) a += 2 * PI; return a; };
function rng32(seed) { let a = (seed >>> 0) || 1; return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
XS.rng32 = rng32;

const ROBOT_R = 0.105, INFLATE = 0.15, CELL = 0.05;
const PARK = [-7, -7], SIT_DROP = 0.45;
const FURN_CN = { sofa: '沙发', table: '桌子', shelf: '架子', plant: '盆栽', chair: '椅子', carton: '箱子', basket: '收纳筐', dock: '充电座' };
const KIND_CN = { ball: '小球', cube: '方块', car: '玩具车', banana: '香蕉玩具', lbrick: 'L形积木', rock: '石头', sock: '袜子', plush: '毛绒玩具' };
// the average power the real simulator's robot drew in each state (watts), measured from the recorded runs
const LOAD_W = { '待命': 13.7, '启动': 14.6, '离开充电座': 21.0, '巡视': 49.7, '导航': 46.3, '抓取': 17.6, '搬运': 49.8, '放置': 18.9, '起身': 47.7, '重新定位': 40.9, '交接': 22.5, '去充电座': 22.5, '对接': 18.0, '扫描': 49.7, '遥控': 30, '暂停': 17, '求助': 13.7, '等待': 17, '回位': 46.3, '完成': 13.7 };
XS.FURN_CN = FURN_CN; XS.KIND_CN = KIND_CN;

// ---------------------------------------------------------------------------------------------------------------- the room
class World {
    constructor(D, seed) {
        this.D = D; this.W = D.world.room[0]; this.H = D.world.room[1];
        this.furn = D.world.furniture.map(f => ({ name: f[0], cx: f[1], cy: f[2], sx: f[3], sy: f[4], h: f[5], rgba: f[6] }));
        this.boxes = (D.layouts[String(seed)].boxes || []).map(b => ({ name: 'carton', cx: b[0], cy: b[1], sx: b[2], sy: b[3], h: b[4], rgba: '0.79 0.64 0.45 1' }));
        this.basket = Object.assign({}, D.world.basket); this.dockPose = D.world.dock.slice(); this.home = D.world.home.slice(); this.doors = D.world.doors;
        this.objects = D.layouts[String(seed)].objects.map((o, i) => ({ id: i, kind: o.kind, x: o.x, y: o.y, th: o.th, r: o.r, z: 0, status: 'on_floor', seen: false, lastSeen: null, fails: 0, gaveUp: false }));
        this.solids = this.furn.concat(this.boxes);
        this.nx = Math.round(this.W / CELL); this.ny = Math.round(this.H / CELL); this.free = new Uint8Array(this.nx * this.ny);
        for (let j = 0; j < this.ny; j++) for (let i = 0; i < this.nx; i++) this.free[j * this.nx + i] = this.clearance((i + 0.5) * CELL, (j + 0.5) * CELL) >= INFLATE ? 1 : 0;
    }
    /** the distance from a point to the nearest wall or piece of furniture (negative inside) */
    clearance(x, y, noBasket) {
        let d = Math.min(x, y, this.W - x, this.H - y);
        for (const f of this.solids) d = Math.min(d, this._dist(f, x, y));
        if (!noBasket) { const b = this.basket; d = Math.min(d, hypot(Math.max(Math.abs(x - b.x) - b.sx / 2, 0), Math.max(Math.abs(y - b.y) - b.sy / 2, 0))); }
        return d;
    }
    _dist(f, x, y) {
        if (f.sy === null || f.sy === undefined) return hypot(x - f.cx, y - f.cy) - f.sx;
        const dx = Math.abs(x - f.cx) - f.sx / 2, dy = Math.abs(y - f.cy) - f.sy / 2;
        return dx > 0 || dy > 0 ? hypot(Math.max(dx, 0), Math.max(dy, 0)) : Math.max(dx, dy);
    }
    cell(x, y) { return [clamp(Math.floor(x / CELL), 0, this.nx - 1), clamp(Math.floor(y / CELL), 0, this.ny - 1)]; }
    isFree(x, y, extra) {
        const [i, j] = this.cell(x, y); if (!this.free[j * this.nx + i]) return false;
        if (extra) for (const z of extra) if (hypot(x - z[0], y - z[1]) < z[2] + (z[3] === undefined ? INFLATE : z[3])) return false;
        return true;
    }
    /** a line of sight between two points that does not cross a piece of furniture taller than `h` */
    sees(x0, y0, x1, y1, h = 0.1) {
        const n = Math.ceil(hypot(x1 - x0, y1 - y0) / 0.04);
        for (let k = 1; k < n; k++) { const x = x0 + (x1 - x0) * k / n, y = y0 + (y1 - y0) * k / n; for (const f of this.solids) if (f.h > h && this._dist(f, x, y) < 0) return false; }
        return true;
    }
    /** A* on the cost map (8 neighbours), then the path is pulled straight where it can be. `zones`: [[x, y, r], ...] that are blocked as well (people). Returns [[x, y], ...] or null */
    plan(a, b, zones) {
        const nx = this.nx, ny = this.ny; let [si, sj] = this.cell(a[0], a[1]), [gi, gj] = this.cell(b[0], b[1]);
        const ok = (i, j) => i >= 0 && j >= 0 && i < nx && j < ny && this.free[j * nx + i] && (!zones || this.isFree((i + 0.5) * CELL, (j + 0.5) * CELL, zones));
        const nearest = (i, j) => {
            if (ok(i, j)) return [i, j];
            for (let r = 1; r < 12; r++) { let best = null, bd = 1e9; for (let dj = -r; dj <= r; dj++) for (let di = -r; di <= r; di++) { if (Math.max(Math.abs(di), Math.abs(dj)) !== r || !ok(i + di, j + dj)) continue; const d = hypot(di, dj); if (d < bd) { bd = d; best = [i + di, j + dj]; } } if (best) return best; }
            return null;
        };
        const s = nearest(si, sj), g = nearest(gi, gj); if (!s || !g) return null;
        const N = nx * ny, gs = new Float64Array(N).fill(1e9), from = new Int32Array(N).fill(-1), closed = new Uint8Array(N);
        const heap = []; const push = (f, n) => { heap.push([f, n]); let k = heap.length - 1; while (k > 0) { const p = (k - 1) >> 1; if (heap[p][0] <= heap[k][0]) break; [heap[p], heap[k]] = [heap[k], heap[p]]; k = p; } };
        const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let k = 0; for (;;) { let l = 2 * k + 1, r = l + 1, m = k; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === k) break; [heap[m], heap[k]] = [heap[k], heap[m]]; k = m; } } return top; };
        const h = (i, j) => { const dx = Math.abs(i - g[0]), dy = Math.abs(j - g[1]); return (dx + dy) + (Math.SQRT2 - 2) * Math.min(dx, dy); };
        const s0 = s[1] * nx + s[0]; gs[s0] = 0; push(h(s[0], s[1]), s0);
        const goal = g[1] * nx + g[0]; let found = false;
        while (heap.length) {
            const [, n] = pop(); if (closed[n]) continue; closed[n] = 1; if (n === goal) { found = true; break; }
            const i = n % nx, j = (n / nx) | 0;
            for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
                if (!di && !dj) continue; const ni = i + di, nj = j + dj; if (!ok(ni, nj)) continue;
                if (di && dj && (!ok(i + di, j) || !ok(i, j + dj))) continue;
                const m = nj * nx + ni, c = gs[n] + (di && dj ? Math.SQRT2 : 1); if (c < gs[m]) { gs[m] = c; from[m] = n; push(c + h(ni, nj), m); }
            }
        }
        if (!found) return null;
        let pts = [], n = goal; while (n !== -1) { pts.push([(n % nx + 0.5) * CELL, (((n / nx) | 0) + 0.5) * CELL]); n = from[n]; } pts.reverse();
        pts[0] = [a[0], a[1]]; if (hypot(b[0] - pts[pts.length - 1][0], b[1] - pts[pts.length - 1][1]) < 0.12 && this.isFree(b[0], b[1], zones)) pts[pts.length - 1] = [b[0], b[1]];
        const clear = (p, q) => { const m = Math.ceil(hypot(q[0] - p[0], q[1] - p[1]) / 0.025); for (let k = 0; k <= m; k++) if (!this.isFree(p[0] + (q[0] - p[0]) * k / m, p[1] + (q[1] - p[1]) * k / m, zones)) return false; return true; };
        const out = [pts[0]]; let k = 0;
        while (k < pts.length - 1) { let far = k + 1; for (let m = pts.length - 1; m > k + 1; m--) if (clear(pts[k], pts[m])) { far = m; break; } out.push(pts[far]); k = far; }
        return out;
    }
}
XS.World = World;

// ---------------------------------------------------------------------------------------------------------------- the people (people.Script, people.People)
class Script {
    constructor(steps) { this.steps = steps.slice(); this.i = 0; this.x = PARK[0]; this.y = PARK[1]; this.yaw = 0; this.s = 0; this.v = 0; this.face = false; this._t = null; this._k = 0; this._started = false; }
    _done() { this.i++; this._k = 0; this._started = false; }
    at(t, rxy, doors) {
        const dt = this._t === null ? 0 : Math.max(t - this._t, 0); this._t = t; this.v = 0;
        let guard = 0;
        while (this.i < this.steps.length && guard < 20) {
            guard++; const st = this.steps[this.i], k = st[0];
            if (k === 'at') { this.x = st[1]; this.y = st[2]; this.yaw = st[3]; this._done(); continue; }
            if (k === 'turn') { this.yaw = st[1]; this._done(); continue; }
            if (k === 'face_robot') { this.face = true; this._done(); continue; }
            if (k === 'hide') { this.x = PARK[0]; this.y = PARK[1]; this._done(); continue; }
            if (k === 'wait') { this._k += dt; if (this._k >= st[1]) { this._done(); continue; } break; }
            if (k === 'enter') {
                let far = doors[0], fd = -1; for (const p of doors) { const d = hypot(p[0] - rxy[0], p[1] - rxy[1]); if (d > fd) { fd = d; far = p; } }
                this.x = far[0]; this.y = far[1]; this.yaw = Math.atan2(1.5 - far[1], 2.0 - far[0]); this._done(); continue;
            }
            if (k === 'exit') {
                let ok = doors.filter(p => hypot(p[0] - rxy[0], p[1] - rxy[1]) > 1.2); if (!ok.length) ok = doors;
                let door = ok[0], bd = 1e9; for (const p of ok) { const d = hypot(p[0] - this.x, p[1] - this.y); if (d < bd) { bd = d; door = p; } }
                this.steps[this.i] = ['walk', door[0], door[1], st[1]]; this.steps.splice(this.i + 1, 0, ['hide']); continue;
            }
            if (k === 'to_robot') {
                const d = hypot(rxy[0] - this.x, rxy[1] - this.y), target = st[1] + 0.105 + 0.14;
                if (d <= target + 0.01) { this._done(); continue; }
                this.steps[this.i] = ['walk_dyn', st[1], st[2]]; continue;
            }
            if (k === 'walk' || k === 'walk_dyn') {
                let tx, ty, sp, goalD = null;
                if (k === 'walk_dyn') { tx = rxy[0]; ty = rxy[1]; const target = st[1] + 0.105 + 0.14, dd = hypot(tx - this.x, ty - this.y); if (dd <= target + 0.01) { this._done(); continue; } sp = st[2]; goalD = dd - target; }
                else { tx = st[1]; ty = st[2]; sp = st[3]; }
                const dx = tx - this.x, dy = ty - this.y; let d = hypot(dx, dy); this._k += dt;
                if (!this._started) { this._d0 = d; this._started = true; }
                if (this._k > 5.0 + 3.0 * this._d0 / Math.max(sp, 0.1)) { this._done(); continue; }
                if (d < 1e-6) { this._done(); continue; }
                if (goalD !== null) d = goalD;
                this.yaw = Math.atan2(dy, dx); const step = Math.min(sp * dt, d), dd = hypot(dx, dy);
                let nx = this.x + dx / dd * step, ny = this.y + dy / dd * step;
                if (k === 'walk') {
                    const rd = hypot(nx - rxy[0], ny - rxy[1]);
                    if (rd < 0.45) { const ux = (nx - rxy[0]) / Math.max(rd, 1e-6), uy = (ny - rxy[1]) / Math.max(rd, 1e-6); let tx_ = -uy, ty_ = ux; if (tx_ * dx + ty_ * dy < 0) { tx_ = -tx_; ty_ = -ty_; } nx = rxy[0] + ux * 0.45 + tx_ * step; ny = rxy[1] + uy * 0.45 + ty_ * step; }
                }
                this.v = dt > 0 ? hypot(nx - this.x, ny - this.y) / dt : 0; this.x = nx; this.y = ny;
                if (k === 'walk' && hypot(st[1] - this.x, st[2] - this.y) < 0.03) this._done();
                break;
            }
            if (k === 'posture') {
                this._k += dt; const a = st[1] === 'sit' ? 1.0 : 0.0; if (!this._started) { this._s0 = this.s; this._started = true; }
                const f = st[2] <= 0 ? 1.0 : Math.min(this._k / st[2], 1.0); this.s = this._s0 + f * (a - this._s0);
                if (f >= 1.0) { this._done(); continue; } break;
            }
            if (k === 'chase') {
                this._k += dt; const dx = rxy[0] - this.x, dy = rxy[1] - this.y, d = hypot(dx, dy), target = st[2] + 0.105 + 0.14; this.yaw = Math.atan2(dy, dx);
                if (d <= target + 0.01 || this._k >= st[3]) { this._done(); continue; }
                const step = Math.min(st[1] * dt, d - target); this.x += dx / d * step; this.y += dy / d * step; this.v = dt > 0 ? step / dt : 0; break;
            }
            this._done();
        }
        if (this.face && this.v < 0.01) this.yaw = Math.atan2(rxy[1] - this.y, rxy[0] - this.x);
        return [this.x, this.y, this.yaw, this.v, this.s];
    }
}
XS.Script = Script;

class People {
    constructor(D, comply, rnd) {
        this.D = D; this.specs = D.people; this.scripts = this.specs.map(() => new Script([])); this.comply = comply || [1.0, 1.0, 0.7, 0.7, 0.0]; this.rnd = rnd; this.state = []; this.t = 0;
        this.contact = this.specs.map(() => 0); this._in = this.specs.map(() => false); this.minGap = this.specs.map(() => 9);
        this.update(0, [2, 1.5]);
    }
    index(name) { return this.specs.findIndex(s => s.name === name); }
    update(t, rxy) {
        this.t = t;
        this.state = this.specs.map((s, i) => {
            const [x, y, yaw, v, sit] = this.scripts[i].at(t, rxy, this.D.world.doors); const hipH = (s.hip || 0.5) * s.h;       // sitting drops the hips by SIT_DROP of the hip height, and the head with them
            return { i, name: s.name, cn: s.cn, cls: s.cls, x, y, yaw, v, sit, h: s.h - SIT_DROP * hipH * sit, hStand: s.h, w: s.w, inside: x > -0.5 };
        });
    }
    truth() { return this.state; }
    askToMove(rxy, reach = 1.4) {
        for (const s of this.state) {
            if (!s.inside || hypot(s.x - rxy[0], s.y - rxy[1]) > reach || this.rnd() >= this.comply[s.i]) continue;
            const dx = s.x - rxy[0], dy = s.y - rxy[1], d = Math.max(hypot(dx, dy), 1e-3);
            const ax = clamp(s.x + dx / d * 1.0, 0.25, 3.75), ay = clamp(s.y + dy / d * 1.0, 0.25, 2.75), sc = this.scripts[s.i];
            sc.steps.splice(sc.i, 0, ['wait', 1.0], ['walk', ax, ay, 0.8]); sc._k = 0; sc._started = false;
        }
    }
    /** contacts between a person and the robot: counted when they start */
    touch(rxy, rr) {
        let n = 0;
        for (const s of this.state) {
            if (!s.inside) { this._in[s.i] = false; continue; }
            const gap = hypot(s.x - rxy[0], s.y - rxy[1]) - rr - (s.w / 2 + 0.02); this.minGap[s.i] = Math.min(this.minGap[s.i], gap);
            const c = gap < 0; if (c && !this._in[s.i]) { this.contact[s.i]++; n++; } this._in[s.i] = c;
        }
        return n;
    }
}
XS.People = People;

// ---------------------------------------------------------------------------------------------------------------- what the robot sees of the people (the head camera tracker and the lidar movers)
class Tracker {
    constructor(K) {
        this.K = K; this.cam = []; this.lid = []; this.nextId = 1; this.t = 0; this.fov = 35 * PI / 180; this.range = 5.0; this.period = 0.2; this.last = -9;
    }
    update(t, pose, state) {
        if (t - this.last < this.period - 1e-9) return; const dt = t - this.last; this.last = t; this.t = t;
        const [x, y, th] = pose, vis = [], mov = [];
        for (const s of state) {
            if (!s.inside) continue; const d = hypot(s.x - x, s.y - y), b = wrap(Math.atan2(s.y - y, s.x - x) - th);
            if (d <= this.range && Math.abs(b) <= this.fov && d > 0.12) vis.push(s); else if (s.v > 0.15 && d < 4.0) mov.push(s);
        }
        this._assoc(this.cam, vis, t, dt, 'cam'); this._assoc(this.lid, mov, t, dt, 'lidar');
        for (const k of this.cam) { k.dist = hypot(k.x - x, k.y - y); }
        for (const k of this.lid) { k.cls = 'unsure'; k.dist = hypot(k.x - x, k.y - y); }
    }
    _assoc(tracks, dets, t, dt, src) {
        const used = new Set();
        for (const k of tracks) {
            let best = null, bd = 0.6; for (const s of dets) { if (used.has(s.i)) continue; const d = hypot(s.x - k.x, s.y - k.y); if (d < bd) { bd = d; best = s; } }
            if (best) {
                used.add(best.i); const vx = (best.x - k.x) / Math.max(dt, 1e-3), vy = (best.y - k.y) / Math.max(dt, 1e-3);
                k.vx = k.hits > 0 ? 0.5 * k.vx + 0.5 * vx : vx; k.vy = k.hits > 0 ? 0.5 * k.vy + 0.5 * vy : vy; k.x = best.x; k.y = best.y; k.tSeen = t; k.hits++; k.misses = 0; k.coast = false; k.truth = best.i; k.h = best.h; k.w = best.w;
                if (src === 'cam') k.cls = XS.classify(best.h, best.w, false, this.K);
            } else { k.misses++; if (t - k.tSeen > 0.3) k.coast = true; }
        }
        for (const s of dets) if (!used.has(s.i)) {
            const k = { id: this.nextId++, x: s.x, y: s.y, vx: 0, vy: 0, hits: 1, misses: 0, coast: false, tSeen: t, truth: s.i, src, h: s.h, w: s.w, cls: src === 'cam' ? XS.classify(s.h, s.w, false, this.K) : 'unsure', dist: 0 };
            k.predict = function (tt) { return [this.x + this.vx * (tt - this.tSeen), this.y + this.vy * (tt - this.tSeen)]; }; tracks.push(k);
        }
        for (let i = tracks.length - 1; i >= 0; i--) { tracks[i].predict = tracks[i].predict || function (tt) { return [this.x + this.vx * (tt - this.tSeen), this.y + this.vy * (tt - this.tSeen)]; }; if (t - tracks[i].tSeen > 1.5) tracks.splice(i, 1); }
    }
    /** confirmed tracks: the camera's (two looks) and the lidar's that the camera does not already have (three looks, moving) */
    confirmed() {
        const cam = this.cam.filter(k => k.hits >= 2), lid = this.lid.filter(k => k.hits >= 3 && !k.coast && !cam.some(c => hypot(k.x - c.x, k.y - c.y) < 0.6));
        return cam.concat(lid);
    }
}
XS.Tracker = Tracker;

// ---------------------------------------------------------------------------------------------------------------- the robot and its supervisor (modes.Supervisor)
const DEFAULTS_CFG = { auto_charge: true, auto_right: true, crit_overrides_teleop: true, deadman: 0.5, fall_tilt: 45.0, fall_hold: 0.3 };

class Sim {
    constructor(D, opts = {}) {
        this.D = D; this.dt = 0.05; this.t = 0; this.seed = opts.seed || 4; this.rnd = rng32(this.seed * 7919 + 13);
        this.cfg = Object.assign({}, D.modes, DEFAULTS_CFG, opts.cfg || {});
        this.world = new World(D, this.seed); this.dockPose = D.world.dock; this.K = D.classify;
        const [dx, dy, dth] = opts.start === 'home' ? D.world.home : D.world.dock;
        this.x = dx; this.y = dy; this.th = dth; this.vx = 0; this.vy = 0; this.wz = 0; this.tilt = 0; this.fallT = null; this.fallKind = null; this.held = null; this.armOut = false; this.armHold = false;
        this.bat = new Battery(D.battery, opts.soc0 === undefined ? 0.9 : opts.soc0, null, this.rnd); this.estBias = this.bat.bias;
        this.people = new People(D, undefined, this.rnd); this.tracker = new Tracker(this.K); this.guard = new Guard(D.guard); this.guardOn = true;
        this.mode = 'IDLE'; this.state = '待命'; this.kw = {}; this.gen = null; this.modeT = 0; this.q = []; this.events = []; this.chat = []; this.logLines = [];
        this.stats = { charges: 0, charge_min: 0, falls: 0, righted: 0, right_failed: 0, takeovers: 0, nav_ok: 0, nav_fail: 0, dock_fail: 0, delivered: 0, contacts: 0 };
        this.taskDone = false; this._taskStarted = false; this._looked = false; this._handback = false; this.navQueue = []; this.navExplore = false; this.navResults = [];
        this.resume = null; this.resumeRight = null; this.scanDone = false; this.scanned = new Set(); this.fault = null; this.warn = null; this._softReq = false; this._fallTicks = 0;
        this.wheelTarget = { vx: 0, vy: 0, wz: 0 }; this.zoneSkipT = -1e9; this.dynZones = []; this.personZones = []; this.abort = null;
        this.tele = { vx: 0, vy: 0, wz: 0, t: -1e9, safety: true }; this.speakerAt = null; this.trail = []; this.path = null; this.targetObj = null; this.failRight = false; this.nowDocked = true;
        this.guard.dt = this.dt; this.pad = 0;
        this.voice = new Voice(D.voice, D.pinyin, this._host());
        this.gen = this._gen_idle(); this._lastTrail = -9;
    }
    // ---- bookkeeping
    pose() { return [this.x, this.y, this.th]; }
    armDone() { return !this._armBusyUntil || this.t >= this._armBusyUntil; }
    get armBusy() { return this._armBusy || false; }
    set armBusy(v) { this._armBusy = v; }
    setTwist() { this.wheelTarget = { vx: 0, vy: 0, wz: 0 }; }
    event(text) { this.events.push([Math.round(this.t * 10) / 10, this.mode, text]); this.logLines.push([this.t, `[${this.mode}] ${text}`]); }
    log(text) { this.logLines.push([this.t, text]); if (this.logLines.length > 300) this.logLines.shift(); }
    say(who, text) { this.chat.push({ t: Math.round(this.t * 10) / 10, who, text }); if (this.chat.length > 200) this.chat.shift(); }
    speak(text) { this.say('机器人', text); }
    onAsk(x, y) { this.people.askToMove([x, y]); }
    tracks() { return this.tracker.confirmed(); }
    socEst() { return this.bat.est; }
    atDock() { return hypot(this.x - this.dockPose[0], this.y - this.dockPose[1]) < 0.12; }
    dockConnected() { return hypot(this.x - this.dockPose[0], this.y - this.dockPose[1]) < 0.05 && this.mode !== 'RIGHT'; }
    delivered() { return this.world.objects.filter(o => o.status === 'in_basket').length; }

    // ---- the operator's side (modes.Supervisor)
    request(mode, kw) { this.q.push([mode.toUpperCase(), kw || {}]); }
    startTask() { this.request('TASK'); } idle() { this.request('IDLE'); } chargeNow() { this.request('CHARGE', { reason: 'requested' }); } teleop() { this.request('TELEOP'); }
    scanHome() { this.request('SCAN'); } explore() { this.request('NAV', { goals: [], explore: true }); } fetch(o) { this.request('FETCH', { k: o.id }); }
    goTo(x, y, th = null, tol = 0.07) { this.request('NAV', { goals: [{ x, y, th, tol }] }); }
    drive(vx, vy, wz) { this.tele.vx = vx; this.tele.vy = vy; this.tele.wz = wz; this.tele.t = this.t; }
    status() { return { t: this.t, mode: this.mode, state: this.state, soc: this.bat.est, fault: this.fault, warn: this.warn }; }
    /** people walking in (the presets of live.py) */
    person(kind) {
        const P = this.people, S = n => P.scripts[P.index(n)].steps;
        if (kind === 'kid') S('child_5').push(['enter'], ['chase', 1.4, 0.12, 15.0], ['face_robot'], ['wait', 6.0], ['exit', 0.9]);
        else if (kind === 'adult') S('adult_m').push(['enter'], ['to_robot', 0.5, 1.0], ['exit', 1.0]);
        else if (kind === 'toddler') S('toddler').push(['enter'], ['walk', 3.2, 2.35, 0.4], ['wait', 25.0], ['walk', 3.05, 2.45, 0.3], ['wait', 15.0], ['exit', 0.4]);
        else if (kind === 'family') {
            S('adult_f').push(['enter'], ['walk', 2.0, 1.1, 0.8], ['wait', 10.0], ['walk', 3.6, 1.5, 0.8], ['wait', 20.0], ['walk', 1.0, 2.6, 0.8], ['wait', 20.0], ['exit', 0.8]);
            S('child_8').push(['wait', 10.0], ['enter'], ['walk', 2.2, 1.9, 1.1], ['walk', 1.0, 1.0, 1.1], ['wait', 5.0], ['walk', 3.0, 2.4, 1.4], ['wait', 5.0], ['walk', 3.6, 1.9, 1.3], ['wait', 5.0], ['exit', 1.0]);
            S('toddler').push(['wait', 20.0], ['enter'], ['walk', 2.6, 1.7, 0.3], ['wait', 8.0], ['walk', 3.4, 2.2, 0.3], ['wait', 15.0], ['exit', 0.4]);
        } else if (kind === 'leave') P.specs.forEach((s, i) => P.scripts[i].steps.push(['exit', 1.0]));
        else if (kind && kind.startsWith('stand:')) { const [, name, x, y] = kind.split(':'); const sc = P.scripts[P.index(name)]; sc.steps.push(['at', +x, +y, 0.0], ['face_robot']); }
    }

    _host() {
        const S = this;
        return {
            now: () => S.t, mode: () => S.mode, socEst: () => S.bat.est, idle: () => S.idle(), startTask: () => S.startTask(), scanHome: () => S.scanHome(), chargeNow: () => S.chargeNow(),
            requestMode: (m, kw) => S.request(m, kw), curKw: () => Object.assign({}, S.kw), explore: () => S.explore(), onSpeak: t => S.speak(t),
            placeKnown: n => n === '收纳筐' || n === '充电座' || (S.scanDone && S.placeNames().includes(n)),
            goToPlace: n => S.goToPlace(n), items: k => S.inventory(k), canFetch: i => i.status === 'on_floor', fetch: i => S.fetch(i), where: i => S.itemWhere(i),
            describePeople: () => S.describePeople(), speakerLoc: () => S.speakerAt, comeHere: loc => { const [x, y] = [S.x, S.y], d = hypot(loc[0] - x, loc[1] - y), k = Math.max(d - 1.0, 0) / Math.max(d, 1e-6); S.goTo(x + (loc[0] - x) * k, y + (loc[1] - y) * k, Math.atan2(loc[1] - y, loc[0] - x), 0.2); },
        };
    }
    placeNames() { return Array.from(new Set(this.world.solids.map(f => FURN_CN[f.name]))); }
    inventory(kind) {
        const W = this.world, out = [];
        for (const o of W.objects) {
            if (!o.seen || (kind && o.kind !== kind)) continue;
            out.push({ id: o.id, kind: o.kind, label: KIND_CN[o.kind], status: o.status === 'in_basket' ? 'in_basket' : (o.status === 'held' ? 'held' : 'on_floor'), place: this.itemPlace(o), lastSeen: o.lastSeen, o });
        }
        return out.sort((a, b) => b.lastSeen - a.lastSeen);
    }
    itemPlace(o) {
        if (o.status === 'in_basket') return '收纳筐里'; if (!this.scanDone) return null;
        let best = null, bd = 0.40; for (const f of this.world.solids) { const d = this.world._dist(f, o.x, o.y); if (d < bd) { bd = d; best = f; } }
        return best ? `${FURN_CN[best.name]}旁边` : '地板中间';
    }
    itemWhere(i) {
        const age = Math.max(this.t - i.lastSeen, 0), a = age < 90 ? `${Math.round(age)} 秒` : (age < 5400 ? `${Math.round(age / 60)} 分钟` : `${(age / 3600).toFixed(1)} 小时`);
        const st = { on_floor: '在地上', in_basket: '在收纳筐里', held: '正拿在手里' }[i.status];
        return `${i.label}：${i.place || '不知道在哪'}（${st}，${a}前看到）`;
    }
    describePeople() {
        const [x, y, th] = this.pose(), out = [];
        for (const k of this.tracks().filter(k => k.src !== 'lidar').sort((a, b) => a.dist - b.dist)) {
            const sn = Math.sin(Math.atan2(k.y - y, k.x - x) - th), side = sn > 0.25 ? '左' : (sn < -0.25 ? '右' : '正');
            const who = k.cls === 'child' ? '一个小孩' : (k.cls === 'adult' ? '一个大人' : '一个人（不确定大人还是小孩）');
            out.push(`${who}，约 ${k.h.toFixed(1)} 米高，在${side}前方 ${k.dist.toFixed(1)} 米`);
        }
        return out;
    }
    goToPlace(name) {
        const W = this.world;
        if (name === '收纳筐') { this.goTo(W.basket.x - 0.5, W.basket.y); return true; }
        let pts = [];
        if (name === '充电座') return false;
        for (const f of W.solids) {
            if (FURN_CN[f.name] !== name) continue;
            if (f.sy === null || f.sy === undefined) for (let a = 0; a < 8; a++) pts.push([f.cx + Math.cos(a * PI / 4) * (f.sx + 0.5), f.cy + Math.sin(a * PI / 4) * (f.sx + 0.5)]);
            else for (const [sx, sy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) pts.push([f.cx + sx * (f.sx / 2 + 0.5), f.cy + sy * (f.sy / 2 + 0.5)]);
        }
        pts = pts.filter(p => W.isFree(p[0], p[1]));
        if (!pts.length) {       // the real code only tries the four spots 0.5 m from the middle of each side; the sandbox looks for any free spot 0.4-0.75 m from the piece, so that it is rarely refused
            const fs = W.solids.filter(f => FURN_CN[f.name] === name);
            for (let x = 0.1; x < W.W; x += 0.1) for (let y = 0.1; y < W.H; y += 0.1) { const d = Math.min(...fs.map(f => W._dist(f, x, y))); if (d > 0.4 && d < 0.75 && W.isFree(x, y)) pts.push([x, y]); }
            if (!pts.length) return false;
        }
        pts.sort((a, b) => hypot(a[0] - this.x, a[1] - this.y) - hypot(b[0] - this.x, b[1] - this.y)); this.goTo(pts[0][0], pts[0][1]); return true;
    }

    // ---- one control period
    tick() {
        this.t = Math.round((this.t + this.dt) * 1e6) / 1e6; const dt = this.dt;
        this.wheelTarget = { vx: 0, vy: 0, wz: 0 };
        this._intake(); this._watch();
        const g = this.guardOn ? this.guard : null; const halted = g ? g.pre(this) : false;
        if (!g) { this.dynZones = []; this.personZones = []; this.armHold = false; }
        if (this.gen && !halted) {
            let r; try { r = this.gen.next(); } catch (e) { throw e; }
            if (r.done) { this.gen = null; this._onDone(); }
        }
        if (g && !halted) g.post(this);
        this.people.update(this.t, [this.x, this.y]); this._physics(dt); this._account(dt);
        this.tracker.update(this.t, this.pose(), this.people.state); this._sense();
        if (this.t - this._lastTrail > 0.5) { this._lastTrail = this.t; this.trail.push([this.x, this.y]); if (this.trail.length > 600) this.trail.shift(); }
    }
    run(seconds) { const n = Math.round(seconds / this.dt); for (let i = 0; i < n; i++) this.tick(); }
    _sense() {            // what the base camera and the lidar see of the things on the floor: within 2.2 m and not behind furniture
        for (const o of this.world.objects) {
            if (o.status !== 'on_floor') continue; const d = hypot(o.x - this.x, o.y - this.y);
            if (d < 2.2 && this.world.sees(this.x, this.y, o.x, o.y, 0.12) && (this._spinning || Math.abs(wrap(Math.atan2(o.y - this.y, o.x - this.x) - this.th)) < 0.9 || d < 0.6)) { o.seen = true; o.lastSeen = this.t; }
        }
    }
    _account(dt) {
        const b = this.bat; if (this.mode === 'CHARGE' && this.state === '充电') return;
        const w = LOAD_W[this.state] !== undefined ? LOAD_W[this.state] : 20; b.step(dt, w, 0);
    }
    _physics(dt) {
        const W = this.world, w = this.wheelTarget;
        const accel = 0.6 * dt, tgt = this.tilt > 8 ? { vx: 0, vy: 0, wz: 0 } : w;
        this.vx += clamp(tgt.vx - this.vx, -accel, accel); this.vy += clamp(tgt.vy - this.vy, -accel, accel); this.wz += clamp(tgt.wz - this.wz, -3 * dt, 3 * dt);
        const c = Math.cos(this.th), s = Math.sin(this.th), dx = (c * this.vx - s * this.vy) * dt, dy = (s * this.vx + c * this.vy) * dt;
        const ok = (x, y) => W.clearance(x, y, false) >= ROBOT_R - 0.005 && !this.people.state.some(p => p.inside && hypot(p.x - x, p.y - y) < ROBOT_R + p.w / 2 * 0.9);
        if (ok(this.x + dx, this.y + dy)) { this.x += dx; this.y += dy; }
        else if (ok(this.x + dx, this.y)) this.x += dx; else if (ok(this.x, this.y + dy)) this.y += dy;
        else { this.vx = this.vy = 0; }
        this.th = wrap(this.th + this.wz * dt);
        this.stats.contacts += this.people.touch([this.x, this.y], ROBOT_R);
        // falling and getting up are driven by the timers below
        if (this.fallT !== null) {
            const e = this.t - this.fallT; this.tilt = e < 0.8 ? 160 * Math.sin(Math.min(e / 0.8, 1) * PI / 2) : (this.fallPhase === 'up' ? Math.max(0, this.tilt - 90 * dt) : this.tilt);
        }
    }

    // ---- switching (modes.Supervisor._intake / _switch / _watch / _on_done)
    _soft() {
        if (this._softReq) return true;
        if (!this.cfg.auto_charge || !['TASK', 'NAV', 'SCAN'].includes(this.mode)) return false;
        return this.bat.est < this.cfg.soc_soft;
    }
    _switch(mode, kw = {}) {
        if (this.gen) { try { this.gen.return(); } catch (e) {} this.gen = null; }
        this.setTwist(); this.path = null; this.armHold = false;
        this.event(`→ ${mode}` + (mode !== 'NAV' && Object.keys(kw).length ? ' ' + JSON.stringify(kw) : ''));
        this.mode = mode; this.kw = kw; this.modeT = this.t; this._softReq = false; this.armBusy = false;
        this.gen = this['_gen_' + mode.toLowerCase()](kw);
    }
    _intake() {
        while (this.q.length) {
            const [mode, kw0] = this.q.shift(); let kw = kw0;
            if (this.mode === 'RIGHT' || this.mode === 'FAULT') { if (this.mode === 'RIGHT') { this.q.unshift([mode, kw]); return; } if (mode !== 'IDLE') continue; }
            if (mode === 'TELEOP' && this.mode !== 'TELEOP') { this.stats.takeovers++; this.resume = null; }
            if (mode !== 'TASK' && this.mode === 'TASK') this._handback = true;
            if (mode === 'NAV') { kw = Object.assign({}, kw); this.navQueue = (kw.goals || []).slice(); delete kw.goals; this.navExplore = !!kw.explore; }
            if (mode === 'IDLE') this.fault = null;
            this._switch(mode, kw);
        }
    }
    _watch() {
        const cfg = this.cfg;
        const frz = this.tilt > 12 ? 2 : (this.mode === 'RIGHT' ? 1 : 0); this.freeze = frz;
        this._fallTicks = this.tilt > cfg.fall_tilt ? this._fallTicks + 1 : 0;
        if (this._fallTicks >= cfg.fall_hold / this.dt && !['RIGHT', 'FAULT'].includes(this.mode)) {
            this.stats.falls++; this.event(`摔倒了（IMU 倾角 ${Math.round(this.tilt)}°）`);
            if (cfg.auto_right) { this.resumeRight = [this.mode, Object.assign({}, this.kw)]; this._handback = this._handback || this.mode === 'TASK'; this._switch('RIGHT'); }
            else this._fault('fall');
            return;
        }
        const b = this.bat; if (this.mode === 'FAULT' || this.mode === 'RIGHT') return;
        if (b.dead) { this._fault('battery'); return; }
        const e = b.est; this.warn = e < cfg.soc_soft ? 'low battery' : null;
        if (e < cfg.soc_crit && this.mode !== 'CHARGE' && cfg.auto_charge && (this.mode !== 'TELEOP' || cfg.crit_overrides_teleop)) {
            this.resume = this.mode === 'TASK' ? ['TASK', { resume: true }] : (this.mode === 'NAV' ? ['NAV', Object.assign({}, this.kw)] : null);
            if (this.mode === 'TASK') this._handback = true;
            this.event(`电量 ${Math.round(100 * e)}%，低于紧急线，放下手里的事去充电`); this._switch('CHARGE', { reason: 'critical' });
        }
    }
    _fault(why) { this.fault = why; this.event(`故障：${why}，需要人来处理`); this._switch('FAULT', { why }); }
    _onDone() {
        const mode = this.mode;
        if (mode === 'TASK') {
            if (this.state === '暂停') { this.resume = ['TASK', { resume: true }]; this.event('电量低，停在两件事之间去充电'); this._switch('CHARGE', { reason: 'low battery' }); }
            else if (this.state === '求助') this._fault('payload stuck');
            else {
                this.taskDone = true; this.event('任务结束');
                if (this.cfg.auto_charge && this.bat.est < this.cfg.soc_full - 0.05) { this.resume = null; this._switch('CHARGE', { reason: 'task done' }); } else this._switch('IDLE');
            }
        } else if (mode === 'SCAN') {
            if (!this.scanDone && this._soft()) { this.resume = ['SCAN', Object.assign({}, this.kw)]; this.event('电量低，停在两个扫描点之间去充电'); this._switch('CHARGE', { reason: 'low battery' }); } else this._switch('IDLE');
        } else if (mode === 'NAV') {
            if ((this.navQueue.length || this.navExplore) && this._soft()) { this.resume = ['NAV', { explore: this.navExplore }]; this.event('电量低，停在两个目标之间去充电'); this._switch('CHARGE', { reason: 'low battery' }); } else this._switch('IDLE');
        } else if (mode === 'CHARGE') {
            if (this.fault) return; const res = this.resume; this.resume = null;
            if (res && !this._soft()) this._switch(res[0], res[1]); else this._switch('IDLE');
        } else if (mode === 'RIGHT') {
            const res = this.resumeRight; this.resumeRight = null; if (this.fault) return;
            if (!res || ['IDLE', 'RIGHT', 'FAULT'].includes(res[0])) this._switch('IDLE');
            else { const kw = Object.assign({}, res[1]); if (res[0] === 'TASK') Object.assign(kw, { resume: true, handback: true }); if (res[0] === 'CHARGE' && kw.reason === undefined) kw.reason = 'after fall'; this._switch(res[0], kw); }
        } else this._switch('IDLE');
    }

    // ---- motion helpers
    cmd(vx, vy, wz) { this.wheelTarget = { vx, vy, wz }; }          // body frame
    stop() { this.cmd(0, 0, 0); }
    *wait(sec) { const t0 = this.t; while (this.t - t0 < sec) yield; }
    /** drive to a point along a planned path (re-planned every second, people's legs are kept out of the map); returns the goal reached, or null (this.abort says why) */
    *goto(goalFn, opts = {}) {
        const vmax = opts.vmax || 0.14, timeout = opts.timeout || 120, t0 = this.t; let path = null, replan = -1e9, stuck = 0, lastD = 1e9, lastT = this.t;
        for (;;) {
            if (this.t - t0 > timeout) { this.abort = 'timeout'; this.path = null; return null; }
            const goal = goalFn(); if (!goal) { this.abort = 'no goal'; return null; }
            const [gx, gy, gth, tol] = goal; const dG = hypot(gx - this.x, gy - this.y);
            if (this.t - replan > 1.0 || !path) { path = this.world.plan([this.x, this.y], [gx, gy], this.dynZones); replan = this.t; if (!path) { this.abort = 'no path'; this.path = null; return null; } this.path = path; }
            const endOk = dG < Math.max(tol || 0.07, 0.04);
            if (endOk) {
                if (gth === null || gth === undefined || Math.abs(wrap(gth - this.th)) < 0.06) { this.stop(); this.path = null; return goal; }
                this.cmd(0, 0, clamp(2.2 * wrap(gth - this.th), -1.0, 1.0)); yield; continue;
            }
            let k = 1; while (k < path.length - 1 && hypot(path[k][0] - this.x, path[k][1] - this.y) < 0.12) k++;
            const wp = path[Math.min(k, path.length - 1)], dx = wp[0] - this.x, dy = wp[1] - this.y, d = Math.max(hypot(dx, dy), 1e-6);
            let dEnd = hypot(path[path.length - 1][0] - this.x, path[path.length - 1][1] - this.y); const v = Math.min(vmax, 0.05 + 0.9 * dEnd);
            const wx = dx / d * v, wy = dy / d * v, c = Math.cos(this.th), s = Math.sin(this.th);
            const heading = (gth !== null && gth !== undefined && dEnd < 0.4) ? gth : (opts.faceGoal ? Math.atan2(gy - this.y, gx - this.x) : Math.atan2(dy, dx));
            this.cmd(c * wx + s * wy, -s * wx + c * wy, clamp(2.2 * wrap(heading - this.th), -1.0, 1.0));
            if (this.t - lastT > 6) { if (lastD - dG < 0.03 && !this.guard.halted) { this.abort = 'stuck'; this.path = null; return null; } lastD = dG; lastT = this.t; }
            yield;
        }
    }
    *spin(sec = 7.6) { this._spinning = true; const t0 = this.t; while (this.t - t0 < sec) { this.cmd(0, 0, 2 * PI / sec); yield; } this._spinning = false; this.stop(); }
    *pullOut() { const prev = this.state; this.state = '离开充电座'; const y1 = this.dockPose[1] - 0.357; while (this.y > y1 + 0.01) { this.cmd(0.067, 0, 0); yield; } this.stop(); this.state = prev; }
    *dockIn() {
        // to the spot in front of the dock, then reverse into it
        this.state = '去充电座'; const fx = this.dockPose[0], fy = this.dockPose[1] - 0.36;
        const r = yield* this.goto(() => [fx, fy, this.dockPose[2], 0.05], { vmax: 0.14, timeout: 90 });
        if (!r) return false;
        this.state = '对接'; const t0 = this.t;
        while (this.y < this.dockPose[1] - 0.003) { this.cmd(-0.04, 0, clamp(2.2 * wrap(this.dockPose[2] - this.th), -0.3, 0.3)); if (this.t - t0 > 30) return false; yield; }
        this.x = this.dockPose[0]; this.y = this.dockPose[1]; this.stop(); return true;
    }

    // ---- the modes
    *_gen_idle() {
        this.state = '待命';
        for (;;) {
            this.stop(); if (this.state !== '待命') this.state = '待命';
            if (this.cfg.auto_charge && this.bat.est < this.cfg.soc_soft) { this.resume = null; this.event('待命时电量低，去充电'); this._switch('CHARGE', { reason: 'low battery while idle' }); return; }
            yield;
        }
    }
    *_gen_fault() { this.state = '求助'; for (;;) { this.stop(); yield; } }
    *_handbackGen() {
        this.state = '交接'; yield* this.wait(0.8);
        if (this.held !== null) { const o = this.world.objects[this.held]; o.status = 'on_floor'; o.x = this.x + 0.2 * Math.cos(this.th); o.y = this.y + 0.2 * Math.sin(this.th); this.held = null; }
        this.targetObj = null; this._handback = false;
    }
    *_gen_task(kw) {
        if (this.atDock()) yield* this.pullOut();
        let resume = kw.resume; if (resume === undefined) resume = this._taskStarted || this._looked;
        if (kw.handback || (kw.handback === undefined && this._handback)) yield* this._handbackGen();
        this._taskStarted = true;
        yield* this._run(resume);
    }
    /** controller.Agent.run: look round once, then for each thing: go to it, pick it up, take it to the basket, let go */
    *_run(resume) {
        const W = this.world, T = this.D.grasp;
        if (!resume && !this._looked) { this.state = '巡视'; yield* this.wait(0.5); yield* this.spin(); this._looked = true; }
        this._looked = true; const visited = this._visited = this._visited || [[this.x, this.y]];
        for (;;) {
            if (this._soft()) { this.state = '暂停'; this.stop(); return; }
            const left = W.objects.filter(o => o.status === 'on_floor' && !o.gaveUp);
            if (!left.length) {
                // everything it knows of is in the basket: look round the rest of the room once, in case something else is lying about, then go back to the start
                if (!this._sweep) this._sweep = [[this.x, this.y]];
                const spot = this._nextLookout(this._sweep, 0.85);
                if (spot) { this._sweep.push(spot); this.state = '巡视'; const r = yield* this.goto(() => [spot[0], spot[1], null, 0.12], { timeout: 80 }); if (r) yield* this.spin(7.6); continue; }
                this._sweep = null; this.state = '回位'; const h = [W.home[0], W.home[1] - 0.1, null, 0.12]; yield* this.goto(() => h, { timeout: 60 });
                this.state = '完成'; this.stop(); return;
            }
            const near = o => this.personZones.some(z => hypot(o.x - z[0], o.y - z[1]) < z[2]);
            const ok = left.filter(o => o.seen && !near(o)); const skipped = left.filter(o => o.seen && near(o));
            let tgt = null, bd = 1e9; for (const o of ok) { const d = hypot(o.x - this.x, o.y - this.y); if (d < bd) { bd = d; tgt = o; } }
            if (!tgt) {
                if (skipped.length && left.every(o => !o.seen || near(o))) { this.zoneSkipT = this.t; this.state = '等待'; yield* this.wait(1.0); continue; }
                this.state = '导航'; const spot = this._nextLookout(visited); if (!spot) { for (const o of left) o.gaveUp = true; continue; }
                visited.push(spot); const r = yield* this.goto(() => [spot[0], spot[1], null, 0.12], { timeout: 80 }); if (r) { this.state = '巡视'; yield* this.spin(7.6); } continue;
            }
            const ok2 = yield* this._pickAndPlace(tgt, T);
            if (ok2 === 'fault') { this.state = '求助'; return; }
        }
    }
    /** where to look next: the nearest place that none of the looks so far covers (it does not know where the things are: it only knows what it has not looked at) */
    _nextLookout(visited, radius = 1.05) {
        const W = this.world; let best = null, bd = 1e9;
        for (let x = 0.4; x < W.W - 0.2; x += 0.4) for (let y = 0.4; y < W.H - 0.2; y += 0.4) {
            if (!W.isFree(x, y)) continue; if (visited.some(v => hypot(v[0] - x, v[1] - y) < radius)) continue;
            const d = hypot(this.x - x, this.y - y); if (d < bd) { bd = d; best = [x, y]; }
        }
        return best;
    }
    stance(o) {       // where the robot stands to pick a thing up: 0.24 m from it on the side the robot comes from, with a free spot
        const W = this.world, a0 = Math.atan2(this.y - o.y, this.x - o.x);
        for (const da of [0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.2, -2.2, PI]) { const a = a0 + da, sx = o.x + 0.24 * Math.cos(a), sy = o.y + 0.24 * Math.sin(a); if (W.isFree(sx, sy)) return [sx, sy, Math.atan2(o.y - sy, o.x - sx)]; }
        return null;
    }
    basketStance() {
        const W = this.world, b = W.basket, sides = [[0, -1], [-1, 0], [0, 1], [1, 0]];
        let best = null, bd = 1e9;
        for (const [sx, sy] of sides) {
            const x = b.x + sx * (b.sx / 2 + 0.29), y = b.y + sy * (b.sy / 2 + 0.29); if (!W.isFree(x, y)) continue;
            const d = hypot(x - this.x, y - this.y); if (d < bd) { bd = d; best = [x, y, Math.atan2(b.y - y, b.x - x)]; }
        }
        return best;
    }
    *_pickAndPlace(o, T) {
        const W = this.world, rate = T[o.kind] ? T[o.kind].success / T[o.kind].n : 0.85;
        const st = this.stance(o); if (!st) { o.gaveUp = true; this.log(`${KIND_CN[o.kind]}：没有能站的位置，放弃`); return false; }
        this.state = '导航'; this.targetObj = o.id; this.log(`前往 ${KIND_CN[o.kind]}`);
        const r = yield* this.goto(() => [st[0], st[1], st[2], 0.05], { vmax: 0.14, timeout: 100 });
        if (!r) { o.fails++; if (o.fails >= 3) { o.gaveUp = true; this.log(`${KIND_CN[o.kind]}：到不了，放弃（${this.abort}）`); } return false; }
        this.state = '抓取'; this._armBusyUntil = this.t + 9.6; this.armBusy = true; yield* this.wait(9.6); this.armBusy = false;
        if (this.rnd() > rate) { o.fails++; this.log(`${KIND_CN[o.kind]}：抓取时滑脱`); if (o.fails >= 4) { o.gaveUp = true; this.log(`${KIND_CN[o.kind]}：放弃`); } return false; }
        o.status = 'held'; this.held = o.id; this.log(`夹住了 ${KIND_CN[o.kind]}`);
        this.state = '搬运'; const bs = this.basketStance(); if (!bs) { this.log('筐前没有位置'); return 'fault'; }
        const r2 = yield* this.goto(() => [bs[0], bs[1], bs[2], 0.05], { vmax: 0.13, timeout: 120 });
        if (!r2) { o.status = 'on_floor'; o.x = this.x; o.y = this.y; this.held = null; o.fails++; return false; }
        this.state = '放置'; this._armBusyUntil = this.t + 8.4; this.armBusy = true; yield* this.wait(8.4); this.armBusy = false;
        const b = W.basket; o.status = 'in_basket'; o.x = b.x + (this.rnd() - 0.5) * (b.sx - 0.1); o.y = b.y + (this.rnd() - 0.5) * (b.sy - 0.1); this.held = null; this.targetObj = null; this.stats.delivered++; (this.deliveries = this.deliveries || []).push(this.t);
        this.log(`已放进收纳筐（${this.delivered()}/${W.objects.length}）`); return true;
    }
    *_gen_nav(kw) {
        const explore = !!kw.explore;
        if (this.atDock()) yield* this.pullOut();
        if (!this._looked && !explore) { this.state = '巡视'; this.event('第一次导航，先原地环顾建图'); yield* this.wait(0.5); yield* this.spin(); this._looked = true; }
        this.state = '导航';
        if (explore) {
            this.event('自主巡视，只建图不捡东西'); const visited = [[this.x, this.y]];
            for (;;) {
                if (this._soft()) { this.stop(); return; }
                const spot = this._nextLookout(visited); if (!spot) break; visited.push(spot);
                const r = yield* this.goto(() => [spot[0], spot[1], null, 0.12], { timeout: 80 }); if (r) { this.state = '巡视'; yield* this.spin(7.6); this.state = '导航'; }
            }
            this.navExplore = false; this.event('整个房间看过了'); this.navResults.push({ t: this.t, goal: 'explore', result: 'done' }); return;
        }
        while (this.navQueue.length) {
            if (this._soft()) return;
            const g = this.navQueue[0]; this.event(`去 (${g.x.toFixed(2)}, ${g.y.toFixed(2)})` + (g.th === null || g.th === undefined ? '' : ` 朝向 ${Math.round(g.th * 180 / PI)}°`)); this.state = '导航'; const t0 = this.t;
            const r = yield* this.goto(() => [g.x, g.y, g.th, g.th === null || g.th === undefined ? g.tol : 0.05], { timeout: this.cfg.nav_timeout });
            const res = { t: this.t, goal: [g.x, g.y, g.th], result: r ? 'arrived' : this.abort, secs: Math.round((this.t - t0) * 10) / 10 }; this.navResults.push(res); this.stats[r ? 'nav_ok' : 'nav_fail']++;
            this.event(`${r ? '到了' : '没到：' + this.abort}（${res.secs} 秒）`); this.navQueue.shift();
        }
    }
    *_gen_scan() {
        if (this.atDock()) yield* this.pullOut();
        const W = this.world, t0 = this.t; this.scanDone = false; let stops = 0;
        if (!this._looked) { this.state = '扫描'; yield* this.wait(0.5); yield* this.spin(); this._looked = true; }
        this.event('开始扫描家里（nbv）'); const visited = [[this.x, this.y]];
        const cand = []; for (let x = 0.5; x < W.W; x += 0.5) for (let y = 0.5; y < W.H; y += 0.5) if (W.isFree(x, y)) cand.push([x, y]);
        while (stops < 6) {
            if (this._soft()) return;
            this.state = '扫描'; let best = null, bs = -1;
            for (const c of cand) { const dv = Math.min(...visited.map(v => hypot(v[0] - c[0], v[1] - c[1]))); if (dv > bs && dv > 1.1) { bs = dv; best = c; } }
            if (!best) break;
            const r = yield* this.goto(() => [best[0], best[1], null, 0.10], { timeout: 60 }); if (!r) { cand.splice(cand.indexOf(best), 1); continue; }
            yield* this.spin(); stops++; visited.push([this.x, this.y]);
        }
        this.scanDone = true; let seen = 0, tot = 0;
        for (let x = 0.1; x < W.W; x += 0.1) for (let y = 0.1; y < W.H; y += 0.1) { if (W.clearance(x, y) < 0.0) continue; tot++; if (visited.some(v => hypot(v[0] - x, v[1] - y) < 2.4 && W.sees(v[0], v[1], x, y, 0.3))) seen++; }
        for (const o of W.objects) if (o.status === 'on_floor' && visited.some(v => hypot(v[0] - o.x, v[1] - o.y) < 2.4 && W.sees(v[0], v[1], o.x, o.y, 0.12))) { o.seen = true; o.lastSeen = this.t; }
        this.event(`扫描完成：${stops} 个停靠点，${Math.round(this.t - t0)} 秒；家具：${this.placeNames().join('、')}`);
    }
    *_gen_fetch(kw) {
        const o = this.world.objects[kw.k]; if (!o) { this.event('要拿的东西不在记忆里'); return; }
        if (this.atDock()) yield* this.pullOut();
        this.event(`去拿 #${o.id}（${KIND_CN[o.kind]}）`);
        const ok = yield* this._pickAndPlace(o, this.D.grasp); this.event(ok === true ? '拿到并送进了收纳筐' : '没拿成');
    }
    *_gen_teleop() {
        const tc = this.tele, cfg = this.cfg; this.state = '遥控'; this.path = null; tc.vx = tc.vy = tc.wz = 0; this.event('遥控开始' + (tc.safety ? '（带安全层）' : '（直连，无安全层）'));
        let stale = true; const sg = this.guardOn;
        for (;;) {
            const live = this.t - tc.t < cfg.deadman; if (live) stale = false; else if (!stale) { stale = true; this.event('指令超时，停车（死人开关）'); }
            let vx = live ? tc.vx : 0, vy = live ? tc.vy : 0, wz = live ? tc.wz : 0; const sp = hypot(vx, vy), vm = this.D.limits.v_max;
            if (sp > vm) { vx *= vm / sp; vy *= vm / sp; } wz = clamp(wz, -this.D.limits.w_max, this.D.limits.w_max);
            this.cmd(vx, vy, wz);
            this.teleSafety = tc.safety;
            yield;
        }
    }
    *_gen_charge(kw) {
        const b = this.bat; this.event(`去充电（${kw.reason || ''}），电量 ${Math.round(100 * b.est)}%`);
        if (!this.dockConnected()) {
            if (this.held !== null) { /* the real robot puts down what it holds before it goes (the job's own safe point does that); here the soft stop already came between two things */ }
            const ok = yield* this.dockIn(); if (!ok) { this.stats.dock_fail++; this._fault('dock unreachable'); return; }
        }
        this.stop(); this.state = '充电'; yield* this.wait(1.0);
        const soc0 = b.soc; let secs = 0;
        while (b.soc < this.cfg.soc_full) { let more = false; for (let i = 0; i < 4; i++) { if (!b.chargeStep(5.0)) break; secs += 5; more = true; } if (!more) break; yield; }
        this.stats.charges++; this.stats.charge_min += secs / 60;
        this.event(`充电 ${Math.round(100 * soc0)}% → ${Math.round(100 * b.soc)}%，用了 ${Math.round(secs / 60)} 分钟（沙盘里快进，物理时间没走）`); this.warn = null;
        if (this.resume !== null) yield* this.pullOut();
    }
    // ---- being knocked over and getting up: the stages and their length are those of the recorded run (docs/playground.md)
    push(kind) { if (this.fallT !== null && this.mode === 'RIGHT') return; this.fallT = this.t; this.fallKind = kind; this.fallPhase = 'down'; this.say('系统', `推了一下（${kind}）。看它是不是自己起来。`); }
    *_gen_right() {
        this.state = '起身'; this.held !== null && (this._dropHeld());
        yield* this.wait(2.2); this.event('静止了，姿态：side_r');
        const tries = this.failRight ? 3 : (this.rnd() < 0.5 ? 1 : 2);
        for (let n = 1; n <= tries; n++) {
            this.event(`3 秒后开始起身（第 ${n} 次）`); yield* this.wait(3.0);
            if (n === tries && !this.failRight) this.fallPhase = 'up'; yield* this.wait(n === tries && !this.failRight ? 11.7 : 9.4);
            if (!(n === tries && !this.failRight)) this.event(`静止了，姿态：side_l`);
        }
        if (this.failRight) { this.stats.right_failed++; this.event('起不来：三次都没成功'); this._fault('fall'); return; }
        this.fallT = null; this.tilt = 0; this.stats.righted++; this.event('站起来了');
        this.state = '重新定位'; yield* this.wait(2.5); this.event('重新定位成功（沙盘里用的是假设的结果）'); yield* this.wait(2.2);
    }
    _dropHeld() { const o = this.world.objects[this.held]; if (!o) return; o.status = 'on_floor'; o.x = this.x + 0.2 * Math.cos(this.th); o.y = this.y + 0.2 * Math.sin(this.th); this.held = null; }
}
XS.Sim = Sim;
})(typeof window !== 'undefined' ? window : globalThis);
