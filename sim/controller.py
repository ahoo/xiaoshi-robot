"""Autonomous fetch controller: lidar mapping, A* planning, holonomic path following, grasp and place.

The control code only reads what a real robot could read: lidar ranges, joint feedback, wheel encoders, a pose
estimate and camera detections. Two of those are *sensor models* fed from simulator ground truth plus noise, because
this project does not simulate images or scan matching:
  * pose estimate  = true pose + slowly varying noise (stands in for lidar localisation such as AMCL)
  * camera objects = true object pose + range-dependent noise, only inside the field of view and with line of sight
Their noise levels are parameters below and are things to measure on the real robot.
"""
import heapq
import math

import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt

from model import BASKET, FURNITURE, HOME, OBJECTS, OBJ_HALF_H, P, ROOM_H, ROOM_W
from robot import LIDAR_RANGE, Q_CARRY
from skills import carry_pose, grasp, holding, place, plan_grasp, wait, wait_arm

CS, GX, GY = 0.05, 80, 60
CCS, CGX, CGY = 0.25, 16, 12
PASS = 0.215                 # m: a cell is drivable if the nearest obstacle is at least this far (robot radius 0.17 + margin)
V_MAX, W_MAX = 0.18, 1.0
CAM_HFOV, CAM_RANGE = math.radians(44), 2.2
STANDOFF = 0.235             # robot centre to object when grasping
LOC_SIGMA_XY, LOC_SIGMA_TH = 0.015, math.radians(1.0)
DET_SIGMA0, DET_SIGMA_K = 0.005, 0.015       # camera position noise: 5 mm + 1.5 % of range
CLOSE_SIGMA, YAW_SIGMA = 0.004, 0.10         # close-range position noise; yaw noise of a far-away look
CLOSE_YAW_SIGMA = 0.05                       # yaw noise of the close-range look
T_MAX = 600.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Agent:
    def __init__(self, robot, seed=0, verbose=False):
        self.r = robot; self.rng = np.random.default_rng(seed + 7); self.verbose = verbose
        self.lo = np.zeros((GY, GX), np.int8)
        self.cov = np.zeros((CGY, CGX), bool)
        self.memory = {}
        self.noise = np.zeros(3)
        self.path = None; self.scan = None; self.state = '启动'; self.target = None; self.log_lines = []
        self.stats = dict(grasp_attempts=0, grasp_fail=0, dropped=0, missed_place=0, replans=0, stuck=0, collisions=0, floor_touch=0, max_tilt=0.0, dist=0.0)
        self._tick = 0; self._last_xy = np.array(robot.true_pose()[:2]); self._coll_t = -9.0; self._floor_t = -9.0
        m = robot.m
        self._world_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self._geomid = np.zeros(1, dtype=np.int32)
        self._obj_geom = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f'obj{i}') for i in range(len(robot.objects))]
        self._floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self._basket_cells = [(j, i) for j in range(GY) for i in range(GX)
                              if abs((i + 0.5) * CS - BASKET['x']) <= BASKET['sx'] / 2 and abs((j + 0.5) * CS - BASKET['y']) <= BASKET['sy'] / 2]
        self._cx, self._cy = np.meshgrid((np.arange(GX) + 0.5) * CS, (np.arange(GY) + 0.5) * CS)
        self.sense()

    # ------------------------------------------------------------------ sensing
    def log(self, msg):
        self.log_lines.append((round(self.r.t, 1), msg))
        if self.verbose: print(f'{self.r.t:6.1f}s  {msg}')

    def pose(self):
        x, y, th = self.r.true_pose()
        return x + self.noise[0], y + self.noise[1], th + self.noise[2]

    def _los(self, p0, p1, target_geom=-1):
        v = np.asarray(p1, float) - p0; d = np.linalg.norm(v)
        if d < 1e-6: return True
        dist = mujoco.mj_ray(self.r.m, self.r.d, np.asarray(p0, float), v / d, self._world_group, 1, -1, self._geomid)
        return dist < 0 or dist >= d - 0.03 or self._geomid[0] == target_geom

    def sense(self):
        r = self.r
        sig = np.array([LOC_SIGMA_XY, LOC_SIGMA_XY, LOC_SIGMA_TH])
        self.noise = 0.97 * self.noise + math.sqrt(1 - 0.97 ** 2) * sig * self.rng.normal(size=3)
        x, y, th = self.pose()
        # --- lidar -> occupancy log-odds
        ang, rng_ = r.lidar()
        lx, ly = x + math.cos(th) * P['lidar_pos'][0], y + math.sin(th) * P['lidar_pos'][0]
        fin = np.isfinite(rng_); a = ang[fin] + th; rr = rng_[fin]
        self.scan = (lx, ly, a, rr)
        if len(rr):
            ts = np.arange(0.06, LIDAR_RANGE, 0.04)[None, :]
            mask = ts < (rr[:, None] - 0.05)
            px = (lx + np.cos(a)[:, None] * ts)[mask]; py = (ly + np.sin(a)[:, None] * ts)[mask]
            ci = np.floor(px / CS).astype(int); cj = np.floor(py / CS).astype(int)
            ok = (ci >= 0) & (ci < GX) & (cj >= 0) & (cj < GY)
            flat = np.unique(cj[ok] * GX + ci[ok]); lo = self.lo.reshape(-1)
            lo[flat] = np.maximum(lo[flat] - 1, -3)
            hi = np.floor((lx + np.cos(a) * (rr + 0.015)) / CS).astype(int); hj = np.floor((ly + np.sin(a) * (rr + 0.015)) / CS).astype(int)
            ok = (hi >= 0) & (hi < GX) & (hj >= 0) & (hj < GY)
            flat = np.unique(hj[ok] * GX + hi[ok])
            lo[flat] = np.minimum(lo[flat].astype(int) + 6, 12).astype(np.int8)
        # --- camera: coverage and object detections (ground truth + noise, gated by field of view and line of sight)
        tx, ty, tth = r.true_pose()
        cam = np.array([tx + math.cos(tth) * 0.108, ty + math.sin(tth) * 0.108, 0.126])
        for j in range(CGY):
            for i in range(CGX):
                if self.cov[j, i]: continue
                cx, cy = (i + 0.5) * CCS, (j + 0.5) * CCS; d = math.hypot(cx - cam[0], cy - cam[1])
                if d < 0.3 or (d < CAM_RANGE and abs(wrap(math.atan2(cy - cam[1], cx - cam[0]) - tth)) < CAM_HFOV and self._los(cam, [cx, cy, 0.05])):
                    self.cov[j, i] = True
        self.visible = {}
        for k in range(len(r.objects)):
            e = self.memory.get(k)
            if e and e['status'] in ('held', 'delivered'): continue
            p = r.obj_pos(k); d = math.hypot(p[0] - cam[0], p[1] - cam[1])
            if d > CAM_RANGE or d < 0.05 or abs(wrap(math.atan2(p[1] - cam[1], p[0] - cam[0]) - tth)) > CAM_HFOV: continue
            if self._in_basket(p) or not self._los(cam, p, self._obj_geom[k]): continue
            s = DET_SIGMA0 + DET_SIGMA_K * d
            # a detection is relative to the robot, so it inherits the robot's own pose error when put on the map
            mx, my = p[0] + self.noise[0] + self.rng.normal(0, s), p[1] + self.noise[1] + self.rng.normal(0, s)
            myaw = r.obj_yaw(k) + self.rng.normal(0, YAW_SIGMA)
            self.visible[k] = d
            if e is None:
                self.memory[k] = dict(kind=r.objects[k][0], x=mx, y=my, yaw=myaw, sigma=s, status='pending', retry_at=0.0, fails=0)
                self.log(f'识别到 {OBJECTS[r.objects[k][0]]["cn"]}，距离 {d:.2f} m')
            elif s <= e['sigma'] * 1.1 or math.hypot(mx - e['x'], my - e['y']) > 3 * e['sigma'] + 0.03:
                e.update(x=mx, y=my, yaw=myaw, sigma=s)

    def _in_basket(self, p):
        return abs(p[0] - BASKET['x']) < BASKET['sx'] / 2 and abs(p[1] - BASKET['y']) < BASKET['sy'] / 2 and p[2] < BASKET['h'] + 0.03

    def after_step(self):
        """Called after every 20 ms physics step: sensors at 10 Hz, plus bookkeeping for the score sheet (ground truth)."""
        r = self.r; self._tick += 1
        if self._tick % 5 == 0: self.sense()
        xy = np.array(r.true_pose()[:2]); self.stats['dist'] += float(np.linalg.norm(xy - self._last_xy)); self._last_xy = xy
        tilt = math.degrees(r.tilt())
        if tilt > 3 and tilt > self.stats['max_tilt'] + 0.5: self.log('机身倾斜 %.1f°（%s）' % (tilt, self.state))
        self.stats['max_tilt'] = max(self.stats['max_tilt'], tilt)
        d = r.d
        for i in range(d.ncon):
            c = d.contact[i]; b1, b2 = r.m.geom_bodyid[c.geom1], r.m.geom_bodyid[c.geom2]
            rob1, rob2 = r.m.body_rootid[b1] == r.chassis, r.m.body_rootid[b2] == r.chassis
            if rob1 == rob2: continue
            other, mine = (c.geom2, c.geom1) if rob1 else (c.geom1, c.geom2)
            if r.m.geom_bodyid[other] != 0: continue                      # only static world geometry counts
            is_roller = r.m.geom_group[mine] == 1
            if other == self._floor:
                if not is_roller and r.t - self._floor_t > 1.0: self._floor_t = r.t; self.stats['floor_touch'] += 1
            elif r.t - self._coll_t > 1.0:
                self._coll_t = r.t; self.stats['collisions'] += 1
                self.log('碰到 ' + (mujoco.mj_id2name(r.m, mujoco.mjtObj.mjOBJ_GEOM, other) or '?'))

    # ------------------------------------------------------------------ planning
    def costmap(self, ignore=None):
        occ = self.lo > 0
        for k, e in self.memory.items():
            if k != ignore and e['status'] == 'pending':
                i, j = int(e['x'] / CS), int(e['y'] / CS)
                if 0 <= i < GX and 0 <= j < GY: occ[j, i] = True
        for j, i in self._basket_cells: occ[j, i] = True
        D = distance_transform_edt(~np.pad(occ, 1, constant_values=True))[1:-1, 1:-1] * CS
        if ignore is not None and ignore in self.memory:
            # the target is not a wall, but the robot must not drive over it on the way to its far side:
            # keep the robot centre at least 19.5 cm away (the grasp stance is at 23.5 cm)
            e = self.memory[ignore]
            D = np.minimum(D, np.hypot(self._cx - e['x'], self._cy - e['y']) + (PASS - 0.195))
        return D

    def astar(self, D, goals):
        """goals: list of (x, y, heading or None, radius). Returns (path, goal) or None."""
        x, y, _ = self.pose()
        si, sj = min(GX - 1, max(0, int(x / CS))), min(GY - 1, max(0, int(y / CS)))
        gx = np.array([g[0] for g in goals]); gy = np.array([g[1] for g in goals]); gr = np.array([max(g[3], 0.036) for g in goals])

        def h(i, j):
            return max(0.0, float(np.min(np.hypot(gx - (i + 0.5) * CS, gy - (j + 0.5) * CS) - gr)))

        def enter(a, b, near):
            if D[b] >= PASS: return True
            return near and D[a] < PASS and D[b] >= D[a] - 1e-6 and D[b] > 0.11
        g = {(sj, si): 0.0}; came = {}; heap = [(h(si, sj), sj, si)]; closed = set(); found = None
        while heap:
            _, j, i = heapq.heappop(heap)
            if (j, i) in closed: continue
            closed.add((j, i))
            cx, cy = (i + 0.5) * CS, (j + 0.5) * CS
            if D[j, i] >= PASS:
                dd = np.hypot(gx - cx, gy - cy) - gr
                if dd.min() <= 0: found = (j, i, int(dd.argmin())); break
            near = abs(i - si) <= 8 and abs(j - sj) <= 8
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ni, nj = i + di, j + dj
                if not (0 <= ni < GX and 0 <= nj < GY) or (nj, ni) in closed: continue
                if not enter((j, i), (nj, ni), near): continue
                if di and dj and (not enter((j, i), (j, ni), near) or not enter((j, i), (nj, i), near)): continue
                dn = D[nj, ni]
                ng = g[(j, i)] + CS * (1.4142 if di and dj else 1.0) * (1 + (2.5 * (0.38 - dn) / 0.165 if dn < 0.38 else 0) + (6 if dn < PASS else 0))
                if ng < g.get((nj, ni), 1e9):
                    g[(nj, ni)] = ng; came[(nj, ni)] = (j, i); heapq.heappush(heap, (ng + h(ni, nj), nj, ni))
        if found is None: return None
        j, i, gi = found; pts = []
        node = (j, i)
        while node in came or node == (sj, si):
            pts.append(((node[1] + 0.5) * CS, (node[0] + 0.5) * CS))
            if node == (sj, si): break
            node = came[node]
        pts.reverse(); pts[0] = (x, y)
        goal = goals[gi]
        if goal[3] <= 0.036 and math.hypot(pts[-1][0] - goal[0], pts[-1][1] - goal[1]) > 1e-6: pts.append((goal[0], goal[1]))
        return self._smooth(D, pts), goal

    def _dat(self, D, x, y):
        i, j = int(x / CS), int(y / CS)
        return D[j, i] if 0 <= i < GX and 0 <= j < GY else 0.0

    def _clear(self, D, a, b):
        da, db = self._dat(D, *a), self._dat(D, *b)
        if da < PASS or db < PASS: return False
        need = min(0.27, da, db) - 1e-4; L = math.hypot(b[0] - a[0], b[1] - a[1]); n = int(L / 0.025) + 1
        return all(self._dat(D, a[0] + (b[0] - a[0]) * s / n, a[1] + (b[1] - a[1]) * s / n) >= need for s in range(1, n))

    def _smooth(self, D, pts):
        out = [pts[0]]; i = 0
        while i < len(pts) - 1:
            j = min(len(pts) - 1, i + 60)
            while j > i + 1 and not self._clear(D, pts[i], pts[j]): j -= 1
            out.append(pts[j]); i = j
        return out

    # ------------------------------------------------------------------ motion
    def stop(self):
        self.r.set_twist(0, 0, 0)

    def drive_world(self, vwx, vwy, th_des):
        x, y, th = self.pose()
        c, s = math.cos(th), math.sin(th)
        self.r.set_twist(c * vwx + s * vwy, -s * vwx + c * vwy, float(np.clip(2.2 * wrap(th_des - th), -W_MAX, W_MAX)))

    def goto(self, goal_fn, ignore=None, check=None, timeout=150.0):
        """Plan and drive to one of the goals produced by goal_fn(D). Returns the goal reached, or None.
        `check` is called every tick; if it returns a string the motion is aborted with that reason."""
        t0 = self.r.t; t_plan = -9.0; fails = 0; goal = None; path = None; wp = 1
        prog_t, prog_xy = self.r.t, np.array(self.pose()[:2])
        while True:
            if self.r.t - t0 > timeout: self.stop(); return None
            if check:
                why = check()
                if why: self.stop(); self.abort = why; return None
            if self.r.t - t_plan > 0.6:
                t_plan = self.r.t; D = self.costmap(ignore); goals = goal_fn(D)
                if goal is not None:                       # stick with the goal already chosen while it stays valid
                    keep = [g for g in goals if abs(g[0] - goal[0]) < 0.03 and abs(g[1] - goal[1]) < 0.03]
                    goals = keep or goals
                all_goals = goal_fn(D) if goal is not None else goals
                res = self.astar(D, goals) if goals else None
                if res is None and len(all_goals) > len(goals): res = self.astar(D, all_goals)
                if res is None:
                    fails += 1; self.stop(); path = None
                    if fails >= 3: self.abort = 'no_path'; return None
                else:
                    fails = 0; path, goal = res; wp = 1; self.path = path; self.stats['replans'] += 1
            x, y, th = self.pose()
            if path is None: yield; continue
            end = path[-1]; d_end = math.hypot(end[0] - x, end[1] - y)
            if d_end < (0.05 if goal[3] <= 0.036 else 0.07): break
            while wp < len(path) - 1 and math.hypot(path[wp][0] - x, path[wp][1] - y) < 0.12: wp += 1
            tx, ty = path[min(wp, len(path) - 1)]; d = math.hypot(tx - x, ty - y) + 1e-9
            v = min(V_MAX, 0.05 + 0.9 * d_end)
            if self._dat(D, x, y) < PASS + 0.04: v *= 0.6
            travel = math.atan2(ty - y, tx - x)
            th_des = goal[2] if (goal[2] is not None and d_end < 0.45) else travel
            if abs(wrap(th_des - th)) > 0.9: v *= 0.3
            self.drive_world(v * (tx - x) / d, v * (ty - y) / d, th_des)
            if self.r.t - prog_t > 6.0:
                now = np.array([x, y])
                if np.linalg.norm(now - prog_xy) < 0.04:
                    self.stats['stuck'] += 1; self.log('原地卡住，后退重新规划')
                    self.r.set_twist(-0.08, 0, 0); yield from wait(self.r, 1.2); self.stop(); t_plan = -9.0
                prog_t, prog_xy = self.r.t, now
            yield
        if goal[2] is not None and goal[3] <= 0.036:
            yield from self.servo(goal)
        self.stop(); self.path = None
        return goal

    def servo(self, goal, timeout=7.0):
        t0 = self.r.t; good = 0
        while self.r.t - t0 < timeout:
            x, y, th = self.pose(); ex, ey, eth = goal[0] - x, goal[1] - y, wrap(goal[2] - th)
            d = math.hypot(ex, ey)
            good = good + 1 if (d < 0.012 and abs(eth) < 0.03) else 0
            if good > 10: break
            v = min(0.08, 1.5 * d)
            self.drive_world(v * ex / (d + 1e-9), v * ey / (d + 1e-9), goal[2])
            yield
        self.stop(); yield from wait(self.r, 0.4)

    def spin(self, turns=1.0):
        acc = 0.0; last = self.pose()[2]
        while acc < 2 * math.pi * turns:
            self.r.set_twist(0, 0, 0.9); yield
            th = self.pose()[2]; acc += abs(wrap(th - last)); last = th
        self.stop(); yield from wait(self.r, 0.3)

    # ------------------------------------------------------------------ task pieces
    def standoffs(self, e, D):
        kind = e['kind']
        if kind == 'ball': dirs = [k * math.pi / 6 for k in range(12)]
        elif kind == 'block': dirs = [e['yaw'] + k * math.pi / 2 for k in range(4)]
        else: dirs = [e['yaw'] + math.pi / 2, e['yaw'] - math.pi / 2]
        for extra in ((0.0,), (0.14, -0.14)):
            out = []
            for a0 in dirs:
                for da in extra:
                    a = a0 + da
                    px, py = e['x'] + STANDOFF * math.cos(a), e['y'] + STANDOFF * math.sin(a)
                    # the moving jaw swings out on the far side of the object: that space must be empty too
                    fx, fy = e['x'] - 0.11 * math.cos(a), e['y'] - 0.11 * math.sin(a)
                    if self._dat(D, px, py) >= PASS and self._dat(D, fx, fy) >= 0.075: out.append((px, py, wrap(a + math.pi), 0.0))
            if out: return out
        return []

    def basket_standoffs(self, D):
        b = BASKET; out = []
        for nx, ny, half in ((0, -1, b['sy'] / 2), (0, 1, b['sy'] / 2), (-1, 0, b['sx'] / 2), (1, 0, b['sx'] / 2)):
            d = half + 0.25
            px, py = b['x'] + nx * d, b['y'] + ny * d
            if self._dat(D, px, py) >= PASS: out.append((px, py, math.atan2(-ny, -nx), 0.0))
        return out

    def measure_close(self, k):
        """Close-range look at the object right before grasping. Returns its centre in the chassis frame, or None."""
        r = self.r; p = r.obj_pos(k); x, y, th = r.true_pose()
        c, s = math.cos(th), math.sin(th); dx, dy = p[0] - x, p[1] - y
        rel = np.array([c * dx + s * dy, -s * dx + c * dy, p[2]])
        if not (0.13 < rel[0] < 0.42 and abs(rel[1]) < 0.16 and p[2] < 0.08): return None
        rel[:2] += self.rng.normal(0, CLOSE_SIGMA, 2)
        return rel, wrap(r.obj_yaw(k) + self.rng.normal(0, CLOSE_YAW_SIGMA) - th)

    def fetch(self, k):
        r = self.r; e = self.memory[k]; cn = OBJECTS[e['kind']]['cn']; self.target = k
        tries = 0; restand = 0
        while True:
            self.state = '导航'; self.log(f'前往 {cn}' + (f'（第 {tries + 1} 次）' if tries else ''))
            goal = yield from self.goto(lambda D: self.standoffs(e, D), ignore=k)
            if goal is None:
                e['fails'] += 1; e['retry_at'] = r.t + 30 * e['fails']; e['why'] = '够不到' if self.abort == 'no_path' else '导航超时'
                self.log(f'{cn} {e["why"]}，先跳过'); return False
            self.state = '抓取'
            m = self.measure_close(k)
            if m is None:
                self.log(f'{cn} 不在预期位置，重新找'); del self.memory[k]; self.cov[:] = False; return False
            rel, yaw_rel = m
            x, y, th = self.pose()
            nx, ny = x + math.cos(th) * rel[0] - math.sin(th) * rel[1], y + math.sin(th) * rel[0] + math.cos(th) * rel[1]
            moved = math.hypot(nx - e['x'], ny - e['y'])
            e.update(x=nx, y=ny, yaw=wrap(yaw_rel + th), sigma=CLOSE_SIGMA + LOC_SIGMA_XY)
            mis = 0.0
            if e['kind'] not in ('ball', 'block'):
                radial = math.atan2(rel[1], rel[0] - 0.069)
                mis = abs(wrap(2 * (yaw_rel + math.pi / 2 - radial))) / 2
            # the first stance was computed from a far-away look; now that the object is seen up close, re-stand if it is off
            if restand < 4 and (mis > 0.14 or moved > 0.03 or abs(rel[1]) > 0.06):
                restand += 1; continue
            plan = plan_grasp(rel, e['kind'], mis)
            if plan is None:
                if restand < 4: restand += 1; self.log('手臂够不到这个位置，换个站位'); continue
                e['fails'] += 1; e['retry_at'] = r.t + 40 * e['fails']; e['why'] = '手臂够不到'; self.log(f'{cn} 手臂够不到，先跳过'); return False
            self.stats['grasp_attempts'] += 1; tries += 1
            ok = yield from grasp(r, plan, self.log)
            if ok:
                yield from carry_pose(r)
                ok = holding(r, plan)
                if not ok: self.log('收臂时掉了')
            if ok: break
            self.stats['grasp_fail'] += 1
            r.set_arm(grip=0.3); yield from carry_pose(r, speed=1.0); yield from wait(r, 0.6)
            if tries >= 3:
                e['fails'] += 1; e['retry_at'] = r.t + 40 * e['fails']; e['why'] = '连续 3 次没夹住'
                self.log(f'{cn} 连续 3 次没夹住，先跳过'); return False
        e['status'] = 'held'; self.state = '搬运'; self.log(f'夹住 {cn}，送往收纳筐')
        goal = yield from self.goto(self.basket_standoffs, check=lambda: None if holding(r, plan) else 'dropped')
        if goal is None:
            if self.abort == 'dropped':
                self.stats['dropped'] += 1; self.log(f'路上把 {cn} 掉了')
                del self.memory[k]; self.cov[:] = False        # it will be found again by looking
                r.set_arm(grip=0.3)
                return False
            self.log(f'去收纳筐的路不通，先把 {cn} 放回地上')
            yield from self.put_down(k); return False
        self.state = '放置'
        x, y, th = self.pose()
        bx, by = BASKET['x'] - x, BASKET['y'] - y; dist = math.hypot(bx, by)
        half = BASKET['sx'] / 2 if abs(math.cos(goal[2])) > 0.7 else BASKET['sy'] / 2
        reach = float(np.clip(dist - half + 0.05, 0.24, 0.31))      # drop 5 cm inside the near wall
        c, s = math.cos(th), math.sin(th)
        ux, uy = bx / dist, by / dist; wx, wy = x + ux * reach, y + uy * reach
        rel = (c * (wx - x) + s * (wy - y), -s * (wx - x) + c * (wy - y))
        ok = yield from place(r, rel, BASKET['h'], self.log)
        yield from wait(r, 0.5)
        if self._in_basket(r.obj_pos(k)):
            e['status'] = 'delivered'; self.log(f'{cn} 已放进收纳筐'); return True
        self.stats['missed_place'] += 1; self.log(f'{cn} 没落进筐里'); del self.memory[k]; self.cov[:] = False
        return False

    def put_down(self, k):
        """Set a held object on the floor in front of the robot and remember where it is."""
        from skills import IK
        r = self.r; e = self.memory[k]
        q, _, _ = IK.solve([0.25, 0, 0.06])
        r.set_arm(q5=q, speed=0.6); yield from wait_arm(r)
        r.set_arm(grip=1.0); yield from wait(r, 0.6)
        yield from carry_pose(r, speed=1.0); r.set_arm(grip=0.3)
        x, y, th = self.pose()
        e.update(status='pending', x=x + 0.25 * math.cos(th), y=y + 0.25 * math.sin(th), sigma=0.03, retry_at=r.t + 5)
        # whatever sits closest to the basket is probably what blocks the way in: deal with that first
        cands = [(math.hypot(o['x'] - BASKET['x'], o['y'] - BASKET['y']), j) for j, o in self.memory.items() if o['status'] == 'pending' and j != k]
        if cands: self.priority = min(cands)[1]

    def explore_step(self):
        x, y, th = self.pose(); D = self.costmap()
        cells = []
        for j in range(CGY):
            for i in range(CGX):
                if self.cov[j, i]: continue
                cx, cy = (i + 0.5) * CCS, (j + 0.5) * CCS
                if self._dat(D, cx, cy) < PASS - 0.06: self.cov[j, i] = True; continue
                cells.append((math.hypot(cx - x, cy - y), j, i, cx, cy))
        if not cells: return False
        cells.sort()
        _, j, i, cx, cy = cells[0]
        self.state = '巡视'
        goal = yield from self.goto(lambda D: [(cx, cy, None, 0.55)], check=lambda: 'found' if self.pick() is not None else None, timeout=60)
        if goal is None and self.abort == 'found': return True
        if goal is not None and not self.cov[j, i]: yield from self.spin()
        self.cov[j, i] = True
        return True

    def pick(self):
        x, y, _ = self.pose(); best = None
        p = getattr(self, 'priority', None)
        if p is not None:
            e = self.memory.get(p)
            if e and e['status'] == 'pending' and e['retry_at'] <= self.r.t: return p
            self.priority = None
        for k, e in self.memory.items():
            if e['status'] == 'pending' and e['retry_at'] <= self.r.t:
                d = math.hypot(e['x'] - x, e['y'] - y)
                if best is None or d < best[0]: best = (d, k)
        return None if best is None else best[1]

    def run(self):
        self.abort = None
        yield from wait(self.r, 0.5)
        self.state = '巡视'; self.log('开机，原地环顾建图')
        yield from self.spin()
        while self.r.t < T_MAX:
            k = self.pick()
            if k is not None:
                yield from self.fetch(k); self.target = None
                continue
            more = yield from self.explore_step()
            if not more:
                waiting = [e for e in self.memory.values() if e['status'] == 'pending']
                if not waiting: break
                t_next = min(e['retry_at'] for e in waiting)
                if t_next > T_MAX: break
                self.state = '等待'; yield from wait(self.r, max(0.5, min(5.0, t_next - self.r.t)))
        self.state = '回位'; self.log('没有待办，回起点')
        yield from self.goto(lambda D: [(HOME[0], HOME[1], HOME[2], 0.0)], timeout=90)
        self.state = '完成'
