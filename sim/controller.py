"""Autonomous fetch controller: lidar SLAM, A* planning, holonomic path following, grasp and place.

The control code only uses what the real robot has: wheel encoders (with calibration error and slip), lidar
ranges, joint feedback, and the front RGB-D camera. Localisation is lidar scan matching against the robot's own map
(localization.py); objects are found and measured in rendered depth images (perception.py). The only remaining
idealisation in sensing is the per-pixel object labels (see perception.py). Ground truth is read only by the score
sheet in after_step(), never to make a decision.
"""
import heapq
import math

import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt

from model import BASKET, FURNITURE, HOME, OBJECTS, OBJ_HALF_H, P, ROOM_H, ROOM_W
from robot import LIDAR_RANGE, Q_CARRY
from skills import carry_pose, grasp, holding, misalignment, place, plan_grasp, wait, wait_arm
from localization import Localizer
from perception import FrontCamera

CS, GX, GY = 0.05, 80, 60
CCS, CGX, CGY = 0.25, 16, 12
PASS = 0.215                 # m: a cell is drivable if the nearest mapped obstacle cell is at least this far (0.14 to the wheel rims, plus cell size, pose error and margin)
SAFE_R = 0.45                # m: obstacle points within this radius are checked by the safety layer
BODY_R = 0.15                # m: chassis + wheel rims, as a circle
ARM_BOX = (0.035, 0.228, 0.04)   # folded arm footprint in the chassis frame: x from, x to, half width (measured from the meshes)
HELD_BOX = (0.035, 0.245, 0.08)  # the same with a payload hanging under the gripper
SAFE_GAP = 0.02              # m: the safety layer never lets a motion bring any part of the robot closer than this
MAP_PT_MARGIN = 0.02         # m: allowance for a mapped cell centre lying behind the obstacle surface it stands for
SAFE_HORIZON = (0.15, 0.3)   # s: look-ahead of the safety layer; 0.3 s covers stopping from full speed under the wheel ramp
ARM_SWEEP, HELD_SWEEP = 0.228, 0.258   # m: radius swept by the folded arm when the robot turns on the spot (empty / holding)
V_MAX, W_MAX = 0.18, 1.0
CAM_RANGE = 1.6             # m: beyond this the smallest object (4 cm cube) is under MIN_PX pixels, so 'seen' stops here
STANDOFF = 0.235             # robot centre to object when grasping
T_MAX = 600.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Agent:
    def __init__(self, robot, seed=0, verbose=False):
        self.r = robot; self.rng = np.random.default_rng(seed + 7); self.verbose = verbose
        self.lo = np.zeros((GY, GX), np.int8)
        self.cov = np.zeros((CGY, CGX), bool)
        self.memory = {}
        self.loc = Localizer(robot.true_pose(), CS, (GX * CS, GY * CS))   # starts docked at a known pose; after that, lidar + odometry only
        self.cam = FrontCamera(robot, np.random.default_rng(seed + 11)); self.visible = {}; self._ncam = 0
        self.loc_err = []                                  # (position error m, heading error rad) once a second, for the score sheet
        self.scan_pts = np.zeros((0, 2)); self.map_pts = np.zeros((0, 2)); self.blocked = 0; self.held = False; self.basket_blocked_t = -99.0
        self.path = None; self.scan = None; self.state = '启动'; self.target = None; self.log_lines = []
        self.stats = dict(grasp_attempts=0, grasp_fail=0, dropped=0, missed_place=0, replans=0, stuck=0, collisions=0, floor_touch=0, max_tilt=0.0, dist=0.0)
        self._tick = 0; self._last_xy = np.array(robot.true_pose()[:2]); self._coll_t = -9.0; self._floor_t = -9.0
        m = robot.m
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
        x, y, th = self.loc.est
        return float(x), float(y), float(th)

    def sense(self):
        r = self.r
        self._ncam += 1
        cam_now = self._ncam % 3 == 1                 # camera at 3.3 Hz, lidar at 10 Hz
        # --- lidar: correct the pose against the map, then update the map from the corrected pose
        if self.state in ('抓取', '放置'):          # the arm is out in front of the lidar: neither match nor map
            if cam_now: self._sense_camera()
            return
        ang, rng_ = r.lidar()
        f = np.isfinite(rng_)
        spx, spy = P['lidar_pos'][0] + rng_[f] * np.cos(ang[f]), rng_[f] * np.sin(ang[f])
        self.loc.correct(spx, spy)
        self.loc.add_scan(spx, spy)
        x, y, th = self.pose()
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
            lo[flat] = np.minimum(lo[flat].astype(int) + 6, 20).astype(np.int8)
            wx_, wy_ = lx + np.cos(a) * rr, ly + np.sin(a) * rr
            near = np.hypot(wx_ - x, wy_ - y) < SAFE_R
            self.scan_pts = np.stack([wx_[near], wy_[near]], 1)
        else:
            self.scan_pts = np.zeros((0, 2))
        # mapped obstacle cells near the robot (world frame). The folded arm hides a strip in front of the lidar, and
        # anything seen there earlier is still in the map, so the safety layer uses both.
        i0, i1 = max(0, int((x - SAFE_R) / CS)), min(GX, int((x + SAFE_R) / CS) + 1)
        j0, j1 = max(0, int((y - SAFE_R) / CS)), min(GY, int((y + SAFE_R) / CS) + 1)
        jj, ii = np.nonzero(self.lo[j0:j1, i0:i1] > 0)
        self.map_pts = np.stack([(ii + i0 + 0.5) * CS, (jj + j0 + 0.5) * CS], 1)
        if cam_now: self._sense_camera()

    def _sense_camera(self):
        """Front RGB-D camera: which floor cells it has looked at, and object detections put on the map."""
        r = self.r
        x, y, th = self.pose(); c, s_ = math.cos(th), math.sin(th)
        dets = self.cam.detect()
        # coverage: unseen floor cells whose centre is in the image, not hidden behind anything, and close enough
        cells = [(j, i) for j in range(CGY) for i in range(CGX) if not self.cov[j, i]]
        if cells:
            W_ = np.array([((i + 0.5) * CCS, (j + 0.5) * CCS) for j, i in cells]) - [x, y]
            B = np.stack([c * W_[:, 0] + s_ * W_[:, 1], -s_ * W_[:, 0] + c * W_[:, 1], np.zeros(len(W_))], 1)
            dist = np.hypot(B[:, 0], B[:, 1])
            seen = self.cam.floor_visible(B) & (dist < CAM_RANGE)
            for (j, i), sn, dd in zip(cells, seen, dist):
                if sn or dd < 0.3: self.cov[j, i] = True
        self.visible = {}
        for k, dt in dets.items():
            rel = dt['rel']
            if rel[2] > 0.10: continue                      # up in the air: the object in our own gripper
            mx, my = x + c * rel[0] - s_ * rel[1], y + s_ * rel[0] + c * rel[1]
            if self._in_basket((mx, my, 0.0), margin=0.03): continue
            kind = r.objects[k][0]; cn = OBJECTS[kind]['cn']
            e = self.memory.get(k)
            if e and e['status'] == 'held': continue
            if e and e['status'] == 'delivered':
                self.log(f'看见 {cn} 在筐外的地上，重新去拿'); e.update(status='pending', retry_at=0.0)
            yaw = wrap(dt['yaw4'] + th)
            sig = 0.008 + 0.007 * dt['dist']                 # detector error (bench) plus localisation error
            self.visible[k] = dt['dist']
            if e is None:
                self.memory[k] = dict(kind=kind, x=mx, y=my, yaw=yaw, sigma=sig, status='pending', retry_at=0.0, fails=0)
                self.log(f'识别到 {cn}，距离 {dt["dist"]:.2f} m')
            elif sig <= e['sigma'] * 1.1 or math.hypot(mx - e['x'], my - e['y']) > 3 * e['sigma'] + 0.03:
                e.update(x=mx, y=my, yaw=yaw, sigma=sig)

    def _in_basket(self, p, margin=0.0):
        return abs(p[0] - BASKET['x']) < BASKET['sx'] / 2 + margin and abs(p[1] - BASKET['y']) < BASKET['sy'] / 2 + margin and p[2] < BASKET['h'] + 0.03

    def after_step(self):
        """Called after every 20 ms physics step: sensors at 10 Hz, plus bookkeeping for the score sheet (ground truth)."""
        r = self.r; self._tick += 1
        self.loc.predict(r.odom)
        if self._tick % 5 == 0: self.sense()
        if self._tick % 50 == 0:
            tx, ty, tth = r.true_pose(); ex, ey, eth = self.loc.est
            self.loc_err.append((math.hypot(ex - tx, ey - ty), abs(wrap(eth - tth))))
        xy = np.array(r.true_pose()[:2]); self.stats['dist'] += float(np.linalg.norm(xy - self._last_xy)); self._last_xy = xy
        tilt = math.degrees(r.tilt())
        if tilt > 3 and tilt > self.stats['max_tilt'] + 0.5 and r.t - getattr(self, '_tilt_t', -9) > 2: self._tilt_t = r.t; self.log('机身倾斜超过 3°（%s）' % self.state)
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
        self.D_tall = distance_transform_edt(~np.pad(self.lo > 0, 1, constant_values=True))[1:-1, 1:-1] * CS
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

    def _clearance(self, P, box=None):
        """Distance (m) from points P (N x 2, chassis frame) to the robot's outline: body circle plus the folded arm."""
        x0, x1, hw = box or (HELD_BOX if self.held else ARM_BOX)
        c_body = np.hypot(P[:, 0], P[:, 1]) - BODY_R
        dx = np.maximum(np.maximum(x0 - P[:, 0], P[:, 0] - x1), 0.0); dy = np.maximum(np.abs(P[:, 1]) - hw, 0.0)
        return np.minimum(c_body, np.hypot(dx, dy))

    def cmd(self, vx, vy, wz):
        """Send a body twist through the safety layer. Every 20 ms it predicts where nearby obstacle points end up
        relative to the robot after 0.25 s and 0.5 s of the commanded motion; if any would come closer to the outline
        than SAFE_GAP (or closer than it already is), it falls back to a gentler command, finally to standing still.
        Rotation matters as much as translation: the folded arm sweeps a wider circle than the body."""
        x, y, th = self.pose()
        W = [self.scan_pts, self.map_pts]
        if not any(len(w) for w in W):
            self.r.set_twist(vx, vy, wz); return
        c, s = math.cos(th), math.sin(th)
        P = np.vstack([w for w in W if len(w)]) - [x, y]
        P = np.stack([c * P[:, 0] + s * P[:, 1], -s * P[:, 0] + c * P[:, 1]], 1)
        margin = np.concatenate([np.zeros(len(self.scan_pts)), np.full(len(self.map_pts), MAP_PT_MARGIN)])
        now = self._clearance(P) - margin
        keep = now < 0.12
        if not keep.any():
            self.r.set_twist(vx, vy, wz); self.blocked = 0; return
        P, now, margin = P[keep], now[keep], margin[keep]
        floor_ = np.minimum(now, SAFE_GAP) - 0.002
        # sliding: drop only the part of the velocity that points at the closest obstacle, keep the part along it
        cl = P[np.argmin(now)]; nrm = cl / max(np.hypot(*cl), 1e-6)
        toward = vx * nrm[0] + vy * nrm[1]
        sx, sy = (vx - toward * nrm[0], vy - toward * nrm[1]) if toward > 0 else (vx, vy)
        cands = ((vx, vy, wz), (vx, vy, 0.0), (sx, sy, wz), (sx, sy, 0.0), (0.5 * sx, 0.5 * sy, 0.0), (0.0, 0.0, wz), (0.0, 0.0, 0.0))
        for k, (cx, cy, cw) in enumerate(cands):
            ok = True
            for dt in SAFE_HORIZON:
                a = -cw * dt; ca, sa = math.cos(a), math.sin(a)
                Q = P - [cx * dt, cy * dt]
                Q = np.stack([ca * Q[:, 0] - sa * Q[:, 1], sa * Q[:, 0] + ca * Q[:, 1]], 1)
                if (self._clearance(Q) - margin < floor_).any(): ok = False; break
            if ok:
                self.r.set_twist(cx, cy, cw); self.blocked = self.blocked + 1 if k else 0
                return
        self.r.set_twist(0.0, 0.0, 0.0); self.blocked += 1

    def drive_world(self, vwx, vwy, th_des):
        x, y, th = self.pose()
        c, s = math.cos(th), math.sin(th)
        self.cmd(c * vwx + s * vwy, -s * vwx + c * vwy, float(np.clip(2.2 * wrap(th_des - th), -W_MAX, W_MAX)))

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
                    for _ in range(60): self.cmd(-0.08, 0, 0); yield
                    self.stop(); t_plan = -9.0
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
        acc = 0.0; last = self.pose()[2]; t0 = self.r.t
        while acc < 2 * math.pi * turns and self.r.t - t0 < 2 * math.pi * turns / 0.9 + 3.0:
            self.cmd(0, 0, 0.9); yield
            if self.blocked > 50: break               # the arm would hit something: look around another time
            th = self.pose()[2]; acc += abs(wrap(th - last)); last = th
        self.stop(); yield from wait(self.r, 0.3)

    # ------------------------------------------------------------------ task pieces
    def pose_ok(self, px, py, heading, box, sweep):
        """Can the robot stand at (px, py) and turn by +-sweep around `heading` without the folded arm touching mapped furniture?"""
        i0, i1 = max(0, int((px - 0.4) / CS)), min(GX, int((px + 0.4) / CS) + 1)
        j0, j1 = max(0, int((py - 0.4) / CS)), min(GY, int((py + 0.4) / CS) + 1)
        jj, ii = np.nonzero(self.lo[j0:j1, i0:i1] > 0)
        if not len(ii): return True
        Q = np.stack([(ii + i0 + 0.5) * CS - px, (jj + j0 + 0.5) * CS - py], 1)
        for h in np.linspace(heading - sweep, heading + sweep, 7):
            c, s = math.cos(h), math.sin(h)
            B = np.stack([c * Q[:, 0] + s * Q[:, 1], -s * Q[:, 0] + c * Q[:, 1]], 1)
            if (self._clearance(B, box) - MAP_PT_MARGIN < 0.03).any(): return False
        return True

    def standoffs(self, e, D):
        kind = e['kind']
        if kind in ('ball', 'block'): dirs = [e['yaw'] + k * math.pi / 6 for k in range(12)]   # a cube can be taken across a diagonal too: the jaws open for it
        else: dirs = [e['yaw'] + math.pi / 2, e['yaw'] - math.pi / 2]
        # Square-on stances plus ones turned 0.10 rad either way (still under the re-stand threshold). All of them go to
        # A*, which picks whichever it can reach: a valid square-on stance on the far side of the object is no use.
        out = []
        for a0 in dirs:
            for da in (0.0, 0.10, -0.10):
                a = a0 + da
                px, py = e['x'] + STANDOFF * math.cos(a), e['y'] + STANDOFF * math.sin(a)
                # the moving jaw swings out on the far side of the object: that space must be empty too
                fx, fy = e['x'] - 0.11 * math.cos(a), e['y'] - 0.11 * math.sin(a)
                # and the folded arm must have room to turn a little there without touching furniture
                if self._dat(D, px, py) >= PASS and self._dat(D, fx, fy) >= 0.075 and self.pose_ok(px, py, a + math.pi, ARM_BOX, 0.3):
                    out.append((px, py, wrap(a + math.pi), 0.0))
        return out

    def basket_standoffs(self, D):
        b = BASKET; out = []
        for nx, ny, half in ((0, -1, b['sy'] / 2), (0, 1, b['sy'] / 2), (-1, 0, b['sx'] / 2), (1, 0, b['sx'] / 2)):
            d = half + 0.25
            px, py = b['x'] + nx * d, b['y'] + ny * d
            h = math.atan2(-ny, -nx)
            if self._dat(D, px, py) >= PASS and self.pose_ok(px, py, h, HELD_BOX, 0.5): out.append((px, py, h, 0.0))
        return out

    def measure_close(self, k):
        """Close look with the front camera right before grasping. Returns (centre in the chassis frame, long-axis
        heading in the chassis frame), or None if the object is not where it should be."""
        dets = self.cam.detect()
        d = dets.get(k)
        if d is None or d['n_px'] < 60: return None
        rel = np.array(d['rel'], float); rel[2] = OBJ_HALF_H[self.r.objects[k][0]]
        if not (0.13 < rel[0] < 0.42 and abs(rel[1]) < 0.16): return None
        return rel, d['yaw4']

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
            e.update(x=nx, y=ny, yaw=wrap(yaw_rel + th), sigma=0.012)
            mis = misalignment(rel, yaw_rel, e['kind'])
            # the first stance was computed from a far-away look; now that the object is seen up close, re-stand if it is off
            if restand < 4 and ((mis > 0.14 and e['kind'] != 'block') or moved > 0.03 or abs(rel[1]) > 0.06):
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
        e['status'] = 'held'; self.held = True; self.state = '搬运'; self.log(f'夹住 {cn}，送往收纳筐')
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
        if not ok:                                   # could not reach over the rim from here: set it down, try later
            yield from self.put_down(k); return False
        yield from wait(r, 0.5)
        # The robot assumes it went in. If it bounced out, the camera will see it on the floor later and reopen it.
        e['status'] = 'delivered'; self.log(f'{cn} 已放进收纳筐')
        if not self._in_basket(r.obj_pos(k)): self.stats['missed_place'] += 1      # score sheet only
        return True

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
        self.basket_blocked_t = r.t
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
            if self.r.t - self.basket_blocked_t < 25.0: return None   # no point picking up anything else yet
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
                yield from self.fetch(k); self.target = None; self.held = False
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
