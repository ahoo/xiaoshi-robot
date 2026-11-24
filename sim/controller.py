"""Autonomous fetch controller: lidar SLAM, A* planning, holonomic path following, grasp and place.

The control code only uses what the real robot has: wheel encoders (with calibration error and slip), lidar
ranges, joint feedback, and the front RGB-D camera. Localisation is lidar scan matching against the robot's own map
(localization.py); objects are found and measured in rendered depth images (perception.py). The only remaining
idealisation in sensing is the per-pixel instance masks (see perception.py). They say neither what an object is nor
which one it was last time: the controller remembers objects by where they are, and knows each one only by the
points the camera has seen on it. Ground truth is read only by the score sheet (after_step, _score_*), never to make
a decision.
"""
import heapq
import math

import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt

from scipy.spatial import ConvexHull, QhullError

from model import BASKET, HOME, P
from robot import LIDAR_RANGE
from skills import ELONG, PULL_MAX, PULL_REACH, best_band, carry_pose, grasp, holding, place, plan_grasp, pull, wait, wait_arm
from localization import Localizer
from perception import FrontCamera, WristCamera, clear_detection

CS, GX, GY = 0.05, 80, 60
CCS, CGX, CGY = 0.25, 16, 12
PASS = 0.215                 # m: a cell is drivable if the nearest mapped obstacle cell is at least this far (0.14 to the wheel rims, plus cell size, pose error and margin)
SAFE_R = 0.45                # m: obstacle points within this radius are checked by the safety layer
BODY_R = 0.15                # m: chassis + wheel rims, as a circle
ARM_BOX = (0.035, 0.228, 0.04)   # folded arm footprint in the chassis frame: x from, x to, half width (measured from the meshes)
HELD_BOX = (0.035, 0.245, 0.08)  # the same with a payload hanging under the gripper
SAFE_GAP = 0.02              # m: the safety layer never lets a motion bring any part of the robot closer than this
MAP_HIT = 4                  # obstacle map: a lidar return adds this to its cell, a ray passing through takes 1 off
MAP_OCC = 4                  # a cell is an obstacle above this, i.e. after two returns: a single stray return (outlier,
                             # mixed pixel) no longer blocks a cell until six rays have seen through it
LIDAR_BIAS_MARGIN = 0.02     # m: extra gap kept from lidar points for the lidar's own range bias. LD19 datasheet: mean
                             # error up to +-45 mm at 0.3-12 m and unspecified below 0.3 m; 2 cm assumes a per-unit
                             # range calibration (docs/validation.md). Without one, set it to 0.045.
BIAS_ON_MAP = False          # mapped cells keep MAP_PT_MARGIN only: stacking the bias margin on top left the planner's
                             # paths (PASS from cell centres) with no room to turn, and the robot froze (round 11: 15
                             # stuck events in 8 runs, against 1 before)
ESCAPE = True                # when every candidate motion is blocked, ease off slowly away from the close points
PLACE_REACH = 0.31           # m: farthest drop point over the basket rim the arm reaches (from the chassis centre)
MAP_PT_MARGIN = 0.02         # m: allowance for a mapped cell centre lying behind the obstacle surface it stands for
SAFE_HORIZON = (0.15, 0.3)   # s: look-ahead of the safety layer; 0.3 s covers stopping from full speed under the wheel ramp
ARM_SWEEP, HELD_SWEEP = 0.228, 0.258   # m: radius swept by the folded arm when the robot turns on the spot (empty / holding)
V_MAX, W_MAX = 0.18, 1.0
CAM_RANGE = 1.2             # m: floor counts as looked at out to here (with real depth-camera holes, small dark or flat
                            # things are often missed beyond ~1 m: perception bench)
CLOSE_FRAMES = 8             # depth frames averaged for the close look before a grasp (standing still, ~0.3 s at 30 fps)
CAM_RANGE_2 = 0.6           # m: second, closer sweep when the first one is done and there is time left
STANDOFFS = (0.235, 0.25, 0.265)   # robot centre to the middle of the strip the jaws close on: preferred, then farther back
KEEP = 0.19                  # m: on the way to a target, the robot centre keeps this far from its outline
ASSOC = 0.10                 # m: a detection this close to a remembered object is taken to be that object
T_MAX = 600.0
BLIND_PAD = 0.05             # m: a thing known only by a no-reading cluster, seen end-on, may be longer than it looks: the
                             # outline kept for the planner and the safety layer is padded this much all round
BLIND_CONFIRM = 2            # sightings (camera frames) of a cluster before it becomes something to go and get
BLIND_OBJ_FRAC = 0.35       # a cluster with more than this many real above-floor readings beside it, per no-reading pixel, is a black or
                             # shiny thing with holes in it, not clear plastic (bench: clear 0.13-0.22 up to 1 m, black 0.2-1.8, shiny 0.7-3)
FRAG_RATIO = 1.5              # a detection with this many times more no-reading pixels beside it than its own is a piece of a clear thing
                             # (black things: about 1.2 times, clear plastic: 6 or more)
BLIND_DRIFT = 0.0005         # m/s: how fast the certainty about where a no-reading thing lies fades while it is not looked at (the wheels
                             # can push it and nothing but the camera would tell); after 100 s a look from 1 m may correct a close-look position
BLIND_LOOK_WIDE = 0.25       # m: after a close look that saw nothing where the thing should be, accept a cluster this far from it
BLIND_MIN_RANGE = 0.14        # m from the camera foot: nearer clusters are only used by the close look in fetch
BLIND_GONE = 3               # looks at the spot with no no-reading pixels there before the memory of it is dropped
SLIP_WINDOW = 1.5            # s: lidar and odometry are compared over this long
SLIP_MIN_PATH = 0.12         # m of odometry path in the window before a mismatch counts (below that: standing, or creeping)
SLIP_RATIO = 0.5             # the scan matcher pulled the pose back by more than this share of the odometry path: the wheels
                             # turn but the robot does not follow (a wheel on a thing it cannot see, or pressed against it)
GHOST_R = 0.04               # m: radius of the invisible obstacle remembered where the robot got stuck
GHOST_LIFE = 60.0            # s: a preference, not a wall: goto forgets it when it leaves no way through (see there)
SLIP_ON = True               # lidar/odometry mismatch -> back off and remember an invisible obstacle
CLEAR_CUE = True             # no-reading clusters in the depth image -> things the camera cannot measure (clear plastic)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Agent:
    def __init__(self, robot, seed=0, verbose=False):
        self.r = robot; self.rng = np.random.default_rng(seed + 7); self.verbose = verbose
        self.lo = np.zeros((GY, GX), np.int8)
        self.cov = np.zeros((CGY, CGX), bool)
        self.memory = {}; self._next_id = 1; self.priority = None; self.score_events = []; self.cam_range = CAM_RANGE
        self.loc = Localizer(robot.true_pose(), CS, (GX * CS, GY * CS))   # starts docked at a known pose; after that, lidar + odometry only
        self.cam = FrontCamera(robot, np.random.default_rng(seed + 11)); self.visible = {}; self._ncam = 0
        self.wrist = WristCamera(robot)
        self.loc_err = []                                  # (position error m, heading error rad) once a second, for the score sheet
        self.cam_low = np.zeros((0, 2)); self.cam_low_t = -9.0; self.help_needed = False
        self.scan_pts = np.zeros((0, 2)); self.map_pts = np.zeros((0, 2)); self.blocked = 0; self.held = False; self.basket_blocked_t = -99.0
        self.path = None; self.scan = None; self.state = '启动'; self.target = None; self.log_lines = []
        self.stats = dict(pulls=0, grasp_attempts=0, grasp_fail=0, dropped=0, missed_place=0, replans=0, stuck=0, collisions=0, floor_touch=0, max_tilt=0.0, dist=0.0)
        self._tick = 0; self._last_xy = np.array(robot.true_pose()[:2]); self._coll_t = -9.0; self._floor_t = -9.0
        self.ghosts = []; self._slip_hist = []; self.slip_flag = False; self._cmd_w = np.zeros(2); self._slip_t = -99.0
        self.pause = None          # set by modes.Supervisor: a callable that says "stop at the next safe point" (low battery); None = never
        self.arm_busy = False      # set by modes.Supervisor while an operator has the arm out in front of the lidar
        self.freeze = False        # set by modes.Supervisor while the robot is tilted or lying down: the lidar and camera see the room at the wrong angle, so no mapping, matching or detecting
        self.pose_before_fall = None
        self.dyn_zones = []        # set by people.Guard: [(x, y, r)] where people are right now; a lidar hit inside one is a person, not a wall: it goes neither into the map nor into the scan matching
        self.zone_skip_t = -99.0   # when pick() last skipped an object because someone stands by it
        self.person_zones = []     # set by people.Guard: [(x, y, r)] around the people in the room; an object inside one is not started on (looked at again in 10 s)
        self.depth_hook = None     # set by home.py: called as hook(agent, x, y, th) with every front-camera frame, after detection (3D scan, object records)
        m = robot.m
        self._floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        # Where the basket is, in the robot's own map. It starts from where the user said it is, and is then corrected
        # every time the camera sees it, so it stays consistent with the map even if the map as a whole has drifted.
        self.basket = dict(BASKET); self.basket_seen = 0
        self._set_basket(BASKET['x'], BASKET['y'])
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
        if self.freeze == 2: return                    # tilted or lying: nothing from the sensors is usable
        r = self.r
        if self.freeze == 1:                            # upright again but the pose is unknown: only what is near the robot, for the safety layer
            ang, rng_ = r.lidar(); rng_ = self._scan_filter(ang, rng_); f = np.isfinite(rng_)
            x, y, th = self.pose(); sx, sy = P['lidar_pos'][0] + rng_[f] * np.cos(ang[f]), rng_[f] * np.sin(ang[f])
            wx_, wy_ = x + math.cos(th) * sx - math.sin(th) * sy, y + math.sin(th) * sx + math.cos(th) * sy
            near = np.hypot(wx_ - x, wy_ - y) < SAFE_R
            self.scan_pts = np.stack([wx_[near], wy_[near]], 1); self.map_pts = np.zeros((0, 2)); return
        self._ncam += 1
        cam_now = self._ncam % 3 == 1                 # camera at 3.3 Hz, lidar at 10 Hz
        # --- lidar: correct the pose against the map, then update the map from the corrected pose
        if self.arm_busy or self.state in ('抓取', '放置', '放下'):   # the arm is out in front of the lidar: neither match nor map
            if cam_now: self._sense_camera()
            return
        ang, rng_ = r.lidar()
        rng_ = self._scan_filter(ang, rng_)
        f = np.isfinite(rng_)
        spx, spy = P['lidar_pos'][0] + rng_[f] * np.cos(ang[f]), rng_[f] * np.sin(ang[f])
        if self.dyn_zones:                           # people are not walls: their legs stay out of the scan matching and out of the map (they would be left behind as ghost obstacles)
            ex, ey, eth = self.loc.est; wx0 = ex + math.cos(eth) * spx - math.sin(eth) * spy; wy0 = ey + math.sin(eth) * spx + math.cos(eth) * spy
            dyn = np.zeros(len(spx), bool)
            for zx, zy, zr in self.dyn_zones: dyn |= np.hypot(wx0 - zx, wy0 - zy) < zr
            if dyn.any(): spx, spy = spx[~dyn], spy[~dyn]
        pre = self.loc.est[:2].copy(); path = self.loc._moved
        moved_ok = self.loc.correct(spx, spy)
        if SLIP_ON and moved_ok: self._track_slip(pre, self.loc.est[:2] - pre, path)
        good = self.loc.map_ok
        if good: self.loc.add_scan(spx, spy)
        else: self.stats['scans_unmapped'] = self.stats.get('scans_unmapped', 0) + 1
        x, y, th = self.pose()
        lx, ly = x + math.cos(th) * P['lidar_pos'][0], y + math.sin(th) * P['lidar_pos'][0]
        fin = np.isfinite(rng_); a = ang[fin] + th; rr = rng_[fin]
        self.scan = (lx, ly, a, rr)
        if len(rr) and good:                          # map update, only from well-matched scans
            ts = np.arange(0.06, LIDAR_RANGE, 0.04)[None, :]
            mask = ts < (rr[:, None] - 0.05)
            px = (lx + np.cos(a)[:, None] * ts)[mask]; py = (ly + np.sin(a)[:, None] * ts)[mask]
            ci = np.floor(px / CS).astype(int); cj = np.floor(py / CS).astype(int)
            ok = (ci >= 0) & (ci < GX) & (cj >= 0) & (cj < GY)
            flat = np.unique(cj[ok] * GX + ci[ok]); lo = self.lo.reshape(-1)
            lo[flat] = np.maximum(lo[flat] - 1, -3)
            hi = np.floor((lx + np.cos(a) * (rr + 0.015)) / CS).astype(int); hj = np.floor((ly + np.sin(a) * (rr + 0.015)) / CS).astype(int)
            ok = (hi >= 0) & (hi < GX) & (hj >= 0) & (hj < GY)
            for zx, zy, zr in self.dyn_zones: ok &= np.hypot(lx + np.cos(a) * rr - zx, ly + np.sin(a) * rr - zy) >= zr      # a hit on a person marks nothing (the free space in front of him is still cleared)
            flat = np.unique(hj[ok] * GX + hi[ok])
            lo[flat] = np.minimum(lo[flat].astype(int) + MAP_HIT, 20).astype(np.int8)
        if len(rr):                                   # the safety layer always gets this scan's points
            wx_, wy_ = lx + np.cos(a) * rr, ly + np.sin(a) * rr
            near = np.hypot(wx_ - x, wy_ - y) < SAFE_R
            self.scan_pts = np.stack([wx_[near], wy_[near]], 1)
        else:
            self.scan_pts = np.zeros((0, 2))
        # mapped obstacle cells near the robot (world frame). The folded arm hides a strip in front of the lidar, and
        # anything seen there earlier is still in the map, so the safety layer uses both.
        i0, i1 = max(0, int((x - SAFE_R) / CS)), min(GX, int((x + SAFE_R) / CS) + 1)
        j0, j1 = max(0, int((y - SAFE_R) / CS)), min(GY, int((y + SAFE_R) / CS) + 1)
        jj, ii = np.nonzero(self.lo[j0:j1, i0:i1] > MAP_OCC)
        self.map_pts = np.stack([(ii + i0 + 0.5) * CS, (jj + j0 + 0.5) * CS], 1)
        if cam_now: self._sense_camera()

    def _track_slip(self, pre, corr, path):
        """Compare what the wheels say with what the lidar says. Each scan: `path` is how far the odometry moved the
        pose since the last scan matched, `corr` what the matcher then changed. Over a window, if the corrections point
        back along the motion and add up to more than SLIP_RATIO of the path, the wheels are turning without the
        robot following: raise the flag (acted on in goto)."""
        t = self.r.t
        mv = pre - self._slip_prev if getattr(self, '_slip_prev', None) is not None else np.zeros(2)
        self._slip_prev = pre + corr
        n = float(np.hypot(*mv))
        back = -float(corr @ mv) / n if n > 1e-4 else 0.0
        self._slip_hist.append((t, path, back))
        self._slip_hist = [h for h in self._slip_hist if t - h[0] <= SLIP_WINDOW]
        P = sum(h[1] for h in self._slip_hist); B = sum(h[2] for h in self._slip_hist)
        self.slip_ratio = B / P if P > 1e-6 else 0.0
        if (P >= SLIP_MIN_PATH and self.slip_ratio > SLIP_RATIO and t - self._slip_t > 4.0
                and self.state in ('导航', '搬运', '巡视', '回位') and float(np.hypot(*self._cmd_w)) > 0.05):
            self.slip_flag = t                        # when it was raised: only acted on while fresh (goto)

    def _slip_recover(self):
        """The wheels turned and the robot did not follow: something it cannot see (a clear bottle, a cable) is under or
        against a wheel. Stop, back away from it, and remember an invisible obstacle just ahead of the body in the
        direction it was driving, so the next plan goes round it."""
        self.slip_flag = False; self._slip_t = self.r.t; self._slip_hist = []
        x, y, th = self.pose(); d = self._cmd_w / max(float(np.hypot(*self._cmd_w)), 1e-6)
        c_ = np.array([x, y]) + d * (BODY_R + GHOST_R - 0.01)
        ang = np.linspace(0, 2 * math.pi, 12, endpoint=False)
        # Only something nothing else explains is remembered: not where a known thing (to be picked up, being carried, the
        # basket) or mapped furniture already is. The robot is often pressed against the very thing it is going to; a
        # remembered obstacle on top of it would only make it unreachable.
        known = any(float(np.min(np.hypot(e['outline'][:, 0] - c_[0], e['outline'][:, 1] - c_[1]))) < GHOST_R + 0.10 for e in self.memory.values() if e['status'] in ('pending', 'held'))
        known = known or self._in_basket((c_[0], c_[1], 0.0), margin=GHOST_R + 0.05)
        gi_, gj_ = int(c_[0] / CS), int(c_[1] / CS)
        known = known or (0 <= gi_ < GX and 0 <= gj_ < GY and self.lo[max(0, gj_ - 1):gj_ + 2, max(0, gi_ - 1):gi_ + 2].max() > MAP_OCC)
        if not known:
            self.ghosts.append(dict(x=float(c_[0]), y=float(c_[1]), t=self.r.t, outline=np.stack([c_[0] + GHOST_R * np.cos(ang), c_[1] + GHOST_R * np.sin(ang)], 1)))
        self.stats['slips'] = self.stats.get('slips', 0) + 1
        self.stats['ghosts'] = self.stats.get('ghosts', 0) + (0 if known else 1)
        self.log('轮子转了、车没跟上（雷达和里程计对不上）：停下后退' + ('，前方已经有已知的东西，不另记' if known else '，前方记一个看不见的障碍'))
        self.stop()
        c, s = math.cos(th), math.sin(th)
        for _ in range(60): self.cmd(-0.08 * (c * d[0] + s * d[1]), -0.08 * (-s * d[0] + c * d[1]), 0.0); yield
        self.stop()

    def _live_ghosts(self):
        self.ghosts = [g for g in self.ghosts if self.r.t - g['t'] < GHOST_LIFE]
        return self.ghosts

    def _scan_filter(self, ang, rr):
        """Drop lidar returns that are neither backed by a neighbour in this scan nor seen at the same spot one scan
        earlier: random outliers and mixed pixels at range jumps (both measured on LD19-class lidars) would otherwise
        put phantom obstacles in the map and in front of the safety layer. A thin chair leg that only one ray hits
        still survives, because it is there again in the next scan."""
        n = len(rr); fin = np.isfinite(rr); R = np.where(fin, rr, 1e6)
        tol = 0.05 + 0.03 * np.where(fin, rr, 0)
        ok = np.zeros(n, bool)
        for k in (1, 2):
            for sgn in (1, -1):
                ok |= np.abs(np.roll(R, sgn * k) - R) < tol
        x, y, th = self.pose(); lx = P['lidar_pos'][0]
        wx = x + np.cos(th + ang) * R + math.cos(th) * lx; wy = y + np.sin(th + ang) * R + math.sin(th) * lx
        lone = np.nonzero(fin & ~ok)[0]
        prev = getattr(self, '_prev_scan_w', None)
        if len(lone) and prev is not None and len(prev):
            d = np.min(np.hypot(wx[lone, None] - prev[None, :, 0], wy[lone, None] - prev[None, :, 1]), 1)
            ok[lone[d < 0.04]] = True
        self._prev_scan_w = np.stack([wx[fin], wy[fin]], 1)
        self.stats['scan_dropped'] = self.stats.get('scan_dropped', 0) + int((fin & ~ok).sum())
        return np.where(ok, rr, np.inf)

    def _sense_camera(self):
        """Front RGB-D camera: which floor cells it has looked at, and object detections put on the map."""
        x, y, th = self.pose(); c, s_ = math.cos(th), math.sin(th)
        dets = self._drop_fragments(self.cam.detect(self.basket_rect()))
        # Low obstacles the camera sees right now (basket walls, things on the floor), for the safety layer. They do
        # not depend on the map, so they still protect when localisation has drifted. Leave out the thing being
        # carried (it hangs in front of the camera) and the thing about to be grasped.
        L = self.cam.low_pts
        if len(L):
            if self.held or self.state in ('放置', '放下'):
                L = L[~((L[:, 0] > -0.02) & (L[:, 0] < 0.42) & (np.abs(L[:, 1]) < 0.13))]
            Lw = np.stack([x + c * L[:, 0] - s_ * L[:, 1], y + s_ * L[:, 0] + c * L[:, 1]], 1)
            if self.target in self.memory and len(Lw):
                ol = self.memory[self.target]['outline']
                Lw = Lw[np.min(np.hypot(Lw[:, None, 0] - ol[None, :, 0], Lw[:, None, 1] - ol[None, :, 1]), 1) > 0.03]
            self.cam_low = Lw
        else:
            self.cam_low = np.zeros((0, 2))
        self.cam_low_t = self.r.t
        if self.cam.basket_obs is not None and not self.cam.basket_obs['cut']:
            self._update_basket(self.cam.basket_obs['pts'])
        # coverage: unseen floor cells whose centre is in the image, not hidden behind anything, and close enough
        cells = [(j, i) for j in range(CGY) for i in range(CGX) if not self.cov[j, i]]
        if cells:
            W_ = np.array([((i + 0.5) * CCS, (j + 0.5) * CCS) for j, i in cells]) - [x, y]
            B = np.stack([c * W_[:, 0] + s_ * W_[:, 1], -s_ * W_[:, 0] + c * W_[:, 1], np.zeros(len(W_))], 1)
            dist = np.hypot(B[:, 0], B[:, 1])
            seen = self.cam.floor_visible(B) & (dist < self.cam_range)
            for (j, i), sn, dd in zip(cells, seen, dist):
                if sn or dd < 0.3: self.cov[j, i] = True
        self.visible = {}
        if self.state in ('抓取', '放置', '放下'): return       # the arm is in front of the camera, and the target is being handled
        seen = set()
        for dt in dets:
            if dt['bottom'] > 0.04: continue               # not lying on the floor: the load in our own gripper
            o = self._to_world(dt, x, y, th)
            if self._in_basket((o['x'], o['y'], 0.0), margin=0.03): continue
            # a strip of basket wall the segmentation let through (the basket estimate is a few cm off): rim height, at the rim
            b = self.basket
            if abs(o['top'] - b['h']) < 0.015 and abs(o['x'] - b['x']) < b['sx'] / 2 + 0.07 and abs(o['y'] - b['y']) < b['sy'] / 2 + 0.07: continue
            gi_, gj_ = int(o['x'] / CS), int(o['y'] / CS)
            if 0 <= gi_ < GX and 0 <= gj_ < GY and self.lo[gj_, gi_] > MAP_OCC: continue     # inside mapped furniture: a sliver of it
            k = self._associate(o['x'], o['y'])
            if k is not None and self.memory[k].get('blind') and k == self.target: seen.add(k); continue      # fetch looks at it itself
            if k is not None: seen.add(k)
            # only whole views are used to learn about an object: one cut by the image edge has a shifted outline
            if dt['cut'] or dt['n_px'] < 40: continue
            sig = 0.008 + 0.007 * dt['dist']                 # detector error (bench) plus localisation error
            if k is None:
                k = self._next_id; self._next_id += 1; seen.add(k)
                self.memory[k] = dict(id=k, status='pending', retry_at=0.0, fails=0, miss=0, sigma=sig, **o)
                self.log(f'发现{self._name(self.memory[k], True)}，距离 {dt["dist"]:.2f} m')
            else:
                e = self.memory[k]
                if e['status'] == 'delivered':
                    self.log(f'{self._name(e)} 那里的地上还有东西，重新去拿'); e.update(status='pending', retry_at=0.0)
                if sig <= e['sigma'] * 1.1:
                    e.update(sigma=sig, **o)
                    if e.pop('blind', None):
                        self.log(f'{self._name(e)} 走近后看清了，按普通物体处理')
                        # everything that belonged to the no-reading estimate goes with it: the core outline (outline_core is what
                        # standoffs, the safety layer and the cost map read for the thing being fetched) and the side-look state
                        for k_ in ('ambiguous', 'outline_core', 'bad_span', 'side_gave_up', 'look_fail', 'looks', 'seen_t'): e.pop(k_, None)
            self.visible[k] = dt['dist']
        for k in seen:
            if k in self.memory: self.memory[k]['miss'] = 0
        if self.depth_hook is not None: self.depth_hook(self, x, y, th)      # after this frame's objects are matched: agent.visible and agent.memory are current
        found = self._sense_clear(dets, x, y, th)
        # Negative evidence: the floor where a remembered object should be is in plain view and nothing is there.
        # (Pushed away by a wheel, or a duplicate.) Only from 0.35 m (closer, a flat object hardly hides the floor
        # point under it) out to where the camera would surely have seen it: 1.2 m, but only 0.7 m for small flat
        # things (a 2 cm stone is seen 1 time in 6 at 1 m). The floor there is marked unseen again, to look later.
        for k, e in list(self.memory.items()):
            if k in seen or e['status'] != 'pending' or k == self.target or e.get('blind'): continue
            wx_, wy_ = e['x'] - x, e['y'] - y
            B = np.array([[c * wx_ + s_ * wy_, -s_ * wx_ + c * wy_, 0.0]])
            r_max = 0.7 if (e['top'] < 0.03 or max(e['size'][:2]) < 0.05) else 1.2
            if 0.35 < math.hypot(B[0, 0], B[0, 1]) < r_max and self.cam.floor_visible(B)[0]:
                e['miss'] += 1
                if e['miss'] >= 2:
                    self.log(f'{self._name(e)} 不在原处了，从记忆里去掉'); del self.memory[k]
                    ci, cj = int(e['x'] / CCS), int(e['y'] / CCS)
                    if 0 <= ci < CGX and 0 <= cj < CGY: self.cov[cj, ci] = False
                    if self.priority == k: self.priority = None
        # Two memories of one object (seen from two sides before the views agreed, or cut in two by holes in the
        # depth of a shiny thing): centres within 7 cm, or outlines within 2 cm. Keep the one seen more completely.
        pend = [(k, e) for k, e in self.memory.items() if e['status'] == 'pending']
        for i, (k1, e1) in enumerate(pend):
            for k2, e2 in pend[i + 1:]:
                if k1 not in self.memory or k2 not in self.memory: continue
                near = math.hypot(e1['x'] - e2['x'], e1['y'] - e2['y']) < 0.07
                if not near and math.hypot(e1['x'] - e2['x'], e1['y'] - e2['y']) < 0.25:
                    o1, o2 = e1['outline'], e2['outline']
                    near = float(np.min(np.hypot(o1[:, None, 0] - o2[None, :, 0], o1[:, None, 1] - o2[None, :, 1]))) < 0.02
                if not near: continue
                if k1 == self.target: drop = k2
                elif k2 == self.target: drop = k1
                else: drop = k2 if len(e1['pts']) >= len(e2['pts']) else k1
                del self.memory[drop]
                if self.priority == drop: self.priority = None

    @staticmethod
    def _pad(outline, r):
        """The outline grown by r all round (convex hull of eight shifted copies)."""
        offs = np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]) * r
        Pp = (outline[:, None, :] + offs[None]).reshape(-1, 2)
        return Pp[ConvexHull(Pp).vertices]

    def _drop_fragments(self, dets):
        """Pieces of a clear thing that did return depth (cap, label, an edge) come out of the segmentation as small objects
        of their own. A no-reading cluster that is clearly bigger than the detection on the same spot says so: the detection
        is only a piece, and the thing is described by the cluster (perception.clear_detection). A cluster inside a
        detection (a few holes in a black or shiny thing that is measured anyway) leaves the detection alone."""
        if not CLEAR_CUE or not dets or not self.cam.holes: return dets
        keep = []
        for dt in dets:
            frag = False
            for h in self.cam.holes:
                if h['n_px'] < 30 or h['obj_frac'] > BLIND_OBJ_FRAC: continue      # (no range limit here: close up is where the pieces are largest)
                d = clear_detection(h, self.cam)
                if math.hypot(dt['rel'][0] - d['rel'][0], dt['rel'][1] - d['rel'][1]) > 0.15: continue
                H_ = np.asarray(dt['hull']); L_dt = max(float(np.hypot(*(a - b))) for a in H_ for b in H_)
                if d['length'] > L_dt + 0.03 or h['n_px'] > FRAG_RATIO * dt['n_px']: frag = True; break
            if not frag: keep.append(dt)
        return keep

    @staticmethod
    def _reset_blind(e):
        """After a fetch that did not end in a grasp, drop the flags that would keep the next try from working: the side look
        that was given up on, the count of close looks that saw nothing, the sides cached for the old geometry. Kept: which
        sides were tried (`bad`) and from where it was looked at (`looks`), without which the second try repeats the first, and
        the position with its certainty, which fades with time since it was last seen (BLIND_DRIFT) rather than being thrown
        away: wiping it made an end-on bottle in patchy depth repeat the same failing round three times, keeping it locked
        lost a bottle that had been pushed (round 12, materials seed 303)."""
        if not e.get('blind'): return
        for k in ('side_gave_up', 'look_fail'): e.pop(k, None)
        e['dirs'] = None

    def _set_blind(self, e, o, ambiguous):
        """Outlines of something known only by a no-reading cluster: `outline_core` as measured, `outline` (what the planner
        and the safety layer keep away from) padded when its length is unknown. The thing being fetched is kept away
        from by its core only: the stance has to be close enough to reach it."""
        e['ambiguous'] = bool(ambiguous); e['outline_core'] = o['outline']
        e['outline'] = self._pad(o['outline'], BLIND_PAD if ambiguous else 0.015)

    def _sense_clear(self, dets, x, y, th):
        """Things the depth camera cannot measure (clear plastic): a cluster of pixels without any reading in the middle
        of floor that does return depth. Nothing else is known of such a thing, so it is handled as an object seen only
        as far as perception.clear_detection can tell: a remembered object (status pending, flagged `blind`) that the
        planner can go to, and, from its outline, something the wheels keep away from. A cluster has to be seen in
        BLIND_CONFIRM frames before it counts, and a remembered one is dropped when the spot is in view and there are
        no missing readings there any more. Returns the set of memory ids that were confirmed by a cluster this frame."""
        c, s_ = math.cos(th), math.sin(th); found = set()
        if not CLEAR_CUE: return found
        cands = getattr(self, '_blind_cands', [])
        cands = [q for q in cands if self.r.t - q['t'] < 2.0]
        if not self.held:
            for h in self.cam.holes:
                if h['cut'] or h['n_px'] < 30: continue          # cut by the image edge: its extent is not whole
                if h['obj_frac'] > BLIND_OBJ_FRAC: continue      # holes in a black or shiny thing that is measured anyway
                if h['near'] - self.cam.t_bc[0] < BLIND_MIN_RANGE: continue     # right under the camera: the geometry is not reliable (fetch looks again from its stance)
                d = clear_detection(h, self.cam)
                if any(math.hypot(dt['rel'][0] - d['rel'][0], dt['rel'][1] - d['rel'][1]) < 0.15 for dt in dets): continue   # a thing with a few holes in it that is measured anyway
                # sight line over the basket: what is seen there is in the basket (or behind it), not on the floor here
                near_pt = np.array([h['near'], 0.5 * (h['y0'] + h['y1'])]); cam_xy = self.cam.t_bc[:2]
                line = [cam_xy + (near_pt - cam_xy) * f_ for f_ in np.linspace(0.05, 1.0, 12)]
                if any(self._in_basket((x + c * p[0] - s_ * p[1], y + s_ * p[0] + c * p[1], 0.0), margin=0.03) for p in line): continue
                o = self._to_world(d, x, y, th)
                if self._in_basket((o['x'], o['y'], 0.0), margin=0.03): continue
                gi_, gj_ = int(o['x'] / CS), int(o['y'] / CS)
                if 0 <= gi_ < GX and 0 <= gj_ < GY and self.lo[gj_, gi_] > MAP_OCC: continue
                sig = 0.02 + 0.04 * d['dist']
                k = self._associate(o['x'], o['y'])
                if k is not None:
                    e = self.memory[k]; found.add(k)
                    if not e.get('blind'): continue
                    e['miss'] = 0
                    eff = e['sigma'] + BLIND_DRIFT * (self.r.t - e.get('seen_t', self.r.t))        # fades while it is out of sight
                    e['seen_t'] = self.r.t
                    if e['status'] == 'delivered':
                        self.log(f'{self._name(e)} 那里的地上还有看不清的东西，重新去拿'); e.update(status='pending', retry_at=0.0, looks=[])
                    # a close, whole look wins; a view that cannot tell the direction does not overwrite one that could
                    if sig <= eff * 1.1 and not (d['ambiguous'] and not e['ambiguous']):
                        e.update(sigma=sig, **o); self._set_blind(e, o, d['ambiguous'])
                    continue
                q = next((q for q in cands if math.hypot(q['x'] - o['x'], q['y'] - o['y']) < 0.10), None)
                if q is None: cands.append(dict(x=o['x'], y=o['y'], n=1, t=self.r.t)); continue
                q.update(x=o['x'], y=o['y'], n=q['n'] + 1, t=self.r.t)
                if q['n'] < BLIND_CONFIRM: continue
                cands.remove(q)
                k = self._next_id; self._next_id += 1; found.add(k)
                self.memory[k] = dict(id=k, status='pending', retry_at=0.0, fails=0, miss=0, sigma=sig, blind=True, looks=[], seen_t=self.r.t, **o)
                self._set_blind(self.memory[k], o, d['ambiguous']); self.stats['blind_found'] = self.stats.get('blind_found', 0) + 1
                self.log(f'发现{self._name(self.memory[k], True)}：深度图上一片没读数的地方，应该是透明的东西，距离 {d["dist"]:.2f} m')
        self._blind_cands = cands
        # gone? The spot is in view, close enough to judge, and nothing is missing from the depth there
        for k, e in list(self.memory.items()):
            if not e.get('blind') or e['status'] != 'pending' or k == self.target or k in found or self.held: continue
            wx_, wy_ = e['x'] - x, e['y'] - y
            B = np.array([c * wx_ + s_ * wy_, -s_ * wx_ + c * wy_, 0.0])
            if not 0.35 < math.hypot(B[0], B[1]) < 1.0: continue
            # the centre of a thing seen end-on can be several cm off along the line of sight: look over the whole core outline
            pts_ = [B] + [np.array([c * (px_ - x) + s_ * (py_ - y), -s_ * (px_ - x) + c * (py_ - y), 0.0]) for px_, py_ in e.get('outline_core', e['outline'])]
            fr_ = [self.cam.hole_fraction(q_, half=4) for q_ in pts_]
            frac = None if all(f_ is None for f_ in fr_) else max(f_ for f_ in fr_ if f_ is not None)
            if frac is not None and frac < 0.02:
                e['miss'] += 1
                if e['miss'] >= BLIND_GONE:
                    self.log(f'{self._name(e)} 不在原处了，从记忆里去掉'); del self.memory[k]
                    ci, cj = int(e['x'] / CCS), int(e['y'] / CCS)
                    if 0 <= ci < CGX and 0 <= cj < CGY: self.cov[cj, ci] = False
                    if self.priority == k: self.priority = None
        return found

    def _to_world(self, dt, x, y, th):
        """A detection (chassis frame) as map-frame measurements of the object: centre, outline, surface points, size."""
        c, s_ = math.cos(th), math.sin(th)
        T = lambda Q: np.stack([x + c * Q[:, 0] - s_ * Q[:, 1], y + s_ * Q[:, 0] + c * Q[:, 1]], 1)
        ctr = T(np.asarray(dt['rel'][None, :2]))[0]; hull = T(np.asarray(dt['hull']))
        pts = np.column_stack([T(dt['pts']), dt['pts'][:, 2]])
        # The far side is hidden from the camera; mirror the seen outline through its centre to cover it.
        both = np.vstack([hull, 2 * ctr - hull])
        try: outline = both[ConvexHull(both).vertices]
        except (QhullError, ValueError): outline = both
        hd = wrap(dt['heading'] + th); a = np.array([math.cos(hd), math.sin(hd)]); b = np.array([-a[1], a[0]])
        L = float(np.ptp(outline @ a)); Wd = float(np.ptp(outline @ b))
        return dict(x=float(ctr[0]), y=float(ctr[1]), heading=hd, elong=dt['elong'], top=dt['top'], pts=pts, outline=outline,
                    size=(L, Wd, dt['top']), dirs=None)

    def _associate(self, mx, my):
        best = None
        for k, e in self.memory.items():
            if e['status'] == 'held': continue
            d = math.hypot(mx - e['x'], my - e['y'])
            if d < max(ASSOC, 3 * e['sigma'] + 0.03) and (best is None or d < best[0]): best = (d, k)
        return None if best is None else best[1]

    @staticmethod
    def _name(e, size=False):
        if not size: return f'物体#{e["id"]}'
        L, Wd, H = (100 * v for v in e['size'])
        return f'物体#{e["id"]}（约 {L:.0f}×{Wd:.0f} cm，高 {H:.0f} cm）'

    def _set_basket(self, bx, by):
        b = self.basket; b['x'], b['y'] = bx, by
        self._basket_cells = [(j, i) for j in range(GY) for i in range(GX)
                              if abs((i + 0.5) * CS - bx) <= b['sx'] / 2 and abs((j + 0.5) * CS - by) <= b['sy'] / 2]
        # the rim as points, for the safety layer (it is below the lidar, so the scan never shows it)
        hx, hy = b['sx'] / 2, b['sy'] / 2
        ex = np.arange(-hx, hx + 1e-9, 0.02); ey = np.arange(-hy, hy + 1e-9, 0.02)
        self.basket_pts = np.vstack([np.stack([bx + ex, by - hy + 0 * ex], 1), np.stack([bx + ex, by + hy + 0 * ex], 1),
                                     np.stack([bx - hx + 0 * ey, by + ey], 1), np.stack([bx + hx + 0 * ey, by + ey], 1)])

    def basket_rect(self):
        """Where the robot believes the basket is, in the chassis frame (for the segmentation: points there belong to
        the basket, not to objects). Its rough place is known from the start; the camera refines it."""
        x, y, th = self.pose(); c, s_ = math.cos(th), math.sin(th); b = self.basket
        dx, dy = b['x'] - x, b['y'] - y
        return (c * dx + s_ * dy, -s_ * dx + c * dy, -th, b['sx'] / 2 + 0.01, b['sy'] / 2 + 0.01)

    def _update_basket(self, pts_body):
        """Camera points on the basket walls (chassis frame) -> basket centre in the map. The known rectangle is slid
        around the current estimate (+-8 cm) to where most points lie on its walls. One wall seen is enough to fix the
        distance across it; along it, the estimate stays where it was unless an end of the wall is seen too."""
        x, y, th = self.pose(); c, s_ = math.cos(th), math.sin(th)
        wx = x + c * pts_body[:, 0] - s_ * pts_body[:, 1]; wy = y + s_ * pts_body[:, 0] + c * pts_body[:, 1]
        b = self.basket; hx, hy = b['sx'] / 2, b['sy'] / 2
        best = None
        for dx in np.arange(-0.08, 0.0801, 0.01):
            for dy in np.arange(-0.08, 0.0801, 0.01):
                ex = np.abs(wx - b['x'] - dx) - hx; ey = np.abs(wy - b['y'] - dy) - hy
                d = np.where((ex <= 0) & (ey <= 0), np.minimum(-ex, -ey), np.hypot(np.maximum(ex, 0), np.maximum(ey, 0)))
                score = float((d < 0.012).sum()) - 0.5 * float((d > 0.03).sum()) - 2.0 * math.hypot(dx, dy)
                if best is None or score > best[0]: best = (score, dx, dy)
        _, dx, dy = best
        a = 0.5 if self.basket_seen == 0 else 0.2
        self._set_basket(b['x'] + a * dx, b['y'] + a * dy); self.basket_seen += 1

    def _outline(self, e):
        """Outline of a remembered floor object (map frame) as measured by the camera. The safety layer keeps the
        wheels off it: below the lidar, the object is otherwise invisible to it."""
        if e.get('outline_core') is not None and e['id'] == self.target: return e['outline_core']
        return e['outline']

    def _in_basket(self, p, margin=0.0):
        """Inside the basket as the robot believes it to be (map frame)."""
        b = self.basket
        return abs(p[0] - b['x']) < b['sx'] / 2 + margin and abs(p[1] - b['y']) < b['sy'] / 2 + margin and p[2] < b['h'] + 0.03

    @staticmethod
    def _in_basket_world(p):
        """Inside the real basket (world frame, ground truth): for the score sheet only."""
        return abs(p[0] - BASKET['x']) < BASKET['sx'] / 2 and abs(p[1] - BASKET['y']) < BASKET['sy'] / 2 and p[2] < BASKET['h'] + 0.03

    def _score_held_obj(self):
        """Score sheet only (ground truth): which real object is in the gripper."""
        p = self.r.pinch_pos()
        return int(np.argmin([np.linalg.norm(self.r.obj_pos(i) - p) for i in range(len(self.r.objects))]))

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
            if mine == r.dock_nose and other in (r.dock_pad, r.dock_plate): continue      # touching the charger is what docking is, not a collision
            is_roller = r.m.geom_group[mine] == 1
            if other == self._floor:
                if not is_roller and r.t - self._floor_t > 1.0: self._floor_t = r.t; self.stats['floor_touch'] += 1
            elif r.t - self._coll_t > 1.0:
                self._coll_t = r.t; self.stats['collisions'] += 1
                self.log('碰到 ' + (mujoco.mj_id2name(r.m, mujoco.mjtObj.mjOBJ_GEOM, other) or '?'))

    # ------------------------------------------------------------------ planning
    def costmap(self, ignore=None):
        occ = self.lo > MAP_OCC
        for j, i in self._basket_cells: occ[j, i] = True
        D = distance_transform_edt(~np.pad(occ, 1, constant_values=True))[1:-1, 1:-1] * CS
        self.D_furn = D
        # Objects on the floor are low: only the chassis can touch them, so the robot centre has to keep KEEP from
        # their outlines, not a full PASS from the cells they are in.
        ol = [e['outline'] for k, e in self.memory.items() if k != ignore and e['status'] == 'pending'] + [g['outline'] for g in self._live_ghosts()]
        if ol:
            ol = np.vstack(ol)
            D = np.minimum(D, np.min(np.hypot(self._cx[..., None] - ol[:, 0], self._cy[..., None] - ol[:, 1]), -1) + (PASS - KEEP))
        self.D_base = D
        if ignore is not None and ignore in self.memory:
            # the target is not a wall, but the robot must not drive over it on the way to its far side:
            # keep the robot centre at least KEEP from its outline
            ol = self.memory[ignore].get('outline_core', self.memory[ignore]['outline'])
            dmin = np.min(np.hypot(self._cx[..., None] - ol[:, 0], self._cy[..., None] - ol[:, 1]), -1)
            D = np.minimum(D, dmin + (PASS - KEEP))
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
        lowp = np.vstack([self.basket_pts] + [self._outline(e) for e in self.memory.values() if e['status'] == 'pending'] + [g['outline'] for g in self.ghosts]
                         + ([self.cam_low] if self.r.t - self.cam_low_t < 0.6 and len(self.cam_low) else []))
        bp = lowp[np.hypot(lowp[:, 0] - x, lowp[:, 1] - y) < SAFE_R]
        W = [self.scan_pts, self.map_pts, bp]
        if not any(len(w) for w in W):
            self.r.set_twist(vx, vy, wz); return
        c, s = math.cos(th), math.sin(th)
        P = np.vstack([w for w in W if len(w)]) - [x, y]
        P = np.stack([c * P[:, 0] + s * P[:, 1], -s * P[:, 0] + c * P[:, 1]], 1)
        margin = np.concatenate([np.full(len(self.scan_pts), LIDAR_BIAS_MARGIN), np.full(len(self.map_pts), MAP_PT_MARGIN + (LIDAR_BIAS_MARGIN if BIAS_ON_MAP else 0.0)), np.full(len(bp), 0.01)])
        low = np.concatenate([np.zeros(len(self.scan_pts) + len(self.map_pts), bool), np.ones(len(bp), bool)])
        clear = lambda Q, lw: np.where(lw, np.hypot(Q[:, 0], Q[:, 1]) - BODY_R, self._clearance(Q))   # low points: only the body can hit them
        now = clear(P, low) - margin
        keep = now < 0.12
        if not keep.any():
            self.r.set_twist(vx, vy, wz); self.blocked = 0; return
        P, now, margin, low = P[keep], now[keep], margin[keep], low[keep]
        floor_ = np.minimum(now, SAFE_GAP) - 0.002
        # sliding: drop only the part of the velocity that points at the closest obstacle, keep the part along it
        cl = P[np.argmin(now)]; nrm = cl / max(np.hypot(*cl), 1e-6)
        toward = vx * nrm[0] + vy * nrm[1]
        sx, sy = (vx - toward * nrm[0], vy - toward * nrm[1]) if toward > 0 else (vx, vy)
        cands = [(vx, vy, wz), (vx, vy, 0.0), (sx, sy, wz), (sx, sy, 0.0), (0.5 * sx, 0.5 * sy, 0.0), (0.0, 0.0, wz)]
        if ESCAPE:
            # away from the close points, weighted by closeness: lets the robot ease off a wall it was squeezed against,
            # so the turn or the path it wants becomes possible again (instead of freezing until the stuck handler)
            close = now < SAFE_GAP + 0.02
            if close.any():
                w = 1.0 / np.maximum(now[close], 0.005); dirs = P[close] / np.maximum(np.hypot(P[close, 0], P[close, 1]), 1e-6)[:, None]
                e = -(w[:, None] * dirs).sum(0); ne = math.hypot(*e)
                if ne > 1e-6: cands.append((0.04 * e[0] / ne, 0.04 * e[1] / ne, 0.0))
        cands.append((0.0, 0.0, 0.0))
        for k, (cx, cy, cw) in enumerate(cands):
            ok = True
            for dt in SAFE_HORIZON:
                a = -cw * dt; ca, sa = math.cos(a), math.sin(a)
                Q = P - [cx * dt, cy * dt]
                Q = np.stack([ca * Q[:, 0] - sa * Q[:, 1], sa * Q[:, 0] + ca * Q[:, 1]], 1)
                if (clear(Q, low) - margin < floor_).any(): ok = False; break
            if ok:
                self.r.set_twist(cx, cy, cw); self.blocked = self.blocked + 1 if k else 0
                return
        self.r.set_twist(0.0, 0.0, 0.0); self.blocked += 1

    def drive_world(self, vwx, vwy, th_des):
        self._cmd_w = np.array([vwx, vwy])
        x, y, th = self.pose()
        c, s = math.cos(th), math.sin(th)
        self.cmd(c * vwx + s * vwy, -s * vwx + c * vwy, float(np.clip(2.2 * wrap(th_des - th), -W_MAX, W_MAX)))

    def goto(self, goal_fn, ignore=None, check=None, timeout=150.0):
        """Plan and drive to one of the goals produced by goal_fn(D). Returns the goal reached, or None.
        `check` is called every tick; if it returns a string the motion is aborted with that reason."""
        t0 = self.r.t; t_plan = -9.0; fails = 0; goal = None; path = None; wp = 1; self.abort = None
        prog_t, prog_xy = self.r.t, np.array(self.pose()[:2])
        while True:
            if self.r.t - t0 > timeout: self.stop(); self.abort = 'timeout'; return None
            if check:
                why = check()
                if why: self.stop(); self.abort = why; return None
            if self.slip_flag and self.r.t - self.slip_flag > 1.0: self.slip_flag = False        # stale: raised somewhere else
            if self.slip_flag:
                yield from self._slip_recover(); t_plan = -9.0; path = None; prog_t, prog_xy = self.r.t, np.array(self.pose()[:2])
            if self.r.t - t_plan > 0.6:
                t_plan = self.r.t; D = self.costmap(ignore); goals = goal_fn(D)
                if goal is not None:                       # stick with the goal already chosen while it stays valid
                    keep = [g for g in goals if abs(g[0] - goal[0]) < 0.03 and abs(g[1] - goal[1]) < 0.03]
                    goals = keep or goals
                all_goals = goal_fn(D) if goal is not None else goals
                res = self.astar(D, goals) if goals else None
                if res is None and len(all_goals) > len(goals): res = self.astar(D, all_goals)
                if res is None and self.ghosts:
                    # The invisible obstacles remembered after a slip are a preference, not a wall: a 4 cm thing blown up by the
                    # robot's own size can shut a 0.8 m passage (round 12, materials seed 301: two objects out of reach for good).
                    # If they leave no way through, forget them and plan again.
                    self.log('绕不开记下的看不见的障碍，先不管它'); self.ghosts = []
                    D = self.costmap(ignore); goals = goal_fn(D); res = self.astar(D, goals) if goals else None
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
        jj, ii = np.nonzero(self.lo[j0:j1, i0:i1] > MAP_OCC)
        if not len(ii): return True
        Q = np.stack([(ii + i0 + 0.5) * CS - px, (jj + j0 + 0.5) * CS - py], 1)
        for h in np.linspace(heading - sweep, heading + sweep, 7):
            c, s = math.cos(h), math.sin(h)
            B = np.stack([c * Q[:, 0] + s * Q[:, 1], -s * Q[:, 0] + c * Q[:, 1]], 1)
            if (self._clearance(B, box) - MAP_PT_MARGIN < 0.03).any(): return False
        return True

    def grasp_dirs(self, e):
        """Which sides the object can be taken from. The jaws always close along the line from the robot to the
        object (no wrist yaw), so for each side to stand on, measure the object along that line inside a
        finger-wide strip (skills.best_band) using the points the camera has seen on it. Keep the sides whose
        narrowest strip fits the gripper and is within 12 mm of the best side: across a banana, not along it; any side
        of a ball; the narrow arm of an L. Returns [(strip length, side angle, jaw direction u, band)]."""
        if e.get('dirs') is not None: return e['dirs']
        rel = np.column_stack([e['pts'][:, 0] - e['x'], e['pts'][:, 1] - e['y'], e['pts'][:, 2]])
        # Jaw lines every 15 deg (plus square to the long axis). One jaw line can be used from either end, and both
        # ends get the same score: the camera has mostly seen one side, so measuring it again from the other end
        # would only add that bias, and the side that is free of furniture is often the far one.
        angs = [k * math.pi / 12 for k in range(12)]
        if e['elong'] >= ELONG: angs += [(e['heading'] + math.pi / 2 + da) % math.pi for da in (0.0, 0.1, -0.1)]
        bad = e.get('bad', [])
        out = []
        for a in angs:
            u = np.array([-math.cos(a), -math.sin(a)])
            b = best_band(rel, u)
            if b is None: continue
            near, far, off, lw, sc = b
            for side, uu, bb in ((a, u, b), (a + math.pi, -u, (-far, -near, -off, lw, sc))):
                if any(abs(wrap(side - b_)) < e.get('bad_span', 0.35) for b_ in bad): continue      # tried from about here and could not
                out.append((sc, wrap(side), uu, bb))
        if out:   # tier 0: the best sides (within 1 cm of score); tier 1: any side where some strip fits the gripper
            smin = min(o[0] for o in out); out = sorted([(0 if o[0] <= smin + 0.010 else 1,) + o for o in out], key=lambda o: o[:2])
        e['dirs'] = out
        return out

    def standoffs(self, e, D, max_tier=0):
        """Grasp stances for the best sides (tier 0), or for every side where the gripper fits (max_tier=1; also used
        when no best side can be stood at). All of them go to A*, which picks whichever it can reach."""
        out = []; ol = e.get('outline_core', e['outline'])
        for tier, _, a, u, (near, far, off, _, _) in self.grasp_dirs(e):
            if tier > max_tier: break
            v = np.array([-u[1], u[0]])
            t = np.array([e['x'], e['y']]) + 0.5 * (near + far) * u + off * v     # middle of the strip the jaws close on
            # the moving jaw swings out on the far side of the object: that space must be free of furniture
            fx, fy = t + (0.5 * (far - near) + 0.09) * u
            if self._dat(self.D_base, fx, fy) < 0.075: continue
            h = math.atan2(u[1], u[0])
            # The chassis must not touch the object itself (checked exactly, not on the 5 cm grid). When the best strip
            # is at the edge of something bulky (a plush toy's arm), stand back a little: the arm reaches 26.5 cm.
            for dist in STANDOFFS:
                px, py = t - dist * u
                if np.min(np.hypot(ol[:, 0] - px, ol[:, 1] - py)) < 0.18: continue
                # the folded arm must have room to turn a little there without touching furniture
                if self._dat(self.D_base, px, py) >= PASS and self.pose_ok(px, py, h, ARM_BOX, 0.3):
                    out.append((float(px), float(py), h, 0.0))
                break
        if not out and max_tier == 0: return self.standoffs(e, D, 1)
        return out

    PULL_BACK = 0.15

    def pull_stances(self, e, D):
        """Where to stand to drag an object out of a tight spot: facing it, close enough that the arm reaches 2 cm past
        its far side (at most PULL_MAX from the robot centre), with room for the paddle there (away from furniture and
        other objects) and room behind the robot to back up. Only directions that leave the object in more open space
        than now are offered."""
        ctr = np.array([e['x'], e['y']]); ol = e['outline']; out = []
        others = [o['outline'] for j, o in self.memory.items() if j != e['id'] and o['status'] == 'pending']
        others = np.vstack(others) if others else np.zeros((0, 2))
        def room(p):          # how open the floor is around a point: furniture and other objects
            r_ = self._dat(self.D_furn, *p)
            if len(others): r_ = min(r_, float(np.min(np.hypot(others[:, 0] - p[0], others[:, 1] - p[1]))) + 0.03)
            return r_
        now = room(ctr)
        for k in range(24):
            a = k * math.pi / 12; dv = np.array([math.cos(a), math.sin(a)]); u = -dv
            far = float(np.max((ol - ctr) @ u)); near = float(np.max((ol - ctr) @ dv))
            dist = max(0.30 - far - PULL_REACH, near + 0.17)
            if dist + far + PULL_REACH > PULL_MAX - 0.01: continue
            P = ctr + dist * dv; finger = ctr + (far + PULL_REACH) * u
            if self._dat(self.D_furn, *finger) < 0.035: continue
            b = self.basket          # the paddle must not come down on or scrape along the basket wall
            if math.hypot(max(abs(finger[0] - b['x']) - b['sx'] / 2, 0), max(abs(finger[1] - b['y']) - b['sy'] / 2, 0)) < 0.06: continue
            if len(others) and np.min(np.hypot(others[:, 0] - finger[0], others[:, 1] - finger[1])) < 0.015: continue
            if room(ctr + self.PULL_BACK * dv) < now + 0.04: continue
            if self._dat(self.D_base, *P) < PASS or not self.pose_ok(P[0], P[1], math.atan2(u[1], u[0]), ARM_BOX, 0.3): continue
            if any(self._dat(self.D_base, *(P + s * dv)) < PASS - 0.04 for s in (0.05, 0.10, 0.15)): continue
            out.append((float(P[0]), float(P[1]), math.atan2(u[1], u[0]), 0.0))
        return out

    def pull_out(self, k):
        """Drag object k into the open so it can be grasped. Returns True if it was dragged."""
        r = self.r; e = self.memory[k]; cn = self._name(e)
        e['pulls'] = e.get('pulls', 0) + 1
        self.state = '导航'; self.log(f'{cn} 太贴着别的东西，站不到能夹的位置：先把它扒出来')
        goal = yield from self.goto(lambda D: self.pull_stances(e, D), ignore=k)
        if goal is None:
            self.log(f'{cn} 也找不到能扒它的位置'); return False
        self.state = '抓取'
        d = self.measure_close(e)
        if d is None:
            self.log(f'{cn} 不在预期位置，重新找'); self.memory.pop(k, None); self.cov[:] = False; return False
        u = d['rel'][:2] / np.linalg.norm(d['rel'][:2])
        t = d['pts'][:, :2] @ u; z = d['pts'][:, 2]
        near, far = np.percentile(t, [1, 99]); tz = t[z >= z.max() - 0.003].mean()
        far = max(far, 2 * tz - near)                     # hidden far side, as in skills.best_band
        finger = (far + PULL_REACH) * u
        if np.linalg.norm(finger) > PULL_MAX + 0.005:
            self.log(f'手臂伸不到 {cn} 后面'); return False
        ok = yield from pull(r, finger, self.PULL_BACK, drive=lambda vx: self.cmd(vx, 0.0, 0.0), log=self.log)
        if not ok: return False
        self.stats['pulls'] += 1
        x, y, th = self.pose(); sh = -self.PULL_BACK * np.array([math.cos(th), math.sin(th)])
        e.update(x=e['x'] + sh[0], y=e['y'] + sh[1], outline=e['outline'] + sh, pts=e['pts'] + np.append(sh, 0.0), dirs=None, sigma=0.03, retry_at=0.0)
        if e.get('outline_core') is not None: e['outline_core'] = e['outline_core'] + sh
        e.pop('bad', None)
        self.log(f'把 {cn} 往外扒了约 {100 * self.PULL_BACK:.0f} cm')
        return True

    def basket_standoffs(self, D):
        """Places to stand to drop into the basket: off each side, at two distances and three positions along the side,
        facing the basket centre (place() reaches over the near wall toward it). More than one per side, so a single
        thing on the floor next to the basket does not shut that side off."""
        b = self.basket; out = []
        for nx, ny, half, along in ((0, -1, b['sy'] / 2, b['sx']), (0, 1, b['sy'] / 2, b['sx']), (-1, 0, b['sx'] / 2, b['sy']), (1, 0, b['sx'] / 2, b['sy'])):
            for d in (half + 0.25, half + 0.28):
                for o in (0.0, -0.3 * along, 0.3 * along):
                    px, py = b['x'] + nx * d + abs(ny) * o, b['y'] + ny * d + abs(nx) * o
                    h = math.atan2(b['y'] - py, b['x'] - px)
                    if self._dat(D, px, py) >= PASS and self.pose_ok(px, py, h, HELD_BOX, 0.5): out.append((float(px), float(py), h, 0.0))
        return out

    def measure_close(self, e):
        """Close look with the front camera right before grasping: the detection nearest where the object should be,
        or None if there is nothing there."""
        x, y, th = self.pose(); c, s_ = math.cos(th), math.sin(th)
        ex, ey = c * (e['x'] - x) + s_ * (e['y'] - y), -s_ * (e['x'] - x) + c * (e['y'] - y)
        blind = bool(e.get('blind'))
        dets = self.cam.detect(self.basket_rect(), n_avg=CLOSE_FRAMES)
        if blind:     # known only by the no-reading cluster it leaves in the depth image
            cands = [clear_detection(h, self.cam) for h in self.cam.holes if h['n_px'] >= 60 and not h['cut']]
        else:
            cands = [d for d in dets if d['n_px'] >= 60 and d['bottom'] < 0.04]
        if not cands: return None
        d = min(cands, key=lambda d: math.hypot(d['rel'][0] - ex, d['rel'][1] - ey))
        if math.hypot(d['rel'][0] - ex, d['rel'][1] - ey) > ((BLIND_LOOK_WIDE if e.get('look_fail') else 0.10) if blind else 0.08): return None
        if not (0.13 < d['rel'][0] < 0.42 and abs(d['rel'][1]) < 0.16): return None
        return d

    def fetch(self, k):
        r = self.r; e = self.memory[k]; cn = self._name(e); self.target = k
        if not self.grasp_dirs(e):
            e['fails'] += 1; e['retry_at'] = r.t + 40 * e['fails']; e.pop('bad', None); e['dirs'] = None
            e['why'] = '找不到能下手的地方' if e['fails'] > 1 else '比夹爪开口宽'
            self.log(f'{cn} 从哪边量都比夹爪开口宽，先跳过'); return False
        tries = 0; restand = 0
        while True:
            self.state = '导航'; self.log(f'前往 {cn}' + (f'（第 {tries + 1} 次）' if tries else ''))
            goal = yield from self.goto(lambda D: self.standoffs(e, D), ignore=k, timeout=75.0)
            if goal is None and self.abort == 'no_path':       # the best sides are cut off: try the others too
                goal = yield from self.goto(lambda D: self.standoffs(e, D, 1), ignore=k, timeout=75.0)
            if goal is None and self.abort in ('no_path', 'timeout') and e.get('pulls', 0) < 2 and not e.get('blind'):
                # nowhere to stand to grasp it: drag it into the open, then try again
                pulled = yield from self.pull_out(k)
                if pulled: continue
                if k not in self.memory: return False
            if goal is None and e.get('blind') and e.get('bad_span') and not e.get('side_gave_up'):
                # no way to the side we wanted to look from (against furniture, a wall): take it from where it can be reached
                e['side_gave_up'] = True; e.pop('bad', None); e.pop('bad_span', None); e['dirs'] = None
                self.log(f'{cn} 侧面站不进去，就从能站的一侧下手'); continue
            if goal is None:
                e['fails'] += 1; e['retry_at'] = r.t + 30 * e['fails']; e['why'] = '够不到' if self.abort == 'no_path' else '导航超时'
                self.log(f'{cn} {e["why"]}，先跳过'); return False
            self.state = '抓取'
            d = self.measure_close(e)
            if d is None and e.get('blind') and e.get('look_fail', 0) < 2:
                # a cluster this close can fall outside what the camera sees; step back and look again before giving up on it
                e['look_fail'] = e.get('look_fail', 0) + 1
                x, y, th = self.pose(); self.log(f'{cn} 近看没看到，退后一点再看')
                yield from self.servo((x - 0.10 * math.cos(th), y - 0.10 * math.sin(th), th), timeout=4.0)
                d = self.measure_close(e)
            if d is None:
                self.log(f'{cn} 不在预期位置，重新找'); self.memory.pop(k, None); self.cov[:] = False; return False
            e.pop('look_fail', None)
            o = self._to_world(d, *self.pose())
            moved = math.hypot(o['x'] - e['x'], o['y'] - e['y'])
            e.update(sigma=0.012, **o)
            if d.get('blind'): e['seen_t'] = r.t
            if d.get('blind'):
                self._set_blind(e, o, d['ambiguous'])
                x, y, _ = self.pose(); ang = math.atan2(y - e['y'], x - e['x'])
                if d['ambiguous']:
                    # Seen end-on, or short: the cluster is only as wide as the thing is thick, and does not say how long it
                    # is. Look once more from the side (a stance 60 degrees or more round from this one); if that is
                    # no better either, it is a short thing and is taken from where we are.
                    looks = e.setdefault('looks', [])
                    if restand < 4 and not e.get('side_gave_up') and all(abs(wrap(ang - a_)) < 1.0 for a_ in looks):
                        looks.append(ang); restand += 1; e['bad'] = list(looks); e['bad_span'] = 1.0; e['dirs'] = None
                        if self.grasp_dirs(e):
                            self.log(f'{cn} 这个角度看不出多长，换到侧面再看'); continue
                        e.pop('bad', None); e.pop('bad_span', None); e['dirs'] = None
                    else:
                        e.pop('bad_span', None)               # looked from two sides and it is short: no more looking round
                else:
                    # a whole look: the direction and length are known. Stand on one of the best sides, if this is not one.
                    e.pop('bad', None); e.pop('bad_span', None); e['dirs'] = None
                    dirs_ = self.grasp_dirs(e)
                    if restand < 4 and dirs_:
                        # Is the strip the jaws would close on from here (best_band along the line from this stance) more than
                        # 1 cm wider than from the best side? The same test grasp_dirs uses for its best sides: a heading that
                        # is off by 20-30 degrees only widens it by a few mm, and is no reason to move.
                        relp = np.column_stack([e['pts'][:, 0] - e['x'], e['pts'][:, 1] - e['y'], e['pts'][:, 2]])
                        cur = best_band(relp, np.array([-math.cos(ang), -math.sin(ang)]))
                        if cur is None or cur[4] > min(o[1] for o in dirs_) + 0.010:
                            restand += 1; e['bad'] = [ang]; e['dirs'] = None
                            self.log(f'{cn} 看清了朝向，换到能夹窄边的一侧'); continue
            plan = plan_grasp(d)
            # the stance was chosen from a farther look; now that the object is seen up close, re-stand if it is off
            if restand < 4 and (moved > 0.04 or abs(d['rel'][1]) > 0.06 or plan is None):
                restand += 1
                if plan is None:
                    x, y, _ = self.pose(); e.setdefault('bad', []).append(math.atan2(y - e['y'], x - e['x'])); e['dirs'] = None
                    self.log('从这边下不了手，换个站位')
                    if not self.grasp_dirs(e): restand = 99
                continue
            if plan is None and e.get('pulls', 0) < 2 and not e.get('blind'):
                pulled = yield from self.pull_out(k)
                if pulled: restand = 0; continue
                if k not in self.memory: return False
            if plan is None:
                e['fails'] += 1; e['retry_at'] = r.t + 40 * e['fails']; e['why'] = '找不到能下手的地方'; e.pop('bad', None); e['dirs'] = None
                self.log(f'{cn} 找不到能下手的地方，先跳过'); return False
            self.stats['grasp_attempts'] += 1; tries += 1
            ok = yield from grasp(r, plan, self.log, look=self.wrist.holding)
            if ok:
                yield from carry_pose(r)
                ok = holding(r, plan)
                if not ok: self.log('收臂时掉了')
            if ok: break
            self.stats['grasp_fail'] += 1
            r.set_arm(grip=0.3); yield from carry_pose(r, speed=1.0); yield from wait(r, 0.6)
            # next attempt from another side, if there is one: a different contact geometry, not the same failure again
            x, y, _ = self.pose(); e.setdefault('bad', []).append(math.atan2(y - e['y'], x - e['x'])); e['dirs'] = None
            if not self.grasp_dirs(e): e.pop('bad', None); e['dirs'] = None
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
        self.state = '放置'; held_true = self._score_held_obj()
        x, y, th = self.pose()
        bx, by = self.basket['x'] - x, self.basket['y'] - y; dist = math.hypot(bx, by)
        half = self.basket['sx'] / 2 if abs(math.cos(goal[2])) > 0.7 else self.basket['sy'] / 2
        if dist - half + 0.05 > PLACE_REACH:
            # standing too far back (the far stance, or stopped short of it): the drop point would land on the rim,
            # and a soft thing released there drapes over the wall and stays between the jaws. Step in first.
            step = dist - half + 0.05 - PLACE_REACH + 0.02
            yield from self.servo((x + bx / dist * step, y + by / dist * step, th), timeout=4.0)
            x, y, th = self.pose()
            bx, by = self.basket['x'] - x, self.basket['y'] - y; dist = math.hypot(bx, by)
        reach = float(np.clip(dist - half + 0.05, 0.24, PLACE_REACH))      # drop 5 cm inside the near wall
        c, s = math.cos(th), math.sin(th)
        ux, uy = bx / dist, by / dist; wx, wy = x + ux * reach, y + uy * reach
        rel = (c * (wx - x) + s * (wy - y), -s * (wx - x) + c * (wy - y))
        ok = yield from place(r, rel, self.basket['h'], self.log, look=self.wrist.holding)
        if not ok:                                   # could not reach over the rim from here: set it down, try later
            yield from self.put_down(k); return False
        if ok == 'stuck':
            # Still on the gripper after shaking and pressing. Nothing more can be grasped with it there, and it hangs
            # in front of the camera: stop and ask for help rather than drive about with it.
            self.log(f'{cn} 粘在夹爪上弄不下来，停下等人帮忙'); self.stats['stuck_payload'] = 1
            self.help_needed = True; return False
        yield from wait(r, 0.5)
        # The robot assumes it went in. If it bounced out, the camera will see it on the floor later and reopen it.
        e['status'] = 'delivered'; self.log(f'{cn} 已放进收纳筐')
        ok_true = self._in_basket_world(r.obj_pos(held_true))                      # score sheet only from here
        if not ok_true: self.stats['missed_place'] += 1
        self.score_events.append((round(r.t, 1), 'place', r.objects[held_true][0], bool(ok_true)))
        return True

    def put_down(self, k):
        """Set a held object on the floor in front of the robot and remember where it is."""
        from skills import IK
        r = self.r; e = self.memory[k]; self.state = '放下'
        q, _, _ = IK.solve([0.25, 0, 0.06])
        r.set_arm(q5=q, speed=0.6); yield from wait_arm(r)
        r.set_arm(grip=1.0); yield from wait(r, 0.6)
        yield from carry_pose(r, speed=1.0); r.set_arm(grip=0.3)
        x, y, th = self.pose()
        nx, ny = x + 0.25 * math.cos(th), y + 0.25 * math.sin(th); sh = np.array([nx - e['x'], ny - e['y']])
        e.update(status='pending', x=nx, y=ny, outline=e['outline'] + sh, pts=e['pts'] + np.append(sh, 0.0), dirs=None, sigma=0.03, retry_at=r.t + 5)
        if e.get('outline_core') is not None: e['outline_core'] = e['outline_core'] + sh
        self.basket_blocked_t = r.t
        # whatever sits closest to the basket is probably what blocks the way in: deal with that first
        cands = [(math.hypot(o['x'] - self.basket['x'], o['y'] - self.basket['y']), j) for j, o in self.memory.items() if o['status'] == 'pending' and j != k]
        if cands: self.priority = min(cands)[1]

    def explore_step(self, stop_on_find=True):
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
        # stop well inside camera range of the cell: standing 0.55 m off with a 0.6 m range (second sweep) meant each
        # spin barely saw the cell, and the sweep crept along 10 cm per spin
        rad = min(0.55, self.cam_range - 0.25)
        goal = yield from self.goto(lambda D: [(cx, cy, None, rad)], check=(lambda: 'found' if self.pick() is not None else None) if stop_on_find else None, timeout=60)
        if goal is None and self.abort == 'found': return True
        if goal is not None and not self.cov[j, i]: yield from self.spin()
        self.cov[j, i] = True
        return True

    def pick(self):
        x, y, _ = self.pose(); best = None
        p = self.priority
        if p is not None:
            e = self.memory.get(p)
            if e and e['status'] == 'pending' and e['retry_at'] <= self.r.t and not (self.person_zones and any(math.hypot(e['x'] - zx, e['y'] - zy) < zr for zx, zy, zr in self.person_zones)): return p
            if self.r.t - self.basket_blocked_t < 25.0: return None   # no point picking up anything else yet
            self.priority = None
        for k, e in self.memory.items():
            if e['status'] == 'pending' and e['retry_at'] <= self.r.t:
                if self.person_zones and any(math.hypot(e['x'] - zx, e['y'] - zy) < zr for zx, zy, zr in self.person_zones):
                    e['retry_at'] = self.r.t + 10.0; self.zone_skip_t = self.r.t; continue                 # someone is standing by it: another one first, and look again in a few seconds
                d = math.hypot(e['x'] - x, e['y'] - y)
                if best is None or d < best[0]: best = (d, k)
        return None if best is None else best[1]

    def run(self, resume=False):
        """The whole job. `resume`: pick it up again after the supervisor had the robot doing something else (charging, an operator, a
        fall): the map, the memory and the clock carry on, so the look-round at start-up is skipped."""
        self.abort = None
        if not resume:
            yield from wait(self.r, 0.5)
            self.state = '巡视'; self.log('开机，原地环顾建图')
            yield from self.spin()
        while True:
            yield from self._work()
            if self.help_needed:
                self.stop(); self.state = '求助'; return
            if self.pause is not None and self.pause():
                self.stop(); self.state = '暂停'; return
            self.state = '回位'; self.log('没有待办，回起点')
            yield from self.goto(lambda D: [(HOME[0], HOME[1], HOME[2], 0.0)], check=lambda: 'found' if self.pick() is not None else None, timeout=90)
            if self.abort == 'found' and self.r.t < T_MAX:
                self.log('回去的路上又看见东西，接着干'); continue
            break
        self.state = '完成'

    def _work(self):
        """Fetch what is known, explore for more, wait for skipped things to come due; until nothing is left or time is up."""
        while self.r.t < T_MAX:
            if self.pause is not None and self.pause(): return         # asked to stop at a safe point (nothing in the gripper, between jobs)
            k = self.pick()
            if k is not None:
                yield from self.fetch(k); self.target = None
                if k in self.memory and self.memory[k]['status'] == 'pending': self._reset_blind(self.memory[k])
                if self.help_needed: return
                self.held = False
                continue
            more = yield from self.explore_step()
            if not more:
                waiting = [e for e in self.memory.values() if e['status'] == 'pending']
                if not waiting and self.cam_range > CAM_RANGE_2 and T_MAX - self.r.t > 120:
                    # small dark things are easy to miss from afar: go round once more and look from closer up
                    self.cam_range = CAM_RANGE_2; self.cov[:] = False; self.log('看完一遍了，再走近仔细看一遍')
                    continue
                if not waiting: return
                t_next = min(e['retry_at'] for e in waiting)
                if t_next > T_MAX: return
                self.state = '等待'; yield from wait(self.r, max(0.5, min(5.0, t_next - self.r.t)))
