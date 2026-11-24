"""People in the home: simulated humans, a head camera, finding them in depth, child or adult from height, who it is (only if enrolled), and the safety layer.

What is real here and what is a stand-in (the simulator cannot test everything; this says which is which):

  Simulated humans    mocap bodies (legs, torso, shoulders, arms, head) of five heights from a toddler to a tall man, walking, standing, sitting,
                      crouching along scripts. The lidar sees their legs, the cameras see the whole body. Contacts with the robot are counted.
  HeadCam             a depth camera on a short mast (0.5 m, tilted 28 degrees up, 91 x 60 degree field of view, the noise of a consumer stereo camera).
                      The base camera sits 12 cm above the floor and sees only legs; a person's height needs something that sees the head.
  ArmCam              the same camera on the robot's arm instead of a mast (model.ARM_CAMS): where it looks follows the joints, so its pose is computed from the servo readings (with the error
                      of their calibration) for every frame, and it is used only while it looks somewhere useful and the arm is not moving.
  PersonDetector      depth only: points 0.15 to 2.1 m above the floor, seen from above on a 4 cm grid, clusters = people when they are tall and compact enough;
                      height = the top of the cluster (a lower bound when it touches the top of the image), shoulder width at 80 % of the height.
                      A static-map check rejects tall furniture the scan has already seen. Tracked over frames (nearest neighbour, constant velocity).
  Age class           from the tracked height (and the shoulder width for a sitting adult): child < 1.30 m, adult >= 1.50 m, in between "unsure" and treated as a child
                      for safety. That is about "under 11 years" - a 12-year-old is counted as an adult. It is a size class, not an age estimate.
  Identity            STAND-IN. The simulator cannot render faces, so the recogniser returns a feature vector drawn per person (siblings share part of it) with noise that grows
                      with distance and when the person is not facing the camera. It tests what is ours: the opt-in registry, thresholds, fusion over a track, the refusal to
                      name strangers, and deleting. It does not say how well a real face or body model works; that has to be measured with the real model and real children.
  Guard               the safety layer between the modes and the wheels/arm: slow near people, stop inside a radius (larger for children), hold the arm still while someone is
                      within reach. Same code runs whatever mode is active.

Nothing here stores an image. The registry keeps a feature vector, a name, a role and when consent was given, in a local file the owner can read and delete.
"""
import json
import math
import os
import time
from contextlib import contextmanager

import mujoco
import numpy as np
from scipy import ndimage

import perturb
from store import LOG_MAX, dump_json, load_log

# ---------------------------------------------------------------------------------------------------------------- the simulated people
SPECS = {
    'adult_m': dict(h=1.78, w=0.46, hip=0.52, head=0.125, cn='成年男性', cls='adult', clothes=('0.25 0.32 0.55 1', '0.20 0.20 0.24 1')),
    'adult_f': dict(h=1.63, w=0.40, hip=0.52, head=0.130, cn='成年女性', cls='adult', clothes=('0.70 0.30 0.35 1', '0.22 0.22 0.30 1')),
    'child_8': dict(h=1.28, w=0.32, hip=0.49, head=0.145, cn='8 岁孩子', cls='child', clothes=('0.95 0.75 0.20 1', '0.25 0.35 0.60 1')),
    'child_5': dict(h=1.10, w=0.27, hip=0.47, head=0.155, cn='5 岁孩子', cls='child', clothes=('0.35 0.70 0.45 1', '0.30 0.30 0.50 1')),
    'toddler': dict(h=0.88, w=0.23, hip=0.45, head=0.170, cn='2~3 岁幼儿', cls='child', clothes=('0.90 0.50 0.70 1', '0.40 0.50 0.70 1')),
}
SKIN = '0.87 0.69 0.58 1'
SIT_DROP = 0.45                 # how far the hips drop when sitting or crouching, as a share of the standing hip height
DOORS = [(0.12, 1.3), (3.88, 1.3), (1.5, 2.88)]           # where people come in and leave the room (the side of the room the robot is not on is picked)
PARK = (-7.0, -7.0)             # where a person is before the script starts: far outside the room (the walls are 0.4 m low and the head camera sees over them, so it must be out of its 5 m range)


def _dims(s):
    h, w = s['h'], s['w']; hip = s['hip'] * h; head = s['head'] * h; neck = 0.04 * h
    return dict(hip=hip, sh=h - head - neck, head=head, ly=0.19 * w, rl=max(0.026 * h, 0.022), ra=0.019 * h, rt=0.30 * w, halfsh=w / 2 - 0.02 * h)


def people_xml(names):
    """MJCF bodies for the people (mocap, so a script moves them) and nothing else. names: list of keys of SPECS."""
    out = []
    for i, n in enumerate(names):
        s = SPECS[n]; D = _dims(s); shirt, trousers = s['clothes']; p = f'p{i}_'
        out.append(f'<body name="{p}torso" mocap="true" pos="{PARK[0]} {PARK[1]} {D["hip"]}">'
                   f'<geom type="capsule" fromto="0 0 0 0 0 {D["sh"] - D["hip"] - 0.03}" size="{D["rt"]}" rgba="{shirt}"/>'
                   f'<geom type="capsule" fromto="0 {-D["halfsh"]} {D["sh"] - D["hip"] - 0.03} 0 {D["halfsh"]} {D["sh"] - D["hip"] - 0.03}" size="{0.04 * s["h"]}" rgba="{shirt}"/></body>')
        out.append(f'<body name="{p}head" mocap="true" pos="{PARK[0]} {PARK[1]} {s["h"] - D["head"] / 2}">'
                   f'<geom type="ellipsoid" size="{0.40 * D["head"]} {0.36 * D["head"]} {0.5 * D["head"]}" rgba="{SKIN}"/></body>')
        for side, sg in (('L', 1), ('R', -1)):
            out.append(f'<body name="{p}leg{side}" mocap="true" pos="{PARK[0]} {PARK[1]} {D["hip"]}">'
                       f'<geom type="capsule" fromto="0 0 0 0 0 {-(D["hip"] - D["rl"])}" size="{D["rl"]}" rgba="{trousers}"/></body>')
            out.append(f'<body name="{p}arm{side}" mocap="true" pos="{PARK[0]} {PARK[1]} {D["sh"]}">'
                       f'<geom type="capsule" fromto="0 0 0 0 0 {-0.36 * s["h"]}" size="{D["ra"]}" rgba="{shirt}"/></body>')
    return '\n'.join(out)


def _quat(yaw, pitch):
    """yaw about z, then pitch about the body's y (positive pitch swings the hanging limb backwards)."""
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])


class Script:
    """A person's day as steps, run in time (the person reacts to the robot only in 'chase' and 'face_robot'):
         ('at', x, y, yaw)            appear there at once
         ('walk', x, y, speed)        walk to the point at that speed (m/s)
         ('wait', seconds)            stand
         ('turn', yaw)                face a direction at once
         ('face_robot',)              from now on, face the robot whenever standing
         ('posture', 'sit'|'stand', seconds)    sit or crouch (the hips drop) or get up, taking that long
         ('chase', speed, gap, max_s) walk straight at the robot until the gap between their surfaces is `gap` m (or max_s seconds), like a child running up to it
         ('enter',)                   come in through the door farthest from the robot
         ('exit', speed)              walk to a door (the nearest one that is not close to the robot) and leave
         ('to_robot', gap, speed)     walk up to the robot until the gap is `gap` m (they walk round it, never through it)
         ('hide',)                    leave the room at once
    Walking people keep 0.45 m between their centre and the robot's: they step round it.
    Before the first step the person is outside the room."""
    def __init__(self, steps, t0=0.0):
        self.steps = list(steps); self.i = 0; self.t0 = t0; self.x, self.y = PARK; self.yaw = 0.0; self.s = 0.0; self.v = 0.0; self.face = False
        self._t = None; self._k = 0.0; self._started = False

    def _step_done(self): self.i += 1; self._k = 0.0; self._started = False

    def at(self, t, robot_xy=None):
        dt = 0.0 if self._t is None else max(t - self._t, 0.0); self._t = t; self.v = 0.0
        if t < self.t0: return self.x, self.y, self.yaw, 0.0, self.s
        guard = 0
        while self.i < len(self.steps) and guard < 20:
            guard += 1; st = self.steps[self.i]; k = st[0]
            if k == 'at': self.x, self.y, self.yaw = st[1], st[2], st[3]; self._step_done(); continue
            if k == 'turn': self.yaw = st[1]; self._step_done(); continue
            if k == 'face_robot': self.face = True; self._step_done(); continue
            if k == 'hide': self.x, self.y = PARK; self._step_done(); continue
            if k == 'wait':
                self._k += dt
                if self._k >= st[1]: self._step_done(); continue
                break
            if k == 'enter':
                far = max(DOORS, key=lambda p: math.hypot(p[0] - robot_xy[0], p[1] - robot_xy[1])) if robot_xy is not None else DOORS[0]
                self.x, self.y = far; self.yaw = math.atan2(1.5 - far[1], 2.0 - far[0]); self._step_done(); continue
            if k == 'exit':
                ok = [p for p in DOORS if robot_xy is None or math.hypot(p[0] - robot_xy[0], p[1] - robot_xy[1]) > 1.2] or DOORS
                door = min(ok, key=lambda p: math.hypot(p[0] - self.x, p[1] - self.y)); self.steps[self.i] = ('walk', door[0], door[1], st[1]); self.steps.insert(self.i + 1, ('hide',)); continue
            if k == 'to_robot':
                if robot_xy is None: break
                dx, dy = robot_xy[0] - self.x, robot_xy[1] - self.y; d = math.hypot(dx, dy); target = st[1] + 0.105 + 0.14
                if d <= target + 0.01: self._step_done(); continue
                self.steps[self.i] = ('walk_dyn', st[1], st[2]); continue
            if k in ('walk', 'walk_dyn'):
                if k == 'walk_dyn':                                     # walk at the robot's current position until the gap is reached
                    if robot_xy is None: break
                    tx, ty = robot_xy; target = st[1] + 0.105 + 0.14; dd = math.hypot(tx - self.x, ty - self.y)
                    if dd <= target + 0.01: self._step_done(); continue
                    sp = st[2]; goal_d = dd - target
                else: tx, ty = st[1], st[2]; sp = st[3]; goal_d = None
                dx, dy = tx - self.x, ty - self.y; d = math.hypot(dx, dy)
                self._k += dt
                if not self._started: self._d0 = d; self._started = True
                if self._k > 5.0 + 3.0 * self._d0 / max(sp, 0.1): self._step_done(); continue          # cannot get there (the robot is in the way): give up and carry on with the day
                if d < 1e-6: self._step_done(); continue
                if goal_d is not None: d = goal_d
                self.yaw = math.atan2(dy, dx); step = min(sp * dt, d); nx, ny = self.x + dx / math.hypot(dx, dy) * step, self.y + dy / math.hypot(dx, dy) * step
                if robot_xy is not None and k == 'walk':                # step round the robot instead of through it
                    rx, ry = robot_xy; rd = math.hypot(nx - rx, ny - ry)
                    if rd < 0.45:
                        ux, uy = (nx - rx) / max(rd, 1e-6), (ny - ry) / max(rd, 1e-6); tx_, ty_ = -uy, ux
                        if tx_ * dx + ty_ * dy < 0: tx_, ty_ = -tx_, -ty_
                        nx, ny = rx + ux * 0.45 + tx_ * step, ry + uy * 0.45 + ty_ * step
                self.v = math.hypot(nx - self.x, ny - self.y) / dt if dt > 0 else 0.0; self.x, self.y = nx, ny
                if k == 'walk' and math.hypot(st[1] - self.x, st[2] - self.y) < 0.03: self._step_done()
                break
            if k == 'posture':
                self._k += dt; a = 1.0 if st[1] == 'sit' else 0.0
                if not self._started: self._s0 = self.s; self._started = True
                f = 1.0 if st[2] <= 0 else min(self._k / st[2], 1.0); self.s = self._s0 + f * (a - self._s0)
                if f >= 1.0: self._step_done(); continue
                break
            if k == 'chase':
                self._k += dt
                if robot_xy is None: break
                dx, dy = robot_xy[0] - self.x, robot_xy[1] - self.y; d = math.hypot(dx, dy); target = st[2] + 0.105 + 0.14
                self.yaw = math.atan2(dy, dx)
                if d <= target + 0.01 or self._k >= st[3]: self._step_done(); continue
                step = min(st[1] * dt, d - target); self.x += dx / d * step; self.y += dy / d * step; self.v = step / dt if dt > 0 else 0.0
                break
            self._step_done()
        if self.face and self.v < 0.01 and robot_xy is not None: self.yaw = math.atan2(robot_xy[1] - self.y, robot_xy[0] - self.x)
        return self.x, self.y, self.yaw, self.v, self.s


class People:
    """Runtime of the simulated people: every control tick moves the mocap bodies along their scripts."""
    def __init__(self, robot, names, scripts, comply=None, seed=0):
        self.r = robot; m = robot.m; self.names = list(names); self.scripts = scripts; self.comply = list(comply) if comply is not None else [1.0] * len(names); self._rng = np.random.default_rng(seed + 77); self.asked = 0; self.moved_aside = 0; self.dist = [0.0] * len(names); self.phase = [0.0] * len(names)
        nid = lambda n: m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)]
        self.mid = [{p: int(nid(f'p{i}_{p}')) for p in ('torso', 'head', 'legL', 'legR', 'armL', 'armR')} for i in range(len(names))]
        self.bodies = [{mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'p{i}_{p}') for p in ('torso', 'head', 'legL', 'legR', 'armL', 'armR')} for i in range(len(names))]
        self.root = int(m.body_rootid[robot.chassis]); self.state = [None] * len(names)
        self.contact_events = [0] * len(names); self._in_contact = [False] * len(names); self.contact_t = [0.0] * len(names); self.contact_speed = []
        self.min_gap = [9.0] * len(names); self._last_xy = None; self.stats_t = 0.0
        self.update()

    def update(self):
        """Place every person for the current simulation time. Call once per control tick, before stepping the physics."""
        r, d = self.r, self.r.d; t = r.t; rxy = d.xpos[r.chassis][:2]
        for i, n in enumerate(self.names):
            s = SPECS[n]; D = _dims(s); x, y, yaw, v, sit = self.scripts[i].at(t, rxy)
            self.phase[i] += v * 0.02 / (0.42 * s['h'] / 1.7) * math.pi                      # one leg swing per stride
            amp = 0.5 * min(v / 1.0, 1.0) * (1 - sit); sw = amp * math.sin(self.phase[i])
            drop = SIT_DROP * D['hip'] * sit; hipz = D['hip'] - drop; shz = D['sh'] - drop
            c, sn = math.cos(yaw), math.sin(yaw); M = self.mid[i]
            def put(part, dx, dy, z, pitch):
                d.mocap_pos[M[part]] = [x + c * dx - sn * dy, y + sn * dx + c * dy, z]; d.mocap_quat[M[part]] = _quat(yaw, pitch)
            put('torso', 0, 0, hipz, 0.0); put('head', 0.0, 0, s['h'] - drop - D['head'] / 2, 0.0)
            leg_pitch = lambda side: (-1.30 * sit) + (sw if side == 'L' else -sw)
            put('legL', 0, D['ly'], hipz, leg_pitch('L')); put('legR', 0, -D['ly'], hipz, leg_pitch('R'))
            put('armL', 0, D['halfsh'] + D['ra'], shz - 0.02, -sw * 0.8 if not sit else 0.0); put('armR', 0, -(D['halfsh'] + D['ra']), shz - 0.02, sw * 0.8 if not sit else 0.0)
            self.state[i] = dict(i=i, name=n, x=x, y=y, yaw=yaw, v=v, sit=sit, cls=s['cls'], h=s['h'] - drop, h_stand=s['h'], w=s['w'], inside=x > -0.5)

    def truth(self): return [dict(s) for s in self.state if s is not None]

    def ask_to_move(self, robot_xy, reach=1.4):
        """The robot said 'please make way': a person close to it steps 1 m further off after a second, with the probability in `comply` (an assumption about people, not a measurement: a grown-up does,
        a toddler does not). Their script goes on afterwards."""
        self.asked += 1
        for i, s in enumerate(self.state):
            if s is None or not s['inside'] or math.hypot(s['x'] - robot_xy[0], s['y'] - robot_xy[1]) > reach or self._rng.random() >= self.comply[i]: continue
            dx, dy = s['x'] - robot_xy[0], s['y'] - robot_xy[1]; d = max(math.hypot(dx, dy), 1e-3); ax, ay = float(np.clip(s['x'] + dx / d * 1.0, 0.25, 3.75)), float(np.clip(s['y'] + dy / d * 1.0, 0.25, 2.75))
            sc = self.scripts[i]; sc.steps[sc.i:sc.i] = [('wait', 1.0), ('walk', ax, ay, 0.8)]; sc._k = 0.0; sc._started = False; self.moved_aside += 1

    def after_step(self):
        """Contacts between a person and the robot, and the closest approach (centre to surface)."""
        r = self.r; m, d = r.m, r.d; self.stats_t += 0.02
        pos = d.xpos[r.chassis][:2].copy(); sp = 0.0 if self._last_xy is None else float(np.hypot(*(pos - self._last_xy))) / 0.02; self._last_xy = pos
        touching = [False] * len(self.names)
        for k in range(d.ncon):
            c = d.contact[k]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for i, bs in enumerate(self.bodies):
                if (b1 in bs and int(m.body_rootid[b2]) == self.root) or (b2 in bs and int(m.body_rootid[b1]) == self.root): touching[i] = True
        for i, t in enumerate(touching):
            if t and not self._in_contact[i]: self.contact_events[i] += 1; self.contact_speed.append(round(sp, 3))
            if t: self.contact_t[i] += 0.02
            self._in_contact[i] = t
            s = self.state[i]
            if s and s['inside']: self.min_gap[i] = min(self.min_gap[i], math.hypot(s['x'] - pos[0], s['y'] - pos[1]) - 0.105 - 0.12)


# ---------------------------------------------------------------------------------------------------------------- the head camera
HW, HH = 320, 180                # image size
HEAD_RANGE = 5.0                 # m
HEAD_SIGMA = (0.002, 0.005)      # depth noise a + b z^2 (m), as for the base camera


@contextmanager
def _zfar(m, z):
    """The model's far clip is 2 m (the room is small); the head camera needs to see people across it. Only while it renders."""
    old = m.vis.map.zfar; m.vis.map.zfar = z
    try: yield
    finally: m.vis.map.zfar = old


class HeadCam:
    """Depth image of the mast camera, as points in the chassis frame (z from the floor) and in the world through the robot's own pose estimate."""
    arm = False                      # True for ArmCam: the pose changes from frame to frame and the camera is not always looking at anything useful

    def __init__(self, robot, rng, name='head'):
        self.r = robot; self.rng = rng; m = robot.m; self.name = name
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        self.dep = mujoco.Renderer(m, HH, HW); self.dep.enable_depth_rendering()
        self.seg = mujoco.Renderer(m, HH, HW); self.seg.enable_segmentation_rendering()
        f = (HH / 2) / math.tan(math.radians(m.cam_fovy[self.cam]) / 2); self.f = f
        u, v = np.meshgrid(np.arange(HW) + 0.5, np.arange(HH) + 0.5)
        self.ray = np.stack([(u - HW / 2) / f, -(v - HH / 2) / f, -np.ones_like(u)], -1)
        d = robot.d; Rb = d.xmat[robot.chassis].reshape(3, 3); pb = d.xpos[robot.chassis]
        Rc = d.cam_xmat[self.cam].reshape(3, 3); pc = d.cam_xpos[self.cam]
        self.R_bc = Rb.T @ Rc; self.t_bc = Rb.T @ (pc - pb); self.t_bc[2] += pb[2]
        ce = perturb.get('head_cam_err', None)           # mount error: (position sigma m, rotation sigma deg)
        if ce:
            er = np.random.default_rng(int(rng.integers(1 << 30))); self.t_bc = self.t_bc + er.normal(0, ce[0], 3)
            a = er.normal(0, math.radians(ce[1]), 3); q = np.zeros(4); Rx = np.zeros(9)
            mujoco.mju_axisAngle2Quat(q, a / max(np.linalg.norm(a), 1e-9), float(np.linalg.norm(a))); mujoco.mju_quat2Mat(Rx, q); self.R_bc = Rx.reshape(3, 3) @ self.R_bc
        root = int(m.body_rootid[robot.chassis]); self.self_geoms = np.array([g for g in range(m.ngeom) if int(m.body_rootid[m.geom_bodyid[g]]) == root])
        self.depth = None; self.self_mask = None; self.self_share = 0.0

    def ready(self): return True

    @property
    def height(self): return float(self.t_bc[2])

    def capture(self):
        d, m = self.r.d, self.r.m
        with _zfar(m, HEAD_RANGE + 0.5):
            self.dep.update_scene(d, self.name); depth = self.dep.render().astype(np.float64)
            self.seg.update_scene(d, self.name); s = self.seg.render()
        lab = np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)
        self.self_mask = np.isin(lab, self.self_geoms); self.self_share = float(self.self_mask.mean())
        depth = depth + self.rng.normal(0, 1, depth.shape) * (HEAD_SIGMA[0] + HEAD_SIGMA[1] * depth ** 2) * perturb.get('depth_noise', 1.0)
        far = depth >= HEAD_RANGE; depth[far] = 0.0                                   # nothing measured there
        e = self._edges(depth); depth[e] = 0.0
        self.depth = depth; return depth

    @staticmethod
    def _edges(d):
        """Pixels next to a depth jump (flying pixels): dropped, as every stereo camera's filter does."""
        tol = 0.04 + 0.03 * d; e = np.zeros(d.shape, bool)
        j = (np.abs(d[1:] - d[:-1]) > np.minimum(tol[1:], 0.25)) & (d[1:] > 0) & (d[:-1] > 0); e[1:] |= j; e[:-1] |= j
        j = (np.abs(d[:, 1:] - d[:, :-1]) > np.minimum(tol[:, 1:], 0.25)) & (d[:, 1:] > 0) & (d[:, :-1] > 0); e[:, 1:] |= j; e[:, :-1] |= j
        return e

    def points(self, pose):
        """-> world points (HH, HW, 3) with z above the floor, valid mask. pose: the robot's estimate (x, y, theta)."""
        z = self.depth; P = (self.ray * z[..., None]) @ self.R_bc.T + self.t_bc
        x, y, th = pose; c, s = math.cos(th), math.sin(th)
        W = np.stack([x + c * P[..., 0] - s * P[..., 1], y + s * P[..., 0] + c * P[..., 1], P[..., 2]], -1)
        return W, (z > 0.1) & ~self.self_mask


class ArmCam(HeadCam):
    """The camera on the arm (model.ARM_CAMS). Its pose in the chassis frame is not fixed: the robot works it out from the angles its servos report (forward kinematics through the same arm
    model), plus the error of its calibration of them (`arm_cam_cal_deg` per joint, one draw for the run, 0.4 degrees: backlash and zero point of the STS3215s) and the usual mount error.
    It is only worth reading while it looks forward and a little up (not at the floor, as the forearm does while the arm reaches for something) and while the arm is not moving
    (blur and a pose that is not what the servos last said)."""
    arm = True
    ELEV_MIN, ELEV_MAX = -8.0, 62.0      # degrees of the optical axis above the horizontal
    MAX_JOINT_SPEED = 0.30               # rad/s
    MAX_SELF = 0.35                      # more of the picture than this is the robot itself: no use

    def __init__(self, robot, rng):
        super().__init__(robot, rng, 'armcam')
        m = robot.m; self.fk = mujoco.MjData(m); self.q0 = m.qpos0.copy()
        er = np.random.default_rng(int(rng.integers(1 << 30))); self.cal = er.normal(0, math.radians(perturb.get('arm_cam_cal_deg', 0.4)), 5)
        self.mount = (np.eye(3), np.zeros(3)); ce = perturb.get('head_cam_err', None)
        if ce:
            a = er.normal(0, math.radians(ce[1]), 3); q = np.zeros(4); Rx = np.zeros(9); mujoco.mju_axisAngle2Quat(q, a / max(np.linalg.norm(a), 1e-9), float(np.linalg.norm(a))); mujoco.mju_quat2Mat(Rx, q)
            self.mount = (Rx.reshape(3, 3), er.normal(0, ce[0], 3))
        self.elev = 0.0; self._set()

    def _set(self):
        """The camera's pose in the chassis frame (z from the floor) for the joint angles the robot reads now."""
        r = self.r; fd = self.fk; fd.qpos[:] = self.q0; fd.qpos[r.arm_qadr[:5]] = r.joints()[:5] + self.cal
        mujoco.mj_kinematics(r.m, fd); mujoco.mj_camlight(r.m, fd)
        Rb = fd.xmat[r.chassis].reshape(3, 3); pb = fd.xpos[r.chassis]; Rc = fd.cam_xmat[self.cam].reshape(3, 3); pc = fd.cam_xpos[self.cam]
        R = Rb.T @ Rc; t = Rb.T @ (pc - pb); t[2] += pb[2]
        self.R_bc = self.mount[0] @ R; self.t_bc = t + self.mount[1]; self.elev = math.degrees(math.asin(float(np.clip(-self.R_bc[2, 2], -1, 1))))

    def ready(self):
        self._set()
        return self.ELEV_MIN <= self.elev <= self.ELEV_MAX and float(np.abs(self.r.arm_vel[:5]).max()) < self.MAX_JOINT_SPEED

    def capture(self):
        self._set(); return super().capture()


# ---------------------------------------------------------------------------------------------------------------- finding people in depth
CELL = 0.04
UPPER = 0.50             # m: clusters are made of the points above this height only, so a person standing against a wall or a sofa is not joined to it
TOP_BIAS = 0.05          # m: the depth-edge filter eats the top of the head; added back (calibrated once with a person of known height)
MIN_TOP = 0.75           # a person candidate is at least this tall (or, when its top is cut off by the picture, at least this much of it is seen)
MIN_SPREAD = 0.22        # and its points above UPPER reach over at least this much height: a low shelf top is a thin slab, a person is not
MAX_FOOT = 1.0           # m: a cluster wider than this is a wall or a sofa, not a person
GROUP_FOOT = 0.70        # m: wider than this it is more than one person (a child by the hand of a grown-up): the pair is called 'unsure', i.e. treated as a child
CHILD_BELOW, ADULT_FROM = 1.30, 1.50
ADULT_WIDTH = 0.37       # shoulder width that marks a sitting adult
IDENT_N = 12             # feature samples fused per track


def classify(h, w=None, lower_bound=False):
    """Size class from a height (m) and a shoulder width (m, may be None): 'child', 'adult' or 'unsure' (treated as a child by everything that has a safety side)."""
    if lower_bound:                                     # the top was not seen: the person is at least this tall
        return 'adult' if h >= ADULT_FROM else 'unsure'
    if h >= ADULT_FROM: return 'adult'
    if h < CHILD_BELOW: return 'child'
    return 'adult' if (w is not None and w >= ADULT_WIDTH) else 'unsure'


class Track:
    _n = 0
    def __init__(self, t, x, y, h, cut, w, grp=False):
        Track._n += 1; self.grp = [grp]; self.id = Track._n; self.t0 = self.t = t; self.x, self.y = x, y; self.vx = self.vy = 0.0; self.hits = 1; self.misses = 0
        self.hs = [] if cut else [h]; self.hmax_cut = h if cut else 0.0; self.ws = [w] if w else []; self.cut = cut; self.emb = []; self.frontal = []; self.who = None; self.who_conf = 0.0; self.truth = None
        self.cls = 'unsure'; self.coast = False; self.dist = 0.0; self.history = [(t, x, y)]

    def update(self, t, x, y, h, cut, w, grp=False):
        self.grp = (self.grp + [grp])[-12:]; dt = max(t - self.t, 1e-3); vx, vy = (x - self.x) / dt, (y - self.y) / dt; a = 0.5
        self.vx, self.vy = (1 - a) * self.vx + a * vx, (1 - a) * self.vy + a * vy; self.x, self.y = x, y; self.t = t; self.hits += 1; self.misses = 0; self.coast = False
        if cut: self.hmax_cut = max(self.hmax_cut, h)
        else: self.hs = (self.hs + [h])[-12:]
        if w: self.ws = (self.ws + [w])[-12:]
        self.history = (self.history + [(t, x, y)])[-40:]

    @property
    def height(self):
        """(height, is_lower_bound): the median of the frames in which the top was in the picture; else the highest part seen."""
        if len(self.hs) >= 2 or (self.hs and not self.hmax_cut): return float(np.median(self.hs)), False
        if self.hs: return float(self.hs[0]), False
        return self.hmax_cut, True

    @property
    def width(self): return float(np.median(self.ws)) if self.ws else None

    def predict(self, t):
        dt = min(t - self.t, 1.5); return self.x + self.vx * dt, self.y + self.vy * dt


class PersonDetector:
    """Head-camera depth -> tracked people with height and size class. Everything it uses is what the robot has on board: the depth image, its own pose estimate and,
    if it has scanned the room, the 3D map (to tell a tall cupboard it has already seen from a person)."""
    def __init__(self, robot, agent, rng, static_map=None, period=0.2):
        self.r, self.a = robot, agent; self.tracks = []; self.period = period; self._t = -9.0; self.static_map = static_map
        self.cam = HeadCam(robot, rng) if mujoco.mj_name2id(robot.m, mujoco.mjtObj.mjOBJ_CAMERA, 'head') >= 0 else ArmCam(robot, rng)          # the mast camera, or else the one on the arm
        self.frames = 0; self.t_ms = 0.0; self.raw = []; self.rng = rng
        self.blind_total = 0.0; self._tb = None; self.blind_s = 0.0; self.seen_s = 0.0      # seconds the camera was not looking anywhere useful / was

    # ---- one frame
    def process(self):
        t = self.r.t
        if t - self._t < self.period - 1e-6 or self.a.freeze: return False
        self._t = t; dt = 0.0 if self._tb is None else t - self._tb; self._tb = t
        if not self.cam.ready():                                       # looking at the floor, or the arm is moving: nothing learned, and the tracks must not be aged for it
            self.blind_total += dt; self.blind_s += dt
            for k in self.tracks:
                if t - k.t > 0.6: k.coast = True
            return False
        self.seen_s += dt; t0 = time.time(); self.frames += 1
        self.cam.capture()
        if self.cam.self_share > getattr(self.cam, 'MAX_SELF', 1.0): return False
        pose = self.a.pose(); W, ok = self.cam.points(pose)
        det = self._cluster(W, ok, pose)
        self._associate(t, det, pose); self.t_ms += (time.time() - t0) * 1000.0
        return True

    def _cluster(self, W, ok, pose):
        pz = W[..., 2]; sel = ok & (pz > UPPER) & (pz < 2.15)
        vv, uu = np.nonzero(sel)
        if len(vv) < 30: return []
        X, Y, Z = W[vv, uu, 0], W[vv, uu, 1], pz[vv, uu]
        gx, gy = ((X - X.min()) / CELL).astype(int), ((Y - Y.min()) / CELL).astype(int)
        grid = np.zeros((gx.max() + 3, gy.max() + 3), np.int32); np.add.at(grid, (gx + 1, gy + 1), 1)
        lab, n = ndimage.label(ndimage.binary_closing(grid >= 2, iterations=1), structure=np.ones((3, 3)))
        out = []
        for c in range(1, n + 1):
            m = lab[gx + 1, gy + 1] == c
            if m.sum() < 40: continue
            x, y, z, v, u = X[m], Y[m], Z[m], vv[m], uu[m]
            top = float(np.percentile(z, 99)) + TOP_BIAS; foot = max(np.ptp(x), np.ptp(y))
            if top < MIN_TOP or foot > MAX_FOOT or (np.percentile(z, 98) - np.percentile(z, 2)) < MIN_SPREAD: continue
            cut = bool((v[z > top - 0.12] <= 2).any())                                     # the top of the picture clips the head: the height is a lower bound
            cx, cy = float(np.median(x)), float(np.median(y))
            band = (z > 0.74 * top) & (z < 0.88 * top)
            w = None
            if band.sum() >= 12:                                                           # shoulder width: the extent across the line of sight at shoulder height
                dx, dy = cx - pose[0], cy - pose[1]; n_ = math.hypot(dx, dy) + 1e-6; px, py = -dy / n_, dx / n_
                s_ = x[band] * px + y[band] * py; w = float(np.percentile(s_, 97) - np.percentile(s_, 3))
            if self._is_furniture(x, y): continue
            out.append(dict(x=cx, y=cy, h=top, cut=cut, w=w, n=int(m.sum()), grp=bool(foot > GROUP_FOOT)))
        return out

    def _is_furniture(self, x, y):
        """A cluster standing where the scan already found a solid thing low down is furniture, not a person. Skipped when there is no map."""
        sm = self.static_map
        if sm is None: return False
        try: return sm(x, y) > 0.6
        except Exception: return False

    def _associate(self, t, det, pose):
        tr = list(self.tracks); used = set()
        for d in sorted(det, key=lambda d: -d['n']):
            best = None
            for k in tr:
                if k in used: continue
                px, py = k.predict(t); dd = math.hypot(px - d['x'], py - d['y'])
                if dd < 0.7 and (best is None or dd < best[0]): best = (dd, k)
            if best: best[1].update(t, d['x'], d['y'], d['h'], d['cut'], d['w'], d.get('grp', False)); best[1].b0 = self.blind_total; used.add(best[1])
            else: nk = Track(t, d['x'], d['y'], d['h'], d['cut'], d['w'], d.get('grp', False)); nk.b0 = self.blind_total; self.tracks.append(nk)
        for k in self.tracks:
            if k not in used:
                k.misses += 1; k.coast = True
                if t - k.t < 1.5: k.x, k.y = k.predict(t)[0], k.predict(t)[1]
        self.tracks = [k for k in self.tracks if t - k.t - min(self.blind_total - getattr(k, 'b0', self.blind_total), 6.0) < 3.0]      # time the camera was not looking is not held against a track (up to 6 s)
        for k in self.tracks:
            h, lb = k.height; k.cls = 'unsure' if any(k.grp) and sum(k.grp) * 2 >= len(k.grp) else classify(h, k.width, lb); k.dist = math.hypot(k.x - pose[0], k.y - pose[1])

    def confirmed(self, min_hits=2): return [k for k in self.tracks if k.hits >= min_hits]


# ---------------------------------------------------------------------------------------------------------------- movers from the lidar (all around, legs)
class LidarMovers:
    """The head camera looks one way; the lidar looks all round, at 12 cm, where it sees legs and nothing else. A leg cannot be told from a chair leg by shape, but it appears: a point of this
    scan that is clearly in front of what the scan 0.4-1.3 s ago saw in the same direction (both in the world frame, through the robot's pose estimate, seen from this scan's origin, so a
    moving robot does not make furniture 'move') is something that was not there. Wall points at long range and grazing angles, which the lidar's own error moves by several centimetres,
    are not in front of the old wall, so they do not count. Points that stay together (3+ within 0.2 m, not wider than 0.6 m) are grouped, tracked, and given to the guard as 'unsure'
    people (treated as children: nothing is known of their size). A person standing still is no longer 'new' after a second; the head camera has them, and the planner has them as an obstacle."""
    BIN = math.radians(1.5); NB = int(round(2 * math.pi / BIN)); IN_FRONT = 0.10; WINDOW = 1; RANGE = 3.0

    def __init__(self, agent): self.a = agent; self.hist = []; self._last = None; self.tracks = []; self.frames = 0

    def _appeared(self, old_world, origin, new_world):
        ox, oy = origin
        do = old_world - [ox, oy]; ro = np.hypot(do[:, 0], do[:, 1]); bo = ((np.arctan2(do[:, 1], do[:, 0]) % (2 * math.pi)) / self.BIN).astype(int) % self.NB
        minr = np.full(self.NB, np.inf); np.minimum.at(minr, bo, ro)
        w = np.min([np.roll(minr, k) for k in range(-self.WINDOW, self.WINDOW + 1)], axis=0)
        dn = new_world - [ox, oy]; rn = np.hypot(dn[:, 0], dn[:, 1]); bn = ((np.arctan2(dn[:, 1], dn[:, 0]) % (2 * math.pi)) / self.BIN).astype(int) % self.NB
        return np.isfinite(w[bn]) & (rn < w[bn] - self.IN_FRONT)

    def update(self, t):
        sc = self.a.scan
        if sc is None or sc is self._last or self.a.freeze: return
        self._last = sc; lx, ly, ang, rr = sc; x, y, th = self.a.pose(); self.frames += 1
        keep = rr < self.RANGE; pts = np.stack([lx + np.cos(ang[keep]) * rr[keep], ly + np.sin(ang[keep]) * rr[keep]], 1)
        self.hist.append((t, pts)); self.hist = [h for h in self.hist if t - h[0] < 1.4]
        olds = [h for h in self.hist if 0.4 <= t - h[0] <= 1.3]; dets = []
        if olds and len(pts) >= 3:
            old = olds[0][1]
            if len(old) >= 20:
                mv = pts[self._appeared(old, (lx, ly), pts)]
                if len(mv) >= 3:
                    from scipy.spatial import cKDTree
                    tr = cKDTree(mv); lab = -np.ones(len(mv), int); n = 0
                    for i in range(len(mv)):
                        if lab[i] >= 0: continue
                        stack = [i]; lab[i] = n
                        while stack:
                            j = stack.pop()
                            for q in tr.query_ball_point(mv[j], 0.2):
                                if lab[q] < 0: lab[q] = n; stack.append(q)
                        n += 1
                    for c in range(n):
                        g = mv[lab == c]
                        if len(g) >= 3 and max(np.ptp(g[:, 0]), np.ptp(g[:, 1])) < 0.6: dets.append((float(g[:, 0].mean()), float(g[:, 1].mean())))
        used = set()
        for (px, py) in dets:
            best = None
            for k in self.tracks:
                if k in used: continue
                qx, qy = k.predict(t); dd = math.hypot(qx - px, qy - py)
                if dd < 0.6 and (best is None or dd < best[0]): best = (dd, k)
            if best: best[1].update(t, px, py, 1.0, False, None); used.add(best[1])
            else: k = Track(t, px, py, 1.0, False, None); k.src = 'lidar'; self.tracks.append(k)
        for k in self.tracks:
            if k not in used: k.misses += 1; k.coast = True
        self.tracks = [k for k in self.tracks if t - k.t < 1.5]
        for k in self.tracks: k.cls = 'unsure'; k.dist = math.hypot(k.x - x, k.y - y)

    def confirmed(self, min_hits=3): return [k for k in self.tracks if k.hits >= min_hits and not k.coast]


# ---------------------------------------------------------------------------------------------------------------- who it is: opt-in registry (the recogniser itself is a stand-in)
EMB_D = 24


class FaceStandIn:
    """What the simulator can say about faces: nothing. This draws, per person, a fixed feature vector (siblings of one family share a part of it) and, per look, a noisy copy:
    the noise grows with distance and when the person is not facing the camera, as it does for a real face model. The numbers it gives test our registry and thresholds,
    never how well a real recogniser works."""
    def __init__(self, who, family, seed=0, family_share=0.55):
        rng = np.random.default_rng(seed); fam = {f: rng.normal(size=EMB_D) for f in set(family)}; self.v = {}
        for w, f in zip(who, family):
            a = rng.normal(size=EMB_D); v = family_share * fam[f] / np.linalg.norm(fam[f]) + math.sqrt(1 - family_share ** 2) * a / np.linalg.norm(a); self.v[w] = v / np.linalg.norm(v)
        self.rng = np.random.default_rng(seed + 1)

    def look(self, who, dist, facing, speed, elev=0.0):
        """One noisy feature vector of a person `who` seen at `dist` m, `facing` = cosine between where they face and the direction to the camera, `elev` = how many degrees upward the
        camera looks at the face (a face seen steeply from below is harder; the arm camera is lower than a person's face, as the mast camera was). -> (vector, sigma)."""
        sig = 0.30 + 0.28 * dist
        if facing < 0.5: sig *= 2.2
        if speed > 0.8: sig *= 1.3
        if elev > 40.0: sig *= 1.0 + 0.02 * (elev - 40.0)
        e = self.v[who] + self.rng.normal(0, sig / math.sqrt(EMB_D), EMB_D)
        return e / np.linalg.norm(e), sig


class Registry:
    """Who the robot has been told it may recognise. Opt-in: nobody is stored without a grown-up's consent, and a person nobody enrolled stays 'unknown' (the robot may still know
    'a child' from the height). It keeps a feature vector, a name, a role and the consent record; no picture. Everything is in one local file the owner can read, and `forget` removes
    one person for good."""
    TAU, MARGIN, MIN_SAMPLES = 0.80, 0.12, 5

    def __init__(self, path=None):
        self.path = path; self.people = {}; self.log = []
        if path and os.path.exists(path):
            try:
                with open(path) as f: d = json.load(f)
                self.people = d['people']; self.log = load_log(d)                                   # the log of the sessions before: 'forget' scrubs those names too
            except (OSError, ValueError, KeyError, TypeError): pass

    def enrol(self, name, role, samples, consent_by, t=0.0):
        """samples: feature vectors (frontal looks). consent_by: the name of the grown-up who agreed (a child cannot consent for a child)."""
        if not consent_by: raise PermissionError('no consent given')
        if len(samples) < self.MIN_SAMPLES: raise ValueError('too few looks')
        m = np.mean(samples, 0); m = m / np.linalg.norm(m)
        self.people[name] = dict(role=role, emb=m.tolist(), n=len(samples), consent_by=consent_by, t=round(t, 1)); self.log.append((round(t, 1), 'enrol', name)); self.save(); return True

    def forget(self, name, t=0.0):
        ok = self.people.pop(name, None) is not None
        if ok:
            self.log = [e for e in self.log if name not in e[2:]]                                  # the name goes from the log as well as from the records
            for p in self.people.values():
                if p.get('consent_by') == name: p['consent_by'] = '(forgotten)'
            self.log.append((round(t, 1), 'forget', None)); self.save()
        return ok

    def forget_all(self, t=0.0):
        n = len(self.people); self.people = {}; self.log = [e for e in self.log if not any(isinstance(v, str) for v in e[2:])]; self.log.append((round(t, 1), 'forget_all', n)); self.save(); return n

    def identify(self, samples):
        """samples: the looks of one track. -> (name or None, best score, runner-up score). Unknown unless clearly one enrolled person."""
        if not self.people or len(samples) < self.MIN_SAMPLES: return None, 0.0, 0.0
        f = np.mean(samples, 0); f = f / np.linalg.norm(f)
        sc = sorted(((float(f @ np.array(p['emb'])), n) for n, p in self.people.items()), reverse=True)
        s1 = sc[0][0]; s2 = sc[1][0] if len(sc) > 1 else -1.0
        return (sc[0][1] if (s1 >= self.TAU and s1 - s2 >= self.MARGIN) else None), s1, s2

    def role(self, name): return self.people.get(name, {}).get('role')
    def save(self):
        self.log = self.log[-LOG_MAX:]
        if self.path: dump_json(self.path, dict(people=self.people, log=self.log))
    def report(self): return [dict(name=n, role=p['role'], looks=p['n'], consent_by=p['consent_by'], stored='feature vector only') for n, p in self.people.items()]


# ---------------------------------------------------------------------------------------------------------------- the safety layer
class Guard:
    """Between the modes and the wheels and arm. It does not plan; it only limits:
       * a person's surface within `slow` of the robot, in the direction it is going: speed capped to what it can stop from, and to a crawl inside `slow`;
       * inside `halt`: it may not drive at the person (turning on the spot and driving away are fine); a person closing in fast, or at touching distance: the mode is paused where it stands (it
         picks up again when the person has been clear for a moment);
       * a person within arm reach while the arm is at work: the whole job pauses, the arm holds its pose.
       Children get bigger distances. 'unsure' counts as a child. Not applied while righting itself or in a fault, and a human teleoperator is only limited, not stopped."""
    CFG = dict(slow=1.0, halt=0.35, slow_child=1.4, halt_child=0.55, v_crawl=0.10, v_crawl_child=0.06, brake=0.30, arm_reach=0.40, arm_reach_child=0.55, zone=0.9, zone_child=1.1, mem_still=90.0, mem_moving=8.0, ask_after=6.0, release=0.8, close_speed=0.6, robot_r=0.105, person_r=0.14)

    def __init__(self, det, cfg=None, speak=None):
        self.d = det; self.extra = lambda: []; self.c = dict(self.CFG, **(cfg or {})); self.halted = False; self._clear_t = 0.0; self.speak = speak
        self.stats = dict(halt_s=0.0, slow_s=0.0, halts=0, arm_pauses=0, max_scale_cut=0.0); self._last_say = -99.0; self.why = ''; self.scale = 1.0; self._halt_t = 0.0
        self.mem = {}; self.on_ask = None; self.episodes = []; self._zone_ask_t = -99.0

    def _people(self, sup):
        out = []; x, y, th = sup.a.pose()
        cam = self.d.confirmed(2); lid = [k for k in self.extra() if not any(math.hypot(k.x - c.x, k.y - c.y) < 0.6 for c in cam)]
        for k in cam + lid:
            kid = k.cls in ('child', 'unsure'); px, py = k.predict(sup.r.t) if k.coast else (k.x, k.y)
            d = math.hypot(px - x, py - y); gap = d - self.c['robot_r'] - self.c['person_r']
            clos = -((px - x) * k.vx + (py - y) * k.vy) / max(d, 1e-3)                  # closing speed (m/s), > 0: coming at the robot
            out.append(dict(k=k, kid=kid, gap=gap, bearing=math.atan2(py - y, px - x) - th, closing=clos))
        return out

    def pre(self, sup):
        """Before the mode runs this tick: True = pause it (and keep the base still, the arm where it is). Only for what is dangerous standing still is not: a person within the arm's reach
        while the arm is at work, a person coming at the robot fast, a person at touching distance. A person merely near is handled in post(): the robot just may not drive at them."""
        r, c = sup.r, self.c
        if sup.mode in ('RIGHT', 'FAULT', 'TELEOP'): self.halted = False; r.hold_arm = False; sup.a.person_zones = []; sup.a.dyn_zones = []; return False
        ppl = self._people(sup); arm_work = sup.a.state in ('抓取', '放置', '放下') or sup.a.arm_busy or not r.arm_done(0.08)
        t = r.t
        sup.a.dyn_zones = [(p['k'].x, p['k'].y, 0.25 if p['kid'] else 0.32) for p in ppl if not p['k'].coast]       # keep their legs out of the map
        for p in ppl:                                              # where people were, and for how long that is still worth believing (a person standing still is probably still there)
            k = p['k']
            if not k.coast: self.mem[(id(k))] = (k.x, k.y, t, p['kid'], math.hypot(k.vx, k.vy) < 0.25)
        self.mem = {i: m for i, m in self.mem.items() if t - m[2] < (c['mem_still'] if m[4] else c['mem_moving'])}
        sup.a.person_zones = [(m[0], m[1], c['zone_child'] if m[3] else c['zone']) for m in self.mem.values()]       # the job does not start on something someone stands (or stood a moment ago) next to
        trig = False; self.why = ''
        if sup.mode == 'TASK' and r.t - sup.a.zone_skip_t < 1.5 and r.t - self._zone_ask_t > 20.0:       # the only things left are by somebody who stands still: ask
            self._zone_ask_t = r.t; self._say(sup, '请让一让，我想拿您旁边的玩具', every=0.0)
        for p in ppl:
            reach = c['arm_reach_child'] if p['kid'] else c['arm_reach']
            if p['gap'] < 0.2: trig = True; self.why = f'{p["k"].cls} at {p["gap"]:.2f} m'
            if p['gap'] < reach and arm_work: trig = True; self.why = f'arm, {p["k"].cls} {p["gap"]:.2f} m'
            if p['closing'] > c['close_speed'] and p['gap'] < (1.2 if p['kid'] else 0.9): trig = True; self.why = f'{p["k"].cls} coming fast {p["closing"]:.1f} m/s'
        if trig:
            if not self.halted:
                self.stats['halts'] += 1; self.stats['arm_pauses'] += int(arm_work); self._say(sup, f'有人在附近（{self.why}），先停一下'); self.episodes.append([round(r.t, 1), None, self.why])
            self.halted = True; self._clear_t = r.t
        elif self.halted and r.t - self._clear_t > c['release']:
            self.halted = False
            if self.episodes and self.episodes[-1][1] is None: self.episodes[-1][1] = round(r.t, 1)
        r.hold_arm = bool(self.halted and arm_work)
        if self.halted:
            self.stats['halt_s'] += 0.02; r.set_twist(0, 0, 0); self._halt_t += 0.02
            if self._halt_t > c['ask_after']: self._say(sup, '请让一让，我要过去拿东西', every=15.0)
        else: self._halt_t = 0.0
        return self.halted

    def post(self, sup):
        """After the mode has set its velocity: it may not drive at a person closer than `halt` (translation only; turning on the spot is fine), and is slowed down on the way there."""
        r, c = sup.r, self.c; self.scale = 1.0
        if sup.mode in ('RIGHT', 'FAULT') or self.halted: return
        vx, vy, wz = wheels_to_twist(r.wheel_target); v = math.hypot(vx, vy)
        if v < 1e-3: return
        blocked = False
        for p in self._people(sup):
            slow = c['slow_child'] if p['kid'] else c['slow']; halt = c['halt_child'] if p['kid'] else c['halt']
            if p['gap'] > slow: continue
            toward = math.cos(math.atan2(vy, vx) - p['bearing'])
            if toward < 0.2: continue
            if p['gap'] <= halt: self.scale = 0.0; blocked = True; break
            vmax = c['v_crawl_child'] if p['kid'] else c['v_crawl']
            vmax = max(vmax, min(v, math.sqrt(2 * c['brake'] * max(p['gap'] - halt, 0.0))))
            vmax = min(vmax, v) if p['gap'] > halt + 0.25 else min(vmax, (c['v_crawl_child'] if p['kid'] else c['v_crawl']))
            self.scale = min(self.scale, vmax / v)
        if self.scale < 1.0:
            r.wheel_target = twist_to_wheels(vx * self.scale, vy * self.scale, wz)            # the turning part of the command is kept
            self.stats['slow_s'] += 0.02; self.stats['max_scale_cut'] = max(self.stats['max_scale_cut'], 1 - self.scale)
            if blocked: self.stats['blocked_s'] = self.stats.get('blocked_s', 0.0) + 0.02

    def _say(self, sup, text, every=10.0):
        if sup.r.t - self._last_say < every: return
        self._last_say = sup.r.t; sup._event(text)
        if self.speak: self.speak(text)
        if self.on_ask is not None and '请让一让' in text: self.on_ask((sup.a.pose()[0], sup.a.pose()[1]))



class PeopleSystem:
    """Everything about people on one object, hooked into the Supervisor (modes.py) by `sup.attach_people(system)`: it moves the simulated people, reads the head camera,
    tracks, sizes and (for enrolled people) names them, and runs the safety guard."""
    def __init__(self, robot, agent, names, scripts, who=None, family=None, seed=0, registry_path=None, guard=True, static_map=None, period=0.2, guard_cfg=None, comply=None, strict_face=None):
        self.r, self.a = robot, agent; self.names = list(names); self.who = list(who) if who else [f'p{i}' for i in range(len(names))]
        self.people = People(robot, names, scripts, comply=comply, seed=seed); self.det = PersonDetector(robot, agent, np.random.default_rng(seed + 11), static_map, period)
        self.face = FaceStandIn(self.who, family or [w for w in self.who], seed + 5); self.registry = Registry(registry_path)
        self.movers = LidarMovers(agent); self.guard = Guard(self.det, guard_cfg) if guard else None
        if self.guard: self.guard.extra = self.movers.confirmed; self.guard.on_ask = self.people.ask_to_move
        self.enabled = True; self.t_identify = 0.0; self.ident_log = []
        self.strict_face = self.det.cam.arm if strict_face is None else bool(strict_face)      # a face is only looked at if the head is in the picture, and a steep look up costs (always so for the arm camera)
        self.n_face_out = 0
        self.n_look = 0; self.n_false = 0; self.cls_votes = [dict() for _ in names]; self.id_votes = [dict() for _ in names]

    # --- called by the supervisor each tick
    def before_physics(self): self.people.update()

    def after_physics(self):
        self.people.after_step(); self.movers.update(self.r.t)
        if self.det.process(): self._look()

    def _look(self):
        """The stand-in recogniser: every track that was updated this frame gets one noisy feature vector of whoever really is there (the simulator's only access to the truth here),
        then the registry is asked who it is. A real model would take a face or body crop instead; nothing of the picture would be kept either way."""
        truth = self.people.truth(); rx, ry, th = self.a.pose()
        for k in self.det.tracks:
            if k.coast or k.hits < 2: continue
            best = None
            for s in truth:
                dd = math.hypot(s['x'] - k.x, s['y'] - k.y)
                if dd < 0.6 and (best is None or dd < best[0]): best = (dd, s)
            self.n_look += 1
            if best is None: k.truth = None; self.n_false += 1; continue
            s = best[1]; k.truth = s['i']; d = math.hypot(s['x'] - rx, s['y'] - ry); elev = 0.0
            facing = math.cos(s['yaw']) * (rx - s['x']) / max(d, 1e-6) + math.sin(s['yaw']) * (ry - s['y']) / max(d, 1e-6)
            if self.strict_face:
                fh = 0.93 * s['h']; cam = self.det.cam; elev = math.degrees(math.atan2(fh - cam.height, max(d, 0.05)))
                if not self._in_picture(s['x'] - rx, s['y'] - ry, th, fh): self.n_face_out += 1; continue
            e, sig = self.face.look(self.who[s['i']], d, facing, s['v'], elev); k.emb = (k.emb + [e])[-IDENT_N:]; k.frontal = (k.frontal + [facing >= 0.5])[-IDENT_N:]
            who, s1, s2 = self.registry.identify(k.emb); k.who, k.who_conf = who, s1
            if k.hits >= 4:
                v = self.cls_votes[s['i']]; v[k.cls] = v.get(k.cls, 0) + 1; v = self.id_votes[s['i']]; key = who or 'unknown'; v[key] = v.get(key, 0) + 1

    def _in_picture(self, dx, dy, th, fh):
        """Is a face at (dx, dy) from the robot and fh m above the floor inside the camera's picture (4 degrees from the edge)? Uses the camera pose the robot believes."""
        cam = self.det.cam; c, sn = math.cos(th), math.sin(th)
        p = cam.R_bc.T @ (np.array([c * dx + sn * dy, -sn * dx + c * dy, fh]) - cam.t_bc)
        if p[2] >= -0.05: return False
        fovy = float(self.r.m.cam_fovy[cam.cam]); fovx = math.degrees(2 * math.atan(math.tan(math.radians(fovy) / 2) * HW / HH))
        return abs(math.degrees(math.atan2(p[0], -p[2]))) <= fovx / 2 - 4.0 and abs(math.degrees(math.atan2(p[1], -p[2]))) <= fovy / 2 - 4.0

    # --- what the rest of the robot asks
    def tracks(self, min_hits=2): return self.det.confirmed(min_hits)

    def nearest(self, kind=None, unidentified=False, truth=None):
        """The nearest tracked person. kind='child': skip adults. unidentified: skip people the registry already names. truth: the track that belongs to this simulated person
        (stand-in for 'the person the voice came from')."""
        x, y, th = self.a.pose(); best = None
        for k in self.tracks():
            if truth is not None and k.truth != truth: continue
            if kind == 'child' and k.cls == 'adult': continue
            if unidentified and k.who: continue
            d = math.hypot(k.x - x, k.y - y)
            if best is None or d < best[0]: best = (d, k)
        return best[1] if best else None

    def enrol(self, track, name, role, consent_by):
        """Opt-in: store this person's feature vector under a name. Needs consent from a named grown-up and some frontal looks."""
        fr = [e for e, f in zip(track.emb, track.frontal) if f]
        if len(fr) < Registry.MIN_SAMPLES: raise ValueError('need more frontal looks')
        return self.registry.enrol(name, role, fr, consent_by, self.r.t)

    def describe(self):
        """For the voice: who is around, in words."""
        x, y, th = self.a.pose(); out = []
        for k in sorted(self.tracks(), key=lambda k: k.dist):
            h, lb = k.height; side = '左' if math.sin(math.atan2(k.y - y, k.x - x) - th) > 0.25 else ('右' if math.sin(math.atan2(k.y - y, k.x - x) - th) < -0.25 else '正')
            who = k.who or ('一个小孩' if k.cls == 'child' else ('一个大人' if k.cls == 'adult' else '一个人（不确定大人还是小孩）'))
            out.append(f'{who}，约 {h:.1f} 米高{"以上" if lb else ""}，在{side}前方 {k.dist:.1f} 米')
        return out


from robot import twist_to_wheels, wheels_to_twist     # noqa: E402  (kept at the bottom: robot.py imports model.py, which imports this module lazily)
