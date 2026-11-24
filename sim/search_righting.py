"""Search for arm-and-wheel maneuvers that put the fallen robot back on its wheels (how sim/righting_maneuvers.json was made).

  python search_righting.py refine CLASS INIT.json|- GENS POP PROCS N STD TAG    cross-entropy search; '-' starts from scratch, N perturbed scenarios per candidate
  python search_righting.py test CLASS CAND.json N                              success rate of a stored candidate on N fresh perturbed scenarios
  python search_righting.py init CLASS OUT.json [LIBRARY.json]                  the stored maneuver of CLASS as a starting point for refine (default library: righting_maneuvers.json)

CLASS: side_l, side_r (on its side), fwd (leaning on the front wheels and the open gripper), back. The robot is dropped into that pose with the arm in the carry
pose and settled; a candidate is 4 phases of (4 arm joint targets, 3 wheel speeds, duration, arm speed) followed by letting go and settling; the cost is the
tilt angle left (plus half the smallest tilt reached on the way, to give the search something to climb on). Every candidate is judged on N scenarios that differ in
arm calibration (1 deg), roller friction (+-20 %), chassis mass (+-8 %), and the exact resting pose (3 deg, any heading), the same N for every candidate.
The arm may use 30 rad/s^2 here (8 for normal driving). Candidates whose arm would pass through the chassis are penalised (the model has no arm-body collisions).
Results: results/righting_search.json. ARM_CAM=forearm_up (any key of model.ARM_CAMS) in the environment puts the arm camera, and its weight, on the robot being searched with; ARM_CAM_COLLIDE=1 also makes its box touch the floor,
and the fitness then has up to 30 points for pressing the camera against the ground (peak force and time).
"""
import json, math, os, sys, time
import numpy as np
import mujoco
from robot import Robot, Q_CARRY, GRIP_CLOSED

CLASSES = dict(side_l=((1, 0, 0), 90), side_r=((1, 0, 0), -90), fwd=((0, 1, 0), 60), back=((0, 1, 0), -90))
NPH = 4; PER = 9
SD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', '')


def quat_axis(axis, ang):
    s = math.sin(ang / 2); return np.array([math.cos(ang / 2), axis[0] * s, axis[1] * s, axis[2] * s])


class Env:
    def __init__(self, cls, arm_acc=30.0):
        self.cls = cls; self.arm_acc = arm_acc
        kw = {'arm_cam': os.environ['ARM_CAM'], 'arm_cam_collide': bool(os.environ.get('ARM_CAM_COLLIDE'))} if os.environ.get('ARM_CAM') else {}      # ARM_CAM=forearm_up: search with the camera's weight on the arm; ARM_CAM_COLLIDE=1: and its box touching the floor (round 16)
        self.r = Robot([], seed=0, robot_pose=(2.0, 1.5, 0.0), **kw)
        r = self.r; m = r.m; self.cam_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'armcam_col'); self._f6 = np.zeros(6)
        self.body_geoms = [g for g in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ('chassis_col', 'lidar_head', 'lidar_post')) if g >= 0] if self.cam_geom >= 0 else []       # what the camera must not run into
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex')]
        self.lo = np.array([m.jnt_range[j][0] for j in jid]) + 0.03; self.hi = np.array([m.jnt_range[j][1] for j in jid]) - 0.03
        self.roller_geoms = [g for g in range(m.ngeom) if m.geom_group[g] == 1 and m.geom_bodyid[g] != 0]
        self.fric0 = m.geom_friction[self.roller_geoms, 0].copy()
        self.mass0 = float(m.body_mass[r.chassis]); self.inert0 = m.body_inertia[r.chassis].copy()
        self.d2 = mujoco.MjData(m); self.d2.qpos[3:7] = [1, 0, 0, 0]; self.d2.qpos[2] = 0.0513
        self.link_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ('shoulder', 'upper_arm', 'lower_arm', 'wrist', 'gripper')]
        self.pinch = r.pinch_site
        self.drop(np.random.default_rng(0), jitter=0.0)

    def drop(self, rng, jitter=0.0, arm=None):
        """Drop the robot from about the nominal orientation and let it settle: the new rest state is the starting point of runs."""
        r = self.r; m = r.m
        axis, ang = CLASSES[self.cls]
        q = quat_axis(axis, math.radians(ang))
        if jitter:
            j = rng.normal(0, math.radians(jitter), 3); qq = np.zeros(4); mujoco.mju_axisAngle2Quat(qq, j / (np.linalg.norm(j) + 1e-12), np.linalg.norm(j)); out = np.zeros(4); mujoco.mju_mulQuat(out, qq, q); q = out
        yaw = rng.uniform(-math.pi, math.pi); qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]); out = np.zeros(4); mujoco.mju_mulQuat(out, qz, q); q = out
        r.d.qpos[:7] = [2.0, 1.5, 0.3, *q]; r.d.qvel[:] = 0
        arm0 = np.concatenate([Q_CARRY, [0.3]]) if arm is None else np.asarray(arm)
        r.arm_cmd[:] = arm0; r.arm_target[:] = arm0; r.arm_vel[:] = 0; r.wheel_cmd[:] = 0; r.wheel_target[:] = 0
        r.d.qpos[r.arm_qadr] = arm0 + r.arm_offset; r.d.ctrl[3:9] = arm0 + r.arm_offset
        mujoco.mj_forward(m, r.d)
        for _ in range(220): r.step()
        self.qpos0 = r.d.qpos.copy(); self.tilt0 = math.degrees(r.tilt()); self.arm0 = r.joints().copy()

    def perturb(self, rng, arm_deg=1.0, fric=0.2, mass=0.08):
        r = self.r; m = r.m
        r.arm_offset[:] = np.append(rng.normal(0, math.radians(arm_deg), 5), 0.0)
        m.geom_friction[self.roller_geoms, 0] = self.fric0 * rng.uniform(1 - fric, 1 + fric)
        f = rng.uniform(1 - mass, 1 + mass); m.body_mass[r.chassis] = self.mass0 * f; m.body_inertia[r.chassis] = self.inert0 * f

    def unperturb(self):
        r = self.r; m = r.m
        r.arm_offset[:] = 0; m.geom_friction[self.roller_geoms, 0] = self.fric0; m.body_mass[r.chassis] = self.mass0; m.body_inertia[r.chassis] = self.inert0

    def arm_hits_chassis(self, q4):
        d2, m = self.d2, self.r.m
        d2.qpos[self.r.arm_qadr[:5]] = np.append(q4, 0.0); mujoco.mj_kinematics(m, d2)
        pts = [d2.xpos[i] - d2.xpos[self.r.chassis] for i in self.link_ids] + [d2.site_xpos[self.pinch] - d2.xpos[self.r.chassis]]
        samples = []
        for a, b in zip(pts[1:-1], pts[2:]): samples += [a + (b - a) * t for t in (0, 0.25, 0.5, 0.75, 1.0)]
        for p in samples:
            if math.hypot(p[0], p[1]) < 0.115 and 0.02 < p[2] < 0.09: return True
        return False

    def decode(self, x):
        out = []
        for k in range(NPH):
            v = x[k * PER:(k + 1) * PER]
            out.append(dict(q=self.lo + v[0:4] * (self.hi - self.lo), w=(v[4:7] * 2 - 1) * 3.6, T=0.2 + v[7] * 1.8, speed=0.6 + v[8] * 4.1))
        return out

    def encode(self, phases):
        """The search vector (36 numbers in 0..1) of stored phases (the inverse of decode): to start a search from a maneuver that is already known."""
        x = []
        for p in phases:
            q = (np.asarray(p['q'], float) - self.lo) / (self.hi - self.lo); w = (np.asarray(p['w'], float) / 3.6 + 1) / 2
            x += list(np.clip(np.concatenate([q, w, [(p['T'] - 0.2) / 1.8, (p['speed'] - 0.6) / 4.1]]), 0, 1))
        return np.array(x)

    def run(self, x, record=False, settle=2.5):
        return self.run_phases(self.decode(x), record, settle)

    def run_phases(self, ph, record=False, settle=2.5):
        r = self.r
        r.d.qpos[:] = self.qpos0; r.d.qvel[:] = 0
        r.arm_cmd[:] = self.arm0 + 0; r.arm_target[:] = r.arm_cmd; r.arm_vel[:] = 0; r.wheel_cmd[:] = 0; r.wheel_target[:] = 0
        r.d.qpos[r.arm_qadr] = r.arm_cmd + r.arm_offset; r.d.ctrl[3:9] = r.arm_cmd + r.arm_offset; r.arm_acc = self.arm_acc
        mujoco.mj_forward(r.m, r.d)
        pen = 0.0; best = 999.0; track = []; cam_peak = 0.0; cam_steps = 0; cam_clear = 0.05
        for p in ph:
            if self.arm_hits_chassis(p['q']): pen += 40.0
            r.arm_target[:4] = p['q']; r.arm_target[4] = 0.0; r.arm_target[5] = GRIP_CLOSED; r.arm_speed = p['speed']; r.wheel_target[:] = p['w']
            for _ in range(int(p['T'] / 0.02)):
                r.step(); t = math.degrees(r.tilt()); best = min(best, t)
                if record: track.append(t)
                if self.cam_geom >= 0:                                                                       # is the camera pressed against the floor?
                    d = r.d; hit = False
                    for i in range(d.ncon):
                        c = d.contact[i]
                        if c.geom1 == self.cam_geom or c.geom2 == self.cam_geom: mujoco.mj_contactForce(r.m, d, i, self._f6); cam_peak = max(cam_peak, float(self._f6[0])); hit = True
                    cam_steps += hit
                    for g in self.body_geoms: cam_clear = min(cam_clear, float(mujoco.mj_geomDistance(r.m, d, self.cam_geom, g, 0.05, self._f6)))      # signed distance to the chassis and the lidar (capped at 5 cm); not part of the fitness
        r.wheel_target[:] = 0; r.arm_target[:5] = Q_CARRY; r.arm_speed = 2.0
        for _ in range(int(settle / 0.02)): r.step()
        final = math.degrees(r.tilt()); r.arm_acc = 8.0
        cam_pen = 15.0 * min(cam_peak / 30.0, 1.0) + 15.0 * min(cam_steps * 0.02, 1.0)                          # the camera is not to be crushed: up to 30 points for a hard, long press
        return dict(final=final, best=best, pen=pen, fit=final + 0.5 * min(best, 90.0) + pen + cam_pen, upright=bool(final < 20 and abs(r.d.qpos[2] - 0.0513) < 0.02), track=track, cam_force=cam_peak, cam_s=cam_steps * 0.02, cam_clear=cam_clear)

    def robust(self, x, n=6, seed=0, **kw):
        """Mean fitness over n perturbed rest states / models (the same n each call: common random numbers)."""
        rng = np.random.default_rng(seed); fits = []; ups = 0
        for k in range(n):
            self.unperturb(); self.drop(rng, jitter=3.0); self.perturb(rng, **kw)
            o = self.run(x); fits.append(o['fit']); ups += o['upright']
        self.unperturb()
        return float(np.mean(fits)), ups


_ENV = None
def _init(cls, n):
    global _ENV, _N; _ENV = Env(cls); _N = n
def _eval(x):
    f, ups = _ENV.robust(x, n=_N); return f, ups


def cem(cls, x0, gens, pop, nproc, n, std0, tag, seed=11):
    import multiprocessing as mp
    D = NPH * PER; rng = np.random.default_rng(seed)
    mean = np.array(x0, float) if x0 is not None else np.full(D, 0.5); std = np.full(D, std0); elite = max(6, pop // 6)
    best = (1e9, None, 0); t0 = time.time()
    with mp.Pool(nproc, initializer=_init, initargs=(cls, n)) as pool:
        for g in range(gens):
            X = np.clip(rng.normal(mean, std, (pop, D)), 0, 1)
            if x0 is not None and g == 0: X[0] = x0
            R = pool.map(_eval, list(X)); F = np.array([a for a, _ in R]); U = np.array([b for _, b in R])
            idx = np.argsort(F)[:elite]
            mean = 0.5 * mean + 0.5 * X[idx].mean(0); std = np.maximum(0.8 * std + 0.2 * X[idx].std(0), 0.02)
            if F[idx[0]] < best[0]: best = (F[idx[0]], X[idx[0]].copy(), int(U[idx[0]]))
            print(f'[{tag}] gen {g:2d} best {F[idx[0]]:6.1f} ({U[idx[0]]}/{n} upright) elite mean {F[idx].mean():6.1f} overall {best[0]:6.1f} ({best[2]}/{n}) {time.time() - t0:.0f}s', flush=True)
            json.dump(dict(cls=cls, x=best[1].tolist(), fit=float(best[0]), upright=best[2], n=n), open(SD + f'righting_{tag}.json', 'w'))
            if best[2] == n and best[0] < float(os.environ.get('SEARCH_STOP_FIT', 25)) and g >= 6: break          # SEARCH_STOP_FIT: stop when all upright and the fitness is below this (default 25)
    return best


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'refine':      # refine CLASS INITJSON GENS POP NPROC N STD TAG
        cls = sys.argv[2]; x0 = json.load(open(sys.argv[3]))['x'] if sys.argv[3] != '-' else None
        cem(cls, x0, int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]), sys.argv[9])
    elif mode == 'init':      # init CLASS OUT.json [LIBRARY.json]: the stored maneuver of this class as a starting point for refine
        import righting
        cls = sys.argv[2]; lib = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else righting.LIB; env = Env(cls)
        json.dump(dict(cls=cls, x=env.encode(lib['maneuvers'][cls][0]['phases']).tolist()), open(sys.argv[3], 'w'))
    elif mode == 'test':      # test CLASS JSON N
        cls = sys.argv[2]; x = np.array(json.load(open(sys.argv[3]))['x']); n = int(sys.argv[4])
        env = Env(cls); f, ups = env.robust(x, n=n, seed=99)
        print(f'{cls} {sys.argv[3]}: mean final tilt {f:.1f}, upright {ups}/{n}')
