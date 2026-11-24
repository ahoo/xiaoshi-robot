"""Self-righting bench (what the supervisor does after a fall, without the rest of the robot).

  python test_righting.py unit   the wait for a calm robot before the arm moves (no physics, a second): a robot that settles a few degrees from where it was read still gets up; one that is carried does not get an arm swung
  python test_righting.py [n]    per rest pose: the stored maneuver on n perturbed scenarios (arm calibration 1 deg, roller friction +-20 %, chassis mass +-8 %,
                                 resting pose +-3 deg and any heading), harsher perturbations too; then knock-overs: 4 directions x 7 strengths of a torque pulse about
                                 the robot's roll or pitch axis, and the maneuvers applied the way righting.recover does (classify by IMU, run, look again, up to 3 times)
                                 -> results/righting.json
Open floor, no furniture: the room's furniture is what the in-supervisor runs (test_modes.py fall) add.
"""
import json
import math
import os
import sys

import numpy as np

import righting
from robot import CTRL_DT, GRIP_CLOSED, Q_CARRY, Robot
from search_righting import Env

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def per_pose(n=24):
    out = {}
    for cls in ('side_l', 'side_r', 'fwd'):
        env = Env(cls); ph = righting.LIB['maneuvers'][cls][0]['phases']
        row = {}
        for name, kw in (('standard', dict()), ('harsh', dict(arm_deg=2.0, fric=0.3, mass=0.15))):
            rng = np.random.default_rng(99 if name == 'standard' else 123); ups = 0; fin = []
            for k in range(n):
                env.unperturb(); env.drop(rng, jitter=3.0); env.perturb(rng, **kw)
                o = env.run_phases([dict(q=np.array(p['q']), w=np.array(p['w']), T=p['T'], speed=p['speed']) for p in ph]); ups += o['upright']; fin.append(o['final'])
            row[name] = dict(upright=int(ups), n=n, mean_final_tilt=round(float(np.mean(fin)), 1))
        env.unperturb(); out[cls] = row; print(cls, row)
    return out


def knock(r, kind, strength, dur=0.25):
    R = r.d.xmat[r.chassis].reshape(3, 3).copy()
    axis = dict(roll_l=(1, 0, 0), roll_r=(-1, 0, 0), pitch_f=(0, 1, 0), pitch_b=(0, -1, 0))[kind]
    tw = R @ (np.array(axis, float) * 3.2 * strength)
    for k in range(int(dur / 0.02)): r.d.xfrc_applied[r.chassis, 3:6] = tw; r.step()
    r.d.xfrc_applied[r.chassis, 3:6] = 0.0


def run_man(r, phases):
    r.arm_acc = righting.ARM_ACC_RIGHT
    for p in phases:
        r.arm_target[:4] = p['q']; r.arm_target[4] = 0.0; r.arm_target[5] = GRIP_CLOSED; r.arm_speed = p['speed']; r.wheel_target[:] = p['w']
        for _ in range(int(p['T'] / 0.02)): r.step()
    r.wheel_target[:] = 0; r.arm_target[:5] = Q_CARRY; r.arm_speed = 2.0
    for _ in range(int(2.5 / 0.02)): r.step()
    r.arm_acc = 8.0


def sweep(cam=None):
    """28 knock-overs (4 directions x 7 strengths), each from a fresh robot, then up to 3 tries the way righting.recover does them (classify, run the maneuver of that class, look again; the first time a class
    comes up its best maneuver, again the best one the second time, the next one the third time). cam: a key of model.ARM_CAMS: the robot has that camera (its weight, and its box touching the floor) and the maneuvers that suit it."""
    rows = []; kw = dict(arm_cam=cam, arm_cam_collide=True) if cam else {}
    for kind in ('roll_l', 'roll_r', 'pitch_f', 'pitch_b'):
        for strength in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4):
            r = Robot([], seed=1, robot_pose=(2.0, 1.5, 0.3), **kw); lib = righting.library_for(r)
            for _ in range(50): r.step()
            knock(r, kind, strength)
            for _ in range(int(3.0 / 0.02)): r.step()
            cls = righting.classify(r.imu(), lib); seq = [cls]; used = {}
            for attempt in range(3):
                if cls == 'upright': break
                if cls == 'unknown': cls = righting.ranked(r.imu(), lib)[0][1]
                mans = lib['maneuvers'][cls]; n_ = used.get(cls, 0); used[cls] = n_ + 1; order = [mans[0]] + list(mans)
                run_man(r, order[n_ % len(order)]['phases']); cls = righting.classify(r.imu(), lib); seq.append(cls)
            rows.append(dict(kind=kind, strength=strength, poses=seq, up=seq[-1] == 'upright', fell=seq[0] != 'upright'))
    fell = [x for x in rows if x['fell']]
    return dict(trials=rows, fell=len(fell), got_up=sum(x['up'] for x in fell), n=len(rows), camera=cam)


class _Stub:
    """Just what righting.recover reads and writes, with the gravity direction a function of the time: no physics."""
    def __init__(self, up_of_t):
        self.t = 0.0; self.up_of_t = up_of_t; self.wheel_target = np.zeros(3); self.arm_target = np.zeros(6); self.arm_speed = 1.0; self.arm_acc = 8.0; self.arm_moved_at = None
        self.events = []; self.stats = dict(righted=0, right_failed=0); self.fault = None; self.r = self
        self.a = type('A', (), dict(stop=lambda s: None, state=''))()
    def imu(self): return np.asarray(self.up_of_t(self.t), float)
    def _event(self, text): self.events.append((round(self.t, 1), text))
    def _fault(self, why): self.fault = why
    def run(self, t_max):
        g = righting.recover(self)
        while self.t < t_max:
            try: next(g)
            except StopIteration: break
            if self.arm_moved_at is None and np.any(self.arm_target[:4] != 0.0): self.arm_moved_at = round(self.t, 1)
            self.t += CTRL_DT
        return self


def unit():
    """righting.recover's wait before the first movement (round 16): the old one compared the IMU with the reading taken when the robot first lay still, so a robot that then settled 4 degrees away
    (a fall in a room: 94 -> 98 degrees) never counted as calm and stayed in the RIGHT mode for the rest of the run. No physics: a stand-in robot whose gravity direction is a function of the time."""
    lib = righting.LIB; v = np.asarray(lib['rest_poses']['side_l'], float); v /= np.linalg.norm(v); t0 = math.acos(v[2]); h = v[:2] / math.hypot(*v[:2])
    vec = lambda th: np.array([math.sin(th) * h[0], math.sin(th) * h[1], math.cos(th)])
    saved = righting.library_for, righting.relocalize
    righting.library_for = lambda r: righting.LIB; righting.relocalize = lambda sup: iter(())
    out = []; ok = lambda name, c: (out.append(bool(c)), print(('ok   ' if c else 'FAIL ') + name))
    try:
        # A: lies still, then settles 5.5 degrees further over 3.6 s (the reading is taken during the settling), then stays there
        creep = lambda t: vec(t0 + math.radians(5.5) * min(t / 3.6, 1.0))
        a = _Stub(creep).run(200.0)
        ok('settles 5 degrees from where it was read: the arm still moves, after the 30 s of waiting and 3 calm seconds', a.arm_moved_at is not None and 30.0 < a.arm_moved_at < 45.0)
        ok('  (and it announced the movement once for that try)', sum('秒后开始起身' in e[1] for e in a.events[:2]) == 1)
        # B: lies still for 2 s and is then carried about for good (swings 10 degrees every 4 s)
        carried = lambda t: vec(t0 + (0.0 if t < 2.0 else math.radians(10.0) * math.sin(2 * math.pi * (t - 2.0) / 4.0)))
        b = _Stub(carried).run(300.0)
        ok('carried about for good: the arm never moves', b.arm_moved_at is None)
        ok('  and after 90 s of waiting it stops and asks for help (fault), it does not wait for ever', b.fault is not None and b.t < 100.0 and b.stats['right_failed'] == 1)
        # C: lies still, a hand puts it back on its wheels during the wait
        hand = lambda t: vec(t0) if t < 3.0 else np.array([0.0, 0.0, 1.0])                  # the wait is from 1.5 s (still for 1.5 s) to 4.5 s
        c = _Stub(hand).run(100.0)
        ok('put on its wheels by a hand during the wait: the arm does not move, counted as up, a second later', c.arm_moved_at is None and c.stats['righted'] == 1 and c.t < 5.0)
        # D: an ordinary fall (lies still, nothing moves it): the arm moves after the announced 3 s, as before
        d = _Stub(lambda t: vec(t0)).run(60.0)
        ok('lying still: the arm moves about 3 s after the announcement, no extra wait', d.arm_moved_at is not None and 4.0 < d.arm_moved_at < 6.5)
    finally:
        righting.library_for, righting.relocalize = saved
    print(f'{sum(out)}/{len(out)} passed'); return all(out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'unit': sys.exit(0 if unit() else 1)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    cam = os.environ.get('SWEEP_CAM')                                    # SWEEP_CAM=forearm_up: only the knock-over sweep, with that camera -> results/righting_sweep_<cam>.json
    if cam:
        out = dict(sweep=sweep(cam)); print('with the %s camera: knock-overs: %d of %d left it down, %d of those got up' % (cam, out['sweep']['fell'], out['sweep']['n'], out['sweep']['got_up']))
        json.dump(out, open(os.path.join(RES, f'righting_sweep_{cam}.json'), 'w'), ensure_ascii=False, indent=1); sys.exit(0)
    out = dict(per_pose=per_pose(n)); out['sweep'] = sweep()
    print('knock-overs: %d of %d left it down, %d of those got up' % (out['sweep']['fell'], out['sweep']['n'], out['sweep']['got_up']))
    json.dump(out, open(os.path.join(RES, 'righting.json'), 'w'), ensure_ascii=False, indent=1)
