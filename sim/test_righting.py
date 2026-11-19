"""Self-righting bench (what the supervisor does after a fall, without the rest of the robot).

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
from robot import GRIP_CLOSED, Q_CARRY, Robot
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


def sweep():
    rows = []
    for kind in ('roll_l', 'roll_r', 'pitch_f', 'pitch_b'):
        for strength in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4):
            r = Robot([], seed=1, robot_pose=(2.0, 1.5, 0.3))
            for _ in range(50): r.step()
            knock(r, kind, strength)
            for _ in range(int(3.0 / 0.02)): r.step()
            cls = righting.classify(r.imu()); seq = [cls]
            for attempt in range(3):
                if cls == 'upright': break
                if cls == 'unknown': cls = righting.ranked(r.imu())[0][1]
                run_man(r, righting.LIB['maneuvers'][cls][0]['phases']); cls = righting.classify(r.imu()); seq.append(cls)
            rows.append(dict(kind=kind, strength=strength, poses=seq, up=seq[-1] == 'upright', fell=seq[0] != 'upright'))
    fell = [x for x in rows if x['fell']]
    return dict(trials=rows, fell=len(fell), got_up=sum(x['up'] for x in fell), n=len(rows))


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    out = dict(per_pose=per_pose(n)); out['sweep'] = sweep()
    print('knock-overs: %d of %d left it down, %d of those got up' % (out['sweep']['fell'], out['sweep']['n'], out['sweep']['got_up']))
    json.dump(out, open(os.path.join(RES, 'righting.json'), 'w'), ensure_ascii=False, indent=1)
