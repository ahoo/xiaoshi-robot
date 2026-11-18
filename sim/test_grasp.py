"""Bench test of grasping unknown objects: robot parked, one object on the floor in front of it, pick it up and hold
it for 2 s. Every trial draws a new random object of the given kind (objects.py).

  python test_grasp.py [trials_per_kind]      ->  results/grasp_bench.json

Everything the planner knows about the object comes from the front RGB-D camera (perception.py).
Run with grip tape on the jaws (pad friction 1.0) and bare PLA jaws (0.35). Soft objects use fabric friction either way.
"""
import json
import math
import sys

import numpy as np

import model
import objects as O
from perception import FrontCamera, WristCamera
from robot import Robot
from skills import carry_pose, grasp, holding, plan_grasp, wait

ALIGN_NOISE = 0.15     # rad, 1 sigma: how far a long object's long axis is from square to the jaw line after the base has lined up
LONG = ('car', 'banana', 'sock', 'bottle')   # kinds whose long axis is the body x axis; the base lines up with these


def one(kind, seed, pad_friction, mat=None):
    rng = np.random.default_rng(seed)
    model.P['pad_friction'] = pad_friction
    spec = O.sample(kind, rng)
    if mat: spec['mat'] = mat
    rpose = (2.2, 1.05, rng.uniform(-3.1, 3.1))          # open floor, at least 0.7 m from any furniture
    # Where the base ends up relative to the object: 0.235 m +- servo and localisation error, a little off to the side.
    # After the object has settled, the robot takes the same close look as the controller and applies the same test:
    # grasp only if the object is within 6 cm of the centre line and the planner finds a finger-wide strip of it that
    # fits the gripper. Otherwise the controller would move to a new stance; here the stance is redrawn.
    cam = None
    for tries in range(8):
        r_obj = rng.uniform(0.205, 0.265); bearing = rng.uniform(-0.3, 0.3)
        rel0 = np.array([r_obj * math.cos(bearing), r_obj * math.sin(bearing)])
        rad = math.atan2(rel0[1], rel0[0] - 0.069)
        yaw_rel = rad + math.pi / 2 + rng.normal(0, ALIGN_NOISE) if kind in LONG else rng.uniform(-math.pi, math.pi)
        c, s_ = math.cos(rpose[2]), math.sin(rpose[2])
        robot = Robot([(kind, rpose[0] + c * rel0[0] - s_ * rel0[1], rpose[1] + s_ * rel0[0] + c * rel0[1], rpose[2] + yaw_rel, spec)],
                      seed=seed, robot_pose=rpose)
        for _ in range(30): robot.step()                 # 0.6 s: let the object and the robot settle
        if cam is not None: cam.close()
        cam = FrontCamera(robot, np.random.default_rng(seed * 10 + tries))
        dets = [d for d in cam.detect() if d['n_px'] >= 60 and d['top'] < 0.10]
        plan = None
        if dets:
            det = min(dets, key=lambda d: abs(d['rel'][1]))
            plan = plan_grasp(det)
            if abs(det['rel'][1]) <= 0.06 and plan is not None:
                break
            plan = None
    logs = [] if plan is not None else ['8 个站位都不满足下手条件']
    wrist = WristCamera(robot)

    def script():
        if plan is None: return False
        ok = yield from grasp(robot, plan, logs.append, look=wrist.holding)
        if not ok: return False
        yield from carry_pose(robot)
        yield from wait(robot, 2.0)
        return holding(robot, plan)

    g = script(); claimed = None
    try:
        while True:
            next(g); robot.step()
    except StopIteration as e:
        claimed = bool(e.value)
    held = robot.obj_pos(0)[2] > 0.08 and np.linalg.norm(robot.obj_pos(0) - robot.pinch_pos()) < 0.10
    cam.close(); wrist.close()
    return dict(kind=kind, held=bool(held), claimed=claimed, logs=logs, stances=tries + 1, size=spec['radius'],
                peak_arm_torque=[float(v) for v in robot.peak_torque[3:9]])


def run(n=10, frictions=(1.0, 0.35), verbose=True, kinds=O.KINDS):
    out = {}
    for mu in frictions:
        out[str(mu)] = {}
        for kind in kinds:
            res = [one(kind, 100 + i, mu) for i in range(n)]
            tried = [r for r in res if r['logs'] != ['8 个站位都不满足下手条件']]
            ok = sum(r['held'] for r in tried); wrong = sum(r['held'] != r['claimed'] for r in tried)
            why = {}
            for r in tried:
                if not r['held']:
                    k = (r['logs'] or ['夹起后掉落'])[-1]; why[k] = why.get(k, 0) + 1
            peak = np.max([r['peak_arm_torque'] for r in res], axis=0)
            out[str(mu)][kind] = dict(success=ok, n=len(tried), no_stance=n - len(tried), failures=why, misjudged=wrong,
                                      peak_arm_torque=[round(float(v), 2) for v in peak])
            if verbose:
                print(f'pad mu={mu} {O.TYPES[kind]["cn"]:<8} {ok}/{len(tried)} held (+{n - len(tried)} no stance) | self-check wrong {wrong} | {why}', flush=True)
    model.P['pad_friction'] = 1.0
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    json.dump(run(n), open('../results/grasp_bench.json', 'w'), ensure_ascii=False, indent=1)
