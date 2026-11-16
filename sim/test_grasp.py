"""Bench test of grasp physics: robot parked, object on the floor in front of it, pick it up and hold it for 2 s.

  python test_grasp.py [trials_per_object]

Position/yaw noise stands in for camera error. Run with grip tape (pad friction 1.0) and bare PLA jaws (0.35).
"""
import json
import math
import sys

import numpy as np

import model
from model import OBJECTS, OBJ_HALF_H
from robot import Robot
from skills import carry_pose, grasp, holding, plan_grasp, to_chassis, wait

POS_NOISE = 0.004      # m, 1 sigma: wrist-camera localisation error assumed by the controller
ALIGN_NOISE = 0.15     # rad, 1 sigma: how far the object's narrow axis is from the shoulder-object line


def one(kind, seed, pad_friction):
    rng = np.random.default_rng(seed)
    model.P['pad_friction'] = pad_friction
    rpose = (1.6, 1.9, rng.uniform(-3.1, 3.1))
    r_obj = rng.uniform(0.20, 0.26); bearing = rng.uniform(-0.3, 0.3)
    ox = rpose[0] + r_obj * math.cos(rpose[2] + bearing); oy = rpose[1] + r_obj * math.sin(rpose[2] + bearing)
    # the base is expected to line the object's narrow side up with the shoulder-object line; ALIGN_NOISE is how well it manages
    px = rpose[0] + 0.069 * math.cos(rpose[2]); py = rpose[1] + 0.069 * math.sin(rpose[2])
    radial = math.atan2(oy - py, ox - px)
    yaw = radial + (0 if kind == 'block' else math.pi / 2) + rng.normal(0, ALIGN_NOISE)
    robot = Robot([(kind, ox, oy, yaw)], seed=seed, robot_pose=rpose)
    logs = []

    def script():
        yield from wait(robot, 0.6)
        p = robot.obj_pos(0) + np.append(rng.normal(0, POS_NOISE, 2), 0)
        rel = to_chassis(robot, p, robot.true_pose())
        plan = plan_grasp(rel, kind)
        if plan is None:
            logs.append('IK 无解'); return False
        ok = yield from grasp(robot, plan, logs.append)
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
    held = robot.obj_pos(0)[2] > 0.10 and np.linalg.norm(robot.obj_pos(0) - robot.pinch_pos()) < 0.08
    return dict(kind=kind, held=bool(held), claimed=claimed, logs=logs, peak_arm_torque=[float(v) for v in robot.peak_torque[3:9]],
                rms_arm_torque=[float(v) for v in np.sqrt(robot.torque_sq[3:9] / robot.n_steps)], sat_frac=[float(v) for v in robot.sat_steps[3:9] / robot.n_steps], t=robot.t)


def run(n=8, frictions=(1.0, 0.35), verbose=True):
    out = {}
    for mu in frictions:
        out[str(mu)] = {}
        for kind in OBJECTS:
            res = [one(kind, 100 + i, mu) for i in range(n)]
            ok = sum(r['held'] for r in res); wrong = sum(r['held'] != r['claimed'] for r in res)
            peak = np.max([r['peak_arm_torque'] for r in res], axis=0); rms = np.max([r['rms_arm_torque'] for r in res], axis=0); sat = np.max([r['sat_frac'] for r in res], axis=0)
            out[str(mu)][kind] = dict(success=ok, n=n, misjudged=wrong, peak_arm_torque=[round(float(v), 2) for v in peak], rms_arm_torque=[round(float(v), 2) for v in rms], time_above_80pct=[round(float(v), 4) for v in sat])
            if verbose:
                fails = [l for r in res if not r['held'] for l in (r['logs'] or ['(掉落)'])]
                print(f'pad mu={mu} {OBJECTS[kind]["cn"]:<6} {ok}/{n} held | self-check wrong {wrong} | shoulder_lift torque peak {peak[1]:.2f} rms {rms[1]:.2f}, elbow peak {peak[2]:.2f} rms {rms[2]:.2f} N*m, >80% stall {100 * sat[:5].max():.1f}% of time | {fails[:3]}')
    model.P['pad_friction'] = 1.0
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    json.dump(run(n), open('../results/grasp_bench.json', 'w'), ensure_ascii=False, indent=1)
