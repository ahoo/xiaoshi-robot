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
from skills import carry_pose, grasp, holding, misalignment, plan_grasp, to_chassis, wait

POS_NOISE = 0.004      # m, 1 sigma: wrist-camera localisation error assumed by the controller
ALIGN_NOISE = 0.15     # rad, 1 sigma: how far the object's narrow axis is from the shoulder-object line
RESTAND_MIS = 0.14     # rad: the controller re-stands above this measured misalignment (controller.py)


def one(kind, seed, pad_friction):
    rng = np.random.default_rng(seed)
    model.P['pad_friction'] = pad_friction
    rpose = (2.2, 1.05, rng.uniform(-3.1, 3.1))          # open floor, at least 0.7 m from any furniture
    # Where the base ends up relative to the object: 0.235 m +- servo and localisation error, a little off to the side,
    # and the object's narrow axis off the jaw line by ALIGN_NOISE. After the object has settled the robot takes the
    # same close look as the controller and applies the same test (controller.fetch): grasp only if the object is within
    # 6 cm of the centre line, a long object is within RESTAND_MIS of square, and the grasp planner finds a solution.
    # Otherwise the controller would move to a new stance; here the stance is redrawn (at most 8 times). So this bench
    # measures exactly what the full runs count as a grasp attempt.
    for tries in range(8):
        r_obj = rng.uniform(0.205, 0.265); bearing = rng.uniform(-0.3, 0.3); off = rng.normal(0, ALIGN_NOISE)
        pos_noise = rng.normal(0, POS_NOISE, 2); yaw_noise = rng.normal(0, 0.05)
        rel0 = np.array([r_obj * math.cos(bearing), r_obj * math.sin(bearing)])
        yaw_rel = math.atan2(rel0[1], rel0[0] - 0.069) + (0 if kind == 'block' else math.pi / 2) + off
        c, s_ = math.cos(rpose[2]), math.sin(rpose[2])
        robot = Robot([(kind, rpose[0] + c * rel0[0] - s_ * rel0[1], rpose[1] + s_ * rel0[0] + c * rel0[1], rpose[2] + yaw_rel)], seed=seed, robot_pose=rpose)
        for _ in range(30): robot.step()                 # 0.6 s: let the object and the robot settle
        rel = to_chassis(robot, robot.obj_pos(0) + np.append(pos_noise, 0), robot.true_pose())
        mis = misalignment(rel, robot.obj_yaw(0) - robot.true_pose()[2] + yaw_noise, kind)
        plan = plan_grasp(rel, kind, mis)
        if abs(rel[1]) <= 0.06 and (kind in ('ball', 'block') or mis < RESTAND_MIS) and plan is not None:
            break
    logs = [] if plan is not None else ['8 个站位都不满足下手条件']

    def script():
        if plan is None: return False
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
    return dict(kind=kind, held=bool(held), claimed=claimed, logs=logs, stances=tries + 1, peak_arm_torque=[float(v) for v in robot.peak_torque[3:9]],
                rms_arm_torque=[float(v) for v in np.sqrt(robot.torque_sq[3:9] / robot.n_steps)], sat_frac=[float(v) for v in robot.sat_steps[3:9] / robot.n_steps], t=robot.t)


def run(n=8, frictions=(1.0, 0.35), verbose=True):
    out = {}
    for mu in frictions:
        out[str(mu)] = {}
        for kind in OBJECTS:
            res = [one(kind, 100 + i, mu) for i in range(n)]
            tried = [r for r in res if r['logs'] != ['8 个站位都不满足下手条件']]
            ok = sum(r['held'] for r in tried); wrong = sum(r['held'] != r['claimed'] for r in tried)
            why = {}
            for r in tried:
                if not r['held']:
                    k = (r['logs'] or ['夹起后掉落'])[-1]; why[k] = why.get(k, 0) + 1
            peak = np.max([r['peak_arm_torque'] for r in res], axis=0); rms = np.max([r['rms_arm_torque'] for r in res], axis=0); sat = np.max([r['sat_frac'] for r in res], axis=0)
            out[str(mu)][kind] = dict(success=ok, n=len(tried), no_stance=n - len(tried), failures=why, misjudged=wrong, peak_arm_torque=[round(float(v), 2) for v in peak], rms_arm_torque=[round(float(v), 2) for v in rms], time_above_80pct=[round(float(v), 4) for v in sat])
            if verbose:
                fails = why
                print(f'pad mu={mu} {OBJECTS[kind]["cn"]:<6} {ok}/{len(tried)} held (+{n - len(tried)} no stance) | self-check wrong {wrong} | shoulder_lift torque peak {peak[1]:.2f} rms {rms[1]:.2f}, elbow peak {peak[2]:.2f} rms {rms[2]:.2f} N*m, >80% stall {100 * sat[:5].max():.1f}% of time | {fails}')
    model.P['pad_friction'] = 1.0
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    json.dump(run(n), open('../results/grasp_bench.json', 'w'), ensure_ascii=False, indent=1)
