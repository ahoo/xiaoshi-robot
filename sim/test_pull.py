"""Bench test of dragging a thing out of a tight spot: one random object in front of the shelf with its edge 2-6 cm
from the shelf face, the robot facing it from the room. The robot looks, puts its closed jaws (side by side across
the pull: a solid paddle) down 3.5 cm past the far side it measured, backs up 15 cm, then looks again and tries an
ordinary grasp from where it now stands.  ->  results/pull_bench.json

  python test_pull.py [trials per kind and gap]
"""
import json
import math
import sys

import numpy as np

import objects as O
from perception import FrontCamera, WristCamera
from robot import Robot
from skills import PULL_MAX, PULL_REACH, grasp, plan_grasp, pull
from test_perception import true_points

SHELF_FACE = 0.35   # y of the shelf's front face (model.FURNITURE 'shelf')


def run_gen(robot, g):
    try:
        while True: next(g); robot.step()
    except StopIteration as e:
        return e.value


def one(kind, gap, seed):
    rng = np.random.default_rng(seed); spec = O.sample(kind, rng)
    ox = rng.uniform(2.35, 2.85); yaw = rng.uniform(-math.pi, math.pi)
    # place it, let it settle, then move it so its real edge is `gap` from the shelf face
    r0 = Robot([(kind, ox, 1.0, yaw, spec)], seed=seed, robot_pose=(ox, 1.6, -math.pi / 2))
    for _ in range(20): r0.step()
    W = true_points(r0, 0); oy = 1.0 - (W[:, 1].min() - SHELF_FACE - gap)
    far_true = 1.0 - W[:, 1].min(); near_true = W[:, 1].max() - 1.0
    # stand where the controller would: arm reaches 2 cm past the far side, body 17 cm clear of the near side
    dist = max(0.30 - far_true - PULL_REACH, near_true + 0.17) + rng.normal(0, 0.01)
    robot = Robot([(kind, ox, oy, yaw, spec)], seed=seed, robot_pose=(ox + rng.normal(0, 0.01), oy + dist, -math.pi / 2 + rng.normal(0, 0.03)))
    for _ in range(40): robot.step()
    cam = FrontCamera(robot, np.random.default_rng(seed)); wrist = WristCamera(robot)
    try:
        ds = [d for d in cam.detect(n_avg=8) if d['n_px'] >= 40 and d['bottom'] < 0.04]
        if not ds: return dict(kind=kind, gap=gap, seen=False, pulled=False, held=False)
        d = min(ds, key=lambda d: abs(d['rel'][1]))
        before = plan_grasp(d) is not None
        u = d['rel'][:2] / np.linalg.norm(d['rel'][:2]); t = d['pts'][:, :2] @ u; z = d['pts'][:, 2]
        near, far = np.percentile(t, [1, 99]); far = max(far, 2 * t[z >= z.max() - 0.003].mean() - near)
        p0 = robot.obj_pos(0).copy(); logs = []
        ok = run_gen(robot, pull(robot, (far + PULL_REACH) * u, 0.15, log=logs.append)) if np.linalg.norm((far + PULL_REACH) * u) <= PULL_MAX + 0.005 else False
        if not ok and not logs: logs.append('伸不到')
        p1 = robot.obj_pos(0); out = float(p1[1] - p0[1])
        for _ in range(20): robot.step()
        ds = [d for d in cam.detect(n_avg=8) if d['n_px'] >= 60 and d['bottom'] < 0.04]
        held = False
        if ds:
            d2 = min(ds, key=lambda d: abs(d['rel'][1])); plan = plan_grasp(d2)
            if plan is not None and 0.13 < d2['rel'][0] < 0.42:
                held = bool(run_gen(robot, grasp(robot, plan, logs.append, look=wrist.holding))) and robot.obj_pos(0)[2] > 0.06
            elif plan is None: logs.append('扒出来后从这边下不了手')
        return dict(kind=str(kind), gap=gap, seen=True, grasp_before=bool(before), pulled=bool(ok) and out > 0.08, out_cm=round(100 * out, 1), held=bool(held), logs=logs,
                    tilt_deg=round(math.degrees(robot.tilt()), 1))
    finally:
        cam.close(); wrist.close()


def run(n=2, gaps=(0.02, 0.04, 0.06)):
    out = {}
    for gap in gaps:
        rs = [one(str(k), gap, 400 + 10 * i + j) for i, k in enumerate(O.KINDS) for j in range(n)]
        out[str(gap)] = dict(n=len(rs), pulled=int(sum(r['pulled'] for r in rs)), held=int(sum(r['held'] for r in rs)),
                             grasp_before=int(sum(bool(r.get('grasp_before')) for r in rs)), trials=rs)
        o = out[str(gap)]
        print(f'gap {100 * gap:.0f} cm: pulled out {o["pulled"]}/{o["n"]}, then grasped {o["held"]}/{o["n"]}', flush=True)
        for r in rs:
            if not r['held']: print('   ', r['kind'], r.get('out_cm'), r.get('logs'), flush=True)
    return out


if __name__ == '__main__':
    json.dump(run(int(sys.argv[1]) if len(sys.argv) > 1 else 2), open('../results/pull_bench.json', 'w'), ensure_ascii=False, indent=1)
