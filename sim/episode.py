"""One full fetch episode: random room layout -> autonomous run -> score sheet.

  python episode.py [seed] [clearance_m] [--video out.mp4]
"""
import json
import math
import sys
import time

import numpy as np

from controller import Agent, T_MAX
from model import BASKET, FURNITURE, HOME, OBJECTS, ROOM_H, ROOM_W
from robot import Robot


def _furn_dist(x, y):
    d = min(x, y, ROOM_W - x, ROOM_H - y)
    for _, cx, cy, sx, sy, h, _ in FURNITURE:
        if sy is None: d = min(d, math.hypot(x - cx, y - cy) - sx)
        else: d = min(d, math.hypot(max(abs(x - cx) - sx / 2, 0), max(abs(y - cy) - sy / 2, 0)))
    return d


def make_layout(seed, clearance=0.30, n_boxes=None):
    """Five objects (one of each kind) at random poses at least `clearance` from walls and furniture, plus 0-2 extra cartons."""
    rng = np.random.default_rng(seed)
    boxes = []
    for _ in range(int(rng.integers(0, 3)) if n_boxes is None else n_boxes):
        for _ in range(200):
            x, y = rng.uniform(0.5, ROOM_W - 0.5), rng.uniform(0.5, ROOM_H - 0.5)
            if _furn_dist(x, y) < 0.62 or math.hypot(x - HOME[0], y - HOME[1]) < 0.8 or math.hypot(x - BASKET['x'], y - BASKET['y']) < 0.9: continue
            if any(math.hypot(x - b[0], y - b[1]) < 1.0 for b in boxes): continue
            boxes.append((round(x, 3), round(y, 3), 0.30, 0.25, 0.25)); break
    objs = []
    for kind in OBJECTS:
        for _ in range(500):
            x, y = rng.uniform(0.2, ROOM_W - 0.2), rng.uniform(0.2, ROOM_H - 0.2)
            if _furn_dist(x, y) < clearance: continue
            if any(math.hypot(max(abs(x - b[0]) - 0.15, 0), max(abs(y - b[1]) - 0.125, 0)) < clearance for b in boxes): continue
            if math.hypot(x - HOME[0], y - HOME[1]) < 0.5 or math.hypot(x - BASKET['x'], y - BASKET['y']) < 0.6: continue
            if any(math.hypot(x - o[1], y - o[2]) < 0.3 for o in objs): continue
            objs.append((kind, round(x, 3), round(y, 3), round(float(rng.uniform(-math.pi, math.pi)), 3))); break
    return objs, boxes


def run_episode(seed, clearance=0.30, verbose=False, on_step=None, pad_friction=None):
    import model
    if pad_friction is not None: model.P['pad_friction'] = pad_friction
    objs, boxes = make_layout(seed, clearance)
    robot = Robot(objs, seed=seed, extra_boxes=boxes)
    agent = Agent(robot, seed=seed, verbose=verbose)
    t0 = time.time(); gen = agent.run()
    try:
        while robot.t < T_MAX + 60:
            next(gen); robot.step(); agent.after_step()
            if on_step: on_step(robot, agent)
    except StopIteration:
        pass
    outcome = {}
    for k, (kind, *_rest) in enumerate(objs):
        e = agent.memory.get(k)
        if agent._in_basket(robot.obj_pos(k)): outcome[kind] = '已送达'
        elif e is None: outcome[kind] = '没找到'
        elif e['status'] == 'held': outcome[kind] = '超时（搬运中）'
        else: outcome[kind] = e.get('why', '超时')
    x, y, th = robot.true_pose()
    res = dict(seed=seed, clearance=clearance, n_boxes=len(boxes), delivered=sum(v == '已送达' for v in outcome.values()), total=len(objs), outcome=outcome,
               sim_time=round(robot.t, 1), finished=agent.state == '完成', wall_time=round(time.time() - t0, 1),
               odom_err=round(float(math.hypot(robot.odom[0] - x, robot.odom[1] - y)), 3),
               loc_err_mean=round(float(np.mean([e[0] for e in agent.loc_err])), 4) if agent.loc_err else None,
               loc_err_max=round(float(np.max([e[0] for e in agent.loc_err])), 4) if agent.loc_err else None,
               loc_head_err_max_deg=round(math.degrees(float(np.max([e[1] for e in agent.loc_err]))), 2) if agent.loc_err else None,
               scan_matches=agent.loc.n_match, scan_skips=agent.loc.n_skip,
               peak_torque=[round(float(v), 2) for v in robot.peak_torque], rms_torque=[round(float(v), 2) for v in np.sqrt(robot.torque_sq / robot.n_steps)],
               time_above_80pct=[round(float(v), 4) for v in robot.sat_steps / robot.n_steps], **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in agent.stats.items()})
    if pad_friction is not None: model.P['pad_friction'] = 1.0
    return res, agent, robot


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    clr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
    res, agent, robot = run_episode(seed, clr, verbose=True)
    print(json.dumps(res, ensure_ascii=False))
