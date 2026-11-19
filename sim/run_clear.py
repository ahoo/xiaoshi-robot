"""The clear plastic bottle on its own: does the robot find it, pick it up and bring it in?  ->  results/clear_bottle.json

  python run_clear.py ab [procs]       seeds 301-312 (the 'materials' layouts), the bottle alone in the room, with the two
                                       remedies on / only the wheel-slip one / neither (round 11 behaviour)
  python run_clear.py forced [procs]   the bottle 0.8 m ahead of the robot on open floor, its axis at 0, 10, 30, 60, 90, -30
                                       and -60 degrees to the line of sight, no turn on the spot at start-up (so that the first
                                       view is the one a robot driving up to it would have); one run each
  python run_clear.py robust [procs]   the forced set again with rougher clear-plastic depth: patches instead of single
                                       pixels, fewer pixels without a reading, both (perturb.py: clear_holes, clear_block)
  python run_clear.py check            a few seconds: one run, to see that everything works

The depth model of clear plastic is the weakest assumption here: how much of a real bottle gives no reading depends on the
camera, the light and the bottle, and has to be measured (docs/validation.md). 'robust' shows how far the method can be pushed.
"""
import json
import math
import multiprocessing as mp
import os
import sys
import types

import numpy as np

import controller as C
import objects as O
import perturb
from controller import Agent, T_MAX
from episode import make_layout
from robot import Robot
from skills import wait

OUT = '../results/clear_bottle.json'
AB = [('both', dict(cue=True, slip=True)), ('slip_only', dict(cue=False, slip=True)), ('neither', dict(cue=False, slip=False))]
ANGLES = (0, 10, 30, 60, 90, -30, -60)
ROBUST = [('patches_4px', dict(clear_block=4)), ('fewer_holes', dict(clear_holes=(0.4, 0.5))), ('both_rougher', dict(clear_holes=(0.35, 0.45), clear_block=4))]


NAV = ('导航', '巡视', '回位')               # driving about, not carrying it (the gripper touches it then) or grasping it


def _go(robot, agent, t_end, stop_when_in=True):
    """Run the episode. Counts, from the simulator's own contacts (score sheet only, the robot never sees this): how often
    the chassis or a wheel touched the bottle while driving (separate touches, more than 1 s apart), and how far the
    bottle ended up from where it started if it was never picked up."""
    m = robot.m; bg = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == robot.obj_body[0]}
    rg = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == robot.chassis}
    p0 = robot.obj_pos(0).copy(); hits = 0; last = -9.0
    gen = agent.run()
    try:
        while robot.t < t_end:
            next(gen); robot.step(); agent.after_step()
            if agent.state in NAV and robot.t - last > 1.0:
                for i in range(robot.d.ncon):
                    c = robot.d.contact[i]
                    if (c.geom1 in rg and c.geom2 in bg) or (c.geom2 in rg and c.geom1 in bg): hits += 1; last = robot.t; break
            if stop_when_in and robot.t > 20 and agent._in_basket_world(robot.obj_pos(0)): break
    except StopIteration:
        pass
    moved = float(np.linalg.norm(robot.obj_pos(0)[:2] - p0[:2])); delivered_ = bool(agent._in_basket_world(robot.obj_pos(0)))
    return dict(delivered=delivered_, sim_t=round(robot.t, 1), bottle_touches=hits, bottle_moved_cm=round(100 * moved, 1) if not delivered_ else None,
                grasp_attempts=agent.stats['grasp_attempts'], grasp_fail=agent.stats['grasp_fail'],
                collisions=agent.stats['collisions'], stuck=agent.stats['stuck'], slips=agent.stats.get('slips', 0), floor_touch=agent.stats['floor_touch'],
                pulls=agent.stats['pulls'], blind_entries=sum(bool(e.get('blind')) for e in agent.memory.values()),
                looks=sum(('侧面' in m or '朝向' in m) for _, m in agent.log_lines), log=agent.log_lines)


def job_ab(args):
    name, seed, cue, slip = args
    C.CLEAR_CUE, C.SLIP_ON = cue, slip; perturb.CUR = {}
    objs, boxes = make_layout(seed, 0.30, materials=True)
    robot = Robot([[o for o in objs if o[0] == 'bottle'][0]], seed=seed, extra_boxes=boxes)
    res = _go(robot, Agent(robot, seed=seed), T_MAX + 60, stop_when_in=False)
    res.update(setting=name, seed=seed); return res


def job_forced(args):
    name, angle, pert = args
    C.CLEAR_CUE, C.SLIP_ON = True, True; perturb.CUR = dict(pert)
    rng = np.random.default_rng(1); spec = O.sample('bottle', rng); rpose = (2.2, 1.05, 0.0)
    robot = Robot([('bottle', rpose[0] + 0.8, rpose[1], math.radians(angle), spec)], seed=1, robot_pose=rpose)
    agent = Agent(robot, seed=1)
    def no_spin(self, turns=1.0):                 # keep the first view: do not turn on the spot at start-up
        yield from wait(self.r, 1.0)
    agent.spin = types.MethodType(no_spin, agent)
    res = _go(robot, agent, 240)
    perturb.CUR = {}
    res.update(setting=name, angle=angle); return res


def summarise(rows, key='setting'):
    out = {}
    for name in dict.fromkeys(r[key] for r in rows):
        sel = [r for r in rows if r[key] == name]
        out[name] = dict(n=len(sel), delivered=sum(r['delivered'] for r in sel), collisions=sum(r['collisions'] for r in sel), stuck=sum(r['stuck'] for r in sel), slips=sum(r['slips'] for r in sel),
                         bottle_touches=sum(r['bottle_touches'] for r in sel), runs_touching=sum(r['bottle_touches'] > 0 for r in sel),
                         grasp_attempts=sum(r['grasp_attempts'] for r in sel), grasp_fail=sum(r['grasp_fail'] for r in sel), floor_touch=sum(r['floor_touch'] for r in sel),
                         mean_sim_t_delivered=round(float(np.mean([r['sim_t'] for r in sel if r['delivered']])), 1) if any(r['delivered'] for r in sel) else None,
                         side_looks=sum(r['looks'] > 0 for r in sel))
    return out


def _save(key, rows):
    data = json.load(open(OUT)) if os.path.exists(OUT) else {}
    data[key] = dict(summary=summarise(rows), runs=[{k: v for k, v in r.items() if k != 'log'} for r in rows])
    data[key]['failure_logs'] = [dict(setting=r['setting'], seed=r.get('seed', r.get('angle')), log=r['log']) for r in rows if not r['delivered']]
    json.dump(data, open(OUT, 'w'), ensure_ascii=False, indent=1)
    for k, v in data[key]['summary'].items(): print(key, k, v, flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if mode == 'check':
        r = job_forced(('check', 60, {})); print({k: v for k, v in r.items() if k != 'log'})
        sys.exit()
    if mode == 'ab': jobs, fn = [(n, s, o['cue'], o['slip']) for n, o in AB for s in range(301, 313)], job_ab
    elif mode == 'forced': jobs, fn = [('forced', a, {}) for a in ANGLES], job_forced
    elif mode == 'robust': jobs, fn = [(n, a, p) for n, p in ROBUST for a in ANGLES], job_forced
    else: sys.exit(__doc__)
    with mp.Pool(procs, maxtasksperchild=1) as pool: rows = pool.map(fn, jobs, chunksize=1)
    _save(mode, rows)
