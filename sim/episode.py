"""One full fetch episode: random room layout -> autonomous run -> score sheet.

  python episode.py [seed] [clearance_m] [--video out.mp4]
"""
import json
import math
import sys
import time

import numpy as np

from controller import Agent, T_MAX
import objects as O
from model import BASKET, FURNITURE, HOME, ROOM_H, ROOM_W
from robot import Robot


def _furn_dist(x, y):
    d = min(x, y, ROOM_W - x, ROOM_H - y)
    for _, cx, cy, sx, sy, h, _ in FURNITURE:
        if sy is None: d = min(d, math.hypot(x - cx, y - cy) - sx)
        else: d = min(d, math.hypot(max(abs(x - cx) - sx / 2, 0), max(abs(y - cy) - sy / 2, 0)))
    return d


def make_layout(seed, clearance=0.30, n_boxes=None, clutter=False, materials=False):
    """Five objects of five different kinds out of eight (each with its own random size, shape and weight, objects.py)
    at random poses at least `clearance` from walls and furniture, plus 0-2 extra cartons.
    clutter: the second and fourth objects are put right next to the first and third (edges 0.5-5 cm apart), the
    way things end up in a heap, so the camera may see each pair as one blob.
    materials: each object is black or shiny (patchy depth), and a sixth object is added: a small clear plastic bottle,
    which a depth camera cannot measure (it shows up as a patch of missing depth, see controller._sense_clear). Drawn from a separate random stream, so the layout is otherwise the same."""
    rng = np.random.default_rng(seed)
    kinds = [O.KINDS[i] for i in rng.choice(len(O.KINDS), 5, replace=False)]
    specs = [O.sample(k, rng) for k in kinds]
    boxes = []
    for _ in range(int(rng.integers(0, 3)) if n_boxes is None else n_boxes):
        for _ in range(200):
            x, y = rng.uniform(0.5, ROOM_W - 0.5), rng.uniform(0.5, ROOM_H - 0.5)
            if _furn_dist(x, y) < 0.62 or math.hypot(x - HOME[0], y - HOME[1]) < 0.8 or math.hypot(x - BASKET['x'], y - BASKET['y']) < 0.9: continue
            if any(math.hypot(x - b[0], y - b[1]) < 1.0 for b in boxes): continue
            boxes.append((round(x, 3), round(y, 3), 0.30, 0.25, 0.25)); break
    objs = []
    for n_, (kind, spec) in enumerate(zip(kinds, specs)):
        for _ in range(500):
            x, y = rng.uniform(0.2, ROOM_W - 0.2), rng.uniform(0.2, ROOM_H - 0.2)
            if clutter and n_ in (1, 3):      # next to the previous one, with a small gap between their outlines
                a = rng.uniform(-math.pi, math.pi); prev = objs[-1]
                sep = 0.75 * (spec['radius'] + specs[n_ - 1]['radius']) + rng.uniform(0.005, 0.05)
                x, y = prev[1] + sep * math.cos(a), prev[2] + sep * math.sin(a)
                if _furn_dist(x, y) < clearance or math.hypot(x - BASKET['x'], y - BASKET['y']) < 0.6: continue
                objs.append((kind, round(x, 3), round(y, 3), round(float(rng.uniform(-math.pi, math.pi)), 3), spec)); break
            if _furn_dist(x, y) < clearance: continue
            if any(math.hypot(max(abs(x - b[0]) - 0.15, 0), max(abs(y - b[1]) - 0.125, 0)) < clearance for b in boxes): continue
            if math.hypot(x - HOME[0], y - HOME[1]) < 0.5 or math.hypot(x - BASKET['x'], y - BASKET['y']) < 0.6: continue
            if any(math.hypot(x - o[1], y - o[2]) < (0.45 if clutter else 0.3) for o in objs): continue
            objs.append((kind, round(x, 3), round(y, 3), round(float(rng.uniform(-math.pi, math.pi)), 3), spec)); break
    if materials:
        r2 = np.random.default_rng(seed + 7000)
        for o in objs: o[4]['mat'] = str(r2.choice(['black', 'shiny', 'matte'], p=[0.45, 0.45, 0.10]))
        spec = O.sample('bottle', r2)
        for _ in range(500):
            x, y = r2.uniform(0.2, ROOM_W - 0.2), r2.uniform(0.2, ROOM_H - 0.2)
            if _furn_dist(x, y) < clearance or math.hypot(x - HOME[0], y - HOME[1]) < 0.5 or math.hypot(x - BASKET['x'], y - BASKET['y']) < 0.6: continue
            if any(math.hypot(max(abs(x - b[0]) - 0.15, 0), max(abs(y - b[1]) - 0.125, 0)) < clearance for b in boxes): continue
            if any(math.hypot(x - o[1], y - o[2]) < 0.3 for o in objs): continue
            objs.append(('bottle', round(x, 3), round(y, 3), round(float(r2.uniform(-math.pi, math.pi)), 3), spec)); break
    return objs, boxes


def score_sheet(seed, clearance, objs, boxes, robot, agent, t0):
    """Score sheet (ground truth). The robot never knew which real object a memory stood for: match by position."""
    outcome = {}
    for k, (kind, *_rest) in enumerate(objs):
        p = robot.obj_pos(k)
        near = [(math.hypot(e['x'] - p[0], e['y'] - p[1]), e) for e in agent.memory.values()]
        e = min(near, key=lambda t: t[0])[1] if near and min(t[0] for t in near) < 0.15 else None
        if agent._in_basket_world(p): outcome[kind] = '已送达'
        elif p[2] > 0.06: outcome[kind] = '超时（搬运中）'
        elif e is None: outcome[kind] = '没找到'
        elif e['status'] == 'delivered': outcome[kind] = '放筐时掉在外面'
        else: outcome[kind] = e.get('why', '超时')
    x, y, th = robot.true_pose()
    res = dict(seed=seed, clearance=clearance, n_boxes=len(boxes), objects={o[0]: dict(mass_g=round(1000 * float(robot.m.body_subtreemass[robot.obj_body[i]]), 1), size_cm=round(200 * o[4]['radius'], 1)) for i, o in enumerate(objs)}, delivered=sum(v == '已送达' for v in outcome.values()), total=len(objs), outcome=outcome,
               sim_time=round(robot.t, 1), finished=agent.state == '完成', wall_time=round(time.time() - t0, 1),
               odom_err=round(float(math.hypot(robot.odom[0] - x, robot.odom[1] - y)), 3),
               loc_err_mean=round(float(np.mean([e[0] for e in agent.loc_err])), 4) if agent.loc_err else None,
               loc_err_max=round(float(np.max([e[0] for e in agent.loc_err])), 4) if agent.loc_err else None,
               loc_head_err_max_deg=round(math.degrees(float(np.max([e[1] for e in agent.loc_err]))), 2) if agent.loc_err else None,
               scan_matches=agent.loc.n_match, scan_skips=agent.loc.n_skip,
               basket_sightings=agent.basket_seen, places=agent.score_events,
               t_last=max([p[0] for p in agent.score_events if p[3]], default=None),   # when the last thing really went in
               peak_torque=[round(float(v), 2) for v in robot.peak_torque], rms_torque=[round(float(v), 2) for v in np.sqrt(robot.torque_sq / robot.n_steps)],
               time_above_80pct=[round(float(v), 4) for v in robot.sat_steps / robot.n_steps], **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in agent.stats.items()})
    return res


def run_episode(seed, clearance=0.30, verbose=False, on_step=None, pad_friction=None, clutter=False, perturbation=None, materials=False):
    import model
    import perturb
    perturb.CUR = dict(perturbation or {})
    if pad_friction is not None: model.P['pad_friction'] = pad_friction
    objs, boxes = make_layout(seed, clearance, clutter=clutter, materials=materials)
    robot = Robot(objs, seed=seed, extra_boxes=boxes)
    agent = Agent(robot, seed=seed, verbose=verbose)
    t0 = time.time(); gen = agent.run()
    try:
        while robot.t < T_MAX + 60:
            next(gen); robot.step(); agent.after_step()
            if on_step: on_step(robot, agent)
    except StopIteration:
        pass
    res = score_sheet(seed, clearance, objs, boxes, robot, agent, t0)
    if pad_friction is not None: model.P['pad_friction'] = 1.0
    perturb.CUR = {}
    return res, agent, robot


def run_supervised(seed, clearance=0.30, verbose=False, dock=False, soc0=None, cfg=None, script=None, t_end=None, start='task', materials=False, clutter=False,
                   on_step=None, perturbation=None, robot_pose=None, objects=None, boxes=None, until=None, right=True, est_bias=None, people=None, setup=None, robot_kw=None):
    """The same room and job as run_episode, but under modes.Supervisor: a battery (soc0 = state of charge at the start; None = no battery model),
    a charging dock (dock=True: the robot starts docked unless robot_pose says otherwise), modes, falls, an operator (`script(sup)` is called every tick).
    start: 'task' (start the fetch job), 'idle' (wait for the script).
    people: None, or dict(names=[people.SPECS keys], steps=[a people.Script step list per person], who=[...], family=[...], guard=True, cam='mast'|'forearm'|'periscope'): simulated people
    walk through the room (people.py) and the supervisor runs the head camera and, if guard, the safety layer."""
    import perturb, righting
    from model import DOCK_POSE
    from modes import Supervisor
    from power import Battery
    perturb.CUR = dict(perturbation or {})
    if objects is None: objects, boxes = make_layout(seed, clearance, clutter=clutter, materials=materials)
    pose = robot_pose or (DOCK_POSE if dock else HOME)
    cam = (people or {}).get('cam', 'mast')                                # 'mast' (the round-14 head camera) or a camera on the arm (model.ARM_CAMS): 'forearm', 'periscope'
    pk = {} if not people else dict(people=list(people['names']), **(dict(head_cam=True) if cam == 'mast' else dict(arm_cam=cam)))
    robot = Robot(objects, seed=seed, extra_boxes=boxes or [], dock=dock, robot_pose=pose, **pk, **(robot_kw or {}))       # robot_kw: more of the model's options, e.g. dict(arm_cam='forearm_up', arm_cam_collide=True)
    agent = Agent(robot, seed=seed, verbose=verbose)
    bat = Battery(soc0, rng=np.random.default_rng(seed + 99), est_bias=est_bias) if soc0 is not None else None
    sup = Supervisor(robot, agent, bat, cfg)
    if people:
        import people as people_lib
        sup.attach_people(people_lib.PeopleSystem(robot, agent, people['names'], [people_lib.Script(st) for st in people['steps']], who=people.get('who'), family=people.get('family'), seed=seed,
                                                  guard=people.get('guard', True), static_map=people.get('static_map'), guard_cfg=people.get('guard_cfg'), registry_path=people.get('registry_path'), comply=people.get('comply'), strict_face=people.get('strict_face')))
    if right: righting.attach(sup)
    if setup: setup(sup)                           # e.g. attach the home features (3D map, object records, voice) before the first tick
    if start == 'task': sup.start_task()
    t0 = time.time()
    def step(r, a):
        if script: script(sup)
        if on_step: on_step(r, a)
    sup.run(t_end if t_end is not None else T_MAX + 60, on_step=step, until=until)
    res = score_sheet(seed, clearance, objects, boxes or [], robot, agent, t0)
    res['finished'] = res['finished'] or sup.task_done
    res.update(mode=sup.mode, fault=sup.fault, sup_stats=dict(sup.stats), charge_log=sup.charge_log, fall_log=sup.fall_log, nav_results=sup.nav_results,
               soc_end=None if bat is None else round(bat.soc, 4), energy_wh=None if bat is None else round(bat.drawn_wh, 2))
    perturb.CUR = {}
    return res, sup


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    clr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
    res, agent, robot = run_episode(seed, clr, verbose=True)
    print(json.dumps(res, ensure_ascii=False))
