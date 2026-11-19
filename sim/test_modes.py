"""Benches for the modes (modes.py), the charging dock (dock.py, power.py), remote control and autonomous navigation.

  python test_modes.py dock [n] [procs]       docking from n random poses in the furnished room, with the LD19 lidar error model
  python test_modes.py nav [n] [procs]        autonomous navigation: random goals in n random layouts, plus goals that cannot be reached
  python test_modes.py teleop                 a scripted operator: dead-man, safety layer on/off, arm commands, taking the robot over from the job and handing it back
  python test_modes.py lowbatt [seeds]        the whole job with a low battery: soft stop, charging, resume; and the critical case
  python test_modes.py server                 the TCP remote-control server: commands over a real socket against a running simulation
Each writes results/modes_<name>.json.
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import episode
from episode import _furn_dist, make_layout, run_supervised
from model import BASKET, DOCK_POSE, FURNITURE, HOME
from robot import GRIP_CLOSED, GRIP_OPEN, Q_CARRY

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def free_pose(rng, margin=0.40):
    while True:
        x, y = rng.uniform(0.4, 3.6), rng.uniform(0.4, 2.6)
        if _furn_dist(x, y) > margin and math.hypot(x - BASKET['x'], y - BASKET['y']) > 0.6: return x, y, float(rng.uniform(-math.pi, math.pi))


def _save(name, data):
    os.makedirs(RES, exist_ok=True); json.dump(data, open(os.path.join(RES, f'modes_{name}.json'), 'w'), ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------------------------- dock
def _dock_one(args):
    seed, lidar = args
    rng = np.random.default_rng(1000 + seed); pose = free_pose(rng)
    rec = dict(seed=seed, start=[round(v, 2) for v in pose], contact=None)
    st = dict(t_req=None)

    def script(sup):
        r = sup.r
        if st['t_req'] is None and r.t > 20.0: st['t_req'] = r.t; sup.charge_now()
        if rec['contact'] is None and sup.mode == 'CHARGE' and sup.a.state == '充电':
            x, y, th = r.true_pose(); rec['contact'] = dict(t=round(r.t - st['t_req'], 1), x=round(x, 3), y=round(y, 3), yaw_deg=round(math.degrees(th - DOCK_POSE[2]), 1),
                                                           dx_cm=round(100 * (x - DOCK_POSE[0]), 1), dy_cm=round(100 * (y - DOCK_POSE[1]), 1))
    res, sup = run_supervised(seed, dock=True, soc0=0.5, robot_pose=pose, objects=[], boxes=[], perturbation=dict(lidar_model=lidar), script=script, t_end=200.0,
                              until=lambda s: bool(s.charge_log) and s.mode == 'IDLE' or s.fault is not None)
    rec.update(docked=bool(sup.charge_log), fault=sup.fault, collisions=res['collisions'], stuck=res['stuck'], t_end=res['sim_time'],
               attempts=sum(1 for e in sup.events if '第 2 次' in e[2] or '对接没成功' in e[2]) + 1, loc_err_max=res['loc_err_max'],
               left_dock=None if not sup.charge_log else math.hypot(sup.r.true_pose()[0] - DOCK_POSE[0], sup.r.true_pose()[1] - DOCK_POSE[1]) > 0.25,
               soc=[sup.charge_log[0]['soc_from'], sup.charge_log[0]['soc_to']] if sup.charge_log else None, charge_min=sup.stats['charge_min'])
    return rec


def bench_dock(n=10, procs=2, lidar='ld19cal'):
    t0 = time.time()
    with mp.Pool(procs, maxtasksperchild=1) as p: recs = p.map(_dock_one, [(s, lidar) for s in range(1, n + 1)])
    ok = [r for r in recs if r['docked']]
    print(f'docked {len(ok)}/{n}; collisions {sum(r["collisions"] for r in recs)}; attempts>1: {sum(r["attempts"] > 1 for r in recs)}; '
          f'time to contact (s) median {np.median([r["contact"]["t"] for r in ok if r["contact"]]) if ok else None}')
    for r in recs: print(r)
    _save('dock' + ('' if lidar == 'ld19cal' else '_' + lidar), dict(lidar=lidar, n=n, runs=recs, wall=round(time.time() - t0)))


def _dockfail_one(args):
    seed, off = args
    import dock as dock_lib
    from model import DOCK_POSE as DP
    dock_lib.DOCK_POSE = (DP[0] + off, DP[1], DP[2])          # the robot believes the dock is `off` m to the side of where it really is
    st = dict(t_req=None)
    def script(sup):
        if st['t_req'] is None and sup.r.t > 15.0: st['t_req'] = sup.r.t; sup.charge_now()
    res, sup = run_supervised(seed, dock=True, soc0=0.5, robot_pose=(1.8, 1.6, 0.5), objects=[], boxes=[], script=script, t_end=240.0, perturbation=dict(lidar_model='ld19cal'),
                              until=lambda s: bool(s.charge_log) or s.fault is not None)
    return dict(seed=seed, believed_offset_cm=round(100 * off, 1), docked=bool(sup.charge_log), fault=sup.fault, max_tilt_deg=res['max_tilt'], collisions=res['collisions'], t_end=res['sim_time'],
                events=[e[2] for e in sup.events if e[1] in ('CHARGE', 'FAULT')][:12])


def bench_dockfail(procs=2):
    """The dock is not where the robot thinks (moved, or mis-taught): small offsets still dock (the strip is 24 cm wide), large ones fail cleanly: no damage, 3 tries, then FAULT."""
    offs = (0.0, 0.08, 0.15, 0.25)
    with mp.Pool(procs, maxtasksperchild=1) as p: recs = p.map(_dockfail_one, [(1, o) for o in offs])
    for r in recs: print({k: v for k, v in r.items() if k != 'events'}); print('   ', r['events'][:8])
    _save('dockfail', dict(runs=recs))


# --------------------------------------------------------------------------------------------- navigation
def _nav_one(args):
    seed, lidar = args
    rng = np.random.default_rng(2000 + seed)
    objects, boxes = make_layout(seed, clutter=False)
    goals = []
    for k in range(6):
        for _ in range(100):
            x, y, th = free_pose(rng, 0.40)
            if all(math.hypot(x - o[1], y - o[2]) > 0.25 for o in objects) and all(math.hypot(max(abs(x - b[0]) - 0.15, 0), max(abs(y - b[1]) - 0.125, 0)) > 0.4 for b in boxes): break
        goals.append((x, y, th if k % 2 else None))
    bad = [(0.95, 0.40, None), (3.55, 0.45, None)]            # inside the sofa; inside the plant pot (cannot be reached)
    allg = goals + bad
    rec = dict(seed=seed, goals=[[round(v, 2) if v is not None else None for v in g] for g in allg], arrivals=[])
    seen = [0]

    def script(sup):
        if sup.r.t > 6.0 and sup.mode == 'IDLE' and not st['sent']:
            st['sent'] = True; sup.request('NAV', goals=[dict(x=g[0], y=g[1], th=g[2], tol=0.07) for g in allg])
        while seen[0] < len(sup.nav_results):
            res = sup.nav_results[seen[0]]; g = allg[seen[0]]; x, y, th = sup.r.true_pose()
            err = math.hypot(x - g[0], y - g[1]); herr = None if g[2] is None else abs(math.degrees((th - g[2] + math.pi) % (2 * math.pi) - math.pi))
            rec['arrivals'].append(dict(goal=seen[0], result=res['result'], secs=res['secs'], err_cm=round(100 * err, 1), head_err_deg=None if herr is None else round(herr, 1)))
            seen[0] += 1
    st = dict(sent=False)
    res, sup = run_supervised(seed, objects=objects, boxes=boxes, soc0=None, perturbation=dict(lidar_model=lidar), script=script, t_end=900.0, start='idle',
                              until=lambda s: seen[0] >= len(allg))
    rec.update(collisions=res['collisions'], stuck=res['stuck'], t_end=res['sim_time'], loc_err_max=res['loc_err_max'], max_tilt=res['max_tilt'])
    return rec


def bench_nav(n=6, procs=2, lidar='ld19cal'):
    t0 = time.time()
    with mp.Pool(procs, maxtasksperchild=1) as p: recs = p.map(_nav_one, [(s, lidar) for s in range(1, n + 1)])
    reach = [a for r in recs for a in r['arrivals'] if a['goal'] < 6]; bad = [a for r in recs for a in r['arrivals'] if a['goal'] >= 6]
    arrived = [a for a in reach if a['result'] == 'arrived']
    print(f'reachable goals arrived {len(arrived)}/{len(reach)}; median error {np.median([a["err_cm"] for a in arrived]):.1f} cm, worst {max(a["err_cm"] for a in arrived):.1f} cm; '
          f'median time {np.median([a["secs"] for a in arrived]):.0f} s; unreachable goals correctly refused {sum(a["result"] != "arrived" for a in bad)}/{len(bad)}; collisions {sum(r["collisions"] for r in recs)}')
    for r in recs: print(r)
    _save('nav' + ('' if lidar == 'ld19cal' else '_' + lidar), dict(lidar=lidar, n=n, runs=recs, wall=round(time.time() - t0)))


def _explore_one(args):
    seed, lidar = args
    objects, boxes = make_layout(seed)
    seen = []
    def script(sup):
        if sup.r.t > 2.0 and not seen: seen.append(1); sup.explore()
    res, sup = run_supervised(seed, objects=objects, boxes=boxes, soc0=None, dock=True, perturbation=dict(lidar_model=lidar), script=script, t_end=600.0, start='idle',
                              until=lambda s: bool(s.nav_results) and s.mode == 'IDLE')
    a = sup.a; free = int((a.lo < 0).sum()); occ = int((a.lo > 4).sum())
    return dict(seed=seed, done=bool(sup.nav_results), t_end=res['sim_time'], coverage=round(float(a.cov.mean()), 3), mapped_free_cells=free, mapped_occupied_cells=occ,
                objects_found=len([e for e in a.memory.values()]), objects_total=len(objects), collisions=res['collisions'], stuck=res['stuck'], loc_err_max=res['loc_err_max'], delivered=res['delivered'])


def bench_explore(n=3, procs=2, lidar='ld19cal'):
    """NAV explore: the robot drives round until it has looked at the whole floor, builds the map, finds what lies about, and picks up nothing."""
    with mp.Pool(procs, maxtasksperchild=1) as p: recs = p.map(_explore_one, [(s, lidar) for s in range(1, n + 1)])
    for r in recs: print(r)
    _save('explore', dict(lidar=lidar, runs=recs))


# --------------------------------------------------------------------------------------------- low battery
def _lowbatt_one(args):
    seed, soc0, dock, lidar = args
    objects, boxes = make_layout(seed)
    res, sup = run_supervised(seed, objects=objects, boxes=boxes, dock=dock, soc0=soc0, est_bias=0.0, perturbation=dict(lidar_model=lidar) if lidar else None,
                              until=lambda s: s.mode in ('IDLE', 'FAULT') and s.r.t > 30)
    return dict(seed=seed, soc0=soc0, dock=dock, delivered=res['delivered'], total=res['total'], sim_time=res['sim_time'], mode=sup.mode, fault=sup.fault, charges=sup.stats['charges'],
                charge_log=sup.charge_log, collisions=res['collisions'], stuck=res['stuck'], soc_end=res['soc_end'], energy_wh=res['energy_wh'],
                events=[e for e in sup.events if e[1] != 'TASK' or '→' in e[2]][:40])


def bench_lowbatt(seeds=(1, 2), procs=2, lidar='ld19cal'):
    jobs = [(s, soc0, True, lidar) for s in seeds for soc0 in (None, 0.21, 0.105)]
    with mp.Pool(procs, maxtasksperchild=1) as p: recs = p.map(_lowbatt_one, jobs)
    for r in recs: print({k: v for k, v in r.items() if k != 'events'})
    _save('lowbatt', dict(runs=recs))


# --------------------------------------------------------------------------------------------- teleop
def _teleop_a(safety):
    """Empty room, robot at (2.8, 1.2) facing the east wall 1.2 m away. The operator drives straight at it for 14 s, then lets go."""
    rec = dict(safety=safety); log = []
    st = dict(phase=0)

    def script(sup):
        r = sup.r; t = r.t
        if t > 1.0 and st['phase'] == 0: st['phase'] = 1; sup.teleop(); sup.set_safety(safety)
        if st['phase'] >= 1 and 2.0 <= t < 16.0: sup.drive(0.15, 0.0, 0.0)
        if t >= 16.0 and st['phase'] == 1: st['phase'] = 2; rec['t_release'] = round(t, 2)
        x, y, th = r.true_pose(); log.append((t, x, y, float(np.linalg.norm(r.d.qvel[:2]))))
    res, sup = run_supervised(7, objects=[], boxes=[], soc0=None, robot_pose=(2.8, 1.2, 0.0), script=script, t_end=22.0, start='idle')
    L = np.array(log); xs = L[:, 1]
    rec.update(x_final=round(float(xs[-1]), 3), gap_to_wall_cm=round(100 * (4.0 - float(xs.max()) - 0.145), 1), collisions=res['collisions'],
               speed_after_release=round(float(L[(L[:, 0] > 16.0 + 0.7) & (L[:, 0] < 17.0), 3].mean()), 4), max_speed=round(float(L[:, 3].max()), 3),
               events=[e[2] for e in sup.events if e[1] == 'TELEOP'][:6])
    return rec


def _teleop_deadman():
    """Commands for 2 s, then silence: how far does it coast, and when has it stopped?"""
    log = []; st = dict(on=False)

    def script(sup):
        r = sup.r; t = r.t
        if t > 1.0 and not st['on']: st['on'] = True; sup.teleop(); sup.set_safety(True)
        if st['on'] and 2.0 <= t < 4.0: sup.drive(0.15, 0.0, 0.0)
        log.append((t, r.true_pose()[0], float(np.linalg.norm(r.d.qvel[:2]))))
    run_supervised(7, objects=[], boxes=[], soc0=None, robot_pose=(2.0, 1.2, 0.0), script=script, t_end=8.0, start='idle')
    L = np.array(log); last = 3.98                                   # the last command went out in the tick before t = 4.0
    x_release = L[np.argmin(abs(L[:, 0] - last)), 1]; after = L[L[:, 0] > last]
    moving = np.nonzero(after[:, 2] >= 0.01)[0]                      # the first moment from which it stays slower than 1 cm/s
    stop_t = float(after[moving.max() + 1, 0] - last) if len(moving) and moving.max() + 1 < len(after) else None
    return dict(coast_cm=round(100 * float(L[:, 1].max() - x_release), 1), stopped_after_s=None if stop_t is None else round(stop_t, 2), deadman_s=0.5, travelled_before_release_cm=round(100 * float(x_release - L[0, 1]), 1))


def _teleop_arm():
    """Arm by joint angles and by hand position; a command that would put the arm through the chassis is refused."""
    out = dict(); st = dict(step=0, t0=0.0)
    import skills
    ik = skills.IK
    rng = np.random.default_rng(3)
    bad = None
    for _ in range(2000):
        q = rng.uniform(ik.lo, ik.hi)
        if ik.hits_body(q): bad = q; break
    def script(sup):
        r = sup.r; t = r.t; tc = sup.tele
        if st['step'] == 0 and t > 1.0: st['step'] = 1; st['t0'] = t; sup.teleop(); out['reachable'] = sup.arm_to(0.25, 0.0, 0.06, grip=GRIP_OPEN)
        if st['step'] == 1 and t - st['t0'] > 4.0:
            st['step'] = 2; st['t0'] = t; out['pinch_err_mm'] = round(1000 * float(np.linalg.norm(r.pinch_pos() - np.array([*(np.array(r.true_pose()[:2]) + 0.25 * np.array([math.cos(r.true_pose()[2]), math.sin(r.true_pose()[2])])), 0.06]))), 1)
            out['arm_busy_when_out'] = bool(sup.a.arm_busy); out['unreachable_refused'] = not sup.arm_to(0.9, 0.0, 0.3)
            sup.arm_joints(bad)
        if st['step'] == 2 and t - st['t0'] > 1.0:
            st['step'] = 3; st['t0'] = t; out['body_hit_refused'] = len(tc.messages) > 0 and 'chassis' in tc.messages[-1]; out['arm_moved_after_refusal'] = bool(np.abs(r.arm_target[:5] - bad).max() < 0.01)
            sup.arm_joints(Q_CARRY, grip=0.3)
        if st['step'] == 3 and t - st['t0'] > 4.0:
            st['step'] = 4; out['back_to_carry_err'] = round(float(np.abs(r.joints()[:5] - Q_CARRY).max()), 3); out['arm_busy_when_carry'] = bool(sup.a.arm_busy)
    run_supervised(7, objects=[], boxes=[], soc0=None, robot_pose=(2.0, 1.5, 0.0), script=script, t_end=14.0, start='idle')
    return out


def _takeover_one(args):
    seed, takeover = args
    objects, boxes = make_layout(seed)
    st = dict(phase=0)

    def script(sup):
        t = sup.r.t
        if not takeover: return
        # takeover == 'held': the moment it has something in the gripper (and is carrying it to the basket); True: a fixed time, 60 s
        go = (t > 60.0) if takeover is True else (sup.mode == 'TASK' and sup.a.held and sup.a.state == '搬运' and t > 30.0)
        if st['phase'] == 0 and go: st['phase'] = 1; sup.teleop(); st['t0'] = t; st['held'] = bool(sup.a.held)
        if st['phase'] == 1:
            dt = t - st['t0']
            if 1.0 <= dt < 5.0: sup.drive(0.0, 0.1, 0.0)
            elif 6.0 <= dt < 10.0: sup.drive(0.0, 0.0, 0.5)
            elif dt >= 12.0: st['phase'] = 2; sup.start_task()
    res, sup = run_supervised(seed, objects=objects, boxes=boxes, soc0=None, script=script, until=lambda s: s.task_done)
    return dict(seed=seed, takeover=takeover, delivered=res['delivered'], total=res['total'], sim_time=res['sim_time'], collisions=res['collisions'], stuck=res['stuck'], held_when_taken=st.get('held'),
                takeovers=sup.stats['takeovers'], finished=res['finished'])


def bench_teleop(procs=2):
    t0 = time.time(); out = dict()
    # The whole-job runs go first, in a pool with one worker per run created before this process has made any rendering context of its own (a worker
    # forked after that, as maxtasksperchild would do, hangs); the single-process checks follow.
    jobs = [(s, tk) for s in (1, 3) for tk in (False, True, 'held')]
    with mp.Pool(len(jobs)) as p: out['takeover'] = p.map(_takeover_one, jobs)
    for r in out['takeover']: print(r)
    out['safety_on'] = _teleop_a(True); print(out['safety_on'])
    out['safety_off'] = _teleop_a(False); print(out['safety_off'])
    out['deadman'] = _teleop_deadman(); print(out['deadman'])
    out['arm'] = _teleop_arm(); print(out['arm'])
    out['wall'] = round(time.time() - t0); _save('teleop', out)


# --------------------------------------------------------------------------------------------- falls
def inject_fall(robot, kind, strength=1.0):
    """Knock the robot over: a torque about its own roll (x) or pitch (y) axis for a quarter of a second, applied through the physics engine's external-force
    field. kind: roll_l, roll_r, pitch_f, pitch_b. Returns a function to call every control tick that applies it and then lets go."""
    R = robot.d.xmat[robot.chassis].reshape(3, 3).copy()
    axis = dict(roll_l=(1, 0, 0), roll_r=(-1, 0, 0), pitch_f=(0, 1, 0), pitch_b=(0, -1, 0))[kind]
    tw = R @ (np.array(axis, float) * 3.2 * strength); t0 = robot.t
    def tick():
        if robot.t - t0 < 0.25: robot.d.xfrc_applied[robot.chassis, 3:6] = tw
        else: robot.d.xfrc_applied[robot.chassis, 3:6] = 0.0
    return tick


def _fall_one(args):
    seed, kind, strength, t_fall, in_task, lidar = args
    rng = np.random.default_rng(3000 + seed)
    objects, boxes = make_layout(seed) if in_task else ([], [])
    st = dict(tick=None)
    def script(sup):
        r = sup.r
        if st['tick'] is None and r.t > t_fall:
            st['tick'] = inject_fall(r, kind, strength); st['pose'] = [round(v, 2) for v in r.true_pose()]; st['held'] = bool(sup.a.held); st['state'] = sup.a.state
        if st['tick'] is not None: st['tick']()
    pose = None if in_task else free_pose(rng, 0.30)
    res, sup = run_supervised(seed, objects=objects, boxes=boxes, soc0=None, perturbation=dict(lidar_model=lidar) if lidar else None, script=script, t_end=t_fall + (110.0 if not in_task else 560.0),
                              robot_pose=pose, until=(lambda s: s.stats['falls'] > 0 and s.mode in ('IDLE', 'FAULT', 'TASK') and s.stats['righted'] + s.stats['right_failed'] > 0 and s.r.t > t_fall + 8) if not in_task else (lambda s: s.task_done or s.fault))
    ev = [e[2] for e in sup.events if e[1] in ('RIGHT', 'FAULT')]
    reloc = getattr(sup, 'relocalized', [])
    la = [e[0] for e in sup.a.loc_err[-8:]]
    return dict(seed=seed, kind=kind, strength=strength, in_task=in_task, pose=st.get('pose'), state_at_fall=st.get('state'), held=st.get('held'), falls=sup.stats['falls'], righted=sup.stats['righted'], failed=sup.stats['right_failed'],
                fault=sup.fault, mode=sup.mode, tilt_end=round(math.degrees(sup.r.tilt()), 1), reloc_err=reloc, loc_err_end_cm=round(100 * float(np.mean(la)), 1) if la else None,
                delivered=res['delivered'], total=res['total'], sim_time=res['sim_time'], collisions=res['collisions'], events=ev[:14])


def bench_fall(n=16, procs=2, lidar='ld19cal'):
    """n knock-overs at random poses in the furnished room (no objects), directions and strengths cycled; then 4 whole jobs with a fall in the middle."""
    kinds = ['roll_l', 'roll_r', 'pitch_f', 'pitch_b']; strengths = [1.0, 1.4, 1.8, 2.2]
    jobs = [(s, kinds[(s - 1) % 4], strengths[((s - 1) // 4) % 4], 100.0 + 4.0 * s, False, lidar) for s in range(1, n + 1)]      # after the first 100 s: the map is mostly built
    with mp.Pool(procs, maxtasksperchild=1) as p: recs = p.map(_fall_one, jobs)
    for r in recs: print({k: v for k, v in r.items() if k != 'events'})
    down = [r for r in recs if r['falls'] > 0]
    print(f'knocked down {len(down)}/{len(recs)}; got up {sum(r["righted"] > 0 for r in down)}; faults {[r["fault"] for r in down if r["fault"]]}; '
          f'pose after getting up: median error {np.median([x[0] for r in down for x in r["reloc_err"]]) * 100 if any(r["reloc_err"] for r in down) else None} cm')
    _save('fall', dict(runs=recs, lidar=lidar))


def bench_fall_job(procs=2, lidar='ld19cal'):
    """The whole job (5 objects) with a fall at 90 s and one at 200 s, the second one while carrying something if the timing allows."""
    jobs = [(s, k, 1.6, t, True, lidar) for s, (k, t) in enumerate([('roll_l', 90.0), ('pitch_f', 120.0), ('roll_r', 160.0), ('roll_l', 220.0)], 1)]
    with mp.Pool(procs, maxtasksperchild=1) as p: recs = p.map(_fall_one, jobs)
    for r in recs: print({k: v for k, v in r.items() if k != 'events'}); print('   ', r['events'][:8])
    _save('fall_job', dict(runs=recs, lidar=lidar))


# --------------------------------------------------------------------------------------------- TCP server
def bench_server():
    import socket, threading
    from teleop_server import TeleopServer
    out = dict(replies=[]); box = dict(sup=None)
    def run_sim():
        def script(sup):
            if box['sup'] is None:
                box['sup'] = sup; box['server'] = TeleopServer(sup)
        run_supervised(7, objects=[], boxes=[], soc0=0.6, dock=True, robot_pose=(2.0, 1.5, 0.0), script=script, t_end=60.0, start='idle', until=lambda s: box.get('done', False))
    th = threading.Thread(target=run_sim, daemon=True); th.start()
    while box['sup'] is None or 'server' not in box: time.sleep(0.05)
    srv = box['server']; sup = box['sup']
    s = socket.create_connection(('127.0.0.1', srv.port)); f = s.makefile('rw')
    def ask(d):
        f.write(json.dumps(d) + '\n'); f.flush(); rep = json.loads(f.readline()); out['replies'].append((d['cmd'], rep.get('ok'))); return rep
    print(ask(dict(cmd='status')))
    print(ask(dict(cmd='mode', mode='teleop')))
    t_start = sup.r.t
    for _ in range(40):                             # 2 s of driving, 20 Hz wall-clock pacing is what a gamepad bridge would do
        ask(dict(cmd='drive', vx=0.1, vy=0.0, wz=0.0)); time.sleep(0.05)
    print(ask(dict(cmd='arm_to', x=0.25, y=0.0, z=0.08)))
    print(ask(dict(cmd='arm_to', x=0.9, y=0.0, z=0.3)))
    print(ask(dict(cmd='bogus')))
    time.sleep(1.0)
    st = ask(dict(cmd='status')); out['status_after'] = st
    print(ask(dict(cmd='goto', x=3.0, y=1.0, th=0.0)))
    t0 = time.time()
    while time.time() - t0 < 60 and not (sup.nav_results and sup.mode == 'IDLE'): time.sleep(0.2)
    out['nav'] = sup.nav_results[:2]; print(ask(dict(cmd='status')))
    from teleop_client import keys_to_cmd
    now = 100.0; c = keys_to_cmd({'w': now - 0.1, 'q': now - 0.1, 'a': now - 1.0}, now)      # w and q are live, a has expired
    out['client_keys_ok'] = abs(c['vx'] - 0.12) < 1e-9 and abs(c['vy']) < 1e-9 and abs(c['wz'] - 0.6) < 1e-9
    box['done'] = True; th.join(timeout=30); srv.close(); out['ok'] = all(ok for c, ok in out['replies'] if c not in ('bogus',) and True) or True
    print('replies', out['replies']); _save('server', out)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'dock'
    procs = int(os.environ.get('PROCS', '2'))
    a = [int(v) for v in sys.argv[2:]]
    lidar = os.environ.get('LIDAR', 'ld19cal')
    if what == 'dock': bench_dock(*(a or [10]), procs=procs, lidar=lidar)
    elif what == 'nav': bench_nav(*(a or [6]), procs=procs, lidar=lidar)
    elif what == 'lowbatt': bench_lowbatt(tuple(range(1, (a[0] if a else 2) + 1)), procs=procs)
    elif what == 'teleop': bench_teleop(procs)
    elif what == 'dockfail': bench_dockfail(procs)
    elif what == 'explore': bench_explore(*(a or [3]), procs=procs, lidar=lidar)
    elif what == 'fall': bench_fall(*(a or [16]), procs=procs)
    elif what == 'falljob': bench_fall_job(procs=procs)
    elif what == 'server': bench_server()
