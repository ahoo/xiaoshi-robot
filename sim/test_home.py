"""Benches for the home features: 3D scan, object records, people and children, voice.

  python test_home.py scan [seeds]        3D scan of the empty furnished room: next-best-view planner against the coverage tour, both against the true geometry
  python test_home.py export [seed]       one scan with the toys on the floor: point cloud (PLY), stand-alone viewer page (HTML), numbers against the truth
  python test_home.py inv1 [seeds]        object records, one session: found, kind guessed (standard layouts and the hard-material ones)
  python test_home.py inventory [seeds]   object records across two sessions: things moved, taken away, new; identity of each
  python test_home.py people_static [n]   people: found, how tall, child or adult, from a standing robot (150 trials), false tracks in an empty room (CAM=forearm|periscope: camera on the arm)
  python test_home.py pairs               a child standing close to a grown-up: are they told apart, is the child ever called an adult
  python test_home.py identity            the registry and the thresholds of the (stand-in) recogniser: consent, enrolment, confusions, forgetting
  python test_home.py safety [seeds]      people and the robot at work: an adult walks up, a child runs at it, a toddler by the basket, a family; guard on and off
  python test_home.py voice               Chinese commands through the simulated recogniser: noise sweep, chatter without the wake word, stop words, permissions, dialogues
  python test_home.py day [seed]          one scripted day with everything: voice, scan, records, a child enrolled by a grown-up, stop and continue, forgetting
Each writes results/home_<name>.json (export also writes the PLY and HTML). Set PROCS for the number of processes and MUJOCO_GL=egl for headless rendering.
Not done: a scan of more than one room (the two-room test planned for round 14 was not built).
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def _save(name, data):
    """results/home_<name>.json; when the colour-image person detector is on (PEOPLE_RGB) the name gets `_rgb` (and PEOPLE_RGB_TAG if given), so that it never overwrites the depth-only run."""
    rgb = os.environ.get('PEOPLE_RGB', '')
    if rgb and rgb not in ('0', 'off'): name += '_rgb' + (('_' + os.environ['PEOPLE_RGB_TAG']) if os.environ.get('PEOPLE_RGB_TAG') else '')
    os.makedirs(RES, exist_ok=True); json.dump(data, open(os.path.join(RES, f'home_{name}.json'), 'w'), ensure_ascii=False, indent=1, default=float)


# --------------------------------------------------------------------------------------------- 3D scan
def _true_visible(robot, step=0.30, zmax=0.6):
    """True surface points (up to zmax) that could be seen from some reachable stop at all: the fixed yardstick both planners are measured against.
    A point counts if a ray from the camera height over some free position reaches it unobstructed and within the camera's vertical field of view."""
    import mapping3d as M
    P, gid = M.true_surface_points(robot.m, robot.d, robot.chassis, zmax=zmax, spacing=0.03)
    from model import ROOM_W, ROOM_H
    import mujoco
    # free positions: not within 0.26 m of a true obstacle (the robot centre can be there)
    occ_pts = P[(P[:, 2] > 0.05) & (P[:, 2] < 0.3)][:, :2]
    from scipy.spatial import cKDTree
    t = cKDTree(occ_pts); xs = np.arange(0.3, ROOM_W - 0.2, step); ys = np.arange(0.3, ROOM_H - 0.2, step)
    X, Y = np.meshgrid(xs, ys); C = np.stack([X.ravel(), Y.ravel()], 1); ok = t.query(C)[0] > 0.26
    seen = np.zeros(len(P), bool); h = 0.126 + 0.05
    for x, y in C[ok]:
        o = np.array([x, y, h]); d = P - o; L = np.linalg.norm(d, axis=1); el = np.degrees(np.arcsin(d[:, 2] / np.maximum(L, 1e-6)))
        cand = (~seen) & (L > 0.15) & (L < M.MAX_RANGE) & (el > -62) & (el < 12)
        idx = np.nonzero(cand)[0]
        for s in range(0, len(idx), 4000):
            sub = idx[s:s + 4000]; vec = (d[sub] / L[sub, None]).reshape(-1); gid_ = np.zeros(len(sub), np.int32); dist = np.zeros(len(sub))
            mujoco.mj_multiRay(robot.m, robot.d, o, vec, np.array([1, 0, 0, 0, 0, 0], np.uint8), 1, -1, gid_, dist, None, len(sub), M.MAX_RANGE)
            seen[sub[(dist > 0) & (dist >= L[sub] - 0.04)]] = True
    return P, seen


def _scan_one(args):
    import mapping3d as M
    import perturb
    from controller import Agent
    from home import Home
    from modes import Supervisor
    from robot import Robot
    planner, seed, lidar = args
    perturb.CUR = dict(lidar_model=lidar)
    robot = Robot([], seed=seed, robot_pose=(0.45, 2.55, -math.pi / 2)); agent = Agent(robot, seed=seed)
    home = Home(robot, agent); sup = Supervisor(robot, agent); sup.home = home
    t0 = time.time(); sup.scan_home(planner=planner)
    sup.run(900.0, until=lambda s: s.scan_done or (s.mode == 'IDLE' and s.r.t > 5))
    info = home.finish_scan()
    P, vis = _true_visible(robot)
    ev = M.evaluate(home.map, P, vis)
    # per-planner time series (observed volume against time), coverage of the yardstick at the end
    out = dict(planner=planner, seed=seed, sim_s=round(robot.t, 1), wall_s=round(time.time() - t0), stops=len(home.scan_log), path_m=round(agent.stats['dist'], 1), frames=home.scan.frames,
               observed=round(home.map.observed_fraction(0.55), 3), recall=round(ev['recall'], 3), precision=round(ev['precision'], 3), n_true_visible=ev['n_true'], n_map=ev['n_map'],
               furniture=len(info['furniture']), walls=len(info['walls']), collisions=agent.stats['collisions'], stuck=agent.stats['stuck'], log=home.scan_log,
               ms_per_frame=round(home.scan.t_ms / max(home.scan.frames, 1)))
    out['furniture_boxes'] = [dict(centre=[round(v, 2) for v in f['centre']], length=round(f['length'], 2), width=round(f['width'], 2), height=round(f['height_seen'], 2)) for f in info['furniture']]
    return out, home


def bench_scan(seeds=(1,), procs=2, lidar='ld19cal'):
    jobs = [(pl, s, lidar) for s in seeds for pl in ('tour', 'nbv')]
    with mp.Pool(procs, maxtasksperchild=1) as p: res = p.map(_scan_job, jobs)
    for r in res: print({k: v for k, v in r.items() if k not in ('log', 'furniture_boxes')})
    _save('scan', dict(lidar=lidar, runs=res))


def _scan_job(args):
    out, home = _scan_one(args); return out


# --------------------------------------------------------------------------------------------- object records
def _gt_objects(robot):
    return [(robot.objects[i][0], robot.obj_pos(i)) for i in range(len(robot.objects))]


def _run_session(objects, boxes, seed, inv_path, lidar, t_end=170.0, perturb_fn=None):
    """One session: the robot explores (picks nothing) with the object records on. Returns the Home, the robot and the agent."""
    import perturb
    from controller import Agent
    from home import Home
    from model import HOME
    from modes import Supervisor
    from robot import Robot
    perturb.CUR = dict(lidar_model=lidar)
    robot = Robot(objects, seed=seed, extra_boxes=boxes, robot_pose=HOME); agent = Agent(robot, seed=seed)
    home = Home(robot, agent, inv_path=inv_path); sup = Supervisor(robot, agent); sup.home = home
    sup.explore(); sup.run(t_end, until=lambda s: bool(s.nav_results))
    home.finish_scan(); home.inv.end_session()
    return home, robot, agent


def _match_gt(home, robot, tol=0.10):
    """For each true object: the item the records hold for it (the nearest item within tol of where it lies), or None."""
    out = []
    for kind, p in _gt_objects(robot):
        best = None
        for it in home.inv.items.values():
            d = math.hypot(it['pos'][0] - p[0], it['pos'][1] - p[1])
            if d < tol and (best is None or d < best[0]): best = (d, it)
        out.append((kind, p, None if best is None else best[1]))
    return out


def _session_one(args):
    """One session on a fresh layout: how complete and how right are the records?"""
    import objects as O
    from episode import make_layout
    seed, lidar, materials = args
    objects, boxes = make_layout(seed, materials=materials)
    path = f'/tmp/inv_{seed}_s1{"m" if materials else ""}.json'
    if os.path.exists(path): os.remove(path)
    home, robot, agent = _run_session(objects, boxes, seed, path, lidar)
    m = _match_gt(home, robot); found = [(k, it) for k, p, it in m if it is not None]
    kind_ok = [it['kind'] == k for k, it in found]
    near = lambda it: any(math.hypot(it['pos'][0] - p[0], it['pos'][1] - p[1]) < 0.10 for k, p in _gt_objects(robot))
    extra = [dict(id=i, kind=it['kind'], pos=it['pos'], status=it['status']) for i, it in home.inv.items.items() if not near(it)]
    return dict(seed=seed, materials=materials, objects=len(objects), found=len(found), items=len(home.inv.items), duplicates_or_ghosts=extra, kind_correct=int(sum(kind_ok)),
                kinds=[(k, it['kind'], it['kind_conf'], objects[i][4].get('mat', 'matte')) for i, (k, p, it) in enumerate(m) if it is not None],
                missed=[(k, objects[i][4].get('mat', 'matte')) for i, (k, p, it) in enumerate(m) if it is None], coloured=sum(it['colour']['n'] > 0 for k, it in found),
                t=round(robot.t, 1), collisions=agent.stats['collisions'], events=home.inv.events[:20])


def bench_inventory(seeds=(1, 2, 3, 4, 5, 6), procs=2, lidar='ld19cal', material_seeds=(1, 2, 3, 4)):
    jobs = [(s, lidar, False) for s in seeds] + [(s, lidar, True) for s in material_seeds]
    with mp.Pool(procs, maxtasksperchild=1) as p: res = p.map(_session_one, jobs)
    for r in res: print({k: v for k, v in r.items() if k not in ('events',)})
    std = [r for r in res if not r['materials']]; mat = [r for r in res if r['materials']]
    summ = dict(standard=dict(runs=len(std), objects=sum(r['objects'] for r in std), found=sum(r['found'] for r in std), kind_correct=sum(r['kind_correct'] for r in std), extra_items=sum(len(r['duplicates_or_ghosts']) for r in std)),
                hard_materials=dict(runs=len(mat), objects=sum(r['objects'] for r in mat), found=sum(r['found'] for r in mat), kind_correct=sum(r['kind_correct'] for r in mat), extra_items=sum(len(r['duplicates_or_ghosts']) for r in mat)))
    print(summ); _save('inventory_s1', dict(runs=res, summary=summ))
def _free_spot(rng, objs, boxes, clearance=0.35, avoid=None, min_move=0.0):
    """A random floor spot that the layout generator would also accept (clear of furniture, cartons, basket, dock, other objects)."""
    from episode import _furn_dist
    from model import BASKET, HOME, ROOM_H, ROOM_W
    for _ in range(800):
        x, y = rng.uniform(0.2, ROOM_W - 0.2), rng.uniform(0.2, ROOM_H - 0.2)
        if _furn_dist(x, y) < clearance: continue
        if any(math.hypot(max(abs(x - b[0]) - 0.15, 0), max(abs(y - b[1]) - 0.125, 0)) < clearance for b in boxes): continue
        if math.hypot(x - HOME[0], y - HOME[1]) < 0.5 or math.hypot(x - BASKET['x'], y - BASKET['y']) < 0.6: continue
        if any(math.hypot(x - o[1], y - o[2]) < 0.45 for o in objs): continue
        if avoid is not None and math.hypot(x - avoid[0], y - avoid[1]) < min_move: continue
        return round(x, 3), round(y, 3)
    raise RuntimeError('no free spot')


def _two_sessions(args):
    """Session 1 on a layout, then the world changes (two things moved, one taken away, one new), session 2 with the records of session 1.
    Scored on identity: the moved things must be the same items, the new thing a new item, the taken one missing, the untouched ones unchanged."""
    import objects as O
    from episode import make_layout
    seed, lidar = args
    objects, boxes = make_layout(seed)
    path = f'/tmp/inv_{seed}_ms.json'
    if os.path.exists(path): os.remove(path)
    h1, r1, a1 = _run_session(objects, boxes, seed, path, lidar)
    m1 = _match_gt(h1, r1); ids1 = [None if it is None else it['id'] for k, p, it in m1]
    inv1 = {i: dict(kind=it['kind'], pos=list(it['pos']), status=it['status']) for i, it in h1.inv.items.items()}
    # --- the world changes
    rng = np.random.default_rng(seed + 4242)
    objs2 = [list(o) for o in objects]; role = ['kept'] * len(objs2)
    pick = list(rng.permutation(len(objs2)))
    moved, removed = pick[:2], pick[2]
    for i in moved:
        others = [o for j, o in enumerate(objs2) if j != i]
        x, y = _free_spot(rng, others, boxes, avoid=(objs2[i][1], objs2[i][2]), min_move=0.8)
        objs2[i][1], objs2[i][2] = x, y; objs2[i][3] = round(float(rng.uniform(-math.pi, math.pi)), 3); role[i] = 'moved'
    role[removed] = 'removed'
    free_kinds = [k for k in O.KINDS if k not in [o[0] for o in objects]]; kind = free_kinds[int(rng.integers(len(free_kinds)))]
    x, y = _free_spot(rng, objs2, boxes); spec = O.sample(kind, rng)
    objs2 = [tuple(o) for j, o in enumerate(objs2) if j != removed] + [(kind, x, y, round(float(rng.uniform(-math.pi, math.pi)), 3), spec)]
    role = [r for j, r in enumerate(role) if j != removed] + ['new']
    exp_ids = [ids1[j] for j in range(len(objects)) if j != removed] + [None]
    h2, r2, a2 = _run_session(objs2, boxes, seed + 1000, path, lidar)
    inv = h2.inv; m2 = _match_gt(h2, r2); out = []
    for (k, p, it), rl, eid in zip(m2, role, exp_ids):
        row = dict(kind=k, role=rl, expected_id=eid, got_id=None if it is None else it['id'], got_kind=None if it is None else it['kind'], status=None if it is None else it['status'])
        row['ok'] = (it is not None and it['status'] == 'on_floor' and ((rl in ('kept', 'moved') and it['id'] == eid) or (rl == 'new' and it['id'] not in inv1)))
        if rl == 'moved' and it is not None: row['moved_flag'] = it.get('moved_session', 0) == inv.session
        out.append(row)
    rem = inv.items.get(ids1[removed]) if ids1[removed] is not None else None
    removed_row = dict(kind=objects[removed][0], role='removed', expected_id=ids1[removed], status=None if rem is None else rem['status'], ok=rem is not None and rem['status'] == 'missing',
                       unchecked=rem is not None and rem['status'] == 'on_floor' and not rem.get('seen_this_session'))
    pos = [(p[0], p[1]) for k, p, it in m2]
    ghosts = [dict(id=i, kind=it['kind'], pos=it['pos'], status=it['status'], size=it['size_cm']) for i, it in inv.items.items()
              if it['status'] == 'on_floor' and it.get('seen_this_session') and not any(math.hypot(it['pos'][0] - px, it['pos'][1] - py) < 0.10 for px, py in pos)]
    return dict(seed=seed, session1=dict(found=sum(i is not None for i in ids1), items=len(inv1)), rows=out + [removed_row], ghosts=ghosts, events=inv.events, new_ids=sorted(i for i in inv.items if i not in inv1),
                questions=dict(new=[inv.label(it) for it in inv.new_since(1)], moved=[inv.label(it) for it in inv.moved_since(1)], missing=[inv.label(it) for it in inv.missing()]),
                t=(round(r1.t), round(r2.t)), collisions=a1.stats['collisions'] + a2.stats['collisions'])


def bench_inventory_sessions(seeds=(1, 2, 3), procs=2, lidar='ld19cal'):
    with mp.Pool(procs, maxtasksperchild=1) as p: res = p.map(_two_sessions, [(s, lidar) for s in seeds])
    tot = dict(kept=[0, 0], moved=[0, 0], new=[0, 0], removed=[0, 0]); dup = 0; ghost = 0; unchecked = 0
    for r in res:
        for row in r['rows']:
            tot[row['role']][1] += 1; tot[row['role']][0] += int(row['ok'])
        unchecked += sum(1 for w in r['rows'] if w.get('unchecked'))
        dup += sum(1 for w in r['rows'] if w['role'] in ('kept', 'moved', 'new') and w.get('got_id') is not None and w['status'] == 'on_floor' and not w['ok']); ghost += len(r['ghosts'])
        print(r['seed'], 'ghosts', len(r['ghosts']), [(w['role'], w['kind'], w['ok']) for w in r['rows']], r['questions'], 'collisions', r['collisions'])
    print('identity right / total:', tot, 'wrong-identity:', dup, 'ghost items on the floor:', ghost, 'removed but not re-checked:', unchecked)
    _save('inventory', dict(runs=res, totals=tot, wrong_identity=dup, ghosts=ghost, removed_unchecked=unchecked))


# --------------------------------------------------------------------------------------------- people: detection, height, size class
class _Stub:
    """Stands in for the agent where only the robot's pose is needed (the true pose: the detector's own error is what is measured here)."""
    freeze = False
    def __init__(self, r): self.r = r
    def pose(self): return self.r.true_pose()


def _teleport(r, x, y, yaw):
    import mujoco
    adr = int(r.m.jnt_qposadr[mujoco.mj_name2id(r.m, mujoco.mjtObj.mjOBJ_JOINT, 'base')])
    r.d.qpos[adr:adr + 2] = [x, y]; r.d.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]; r.d.qvel[:6] = 0; r.d.qacc[:] = 0
    mujoco.mj_forward(r.m, r.d)


TRACK_OF_PERSON = 0.30          # m: a track this close to the person is the person (position error of the 400+ people found in the benches: median 8 cm, largest 17 cm)


def _static_job(args):
    """Robot standing still, one person placed at a random spot 1-3 m ahead inside its field of view: is it found, how tall, which size class."""
    import mujoco
    import people as P
    from episode import _furn_dist
    from robot import Robot
    seed, n_trials = args[:2]; cam = args[2] if len(args) > 2 else 'mast'                  # 'mast' (round 14), or a camera on the arm: 'forearm', 'periscope' (model.ARM_CAMS)
    rng = np.random.default_rng(seed); names = list(P.SPECS)
    r = Robot([], seed=seed, people=names, robot_pose=(2.0, 1.5, 0.0), **(dict(head_cam=True) if cam == 'mast' else dict(arm_cam=cam))); stub = _Stub(r)
    scripts = [P.Script([]) for _ in names]; ps = P.People(r, names, scripts); det = P.PersonDetector(r, stub, np.random.default_rng(seed + 3), boxes=P.make_boxes(r, names, np.random.default_rng(seed + 29)))
    look = None if cam == 'mast' else __import__('model').ARM_CAMS[cam]['look']
    if look is not None:                                                                       # the periscope: raise the arm and let it settle before anyone is looked at
        r.set_arm(q5=np.array(look))
        for _ in range(600):
            ps.update(); r.step()
            if r.arm_done(0.02): break
        for _ in range(50): ps.update(); r.step()
    rows = []; fp = []
    def place(i, x, y, yaw, posture):
        scripts[i] = P.Script([('at', x, y, yaw)] + ([('posture', 'sit', 0.01)] if posture == 'sit' else [])); ps.scripts = scripts
    def run(frames=10):
        det.tracks = []; det._t = -9.0
        for _ in range(frames * 10): ps.update(); r.step(); det.process()
    for trial in range(n_trials):
        i = trial % len(names); posture = 'sit' if (trial // len(names)) % 3 == 2 else 'stand'
        while True:
            rx, ry, yaw = rng.uniform(0.4, 3.6), rng.uniform(0.4, 2.6), rng.uniform(-math.pi, math.pi)
            if _furn_dist(rx, ry) < 0.3: continue
            d, b = rng.uniform(1.0, 3.0), math.radians(rng.uniform(-28, 28)); px, py = rx + d * math.cos(yaw + b), ry + d * math.sin(yaw + b)
            if not (0.3 < px < 3.7 and 0.3 < py < 2.7) or _furn_dist(px, py) < 0.3: continue
            break
        face = rng.random() < 0.5; pyaw = math.atan2(ry - py, rx - px) if face else float(rng.uniform(-math.pi, math.pi))
        for j in range(len(names)): scripts[j] = P.Script([]) if j != i else scripts[j]
        place(i, px, py, pyaw, posture); _teleport(r, rx, ry, yaw); mujoco.mj_forward(r.m, r.d); ps.scripts = scripts
        run(10)
        s = ps.truth()[i]; got = [k for k in det.confirmed(2) if math.hypot(k.x - px, k.y - py) < TRACK_OF_PERSON]          # (people who are found are found within 0.17 m; a cupboard next to somebody the camera does not see is not them)
        row = dict(person=names[i], posture=posture, dist=round(d, 2), bearing=round(math.degrees(b)), facing=face, truth_cls=s['cls'], truth_h=round(s['h'], 3), found=bool(got))
        if got:
            k = min(got, key=lambda k: math.hypot(k.x - px, k.y - py)); h, lb = k.height
            row.update(pos_err=round(math.hypot(k.x - px, k.y - py), 3), h=round(h, 3), lower_bound=lb, w=None if k.width is None else round(k.width, 3), cls=k.cls, true_w=s['w'])
        extra = [k for k in det.confirmed(2) if math.hypot(k.x - px, k.y - py) >= TRACK_OF_PERSON]
        row['extra_tracks'] = len(extra); rows.append(row)
        scripts[i] = P.Script([])
    # no one in the room: anything found is false
    ps.scripts = [P.Script([]) for _ in names]
    for trial in range(max(n_trials // 3, 10)):
        while True:
            rx, ry, yaw = rng.uniform(0.4, 3.6), rng.uniform(0.4, 2.6), rng.uniform(-math.pi, math.pi)
            if _furn_dist(rx, ry) >= 0.3: break
        _teleport(r, rx, ry, yaw); run(10); fp.append(len(det.confirmed(2)))
    return rows, fp, det.t_ms / max(det.frames, 1)


def bench_people_static(procs=2, trials=150, cam='mast'):
    per = trials // procs
    seed0 = int(os.environ.get('STATIC_SEED0', 100))                                                 # other seeds: to see how much the numbers move
    with mp.Pool(procs, maxtasksperchild=1) as p: out = p.map(_static_job, [(seed0 + i, per, cam) for i in range(procs)])
    rows = [r for o in out for r in o[0]]; fp = [v for o in out for v in o[1]]; ms = float(np.mean([o[2] for o in out]))
    def stat(sel):
        a = [r for r in rows if sel(r)]
        if not a: return None
        f = [r for r in a if r['found']]; nb = [r for r in f if not r['lower_bound']]
        he = [abs(r['h'] - r['truth_h']) for r in nb]
        return dict(n=len(a), found=round(len(f) / len(a), 3), pos_err_median=round(float(np.median([r['pos_err'] for r in f])), 3) if f else None,
                    h_err_median=round(float(np.median(he)), 3) if he else None, h_err_p90=round(float(np.percentile(he, 90)), 3) if he else None, lower_bound_share=round(1 - len(nb) / max(len(f), 1), 3))
    res = dict(by_distance={f'{a}-{b} m': stat(lambda r, a=a, b=b: a <= r['dist'] < b and r['posture'] == 'stand') for a, b in ((1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.01))},
               by_person={n: stat(lambda r, n=n: r['person'] == n and r['posture'] == 'stand') for n in sorted({r['person'] for r in rows})}, sitting=stat(lambda r: r['posture'] == 'sit'),
               false_tracks_empty_room=dict(frames=len(fp), with_a_track=int(sum(1 for v in fp if v > 0))), extra_tracks_with_a_person=int(sum(r['extra_tracks'] for r in rows)), ms_per_frame=round(ms))
    conf = {}
    for posture in ('stand', 'sit'):
        for tc in ('child', 'adult'):
            sel = [r for r in rows if r['posture'] == posture and r['truth_cls'] == tc and r['found']]
            conf[f'{posture}/{tc}'] = {c: sum(1 for r in sel if r['cls'] == c) for c in ('child', 'unsure', 'adult')}
    res['size_class'] = conf
    res['child_called_adult'] = int(sum(1 for r in rows if r['found'] and r['truth_cls'] == 'child' and r['cls'] == 'adult'))
    print(json.dumps(res, ensure_ascii=False, indent=1)); _save('people_static' if cam == 'mast' else f'people_static_{cam}', dict(summary=res, rows=rows, camera=cam))

def _pair_job(args):
    """A child and a grown-up standing close together, at different gaps and in two arrangements (side by side, one in front of the other): are both found, and what is the pair called?"""
    import mujoco
    import people as P
    from episode import _furn_dist
    from robot import Robot
    seed, reps = args
    rng = np.random.default_rng(seed); names = ['adult_m', 'child_8']
    r = Robot([], seed=seed, people=names, head_cam=True, robot_pose=(2.0, 1.5, 0.0)); stub = _Stub(r)
    scripts = [P.Script([]) for _ in names]; ps = P.People(r, names, scripts); det = P.PersonDetector(r, stub, np.random.default_rng(seed + 3), boxes=P.make_boxes(r, names, np.random.default_rng(seed + 29))); rows = []
    for gap in (0.10, 0.25, 0.45, 0.70):
        for arr in ('side', 'inline'):
            for rep in range(reps):
                for _try in range(3000):
                    rx, ry, yaw = rng.uniform(0.9, 3.0), rng.uniform(0.7, 2.3), rng.uniform(-math.pi, math.pi); d = rng.uniform(1.6, 2.6)
                    ux, uy = math.cos(yaw), math.sin(yaw); px, py = rx + d * ux, ry + d * uy                      # the adult, straight ahead
                    off = (0.23 + 0.16 + gap)                                                                       # shoulder to shoulder with the stated gap between the bodies
                    cx, cy = (px - uy * off, py + ux * off) if arr == 'side' else (px - ux * (0.14 + 0.14 + gap), py - uy * (0.14 + 0.14 + gap))
                    if 0.3 < px < 3.7 and 0.3 < py < 2.7 and 0.3 < cx < 3.7 and 0.3 < cy < 2.7 and min(_furn_dist(x, y) for x, y in ((rx, ry), (px, py), (cx, cy))) >= 0.3: break      # nobody on or inside the furniture, the robot neither
                else: continue
                scripts[0] = P.Script([('at', px, py, yaw + math.pi)]); scripts[1] = P.Script([('at', cx, cy, yaw + math.pi)]); ps.scripts = scripts
                ps.update(); mujoco.mj_forward(r.m, r.d)                                                    # the two people first: put down where they stood last time, the robot would be shoved by them (and look crooked)
                _teleport(r, rx, ry, yaw); mujoco.mj_forward(r.m, r.d); det.tracks = []; det._t = -9.0
                tilt = 0.0
                for _ in range(100): ps.update(); r.step(); det.process(); tilt = max(tilt, float(r.tilt()))
                tr = det.confirmed(2); dA = lambda k: math.hypot(k.x - px, k.y - py); dC = lambda k: math.hypot(k.x - cx, k.y - cy)
                ka = sorted([k for k in tr if dA(k) < 0.6 and dA(k) <= dC(k)], key=dA); kc = sorted([k for k in tr if dC(k) < 0.6 and dC(k) < dA(k)], key=dC)      # the nearest one is theirs                       # a track belongs to whichever of the two it is nearer to, if within 0.6 m (with a 10 cm gap the two are 0.49 m apart)
                rows.append(dict(gap=gap, arrangement=arr, tracks=len(tr), adult_found=bool(ka), child_found=bool(kc), adult_cls=ka[0].cls if ka else None, child_cls=kc[0].cls if kc else None,
                                 pair_in_one=bool(len(tr) == 1), tipped=bool(tilt > 0.2)))
    return rows


def bench_pairs(procs=2, reps=8):
    seed0 = int(os.environ.get('PAIRS_SEED0', 300))                                                  # other seeds: to see how much the numbers move
    with mp.Pool(procs, maxtasksperchild=1) as p: out = p.map(_pair_job, [(seed0 + i, reps) for i in range(procs)])
    rows = [r for o in out for r in o]; res = {}
    for gap in (0.10, 0.25, 0.45, 0.70):
        for arr in ('side', 'inline'):
            a = [r for r in rows if r['gap'] == gap and r['arrangement'] == arr]
            if not a: continue
            res[f'{arr} gap {int(gap * 100)} cm'] = dict(n=len(a), two_tracks=sum(1 for r in a if r['tracks'] >= 2 and r['adult_found'] and r['child_found']), merged_into_one=sum(1 for r in a if r['pair_in_one']),
                                                         child_called_child=sum(1 for r in a if r['child_cls'] in ('child', 'unsure')), child_called_adult_or_missing=sum(1 for r in a if r['child_cls'] not in ('child', 'unsure')),
                                                         pair_called=dict(sorted({c: sum(1 for r in a if r['pair_in_one'] and r['adult_cls'] == c) for c in ('adult', 'unsure', 'child')}.items())))
    res['robot_tipped_trials'] = int(sum(1 for r in rows if r.get('tipped')))                    # a sanity check on the bench itself: the robot must stand still and upright
    print(json.dumps(res, ensure_ascii=False, indent=1)); _save('people_pairs', dict(summary=res, rows=rows))



# --------------------------------------------------------------------------------------------- people near the robot at work
def _scenario(name, seed):
    """People that walk through the room while the robot does its job. -> people dict for episode.run_supervised. They come in by the door farthest from the robot, step round it, leave by a door."""
    if name == 'none': return None
    if name == 'adult_cross':      # a grown-up walks up to the robot's working area twice and leaves
        return dict(names=['adult_m'], who=['dad'], family=['A'], comply=[1.0],
                    steps=[[('wait', 20.0), ('enter',), ('to_robot', 0.5, 1.0), ('exit', 1.0), ('wait', 35.0), ('enter',), ('to_robot', 0.5, 1.0), ('exit', 1.0)]])
    if name == 'child_run':        # a 5-year-old runs straight at the robot, stops close, stays a while, goes; and once more
        return dict(names=['child_5'], who=['kid'], family=['A'], comply=[0.7],
                    steps=[[('wait', 25.0), ('enter',), ('chase', 1.4, 0.12, 15.0), ('face_robot',), ('wait', 6.0), ('exit', 0.9), ('wait', 30.0), ('enter',), ('chase', 1.2, 0.20, 15.0), ('wait', 4.0), ('exit', 0.9)]])
    if name == 'toddler_basket':   # a toddler plays next to the basket for most of the job, crawling about slowly
        return dict(names=['toddler'], who=['baby'], family=['A'], comply=[0.0],
                    steps=[[('wait', 15.0), ('enter',), ('walk', 3.2, 2.35, 0.4), ('wait', 8.0), ('walk', 3.05, 2.45, 0.3), ('wait', 6.0), ('walk', 3.3, 2.65, 0.3), ('wait', 40.0), ('exit', 0.4)]])
    if name == 'family':           # three people moving about at once
        return dict(names=['adult_f', 'child_8', 'toddler'], who=['mum', 'ming', 'baby'], family=['A', 'A', 'A'], comply=[1.0, 0.7, 0.0],
                    steps=[[('wait', 10.0), ('enter',), ('walk', 2.0, 1.1, 0.8), ('wait', 10.0), ('walk', 3.6, 1.5, 0.8), ('wait', 20.0), ('walk', 1.0, 2.6, 0.8), ('wait', 20.0), ('walk', 3.6, 0.4, 0.8), ('exit', 0.8)],
                           [('wait', 20.0), ('enter',), ('walk', 2.2, 1.9, 1.1), ('walk', 1.0, 1.0, 1.1), ('wait', 5.0), ('walk', 3.0, 2.4, 1.4), ('wait', 5.0), ('walk', 1.5, 2.9, 1.2), ('walk', 3.6, 1.9, 1.3), ('wait', 5.0), ('exit', 1.0)],
                           [('wait', 30.0), ('enter',), ('walk', 2.6, 1.7, 0.3), ('wait', 8.0), ('walk', 3.4, 2.2, 0.3), ('wait', 15.0), ('walk', 2.0, 2.0, 0.3), ('wait', 15.0), ('walk', 1.6, 1.0, 0.3), ('wait', 10.0), ('exit', 0.4)]])
    raise ValueError(name)


def _safety_job(args):
    from episode import run_supervised
    seed, scen, guard, t_end = args[:4]; cam = args[4] if len(args) > 4 else 'mast'
    pe = _scenario(scen, seed)
    if pe is not None: pe['guard'] = guard; pe['cam'] = cam
    t0 = time.time(); res, sup = run_supervised(seed, start='task', people=pe, t_end=t_end, perturbation=dict(lidar_model='ld19cal'))
    out = dict(scenario=scen, seed=seed, guard=guard, camera=cam, delivered=res['delivered'], total=res['total'], finished=res['finished'], sim_time=res['sim_time'], collisions=res['collisions'], stuck=res['stuck'],
               wall_s=round(time.time() - t0), mode_end=res['mode'])
    pp = sup.ppl
    if pp is not None:
        g = pp.guard; ps = pp.people
        out.update(contacts=[int(v) for v in ps.contact_events], contact_s=[round(v, 2) for v in ps.contact_t], contact_speed=ps.contact_speed, min_gap_cm=[round(100 * v, 1) for v in ps.min_gap],
                   halts=None if g is None else g.stats['halts'], halt_s=None if g is None else round(g.stats['halt_s'], 1), slow_s=None if g is None else round(g.stats['slow_s'], 1),
                   arm_pauses=None if g is None else g.stats['arm_pauses'], false_track_share=round(pp.n_false / max(pp.n_look, 1), 3), cls_votes=pp.cls_votes, ms_per_frame=round(pp.det.t_ms / max(pp.det.frames, 1)),
                   cam_blind_share=round(pp.det.blind_s / max(pp.det.blind_s + pp.det.seen_s, 1e-9), 3), face_out=pp.n_face_out)
        out['events'] = [e for e in sup.events if '有人' in e[2]][:12]
    return out


def bench_people_safety(seeds=(1, 2), procs=2, scenarios=('none', 'adult_cross', 'child_run', 'toddler_basket', 'family'), t_end=420.0, cam='mast'):
    jobs = [(s, sc, g, t_end, cam) for s in seeds for sc in scenarios for g in ((True,) if sc == 'none' else (False, True))]
    with mp.Pool(procs, maxtasksperchild=1) as p: res = p.map(_safety_job, jobs)
    for r in res: print({k: v for k, v in r.items() if k not in ('events', 'cls_votes')})
    name = 'people_safety' if cam == 'mast' else f'people_safety_{cam}'
    fn = os.path.join(RES, f'home_{name}.json'); old = json.load(open(fn))['runs'] if os.path.exists(fn) else []
    keys = {(r['scenario'], r['seed'], r['guard']) for r in res}; _save(name, dict(runs=[r for r in old if (r['scenario'], r['seed'], r['guard']) not in keys] + res))


# --------------------------------------------------------------------------------------------- voice (language side, no robot)
def _slots_ok(got, want):
    return all((got or {}).get(k) == v for k, v in want.items())


def _outcome(vs, rec, intent, slots):
    act = rec.get('action') or ''
    if act.startswith('ignored'): return 'missed_wake'
    pend = vs.pending
    if act == 'reask': return 'confirmed' if (pend and pend.get('intent') == intent) else 'reask'
    if act == 'confirm' and pend and pend.get('intent') == intent: return 'confirmed'
    if rec.get('intent') == intent and _slots_ok(rec.get('slots'), slots): return 'ok'
    return 'wrong'


def _voice_sweep(args):
    from voice import MOTION, SENSITIVE, SimASR, VoiceSystem
    from voice_data import COMMANDS
    cer, child, seeds, n_sub = args
    cnt = dict(ok=0, confirmed=0, reask=0, wrong=0, wrong_motion=0, missed_wake=0, n=0); by_intent = {}
    for text, intent, slots in COMMANDS:
        if intent == 'stop': continue                      # the always-on stop words are measured on their own
        for sd in range(seeds):
            vs = VoiceSystem(asr=SimASR(cer, 1000 * sd + len(text))); vs.extra_persons = ['小明', '小红']
            wake = '小拾小拾，' if sd % 2 == 0 else '你好小拾，'
            rec = vs.hear(wake + text, role='child' if child else 'owner'); o = _outcome(vs, rec, intent, slots)
            if o == 'wrong' and rec.get('intent') in MOTION and rec.get('action') == rec.get('intent'): cnt['wrong_motion'] += 1      # the robot would have moved, and not as asked
            cnt[o] += 1; cnt['n'] += 1; b = by_intent.setdefault(intent, dict(n=0, ok=0)); b['n'] += 1; b['ok'] += int(o in ('ok', 'confirmed'))
    return dict(cer=cer, child=child, **cnt, by_intent=by_intent)


def _voice_noise_tests():
    from voice import NLU, SimASR, VoiceSystem
    from voice_data import CHATTER, COMMANDS
    out = {}
    # speech that is not for the robot, no wake word: nothing may be acted on
    acted = 0; n = 0; stops = 0
    for cer in (0.0, 0.10, 0.20):
        for text in CHATTER:
            for sd in range(8):
                vs = VoiceSystem(asr=SimASR(cer, sd * 37 + len(text))); rec = vs.hear(text, role='owner'); n += 1
                if not (rec.get('action') or '').startswith('ignored'): acted += 1
                if rec.get('action') == 'stop': stops += 1
    out['chatter_no_wake'] = dict(n=n, not_ignored=acted, false_stops=stops)
    # the always-on stop words, said without the wake word, by an adult and by a child, in noise
    st = {}
    for name, cer, child in (('adult_quiet', 0.04, False), ('adult_noisy', 0.15, False), ('child_quiet', 0.04, True), ('child_noisy', 0.15, True)):
        hit = 0; n = 0
        for w in ('停', '停下', '别动', '停下来', '暂停', '站住', '停止', '不要动'):
            for sd in range(25):
                vs = VoiceSystem(asr=SimASR(cer, sd * 13 + len(w))); rec = vs.hear(w, role='child' if child else 'owner', dist=2.5); n += 1; hit += int(rec.get('action') == 'stop')
        st[name] = round(hit / n, 3)
    out['stop_words_detected'] = st
    # a wake word said, then the rest: how often is the wake word caught
    wk = {}
    for cer in (0.0, 0.05, 0.10, 0.20):
        hit = 0; n = 0
        for sd in range(200):
            vs = VoiceSystem(asr=SimASR(cer, sd)); rec = vs.hear('小拾小拾，你好' if sd % 2 == 0 else '你好小拾，你是谁', role='owner'); n += 1; hit += int(not (rec.get('action') or '').startswith('ignored'))
        wk[str(cer)] = round(hit / n, 3)
    out['wake_word_caught'] = wk
    return out


def _voice_permissions():
    from voice import MIN_ROLE, ROLE_RANK, SimASR, VoiceSystem
    from voice_data import COMMANDS
    sample = {}
    for text, intent, slots in COMMANDS: sample.setdefault(intent, (text, slots))
    rows = []; bad = 0
    for intent, (text, slots) in sample.items():
        if intent in ('stop',): continue
        for role in ('owner', 'adult', 'child', 'unknown'):
            vs = VoiceSystem(asr=SimASR(0.0, 1)); vs.extra_persons = ['小明', '小红']; rec = vs.hear('小拾小拾，' + text, role=role)
            allowed = ROLE_RANK[role] >= ROLE_RANK[MIN_ROLE[intent]]; denied = rec.get('action') == 'denied'
            ok = (denied != allowed)
            if intent in ('cancel', 'resume') and False: ok = True
            rows.append(dict(intent=intent, role=role, expected_allowed=allowed, denied=denied, ok=ok)); bad += int(not ok)
    return dict(rows=rows, wrong=bad, n=len(rows))


def _voice_dialogues():
    """Small multi-turn scripts. Each returns (passed, what happened)."""
    from voice import MockLLM, SimASR, VoiceSystem
    res = []
    def new(**kw): return VoiceSystem(asr=SimASR(0.0, 1), **kw)
    def chk(name, ok, detail): res.append(dict(name=name, ok=bool(ok), detail=detail))
    # follow-up without the wake word within the window, not after it
    vs = new(); vs.hear('小拾小拾，你好', role='owner'); vs.advance(5.0); a = vs.hear('你是谁', role='owner'); vs.advance(20.0); b = vs.hear('你是谁', role='owner')
    chk('follow-up window', a.get('intent') == 'identity' and (b.get('action') or '').startswith('ignored'), (a.get('intent'), b.get('action')))
    # a missing slot is asked for, and the answer completes the command
    vs = new(); a = vs.hear('小拾小拾，帮我找一下', role='owner'); vs.advance(3.0); b = vs.hear('小球', role='owner')
    chk('asks for the missing item', a.get('action') == 'ask' and b.get('intent') == 'where_is' and vs.actions and vs.actions[-1][1].get('item') == 'ball', (a.get('reply'), b.get('action'), vs.actions[-1:]))
    # 'it' is the thing mentioned last
    vs = new(); vs.ctx['item'] = 7; a = vs.hear('小拾小拾，把它放进筐里', role='owner')
    chk('"it" = the last item', vs.actions and vs.actions[-1][0] == 'fetch_item' and vs.actions[-1][1].get('item') == 7, vs.actions[-1:])
    # unsure -> confirm -> yes executes / no cancels
    vs = new(); vs.asr = SimASR(0.0, 1); a = vs.hear('小拾小拾，去沙发那边', role='adult')
    from voice import NLU
    p = vs.nlu.parse('去沙发那边'); vs.pending = dict(kind='confirm', intent='go_to', slots=dict(place='沙发'), role='adult', until=vs.now() + 12); vs.advance(2.0); b = vs.hear('好的', role='adult')
    chk('confirm then yes', b.get('via') == 'confirmed' and vs.actions[-1][0] == 'go_to', (b.get('action'), vs.actions[-1:]))
    vs = new(); vs.pending = dict(kind='confirm', intent='go_to', slots=dict(place='沙发'), role='adult', until=vs.now() + 12); a = vs.hear('不是', role='adult')
    chk('confirm then no', a.get('action') == 'cancelled' and not vs.actions, (a.get('action'), vs.actions))
    # a child cannot confirm a grown-up's command
    vs = new(); vs.pending = dict(kind='confirm', intent='scan_home', slots={}, role='adult', until=vs.now() + 12); a = vs.hear('好的', role='child')
    chk('a child cannot confirm a scan', a.get('action') == 'denied' and not vs.actions, (a.get('action'), vs.actions))
    # privacy: mute, then deaf; the unmute is the button
    vs = new(); a = vs.hear('小拾小拾，别听了，静音', role='adult'); b = vs.hear('小拾小拾，你好', role='owner')
    chk('mute works and sticks', vs.muted and (b.get('action') or '').startswith('ignored'), (a.get('action'), b.get('action')))
    # forget: asks, then deletes
    vs = new(); vs.extra_persons = ['小明']; a = vs.hear('小拾小拾，忘掉小明', role='owner'); vs.advance(2.0); b = vs.hear('好的', role='owner')
    chk('forget asks and then forgets', a.get('action') == 'confirm' and b.get('action') == 'forgot' and ('forget', '小明') in vs.actions, (a.get('reply'), b.get('reply')))
    vs = new(); vs.extra_persons = ['小明']; a = vs.hear('小拾小拾，忘掉小明', role='child')
    chk('a child cannot make it forget', a.get('action') == 'denied', a.get('reply'))
    # 'go to the sofa' after the house has been scanned: a free spot is found even when the four standard ones (0.5 m from the middle of each side) are taken by the table and the shelf; when there really is none it says so,
    # and it only says 'scan the house first' when the house has not been scanned
    import numpy as np
    from episode import _furn_dist
    from model import BASKET, FURNITURE
    class _O: pass
    def stub(blocked=False, scanned=True):
        sup = _O(); sup.calls = []; sup.go_to = lambda x, y, **kw: sup.calls.append((round(x, 2), round(y, 2))); sup.r = _O(); sup.r.t = 0.0; sup.mode = 'IDLE'; sup.idle = lambda: None; sup.charge_now = lambda: None
        home = _O(); home.inv = None; home.a = _O(); home.a.basket = dict(BASKET); home.a.pose = lambda: (3.0, 2.0, 0.0)
        home.a.costmap = lambda: np.zeros((60, 80)) if blocked else np.array([[_furn_dist((i + 0.5) * 0.05, (j + 0.5) * 0.05) for i in range(80)] for j in range(60)])
        home.analysis = dict(furniture=[dict(centre=(f[1], f[2], 0.0), length=max(f[3], f[4] or f[3]), width=min(f[3], f[4] or f[3]), yaw=0.0 if (f[4] is None or f[3] >= f[4]) else math.pi / 2, height_seen=f[5]) for f in FURNITURE]) if scanned else None
        return sup, home
    sup, home = stub(); vs = VoiceSystem(sup, home, None, asr=SimASR(0.0, 1)); vs.hear('小拾小拾，过去沙发', role='owner')
    spot = sup.calls[0] if sup.calls else None; edge = None if spot is None else math.hypot(max(abs(spot[0] - 0.95) - 0.75, 0), max(abs(spot[1] - 0.4) - 0.3, 0))       # the sofa: centre (0.95, 0.4), 1.5 x 0.6 m
    ok_spot = spot is not None and 0.4 <= edge <= 0.75 and _furn_dist(*spot) >= 0.235
    chk('go to the sofa finds a spot when the four standard ones are taken', ok_spot, sup.calls)
    sup, home = stub(blocked=True); vs = VoiceSystem(sup, home, None, asr=SimASR(0.0, 1)); a = vs.hear('小拾小拾，过去沙发', role='owner')
    chk('no free spot at all: says so, does not say the house is unknown', not sup.calls and '找不到能站的位置' in (a.get('reply') or ''), a.get('reply'))
    sup, home = stub(scanned=False); vs = VoiceSystem(sup, home, None, asr=SimASR(0.0, 1)); a = vs.hear('小拾小拾，过去沙发', role='owner')
    chk('before a scan: asks for the scan', not sup.calls and '要先扫描' in (a.get('reply') or ''), a.get('reply'))
    # remembering / forgetting somebody and muting need the sound to be right, not nearly right: sentences that only sound a little like '记一下', '这是我的', '忘掉', '静音' are not those intents
    #   ('巡视一下' was heard as '记一下', the remember-a-person flow, instead of 'scan the house': found by checking the web playground against this code)
    near = [('巡视一下', 'scan_home'), ('去巡视一下家里', 'scan_home'), ('吃一下', None), ('试一下', None), ('这是谁的', None), ('那个球是我的', None)]       # (a word that sounds exactly the same, like 清楚 / 清除 or 路 / 录, cannot be told apart by sound and is not in this list)
    got = []
    for text, want in near:
        vs = new(); rec = vs.hear('小拾小拾，' + text, role='owner'); got.append((text, rec.get('intent'), rec.get('action')))
    bad = [g for g, (text, want) in zip(got, near) if (g[1] != want if want else g[1] in ('enrol', 'forget', 'privacy'))]
    chk('a near miss of the sound is not a sensitive intent', not bad, bad or got)
    vs = new(); ok_all = []
    for text, want in (('记一下我女儿', 'enrol'), ('记住这个孩子', 'enrol'), ('忘掉小明', 'forget'), ('静音', 'privacy'), ('你在录音吗', 'privacy')):
        vs = new(); vs.extra_persons = ['小明']; rec = vs.hear('小拾小拾，' + text, role='owner'); ok_all.append((text, rec.get('intent')) if rec.get('intent') != want else None)
    chk('the sensitive intents are still understood when said right', not any(ok_all), [x for x in ok_all if x])
    # language model: a proposed intent goes through the permission check and a confirmation; garbage and timeouts fall back
    llm = MockLLM({'乱糟糟': dict(intent='tidy'), '去客厅扫地图': dict(intent='scan_home'), '胡说': dict(intent='rm_rf'), '讲个故事': dict(say='从前有一只小机器人，它每天都在家里捡玩具。')})
    vs = new(llm=llm); a = vs.hear('小拾小拾，房间乱糟糟的', role='owner'); b = vs.hear('好的', role='owner')
    chk('LLM proposal still has to be confirmed', a.get('action') == 'confirm' and a.get('via') == 'llm' and b.get('via') == 'confirmed' and vs.actions[-1][0] == 'tidy', (a.get('action'), b.get('via'), vs.actions))
    vs = new(llm=llm); a = vs.hear('小拾小拾，去客厅扫地图', role='child')
    chk('LLM proposal is refused for a child', a.get('action') == 'denied', a.get('action'))
    vs = new(llm=llm); a = vs.hear('小拾小拾，胡说八道', role='owner')
    chk('LLM nonsense intent is refused', a.get('action') == 'reask', a.get('action'))
    vs = new(llm=MockLLM(fail=True)); a = vs.hear('小拾小拾，随便说点什么吧', role='owner')
    chk('LLM timeout falls back', a.get('action') == 'reask', a.get('action'))
    vs = new(llm=llm); a = vs.hear('小拾小拾，讲个故事', role='child')
    chk('LLM free text is not spoken to a child', a.get('action') == 'reask', a.get('action'))
    vs = new(llm=llm); a = vs.hear('小拾小拾，讲个故事', role='owner')
    chk('LLM free text is spoken to a grown-up', a.get('action') == 'llm reply', a.get('action'))
    return res


def bench_voice(procs=2):
    t0 = time.time(); res = {}
    jobs = [(cer, child, 6, 0) for cer in (0.0, 0.04, 0.08, 0.15, 0.25) for child in (False,)] + [(cer, True, 6, 0) for cer in (0.04, 0.08)]
    with mp.Pool(procs) as p: sweep = p.map(_voice_sweep, jobs)
    for r in sweep: print({k: v for k, v in r.items() if k != 'by_intent'})
    res['sweep'] = sweep; res['noise'] = _voice_noise_tests(); print(json.dumps(res['noise'], ensure_ascii=False))
    res['permissions'] = _voice_permissions(); print('permission matrix wrong:', res['permissions']['wrong'], 'of', res['permissions']['n'])
    res['dialogues'] = _voice_dialogues()
    for d in res['dialogues']: print('  ', 'PASS' if d['ok'] else 'FAIL', d['name'], '' if d['ok'] else d['detail'])
    from voice import NLU
    n = NLU(); t1 = time.time()
    for i in range(50): n.parse('把小球放进筐里')
    res['ms_per_parse'] = round(1000 * (time.time() - t1) / 50, 1); print('ms per parse', res['ms_per_parse'], 'total', round(time.time() - t0), 's')
    _save('voice', res)


# --------------------------------------------------------------------------------------------- people: identity (stand-in recogniser: tests our registry, not a face model)
def bench_identity(n=300):
    import people as P
    rng = np.random.default_rng(5)
    who = ['xiaoming', 'xiaohong', 'baba', 'lele', 'cousin']; family = ['A', 'A', 'A', 'B', 'A']; enrolled = ['xiaoming', 'xiaohong', 'baba']
    role = dict(xiaoming='child', xiaohong='child', baba='adult')
    path = '/tmp/registry_test.json'
    if os.path.exists(path): os.remove(path)
    face = P.FaceStandIn(who, family, seed=7); reg = P.Registry(path)
    out = dict(note='stand-in recogniser: feature vectors drawn per person (siblings share 55 % of theirs), noise grows with distance and when not facing the camera; tests the registry and the thresholds, not a face model')
    # consent and sample count are enforced
    checks = {}
    try: reg.enrol('x', 'child', [face.look('xiaoming', 1.2, 1.0, 0.0)[0]] * 12, consent_by=None); checks['enrol without consent'] = 'ACCEPTED (wrong)'
    except PermissionError: checks['enrol without consent'] = 'refused'
    try: reg.enrol('x', 'child', [face.look('xiaoming', 1.2, 1.0, 0.0)[0]] * 2, consent_by='baba'); checks['enrol with 2 looks'] = 'ACCEPTED (wrong)'
    except ValueError: checks['enrol with 2 looks'] = 'refused'
    for w in enrolled: reg.enrol(w, role[w], [face.look(w, 1.2, 1.0, 0.0)[0] for _ in range(12)], consent_by='baba', t=0.0)
    out['consent_checks'] = checks
    rows = []
    for dist in (1.0, 2.0, 3.0, 4.0):
        for facing in (1.0, 0.0):
            for k in (5, 12):
                for w in who:
                    hit = wrong = unk = 0
                    for _ in range(n):
                        looks = [face.look(w, dist, facing, 0.0)[0] for _ in range(k)]; got, s1, s2 = reg.identify(looks)
                        if got is None: unk += 1
                        elif got == w: hit += 1
                        else: wrong += 1
                    rows.append(dict(dist=dist, frontal=facing > 0.5, looks=k, who=w, enrolled=w in enrolled, right=hit / n, wrong_name=wrong / n, unknown=unk / n))
    out['rows'] = rows
    def agg(sel):
        a = [r for r in rows if sel(r)]
        return dict(right=round(float(np.mean([r['right'] for r in a])), 3), wrong_name=round(float(np.mean([r['wrong_name'] for r in a])), 4), unknown=round(float(np.mean([r['unknown'] for r in a])), 3)) if a else None
    out['enrolled_by_distance_frontal_12looks'] = {str(d): agg(lambda r, d=d: r['enrolled'] and r['dist'] == d and r['frontal'] and r['looks'] == 12) for d in (1.0, 2.0, 3.0, 4.0)}
    out['enrolled_by_distance_side_12looks'] = {str(d): agg(lambda r, d=d: r['enrolled'] and r['dist'] == d and not r['frontal'] and r['looks'] == 12) for d in (1.0, 2.0, 3.0, 4.0)}
    out['enrolled_5looks_vs_12'] = dict(five=agg(lambda r: r['enrolled'] and r['looks'] == 5 and r['frontal'] and r['dist'] <= 2.0), twelve=agg(lambda r: r['enrolled'] and r['looks'] == 12 and r['frontal'] and r['dist'] <= 2.0))
    out['stranger_visiting_child'] = dict(by_distance={str(d): agg(lambda r, d=d: r['who'] == 'lele' and r['dist'] == d and r['looks'] == 12) for d in (1.0, 2.0, 3.0, 4.0)})
    out['stranger_cousin_shares_family_part'] = dict(by_distance={str(d): agg(lambda r, d=d: r['who'] == 'cousin' and r['dist'] == d and r['looks'] == 12) for d in (1.0, 2.0, 3.0, 4.0)})
    out['sibling_swap'] = dict(xiaoming_called_xiaohong=round(float(np.mean([r['wrong_name'] for r in rows if r['who'] == 'xiaoming' and r['looks'] == 12])), 4))
    # forgetting
    lk = [face.look('xiaoming', 1.5, 1.0, 0.0)[0] for _ in range(12)]; before = reg.identify(lk)[0]
    reg.forget('xiaoming'); after = reg.identify(lk)[0]; on_disk = 'xiaoming' in open(path).read() and json.load(open(path))['people'].get('xiaoming') is not None
    n_all = reg.forget_all()
    out['forget'] = dict(recognised_before=before, recognised_after=after, still_in_file=on_disk, forget_all_removed=n_all, left=len(reg.people), file_after=json.load(open(path))['people'])
    out['stored_per_person'] = ['role', 'emb (a %d-number vector)' % P.EMB_D, 'n looks', 'consent_by', 't']
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, ensure_ascii=False, indent=1)); _save('people_identity', out)


# --------------------------------------------------------------------------------------------- one day: everything together
class Director:
    """Plays a script of steps against a running supervisor: people walking, people talking. A step runs once its condition holds; steps run in order."""
    def __init__(self, sup, vs, ppl):
        self.sup, self.vs, self.ppl = sup, vs, ppl; self.steps = []; self.i = 0; self.t_prev = 0.0; self.timeline = []; self.done = False; self.notes = {}
        self.snap = []; self._t_snap = -99.0; self._coll = 0; self.collisions = []

    def add(self, name, when, act): self.steps.append((name, when, act)); return self
    def after(self, dt): return lambda s: s.r.t >= self.t_prev + dt

    def __call__(self, sup):
        self.vs.tick(); t = sup.r.t
        if t - self._t_snap >= 10.0:                                     # what the robot sees and where the people really are, every 10 s
            self._t_snap = t; x, y, th = sup.r.true_pose()
            self.snap.append(dict(t=round(t, 1), mode=sup.mode, robot=[round(x, 2), round(y, 2), round(th, 2)], tracks=[(k.id, round(k.x, 2), round(k.y, 2), k.hits, k.cls, k.truth, bool(k.coast), getattr(k, 'src', 'cam')) for k in self.ppl.det.tracks + self.ppl.movers.tracks],
                                  people=[(s['name'], round(s['x'], 2), round(s['y'], 2)) for s in self.ppl.people.truth() if s['inside']]))
        c = sup.a.stats['collisions']
        if c > self._coll:
            x, y, th = sup.r.true_pose(); near = [(s['name'], round(math.hypot(s['x'] - x, s['y'] - y), 2)) for s in self.ppl.people.truth() if s['inside']]
            self.collisions.append(dict(t=round(t, 1), n=c, robot=[round(x, 2), round(y, 2)], mode=sup.mode, state=sup.a.state, people=near)); self._coll = c
        if self.i < len(self.steps):
            name, when, act = self.steps[self.i]
            if when(sup): self.t_prev = sup.r.t; self.i += 1; self.timeline.append((round(sup.r.t, 1), name)); act(sup)
        else: self.done = True


def _spot_in_front(sup, d, side=0.0, avoid=()):
    """A free spot about d m from the robot, roughly in front of it (side: radians to the left/right), inside the room, clear of furniture and at least 0.8 m from the spots in `avoid`."""
    from episode import _furn_dist
    x, y, th = sup.r.true_pose()
    for dd in (d, 0.8 * d, 1.2 * d, 0.6 * d, 1.5 * d):
        for sd in (side, -side, 0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5):
            a = th + sd; px, py = x + dd * math.cos(a), y + dd * math.sin(a)
            if 0.3 < px < 3.7 and 0.3 < py < 2.7 and _furn_dist(px, py) > 0.3 and all(math.hypot(px - q[0], py - q[1]) >= 0.8 for q in avoid): return float(px), float(py)
    return float(np.clip(x, 0.4, 3.6)), float(np.clip(y - d, 0.4, 2.6))


def _day_job(seed=3):
    import perturb
    from episode import run_supervised
    from home import Home
    from voice import ITEM_WORDS, SimASR, VoiceSystem
    from inventory import PROTO
    box = {}
    pe = dict(names=['adult_m', 'child_8'], who=['baba', 'xiaoming'], family=['A', 'A'], guard=True, comply=[1.0, 0.7],
              steps=[[('at', 0.45, 1.5, math.pi / 2), ('face_robot',)], []], registry_path=f'/tmp/day_registry_{seed}.json')
    if os.path.exists(pe['registry_path']): os.remove(pe['registry_path'])
    inv_path = f'/tmp/day_inv_{seed}.json'
    if os.path.exists(inv_path): os.remove(inv_path)
    checks = []
    def chk(name, ok, detail=''): checks.append(dict(name=name, ok=bool(ok), detail=detail))

    def setup(sup):
        home = Home(sup.r, sup.a, inv_path=inv_path); sup.home = home; vs = VoiceSystem(sup, home, sup.ppl, asr=SimASR(0.04, seed), seed=seed, setup_mode=True)
        d = Director(sup, vs, sup.ppl); box.update(home=home, vs=vs, d=d); build(d, sup, home, vs)
        sup.day = d

    def say(d, text, speaker, label=None):
        return lambda s: box.setdefault('last', {}).update(rec=box['vs'].hear(text, speaker=speaker), text=text)

    def build(d, sup, home, vs):
        ppl = sup.ppl; reg = ppl.registry; L = box.setdefault('last', {})
        def hear(text, who):
            def f(s): L['rec'] = vs.hear(text, speaker=who); L['text'] = text
            return f
        def push(i, steps): return lambda s: ppl.people.scripts[i].steps.extend(steps)
        d.add('greet', d.after(3.0), hear('小拾小拾，你好', 0))
        d.add('check greet', d.after(1.0), lambda s: chk('greeting answered', 'Hello' in str(L['rec'].get('reply')) or '小拾' in str(L['rec'].get('reply')), L['rec'].get('reply')))
        d.add('dad: remember me', d.after(3.0), hear('小拾小拾，记住我，我叫爸爸', 0))
        d.add('dad: yes (name)', d.after(3.0), hear('对', 0))
        d.add('dad: agree (consent)', d.after(4.0), hear('同意', 0))
        d.add('wait enrolled', lambda s: '爸爸' in reg.people or s.r.t > d.t_prev + 25.0, lambda s: chk('dad enrolled with consent', '爸爸' in reg.people and reg.people['爸爸']['consent_by'], reg.report()))
        d.add('dad: scan', d.after(2.0), hear('小拾小拾，把家里扫描一遍', 0))
        d.add('dad leaves', d.after(2.0), lambda s: (chk('scan started by voice', s.mode == 'SCAN', s.mode), ppl.people.scripts[0].steps.extend([('exit', 1.0)])))
        d.add('scan done', lambda s: s.scan_done or s.r.t > d.t_prev + 330.0, lambda s: (chk('scan finished', s.scan_done, round(s.r.t, 1)), chk('scan found the furniture and the walls', bool(home.analysis) and len(home.analysis['furniture']) >= 5 and len(home.analysis['walls']) >= 3, (len(home.analysis['furniture']), len(home.analysis['walls'])) if home.analysis else None), box.update(halts_scan=ppl.guard.stats['halts'], halt_s_scan=round(ppl.guard.stats['halt_s'], 1)),
                                                                                         chk('no false halts while nobody was there', ppl.guard.stats['halts'] <= 1, ppl.guard.stats)))
        def enter(s):
            gx, gy = _spot_in_front(s, 1.2, side=0.0); fx, fy = _spot_in_front(s, 1.5, side=0.6, avoid=[(gx, gy)])         # both well inside the head camera's view: the child ahead, the grown-up a little to the side
            ppl.people.scripts[0].steps.extend([('enter',), ('walk', fx, fy, 1.0), ('face_robot',)])
            ppl.people.scripts[1].steps.extend([('wait', 4.0), ('enter',), ('walk', gx, gy, 1.0), ('face_robot',)])
        d.add('dad and child come in', d.after(3.0), enter)
        d.add('who is here', d.after(14.0), hear('小拾小拾，现在有谁', 0))
        d.add('check who', d.after(0.5), lambda s: chk('sees the child, as a child', any(k.cls in ('child', 'unsure') and k.truth == 1 for k in ppl.tracks()), [(k.cls, round(k.dist, 2), k.who, k.truth) for k in ppl.tracks()]))
        d.add('dad: remember the child', d.after(3.0), hear('小拾小拾，记住这个孩子，他叫小明', 0))
        d.add('dad: yes (name 2)', d.after(3.0), hear('对', 0))
        d.add('child tries to agree', d.after(4.0), hear('同意', 1))
        d.add('check child consent refused', d.after(0.2), lambda s: chk('a child cannot give the consent', '小明' not in reg.people and L['rec'].get('action') == 'denied', L['rec'].get('reply')))
        d.add('dad: agree 2', d.after(2.0), hear('同意', 0))
        d.add('wait enrolled child', lambda s: '小明' in reg.people or s.r.t > d.t_prev + 25.0, lambda s: chk('child enrolled by a grown-up', '小明' in reg.people and reg.people['小明']['role'] == 'child', reg.report()))
        d.add('child recognised', d.after(6.0), lambda s: chk('the child is recognised by name', any(k.who == '小明' for k in ppl.tracks()), [(k.who, round(k.who_conf, 2)) for k in ppl.tracks()]))
        def ask_where(s):
            its = [it for it in home.inv.items.values() if it['kind'] and it['kind_conf'] >= 0.25]
            if its: box['asked'] = its[0]; L['rec'] = vs.hear(f'小拾小拾，{ITEM_WORDS[its[0]["kind"]][0]}在哪里', speaker=1)
            else: L['rec'] = dict(reply=None)
        d.add('child: where is X', d.after(2.0), ask_where)
        d.add('check where', d.after(0.3), lambda s: chk('child gets an answer about a toy', bool(L['rec'].get('reply')) and '没见过' not in L['rec']['reply'], L['rec'].get('reply')))
        d.add('child: go charge', d.after(4.0), hear('小拾小拾，去充电', 1))
        d.add('check denied', d.after(0.2), lambda s: chk('a child cannot send the robot to charge', L['rec'].get('action') == 'denied' and s.mode != 'CHARGE', (L['rec'].get('reply'), s.mode)))
        d.add('dad: tidy', d.after(3.0), hear('小拾小拾，收拾一下玩具', 0))
        d.add('dad steps aside', d.after(3.0), lambda s: ppl.people.scripts[0].steps.extend([('walk', 3.6, 1.3, 0.8), ('face_robot',)]))
        d.add('child runs up', d.after(12.0), lambda s: ppl.people.scripts[1].steps.extend([('chase', 1.2, 0.2, 10.0), ('wait', 8.0), ('exit', 0.8)]))
        def stop(s): box['t_stop'] = s.r.t; box['v_stop'] = float(np.hypot(*s.r.d.qvel[:2])); L['rec'] = vs.hear('停下', speaker=0)
        d.add('dad: stop', d.after(20.0), stop)
        def stopped(s): return float(np.hypot(*s.r.d.qvel[:2])) < 0.02 and s.r.t > box['t_stop'] + 0.05 or s.r.t > box['t_stop'] + 5.0
        d.add('measure stop', stopped, lambda s: (box.update(stop_latency=round(s.r.t - box['t_stop'], 2)), chk('stop word stops the robot', L['rec'].get('action') == 'stop' and s.mode == 'IDLE', (L['rec'].get('action'), s.mode, box['stop_latency']))))
        d.add('dad: continue', d.after(1.0), hear('小拾小拾，继续', 0))
        d.add('check resumed', d.after(6.0), lambda s: chk('continue resumes the job', s.mode == 'TASK' or any(a[0] == 'resume' for a in vs.actions), (s.mode, vs.tts[-2:])))
        d.add('task done', lambda s: s.task_done or s.r.t > d.t_prev + 260.0, lambda s: box.update(t_task_end=round(s.r.t, 1)))
        d.add('dad: stop (end of the job)', d.after(1.0), hear('停下', 0))
        d.add('dad comes back', d.after(2.0), lambda s: ppl.people.scripts[0].steps.extend([('to_robot', 1.2, 1.0), ('face_robot',)]))
        d.add('dad: forget the child', d.after(10.0), hear('小拾小拾，忘掉小明', 0))
        d.add('dad: yes (forget)', d.after(6.5), hear('好的', 0))                          # the robot turns to him first (3.5 s), then asks
        d.add('check forgotten', d.after(0.5), lambda s: chk('forgetting removes the child', '小明' not in reg.people, reg.report()))
        d.add('who again', d.after(8.0), hear('小拾小拾，现在有谁', 0))
        d.add('end', d.after(1.0), lambda s: chk('after forgetting the child is only "a child"', '小明' not in str(L['rec'].get('reply')), L['rec'].get('reply')))

    t0 = time.time()
    res, sup = run_supervised_day(seed, pe, setup, box)
    d = box['d']; vs = box['vs']; pp = sup.ppl
    out = dict(seed=seed, sim_s=round(sup.r.t), wall_s=round(time.time() - t0), checks=checks, passed=sum(c['ok'] for c in checks), n_checks=len(checks), timeline=d.timeline, finished_script=d.done,
               delivered=res['delivered'], total=res['total'], collisions=res['collisions'], stuck=res['stuck'], contacts=[int(v) for v in pp.people.contact_events], min_gap_cm=[round(100 * v, 1) for v in pp.people.min_gap],
               guard=pp.guard.stats, guard_episodes=pp.guard.episodes, asked_to_make_way=pp.people.asked, moved_aside=pp.people.moved_aside, stop_latency_s=box.get('stop_latency'), t_task_end=box.get('t_task_end'), halts_during_scan=box.get('halts_scan'),
               snapshots=d.snap, collision_log=d.collisions, scan_log=[dict(t=round(e['t'], 1), stops=e.get('stops'), observed=round(e.get('observed', 0), 3)) for e in box['home'].scan_log][-12:], voice_stats=vs.stats, tts=vs.tts, actions=[a for a in vs.actions], registry_file_keys=sorted({k for p in json.load(open(pe['registry_path']))['people'].values() for k in p}) if os.path.exists(pe['registry_path']) else None,
               inventory=dict(summary=box['home'].inv.summary(), items=[(it['id'], it['kind'], it['place'], it['status']) for it in box['home'].inv.items.values()]), scan=dict(furniture=len(box['home'].analysis['furniture']) if box['home'].analysis else None,
               walls=len(box['home'].analysis['walls']) if box['home'].analysis else None), cls_votes=pp.cls_votes, id_votes=pp.id_votes, false_track_share=round(pp.n_false / max(pp.n_look, 1), 3))
    return out


def run_supervised_day(seed, pe, setup, box):
    from episode import run_supervised
    import perturb
    holder = {}
    def script(sup):
        sup.day(sup)
    res, sup = run_supervised(seed, start='idle', people=pe, setup=setup, script=script, t_end=1100.0, perturbation=dict(lidar_model='ld19cal'), until=lambda s: box['d'].done)
    return res, sup


def bench_day(seeds=(3,)):
    res = []
    for s in seeds:
        r = _day_job(s); res.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ('timeline', 'tts', 'actions', 'inventory', 'checks')}, ensure_ascii=False, default=float))
        for c in r['checks']: print('  ', 'PASS' if c['ok'] else 'FAIL', c['name'], '' if c['ok'] else c['detail'])
    _save('day', dict(runs=res))


# --------------------------------------------------------------------------------------------- export of a scan for the page
def export_scan(seed=1, planner='nbv'):
    """One scan of the furnished room with the toys on the floor: the point cloud (PLY), a stand-alone viewer page (HTML) and the numbers, into results/."""
    import mapping3d as M
    import perturb
    from controller import Agent
    from episode import make_layout
    from home import Home
    from inventory import PROTO, furniture_name
    from model import HOME
    from modes import Supervisor
    from robot import Robot
    perturb.CUR = dict(lidar_model='ld19cal')
    objects, boxes = make_layout(seed)
    robot = Robot(objects, seed=seed, extra_boxes=boxes, robot_pose=HOME); agent = Agent(robot, seed=seed)
    home = Home(robot, agent); sup = Supervisor(robot, agent); sup.home = home
    t0 = time.time(); sup.scan_home(planner=planner); sup.run(900.0, until=lambda s: s.scan_done or (s.mode == 'IDLE' and s.r.t > 5))
    info = home.finish_scan(); P, vis = _true_visible(robot); ev = M.evaluate(home.map, P, vis)
    objs = [dict(label=home.inv.label(it), x=it['pos'][0], y=it['pos'][1], kind=it['kind']) for it in home.inv.items.values()]
    for f in info['furniture']: f['name'] = furniture_name(f)
    data = M.viewer_data(home.map, info, objects=objs, name=f'家里的三维地图（仿真，{len(home.inv.items)} 件东西）')
    os.makedirs(RES, exist_ok=True)
    n = M.export_ply(home.map, os.path.join(RES, f'home_scan_seed{seed}.ply'))
    open(os.path.join(RES, f'home_scan_seed{seed}.html'), 'w').write(M.viewer_html(data))
    from model import FURNITURE
    truth = [dict(name=n, centre=(cx, cy), size=((2 * sx, 2 * sx) if sy is None else (sx, sy)), h=h) for n, cx, cy, sx, sy, h, _ in FURNITURE] + [dict(name=f'carton{i}', centre=(b[0], b[1]), size=(b[2], b[3]), h=b[4]) for i, b in enumerate(boxes)]
    vs_truth = []
    for f in info['furniture']:
        tt = min(truth, key=lambda q: math.hypot(q['centre'][0] - f['centre'][0], q['centre'][1] - f['centre'][1]))
        L, W = max(f['length'], f['width']), min(f['length'], f['width']); tl, tw = max(tt['size']), min(tt['size'])
        vs_truth.append(dict(guess=f['name'], truth=tt['name'], centre_err_cm=round(100 * math.hypot(tt['centre'][0] - f['centre'][0], tt['centre'][1] - f['centre'][1]), 1), long_side_cm=round(100 * (L - tl), 1),
                             short_side_cm=round(100 * (W - tw), 1), height_cm=round(100 * (f['height_seen'] - tt['h']), 1)))
    meta = dict(seed=seed, planner=planner, vs_truth=vs_truth, true_furniture=len(truth), sim_s=round(robot.t, 1), wall_s=round(time.time() - t0), stops=len(home.scan_log), path_m=round(agent.stats['dist'], 1), recall=round(ev['recall'], 3), precision=round(ev['precision'], 3),
                points=n, voxel_cm=M.VOX * 100, walls=len(info['walls']), furniture=[dict(name=f['name'], centre=[round(v, 2) for v in f['centre']], length=round(f['length'], 2), width=round(f['width'], 2), height=round(f['height_seen'], 2)) for f in info['furniture']],
                objects=objs, collisions=agent.stats['collisions'], inventory=home.inv.summary(), viewer_bytes=os.path.getsize(os.path.join(RES, f'home_scan_seed{seed}.html')))
    _save('scan_export', meta); print(json.dumps(meta, ensure_ascii=False, default=float)[:1500])


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    procs = int(os.environ.get('PROCS', '2')); a = [int(v) for v in sys.argv[2:]]
    if what == 'scan': bench_scan(tuple(a) or (1,), procs=procs)
    elif what == 'inv1': bench_inventory(tuple(a) or (1, 2, 3), procs=procs)
    elif what == 'inventory': bench_inventory_sessions(tuple(a) or (1, 2, 3), procs=procs)
    elif what == 'voice': bench_voice(procs=procs)
    elif what == 'export': export_scan(a[0] if a else 1)
    elif what == 'day': bench_day(tuple(a) or (3,))
    elif what == 'identity': bench_identity()
    elif what == 'pairs': bench_pairs(procs=procs)
    elif what == 'safety': bench_people_safety(tuple(a) or (1, 2), procs=procs, cam=os.environ.get('CAM', 'mast'))
    elif what == 'people_static': bench_people_static(procs=procs, trials=a[0] if a else 150, cam=os.environ.get('CAM', 'mast'))
