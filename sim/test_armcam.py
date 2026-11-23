"""The camera on the arm instead of a mast (round 16): can the robot still find people, tell children from grown-ups, and know who they are?

  python test_armcam.py pose        where the camera is and where the robot believes it is (calibration error of the joints), for each mount
  python test_armcam.py faces [n]   who-is-it, per mount: n people (default 60) placed 1-3 m ahead, enrolled or not, facing the robot or not: how often and how fast the right name comes
  python test_armcam.py mic [n]     the microphone array at 0.5 m (on the mast), 0.3 and 0.2 m (on the upper plate): the direction of a voice
  python test_armcam.py righting    the self-righting maneuvers again with the camera's 75 g on the forearm
  (the people bench with the camera on the arm is test_home.py people_static / safety with CAM=forearm|periscope)
Writes results/armcam_<name>.json. CAM mounts: mast (round 14), forearm, periscope (model.ARM_CAMS).

The recogniser is the stand-in of round 14 (people.FaceStandIn: a feature vector per person, noise that grows with distance and when the face is turned away, and, for the arm camera, when the
camera looks steeply up at the face or the head is not in the picture). So this bench says whether the arm camera gets the faces into the picture often enough and in time; how well a real
face model reads them is not something it can say.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.path.join(HERE, '..', 'results')


def _save(name, data):
    os.makedirs(RES, exist_ok=True); json.dump(data, open(os.path.join(RES, f'armcam_{name}.json'), 'w'), ensure_ascii=False, indent=1, default=float)


class Stub:
    """The agent where only the pose is needed (the true one: what is measured is the camera side)."""
    freeze = False; scan = None
    def __init__(self, r): self.r = r
    def pose(self): return self.r.true_pose()


def _robot(cam, seed=0, people=(), pose=(2.0, 1.5, 0.0), **kw):
    from robot import Robot
    return Robot([], seed=seed, people=list(people), robot_pose=pose, **(dict(head_cam=True) if cam == 'mast' else dict(arm_cam=cam)), **kw)


def _raise(r, cam, ps=None):
    import model
    look = None if cam == 'mast' else model.ARM_CAMS[cam]['look']
    if look is None: return 0.0
    r.set_arm(q5=np.array(look)); n = 0
    while not r.arm_done(0.02) and n < 800:
        if ps is not None: ps.before_physics()
        r.step(); n += 1
        if ps is not None: ps.after_physics()
    for _ in range(50):
        if ps is not None: ps.before_physics()
        r.step()
        if ps is not None: ps.after_physics()
    return n * 0.02


def pose_bench():
    import mujoco
    import people as P
    out = {}
    for cam in CAMS:
        rows = []
        for seed in range(8):
            r = _robot(cam, seed=seed); t_raise = _raise(r, cam); stub = Stub(r); det = P.PersonDetector(r, stub, np.random.default_rng(seed)); c = det.cam
            if c.arm: c._set()
            cid = c.cam; Rb = r.d.xmat[r.chassis].reshape(3, 3); pb = r.d.xpos[r.chassis]; Rc = r.d.cam_xmat[cid].reshape(3, 3); tp = Rb.T @ (r.d.cam_xpos[cid] - pb); tp[2] += pb[2]
            fwd_true = -(Rb.T @ Rc)[:, 2]; fwd_bel = -c.R_bc[:, 2]
            rows.append(dict(height=float(tp[2]), elev=math.degrees(math.asin(fwd_true[2])), ang_err=math.degrees(math.acos(float(np.clip(fwd_true @ fwd_bel, -1, 1)))), pos_err_mm=1000 * float(np.linalg.norm(tp - c.t_bc)), t_raise=t_raise))
        out[cam] = dict(height_m=round(float(np.mean([x['height'] for x in rows])), 3), elevation_deg=round(float(np.mean([x['elev'] for x in rows])), 1), time_to_raise_s=round(rows[0]['t_raise'], 1),
                        believed_pose_angle_error_deg=dict(mean=round(float(np.mean([x['ang_err'] for x in rows])), 2), max=round(float(np.max([x['ang_err'] for x in rows])), 2)),
                        believed_pose_position_error_mm=dict(mean=round(float(np.mean([x['pos_err_mm'] for x in rows])), 1), max=round(float(np.max([x['pos_err_mm'] for x in rows])), 1)))
        print(cam, out[cam])
    _save('pose' if len(CAMS) > 1 else 'pose_' + CAMS[0], out); return out


def _faces_job(args):
    import people as P
    from episode import _furn_dist
    cam, seed, n = args[:3]
    names = ['adult_m', 'adult_f', 'child_8', 'child_5', 'toddler']; who = ['baba', 'mama', 'xiaoming', 'lele', 'baby']; family = ['A', 'A', 'A', 'B', 'A']; enrolled = ['baba', 'mama', 'xiaoming', 'baby']
    rng = np.random.default_rng(seed)
    r = _robot(cam, seed=seed, people=names); stub = Stub(r); scripts = [P.Script([]) for _ in names]
    ps = P.PeopleSystem(r, stub, names, scripts, who=who, family=family, seed=seed, guard=False, strict_face=True)       # the same rules for every camera: the head has to be in the picture, a steep look up costs
    for w, nm in zip(who, names):
        if w in enrolled: ps.registry.enrol(w, 'adult' if nm.startswith('adult') else 'child', [ps.face.look(w, 1.2, 1.0, 0.0)[0] for _ in range(12)], consent_by='baba', t=0.0)
    _raise(r, cam, ps); rows = []
    import test_home as T
    for trial in range(n):
        i = trial % len(names); face = rng.random() < 0.7
        while True:
            rx, ry, yaw = rng.uniform(0.5, 3.5), rng.uniform(0.5, 2.5), rng.uniform(-math.pi, math.pi)
            if _furn_dist(rx, ry) < 0.3: continue
            d, b = rng.uniform(1.0, 3.0), math.radians(rng.uniform(-30, 30)); px, py = rx + d * math.cos(yaw + b), ry + d * math.sin(yaw + b)
            if 0.3 < px < 3.7 and 0.3 < py < 2.7 and _furn_dist(px, py) >= 0.3: break
        pyaw = math.atan2(ry - py, rx - px) if face else float(rng.uniform(-math.pi, math.pi))
        for j in range(len(names)): scripts[j] = P.Script([]) if j != i else P.Script([('at', px, py, pyaw)])
        ps.people.scripts = scripts; T._teleport(r, rx, ry, yaw); ps.det.tracks = []; ps.det._t = -9.0
        t_ok = None; wrong = False; first_track = None
        for k in range(250):                                                                  # 5 s
            ps.before_physics(); r.step(); ps.after_physics()
            for tk in ps.det.confirmed(2):
                if math.hypot(tk.x - px, tk.y - py) < 0.6:
                    if first_track is None: first_track = r.t
                    if tk.who is not None:
                        if tk.who == who[i] and t_ok is None: t_ok = r.t
                        if tk.who != who[i]: wrong = True
        rows.append(dict(cam=cam, person=who[i], enrolled=who[i] in enrolled, dist=round(d, 2), facing=face, found=first_track is not None, named=t_ok is not None, wrong=wrong,
                         t_named=None if t_ok is None else round(t_ok - (first_track or 0.0), 2)))
        scripts[i] = P.Script([])
    return rows, ps.n_face_out, ps.n_look


CAMS = tuple(os.environ.get('CAMS', 'mast,forearm,periscope').split(','))


def faces_bench(n=60, procs=2):
    import multiprocessing as mp
    out = {}
    for cam in CAMS:
        per = max(n // procs, 5)
        with mp.Pool(procs, maxtasksperchild=1) as p: res = p.map(_faces_job, [(cam, 500 + k, per) for k in range(procs)])
        rows = [x for o in res for x in o[0]]; pic_out = sum(o[1] for o in res); looks = sum(o[2] for o in res)
        def agg(sel):
            a = [x for x in rows if sel(x)]
            if not a: return None
            named = [x for x in a if x['named']]
            return dict(n=len(a), found=round(sum(x['found'] for x in a) / len(a), 3), named_right=round(len(named) / len(a), 3), named_within_2s=round(sum(1 for x in named if x['t_named'] is not None and x['t_named'] <= 2.0) / len(a), 3),
                        median_time_to_name_s=None if not named else round(float(np.median([x['t_named'] for x in named])), 2), wrong_name=round(sum(x['wrong'] for x in a) / len(a), 3))
        out[cam] = dict(enrolled_facing=agg(lambda x: x['enrolled'] and x['facing']), enrolled_turned_away=agg(lambda x: x['enrolled'] and not x['facing']), enrolled_facing_by_distance={f'{a}-{b} m': agg(lambda x, a=a, b=b: x['enrolled'] and x['facing'] and a <= x['dist'] < b) for a, b in ((1.0, 1.7), (1.7, 2.4), (2.4, 3.1))},
                        stranger=agg(lambda x: not x['enrolled']), looks_where_the_head_was_out_of_the_picture=f'{pic_out}/{looks + pic_out}', rows=rows)
        print(cam, {k: v for k, v in out[cam].items() if k != 'rows'})
    _save('faces' if len(CAMS) > 1 else 'faces_' + CAMS[0], out); return out


def _mic_job(args):
    import sound_lab as SL
    height, cond_name, seed, n = args; cond = dict(mic_height=height, rt60=0.5, level=62.0, array='4mic'); cond.update(dict(idle={}, driving=dict(speed=0.15), tv=dict(tv=60.0))[cond_name])
    rng = np.random.default_rng(seed); errs = []; found = 0
    for k in range(n):
        spec = SL.voice_spec(rng, 'child' if k % 4 == 3 else 'adult')
        try: o = SL.chain(spec, rng, cond=cond)
        except RuntimeError: continue
        if o['found']: found += 1; errs.append(o['err_deg'])
        else: errs.append(180.0)
    return dict(height=height, cond=cond_name, n=len(errs), found=found, errs=errs)


def mic_bench(n=48, procs=2):
    """The microphone array was on the mast (0.5 m). Without the mast it goes on the upper plate by the lidar (about 0.2 m): does the direction of a voice get worse? (Free-field room model: no shadow of the
    robot's body or arm, and no louder fan next to it; those are not in the simulator.)"""
    import multiprocessing as mp
    jobs = [(h, c, 700 + i, n // procs) for h in (0.50, 0.30, 0.20) for c in ('idle', 'driving', 'tv') for i in range(procs)]
    with mp.Pool(procs, maxtasksperchild=1) as p: res = p.map(_mic_job, jobs)
    out = {}
    for h in (0.50, 0.30, 0.20):
        for c in ('idle', 'driving', 'tv'):
            e = np.array([x for r in res if r['height'] == h and r['cond'] == c for x in r['errs']]); nf = sum(r['found'] for r in res if r['height'] == h and r['cond'] == c)
            out[f'{h:.2f} m / {c}'] = dict(n=len(e), found=round(nf / max(len(e), 1), 3), median_err_deg=round(float(np.median(e)), 1), within_20deg=round(float((e <= 20).mean()), 3), within_45deg=round(float((e <= 45).mean()), 3))
        print(h, {k: v for k, v in out.items() if k.startswith(f'{h:.2f}')})
    _save('mic_height', out); return out


def _right_job(args):
    import model
    import righting
    import search_righting
    from search_righting import Env
    cls, mass, n, seed = args
    from robot import Robot as R0; model.ARM_CAMS['forearm']['mass'] = mass                       # (a worker may run several jobs: always start from the real Robot)
    search_righting.Robot = (lambda *a, **k: R0(*a, arm_cam='forearm', **k)) if mass > 0 else R0
    env = Env(cls); ph = righting.LIB['maneuvers'][cls][0]['phases']; rng = np.random.default_rng(seed); ups = 0; fin = []
    for k in range(n):
        env.unperturb(); env.drop(rng, jitter=3.0); env.perturb(rng)
        o = env.run_phases([dict(q=np.array(p['q']), w=np.array(p['w']), T=p['T'], speed=p['speed']) for p in ph]); ups += o['upright']; fin.append(o['final'])
    return dict(cls=cls, mass=mass, upright=int(ups), n=n, mean_final_tilt=round(float(np.mean(fin)), 1))


def righting_bench(n=12, procs=2):
    """The stored maneuvers (sim/righting_maneuvers.json, searched for the arm without a camera) with a camera of 0 to 75 g on the forearm: standard perturbations (arm calibration 1 degree,
    roller friction +-20 %, chassis mass +-8 %, resting pose +-3 degrees), the same scenarios for every mass."""
    import multiprocessing as mp
    masses = (0.0, 0.02, 0.04, 0.05, 0.06, 0.075)
    jobs = [(c, m, n, 99) for c in ('side_l', 'side_r', 'fwd') for m in masses]
    with mp.Pool(procs, maxtasksperchild=1) as p: res = p.map(_right_job, jobs)
    out = {c: {f'{int(1000 * m)} g': next(f'{r["upright"]}/{r["n"]}' for r in res if r['cls'] == c and r['mass'] == m) for m in masses} for c in ('side_l', 'side_r', 'fwd')}
    print(json.dumps(out, ensure_ascii=False)); _save('righting', dict(camera='forearm, camera mass swept', upright_of_n=out, rows=res)); return out


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'pose'; procs = int(os.environ.get('PROCS', '2'))
    if what == 'pose': pose_bench()
    elif what == 'faces': faces_bench(int(sys.argv[2]) if len(sys.argv) > 2 else 60, procs)
    elif what == 'righting': righting_bench(int(sys.argv[2]) if len(sys.argv) > 2 else 12, procs)
    elif what == 'mic': mic_bench(int(sys.argv[2]) if len(sys.argv) > 2 else 48, procs)
