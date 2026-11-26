"""Whole fetch jobs for the acceptance (accept.py items B1, B2): five (or six) things to put in the basket, the robot as it is meant to be built (LD19 lidar with its calibration, the camera on the
forearm), a few layouts per condition.

  python accept_job.py standard 6                                   six layouts of the standard condition
  python accept_job.py near_walls,clutter,materials,bare_jaws 3     three layouts of each of the four hard conditions
  CAMERA=none python accept_job.py standard 6                       the same without the camera

Conditions (as in run_trials.py): standard (things at least 30 cm from the walls and the furniture), near_walls (12 cm), clutter (two pairs of things side by side), materials (black or shiny things
and a clear plastic bottle), bare_jaws (no anti-slip tape on the gripper). Writes ../results/accept_job.json; the conditions run now replace the earlier rows of the same conditions.
"""
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = os.path.join(HERE, '..', 'results', 'accept_job.json')
COND = dict(standard=dict(clearance=0.30, pad=1.0, seed0=1), near_walls=dict(clearance=0.12, pad=1.0, seed0=101), clutter=dict(clearance=0.30, pad=1.0, seed0=201, clutter=True),
            materials=dict(clearance=0.30, pad=1.0, seed0=301, materials=True), bare_jaws=dict(clearance=0.30, pad=0.35, seed0=1))


def one(args):
    cond, seed, cam = args
    import model
    from episode import make_layout, run_supervised
    c = COND[cond]; model.P['pad_friction'] = c['pad']; t0 = time.time()
    objects, boxes = make_layout(seed, c['clearance'], clutter=c.get('clutter', False), materials=c.get('materials', False))
    res, sup = run_supervised(seed, objects=objects, boxes=boxes, soc0=None, perturbation=dict(lidar_model='ld19cal'), robot_kw=None if cam == 'none' else dict(arm_cam=cam),
                              until=lambda s: s.task_done or s.mode == 'FAULT', t_end=720.0)
    return dict(condition=cond, seed=seed, camera=cam, delivered=res['delivered'], total=res['total'], sim_time=res['sim_time'], collisions=res['collisions'], stuck=res['stuck'], fault=sup.fault, wall_s=round(time.time() - t0))


if __name__ == '__main__':
    conds = sys.argv[1].split(','); n = int(sys.argv[2]); cam = os.environ.get('CAMERA', 'forearm_up'); procs = int(os.environ.get('PROCS', '4'))
    jobs = [(c, COND[c]['seed0'] + k, cam) for k in range(n) for c in conds]
    rows = []
    with mp.Pool(procs, maxtasksperchild=1) as p:
        for r in p.imap_unordered(one, jobs): print(r, flush=True); rows.append(r)
    old = json.load(open(OUT))['runs'] if os.path.exists(OUT) else []
    keep = [r for r in old if r['condition'] not in conds]
    json.dump(dict(runs=sorted(keep + rows, key=lambda r: (r['condition'], r['seed']))), open(OUT, 'w'), ensure_ascii=False, indent=1)
