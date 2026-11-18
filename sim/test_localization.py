"""Offline localisation bench: replay one recorded trajectory with different lidar error models.

The trajectory (true pose, wheel odometry and the noise-free lidar ranges with the geom each ray hit) is recorded once
from a nominal episode; replaying it through the localiser takes ~20 s instead of a 10-minute episode. The replay is
open loop (the estimate does not steer the robot), which is what makes the comparison fair between settings.

  python test_localization.py record 5          # records results/../cache/lidarlog_5.pkl (one nominal episode)
  python test_localization.py bench 5,9         # writes results/loc_bench.json

Lidar error models:
  gauss10 / gauss30  independent Gaussian range noise, 1 / 3 cm (the old perturbation 'lidar_noise' 1 and 3)
  ld19 / ld19b / ld19c  a realistic LDROBOT LD19 model, three draws of the per-surface bias:
                     a constant bias per surface, uniform within +-45 mm (the datasheet's accuracy figure, taken as
                     worst case at every range), 10 mm noise, 15 mm range steps, mixed pixels at range jumps (the
                     return lands between the near and far surface, half the time), 1 % random outliers
  ld19x2             the same with 20 mm noise and 3 % outliers
"""
import json
import math
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', 'cache')
ARM_STATES = ('抓取', '放置', '放下')


from lidar_models import MODELS, gauss, ld19  # noqa: F401


def record(seed):
    import episode
    log, ticks = [], []

    def hook(robot, agent):
        sense = agent._tick % 5 == 0
        ticks.append((robot.odom.copy(), sense, agent.state))
        if sense:
            log.append(dict(t=robot.t, true=np.array(robot.true_pose()), state=agent.state,
                            raw=getattr(robot, 'last_raw', None), geom=getattr(robot, 'last_ray_geom', None)))
    res, agent, robot = episode.run_episode(seed, 0.30, on_step=hook)
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump(dict(log=log, ticks=ticks, ang=robot.ray_ang), open(os.path.join(CACHE, f'lidarlog_{seed}.pkl'), 'wb'))
    print('recorded seed', seed, len(log), 'scans; delivered', res['delivered'])


def replay(path, model, seed=0, **settings):
    import importlib
    import localization as L
    importlib.reload(L)
    for k, v in settings.items(): setattr(L, k, v)
    import controller as C
    from model import P
    D = pickle.load(open(path, 'rb')); log = D['log']; ang = D['ang']; lx = P['lidar_pos'][0]
    rng = np.random.default_rng(seed)
    loc = L.Localizer(log[0]['true'], C.CS, (C.GX * C.CS, C.GY * C.CS))
    err, li, t0 = [], 0, time.time()
    for odom, sense, state in D['ticks']:
        loc.predict(odom)
        if not sense: continue
        e = log[li]; li += 1
        if e['state'] in ARM_STATES or e['raw'] is None: continue
        r = model(e['raw'], e['geom'], rng); f = np.isfinite(r)
        px, py = lx + r[f] * np.cos(ang[f]), r[f] * np.sin(ang[f])
        loc.correct(px, py); loc.add_scan(px, py)
        if li % 10 == 0:
            tx, ty, _ = e['true']; err.append(math.hypot(loc.est[0] - tx, loc.est[1] - ty))
    err = np.array(err)
    return dict(mean_cm=round(100 * err.mean(), 2), p95_cm=round(100 * np.percentile(err, 95), 2), max_cm=round(100 * err.max(), 2),
                matched=loc.n_match, skipped=loc.n_skip, wall_s=round(time.time() - t0, 1))


SETTINGS = {'round10': dict(OCC_MODE='abs', SCORE='mean'), 'round11': {}}


def _job(j):
    seed, name, model = j
    return seed, name, model, replay(os.path.join(CACHE, f'lidarlog_{seed}.pkl'), MODELS[model], **SETTINGS[name])


if __name__ == '__main__':
    if sys.argv[1] == 'record':
        record(int(sys.argv[2]))
    else:
        from multiprocessing import Pool
        seeds = [int(s) for s in sys.argv[2].split(',')]
        jobs = [(s, n, m) for s in seeds for m in MODELS for n in SETTINGS]
        out = []
        with Pool(int(os.environ.get('JOBS', 2))) as p:
            for seed, name, model, r in p.imap(_job, jobs):
                print(f'seed {seed} {model:8s} {name}: mean {r["mean_cm"]:5.1f} p95 {r["p95_cm"]:5.1f} max {r["max_cm"]:5.1f} cm', flush=True)
                out.append(dict(seed=seed, model=model, localiser=name, **r))
        json.dump(out, open(os.path.join(HERE, '..', 'results', 'loc_bench.json'), 'w'), indent=1)
