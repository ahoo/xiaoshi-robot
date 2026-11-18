"""Batch of randomised fetch episodes -> results/trials.json

  python run_trials.py            # the three conditions reported in docs/validation.md
"""
import json
import multiprocessing as mp
import os
import shutil
import sys

from episode import run_episode

CONDITIONS = [
    dict(name='standard', label='标准：物体离墙和家具至少 30 cm，夹爪贴防滑胶带', clearance=0.30, pad_friction=1.0, seeds=range(1, 21)),
    dict(name='near_walls', label='贴边：物体离墙和家具只有 12 cm 起', clearance=0.12, pad_friction=1.0, seeds=range(101, 113)),
    dict(name='bare_jaws', label='裸夹爪：不贴防滑胶带（摩擦系数 0.35）', clearance=0.30, pad_friction=0.35, seeds=range(1, 9)),
    dict(name='clutter', label='扎堆：有两对物体挨着放，边缘只隔 0.5–5 cm', clearance=0.30, pad_friction=1.0, seeds=range(201, 209), clutter=True),
    dict(name='materials', label='难材质：五件都是黑色或反光的，另加一个透明塑料瓶', clearance=0.30, pad_friction=1.0, seeds=range(301, 309), materials=True),
]


CACHE = '../results/.trials_cache'      # one file per finished episode, so an interrupted batch resumes


def _one(args):
    name, seed, clearance, mu, clutter, materials = args
    fn = f'{CACHE}/{name}_{seed}.json'
    if os.path.exists(fn): return json.load(open(fn))
    res, agent, robot = run_episode(seed, clearance, pad_friction=mu, clutter=clutter, materials=materials)
    res['condition'] = name; res['log'] = agent.log_lines
    json.dump(res, open(fn, 'w'), ensure_ascii=False)
    return res


if __name__ == '__main__':
    os.makedirs(CACHE, exist_ok=True)
    jobs = [(c['name'], s, c['clearance'], c['pad_friction'], c.get('clutter', False), c.get('materials', False)) for c in CONDITIONS for s in c['seeds']]
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else max(1, mp.cpu_count() - 1), maxtasksperchild=1) as pool:   # fresh process per episode: GL memory is not all returned
        out = []
        for r in pool.imap_unordered(_one, jobs):
            out.append(r); print(r['condition'], r['seed'], f"{r['delivered']}/{r['total']}", r['sim_time'], flush=True)
    out.sort(key=lambda r: (r['condition'], r['seed']))
    json.dump(dict(conditions=[dict(name=c['name'], label=c['label'], clearance=c['clearance'], pad_friction=c['pad_friction'], n=len(c['seeds'])) for c in CONDITIONS], episodes=out),
              open('../results/trials.json', 'w'), ensure_ascii=False, indent=1)
    shutil.rmtree(CACHE)
