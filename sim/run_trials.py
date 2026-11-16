"""Batch of randomised fetch episodes -> results/trials.json

  python run_trials.py            # the three conditions reported in docs/validation.md
"""
import json
import multiprocessing as mp
import sys

from episode import run_episode

CONDITIONS = [
    dict(name='standard', label='标准：物体离墙和家具至少 30 cm，夹爪贴防滑胶带', clearance=0.30, pad_friction=1.0, seeds=range(1, 21)),
    dict(name='near_walls', label='贴边：物体离墙和家具只有 12 cm 起', clearance=0.12, pad_friction=1.0, seeds=range(101, 113)),
    dict(name='bare_jaws', label='裸夹爪：不贴防滑胶带（摩擦系数 0.35）', clearance=0.30, pad_friction=0.35, seeds=range(1, 9)),
]


def _one(args):
    name, seed, clearance, mu = args
    res, agent, robot = run_episode(seed, clearance, pad_friction=mu)
    res['condition'] = name; res['log'] = agent.log_lines
    return res


if __name__ == '__main__':
    jobs = [(c['name'], s, c['clearance'], c['pad_friction']) for c in CONDITIONS for s in c['seeds']]
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else max(1, mp.cpu_count() - 1)) as pool:
        out = []
        for r in pool.imap_unordered(_one, jobs):
            out.append(r); print(r['condition'], r['seed'], f"{r['delivered']}/{r['total']}", r['sim_time'], flush=True)
    out.sort(key=lambda r: (r['condition'], r['seed']))
    json.dump(dict(conditions=[dict(name=c['name'], label=c['label'], clearance=c['clearance'], pad_friction=c['pad_friction'], n=len(c['seeds'])) for c in CONDITIONS], episodes=out),
              open('../results/trials.json', 'w'), ensure_ascii=False, indent=1)
