"""The whole second day (test_day2.py) for several seeds at once, for the acceptance (accept.py item G4).

  python accept_day2.py 5 6 7      three seeds, four processes at a time (about six minutes each) -> ../results/accept_day2.json
"""
import json
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = os.path.join(HERE, '..', 'results', 'accept_day2.json')


def one(seed):
    import test_day2 as T
    r = T._day2_job(seed)
    return dict(seed=seed, passed=r['passed'], n=r['n_checks'], checks=[dict(name=c['name'], ok=bool(c['ok'])) for c in r['checks']], collisions=r.get('collisions'), sim_s=r.get('sim_s'), wall_s=r.get('wall_s'))


if __name__ == '__main__':
    seeds = [int(a) for a in sys.argv[1:]] or [5]; procs = int(os.environ.get('PROCS', '4')); rows = []
    with mp.Pool(min(procs, len(seeds)), maxtasksperchild=1) as p:
        for r in p.imap_unordered(one, seeds): print(r['seed'], f'{r["passed"]}/{r["n"]}', [c['name'] for c in r['checks'] if not c['ok']], flush=True); rows.append(r)
    json.dump(dict(runs=sorted(rows, key=lambda r: r['seed'])), open(OUT, 'w'), ensure_ascii=False, indent=1)
