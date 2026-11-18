"""Sensitivity of the whole task to real-world errors: the standard condition (seeds 1-10), once per setting, each
setting making one thing worse than the nominal model (perturb.py), plus one with everything a little worse at once.
-> results/sensitivity.json. The settings whose delivery rate drops most are the ones to measure first on the robot.

  python run_sensitivity.py [processes]
"""
import json
import multiprocessing as mp
import os
import shutil
import sys

from episode import run_episode

SEEDS = range(1, 11)
SETTINGS = [
    dict(name='nominal', label='名义模型', pert={}, pad=None),
    dict(name='pad_mu', label='指垫摩擦减半（1.0 → 0.5）', pert={}, pad=0.5),
    dict(name='obj_mass', label='物体重一倍', pert=dict(obj_mass=2.0), pad=None),
    dict(name='arm_offset', label='每个舵机零位偏 2°（1σ）', pert=dict(arm_offset_deg=2.0), pad=None),
    dict(name='cam_err', label='前置相机安装位置偏 1 cm、角度偏 1°（1σ）', pert=dict(cam_err=(0.01, 1.0)), pad=None),
    dict(name='chassis_light', label='底盘轻 20%', pert=dict(chassis_mass=0.8), pad=None),
    dict(name='odom', label='里程计误差 ×3', pert=dict(odom=3.0), pad=None),
    dict(name='lidar', label='雷达噪声 ×3（3 cm）', pert=dict(lidar_noise=3.0), pad=None),
    dict(name='depth', label='深度噪声 ×2', pert=dict(depth_noise=2.0), pad=None),
    dict(name='combined', label='以上各项都差一点（摩擦 0.7、重 1.5 倍、零位 1°、相机 0.5 cm/0.5°、底盘轻 10%、里程计 ×2、雷达 ×2、深度 ×1.5）',
         pert=dict(obj_mass=1.5, arm_offset_deg=1.0, cam_err=(0.005, 0.5), chassis_mass=0.9, odom=2.0, lidar_noise=2.0, depth_noise=1.5), pad=0.7),
]
CACHE = '../results/.sens_cache'


def _one(args):
    name, seed = args
    st = next(s for s in SETTINGS if s['name'] == name)
    fn = f'{CACHE}/{name}_{seed}.json'
    if os.path.exists(fn): return json.load(open(fn))
    if name == 'nominal' and os.path.exists('../results/trials.json'):     # same code, same seeds: the main batch's standard runs
        for e in json.load(open('../results/trials.json'))['episodes']:
            if e['condition'] == 'standard' and e['seed'] == seed:
                return dict(e, setting='nominal')
    res, agent, robot = run_episode(seed, 0.30, pad_friction=st['pad'], perturbation=st['pert'])
    res['setting'] = name; res['log'] = agent.log_lines
    json.dump(res, open(fn, 'w'), ensure_ascii=False)
    return res


if __name__ == '__main__':
    os.makedirs(CACHE, exist_ok=True)
    jobs = [(s['name'], seed) for s in SETTINGS for seed in SEEDS]
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2, maxtasksperchild=1) as pool:
        out = []
        for r in pool.imap_unordered(_one, jobs):
            out.append(r); print(r['setting'], r['seed'], f"{r['delivered']}/{r['total']}", r['sim_time'], flush=True)
    out.sort(key=lambda r: (r['setting'], r['seed']))
    json.dump(dict(settings=[{k: v for k, v in s.items()} for s in SETTINGS], episodes=out), open('../results/sensitivity.json', 'w'), ensure_ascii=False, indent=1)
    shutil.rmtree(CACHE)
