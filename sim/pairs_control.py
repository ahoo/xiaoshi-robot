"""Control for `python test_home.py pairs` (acceptance item E2): the same 256 random placements, but only the grown-up is there. How often is the grown-up found at all?

In the pairs bench someone is found next to the pair in only 71% of the trials; this shows that is a property of the bench (random robot pose among the furniture, a 2 s look,
confirmation needs two frames) and not of the pair: a lone grown-up is found in 67% of the same placements. Single-person detection is measured in `people_static` (acceptance E1).

  PROCS=4 python pairs_control.py
"""
import sys, math, multiprocessing as mp, numpy as np
sys.path.insert(0, '/home/user/xiaoshi-robot/sim')
def job(args):
    seed, reps = args
    import mujoco, people as P
    from robot import Robot
    from episode import _furn_dist
    from test_home import _Stub, _teleport
    rng = np.random.default_rng(seed); names = ['adult_m']
    r = Robot([], seed=seed, people=names, head_cam=True, robot_pose=(2.0, 1.5, 0.0)); stub = _Stub(r)
    scripts = [P.Script([])]; ps = P.People(r, names, scripts); det = P.PersonDetector(r, stub, np.random.default_rng(seed + 3)); out = []
    for gap in (0.10, 0.25, 0.45, 0.70):
        for arr in ('side', 'inline'):
            for rep in range(reps):
                for _try in range(3000):
                    rx, ry, yaw = rng.uniform(0.9, 3.0), rng.uniform(0.7, 2.3), rng.uniform(-math.pi, math.pi); d = rng.uniform(1.6, 2.6)
                    ux, uy = math.cos(yaw), math.sin(yaw); px, py = rx + d * ux, ry + d * uy
                    off = (0.23 + 0.16 + gap)
                    cx, cy = (px - uy * off, py + ux * off) if arr == 'side' else (px - ux * (0.14 + 0.14 + gap), py - uy * (0.14 + 0.14 + gap))
                    if 0.3 < px < 3.7 and 0.3 < py < 2.7 and 0.3 < cx < 3.7 and 0.3 < cy < 2.7 and min(_furn_dist(x, y) for x, y in ((rx, ry), (px, py), (cx, cy))) >= 0.3: break
                else: continue
                scripts[0] = P.Script([('at', px, py, yaw + math.pi)]); ps.scripts = scripts
                ps.update(); mujoco.mj_forward(r.m, r.d)
                _teleport(r, rx, ry, yaw); mujoco.mj_forward(r.m, r.d); det.tracks = []; det._t = -9.0
                for _ in range(100): ps.update(); r.step(); det.process()
                tr = det.confirmed(2); ka = [k for k in tr if math.hypot(k.x - px, k.y - py) < 0.5]
                out.append((bool(ka), len(tr), d))
    return out
if __name__ == '__main__':
    with mp.Pool(4, maxtasksperchild=1) as p: res = p.map(job, [(300 + i, 8) for i in range(4)])
    rows = [x for o in res for x in o]; f = sum(1 for r in rows if r[0]); print('adult alone: found', f, '/', len(rows), f'({100 * f / len(rows):.0f}%)', ' no track at all:', sum(1 for r in rows if r[1] == 0))
