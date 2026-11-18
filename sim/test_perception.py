"""Bench test of the RGB-D detector on objects it knows nothing about: each kind (random size and shape every trial) at
several distances and headings in front of the robot. The detector's measurements are compared with the true shape,
which is sampled from the simulator's geometry and used only here.  ->  results/perception_bench.json

  python test_perception.py

Reported per kind and distance: detection rate, and the errors of the quantities the grasp planner and the map use:
  near edge  - where the object begins along the line from the robot (the fixed jaw goes there)
  width      - extent across that line
  top        - height
  centre     - middle of the outline (what the map remembers)
Then the mistakes of the segmentation itself:
  pairs      - two objects side by side at a given gap, 0.6 m ahead: seen as two, or merged into one
  furniture  - one object at a given clearance from the front of the shelf, 0.6 m ahead: still found?
  empty      - the robot looks around a room with no objects in it: anything reported is a false detection
"""
import json
import math

import mujoco
import numpy as np

import objects as O
from perception import FrontCamera
from robot import Robot

_S = np.array([[math.cos(a) * math.cos(b), math.sin(a) * math.cos(b), math.sin(b)] for a in np.linspace(0, 2 * math.pi, 24, endpoint=False) for b in np.linspace(-math.pi / 2, math.pi / 2, 9)])


def true_points(r, k):
    """Points on the surface of object k (world frame), sampled from its geoms."""
    m, d = r.m, r.d; out = []
    for g in range(m.ngeom):
        if m.body_rootid[m.geom_bodyid[g]] != r.obj_body[k]: continue
        t = m.geom_type[g]; sz = m.geom_size[g]; R = d.geom_xmat[g].reshape(3, 3); p = d.geom_xpos[g]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE: L = _S * sz[0]
        elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID: L = _S * sz
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE: L = np.vstack([_S * sz[0] + [0, 0, sz[1]], _S * sz[0] - [0, 0, sz[1]]])
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER: L = np.array([[sz[0] * math.cos(a), sz[0] * math.sin(a), s * sz[1]] for a in np.linspace(0, 2 * math.pi, 24, endpoint=False) for s in (-1, 1)])
        elif t == mujoco.mjtGeom.mjGEOM_BOX: L = np.array([[sx, sy, sz_] for sx in (-1, 1) for sy in (-1, 1) for sz_ in (-1, 1)]) * sz
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            i = m.geom_dataid[g]; L = m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]]
        else: continue
        out.append(L @ R.T + p)
    return np.vstack(out)


HARD_MATS = (('sock', 'black'), ('cube', 'black'), ('plush', 'black'), ('ball', 'shiny'), ('car', 'shiny'), ('bottle', 'clear'))


def run(trials=6, dists=(0.235, 0.6, 1.0, 1.5), kinds=None):
    rng = np.random.default_rng(0); out = {}
    rpose = (2.2, 1.05, 0.0)      # facing +x: open floor out to about 1.7 m
    for kind in (kinds or O.KINDS):
        mat = None
        if isinstance(kind, tuple): kind, mat = kind
        key = kind if mat is None else f'{kind}:{mat}'
        out[key] = {}
        for dist in dists:
            det = 0; e_near = []; e_wid = []; e_top = []; e_ctr = []
            for t in range(trials):
                bearing = rng.uniform(-0.25, 0.25) if dist > 0.3 else rng.uniform(-0.1, 0.1)
                yaw = rng.uniform(-math.pi, math.pi); spec = O.sample(kind, rng)
                if mat: spec['mat'] = mat
                ox, oy = rpose[0] + dist * math.cos(bearing), rpose[1] + dist * math.sin(bearing)
                r = Robot([(kind, ox, oy, yaw, spec)], seed=t, robot_pose=rpose)
                for _ in range(60): r.step()
                cam = FrontCamera(r, np.random.default_rng(t)); ds = [d for d in cam.detect() if not d['cut']]; cam.close()
                if not ds: continue
                d = ds[0]; det += 1
                x, y, th = r.true_pose(); c, s = math.cos(th), math.sin(th)
                W = true_points(r, 0); B = np.stack([c * (W[:, 0] - x) + s * (W[:, 1] - y), -s * (W[:, 0] - x) + c * (W[:, 1] - y)], 1)
                u = d['rel'][:2] / np.linalg.norm(d['rel'][:2]); v = np.array([-u[1], u[0]])
                e_near.append(float((d['hull'] @ u).min() - (B @ u).min()))
                e_wid.append(float(np.ptp(d['hull'] @ v) - np.ptp(B @ v)))
                e_top.append(float(d['top'] - W[:, 2].max()))
                tc = 0.5 * (B.min(0) + B.max(0)); e_ctr.append(float(np.linalg.norm(d['rel'][:2] - tc)))
            mm = lambda a, f: round(1000 * float(f(np.abs(a))), 1) if a else None
            out[key][str(dist)] = dict(detected=det, n=trials, near_edge_err_mm=mm(e_near, np.mean), near_edge_err_max_mm=mm(e_near, np.max), width_err_mm=mm(e_wid, np.mean),
                                        top_err_mm=mm(e_top, np.mean), centre_err_mm=mm(e_ctr, np.mean))
            o = out[key][str(dist)]
            print(f'{O.TYPES[kind]["cn"] + (":" + mat if mat else ""):<14} {dist:.3f} m  detected {det}/{trials}  near edge {o["near_edge_err_mm"]} mm (max {o["near_edge_err_max_mm"]})  width {o["width_err_mm"]} mm  top {o["top_err_mm"]} mm  centre {o["centre_err_mm"]} mm', flush=True)
    return out


def gap_between(r, i, j):
    A, B = true_points(r, i)[:, :2], true_points(r, j)[:, :2]
    return float(np.min(np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])))


def run_pairs(trials=40):
    """Two random objects side by side, 0.6 m ahead, at random small spacings; results grouped by the real
    edge-to-edge gap. 'apart': two detections, one centred on each object. 'merged': one detection spanning both."""
    rng = np.random.default_rng(1); rec = []
    rpose = (2.2, 1.05, 0.0)
    for t in range(trials):
        k1, k2 = rng.choice(O.KINDS, 2, replace=True); s1, s2 = O.sample(k1, rng), O.sample(k2, rng)
        sep = 0.75 * (s1['radius'] + s2['radius']) + rng.uniform(-0.01, 0.06)
        r = Robot([(k1, rpose[0] + 0.6, rpose[1] + sep / 2, rng.uniform(-math.pi, math.pi), s1),
                   (k2, rpose[0] + 0.6, rpose[1] - sep / 2, rng.uniform(-math.pi, math.pi), s2)], seed=t, robot_pose=rpose)
        for _ in range(60): r.step()
        gap = gap_between(r, 0, 1)
        cam = FrontCamera(r, np.random.default_rng(t)); ds = [d for d in cam.detect() if d['n_px'] >= 25]; cam.close()
        x, y, th = r.true_pose(); c, s_ = math.cos(th), math.sin(th)
        cen = []
        for i in (0, 1):
            B = true_points(r, i)[:, :2]; m = 0.5 * (B.min(0) + B.max(0))
            cen.append(np.array([c * (m[0] - x) + s_ * (m[1] - y), -s_ * (m[0] - x) + c * (m[1] - y)]))
        hit = [min((np.hypot(*(d['rel'][:2] - cc)) for d in ds), default=9) < 0.035 for cc in cen]
        res = 'apart' if all(hit) and len(ds) >= 2 else ('merged' if len(ds) >= 1 else 'missed')
        rec.append((gap, res))
    out = {}
    for lo, hi, name in ((0, 0.005, '<5mm'), (0.005, 0.015, '5-15mm'), (0.015, 0.03, '15-30mm'), (0.03, 1, '>30mm')):
        sel = [r_ for g, r_ in rec if lo <= g < hi]
        out[name] = dict(n=len(sel), apart=sel.count('apart'), merged=sel.count('merged'), missed=sel.count('missed'))
        print(f'pair gap {name:>8}: {len(sel)} trials, apart {sel.count("apart")}, merged {sel.count("merged")}, missed {sel.count("missed")}', flush=True)
    return out


def run_furniture(trials=8, clears=(0.02, 0.05, 0.10)):
    """One random object in front of the shelf (front face at y = 0.35), robot 0.6 m away facing it."""
    rng = np.random.default_rng(2); out = {}
    for cl in clears:
        found = 0
        for t in range(trials):
            k = rng.choice(O.KINDS); s = O.sample(k, rng); ox = rng.uniform(2.35, 2.85)
            oy = 0.35 + cl + 0.7 * s['radius']
            r = Robot([(k, ox, oy, rng.uniform(-math.pi, math.pi), s)], seed=t, robot_pose=(ox, oy + 0.6, -math.pi / 2))
            for _ in range(60): r.step()
            cam = FrontCamera(r, np.random.default_rng(t)); ds = [d for d in cam.detect() if d['n_px'] >= 25]; cam.close()
            x, y, th = r.true_pose(); c, s_ = math.cos(th), math.sin(th); p = r.obj_pos(0)
            rel = np.array([c * (p[0] - x) + s_ * (p[1] - y), -s_ * (p[0] - x) + c * (p[1] - y)])
            if any(np.hypot(*(d['rel'][:2] - rel)) < 0.06 for d in ds): found += 1
        out[str(cl)] = dict(n=trials, found=found)
        print(f'{1000 * cl:.0f} mm from the shelf: found {found}/{trials}', flush=True)
    return out


def run_empty(poses=24):
    """No objects at all; the robot looks from random places and headings. Every detection is a false one."""
    rng = np.random.default_rng(3); n_false = 0; looks = 0; basket = 0
    from model import FURNITURE, ROOM_H, ROOM_W
    while looks < poses:
        x, y = rng.uniform(0.3, ROOM_W - 0.3), rng.uniform(0.3, ROOM_H - 0.3)
        if any(abs(x - f[1]) < (f[3] / 2 if f[4] else f[3]) + 0.25 and abs(y - f[2]) < (f[4] / 2 if f[4] else f[3]) + 0.25 for f in FURNITURE): continue
        r = Robot([], seed=looks, robot_pose=(x, y, rng.uniform(-math.pi, math.pi)))
        for _ in range(20): r.step()
        cam = FrontCamera(r, np.random.default_rng(looks)); ds = [d for d in cam.detect() if d['n_px'] >= 25 and not d['cut']]
        basket += cam.basket_obs is not None; cam.close()
        n_false += len(ds); looks += 1
    print(f'empty room: {n_false} false detections in {looks} looks (basket recognised in {basket})', flush=True)
    return dict(looks=looks, false_detections=n_false, basket_recognised=basket)


if __name__ == '__main__':
    res = run()
    res['_pairs'] = run_pairs(); res['_furniture'] = run_furniture(); res['_empty'] = run_empty()
    res['_materials'] = run(trials=6, dists=(0.235, 0.6, 1.0), kinds=HARD_MATS)
    json.dump(res, open('../results/perception_bench.json', 'w'), ensure_ascii=False, indent=1)
