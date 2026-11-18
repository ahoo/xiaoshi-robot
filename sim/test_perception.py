"""Bench test of the RGB-D detector on objects it knows nothing about: each kind (random size and shape every trial) at
several distances and headings in front of the robot. The detector's measurements are compared with the true shape,
which is sampled from the simulator's geometry and used only here.  ->  results/perception_bench.json

  python test_perception.py

Reported per kind and distance: detection rate, and the errors of the quantities the grasp planner and the map use:
  near edge  - where the object begins along the line from the robot (the fixed jaw goes there)
  width      - extent across that line
  top        - height
  centre     - middle of the outline (what the map remembers)
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


def run(trials=6, dists=(0.235, 0.6, 1.0, 1.5)):
    rng = np.random.default_rng(0); out = {}
    rpose = (2.2, 1.05, 0.0)      # facing +x: open floor out to about 1.7 m
    for kind in O.KINDS:
        out[kind] = {}
        for dist in dists:
            det = 0; e_near = []; e_wid = []; e_top = []; e_ctr = []
            for t in range(trials):
                bearing = rng.uniform(-0.25, 0.25) if dist > 0.3 else rng.uniform(-0.1, 0.1)
                yaw = rng.uniform(-math.pi, math.pi); spec = O.sample(kind, rng)
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
            out[kind][str(dist)] = dict(detected=det, n=trials, near_edge_err_mm=mm(e_near, np.mean), near_edge_err_max_mm=mm(e_near, np.max), width_err_mm=mm(e_wid, np.mean),
                                        top_err_mm=mm(e_top, np.mean), centre_err_mm=mm(e_ctr, np.mean))
            o = out[kind][str(dist)]
            print(f'{O.TYPES[kind]["cn"]:<10} {dist:.3f} m  detected {det}/{trials}  near edge {o["near_edge_err_mm"]} mm (max {o["near_edge_err_max_mm"]})  width {o["width_err_mm"]} mm  top {o["top_err_mm"]} mm  centre {o["centre_err_mm"]} mm', flush=True)
    return out


if __name__ == '__main__':
    json.dump(run(), open('../results/perception_bench.json', 'w'), ensure_ascii=False, indent=1)
