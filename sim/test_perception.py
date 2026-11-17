"""Bench test of the RGB-D detector: each object at several distances and headings in front of the robot.
Reports detection rate, position error and heading error.  ->  results/perception_bench.json

  python test_perception.py
"""
import json
import math

import numpy as np

from model import OBJECTS
from perception import FrontCamera
from robot import Robot


def wrap_pi(a):          # heading of a long axis is only defined modulo 180 deg
    return (a + math.pi / 2) % math.pi - math.pi / 2


def run(trials=6):
    rng = np.random.default_rng(0); out = {}
    rpose = (2.2, 1.05, 0.0)      # facing +x: open floor out to about 1.7 m
    for kind in OBJECTS:
        out[kind] = {}
        for dist in (0.235, 0.6, 1.0, 1.5):
            det = 0; perr = []; yerr = []
            for t in range(trials):
                bearing = rng.uniform(-0.25, 0.25) if dist > 0.3 else rng.uniform(-0.1, 0.1)
                yaw = rng.uniform(-math.pi, math.pi)
                ox, oy = rpose[0] + dist * math.cos(bearing), rpose[1] + dist * math.sin(bearing)
                r = Robot([(kind, ox, oy, yaw)], seed=t, robot_pose=rpose)
                for _ in range(30): r.step()
                cam = FrontCamera(r, np.random.default_rng(t))
                d = cam.detect().get(0)
                if d is None: continue
                det += 1
                x, y, th = r.true_pose(); p = r.obj_pos(0)
                c, s = math.cos(th), math.sin(th)
                true_rel = np.array([c * (p[0] - x) + s * (p[1] - y), -s * (p[0] - x) + c * (p[1] - y)])
                perr.append(float(np.linalg.norm(d['rel'][:2] - true_rel)))
                if kind != 'ball':
                    sym = math.pi / 2 if kind == 'block' else math.pi
                    e = (d['yaw4'] - (r.obj_yaw(0) - th) + sym / 2) % sym - sym / 2
                    yerr.append(abs(e))
            out[kind][str(dist)] = dict(detected=det, n=trials, pos_err_mean=round(float(np.mean(perr)), 4) if perr else None,
                                        pos_err_max=round(float(np.max(perr)), 4) if perr else None,
                                        yaw_err_deg_mean=round(math.degrees(np.mean(yerr)), 1) if yerr else None)
            print(f'{OBJECTS[kind]["cn"]:<6} {dist:.3f} m  detected {det}/{trials}  pos err mean {1000 * np.mean(perr) if perr else float("nan"):.1f} mm '
                  f'max {1000 * np.max(perr) if perr else float("nan"):.1f} mm  heading err {math.degrees(np.mean(yerr)) if yerr else float("nan"):.1f} deg')
    return out


if __name__ == '__main__':
    json.dump(run(), open('../results/perception_bench.json', 'w'), ensure_ascii=False, indent=1)
