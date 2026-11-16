"""Bench tests of the base physics. Each one has a real-robot counterpart in docs/validation.md.

  python test_base.py
"""
import math
import time

import numpy as np

from robot import Robot, CTRL_DT


def drive(vx, vy, wz, T=4.0, pose=(0.8, 2.05, 0.0), label=''):
    r = Robot([], robot_pose=pose)
    for _ in range(25): r.step()
    x0, y0, th0 = r.true_pose(); r.odom[:] = [x0, y0, th0]
    r.set_twist(vx, vy, wz)
    t0 = time.time(); tilt = 0; yaw_acc = 0; last = th0
    for _ in range(int(T / CTRL_DT)):
        r.step(); tilt = max(tilt, r.tilt())
        th = r.true_pose()[2]; yaw_acc += (th - last + math.pi) % (2 * math.pi) - math.pi; last = th
    x1, y1, th1 = r.true_pose()
    c, s = math.cos(th0), math.sin(th0)
    bx, by = (c * (x1 - x0) + s * (y1 - y0)) / T, (-s * (x1 - x0) + c * (y1 - y0)) / T
    odo_err = math.hypot(r.odom[0] - x1, r.odom[1] - y1); dist = math.hypot(x1 - x0, y1 - y0)
    print(f'{label:<10} cmd ({vx:+.2f},{vy:+.2f},{wz:+.2f}) -> got ({bx:+.3f},{by:+.3f},{yaw_acc / T:+.3f}) | '
          f'odometry error {100 * odo_err:.1f} cm over {dist:.2f} m | max tilt {math.degrees(tilt):.2f} deg | '
          f'wheel torque peak {r.peak_torque[:3].max():.2f} rms {np.sqrt(r.torque_sq[:3] / r.n_steps).max():.2f} N*m | {T / (time.time() - t0):.0f}x realtime')
    return dict(cmd=[vx, vy, wz], got=[bx, by, yaw_acc / T], odom_err_m=odo_err, dist_m=dist, tilt_deg=math.degrees(tilt), peak_wheel_torque=float(r.peak_torque[:3].max()))


def run_all():
    out = {}
    out['forward'] = drive(0.15, 0, 0, label='forward')
    out['sideways'] = drive(0, 0.15, 0, label='sideways')
    out['diagonal'] = drive(0.1, 0.1, 0, label='diagonal')
    out['spin'] = drive(0, 0, 0.8, label='spin')
    out['overspeed'] = drive(0.4, 0, 0, T=3, label='overspeed')
    return out


if __name__ == '__main__':
    import json
    json.dump(run_all(), open('../results/base_bench.json', 'w'), indent=1)
