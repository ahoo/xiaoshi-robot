"""Docking with the charger, as generators (one `yield` per 20 ms control tick, like skills.py).

The dock is a wall-mounted plate with one wide contact strip (model.DOCK). The robot carries a charging probe on its rear. It drives to a
pre-dock pose in front of the dock with the ordinary planner, then reverses along the dock axis, steering with its own pose estimate, until the
probe touches the strip (the charge-detect signal of the charger board: robot.dock_connected()). The strip is 24 cm wide, so the pose estimate
has to be good to a few centimetres and a few degrees, not to millimetres.
"""
import math

import numpy as np

from model import DOCK, DOCK_POSE
from skills import wait

PRE_DIST = 0.40          # m: the pre-dock pose is this far in front of the docked pose, on the dock axis
V_REV, V_REV_SLOW = 0.06, 0.03   # m/s: reversing speed, and the last 8 cm before the strip
SLOW_ZONE = 0.08         # m
OVERRUN = 0.05           # m: no contact this far past where the strip should have been: give up this attempt
UNDOCK_SPEED = 0.08


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def axis():
    """Unit vector from the dock out into the room (the direction the robot drives away along)."""
    th = DOCK_POSE[2]
    return np.array([math.cos(th), math.sin(th)])


def predock_pose(dist=PRE_DIST):
    a = axis()
    return (DOCK_POSE[0] + dist * a[0], DOCK_POSE[1] + dist * a[1], DOCK_POSE[2])


def _along(est):
    """(distance of the robot centre from the docked pose along the axis, lateral offset from the axis), from a pose estimate."""
    a = axis(); d = np.array([est[0] - DOCK_POSE[0], est[1] - DOCK_POSE[1]])
    return float(d @ a), float(-d[0] * a[1] + d[1] * a[0])


def reverse_in(agent, timeout=40.0):
    """Reverse along the dock axis until the probe touches the strip. Returns True once the contact has held for a second. The safety layer is not
    used here (the plate it would keep away from is the thing to touch); the motion is slow, straight, and stops on contact."""
    r = agent.r; t0 = r.t; touched = None; a = axis(); agent.state = '对接'
    while r.t - t0 < timeout:
        est = agent.pose()
        along, lat = _along(est)                       # along > 0: still in front of the docked pose
        if touched is None:
            if r.dock_connected(hold=0.0): touched = r.t
            elif along < -OVERRUN or math.degrees(r.tilt()) > 6.0:
                agent.stop(); agent.log('倒车过了头还没碰到触片，这次作废' if along < -OVERRUN else '对接时机身倾斜，停下'); return False
        if touched is not None:
            agent.stop()
            if r.t - touched > 1.0: return r.dock_connected()
            yield; continue
        v = V_REV_SLOW if along < SLOW_ZONE else V_REV
        # world velocity: back along the axis, plus a lateral correction toward it; heading held at the dock heading
        lat_dir = np.array([-a[1], a[0]])
        vw = -v * a + float(np.clip(-1.0 * lat, -0.04, 0.04)) * lat_dir
        c, s = math.cos(est[2]), math.sin(est[2])
        r.set_twist(c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1], float(np.clip(2.2 * wrap(DOCK_POSE[2] - est[2]), -0.5, 0.5)))
        yield
    agent.stop(); agent.log('倒车超时'); return False


def pull_out(agent, dist=PRE_DIST - 0.05, timeout=15.0):
    """Drive forward off the dock, straight along the axis."""
    r = agent.r; t0 = r.t; agent.state = '离开充电座'
    start = _along(agent.pose())[0]
    while r.t - t0 < timeout:
        along, lat = _along(agent.pose())
        if along - start >= dist: break
        agent.cmd(UNDOCK_SPEED, 0.0, float(np.clip(2.2 * wrap(DOCK_POSE[2] - agent.pose()[2]), -0.4, 0.4)))
        yield
    agent.stop(); yield from wait(r, 0.3)
    return _along(agent.pose())[0] - start >= dist - 0.05


def dock(agent, tries=3):
    """The whole docking: pre-dock pose, reverse in, retry from the pre-dock pose if the contact was missed. Returns True when connected."""
    r = agent.r
    for k in range(tries):
        pre = predock_pose()
        agent.state = '去充电座'; agent.log('去充电座' + (f'（第 {k + 1} 次）' if k else ''))
        goal = yield from agent.goto(lambda D: [(pre[0], pre[1], pre[2], 0.0)], timeout=90.0)
        if goal is None:
            agent.log(f'去不了充电座前面（{agent.abort}）'); yield from wait(r, 2.0); continue
        ok = yield from reverse_in(agent)
        if ok:
            agent.log('对接成功，开始充电'); return True
        agent.log('对接没成功，前进离开再来'); yield from pull_out(agent, 0.15)
    return False
