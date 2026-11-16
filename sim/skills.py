"""Arm skills written as generators: each `yield` is one 20 ms control tick, so they can be driven by any loop."""
import math

import numpy as np

from arm import ArmIK
from model import OBJECTS, OBJ_HALF_H, P
from robot import GRIP_CLOSED, Q_CARRY

IK = ArmIK()
PINCH_MIN_Z = 0.025        # keeps the fingertips clear of the floor
HOLD_MIN = GRIP_CLOSED + 0.10   # jaw angle above this means something at least ~1 cm thick is between the jaws
SOFT_GRIP = 0.4               # N*m while closing on the floor
TILT_ABORT = 0.07             # rad of chassis tilt at which a grasp is abandoned
CLEAR = 0.007              # gap left between the fixed jaw and the object before closing


def jaw_gap(q, depth=0.0626):
    """Opening between the jaw faces (m) at `depth` below the jaw hinge, for moving-jaw angle q."""
    return 0.0292 - 0.0122 * math.cos(q) + depth * math.sin(q)


def jaw_angle_for_gap(gap, depth=0.0626):
    for q in np.linspace(-0.17, 1.3, 300):
        if jaw_gap(q, depth) >= gap:
            return float(q)
    return 1.3


def wait(robot, seconds):
    for _ in range(int(seconds / 0.02)):
        yield


def wait_arm(robot, timeout=4.0, tol=0.03):
    for _ in range(int(timeout / 0.02)):
        yield
        if robot.arm_done(tol):
            return True
    return False


def to_chassis(robot, p_world, pose):
    """World point -> chassis frame (z stays height above floor). `pose` is the pose estimate the caller trusts."""
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    dx, dy = p_world[0] - x, p_world[1] - y
    return np.array([c * dx + s * dy, -s * dx + c * dy, p_world[2]])


def misalignment(rel, yaw_rel, kind):
    """Angle (rad) between the jaw closing line and the object's best grasp axis. The jaws always close along the
    line from the shoulder to the object (5-axis arm, no wrist yaw), so this is what the base has to fix by standing elsewhere.
    Long objects should be gripped across their narrow side (symmetry 180 deg); the cube across any face (symmetry 90 deg)."""
    if kind == 'ball':
        return 0.0
    radial = math.atan2(rel[1], rel[0] - 0.069)
    if kind == 'block':
        m = abs(wrap(4 * (yaw_rel - radial))) / 4
    else:
        m = abs(wrap(2 * (yaw_rel + math.pi / 2 - radial))) / 2
    return m


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def plan_grasp(rel, kind, mis=0.0):
    """rel: object centre in the chassis frame (z = height above floor). Returns joint waypoints and jaw angles, or None.

    Two constraints of this hardware shape the plan:
    * The jaw swings on a hinge, so the two faces form a V. The gripper is tilted by half the contact angle so the V
      is symmetric about vertical; both fingertips then reach the same height, which is what lets it pick up flat things.
    * The arm has 5 axes and no wrist yaw, so that tilt can only happen in the arm's own vertical plane. The jaws
      therefore always close along the line from the shoulder to the object, and it is the base's job to stand where
      that line crosses the object's narrow side."""
    o = OBJECTS[kind]; w = o['width']
    # `mis` = angle between the jaw line and the object's narrow axis. A long object that is not square to the jaws
    # is wider across them, so the jaws must open for that width or they come down on top of it.
    if kind == 'ball':
        w_open = w
    elif kind == 'block':
        w_open = w * (math.cos(mis) + math.sin(mis))           # a cube turned by `mis` is wider across the jaws
    else:
        w_open = w * math.cos(mis) + o['length'] * abs(math.sin(mis))
    if w_open + CLEAR + 0.02 > 0.10:
        return None
    depth = 0.075 if kind == 'ball' else 0.0626          # a ball is held nearer the fingertips, where the jaws open wider
    q_c = jaw_angle_for_gap(w, depth)                    # jaw angle when both faces touch the object
    beta = q_c / 2
    c = np.array([rel[0], rel[1], max(rel[2], PINCH_MIN_Z)])
    u = np.array([rel[0] - 0.069, rel[1], 0.0]); u /= np.linalg.norm(u)      # radial direction from the pan axis
    yaw = math.atan2(u[1], u[0])
    ap = -math.sin(beta) * u - math.cos(beta) * np.array([0, 0, 1.0])
    q, e, pe = IK.solve(c, approach=ap, yaw=yaw)
    if pe > 0.006:
        return None
    _, a, xg = IK.fk(q)
    if xg @ u < 0:                                       # fixed jaw must be on the robot side
        return None
    touch = c - (w_open / 2) * xg - (depth - 0.0626) * a      # pinch site (on the fixed jaw face) when that face touches the object
    touch[2] = max(touch[2], PINCH_MIN_Z)
    back = touch - CLEAR * u
    q_touch, e0, p0 = IK.solve(touch, approach=ap, yaw=yaw, seed=q)
    q_grasp, e1, p1 = IK.solve(back, approach=ap, yaw=yaw, seed=q_touch)
    errs = [e0, e1]; perr = [p0, p1]
    # come in along the gripper's own axis in short steps, so the fixed jaw slides down past the object instead of landing on it
    approach, seed = [], q_grasp
    for dist in (0.025, 0.05, 0.085):
        seed, e, p = IK.solve(back - dist * a, approach=ap, yaw=yaw, seed=seed); approach.insert(0, seed); errs.append(e); perr.append(p)
    lift, seed = [], q_touch
    for dz in (0.03, 0.08):
        seed, e, p = IK.solve(touch + [0, 0, dz], approach=ap, yaw=yaw, seed=seed); lift.append(seed); errs.append(e); perr.append(p)
    if max(perr) > 0.004 or max(errs) > 0.03:
        return None
    return dict(approach=approach, grasp=q_grasp, touch=q_touch, lift=lift, open=jaw_angle_for_gap(w_open + CLEAR + 0.02, 0.08), expect=q_c)


def grasp(robot, plan, log=None):
    """Executes a planned top-down grasp. Returns True if the jaw stopped on something and still holds it after lifting."""
    robot.set_arm(grip=plan['open'], speed=1.2)
    robot.set_arm(q5=plan['approach'][0]); yield from wait_arm(robot)
    for q, speed in [(q, 0.6) for q in plan['approach'][1:] + [plan['grasp']]] + [(plan['touch'], 0.3)]:   # last step slides the fixed jaw up to the object
        robot.set_arm(q5=q, speed=speed)
        for _ in range(200):
            yield
            if robot.tilt() > 0.045:                 # the arm is pressing on something and lifting the chassis: back off
                if log: log('手臂压到东西了，抬起重来')
                robot.set_arm(q5=plan['approach'][0], speed=1.0); yield from wait_arm(robot)
                return False
            if robot.arm_done(0.015): break
    yield from wait(robot, 0.15)
    # close gently while the object is still on the floor: at full torque the jaw can press down hard enough to
    # lever this 2.3 kg robot off its front wheels
    robot.set_grip_torque(SOFT_GRIP); robot.set_arm(grip=GRIP_CLOSED)
    last = 9.0
    for _ in range(100):                        # close until the jaw stalls
        yield
        q = robot.joints()[5]
        if (abs(q - last) < 0.0006 and robot.arm_cmd[5] <= GRIP_CLOSED + 0.02) or robot.tilt() > TILT_ABORT:
            break
        last = q
    if robot.joints()[5] < HOLD_MIN or robot.tilt() > TILT_ABORT:
        if log: log('机身被顶起，松手重来' if robot.tilt() > TILT_ABORT else '夹爪合拢到底，没夹到东西')
        robot.set_arm(grip=0.5); yield from wait(robot, 0.3)
        robot.set_grip_torque(P['gripper_torque'])
        robot.set_arm(q5=plan['lift'][1], grip=0.3, speed=1.0); yield from wait_arm(robot)
        return False
    robot.set_grip_torque(P['gripper_torque'])   # both jaws are on the object now: squeeze properly before lifting
    for _ in range(15):
        yield
        if robot.tilt() > TILT_ABORT:            # the jaw is sitting on top of the object and pushing the robot up
            if log: log('机身被顶起，松手重来')
            robot.set_arm(grip=0.5); yield from wait(robot, 0.3)
            robot.set_arm(q5=plan['lift'][1], grip=0.3, speed=1.0); yield from wait_arm(robot)
            return False
    for q in plan['lift']:
        robot.set_arm(q5=q, speed=0.4); yield from wait_arm(robot)
    yield from wait(robot, 0.3)
    if robot.joints()[5] < HOLD_MIN:
        if log: log('抬起时物体滑脱')
        return False
    return True


def carry_pose(robot, speed=0.6):
    robot.set_arm(q5=Q_CARRY, speed=speed); yield from wait_arm(robot)


def place(robot, rel_xy, rim_z, log=None):
    """Reach over the basket rim at rel_xy (chassis frame), lower, open, retreat."""
    r = math.hypot(rel_xy[0], rel_xy[1])
    hi = np.array([rel_xy[0], rel_xy[1], rim_z + 0.10]); lo = np.array([rel_xy[0], rel_xy[1], rim_z + 0.045])
    q_hi, e1, p1 = IK.solve(hi, approach=(0.5 * rel_xy[0] / r, 0.5 * rel_xy[1] / r, -1))
    q_lo, e2, p2 = IK.solve(lo, approach=(0.5 * rel_xy[0] / r, 0.5 * rel_xy[1] / r, -1), seed=q_hi)
    if max(p1, p2) > 0.01:
        if log: log('放置点够不到')
        return False
    robot.set_arm(q5=q_hi, speed=1.0); yield from wait_arm(robot)
    robot.set_arm(q5=q_lo, speed=0.6); yield from wait_arm(robot)
    robot.set_arm(grip=1.0); yield from wait(robot, 0.7)
    robot.set_arm(q5=q_hi, speed=1.0); yield from wait_arm(robot)
    robot.set_arm(q5=Q_CARRY, grip=0.3, speed=1.2); yield from wait_arm(robot)
    return True


def holding(robot, plan):
    """Servo feedback check: the jaw is still propped open by something."""
    return robot.joints()[5] > HOLD_MIN
