"""Arm skills written as generators: each `yield` is one 20 ms control tick, so they can be driven by any loop."""
import math

import numpy as np

from arm import ArmIK
from model import P
from robot import GRIP_CLOSED, Q_CARRY

IK = ArmIK()
PINCH_MIN_Z = 0.025        # keeps the fingertips clear of the floor
HOLD_MIN = GRIP_CLOSED + 0.03   # jaw stopped short of its end stop (~8 mm gap or more): something is between the jaws, even a squashed soft one
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


ELONG = 1.3            # outline long/short ratio above which an object is treated as long (grip across its narrow side)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def radial(det):
    """Unit vector from the arm's pan axis to the object's outline centre (chassis frame): the line the jaws close along."""
    c = det['rel']; u = np.array([c[0] - 0.069, c[1]]); return u / np.linalg.norm(u)


def misalignment(det):
    """How far (rad) a long object's long axis is from square to the jaw line. The jaws always close along the line
    from the shoulder to the object (5-axis arm, no wrist yaw), so the base has to fix this by standing elsewhere.
    Objects that are not long (outline ratio under ELONG) can be taken from any side: 0."""
    if det['elong'] < ELONG: return 0.0
    u = radial(det)
    return abs(wrap(2 * (det['heading'] - math.atan2(u[1], u[0]) - math.pi / 2))) / 2


BAND = 0.012          # half width of the strip the fingers close on (pads are 14 mm wide, plus some slack)


def best_band(pts, u):
    """Where to close the jaws on an object of unknown shape. The jaws close along u; try strips across the object at
    lateral offsets 1 cm apart, measure how much of the object lies inside each strip along u, and pick the narrowest
    strip that fits the gripper, preferring strips near the middle (balance). Returns (near, far, offset, object width
    across u, score) or None; lower score is better. pts: points seen on the object, chassis frame, N x 3 (z up from the floor)."""
    v = np.array([-u[1], u[0]])
    t = pts[:, :2] @ u; s = pts[:, :2] @ v; z = pts[:, 2]
    lo, hi = np.percentile(s, [2, 98]); mid = np.median(s)
    best = None
    for off in np.arange(lo + BAND / 2, hi - BAND / 2 + 1e-9, 0.01) if hi - lo > BAND else [0.5 * (lo + hi)]:
        m = np.abs(s - off) < BAND
        if m.sum() < 8: continue
        near, far = np.percentile(t[m], [1, 99])
        # The camera sees the object from the robot's side, so the far side of anything rounded is hidden and the far
        # edge comes out short. The highest part of the strip sits above its middle (flat tops: the whole top face),
        # so mirror the near edge about it.
        tz = t[m][z[m] >= z[m].max() - 0.003].mean()
        far = max(far, 2 * tz - near)
        w = far - near
        if w < 0.006 or w > 0.068: continue
        score = w + 1.2 * abs(off - mid)        # off-centre strips must be clearly narrower to win (a ball's edge is narrow too)
        if best is None or score < best[0]: best = (score, float(near), float(far), float(off))
    if best is None: return None
    return best[1], best[2], best[3], float(hi - lo), best[0]


def plan_grasp(det):
    """det: a detection from perception (chassis frame). Returns joint waypoints and jaw angles, or None.

    Nothing about the object is known except what the camera measured: its outline seen from above and its height.

    Two constraints of this hardware shape the plan:
    * The jaw swings on a hinge, so the two faces form a V. The gripper is tilted by half the contact angle so the V
      is symmetric about vertical; both fingertips then reach the same height, which is what lets it pick up flat things.
    * The arm has 5 axes and no wrist yaw, so that tilt can only happen in the arm's own vertical plane. The jaws
      therefore always close along the line from the shoulder to the object, and it is the base's job to stand where
      that line crosses the object's narrow side."""
    top = det['top']
    u2 = radial(det); v2 = np.array([-u2[1], u2[0]])
    band = best_band(det['pts'], u2)
    if band is None:
        return None
    near, far, off, _, _ = band
    ext = far - near + 0.004
    if ext + CLEAR + 0.02 > 0.10:
        return None
    depth = 0.075 if (top > 0.045 and det['elong'] < ELONG) else 0.0626   # tall round things: hold nearer the fingertips
    q_c = jaw_angle_for_gap(ext, depth)                  # jaw angle when both faces touch the object
    beta = q_c / 2
    zg = min(max(0.5 * top, PINCH_MIN_Z), max(PINCH_MIN_Z, top - 0.006))
    cxy = (near + ext / 2) * u2 + off * v2
    c = np.array([cxy[0], cxy[1], zg])
    u = np.array([u2[0], u2[1], 0.0])
    yaw = math.atan2(u[1], u[0])
    ap = -math.sin(beta) * u - math.cos(beta) * np.array([0, 0, 1.0])
    q, e, pe = IK.solve(c, approach=ap, yaw=yaw)
    if pe > 0.006:
        return None
    _, a, xg = IK.fk(q)
    if xg @ u < 0:                                       # fixed jaw must be on the robot side
        return None
    touch = c - (ext / 2) * xg - (depth - 0.0626) * a    # pinch site (on the fixed jaw face) when that face touches the near side
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
    return dict(approach=approach, grasp=q_grasp, touch=q_touch, lift=lift, open=jaw_angle_for_gap(ext + CLEAR + 0.02, 0.08), expect=q_c, width=ext)


def pinch_now(robot):
    """Where the pinch point is now (chassis frame), from joint feedback."""
    return IK.fk(robot.joints()[:5])[0]


def grasp(robot, plan, log=None, look=None):
    """Executes a planned top-down grasp. Returns True if it holds the object after lifting.

    The jaw angle tells whether something props the jaws open. A soft object squeezed thinner than the jaw's end stop
    (a plush toy's arm, a sock) leaves the jaw at the stop, exactly like an empty gripper. So when the jaw closes to the
    stop, the arm lifts anyway and look() (the front camera) checks whether anything hangs in the gripper."""
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
    closed_full = robot.joints()[5] < HOLD_MIN
    if (closed_full and look is None) or robot.tilt() > TILT_ABORT:
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
        if look is not None and look():
            plan['closed_hold'] = True
            if log: log('夹爪合到了底，但摄像头看到夹爪下挂着东西：是软的，被压扁了')
            return True
        if log: log('夹爪合拢到底，没夹到东西' if closed_full else '抬起时物体滑脱')
        return False
    return True


def carry_pose(robot, speed=0.6):
    robot.set_arm(q5=Q_CARRY, speed=speed); yield from wait_arm(robot)


def place(robot, rel_xy, rim_z, log=None, peek=None):
    """Reach over the basket rim at rel_xy (chassis frame), lower, open, retreat. A squashed soft object can stay
    stuck to the jaws after they open. peek() (front camera) gives the top of whatever is under the gripper; it is read
    before and after lifting the open gripper 3 cm, and if it rose with the gripper, the gripper shakes it off."""
    r = math.hypot(rel_xy[0], rel_xy[1])
    hi = np.array([rel_xy[0], rel_xy[1], rim_z + 0.10]); lo = np.array([rel_xy[0], rel_xy[1], rim_z + 0.045])
    q_hi, e1, p1 = IK.solve(hi, approach=(0.5 * rel_xy[0] / r, 0.5 * rel_xy[1] / r, -1))
    q_lo, e2, p2 = IK.solve(lo, approach=(0.5 * rel_xy[0] / r, 0.5 * rel_xy[1] / r, -1), seed=q_hi)
    q_mid, _, _ = IK.solve(lo + [0, 0, 0.03], approach=(0.5 * rel_xy[0] / r, 0.5 * rel_xy[1] / r, -1), seed=q_lo)
    if max(p1, p2) > 0.01:
        if log: log('放置点够不到')
        return False
    robot.set_arm(q5=q_hi, speed=1.0); yield from wait_arm(robot)
    robot.set_arm(q5=q_lo, speed=0.6); yield from wait_arm(robot)
    robot.set_arm(grip=1.0); yield from wait(robot, 0.7)
    for _ in range(3 if peek is not None else 0):
        t0 = peek()
        robot.set_arm(q5=q_mid, speed=0.6); yield from wait_arm(robot)
        t1 = peek()
        # It let go if nothing was hanging there, or whatever was there stayed put. Something that rose with the
        # gripper, or rose out of the picture, is stuck to the jaws.
        if t0 == 0.0 or (0.0 < t1 < t0 + 0.02): break
        if log: log('松手后东西还粘在夹爪上，抖一抖')
        for g_, roll in ((0.2, 0.6), (1.6, -0.6), (1.6, 0.6), (1.6, 0.0)):
            q = q_lo.copy(); q[4] += roll
            robot.set_arm(q5=q, grip=g_, speed=2.0); yield from wait_arm(robot, timeout=1.5)
        yield from wait(robot, 0.5)
    robot.set_arm(q5=q_hi, speed=1.0); yield from wait_arm(robot)
    robot.set_arm(q5=Q_CARRY, grip=0.3, speed=1.2); yield from wait_arm(robot)
    return True


def holding(robot, plan):
    """Servo feedback check: the jaw is still propped open by something. After a hold that was confirmed by the camera
    with the jaw at its stop, the jaw angle says nothing: assume it is still there (a drop is found later by looking)."""
    return plan.get('closed_hold', False) or robot.joints()[5] > HOLD_MIN
