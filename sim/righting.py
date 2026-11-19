"""Getting back on the wheels after a fall.

Detecting the fall is the supervisor's job (IMU tilt held beyond `fall_tilt`). What happens here, in order:
  1. stop, announce, and wait until the robot has been at rest for a while (someone lifting it must not get an arm swung at them);
  2. read the resting pose from the IMU's gravity direction and classify it (it comes to rest in one of a few stable poses: on its side, leaning
     on the front wheels and the open gripper, on its back);
  3. run the maneuver stored for that pose (arm joint targets and wheel speeds in a few phases, found by search in the simulator: see
     docs/validation.md and results/righting_search.json), look at the IMU, and try again, from whatever pose it is in then, up to `tries` times;
  4. on success tidy up the robot's beliefs: nothing is held any more, and the position is lost: find it again by matching the lidar scan against
     the whole map (Localizer.global_search);
  5. on failure give up: the supervisor puts the robot in FAULT (wheels and arm stopped) and a person is needed.
The maneuvers need a joint acceleration far above the one used for normal driving (the STS3215's acceleration register allows it); they are
the part of this that is least certain on real hardware (docs/validation.md).
"""
import json
import math
import os

import numpy as np

from robot import CTRL_DT, GRIP_CLOSED, Q_CARRY
from skills import IK, wait

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_ACC_RIGHT = 30.0         # rad/s^2 while a maneuver runs (8 for normal motion)
REST_FOR = 1.5               # s the IMU must have been steady before anything moves
WARN_S = 3.0                 # s between the announcement (beep, light) and the first movement; a disturbance restarts it
UPRIGHT_DEG = 15.0           # below this the robot counts as back on its wheels
CLASS_MIN_COS = 0.85         # closer than ~32 degrees to a stored rest pose, or it is 'unknown'
PRIOR_W = 0.04               # score taken off per metre between a relocalisation candidate and where the robot fell
FEASIBLE = 0.09              # m: the robot centre must be at least this far from every mapped obstacle (the chassis radius is 0.105; after a fall it may be touching furniture)


def load():
    fn = os.path.join(HERE, 'righting_maneuvers.json')
    return json.load(open(fn)) if os.path.exists(fn) else {'rest_poses': {}, 'maneuvers': {}}


LIB = load()


def ranked(up):
    """Stored rest poses, closest first, as (cosine of the angle between them and the measurement, name)."""
    up = np.asarray(up, float)
    return sorted(((float(up @ np.asarray(v) / np.linalg.norm(v)), k) for k, v in LIB['rest_poses'].items()), reverse=True)


def classify(up):
    """Name of the stored rest pose closest to the measured 'up' direction (body frame), or 'upright' / 'unknown'."""
    up = np.asarray(up, float)
    if up[2] > math.cos(math.radians(UPRIGHT_DEG)): return 'upright'
    best, name = -1.0, 'unknown'
    for k, v in LIB['rest_poses'].items():
        c = float(up @ np.asarray(v) / np.linalg.norm(v))
        if c > best: best, name = c, k
    return name if best >= CLASS_MIN_COS else 'unknown'


def _tilt(up): return math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))


def _rest(sup, seconds, jitter=3.0, timeout=90.0):
    """Wait until the IMU direction has stayed within `jitter` degrees for `seconds`. Returns the mean direction, or None if it never settles within
    `timeout` (somebody is carrying or shaking the robot: nothing is moved then)."""
    r = sup.r; ups = []; n = int(seconds / CTRL_DT); t0 = r.t
    while r.t - t0 < timeout:
        ups.append(r.imu()); yield
        if len(ups) > n:
            ups.pop(0); m = np.mean(ups, 0); m /= np.linalg.norm(m)
            if max(math.degrees(math.acos(max(-1.0, min(1.0, float(u @ m))))) for u in ups) < jitter: return m
    return None


def _run_maneuver(sup, phases):
    r = sup.r; r.arm_acc = ARM_ACC_RIGHT
    try:
        for p in phases:
            q = np.asarray(p['q'], float)
            r.arm_target[:4] = q; r.arm_target[4] = 0.0; r.arm_target[5] = GRIP_CLOSED; r.arm_speed = p['speed']; r.wheel_target[:] = p['w']
            for _ in range(int(p['T'] / CTRL_DT)): yield
        r.wheel_target[:] = 0.0; r.arm_target[:5] = Q_CARRY; r.arm_speed = 2.0
        yield from wait(r, 2.5)
    finally:
        r.arm_acc = 8.0


def recover(sup, tries=3):
    """The RIGHT mode. Leaves the robot upright and relocalised (returns normally), or sets the supervisor's FAULT."""
    r, a = sup.r, sup.a
    a.stop(); a.state = '起身'; r.wheel_target[:] = 0.0
    attempts = []; up = r.imu()
    for k in range(tries):
        up = yield from _rest(sup, REST_FOR)
        if up is None: attempts.append(('never still', None)); break
        cls = classify(up)
        sup._event(f'静止了，姿态：{cls}（倾角 {_tilt(up):.0f}°）')
        if cls == 'upright': break
        mans = LIB['maneuvers'].get(cls)
        if not mans and cls == 'unknown':                      # a pose nobody stored: try what suits the closest one
            cls = ranked(up)[0][1]; mans = LIB['maneuvers'].get(cls); sup._event(f'没见过的姿态，按最接近的「{cls}」试试')
        if not mans: attempts.append((cls, 'no maneuver')); break
        # announce, then wait out a calm WARN_S: anything that moves the robot meanwhile (a hand) starts the count again
        sup._event(f'{WARN_S:.0f} 秒后开始起身（第 {k + 1} 次，{cls}）')
        t_ok = 0.0
        while t_ok < WARN_S:
            t_ok = t_ok + CTRL_DT if abs(_tilt(r.imu()) - _tilt(up)) < 3.0 else 0.0
            yield
        yield from _run_maneuver(sup, mans[min(k, len(mans) - 1)]['phases'])
        up = r.imu(); attempts.append((cls, round(_tilt(up), 1)))
        if _tilt(up) < UPRIGHT_DEG: break
    if _tilt(r.imu()) >= UPRIGHT_DEG:
        sup.stats['right_failed'] += 1; sup._event(f'起不来：{attempts}'); sup._fault('fallen, could not get up'); return
    sup.stats['righted'] += 1; sup._event(f'站起来了（{attempts}）')
    yield from wait(r, 1.0)
    yield from relocalize(sup)


def forget_what_was_in_hand(a):
    for k in [k for k, e in a.memory.items() if e['status'] == 'held']: del a.memory[k]
    a.held = False; a.target = None; a.priority = None; a.path = None; a.slip_flag = False; a.ghosts = []


def _wrap(x): return (x + math.pi) % (2 * math.pi) - math.pi


def _compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2]); return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], _wrap(a[2] + b[2])])


def _inv(a):
    c, s = math.cos(a[2]), math.sin(a[2]); return np.array([-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), _wrap(-a[2])])


def _scan(sup):
    r, a = sup.r, sup.a
    ang, rr = r.lidar(); rr = a._scan_filter(ang, rr); f = np.isfinite(rr)
    return -0.07 + rr[f] * np.cos(ang[f]), rr[f] * np.sin(ang[f]), (float(rr[f].min()) if f.any() else 9.0)


def _best_consistent(views, pos_tol=0.15, head_tol=math.radians(8)):
    """views: [(odometry pose when the scan was taken, candidates [(score, x, y, th), ...])]. The latest view's candidates are tested against every earlier
    one: the earlier candidate moved on by the odometry between the two must land on the later one. Returns (best, runner_up) as (summed score, later candidate)."""
    j = len(views) - 1; odj, cj = views[j]; found = []
    for i in range(j):
        odi, ci = views[i]; delta = _compose(_inv(odi), odj)
        for s_i, *p_i in ci:
            pred = _compose(np.array(p_i), delta)
            for s_j, *p_j in cj:
                if math.hypot(pred[0] - p_j[0], pred[1] - p_j[1]) < pos_tol and abs(_wrap(pred[2] - p_j[1 + 1])) < head_tol: found.append((s_i + s_j, (s_j, *p_j)))
    found.sort(key=lambda t: -t[0])
    best = found[0] if found else None
    rival = next((f for f in found[1:] if math.hypot(f[1][1] - best[1][1], f[1][2] - best[1][2]) > 0.3 or abs(_wrap(f[1][3] - best[1][3])) > math.radians(30)), None) if best else None
    return best, rival


def _new_view(sup):
    """Move a little so the next scan is taken from somewhere else (known by the odometry): a turn on the spot if the safety layer allows it, else a step away."""
    r, a = sup.r, sup.a
    for cmd in ((0.0, 0.0, 0.8), (0.0, 0.0, -0.8), (0.07, 0.0, 0.0), (-0.07, 0.0, 0.0), (0.0, 0.07, 0.0), (0.0, -0.07, 0.0)):
        o0 = r.odom.copy(); t0 = r.t
        while r.t - t0 < 2.0:
            a.cmd(*cmd); yield
            if abs(_wrap(r.odom[2] - o0[2])) > 1.4 or math.hypot(*(r.odom[:2] - o0[:2])) > 0.12: break
        a.stop(); yield from wait(r, 0.3)
        if abs(_wrap(r.odom[2] - o0[2])) > 0.35 or math.hypot(*(r.odom[:2] - o0[:2])) > 0.05: return True
    return False


def relocalize(sup, views=5, min_score=0.55, margin=0.06, max_jump=1.2):     # max_jump: radius of the first, near search
    """After a fall the odometry says nothing about where the robot is. Find the pose by matching lidar scans against the whole map, looking first
    near where it was; one scan can fit several places in a room with so little in it, so it moves and looks again, and takes the pose only when two
    views agree with each other through the odometry between them and no other pair does about as well.
    A wrong pose is worse than none (it corrupts the map), so if the views never settle it, the robot stays put and asks for help."""
    r, a = sup.r, sup.a
    forget_what_was_in_hand(a); a.arm_busy = False
    r.arm_target[:5] = Q_CARRY; r.arm_target[5] = 0.3; r.arm_speed = 1.6; yield from wait(r, 1.5)
    a.state = '重新定位'; prior = a.pose_before_fall if a.pose_before_fall is not None else a.loc.est.copy()
    seen = []; D = a.costmap()
    for k in range(views):
        px, py, rmin = _scan(sup)
        near = a.loc.global_search(px, py, center=prior[:2], radius=max_jump) or []           # near where it fell first, then anywhere on the map
        anywhere = a.loc.global_search(px, py) or []
        cands = sorted(near + [c for c in anywhere if all(math.hypot(c[1] - o[1], c[2] - o[2]) > 0.3 or abs(_wrap(c[3] - o[3])) > math.radians(30) for o in near)], reverse=True)
        # a candidate has to be a place the robot can be (not in a wall or on a thing it has seen), and nearer to where it fell is a little more likely
        cands = [(c[0] - PRIOR_W * math.hypot(c[1] - prior[0], c[2] - prior[1]), c[1], c[2], c[3]) for c in cands if a._dat(D, c[1], c[2]) >= FEASIBLE]
        cands.sort(reverse=True)
        if not cands: break
        seen.append((r.odom.copy(), cands))
        best, rival = _best_consistent(seen) if len(seen) > 1 else (None, None)
        tx, ty, tth = r.true_pose()
        tr = lambda c: f'{100 * math.hypot(c[1] - tx, c[2] - ty):.0f} cm {math.degrees(abs(_wrap(c[3] - tth))):.0f}°'
        sup._event(f'重新定位第 {k + 1} 次看：最好的得分 {cands[0][0]:.2f}' + (f'，一致的一对 {best[0]:.2f}' + ('' if rival is None else f'，对手 {rival[0]:.2f}') if best else '，还没有一致的一对') + f'（真值误差 {tr(cands[0]) if best is None else tr(best[1])}）')
        if best is not None and best[1][0] > min_score and (rival is None or best[0] - rival[0] > margin):
            a.loc.reset_pose(np.array(best[1][1:])); a.pose_before_fall = None
            sup.relocalized = getattr(sup, 'relocalized', []) + [(round(math.hypot(best[1][1] - tx, best[1][2] - ty), 3), round(abs(_wrap(best[1][3] - tth)), 3))]
            return True
        moved = yield from _new_view(sup)
        if not moved: break
    sup._event('位置找不回来，不敢瞎猜（猜错会把地图搞乱）'); sup._fault('lost after a fall'); return False


def attach(sup):
    sup.right_fn = recover
