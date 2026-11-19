"""Bench for the clear plastic bottle, which the depth camera cannot measure (it returns nothing, or the floor behind it).

  python test_clear.py all [n]      everything below, saved to results/clear_bench.json (a few minutes)
  python test_clear.py grasp [n]    the bottle parked at a grasp stance, lying across the jaw line (as the controller arranges it: it
                                    stands where the cluster shows the narrow side); the robot knows it only from the no-reading
                                    cluster in the depth image (perception.clear_detection); pick it up, hold 2 s
  python test_clear.py see [n]      how well the cluster gives the position and size of the bottle, by distance and orientation
                                    (true shape sampled from the simulator, used only here)
  python test_clear.py signal       what a clear bottle leaves in the depth image, by distance (no reading / floor behind / bottle surface, and how
                                    many of the real readings the flying-pixel filter keeps)
  python test_clear.py empty        no bottle: do the clusters ever appear on bare floor (false alarms)
  python test_clear.py poserough [n]   the stance sweep again with a rougher clear-plastic depth image (35 % no reading, 45 % floor behind, in 4-pixel patches)
  python test_clear.py pose|poseclose|posestance [n]   direction and length of the bottle at any angle to the line of sight and with the
                                    robot turned, far (0.4-1.0 m), close (0.28-0.45 m) and at a grasp stance (0.21-0.29 m)
"""
import json
import math
import sys

import numpy as np

import model
import objects as O
from perception import FrontCamera, WristCamera, clear_detection
from robot import Robot
from skills import carry_pose, grasp, holding, plan_grasp, wait
from test_perception import true_points

ORIENT = dict(across=math.pi / 2)       # long axis relative to the jaw line (from robot to bottle): the controller stands on the narrow side
SEE_ORIENT = dict(across=math.pi / 2, diag=math.pi / 4, along=0.0)       # long axis relative to the line of sight, for the see bench


def one(seed, orient, pad_friction=1.0, verbose=False):
    rng = np.random.default_rng(seed)
    model.P['pad_friction'] = pad_friction
    spec = O.sample('bottle', rng)
    rpose = (2.2, 1.05, rng.uniform(-3.1, 3.1))
    cam = None; plan = None; logs = []; det = None
    for tries in range(8):
        r_obj = rng.uniform(0.205, 0.265); bearing = rng.uniform(-0.3, 0.3)
        rel0 = np.array([r_obj * math.cos(bearing), r_obj * math.sin(bearing)])
        rad = math.atan2(rel0[1], rel0[0] - 0.069)
        yaw_rel = rad + ORIENT[orient] + rng.normal(0, 0.15) * (orient == 'across')
        c, s_ = math.cos(rpose[2]), math.sin(rpose[2])
        robot = Robot([('bottle', rpose[0] + c * rel0[0] - s_ * rel0[1], rpose[1] + s_ * rel0[0] + c * rel0[1], rpose[2] + yaw_rel, spec)], seed=seed, robot_pose=rpose)
        for _ in range(30): robot.step()
        if cam is not None: cam.close()
        cam = FrontCamera(robot, np.random.default_rng(seed * 10 + tries))
        cam.detect(n_avg=8)
        hs = [h for h in cam.holes if h['n_px'] >= 60]
        plan = None
        if hs:
            h = max(hs, key=lambda q: q['n_px']); det = clear_detection(h, cam)
            plan = plan_grasp(det)
            if abs(det['rel'][1]) <= 0.06 and plan is not None: break
            plan = None
    if plan is None: logs = ['no stance']
    wrist = WristCamera(robot)

    def script():
        if plan is None: return False
        ok = yield from grasp(robot, plan, logs.append, look=wrist.holding)
        if not ok: return False
        yield from carry_pose(robot)
        yield from wait(robot, 2.0)
        return holding(robot, plan)
    g = script(); claimed = None
    try:
        while True:
            next(g); robot.step()
    except StopIteration as e:
        claimed = bool(e.value)
    held = robot.obj_pos(0)[2] > 0.08 and np.linalg.norm(robot.obj_pos(0) - robot.pinch_pos()) < 0.10
    cam.close(); wrist.close()
    return dict(held=bool(held), claimed=claimed, logs=logs, stances=tries + 1, no_stance=plan is None and logs == ['no stance'])


def run_grasp(n=10):
    out = {}
    for orient in ORIENT:
        res = [one(200 + i, orient) for i in range(n)]
        tried = [r for r in res if not r['no_stance']]
        why = {}
        for r in tried:
            if not r['held']:
                k = (r['logs'] or ['夹起后掉落'])[-1]; why[k] = why.get(k, 0) + 1
        out[orient] = dict(success=sum(r['held'] for r in tried), n=len(tried), no_stance=n - len(tried), misjudged=sum(r['held'] != r['claimed'] for r in tried), failures=why)
        print(f'{orient:<7} held {out[orient]["success"]}/{len(tried)} (+{n - len(tried)} no stance)  self-check wrong {out[orient]["misjudged"]}  {why}', flush=True)
    return out


def run_see(n=6, dists=(0.4, 0.6, 0.8, 1.0, 1.2)):
    out = {}; rpose = (2.2, 1.05, 0.0); rng = np.random.default_rng(5)
    for orient, yaw in SEE_ORIENT.items():
        for dist in dists:
            rows = []
            for t in range(n):
                spec = O.sample('bottle', rng); bear = rng.uniform(-0.15, 0.15)
                r = Robot([('bottle', rpose[0] + dist * math.cos(bear), rpose[1] + dist * math.sin(bear), yaw + rng.uniform(-0.1, 0.1), spec)], seed=t, robot_pose=rpose)
                for _ in range(60): r.step()
                cam = FrontCamera(r, np.random.default_rng(t)); cam.detect(); hs = cam.holes
                if not hs: cam.close(); rows.append(None); continue
                h = max(hs, key=lambda q: q['n_px']); d = clear_detection(h, cam); cam.close()
                x, y, th = r.true_pose(); c, s = math.cos(th), math.sin(th)
                W_ = true_points(r, 0); B = np.stack([c * (W_[:, 0] - x) + s * (W_[:, 1] - y), -s * (W_[:, 0] - x) + c * (W_[:, 1] - y)], 1)
                # true footprint centre from the sampled surface (the middle of its bounding box: it is a cylinder)
                tc = 0.5 * (B.min(0) + B.max(0)); ec = d['rel'][:2]
                dh = abs(((d['heading'] - (spec_axis(r, th))) + math.pi / 2) % math.pi - math.pi / 2)
                rows.append((d['hull'][:, 0].min() - B[:, 0].min(), float(np.linalg.norm(ec - tc)), d['length'] - spec['p']['L'], d['top'] - W_[:, 2].max(), math.degrees(dh) if not d['ambiguous'] else np.nan, float(d['ambiguous'])))
            got = [q for q in rows if q]
            a = np.array(got) if got else np.zeros((0, 6))
            f = lambda col, absval=True: round(1000 * float(np.nanmean(np.abs(a[:, col]) if absval else a[:, col])), 1) if len(a) else None
            near_bias = f(0, False)
            hd = a[:, 4][~np.isnan(a[:, 4])] if len(a) else []
            out[f'{orient}:{dist}'] = dict(seen=len(got), n=n, near_bias_mm=near_bias, near_err_mm=f(0), centre_err_mm=f(1), length_err_mm=f(2), top_err_mm=f(3),
                                          heading_err_deg=round(float(np.mean(hd)), 1) if len(hd) else None, ambiguous=int(a[:, 5].sum()) if len(a) else 0)
            o = out[f'{orient}:{dist}']
            print(f'{orient:<7} {dist:.1f} m  seen {o["seen"]}/{n}  ambiguous {o["ambiguous"]}  near edge {o["near_bias_mm"]:+} mm (abs {o["near_err_mm"]})  centre {o["centre_err_mm"]} mm  length {o["length_err_mm"]} mm  height {o["top_err_mm"]} mm  heading err {o["heading_err_deg"]} deg', flush=True)
    return out


def run_signal(n=6, dists=(0.235, 0.4, 0.6, 0.8, 1.0, 1.2)):
    """What a clear bottle leaves in the depth image, per distance (mean over n random bottles, any direction): pixels in its
    outline, of them without a reading / showing the floor behind it / showing the bottle itself, and how many of those
    real readings survive the depth-edge filter that removes flying pixels."""
    from perception import depth_edges
    rng = np.random.default_rng(0); out = {}
    for dist in dists:
        rows = []
        for t in range(n):
            spec = O.sample('bottle', rng); yaw = rng.uniform(-math.pi, math.pi)
            r = Robot([('bottle', 2.2 + dist, 1.05, yaw, spec)], seed=t, robot_pose=(2.2, 1.05, 0.0))
            for _ in range(60): r.step()
            cam = FrontCamera(r, np.random.default_rng(t))
            cam.seg.update_scene(r.d, 'front'); sg = cam.seg.render(); lab = np.where(sg[..., 1] == 5, sg[..., 0], -1)
            cam.dep.update_scene(r.d, 'front'); true = cam.dep.render()
            cam.dep.update_scene(r.d, 'front', scene_option=cam.bg_opt); bg = cam.dep.render()
            cam.capture(); d = cam.depth
            sil = (lab >= 0) & (cam.gmat[np.maximum(lab, 0)] == 3)
            surf = sil & (d > 0.05) & (np.abs(d - true) < 1e-6)
            rows.append((int(sil.sum()), int((sil & (d <= 0.05)).sum()), int((sil & (d > 0.05) & (np.abs(d - bg) < 1e-6)).sum()), int(surf.sum()), int((surf & ~depth_edges(d)).sum())))
            cam.close()
        a = np.array(rows, float).mean(0)
        out[str(dist)] = dict(silhouette_px=round(a[0]), no_reading=round(a[1]), floor_behind=round(a[2]), bottle_surface=round(a[3]), bottle_surface_after_edge_filter=round(float(a[4]), 1))
        print(dist, out[str(dist)], flush=True)
    return out


def run_empty():
    """False alarms: the robot turns on the spot at four places in the empty room (12 headings each); any cluster is false."""
    tot = 0; frames = 0
    for (px, py) in ((1.0, 1.0), (2.2, 1.05), (3.0, 2.0), (1.5, 2.2)):
        for yaw in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            r = Robot([], seed=0, robot_pose=(px, py, yaw))
            for _ in range(20): r.step()
            cam = FrontCamera(r, np.random.default_rng(1)); cam.detect(); tot += len(cam.holes); frames += 1; cam.close()
    print(f'empty room: {frames} frames, {tot} clusters')
    return dict(frames=frames, false_clusters=tot)


def run_pose_sweep(n=24, seed=11, dlo=0.4, dhi=1.0, turn_max=0.6, pert=None):
    """Heading and length of the bottle from the cluster when the robot is not squarely facing it: the robot turned by up to
    +-35 degrees (so the bottle sits toward the side of the image), the bottle at any angle to the line of sight, 0.4-1.0 m.
    A wrong direction reported as sure (not 'ambiguous') is the dangerous case: the planner would stand on the wrong side."""
    import perturb
    perturb.CUR = dict(pert or {})
    rng = np.random.default_rng(seed); rows = []
    for t in range(n):
        spec = O.sample('bottle', rng); dist = rng.uniform(dlo, dhi); turn = rng.uniform(-turn_max, turn_max)
        rpose = (2.2, 1.05, turn); ax = rng.uniform(0, math.pi)      # bottle axis angle, world
        r = Robot([('bottle', 2.2 + dist, 1.05, ax, spec)], seed=t, robot_pose=rpose)
        for _ in range(40): r.step()
        cam = FrontCamera(r, np.random.default_rng(t)); cam.detect(); hs = cam.holes
        if not hs: cam.close(); rows.append(dict(seen=False)); continue
        h = max(hs, key=lambda q: q['n_px']); d = clear_detection(h, cam); cam.close()
        x, y, th = r.true_pose(); true_ax = spec_axis(r, th)
        sight = math.atan2(0.0, 1.0) - th                                       # line of sight in the chassis frame (bottle is straight ahead in the world)
        ang_to_sight = abs(((true_ax - sight) + math.pi / 2) % math.pi - math.pi / 2)     # 0 = end-on, 90 = across
        err = abs(((d['heading'] - true_ax) + math.pi / 2) % math.pi - math.pi / 2)
        rows.append(dict(seen=True, ambiguous=d['ambiguous'], head_err=math.degrees(err), len_err=100 * (d['length'] - spec['p']['L']), true_angle=math.degrees(ang_to_sight), turn=math.degrees(turn), dist=dist))
    perturb.CUR = {}
    seen = [q for q in rows if q['seen']]; sure = [q for q in seen if not q['ambiguous']]
    out = dict(n=len(rows), seen=len(seen), sure=len(sure), ambiguous=len(seen) - len(sure), range_m=(dlo, dhi), turn_max_deg=round(math.degrees(turn_max)))
    print(f'seen {len(seen)}/{len(rows)}; sure (not ambiguous) {len(sure)}; ambiguous {len(seen) - len(sure)}')
    if sure:
        he = np.array([q['head_err'] for q in sure]); le = np.array([q['len_err'] for q in sure])
        out.update(heading_err_median_deg=round(float(np.median(he)), 1), heading_err_p90_deg=round(float(np.percentile(he, 90)), 1), heading_err_worst_deg=round(float(he.max()), 1),
                   wrong_over_25deg=int((he > 25).sum()), length_err_median_cm=round(float(np.median(le)), 1))
        print(f'  sure ones: heading error median {np.median(he):.0f} deg, 90th pct {np.percentile(he, 90):.0f}, worst {he.max():.0f}; wrong by more than 25 deg: {(he > 25).sum()}/{len(he)}; length error (cm) median {np.median(le):+.1f} worst {le[np.argmax(np.abs(le))]:+.1f}')
    near_end_on = [q for q in seen if q['true_angle'] <= 10]
    out['end_on_within_10deg'] = dict(n=len(near_end_on), flagged_ambiguous=sum(q['ambiguous'] for q in near_end_on))
    for q in sorted(seen, key=lambda q: q['true_angle']):
        print(f"  axis-vs-sightline {q['true_angle']:4.0f} deg  robot turned {q['turn']:+4.0f}  dist {q['dist']:.2f}  " + ('ambiguous' if q['ambiguous'] else f"sure: heading err {q['head_err']:4.0f} deg  length err {q['len_err']:+5.1f} cm"))
    return out


def spec_axis(r, th):
    """True long axis of bottle 0 (its body x axis, cylinder axis) in the chassis frame, from the simulator."""
    R = r.d.xmat[r.obj_body[0]].reshape(3, 3); ax = R[:, 0]
    return math.atan2(-math.sin(th) * ax[0] + math.cos(th) * ax[1], math.cos(th) * ax[0] + math.sin(th) * ax[1])


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'; n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out = {}
    if what in ('all', 'grasp'): out['grasp'] = run_grasp(max(n, 10) if what == 'all' else n)
    if what in ('all', 'see'): out['see'] = run_see(6 if what == 'all' else n)
    if what in ('all', 'empty'): out['empty'] = run_empty()
    if what in ('all', 'signal'): out['signal'] = run_signal()
    if what in ('all', 'pose'): out['pose_far'] = run_pose_sweep(30 if what == 'all' else n)
    if what in ('all', 'poseclose'): out['pose_close'] = run_pose_sweep(30 if what == 'all' else n, seed=12, dlo=0.28, dhi=0.45)
    if what in ('all', 'poserough'): out['pose_stance_rough'] = run_pose_sweep(30 if what == 'all' else n, seed=13, dlo=0.21, dhi=0.29, turn_max=0.12, pert=dict(clear_holes=(0.35, 0.45), clear_block=4))
    if what in ('all', 'posestance'): out['pose_stance'] = run_pose_sweep(30 if what == 'all' else n, seed=13, dlo=0.21, dhi=0.29, turn_max=0.12)     # where the robot stands to grasp
    if what == 'all': json.dump(out, open('../results/clear_bench.json', 'w'), indent=1)
