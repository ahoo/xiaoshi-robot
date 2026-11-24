"""SO-101 inverse kinematics (damped least squares on the MuJoCo model) in the chassis frame."""
import math

import mujoco
import numpy as np

from model import build

ARM_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
ALL_JOINTS = ARM_JOINTS + ['gripper']


class ArmIK:
    """Solves for the 5 arm joints so the 'pinch' site reaches a point (chassis frame, z measured from the floor)
    with the gripper approach axis along `approach` and, optionally, the jaw closing axis at `yaw`."""

    def __init__(self):
        self.m, _ = build([], robot_pose=(0.0, 0.0, 0.0))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        self.qadr = [m.jnt_qposadr[j] for j in self.jid]
        self.dadr = [m.jnt_dofadr[j] for j in self.jid]
        self.lo = np.array([m.jnt_range[j][0] for j in self.jid]) + 0.02
        self.hi = np.array([m.jnt_range[j][1] for j in self.jid]) - 0.02
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'pinch')
        self.jp = np.zeros((3, m.nv)); self.jr = np.zeros((3, m.nv))
        # the chassis sits with the axle plane 0.0508 above the floor; z in this class is height above the floor
        self.d.qpos[:3] = [0, 0, 0.0508]; self.d.qpos[3:7] = [1, 0, 0, 0]

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
        R = self.d.site_xmat[self.site].reshape(3, 3)
        return self.d.site_xpos[self.site].copy(), -R[:, 2].copy(), R[:, 0].copy()   # position, approach axis, jaw axis

    def hits_body(self, q5):
        """Would the arm, at joint angles q5, run into the chassis? (The simulator's arm geoms do not collide with the chassis, a real arm does.)
        Samples points along the links and tests them against the chassis cylinder (radius 0.105, 0.02-0.09 m above the axle plane)."""
        q = np.asarray(q5, float).copy(); self.d.qpos[self.qadr] = q[:5]
        mujoco.mj_kinematics(self.m, self.d)
        names = ('shoulder', 'upper_arm', 'lower_arm', 'wrist', 'gripper')
        pts = [self.d.xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)] - [0, 0, 0.0508] for n in names] + [self.d.site_xpos[self.site] - [0, 0, 0.0508]]
        for a_, b_ in zip(pts[1:-1], pts[2:]):
            for t in (0, 0.25, 0.5, 0.75, 1.0):
                p = a_ + (b_ - a_) * t
                if math.hypot(p[0], p[1]) < 0.115 and 0.02 < p[2] < 0.09: return True
        return False

    def solve(self, pos, approach=(0, 0, -1), yaw=None, seed=None, iters=200, w_rot=0.12):
        pos = np.asarray(pos, float); a_t = np.asarray(approach, float); a_t /= np.linalg.norm(a_t)
        seeds = [seed] if seed is not None else []
        pan = -math.atan2(pos[1], pos[0] - 0.069)
        seeds += [np.array([pan, -0.3, 0.8, 1.0, 0.0]), np.array([pan, -1.2, 1.5, 0.9, 0.0]), np.array([pan, 0.3, 0.3, 0.9, 0.0])]
        best = None
        for s in seeds:
            q = np.clip(np.array(s, float), self.lo, self.hi)
            for _ in range(iters):
                p, a, jx = self.fk(q)
                mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
                e_p = pos - p
                e_r = np.cross(a, a_t)
                rows = [self.jp[:, self.dadr], w_rot * self.jr[:, self.dadr]]; errs = [e_p, w_rot * e_r]
                if yaw is not None:
                    # rotate about the approach axis until the jaw axis projects onto the requested heading (mod 180 deg)
                    jh = math.atan2(jx[1], jx[0]); dy = (yaw - jh + math.pi / 2) % math.pi - math.pi / 2
                    rows.append(w_rot * self.jr[2:3, self.dadr]); errs.append(np.array([w_rot * dy]))
                J = np.vstack(rows); e = np.concatenate(errs)
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(J.shape[0]), e)
                q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
                if np.linalg.norm(e_p) < 4e-4 and np.linalg.norm(e_r) < 4e-3 and (yaw is None or abs(dy) < 0.01):
                    break
            p, a, jx = self.fk(q)
            err = np.linalg.norm(pos - p) + 0.05 * np.linalg.norm(np.cross(a, a_t))
            if yaw is not None:
                jh = math.atan2(jx[1], jx[0]); err += 0.02 * abs((yaw - jh + math.pi / 2) % math.pi - math.pi / 2)
            if best is None or err < best[1]:
                best = (q.copy(), err, np.linalg.norm(pos - p))
            if err < 2e-3:
                break
        return best  # (q, combined error, position error in metres)


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    ik = ArmIK()
    print('--- floor grasps (pinch 2.5 cm above floor, pointing down)')
    for r in [0.16, 0.18, 0.20, 0.24, 0.28, 0.30, 0.32, 0.34]:
        q, err, pe = ik.solve([r, 0, 0.025])
        print(f'r={r:.2f} q={q} pos err {1000 * pe:.1f} mm total {err:.4f}')
    print('--- lateral: object 0.24 ahead, y offset')
    for y in [-0.10, -0.05, 0.05, 0.10]:
        q, err, pe = ik.solve([0.24, y, 0.025], yaw=0.5)
        print(f'y={y:+.2f} q={q} pos err {1000 * pe:.1f} mm total {err:.4f}')
    print('--- carry over base / place over basket rim')
    for name, p, ap in [('carry', [0.10, 0, 0.26], (0, 0, -1)), ('carry2', [0.13, 0, 0.22], (0, 0, -1)), ('place', [0.30, 0, 0.17], (0.4, 0, -1)), ('place2', [0.33, 0, 0.17], (0.7, 0, -1)), ('pre', [0.24, 0, 0.12], (0, 0, -1))]:
        q, err, pe = ik.solve(p, approach=ap)
        print(f'{name} target {p} q={q} pos err {1000 * pe:.1f} mm total {err:.4f}')
