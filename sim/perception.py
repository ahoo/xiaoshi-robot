"""Object detection from the front RGB-D camera, rendered by MuJoCo.

What is real here: the images. Objects are only found if enough of their pixels are visible (resolution, distance,
occlusion by furniture or by the robot's own arm all count), their position comes from back-projecting noisy depth
pixels, and their orientation from the shape of those points on the floor.

What is assumed: the per-pixel class labels. A segmentation render stands in for a trained detector that labels
every visible pixel of a known object correctly and never labels anything else. Each object class appears once per
room, so the class is also the object's identity.
"""
import math

import mujoco
import numpy as np

from model import OBJECTS

W, H = 320, 240
MIN_PX = 25                  # fewer visible pixels than this: not detected
DEPTH_SIGMA = (0.002, 0.005)  # per-pixel depth noise: a + b * depth^2 (m), roughly a consumer stereo depth camera


class FrontCamera:
    def __init__(self, robot, rng):
        self.r = robot; self.rng = rng
        m = robot.m
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'front')
        self.seg = mujoco.Renderer(m, H, W); self.seg.enable_segmentation_rendering()
        self.dep = mujoco.Renderer(m, H, W); self.dep.enable_depth_rendering()
        f = (H / 2) / math.tan(math.radians(m.cam_fovy[self.cam]) / 2)
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        self.ray = np.stack([(u - W / 2) / f, -(v - H / 2) / f, -np.ones_like(u)], -1)   # camera frame, per unit depth
        # camera mount relative to the chassis: a calibration constant on the real robot
        d = robot.d; Rb = d.xmat[robot.chassis].reshape(3, 3); pb = d.xpos[robot.chassis]
        Rc = d.cam_xmat[self.cam].reshape(3, 3); pc = d.cam_xpos[self.cam]
        self.R_bc = Rb.T @ Rc; self.t_bc = Rb.T @ (pc - pb); self.t_bc[2] += pb[2]    # z measured from the floor
        self.obj_geom = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f'obj{i}'): i for i in range(len(robot.objects))}
        self.kinds = [o[0] for o in robot.objects]
        self.depth = None

    def capture(self):
        d = self.r.d
        self.seg.update_scene(d, 'front'); s = self.seg.render()
        self.dep.update_scene(d, 'front'); self.depth = self.dep.render()   # noise is added per pixel where it is used
        self.label = np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)
        return self.depth

    def points_body(self, mask):
        """Back-project masked pixels to the chassis frame (x forward, y left, z up from the floor)."""
        z = self.depth[mask].astype(np.float64)
        z = z + self.rng.normal(0, 1, z.shape) * (DEPTH_SIGMA[0] + DEPTH_SIGMA[1] * z * z)
        P = self.ray[mask] * z[:, None]
        return P @ self.R_bc.T + self.t_bc

    def detect(self):
        """Returns {object index: dict(rel=centre in chassis frame, yaw=long-axis heading in chassis frame or None, n_px, dist)}."""
        self.capture()
        out = {}
        for gid, k in self.obj_geom.items():
            mask = self.label == gid
            n = int(mask.sum())
            if n < MIN_PX: continue
            P = self.points_body(mask)
            P = P[np.isfinite(P).all(1)]
            if len(P) < MIN_PX: continue
            # Only the side facing the camera is visible, so the centre of all points is pulled towards the robot.
            # Seen from above, the top face (or the top cap of a round object) is complete: use it for centre and heading.
            zt = np.percentile(P[:, 2], 97)
            top = P[P[:, 2] >= zt - 0.012]
            if len(top) < 8: top = P[P[:, 2] >= np.percentile(P[:, 2], 60)]
            yaw, elong = _heading(top[:, :2] if len(top) >= 10 else P[:, :2])
            # Along the line of sight, take the near edge (the best-seen part) and add the class's known half size in
            # that direction; across it, the visible points are symmetric, so their mean is the centre.
            ux, uy = np.mean(P[:, 0]) - self.t_bc[0], np.mean(P[:, 1]) - self.t_bc[1]
            un = math.hypot(ux, uy); ux, uy = ux / un, uy / un
            along = P[:, 0] * ux + P[:, 1] * uy; across = -P[:, 0] * uy + P[:, 1] * ux
            h = _half_extent(self.kinds[k], math.atan2(uy, ux) - yaw)
            hc = _half_extent(self.kinds[k], math.atan2(uy, ux) + math.pi / 2 - yaw)
            vv, uu = np.nonzero(mask)
            lo_ = lambda v: np.partition(v, 2)[2]     # robust extremes: ignore the two most extreme pixels
            hi_ = lambda v: np.partition(v, -3)[-3]
            if vv.max() >= H - 2:                    # near end runs off the bottom of the image: use the far edge
                ca = hi_(along) - h
            else:
                ca = lo_(along) + h
            # Across the line of sight both outline edges of a convex object are visible: take their midpoint.
            # Image left = robot's left = +across; an edge cut off by the image border is replaced by the other one.
            cut_l, cut_r = uu.min() <= 1, uu.max() >= W - 2
            if cut_l and not cut_r: cc = lo_(across) + hc
            elif cut_r and not cut_l: cc = hi_(across) - hc
            else: cc = 0.5 * (lo_(across) + hi_(across))
            c = np.array([ca * ux - cc * uy, ca * uy + cc * ux, np.median(P[:, 2])])
            dist = float(np.hypot(c[0] - self.t_bc[0], c[1] - self.t_bc[1]))
            out[k] = dict(rel=c, yaw=yaw if elong > 1.25 else None, yaw4=yaw, n_px=n, dist=dist, cut=bool(vv.max() >= H - 2 or cut_l or cut_r))
        return out

    def floor_visible(self, pts_body):
        """For floor points (chassis frame, z=0): is each one inside the image and not hidden behind something?"""
        Q = (np.asarray(pts_body) - self.t_bc) @ self.R_bc        # camera frame
        z = -Q[:, 2]; ok = z > 0.05
        f = (H / 2) / math.tan(math.radians(self.r.m.cam_fovy[self.cam]) / 2)
        u = (Q[:, 0] / np.maximum(z, 1e-6)) * f + W / 2; v = -(Q[:, 1] / np.maximum(z, 1e-6)) * f + H / 2
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        res = np.zeros(len(Q), bool)
        for i in np.nonzero(ok)[0]:
            res[i] = self.depth[int(v[i]), int(u[i])] >= z[i] - 0.06
        return res


_ANG = np.radians(np.arange(0, 180, 3))
_CS = np.stack([np.cos(_ANG), np.sin(_ANG)])


def _heading(xy):
    """Orientation of the minimum-area rectangle around 2-D points: (heading of its long side, long/short ratio)."""
    p1 = xy @ _CS                                    # N x angles
    p2 = xy @ np.stack([-_CS[1], _CS[0]])
    q1 = np.percentile(p1, [3, 97], axis=0); q2 = np.percentile(p2, [3, 97], axis=0)
    l1 = q1[1] - q1[0]; l2 = q2[1] - q2[0]
    i = int(np.argmin(l1 * l2))
    a = _ANG[i] if l1[i] >= l2[i] else _ANG[i] + math.pi / 2
    return float(a), float(max(l1[i], l2[i]) / max(min(l1[i], l2[i]), 1e-3))


def _half_extent(kind, phi):
    """Half size of a known object class along a horizontal direction at angle phi from the object's long axis."""
    o = OBJECTS[kind]; a = o['size'][0]
    if o['type'] == 'sphere':
        return a
    b = o['size'][1]
    if o['type'] == 'ellipsoid':
        return math.sqrt((a * math.cos(phi)) ** 2 + (b * math.sin(phi)) ** 2)
    return a * abs(math.cos(phi)) + b * abs(math.sin(phi))
