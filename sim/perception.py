"""Object detection from the front RGB-D camera, rendered by MuJoCo.

What is real here: the images. Objects are only found if enough of their pixels are visible (resolution, distance,
occlusion by furniture or by the robot's own arm all count). Everything the robot learns about an object comes from
back-projecting noisy depth pixels: where it is, how tall it is, its outline seen from above, its long axis.
Nothing is known in advance about what the object is or how big it is.

What is assumed: the instance masks. A segmentation render stands in for a class-agnostic instance segmentation
model ("this blob of pixels is one thing lying on the floor") that never misses, merges or splits objects. It does
not say what the object is, and it gives no identity across frames: matching detections to remembered objects is
the controller's job, by position.
"""
import math

import mujoco
import numpy as np

from scipy.spatial import ConvexHull, QhullError

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
        # pixel label (geom id) -> object instance, through the geom's top-level body
        self.geom2obj = np.full(m.ngeom, -1)
        roots = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'obj{i}'): i for i in range(len(robot.objects))}
        for gid in range(m.ngeom):
            self.geom2obj[gid] = roots.get(int(m.body_rootid[m.geom_bodyid[gid]]), -1)
        self.basket_geoms = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ('basket_floor', 'basket_w0', 'basket_w1', 'basket_w2', 'basket_w3')])
        self.basket_obs = None
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
        """List of detections, each a dict in the chassis frame (x forward, y left, z up from the floor):
        rel (centre of the visible outline, z = half the height), top (height), bottom (lowest visible point: above ~4 cm
        it is not lying on the floor), hull (outline seen from above, k x 2), pts (up to 400 visible surface points, N x 3),
        heading (long axis), elong (long/short), width (narrowest outline width), n_px, dist, cut (touches the image edge)."""
        self.capture()
        out = []
        bm = np.isin(self.label, self.basket_geoms); nb = int(bm.sum()); self.basket_obs = None
        if nb >= 80:
            vv, uu = np.nonzero(bm); Pb = self.points_body(bm); Pb = Pb[np.isfinite(Pb).all(1)]
            self.basket_obs = dict(pts=Pb, n_px=nb, cut=bool(vv.max() >= H - 2 or uu.min() <= 1 or uu.max() >= W - 2))
        inst = np.where(self.label >= 0, self.geom2obj[np.maximum(self.label, 0)], -1)
        for k in np.unique(inst[inst >= 0]):
            mask = inst == k
            n = int(mask.sum())
            if n < MIN_PX: continue
            P = self.points_body(mask); P = P[np.isfinite(P).all(1)]
            if len(P) < MIN_PX: continue
            vv, uu = np.nonzero(mask)
            top = float(np.percentile(P[:, 2], 98))
            xy = P[:, :2]
            heading, elong = _heading(xy)
            try:
                hull = xy[ConvexHull(xy).vertices]
            except (QhullError, ValueError):
                hull = xy[[np.argmin(xy[:, 0]), np.argmax(xy[:, 0]), np.argmin(xy[:, 1]), np.argmax(xy[:, 1])]]
            c = 0.5 * (hull.min(0) + hull.max(0))
            dist = float(np.hypot(c[0] - self.t_bc[0], c[1] - self.t_bc[1]))
            pts = P if len(P) <= 400 else P[np.linspace(0, len(P) - 1, 400).astype(int)]
            out.append(dict(rel=np.array([c[0], c[1], top / 2]), top=top, bottom=float(np.percentile(P[:, 2], 2)), hull=hull, pts=pts, heading=heading, elong=elong,
                            width=min_width(hull), n_px=n, dist=dist, cut=bool(vv.max() >= H - 2 or uu.min() <= 1 or uu.max() >= W - 2)))
        return out

    def close(self):
        """Free the two offscreen renderers (each holds a GL context; they are not released with the object)."""
        self.seg.close(); self.dep.close()

    def sees_load(self, p_pinch):
        """Is something hanging in the gripper? p_pinch: the gripper's pinch point (chassis frame, from joint feedback).
        Used when the jaw has closed to its stop: a squashed soft object and an empty gripper then read the same angle."""
        for d in self.detect():
            # off the floor, under the gripper, and reaching up to the jaws (not something lying in the basket below)
            if d['bottom'] > 0.03 and d['top'] > p_pinch[2] - 0.04 and math.hypot(d['rel'][0] - p_pinch[0], d['rel'][1] - p_pinch[1]) < 0.07:
                return True
        return False

    def top_under(self, p_pinch):
        """Highest top among things off the floor right under the gripper (0 if none). Called before and after a small
        lift of the open gripper: a thing that rises with it is stuck to the jaws."""
        tops = [d['top'] for d in self.detect() if d['bottom'] > 0.02 and math.hypot(d['rel'][0] - p_pinch[0], d['rel'][1] - p_pinch[1]) < 0.08]
        return max(tops, default=0.0)

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


def min_width(hull):
    """Narrowest width of a 2-D outline (over directions every 3 degrees)."""
    p = hull @ _CS
    return float((p.max(0) - p.min(0)).min())


def extent_along(hull, u):
    """(near, far) of an outline along the unit direction u."""
    t = hull @ np.asarray(u)
    return float(t.min()), float(t.max())
