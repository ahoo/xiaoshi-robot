"""Object detection from the front RGB-D camera, and a load check from the wrist camera, rendered by MuJoCo.

Everything here works from depth images alone, the way it would on the robot:

* the floor is found by fitting a plane to the depth points near it, every frame;
* the robot's own arm and body are masked out from its known kinematics (a "self filter"; in the simulator this
  mask comes from the segmentation render of the robot's own geoms, which carries the same information);
* anything that reaches above 13 cm is furniture, a wall or a carton (objects to pick up are lower), and is removed
  together with everything directly beneath it;
* what is left, low things on the floor, is grouped by connectivity seen from above, on a 1.5 cm grid.

That makes no use of what the objects are. It also makes the mistakes such a method makes: two things closer than
about 1.5 cm come out as one blob, a thing pushed against furniture can lose its near side or vanish, flat things
under ~1 cm are lost in the floor, and small things far away have too few pixels. The basket is recognised by its
known size and rim height.
"""
import math

import mujoco
import numpy as np

from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

W, H = 320, 240
MIN_PX = 25                  # fewer visible pixels than this: not detected
DEPTH_SIGMA = (0.002, 0.005)  # per-pixel depth noise: a + b * depth^2 (m), roughly a consumer stereo depth camera
MAX_RANGE = 3.0              # m
FLOOR_TOL = 0.008            # m above the fitted floor plane: below this a point is floor
TALL = 0.13                  # m: points above this belong to furniture, walls, cartons
CELL = 0.015                 # m: grid for grouping points seen from above
MAX_SIZE = 0.20              # m: a blob longer than this is not something to pick up (or two things merged)


def robot_geoms(m, robot):
    root = m.body_rootid[robot.chassis]
    return np.array([g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root])


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
        self.self_geoms = robot_geoms(m, robot)
        self.basket_obs = None
        self.depth = None
        self.floor = np.zeros(3)

    def capture(self):
        d = self.r.d
        self.seg.update_scene(d, 'front'); s = self.seg.render()
        self.dep.update_scene(d, 'front'); self.depth = self.dep.render()
        lab = np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)
        self.self_mask = np.isin(lab, self.self_geoms)          # the robot's own arm and body
        return self.depth

    def _cloud(self):
        """All pixels as noisy points in the chassis frame, with height above the fitted floor plane."""
        z = self.depth.astype(np.float64)
        z = z + self.rng.normal(0, 1, z.shape) * (DEPTH_SIGMA[0] + DEPTH_SIGMA[1] * z * z)
        P = (self.ray * z[..., None]) @ self.R_bc.T + self.t_bc
        valid = (z < MAX_RANGE) & ~self.self_mask
        # floor plane z = a x + b y + c, fitted to points near the nominal floor, then refit without outliers
        cand = valid & (np.abs(P[..., 2]) < 0.03)
        hgt = P[..., 2].copy(); self.floor = np.zeros(3)
        if cand.sum() > 300:
            Q = P[cand]
            for tol in (0.03, 0.006):
                Q = Q[np.abs(Q[:, 2] - (Q[:, :2] @ self.floor[:2] + self.floor[2])) < tol]
                if len(Q) < 200: break
                A = np.column_stack([Q[:, :2], np.ones(len(Q))])
                self.floor = np.linalg.lstsq(A, Q[:, 2], rcond=None)[0]
            hgt = P[..., 2] - (P[..., 0] * self.floor[0] + P[..., 1] * self.floor[1] + self.floor[2])
        P[..., 2] = hgt
        return P, valid

    def detect(self, basket_rect=None):
        """List of detections, each a dict in the chassis frame (x forward, y left, z up from the floor):
        rel (centre of the visible outline, z = half the height), top (height), bottom (lowest visible point: above ~4 cm
        it is not lying on the floor), hull (outline seen from above, k x 2), pts (up to 400 visible surface points, N x 3),
        heading (long axis), elong (long/short), width (narrowest outline width), n_px, dist, cut (touches the image edge).
        basket_rect: (cx, cy, yaw, half_x, half_y) where the robot believes the basket is (chassis frame); points there
        go to the basket estimate and are not grouped into objects. Without it the basket is recognised by its size."""
        self.capture(); self.basket_obs = None
        P, valid = self._cloud()
        hgt = P[..., 2]
        above = valid & (hgt > FLOOR_TOL)
        tall = above & (hgt > TALL)
        low = above & ~tall
        vv_all, uu_all = np.nonzero(low)
        if not len(vv_all): return []
        # Blobs whose upper edge is hidden count as only partly seen: something closer sits right above them in the
        # image (the robot's own arm, or the thing it is carrying), so they may continue upward behind it. A table leg
        # whose upper part is hidden looks just like a short thing on the floor. (A low thing on the floor can only be
        # hidden from below by things in front of it, never from above.)
        dz = np.full(self.depth.shape, np.inf); dz[3:] = self.depth[:-3]
        self.occl_top = ndimage.binary_dilation(self.self_mask, iterations=3) | (dz < self.depth - 0.05)
        X = P[vv_all, uu_all]
        if basket_rect is not None:
            cx, cy, yaw, hx, hy = basket_rect; c, s = math.cos(yaw), math.sin(yaw)
            bx = c * (X[:, 0] - cx) + s * (X[:, 1] - cy); by = -s * (X[:, 0] - cx) + c * (X[:, 1] - cy)
            inb = (np.abs(bx) < hx + 0.03) & (np.abs(by) < hy + 0.03)
            if inb.sum() >= 80:
                self.basket_obs = dict(pts=X[inb], n_px=int(inb.sum()), cut=self._cut(vv_all[inb], uu_all[inb]))
            keep = ~inb; vv_all, uu_all, X = vv_all[keep], uu_all[keep], X[keep]
            if not len(X): return []
        # grid seen from above
        x0, y0 = X[:, 0].min() - 2 * CELL, X[:, 1].min() - 2 * CELL
        gi = ((X[:, 0] - x0) / CELL).astype(int); gj = ((X[:, 1] - y0) / CELL).astype(int)
        G = np.zeros((gi.max() + 3, gj.max() + 3), np.int32)
        np.add.at(G, (gi, gj), 1)
        T = tall & (P[..., 0] > x0) & (P[..., 1] > y0)
        if T.any():
            Y = P[T]; ti = ((Y[:, 0] - x0) / CELL).astype(int); tj = ((Y[:, 1] - y0) / CELL).astype(int)
            ok = (ti < G.shape[0]) & (tj < G.shape[1])
            TG = np.zeros(G.shape, bool); TG[ti[ok], tj[ok]] = True
            G[ndimage.binary_dilation(TG, iterations=1)] = 0          # furniture and whatever is right beneath it
        lab, n = ndimage.label(G >= 2, structure=np.ones((3, 3)))
        pl = lab[gi, gj]
        out = []
        for k in range(1, n + 1):
            sel = pl == k
            if sel.sum() < MIN_PX: continue
            Pk = X[sel]; vv, uu = vv_all[sel], uu_all[sel]
            xy = Pk[:, :2]
            heading, elong = _heading(xy)
            try:
                hull = xy[ConvexHull(xy).vertices]
            except (QhullError, ValueError):
                hull = xy[[np.argmin(xy[:, 0]), np.argmax(xy[:, 0]), np.argmin(xy[:, 1]), np.argmax(xy[:, 1])]]
            top = float(np.percentile(Pk[:, 2], 98))
            ext = np.ptp(hull, 0); L = float(max(np.hypot(*ext), 0))
            if basket_rect is None and self.basket_obs is None and _looks_like_basket(xy, top):
                self.basket_obs = dict(pts=Pk, n_px=int(sel.sum()), cut=self._cut(vv, uu)); continue
            if L > MAX_SIZE * 1.42: continue                    # too big to be one thing to pick up
            c = 0.5 * (hull.min(0) + hull.max(0))
            dist = float(np.hypot(c[0] - self.t_bc[0], c[1] - self.t_bc[1]))
            pts = Pk if len(Pk) <= 400 else Pk[np.linspace(0, len(Pk) - 1, 400).astype(int)]
            out.append(dict(rel=np.array([c[0], c[1], top / 2]), top=top, bottom=float(np.percentile(Pk[:, 2], 2)), hull=hull, pts=pts, heading=heading, elong=elong,
                            width=min_width(hull), n_px=int(sel.sum()), dist=dist, cut=self._cut(vv, uu)))
        return out

    def _cut(self, vv, uu):
        return bool(vv.max() >= H - 2 or vv.min() <= 1 or uu.min() <= 1 or uu.max() >= W - 2 or self.occl_top[vv, uu].sum() >= 3)

    def close(self):
        """Free the two offscreen renderers (each holds a GL context; they are not released with the object)."""
        self.seg.close(); self.dep.close()

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


def _looks_like_basket(xy, top, size=(0.30, 0.22), h=0.09):
    """The robot's own basket: a blob about its known size (at least one full side seen) with the rim at its known height."""
    a, _ = _heading(xy)
    u = np.array([math.cos(a), math.sin(a)]); v = np.array([-u[1], u[0]])
    L = np.ptp(xy @ u); Wd = np.ptp(xy @ v)
    return abs(top - h) < 0.025 and size[0] - 0.08 < L < size[0] + 0.05 and Wd < size[0] + 0.05 and L > 0.18


class WristCamera:
    """Short-range depth camera on the side of the gripper, looking into the gap between the jaws. Answers one
    question: is anything (other than the robot's own fingers) between the jaws?"""
    WS, HS = 96, 72

    def __init__(self, robot):
        self.r = robot; m = robot.m
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'wrist')
        self.body = m.cam_bodyid[self.cam]
        self.seg = mujoco.Renderer(m, self.HS, self.WS); self.seg.enable_segmentation_rendering()
        self.dep = mujoco.Renderer(m, self.HS, self.WS); self.dep.enable_depth_rendering()
        f = (self.HS / 2) / math.tan(math.radians(m.cam_fovy[self.cam]) / 2)
        u, v = np.meshgrid(np.arange(self.WS) + 0.5, np.arange(self.HS) + 0.5)
        self.ray = np.stack([(u - self.WS / 2) / f, -(v - self.HS / 2) / f, -np.ones_like(u)], -1)
        # camera mount in the gripper frame: fixed by the bracket, a calibration constant
        Rq = np.zeros(9); mujoco.mju_quat2Mat(Rq, m.cam_quat[self.cam]); self.R = Rq.reshape(3, 3); self.t = m.cam_pos[self.cam].copy()
        self.self_geoms = robot_geoms(m, robot)
        self.rng = np.random.default_rng(1)

    def load(self):
        """Number of pixels that see something in the space between the jaws (gripper frame: fingers along -z,
        fixed jaw face at x = -0.009, jaws close along x)."""
        d = self.r.d
        self.seg.update_scene(d, self.cam); s = self.seg.render()
        self.dep.update_scene(d, self.cam); z = self.dep.render().astype(np.float64)
        lab = np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)
        ok = ~np.isin(lab, self.self_geoms) & (z < 0.5)
        z = z + self.rng.normal(0, 0.001, z.shape)
        Pg = (self.ray[ok] * z[ok][:, None]) @ self.R.T + self.t
        inside = (Pg[:, 0] > -0.012) & (Pg[:, 0] < 0.075) & (np.abs(Pg[:, 1]) < 0.03) & (Pg[:, 2] < -0.035) & (Pg[:, 2] > -0.112)
        return int(inside.sum())

    def holding(self):
        return self.load() >= 20

    def close(self):
        self.seg.close(); self.dep.close()


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
