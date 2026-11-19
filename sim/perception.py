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

import perturb
from scipy.spatial import ConvexHull, QhullError

W, H = 320, 240
MIN_PX = 25                  # fewer visible pixels than this: not detected
DEPTH_SIGMA = (0.002, 0.005)  # per-pixel depth noise: a + b * depth^2 (m), roughly a consumer stereo depth camera
MAX_RANGE = 3.0              # m
FLOOR_TOL = 0.008            # m above the fitted floor plane: below this a point is floor
TALL = 0.13                  # m: points above this belong to furniture, walls, cartons
CELL = 0.015                 # m: grid for grouping points seen from above
MAX_SIZE = 0.20              # m: a blob longer than this is not something to pick up (or two things merged)
# No-reading clusters on the floor (clear plastic, and anything else the depth camera cannot measure): a patch of
# pixels without depth, in the middle of floor that does return depth, is a thing we cannot see but know is there.
NO_READING = 0.02            # m: a depth below this is no reading (spoil_depth writes 0; the camera's own near edge shows up as 0.05)
HOLE_MIN_PX = 18             # no-reading pixels in one cluster (a 12 cm bottle at 1.2 m has about 70)
HOLE_DENS = 0.30             # fraction of no-reading pixels in a 5x5 window that counts as inside a cluster
HOLE_RANGE = (0.04, 1.30)    # m: the floor point under a cluster's lowest pixel must be this far from the camera foot
HOLE_MAX_W = 0.22            # m: wider than this on the floor is a dropout of the whole floor, not a thing to pick up
LONG_MIN = 0.09             # m: a cluster this wide (sideways) shows the long axis of what lies there; narrower: end-on, or short
SIDE_MIN = 40.0              # degrees between the direction found and the line of sight: nearer end-on than this is not believed
FAR_LIMIT = 0.55             # m from the camera: beyond this the lower edge of a cluster is too coarse to give a direction


def object_materials(m, robot):
    """Per geom: 0 matte (or not an object), 1 black, 2 shiny, 3 clear. Used only to spoil the rendered depth the way a
    real stereo depth camera would; the algorithms never see it."""
    code = dict(matte=0, black=1, shiny=2, clear=3)
    out = np.zeros(m.ngeom, np.int8)
    roots = {robot.obj_body[i]: code[o[4].get('mat', 'matte')] for i, o in enumerate(robot.objects)}
    for g in range(m.ngeom):
        out[g] = roots.get(int(m.body_rootid[m.geom_bodyid[g]]), 0)
    return out


def spoil_depth(depth, lab, gmat, rng, background=None):
    """What a stereo depth camera returns instead of the true depth (0 = no reading):
    black surfaces: no reading over half of them, in patches; shiny: holes in about a quarter, in patches;
    clear plastic: mostly no reading, often the floor behind it (background), sometimes the surface;
    every depth edge: some pixels take a value between the near and the far side ("flying pixels")."""
    Hh, Ww = depth.shape
    mat = np.where(lab >= 0, gmat[np.maximum(lab, 0)], 0)
    d = depth.copy()
    if mat.any():
        f = np.kron(rng.random((Hh // 6 + 1, Ww // 6 + 1)), np.ones((6, 6)))[:Hh, :Ww]      # patchy
        px = rng.random(d.shape)
        d[(mat == 1) & ((f < 0.5) | (px < 0.1))] = 0.0
        d[(mat == 2) & ((f < 0.25) | (px < 0.05))] = 0.0
        clear = mat == 3
        if clear.any():
            # share of a clear thing's pixels without a reading / showing the floor behind it (the rest: its real surface);
            # block: side of the squares the choice is made in (1: every pixel on its own; real holes come in patches)
            ph, pf = perturb.get('clear_holes', (0.6, 0.3)); blk = int(perturb.get('clear_block', 1))
            pc = px if blk <= 1 else np.kron(rng.random((Hh // blk + 1, Ww // blk + 1)), np.ones((blk, blk)))[:Hh, :Ww]
            d[clear & (pc < ph)] = 0.0
            if background is not None:
                see = clear & (pc >= ph) & (pc < ph + pf); d[see] = background[see]
    # flying pixels on depth edges (left/right and up/down neighbours differ by more than 5 cm)
    for ax in (0, 1):
        a = np.take(d, range(0, d.shape[ax] - 1), axis=ax); b = np.take(d, range(1, d.shape[ax]), axis=ax)
        edge = (np.abs(a - b) > 0.05) & (a > 0) & (b > 0)
        pick = edge & (rng.random(edge.shape) < 0.3)
        mix = a + (b - a) * rng.random(edge.shape)
        if ax == 0: d[:-1][pick] = mix[pick]
        else: d[:, :-1][pick] = mix[pick]
    return d


def depth_edges(d):
    """Pixels on a depth edge (a neighbour differs by more than 3 cm + 2 % of the depth): the standard filter against
    flying pixels. It also trims a pixel off every silhouette, which costs about 1 mm up close and 6 mm at 1.5 m."""
    tol = 0.03 + 0.02 * d
    e = np.zeros(d.shape, bool)
    for a, b in ((d[1:], d[:-1]), (d[:, 1:], d[:, :-1])):
        jump = np.abs(a - b) > np.minimum(tol[: a.shape[0], : a.shape[1]], 0.2)
        if a.shape[0] < d.shape[0]: e[1:] |= jump; e[:-1] |= jump
        else: e[:, 1:] |= jump; e[:, :-1] |= jump
    return e


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
        ce = perturb.get('cam_err', None)
        if ce:      # the robot's belief about the mount is off (perturb.py): position, then a small rotation
            er = np.random.default_rng(int(rng.integers(1 << 30)))
            self.t_bc = self.t_bc + er.normal(0, ce[0], 3)
            a = er.normal(0, math.radians(ce[1]), 3); Rx = np.zeros(9)
            q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, a / max(np.linalg.norm(a), 1e-9), float(np.linalg.norm(a))); mujoco.mju_quat2Mat(Rx, q)
            self.R_bc = Rx.reshape(3, 3) @ self.R_bc
        self.self_geoms = robot_geoms(m, robot)
        self.gmat = object_materials(m, robot); self.has_clear = bool((self.gmat == 3).any())
        self.bg_opt = mujoco.MjvOption(); self.bg_opt.geomgroup[2] = 0
        self.basket_obs = None
        self.depth = None
        self.floor = np.zeros(3)

    def capture(self):
        d = self.r.d
        self.seg.update_scene(d, 'front'); s = self.seg.render()
        self.dep.update_scene(d, 'front'); depth = self.dep.render()
        lab = np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)
        bg = None
        if self.has_clear:
            self.dep.update_scene(d, 'front', scene_option=self.bg_opt); bg = self.dep.render()
        self.depth = spoil_depth(depth, lab, self.gmat, self.rng, bg)
        self.self_mask = np.isin(lab, self.self_geoms)          # the robot's own arm and body
        return self.depth

    def _cloud(self, n_avg=1):
        """All pixels as noisy points in the chassis frame, with height above the fitted floor plane. n_avg: frames
        averaged while standing still (a temporal filter): the random part of the depth noise shrinks by sqrt(n_avg)."""
        z = self.depth.astype(np.float64)
        z = z + self.rng.normal(0, 1, z.shape) * (DEPTH_SIGMA[0] + DEPTH_SIGMA[1] * z * z) * perturb.get('depth_noise', 1.0) / np.sqrt(n_avg)
        P = (self.ray * z[..., None]) @ self.R_bc.T + self.t_bc
        valid = (z < MAX_RANGE) & (self.depth > 0.05) & ~self.self_mask & ~depth_edges(self.depth)   # 0: no reading there
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

    def detect(self, basket_rect=None, n_avg=1):
        """List of detections, each a dict in the chassis frame (x forward, y left, z up from the floor):
        rel (centre of the visible outline, z = half the height), top (height), bottom (lowest visible point: above ~4 cm
        it is not lying on the floor), hull (outline seen from above, k x 2), pts (up to 400 visible surface points, N x 3),
        heading (long axis), elong (long/short), width (narrowest outline width), n_px, dist, cut (touches the image edge).
        basket_rect: (cx, cy, yaw, half_x, half_y) where the robot believes the basket is (chassis frame); points there
        go to the basket estimate and are not grouped into objects. Without it the basket is recognised by its size.
        n_avg: frames averaged (only while the robot stands still, e.g. the close look before a grasp)."""
        self.capture(); self.basket_obs = None; self.low_pts = np.zeros((0, 2))
        P, valid = self._cloud(n_avg)
        hgt = P[..., 2]
        self.holes = self._find_holes(hgt)
        above = valid & (hgt > FLOOR_TOL)
        tall = above & (hgt > TALL)
        low = above & ~tall
        vv_all, uu_all = np.nonzero(low)
        if not len(vv_all): return []
        # Blobs whose upper edge is hidden count as only partly seen: something closer sits right above them in the
        # image (the robot's own arm, or the thing it is carrying), so they may continue upward behind it. A table leg
        # whose upper part is hidden looks just like a short thing on the floor. (A low thing on the floor can only be
        # hidden from below by things in front of it, never from above.)
        dd = np.where(self.depth > 0.05, self.depth, np.inf)            # holes in the depth are unknown, not near
        dz = np.full(dd.shape, np.inf); dz[3:] = dd[:-3]
        self.occl_top = ndimage.binary_dilation(self.self_mask, iterations=3) | (dz < dd - 0.05)
        X = P[vv_all, uu_all]
        # low obstacles as seen right now (chassis frame, one point per 2 cm cell, within 0.8 m): for the safety layer
        Lo = X[(X[:, 2] > 0.015) & (np.hypot(X[:, 0], X[:, 1]) < 0.8), :2]
        if len(Lo): self.low_pts = np.unique(np.round(Lo / 0.02), axis=0) * 0.02
        if basket_rect is not None:
            cx, cy, yaw, hx, hy = basket_rect; c, s = math.cos(yaw), math.sin(yaw)
            bx = c * (X[:, 0] - cx) + s * (X[:, 1] - cy); by = -s * (X[:, 0] - cx) + c * (X[:, 1] - cy)
            inb = (np.abs(bx) < hx + 0.05) & (np.abs(by) < hy + 0.05)
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
        # close one-cell gaps: holes in the depth of black or shiny things otherwise cut one thing into pieces
        lab, n = ndimage.label(ndimage.binary_closing(G >= 2, structure=np.ones((3, 3))) | (G >= 2), structure=np.ones((3, 3)))
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

    def floor_hits(self, vv, uu):
        """Where the rays through pixels (row vv, column uu) meet the fitted floor plane: (x, y) in the chassis frame and
        a flag for rays that do meet it ahead of the camera."""
        ray = self.ray[vv, uu] @ self.R_bc.T; o = self.t_bc; a, b, c = self.floor
        den = ray[:, 2] - a * ray[:, 0] - b * ray[:, 1]
        t = (a * o[0] + b * o[1] + c - o[2]) / np.where(np.abs(den) < 1e-6, -1e-6, den)
        return o[:2] + ray[:, :2] * t[:, None], (den < -1e-3) & (t > 0)

    def depth_ok_above(self, hgt):
        """Pixels with a depth reading that are above the floor but not tall (a thing to pick up), not the robot, not edge-filtered."""
        return (self.depth > 0.05) & (hgt > FLOOR_TOL) & (hgt < TALL) & ~self.self_mask

    def _find_holes(self, hgt=None):
        """Clusters of pixels with no depth reading whose rays reach the floor close by. Returns a list of dicts in the
        chassis frame: n_px; near (x of the near edge: the floor point under the lowest no-reading pixel of each image
        column, which is where the thing meets the floor on the near side, to about a pixel); y0, y1 (its sideways
        extent there); far (x where the ray over the top of the cluster meets the floor: always beyond the thing, since
        the camera looks down on it at a shallow angle); cut (touches the image edge); obj_frac (readings of something above
        the floor within 3 pixels of the cluster, per pixel of the cluster: about 0.2 around clear plastic, 0.5-3 around
        black or shiny things, whose holes sit among real surface). The robot's own arm is not counted."""
        hole = (self.depth < NO_READING) & ~self.self_mask          # a real zero, not a small depth (the robot's own front edge is 5 cm from the lens)
        if hole.sum() < HOLE_MIN_PX: return []
        dens = ndimage.uniform_filter(hole.astype(np.float32), size=5)
        core = ndimage.binary_closing(dens > HOLE_DENS, structure=np.ones((3, 3)))
        lab, n = ndimage.label(core, structure=np.ones((3, 3)))
        out = []
        for k in range(1, n + 1):
            m = hole & (lab == k)
            if m.sum() < HOLE_MIN_PX: continue
            vv, uu = np.nonzero(m)
            obj_frac = 0.0
            if hgt is not None:
                reg = ndimage.binary_dilation(lab == k, iterations=3)
                obj_frac = float((self.depth_ok_above(hgt) & reg).sum() / m.sum())
            cols = np.unique(uu)
            low = np.array([vv[uu == c].max() for c in cols]); high = np.array([vv[uu == c].min() for c in cols])
            Q, ok = self.floor_hits(low, cols)
            if ok.sum() < 3: continue
            Q = Q[ok]; dist = np.hypot(Q[:, 0] - self.t_bc[0], Q[:, 1] - self.t_bc[1])
            if not (HOLE_RANGE[0] < np.percentile(dist, 20) < HOLE_RANGE[1]): continue
            y0, y1 = float(np.percentile(Q[:, 1], 2)), float(np.percentile(Q[:, 1], 98))
            if y1 - y0 > HOLE_MAX_W: continue
            Qt, okt = self.floor_hits(high, cols)
            far = float(np.percentile(Qt[okt, 0], 90)) if okt.any() else float(Q[:, 0].max())
            out.append(dict(n_px=int(m.sum()), near=float(np.percentile(Q[:, 0], 10)), y0=y0, y1=y1, far=far, edge=Q, obj_frac=obj_frac,
                            cut=bool(vv.max() >= H - 2 or uu.min() <= 1 or uu.max() >= W - 2),
                            rows=(int(vv.min()), int(vv.max())), cols=(int(uu.min()), int(uu.max()))))
        return out

    def close(self):
        """Free the two offscreen renderers (each holds a GL context; they are not released with the object)."""
        self.seg.close(); self.dep.close()

    def hole_fraction(self, pt_body, half=3):
        """Share of pixels without a reading in a (2 half + 1)^2 window around where the floor point pt_body (chassis
        frame) shows up in the image; None if it is outside the image or behind the camera."""
        Q = (np.asarray(pt_body, float).reshape(1, 3) - self.t_bc) @ self.R_bc
        z = -Q[0, 2]
        if z < 0.05: return None
        f = (H / 2) / math.tan(math.radians(self.r.m.cam_fovy[self.cam]) / 2)
        u = int((Q[0, 0] / z) * f + W / 2); v = int(-(Q[0, 1] / z) * f + H / 2)
        if not (half <= u < W - half and half <= v < H - half): return None
        win = self.depth[v - half:v + half + 1, u - half:u + half + 1]
        return float((win < NO_READING).mean())

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


def clear_detection(h, cam):
    """A detection (same fields as FrontCamera.detect gives) for something known only by its no-reading cluster `h`.
    What the cluster tells: the near edge (where it meets the floor), the sideways extent, and the ray that just clears
    its top. Not known: the depth of it along the line of sight. The assumption made here: it is as high as it is
    thick (a bottle or a ball, a lump), with its top at the middle, as for every other thing the camera sees only from
    one side (skills.best_band). With the camera at height hc and x = c, near edge n and far-ray floor point f (both
    measured from the camera foot), the ray over the top touches it at height w above the floor, w/2 beyond the near
    edge:  w = hc (1 - (n + w/2) / f)  ->  w = hc (1 - n/f) / (1 + hc / (2 f)).
    Its direction: a thing lying on the floor shows a long side as a line of floor points along the lower edge of the
    cluster; that line's direction is the long axis and its length the length (when the cluster is wide enough to tell,
    LONG_MIN). Seen end-on, or short, the cluster is as wide as the thing is thick and says nothing about its length:
    `ambiguous` is set, and the controller looks again from the side.
    The detection describes the whole footprint (a half-round cylinder on its long axis), not just the visible half."""
    c, hc = cam.t_bc[0], cam.t_bc[2]
    f = max(h['far'] - c, 0.1); lw = h['y1'] - h['y0']; Q = h['edge'].copy()
    # The lowest ray that still meets a round thing of radius r touches it low on the near side and continues to the
    # floor, landing (1 - sin a + (1 - cos a) cos a / sin a) r beyond where the thing touches the floor (a: the ray's
    # angle below horizontal; about 0.7 r at 35 degrees). Take that off the near edge, with r = w/2 from the round prior.
    near = h['near']
    for _ in range(2):
        n = max(near - c, 0.02); f = max(f, n + 0.02)
        w = float(np.clip(hc * (1 - n / f) / (1 + hc / (2 * f)), 0.025, 0.07))
        al = math.atan2(hc, max(h['near'] - c, 0.05))
        near = h['near'] - (w / 2) * (1 - math.sin(al) + (1 - math.cos(al)) * math.cos(al) / math.sin(al))
    Q[:, 0] += near - h['near']
    far_off = (h['near'] - c) > FAR_LIMIT                    # too far for the lower edge to be more than a pixel or two high
    a = np.array([0.0, 1.0]); L = max(lw, w); ambiguous = lw < LONG_MIN or far_off; fit = None
    mid = np.array([near, 0.5 * (h['y0'] + h['y1'])])                      # middle of the lower-edge line
    if not ambiguous and len(Q) >= 6:
        q = Q - Q.mean(0); _, evec = np.linalg.eigh(q.T @ q / len(q)); ev_ = _
        a_ = evec[:, 1]                                                      # principal direction of the edge points
        if a_[1] < 0: a_ = -a_
        s_ = q @ a_; lo_, hi_ = np.percentile(s_, [2, 98]); L_ = float(max(hi_ - lo_, w)); mid_ = Q.mean(0) + a_ * 0.5 * (lo_ + hi_)
        fit = (float(np.sqrt(max(ev_[0], 0.0))), float(ev_[1] / max(ev_[0], 1e-9)))     # spread across the line (m), ratio along/across
        # Seen nearly end-on (the direction within SIDE_MIN of the line of sight), the lower edge is the near end of the
        # thing plus a little of its side, and the fitted direction is not to be trusted (bench: 10-30 degrees off, and
        # flipping between looks): then it is as unknown as if it were short.
        sight = mid_ - np.array([c, 0.0]); sight = sight / max(float(np.hypot(*sight)), 1e-6)
        if abs(float(a_ @ sight)) < math.cos(math.radians(SIDE_MIN)): a, L, mid = a_, L_, mid_
        else: ambiguous = True
    # the lower-edge line sits on the near side of the thing; the middle of the thing is w/2 farther from the camera
    nrm = np.array([-a[1], a[0]])
    if nrm @ (mid - np.array([c, 0.0])) < 0: nrm = -nrm
    ctr = mid + nrm * w / 2
    ss = np.linspace(-L / 2, L / 2, max(3, int(L / 0.01) + 1)); tt = np.linspace(-w / 2, w / 2, 9)
    z = np.maximum(w / 2 + np.sqrt(np.maximum(0.0, (w / 2) ** 2 - tt ** 2)), 0.004)
    pts = np.array([[*(ctr + s_ * a + t_ * nrm), zz] for s_ in ss for t_, zz in zip(tt, z)])
    corners = np.array([ctr + sa * (L / 2) * a + sn * (w / 2) * nrm for sa in (-1, 1) for sn in (-1, 1)])
    hull = corners[ConvexHull(corners).vertices]
    heading = float(math.atan2(a[1], a[0]) % math.pi)
    return dict(rel=np.array([ctr[0], ctr[1], w / 2]), top=w, bottom=0.0, hull=hull, pts=pts, heading=heading, elong=L / w if L > 1.3 * w else 1.0,
                width=w, n_px=h['n_px'], dist=float(np.hypot(ctr[0] - c, ctr[1])), cut=h['cut'], blind=True, ambiguous=bool(ambiguous), length=float(L), fit=fit)


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
        self.gmat = object_materials(m, robot); self.has_clear = bool((self.gmat == 3).any())
        self.bg_opt = mujoco.MjvOption(); self.bg_opt.geomgroup[2] = 0
        self.rng = np.random.default_rng(1)

    def load(self):
        """Number of pixels that see something between the two finger pads (gripper frame: fingers along -z, fixed pad
        face at x = -0.009, jaws close along x; the moving pad's face is where the jaw angle puts it). Only the space
        between the pads counts: next to furniture, the region around the fingers is full of shelf."""
        from skills import jaw_gap
        gap = jaw_gap(self.r.joints()[5], 0.075)
        d = self.r.d
        self.seg.update_scene(d, self.cam); s = self.seg.render()
        self.dep.update_scene(d, self.cam); z = self.dep.render()
        lab = np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)
        bg = None
        if self.has_clear:
            self.dep.update_scene(d, self.cam, scene_option=self.bg_opt); bg = self.dep.render()
        z = spoil_depth(z, lab, self.gmat, self.rng, bg).astype(np.float64)
        ok = ~np.isin(lab, self.self_geoms) & (z < 0.5) & (z > 0.01) & ~depth_edges(z)
        z = z + self.rng.normal(0, 0.001, z.shape)
        Pg = (self.ray[ok] * z[ok][:, None]) @ self.R.T + self.t
        inside = (Pg[:, 0] > -0.011) & (Pg[:, 0] < -0.009 + gap + 0.003) & (np.abs(Pg[:, 1]) < 0.012) & (Pg[:, 2] < -0.045) & (Pg[:, 2] > -0.106)
        return int(inside.sum())

    def holding(self):
        return self.load() >= 10        # empty: 0; a plush toy squashed to the jaw stop: ~30

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
