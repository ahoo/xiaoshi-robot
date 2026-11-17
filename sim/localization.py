"""Lidar localisation against the robot's own map (the front end of a 2D SLAM system, no loop closure).

Every control tick the pose is predicted from wheel odometry. Every lidar scan (10 Hz) the pose is corrected by a
two-stage correlative search: scan end points are scored against a likelihood field built from the occupied cells
of the map, over a grid of small shifts and turns around the prediction. Then the map is updated from the corrected
pose (in controller.Agent.sense). For matching the localiser keeps its own finer grid of lidar returns (2.5 cm),
built from the corrected poses. Nothing here reads the simulator's ground truth.
"""
import math

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

MATCH_SIGMA = 0.025         # m: width of the likelihood field around mapped walls
FINE = 0.025                # m: cell size of the localiser's own grid of lidar returns
MIN_MAP_CELLS = 40          # do not match until the map has this many confident cells
MIN_POINTS = 40             # nor when the scan has fewer usable end points than this
MIN_HITS = 2                # a fine cell counts as a wall once it has been hit this many times


def _grid(half, step):
    return np.arange(-half, half + 1e-9, step)


class Localizer:
    def __init__(self, pose, cs, size):
        self.est = np.array(pose, float)
        self.hits = np.zeros((int(round(size[1] / FINE)), int(round(size[0] / FINE))), np.int16)
        self._odom_prev = None; self._field = None; self._dirty = True
        self.n_match = 0; self.n_skip = 0
        self.last_score = 0.0

    def add_scan(self, px, py):
        """Record scan end points (robot frame) at the current estimate in the fine grid of returns."""
        x, y, th = self.est; c, s = math.cos(th), math.sin(th)
        i = np.floor((x + c * px - s * py) / FINE).astype(int); j = np.floor((y + s * px + c * py) / FINE).astype(int)
        ok = (i >= 0) & (i < self.hits.shape[1]) & (j >= 0) & (j < self.hits.shape[0])
        np.add.at(self.hits, (j[ok], i[ok]), 1)
        np.minimum(self.hits, 1000, out=self.hits); self._dirty = True

    def predict(self, odom):
        """Apply the change in the odometry pose since the last call, expressed in the robot frame."""
        odom = np.asarray(odom, float)
        if self._odom_prev is not None:
            dx, dy = odom[:2] - self._odom_prev[:2]; t0 = self._odom_prev[2]
            bx, by = math.cos(t0) * dx + math.sin(t0) * dy, -math.sin(t0) * dx + math.cos(t0) * dy
            th = self.est[2]
            self.est[0] += math.cos(th) * bx - math.sin(th) * by
            self.est[1] += math.sin(th) * bx + math.cos(th) * by
            self.est[2] = (self.est[2] + odom[2] - t0 + math.pi) % (2 * math.pi) - math.pi
        self._odom_prev = odom.copy()

    def correct(self, px, py):
        """(px, py): scan end points in the robot frame. Updates self.est in place; returns True if it moved it."""
        occ = self.hits >= MIN_HITS
        if occ.sum() < MIN_MAP_CELLS or len(px) < MIN_POINTS:
            self.n_skip += 1; return False
        if len(px) > 120:
            k = np.linspace(0, len(px) - 1, 120).astype(int); px, py = px[k], py[k]
        if self._dirty:
            self._field = distance_transform_edt(~occ) * FINE         # m to the nearest wall cell centre
            self._dirty = False
        field = self._field
        x0, y0, t0 = self.est

        def score(xs, ys, ts):
            best = (-1.0, x0, y0, t0)
            for t in ts:
                c, s = math.cos(t), math.sin(t)
                wx = c * px - s * py; wy = s * px + c * py
                X = (xs[:, None, None] + wx[None, None, :]); Y = (ys[None, :, None] + wy[None, None, :])
                X, Y = np.broadcast_arrays(X, Y)
                d = map_coordinates(field, [Y.ravel() / FINE - 0.5, X.ravel() / FINE - 0.5], order=1, mode='nearest')
                sc = np.exp(-0.5 * (d / MATCH_SIGMA) ** 2).reshape(X.shape).mean(-1)
                i, j = np.unravel_index(np.argmax(sc), sc.shape)
                if sc[i, j] > best[0]: best = (float(sc[i, j]), float(xs[i]), float(ys[j]), float(t))
            return best

        here = score(np.array([x0]), np.array([y0]), [t0])[0]
        sc, bx, by, bt = score(x0 + _grid(0.06, 0.02), y0 + _grid(0.06, 0.02), t0 + _grid(math.radians(3), math.radians(1)))
        sc, bx, by, bt = score(bx + _grid(0.01, 0.005), by + _grid(0.01, 0.005), bt + _grid(math.radians(0.5), math.radians(0.25)))
        self.last_score = sc
        if sc < 0.3 or sc <= here:
            self.n_skip += 1; return False
        self.est[:] = [bx, by, (bt + math.pi) % (2 * math.pi) - math.pi]
        self.n_match += 1
        return True
