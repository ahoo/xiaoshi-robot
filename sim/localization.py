"""Lidar localisation against the robot's own map (the front end of a 2D SLAM system, no loop closure).

Every control tick the pose is predicted from wheel odometry. Every lidar scan (10 Hz) the pose is corrected by a
two-stage correlative search: scan end points are scored against a likelihood field built from the occupied cells
of the map, over a grid of small shifts and turns around the prediction. Then the map is updated from the corrected
pose (in controller.Agent.sense). For matching the localiser keeps its own finer grid of lidar returns (2.5 cm),
built from the corrected poses. Nothing here reads the simulator's ground truth.

Round 11: two changes made it hold up under noisy lidar (offline bench: test_localization.py).
  * Walls are thinned: a cell counts only if it has at least half the hits of the busiest cell near it. Before, hits
    piled up over thousands of scans, every cell noise had ever touched became wall, walls grew 15-20 cm thick, the
    score went flat across them and the estimate wandered, which thickened them further.
  * The score is a robust log-likelihood summed over the end points plus an odometry prior, so a correction has to be
    earned by the scan instead of taking the best of 300+ noisy candidates.
With 3 cm lidar noise this took the mean error from 20-25 cm to under 3 cm; with a realistic LD19 error model
(per-surface bias up to 4.5 cm, 1.5 cm steps, mixed pixels, 1 % outliers) from 3-5 cm (max 17) to 1.4-2.5 cm.
Neither change works on its own.
"""
import math

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates, maximum_filter

MATCH_SIGMA = 0.025         # m: width of the likelihood field around mapped walls
FINE = 0.025                # m: cell size of the localiser's own grid of lidar returns
MIN_MAP_CELLS = 40          # do not match until the map has this many confident cells
MIN_POINTS = 40             # nor when the scan has fewer usable end points than this
MIN_HITS = 2                # a fine cell counts as a wall once it has been hit this many times
MAX_PTS = 120               # scan end points used per match (evenly subsampled)
OCC_MODE = 'nms'            # 'abs': a cell is wall once hit MIN_HITS times; 'nms': also needs NMS_FRAC of the busiest cell within NMS_CELLS
NMS_FRAC = 0.5
NMS_CELLS = 5
SCORE = 'll'                # 'mean': mean Gaussian kernel; 'll': summed robust log-likelihood plus an odometry prior
LL_FLOOR = 0.05             # robust floor of the per-point likelihood (outliers, mixed pixels, things not yet mapped)
LL_DECORR = 2.0             # neighbouring end points are not independent: divide the summed log-likelihood by this
PRIOR_XY = (0.004, 0.05)    # odometry prior, 1 sigma: fixed part (m) + fraction of the distance driven since the last match
PRIOR_TH = (math.radians(0.3), 0.05)   # rad + fraction of the turn since the last match
RECOVER_BELOW = 0.75        # kernel score below this: also search wide (+-15 cm, +-12 deg) without the odometry prior,
RECOVER_GAIN = 0.10         # and jump there if it scores this much better. A bump into furniture turns the robot in a
                            # way the wheels do not see; the prior must not then hold the estimate where the wheels say.
DROP = 0.15                 # a scan scoring this far below the running average of recent scores is suspect: a bump or
                            # slip has misaligned it. It is not added to the map (it would corrupt it) and triggers the
                            # wide search. Normal scans dip at most ~0.09 below it, even with LD19 errors; exploring new
                            # ground lowers the score gradually, and the average follows, so mapping never stalls.


def _grid(half, step):
    return np.arange(-half, half + 1e-9, step)


class Localizer:
    def __init__(self, pose, cs, size):
        self.est = np.array(pose, float)
        self.hits = np.zeros((int(round(size[1] / FINE)), int(round(size[0] / FINE))), np.int16)
        self._odom_prev = None; self._field = None; self._dirty = True
        self.n_match = 0; self.n_skip = 0
        self.last_score = 0.0; self.n_recover = 0; self.base = None; self.map_ok = True
        self._moved = 0.0; self._turned = 0.0

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
            self._moved += math.hypot(bx, by); self._turned += abs((odom[2] - t0 + math.pi) % (2 * math.pi) - math.pi)
            self.est[0] += math.cos(th) * bx - math.sin(th) * by
            self.est[1] += math.sin(th) * bx + math.cos(th) * by
            self.est[2] = (self.est[2] + odom[2] - t0 + math.pi) % (2 * math.pi) - math.pi
        self._odom_prev = odom.copy()

    def correct(self, px, py):
        """(px, py): scan end points in the robot frame. Updates self.est in place; returns True if it moved it."""
        occ = self.hits >= MIN_HITS
        if OCC_MODE == 'nms':
            occ &= self.hits >= NMS_FRAC * maximum_filter(self.hits, NMS_CELLS)
        if occ.sum() < MIN_MAP_CELLS or len(px) < MIN_POINTS:
            self.n_skip += 1; return False
        if len(px) > MAX_PTS:
            k = np.linspace(0, len(px) - 1, MAX_PTS).astype(int); px, py = px[k], py[k]
        if self._dirty:
            self._field = distance_transform_edt(~occ) * FINE         # m to the nearest wall cell centre
            self._dirty = False
        field = self._field
        x0, y0, t0 = self.est
        sxy = PRIOR_XY[0] + PRIOR_XY[1] * self._moved; sth = PRIOR_TH[0] + PRIOR_TH[1] * self._turned

        def score(xs, ys, ts, ll=SCORE == 'll'):
            best = (-1.0, x0, y0, t0)
            for t in ts:
                c, s = math.cos(t), math.sin(t)
                wx = c * px - s * py; wy = s * px + c * py
                X = (xs[:, None, None] + wx[None, None, :]); Y = (ys[None, :, None] + wy[None, None, :])
                X, Y = np.broadcast_arrays(X, Y)
                d = map_coordinates(field, [Y.ravel() / FINE - 0.5, X.ravel() / FINE - 0.5], order=1, mode='nearest')
                k = np.exp(-0.5 * (d / MATCH_SIGMA) ** 2).reshape(X.shape)
                if ll:
                    sc = np.log(LL_FLOOR + k).sum(-1) / LL_DECORR
                    sc = sc - 0.5 * (((xs[:, None] - x0) ** 2 + (ys[None, :] - y0) ** 2) / sxy ** 2 + ((t - t0 + math.pi) % (2 * math.pi) - math.pi) ** 2 / sth ** 2)
                else:
                    sc = k.mean(-1)
                i, j = np.unravel_index(np.argmax(sc), sc.shape)
                if best[0] == -1.0 or sc[i, j] > best[0]: best = (float(sc[i, j]), float(xs[i]), float(ys[j]), float(t))
            return best

        def kernel(bx, by, bt):
            c, s = math.cos(bt), math.sin(bt)
            d = map_coordinates(field, [(by + s * px + c * py) / FINE - 0.5, (bx + c * px - s * py) / FINE - 0.5], order=1, mode='nearest')
            return float(np.exp(-0.5 * (d / MATCH_SIGMA) ** 2).mean())

        here = score(np.array([x0]), np.array([y0]), [t0])[0]
        sc, bx, by, bt = score(x0 + _grid(0.06, 0.02), y0 + _grid(0.06, 0.02), t0 + _grid(math.radians(3), math.radians(1)))
        sc, bx, by, bt = score(bx + _grid(0.01, 0.005), by + _grid(0.01, 0.005), bt + _grid(math.radians(0.5), math.radians(0.25)))
        if SCORE == 'll':
            # acceptance on the plain kernel score at the chosen pose, so the 0.3 floor means the same as before
            self.last_score = kernel(bx, by, bt)
            suspect = self.base is not None and self.last_score < self.base - DROP
            if self.last_score < RECOVER_BELOW or suspect:
                _, wx_, wy_, wt_ = score(x0 + _grid(0.15, 0.03), y0 + _grid(0.15, 0.03), t0 + _grid(math.radians(12), math.radians(2)), ll=False)
                _, wx_, wy_, wt_ = score(wx_ + _grid(0.02, 0.01), wy_ + _grid(0.02, 0.01), wt_ + _grid(math.radians(1.5), math.radians(0.5)), ll=False)
                _, wx_, wy_, wt_ = score(wx_ + _grid(0.01, 0.005), wy_ + _grid(0.01, 0.005), wt_ + _grid(math.radians(0.5), math.radians(0.25)), ll=False)
                ws = kernel(wx_, wy_, wt_)
                if ws > self.last_score + RECOVER_GAIN:
                    bx, by, bt = wx_, wy_, wt_; self.last_score = ws; self.n_recover += 1
            ok = self.last_score >= 0.3
            self.map_ok = self.n_match < 20 or self.base is None or self.last_score >= self.base - DROP
            self.base = self.last_score if self.base is None else 0.95 * self.base + 0.05 * self.last_score
        else:
            self.last_score = sc; ok = sc >= 0.3 and sc > here
        if not ok:
            self.n_skip += 1; return False
        self.est[:] = [bx, by, (bt + math.pi) % (2 * math.pi) - math.pi]
        self.n_match += 1; self._moved = 0.0; self._turned = 0.0
        return True
