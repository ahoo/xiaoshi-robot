"""The home as a 3D map: a voxel grid fused from the front depth camera, with what the robot found in it.

  VoxelMap      a dense grid of log-odds (occupied / free), how often each voxel was seen, and the mean colour of what hit it
  Scan3D        takes every front-camera frame (the agent's depth_hook), puts the points in the world with the robot's own pose estimate
                and adds them to the map; the free space along each ray is carved out as well
  analyze()     floor, walls, furniture and small things out of the voxels. Walls come from the 2D lidar map (long straight lines),
                furniture is what the lidar map shows that is not wall, with its height from the voxels. A furniture box is the box round
                what was SEEN: a sofa against a wall shows its front and sides, not how deep it is
  export_*      a coloured point cloud (PLY), a JSON of the rooms, furniture boxes and things, and a viewer page you can turn with the mouse
  evaluate()    against the simulator's true geometry (only the benches use it): how much of the surface the robot could have seen it did
                see (recall), how much of what it mapped really is there (precision), and the size error of each furniture box
  ScanPlanner   which stop to drive to next: the one from which most not-yet-seen volume lies in view (next-best-view), against the plain
                coverage tour the job uses

Heights: the low front camera (12 cm above the floor, looking 25 degrees down) sees the lower half metre of the room. Taller things need a
second camera that looks up (BOM); the map grid is as tall as you set it.
"""
import base64
import json
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

VOX = 0.04                    # m, voxel edge
FLOOR_LAYER = 1               # voxels below VOX * this (z < 4 cm) are floor
OCC_LOG = 4                   # log-odds above this: occupied (two hits)
L_HIT, L_FREE, L_MIN, L_MAX = 3, -1, -8, 40
MAX_RANGE = 3.0


def min_area_rect(xy):
    """Smallest rectangle round 2D points: (cx, cy, length along yaw, width across, yaw)."""
    xy = np.asarray(xy, float)
    if len(xy) < 3: c = xy.mean(0); return c[0], c[1], 0.04, 0.04, 0.0
    try: h = xy[ConvexHull(xy).vertices]
    except (QhullError, ValueError): c = xy.mean(0); return c[0], c[1], float(np.ptp(xy[:, 0])) + 0.04, float(np.ptp(xy[:, 1])) + 0.04, 0.0
    best = None
    for k in range(len(h)):
        e = h[(k + 1) % len(h)] - h[k]; a = math.atan2(e[1], e[0]); c, s = math.cos(a), math.sin(a)
        u = h @ np.array([[c, -s], [s, c]]); lo, hi = u.min(0), u.max(0); area = (hi[0] - lo[0]) * (hi[1] - lo[1])
        if best is None or area < best[0]: best = (area, a, lo, hi)
    _, a, lo, hi = best; c, s = math.cos(a), math.sin(a)
    mid = (lo + hi) / 2; cx, cy = c * mid[0] - s * mid[1], s * mid[0] + c * mid[1]
    L, Wd = hi[0] - lo[0], hi[1] - lo[1]
    if Wd > L: L, Wd, a = Wd, L, a + math.pi / 2
    return float(cx), float(cy), float(L), float(Wd), float((a + math.pi / 2) % math.pi - math.pi / 2)


class VoxelMap:
    def __init__(self, size=(4.0, 3.0, 1.0), vox=VOX, origin=(0.0, 0.0, 0.0)):
        self.vox = vox; self.origin = np.array(origin, float)
        self.n = np.ceil(np.array(size) / vox).astype(int)
        self.lo = np.zeros(self.n, np.int16)
        self.seen = np.zeros(self.n, np.uint8)
        self.csum = np.zeros(tuple(self.n) + (3,), np.float32); self.ccnt = np.zeros(self.n, np.uint16)
        self.frames = 0; self.poses = []

    # ---- indexing
    def idx(self, P):
        ij = np.floor((np.asarray(P, float) - self.origin) / self.vox).astype(int)
        return ij, np.all((ij >= 0) & (ij < self.n), -1)

    def centre(self, ij):
        return self.origin + (np.asarray(ij, float) + 0.5) * self.vox

    # ---- fusing a frame
    def integrate(self, P, origin, color=None, stride=4, step=None):
        """P: Nx3 hit points (world), origin: the sensor (world). Occupied at the hits, free along the rays before them."""
        step = step or self.vox / 2
        ij, ok = self.idx(P); P = P[ok]; ij = ij[ok]
        if not len(P): return
        flat = np.ravel_multi_index(ij.T, self.n)
        u, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
        self.lo.flat[u] = np.clip(self.lo.flat[u] + L_HIT * np.minimum(cnt, 3), L_MIN, L_MAX)
        self.seen.flat[u] = np.minimum(self.seen.flat[u].astype(int) + 1, 255)
        if color is not None:
            c = color[ok]; sums = np.zeros((len(u), 3)); np.add.at(sums, inv, c)
            cs = self.csum.reshape(-1, 3); cs[u] += sums.astype(np.float32); self.ccnt.flat[u] += np.minimum(cnt, 60000).astype(np.uint16)
        sel = np.arange(0, len(P), stride); d = P[sel] - origin; L = np.linalg.norm(d, axis=1); k = L > 0.15
        d, L = d[k] / L[k, None], L[k]
        if not len(L): self.frames += 1; return
        t = np.arange(0.10, min(MAX_RANGE, L.max()) - 0.06, step)
        pts = origin + d[:, None, :] * t[None, :, None]
        keep = t[None, :] < (L[:, None] - 0.06)
        q, inside = self.idx(pts.reshape(-1, 3)); m = keep.reshape(-1) & inside
        f = np.unique(np.ravel_multi_index(q[m].T, self.n))
        self.lo.flat[f] = np.clip(self.lo.flat[f] + L_FREE, L_MIN, L_MAX); self.seen.flat[f] = np.minimum(self.seen.flat[f].astype(int) + 1, 255)
        self.frames += 1

    # ---- reading it
    def occupied(self): return self.lo > OCC_LOG

    def points(self, with_floor=True):
        occ = self.occupied()
        if not with_floor: occ = occ.copy(); occ[:, :, :FLOOR_LAYER] = False
        ij = np.argwhere(occ); c = self.centre(ij)
        cnt = self.ccnt[tuple(ij.T)].astype(np.float32)
        rgb = np.where(cnt[:, None] > 0, self.csum[tuple(ij.T)] / np.maximum(cnt[:, None], 1), 160.0)
        return c, np.clip(rgb, 0, 255).astype(np.uint8), ij

    def observed(self): return self.seen > 0

    def observed_fraction(self, zmax=0.6, interior=None):
        """Share of the volume between the floor layer and zmax (inside `interior`, a 2D mask over x, y) that has been looked at."""
        nz = int(zmax / self.vox); sl = self.seen[:, :, FLOOR_LAYER:nz] > 0
        if interior is not None: sl = sl[interior]; return float(sl.mean()) if sl.size else 0.0
        return float(sl.mean())


class Scan3D:
    """The depth hook: every front-camera frame becomes points in the world, with colours when the colour image is wanted."""
    def __init__(self, vmap, colour=True, stride=3, min_range=0.15):
        self.map, self.colour, self.stride, self.min_range = vmap, colour, stride, min_range
        self.on = True; self.frames = 0; self.t_ms = 0.0

    def hook(self, agent, x, y, th):
        if not self.on or agent.freeze: return
        import time
        t0 = time.time(); cam = agent.cam
        P, valid = getattr(cam, 'last_cloud', None) or cam._cloud()      # chassis frame, z above the fitted floor, noise as the sensor has it
        v = valid & (np.hypot(P[..., 0] - cam.t_bc[0], P[..., 1] - cam.t_bc[1]) > self.min_range)
        v[::, :] &= (np.arange(P.shape[0])[:, None] % self.stride == 0) & (np.arange(P.shape[1])[None, :] % self.stride == 0)
        pts = P[v]
        if not len(pts): return
        c, s = math.cos(th), math.sin(th)
        W = np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1], pts[:, 2]], 1)
        o = np.array([x + c * cam.t_bc[0] - s * cam.t_bc[1], y + s * cam.t_bc[0] + c * cam.t_bc[1], cam.t_bc[2]])
        col = None
        if self.colour:
            img = cam.rgb(); col = img[v].astype(np.float32)
        self.map.integrate(W, o, col, stride=2)
        self.map.poses.append((float(x), float(y), float(th)))
        self.frames += 1; self.t_ms += 1000 * (time.time() - t0)


# ---------------------------------------------------------------------------------------------- what is in the map
def wall_lines(occ2d, cell=0.05, min_len=1.8, tol=0.06, gap=1.2, iters=10, rng=None):
    """Long straight lines in a 2D occupancy grid (occ2d[j, i] -> x = (i + .5) cell, y = (j + .5) cell): RANSAC on the occupied cells, repeated.
    Returns [(p0, p1)] end points; a wall is whatever lies along a line at least min_len long, with gaps up to `gap` allowed (furniture in front of it)."""
    rng = rng or np.random.default_rng(0)
    ys, xs = np.nonzero(occ2d); pts = np.stack([(xs + 0.5) * cell, (ys + 0.5) * cell], 1)
    left = np.ones(len(pts), bool); lines = []
    for _ in range(iters):
        idx = np.nonzero(left)[0]
        if len(idx) < 12: break
        best = None
        for _ in range(150):
            a, b = pts[rng.choice(idx, 2, replace=False)]; e = b - a; n = np.hypot(*e)
            if n < 0.4: continue
            nrm = np.array([-e[1], e[0]]) / n; inl = idx[np.abs((pts[idx] - a) @ nrm) < tol]
            if best is None or len(inl) > len(best): best = inl
        if best is None or len(best) * cell < 0.4 * min_len: break
        p = pts[best]; c0 = p.mean(0); w = np.linalg.svd(p - c0, full_matrices=False)[2][0]
        # keep the longest run along the line, where a gap of up to `gap` (something stands in front of the wall) does not break it
        s = (p - c0) @ w; o = np.argsort(s); s_sorted = s[o]; cut = np.nonzero(np.diff(s_sorted) > gap)[0]
        runs = np.split(np.arange(len(s_sorted)), cut + 1); run = max(runs, key=lambda r: s_sorted[r[-1]] - s_sorted[r[0]])
        s0, s1 = s_sorted[run[0]], s_sorted[run[-1]]
        left[best] = False
        if s1 - s0 < min_len: continue
        lines.append((c0 + w * s0, c0 + w * s1))
    return lines


def _near_line(P, lines, tol):
    """Distance of 2D points to the nearest of the segments (inf if none)."""
    d = np.full(len(P), np.inf)
    for a, b in lines:
        e = b - a; L2 = float(e @ e); t = np.clip(((P - a) @ e) / max(L2, 1e-9), 0, 1)
        d = np.minimum(d, np.linalg.norm(P - (a + t[:, None] * e), axis=1))
    return d


def _outer_ok(vmap, a, b, centre, depth=0.8, max_free=6):
    """A wall is the outer edge of the space the robot has been in: on its far side nothing has been seen to be free. Furniture standing in front of
    something is not: the robot sees the gap beside it, and behind it the free floor continues."""
    e = b - a; L = float(np.hypot(*e)); u = e / max(L, 1e-9); nrm = np.array([-u[1], u[0]])
    if nrm @ (centre - (a + b) / 2) > 0: nrm = -nrm
    ss = np.arange(0.1, max(L - 0.1, 0.11), vmap.vox); dd = np.arange(0.12, depth, vmap.vox)
    P = a[None, None, :] + u[None, None, :] * ss[:, None, None] + nrm[None, None, :] * dd[None, :, None]
    free = 0
    for z in (FLOOR_LAYER + 1, FLOOR_LAYER + 2):
        q, ok = vmap.idx(np.concatenate([P.reshape(-1, 2), np.full((P.shape[0] * P.shape[1], 1), (z + 0.5) * vmap.vox)], 1))
        q = q[ok]; free += int(((vmap.lo[tuple(q.T)] < 0) & (vmap.seen[tuple(q.T)] > 0)).sum())
    return free <= max_free


def analyze(vmap, lidar_occ=None, lidar_cell=0.05, furn_min_h=0.12, min_vox=30, wall_tol=0.10):
    """-> dict(walls, furniture, clutter, stats). Walls: long straight lines of the agent's 2D lidar map (lidar_occ, bool [j, i]), with the height the
    voxels reach along them. Furniture: connected clumps of the remaining occupied voxels at least furn_min_h tall; each as the smallest rectangle round
    what was seen, and the height seen. Clutter: low clumps (the object records follow those up)."""
    occ = vmap.occupied().copy(); occ[:, :, :FLOOR_LAYER] = False
    ij = np.argwhere(occ); out = dict(walls=[], furniture=[], clutter=[])
    if not len(ij):
        out['stats'] = dict(frames=vmap.frames, occupied=0, observed=int(vmap.observed().sum()), voxels=int(np.prod(vmap.n))); return out
    C = vmap.centre(ij); xy = C[:, :2]
    cols = np.zeros(vmap.n[:2], bool); hi = ij[:, 2] * vmap.vox + vmap.vox >= furn_min_h; cols[tuple(ij[hi][:, :2].T)] = True       # columns that reach 12 cm: dense, unlike the lidar's 1.5-degree rays
    lines = wall_lines(cols.T, vmap.vox) if lidar_occ is None else wall_lines(lidar_occ, lidar_cell)
    dw = _near_line(xy, lines, wall_tol) if lines else np.full(len(xy), np.inf); is_wall = dw < wall_tol
    fr = np.argwhere((vmap.lo[:, :, FLOOR_LAYER + 1] < 0) & (vmap.seen[:, :, FLOOR_LAYER + 1] > 0))
    centre = vmap.centre(np.column_stack([fr, np.full(len(fr), FLOOR_LAYER + 1)]))[:, :2].mean(0) if len(fr) else xy.mean(0)
    lines = [(a, b) for a, b in lines if _outer_ok(vmap, a, b, centre)]
    dw = _near_line(xy, lines, wall_tol) if lines else np.full(len(xy), np.inf); is_wall = dw < wall_tol
    for a, b in lines:
        near = _near_line(xy, [(a, b)], wall_tol) < wall_tol
        top = float(np.percentile(C[near, 2], 95) + vmap.vox / 2) if near.sum() else 0.0
        out['walls'].append(dict(p0=[float(a[0]), float(a[1])], p1=[float(b[0]), float(b[1])], length=float(np.linalg.norm(b - a)), height_seen=top))
    rest = ~is_wall; keep = np.zeros(vmap.n, bool); keep[tuple(ij[rest].T)] = True
    lab, n = ndimage.label(ndimage.binary_dilation(keep, structure=np.ones((3, 3, 3))), structure=np.ones((3, 3, 3)))     # clumps: one voxel of gap is still one thing
    lab = np.where(keep, lab, 0)
    for k in range(1, n + 1):
        vv = np.argwhere(lab == k)
        if len(vv) < 2: continue
        P = vmap.centre(vv); hh = float(P[:, 2].max() + vmap.vox / 2)
        cx, cy, L, Wd, yaw = min_area_rect(P[:, :2]); L, Wd = L + vmap.vox, Wd + vmap.vox
        near_wall = bool(lines) and float(_near_line(P[:, :2], lines, 1e9).max()) < 0.25
        if near_wall and max(L, Wd) < 0.5: continue                            # what is left of a wall at the ends of its line
        if hh >= furn_min_h and len(vv) >= min_vox:
            out['furniture'].append(dict(centre=[cx, cy], length=L, width=Wd, yaw=yaw, height_seen=hh, voxels=int(len(vv))))
        elif hh < furn_min_h:
            out['clutter'].append(dict(centre=[float(P[:, 0].mean()), float(P[:, 1].mean())], voxels=int(len(vv)), height=hh))
    out['furniture'].sort(key=lambda f: -f['length'] * f['width'] * f['height_seen'])
    for k, f in enumerate(out['furniture']): f['id'] = k + 1
    out['stats'] = dict(frames=vmap.frames, occupied=int(vmap.occupied().sum()), observed=int(vmap.observed().sum()), voxels=int(np.prod(vmap.n)))
    return out


# ---------------------------------------------------------------------------------------------- where to look next
class ScanPlanner:
    """Next-best-view: from every stop the robot could drive to, how much not-yet-seen volume (between the floor layer and zmax) lies in view,
    counting along rays that stop at the first occupied voxel or at the edge of the grid. The score is that volume over the distance to get there.
    A full turn on the spot is assumed at each stop (the camera sees 88 degrees wide; the robot turns), so all directions count.
    The volume is counted in blocks of 3 x 3 x 3 voxels (12 cm): a block is seen if any voxel in it was, because the rays of one frame are a few
    centimetres apart far away and leave single voxels between them unvisited even though the camera looked there.
    Stops already made (and stops that could not be reached) are not asked for again within `apart`."""
    def __init__(self, vmap, zmin=0.06, zmax=0.55, reach=2.5, n_az=24, elev_deg=(-40, -22, -6, 8), cam_h=0.126, grid=0.30, min_gain=15, apart=0.6, block=3):
        self.map = vmap; self.zmin, self.zmax, self.reach, self.cam_h, self.grid, self.min_gain, self.apart, self.b = zmin, zmax, reach, cam_h, grid, min_gain, apart, block
        az = np.arange(n_az) * 2 * math.pi / n_az; el = np.radians(elev_deg)
        A, E = np.meshgrid(az, el); self.dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        self.cv = vmap.vox * block; self.steps = np.arange(0.2, reach, self.cv)
        self.bad = []; self.visited = []

    def _coarse(self):
        vm, b = self.map, self.b; n = (vm.n // b) * b
        seen = (vm.seen[:n[0], :n[1], :n[2]] > 0).reshape(n[0] // b, b, n[1] // b, b, n[2] // b, b).max((1, 3, 5))
        occ = (vm.lo[:n[0], :n[1], :n[2]] > OCC_LOG).reshape(n[0] // b, b, n[1] // b, b, n[2] // b, b).max((1, 3, 5))
        return seen, occ

    def gains(self, origins):
        """Unseen blocks in view from each origin (N x 3, world, at camera height)."""
        vm = self.map; seen, occ = self._coarse(); nb = np.array(seen.shape); out = np.zeros(len(origins))
        z0, z1 = max(int(self.zmin / self.cv), 0), int(self.zmax / self.cv) + 1
        for s in range(0, len(origins), 60):
            O = origins[s:s + 60]
            pts = O[:, None, None, :] + self.dirs[None, :, None, :] * self.steps[None, None, :, None]       # C x R x S x 3
            ij = np.floor((pts - vm.origin) / self.cv).astype(int); inside = np.all((ij >= 0) & (ij < nb), -1)
            safe = np.where(inside[..., None], ij, 0); sn = seen[safe[..., 0], safe[..., 1], safe[..., 2]]; oc = occ[safe[..., 0], safe[..., 1], safe[..., 2]]
            blocked = (~inside) | oc | (pts[..., 2] < 0.02)
            before = ~(np.maximum.accumulate(blocked, axis=2))                       # still in front of the first thing the ray hits
            unk = before & (~sn) & (safe[..., 2] >= z0) & (safe[..., 2] < z1)
            # count each block once per origin, not once per ray that crosses it
            for c in range(len(O)):
                idx = safe[c][unk[c]]
                out[s + c] = len(np.unique(idx, axis=0)) if len(idx) else 0
        return out

    def choose(self, agent, free_dist, pose=None, exclude=()):
        """free_dist: the agent's cost map (distance to the nearest obstacle, 0.05 m cells). Returns (x, y, gain) or None when nowhere shows enough that is new."""
        x0, y0, _ = pose or agent.pose(); cs = 0.05; step = max(1, int(round(self.grid / cs)))
        jj, ii = np.mgrid[0:free_dist.shape[0]:step, 0:free_dist.shape[1]:step]
        ok = free_dist[jj, ii] >= 0.26; xs = (ii[ok] + 0.5) * cs; ys = (jj[ok] + 0.5) * cs
        keep = np.ones(len(xs), bool)
        for bx, by in list(self.bad) + list(self.visited) + list(exclude): keep &= np.hypot(xs - bx, ys - by) > self.apart
        xs, ys = xs[keep], ys[keep]
        if not len(xs): return None
        g = self.gains(np.stack([xs, ys, np.full(len(xs), self.cam_h)], 1)); dist = np.hypot(xs - x0, ys - y0)
        score = np.where(g >= self.min_gain, g / (dist + 0.6), -1.0); k = int(np.argmax(score))
        if score[k] < 0: return None                      # nowhere left that shows enough that has not been seen
        return float(xs[k]), float(ys[k]), float(g[k])


# ---------------------------------------------------------------------------------------------- export
def export_ply(vmap, path, with_floor=True):
    c, rgb, _ = vmap.points(with_floor)
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    a = np.zeros(len(c), dt); a['x'], a['y'], a['z'] = c[:, 0], c[:, 1], c[:, 2]; a['r'], a['g'], a['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, 'wb') as f:
        f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(c)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode())
        f.write(a.tobytes())
    return len(c)


def viewer_data(vmap, info, objects=(), people=(), max_points=40000, name='家里'):
    """What the viewer page needs: quantised points (int16 x, y, z in mm from the map origin) with colours, boxes, objects."""
    c, rgb, _ = vmap.points(True)
    if len(c) > max_points: sel = np.random.default_rng(0).choice(len(c), max_points, replace=False); c, rgb = c[sel], rgb[sel]
    q = np.round((c - vmap.origin) * 1000).astype(np.int16)
    return dict(name=name, size=[float(v) for v in vmap.n * vmap.vox], n=int(len(c)), pts=base64.b64encode(q.tobytes()).decode(), rgb=base64.b64encode(rgb.tobytes()).decode(),
                walls=info['walls'], furniture=info['furniture'], objects=list(objects), people=list(people))


VIEWER_JS = r'''
(function(){
  var D = JSON.parse(document.getElementById('home3d-data').textContent);
  function b64(s, T){ var bin = atob(s), n = bin.length, u = new Uint8Array(n); for (var i = 0; i < n; i++) u[i] = bin.charCodeAt(i); return new T(u.buffer); }
  var P = b64(D.pts, Int16Array), C = b64(D.rgb, Uint8Array), N = D.n, S = D.size;
  var cv = document.getElementById('home3d'), g = cv.getContext('2d');
  var yaw = -0.9, pitch = 0.75, zoom = 1, drag = null, showLabels = true;
  var cx = S[0] / 2, cy = S[1] / 2, cz = 0.15;
  function resize(){ var r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1; cv.width = r.width * d; cv.height = r.height * d; draw(); }
  function proj(x, y, z){
    var dx = x - cx, dy = y - cy, dz = z - cz, c = Math.cos(yaw), s = Math.sin(yaw);
    var rx = c * dx - s * dy, ry = s * dx + c * dy, cp = Math.cos(pitch), sp = Math.sin(pitch);
    var py = cp * ry - sp * dz, pz = sp * ry + cp * dz;
    var f = Math.min(cv.width, cv.height * 1.4) / (Math.max(S[0], S[1]) * 1.25) * zoom;
    return [cv.width / 2 + rx * f, cv.height / 2 + py * f, pz];
  }
  function box(f){
    var c = Math.cos(f.yaw), s = Math.sin(f.yaw), L = f.length / 2, W = f.width / 2, H = f.height_seen, out = [];
    [[-1,-1],[1,-1],[1,1],[-1,1]].forEach(function(k){ out.push([f.centre[0] + c*k[0]*L - s*k[1]*W, f.centre[1] + s*k[0]*L + c*k[1]*W]); });
    g.strokeStyle = 'rgba(255,107,44,.95)'; g.lineWidth = 2;
    var lo = out.map(function(p){ return proj(p[0], p[1], 0.0); }), hi = out.map(function(p){ return proj(p[0], p[1], H); });
    g.beginPath(); for (var i = 0; i < 4; i++){ g.moveTo(lo[i][0], lo[i][1]); g.lineTo(lo[(i+1)%4][0], lo[(i+1)%4][1]); g.moveTo(hi[i][0], hi[i][1]); g.lineTo(hi[(i+1)%4][0], hi[(i+1)%4][1]); g.moveTo(lo[i][0], lo[i][1]); g.lineTo(hi[i][0], hi[i][1]); } g.stroke();
    if (showLabels){ var t = proj(f.centre[0], f.centre[1], H + 0.04); g.fillStyle = '#fff'; g.font = (13 * (window.devicePixelRatio||1)) + 'px sans-serif'; g.fillText(f.name || ('家具' + f.id), t[0] - 20, t[1]); }
  }
  function draw(){
    g.fillStyle = '#0c221e'; g.fillRect(0, 0, cv.width, cv.height);
    var f = Math.min(cv.width, cv.height * 1.4) / (Math.max(S[0], S[1]) * 1.25) * zoom, r = Math.max(1.5, 0.045 * f), idx = [], zs = new Float32Array(N);
    for (var i = 0; i < N; i++){ var p = proj(P[3*i]/1000, P[3*i+1]/1000, P[3*i+2]/1000); zs[i] = p[2]; }
    for (var i = 0; i < N; i++) idx.push(i);
    idx.sort(function(a, b){ return zs[a] - zs[b]; });
    for (var k = 0; k < N; k++){
      var i = idx[k], p = proj(P[3*i]/1000, P[3*i+1]/1000, P[3*i+2]/1000), sh = 0.75 + 0.25 * Math.min(1, P[3*i+2] / 600);
      g.fillStyle = 'rgb(' + (C[3*i]*sh|0) + ',' + (C[3*i+1]*sh|0) + ',' + (C[3*i+2]*sh|0) + ')'; g.fillRect(p[0] - r/2, p[1] - r/2, r, r);
    }
    D.furniture.forEach(box);
    (D.objects||[]).forEach(function(o){ var p = proj(o.x, o.y, 0.03); g.fillStyle = '#ffd166'; g.beginPath(); g.arc(p[0], p[1], 5*(window.devicePixelRatio||1), 0, 6.3); g.fill();
      if (showLabels && o.name){ g.fillStyle = '#ffd166'; g.font = (12*(window.devicePixelRatio||1)) + 'px sans-serif'; g.fillText(o.name, p[0] + 8, p[1] - 6); } });
    (D.people||[]).forEach(function(o){ var p = proj(o.x, o.y, 0.9), q = proj(o.x, o.y, 0); g.strokeStyle = '#7bdff2'; g.lineWidth = 3; g.beginPath(); g.moveTo(q[0], q[1]); g.lineTo(p[0], p[1]); g.stroke(); });
  }
  cv.addEventListener('mousedown', function(e){ drag = [e.clientX, e.clientY]; });
  window.addEventListener('mouseup', function(){ drag = null; });
  window.addEventListener('mousemove', function(e){ if (!drag) return; yaw += (e.clientX - drag[0]) * 0.008; pitch = Math.max(0.05, Math.min(1.5, pitch + (e.clientY - drag[1]) * 0.008)); drag = [e.clientX, e.clientY]; draw(); });
  cv.addEventListener('wheel', function(e){ e.preventDefault(); zoom = Math.max(0.5, Math.min(4, zoom * (e.deltaY < 0 ? 1.1 : 0.9))); draw(); }, {passive: false});
  var tch = null;
  cv.addEventListener('touchstart', function(e){ tch = [e.touches[0].clientX, e.touches[0].clientY]; });
  cv.addEventListener('touchmove', function(e){ if (!tch) return; e.preventDefault(); yaw += (e.touches[0].clientX - tch[0]) * 0.01; pitch = Math.max(0.05, Math.min(1.5, pitch + (e.touches[0].clientY - tch[1]) * 0.01)); tch = [e.touches[0].clientX, e.touches[0].clientY]; draw(); }, {passive: false});
  var b = document.getElementById('home3d-labels'); if (b) b.addEventListener('click', function(){ showLabels = !showLabels; draw(); });
  window.addEventListener('resize', resize); resize();
})();
'''


def viewer_html(data, title='家里的三维地图'):
    """A page of its own (no libraries): the scanned home as a point cloud you can turn with the mouse or a finger, with furniture boxes and what was found on the floor."""
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title>'
            '<style>body{margin:0;background:#0c221e;color:#eef0e4;font:14px sans-serif}#home3d{display:block;width:100%;height:70vh;touch-action:none;cursor:grab}'
            '.bar{padding:8px 12px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}button{background:#ff6b2c;border:0;color:#1c0c03;padding:6px 12px;border-radius:4px;font-weight:700}</style></head><body>'
            '<div class="bar"><b>' + title + '</b><span>拖动旋转，滚轮缩放</span><button id="home3d-labels">标签 开/关</button></div><canvas id="home3d"></canvas>'
            '<script type="application/json" id="home3d-data">' + json.dumps(data, ensure_ascii=False) + '</script><script>' + VIEWER_JS + '</script></body></html>')


# ---------------------------------------------------------------------------------------------- against the truth (benches only)
def true_surface_points(model, data, robot_root, zmax=0.65, spacing=0.03, names=None):
    """Points on the faces of the static world geometry (walls, furniture, basket, dock), up to zmax. Returns (points, geom ids)."""
    import mujoco
    pts, gid = [], []
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        if b != 0 or int(model.geom_type[g]) not in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CYLINDER)): continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        if nm == 'floor' or (names is not None and not any(nm.startswith(p) for p in names)): continue
        pos, R, sz = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            for ax in range(3):
                o = [a for a in range(3) if a != ax]
                for sgn in (-1, 1):
                    u = np.arange(-sz[o[0]] + spacing / 2, sz[o[0]], spacing); v = np.arange(-sz[o[1]] + spacing / 2, sz[o[1]], spacing)
                    if not len(u): u = np.array([0.0])
                    if not len(v): v = np.array([0.0])
                    uu, vv = np.meshgrid(u, v); loc = np.zeros((uu.size, 3)); loc[:, ax] = sgn * sz[ax]; loc[:, o[0]] = uu.ravel(); loc[:, o[1]] = vv.ravel()
                    w = pos + loc @ R.T; pts.append(w[(w[:, 2] <= zmax) & (w[:, 2] > 0.03)]); gid += [g] * len(pts[-1])
        else:   # cylinder: the side only
            r, hh = sz[0], sz[1]; th = np.arange(0, 2 * math.pi, spacing / max(r, 1e-3)); zz = np.arange(-hh + spacing / 2, hh, spacing)
            T, Z = np.meshgrid(th, zz); loc = np.stack([r * np.cos(T).ravel(), r * np.sin(T).ravel(), Z.ravel()], 1)
            w = pos + loc @ R.T; pts.append(w[(w[:, 2] <= zmax) & (w[:, 2] > 0.03)]); gid += [g] * len(pts[-1])
    return (np.concatenate(pts) if pts else np.zeros((0, 3))), np.array(gid)


def observable_mask(model, data, P, origins, tol=0.025, chunk=4000):
    """Which true surface points could be seen from at least one of the camera origins (a ray from the origin reaches the point unobstructed).
    The robot's own geoms are excluded from the ray test."""
    import mujoco
    seen = np.zeros(len(P), bool)
    gm = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    for o in origins:
        d = P - o; L = np.linalg.norm(d, axis=1); ok = (L > 0.12) & (L < MAX_RANGE) & ~seen
        idx = np.nonzero(ok)[0]
        for s in range(0, len(idx), chunk):
            sub = idx[s:s + chunk]; vec = (d[sub] / L[sub, None]).reshape(-1)
            gid = np.zeros(len(sub), np.int32); dist = np.zeros(len(sub))
            mujoco.mj_multiRay(model, data, np.asarray(o, float), vec, gm, 1, -1, gid, dist, None, len(sub), MAX_RANGE)
            seen[sub[(dist > 0) & (dist >= L[sub] - tol - 0.01)]] = True
    return seen


def evaluate(vmap, P_true, visible, tol=0.07):
    """recall: how many of the true surface points that could be seen are within `tol` of a mapped occupied voxel;
    precision: how many occupied voxels above the floor layer are within `tol` of a true surface point (or of `extra`, the things on the floor)."""
    from scipy.spatial import cKDTree
    occ_c, _, _ = vmap.points(with_floor=False)
    if not len(occ_c): return dict(recall=0.0, precision=0.0, n_true=int(visible.sum()), n_map=0)
    tree_map = cKDTree(occ_c); d, _ = tree_map.query(P_true[visible]); recall = float((d < tol).mean()) if visible.any() else float('nan')
    tree_true = cKDTree(P_true); d2, _ = tree_true.query(occ_c); prec = float((d2 < tol).mean())
    return dict(recall=recall, precision=prec, n_true=int(visible.sum()), n_map=int(len(occ_c)))
