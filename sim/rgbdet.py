"""People in the colour image: a person detector (boxes) and how its boxes are put together with the depth image.

Why. Depth alone cannot tell two people who stand close from one: on the 4 cm top-down grid they are one blob, and a child behind a grown-up is simply not there
(status.md round 14, acceptance item E2). A detector that works on the colour picture finds each person on its own; the depth image then gives where each one stands and how tall.

What is real here and what is a stand-in:

  BoxDetector         the interface: `detect(cam) -> [(u0, v0, u1, v1, score), ...]`, pixel boxes in the camera's image.
  SimBoxes            STAND-IN. The simulator cannot render people the way a trained network sees them, so this takes the true outline of every person in the segmentation image and
                      returns a box with the mistakes a small detector (YOLO/NanoDet class, on the Pi 5's CPU or an accelerator) is assumed to make: it misses people that are small in the
                      picture or mostly hidden behind somebody, it shakes the box a little, now and then it sees a person where there is none. THE NUMBERS ARE ASSUMED, NOT MEASURED: what
                      this tests is what is ours (the fusion, what the safety layer does with a detector that misses, how much the outcome depends on how good the detector is), not how
                      good a real network is on a real child; that has to be measured on the real camera with the real model (validation.md).
  OnnxBoxes           the real thing, for the day there is a model file: an ONNX person detector (YOLOv5/v8 style output) run with onnxruntime on the colour picture of the same camera.
  box_detections      boxes + depth -> detections for the tracker (people.PersonDetector): position, height, shoulder width from the depth points inside each box.

The depth-only clusters that no box explains are kept, but as 'unsure' (the safety layer treats those as a child): a person the detector missed in one frame is still slowed for.
"""
import math

import mujoco
import numpy as np

SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-x))


class SimBoxes:
    """The stand-in detector (see the module text). cam: the people camera (people.HeadCam / ArmCam), which keeps its last segmentation image in `cam.lab`.
    recall: the best case (a clear, large, unobstructed person); h50_px: box height in pixels at which half of those are found; vis50: visible share of the outline at which half are found;
    fp_per_frame: boxes on nothing per frame (they are thrown out later for having no person-like depth under them); jitter: box edges shake by this share of the box size."""
    def __init__(self, robot, names, rng, recall=0.97, h50_px=28.0, h_slope=6.0, vis50=0.30, vis_slope=0.08, fp_per_frame=0.02, jitter=0.04):
        from people import SPECS
        self.rng = rng; m = robot.m; self.p = dict(recall=recall, h50=h50_px, hs=h_slope, v50=vis50, vs=vis_slope, fp=fp_per_frame, jit=jitter)
        self.geom_person = np.full(m.ngeom, -1, np.int32)
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ''
            if nm.startswith('p') and '_' in nm and nm[1:nm.index('_')].isdigit(): self.geom_person[g] = int(nm[1:nm.index('_')])
        self.size = [(SPECS[n]['h'], SPECS[n]['w']) for n in names]
        self.last = []

    def recall_at(self, h_px, vis):
        p = self.p; return p['recall'] * float(SIGMOID((h_px - p['h50']) / p['hs'])) * float(SIGMOID((vis - p['v50']) / p['vs']))

    def detect(self, cam):
        lab = cam.lab; H, W = lab.shape; pid = np.where(lab >= 0, self.geom_person[np.maximum(lab, 0)], -1); f = cam.f; rng = self.rng; out = []; self.last = []
        for i in np.unique(pid[pid >= 0]):
            mask = pid == i; n = int(mask.sum())
            if n < 30: continue
            v, u = np.nonzero(mask); v0, v1, u0, u1 = int(v.min()), int(v.max()), int(u.min()), int(u.max())
            z = cam.depth[mask]; z = z[z > 0.1]
            if not len(z): continue
            d = float(np.median(z)); h_true, w_true = self.size[i]
            area_full = 0.5 * (f * h_true / d) * (f * w_true / d)                      # the outline of the whole person in pixels (a body fills about half of its box)
            vis = min(1.0, n / max(area_full, 1.0)); h_px = v1 - v0 + 1
            if rng.random() >= self.recall_at(h_px, vis): self.last.append((int(i), False, vis, h_px)); continue
            bw, bh = u1 - u0 + 1, v1 - v0 + 1; j = self.p['jit']
            box = (float(np.clip(u0 + rng.normal(0, j * bw), 0, W - 1)), float(np.clip(v0 + rng.normal(0, j * bh), 0, H - 1)), float(np.clip(u1 + rng.normal(0, j * bw), 0, W - 1)), float(np.clip(v1 + rng.normal(0, j * bh), 0, H - 1)))
            out.append(box + (float(np.clip(0.55 + 0.4 * vis + rng.normal(0, 0.05), 0.3, 0.99)),)); self.last.append((int(i), True, vis, h_px))
        for _ in range(int(rng.poisson(self.p['fp']))):                                 # a box on nothing
            bh = rng.uniform(30, 0.7 * H); bw = 0.45 * bh; cu = rng.uniform(bw / 2, W - bw / 2); cv = rng.uniform(bh / 2, H - bh / 2)
            out.append((cu - bw / 2, cv - bh / 2, cu + bw / 2, cv + bh / 2, float(rng.uniform(0.35, 0.6))))
        return out


class OnnxBoxes:
    """A real person detector: an ONNX model with YOLOv5/YOLOv8-style output run on the camera's colour picture. Nothing in the repository ships a model; give the path of one
    (`OnnxBoxes('yolov8n.onnx')`, 80-class COCO models work: only class 0, person, is used). Untested without a model file: it exists so that the day there is one, this is the only thing to write."""
    def __init__(self, path, conf=0.35, iou=0.5, size=640):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=['CPUExecutionProvider']); self.inp = self.sess.get_inputs()[0].name; self.conf, self.iou, self.size = conf, iou, size

    def detect(self, cam):
        rgb = cam.rgb(); H, W = rgb.shape[:2]; s = self.size / max(H, W); nh, nw = int(round(H * s)), int(round(W * s))
        yy = (np.arange(nh) / s).astype(int).clip(0, H - 1); xx = (np.arange(nw) / s).astype(int).clip(0, W - 1)
        img = np.full((self.size, self.size, 3), 114, np.uint8); img[:nh, :nw] = rgb[yy][:, xx]
        x = img[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        y = self.sess.run(None, {self.inp: x})[0][0]                                    # (84, N) YOLOv8 or (N, 85) YOLOv5
        if y.shape[0] < y.shape[1]: y = y.T                                             # -> (N, 4 + classes) or (N, 5 + classes)
        if y.shape[1] == 85: scores = y[:, 4] * y[:, 5]; cx, cy, bw, bh = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        else: scores = y[:, 4]; cx, cy, bw, bh = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        keep = scores > self.conf; boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)[keep] / s; scores = scores[keep]
        order = np.argsort(-scores); out = []
        for k in order:                                                                  # non-maximum suppression
            b = boxes[k]
            if all(_iou(b, o[:4]) < self.iou for o in out): out.append(tuple(float(v) for v in b) + (float(scores[k]),))
        return out


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); inter = iw * ih
    return inter / (max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9))


def box_detections(cam, boxes, W, ok, pose, UPPER=0.5, TOP_BIAS=0.05, MIN_TOP=0.6, GROUP_FOOT=0.70, MIN_PTS=25, WINDOW=0.30):
    """One detection (the dict people.PersonDetector tracks) for every box that has a person under it in the depth image, and the mask of the pixels they took (the foreground of their boxes).
    W: world points (H, W, 3) of the depth image through the robot's pose estimate, ok: valid mask. The person is the foreground of the box: the depth readings within WINDOW (0.30 m, about a body's thickness)
    of the median depth of its middle. Boxes are taken nearest first and a pixel belongs to the nearest person that claims it, so that somebody standing in front of somebody else (their boxes overlap)
    does not hide the one behind: that one's depth is measured on what is left."""
    out = []
    z = cam.depth; Hh, Ww = z.shape; pz = W[..., 2]; claimed = np.zeros(z.shape, bool)
    rr = np.hypot(W[..., 0] - pose[0], W[..., 1] - pose[1])                                         # how far each point is from the robot, along the floor: the window below is on this, not on the depth along the
                                                                                                    # optical axis, which grows with height when the camera looks up (a head would fall outside the window)
    def window(box):
        u0, v0, u1, v1, _ = box
        return int(max(0, math.floor(u0))), int(max(0, math.floor(v0))), int(min(Ww - 1, math.ceil(u1))), int(min(Hh - 1, math.ceil(v1)))
    def middle_depth(a, b, c, d, free):
        """The depth of the person in the box: the median of what is free (not taken by a nearer person) in the middle of the box; if too little of the middle is free (somebody stands in
        front of the body) the head and shoulders at the top of the box."""
        sz = rr[b:d + 1, a:c + 1]; sok = ok[b:d + 1, a:c + 1] & (z[b:d + 1, a:c + 1] > 0.1) & free[b:d + 1, a:c + 1]
        for fr0, fr1, fc0, fc1 in ((0.2, 0.7, 0.2, 0.8), (0.0, 0.3, 0.25, 0.75)):
            r0, r1, c0, c1 = int(fr0 * (d - b)), int(fr1 * (d - b)) + 1, int(fc0 * (c - a)), int(fc1 * (c - a)) + 1
            mid = sok[r0:r1, c0:c1]
            if mid.sum() >= 10: return float(np.median(sz[r0:r1, c0:c1][mid])), sz, sok
        return None, sz, sok
    free0 = np.ones(z.shape, bool); order = []
    for bx in boxes:
        a, b, c, d = window(bx)
        if c - a < 4 or d - b < 8: continue
        zm, _, _ = middle_depth(a, b, c, d, free0)
        if zm is not None: order.append((zm, (a, b, c, d)))
    for _, (a, b, c, d) in sorted(order, key=lambda t: t[0]):
        zm, sz, sok = middle_depth(a, b, c, d, ~claimed)                                    # the middle of the box without what a nearer person already took
        if zm is None: continue
        fg = sok & (np.abs(sz - zm) < WINDOW) & (pz[b:d + 1, a:c + 1] > 0.15)
        if fg.sum() < MIN_PTS: continue
        X, Y, Z = W[b:d + 1, a:c + 1, 0][fg], W[b:d + 1, a:c + 1, 1][fg], pz[b:d + 1, a:c + 1][fg]
        top = float(np.percentile(Z, 99)) + TOP_BIAS
        if top < MIN_TOP: continue
        cx, cy = float(np.median(X)), float(np.median(Y)); foot = max(np.percentile(X, 97) - np.percentile(X, 3), np.percentile(Y, 97) - np.percentile(Y, 3))
        band = (Z > 0.74 * top) & (Z < 0.88 * top); w = None
        if band.sum() >= 12:
            dx, dy = cx - pose[0], cy - pose[1]; n_ = math.hypot(dx, dy) + 1e-6; px, py = -dy / n_, dx / n_
            s_ = X[band] * px + Y[band] * py; w = float(np.percentile(s_, 97) - np.percentile(s_, 3))
        claimed[b:d + 1, a:c + 1] |= fg
        out.append(dict(x=cx, y=cy, h=top, cut=bool(b <= 2), w=w, n=int(fg.sum()), grp=bool(foot > GROUP_FOOT), src='box'))
    return out, claimed
