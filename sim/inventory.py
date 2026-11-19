"""Records of the things in the home: what they look like, where they were seen, and where they went.

The robot's job memory (controller.Agent.memory) forgets: an object that is delivered or not found any more is dropped. This keeps what a person would
want to ask later: "where is the ball", "what is new on the floor since yesterday", "what did you pick up today". It survives from one session to the next
(a JSON file), and an item that was moved between sessions is recognised again by how it looks and how big it is, not by where it lies.

  Item        id, name (given by a person) and kind guess (with how sure), size, appearance (colour histogram), first/last seen, place, status, history
  observe()   called with every front-camera frame: the objects the agent has just seen are matched to items or become new ones
  where(), new_since(), moved_since(), missing(), summary()   the questions
  name_item() a person teaches a name: "this is Xiaoming's brick"

What the simulator cannot say: the kind guess uses size, shape and colour with hand-set prototypes. The simulator's eight kinds all have different colours, so
the accuracy it shows is the best case; real toys need a recognition model (the Recognizer hook), and identical items (two of the same ball) cannot be told apart.
"""
import json
import math

import numpy as np

# prototypes of the kinds the simulator has: (hue degrees or None for no colour, saturation, value), size L, W, H in cm (typical), soft
PROTO = {
    'ball':   dict(cn='小球', hue=70, L=(5.2, 6.8), H=(5.2, 6.8), round=True),
    'cube':   dict(cn='方块', hue=205, L=(3.5, 7.0), H=(3.0, 5.0), square=True),
    'car':    dict(cn='玩具车', hue=40, L=(8.0, 10.0), H=(3.5, 5.5), elong=(1.6, 3.0)),
    'banana': dict(cn='香蕉玩具', hue=52, L=(11.0, 16.5), H=(2.5, 3.8), elong=(2.5, 6.0)),
    'lbrick': dict(cn='L 形积木', hue=135, L=(5.0, 9.5), H=(1.8, 3.0)),
    'rock':   dict(cn='石头', hue=None, sat=(0.0, 0.2), L=(2.5, 6.5), H=(1.2, 4.5)),
    'sock':   dict(cn='袜子', hue=3, L=(9.0, 16.0), H=(3.0, 5.0), elong=(2.0, 5.0)),
    'plush':  dict(cn='毛绒玩具', hue=320, L=(7.0, 13.0), H=(4.0, 7.0)),
}
HUE_BINS = 12


def _hsv(rgb):
    rgb = np.asarray(rgb, float) / 255.0; mx = rgb.max(-1); mn = rgb.min(-1); d = mx - mn
    h = np.zeros_like(mx); nz = d > 1e-6
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.where(nz & (mx == r), ((g - b) / np.where(nz, d, 1)) % 6, h); h = np.where(nz & (mx == g), (b - r) / np.where(nz, d, 1) + 2, h); h = np.where(nz & (mx == b), (r - g) / np.where(nz, d, 1) + 4, h)
    return (h * 60.0) % 360.0, np.where(mx > 0, d / np.maximum(mx, 1e-6), 0.0), mx


def colour_descriptor(rgb):
    """rgb: Nx3 pixels of the object. -> dict(hist (hue histogram weighted by saturation), s, v, n)."""
    h, s, v = _hsv(rgb); w = s * (v > 0.15)
    hist = np.zeros(HUE_BINS)
    np.add.at(hist, np.minimum((h / 360.0 * HUE_BINS).astype(int), HUE_BINS - 1), w)
    tot = hist.sum(); hist = hist / tot if tot > 1e-6 else hist
    return dict(hist=hist.tolist(), s=float(s.mean()), v=float(v.mean()), n=int(len(rgb)))


def colour_distance(a, b):
    """0 (same) .. 1 (different): hue histograms circularly blurred and compared, plus how saturated and how bright."""
    ha, hb = np.array(a['hist']), np.array(b['hist'])
    k = np.array([0.25, 0.5, 0.25])
    ba = ha + np.roll(ha, 1) * 0.5 + np.roll(ha, -1) * 0.5; bb = hb + np.roll(hb, 1) * 0.5 + np.roll(hb, -1) * 0.5
    na, nb = np.linalg.norm(ba), np.linalg.norm(bb)
    cos = float(ba @ bb / (na * nb)) if na > 1e-6 and nb > 1e-6 else (1.0 if na < 1e-6 and nb < 1e-6 else 0.0)
    return 0.55 * (1 - cos) + 0.25 * abs(a['s'] - b['s']) + 0.20 * abs(a['v'] - b['v'])


def guess_kind(size_cm, elong, colour):
    """Hand-set prototypes: how well do size, shape and colour fit each kind. -> (kind, confidence 0..1, scores)."""
    L, W, H = size_cm; Ls = max(L, W); out = {}
    hist = np.array(colour['hist']); hue = None
    if hist.sum() > 1e-6 and colour['s'] > 0.18:
        ang = (np.arange(HUE_BINS) + 0.5) * 360.0 / HUE_BINS; hue = math.degrees(math.atan2((hist * np.sin(np.radians(ang))).sum(), (hist * np.cos(np.radians(ang))).sum())) % 360.0
    for k, p in PROTO.items():
        sc = 0.0
        # size: inside the range 1, falling off outside
        for val, rng in ((Ls, p['L']), (H, p['H'])):
            lo, hi = rng; d = max(lo - val, val - hi, 0.0) / max(0.4 * (hi - lo), 0.5); sc += math.exp(-d * d)
        if 'elong' in p:
            lo, hi = p['elong']; d = max(lo - elong, elong - hi, 0.0) / 0.8; sc += 0.8 * math.exp(-d * d)
        if p.get('round'): sc += 0.6 * math.exp(-((elong - 1.0) / 0.3) ** 2) * math.exp(-((H - Ls) / 1.2) ** 2)
        if p.get('square'): sc += 0.4 * math.exp(-((elong - 1.0) / 0.35) ** 2)
        # colour: the biggest weight where it is known
        if p.get('hue') is not None:
            if hue is not None:
                d = abs((hue - p['hue'] + 180) % 360 - 180); sc += 3.0 * math.exp(-(d / 22.0) ** 2)
            else: sc += 0.0
        else:   # grey things: low saturation
            sc += 2.0 * (1.0 if colour['s'] < 0.25 and colour['v'] > 0.25 else 0.1)
        out[k] = sc
    best = max(out, key=out.get); srt = sorted(out.values(), reverse=True)
    conf = float(np.clip((srt[0] - srt[1]) / 2.5, 0, 1)) * float(np.clip(srt[0] / 6.0, 0, 1))
    return best, conf, out


def furniture_name(f):
    """A first guess at what a piece of furniture is, from the box round what was seen. People rename it by voice."""
    L, W, H = max(f['length'], f['width']), min(f['length'], f['width']), f['height_seen']
    if H >= 0.55 and L < 1.3: return '架子'
    if L >= 1.2 and H <= 0.5: return '沙发'
    if 0.25 <= H <= 0.36 and 0.6 <= L <= 1.4: return '桌子'
    if H >= 0.38 and L <= 0.6 and W <= 0.6 and abs(L - W) < 0.2: return '椅子'
    if H <= 0.15: return '收纳筐'
    if L <= 0.5 and W <= 0.3 and H >= 0.28: return '盆栽'
    return '箱子'


class Inventory:
    def __init__(self, path=None, session=0):
        self.items = {}; self.next_id = 1; self.session = session; self.path = path; self.track = {}      # track: this session's memory id -> item id
        self.furniture = []; self.t = 0.0; self.events = []; self.unseen = {}; self._active = set()
        if path:
            try: self.load(path)
            except (OSError, ValueError): pass

    # ------------------------------------------------------------------ session handling and persistence
    def start_session(self):
        self.session += 1; self.track = {}; self.unseen = {}
        for it in self.items.values(): it['seen_this_session'] = False
        return self.session

    def end_session(self):
        for it in list(self.items.values()):                                  # items made this session that never got a good enough look to be compared: last try with what they have
            if it['id'] in self.items and it.get('merge_pending') and it['colour']['n'] >= 12: self._try_merge(it, None)
        if self.path: self.save(self.path)

    def save(self, path):
        json.dump(dict(items=self.items, next_id=self.next_id, session=self.session), open(path, 'w'), ensure_ascii=False)

    def load(self, path):
        d = json.load(open(path)); self.items = {int(k): v for k, v in d['items'].items()}; self.next_id = d['next_id']; self.session = d['session']
        for it in self.items.values(): it['seen_this_session'] = False

    # ------------------------------------------------------------------ appearance from the colour image
    @staticmethod
    def pixels_of(agent, e, x, y, th, img):
        """RGB pixels of object entry `e` in the current front-camera image: its surface points projected into the image, filled, and kept where the depth agrees."""
        cam = agent.cam; Hh, Ww = cam.depth.shape; f = (Hh / 2) / math.tan(math.radians(agent.r.m.cam_fovy[cam.cam]) / 2)
        c, s = math.cos(th), math.sin(th); pw = np.asarray(e['pts'])
        Pb = np.stack([c * (pw[:, 0] - x) + s * (pw[:, 1] - y), -s * (pw[:, 0] - x) + c * (pw[:, 1] - y), pw[:, 2]], 1)
        Pc = (Pb - cam.t_bc) @ cam.R_bc; depth = -Pc[:, 2]; ok = depth > 0.1
        if ok.sum() < 6: return np.zeros((0, 3), np.uint8)
        u = Pc[ok, 0] / depth[ok] * f + Ww / 2; v = -Pc[ok, 1] / depth[ok] * f + Hh / 2; d0 = depth[ok]
        from PIL import Image, ImageDraw
        try:
            from scipy.spatial import ConvexHull
            q = np.stack([u, v], 1); hull = q[ConvexHull(q).vertices]
        except Exception: return np.zeros((0, 3), np.uint8)
        m = Image.new('1', (Ww, Hh), 0); ImageDraw.Draw(m).polygon([tuple(p) for p in hull], fill=1); mask = np.array(m, bool)
        vv, uu = np.nonzero(mask); dd = cam.depth[vv, uu]; near = float(np.median(d0))
        keep = (dd > 0.05) & (np.abs(dd - near) < 0.06) & ~cam.self_mask[vv, uu]
        return img[vv[keep], uu[keep]]

    # ------------------------------------------------------------------ observing
    def observe(self, agent, x, y, th):
        """Every front-camera frame (after the agent has matched this frame's objects): record what it sees on the floor."""
        r = agent.r; self.t = r.t; img = None
        self._active = {self.track[k] for k, e in agent.memory.items() if k in self.track and e['status'] == 'pending'}     # items a live memory entry is tracking right now
        for k, dist in list(agent.visible.items()):
            e = agent.memory.get(k)
            if e is None or e['status'] != 'pending': continue
            it = self.items.get(self.track.get(k))
            need_look = it is None or (dist < 1.0 and it['colour']['n'] < 400 and not e.get('blind'))
            col = None
            if need_look and dist < 1.1 and not e.get('blind'):
                if img is None: img = agent.cam.rgb()
                px = self.pixels_of(agent, e, x, y, th, img)
                if len(px) >= 12: col = colour_descriptor(px)
            size = [100 * v for v in e['size']]
            if it is None:
                it = self._match_or_new(e, size, col, k, r.t)
            else:
                self._update(it, e, size, col, r.t)
                if it.get('merge_pending') and it['colour']['n'] >= 40: it = self._try_merge(it, k)
            self.track[k] = it['id']; self._active.add(it['id']); it['seen_this_session'] = True; self.unseen.pop(it['id'], None)
        for k, e in agent.memory.items():                                   # the job's own bookkeeping: carried, delivered
            i = self.track.get(k)
            if i is None: continue
            it = self.items[i]
            if e['status'] == 'held' and it['status'] != 'held': it['status'] = 'held'; self._log(it, '被夹起来')
            elif e['status'] == 'delivered' and it['status'] != 'in_basket':
                it['status'] = 'in_basket'; it['pos'] = [agent.basket['x'], agent.basket['y']]; it['place'] = '收纳筐里'; it['last_seen'] = r.t; self._log(it, '放进了收纳筐')
        self._negative_evidence(agent, x, y, th)

    def _fresh(self, e, size, col, k, t):
        it = dict(id=self.next_id, name=None, kind=None, kind_conf=0.0, size_cm=[round(v, 1) for v in size], elong=float(e['elong']), colour=col or dict(hist=[0.0] * HUE_BINS, s=0.0, v=0.5, n=0),
                  first_seen=t, last_seen=t, first_session=self.session, last_session=self.session, pos=[round(e['x'], 3), round(e['y'], 3)], place='', status='on_floor', history=[(round(t, 1), round(e['x'], 3), round(e['y'], 3), self.session)],
                  seen_count=1, seen_this_session=True)
        self.next_id += 1; self.items[it['id']] = it; self._refresh_kind(it); return it

    def _refresh_kind(self, it):
        if it['colour']['n'] > 0:
            kind, conf, _ = guess_kind(it['size_cm'], it['elong'], it['colour']); it['kind'], it['kind_conf'] = kind, round(conf, 2)

    def _match_or_new(self, e, size, col, k, t):
        """A thing the agent has just found: the item that lies there (and no live memory entry is tracking), or the item of that look that is not where it was, or a new item.
        The agent drops a memory entry after two frames in which it was not seen and makes a new one when it is seen again, so an item that was seen earlier this session
        is a fair match for a new entry on the same spot; only an item another live entry holds is not."""
        near = [it for it in self.items.values() if it['status'] in ('on_floor', 'missing', 'held') and math.hypot(it['pos'][0] - e['x'], it['pos'][1] - e['y']) < 0.12 and it['id'] not in self._active]
        best = None
        for it in near:
            d = math.hypot(it['pos'][0] - e['x'], it['pos'][1] - e['y']); sc = d / 0.12 + self._look_distance(it, size, col)
            if best is None or sc < best[0]: best = (sc, it)
        if best is not None and best[0] < 1.6:
            it = best[1]; self._update(it, e, size, col, t); return it
        # not where any item was: perhaps an item that was moved. Its look has to fit, and it has to be an item that is not seen at its old place now
        if col is not None:
            it = self._moved_match(e, size, col, exclude=None)
            if it is not None: self._claim_moved(it, e, size, col, t); return it
        it = self._fresh(e, size, col, k, t)
        if col is None: it['merge_pending'] = True               # no colour yet: try again when there is one
        return it

    def _moved_match(self, e, size, col, exclude):
        """The one old item (not seen this session, on the floor or missing) that looks like this and lies somewhere else; None if none fits clearly."""
        cand = [(self._look_distance(it, size, col), it) for it in self.items.values() if it is not exclude and math.hypot(it['pos'][0] - e['x'], it['pos'][1] - e['y']) >= 0.12
                and ((it['status'] in ('on_floor', 'missing') and not it.get('seen_this_session') and it['first_session'] < self.session) or it['status'] == 'held')]      # a thing the robot carried and put down is the same thing
        cand.sort(key=lambda c: c[0])
        if not cand: return None
        best = cand[0][1]
        dc = colour_distance(best['colour'], col) if (col is not None and best['colour']['n'] > 0 and col['n'] > 0) else 0.0
        if cand[0][0] < 0.6 and dc < 0.2 and (len(cand) == 1 or cand[1][0] - cand[0][0] > 0.15): return best
        return None

    def _claim_moved(self, it, e, size, col, t):
        old = (it['pos'][0], it['pos'][1]); it['status'] = 'on_floor'; self._update(it, e, size, col, t)
        self._log(it, f'从 ({old[0]:.2f}, {old[1]:.2f}) 挪到了 ({e["x"]:.2f}, {e["y"]:.2f})', moved=True)

    def _try_merge(self, fresh, k):
        """A new item that had no colour when it was made now has one: is it really an old item that was moved? Then the old item takes over and the new one goes."""
        fresh['merge_pending'] = False
        e = dict(x=fresh['pos'][0], y=fresh['pos'][1])
        it = self._moved_match(e, fresh['size_cm'], fresh['colour'], exclude=fresh)
        if it is None or it['id'] == fresh['id']: return fresh
        for kk, ii in list(self.track.items()):
            if ii == fresh['id']: self.track[kk] = it['id']
        del self.items[fresh['id']]
        old = (it['pos'][0], it['pos'][1]); it['status'] = 'on_floor'; it['seen_this_session'] = True; it['last_seen'] = self.t; it['last_session'] = self.session; it['seen_count'] += 1
        it['pos'] = list(fresh['pos']); it['history'].append((round(self.t, 1), it['pos'][0], it['pos'][1], self.session))
        it['colour'] = fresh['colour'] if fresh['colour']['n'] > it['colour']['n'] // 2 else it['colour']
        self._log(it, f'从 ({old[0]:.2f}, {old[1]:.2f}) 挪到了 ({it["pos"][0]:.2f}, {it["pos"][1]:.2f})', moved=True); self.unseen.pop(it['id'], None)
        return it

    def _look_distance(self, it, size, col):
        """How different an observation looks from an item: size (long side, height; compared against the size of the thing but never finer than 5 cm / 3 cm, since a stone seen from
        another side is easily 1.5 cm off) and colour."""
        ls, lh = max(size[0], size[1]), size[2]; il, ih = max(it['size_cm'][0], it['size_cm'][1]), it['size_cm'][2]
        ds = abs(ls - il) / max(ls, il, 5.0) + 0.6 * abs(lh - ih) / max(lh, ih, 3.0)
        dc = colour_distance(it['colour'], col) if (col is not None and it['colour']['n'] > 0) else 0.2
        return 1.2 * ds + 1.5 * dc

    def _update(self, it, e, size, col, t):
        it['last_seen'] = t; it['last_session'] = self.session; it['seen_count'] += 1
        if it['status'] in ('missing', 'held', 'in_basket') or math.hypot(it['pos'][0] - e['x'], it['pos'][1] - e['y']) > 0.10 and it['status'] == 'on_floor':
            if it['status'] == 'missing': self._log(it, '又看到了')
            it['status'] = 'on_floor'
        if math.hypot(it['pos'][0] - e['x'], it['pos'][1] - e['y']) > 0.03:
            it['pos'] = [round(e['x'], 3), round(e['y'], 3)]
            h = it['history']
            if not h or math.hypot(h[-1][1] - e['x'], h[-1][2] - e['y']) > 0.10: h.append((round(t, 1), round(e['x'], 3), round(e['y'], 3), self.session))
        a = 0.2; it['size_cm'] = [round((1 - a) * p + a * q, 1) for p, q in zip(it['size_cm'], size)]
        if col is not None and col['n'] > 0:
            old = it['colour']; n0, n1 = old['n'], col['n']; w = n1 / (n0 + n1) if n0 else 1.0
            it['colour'] = dict(hist=[(1 - w) * p + w * q for p, q in zip(old['hist'], col['hist'])], s=(1 - w) * old['s'] + w * col['s'], v=(1 - w) * old['v'] + w * col['v'], n=min(n0 + n1, 3000))
            self._refresh_kind(it)

    def _log(self, it, text, moved=False):
        self.events.append((round(self.t, 1), it['id'], text))
        if moved: it['moved_session'] = self.session

    def _negative_evidence(self, agent, x, y, th):
        """An item that should be in plain view and is not (0.35 to 1.2 m away, 0.7 m for small flat things, well inside the picture) is marked missing, but only after
        several frames over a second or more, and after more of them when it was in view a moment ago: the detector flickers, the item is not gone."""
        cam = agent.cam; c, s = math.cos(th), math.sin(th)
        vis_pos = [(agent.memory[k]['x'], agent.memory[k]['y']) for k in agent.visible if k in agent.memory]
        for i, it in self.items.items():
            if it['status'] != 'on_floor' or (it.get('seen_this_session') and it['last_seen'] > self.t - 0.5): continue
            wx, wy = it['pos'][0] - x, it['pos'][1] - y; B = np.array([[c * wx + s * wy, -s * wx + c * wy, 0.0]]); d = math.hypot(B[0, 0], B[0, 1])
            small = max(it['size_cm'][:2]) < 5 or it['size_cm'][2] < 3
            if not (0.35 < d < (0.7 if small else 1.2)) or abs(math.atan2(B[0, 1], B[0, 0])) > math.radians(32) or not cam.floor_visible(B)[0]: continue
            if any(math.hypot(px - it['pos'][0], py - it['pos'][1]) < 0.12 for px, py in vis_pos): continue
            n, t0, tl = self.unseen.get(i, (0, self.t, self.t))
            if self.t - tl > 2.0: n, t0 = 0, self.t                  # the frames have to follow each other, not add up over minutes
            self.unseen[i] = (n + 1, t0, self.t)
            need = 10 if (it.get('seen_this_session') and it['last_seen'] > self.t - 6.0) else 5
            if n + 1 >= need and self.t - t0 >= 1.0:
                it['status'] = 'missing'; self._log(it, f'本该在 ({it["pos"][0]:.2f}, {it["pos"][1]:.2f}) 却不在了')

    # ------------------------------------------------------------------ naming and places
    def name_item(self, i, name):
        if i in self.items: self.items[i]['name'] = name; self._log(self.items[i], f'叫它「{name}」'); return True
        return False

    def label(self, it):
        if it['name']: return it['name']
        if it['kind'] and it['kind_conf'] >= 0.25: return PROTO[it['kind']]['cn']
        return f'东西#{it["id"]}'

    def set_places(self, furniture, names=None):
        """furniture: from mapping3d.analyze. Each item's place is the nearest piece of furniture within 40 cm ('next to the sofa')."""
        self.furniture = furniture
        for it in self.items.values():
            if it['status'] == 'in_basket': it['place'] = '收纳筐里'; continue
            best = None
            for f in furniture:
                c, s = math.cos(f['yaw']), math.sin(f['yaw']); dx, dy = it['pos'][0] - f['centre'][0], it['pos'][1] - f['centre'][1]
                u, v = c * dx + s * dy, -s * dx + c * dy; ex, ey = max(abs(u) - f['length'] / 2, 0), max(abs(v) - f['width'] / 2, 0); d = math.hypot(ex, ey)
                if best is None or d < best[0]: best = (d, f)
            if best and best[0] < 0.40: it['place'] = f'{(names or {}).get(best[1]["id"]) or furniture_name(best[1])}旁边'
            else: it['place'] = '地板中间'

    # ------------------------------------------------------------------ the questions
    def find(self, text):
        """Items whose name or kind name is in `text` (a person's words)."""
        out = []
        for it in self.items.values():
            names = [it['name']] if it['name'] else []
            if it['kind']: names.append(PROTO[it['kind']]['cn'])
            if any(n and n in text for n in names): out.append(it)
        return out

    def where(self, it):
        age = self.t - it['last_seen']
        st = dict(on_floor='在地上', in_basket='在收纳筐里', held='正拿在手里', missing='本来在这里，现在不见了')[it['status']]
        return f'{self.label(it)}：{it["place"] or "不知道在哪"}（{st}，{self._age(age)}前看到）'

    @staticmethod
    def _age(s):
        s = max(s, 0)
        return f'{s:.0f} 秒' if s < 90 else (f'{s / 60:.0f} 分钟' if s < 5400 else f'{s / 3600:.1f} 小时')

    def new_since(self, session): return [it for it in self.items.values() if it['first_session'] > session]
    def moved_since(self, session): return [it for it in self.items.values() if it.get('moved_session', 0) > session]
    def missing(self): return [it for it in self.items.values() if it['status'] == 'missing']
    def unverified(self):
        """On the floor in the records, but not looked at in this session: where it was last seen, not where it is."""
        return [it for it in self.items.values() if it['status'] == 'on_floor' and not it.get('seen_this_session')]
    def on_floor(self): return [it for it in self.items.values() if it['status'] == 'on_floor']

    def summary(self):
        by = {}
        for it in self.items.values(): by[it['status']] = by.get(it['status'], 0) + 1
        return dict(total=len(self.items), by_status=by, named=sum(1 for it in self.items.values() if it['name']))
