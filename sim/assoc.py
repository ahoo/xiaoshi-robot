"""Which person said it: the direction of the voice, the voiceprint and what the camera sees, put together - and what is a television.

Three witnesses, each unreliable in its own way:
  direction   the microphone array says where the sound came from (an angle, wrong by 8 degrees typically and by more than 20 now and then; see test_sound.py ssl)
  voice       the voiceprint says who it sounds like (a name, 'a voice nobody enrolled', or 'not sure'), independent of where the person stands
  sight       the head camera says who is in view, where, and (for enrolled faces) who they are; it sees only a part of the room and nothing behind the robot
A speaker can be a visible person, a person out of view (the kitchen, behind the robot), or a device (the television, a phone). Every one of those is a hypothesis with a prior; the direction and the voice
change their odds by likelihood ratios; the most probable one wins if it is probable enough ('not sure' otherwise, and 'not sure' is treated like a stranger: the lowest permissions).

Devices are learned, not configured: speech that no visible person explains is a ray in the world (robot position + bearing); rays from several robot positions that meet in one point, at a place where no
person is ever seen, are a television. A wake word that comes from there is ignored. (A wake word spoken *by a person* while the television is on still works: the person is the better explanation.)

Safety rules that sit here, not in the dialogue:
  - the 'owner' role (enrol, forget, switch the privacy settings, see a profile) needs two witnesses: a visible, face-identified adult at the direction of the voice AND a voiceprint that does not say it is
    somebody else (a sentence too short to give any voice evidence is not enough), and no known device (television, phone) in that direction.
    A voice alone - however well it matches - buys no permission above a stranger's (the lowest): it names the speaker (for the language and the profile), nothing more. A recording played by a television
    or a phone must not be able to make the robot forget the family, and a stranger standing where the owner is must not inherit the owner's rights.
  - an anonymous voice (nobody enrolled) is never given more than the child level, and its id lives in RAM only.
  - from a device, only the always-on stop word counts.
"""
import math

import numpy as np


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


CFG = dict(sigma_deg=14.0, outlier=0.12, outlier_tv=0.30, fov_deg=100.0, range_m=4.5, p_unseen=0.18, p_unseen_alone=0.5, p_device=0.50, lr_hit=10.0, lr_miss=0.1, lr_device_named=0.04, lr_unknown_vs_voiced=0.15,
           accept=0.6, near_scale=0.35, dev_min_rays=4, dev_min_baseline_m=0.4, dev_resid_deg=25.0, dev_rms_deg=12.0, dev_near_m=0.9, permit_posterior=0.85)


class Associator:
    """resolve() for each speech event; observe_speech() for speech that nobody is asked about (to learn devices)."""
    def __init__(self, cfg=None, has_voice=None, role_of=None, bounds=None):
        """bounds: (x0, x1, y0, y1) of the room if known (the 3D map has the walls); a device outside it is not believed."""
        self.bounds = bounds; self.c = dict(CFG, **(cfg or {})); self.rays = []; self.devices = []; self.has_voice = has_voice or (lambda name: False); self.role_of = role_of or (lambda name: None); self.log = []

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def bearing(pose, x, y): return wrap(math.atan2(y - pose[1], x - pose[0]) - pose[2])

    def _visible(self, pose, x, y):
        d = math.hypot(x - pose[0], y - pose[1]); return d <= self.c['range_m'] and abs(self.bearing(pose, x, y)) <= math.radians(self.c['fov_deg']) / 2

    def _loglik_doa(self, delta, sigma):
        o = self.c['outlier_tv'] if self.devices else self.c['outlier']                              # with a television on, the direction goes wrong more often (see test_sound.py ssl)
        p = (1 - o) * math.exp(-0.5 * (delta / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)) + o / (2 * math.pi); return math.log(p)

    # ------------------------------------------------------------------ the decision
    def resolve(self, ev, pose, tracks, voice=None, wake=True):
        """ev: dict(az (rad, robot frame), snr_db (optional)). pose (x, y, yaw). tracks: visible people (x, y, who, cls, dist). voice: dict from speaker.Voiceprints.identify or None.
        -> dict(source 'person'|'unseen'|'device'|'ambiguous', track, name, anon, loc (x, y) or None, posterior, role, why, hypotheses)"""
        c = self.c; az = ev['az']; sigma = math.radians(c['sigma_deg']) * (1.0 + 0.6 * max(0.0, (6.0 - (ev.get('snr_db') or 10.0))) / 6.0)
        vname = voice['name'] if (voice and voice['status'] == 'known') else None; vstat = voice['status'] if voice else None
        llr = (voice or {}).get('llr') or {}                                                           # {enrolled name: log-likelihood ratio from the voiceprint}; empty without a usable voice
        def voice_term(person):
            """log odds from the voice for 'the speaker is `person`' (None: a face nobody enrolled, so anybody who is not named in view or a guest)."""
            if not llr: return 0.0
            if person is not None: return llr.get(person, 0.0)
            vis = {k.who for k in tracks if k.who}; cand = [math.exp(v) for n, v in llr.items() if n not in vis] + [1.0]; return math.log(sum(cand) / len(cand))
        H = []                                                                                       # (log-posterior, kind, payload); every term is a log odds against "the direction is arbitrary"
        visible_names = {k.who for k in tracks if k.who}; U = -math.log(2 * math.pi); n = max(len(tracks), 1); f = c['fov_deg'] / 360.0
        for k in tracks:
            b = self.bearing(pose, k.x, k.y); ll = self._loglik_doa(wrap(az - b), sigma) - U - c['near_scale'] * k.dist - math.log(n)
            ll += voice_term(k.who) if llr else (math.log(c['lr_hit']) if (vname is not None and k.who == vname) else (math.log(c['lr_miss']) if (vname is not None and k.who and k.who != vname) else 0.0))
            H.append((ll, 'track', k))
        # a person out of view: more likely in a direction the camera does not cover
        inside = abs(az) <= math.radians(c['fov_deg']) / 2
        pu = c['p_unseen_alone'] if not tracks else c['p_unseen']
        ll = math.log(pu) + math.log(0.35 if inside else 1.0 / (1.0 - f)); who_unseen = None; share = 0.0
        if llr:
            vis = {k.who for k in tracks if k.who}; away = {n: v for n, v in llr.items() if n not in vis}                 # whoever is in view by name cannot also be out of it
            cand = [math.exp(v) for v in away.values()] + [1.0]; ll += math.log(sum(cand) / len(cand))
            if away:
                n1 = max(away, key=away.get); share = math.exp(away[n1]) / sum(cand)                                      # how much of the voice's evidence is for this one person
                if share >= 0.75: who_unseen = n1                                                                       # naming somebody out of view needs the voice to be fairly sure
        elif vname is not None:
            if vname in visible_names: ll += math.log(0.03)
            else: ll += math.log(c['lr_hit']); who_unseen = vname; share = 0.95
        H.append((ll, 'unseen', (who_unseen, share)))
        for d in self.devices:
            b = self.bearing(pose, d['xy'][0], d['xy'][1]); person_there = any(math.hypot(k.x - d['xy'][0], k.y - d['xy'][1]) < c['dev_near_m'] for k in tracks)
            named = (max(llr.values()) > 1.5) if llr else (vname is not None)                           # a television that sounds like an enrolled person is a rare thing (a recording of them)
            ll = self._loglik_doa(wrap(az - b), sigma) - U + math.log(c['p_device']) + (math.log(c['lr_device_named']) if named else 0.0) + (math.log(0.15) if person_there else 0.0)
            H.append((ll, 'device', d))
        lp = np.array([h[0] for h in H]); p = np.exp(lp - lp.max()); p /= p.sum(); i = int(p.argmax()); kind, pay = H[i][1], H[i][2]; post = float(p[i])
        out = dict(source='ambiguous', track=None, name=None, anon=None, loc=None, posterior=round(post, 3), why=[], hypotheses=[(h[1], round(float(q), 3), (getattr(h[2], 'who', None) or getattr(h[2], 'cls', None)) if h[1] == 'track' else None) for h, q in zip(H, p)])
        if post < c['accept']: out['why'].append('no hypothesis probable enough'); out['role'] = 'unknown'; self._note(ev, pose, out); return out
        if kind == 'device': out.update(source='device', loc=tuple(pay['xy']), why=['direction of a learned device']); out['role'] = 'device'; self._note(ev, pose, out); return out
        if kind == 'track':
            nm = pay.who
            if nm is None and llr:                                                                       # a face nobody named: the voice may name it
                vis = {k.who for k in tracks if k.who}; away = {n: v for n, v in llr.items() if n not in vis}
                if away and vname in away and vstat == 'known': nm = vname
            elif nm is None and vname and vstat == 'known' and not any(k.who == vname for k in tracks): nm = vname
            out.update(source='person', track=pay, loc=(pay.x, pay.y), name=nm)
            if out['name'] is None and voice and voice.get('anon'): out['anon'] = voice['anon']
            out['why'].append('visible person in the direction of the voice')
        else:
            out.update(source='unseen', name=pay[0], name_conf=round(pay[1], 3), anon=(voice or {}).get('anon'), loc=None, why=['voice of a person out of view'])
        out['role'] = self._role(out, voice, tracks, pose, az, sigma)
        if out['role'] == 'owner' and any(abs(wrap(az - self.bearing(pose, d['xy'][0], d['xy'][1]))) < 2 * sigma for d in self.devices):
            out['role'] = 'adult'; out['why'].append('a known device is in the same direction: not trusted with anything privileged')         # a recording from the television could look exactly like the person standing in front of it
        self._note(ev, pose, out); return out

    def _role(self, out, voice, tracks, pose, az, sigma):
        """The permission level of whoever it was. 'owner' needs the voice (or the face) AND a face-identified person standing where the voice came from."""
        name = out.get('name'); r = self.role_of(name) if name else None; trk = out['track']; llr = (voice or {}).get('llr') or {}
        sure = out['posterior'] >= self.c['permit_posterior']                                          # to act on somebody's authority the robot has to be surer which person spoke than it needs to be to answer
        if out['source'] == 'person' and trk is not None:
            face = bool(trk.who)
            if face and r == 'adult':                                                                 # a face the robot knows, in the direction of the voice ...
                aligned = abs(wrap(az - self.bearing(pose, trk.x, trk.y))) < 2 * sigma                  # ... standing where the sound came from, to the accuracy of the direction finder (a recording from a television somewhere else, whose
                                                                                                         # voice matches a person who happens to be in view, is explained by that person only if the direction is allowed to be far off)
                if sure and aligned and trk.who in llr and llr[trk.who] >= 0.0: return 'owner'        # ... and a voice that does not say it is somebody else (no voice evidence at all - too short a sentence - is not enough)
                return 'adult' if sure else 'unknown'
            if face and r == 'child': return 'child'
            if trk.cls == 'adult': return 'adult' if sure else 'unknown'                              # a grown-up in view that the robot cannot name
            return 'child'
        return 'child' if (name and r == 'child') else 'unknown'                                      # out of view: a voice, however well it matches, buys no permission above a stranger's; it only gives a name

    # ------------------------------------------------------------------ learning where the devices are
    def _note(self, ev, pose, out):
        self.log.append(dict(t=ev.get('t'), source=out['source'], posterior=out['posterior'], name=out.get('name')))

    def observe_speech(self, ev, pose, tracks, t=0.0):
        """Speech heard that no visible person explains (the television, someone in the next room): a ray from the robot. Rays that agree with a known device refine it; several rays from different places
        that meet where no one is seen make a new one."""
        az = ev['az']; sigma = math.radians(self.c['sigma_deg'])
        if any(abs(wrap(az - self.bearing(pose, k.x, k.y))) < 2 * sigma for k in tracks): return None                      # someone in view in that direction explains it
        self.rays.append((pose[0], pose[1], wrap(az + pose[2]), t)); self.rays = self.rays[-60:]; return self._try_devices(tracks)

    def _agrees(self, r, pt):
        """The ray points at the point, within the angle the direction finder can be trusted to."""
        return abs(wrap(math.atan2(pt[1] - r[1], pt[0] - r[0]) - r[2])) < math.radians(self.c['dev_resid_deg'])

    def _inside(self, pt):
        b = self.bounds
        return b is None or (b[0] - 0.3 <= pt[0] <= b[1] + 0.3 and b[2] - 0.3 <= pt[1] <= b[3] + 0.3)

    def _clip(self, pt):
        b = self.bounds
        return np.asarray(pt, float) if b is None else np.array([min(max(pt[0], b[0] + 0.05), b[1] - 0.05), min(max(pt[1], b[2] + 0.05), b[3] - 0.05)])

    def _try_devices(self, tracks):
        c = self.c; R = self.rays; new = None
        for d in self.devices:                                                                          # more rays for a device we know: a better fix
            near = [r for r in R if self._agrees(r, d['xy'])]
            if len(near) >= c['dev_min_rays']:
                pt = self._clip(self._refine(near, np.array(d['xy'])))
                if self._inside(pt): d['xy'] = (float(pt[0]), float(pt[1])); d['n'] = len(near)
        if len(R) < c['dev_min_rays']: return None
        for i in range(len(R)):                                                                         # seed a candidate from one ray pair, then count the rays that agree
            for j in range(i + 1, len(R)):
                a, b = R[i], R[j]
                if math.hypot(a[0] - b[0], a[1] - b[1]) < c['dev_min_baseline_m']: continue
                A = np.array([[math.cos(a[2]), -math.cos(b[2])], [math.sin(a[2]), -math.sin(b[2])]])
                if abs(np.linalg.det(A)) < 0.25: continue                                                    # nearly parallel: no fix
                t = np.linalg.solve(A, np.array([b[0] - a[0], b[1] - a[1]]))
                if t[0] < 0.3 or t[1] < 0.3 or t[0] > 8 or t[1] > 8: continue
                pt = np.array([a[0] + t[0] * math.cos(a[2]), a[1] + t[0] * math.sin(a[2])])
                for _ in range(3):                                                                           # collect the rays that agree, refit to all of them, again
                    near = [r for r in R if self._agrees(r, pt)]
                    if len(near) < c['dev_min_rays']: break
                    pt = self._refine(near, pt)
                near = [r for r in R if self._agrees(r, pt)]; poses = {(round(r[0] / 0.4), round(r[1] / 0.4)) for r in near}
                rms = math.degrees(math.sqrt(np.mean([wrap(math.atan2(pt[1] - r[1], pt[0] - r[0]) - r[2]) ** 2 for r in near]))) if near else 99.0              # how well the rays agree with the fix
                if len(near) >= c['dev_min_rays'] and len(poses) >= 2 and rms <= c['dev_rms_deg'] and self._inside(pt) and not any(math.hypot(k.x - pt[0], k.y - pt[1]) < c['dev_near_m'] for k in tracks):
                    if not any(math.hypot(d['xy'][0] - pt[0], d['xy'][1] - pt[1]) < 0.8 for d in self.devices):
                        pt = self._clip(pt); self.devices.append(dict(xy=(float(pt[0]), float(pt[1])), n=len(near))); new = self.devices[-1]
        return new

    @staticmethod
    def _refine(rays, pt):
        """Least-squares point nearest to all the rays (starting from pt)."""
        A = np.zeros((2, 2)); b = np.zeros(2)
        for r in rays:
            n = np.array([-math.sin(r[2]), math.cos(r[2])]); A += np.outer(n, n); b += np.outer(n, n) @ np.array([r[0], r[1]])
        try: return np.linalg.solve(A, b)
        except np.linalg.LinAlgError: return np.asarray(pt)
