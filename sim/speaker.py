"""Who is talking, from the voice alone: a voiceprint.

What it is: a vector computed from the speech of one utterance (the sound of the voice, not the words), compared with the vectors of the people who agreed to be remembered. Text-independent
(any sentence, either language), so a person can be recognised from "put the ball in the basket" after having been enrolled with "my name is Tom".

What is here, and what is not:
  features / VoiceModel   MFCC statistics + pitch of the speech frames of an utterance, then a projection (LDA) trained on a *background population* of voices rendered through rooms and noise, so that
                          the room and the language count for little and the speaker for much. A real device would use a neural speaker embedding (ECAPA / x-vector) trained on thousands of real
                          speakers; this is the classical stand-in that runs without one.
  Voiceprints             enrolment (needs a named consenting grown-up, three utterances of 1.5 s or more of speech), identification with three answers: a name, 'nobody I know' (an anonymous voice), or
                          'not sure' (two people too close, or too little speech), forgetting one person or all. Per-person score normalisation from a cohort, so one threshold serves everyone.
  Anonymous               a voice nobody enrolled is kept apart from the others in RAM for the rest of the day only (to tell "the same stranger again" from "another one"), never written down, never named.
  What the simulator can't say: the voices here are espeak-ng speakers (a few synthesiser variants and a pitch, with a random vocal-tract warp, speed and pitch jitter per utterance). They are far easier
  to tell apart than people who sound alike (siblings, a parent and a child), and the same voice is rendered the same way every time. The numbers say that the pipeline works and where it breaks when the
  *acoustics* get bad (noise, reverberation, short speech, a television); how well it does on a real family is for real recordings to say.

Privacy: a voiceprint is biometric data. It is stored only with an adult's consent (a child's: a guardian's), as a vector (no audio), in a local file the owner can read and delete; 'forget my voice' removes it;
the robot does not guess age, sex, mood or health from a voice.
"""
import json
import math
import os

import numpy as np
import scipy.fft as sfft

from store import LOG_MAX, dump_json, load_log

FS = 16000
WIN, HOP, NFFT, NMEL, NMFCC = 400, 160, 512, 40, 30
MIN_SPEECH_S = 1.2                 # less speech than this and the robot says 'not sure' rather than guess


# ---------------------------------------------------------------------------------------------------------------- features
_FB = {}


def _mel_fb():
    if 'fb' not in _FB:
        hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0); mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
        pts = mel2hz(np.linspace(hz2mel(60.0), hz2mel(7600.0), NMEL + 2)); bins = np.floor((NFFT + 1) * pts / FS).astype(int); fb = np.zeros((NMEL, NFFT // 2 + 1))
        for m in range(1, NMEL + 1):
            a, b, c = bins[m - 1], bins[m], bins[m + 1]
            for k in range(a, b): fb[m - 1, k] = (k - a) / max(b - a, 1)
            for k in range(b, c): fb[m - 1, k] = (c - k) / max(c - b, 1)
        _FB['fb'] = fb; _FB['win'] = np.hamming(WIN)
    return _FB['fb'], _FB['win']


def _frames(x):
    n = 1 + (len(x) - WIN) // HOP
    if n < 1: return np.zeros((0, WIN))
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]; return x[idx]


def _pitch(F, voiced_floor=0.35):
    """Autocorrelation pitch of frames (40 ms) -> f0 array (nan where unvoiced)."""
    out = np.full(len(F), np.nan)
    if len(F) == 0: return out
    n = 640; x = F if F.shape[1] >= n else np.pad(F, ((0, 0), (0, n - F.shape[1])))
    x = x[:, :n] - x[:, :n].mean(1, keepdims=True); X = sfft.rfft(x * np.hanning(n), 2 * n); r = sfft.irfft(np.abs(X) ** 2)[:, :n]; r0 = r[:, 0] + 1e-9
    lo, hi = int(FS / 420), int(FS / 70); seg = r[:, lo:hi] / r0[:, None]; k = seg.argmax(1); pk = seg[np.arange(len(seg)), k]
    ok = pk > voiced_floor; out[ok] = FS / (k[ok] + lo); return out


def raw_features(x):
    """One utterance (float array, 16 kHz, speech enhanced or not) -> (raw feature vector, seconds of speech). None if there is almost no speech."""
    x = np.asarray(x, float)
    if len(x) < WIN * 3: return None, 0.0
    x = np.append(x[0], x[1:] - 0.97 * x[:-1]); F = _frames(x); w = _mel_fb()[1]
    e = 10 * np.log10(np.mean(F ** 2, 1) + 1e-14); floor = np.percentile(e, 10); sp = e > max(floor + 9.0, e.max() - 35.0)
    if sp.sum() * HOP / FS < 0.5: return None, float(sp.sum() * HOP / FS)
    S = np.abs(sfft.rfft(F[sp] * w, NFFT)) ** 2; mel = np.log(S @ _mel_fb()[0].T + 1e-12); C = sfft.dct(mel, type=2, norm='ortho', axis=1)[:, :NMFCC]
    f0all = _pitch(_frames(np.asarray(x, float))); voiced = ~np.isnan(f0all) & sp; f0v = f0all[voiced]
    if voiced.sum() >= 12:                                                    # the statistics of the voiced frames only: they carry the voice, the fricatives mostly carry the words
        S = np.abs(sfft.rfft(F[voiced] * w, NFFT)) ** 2; mel = np.log(S @ _mel_fb()[0].T + 1e-12); C = sfft.dct(mel, type=2, norm='ortho', axis=1)[:, :NMFCC]
    if len(f0v) >= 8: lf = np.log(f0v); pit = [np.median(lf), np.percentile(lf, 10), np.percentile(lf, 90), len(f0v) / max(sp.sum(), 1)]
    else: pit = [np.log(150.0), np.log(120.0), np.log(190.0), 0.0]
    v = np.concatenate([C[:, 1:].mean(0), C[:, 1:].std(0), mel.mean(0), pit]); return v, float(sp.sum() * HOP / FS)


# ---------------------------------------------------------------------------------------------------------------- the projection
class VoiceModel:
    """Standardise, then LDA on a background population (labelled by speaker): the directions in which speakers differ much more than one speaker does between utterances (rooms, noise, languages
    are inside the utterances of a speaker in the training data, so they are what the projection learns to ignore)."""
    def __init__(self, k=24): self.k = k; self.mu = self.sd = self.W = None; self.cohort = None; self.cal = None

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y); self.mu = X.mean(0); self.sd = X.std(0) + 1e-6; Z = (X - self.mu) / self.sd; classes = sorted(set(y.tolist())); d = Z.shape[1]
        gm = Z.mean(0); Sw = np.zeros((d, d)); Sb = np.zeros((d, d))
        for c in classes:
            Zc = Z[y == c]; m = Zc.mean(0); Sw += (Zc - m).T @ (Zc - m); Sb += len(Zc) * np.outer(m - gm, m - gm)
        Sw /= len(Z); Sb /= len(Z); Sw += 0.15 * np.trace(Sw) / d * np.eye(d)
        ev, V = np.linalg.eig(np.linalg.solve(Sw, Sb)); o = np.argsort(-ev.real)[:min(self.k, len(classes) - 1, d)]; self.W = V[:, o].real
        E = self.embed_many(X); self.cohort = E
        return self

    def embed(self, raw):
        z = ((np.asarray(raw, float) - self.mu) / self.sd) @ self.W; return z / (np.linalg.norm(z) + 1e-12)

    def embed_many(self, X): return np.array([self.embed(r) for r in X])

    def calibrate(self, target_z, nontarget_z):
        """From trial scores of speakers the projection was not trained on (z-normalised cosine): how the score of the right person and of anybody else are distributed. Gaussians are enough for a
        log-likelihood ratio; the thresholds of Voiceprints are put where the wrong people are accepted 1 time in 100."""
        t = np.asarray(target_z); n = np.asarray(nontarget_z)
        self.cal = np.array([t.mean(), t.std() + 1e-6, n.mean(), n.std() + 1e-6, float(np.percentile(n, 99)), float(np.percentile(n, 90))]); return self.cal

    def save(self, path): np.savez(path, mu=self.mu, sd=self.sd, W=self.W, cohort=self.cohort, cal=self.cal if self.cal is not None else np.zeros(0))

    @staticmethod
    def load(path):
        z = np.load(path); m = VoiceModel(); m.mu, m.sd, m.W, m.cohort = z['mu'], z['sd'], z['W'], z['cohort']; m.cal = z['cal'] if len(z['cal']) else None; return m


# ---------------------------------------------------------------------------------------------------------------- who the voice belongs to
class Voiceprints:
    """The enrolled voices (a vector per person, with the consent record) and the three answers. Scores are cosine similarities between embeddings, normalised per person against a cohort of other
    voices (z = (score - cohort mean) / cohort std), so one threshold fits everybody.
    status: 'known' (name given), 'unknown' (a voice nobody enrolled: anonymous id), 'unsure' (too little speech, or two people too close)."""
    MIN_UTT = 3; MERGE_COS = 0.60

    def __init__(self, model, path=None):
        self.model = model; self.path = path; self.people = {}; self.log = []; self.anon = []; self.n_anon = 0
        c = model.cal if model.cal is not None else np.array([2.0, 1.4, -0.2, 1.0, 2.2, 1.1])
        self.cal = c; self.Z_ACCEPT = float(c[4]); self.Z_MARGIN = 0.8; self.Z_UNKNOWN = float(c[5])           # accept where one wrong person in a hundred would pass
        if path and os.path.exists(path):
            try:
                with open(path) as f: d = json.load(f)
                self.people = d['people']; self.log = load_log(d)                                   # the log of the sessions before: 'forget' scrubs those names too
            except (OSError, ValueError, KeyError, TypeError): pass

    # ---- enrolment, with consent
    def enrol(self, name, role, embs, consent_by, t=0.0, langs=()):
        if not consent_by: raise PermissionError('no consent given')
        if len(embs) < self.MIN_UTT: raise ValueError('too few utterances')
        c = np.mean(embs, 0); c /= np.linalg.norm(c) + 1e-12; s = self.model.cohort @ c
        self.people[name] = dict(role=role, emb=c.tolist(), n=len(embs), consent_by=consent_by, t=round(t, 1), cohort_mu=float(s.mean()), cohort_sd=float(s.std() + 1e-6), langs=sorted(set(langs)))
        self.log.append((round(t, 1), 'enrol', name)); self.save(); return True

    def forget(self, name, t=0.0):
        ok = self.people.pop(name, None) is not None
        if ok:
            self.log = [e for e in self.log if name not in e[2:]]                                  # the name goes from the log as well as from the records
            for p in self.people.values():
                if p.get('consent_by') == name: p['consent_by'] = '(forgotten)'
            self.log.append((round(t, 1), 'forget', None)); self.save()
        return ok

    def forget_all(self, t=0.0):
        n = len(self.people); self.people = {}; self.anon = []; self.log = [e for e in self.log if not any(isinstance(v, str) for v in e[2:])]; self.log.append((round(t, 1), 'forget_all', n)); self.save(); return n

    def save(self):
        self.log = self.log[-LOG_MAX:]
        if self.path: dump_json(self.path, dict(people=self.people, log=self.log))

    def role(self, name): return self.people.get(name, {}).get('role')

    def report(self): return [dict(name=n, role=p['role'], utterances=p['n'], consent_by=p['consent_by'], stored='a vector, no audio') for n, p in self.people.items()]

    # ---- identification
    def scores(self, emb): return {n: (float(emb @ np.array(p['emb'])) - p['cohort_mu']) / p['cohort_sd'] for n, p in self.people.items()}

    def llr(self, z):
        """Log-likelihood ratio 'this is the enrolled person' against 'this is somebody else' for a score z (Gaussian models of the two, from the calibration); bounded so that no single voice settles anything."""
        mt, st, mn, sn = self.cal[:4]; sp = max(math.sqrt(0.5 * (st ** 2 + sn ** 2)), 0.8)                       # two Gaussians of one width (the pooled one, not below 0.8): a monotone ratio
        return float(np.clip((mt - mn) / sp ** 2 * (z - 0.5 * (mt + mn)), -3.0, 2.2))                   # the impostor scores have a heavier tail than a Gaussian: no single voice counts for more than 9 to 1

    def evidence(self, emb):
        """{name: log-likelihood ratio} for every enrolled person: what the voice says to a decision that has other witnesses."""
        return {n: self.llr(z) for n, z in self.scores(emb).items()}

    def identify(self, emb, speech_s=None):
        """-> dict(status, name, z, z2, anon). The anonymous id of a stranger is only kept in RAM."""
        if emb is None or (speech_s is not None and speech_s < 0.6): return dict(status='unsure', name=None, z=0.0, z2=0.0, anon=None, why='too little speech', llr={})
        if speech_s is not None and speech_s < MIN_SPEECH_S:                                         # a short sentence names nobody by itself, but it is evidence, weighed down in proportion
            f = speech_s / MIN_SPEECH_S; return dict(status='unsure', name=None, z=0.0, z2=0.0, anon=None, why='too little speech', llr={n: f * v for n, v in self.evidence(emb).items()})
        sc = sorted(((z, n) for n, z in self.scores(emb).items()), reverse=True); z1, n1 = sc[0] if sc else (-9.0, None); z2 = sc[1][0] if len(sc) > 1 else -9.0; ev = self.evidence(emb)
        if n1 is not None and z1 >= self.Z_ACCEPT and z1 - z2 >= self.Z_MARGIN: return dict(status='known', name=n1, z=z1, z2=z2, anon=None, llr=ev)
        if n1 is not None and z1 >= self.Z_ACCEPT and z1 - z2 < self.Z_MARGIN: return dict(status='unsure', name=None, z=z1, z2=z2, anon=None, why='two voices too close', llr=ev)
        if n1 is not None and z1 >= self.Z_UNKNOWN: return dict(status='unsure', name=None, z=z1, z2=z2, anon=None, why='between known and unknown', llr=ev)      # may well be a family member: not a stranger to be clustered
        return dict(status='unknown', name=None, z=z1, z2=z2, anon=self._anon(emb), llr=ev)

    def _anon(self, emb):
        """The same stranger gets the same anonymous id for the rest of the day (RAM only)."""
        best = None
        for a in self.anon:
            c = float(emb @ a['c']) / (np.linalg.norm(a['c']) + 1e-12)
            if best is None or c > best[0]: best = (c, a)
        if best is not None and best[0] >= self.MERGE_COS:
            a = best[1]; a['c'] = (a['c'] * a['n'] + emb) / (a['n'] + 1); a['n'] += 1; return a['id']
        self.n_anon += 1; a = dict(id=f'voice#{self.n_anon}', c=emb.copy(), n=1); self.anon.append(a); return a['id']

    def end_of_day(self): self.anon = []; self.n_anon = 0                                     # strangers are forgotten overnight


def utterance_embedding(model, x):
    """Audio -> (embedding or None, seconds of speech)."""
    v, s = raw_features(x)
    return (None if v is None else model.embed(v)), s
