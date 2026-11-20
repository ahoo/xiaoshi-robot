"""Finding where a voice comes from, with the microphones of the array and nothing else.

  stft / vad     frames of 32 ms every 10 ms; a frame is speech-like when its band energy (300-3500 Hz) is clearly above a slowly tracked noise floor (with a hangover)
  SRP-PHAT       for every frame the steered response power over 180 azimuths (2 degrees): the phase-transformed cross-spectrum of every microphone pair, aligned for each direction
                 and summed over the band. PHAT keeps only the phase, so a loud television or a loud talker does not drown a quiet one by level alone, and a reverberant room smears
                 the peak less than it would a plain cross-correlation.
  estimate       the weighted map over the speech frames gives the direction of the dominant voice; a histogram of every frame's two strongest peaks gives a second voice when two
                 people talk at once (speech is sparse in time and frequency, so in many frames one of them dominates)
  beamformers    delay-and-sum and MVDR (with the noise covariance from the non-speech frames) steered to a direction: the cleaner signal for the speaker-recognition and for the
                 recogniser, and a measure of how much cleaner it is.
Azimuths are in the robot frame: counter-clockwise from the robot's forward axis (x forward, y left), in radians.
"""
import math

import numpy as np
from scipy import signal as sig

from acoustics import C, FS


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


class DOA:
    def __init__(self, positions, fs=FS, nfft=512, hop=160, fmin=300.0, fmax=3500.0, n_az=180):
        self.P = np.asarray(positions, float); self.M = len(self.P); self.fs, self.nfft, self.hop = fs, nfft, hop
        self.freqs = np.fft.rfftfreq(nfft, 1.0 / fs); self.band = (self.freqs >= fmin) & (self.freqs <= fmax); self.fk = self.freqs[self.band]
        self.az = np.arange(n_az) * (2 * math.pi / n_az); self.n_az = n_az
        self.pairs = [(i, j) for i in range(self.M) for j in range(i + 1, self.M)]
        u = np.stack([np.cos(self.az), np.sin(self.az)], 1)                                         # (A, 2) directions the sound may come from
        tau = np.stack([-((self.P[i] - self.P[j]) @ u.T) / C for i, j in self.pairs])               # (Pairs, A): arrival-time difference of mic i relative to mic j
        self.E = np.exp(1j * 2 * math.pi * self.fk[None, None, :] * tau[:, :, None])                # (Pairs, A, K)
        self.E_flat = self.E.transpose(0, 2, 1).reshape(-1, n_az)                                   # (Pairs*K, A)

    # ---- front end
    def stft(self, x):
        f, t, Z = sig.stft(x, self.fs, window='hann', nperseg=self.nfft, noverlap=self.nfft - self.hop, boundary=None, padded=False, axis=-1)
        return Z                                                                                    # (M, F, T)

    def vad(self, Z, margin_db=4.0, hang=8, floor_win=100):
        """Boolean per frame and the band energy per frame. The floor is the 20th percentile of the energy over a sliding second: stationary noise (fan, television hum) sets it."""
        e = np.sum(np.abs(Z[:, self.band, :]) ** 2, axis=(0, 1)); le = 10 * np.log10(e + 1e-20); T = len(le); fl = np.empty(T)
        for t in range(T): a = max(0, t - floor_win // 2); b = min(T, t + floor_win // 2 + 1); fl[t] = np.percentile(le[a:b], 20)
        v = le > fl + margin_db
        out = v.copy()
        for t in range(T):                                                                          # hangover: a word does not end the moment its energy dips
            if v[t]: out[t:t + hang] = True
        return out, le

    # ---- direction
    def frame_maps(self, Z):
        """SRP-PHAT per frame: (T, A), about 1 for a single coherent source, near 0 for noise."""
        T = Z.shape[2]; Zb = Z[:, self.band, :]; R = np.empty((T, len(self.pairs), len(self.fk)), complex)
        for p, (i, j) in enumerate(self.pairs):
            cs = Zb[i] * np.conj(Zb[j]); R[:, p, :] = (cs / (np.abs(Zb[i]) * np.abs(Zb[j]) + 1e-12)).T
        return np.real(R.reshape(T, -1) @ self.E_flat) / R.shape[1] / R.shape[2]

    @staticmethod
    def _peaks(m, k, min_sep):
        """The k highest circular local maxima of a 1-D map, at least min_sep bins apart: [(index, value)]."""
        n = len(m); loc = np.where((m >= np.roll(m, 1)) & (m >= np.roll(m, -1)))[0]; loc = loc[np.argsort(-m[loc])]; out = []
        for i in loc:
            if all(min((i - j) % n, (j - i) % n) >= min_sep for j, _ in out): out.append((int(i), float(m[i])))
            if len(out) >= k: break
        return out

    def _refine(self, m, i):
        """Parabolic interpolation of the peak at index i on the circular map."""
        a, b, c = m[(i - 1) % len(m)], m[i], m[(i + 1) % len(m)]; d = a - 2 * b + c
        off = 0.0 if abs(d) < 1e-12 else 0.5 * (a - c) / d
        return self.az[i] + off * (self.az[1] - self.az[0])

    def estimate(self, x, max_sources=2, min_sep_deg=30.0, Z=None, active=None):
        """-> dict(sources=[dict(az, conf, frames, level_db)], active=bool mask of frames, energy_db, map). x: (M, N) signals.
        conf: how much of the speech frames agree on this direction (0..1) times how sharp the peak is."""
        Z = self.stft(x) if Z is None else Z; act, le = self.vad(Z) if active is None else (active, np.zeros(Z.shape[2])); maps = self.frame_maps(Z); sep = int(round(min_sep_deg / (360.0 / self.n_az)))
        if act.sum() < 3: return dict(sources=[], active=act, energy_db=le, map=None)
        w = np.where(act, 10 ** ((le - le.max()) / 20.0), 0.0) * np.maximum(maps.max(1), 0) ** 2; gmap = (w[:, None] * maps).sum(0) / (w.sum() + 1e-12)           # frames that agree on one sharp direction count more (the direct sound, not the tail)
        srcs = []; top = self._peaks(gmap, max_sources, sep)
        if not top: return dict(sources=[], active=act, energy_db=le, map=gmap)
        # frames voting for each global peak: their own best direction within 15 degrees of it
        best = np.argmax(maps, 1); tol = int(round(15.0 / (360.0 / self.n_az)))
        for rank, (i, v) in enumerate(top):
            if rank > 0 and v < 0.35 * top[0][1]: continue
            votes = act & (np.minimum((best - i) % self.n_az, (i - best) % self.n_az) <= tol); frac = votes.sum() / max(act.sum(), 1)
            sharp = float(np.clip((v - np.median(gmap)) / (abs(v) + 1e-9), 0, 1))
            srcs.append(dict(az=float(wrap(self._refine(gmap, i))), conf=float(np.clip(frac * 1.2, 0, 1) * (0.5 + 0.5 * sharp)), frames=int(votes.sum()), peak=float(v), level_db=float(le[act].mean())))
        # a second talker is looked for in the frames the first one does not explain, as a histogram of those frames' own peaks (speech is sparse: each talker dominates some frames)
        if max_sources > 1 and len(srcs) <= 1 and srcs:
            i0 = int(round(srcs[0]['az'] / (2 * math.pi) * self.n_az)) % self.n_az; hist = np.zeros(self.n_az)
            for t in np.nonzero(act)[0]:
                for i, v in self._peaks(maps[t], 2, sep):
                    if min((i - i0) % self.n_az, (i0 - i) % self.n_az) >= sep and v > 0.25: hist[i] += v * w[t]
            if hist.sum() > 0:
                hs = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(5) / 5, 'same')[3:-3]; j = int(np.argmax(hs))
                if hs[j] > 0.18 * (w.sum() + 1e-9) * 0.25: srcs.append(dict(az=float(wrap(self.az[j])), conf=float(np.clip(hs[j] / (w.sum() * 0.25 + 1e-9), 0, 1)) * 0.6, frames=int((hist > 0).sum()), peak=float(hs[j]), level_db=float(le[act].mean())))
        return dict(sources=srcs, active=act, energy_db=le, map=gmap)


    # ---- an event against the background before it
    def estimate_event(self, x, t0, t1, base_len=1.5, base_gap=0.15, max_sources=2, min_sep_deg=30.0, Z=None):
        """The direction of what starts at t0 (a wake word, a command) against the sound that was already there: the covariance of the frames before the event (the television, the
        fan, the wheels) is subtracted from the covariance during it, and SRP-PHAT is run on what is left, bin by bin weighted by how much of that bin is new. This is what one does
        with a wake word: the keyword spotter says when, the array says where. Falls back to the whole-segment estimate when there is no usable background.
        -> dict(sources=[dict(az, conf, snr_db)], base_frames, event_frames, map)"""
        Z = self.stft(x) if Z is None else Z; T = Z.shape[2]; tc = (np.arange(T) * self.hop + self.nfft / 2) / self.fs
        ev = (tc >= t0) & (tc <= t1); bs = (tc >= t0 - base_len) & (tc <= t0 - base_gap)
        if ev.sum() < 4: return dict(sources=[], base_frames=int(bs.sum()), event_frames=int(ev.sum()), map=None)
        Zb = Z[:, self.band, :]; Zev = Zb[:, :, ev]; Rx = np.einsum('mkt,nkt->kmn', Zev, Zev.conj()) / Zev.shape[2]
        if bs.sum() >= 12: Zbs = Zb[:, :, bs]; Rb = np.einsum('mkt,nkt->kmn', Zbs, Zbs.conj()) / Zbs.shape[2]
        else: Rb = np.zeros_like(Rx)
        Rs = Rx - Rb; tr_x = np.real(np.trace(Rx, axis1=1, axis2=2)) + 1e-20; tr_s = np.real(np.trace(Rs, axis1=1, axis2=2)); frac = np.clip(tr_s / tr_x, 0.0, 1.0)             # how much of each bin is new
        acc = np.zeros(len(self.fk) * len(self.pairs), complex)
        Phi = np.empty((len(self.pairs), len(self.fk)), complex)
        for p, (i, j) in enumerate(self.pairs): v = Rs[:, i, j]; Phi[p] = v / (np.abs(v) + 1e-20) * frac
        gmap = np.real(Phi.reshape(-1) @ self.E_flat) / Phi.shape[0] / max(frac.mean(), 1e-6) / Phi.shape[1] * 1.0
        sep = int(round(min_sep_deg / (360.0 / self.n_az))); srcs = []
        for rank, (i, v) in enumerate(self._peaks(gmap, max_sources, sep)):
            if rank > 0 and v < 0.4 * gmap.max(): continue
            sharp = float(np.clip((v - np.median(gmap)) / (abs(v) + 1e-9), 0, 1)); srcs.append(dict(az=float(wrap(self._refine(gmap, i))), conf=float(np.clip(v * 1.6, 0, 1) * (0.5 + 0.5 * sharp)), peak=float(v), snr_db=float(10 * np.log10(max(frac.mean(), 1e-3) / max(1 - frac.mean(), 1e-3)))))
        return dict(sources=srcs, base_frames=int(bs.sum()), event_frames=int(ev.sum()), map=gmap)

    # ---- beamforming
    def _steer(self, az, f):
        u = np.array([math.cos(az), math.sin(az)]); tau = -(self.P @ u) / C                          # arrival time of each microphone relative to the centre
        return np.exp(-1j * 2 * math.pi * f[None, :] * tau[:, None])                                # (M, F)

    def beamform_ds(self, x, az):
        """Delay-and-sum toward az on the whole signal: (N,)"""
        X = np.fft.rfft(x, axis=1); f = np.fft.rfftfreq(x.shape[1], 1.0 / self.fs); d = self._steer(az, f)
        return np.fft.irfft((np.conj(d) * X).mean(0), n=x.shape[1])

    def beamform_mvdr(self, x, az, noise_mask=None, load=1e-2):
        """MVDR toward az in the STFT domain; the noise covariance comes from the frames marked as noise (default: the frames the VAD calls non-speech). Falls back to delay-and-sum if
        there are too few noise frames. Returns (N,)"""
        Z = self.stft(x); F, T = Z.shape[1], Z.shape[2]
        if noise_mask is None: noise_mask = ~self.vad(Z)[0]
        if noise_mask.sum() < 8: return self.beamform_ds(x, az)
        d = self._steer(az, self.freqs); Y = np.zeros((F, T), complex)
        for k in range(F):
            Zn = Z[:, k, noise_mask]; Rn = Zn @ Zn.conj().T / Zn.shape[1]; Rn += load * np.trace(Rn).real / self.M * np.eye(self.M) + 1e-18 * np.eye(self.M)
            dk = d[:, k]; wv = np.linalg.solve(Rn, dk); wv /= (dk.conj() @ wv + 1e-18); Y[k] = wv.conj() @ Z[:, k, :]
        _, y = sig.istft(Y, self.fs, window='hann', nperseg=self.nfft, noverlap=self.nfft - self.hop)
        return np.pad(y, (0, max(0, x.shape[1] - len(y))))[:x.shape[1]]


def segments(active, hop, fs=FS, min_len=0.25, gap=0.35):
    """Speech segments (t0, t1) in seconds from a per-frame activity mask: runs shorter than min_len are dropped, runs closer than gap are joined."""
    runs = []; t = 0; T = len(active)
    while t < T:
        if active[t]:
            a = t
            while t < T and active[t]: t += 1
            runs.append([a, t])
        else: t += 1
    merged = []
    for r in runs:
        if merged and (r[0] - merged[-1][1]) * hop / fs < gap: merged[-1][1] = r[1]
        else: merged.append(r)
    return [(a * hop / fs, b * hop / fs) for a, b in merged if (b - a) * hop / fs >= min_len]
