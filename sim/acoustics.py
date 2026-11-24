"""Sound in the simulated room: what a microphone array on the robot would hear.

MuJoCo has no sound, so this is a small acoustic simulator of our own:
  Room        a shoebox (the room is 4 x 3 x 2.5 m) with one wall absorption chosen from the reverberation time (Sabine). The room impulse response from every source to every
              microphone is the early part from the image-source method (reflections up to order 3, each a fractional delay by a windowed sinc) plus a diffuse tail of decaying
              noise whose level follows the critical distance. A talker's directivity is a simple cardioid-like gain on the direct path (somebody facing away is quieter).
  Sources     talkers (speech synthesised by espeak-ng in Mandarin and English with different voices, or a speech-like generator if espeak-ng is not installed), a television (speech or
              music, at a fixed place), the robot's own noise (a fan, servo whine and wheel noise from the three wheels, louder when it drives or turns) and a diffuse background.
  MicArray    2, 4 (ReSpeaker-like square, 32 mm radius) or 6 (circle, 46 mm radius) microphones in the robot frame, on the mast; gain and position errors can be added.
What it cannot say: real rooms have furniture that blocks and scatters sound, real mouths radiate differently at every frequency, real speech has more variety than a formant
synthesiser's. The levels (speech 62 dB SPL at 1 m, television 55-70 dB, robot idle 38 dB and moving 50-55 dB at 0.3 m, background 35 dB, microphone self-noise 25 dB) are typical
values, not measurements; they are the knobs to change when the real array is measured (validation table).
"""
import hashlib
import math
import os
import subprocess

import numpy as np
from scipy import signal as sig

FS = 16000
C = 343.0
CACHE = os.environ.get('XIAOSHI_AUDIO_CACHE', '/tmp/xiaoshi_audio_cache')
P_REF = 20e-6                                  # Pa, 0 dB SPL


def spl_to_pa(db): return P_REF * 10 ** (db / 20.0)


# ---------------------------------------------------------------------------------------------------------------- microphone arrays
def array_positions(kind):
    """Microphone positions (M, 2) in metres, robot frame (x forward, y left), relative to the array centre."""
    if kind == '2mic': return np.array([[0.0, 0.04], [0.0, -0.04]])
    if kind == '4mic': a = np.radians([45, 135, 225, 315]); return 0.0323 * np.stack([np.cos(a), np.sin(a)], 1)
    if kind == '6mic': a = np.radians([0, 60, 120, 180, 240, 300]); return 0.0463 * np.stack([np.cos(a), np.sin(a)], 1)
    raise ValueError(kind)


class MicArray:
    def __init__(self, kind='4mic', height=0.50, offset=(-0.08, 0.0), pos_err_mm=0.0, gain_err_db=0.0, seed=0):
        self.kind = kind; self.nominal = array_positions(kind); self.height = height; self.offset = np.array(offset)       # array centre in the robot frame (on the mast)
        rng = np.random.default_rng(seed)
        self.true = self.nominal + rng.normal(0, pos_err_mm / 1000.0, self.nominal.shape)          # where the capsules really are; the algorithm believes `nominal`
        self.gain = 10 ** (rng.normal(0, gain_err_db, len(self.nominal)) / 20.0)
        self.M = len(self.nominal)

    def world_positions(self, robot_xy, yaw, true=True):
        """(M, 3) world positions of the capsules for a robot at robot_xy facing yaw."""
        p = (self.true if true else self.nominal) + self.offset; c, s = math.cos(yaw), math.sin(yaw)
        wx = robot_xy[0] + c * p[:, 0] - s * p[:, 1]; wy = robot_xy[1] + s * p[:, 0] + c * p[:, 1]
        return np.stack([wx, wy, np.full(len(p), self.height)], 1)


# ---------------------------------------------------------------------------------------------------------------- speech material
def have_espeak(): return subprocess.run(['which', 'espeak-ng'], capture_output=True).returncode == 0


def _resample(x, sr):
    if sr == FS: return x.astype(np.float64)
    g = math.gcd(FS, sr); return sig.resample_poly(x.astype(np.float64), FS // g, sr // g)


def _speechlike(dur, f0, formant_scale, seed):
    """Fallback when espeak-ng is missing: a harmonic source with formant-like resonances and a syllable-rate envelope. Enough for direction finding, not for anything about language."""
    rng = np.random.default_rng(seed); n = int(dur * FS); t = np.arange(n) / FS
    f = f0 * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)) + 0.04 * np.sin(2 * np.pi * 3.1 * t))
    ph = 2 * np.pi * np.cumsum(f) / FS; src = sum(np.sin(k * ph) / k ** 0.8 for k in range(1, 30)) + 0.05 * rng.normal(size=n)
    env = np.zeros(n); pos = 0
    while pos < n:
        L = int(rng.uniform(0.12, 0.28) * FS); env[pos:pos + L] = np.hanning(min(L, n - pos)) ** 0.7 * rng.uniform(0.5, 1.0); pos += L + int(rng.uniform(0.0, 0.06) * FS)
    out = np.zeros(n); pos = 0
    while pos < n:
        L = int(0.2 * FS); F = np.array([rng.uniform(300, 800), rng.uniform(900, 2300), rng.uniform(2400, 3300)]) * formant_scale; seg = src[pos:pos + L]; y = np.zeros(len(seg))
        for fc, bw in zip(F, (80, 120, 200)):
            b, a = sig.iirpeak(min(fc, 7500) / (FS / 2), min(fc, 7500) / bw); y += sig.lfilter(b, a, seg)
        out[pos:pos + len(seg)] = y; pos += L
    return out * env


def synth(text, lang='en', voice='m3', pitch=50, speed=170, seed=0):
    """Speech for `text`, mono at 16 kHz, unit RMS over the active samples. lang 'en' or 'zh'; voice: an espeak-ng variant (m1-m7 men, f1-f5 women), pitch 0-99, speed in words per minute.
    Cached on disk. Without espeak-ng: a speech-like signal of about the same length."""
    key = hashlib.md5(f'{text}|{lang}|{voice}|{pitch}|{speed}'.encode()).hexdigest(); os.makedirs(CACHE, exist_ok=True); fn = os.path.join(CACHE, key + '.npy')
    if os.path.exists(fn): return np.load(fn)
    if have_espeak():
        import scipy.io.wavfile as wf
        wav = os.path.join(CACHE, key + '.wav'); v = ('cmn' if lang == 'zh' else 'en-us') + '+' + voice
        subprocess.run(['espeak-ng', '-v', v, '-p', str(int(pitch)), '-s', str(int(speed)), '-w', wav, text], check=True, capture_output=True)
        sr, x = wf.read(wav); x = _resample(x / 32768.0, sr); os.remove(wav)
    else:
        words = len(text.split()) if lang == 'en' else len(text) / 2.2
        x = _speechlike(max(0.8, words * 0.35), 120 + pitch * 2.0, 1.0 + (0.2 if voice.startswith('f') else 0.0), seed)
    act = np.abs(x) > 0.1 * np.abs(x).max(); x = x / (np.sqrt(np.mean(x[act] ** 2)) + 1e-9)
    np.save(fn, x); return x


def music_like(dur, seed=0):
    """A noise source that is not speech: a few harmonic notes over a rhythm (for the television playing music)."""
    rng = np.random.default_rng(seed); n = int(dur * FS); t = np.arange(n) / FS; y = np.zeros(n)
    for k in range(int(dur / 0.4) + 1):
        f0 = 220 * 2 ** (rng.integers(0, 12) / 12.0); a = int(k * 0.4 * FS); b = min(n, a + int(0.38 * FS))
        if a >= n: break
        tt = t[a:b] - t[a]; y[a:b] += sum(np.sin(2 * np.pi * h * f0 * tt) / h for h in range(1, 7)) * np.exp(-3 * tt)
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-9)


# ---------------------------------------------------------------------------------------------------------------- room
class Room:
    """A shoebox room. rt60: reverberation time in seconds (a furnished living room: 0.3-0.6)."""
    def __init__(self, size=(4.0, 3.0, 2.5), rt60=0.45, order=3):
        self.L = np.array(size, float); self.rt60 = rt60; self.order = order
        V = float(np.prod(self.L)); S = 2 * (self.L[0] * self.L[1] + self.L[0] * self.L[2] + self.L[1] * self.L[2])
        self.alpha = float(np.clip(0.161 * V / (S * rt60), 0.02, 0.9)); self.beta = math.sqrt(1 - self.alpha)
        self.d_crit = 0.057 * math.sqrt(V / rt60)

    def _images(self, s):
        """Image sources of the point s: (positions (N,3), number of reflections (N,)) up to self.order."""
        axes = []
        for d in range(3):
            lst = []
            for n in range(-2, 3):
                lst.append((2 * n * self.L[d] + s[d], abs(2 * n))); lst.append((2 * n * self.L[d] - s[d], abs(2 * n - 1)))
            axes.append([(p, r) for p, r in lst if r <= self.order])
        pos, nref = [], []
        for px, rx in axes[0]:
            for py, ry in axes[1]:
                for pz, rz in axes[2]:
                    if rx + ry + rz <= self.order: pos.append((px, py, pz)); nref.append(rx + ry + rz)
        return np.array(pos), np.array(nref)

    def rir(self, src, mics, facing=None, directivity=0.6, length=None, rng=None, taps=41):
        """Impulse responses (M, n) from a point source to the microphones. facing: the talker's horizontal facing direction (radians) or None for an omnidirectional source."""
        rng = rng or np.random.default_rng(0); length = length or min(1.0, 1.2 * self.rt60); n = int(length * FS)
        img, nref = self._images(np.asarray(src, float)); M = len(mics); h = np.zeros((M, n)); half = taps // 2; win = np.hanning(taps)
        for m in range(M):
            d = np.linalg.norm(img - mics[m], axis=1); t = d / C * FS; amp = (self.beta ** nref) / np.maximum(d, 0.1)
            if facing is not None:                                   # the direct path (nref 0) follows the mouth's directivity, the reflections its average
                v = mics[m][:2] - np.asarray(src[:2]); ang = math.atan2(v[1], v[0]) - facing; g = (1 - directivity) + directivity * (1 + math.cos(ang)) / 2
                amp = np.where(nref == 0, amp * g, amp * (1 - 0.4 * directivity))
            i0 = np.floor(t).astype(int); frac = t - i0
            for k in range(len(t)):
                a0 = i0[k] - half
                if a0 < 0 or a0 + taps > n: continue
                tt = (np.arange(taps) - half) - frac[k]; h[m, a0:a0 + taps] += amp[k] * np.sinc(tt) * win
        # the diffuse tail: noise decaying 60 dB over the reverberation time, starting where the image-source part stops mattering
        d0 = float(np.linalg.norm(np.asarray(src) - mics.mean(0))); e_direct = 1.0 / max(d0, 0.1) ** 2; e_target = e_direct * (max(d0, 0.1) / self.d_crit) ** 2
        e_ism = float(np.sum(h[:, :int(0.05 * FS)] ** 2, axis=1).mean() - e_direct); t_axis = np.arange(n) / FS; env = np.exp(-6.91 * t_axis / self.rt60)
        e_tail = max(0.3 * e_target, e_target - max(e_ism, 0.0)); tail_gain = math.sqrt(e_tail / (np.sum(env ** 2) + 1e-9))
        start = int((d0 / C + 0.02) * FS)
        for m in range(M):
            tail = rng.normal(size=n) * env * tail_gain; tail[:start] = 0.0; h[m] += tail
        return h


# ---------------------------------------------------------------------------------------------------------------- the scene
class Source:
    def __init__(self, signal, pos, level_db=62.0, facing=None, kind='talker', label='', start=0.0):
        self.x = np.asarray(signal, float); self.pos = np.asarray(pos, float); self.level = level_db; self.facing = facing; self.kind = kind; self.label = label; self.start = start


def ego_noise(dur, speed=0.0, turn=0.0, seed=0):
    """The robot's own sound (one signal per wheel and one for the fan and the arm servos): fan 38 dB at 0.3 m; wheel noise and servo whine grow with the speed (translation m/s, turning rad/s)."""
    rng = np.random.default_rng(seed); n = int(dur * FS); t = np.arange(n) / FS
    fan = sig.lfilter(*sig.butter(2, [150 / (FS / 2), 3500 / (FS / 2)], 'band'), rng.normal(size=n)); fan /= fan.std() + 1e-9
    act = min(1.0, abs(speed) / 0.15 + abs(turn) / 1.0)
    wheels = []
    for w in range(3):
        f1 = 380 + 40 * w + 900 * act; tone = sum(np.sin(2 * np.pi * k * f1 * t + rng.uniform(0, 6)) / k for k in (1, 2, 3)); br = sig.lfilter(*sig.butter(2, [300 / (FS / 2), 4000 / (FS / 2)], 'band'), rng.normal(size=n))
        wheels.append((tone / (tone.std() + 1e-9) * 0.6 + br / (br.std() + 1e-9) * 0.8) * act)
    return fan, wheels, act


class Scene:
    """A room, a microphone array on a robot (pose given at mix time) and sources. mix() returns what the array hears."""
    def __init__(self, room=None, array=None, bg_db=35.0, self_noise_db=25.0, seed=0):
        self.room = room or Room(); self.array = array or MicArray('4mic'); self.bg_db = bg_db; self.self_noise_db = self_noise_db; self.rng = np.random.default_rng(seed)

    def mix(self, sources, robot_xy, yaw, dur=None, speed=0.0, turn=0.0, ego=True):
        """-> (M, N) signals in Pa and a dict of the pieces (per source clean signal at the array, for the benches' ground truth). Sources start at their .start (s)."""
        mics = self.array.world_positions(robot_xy, yaw, true=True); M = len(mics)
        N = int((dur or max(s.start + len(s.x) / FS for s in sources) + 0.6) * FS); out = np.zeros((M, N)); parts = {}
        def add(x, pos, level, facing, start, tag):
            x = np.asarray(x)
            if not len(x) or np.abs(x).max() < 1e-12: return                                  # an idle wheel makes no sound
            rms = np.sqrt(np.mean(x[np.abs(x) > 0.1 * np.abs(x).max()] ** 2)) + 1e-9
            sc = spl_to_pa(level) / rms; h = self.room.rir(pos, mics, facing, rng=self.rng); y = np.stack([sig.fftconvolve(x * sc, h[m])[:N - int(start * FS)] for m in range(M)])
            a = int(start * FS); seg = np.zeros((M, N)); seg[:, a:a + y.shape[1]] = y[:, :N - a]; out[:] += seg; parts[tag] = seg
        for s in sources: add(s.x, s.pos, s.level, s.facing, s.start, s.label or s.kind)
        if ego:
            rp = np.array([robot_xy[0], robot_xy[1]]); fan, wheels, act = ego_noise(N / FS, speed, turn, seed=int(self.rng.integers(1 << 30)))
            c, sn = math.cos(yaw), math.sin(yaw); ref = 20 * math.log10(0.3); level_fan = 38.0 + ref; level_w = 38.0 + 14.0 * act + ref            # levels are quoted at 0.3 m; Source levels are at 1 m
            for k, (sg, ang) in enumerate(zip(wheels, (math.radians(90), math.radians(210), math.radians(330)))):
                px = rp[0] + 0.12 * (c * math.cos(ang) - sn * math.sin(ang)); py = rp[1] + 0.12 * (sn * math.cos(ang) + c * math.sin(ang)); add(sg, (px, py, 0.06), level_w - 5.0, None, 0.0, f'ego_wheel{k}')
            add(fan, (rp[0] - 0.02 + 0.0, rp[1], 0.25), level_fan, None, 0.0, 'ego_fan')
        bg = sig.lfilter(*sig.butter(1, 2000 / (FS / 2)), self.rng.normal(size=(M, N)), axis=1)                           # diffuse background (ventilation, street): low-passed, independent per capsule
        out += bg / (bg.std() + 1e-9) * spl_to_pa(self.bg_db); out += self.rng.normal(size=(M, N)) * spl_to_pa(self.self_noise_db)
        out *= self.array.gain[:, None]
        return out, parts
