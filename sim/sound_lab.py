"""Shared pieces of the sound benches: synthetic voices, and a talker in a room heard through the whole chain (microphone array -> direction of the keyword -> beamformer -> audio of that talker only).

A voice here is an espeak-ng synthesiser variant (m1-m7, f1-f5), a pitch, and a vocal-tract warp (a resampling that moves pitch and formants together, children's voices have a larger one); every utterance adds
a random jitter of pitch, speed and warp, so that one speaker never says anything the same way twice. Sentences are the commands of voice_data.py / voice_data_en.py, so both languages are spoken.
"""
import math

import numpy as np
import scipy.signal as sig

import acoustics as A
import doa as D
import voice_data as VD
import voice_data_en as VE

FS = A.FS
ROOM = (4.0, 3.0, 2.5)
LEAD = 2.0
VARIANTS = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'f1', 'f2', 'f3', 'f4', 'f5']
TEXT_EN = [t for t, _, _ in VE.COMMANDS_EN_DEV + VE.COMMANDS_EN_HOLDOUT + VE.COMMANDS_EN_TEST]
TEXT_ZH = [t for t, _, _ in VD.COMMANDS]


def voice_spec(rng, kind='adult'):
    if kind == 'child': return dict(v=str(rng.choice(['f1', 'f2', 'f3', 'f4', 'f5', 'm7'])), p=float(rng.uniform(78, 99)), warp=float(rng.uniform(0.07, 0.14)), kind='child')
    return dict(v=str(rng.choice(VARIANTS)), p=float(rng.uniform(8, 72)), warp=float(np.clip(rng.normal(0.0, 0.04), -0.09, 0.09)), kind='adult')


def render(spec, rng, lang=None, text=None):
    """One utterance of a speaker: (audio at 16 kHz, language, text)."""
    lang = lang or ('zh' if rng.random() < 0.5 else 'en'); text = text or str(rng.choice(TEXT_ZH if lang == 'zh' else TEXT_EN))
    pitch = int(np.clip(spec['p'] + rng.normal(0, 4), 0, 99)); speed = int(rng.uniform(150, 195)) + (15 if spec['kind'] == 'child' else 0)
    x = A.synth(text, lang, spec['v'], pitch, speed); warp = spec['warp'] + rng.normal(0, 0.012)
    if abs(warp) > 1e-3 and len(x) > 100: x = sig.resample(x, int(len(x) / (1.0 + warp)))
    return x, lang, text


def place(rng, robot, dmin=0.8, dmax=3.5):
    for _ in range(500):
        p = (rng.uniform(0.3, ROOM[0] - 0.3), rng.uniform(0.3, ROOM[1] - 0.3)); d = math.hypot(p[0] - robot[0], p[1] - robot[1])
        if dmin <= d <= dmax: return p, d
    raise RuntimeError('no spot')


def chain(spec, rng, cond=None, lang=None, text=None, robot=None, yaw=None, pos=None, bf='mvdr', tv_text=None):
    """A talker says something; the robot hears it with its array. -> dict(audio=the beamformed utterance (16 kHz), az_est, az_true (robot frame), dist, speech_s, lang, text, found) or found=False.
    cond: rt60, array, level (dB SPL at 1 m), speed/turn (robot moving), tv (television level at 1 m), dmin/dmax, pos_err_mm, gain_err_db."""
    cond = cond or {}; seed = int(rng.integers(1 << 30))
    room = A.Room(ROOM, rt60=cond.get('rt60', 0.5)); arr = A.MicArray(cond.get('array', '4mic'), height=cond.get('mic_height', 0.50), pos_err_mm=cond.get('pos_err_mm', 0.0), gain_err_db=cond.get('gain_err_db', 0.0), seed=seed); sc = A.Scene(room, arr, seed=seed + 1); est = D.DOA(arr.nominal)
    yaw = rng.uniform(-math.pi, math.pi) if yaw is None else yaw; fixed = robot is not None
    for _ in range(60):                                                                       # a far talker (3-4 m) needs the robot near a wall: try other places for it
        rb = robot or (rng.uniform(0.7, 3.3), rng.uniform(0.7, 2.3))
        try:
            if pos is None: pos_, dist = place(rng, rb, cond.get('dmin', 0.8), cond.get('dmax', 3.5))
            else: pos_ = pos; dist = math.hypot(pos[0] - rb[0], pos[1] - rb[1])
            break
        except RuntimeError:
            if fixed: raise
    else: raise RuntimeError('no spot')
    robot, pos = rb, pos_
    x, lang, text = render(spec, rng, lang, text); child = spec['kind'] == 'child'; z = rng.uniform(0.95, 1.2) if child else rng.uniform(1.4, 1.65)
    facing = math.atan2(robot[1] - pos[1], robot[0] - pos[0]) if rng.random() < 0.5 else rng.uniform(-math.pi, math.pi)
    srcs = [A.Source(x, (pos[0], pos[1], z), cond.get('level', 62.0) - (2 if child else 0) + rng.normal(0, 2), facing, 'talker', 'talker', start=LEAD)]
    if cond.get('tv') is not None:
        tvpos = cond.get('tv_pos') or (rng.uniform(0.3, 3.7), rng.uniform(0.3, 2.7), 1.0)
        while math.hypot(tvpos[0] - pos[0], tvpos[1] - pos[1]) < 1.0: tvpos = (rng.uniform(0.3, 3.7), rng.uniform(0.3, 2.7), 1.0)
        tx, _, _ = render(voice_spec(rng, 'adult'), rng, 'en' if lang == 'zh' else 'zh', tv_text); tx = np.tile(tx, 4); srcs.append(A.Source(tx, tvpos, cond['tv'], None, 'tv', 'tv'))
    y, parts = sc.mix(srcs, robot, yaw, speed=cond.get('speed', 0.0), turn=cond.get('turn', 0.0)); n_s = y.shape[1] / FS
    c = arr.world_positions(robot, yaw, true=False).mean(0); t_az = D.wrap(math.atan2(pos[1] - c[1], pos[0] - c[0]) - yaw)
    out = dict(found=False, az_true=t_az, dist=dist, lang=lang, text=text, pos=pos, robot=robot, yaw=yaw)
    r = est.estimate_event(y, LEAD + 0.05, n_s - 0.5, max_sources=2)
    if not r['sources']: return out
    az = r['sources'][0]['az']; t0 = int((LEAD + 0.02) * FS); t1 = int((LEAD + len(x) / FS + 0.1) * FS)
    norm = lambda b: (lambda seg: seg / (np.sqrt(np.mean(seg ** 2)) + 1e-12))(b[t0:t1])
    ch = {}
    if bf in ('mvdr', 'all'):
        Z = est.stft(y); T = Z.shape[2]; tc = (np.arange(T) * est.hop + est.nfft / 2) / FS; ch['mvdr'] = norm(est.beamform_mvdr(y, az, noise_mask=(tc >= LEAD - 1.5) & (tc <= LEAD - 0.15)))
    if bf in ('ds', 'all'): ch['ds'] = norm(est.beamform_ds(y, az))
    if bf in ('mic', 'all'): ch['mic'] = norm(y[0])
    out.update(found=True, audio=ch[bf if bf != 'all' else 'mvdr'], audios=ch, az_est=az, conf=r['sources'][0]['conf'], snr_db=r['sources'][0].get('snr_db'), robot=robot, yaw=yaw, err_deg=abs(math.degrees(D.wrap(az - t_az))))
    return out
