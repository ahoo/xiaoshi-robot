"""Benches for sound: where a voice comes from, who it belongs to, in Chinese and English, in rooms with a television.

  python test_sound.py ssl            direction of a voice from the microphone array: 2, 4 and 6 microphones; reverberation, distance, the robot's own noise, a television, children, two talkers
  python test_sound.py ssl_calib      how much does the array's own error (capsule positions, gains) cost
  python test_sound.py train_voice    train the voiceprint projection (60 background voices) and calibrate its scores (20 other voices) -> results/sound_voicemodel.npz
  python test_sound.py voiceprint     who is talking, from the voice alone: households of 2-5 enrolled people, strangers, the television, the robot moving, far talkers, reverberation,
                                      short speech, siblings, enrolling in one language and being recognised in the other, strangers kept apart in RAM
  python test_sound.py anon           only the anonymous strangers (updates that entry of sound_voiceprint.json)
Each writes results/sound_<name>.json. Needs espeak-ng for the speech (apt-get install espeak-ng); without it a speech-like generator is used and the numbers are only about direction.
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import acoustics as A
import doa as D
import sound_lab as SL
import speaker as SP

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
ROOM = (4.0, 3.0, 2.5)

EN_TEXT = ['Where is my ball', 'Put the red ball in the basket please', 'Go to the sofa', 'Stop', 'What is on the floor right now', 'Hey Xiaoshi, scan the house', 'Come here', 'Can you find the sock',
           'Remember this child, his name is Tom', 'How much battery do you have left', 'Tidy up the toys', 'Is anyone home', 'Good morning, how are you today', 'I think the plush toy is under the table',
           'Please go and charge now', 'What is new since yesterday']
ZH_TEXT = ['小球在哪里', '把红色的小球放进筐里', '去沙发那边', '停下', '地上现在有什么', '小拾小拾，把家里扫描一遍', '过来', '帮我找一下袜子', '记住这个孩子，他叫小明', '你还有多少电', '收拾一下玩具',
           '谁在家', '早上好，今天怎么样', '我觉得毛绒玩具在桌子下面', '请去充电', '昨天以后有什么新东西']
# a population of voices: espeak-ng variants and pitches. Children: high pitch on a female variant.
VOICES = {f'{v}_{p}': (v, p) for v in ('m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'f1', 'f2', 'f3', 'f4', 'f5') for p in (25, 45, 65, 85)}
ADULT = [k for k in VOICES if VOICES[k][1] <= 65]; CHILD = [k for k in VOICES if VOICES[k][0].startswith('f') and VOICES[k][1] >= 85]


def talker_signal(rng, child=False, lang=None):
    lang = lang or ('zh' if rng.random() < 0.5 else 'en'); vk = rng.choice(CHILD if child else ADULT); v, p = VOICES[vk]
    text = rng.choice(ZH_TEXT if lang == 'zh' else EN_TEXT); return A.synth(text, lang, v, p + (10 if child else 0), 190 if child else 170), lang, vk


def az_truth(array, robot, yaw, pos):
    c = array.world_positions(robot, yaw, true=False).mean(0); return D.wrap(math.atan2(pos[1] - c[1], pos[0] - c[0]) - yaw)


def place(rng, robot, dmin=0.7, dmax=4.5):
    for _ in range(500):
        p = (rng.uniform(0.3, ROOM[0] - 0.3), rng.uniform(0.3, ROOM[1] - 0.3)); d = math.hypot(p[0] - robot[0], p[1] - robot[1])
        if dmin <= d <= dmax: return p, d
    raise RuntimeError('no spot')


def trial(args):
    """One talker, one condition. -> dict(error_deg, conf, distance, ...)"""
    seed, cfg = args
    rng = np.random.default_rng(seed); room = A.Room(ROOM, rt60=cfg.get('rt60', 0.5)); arr = A.MicArray(cfg.get('array', '4mic'), pos_err_mm=cfg.get('pos_err_mm', 0.0), gain_err_db=cfg.get('gain_err_db', 0.0), seed=seed)
    sc = A.Scene(room, arr, seed=seed + 1); est = D.DOA(arr.nominal)
    robot = (rng.uniform(0.7, 3.3), rng.uniform(0.7, 2.3)); yaw = rng.uniform(-math.pi, math.pi); pos, dist = place(rng, robot); child = cfg.get('child', False)
    x, lang, vk = talker_signal(rng, child, cfg.get('lang')); z = rng.uniform(0.95, 1.2) if child else rng.uniform(1.4, 1.65); facing = math.atan2(robot[1] - pos[1], robot[0] - pos[0]) if rng.random() < 0.5 else rng.uniform(-math.pi, math.pi)
    LEAD = 2.0; src = [A.Source(x, (pos[0], pos[1], z), cfg.get('level', 62.0) - (2 if child else 0) + rng.normal(0, 2), facing, 'talker', 'talker', start=LEAD)]
    tvpos = None
    if cfg.get('tv') is not None:
        tvpos = (rng.uniform(0.3, 3.7), rng.uniform(0.3, 2.7), 1.0)
        while math.hypot(tvpos[0] - pos[0], tvpos[1] - pos[1]) < 1.0: tvpos = (rng.uniform(0.3, 3.7), rng.uniform(0.3, 2.7), 1.0)
        tv, _, _ = talker_signal(rng, False, 'en' if lang == 'zh' else 'zh'); tv = np.tile(tv, 4); src.append(A.Source(tv, tvpos, cfg['tv'], None, 'tv', 'tv'))
    y, parts = sc.mix(src, robot, yaw, speed=cfg.get('speed', 0.0), turn=cfg.get('turn', 0.0))
    t = az_truth(arr, robot, yaw, pos); n_s = y.shape[1] / A.FS
    r = est.estimate_event(y, LEAD + 0.05, n_s - 0.5, max_sources=2); r0 = est.estimate(y)                  # with the background before the keyword; and without (the whole recording, no reference)
    out = dict(dist=dist, lang=lang, child=child, voice=vk, found=False, error_deg=None, conf=None, error_no_bg_deg=None)
    if r0['sources']: out['error_no_bg_deg'] = abs(math.degrees(D.wrap(r0['sources'][0]['az'] - t)))
    if r['sources']:
        s = r['sources'][0]; out.update(found=True, error_deg=abs(math.degrees(D.wrap(s['az'] - t))), conf=s['conf'], n_sources=len(r['sources']), in_top2=min(abs(math.degrees(D.wrap(q['az'] - t))) for q in r['sources']))
    return out


def summarise(rows, label=''):
    e = np.array([r['error_deg'] for r in rows if r['found']]); n = len(rows)
    if not len(e): return dict(label=label, n=n, found=0)
    e0 = np.array([180.0 if r['error_no_bg_deg'] is None else r['error_no_bg_deg'] for r in rows])
    return dict(label=label, n=n, found=round(len(e) / n, 3), median_deg=round(float(np.median(e)), 1), mean_deg=round(float(e.mean()), 1), p90_deg=round(float(np.percentile(e, 90)), 1),
                within_10=round(float((e <= 10).sum() / n), 3), within_20=round(float((e <= 20).sum() / n), 3), within_30=round(float((e <= 30).sum() / n), 3),
                no_background_median_deg=round(float(np.median(e0)), 1), no_background_within_20=round(float((e0 <= 20).sum() / n), 3))


def two_talkers(args):
    """Two people talk at the same time, voices of equal loudness at the robot's side; are both directions found?"""
    seed, cfg = args
    rng = np.random.default_rng(seed); room = A.Room(ROOM, rt60=cfg.get('rt60', 0.5)); arr = A.MicArray(cfg.get('array', '4mic'), seed=seed); sc = A.Scene(room, arr, seed=seed + 1); est = D.DOA(arr.nominal)
    robot = (rng.uniform(0.9, 3.1), rng.uniform(0.8, 2.2)); yaw = rng.uniform(-math.pi, math.pi); srcs = []; truths = []
    while len(srcs) < 2:
        pos, d = place(rng, robot, 0.8, 2.6); t = az_truth(arr, robot, yaw, pos)
        if all(abs(math.degrees(D.wrap(t - u))) >= cfg.get('min_sep', 30) for u in truths):
            x, lang, vk = talker_signal(rng, False); srcs.append(A.Source(x, (pos[0], pos[1], 1.5), 62.0 + rng.normal(0, 1.0), math.atan2(robot[1] - pos[1], robot[0] - pos[0]), 'talker', f'talker{len(srcs)}', start=2.0)); truths.append(t)
    y, parts = sc.mix(srcs, robot, yaw); r = est.estimate_event(y, 2.05, y.shape[1] / A.FS - 0.5, max_sources=2); found = [s for s in r['sources'] if s['conf'] > 0.1]
    hit = [any(abs(math.degrees(D.wrap(s['az'] - t))) <= 20 for s in found) for t in truths]; sep = abs(math.degrees(D.wrap(truths[0] - truths[1])))
    return dict(sep=sep, hits=hit, n_found=len(found), both=all(hit), first=hit[0], second=hit[1])


def bench_ssl(procs=3, n=120):
    t0 = time.time(); results = {}
    base = dict(array='4mic', rt60=0.5)
    cases = [('base: 4 microphones, reverberation 0.5 s, robot idle', base),
             ('2 microphones', dict(base, array='2mic')), ('6 microphones', dict(base, array='6mic')),
             ('reverberation 0.3 s', dict(base, rt60=0.3)), ('reverberation 0.8 s', dict(base, rt60=0.8)),
             ('robot driving (0.15 m/s)', dict(base, speed=0.15)), ('robot driving slowly (0.05 m/s)', dict(base, speed=0.05)),
             ('television at 55 dB (1 m)', dict(base, tv=55.0)), ('television at 62 dB', dict(base, tv=62.0)), ('television at 70 dB', dict(base, tv=70.0)),
             ('child talking', dict(base, child=True)), ('Chinese only', dict(base, lang='zh')), ('English only', dict(base, lang='en')),
             ('quiet talker (52 dB at 1 m)', dict(base, level=52.0)), ('driving and television 62 dB', dict(base, speed=0.15, tv=62.0))]
    with mp.Pool(procs) as p:
        for name, cfg in cases:
            rows = p.map(trial, [(1000 * i + 7, cfg) for i in range(n)]); results[name] = summarise(rows, name); results[name]['rows'] = rows; results[name]['in_top2_within_20'] = round(float(np.mean([r.get('in_top2', 999) <= 20 for r in rows])), 3); print(results[name]['label'], {k: v for k, v in results[name].items() if k not in ('rows', 'label')}, flush=True)
        # by distance, from the base case
        rows = results[cases[0][0]]['rows']; by = {}
        for lo, hi in ((0.7, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.6)): by[f'{lo}-{hi} m'] = summarise([r for r in rows if lo <= r['dist'] < hi], f'{lo}-{hi} m')
        results['by_distance'] = by; print('by distance', by)
        tt = {}
        for sep in (30, 60, 100):
            rr = p.map(two_talkers, [(5000 + 31 * i, dict(min_sep=sep)) for i in range(60)]); tt[f'at least {sep} degrees apart'] = dict(n=len(rr), both_found=round(float(np.mean([r['both'] for r in rr])), 3), first=round(float(np.mean([r['first'] for r in rr])), 3), second=round(float(np.mean([r['second'] for r in rr])), 3))
        results['two_talkers'] = tt; print('two talkers', tt)
        for k in ('2mic', '6mic'):
            rr = p.map(two_talkers, [(7000 + 31 * i, dict(min_sep=60, array=k)) for i in range(60)]); results[f'two_talkers_{k}'] = dict(n=len(rr), both_found=round(float(np.mean([r['both'] for r in rr])), 3), first=round(float(np.mean([r['first'] for r in rr])), 3))
    os.makedirs(RES, exist_ok=True); json.dump(results, open(os.path.join(RES, 'sound_ssl.json'), 'w'), ensure_ascii=False, indent=1, default=float); print('done in', round(time.time() - t0), 's')


# --------------------------------------------------------------------------------------------- voiceprint
MODEL_FILE = os.path.join(RES, 'sound_voicemodel.npz')


def _speaker_utts(args):
    """One speaker, several utterances in a given condition -> list of dict(raw features, speech_s, lang, dist, snr, err_deg) for each beamformer variant."""
    seed, kind, n, cond, langs = args[:5]; spec_seed = args[5] if len(args) > 5 else seed                    # the voice comes from spec_seed, what it says and how from seed
    cond = CONDS.get(cond, cond) if isinstance(cond, str) else cond
    spec = SL.voice_spec(np.random.default_rng(spec_seed), kind); rng = np.random.default_rng(seed); out = []
    for i in range(n):
        c = dict(cond(rng)) if callable(cond) else dict(cond); lang = langs[i % len(langs)] if langs else None
        o = SL.chain(spec, rng, c, lang=lang, bf='all')
        if not o['found']: continue
        row = dict(lang=o['lang'], dist=o['dist'], err_deg=o['err_deg'], snr_db=o.get('snr_db'), raw={})
        for k, a in o['audios'].items():
            raw, sp = SP.raw_features(a)
            if raw is not None: row['raw'][k] = raw; row['speech_s'] = sp
        if row['raw']: out.append(row)
    return dict(seed=seed, spec=spec, utts=out)


def _train_cond(rng):
    return dict(rt60=float(rng.uniform(0.3, 0.8)), speed=0.15 if rng.random() < 0.2 else 0.0, tv=float(rng.uniform(45, 58)) if rng.random() < 0.3 else None, level=float(rng.uniform(56, 66)))


CONDS = {'enrol': lambda rng: dict(rt60=0.45, level=62.0, dmin=1.0, dmax=2.2), 'quiet': lambda rng: dict(rt60=float(rng.uniform(0.35, 0.6))), 'robot moving': lambda rng: dict(rt60=0.5, speed=0.15),
         'television 55 dB': lambda rng: dict(rt60=0.5, tv=55.0), 'television 62 dB': lambda rng: dict(rt60=0.5, tv=62.0), 'reverberation 0.8 s': lambda rng: dict(rt60=0.8),
         'far (3-4 m)': lambda rng: dict(rt60=0.5, dmin=2.6, dmax=4.2), 'quiet talker 52 dB': lambda rng: dict(rt60=0.5, level=52.0)}


def train_voice_model(procs=3, n_spk=60, n_dev=20, n_utt=12, seed=500):
    """60 background voices for the projection, 20 more (not in the projection) to calibrate the scores; 12 utterances each, in random rooms, noise, television and driving."""
    t0 = time.time()
    with mp.Pool(procs) as p: rows = p.map(_speaker_utts, [(seed + 17 * i, 'child' if i % 5 == 0 else 'adult', n_utt, _train_cond, None) for i in range(n_spk + n_dev)])
    X = []; y = []
    for i, r in enumerate(rows[:n_spk]):
        for u in r['utts']:
            if 'mvdr' in u['raw']: X.append(u['raw']['mvdr']); y.append(i)
    m = SP.VoiceModel(k=24).fit(np.array(X), np.array(y))
    P = SP.Voiceprints(m); dev = {}
    for j, r in enumerate(rows[n_spk:]):
        us = [u for u in r['utts'] if 'mvdr' in u['raw']]; en = [u for u in us if u['lang'] == 'en'][:2] + [u for u in us if u['lang'] == 'zh'][:1]
        if len(en) < 3: continue
        P.enrol(f'd{j}', 'adult', [m.embed(u['raw']['mvdr']) for u in en], 'owner'); dev[j] = (en, us)
    tz = []; nz = []
    for j, (en, us) in dev.items():
        for u in us:
            if any(u is e for e in en): continue
            for nme, z in P.scores(m.embed(u['raw']['mvdr'])).items(): (tz if nme == f'd{j}' else nz).append(z)
    cal = m.calibrate(tz, nz); os.makedirs(RES, exist_ok=True); m.save(MODEL_FILE)
    print('voice model trained on', len(set(y)), 'speakers,', len(X), 'utterances in', round(time.time() - t0), 's; calibration on', len(dev), 'other speakers: target z', round(float(cal[0]), 2), '+-', round(float(cal[1]), 2), ', others', round(float(cal[2]), 2), '+-', round(float(cal[3]), 2),
          ', accept above', round(float(cal[4]), 2), '(1 wrong person in 100)')
    return m


def _household_stats(model, pool, rng, n_house, sizes, enrol_mode, key, thr=None):
    """pool: list of dict(enrol={lang: [raw..]}, utts=[...]). Random households from the pool: enrol each member, test the members' utterances and the others' as strangers."""
    P = SP.Voiceprints(model); res = dict(member=0, correct=0, wrong=0, member_unsure=0, member_unknown=0, guest=0, guest_named=0, guest_unknown=0, guest_unsure=0, too_short=0, closed=0)
    zt = []; zn = []
    for h in range(n_house):
        n = int(rng.choice(sizes)); idx = rng.choice(len(pool), n, replace=False); P = SP.Voiceprints(model)
        for j in idx:
            en = pool[j]['enrol']; langs = ['en', 'zh'] if enrol_mode == 'both' else [enrol_mode]
            raws = []
            for lg in langs: raws += en[lg]
            raws = raws[:3] if enrol_mode != 'both' else (en['en'][:2] + en['zh'][:1])
            if len(raws) < 3: break
            P.enrol(f's{j}', 'adult', [model.embed(r) for r in raws], 'owner')
        if len(P.people) < n: continue
        if thr: P.Z_ACCEPT, P.Z_MARGIN, P.Z_UNKNOWN = thr
        others = [k for k in range(len(pool)) if k not in set(idx.tolist())]
        for j in idx:
            for u in pool[j]['utts']:
                r = u['raw'].get(key)
                if r is None: continue
                e = model.embed(r); sc = P.scores(e); zt.append(sc[f's{j}']); zn += [z for nme, z in sc.items() if nme != f's{j}']
                res['member'] += 1; res['closed'] += int(max(sc, key=sc.get) == f's{j}'); o = P.identify(e, u.get('speech_s'))
                if u.get('speech_s', 9) < SP.MIN_SPEECH_S: res['too_short'] += 1
                if o['status'] == 'known': res['correct' if o['name'] == f's{j}' else 'wrong'] += 1
                elif o['status'] == 'unsure': res['member_unsure'] += 1
                else: res['member_unknown'] += 1
        for k in rng.choice(others, min(2, len(others)), replace=False):
            for u in pool[k]['utts']:
                r = u['raw'].get(key)
                if r is None: continue
                e = model.embed(r); o = P.identify(e, u.get('speech_s')); res['guest'] += 1
                if o['status'] == 'known': res['guest_named'] += 1
                elif o['status'] == 'unsure': res['guest_unsure'] += 1
                else: res['guest_unknown'] += 1
    m = max(res['member'], 1); g = max(res['guest'], 1)
    return dict(members=res['member'], closed_set_correct=round(res['closed'] / m, 3), correct=round(res['correct'] / m, 3), wrong_person=round(res['wrong'] / m, 3), unsure=round(res['member_unsure'] / m, 3), unknown=round(res['member_unknown'] / m, 3),
                guests=res['guest'], guest_named_as_member=round(res['guest_named'] / g, 3), guest_unknown=round(res['guest_unknown'] / g, 3), guest_unsure=round(res['guest_unsure'] / g, 3)), (np.array(zt), np.array(zn))


def _eer(zt, zn):
    best = (1.0, 0.0)
    for th in np.linspace(min(zt.min(), zn.min()), max(zt.max(), zn.max()), 400):
        far = float((zn >= th).mean()); frr = float((zt < th).mean())
        if abs(far - frr) < best[0]: best = (abs(far - frr), (far + frr) / 2)
    return round(best[1], 4)


def bench_voiceprint(procs=3, n_pool=24):
    t0 = time.time(); results = {}
    model = SP.VoiceModel.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else train_voice_model(procs)
    # the test voices: disjoint from the training voices (other seeds, continuous parameters)
    enrol_cond = 'enrol'; conds = {k: k for k in CONDS if k != 'enrol'}
    with mp.Pool(procs) as p:
        kinds = ['child' if i % 6 == 0 else 'adult' for i in range(n_pool)]
        en = p.map(_speaker_utts, [(90000 + 31 * i, kinds[i], 3, enrol_cond, ['en'], 20000 + 31 * i) for i in range(n_pool)]); zh = p.map(_speaker_utts, [(91000 + 31 * i, kinds[i], 3, enrol_cond, ['zh'], 20000 + 31 * i) for i in range(n_pool)])
        base_pool = [dict(enrol=dict(en=[u['raw']['mvdr'] for u in en[i]['utts'] if 'mvdr' in u['raw']], zh=[u['raw']['mvdr'] for u in zh[i]['utts'] if 'mvdr' in u['raw']])) for i in range(n_pool)]
        print('enrolment utterances rendered', round(time.time() - t0), 's', flush=True)
        for name, cf in conds.items():
            rows = p.map(_speaker_utts, [(50000 + 31 * i, kinds[i], 6, cf, None, 20000 + 31 * i) for i in range(n_pool)])      # the same speakers (same voice seed), other utterances
            pool = [dict(enrol=base_pool[i]['enrol'], utts=rows[i]['utts']) for i in range(n_pool)]
            for key in ('mvdr', 'ds', 'mic'):
                if name not in ('quiet', 'television 55 dB') and key != 'mvdr': continue
                r, (zt, zn) = _household_stats(model, pool, np.random.default_rng(1), 150, [2, 3, 4, 5], 'both', key)
                r['eer'] = _eer(zt, zn); r['direction_error_median_deg'] = round(float(np.median([u['err_deg'] for q in pool for u in q['utts']])), 1)
                results[f'{name} [{ {"mvdr": "beamformer MVDR", "ds": "beamformer delay-and-sum", "mic": "one microphone, no beamformer"}[key] }]'] = r; print(name, key, r, flush=True)
            if name == 'quiet': quiet_pool = pool
        # enrol in one language, recognised in both
        for mode in ('en', 'zh', 'both'):
            r, _ = _household_stats(model, quiet_pool, np.random.default_rng(2), 150, [2, 3, 4, 5], mode, 'mvdr'); results[f'enrolled with {mode} only' if mode != 'both' else 'enrolled with both languages'] = r; print('enrol', mode, r, flush=True)
        # by length of speech
        by = {}
        for lo, hi in ((0.5, 1.2), (1.2, 2.0), (2.0, 9.0)):
            pl = [dict(enrol=q['enrol'], utts=[u for u in q['utts'] if lo <= u.get('speech_s', 0) < hi]) for q in quiet_pool]
            r, _ = _household_stats(model, pl, np.random.default_rng(3), 150, [3, 4], 'both', 'mvdr'); by[f'speech {lo}-{hi} s'] = r
        results['by_length_of_speech'] = by; print('by length', by, flush=True)
        # siblings: a second voice made from the first by a small change of pitch and warp
        sib = p.map(_sibling, [(70000 + 17 * i,) for i in range(40)]); results['siblings'] = dict(n=len(sib), mixed_up=round(float(np.mean([s['mixed_up'] for s in sib])), 3), named_right=round(float(np.mean([s['right'] for s in sib])), 3), unsure=round(float(np.mean([s['unsure'] for s in sib])), 3), z_gap_median=round(float(np.median([s['gap'] for s in sib])), 2))
        print('siblings', results['siblings'], flush=True)
        # strangers kept apart in RAM
        an = p.map(_anon_cluster, [(80000 + 13 * i,) for i in range(20)]); results['anonymous_strangers'] = dict(n=len(an), clusters_per_4_strangers=round(float(np.mean([a['clusters'] for a in an])), 2), purity=round(float(np.mean([a['purity'] for a in an])), 3))
        print('anonymous', results['anonymous_strangers'], flush=True)
    os.makedirs(RES, exist_ok=True); json.dump(results, open(os.path.join(RES, 'sound_voiceprint.json'), 'w'), ensure_ascii=False, indent=1, default=float); print('done in', round(time.time() - t0), 's')


def _calib_trial(args):
    seed, cfg = args; return trial((seed, cfg))


def bench_ssl_calib(procs=3, n=100):
    """How much does an array that is not quite as drawn cost: capsule positions off by a few tenths of a millimetre (the direction finder assumes the nominal geometry) and capsule gains that differ."""
    t0 = time.time(); results = {}; base = dict(array='4mic', rt60=0.5)
    cases = [('nominal array', base), ('capsule positions 0.5 mm off (1 sigma)', dict(base, pos_err_mm=0.5)), ('capsule positions 1 mm off', dict(base, pos_err_mm=1.0)), ('capsule positions 2 mm off', dict(base, pos_err_mm=2.0)),
             ('capsule gains 1 dB apart (1 sigma)', dict(base, gain_err_db=1.0)), ('capsule gains 3 dB apart', dict(base, gain_err_db=3.0)), ('positions 1 mm and gains 2 dB', dict(base, pos_err_mm=1.0, gain_err_db=2.0))]
    with mp.Pool(procs) as p:
        for name, cfg in cases:
            rows = p.map(_calib_trial, [(1000 * i + 7, cfg) for i in range(n)]); results[name] = summarise(rows, name); print(name, {k: v for k, v in results[name].items() if k != 'label'}, flush=True)
    os.makedirs(RES, exist_ok=True); json.dump(results, open(os.path.join(RES, 'sound_ssl_calib.json'), 'w'), ensure_ascii=False, indent=1, default=float); print('done in', round(time.time() - t0), 's')


def _sibling(args):
    """Two voices that are alike (same synthesiser, pitch within 4, warp within 0.02): enrol both, say things with each: are they mixed up?"""
    seed, = args; rng = np.random.default_rng(seed); model = SP.VoiceModel.load(MODEL_FILE); a = SL.voice_spec(rng, 'adult'); b = dict(a, p=float(np.clip(a['p'] + rng.choice([-4, 4]), 0, 99)), warp=a['warp'] + float(rng.choice([-0.02, 0.02])))
    P = SP.Voiceprints(model); cond = dict(rt60=0.45, dmin=1.0, dmax=2.2)
    for nm, sp in (('a', a), ('b', b)):
        raws = [SL.chain(sp, rng, cond, lang=lg, bf='mvdr')['audio'] for lg in ('en', 'zh', 'en')]; P.enrol(nm, 'adult', [model.embed(SP.raw_features(r)[0]) for r in raws], 'owner')
    res = dict(mixed_up=0, right=0, unsure=0, gap=[]); n = 0
    for nm, sp in (('a', a), ('b', b)):
        for k in range(5):
            o = SL.chain(sp, rng, dict(rt60=0.5), lang=('zh' if k % 2 else 'en'), bf='mvdr')
            if not o['found']: continue
            raw, s = SP.raw_features(o['audio']); r = P.identify(model.embed(raw), s); n += 1
            if r['status'] == 'known': res['right' if r['name'] == nm else 'mixed_up'] += 1
            else: res['unsure'] += 1
            res['gap'].append(r['z'] - r['z2'])
    return dict(mixed_up=res['mixed_up'] / max(n, 1), right=res['right'] / max(n, 1), unsure=res['unsure'] / max(n, 1), gap=float(np.median(res['gap'])) if res['gap'] else 0.0)


def bench_anon(procs=3):
    """Only the anonymous strangers (the merge threshold was recalibrated after the full voiceprint run): 4 strangers x 5 sentences, shuffled; how many clusters, how pure. Updates that entry of sound_voiceprint.json."""
    t0 = time.time()
    with mp.Pool(procs) as p: an = p.map(_anon_cluster, [(80000 + 13 * i,) for i in range(30)])
    r = dict(n=len(an), merge_cosine=SP.Voiceprints.MERGE_COS, clusters_per_4_strangers=round(float(np.mean([a['clusters'] for a in an])), 2), purity=round(float(np.mean([a['purity'] for a in an])), 3)); print('anonymous', r)
    fn = os.path.join(RES, 'sound_voiceprint.json'); d = json.load(open(fn)); d['anonymous_strangers'] = r; json.dump(d, open(fn, 'w'), ensure_ascii=False, indent=1, default=float); print('done in', round(time.time() - t0), 's')


def _anon_cluster(args):
    seed, = args; rng = np.random.default_rng(seed); model = SP.VoiceModel.load(MODEL_FILE); P = SP.Voiceprints(model); specs = [SL.voice_spec(rng, 'adult') for _ in range(3)] + [SL.voice_spec(rng, 'child')]; items = []
    for si, sp in enumerate(specs):
        for k in range(5):
            o = SL.chain(sp, rng, dict(rt60=0.5), bf='mvdr')
            if o['found']:
                raw, s = SP.raw_features(o['audio'])
                if raw is not None and s >= SP.MIN_SPEECH_S: items.append((si, model.embed(raw)))
    order = rng.permutation(len(items)); ids = {}; lab = []
    for i in order: lab.append((items[i][0], P.identify(items[i][1], 3.0)['anon']))
    clusters = {}
    for si, a in lab: clusters.setdefault(a, []).append(si)
    purity = sum(max(np.bincount(v)) for v in clusters.values()) / max(len(lab), 1)
    return dict(clusters=len(clusters), purity=purity)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'ssl'; procs = int(os.environ.get('PROCS', '3')); a = [int(v) for v in sys.argv[2:]]
    if what == 'ssl': bench_ssl(procs=procs, n=a[0] if a else 120)
    elif what == 'ssl_calib': bench_ssl_calib(procs=procs, n=a[0] if a else 100)
    elif what == 'voiceprint': bench_voiceprint(procs=procs)
    elif what == 'anon': bench_anon(procs=procs)
    elif what == 'train_voice': train_voice_model(procs)
