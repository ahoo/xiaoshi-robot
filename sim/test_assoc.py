"""Benches for putting people and voices together: who said it (direction + voiceprint + camera), what is a television, and what the robot keeps about a person (profiles).

  python test_assoc.py assoc      200 rooms, each with 2-4 people (enrolled or guests), a television half of the time; real audio through the real array / direction finder / beamformer / voiceprint;
                                  events: a person says the wake word and a command, the television says it, the television replays an enrolled person's voice. Fused against sight alone, voice alone, direction alone.
  python test_assoc.py profile    a made-up family over 14 days (bilingual speech, hours, intents, two children): does the language preference settle, do the hours converge, do attribution mistakes wash out,
                                  is anything kept that must not be
Each writes results/sound_<name>.json. Needs results/sound_voicemodel.npz (python test_sound.py train_voice).
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import acoustics as A
import assoc as AS
import doa as D
import profiles as PR
import sound_lab as SL
import speaker as SP

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
MODEL_FILE = os.path.join(RES, 'sound_voicemodel.npz')
CMD_EN = ['Hey Xiaoshi, put the red ball in the basket', 'Hey Xiaoshi, where is my teddy bear', 'Hey Xiaoshi, go to the sofa', 'Xiaoshi Xiaoshi, tidy up the toys', 'Hey Xiaoshi, scan the house', 'Hey Xiaoshi, forget everyone']
CMD_ZH = ['小拾小拾，把小球放进筐里', '小拾小拾，小球在哪里', '小拾小拾，去沙发那边', '小拾小拾，收拾一下玩具', '小拾小拾，把家里扫描一遍', '小拾小拾，忘掉所有人']


class Tk:
    """What the camera pipeline hands over for one person in view (the fields of people.Track that matter here)."""
    def __init__(self, x, y, who, cls, dist): self.x, self.y, self.who, self.cls, self.dist = x, y, who, cls, dist


def camera(rng, pose, people, fov=100.0, rng_m=4.5):
    """The head camera: people inside the field of view and range are seen (a detector misses 8 %), positions are 10 cm off; an enrolled face is named right at 90 % within 2 m and 70 % further out
    (otherwise left unnamed, 3 % named as someone else); a grown-up/child call is right 95 % of the time."""
    out = []
    for p in people:
        d = math.hypot(p['pos'][0] - pose[0], p['pos'][1] - pose[1]); b = AS.wrap(math.atan2(p['pos'][1] - pose[1], p['pos'][0] - pose[0]) - pose[2])
        if d > rng_m or abs(b) > math.radians(fov) / 2 or rng.random() < 0.08: continue
        who = None
        if p['enrolled'] and rng.random() < (0.9 if d < 2.0 else 0.7): who = p['name']
        elif rng.random() < 0.03: who = str(rng.choice([q['name'] for q in people if q['enrolled']] or [None]))
        cls = p['kind'] if rng.random() < 0.95 else ('adult' if p['kind'] == 'child' else 'child')
        out.append(Tk(p['pos'][0] + rng.normal(0, 0.1), p['pos'][1] + rng.normal(0, 0.1), who, cls, d))
    return out


def render_event(rng, arr, est, room, pose, sources, tv_on_background=None, moving=False):
    """sources: [(spec_or_None, pos (x, y, z), text or None, lang, level, kind)] speaking at LEAD s; a television already playing speech before it (background) when tv_on_background = (pos, level).
    -> (y (M, N), est azimuth list, speech_start)"""
    sc = A.Scene(room, arr, seed=int(rng.integers(1 << 30))); srcs = []
    if tv_on_background is not None:
        tp, tl = tv_on_background; tx, _, _ = SL.render(SL.voice_spec(rng, 'adult'), rng, 'en' if rng.random() < 0.5 else 'zh'); srcs.append(A.Source(np.tile(tx, 5), tp, tl, None, 'tv', 'tv'))
    for spec, pos, text, lang, level, kind in sources:
        x, lg, tx = SL.render(spec, rng, lang, text); srcs.append(A.Source(x, pos, level, math.atan2(pose[1] - pos[1], pose[0] - pos[0]) if kind == 'talker' else None, kind, kind, start=SL.LEAD))
    y, parts = sc.mix(srcs, (pose[0], pose[1]), pose[2], speed=0.15 if moving else 0.0)
    return y, srcs


def world(rng, model, n_enrolled, n_guests, tv):
    """-> dict(pose, arr, est, room, people, tv, vp). People are placed on a 4x3 m floor at least 0.8 m apart; enrolled ones have their voiceprint made from three quiet sentences."""
    room = A.Room(SL.ROOM, rt60=float(rng.uniform(0.35, 0.7))); arr = A.MicArray('4mic', seed=int(rng.integers(1 << 30))); est = D.DOA(arr.nominal)
    pose = (rng.uniform(0.9, 3.1), rng.uniform(0.8, 2.2), rng.uniform(-math.pi, math.pi)); people = []; vp = SP.Voiceprints(model); roles = {}
    names = ['Tom', 'Emma', 'Mia', 'Leo', 'Ann', 'Ben']
    for i in range(n_enrolled + n_guests):
        for _ in range(100):
            pos = (rng.uniform(0.4, 3.6), rng.uniform(0.4, 2.6))
            if math.hypot(pos[0] - pose[0], pos[1] - pose[1]) > 0.9 and all(math.hypot(pos[0] - q['pos'][0], pos[1] - q['pos'][1]) > 0.8 for q in people): break
        kind = 'child' if (i == 1 and rng.random() < 0.5) else 'adult'; spec = SL.voice_spec(rng, kind); enrolled = i < n_enrolled
        people.append(dict(name=names[i] if enrolled else f'guest{i}', spec=spec, pos=pos, kind=kind, enrolled=enrolled, z=float(rng.uniform(0.95, 1.2) if kind == 'child' else rng.uniform(1.4, 1.65))))
        if enrolled:
            embs = []
            for lg in ('en', 'zh', 'en'):
                o = SL.chain(spec, rng, dict(rt60=0.45, dmin=1.0, dmax=2.2), lang=lg, bf='mvdr')
                if o['found']:
                    e, s = SP.utterance_embedding(model, o['audio'])
                    if e is not None: embs.append(e)
            if len(embs) < 3: embs = (embs * 3)[:3] if embs else None
            if embs is None: people[-1]['enrolled'] = False; continue
            vp.enrol(names[i], kind, embs, 'owner'); roles[names[i]] = kind
    tvp = None
    if tv:
        for _ in range(100):
            tvp = (rng.uniform(0.3, 3.7), rng.uniform(0.3, 2.7), 1.0)
            if all(math.hypot(tvp[0] - q['pos'][0], tvp[1] - q['pos'][1]) > 1.0 for q in people) and math.hypot(tvp[0] - pose[0], tvp[1] - pose[1]) > 1.0: break
    return dict(pose=pose, arr=arr, est=est, room=room, people=people, tv=tvp, vp=vp, roles=roles)


def locate_tv(rng, w, model, asc, n_obs=6):
    """The television has been on for a while; the robot goes about the room and hears it from several places. -> learned device (xy) or None, and the true one."""
    for k in range(n_obs):
        pose = (rng.uniform(0.7, 3.3), rng.uniform(0.7, 2.3), rng.uniform(-math.pi, math.pi)); sc = A.Scene(w['room'], w['arr'], seed=int(rng.integers(1 << 30))); tx, _, _ = SL.render(SL.voice_spec(rng, 'adult'), rng, 'en' if k % 2 else 'zh')
        y, _ = sc.mix([A.Source(np.tile(tx, 3), w['tv'], 58.0, None, 'tv', 'tv')], (pose[0], pose[1]), pose[2], dur=4.0); r = w['est'].estimate(y, max_sources=1)
        if r['sources']:
            tr = camera(rng, pose, w['people']); asc.observe_speech(dict(az=r['sources'][0]['az']), pose, tr, t=float(k))


def identify(model, vp, est, y, az, t_start=None):
    Z = est.stft(y); T = Z.shape[2]; tc = (np.arange(T) * est.hop + est.nfft / 2) / SL.FS
    b = est.beamform_mvdr(y, az, noise_mask=(tc >= SL.LEAD - 1.5) & (tc <= SL.LEAD - 0.15)); seg = b[int((SL.LEAD + 0.02) * SL.FS):]; seg = seg / (np.sqrt(np.mean(seg ** 2)) + 1e-12)
    e, s = SP.utterance_embedding(model, seg)
    return vp.identify(e, s) if e is not None else dict(status='unsure', name=None, z=0.0, z2=0.0, anon=None, why='no speech'), e, s


def scenario(args):
    seed, cfg = args
    rng = np.random.default_rng(seed); model = SP.VoiceModel.load(MODEL_FILE)
    w = world(rng, model, cfg['n_enrolled'], cfg['n_guests'], cfg['tv']); asc = AS.Associator(role_of=lambda n: w['roles'].get(n), has_voice=lambda n: n in w['vp'].people, bounds=(0.0, SL.ROOM[0], 0.0, SL.ROOM[1])); rows = []
    if w['tv'] is not None: locate_tv(rng, w, model, asc)
    tv_err = None
    if w['tv'] is not None and asc.devices: tv_err = min(math.hypot(d['xy'][0] - w['tv'][0], d['xy'][1] - w['tv'][1]) for d in asc.devices)
    pose = w['pose']; tvbg = (w['tv'], 56.0) if w['tv'] is not None else None
    def run(kind, talker=None, tv_voice=None):
        srcs = []
        if talker is not None:
            lang = 'zh' if rng.random() < 0.5 else 'en'; text = str(rng.choice(CMD_ZH if lang == 'zh' else CMD_EN)); srcs.append((talker['spec'], (talker['pos'][0], talker['pos'][1], talker['z']), text, lang, float(62 + rng.normal(0, 2) - (2 if talker['kind'] == 'child' else 0)), 'talker'))
        if tv_voice is not None:
            lang = 'zh' if rng.random() < 0.5 else 'en'; text = str(rng.choice(CMD_ZH if lang == 'zh' else CMD_EN)); srcs.append((tv_voice, w['tv'], text, lang, 60.0, 'tv'))
        y, _ = render_event(rng, w['arr'], w['est'], w['room'], pose, srcs, tv_on_background=tvbg if tv_voice is None else None, moving=False)
        r = w['est'].estimate_event(y, SL.LEAD + 0.05, y.shape[1] / SL.FS - 0.4, max_sources=2)
        if not r['sources']: return None
        az = r['sources'][0]['az']; ev = dict(az=az, snr_db=r['sources'][0].get('snr_db'), t=0.0); voice, e, s = identify(model, w['vp'], w['est'], y, az); tracks = camera(rng, pose, w['people'])
        fused = asc.resolve(ev, pose, tracks, voice); vis = asc.resolve(ev, pose, tracks, None); voi = asc.resolve(ev, pose, [], voice)
        return dict(ev=ev, voice=voice, fused=fused, vision=vis, voice_only=voi, n_tracks=len(tracks), tracks=tracks)
    # events by people
    for p in w['people']:
        if rng.random() < 0.7 or p is w['people'][0]:
            o = run('person', talker=p)
            if o is None: continue
            f = o['fused']; truth_vis = any(math.hypot(t.x - p['pos'][0], t.y - p['pos'][1]) < 0.5 for t in o['tracks']); err = abs(math.degrees(AS.wrap(o['ev']['az'] - AS.Associator.bearing(pose, p['pos'][0], p['pos'][1]))))
            rows.append(dict(kind='person', enrolled=p['enrolled'], pkind=p['kind'], seen=truth_vis, doa_err=err, tv=w['tv'] is not None,
                             fused_name=f['name'], fused_source=f['source'], fused_role=f['role'], fused_post=f['posterior'], vision_name=o['vision']['name'], vision_source=o['vision']['source'], voice_name=o['voice']['name'], voice_status=o['voice']['status'],
                             name=p['name'], fused_track_ok=(f['source'] == 'person' and f['track'] is not None and math.hypot(f['track'].x - p['pos'][0], f['track'].y - p['pos'][1]) < 0.6)))
    if w['tv'] is not None:
        o = run('tv', tv_voice=SL.voice_spec(rng, 'adult'))
        if o is not None: rows.append(dict(kind='tv', learned=bool(asc.devices), fused_source=o['fused']['source'], fused_role=o['fused']['role'], fused_post=o['fused']['posterior'], vision_source=o['vision']['source'], voice_status=o['voice']['status']))
        enrolled_adults = [p for p in w['people'] if p['enrolled'] and p['kind'] == 'adult']
        if enrolled_adults:
            o = run('replay', tv_voice=enrolled_adults[0]['spec'])
            if o is not None: rows.append(dict(kind='replay', learned=bool(asc.devices), fused_source=o['fused']['source'], fused_role=o['fused']['role'], fused_name=o['fused']['name'], voice_name=o['voice']['name'], voice_status=o['voice']['status'], target=enrolled_adults[0]['name']))
    return dict(rows=rows, tv_err=tv_err, learned=bool(asc.devices) if w['tv'] is not None else None, n_rays=len(asc.rays))


def rate(xs): return round(float(np.mean(xs)), 3) if len(xs) else None


def bench_assoc(procs=3, n=200, seed0=10000):
    t0 = time.time()
    if not os.path.exists(MODEL_FILE): raise SystemExit('train the voice model first: python test_sound.py train_voice')
    cfgs = [dict(n_enrolled=int(2 + (i % 3 == 0)), n_guests=int(i % 4 == 0), tv=bool(i % 2)) for i in range(n)]
    with mp.Pool(procs) as p: res = p.map(scenario, [(seed0 + 13 * i, cfgs[i]) for i in range(n)])
    rows = [r for s in res for r in s['rows']]; out = {}
    P = [r for r in rows if r['kind'] == 'person']; E = [r for r in P if r['enrolled']]; G = [r for r in P if not r['enrolled']]
    out['n_rooms'] = n; out['seed0'] = seed0; out['n_person_events'] = len(P)
    def acc(rs, key): return rate([r[key] == r['name'] for r in rs])
    out['enrolled_speaker'] = dict(n=len(E), fused_right_name=acc(E, 'fused_name'), sight_only_right_name=acc(E, 'vision_name'), voice_only_right_name=acc(E, 'voice_name'),
                                  fused_wrong_name=rate([(r['fused_name'] is not None and r['fused_name'] != r['name']) for r in E]), sight_only_wrong_name=rate([(r['vision_name'] is not None and r['vision_name'] != r['name']) for r in E]),
                                  voice_only_wrong_name=rate([(r['voice_name'] is not None and r['voice_name'] != r['name']) for r in E]), fused_no_name=rate([r['fused_name'] is None for r in E]))
    for lab, rs in (('seen', [r for r in E if r['seen']]), ('out_of_view', [r for r in E if not r['seen']])):
        out[f'enrolled_speaker_{lab}'] = dict(n=len(rs), fused_right_name=acc(rs, 'fused_name'), sight_only_right_name=acc(rs, 'vision_name'), voice_only_right_name=acc(rs, 'voice_name'), fused_wrong_name=rate([(r['fused_name'] is not None and r['fused_name'] != r['name']) for r in rs]),
                                             owner_role=rate([r['fused_role'] == 'owner' for r in rs if r['pkind'] == 'adult']), adult_role=rate([r['fused_role'] == 'adult' for r in rs if r['pkind'] == 'adult']))
    out['guest_speaker'] = dict(n=len(G), named_as_enrolled=rate([r['fused_name'] is not None for r in G]), owner_role=rate([r['fused_role'] == 'owner' for r in G]), role_counts={k: sum(1 for r in G if r['fused_role'] == k) for k in ('owner', 'adult', 'child', 'unknown', 'device')})
    out['with_television_on'] = dict(n=len([r for r in E if r['tv']]), fused_right_name=acc([r for r in E if r['tv']], 'fused_name'), sight_only_right_name=acc([r for r in E if r['tv']], 'vision_name'),
                                    person_wrongly_called_device=rate([r['fused_source'] == 'device' for r in P if r['tv']]), person_wrongly_called_device_no_tv=rate([r['fused_source'] == 'device' for r in P if not r['tv']]))
    out['direction_error_deg_median'] = round(float(np.median([r['doa_err'] for r in P])), 1)
    T = [r for r in rows if r['kind'] == 'tv']; R = [r for r in rows if r['kind'] == 'replay']
    out['television_says_the_wake_word'] = dict(n=len(T), learned_the_television_first=rate([r['learned'] for r in T]), rejected_as_device=rate([r['fused_source'] == 'device' for r in T]), accepted_as_person_or_unseen=rate([r['fused_source'] in ('person', 'unseen') for r in T]),
                                                owner_role=rate([r['fused_role'] == 'owner' for r in T]), only_with_learned_device_rejected=rate([r['fused_source'] == 'device' for r in T if r['learned']]),
                                                sight_only_accepted=rate([r['vision_source'] in ('person', 'unseen') for r in T]))
    out['television_replays_an_enrolled_adult'] = dict(n=len(R), voiceprint_fooled=rate([r['voice_name'] == r['target'] for r in R]), got_owner_role=rate([r['fused_role'] == 'owner' for r in R]), got_adult_role=rate([r['fused_role'] == 'adult' for r in R]),
                                                       rejected_as_device=rate([r['fused_source'] == 'device' for r in R]))
    tvs = [s for s in res if s['learned'] is not None]
    out['television_learning'] = dict(n=len(tvs), learned=rate([s['learned'] for s in tvs]), position_error_m_median=round(float(np.median([s['tv_err'] for s in tvs if s['tv_err'] is not None])), 2) if any(s['tv_err'] is not None for s in tvs) else None)
    print(json.dumps(out, ensure_ascii=False, indent=1)); out['seconds'] = round(time.time() - t0)
    os.makedirs(RES, exist_ok=True); json.dump(out, open(os.path.join(RES, 'sound_assoc.json'), 'w'), ensure_ascii=False, indent=1, default=float)


# --------------------------------------------------------------------------------------------- profiles over a fortnight
FAMILY = {
    'Tom': dict(role='adult', lang=dict(zh=0.70, en=0.15, mixed=0.15), hours={18: 3, 19: 5, 20: 5, 21: 3, 7: 2, 8: 1}, zone={'living room': 0.7, 'kitchen': 0.2, 'study': 0.1}, n=6,
                intents=dict(where_is=0.25, tidy=0.2, status=0.1, charge=0.1, who_here=0.1, greet=0.25)),
    'Emma': dict(role='adult', lang=dict(zh=0.15, en=0.65, mixed=0.20), hours={7: 4, 8: 4, 9: 2, 17: 3, 18: 4, 19: 4, 20: 3}, zone={'kitchen': 0.6, 'living room': 0.35, 'hall': 0.05}, n=7,
                intents=dict(where_is=0.35, list_items=0.2, go_to=0.1, greet=0.25, status=0.1)),
    'Mia': dict(role='child', lang=dict(zh=0.40, en=0.30, mixed=0.30), hours={16: 3, 17: 4, 18: 4, 19: 2}, zone={'living room': 0.5, 'study': 0.5}, n=4, intents=dict(where_is=0.4, tidy=0.3, greet=0.3)),
    'Leo': dict(role='child', lang=dict(zh=1.0, en=0.0, mixed=0.0), hours={8: 2, 9: 3, 10: 3, 15: 3, 16: 3, 17: 3}, zone={'living room': 0.9, 'hall': 0.1}, n=3, intents=dict(where_is=0.5, greet=0.5)),
}


def _sentence(rng, kind, intent):
    """A real sentence of the kind and intent (from the labelled sets), so that the language the profile sees is what detect_lang says about real text."""
    import voice_data as VD
    import voice_data_en as VE
    pool = dict(zh=VD.COMMANDS, en=VE.COMMANDS_EN_DEV + VE.COMMANDS_EN_HOLDOUT, mixed=VE.COMMANDS_MIX_DEV + VE.COMMANDS_MIX_HOLDOUT)[kind]
    c = [t for t, i, _ in pool if i == intent] or [t for t, _, _ in pool]; return str(rng.choice(c))


def household(seed, days=14, misattribution=0.0, seen_p=0.35, face_acc=0.9, snaps=(3, 7, 14)):
    import lang as Lg
    rng = np.random.default_rng(seed); pr = PR.Profiles(); names = list(FAMILY); out = {}
    for n, f in FAMILY.items(): pr.enrol(n, f['role'], 'owner' if f['role'] == 'adult' else 'Tom', 0.0, voice=True, face=True)
    truth_lang = {n: (max(f['lang'], key=f['lang'].get) if max(f['lang'].values()) >= 0.6 else None) for n, f in FAMILY.items()}
    def cos(a, b): a = np.asarray(a, float); b = np.asarray(b, float); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)) if a.any() and b.any() else 0.0
    for day in range(1, days + 1):
        for n, f in FAMILY.items():
            hrs = list(f['hours']); w = np.array([f['hours'][h] for h in hrs], float); w /= w.sum()
            for _ in range(int(rng.poisson(f['n']))):
                kind = str(rng.choice(list(f['lang']), p=list(f['lang'].values()))); intent = str(rng.choice(list(f['intents']), p=list(f['intents'].values()))); text = _sentence(rng, kind, intent)
                seen_kind = Lg.detect_lang(text)[0]; t = (day - 1) * 86400 + 3600 * int(rng.choice(hrs, p=w)) + 600
                who = n if rng.random() >= misattribution else str(rng.choice([m for m in names if m != n] + [None]))              # the audio-visual stage got it wrong
                if who is not None: pr.observe_utterance(who, seen_kind, intent, t)
            for h in hrs:                                                                                                          # seen now and then, named by the face with its accuracy
                if rng.random() < seen_p * f['hours'][h] / 3.0:
                    z = str(rng.choice(list(f['zone']), p=list(f['zone'].values()))); who = n if rng.random() < face_acc else str(rng.choice([m for m in names if m != n]))
                    pr.observe_presence(who, (day - 1) * 86400 + 3600 * h + 900, z)
        pr.decay(day * 86400.0)
        if day in snaps:
            row = {}
            for n, f in FAMILY.items():
                p = pr.people[n]; tl = np.zeros(24)
                for h, v in f['hours'].items(): tl[h] = v
                row[n] = dict(lang_pref=pr.lang_pref(n), lang_pref_right=(pr.lang_pref(n) == truth_lang[n]), n_utt=p['n_utt'],
                              hours_cosine=round(cos(p['hours'], tl), 3) if p.get('hours') else None, top_zone=pr.top_zone(n), top_zone_right=(pr.top_zone(n) == max(f['zone'], key=f['zone'].get)) if p.get('hours') else None,
                              top_intents=pr.top_intents(n), top_intents_overlap=(len(set(pr.top_intents(n)) & set(sorted(f['intents'], key=f['intents'].get, reverse=True)[:3])) / 3.0) if p.get('habits') else None)
            out[f'day {day}'] = row
    return out, pr


def bench_profile(n=40):
    t0 = time.time(); res = {}
    for m in (0.0, 0.1, 0.2, 0.3):
        runs = [household(1000 + 7 * i, misattribution=m)[0] for i in range(n)]; agg = {}
        for day in ('day 3', 'day 7', 'day 14'):
            a = {}
            for who in FAMILY:
                rows = [r[day][who] for r in runs]
                a[who] = dict(language_preference_right=rate([r['lang_pref_right'] for r in rows]), language_preference_set=rate([r['lang_pref'] is not None for r in rows]), hours_cosine=(round(float(np.mean([r['hours_cosine'] for r in rows if r['hours_cosine'] is not None])), 3) if any(r['hours_cosine'] is not None for r in rows) else None),
                              top_zone_right=rate([r['top_zone_right'] for r in rows if r['top_zone_right'] is not None]) if any(r['top_zone_right'] is not None for r in rows) else None,
                              top_intents_overlap=(round(float(np.mean([r['top_intents_overlap'] for r in rows if r['top_intents_overlap'] is not None])), 3) if any(r['top_intents_overlap'] is not None for r in rows) else None))
            agg[day] = a
        res[f'misattribution {int(100 * m)} %'] = agg; print('misattribution', m, json.dumps(agg['day 14'], ensure_ascii=False))
    # invariants: what must never be in a profile, and what deleting does
    _, pr = household(7, days=14); checks = []
    def chk(name, ok, detail=''): checks.append(dict(name=name, ok=bool(ok), detail=detail))
    for ch in ('Mia', 'Leo'):
        e = pr.export(ch); chk(f'{ch} (child): no hours, zones, intents or items kept', not any(k in e for k in ('hours', 'zones', 'intents', 'items')), sorted(e))
    e = pr.export('Tom'); chk('adult profile holds counters only (no text field except the name)', all(not isinstance(v, str) or k in ('role', 'consent_by', 'lang_set') for k, v in e.items()), sorted(e))
    blob = json.dumps(pr.people, ensure_ascii=False); chk('no sentence from the household appears in the stored profiles', not any(w in blob for w in ('Put the red ball', 'where is my', '收拾一下', 'teddy')), len(blob))
    pr.set_habits('Tom', False); chk('"stop learning habits" removes the habit counters', not any(k in pr.export('Tom') for k in ('hours', 'zones', 'intents', 'items')), sorted(pr.export('Tom')))
    pr.set_language('Emma', 'zh'); chk('"speak Chinese" overrides what was learned', pr.lang_pref('Emma') == 'zh', pr.lang_pref('Emma'))
    c = pr.card('Mia', 'en'); chk('a child card says what is kept and how to delete', 'forget Mia' in c and 'only the name' in c, c[:120])
    pr.forget('Emma'); chk('forgetting a person deletes the whole profile', pr.export('Emma') is None and "don't keep" in pr.card('Emma', 'en'), pr.card('Emma', 'en')[:80])
    pr.forget_all(); chk('forgetting everyone empties the store', not pr.people, len(pr.people))
    _, pr2 = household(9, days=2); before = sum(pr2.people['Tom']['hours']); pr2.decay(2 * 86400 + 60 * 86400); after = sum(pr2.people['Tom']['hours']); chk('habit counts fade (half-life 30 days: 60 days quarter them)', 0.2 < after / before < 0.3, round(after / before, 3))
    res['invariants'] = checks
    for c in checks: print('  ', 'PASS' if c['ok'] else 'FAIL', c['name'], '' if c['ok'] else c['detail'])
    _, pr3 = household(5, days=14)
    res['cards'] = {n: dict(en=pr3.card(n, 'en'), zh=pr3.card(n, 'zh')) for n in ('Tom', 'Emma', 'Mia')}
    for n in ('Tom', 'Mia'): print(pr3.card(n, 'en')); print(pr3.card(n, 'zh'))
    res['seconds'] = round(time.time() - t0); os.makedirs(RES, exist_ok=True); json.dump(res, open(os.path.join(RES, 'sound_profile.json'), 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'assoc'; procs = int(os.environ.get('PROCS', '3')); a = [int(v) for v in sys.argv[2:]]
    if what == 'assoc': bench_assoc(procs=procs, n=a[0] if a else 200, seed0=a[1] if len(a) > 1 else 10000)
    elif what == 'profile': bench_profile(n=a[0] if a else 40)
