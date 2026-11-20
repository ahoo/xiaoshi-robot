"""Day 2: the whole sound side in the MuJoCo home. A family that speaks Chinese and English, a television that talks (and says the robot's name), a voice from out of view, a stranger, a recording of the
father's voice played by the television, a child. Real audio is rendered for every utterance (soundfront.py) and goes through the direction finder, the beamformer and the voiceprint; the camera tracks and
the face stand-in come from the people system of round 14; the dialogue is voice.py.

  python test_day2.py [seed]       writes results/sound_day2.json (MuJoCo, about 6 minutes of wall time)
"""
import json
import math
import os
import sys
import time

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
HAN = lambda t: any('一' <= c <= '鿿' for c in (t or ''))


def _day2_job(seed=5):
    import assoc as AS
    import profiles as PR
    import sound_lab as SL
    import soundfront as SF
    import speaker as SP
    from home import Home
    from test_home import Director, _spot_in_front, run_supervised_day
    from voice import SimASR, VoiceSystem
    box = {}; checks = []; table = []
    pe = dict(names=['adult_m', 'adult_f', 'child_8', 'adult_m'], who=['Tom', 'Emma', 'Mia', 'guest'], family=['A', 'B', 'A', 'C'], guard=True, comply=[1.0, 1.0, 0.7, 1.0], steps=[[], [], [], []], registry_path=f'/tmp/day2_registry_{seed}.json')
    for pth in (pe['registry_path'], f'/tmp/day2_inv_{seed}.json'):
        if os.path.exists(pth): os.remove(pth)
    def chk(name, ok, detail=''): checks.append(dict(name=name, ok=bool(ok), detail=detail))

    def setup(sup):
        rng = np.random.default_rng(seed + 1); home = Home(sup.r, sup.a, inv_path=f'/tmp/day2_inv_{seed}.json'); sup.home = home
        vs = VoiceSystem(sup, home, sup.ppl, asr=SimASR(0.04, seed, accent=0.3), seed=seed); ppl = sup.ppl; reg = ppl.registry
        model = SP.VoiceModel.load(os.path.join(RES, 'sound_voicemodel.npz')); vp = SP.Voiceprints(model); pr = PR.Profiles(); vs.voiceprints = vp; vs.profiles = pr
        asc = AS.Associator(role_of=lambda n: vp.role(n) or reg.role(n), has_voice=lambda n: n in vp.people, bounds=(0.0, 4.0, 0.0, 3.0))
        voices = {0: SL.voice_spec(rng, 'adult'), 1: SL.voice_spec(rng, 'adult'), 2: SL.voice_spec(rng, 'child'), 3: SL.voice_spec(rng, 'adult')}
        # yesterday: Tom and Mia were enrolled by face (with consent), and Tom, Emma and Mia by voice
        for nm, i, role, by in (('Tom', 0, 'adult', 'owner'), ('Mia', 2, 'child', 'Tom')):
            reg.enrol(nm, role, [ppl.face.look(nm, 1.5, 1.0, 0.0)[0] for _ in range(8)], by, 0.0)
        for nm, i, role, by in (('Tom', 0, 'adult', 'owner'), ('Emma', 1, 'adult', 'owner'), ('Mia', 2, 'child', 'Tom')):
            embs = []
            for lg in ('en', 'zh', 'en'):
                o = SL.chain(voices[i], rng, dict(rt60=0.5, dmin=1.0, dmax=2.2), lang=lg, bf='mvdr'); e, s = SP.utterance_embedding(model, o['audio']); embs.append(e)
            vp.enrol(nm, role, embs, by, 0.0, langs=('en', 'zh')); pr.enrol(nm, role, by, 0.0, voice=True, face=nm in reg.people)
        front = SF.SoundFront(sup, ppl, vs, model, vp, asc, voices, seed=seed); d = Director(sup, vs, ppl); box.update(home=home, vs=vs, d=d, front=front, asc=asc, vp=vp, pr=pr, reg=reg, voices=voices)
        build(d, sup, home, vs, front, asc, vp, pr, reg, ppl); sup.day = d

    def build(d, sup, home, vs, front, asc, vp, pr, reg, ppl):
        L = box.setdefault('last', {}); P = box.setdefault('P', {})
        def say(i, text, label, truth, **kw):
            def f(s):
                rec, ident = front.speak(i, text, **kw); L['rec'] = rec; L['ident'] = ident
                table.append(dict(t=round(s.r.t, 1), label=label, said=text, truth=truth, source=ident.get('source'), name=ident.get('name'), role=ident.get('role'), posterior=ident.get('posterior'), direction_error_deg=round(ident.get('direction_error_deg', -1), 1),
                                  voice=(ident.get('voice') or {}).get('status'), voice_z=round((ident.get('voice') or {}).get('z', 0.0), 2), action=rec.get('action'), intent=rec.get('intent'), reply=rec.get('reply'), lang=vs.lang,
                                  hypotheses=ident.get('hypotheses'), tracks=[(k.who, k.cls, round(math.degrees(AS.Associator.bearing(s.a.pose(), k.x, k.y))), round(k.dist, 1)) for k in ppl.tracks()],
                                  true_bearing_deg=(round(math.degrees(AS.wrap(math.atan2((front.tv['pos'][1] if i is None else ppl.people.state[i]['y']) - s.r.true_pose()[1], (front.tv['pos'][0] if i is None else ppl.people.state[i]['x']) - s.r.true_pose()[0]) - s.r.true_pose()[2]))) if (i is not None or front.tv) else None)))
            return f
        def tv_on(s):
            rx, ry, th = s.r.true_pose(); cands = [(3.7, 0.3), (3.7, 2.7), (0.3, 2.7), (0.3, 0.3)]; tv = max(cands, key=lambda c: math.hypot(c[0] - rx, c[1] - ry)); front.tv = dict(pos=(tv[0], tv[1], 1.0), level=55.0, on=True); box['tv_true'] = tv
        def hidden(s, avoid, want=1.7):
            """A spot the head camera cannot see: more than 75 degrees off the robot's heading, 1.0-2.6 m away, inside the room, 0.8 m from the others; the one nearest `want` m away."""
            rx, ry, th = s.r.true_pose(); best = None
            for gx in np.arange(0.45, 3.6, 0.15):
                for gy in np.arange(0.45, 2.6, 0.15):
                    dd = math.hypot(gx - rx, gy - ry); b = abs(AS.wrap(math.atan2(gy - ry, gx - rx) - th))
                    if 1.0 <= dd <= 2.6 and b >= math.radians(75) and all(math.hypot(gx - q[0], gy - q[1]) >= 0.8 for q in avoid):
                        sc_ = abs(dd - want) + 0.3 * abs(b - math.radians(110))
                        if best is None or sc_ < best[0]: best = (sc_, float(gx), float(gy))
            return (best[1], best[2]) if best else None
        def polar(s, d_, ang, avoid):
            """A spot d_ m from the robot at `ang` radians from where it faces, inside the room (0.4 m from the walls), 0.8 m from the others; closer in and 10 degrees round if it does not fit."""
            rx, ry, th = s.r.true_pose()
            for dd in (d_, 0.85 * d_, 0.7 * d_, 1.15 * d_):
                for da in (0.0, 0.17, -0.17, 0.35, -0.35):
                    px, py = rx + dd * math.cos(th + ang + da), ry + dd * math.sin(th + ang + da)
                    if 0.4 < px < 3.6 and 0.4 < py < 2.6 and all(math.hypot(px - q[0], py - q[1]) >= 0.8 for q in avoid): return float(px), float(py)
            return _spot_in_front(s, d_, side=ang, avoid=avoid)
        def place(s):
            rx, ry, th = s.r.true_pose(); avoid = []; box['pose0'] = (rx, ry, th)
            for i, (d_, ang) in enumerate(((1.4, 0.55), (0.0, None), (1.3, -0.6), (0.0, None))):          # Tom ahead left, Mia ahead right (both in the camera's view); Emma and the guest where the camera cannot see
                if ang is None: pxy = hidden(s, avoid, 1.7 if i == 1 else 2.0)
                else: pxy = polar(s, d_, ang, avoid)
                if pxy is None: pxy = polar(s, 1.5, 2.0, avoid)
                px, py = pxy; avoid.append((px, py)); P[i] = (px, py)
                ppl.people.scripts[i].steps.extend([('at', px, py, math.atan2(ry - py, rx - px)), ('face_robot',)])
        def restore(s):
            p = s.a.pose(); q = box['pose0']
            if math.hypot(p[0] - q[0], p[1] - q[1]) > 0.25 or abs(AS.wrap(p[2] - q[2])) > 0.3: s.go_to(q[0], q[1], th=q[2], tol=0.15); box['restored'] = s.r.t
        def look_back(label):
            d.add(label + ' (the robot looks back where it was)', d.after(0.5), restore)
            d.add('wait for the tracks and the faces', lambda s: s.r.t >= box.get('restored', -99) + 7.0 or box.get('restored', -99) < 0, lambda s: box.pop('restored', None))
        d.add('switch the television on (nobody else is in the room)', d.after(1.0), tv_on)
        d.add('listen to the television (1)', d.after(2.0), lambda s: front.listen_tv(4.0))
        d.add('listen to the television (1b)', d.after(5.0), lambda s: front.listen_tv(4.0))
        def hop(s):
            """Go to the free spot farthest from where the robot has been (a long baseline is what pins a far television down; a metre of parallax at 3 m is five degrees)."""
            from episode import _furn_dist
            been = box.setdefault('been', [s.r.true_pose()[:2]]); best = None
            for gx in np.arange(0.6, 3.5, 0.2):
                for gy in np.arange(0.6, 2.5, 0.2):
                    if _furn_dist(gx, gy) <= 0.3: continue
                    dm = min(math.hypot(gx - p[0], gy - p[1]) for p in been)
                    if best is None or dm > best[0]: best = (dm, float(gx), float(gy))
            been.append((best[1], best[2])); s.go_to(best[1], best[2], tol=0.25)
        d.add('go to another spot', d.after(4.0), hop)
        d.add('listen to the television (2)', d.after(22.0), lambda s: front.listen_tv(4.0))
        d.add('go to a third spot', d.after(4.0), hop)
        d.add('listen to the television (3)', d.after(22.0), lambda s: front.listen_tv(4.0))
        d.add('go to a fourth spot', d.after(4.0), hop)
        d.add('listen to the television (4)', d.after(22.0), lambda s: front.listen_tv(4.0))
        d.add('the family comes in (placed where the robot now stands)', d.after(1.0), place)
        d.add('wait for the tracks and the faces', d.after(9.0), lambda s: None)
        d.add('check the television', d.after(0.2), lambda s: chk('the television is found from its sound alone', bool(asc.devices) and min(math.hypot(dv['xy'][0] - box['tv_true'][0], dv['xy'][1] - box['tv_true'][1]) for dv in asc.devices) < 1.0,
                                                                 (asc.devices, box['tv_true'], len(asc.rays))))
        # the family speaks
        d.add('Tom (zh, in view): who is here', d.after(3.0), say(0, '小拾小拾，现在有谁', 'Tom, in view, Chinese', 'Tom'))
        d.add('check Tom', d.after(0.2), lambda s: chk('Tom, in view, is recognised as a person with the owner role and answered in Chinese', L['ident']['source'] == 'person' and L['ident']['role'] == 'owner' and HAN(L['rec'].get('reply')),
                                                        (L['ident']['source'], L['ident']['name'], L['ident']['role'], L['ident']['posterior'], L['rec'].get('action'), L['rec'].get('reply'))))
        d.add('Emma (en, out of view): where is my teddy bear', d.after(4.0), say(1, 'Hey Xiaoshi, where is my teddy bear', 'Emma, out of view, English', 'Emma'))
        d.add('check Emma', d.after(0.2), lambda s: chk('Emma, out of view, is answered in English and is never given the owner role', not HAN(L['rec'].get('reply')) and L['ident']['role'] != 'owner' and L['ident']['source'] in ('unseen', 'ambiguous'), (L['ident']['source'], L['ident']['name'], L['ident']['role'], L['rec'].get('reply'))))
        d.add('television: "Hey Xiaoshi, scan the house"', d.after(4.0), say(None, 'Hey Xiaoshi, scan the house', 'the television', 'device', from_tv=True))
        d.add('check the television', d.after(0.2), lambda s: chk('the television saying the robot\'s name does not start a scan', s.mode != 'SCAN' and not any(a[0] == 'scan_home' for a in vs.actions), (L['ident']['source'], L['rec'].get('action'), s.mode)))
        d.add('Emma (voice only): forget Mia', d.after(4.0), say(1, 'Hey Xiaoshi, forget Mia', 'Emma, out of view, owner-level request', 'Emma'))
        d.add('check Emma cannot forget', d.after(0.2), lambda s: chk('a voice alone (Emma out of view) cannot make the robot forget somebody', 'Mia' in reg.people and 'Mia' in vp.people and not any(a[0] == 'forget' for a in vs.actions), (L['rec'].get('action'), L['rec'].get('reply'))))
        d.add('stranger (out of view): go and charge', d.after(4.0), say(3, 'Hey Xiaoshi, go and charge', 'a stranger, out of view', 'guest'))
        d.add('check the stranger', d.after(0.2), lambda s: chk('a stranger out of view cannot send the robot to charge', s.mode != 'CHARGE' and not any(a[0] == 'charge' for a in vs.actions), (L['ident']['source'], L['ident']['role'], L['rec'].get('action'), L['rec'].get('reply'))))
        look_back('after the stranger')
        look_back('before Mia')
        d.add('Mia (zh, in view): where is the ball', d.after(2.0), say(2, '小拾小拾，小球在哪里', 'Mia, in view, Chinese', 'Mia'))
        d.add('check Mia', d.after(0.2), lambda s: chk('a child in view is given the child role and is answered', L['ident']['role'] in ('child', 'unknown') and bool(L['rec'].get('reply')) and L['rec'].get('action') != 'denied', (L['ident']['role'], L['rec'].get('action'), L['rec'].get('reply'))))
        look_back('before Tom (mixed)')
        d.add('Tom (mixed): put the ball in the basket', d.after(2.0), say(0, 'Hey Xiaoshi, 把 ball 放进筐里', 'Tom, in view, mixed Chinese and English', 'Tom'))
        look_back('before the recording')
        d.add('recording of Tom played by the television', d.after(2.0), say(None, 'Hey Xiaoshi, forget everyone', 'the television plays a recording of Tom', 'device', from_tv=True, tv_voice_of=0))
        d.add('check the replay', d.after(0.2), lambda s: chk('a recording of the owner played by the television cannot make the robot forget the family', 'Tom' in reg.people and 'Mia' in reg.people and len(vp.people) == 3 and not any(a[0] == 'forget' for a in vs.actions),
                                                             (L['ident']['source'], L['ident']['name'], L['ident']['role'], L['rec'].get('action'))))
        look_back('before the profile')
        d.add('Tom: what do you know about Mia (English)', d.after(2.0), say(0, 'Hey Xiaoshi, what do you know about Mia', 'Tom, in view, asks for a profile', 'Tom'))
        d.add('check the card', d.after(0.2), lambda s: chk('Tom (owner, in view) is shown the profile of the child, in English, with only what is kept', L['rec'].get('intent') == 'profile_show' and 'child' in str(L['rec'].get('reply')) and 'only the name' in str(L['rec'].get('reply')),
                                                           (L['rec'].get('action'), str(L['rec'].get('reply'))[:200])))
        look_back('before come here')
        d.add('Tom: come here', d.after(2.0), say(0, 'Hey Xiaoshi, come here', 'Tom calls the robot over', 'Tom'))
        d.add('check come here', d.after(0.5), lambda s: chk('"come here" from the owner in view is understood and the robot heads for him', L['rec'].get('intent') == 'come_here' and L['rec'].get('action') in ('come_here', 'confirm') and bool(L['rec'].get('reply')),
                                                              (L['rec'].get('intent'), L['rec'].get('action'), L['rec'].get('reply'), s.mode)))
        look_back('before the conversation')
        d.add('Tom speaks, then Emma joins in without the wake word', d.after(2.0), say(0, 'Hey Xiaoshi, what is on the floor', 'Tom opens a conversation', 'Tom'))
        d.add('Emma joins in', d.after(2.0), say(1, 'what is new today', 'Emma speaks without the wake word, in the same conversation', 'Emma'))
        d.add('check the follow-up', d.after(0.2), lambda s: chk('a follow-up from somebody else is not taken as part of Tom\'s conversation', str(L['rec'].get('action', '')).startswith('ignored'), (L['rec'].get('action'), L['ident']['name'])))
        look_back('before the job')
        d.add('Tom: tidy up', d.after(2.0), say(0, 'Hey Xiaoshi, tidy up the toys', 'Tom starts the job, in English', 'Tom'))
        d.add('check the job', d.after(2.0), lambda s: chk('Tom (owner, in view) starts the tidy-up job in English', s.mode == 'TASK' and not HAN(L['rec'].get('reply')), (s.mode, L['rec'].get('action'), L['rec'].get('reply'))))
        d.add('Tom: stop (English) while the wheels turn', d.after(6.0), say(0, 'stop', 'Tom, the English stop word, robot driving', 'Tom'))
        d.add('check stop', d.after(2.0), lambda s: chk('the English stop word stops the robot while it drives', s.mode == 'IDLE' or L['rec'].get('action') == 'stop', (s.mode, L['rec'].get('action'))))
        d.add('Tom: speak Chinese', d.after(3.0), say(0, 'Hey Xiaoshi, speak English', 'Tom asks for English', 'Tom'))
        d.add('end', d.after(2.0), lambda s: chk('the language Tom asked for is kept in his profile', pr.people.get('Tom', {}).get('lang_set') == 'en' or L['rec'].get('action') != 'set_lang', (pr.people.get('Tom', {}).get('lang_set'), L['rec'].get('action'))))

    t0 = time.time(); res, sup = run_supervised_day(seed, pe, setup, box)
    d = box['d']; vs = box['vs']; pr = box['pr']; asc = box['asc']; ppl = sup.ppl
    out = dict(seed=seed, sim_s=round(sup.r.t), wall_s=round(time.time() - t0), checks=checks, passed=sum(c['ok'] for c in checks), n_checks=len(checks), events=table, finished_script=d.done, devices=asc.devices, tv_true=box.get('tv_true'),
               collisions=res['collisions'], stuck=res['stuck'], contacts=[int(v) for v in ppl.people.contact_events], voice_stats=vs.stats, tts=vs.tts, actions=[a for a in vs.actions],
               profile_cards={n: dict(en=pr.card(n, 'en'), zh=pr.card(n, 'zh')) for n in pr.people}, rays=[tuple(round(v, 2) for v in r[:3]) for r in asc.rays], front_events=box['front'].n_events, registry=box['reg'].report(), voiceprints=box['vp'].report(), timeline=d.timeline)
    return out


if __name__ == '__main__':
    seeds = [int(v) for v in sys.argv[1:]] or [5]; r = _day2_job(seeds[0])
    print(json.dumps({k: v for k, v in r.items() if k not in ('timeline', 'tts', 'actions', 'checks', 'events', 'profile_cards')}, ensure_ascii=False, default=float))
    for c in r['checks']: print('  ', 'PASS' if c['ok'] else 'FAIL', c['name'], '' if c['ok'] else c['detail'])
    for e in r['events']: print('  ', e['label'], '|', e['source'], e['name'], e['role'], e['posterior'], '|', e['voice'], e['voice_z'], '|', e['action'], '|', str(e['reply'])[:70], '| doa err', e['direction_error_deg'], 'true bearing', e['true_bearing_deg'], 'tracks', e['tracks'], 'hyp', e['hypotheses'])
    others = []
    for sd in seeds[1:]:                                                                                          # the same script with other random voices, television spots and perceptions
        o = _day2_job(sd); others.append(dict(seed=sd, passed=o['passed'], n_checks=o['n_checks'], failed=[c['name'] + ' ' + str(c['detail'])[:160] for c in o['checks'] if not c['ok']], collisions=o['collisions'], contacts=sum(o['contacts']), sim_s=o['sim_s']))
        print('seed', sd, o['passed'], '/', o['n_checks'], others[-1]['failed'], flush=True)
    r['other_seeds'] = others
    os.makedirs(RES, exist_ok=True); json.dump(r, open(os.path.join(RES, os.environ.get('DAY2_OUT', 'sound_day2.json')), 'w'), ensure_ascii=False, indent=1, default=float)
