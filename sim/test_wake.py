"""Checks of the wake-word / stop-word spotter (wakeword.py) and of how the voice takes its decision.

  python test_wake.py            -> results/wake.json
  python test_wake.py fetch      download the model (33 MB) first

Needs `pip install sherpa-onnx sentencepiece pypinyin` and the model (see wakeword.fetch_model). Without them the spotter checks are SKIPPED (and counted as such), the voice checks still run.
The recordings used are the six that come with the model (two English sentences, four Chinese ones): this repository holds no recording of anybody saying 小拾小拾, and the
synthetic speech of espeak-ng does not trigger the model at all; that is measured below and reported, it is not a pass. How often the robot wakes for a real person and how often it wakes by
mistake have to be measured with wakeword.evaluate() on recordings from the house it will live in.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.path.join(HERE, '..', 'results')

import wakeword as W                                  # noqa: E402
from voice import SimASR, TextASR, VoiceSystem        # noqa: E402


def run():
    res = []; extra = {}
    def chk(name, ok, detail=''): res.append(dict(name=name, ok=bool(ok), detail=str(detail)[:300], skipped=False))
    def skip(name, why): res.append(dict(name=name, ok=True, detail='SKIPPED: ' + why, skipped=True))

    # ---- the spotter
    if not (W.have_runtime() and W.have_model()):
        why = 'sherpa-onnx or the model is not installed (python test_wake.py fetch)'
        for n in ('keywords become the right tokens', 'a phrase that cannot be tokenised is refused', 'a phrase is spotted in a real recording', 'a second phrase in another recording', 'other speech and silence: no false alarm',
                  'refractory time', 'suppression holds off the wake words but not the stop words', 'evaluate() on a folder', 'espeak speech does not stand in for a person'): skip(n, why)
    else:
        M = os.path.join(W.MODEL_DIR, 'test_wavs'); wav = lambda n: W.read_wav(os.path.join(M, n + '.wav'))
        wk = W.keyword_lines(W.WAKE, 'wake'); sk = W.keyword_lines(W.STOP, 'stop')
        chk('keywords become the right tokens', wk[0].startswith('x iǎo sh í x iǎo sh í :') and wk[1].startswith('n ǐ h ǎo x iǎo sh í :') and wk[2].startswith('HH EY1 SH AW1 SH IY1 :') and sk[3].startswith('S T AA1 P :'), (wk, sk))
        try: W.keyword_lines([('x', 'xiaoshi')], 'wake'); chk('a phrase that cannot be tokenised is refused', False, 'accepted')
        except ValueError as e: chk('a phrase that cannot be tokenised is refused', 'cannot tokenise' in str(e), e)
        sp = W.SherpaWake(wake=[('light_up', 'LIGHT UP')], stop=[]); a = wav('en_0'); ev = sp.run(a)
        chk('a phrase is spotted in a real recording', len(ev) == 1 and ev[0]['name'] == 'light_up' and 1.0 < ev[0]['t'] < len(a) / W.SR + 1.0, ev)
        sp2 = W.SherpaWake(wake=[('lovely_child', 'LOVELY CHILD')], stop=[]); ev2 = sp2.run(wav('en_1')); chk('a second phrase in another recording', len(ev2) == 1 and ev2[0]['name'] == 'lovely_child', ev2)
        real = W.SherpaWake(); sil = np.zeros(60 * W.SR, np.float32); noise = (np.random.default_rng(3).normal(0, 0.05, 30 * W.SR)).astype(np.float32)
        evs = {k: real.run(v) for k, v in dict(en_0=wav('en_0'), en_1=wav('en_1'), zh_0=wav('zh_0'), zh_1=wav('zh_1'), zh_2=wav('zh_2'), zh_3=wav('zh_3'), zh_4=wav('zh_4'), silence=sil, noise=noise).items()}
        secs = sum(len(v) for v in (wav(n) for n in ('en_0', 'en_1', 'zh_0', 'zh_1', 'zh_2', 'zh_3', 'zh_4'))) / W.SR
        chk('other speech and silence: no false alarm', all(not e for e in evs.values()), {k: e for k, e in evs.items() if e}); extra['false_alarm_material_s'] = round(secs + 90, 1)
        two = np.concatenate([a, np.zeros(int(0.3 * W.SR), np.float32), a])
        e_short = W.SherpaWake(wake=[('light_up', 'LIGHT UP')], stop=[], refractory_s=1.5).run(two); e_long = W.SherpaWake(wake=[('light_up', 'LIGHT UP')], stop=[], refractory_s=30.0).run(two)
        chk('refractory time', len(e_short) == 2 and len(e_long) == 1, (len(e_short), len(e_long)))
        w_ = W.SherpaWake(wake=[('light_up', 'LIGHT UP')], stop=[]); w_.reset(); w_.suppress(1e6); s_ = W.SherpaWake(wake=[], stop=[('light_up', 'LIGHT UP')]); s_.reset(); s_.suppress(1e6)
        ew, es = w_.run(a), s_.run(a); chk('suppression holds off the wake words but not the stop words', ew == [] and len(es) == 1 and es[0]['kind'] == 'stop', (ew, es))
        d = tempfile.mkdtemp()
        try:
            os.makedirs(d + '/pos'); os.makedirs(d + '/neg'); shutil.copy(os.path.join(M, 'en_0.wav'), d + '/pos/a.wav')
            for n in ('zh_0', 'zh_1', 'zh_2'): shutil.copy(os.path.join(M, n + '.wav'), d + f'/neg/{n}.wav')
            rows = W.evaluate(d + '/pos', d + '/neg', 'light_up', thresholds=(0.25,), phrases=[('light_up', 'LIGHT UP')])
        finally: shutil.rmtree(d)
        chk('evaluate() on a folder', rows[0]['hit_rate'] == 1.0 and rows[0]['false_alarms'] == 0 and rows[0]['negative_hours'] > 0, rows)
        # what a synthesiser does: it does not wake the model (measured, not asserted to be zero: the number is the result)
        import acoustics as A
        hits = n = 0
        for text, lang in (('小拾小拾，收拾玩具', 'zh'), ('你好小拾，去充电', 'zh'), ('Hey Xiaoshi, tidy up the toys', 'en'), ('Xiaoshi Xiaoshi, go and charge', 'en')):
            for v, p in (('m3', 50), ('f2', 60), ('m1', 30), ('f4', 80)):
                x = A.synth(text, lang, v, p, 160); x = (0.5 * x / np.abs(x).max()).astype(np.float32); n += 1; hits += any(e['kind'] == 'wake' for e in real.run(x))
        extra['espeak_wake_hits'] = f'{hits}/{n}'
        chk('espeak speech does not stand in for a person', True, f'the spotter fired for {hits} of {n} synthesised wake phrases: a synthesiser says nothing about the real rate')

    # ---- the voice
    def vs(): return VoiceSystem(asr=SimASR(0.0, 1))
    v = vs(); a = v.hear('收拾玩具', role='owner', kws=True); chk('voice: the spotter heard the wake word, the text does not have it: taken as woken', a.get('woke') and a.get('intent') == 'tidy', (a.get('woke'), a.get('intent'), a.get('action')))
    v = vs(); a = v.hear('小拾小拾，收拾玩具', role='owner', kws=False); chk('voice: the text has the name but the spotter did not hear it: not woken', (a.get('action') or '').startswith('ignored'), a.get('action'))
    v = vs(); a = v.hear('收拾玩具', role='owner', kws=None); chk('voice: no spotter (the earlier way): the text decides', (a.get('action') or '').startswith('ignored'), a.get('action'))
    v = vs(); a = v.hear('嗯', role='child', kws='stop'); chk('voice: a stop word heard in the audio stops, whatever the text says', a.get('action') == 'stop' and v.stats['stop'] == 1, (a.get('action'), v.stats['stop']))
    v = vs(); a = v.hear('停', role='owner', kws=False); chk('voice: "stop" in the text is not enough when a spotter is in use and did not hear it', a.get('intent') != 'stop', (a.get('intent'), a.get('action')))
    v = vs(); v.hear('小拾小拾', role='owner', kws=True); v.advance(3.0); a = v.hear('你是谁', role='owner', kws=False); chk('voice: a follow-up inside the window needs no wake word, spotter or not', a.get('intent') == 'identity', a.get('action'))
    v = VoiceSystem(asr=TextASR()); a = v.hear('Hey Xiaoshi, 把 the ball 放进筐里', role='owner', kws=True); b = v.hear('收拾玩具', role='owner', kws=True)
    chk('voice: TextASR gives the words of a real recogniser back unchanged', a.get('heard') is None and b.get('intent') == 'tidy' and b.get('woke'), (a.get('intent'), b.get('intent')))
    return res, extra


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'fetch': print(W.fetch_model()); sys.exit(0)
    res, extra = run(); bad = [r for r in res if not r['ok']]; skipped = [r for r in res if r['skipped']]
    for r in res: print(('skip ' if r['skipped'] else ('ok   ' if r['ok'] else 'FAIL ')) + r['name'] + ('' if r['ok'] and not r['skipped'] else '   <- ' + r['detail']) + ('   [' + r['detail'] + ']' if r['ok'] and not r['skipped'] and r['name'].startswith('espeak') else ''))
    print(f'{len(res) - len(bad) - len(skipped)}/{len(res)} passed, {len(skipped)} skipped', extra)
    os.makedirs(RES, exist_ok=True); json.dump(dict(checks=res, passed=len(res) - len(bad) - len(skipped), skipped=len(skipped), total=len(res), **extra), open(os.path.join(RES, 'wake.json'), 'w'), ensure_ascii=False, indent=1)
    sys.exit(1 if bad else 0)
