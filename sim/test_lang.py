"""Benches for talking to the robot in Chinese, English and both at once (text level: what the simulated recogniser writes down, what the robot makes of it).

  python test_lang.py lang       commands in three sets of each language (dev / holdout / test) through the simulated recogniser at several error rates and accents; a recogniser of one language only;
                                 wake words and stop words in both languages; speech that is not for the robot; the language of the reply; multi-turn dialogues in English
Writes results/lang.json. The sound itself (the microphones, the room, the television) is test_sound.py; here the recogniser is SimASR, which is a stand-in: how well a real one does with a Mandarin-accented
child in a living room is not something this can say.
"""
import json
import multiprocessing as mp
import os
import re
import sys
import time

import voice_data as VD
import voice_data_en as VE
from test_home import _outcome

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
PERSONS = ['Tom', 'Emma', 'Mia', '小明', '小红']
WAKES = dict(zh=['小拾小拾，', '你好小拾，'], en=['Hey Xiaoshi, ', 'Xiaoshi Xiaoshi, ', 'Hi Xiaoshi, ', 'Hello Xiaoshi, '])
_HAN = re.compile(r'[一-鿿]')
SETS = {'zh': VD.COMMANDS, 'en_dev': VE.COMMANDS_EN_DEV, 'en_holdout': VE.COMMANDS_EN_HOLDOUT, 'en_test': VE.COMMANDS_EN_TEST, 'mix_dev': VE.COMMANDS_MIX_DEV, 'mix_holdout': VE.COMMANDS_MIX_HOLDOUT, 'mix_test': VE.COMMANDS_MIX_TEST}


def _reply_lang(text):
    return 'zh' if _HAN.search(text or '') else 'en'


def _matrix(text):
    han = len(_HAN.findall(text)) / 1.7; en = len(re.findall(r'[A-Za-z]{2,}', text)); return 'zh' if han >= en else 'en'


def sweep(args):
    """One set of commands, one recogniser setting. -> counts"""
    from voice import MOTION, SimASR, VoiceSystem
    setname, cer, accent, mode, seeds = args
    cmds = [c for c in SETS[setname] if c[1] != 'stop']
    cnt = dict(ok=0, confirmed=0, reask=0, wrong=0, wrong_motion=0, missed_wake=0, n=0, reply_lang_ok=0, reply_n=0); by = {}
    for ci, (text, intent, slots) in enumerate(cmds):
        for sd in range(seeds):
            vs = VoiceSystem(asr=SimASR(cer, 1000 * sd + ci, mode=mode, accent=accent)); vs.extra_persons = PERSONS
            zhm = setname == 'zh' or (setname.startswith('mix') and _matrix(text) == 'zh')
            wake = (WAKES['zh'] if zhm else WAKES['en'])[sd % (2 if zhm else 4)]
            rec = vs.hear(wake + text, role='owner'); o = _outcome(vs, rec, intent, slots)
            if o == 'wrong' and rec.get('intent') in MOTION and rec.get('action') == rec.get('intent'): cnt['wrong_motion'] += 1
            cnt[o] += 1; cnt['n'] += 1; b = by.setdefault(intent, dict(n=0, ok=0)); b['n'] += 1; b['ok'] += int(o in ('ok', 'confirmed'))
            if rec.get('reply') and o in ('ok', 'confirmed') and not setname.startswith('mix'):                       # a mixed sentence may be answered in either language
                cnt['reply_n'] += 1; cnt['reply_lang_ok'] += int(_reply_lang(rec['reply']) == ('zh' if zhm else 'en'))
    return dict(set=setname, cer=cer, accent=accent, mode=mode, **cnt, by_intent=by)


def _rates(r):
    n = r['n']; return dict(understood=round((r['ok'] + r['confirmed']) / n, 3), ok=round(r['ok'] / n, 3), confirmed=round(r['confirmed'] / n, 3), reask=round(r['reask'] / n, 3), wrong=round(r['wrong'] / n, 3),
                            wrong_motion=round(r['wrong_motion'] / n, 3), missed_wake=round(r['missed_wake'] / n, 3), reply_language=(round(r['reply_lang_ok'] / r['reply_n'], 3) if r['reply_n'] else None), n=n)


def wake_stop_chatter():
    from voice import NLU, SimASR, VoiceSystem
    out = {}
    # wake word caught
    wk = {}
    for lang, phrases in (('zh', ['小拾小拾，你好', '你好小拾，你是谁']), ('en', ['Hey Xiaoshi, hello', 'Xiaoshi Xiaoshi, who are you', 'Hi Xiaoshi, what can you do', 'Hello Xiaoshi, good morning']), ('mix', ['Hey Xiaoshi, 你好', '小拾小拾，hello'])):
        for accent in ((0.0,) if lang == 'zh' else (0.0, 1.0)):
            for cer in (0.0, 0.05, 0.10, 0.20):
                hit = 0; n = 0
                for sd in range(200):
                    vs = VoiceSystem(asr=SimASR(cer, sd, accent=accent)); rec = vs.hear(phrases[sd % len(phrases)], role='owner'); n += 1; hit += int(not (rec.get('action') or '').startswith('ignored'))
                wk[f'{lang}{"" if accent == 0.0 else " accented"} @cer {cer}'] = round(hit / n, 3)
    out['wake_word_caught'] = wk
    # stop words without the wake word
    st = {}
    zh_stop = ('停', '停下', '别动', '停下来', '暂停', '站住', '停止', '不要动'); en_stop = ('stop', 'stop it', 'freeze', 'wait', 'hold on', 'halt', 'stop now', 'stay', 'help', "don't move")
    for name, words, cer, child, accent in (('zh adult quiet', zh_stop, 0.04, False, 0.0), ('zh adult noisy', zh_stop, 0.15, False, 0.0), ('zh child noisy', zh_stop, 0.15, True, 0.0),
                                           ('en adult quiet', en_stop, 0.04, False, 0.0), ('en adult noisy', en_stop, 0.15, False, 0.0), ('en child noisy', en_stop, 0.15, True, 0.0),
                                           ('en accented adult quiet', en_stop, 0.04, False, 1.0), ('en accented adult noisy', en_stop, 0.15, False, 1.0)):
        hit = 0; n = 0
        for w in words:
            for sd in range(25):
                vs = VoiceSystem(asr=SimASR(cer, sd * 13 + len(w), accent=accent)); rec = vs.hear(w, role='child' if child else 'owner', dist=2.5); n += 1; hit += int(rec.get('action') == 'stop')
        st[name] = round(hit / n, 3)
    out['stop_words_detected'] = st
    # speech that is not for the robot, no wake word
    ch = {}
    for name, lines in (('zh', VD.CHATTER), ('en', VE.CHATTER_EN), ('mix', VE.CHATTER_MIX)):
        acted = 0; n = 0; stops = 0; wakes = 0
        for cer in (0.0, 0.10, 0.20):
            for text in lines:
                for sd in range(6):
                    vs = VoiceSystem(asr=SimASR(cer, sd * 37 + len(text), accent=0.5 if name != 'zh' else 0.0)); rec = vs.hear(text, role='owner'); n += 1
                    a = rec.get('action') or ''
                    if not a.startswith('ignored'): acted += 1
                    if a == 'stop': stops += 1
        ch[name] = dict(n=n, not_ignored=acted, false_stops=stops, false_wake_rate=round(acted / n, 4))
    out['chatter_no_wake'] = ch
    # a stop word hidden in a sentence ('bus stop') must not stop the robot, 'stop' alone must
    tricky = ['where is the bus stop', 'I will stop by the store', 'stop talking to your brother', 'wait for me by the door', 'we stop at the corner', 'do not stop the music', 'the clock stopped']
    fs = 0; n = 0
    for t in tricky:
        for sd in range(20):
            vs = VoiceSystem(asr=SimASR(0.0, sd)); rec = vs.hear(t, role='owner'); n += 1; fs += int(rec.get('action') == 'stop')
    out['stop_word_inside_a_sentence'] = dict(n=n, false_stops=fs)
    return out


def dialogues():
    from voice import SimASR, VoiceSystem
    res = []
    def new(**kw): return VoiceSystem(asr=SimASR(0.0, 1, cs_rate=0.0), **kw)
    def chk(name, ok, detail): res.append(dict(name=name, ok=bool(ok), detail=detail))
    vs = new(); a = vs.hear('Hey Xiaoshi', role='owner'); chk('English wake word alone: "I\'m here"', a.get('intent') == 'wake' and a.get('reply') == "I'm here.", a.get('reply'))
    vs = new(); vs.hear('Hey Xiaoshi, hello', role='owner'); vs.advance(5.0); a = vs.hear('who are you', role='owner'); vs.advance(20.0); b = vs.hear('who are you', role='owner')
    chk('English follow-up without the wake word, then not after the window', a.get('intent') == 'identity' and (b.get('action') or '').startswith('ignored'), (a.get('intent'), b.get('action')))
    vs = new(); a = vs.hear('Hey Xiaoshi, find it please', role='owner'); chk('asks which one (English)', a.get('action') == 'ask' and a.get('reply') == 'Which one do you mean?', a.get('reply'))
    vs = new(); a = vs.hear('Hey Xiaoshi, can you find', role='owner'); vs.advance(3.0); b = vs.hear('the red ball', role='owner')
    chk('missing item asked for in English, answered in English', a.get('action') == 'ask' and b.get('intent') == 'where_is' and vs.actions and vs.actions[-1][1].get('item') == 'ball', (a.get('reply'), b.get('action'), vs.actions[-1:]))
    vs = new(); vs.ctx['item'] = 7; a = vs.hear('Hey Xiaoshi, put it in the basket', role='owner')
    chk('"it" = the last item (English)', vs.actions and vs.actions[-1][0] == 'fetch_item' and vs.actions[-1][1].get('item') == 7, vs.actions[-1:])
    vs = new(); vs.pending = dict(kind='confirm', intent='go_to', slots=dict(place='沙发'), role='adult', until=vs.now() + 12); vs.advance(2.0); b = vs.hear('yes please', role='adult')
    chk('confirm then "yes please"', b.get('via') == 'confirmed' and vs.actions[-1][0] == 'go_to', (b.get('action'), vs.actions[-1:]))
    vs = new(); vs.pending = dict(kind='confirm', intent='go_to', slots=dict(place='沙发'), role='adult', until=vs.now() + 12); a = vs.hear('no thanks', role='adult')
    chk('confirm then "no thanks"', a.get('action') == 'cancelled' and not vs.actions, (a.get('action'), vs.actions))
    vs = new(); vs.pending = dict(kind='confirm', intent='scan_home', slots={}, role='adult', until=vs.now() + 12); a = vs.hear('yes', role='child')
    chk('a child cannot confirm a scan (English)', a.get('action') == 'denied' and a.get('reply') == 'That one needs a grown-up to ask.', (a.get('action'), a.get('reply')))
    vs = new(); a = vs.hear('Hey Xiaoshi, mute', role='adult'); b = vs.hear('Hey Xiaoshi, hello', role='owner')
    chk('mute in English works and sticks', vs.muted and (b.get('action') or '').startswith('ignored'), (a.get('action'), b.get('action')))
    vs = new(); vs.extra_persons = ['Tom']; a = vs.hear('Hey Xiaoshi, forget Tom', role='owner'); vs.advance(2.0); b = vs.hear('yes', role='owner')
    chk('forget Tom: asks, then forgets', a.get('action') == 'confirm' and b.get('action') == 'forgot' and ('forget', 'Tom') in vs.actions, (a.get('reply'), b.get('reply')))
    vs = new(); vs.extra_persons = ['Tom']; a = vs.hear('Hey Xiaoshi, forget Tom', role='child'); chk('a child cannot make it forget (English)', a.get('action') == 'denied', a.get('reply'))
    vs = new(); a = vs.hear('Hey Xiaoshi, hello', role='owner'); b = vs.hear('小拾小拾，你好', role='owner'); c = vs.hear('Hey Xiaoshi, 你好', role='owner')
    chk('the reply follows the language of the question (en, zh, mixed)', a['reply'].startswith('Hello') and b['reply'].startswith('你好') and _HAN.search(c['reply']) is not None, (a['reply'], b['reply'], c['reply']))
    vs = new(); a = vs.hear('Hey Xiaoshi, speak Chinese', role='owner'); b = vs.hear('你好', role='owner'); chk('"speak Chinese" is answered in Chinese', _HAN.search(a['reply'] or '') is not None, (a.get('reply'), b.get('reply')))
    vs = new(); a = vs.hear('Hey Xiaoshi, put the 袜子 in the basket', role='owner'); b = vs.hear('小拾小拾，把 ball 放进筐里', role='owner')
    chk('code-switching in both directions', a.get('intent') == 'fetch_item' and a['slots'].get('item') == 'sock' and b.get('intent') == 'fetch_item' and b['slots'].get('item') == 'ball', (a.get('slots'), b.get('slots')))
    vs = new(); a = vs.hear('Hey Xiaoshi, enrol', role='owner'); vs2 = new(); vs2.extra_persons = ['小明']; x = vs2.hear('Hey Xiaoshi, where is xiao ming', role='owner')
    chk('a Chinese name spelled in pinyin is found in an English sentence', x['slots'].get('person') == '小明', x.get('slots'))
    vs = new(); a = vs.hear('Hey Xiaoshi, remember this child, his name is Tom', role='adult'); chk('enrolment needs the owner (English)', a.get('action') == 'denied', (a.get('action'), a.get('reply')))
    from voice import MockLLM
    llm = MockLLM({'messy': dict(intent='tidy'), 'story': dict(say='Once upon a time there was a little robot that tidied up every day.')})
    vs = new(llm=llm); a = vs.hear('Hey Xiaoshi, this room is so messy', role='owner'); b = vs.hear('yes', role='owner')
    chk('English: a language-model proposal still has to be confirmed', a.get('action') == 'confirm' and a.get('via') == 'llm' and b.get('via') == 'confirmed' and vs.actions[-1][0] == 'tidy', (a.get('action'), a.get('reply'), b.get('via')))
    vs = new(llm=llm); a = vs.hear('Hey Xiaoshi, tell me a story', role='child'); chk('English: free text is not spoken to a child', a.get('action') == 'reask', a.get('action'))
    return res


def examples():
    from voice import SimASR, VoiceSystem
    rows = []
    for said, accent in (('Hey Xiaoshi, put the red ball in the basket', 0.5), ('小拾小拾，把红色的小球放进筐里', 0.0), ('Hey Xiaoshi, 把 ball 放进筐里', 0.5), ("Hey Xiaoshi, where's my teddy bear", 1.0), ('小拾小拾，go to the 沙发', 0.0),
                         ('Xiaoshi Xiaoshi, what can you do', 0.0), ('Hey Xiaoshi, scan the house', 0.5), ('Hey Xiaoshi, 你好', 0.0), ('Hey Xiaoshi, speak Chinese', 0.0), ('stop', 0.0), ('where is the bus stop', 0.0),
                         ('Hey Xiaoshi, are you recording', 0.0)):
        vs = VoiceSystem(asr=SimASR(0.05, len(said), accent=accent), keep_transcripts=True); vs.extra_persons = PERSONS; r = vs.hear(said, role='owner')
        rows.append(dict(said=said, heard=r.get('heard'), intent=r.get('intent'), slots=r.get('slots'), action=r.get('action'), reply=r.get('reply')))
    return rows


def bench_lang(procs=3):
    t0 = time.time(); res = {}
    jobs = []
    for sname in SETS:
        for cer in (0.0, 0.04, 0.08, 0.15, 0.25): jobs.append((sname, cer, 0.0, 'bilingual', 6))
    for sname in ('en_test', 'mix_test'):
        for accent in (0.5, 1.0):
            for cer in (0.04, 0.08): jobs.append((sname, cer, accent, 'bilingual', 6))
    for sname in ('en_test', 'mix_test'): jobs.append((sname, 0.04, 0.0, 'zh_only', 6))                       # a robot that only speaks Chinese
    jobs.append(('zh', 0.04, 0.0, 'en_only', 6)); jobs.append(('zh', 0.04, 0.0, 'bilingual', 6))
    with mp.Pool(procs) as p: rows = p.map(sweep, jobs)
    table = []
    for r in rows:
        t = dict(set=r['set'], cer=r['cer'], accent=r['accent'], mode=r['mode'], **_rates(r)); table.append(t); print(t)
    res['sweep'] = table; res['by_intent_test_sets'] = {f'{r["set"]} @cer {r["cer"]}': r['by_intent'] for r in rows if r['cer'] == 0.04 and r['accent'] == 0.0 and r['mode'] == 'bilingual'}
    res['noise'] = wake_stop_chatter(); print(json.dumps(res['noise'], ensure_ascii=False))
    res['dialogues'] = dialogues()
    for d in res['dialogues']: print('  ', 'PASS' if d['ok'] else 'FAIL', d['name'], '' if d['ok'] else d['detail'])
    res['examples'] = examples()
    for e in res['examples']: print(e)
    from voice import NLU
    n = NLU(); t1 = time.time()
    for i in range(30): n.parse('Hey put the red ball in the basket')
    res['ms_per_parse_en'] = round(1000 * (time.time() - t1) / 30, 1); t1 = time.time()
    for i in range(30): n.parse('把小球放进筐里')
    res['ms_per_parse_zh'] = round(1000 * (time.time() - t1) / 30, 1); print('ms per parse en/zh', res['ms_per_parse_en'], res['ms_per_parse_zh'], 'total', round(time.time() - t0), 's')
    os.makedirs(RES, exist_ok=True); json.dump(res, open(os.path.join(RES, 'lang.json'), 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'lang'; procs = int(os.environ.get('PROCS', '3'))
    if what == 'lang': bench_lang(procs=procs)
