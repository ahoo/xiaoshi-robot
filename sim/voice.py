"""Talking to the robot in Chinese: wake word, speech recogniser (simulated), understanding, dialogue, who may ask what, and doing it.

Where the simulator ends. It has no sound, so the speech recogniser (ASR) is replaced by SimASR: it takes the words a person means to say and returns what a recogniser would
write down, with the kinds of mistakes Mandarin recognisers make (a homophone, a character with the wrong tone, zh/z, n/l, ang/an confusions, a dropped or added syllable, a worse
error rate for children and for far-away speakers) and a confidence. Everything after that is real code: the wake word and the always-on stop words, the understanding (intent and
slots, matched on pinyin so that a wrong character that sounds right still works), the dialogue (asking when something is missing, confirming before it moves or stores
anything about a person, follow-ups without the wake word, 'it' and 'that one'), the permission model (a child may ask where the ball is and tidy up, may not enrol anybody or send the
robot across the house), and the executor that drives the robot (modes.Supervisor, the object records, the 3D map, the people tracker). What the simulator cannot say: how well a real
microphone array and a real recogniser do in a living room with a television on and a six-year-old talking, how the text-to-speech sounds, and what a language model (the optional
LLMBackend) makes of a free-form request. Those need the real parts and real people.

A language model is an optional extra and never in control: it may only propose one of the fixed intents (or a sentence to say), and a proposal goes through the same permission check,
confirmation and executor as anything else. Without it the robot still does everything above.

Privacy by design: the wake-word and stop-word spotting run on the robot; nothing is sent anywhere before the wake word; no audio is stored; the transcript of a command is kept in a local
log only if the owner switches that on; a hardware mute switch cuts the microphones; 'privacy' commands (mute, stop looking, delete) are owner-only to switch off, anyone to switch on.
"""
import math
import random
import re
import time

import numpy as np
from pypinyin import Style, lazy_pinyin

WAKE = '小拾小拾'            # four syllables on purpose: 小拾 alone sounds like 小时 (hour), which is in every other sentence
WAKE_ALT = '你好小拾'
SESSION_S = 8.0               # after the robot answers, a follow-up needs no wake word for this long
PENDING_S = 12.0              # an open question waits this long for its answer

# ---------------------------------------------------------------------------------------------------------------- pinyin and fuzzy matching
_INIT = (('zh', 'z'), ('ch', 'c'), ('sh', 's'))
_FIN = (('iang', 'ian'), ('uang', 'uan'), ('ang', 'an'), ('eng', 'en'), ('ing', 'in'))


def fz(syl):
    """A pinyin syllable (no tone) reduced to what is easily confused: zh/ch/sh -> z/c/s, n -> l, f -> h, ang/eng/ing -> an/en/in."""
    s = syl
    for a, b in _INIT:
        if s.startswith(a): s = b + s[len(a):]; break
    if len(s) > 1 and s[0] in 'nf' and s[1] in 'aeiouüv': s = ('l' if s[0] == 'n' else 'h') + s[1:]
    for a, b in _FIN:
        if s.endswith(a) and len(s) > len(a) - 1: s = s[:-len(a)] + b; break
    return s


def syl(text):
    """Toneless syllables of a text (non-Chinese characters stay as they are)."""
    return lazy_pinyin(text, style=Style.NORMAL, errors=lambda x: list(x))


def fkey(text): return [fz(s) for s in syl(text)]


def _sd(a, b):
    """Distance of two reduced syllables: 0 equal, 0.5 same initial or same final, 1 different."""
    if a == b: return 0.0
    ia, ib = re.match(r'[bpmfdtlnghjqxzcsryw]*', a).group(), re.match(r'[bpmfdtlnghjqxzcsryw]*', b).group()
    return 0.5 if (ia == ib and ia) or (a[len(ia):] == b[len(ib):] and a[len(ia):]) else 1.0


def match(pattern, text, strict=False):
    """Best approximate occurrence of the syllable list `pattern` in the syllable list `text`: (score 0..1, start, end). Substitution by _sd (strict: 0 or 1), insertion and deletion 1."""
    m, n = len(pattern), len(text)
    if m == 0 or n == 0: return 0.0, 0, 0
    D = np.zeros((m + 1, n + 1)); D[:, 0] = np.arange(m + 1); S = np.zeros((m + 1, n + 1), int)
    for j in range(n + 1): S[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c = (D[i - 1, j - 1] + ((0.0 if pattern[i - 1] == text[j - 1] else 1.0) if strict else _sd(pattern[i - 1], text[j - 1])), D[i - 1, j] + 1.0, D[i, j - 1] + 1.0)
            k = int(np.argmin(c)); D[i, j] = c[k]; S[i, j] = S[i - 1, j - 1] if k == 0 else (S[i - 1, j] if k == 1 else S[i, j - 1])
    j = int(np.argmin(D[m, 1:])) + 1
    return max(0.0, 1.0 - D[m, j] / m), int(S[m, j]), j


def clean(text):
    return re.sub(r'[\s，。！？、,.!?；;：:“”"\'（）()…~～—-]', '', text)


# ---------------------------------------------------------------------------------------------------------------- the simulated recogniser
COMMON = ('的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从'
          '动两长知民样现分将外但身些与高意进把法此实家定对小同关机力理水化问最重被等体点以其间做都话已度种次相正通面带位给让它哪去找拿放收拾整扫描地图球块车蕉积木石袜毛绒玩具筐沙发桌架椅栽充电座谁孩'
          '认识记住忘掉删除停止暂继续取消状态量干什么新旧少多哪里叫名字同意好吧嗯吗呢啊呀哦嘛吧快慢请帮忙一下上下左右前后边旁中间东西今天明天现在刚才昨天早晚饭觉玩笑故事歌曲唱')
_BY_PY = {}
for _c in COMMON:
    _p = lazy_pinyin(_c, style=Style.TONE3)[0]
    _BY_PY.setdefault(_p, []).append(_c)
_BY_PYN = {}
for _c in COMMON:
    _p = lazy_pinyin(_c, style=Style.NORMAL)[0]
    _BY_PYN.setdefault(_p, []).append(_c)
_BY_FZ = {}
for _c in COMMON:
    _p = fz(lazy_pinyin(_c, style=Style.NORMAL)[0])
    _BY_FZ.setdefault(_p, []).append(_c)
FILLERS = '呃嗯啊那个就是'


class SimASR:
    """Words meant -> words written down. `cer` is the character error rate (clean room, adult: ~0.03; a noisy room or a child: 0.10-0.25). Errors: 55 % a homophone,
    25 % the same sound with another tone, 20 % a near sound (zh/z, n/l, ang/an ...); in addition syllables are dropped (6 % of the errors) or a filler is added (5 %).
    A child's voice and a distance over 3 m raise the rate. The confidence it returns falls with the number of mistakes, with noise, as a real recogniser's does."""
    def __init__(self, cer=0.05, seed=0): self.cer = cer; self.rng = random.Random(seed)

    def hear(self, text, child=False, dist=1.5):
        r = self.rng; rate = self.cer * (1.8 if child else 1.0) * (1.0 + max(dist - 2.0, 0.0) * 0.35)
        out = []; errs = 0; n = 0
        for ch in text:
            if not ('一' <= ch <= '鿿'): out.append(ch); continue
            n += 1
            if r.random() >= rate: out.append(ch); continue
            errs += 1; u = r.random()
            if u < 0.06: continue                                   # dropped
            p = lazy_pinyin(ch, style=Style.TONE3)[0]; pn = lazy_pinyin(ch, style=Style.NORMAL)[0]; v = r.random(); cand = []
            if v < 0.55: cand = [c for c in _BY_PY.get(p, []) if c != ch]
            if not cand and v < 0.80: cand = [c for c in _BY_PYN.get(pn, []) if c != ch]
            if not cand: cand = [c for c in _BY_FZ.get(fz(pn), []) if c != ch]
            out.append(r.choice(cand) if cand else ch)
            if r.random() < 0.05: out.append(r.choice(FILLERS))
        conf = float(np.clip(0.97 - 2.2 * (errs / max(n, 1)) - abs(r.gauss(0, 0.02)), 0.05, 0.99)) if errs else float(np.clip(0.95 + r.gauss(0, 0.02), 0.6, 0.99))
        return ''.join(out), conf


# ---------------------------------------------------------------------------------------------------------------- what can be said
ITEM_WORDS = {'ball': ['小球', '球', '皮球'], 'cube': ['方块', '木块', '积木块'], 'car': ['玩具车', '小汽车', '小车'], 'banana': ['香蕉玩具', '香蕉'], 'lbrick': ['L形积木', '积木', '拐角积木'],
              'rock': ['石头', '小石头'], 'sock': ['袜子', '小袜子'], 'plush': ['毛绒玩具', '玩偶', '娃娃', '小熊']}
PLACE_WORDS = {'沙发': ['沙发'], '桌子': ['桌子', '茶几'], '架子': ['架子', '书架', '柜子'], '盆栽': ['盆栽', '花盆', '植物'], '椅子': ['椅子', '凳子'], '箱子': ['箱子', '纸箱'],
               '收纳筐': ['收纳筐', '筐子', '筐', '篮子', '收纳篮'], '充电座': ['充电座', '充电桩', '充电的地方', '基座']}
# (trigger phrase, weight): a phrase that says this is what the person wants
TRIG = {
    'stop': [('停下', 1.0), ('停下来', 1.0), ('别动', 1.0), ('不要动', 1.0), ('暂停', 1.0), ('停止', 1.0), ('站住', 1.0), ('等一下', 0.8), ('先别', 0.8)],
    'resume': [('继续', 1.0), ('接着', 0.9), ('接着干', 1.0), ('继续吧', 1.0), ('继续做', 1.0)],
    'cancel': [('算了', 1.0), ('取消', 1.0), ('不用了', 1.0), ('不要了', 1.0), ('不用', 0.8)],
    'tidy': [('收拾', 1.0), ('整理', 1.0), ('收一收', 1.0), ('收玩具', 1.0), ('收拾玩具', 1.0), ('整理房间', 1.0), ('把玩具收', 1.0), ('把地上的东西', 0.9), ('捡玩具', 1.0), ('打扫', 0.8), ('收起来', 0.7)],
    'fetch_item': [('放进筐', 1.0), ('放到筐', 1.0), ('放筐里', 1.0), ('捡起来', 1.0), ('捡一下', 1.0), ('收到筐', 1.0), ('收好', 0.9), ('拿到筐', 1.0), ('放回筐', 1.0), ('收起来', 0.7)],
    'where_is': [('在哪', 1.0), ('在哪里', 1.0), ('去哪了', 1.0), ('在什么地方', 1.0), ('哪去了', 1.0), ('找一下', 0.9), ('找找', 0.9), ('找一找', 0.9), ('帮我找', 1.0), ('在不在', 0.7)],
    'list_items': [('有什么东西', 1.0), ('有哪些东西', 1.0), ('有什么玩具', 1.0), ('地上有什么', 1.0), ('都有什么', 0.9), ('有哪些玩具', 1.0), ('清点', 1.0), ('盘点', 1.0), ('看到了什么', 0.8)],
    'what_new': [('有什么新', 1.0), ('多了什么', 1.0), ('新来了什么', 1.0), ('什么新东西', 1.0), ('少了什么', 1.0), ('什么不见了', 1.0), ('什么挪了', 1.0), ('动过', 0.9), ('被挪过', 1.0), ('挪过位置', 1.0), ('挪过', 0.9), ('有什么变化', 1.0), ('什么变了', 0.9)],
    'scan_home': [('扫描', 1.0), ('扫一扫', 1.0), ('建地图', 1.0), ('三维地图', 1.0), ('三d地图', 1.0), ('看看家里', 1.0), ('看一看家里', 1.0), ('把家里扫', 1.0), ('巡视', 0.9), ('建个地图', 1.0), ('画地图', 0.9), ('地图', 0.6)],
    'go_to': [('去', 0.8), ('到', 0.5), ('过去', 0.9), ('走到', 1.0), ('开到', 1.0), ('去一下', 0.9), ('去那边', 0.9), ('到那边', 0.9), ('过来', 0.4)],
    'charge': [('充电', 1.0), ('去充电', 1.0), ('回去充电', 1.0), ('回充', 1.0), ('回充电座', 1.0), ('吃点电', 0.9)],
    'who_here': [('谁在', 1.0), ('有谁', 1.0), ('谁在家', 1.0), ('看到谁', 1.0), ('有人吗', 1.0), ('有没有人', 1.0), ('有小孩吗', 1.0), ('孩子在哪', 1.0), ('小孩在哪', 1.0), ('看到小孩', 1.0), ('谁来了', 1.0)],
    'enrol': [('记住', 1.0), ('记一下', 1.0), ('认识一下', 1.0), ('认识', 0.8), ('这是我的', 0.9), ('以后认得', 1.0)],
    'forget': [('忘掉', 1.0), ('忘记', 1.0), ('删除', 1.0), ('删掉', 1.0), ('不要记', 0.9), ('清除', 1.0)],
    'name_item': [('叫做', 1.0), ('起名', 1.0), ('取名', 1.0), ('叫它', 1.0), ('它的名字', 1.0), ('这个叫', 1.0), ('起个名字', 1.0)],
    'status': [('在干什么', 1.0), ('在做什么', 1.0), ('什么状态', 1.0), ('电量', 1.0), ('还有多少电', 1.0), ('怎么样了', 0.8), ('忙什么', 1.0), ('干嘛呢', 1.0), ('在干嘛', 1.0)],
    'privacy': [('录音', 1.0), ('在听', 0.8), ('摄像头', 1.0), ('隐私', 1.0), ('静音', 1.0), ('别听', 1.0), ('关掉麦克风', 1.0), ('别看', 1.0), ('你在录', 1.0), ('一直听', 1.0), ('听着', 0.8), ('偷听', 1.0), ('录像', 1.0), ('拍照', 0.9)],
    'greet': [('你好', 1.0), ('早上好', 1.0), ('晚上好', 1.0), ('在吗', 1.0), ('下午好', 1.0)],
    'identity': [('你是谁', 1.0), ('你叫什么', 1.0), ('自我介绍', 1.0)],
    'help': [('你会什么', 1.0), ('你能做什么', 1.0), ('能干什么', 1.0), ('帮助', 1.0), ('怎么用', 1.0), ('你会做什么', 1.0)],
}
YES = ['好', '好的', '是', '是的', '对', '嗯', '可以', '同意', '行', '没错', '确认', '对的']
NO = ['不', '不是', '不对', '不要', '不行', '不同意', '错了', '不好', '没有']
NEEDS = {'where_is': ['item'], 'fetch_item': ['item'], 'go_to': ['place'], 'name_item': ['item', 'newname'], 'enrol': [], 'forget': []}
MOTION = {'tidy', 'fetch_item', 'go_to', 'scan_home', 'charge', 'resume'}
SENSITIVE = {'enrol', 'forget', 'privacy'}
# who may ask what: the lowest role that may. owner > adult (a grown-up the robot does not know) > child (also: unsure, nobody seen)
ROLE_RANK = {'owner': 3, 'adult': 2, 'child': 1, 'unknown': 1}
MIN_ROLE = {'stop': 'unknown', 'status': 'unknown', 'greet': 'unknown', 'identity': 'unknown', 'help': 'unknown', 'where_is': 'unknown', 'list_items': 'unknown', 'what_new': 'unknown',
            'who_here': 'unknown', 'cancel': 'unknown', 'name_item': 'child', 'tidy': 'child', 'fetch_item': 'child', 'resume': 'adult', 'scan_home': 'adult', 'go_to': 'adult', 'charge': 'adult',
            'enrol': 'owner', 'forget': 'owner', 'privacy': 'unknown'}


class NLU:
    """Utterance -> intent, slots, confidence. Matching is on pinyin (so a homophone is no mistake), phrases first, then the things named in the sentence (items, places, people)."""
    def __init__(self):
        self.trig = {k: [(fkey(p), p, w) for p, w in v] for k, v in TRIG.items()}
        self.items = {}; self.places = {k: [(fkey(w), w) for w in v] for k, v in PLACE_WORDS.items()}
        self.items_static = {k: [(fkey(w), w) for w in v] for k, v in ITEM_WORDS.items()}
        self.names = {}              # people and learned item names: label -> ('person'|'item', ref)
        self.wake = fkey(WAKE); self.wake_alt = fkey(WAKE_ALT)

    def set_names(self, persons=(), item_names=None):
        self.names = {n: ('person', n) for n in persons}
        for ref, n in (item_names or {}).items(): self.names[n] = ('item', ref)

    # ---- pieces
    @staticmethod
    def _thr(n): return 1.0 if n <= 2 else (0.8 if n == 3 else (0.74 if n == 4 else 0.7))

    def _wake_at(self, t):
        """(score, start, end) of the best wake phrase in the first syllables of t."""
        best = (0.0, 0, 0)
        for w in (self.wake, self.wake_alt):
            sc, a, b = match(w, t[:12], strict=False)
            if sc > best[0]: best = (sc, a, b)
        return best

    def has_wake(self, text):
        sc, a, b = self._wake_at(fkey(clean(text)))
        return sc >= 0.74 and a <= 4

    def strip_wake(self, text):
        t = clean(text); sc, a, b = self._wake_at(fkey(t))
        return t[b:] if (sc >= 0.74 and a <= 4) else t

    STOP_WORDS = ('停', '停下', '停下来', '别动', '不要动', '暂停', '停止', '站住', '小拾停', '小拾别动', '小拾停下', '救命')

    def is_stop_word(self, text):
        """The always-on stop spotter: a short utterance that is a stop word (reduced pinyin, exact). Needs no wake word."""
        t = clean(text)
        if len(t) > 7: return False
        ft = fkey(t)
        for w in self.STOP_WORDS:
            p = fkey(w)
            sc, a, b = match(p, ft, strict=True)
            if sc >= 1.0 and (len(p) >= 2 or len(ft) <= 2): return True
        return False

    def entities(self, text, ctx_names=None):
        """Items, places and names in the text: [(kind, ref, surface, score, start, end)] in syllables, longest and best first, no overlaps."""
        ft = fkey(text); found = []
        lex = []
        for k, ws in self.items_static.items():
            for p, w in ws: lex.append(('item', k, w, p))
        for k, ws in self.places.items():
            for p, w in ws: lex.append(('place', k, w, p))
        for n, (typ, ref) in self.names.items(): lex.append((typ if typ == 'person' else 'item', ref, n, fkey(n)))
        for kind, ref, w, p in lex:
            sc, a, b = match(p, ft, strict=True)
            if sc >= self._thr(len(p)) and b - a >= max(len(p) - 1, 1): found.append((kind, ref, w, sc, a, b))
        found.sort(key=lambda f: (-(f[5] - f[4]), -f[3]))
        out = []
        for f in found:
            if not any(not (f[5] <= o[4] or f[4] >= o[5]) for o in out): out.append(f)
        return sorted(out, key=lambda f: f[4])

    def parse(self, text):
        """-> dict(intent, conf, slots, missing, alts, text). conf is the phonetic match of the best phrase, weighed by its weight and length and by whether the slots are there."""
        t = clean(text); ft = fkey(t); ents = self.entities(t)
        slots = {}
        for kind, ref, w, sc, a, b in ents:
            if kind == 'item' and 'item' not in slots: slots['item'] = ref; slots['item_word'] = w
            elif kind == 'place' and 'place' not in slots: slots['place'] = ref
            elif kind == 'person' and 'person' not in slots: slots['person'] = ref
        for w in ('它', '这个', '那个', '这个东西', '那个东西'):                            # 'it', 'that one': filled from the context by the dialogue manager
            if w in t and 'item' not in slots: slots['anaphora'] = True; break
        m = re.search(r'(?:叫做|叫它|叫|起名叫|取名叫|名字是|名字叫)([一-鿿A-Za-z0-9]{1,6})', t)
        if m: slots['newname'] = m.group(1)
        scored = []
        for intent, trigs in self.trig.items():
            best = (0.0, None)
            for p, phrase, w in trigs:
                sc, a, b = match(p, ft)
                thr = self._thr(len(p))
                if intent == 'stop': thr = min(thr, 0.7 if len(p) > 1 else 0.5)
                if sc >= thr:
                    v = sc * w * (0.62 + 0.13 * min(len(p), 3))
                    if v > best[0]: best = (v, phrase)
            if best[1] is None: continue
            conf = best[0]; need = NEEDS.get(intent, []); missing = [s for s in need if not (s in slots or (s == 'item' and slots.get('anaphora')))]
            if intent == 'name_item' and 'newname' not in slots: missing = list(dict.fromkeys(missing + ['newname']))
            if need and not missing: conf = min(1.0, conf + 0.12)
            if missing: conf *= 0.8
            if intent == 'go_to' and 'place' not in slots: conf *= 0.3
            scored.append((conf, intent, best[1], missing))
        # a short yes/no answers an open question: the pattern has to be the whole utterance (give or take one syllable), the longest pattern wins
        if len(t) <= 4:
            best = None
            for kind, words in (('yes', YES), ('no', NO)):
                for w in words:
                    p = fkey(w); sc, a, b = match(p, ft, strict=True)
                    if sc >= 1.0 and a == 0 and b >= len(ft) - 0 and (best is None or len(p) > best[0]): best = (len(p), kind)
            if best: scored.append((0.9, best[1], t, []))
        person_noun = bool(re.search(r'(孩子|小孩|宝宝|女儿|儿子|妈妈|爸爸|爷爷|奶奶|哥哥|姐姐|弟弟|妹妹|宝贝|这个人|他|她)', t))
        if person_noun: slots['person_noun'] = True
        scored = [(c * 0.5 if (i == 'name_item' and person_noun) else c, i, p, m) for c, i, p, m in scored]
        if slots.get('item_word') and slots.get('newname') and not person_noun and not any(i == 'name_item' for _, i, _, _ in scored): scored.append((0.85, 'name_item', '叫', []))
        if re.match(r'^(别|不要|不用|先别)', t) and any(i in ('tidy', 'scan_home', 'go_to', 'charge', 'fetch_item') for _, i, _, _ in scored): scored.append((0.9, 'stop', t, []))
        scored.sort(key=lambda s: -s[0])
        if not scored: return dict(intent=None, conf=0.0, slots=slots, missing=[], alts=[], text=t)
        top = scored[0]
        return dict(intent=top[1], conf=round(top[0], 3), slots=slots, missing=top[3], alts=[(round(c, 2), i) for c, i, _, _ in scored[1:4]], text=t, phrase=top[2])


# ---------------------------------------------------------------------------------------------------------------- an optional language model (never in control)
class LLMBackend:
    """Interface for a language model that may help with sentences the phrase matcher did not understand. It sees only the text of what was said (no audio, no images, no names unless the owner
    allowed them) and the speaker's role. It may return {'intent': one of MIN_ROLE's keys, 'slots': {...}} or {'say': a short sentence}. The robot treats an intent like any other: permission,
    confirmation, executor. Nothing here is tested against a real model."""
    def propose(self, text, ctx, role): return None


class MockLLM(LLMBackend):
    """Canned answers, to test the plumbing: that a proposed intent goes through the permission check, that nonsense is refused, that a timeout falls back."""
    def __init__(self, table=None, fail=False): self.table = table or {}; self.fail = fail; self.calls = 0
    def propose(self, text, ctx, role):
        self.calls += 1
        if self.fail: raise TimeoutError
        for k, v in self.table.items():
            if k in text: return v
        return None


# ---------------------------------------------------------------------------------------------------------------- the dialogue
MODE_CN = dict(IDLE='待命', TASK='收玩具', NAV='自己走动', SCAN='扫描家里', FETCH='捡东西', CHARGE='充电', TELEOP='被遥控', RIGHT='爬起来', FAULT='出了故障')


class VoiceSystem:
    """Wake word -> recogniser -> understanding -> permission -> (question | confirmation | action) -> spoken answer. `hear()` takes the words a person means to say, the simulated recogniser
    writes them down with errors. sup/home/ppl are the robot's parts; without them the actions are only recorded (self.actions), which is how the language side is tested alone."""
    def __init__(self, sup=None, home=None, ppl=None, asr=None, llm=None, seed=0, setup_mode=False, keep_transcripts=False):
        self.sup, self.home, self.ppl = sup, home, ppl; self.nlu = NLU(); self.asr = asr or SimASR(0.04, seed); self.llm = llm
        self._t = 0.0; self.session_until = -1.0; self.pending = None; self.ctx = dict(item=None, place=None, person=None)
        self.tts = []; self.log = []; self.actions = []; self.muted = False; self.resume_cmd = None; self.setup_mode = setup_mode; self.keep_transcripts = keep_transcripts
        if ppl is not None and getattr(ppl, 'guard', None) is not None: ppl.guard.speak = self.speak          # what the guard says goes through the same mouth
        self.extra_persons = []; self._retry = None; self.stats = dict(heard=0, ignored=0, reask=0, confirm=0, denied=0, acted=0, stop=0); self.owner_names = set(); self._enrol = None; self.session_mark = 0

    # ---- time and speech
    def now(self): return self.sup.r.t if self.sup is not None else self._t
    def advance(self, dt): self._t += dt
    def speak(self, text): self.tts.append((round(self.now(), 1), text)); return text

    # ---- who is talking
    def role_of(self, speaker):
        """The speaker's role. In the simulator the speaker is one of the simulated people and the track that belongs to them stands in for 'the person the sound came from'
        (a real robot: sound-source direction from the microphone array, matched to the person seen in that direction)."""
        if self.ppl is None or speaker is None: return 'unknown'
        k = next((k for k in self.ppl.tracks() if k.truth == speaker), None)
        if k is None: return 'unknown'
        reg = self.ppl.registry
        if k.who: return 'owner' if reg.role(k.who) == 'adult' else 'child'
        if k.cls == 'adult': return 'owner' if (self.setup_mode and not any(p['role'] == 'adult' for p in reg.people.values())) else 'adult'
        return 'child'

    def _names(self):
        persons = (list(self.ppl.registry.people) if self.ppl is not None else []) + list(self.extra_persons)
        items = {}
        if self.home is not None and self.home.inv is not None: items = {i: it['name'] for i, it in self.home.inv.items.items() if it['name']}
        self.nlu.set_names(persons, items)

    # ---- one utterance
    def hear(self, said, speaker=None, role=None, dist=1.5, _retry=False):
        now = self.now(); role = role or self.role_of(speaker); self.stats['heard'] += 1
        rec = dict(t=round(now, 1), said=said if self.keep_transcripts else None, role=role, speaker=speaker)
        if self.muted: self.stats['ignored'] += 1; rec.update(action='ignored: muted'); self.log.append(rec); return rec
        child_voice = role == 'child'
        if speaker is not None and self.ppl is not None and self.ppl.people.state[speaker] is not None:     # the sound itself: a child's voice, how far away (the simulator knows; the robot does not)
            st = self.ppl.people.state[speaker]; rx, ry, _ = self.ppl.r.true_pose(); child_voice = st['cls'] == 'child'; dist = math.hypot(st['x'] - rx, st['y'] - ry)
        text, aconf = self.asr.hear(said, child=child_voice, dist=dist); rec.update(heard=text if self.keep_transcripts else None, asr_conf=round(aconf, 2)); self._names()
        if self.nlu.is_stop_word(text):                                       # always on, no wake word, anyone
            self._do_stop(); self.stats['stop'] += 1; rec.update(intent='stop', action='stop', reply=self.speak('好的，我停下来了')); self.session_until = now + SESSION_S; self.log.append(rec); return rec
        woke = self.nlu.has_wake(text); live = now < self.session_until or (self.pending is not None and now <= self.pending['until'])
        if not woke and not live:
            self.stats['ignored'] += 1; rec.update(action='ignored: no wake word'); self.log.append(rec); return rec
        body = self.nlu.strip_wake(text) if woke else clean(text)
        if not body:
            self.session_until = now + SESSION_S; rec.update(intent='wake', action='listening', reply=self.speak('我在')); self.log.append(rec); return rec
        if role == 'unknown' and speaker is not None and not _retry and self.sup is not None and self.ppl is not None and self.sup.mode == 'IDLE' and self.pending is None:
            st = self.ppl.people.state[speaker]                       # nobody seen where the voice came from: turn that way (sound-source direction; here the simulator's truth) and listen again
            if st is not None and st['inside']:
                x, y, th = self.sup.a.pose(); self.sup.go_to(x, y, th=math.atan2(st['y'] - y, st['x'] - x), tol=0.15)
                self._retry = dict(said=said, speaker=speaker, t_ready=now + 3.5); self.session_until = now + SESSION_S
                rec.update(action='turning toward the speaker', reply=self.speak('我转过来看看')); self.log.append(rec); return rec
        if self.pending is not None and now > self.pending['until']: self.pending = None
        if self.pending is not None:
            done = self._answer(body, role, aconf, speaker, rec)
            if done: self.session_until = now + SESSION_S; self.log.append(rec); return rec
        rec['_body'] = body; self._command(body, role, aconf, speaker, rec); rec.pop('_body', None); self.session_until = now + (PENDING_S if self.pending else SESSION_S); self.log.append(rec); return rec

    # ---- commands
    def _command(self, body, role, aconf, speaker, rec):
        p = self.nlu.parse(body); conf = p['conf'] * (0.55 + 0.45 * aconf); intent = p['intent']; slots = dict(p['slots'])
        if intent == 'privacy' and p.get('phrase') in ('静音', '别听', '关掉麦克风', '别看'): slots['mute'] = True
        if intent == 'forget' and re.search(r'(所有|全部|大家|每个人)', body): slots['all'] = True
        rec.update(intent=intent, conf=round(conf, 2), slots={k: v for k, v in slots.items() if k != 'item_word'})
        weak = intent is None or conf < 0.45 or (bool(p['missing']) and conf < 0.7 and not slots.get('anaphora'))
        if weak and self.llm is not None and self._ask_llm(body, role, rec): return
        if intent is None or conf < 0.45:
            self.stats['reask'] += 1; rec.update(action='reask'); rec['reply'] = self.speak('我没听清，请再说一遍' if intent is None else f'我没太听清，你是说“{self._intent_cn(intent)}”吗？'); 
            if intent is not None: self.pending = dict(kind='confirm', intent=intent, slots=slots, role=role, until=self.now() + PENDING_S)
            return
        if intent in ('yes', 'no'): rec.update(action='nothing to answer'); rec['reply'] = self.speak('嗯？'); return
        # 'it' / 'that one'
        if intent in NEEDS and 'item' in NEEDS[intent] and 'item' not in slots and slots.get('anaphora') and self.ctx['item'] is not None: slots['item'] = self.ctx['item']
        missing = [s for s in NEEDS.get(intent, []) if s not in slots or slots.get(s) is None]
        if not self._allowed(intent, role, rec): return
        if intent in ('enrol',): return self._start_enrol(slots, role, speaker, rec)
        if intent == 'forget': return self._start_forget(slots, role, rec)
        if missing:
            ask = dict(item='要哪一个东西？', place='要去哪里？', newname='想叫它什么名字？')[missing[0]]
            self.pending = dict(kind='slot', intent=intent, slots=slots, slot=missing[0], role=role, until=self.now() + PENDING_S); rec.update(action='ask', reply=self.speak(ask)); return
        if intent in MOTION and conf < 0.7:                                  # not sure enough to move: ask first
            self.stats['confirm'] += 1; self.pending = dict(kind='confirm', intent=intent, slots=slots, role=role, until=self.now() + PENDING_S)
            rec.update(action='confirm', reply=self.speak(f'你是要我{self._intent_cn(intent, slots)}吗？')); return
        self._execute(intent, slots, role, rec)

    def _intent_cn(self, intent, slots=None):
        s = slots or {}; it = s.get('item_word') or ''
        return dict(tidy='收拾玩具', fetch_item=f'把{it or "它"}放进筐里', scan_home='扫描一遍家里', go_to=f'去{s.get("place", "那里")}', charge='去充电', where_is=f'找{it or "东西"}', resume='继续刚才的事',
                    enrol='记住一个人', forget='忘掉一个人', stop='停下', status='说说我在做什么', list_items='数数地上的东西', who_here='看看谁在').get(intent, intent)

    def _allowed(self, intent, role, rec):
        need = MIN_ROLE.get(intent, 'adult')
        if ROLE_RANK[role] >= ROLE_RANK[need]: return True
        self.stats['denied'] += 1; rec.update(action='denied')
        who = {'child': '这个要大人来说哦', 'unknown': '我没看清是谁在说话，这个要大人在我面前说才行'}.get(role, '这个需要主人来确认')
        rec['reply'] = self.speak(who); return False

    def _ask_llm(self, body, role, rec):
        try: out = self.llm.propose(body, dict(self.ctx), role)
        except Exception: return False
        if not isinstance(out, dict): return False
        if 'intent' in out and out['intent'] in MIN_ROLE:
            rec.update(intent=out['intent'], via='llm'); slots = dict(out.get('slots') or {})
            if not self._allowed(out['intent'], role, rec): return True
            if out['intent'] in MOTION or out['intent'] in SENSITIVE:           # a model's guess is never enough to move or to store anything about a person: ask
                self.stats['confirm'] += 1; self.pending = dict(kind='confirm', intent=out['intent'], slots=slots, role=role, until=self.now() + PENDING_S)
                rec.update(action='confirm', reply=self.speak(f'你是要我{self._intent_cn(out["intent"], slots)}吗？')); return True
            self._execute(out['intent'], slots, role, rec); return True
        if 'say' in out and isinstance(out['say'], str) and role in ('owner', 'adult') and len(out['say']) <= 80:
            rec.update(action='llm reply', via='llm', reply=self.speak(out['say'])); return True
        return False

    # ---- answers to an open question
    def _answer(self, body, role, aconf, speaker, rec):
        pd = self.pending; p = self.nlu.parse(body); yes = p['intent'] == 'yes'; no = p['intent'] == 'no'; now = self.now(); k = pd['kind']
        if k == 'confirm':
            if no: self.pending = None; rec.update(intent='no', action='cancelled', reply=self.speak('好的，不做了')); return True
            if yes:
                if ROLE_RANK[role] < ROLE_RANK[MIN_ROLE.get(pd['intent'], 'adult')]: self._allowed(pd['intent'], role, rec); return True
                self.pending = None; rec.update(intent=pd['intent'], via='confirmed'); self._execute(pd['intent'], pd['slots'], role, rec); return True
            self.pending = None; return False                               # something else: a new command
        if k == 'slot':
            slots = dict(pd['slots']); s = pd['slot']
            if s == 'newname':
                nm = re.sub(r'^(叫|就叫|叫做|名字是|它叫|叫它)', '', clean(body))
                if not nm or len(nm) > 8 or p['intent'] in ('stop', 'cancel'): self.pending = None; return False
                slots['newname'] = nm
            else:
                got = {e[0]: e[1] for e in self.nlu.entities(body)}
                if s not in got: self.pending = None; return False
                slots[s] = got[s]
                if s == 'item': slots['item_word'] = next((e[2] for e in self.nlu.entities(body) if e[0] == 'item'), '')
            self.pending = None; rec.update(intent=pd['intent'], via='answer')
            if pd['intent'] == 'forget': self._start_forget(slots, role, rec)
            else: self._execute(pd['intent'], slots, role, rec)
            return True
        if k in ('enrol_name', 'enrol_name_ok', 'enrol_consent', 'forget_ok'): return self._answer_enrol_forget(k, pd, body, p, yes, no, role, speaker, rec)
        self.pending = None; return False

    # ---- remembering a person (opt-in, with a grown-up's consent) and forgetting
    def _start_enrol(self, slots, role, speaker, rec):
        if self.ppl is None: rec.update(action='no people tracker'); rec['reply'] = self.speak('我现在看不到人，没法记'); return
        mine = bool(re.search(r'记住我|认识我|记一下我', rec.get('_body', '')))                  # "remember me": the person who is talking
        tr = self.ppl.nearest(truth=speaker) if (mine and speaker is not None) else self.ppl.nearest(kind='child' if slots.get('person_noun') else None, unidentified=True)
        if tr is None or tr.dist > 3.0: rec.update(action='nobody in view'); rec['reply'] = self.speak('我现在没看到人。请让要记住的人站在我前面，看着我'); return
        who = slots.get('newname') or slots.get('person')
        self._enrol = dict(track=tr.id, role='child' if tr.cls in ('child', 'unsure') else 'adult', name=who, by=None, t0=self.now(), pos=(tr.x, tr.y))
        if not who: self.pending = dict(kind='enrol_name', role=role, until=self.now() + PENDING_S); rec.update(action='ask', reply=self.speak('好的。请问他叫什么名字？')); return
        self.pending = dict(kind='enrol_name_ok', role=role, until=self.now() + PENDING_S); rec.update(action='confirm', reply=self.speak(f'是叫“{who}”对吗？'))

    def _answer_enrol_forget(self, k, pd, body, p, yes, no, role, speaker, rec):
        now = self.now(); E = self._enrol
        if no and k in ('enrol_name_ok', 'enrol_consent', 'forget_ok'):
            self.pending = None; self._enrol = None; rec.update(action='cancelled', reply=self.speak('好的，那就不记了' if k != 'forget_ok' else '好的，不删')); return True
        if k == 'enrol_name':
            nm = re.sub(r'^(他叫|她叫|叫|就叫|名字是)', '', clean(body))
            if not nm or len(nm) > 8: self.pending = None; return False
            E['name'] = nm; self.pending = dict(kind='enrol_name_ok', role=role, until=now + PENDING_S); rec.update(action='confirm', reply=self.speak(f'是叫“{nm}”对吗？')); return True
        if k == 'enrol_name_ok':
            if not yes: self.pending = None; self._enrol = None; return False
            self.pending = dict(kind='enrol_consent', role=role, until=now + PENDING_S + 8)
            rec.update(action='consent', reply=self.speak(f'我会记住“{E["name"]}”的样子特征，不保存照片，只存在我自己里面，随时可以说“忘掉{E["name"]}”删除。需要大人同意，请大人说“同意”。')); return True
        if k == 'enrol_consent':
            if not yes: self.pending = None; self._enrol = None; return False
            if role not in ('owner', 'adult'):
                rec.update(action='denied', reply=self.speak('需要大人说同意才可以哦')); self.stats['denied'] += 1; return True
            E['by'] = 'owner' if role == 'owner' else f'guest adult @ {now:.0f}s'; E['wait'] = now; self.pending = None
            rec.update(action='enrolling', reply=self.speak('好，请让他看着我，大概三秒钟')); return True
        if k == 'forget_ok':
            self.pending = None
            if not yes: return False
            n = pd['who']; ok = self._forget(n); rec.update(action='forgot', reply=self.speak(f'好了，已经忘掉{n}了' if ok else f'我本来就没记着{n}'))
            return True
        return False

    def _forget(self, who):
        self.actions.append(('forget', who))
        if self.ppl is None: return True
        if who == '__all__': return self.ppl.registry.forget_all(self.now()) > 0
        ok = self.ppl.registry.forget(who, self.now())
        for k in self.ppl.det.tracks:
            if k.who == who: k.who = None; k.emb = []
        return ok

    def _start_forget(self, slots, role, rec):
        who = slots.get('person')
        if who is None:
            if slots.get('all'): who = '__all__'
            else: rec.update(action='ask'); rec['reply'] = self.speak('要忘掉谁？'); self.pending = dict(kind='slot', intent='forget', slots=slots, slot='person', role=role, until=self.now() + PENDING_S); return
        self.pending = dict(kind='forget_ok', who=who, role=role, until=self.now() + PENDING_S); rec.update(action='confirm', reply=self.speak(f'要我忘掉{"所有人" if who == "__all__" else who}的特征吗？删掉就找不回来了'))

    def tick(self):
        """Per control tick (the supervisor calls it): finish an enrolment once there are enough frontal looks; drop an expired question."""
        now = self.now()
        if self._retry is not None and now >= self._retry['t_ready']:
            rt, self._retry = self._retry, None; self.hear(rt['said'], speaker=rt['speaker'], _retry=True)
        if self.pending is not None and now > self.pending['until']: self.pending = None
        E = self._enrol
        if E is not None and E.get('by') and self.ppl is not None:
            tr = next((k for k in self.ppl.det.tracks if k.id == E['track']), None)
            if tr is None and E.get('pos'):                                   # lost the track (the robot turned to the grown-up): the same child, seen again near where he was
                tr = next((k for k in self.ppl.tracks() if k.cls != 'adult' and math.hypot(k.x - E['pos'][0], k.y - E['pos'][1]) < 1.0), None)
                if tr is not None: E['track'] = tr.id
            if tr is not None: E['pos'] = (tr.x, tr.y)
            if (tr is None or tr.coast) and self.sup is not None and self.sup.mode == 'IDLE' and E.get('pos') and now - E.get('turned', -99.0) > 4.0:
                x, y, th = self.sup.a.pose(); self.sup.go_to(x, y, th=math.atan2(E['pos'][1] - y, E['pos'][0] - x), tol=0.15); E['turned'] = now      # look at the person being remembered
            if now - E['wait'] > 30.0: self._enrol = None; self.speak('我没看清楚脸，下次再试'); return
            if tr is None: return
            if sum(tr.frontal) >= 8:
                try: self.ppl.enrol(tr, E['name'], E['role'], E['by']); self.actions.append(('enrol', E['name'], E['role'])); self.speak(f'记住了，{E["name"]}。')
                except Exception as e: self.speak('没记成功，下次再试')
                self._enrol = None

    # ---- doing things
    def _do_stop(self):
        self.actions.append(('stop',))
        if self.sup is not None:
            if self.sup.mode in ('TASK', 'NAV', 'SCAN', 'CHARGE', 'FETCH'): self.resume_cmd = (self.sup.mode, dict(self.sup._kw))
            self.sup.idle()

    def _items_for(self, ref):
        inv = self.home.inv if (self.home is not None and self.home.inv is not None) else None
        if inv is None: return []
        if isinstance(ref, int): return [inv.items[ref]] if ref in inv.items else []
        out = [it for it in inv.items.values() if it['kind'] == ref and it['kind_conf'] >= 0.2]
        return sorted(out, key=lambda it: -it['last_seen'])

    def _approach(self, name):
        """A free spot 0.5 m from the named furniture, nearest to the robot (from the 3D map's furniture boxes and the dock)."""
        if self.home is None or self.home.analysis is None: return None
        from inventory import furniture_name
        agent = self.home.a; x0, y0, _ = agent.pose(); D = agent.costmap(); best = None
        boxes = [(furniture_name(f), f) for f in self.home.analysis['furniture']]
        for nm, f in boxes:
            if nm != name: continue
            c, s = math.cos(f['yaw']), math.sin(f['yaw'])
            for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                px = f['centre'][0] + c * sx * (f['length'] / 2 + 0.5) - s * sy * (f['width'] / 2 + 0.5); py = f['centre'][1] + s * sx * (f['length'] / 2 + 0.5) + c * sy * (f['width'] / 2 + 0.5)
                i, j = int(px / 0.05), int(py / 0.05)
                if not (0 <= j < D.shape[0] and 0 <= i < D.shape[1]) or D[j, i] < 0.235: continue
                d = math.hypot(px - x0, py - y0)
                if best is None or d < best[0]: best = (d, px, py)
        return None if best is None else (best[1], best[2])

    def _execute(self, intent, slots, role, rec):
        kid = role in ('child', 'unknown'); inv = self.home.inv if (self.home is not None and self.home.inv is not None) else None
        self.stats['acted'] += 1; rec['action'] = intent; sup = self.sup; self.actions.append((intent, {k: v for k, v in slots.items() if k in ('item', 'place', 'person', 'newname')}))
        say = lambda a, c=None: rec.__setitem__('reply', self.speak(c if (kid and c) else a))
        if intent == 'stop': self._do_stop(); say('好的，我停下来了')
        elif intent == 'resume':
            if self.resume_cmd and sup is not None: mode, kw = self.resume_cmd; sup.request(mode, **(dict(resume=True) if mode == 'TASK' else kw)); say(f'好的，接着{MODE_CN.get(mode, "刚才的事")}'); self.resume_cmd = None
            else: say('我没有要接着做的事')
        elif intent == 'cancel': self.pending = None; say('好的，不做了')
        elif intent == 'tidy':
            if sup is not None: sup.start_task()
            say('好的，我去收地上的玩具', '好的！我去收玩具啦')
        elif intent == 'fetch_item':
            its = self._items_for(slots.get('item')); k = None
            if sup is not None and its and self.home is not None:
                ks = [kk for kk, i in inv.track.items() if i == its[0]['id'] and kk in self.home.a.memory and self.home.a.memory[kk]['status'] == 'pending']
                k = ks[0] if ks else None
            if self.sup is not None and k is None: say(f'我现在不知道{slots.get("item_word", "它")}在哪里，我先找一找'); sup.explore() if sup else None
            else:
                if sup is not None: sup.fetch_item(k)
                self.ctx['item'] = its[0]['id'] if its else slots.get('item'); say(f'好的，我去把{slots.get("item_word", "它")}放进筐里', '好的！')
        elif intent == 'where_is':
            its = self._items_for(slots.get('item'))
            if inv is None: say(f'{slots.get("item_word", "它")}？我现在还没有物品记录')
            elif not its: say(f'我没见过{slots.get("item_word", "这个东西")}')
            else:
                self.ctx['item'] = its[0]['id']; say('；'.join(inv.where(it) for it in its[:2]))
        elif intent == 'list_items':
            if inv is None: say('我还没有物品记录')
            else:
                fl = [it for it in inv.on_floor() if it.get('seen_this_session')]
                say('地上现在有：' + '、'.join(f'{inv.label(it)}（{it["place"] or "地板上"}）' for it in fl) if fl else '地上现在没看到东西')
        elif intent == 'what_new':
            if inv is None: say('我还没有物品记录')
            else:
                new = inv.new_since(inv.session - 1); mv = inv.moved_since(inv.session - 1); ms = inv.missing(); un = inv.unverified(); parts = []
                if new: parts.append('新来的：' + '、'.join(inv.label(i) for i in new))
                if mv: parts.append('挪过位置的：' + '、'.join(inv.label(i) for i in mv))
                if ms: parts.append('不见了的：' + '、'.join(inv.label(i) for i in ms))
                if un: parts.append('还没核实的：' + '、'.join(inv.label(i) for i in un))
                say('；'.join(parts) if parts else '和上次一样，没什么变化')
        elif intent == 'scan_home':
            if sup is not None: sup.scan_home()
            say('好的，我去把家里扫描一遍，大概要两分钟')
        elif intent == 'go_to':
            name = slots.get('place'); pt = None
            if name == '充电座': return self._execute('charge', slots, role, rec)
            if name == '收纳筐' and self.home is not None: b = self.home.a.basket; pt = (b['x'] - 0.5, b['y'])
            elif sup is not None: pt = self._approach(name)
            if sup is None: say(f'好的，我去{name}那边')
            elif pt is None: say(f'我还不知道{name}在哪里，要先扫描一下家里')
            else: sup.go_to(pt[0], pt[1]); self.ctx['place'] = name; say(f'好的，我去{name}那边')
        elif intent == 'charge':
            if sup is not None: sup.charge_now()
            say('好的，我去充电')
        elif intent == 'who_here':
            ds = self.ppl.describe() if self.ppl is not None else []
            say('我看到：' + '；'.join(ds) if ds else '我现在没看到人')
        elif intent == 'status':
            if sup is None: say('我在待命')
            else:
                st = sup.status(); soc = f'，电量大约百分之{int(round(100 * st["soc"]))}' if 'soc' in st else ''
                say(f'我现在在{MODE_CN.get(st["mode"], st["mode"])}{soc}')
        elif intent == 'name_item':
            its = self._items_for(slots.get('item')) if slots.get('item') is not None else []
            if inv is None or not its: say('我不知道你说的是哪一个')
            else: inv.name_item(its[0]['id'], slots['newname']); self.ctx['item'] = its[0]['id']; say(f'好的，以后叫它“{slots["newname"]}”')
        elif intent == 'privacy':
            if slots.get('mute'): self.muted = True; say('好的，我不听了。要我重新听，请按机身上的静音键')
            else: say('我只在听到“小拾小拾”以后才处理你说的话，不保存录音，也不会把声音传出去。要让我完全不听，请按机身上的静音键')
        elif intent == 'greet': say('你好！我是小拾', '你好呀！')
        elif intent == 'identity': say('我是小拾，家里的收纳小助手', '我是小拾！')
        elif intent == 'help': say('我可以收拾地上的玩具、扫描家里、记住东西放在哪里、告诉你东西在哪儿，也可以去充电。', '我会收玩具，还会找东西！')
        else: say('这个我还不会')
