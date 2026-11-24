"""English and mixed-language support for the voice interface (the Chinese side lives in voice.py).

A home where people speak both languages (and mix them in one sentence: "把 ball 放进筐里", "go to the 沙发") needs more than a second phrase list:

  detect_lang      Chinese, English or mixed, from what the recogniser wrote (script and word counts) - the acoustic language-ID of a real recogniser is not simulated.
  phones           English words as phonemes (the CMU pronouncing dictionary; a rough letter-to-sound rule for names and words it does not have).
  en_match         approximate matching of a phrase in the heard words on phonemes, so "bawl" and "ball", "sheep" and "ship", "tree" and "three" still match: the same idea as the
                   pinyin matching for Chinese (a wrong word that sounds right is no mistake), with the confusions a Mandarin speaker of English makes counted as small (th/s, v/w, r/l,
                   final stops dropped, ship/sheep).
  latin_pinyin     a Latin word that is a run of pinyin syllables ("xiaoshi") is read as Chinese: the robot's name is spelled that way in an English sentence.
  lexicons         English phrase triggers per intent, English names of the items and places, yes/no, stop words, the wake phrases.
  reply            every spoken answer has a Chinese and an English text; the answer is in the language of the question (or the speaker's profile).
"""
import functools
import re

import numpy as np

try:
    import cmudict as _cmudict
    _CMU = _cmudict.dict()
except Exception:                                      # the dictionary is a pip package; without it names and words fall back to the letter rules
    _CMU = {}

# ---------------------------------------------------------------------------------------------------------------- language
_HAN = re.compile(r'[一-鿿]')
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def split_runs(text):
    """[('zh'|'en'|'other', string)] runs of the text by script."""
    out = []; cur = None; buf = ''
    for ch in text:
        k = 'zh' if _HAN.match(ch) else ('en' if (ch.isascii() and (ch.isalpha() or ch in "'-")) else 'other')
        if k == 'other' and cur in ('zh', 'en'): buf += ch; continue
        if k != cur and buf:
            out.append((cur, buf)); buf = ''
        cur = k; buf += ch
    if buf: out.append((cur, buf))
    return out


def detect_lang(text, prior=None):
    """-> (lang, zh_share): 'zh', 'en' or 'mixed'. A Chinese word is counted by its characters / 1.7 (about a character-pair per word), an English word by itself; pinyin-like Latin tokens
    (the robot's name) do not count as English. prior: the speaker's usual language, used when there is nothing to go on."""
    zh = len(_HAN.findall(text)) / 1.7; en = sum(1 for w in _WORD.findall(text) if latin_pinyin(w.lower()) is None and w.lower() not in ('hey', 'hi', 'ok', 'okay'))
    if zh + en < 0.5: return (prior or 'zh'), 0.5
    share = zh / (zh + en)
    return ('zh' if share >= 0.8 else ('en' if share <= 0.2 else 'mixed')), share


# ---------------------------------------------------------------------------------------------------------------- pinyin-like Latin words
def _pinyin_set():
    from pypinyin import Style, lazy_pinyin
    from pypinyin.pinyin_dict import pinyin_dict
    s = set()
    for cp in pinyin_dict:
        for p in lazy_pinyin(chr(cp), style=Style.NORMAL): s.add(p)
    return {p for p in s if p.isascii() and p.isalpha()}


PINYIN = _pinyin_set()


@functools.lru_cache(maxsize=4096)
def pinyin_segments(token):
    """A Latin token that splits into one to four pinyin syllables -> the syllable list, whatever else it might be ('ming', 'shi'); otherwise None."""
    t = token.lower()
    if not t or not t.isalpha(): return None
    best = {0: []}
    for i in range(len(t)):
        if i not in best: continue
        for j in range(i + 1, min(len(t), i + 6) + 1):
            if t[i:j] in PINYIN and (j not in best or len(best[i]) + 1 < len(best[j])): best[j] = best[i] + [t[i:j]]
    sy = best.get(len(t)); return sy if sy and len(sy) <= 4 else None


@functools.lru_cache(maxsize=4096)
def latin_pinyin(token):
    """A Latin token that is one to four pinyin syllables written together ('xiaoshi', 'shao', 'nihao') -> the syllable list; otherwise None. Plain English words that are also pinyin
    ('she', 'me', 'a') are not taken as pinyin unless the token has two or more syllables or the caller asks."""
    t = token.lower()
    if len(t) < 3 and t not in ('ai', 'ao', 'er'): return None
    if t in _CMU: return None                                      # an English word is an English word, even if it can be cut into syllables (house = hou + se, anyone = an + yo + ne)
    best = {0: []}
    for i in range(len(t)):
        if i not in best: continue
        for j in range(i + 1, min(len(t), i + 6) + 1):
            if t[i:j] in PINYIN and (j not in best or len(best[i]) + 1 < len(best[j])): best[j] = best[i] + [t[i:j]]
    sy = best.get(len(t)); return sy if sy and len(sy) <= 4 and (len(sy) >= 2 or t in PINYIN and len(t) >= 3 and t not in _CMU) else None


# ---------------------------------------------------------------------------------------------------------------- English phonemes
_G2P = [('tch', 'CH'), ('sch', 'S K'), ('sh', 'SH'), ('ch', 'CH'), ('th', 'TH'), ('ng', 'NG'), ('ph', 'F'), ('ck', 'K'), ('qu', 'K W'), ('wh', 'W'), ('kn', 'N'), ('ee', 'IY'), ('ea', 'IY'),
        ('oo', 'UW'), ('ai', 'EY'), ('ay', 'EY'), ('oa', 'OW'), ('ou', 'AW'), ('ow', 'OW'), ('oi', 'OY'), ('ie', 'IY'), ('au', 'AO'), ('aw', 'AO'),
        ('a', 'AE'), ('b', 'B'), ('c', 'K'), ('d', 'D'), ('e', 'EH'), ('f', 'F'), ('g', 'G'), ('h', 'HH'), ('i', 'IH'), ('j', 'JH'), ('k', 'K'), ('l', 'L'), ('m', 'M'), ('n', 'N'), ('o', 'AA'),
        ('p', 'P'), ('q', 'K'), ('r', 'R'), ('s', 'S'), ('t', 'T'), ('u', 'AH'), ('v', 'V'), ('w', 'W'), ('x', 'K S'), ('y', 'Y'), ('z', 'Z')]


@functools.lru_cache(maxsize=20000)
def phones(word):
    """Phonemes of an English word (stress removed): the CMU dictionary, else a rough letter-to-sound rule. Returns a tuple."""
    w = word.lower().strip("'-")
    if w in _CMU: return tuple(re.sub(r'\d', '', p) for p in _CMU[w][0])
    out = []; i = 0
    while i < len(w):
        for g, p in _G2P:
            if w.startswith(g, i): out += p.split(); i += len(g); break
        else: i += 1
    return tuple(out)


_PAIRS = {frozenset(p): c for p, c in [
    (('P', 'B'), 0.35), (('T', 'D'), 0.35), (('K', 'G'), 0.35), (('F', 'V'), 0.35), (('S', 'Z'), 0.3), (('SH', 'ZH'), 0.35), (('CH', 'JH'), 0.35), (('TH', 'DH'), 0.3),
    (('TH', 'S'), 0.25), (('TH', 'F'), 0.3), (('DH', 'D'), 0.25), (('DH', 'Z'), 0.25), (('TH', 'T'), 0.3), (('V', 'W'), 0.25), (('R', 'L'), 0.3), (('L', 'N'), 0.4), (('M', 'N'), 0.4), (('N', 'NG'), 0.35),
    (('S', 'SH'), 0.4), (('Z', 'JH'), 0.5), (('CH', 'SH'), 0.4), (('B', 'V'), 0.4), (('W', 'UW'), 0.4), (('HH', 'F'), 0.5),
    (('IY', 'IH'), 0.3), (('EH', 'AE'), 0.3), (('AA', 'AO'), 0.3), (('AH', 'AA'), 0.35), (('UW', 'UH'), 0.3), (('OW', 'AO'), 0.4), (('ER', 'AH'), 0.4), (('AW', 'AO'), 0.4), (('AY', 'EY'), 0.4), (('EH', 'IH'), 0.4),
    (('AH', 'UH'), 0.4), (('AH', 'EH'), 0.4), (('ER', 'R'), 0.4), (('AO', 'AH'), 0.4), (('AE', 'AA'), 0.4), (('EY', 'EH'), 0.4)]}
_WEAK = {'AH', 'HH', 'ER', 'IH'}


def _sub(a, b):
    if a == b: return 0.0
    return _PAIRS.get(frozenset((a, b)), 1.0)


@functools.lru_cache(maxsize=200000)
def ph_dist(pa, pb):
    """Weighted edit distance of two phoneme tuples (substitutions by confusability, insertions and deletions 1, or 0.5 for weak phonemes), normalised by the longer: 0 same, 1 unrelated."""
    n, m = len(pa), len(pb)
    if n == 0 and m == 0: return 0.0
    D = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1): D[i, 0] = D[i - 1, 0] + (0.5 if pa[i - 1] in _WEAK else 1.0)
    for j in range(1, m + 1): D[0, j] = D[0, j - 1] + (0.5 if pb[j - 1] in _WEAK else 1.0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + _sub(pa[i - 1], pb[j - 1]), D[i - 1, j] + (0.5 if pa[i - 1] in _WEAK else 1.0), D[i, j - 1] + (0.5 if pb[j - 1] in _WEAK else 1.0))
    return float(D[n, m] / max(n, m))


@functools.lru_cache(maxsize=200000)
def word_dist(a, b):
    """0 (the same word or the same sound) .. 1 (different). Words of one or two phonemes ('go', 'to') have to match exactly."""
    if a == b: return 0.0
    pa, pb = phones(a), phones(b)
    if not pa or not pb: return 1.0
    if pa == pb: return 0.0
    if min(len(pa), len(pb)) <= 2: return 1.0
    return min(1.0, ph_dist(pa, pb) * 1.25)


FUNCTION = {'the', 'a', 'an', 'my', 'your', 'our', 'his', 'her', 'please', 'to', 'of', 'it', 'that', 'this', 'is', 'are', 'do', 'you', 'can', 'could', 'would', 'just', 'now', 'then', 'and', 'for', 'me',
            'um', 'uh', 'er', 'like', 'well', 'so', 'ok', 'okay', 'hey', 'hi', 'there', 'some', 'any', 'kindly'}


_CONTR = {"where's": 'where is', "what's": 'what is', "who's": 'who is', "it's": 'it is', "that's": 'that is', "there's": 'there is', "here's": 'here is', "let's": 'let us', "how's": 'how is',
          "don't": 'do not', "doesn't": 'does not', "didn't": 'did not', "can't": 'can not', "cannot": 'can not', "won't": 'will not', "isn't": 'is not', "aren't": 'are not', "wasn't": 'was not',
          "haven't": 'have not', "i'm": 'i am', "you're": 'you are', "i'll": 'i will', "you'll": 'you will', "we're": 'we are', "they're": 'they are', "i've": 'i have', "i'd": 'i would',
          "he's": 'he is', "she's": 'she is', "what're": 'what are', "wouldn't": 'would not', "shouldn't": 'should not', "couldn't": 'could not'}


def en_tokens(text):
    """Lower-case words of the English part of a text; contractions are expanded ("where's" -> where is, "don't" -> do not), as they are in the phrase lists."""
    out = []
    for w in _WORD.findall(text.replace('\u2019', "'")):
        w = w.lower().strip("'-")
        if not w: continue
        out += _CONTR.get(w, w).split()
    return out


def en_match(pattern, text, strict_short=True):
    """Best approximate occurrence of the word list `pattern` in the word list `text` on phonemes: (score 0..1, start, end). Substitution by word_dist, insertion/deletion 1 (0.25 for
    function words). The score is 1 - cost / weight of the pattern's content words."""
    m, n = len(pattern), len(text)
    if m == 0 or n == 0: return 0.0, 0, 0
    wp = [0.35 if w in FUNCTION else 1.0 for w in pattern]; tot = sum(wp)
    D = np.zeros((m + 1, n + 1)); S = np.zeros((m + 1, n + 1), int)
    for i in range(1, m + 1): D[i, 0] = D[i - 1, 0] + wp[i - 1]
    for j in range(n + 1): S[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c = (D[i - 1, j - 1] + wp[i - 1] * word_dist(pattern[i - 1], text[j - 1]), D[i - 1, j] + wp[i - 1], D[i, j - 1] + (0.25 if text[j - 1] in FUNCTION else 1.0))
            k = int(np.argmin(c)); D[i, j] = c[k]; S[i, j] = S[i - 1, j - 1] if k == 0 else (S[i - 1, j] if k == 1 else S[i, j - 1])
    j = int(np.argmin(D[m, 1:])) + 1
    return max(0.0, 1.0 - D[m, j] / tot), int(S[m, j]), j


# ---------------------------------------------------------------------------------------------------------------- English lexicons
TRIG_EN = {
    'stop': [('stop', 1.0), ('stop it', 1.0), ('stop that', 1.0), ('stop now', 1.0), ("don't move", 1.0), ('do not move', 1.0), ('freeze', 1.0), ('halt', 1.0), ('hold on', 0.9), ('wait', 0.9), ('wait a second', 1.0), ('stay there', 0.9)],
    'resume': [('continue', 1.0), ('carry on', 1.0), ('go on', 0.9), ('resume', 1.0), ('keep going', 1.0), ('go ahead', 0.9)],
    'cancel': [('never mind', 1.0), ('nevermind', 1.0), ('cancel', 1.0), ('cancel that', 1.0), ('forget it', 1.0), ('no thanks', 1.0), ('leave it', 0.9)],
    'tidy': [('tidy up', 1.0), ('clean up', 1.0), ('pick up the toys', 1.0), ('put the toys away', 1.0), ('tidy the room', 1.0), ('collect the toys', 1.0), ('clean the floor', 0.9), ('tidy', 0.9)],
    'fetch_item': [('put it in the basket', 1.0), ('in the basket', 1.0), ('pick it up', 1.0), ('pick up', 0.8), ('put away', 0.9), ('bring it to the basket', 1.0), ('put in the basket', 1.0)],
    'where_is': [('where is', 1.0), ("where's", 1.0), ('where are', 1.0), ('find the', 0.9), ('find my', 1.0), ('look for', 1.0), ('have you seen', 1.0), ('can you find', 1.0), ('do you know where', 1.0), ('find', 0.7)],
    'list_items': [("what's on the floor", 1.0), ('what is on the floor', 1.0), ('what toys are there', 1.0), ('list the toys', 1.0), ('what do you see', 1.0), ('what can you see', 1.0), ('what is there', 0.8), ('what toys', 0.9)],
    'what_new': [("what's new", 1.0), ('what is new', 1.0), ('anything new', 1.0), ('what changed', 1.0), ('what has changed', 1.0), ('what moved', 1.0), ("what's missing", 1.0), ('what is missing', 1.0), ('anything missing', 1.0), ('anything moved', 1.0)],
    'scan_home': [('scan the house', 1.0), ('scan the room', 1.0), ('scan the home', 1.0), ('map the house', 1.0), ('map the home', 1.0), ('make a map', 1.0), ('make a 3d map', 1.0), ('look around the house', 1.0), ('explore the house', 1.0), ('scan', 0.7)],
    'go_to': [('go to', 0.9), ('go over to', 1.0), ('move to', 0.9), ('drive to', 1.0), ('head to', 1.0), ('go near', 1.0), ('go by', 0.8), ('walk to', 1.0)],
    'charge': [('go charge', 1.0), ('go and charge', 1.0), ('charge yourself', 1.0), ('go to the charger', 1.0), ('go to your charger', 1.0), ('recharge', 1.0), ('time to charge', 1.0), ('go to the dock', 1.0), ('charge now', 1.0), ('go and recharge', 1.0)],
    'who_here': [("who's home", 1.0), ("who's here", 1.0), ('who is home', 1.0), ('who is here', 1.0), ('is anyone home', 1.0), ('is anybody home', 1.0), ('is there anyone', 1.0), ('who do you see', 1.0), ('is there a child', 1.0), ('where are the kids', 0.8)],
    'enrol': [('remember this', 1.0), ('remember me', 1.0), ('remember him', 1.0), ('remember her', 1.0), ('learn my face', 1.0), ('learn my voice', 1.0), ('learn his face', 1.0), ('get to know', 0.9), ('remember my voice', 1.0)],
    'forget': [('forget', 0.8), ('delete', 1.0), ('erase', 1.0), ('remove', 0.8), ('delete everything about', 1.0)],
    'name_item': [('call this', 1.0), ('call it', 1.0), ('name this', 1.0), ('name it', 1.0), ('name the', 0.9), ('this is called', 1.0), ("let's call", 1.0)],
    'status': [('what are you doing', 1.0), ('what are you up to', 1.0), ('how much battery', 1.0), ('battery level', 1.0), ('how is your battery', 1.0), ('are you busy', 1.0), ("what's your status", 1.0)],
    'privacy': [('are you recording', 1.0), ('are you listening', 1.0), ('stop listening', 1.0), ('mute', 1.0), ('turn off the microphone', 1.0), ('privacy', 1.0), ('do you record', 1.0), ('are you always listening', 1.0)],
    'greet': [('hello', 1.0), ('good morning', 1.0), ('good evening', 1.0), ('good afternoon', 1.0), ('hi there', 1.0), ('good night', 0.9)],
    'identity': [('who are you', 1.0), ("what's your name", 1.0), ('what is your name', 1.0), ('introduce yourself', 1.0)],
    'help': [('what can you do', 1.0), ('help me', 0.8), ('how do you work', 1.0), ('what are you able to do', 1.0)],
    'come_here': [('come here', 1.0), ('come to me', 1.0), ('come over', 0.9), ('over here', 0.9), ('come closer', 1.0)],
    'profile_show': [('what do you know about', 1.0), ('show me the profile', 1.0), ('tell me about', 0.8), ('profile of', 1.0), ('what have you learned about', 1.0)],
    'set_lang': [('speak english', 1.0), ('speak chinese', 1.0), ('speak mandarin', 1.0), ('reply in english', 1.0), ('reply in chinese', 1.0), ('talk in english', 1.0), ('talk in chinese', 1.0)],
}
YES_EN = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'right', 'correct', 'agree', 'i agree', 'consent', 'i consent', 'go ahead', 'that is right', 'do it', 'please do', 'of course', 'absolutely', 'definitely', 'sounds good']
NO_EN = ['no', 'nope', 'not', 'wrong', 'no thanks', "don't", 'do not', 'disagree', 'i disagree', 'i do not agree', "that's wrong", 'not that', 'incorrect', 'no way']
STOP_EN = ['stop', 'stop it', 'stop that', 'stop now', 'freeze', 'halt', "don't move", 'do not move', 'wait', 'hold on', 'stay', 'help']
WAKE_PREFIX = ['hey', 'hi', 'ok', 'okay', 'hello', 'yo']
# the name 小拾 (xiao shi) as a recogniser of English may write it, in pinyin-like spelling and as English sounds
WAKE_NAMES_PINYIN = [['xiao', 'shi'], ['shao', 'shi'], ['siao', 'shi'], ['xiao', 'she'], ['shiao', 'shi'], ['xiao', 'si']]
WAKE_NAMES_PHONES = [('SH', 'AW', 'SH', 'IY'), ('SH', 'AW', 'SH', 'ER'), ('S', 'AW', 'SH', 'IY'), ('SH', 'IY', 'AW', 'SH', 'IY'), ('SH', 'AO', 'SH', 'IY'), ('SH', 'AW', 'S', 'IY')]

ITEM_EN = {'ball': ['ball', 'red ball', 'little ball'], 'cube': ['cube', 'block', 'wooden block'], 'car': ['toy car', 'car'], 'banana': ['toy banana', 'banana'],
           'lbrick': ['brick', 'building block', 'lego', 'l shaped brick'], 'rock': ['rock', 'stone', 'pebble'], 'sock': ['sock', 'socks'], 'plush': ['plush toy', 'teddy bear', 'teddy', 'stuffed animal', 'soft toy', 'doll', 'plush']}
PLACE_EN = {'沙发': ['sofa', 'couch'], '桌子': ['table', 'coffee table'], '架子': ['shelf', 'bookshelf', 'cabinet'], '盆栽': ['plant', 'flower pot', 'pot plant'], '椅子': ['chair', 'stool'], '箱子': ['box', 'carton'],
            '收纳筐': ['basket', 'toy basket', 'bin'], '充电座': ['charger', 'dock', 'charging dock', 'charging station']}
ITEM_NAME_EN = {'ball': 'ball', 'cube': 'cube', 'car': 'toy car', 'banana': 'toy banana', 'lbrick': 'L-shaped brick', 'rock': 'rock', 'sock': 'sock', 'plush': 'plush toy'}
PLACE_NAME_EN = {'沙发': 'sofa', '桌子': 'table', '架子': 'shelf', '盆栽': 'plant', '椅子': 'chair', '箱子': 'box', '收纳筐': 'basket', '充电座': 'charger'}
# words a recogniser of one language writes for a word of the other (code switching); both directions, the words that matter here
CS_EN_TO_ZH = {'ball': ['波', '玻'], 'sofa': ['搜发', '收发'], 'basket': ['八斯盖特', '巴斯克'], 'table': ['太博', '特博'], 'charger': ['差击', '察尔'], 'robot': ['肉不特'], 'sock': ['扫克', '搜克']}
CS_ZH_TO_EN = {'沙发': ['show far', 'sir far'], '小球': ['shaw chew', 'show chill'], '袜子': ['why see', 'what see'], '积木': ['gee moo', 'g moo'], '充电': ['chong dian', 'trunk dian'], '小拾': ['shaw she', 'show she']}


# ---------------------------------------------------------------------------------------------------------------- replies in two languages
def pick(zh, en, lang):
    """The text in the requested language. `lang` 'zh' or 'en'; mixed is answered in the language the caller decided on."""
    return en if lang == 'en' else zh


# ---------------------------------------------------------------------------------------------------------------- understanding English
_VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
STOP_SEQ = [('stop',), ('stop', 'it'), ('stop', 'that'), ('stop', 'now'), ('freeze',), ('halt',), ('do', 'not', 'move'), ('wait',), ('hold', 'on'), ('stay',), ('help',), ('emergency',)]
STOP_SKIP = {'please', 'robot', 'hey', 'hi', 'ok', 'okay', 'oh', 'uh', 'um'}
PERSON_EN = {'child', 'kid', 'kids', 'boy', 'girl', 'son', 'daughter', 'baby', 'mum', 'mom', 'mommy', 'dad', 'daddy', 'mother', 'father', 'grandma', 'grandpa', 'brother', 'sister', 'toddler'}
NAME_EN_RE = re.compile(r"(?:call (?:it|this|that|him|her|them)|named|name is|name's|called|call him|call her)\s+([A-Za-z][A-Za-z'\-]{1,14})", re.I)
LANG_EN_RE = re.compile(r'(speak|talk|reply|answer|respond|switch|use)\b[^.?!]{0,16}\b(english|chinese|mandarin)', re.I)


def _syl(ws):
    """Vowel count of the content words (function words count half): how long the phrase sounds, as the number of syllables does in Chinese."""
    c = sum(sum(1 for p in phones(w) if p in _VOWELS) or 1 for w in ws if w not in FUNCTION); f = sum(sum(1 for p in phones(w) if p in _VOWELS) or 1 for w in ws if w in FUNCTION)
    return max(1, int(round(c + 0.5 * f)))


class EnglishNLU:
    """The English half of the understanding: wake phrase, stop words, phrase triggers per intent, items and places (on phonemes), yes/no. `fkey` is the Chinese side's reduced-pinyin key (to read
    a Chinese name that came out spelled in pinyin) and `fz` the reduction of one syllable."""
    def __init__(self, fkey, fz, trig=None):
        self.fkey = fkey; self.fz = fz; self.trig = {i: [(tuple(en_tokens(p)), p, w, _syl(en_tokens(p))) for p, w in v] for i, v in (trig or TRIG_EN).items()}
        self.lex = [('item', k, w, tuple(en_tokens(w))) for k, ws in ITEM_EN.items() for w in ws] + [('place', k, w, tuple(en_tokens(w))) for k, ws in PLACE_EN.items() for w in ws]
        self.names_fz = {tuple(fkey_syl) for fkey_syl in ([self._fzs(v) for v in WAKE_NAMES_PINYIN])}

    def _fzs(self, syls): return tuple(self.fz(s) for s in syls)

    # ---- the wake phrase: 'hey <name>', or the name twice ('xiaoshi xiaoshi'); a lone 'xiaoshi' is not enough
    def is_name(self, toks):
        """'pinyin' if the words are the name spelled in pinyin ('xiaoshi', 'shao shi'), 'sound' if they sound like it ('show she'), else False."""
        j = ''.join(toks); sy = pinyin_segments(j)
        if sy and len(sy) == 2 and self._fzs(sy) in self.names_fz: return 'pinyin'
        ph = tuple(p for w in toks for p in phones(w))
        return 'sound' if (bool(ph) and any(ph_dist(ph, n) <= 0.3 for n in WAKE_NAMES_PHONES) and len(ph) >= 3) else False

    @staticmethod
    def _is_prefix(w):
        """'hey', 'hi', 'ok', 'hello', 'yo', or a word that sounds like one ('hay', 'high')."""
        return w in WAKE_PREFIX or any(len(phones(p)) >= 2 and phones(w) == phones(p) for p in WAKE_PREFIX)

    def wake_span(self, text):
        """End (character index in text) of an English wake phrase at the start of the text, else None."""
        ws = [(m.group().lower().strip("'-"), m.start(), m.end()) for m in _WORD.finditer(text.replace('’', "'"))][:7]
        for i in range(0, min(len(ws), 3)):
            if i and not all(self._is_prefix(w) for w, _, _ in ws[:i]): break
            for n in (1, 2, 3):
                seg = ws[i:i + n]
                if len(seg) < n: break
                route = self.is_name([w for w, _, _ in seg])
                if not route: continue
                if i > 0 or route == 'pinyin': return seg[-1][2]
                nxt = ws[i + n:i + 2 * n]
                if len(nxt) == n and self.is_name([w for w, _, _ in nxt]): return nxt[-1][2]
        return None

    # ---- the always-on stop spotter: the whole (short) utterance is a stop word
    def is_stop(self, text):
        t = [w for w in en_tokens(text) if w not in STOP_SKIP and latin_pinyin(w) is None]
        if not t or len(t) > 3: return False
        for seq in STOP_SEQ:
            if len(seq) == len(t) and all(a == b or phones(a) == phones(b) or (len(phones(a)) >= 3 and ph_dist(phones(a), phones(b)) <= (0.26 if len(phones(a)) >= 4 else 0.34)) for a, b in zip(seq, t)): return True
        return False

    # ---- items, places, people
    @staticmethod
    def _ethr(p):
        c = [w for w in p if w not in FUNCTION] or list(p)
        if len(c) == 1: return 0.99 if len(phones(c[0])) <= 3 else 0.85
        return 0.78

    def entities(self, toks, names=None):
        found = []
        for kind, ref, w, p in self.lex:
            sc, a, b = en_match(p, toks)
            if sc >= self._ethr(p) and b - a >= max(len(p) - 1, 1): found.append((kind, ref, w, float(sc), a, b))
        for n, (typ, ref) in (names or {}).items():
            if _HAN.search(n):                                                              # a Chinese name written out in pinyin ('xiao ming')
                want = tuple(self.fkey(n))
                for i in range(len(toks)):
                    for j in range(i + 1, min(len(toks), i + 3) + 1):
                        sy = pinyin_segments(''.join(toks[i:j]))
                        if sy and self._fzs(sy) == want: found.append((typ if typ == 'person' else 'item', ref, n, 0.9, i, j))
            else:
                p = tuple(en_tokens(n))
                if not p: continue
                sc, a, b = en_match(p, toks)
                if sc >= max(self._ethr(p), 0.85) and b - a >= max(len(p) - 1, 1): found.append((typ if typ == 'person' else 'item', ref, n, float(sc), a, b))
        found.sort(key=lambda f: (-(f[5] - f[4]), -f[3])); out = []
        for f in found:
            if not any(not (f[5] <= o[4] or f[4] >= o[5]) for o in out): out.append(f)
        return sorted(out, key=lambda f: f[4])

    # ---- intents
    def triggers(self, toks):
        """{intent: (value, phrase)} for every intent with a phrase in the words. value = match * weight * length factor, as on the Chinese side."""
        out = {}
        for intent, trigs in self.trig.items():
            best = (0.0, None)
            for p, phrase, w, n in trigs:
                sc, a, b = en_match(p, toks)
                if intent == 'stop' and (a > 1 or len(toks) > 3): continue                    # 'bus stop', 'stop by the shop': a stop word counts at the start of a short sentence
                if sc >= self._ethr(p) - (0.07 if len(p) >= 3 else 0.0):
                    v = sc * w * (0.62 + 0.13 * min(n, 3))
                    if v > best[0]: best = (v, phrase)
            if best[1] is not None: out[intent] = best
        return out

    # ---- sentence frames: a verb of a kind and a thing of a kind, whatever the exact words ("get the teddy into the basket", "walk over to the couch"). They catch what the phrase lists
    # did not think of, and score below a phrase that matches.
    PICK = {'put', 'get', 'pick', 'throw', 'drop', 'place', 'bring', 'take', 'collect', 'toss', 'fetch', 'grab', 'carry', 'move', 'tidy', 'clean', 'clear', 'straighten', 'stash', 'tuck'}
    MOVE = {'go', 'walk', 'head', 'drive', 'move', 'roll', 'travel', 'proceed', 'navigate', 'get'}
    WHQ = {'where', 'whereabouts'}

    def frames(self, toks, slots):
        t = set(toks); out = {}; item = 'item' in slots; place = slots.get('place'); wh = bool(t & self.WHQ)
        def put(i, v):
            if v > out.get(i, 0.0): out[i] = v
        if item and not wh and t & self.PICK: put('fetch_item', 0.82)
        if (wh or t & {'seen', 'find', 'found', 'locate', 'missing'}) and item: put('where_is', 0.85)
        if place == '充电座' and (t & self.MOVE or t & {'charge', 'recharge', 'dock'}): put('charge', 0.88)
        elif place and place != '充电座' and t & self.MOVE and not wh: put('go_to', 0.82)
        if t & {'charge', 'recharge', 'charging'} and not (t & {'phone', 'phones', 'my', 'battery'}): put('charge', 0.85)
        if not item and t & {'toys', 'toy', 'floor', 'room', 'house', 'mess', 'things'} and t & {'tidy', 'clean', 'clear', 'cleared', 'collect', 'straighten', 'pick', 'picking', 'put', 'get', 'tidying'}: put('tidy', 0.8)
        if 'what' in t and t & {'see', 'floor', 'lying', 'around', 'there'} and not t & {'new', 'changed', 'moved', 'missing'}: put('list_items', 0.8)
        if t & {'show'} and t & {'around', 'floor', 'lying'}: put('list_items', 0.8)
        if t & {'changed', 'new', 'moved', 'missing', 'different', 'gone'} and t & {'anything', 'what', 'something', 'is', 'has', 'was', 'since'} and not item: put('what_new', 0.8)
        if (('who' in t) and t & {'home', 'here', 'around', 'house', 'there'}) or (t & {'anyone', 'anybody', 'somebody', 'someone', 'kids', 'children', 'child'} and t & {'home', 'here', 'around', 'in'}): put('who_here', 0.85)
        if t & {'scan', 'map', 'explore', 'mapping', 'scanning'} and t & {'house', 'home', 'room', 'rooms', 'apartment', 'flat', 'place'}: put('scan_home', 0.88)
        if t & {'english', 'chinese', 'mandarin'} and t & {'speak', 'talk', 'reply', 'answer', 'respond', 'switch', 'use', 'say', 'change', 'in', 'to'}: put('set_lang', 0.95)
        if t & {'hi', 'hello', 'hey', 'morning', 'afternoon', 'evening'} and len(toks) <= 3: put('greet', 0.9)
        if t & {'battery', 'status'} and not t & {'phone', 'phones'}: put('status', 0.8)
        return out

    def yes_no(self, toks):
        t = [w for w in toks if w not in ('please', 'thanks', 'thank', 'you', 'very', 'much', 'um', 'uh', 'well')]
        if not t or len(t) > 4: return None
        ys = {w for p in YES_EN for w in en_tokens(p)} | {'yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'right', 'correct', 'agree', 'consent', 'ahead', 'do', 'it', 'that', 'is'}
        yes = any(tuple(en_tokens(p)) == tuple(t) for p in YES_EN) or all(any(phones(w) == phones(y) for y in ys) for w in t) and t[0] in ('yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'right', 'correct', 'agree', 'consent', 'absolutely', 'definitely')
        no = any(tuple(en_tokens(p)) == tuple(t) for p in NO_EN) or t[0] in ('no', 'nope', 'wrong', 'incorrect')
        return 'no' if (no and not yes) else ('yes' if yes else None)


# ---------------------------------------------------------------------------------------------------------------- how a recogniser gets English wrong (for the simulated recogniser)
_REV = None
_PH_ALL = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
_ACCENT_SUB = {'TH': ['S', 'F', 'T'], 'DH': ['D', 'Z'], 'V': ['W', 'B'], 'R': ['L'], 'L': ['N', 'R'], 'AE': ['EH'], 'IH': ['IY'], 'Z': ['S'], 'JH': ['CH'], 'ZH': ['SH'], 'NG': ['N'], 'W': ['V']}
_FINAL_STOPS = {'P', 'T', 'K', 'B', 'D', 'G'}


def _rev():
    global _REV
    if _REV is None:
        _REV = {}
        for w, prons in _CMU.items():
            if not (2 <= len(w) <= 9 and w.isalpha()): continue
            for pr in prons[:1]: _REV.setdefault(tuple(re.sub(r'\d', '', p) for p in pr), []).append(w)
    return _REV


@functools.lru_cache(maxsize=20000)
def confusions(word, accent=False):
    """Words a recogniser might write for `word`: homophones, words one phoneme away (a substitution within vowels or within consonants twice as likely as across), and with accent=True the
    substitutions a Mandarin speaker of English makes (th -> s/f/t, v -> w, r -> l, a lost final stop). -> (homophones, neighbours)"""
    rev = _rev(); ph = phones(word)
    if not ph or not rev: return (), ()
    hom = tuple(w for w in rev.get(ph, []) if w != word.lower()); near = []
    def add(q, wt):
        for w in rev.get(tuple(q), []):
            if w != word.lower(): near.extend([w] * wt)
    for i, p in enumerate(ph):
        for q in _PH_ALL:
            if q != p: add(ph[:i] + (q,) + ph[i + 1:], 2 if ((p in _VOWELS) == (q in _VOWELS)) else 1)
        add(ph[:i] + ph[i + 1:], 1)
    for i in range(len(ph) + 1):
        for q in _PH_ALL: add(ph[:i] + (q,) + ph[i:], 1)
    if accent:
        for i, p in enumerate(ph):
            for q in _ACCENT_SUB.get(p, []): add(ph[:i] + (q,) + ph[i + 1:], 6)
        if ph and ph[-1] in _FINAL_STOPS: add(ph[:-1], 6)
    return hom, tuple(near)


# every word that names an item or a place (so that a name is not mistaken for one: "call this brick Bruno")
WORDS_OF_THINGS = {w for ws in list(ITEM_EN.values()) + list(PLACE_EN.values()) for p in ws for w in p.split()} | {'toy', 'toys', 'thing', 'one'}
