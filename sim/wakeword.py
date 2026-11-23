"""The always-on part of the voice (round 16): spotting the wake word and the stop words in the robot's own audio, on the robot, with no network.

Until now the robot found the wake word in the TEXT the (simulated) recogniser wrote, which is how the language side could be tested without sound, and which a real robot must not do: a
recogniser that runs all day on everything it hears is a recording device, and its text is wrong about the name of the robot in exactly the cases that matter (a noisy room, a child, an accent).
A small keyword spotter listens instead; only after it fires is the sentence recorded and read (by the recogniser, and, if the owner has allowed it, by a language model: llm.py).

  SherpaWake   the real thing: sherpa-onnx's streaming keyword spotter with its bilingual (Chinese + English) 3 M-parameter model, run on the robot's CPU. The words it listens for are given
               as TEXT, not trained: '小拾小拾', 'Hey Xiaoshi', '停下来'. A different name for the robot is a different string.
  keyword_lines(...)  turns the phrases into the file the model reads (pinyin with tones, English phones).
  evaluate(...)  a harness to run on YOUR recordings (a folder of 16 kHz wav files of people saying the phrase, and one of everything else): hits, false alarms per hour, over a range of
               thresholds. The thresholds below are starting points; they have to be set on the voices and the room they will be used in.

What has been verified here, and what not (this repository has no recording of anybody): the model loads, the phrases become the right tokens, a phrase is spotted in a real recording (the
model's own English sample), silence and other speech give no false alarm, the refractory time, the stop words that go through while the wake word is held off. NOT verified: how often it
wakes for '小拾小拾' said by real people in a real room, or how often it wakes by mistake in a day of television: espeak-ng speech does not trigger the model at all (0 of 16 here), so a
synthesiser cannot stand in for a person; that measurement is for the harness and the real hardware. `fetch_model()` downloads the model (33 MB, from the sherpa-onnx project's releases).
"""
import glob
import os
import subprocess
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20'
MODEL_DIR = os.environ.get('WAKE_MODEL_DIR') or os.path.join(HERE, '..', 'third_party', 'sherpa-kws', MODEL_NAME)
MODEL_URL = f'https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/{MODEL_NAME}.tar.bz2'
SR = 16000

# (name, text or explicit tokens): the robot's name is 小拾 (xiao shi); four syllables on purpose: 小拾 alone sounds like 小时 (hour) and would wake it in every other sentence.
WAKE = [('xiaoshi', '小拾小拾'), ('nihao_xiaoshi', '你好小拾'), ('hey_xiaoshi', 'HH EY1 SH AW1 SH IY1')]            # 'Xiaoshi' is not in the English lexicon: its sounds are given
STOP = [('stop_zh1', '停下来'), ('stop_zh2', '别动'), ('stop_zh3', '站住'), ('stop_en1', 'stop'), ('stop_en2', 'freeze')]
# per-kind starting points: (boost, threshold). A longer phrase can have a lower threshold than a short one; the stop words are few sounds and so are held to more.
DEFAULTS = dict(wake=(2.0, 0.25), stop=(2.0, 0.30))


def have_runtime():
    try: import sherpa_onnx  # noqa: F401
    except Exception: return False
    return True


def have_model(model_dir=MODEL_DIR): return os.path.exists(os.path.join(model_dir, 'tokens.txt')) and os.path.exists(os.path.join(model_dir, 'en.phone'))


def fetch_model(dest=None):
    """Download and unpack the model (33 MB) next to the other third-party files (not in the repository: .gitignore). Returns the model directory."""
    dest = dest or os.path.dirname(MODEL_DIR); os.makedirs(dest, exist_ok=True); tgz = os.path.join(dest, MODEL_NAME + '.tar.bz2')
    subprocess.run(['curl', '-sSL', '-o', tgz, MODEL_URL], check=True); subprocess.run(['tar', 'xjf', tgz, '-C', dest], check=True); os.remove(tgz)
    return os.path.join(dest, MODEL_NAME)


def keyword_lines(phrases, kind='wake', model_dir=MODEL_DIR, score=None, threshold=None):
    """phrases: [(name, text)] -> the lines of the keyword file: tokens, ':boost', '#threshold', '@name'. A text of Chinese characters and words in the English lexicon is tokenised by
    sherpa-onnx (pinyin with tones; CMU phones); a text that is already tokens of the model (all of them in tokens.txt, as 'HH EY1 SH AW1 SH IY1') is taken as it is."""
    import sherpa_onnx
    tokens = os.path.join(model_dir, 'tokens.txt'); table = {l.split()[0] for l in open(tokens, encoding='utf-8') if l.strip()}
    sc, th = DEFAULTS[kind]; sc = sc if score is None else score; th = th if threshold is None else threshold; lines = []
    for name, text in phrases:
        parts = text.split()
        if len(parts) > 1 and all(p in table for p in parts) and not any('一' <= c <= '鿿' for c in text): toks = parts
        else:
            res = sherpa_onnx.text2token([text.upper()], tokens=tokens, tokens_type='phone+ppinyin', lexicon=os.path.join(model_dir, 'en.phone'))      # the English lexicon is in capitals
            out = res[0] if res else None
            if not out or any(t not in table for t in out): raise ValueError(f'cannot tokenise {text!r}: give its sounds as tokens of the model')
            toks = out
        lines.append(' '.join(toks) + f' :{sc} #{th} @{name}')
    return lines


class SherpaWake:
    """Streaming wake-word and stop-word spotter. `feed(samples)` takes mono 16 kHz audio of any chunk size and returns the events that fired in it: dicts name, kind ('wake' or 'stop'), t (s from
    the start of the stream). A keyword that has fired is held off for `refractory_s`. `suppress(until)` holds off the WAKE words (not the stop words) until a time: the robot's own voice comes
    back through the microphones; a real robot needs echo cancellation in the microphone chip as well, so that a person can still say 'stop' while it talks."""
    def __init__(self, model_dir=MODEL_DIR, wake=WAKE, stop=STOP, chunk=16, threads=1, refractory_s=1.5, quantized=True, wake_cfg=None, stop_cfg=None):
        import sherpa_onnx
        self.kind = {}; lines = []
        for k, ph, cfg in (('wake', wake, wake_cfg or {}), ('stop', stop, stop_cfg or {})):
            ls = keyword_lines(ph, k, model_dir, **cfg); lines += ls
            for name, _ in ph: self.kind[name] = k
        self.keywords = '\n'.join(lines); self.refractory = refractory_s; self.chunk = chunk
        kf = os.path.join(model_dir, '.keywords.txt'); open(kf, 'w', encoding='utf-8').write(self.keywords + '\n')
        q = '.int8' if quantized else ''; m = lambda n, s='': os.path.join(model_dir, f'{n}-epoch-13-avg-2-chunk-{chunk}-left-64{s}.onnx')
        self.ks = sherpa_onnx.KeywordSpotter(tokens=os.path.join(model_dir, 'tokens.txt'), encoder=m('encoder', q), decoder=m('decoder'), joiner=m('joiner', q), keywords_file=kf, num_threads=threads,
                                             keywords_score=DEFAULTS['wake'][0], keywords_threshold=DEFAULTS['wake'][1])
        self.reset()

    def reset(self): self._new_stream(); self.hold = -1.0

    def _new_stream(self): self.st = self.ks.create_stream(); self.n = 0; self.last = {}

    def suppress(self, until): self.hold = max(self.hold, until)

    def feed(self, x):
        x = np.asarray(x, np.float32); ev = []
        self.st.accept_waveform(SR, x); self.n += len(x)
        while self.ks.is_ready(self.st):
            self.ks.decode_stream(self.st); r = self.ks.get_result(self.st)
            if not r: continue
            self.ks.reset_stream(self.st); t = self.n / SR; k = self.kind.get(r, 'wake')
            if t - self.last.get(r, -99.0) < self.refractory: continue
            self.last[r] = t
            if k == 'wake' and t < self.hold: continue
            ev.append(dict(name=r, kind=k, t=round(t, 2)))
        return ev

    def run(self, x, chunk_s=0.1, tail_s=1.0):
        """All the events in a whole recording (with a second of silence after it so that the last word is finished)."""
        self._new_stream(); x = np.concatenate([np.asarray(x, np.float32), np.zeros(int(tail_s * SR), np.float32)]); ev = []; n = int(chunk_s * SR)       # a new stream; a hold from suppress() stays
        for i in range(0, len(x), n): ev += self.feed(x[i:i + n])
        return ev


def read_wav(fn):
    w = wave.open(fn); n = w.getnframes(); a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if w.getnchannels() > 1: a = a.reshape(-1, w.getnchannels()).mean(1)
    if w.getframerate() != SR:
        import scipy.signal as sg; a = sg.resample_poly(a, SR, w.getframerate())
    return a.astype(np.float32)


def evaluate(pos_dir, neg_dir, name, thresholds=(0.1, 0.15, 0.2, 0.25, 0.3, 0.4), model_dir=MODEL_DIR, phrases=None, kind='wake'):
    """On your recordings: pos_dir = wav files each with one person saying the phrase `name` (with whatever they would say after it); neg_dir = wav files of everything else (conversation, the
    television, music, the robot's own voice, the vacuum cleaner). For every threshold: the share of positives in which `name` fired, and the false alarms per hour of negatives."""
    phrases = phrases or [p for p in (WAKE if kind == 'wake' else STOP)]; pos = sorted(glob.glob(os.path.join(pos_dir, '*.wav'))); neg = sorted(glob.glob(os.path.join(neg_dir, '*.wav')))
    P = [read_wav(f) for f in pos]; N = [read_wav(f) for f in neg]; hours = sum(len(a) for a in N) / SR / 3600.0; rows = []
    for th in thresholds:
        sp = SherpaWake(model_dir, wake=phrases if kind == 'wake' else [], stop=phrases if kind == 'stop' else [], **({'wake_cfg': dict(threshold=th)} if kind == 'wake' else {'stop_cfg': dict(threshold=th)}))
        hit = sum(any(e['name'] == name for e in sp.run(a)) for a in P); fa = sum(sum(1 for e in sp.run(a) if e['name'] == name) for a in N)
        rows.append(dict(threshold=th, hit_rate=round(hit / max(len(P), 1), 3), false_alarms=fa, false_alarms_per_hour=round(fa / hours, 2) if hours else None, positives=len(P), negative_hours=round(hours, 3)))
    return rows
