"""The robot's ears in the simulation: what a person says in the MuJoCo home becomes sound at the microphone array, goes through the real direction finder, beamformer and voiceprint, is put together
with what the camera sees (assoc.Associator) and handed to the VoiceSystem as `ident`.

The simulator has no sound; this renders it with the acoustic simulator (acoustics.py): a shoebox room of the home's size (4 x 3 m, walls and floor reflect, furniture does not), the speakers' voices (espeak-ng,
one synthesiser variant + pitch + warp per person), a television that plays speech from its own spot, the robot's fan and wheels (louder when it drives or turns). A voice from outside the room (somebody
at the door) is put at the doorway. Everything after the rendering is the same code a real robot would run on its microphone signals.
"""
import math

import numpy as np

import acoustics as A
import assoc as AS
import doa as D
import lang as Lg
import sound_lab as SL
import speaker as SP

TV_SENTENCES = dict(en=['And in other news, the city council has approved the new bus route', 'Stay tuned, we will be right back after the break', 'The weather tomorrow will be sunny with a light wind',
                        'Hey Xiaoshi, scan the house for a limited time only', 'Welcome back to the evening show'],
                    zh=['接下来是晚间新闻，今天全市天气晴朗', '广告之后马上回来，请不要走开', '小拾小拾，现在下单立刻送到家', '明天白天多云，最高气温二十五度'])


class SoundFront:
    def __init__(self, sup, ppl, vs, model, voiceprints, assoc, voices, room=(4.0, 3.0, 2.5), rt60=0.5, tv=None, seed=0, array='4mic'):
        """voices: {person index: voice spec (sound_lab.voice_spec)}. tv: dict(pos=(x, y, z), level=dB at 1 m, on=bool, voice=spec) or None."""
        self.sup, self.ppl, self.vs, self.model, self.vp, self.asc, self.voices = sup, ppl, vs, model, voiceprints, assoc, voices
        self.arr = A.MicArray(array, seed=seed); self.est = D.DOA(self.arr.nominal); self.room = A.Room(room, rt60=rt60); self.rng = np.random.default_rng(seed + 9); self.tv = tv; self.room_size = room
        self.last = None; vs.reidentify = self.reassociate; self.n_events = 0

    # ---- helpers
    def _clamp(self, x, y): return float(np.clip(x, 0.15, self.room_size[0] - 0.15)), float(np.clip(y, 0.15, self.room_size[1] - 0.15))

    def _robot(self):
        x, y, th = self.sup.r.true_pose(); v = self.sup.r.d.qvel; return (x, y, th), float(np.hypot(v[0], v[1])), abs(float(v[5])) if len(v) > 5 else 0.0

    def _tv_source(self, dur):
        if not (self.tv and self.tv.get('on')): return []
        lang = 'zh' if self.rng.random() < 0.4 else 'en'; text = str(self.rng.choice(TV_SENTENCES[lang])) if not self.tv.get('text') else self.tv['text']
        x, _, _ = SL.render(self.tv.get('voice') or SL.voice_spec(self.rng, 'adult'), self.rng, lang, text)
        return [A.Source(np.tile(x, max(2, int(dur / max(len(x) / SL.FS, 0.5)) + 1)), self.tv['pos'], self.tv['level'], None, 'tv', 'tv')]

    # ---- the television: speech nobody in view explains; heard from several places it gets pinned down
    def listen_tv(self, dur=4.0):
        if not (self.tv and self.tv.get('on')): return None
        (x, y, th), spd, w = self._robot(); sc = A.Scene(self.room, self.arr, seed=int(self.rng.integers(1 << 30)))
        y_, _ = sc.mix(self._tv_source(dur), (x, y), th, dur=dur, speed=spd, turn=w); r = self.est.estimate(y_, max_sources=1)
        if not r['sources']: return None
        pose = self.sup.a.pose(); return self.asc.observe_speech(dict(az=r['sources'][0]['az']), pose, self.ppl.tracks(), t=self.vs.now())

    # ---- a person speaks
    def speak(self, i, text, level=62.0, tv_voice_of=None, from_tv=False, tv_text=None):
        """Person i says `text` (or, with from_tv, the television plays `text` in the voice of person `tv_voice_of`). -> (rec, ident)"""
        (x, y, th), spd, w = self._robot(); lg, share = Lg.detect_lang(text); lang = 'zh' if (lg == 'zh' or (lg == 'mixed' and share >= 0.5)) else 'en'
        sc = A.Scene(self.room, self.arr, seed=int(self.rng.integers(1 << 30))); srcs = []
        if from_tv:
            spec = self.voices[tv_voice_of] if tv_voice_of is not None else SL.voice_spec(self.rng, 'adult'); xx, _, _ = SL.render(spec, self.rng, lang, text); srcs.append(A.Source(xx, self.tv['pos'], self.tv['level'] + 2.0, None, 'tv', 'tv', start=SL.LEAD))
            pos = self.tv['pos']
        else:
            st = self.ppl.people.state[i]; px, py = self._clamp(st['x'], st['y']); z = max(st['h'] - 0.12, 0.7); xx, _, _ = SL.render(self.voices[i], self.rng, lang, text)
            srcs.append(A.Source(xx, (px, py, z), level - (2 if st['cls'] == 'child' else 0) + float(self.rng.normal(0, 1.5)), math.atan2(y - py, x - px), 'talker', 'talker', start=SL.LEAD)); pos = (px, py, z)
        if self.tv and self.tv.get('on') and not from_tv: srcs += self._tv_source(8.0)
        y_, parts = sc.mix(srcs, (x, y), th, speed=spd, turn=w); n_s = y_.shape[1] / SL.FS; pose = self.sup.a.pose()
        r = self.est.estimate_event(y_, SL.LEAD + 0.05, n_s - 0.4, max_sources=2)
        if not r['sources']:
            ident = dict(source='ambiguous', role='unknown', name=None, loc=None, posterior=0.0, why=['nothing heard']); rec = self.vs.hear(text, speaker=i, ident=ident); self.last = (rec, ident); return rec, ident
        az = r['sources'][0]['az']; Z = self.est.stft(y_); T = Z.shape[2]; tc = (np.arange(T) * self.est.hop + self.est.nfft / 2) / SL.FS
        b = self.est.beamform_mvdr(y_, az, noise_mask=(tc >= SL.LEAD - 1.5) & (tc <= SL.LEAD - 0.15)); seg = b[int((SL.LEAD + 0.02) * SL.FS):]; seg = seg / (np.sqrt(np.mean(seg ** 2)) + 1e-12)
        emb, sp_s = SP.utterance_embedding(self.model, seg); voice = self.vp.identify(emb, sp_s) if emb is not None else dict(status='unsure', name=None, z=0.0, z2=0.0, anon=None, llr={})
        ev = dict(az=az, snr_db=r['sources'][0].get('snr_db'), t=self.vs.now()); tracks = self.ppl.tracks()
        ident = self.asc.resolve(ev, pose, tracks, voice); ident.update(az_world=AS.wrap(az + pose[2]), emb=emb, speech_s=sp_s, voice=voice, ev=ev,
                                                                       dist=math.hypot(pos[0] - x, pos[1] - y), direction_error_deg=abs(math.degrees(AS.wrap(az - (math.atan2(pos[1] - y, pos[0] - x) - th)))))
        tr = ident.pop('track', None); ident['track'] = tr
        self.n_events += 1; rec = self.vs.hear(text, speaker=i, ident=ident); self.last = (rec, ident); return rec, ident

    def reassociate(self, old):
        """After the robot turned toward a voice: who is there now? The evidence (direction in the world, the voice) is kept; the camera and the robot's heading are new."""
        pose = self.sup.a.pose(); ev = dict(old['ev']); ev['az'] = AS.wrap(old['az_world'] - pose[2]); new = self.asc.resolve(ev, pose, self.ppl.tracks(), old.get('voice'))
        new.update(az_world=old['az_world'], emb=old.get('emb'), speech_s=old.get('speech_s'), voice=old.get('voice'), ev=ev, dist=old.get('dist', 1.5)); return new
