"""Person profiles (人物画像): what the robot keeps about a person it was allowed to remember, so that it can answer in their language, greet them by name, know which toy is theirs and when they are usually
around. Local, opt-in, small, readable, deletable.

What is kept (a handful of counters, no text):
  languages    how many of their utterances were Chinese, English or mixed -> the language to answer in (or what they asked for: "speak English")
  hours/zones  when and where in the home they were seen (24 hour-bins x zone), counts that fade with a half-life of 30 days
  intents      how often they asked for each kind of thing (where is it, tidy up, ...), top few
  items        things they said were theirs ("that's my ball"); only by their own statement
  flags        a voiceprint exists / a face vector exists (the vectors themselves live in speaker.py / people.py), who consented and when
What is not kept: audio, transcripts, pictures, anything about health, mood, religion, politics, finances, sex or age guessed from a voice or a face, and no 'score' of any kind about a person.
Children: only the name, the language and the two flags - no hours, no intents, no items - unless a guardian switches the habit learning on for that child.
Control: anyone can ask what is kept about them (a card in their language), the owner can ask about anyone; "forget X" deletes everything about X; "stop learning habits" turns the hours and intents off; the whole
file is one JSON that can be read, exported and deleted; nothing leaves the robot.
What the simulator can and cannot say: the counters, the decay and the rules are real code; how well they describe real people is not tested here - the multi-day household in test_sound.py profile is a
made-up family, and its only purpose is to check the logic (does the language preference settle, do the hours converge, do attribution errors from the audio-visual stage wash out, is nothing kept that must not be).
"""
import json
import os

from store import LOG_MAX, dump_json, load_log

DAY = 86400.0
HALF_LIFE_DAYS = 30.0
PURGE_DAYS = 120.0
ZONES = ['living room', 'kitchen', 'bedroom', 'hall', 'study']
ZONE_ZH = {'living room': '客厅', 'kitchen': '厨房', 'bedroom': '卧室', 'hall': '门厅', 'study': '书房'}
INTENT_EN = dict(where_is='asks where things are', tidy='asks to tidy up', fetch_item='asks to put something away', list_items="asks what is on the floor", what_new='asks what changed', scan_home='asks for a house scan',
                 go_to='sends the robot somewhere', charge='sends the robot to charge', who_here='asks who is home', status='asks what the robot is doing', come_here='calls the robot over', greet='says hello')
INTENT_ZH = dict(where_is='问东西在哪', tidy='让收拾玩具', fetch_item='让把东西放好', list_items='问地上有什么', what_new='问有什么变化', scan_home='让扫描家里', go_to='让去某个地方', charge='让去充电',
                 who_here='问谁在家', status='问在做什么', come_here='叫过来', greet='打招呼')
TRACKED_INTENTS = tuple(INTENT_EN)


def _hour(t, tz_h=0.0): return int(((t + 3600.0 * tz_h) % DAY) // 3600)                       # the hour of the day in the household's time zone (the simulator's clock starts at local midnight)


class Profiles:
    def __init__(self, path=None, half_life_days=HALF_LIFE_DAYS, tz_offset_h=0.0):
        """tz_offset_h: hours the household's clock is ahead of the clock that times are given in (UTC for a real clock; 0 in the simulator)."""
        self.path = path; self.half = half_life_days; self.people = {}; self.log = []; self._t_decay = None; self.tz = tz_offset_h
        if path and os.path.exists(path):
            try:
                with open(path) as f: d = json.load(f)
                self.people = d['people']; self._t_decay = d.get('t_decay'); self.log = load_log(d)                    # where the fading had got to (a restart must not fade everything again from the epoch); the log of the sessions before
            except (OSError, ValueError, KeyError, TypeError): pass

    # ---------------------------------------------------------------- creating and deleting (the consent record is in the registries; here only that it exists)
    def enrol(self, name, role, consent_by, t=0.0, voice=False, face=False, aliases=(), habits=None):
        if not consent_by: raise PermissionError('no consent given')
        habits = (role == 'adult') if habits is None else bool(habits)                          # a child's habits are not learned unless a guardian says so
        p = self.people.setdefault(name, dict(role=role, created=round(t, 1)))
        p.update(role=role, consent_by=consent_by, habits=habits, voice=bool(voice), face=bool(face), aliases=sorted(set(list(p.get('aliases', [])) + list(aliases))), lang_set=p.get('lang_set'),
                 langs=p.get('langs', dict(zh=0, en=0, mixed=0)), n_utt=p.get('n_utt', 0), first_t=p.get('first_t', round(t, 1)), last_t=round(t, 1))
        if habits: p.setdefault('hours', [0.0] * 24); p.setdefault('zones', {}); p.setdefault('intents', {}); p.setdefault('items', [])
        else:
            for k in ('hours', 'zones', 'intents', 'items'): p.pop(k, None)
        self.log.append((round(t, 1), 'enrol', name)); self.save(); return p

    def forget(self, name, t=0.0):
        ok = self.people.pop(name, None) is not None
        if ok:
            self.log = [e for e in self.log if name not in e[2:]]                                  # the name goes from the log as well as from the records
            for p in self.people.values():
                if p.get('consent_by') == name: p['consent_by'] = '(forgotten)'
            self.log.append((round(t, 1), 'forget', None)); self.save()
        return ok

    def forget_all(self, t=0.0):
        n = len(self.people); self.people = {}; self.log = [e for e in self.log if not any(isinstance(v, str) for v in e[2:])]; self.log.append((round(t, 1), 'forget_all', n)); self.save(); return n

    def set_habits(self, name, on, t=0.0):
        p = self.people.get(name)
        if p is None: return False
        p['habits'] = bool(on)
        if on: p.setdefault('hours', [0.0] * 24); p.setdefault('zones', {}); p.setdefault('intents', {}); p.setdefault('items', [])
        else:
            for k in ('hours', 'zones', 'intents', 'items'): p.pop(k, None)
        self.log.append((round(t, 1), 'habits_on' if on else 'habits_off', name)); self.save(); return True

    def set_language(self, name, lang, t=0.0):
        p = self.people.get(name)
        if p is None: return False
        p['lang_set'] = lang; self.save(); return True

    def save(self):
        self.log = self.log[-LOG_MAX:]
        if self.path: dump_json(self.path, dict(people=self.people, log=self.log, t_decay=self._t_decay))

    # ---------------------------------------------------------------- learning (only about people who were enrolled)
    def observe_utterance(self, name, lang, intent=None, t=0.0, item=None):
        """One thing a person said: its language ('zh'|'en'|'mixed') and, if they have habit learning on, what they asked for."""
        p = self.people.get(name)
        if p is None or lang not in ('zh', 'en', 'mixed'): return False
        p['langs'][lang] = p['langs'].get(lang, 0) + 1; p['n_utt'] += 1; p['last_t'] = round(t, 1)
        if p.get('habits') and intent in TRACKED_INTENTS: p['intents'][intent] = p['intents'].get(intent, 0.0) + 1.0
        return True

    def observe_presence(self, name, t, zone):
        p = self.people.get(name)
        if p is None or not p.get('habits'): return False
        p['hours'][_hour(t, self.tz)] += 1.0; p['zones'][zone] = p['zones'].get(zone, 0.0) + 1.0; p['last_t'] = round(t, 1); return True

    def claim_item(self, name, item, t=0.0):
        """'That's my ball': only the person's own statement makes a thing theirs."""
        p = self.people.get(name)
        if p is None or not p.get('habits'): return False
        if item not in p['items']: p['items'].append(item)
        return True

    def owner_of(self, item):
        for n, p in self.people.items():
            if item in p.get('items', []): return n
        return None

    def decay(self, t):
        """Counts fade with a half-life (the first call only sets where the fading starts); the hours, rooms and intents of a person not seen or heard for PURGE_DAYS are cleared altogether
        (the languages and the things they said were theirs stay: those are not habits). The file is written only when something in it changed: a household with no habit counters
        does not have its file rewritten every hour."""
        changed = False
        if self._t_decay is None: self._t_decay = t; changed = True
        dt = (t - self._t_decay) / DAY
        if dt > 0:
            f = 0.5 ** (dt / self.half); self._t_decay = t
            for p in self.people.values():
                if p.get('hours') and any(p['hours']): p['hours'] = [h * f for h in p['hours']]; changed = True
                for d in ('zones', 'intents'):
                    if p.get(d): p[d] = {k: v * f for k, v in p[d].items() if v * f > 0.05}; changed = True
        for n, p in self.people.items():
            if p.get('habits') and t - p.get('last_t', t) > PURGE_DAYS * DAY and (any(p.get('hours') or []) or p.get('zones') or p.get('intents')):
                p['hours'] = [0.0] * 24; p['zones'] = {}; p['intents'] = {}; self.log.append((round(t, 1), 'habits_purged', None)); changed = True
        if changed: self.save()

    # ---------------------------------------------------------------- what is known
    def lang_pref(self, name):
        """The language to answer in: what they asked for, else the language they mostly use (needs 8 utterances and 70 % of them, mixed counting half); mixed counts for neither; None when there is no basis."""
        p = self.people.get(name)
        if p is None: return None
        if p.get('lang_set'): return p['lang_set']
        L = p['langs']; n = sum(L.values())
        if n < 8: return None
        zh, en = L.get('zh', 0) + 0.5 * L.get('mixed', 0), L.get('en', 0) + 0.5 * L.get('mixed', 0)
        if zh / n >= 0.7: return 'zh'
        if en / n >= 0.7: return 'en'
        return None                                                                                      # a person who uses both: the answer follows each sentence, not a fixed language

    def usual_hours(self, name, share=0.6):
        """The smallest set of whole hours that holds `share` of the person's presence (a list of hours), or []."""
        p = self.people.get(name)
        if p is None or not p.get('hours') or sum(p['hours']) < 20: return []
        tot = sum(p['hours']); order = sorted(range(24), key=lambda h: -p['hours'][h]); acc = 0.0; out = []
        for h in order:
            acc += p['hours'][h]; out.append(h)
            if acc >= share * tot: break
        return sorted(out)

    @staticmethod
    def _ranges(hours):
        out = []; s = None; prev = None
        for h in hours:
            if s is None: s = prev = h
            elif h == prev + 1: prev = h
            else: out.append((s, prev + 1)); s = prev = h
        if s is not None: out.append((s, prev + 1))
        return out

    def top_zone(self, name):
        p = self.people.get(name)
        if p is None or not p.get('zones'): return None
        z = max(p['zones'], key=p['zones'].get); return z if p['zones'][z] >= 0.4 * sum(p['zones'].values()) else None

    def top_intents(self, name, k=3):
        p = self.people.get(name)
        if p is None or not p.get('intents'): return []
        return [i for i, v in sorted(p['intents'].items(), key=lambda kv: -kv[1])[:k] if v >= 3]

    def card(self, name, lang='en'):
        """What is kept about `name`, in words (en or zh)."""
        p = self.people.get(name)
        if p is None: return (f"I don't keep a profile of {name}." if lang == 'en' else f'我没有保存{name}的资料。') if name else ("I don't know whom you mean." if lang == 'en' else '你问的是谁？')
        L = p['langs']; n = max(sum(L.values()), 1); pref = self.lang_pref(name); kid = p['role'] == 'child'
        lname = dict(zh='Chinese', en='English', mixed='mixed'); lname_zh = dict(zh='中文', en='英文', mixed='中英混合')
        stored_en = ('a few counters' + (', a voiceprint vector' if p.get('voice') else '') + (', a face vector' if p.get('face') else '') + '. No recordings, no transcripts, no photos.')
        stored_zh = ('几个计数器' + ('、一个声纹向量' if p.get('voice') else '') + ('、一个人脸特征向量' if p.get('face') else '') + '。没有录音、没有文字记录、没有照片。')
        if lang == 'en':
            s = f'{name} ({"child" if kid else "adult"}). Languages: ' + ', '.join(f'{lname[k]} {round(100 * L.get(k, 0) / n)}%' for k in ('zh', 'en', 'mixed') if L.get(k, 0)) + (f' - I answer in {"Chinese" if pref == "zh" else "English"}.' if pref else ' - not enough to tell yet.')
            if p.get('habits'):
                hr = self._ranges(self.usual_hours(name)); z = self.top_zone(name); ti = self.top_intents(name)
                if hr: s += ' Usually around ' + ', '.join(f'{a:02d}:00-{b:02d}:00' for a, b in hr) + (f' in the {z}' if z else '') + '.'
                if ti: s += ' Often: ' + '; '.join(INTENT_EN[i] for i in ti) + '.'
                if p.get('items'): s += f' Their things: {len(p["items"])}.'
            elif kid: s += ' For a child I keep only the name, the language and that I can recognise them.'
            return s + f' Kept: {stored_en} Say "forget {name}" to delete all of it.'
        s = f'{name}（{"小孩" if kid else "大人"}）。语言：' + '、'.join(f'{lname_zh[k]} {round(100 * L.get(k, 0) / n)}%' for k in ('zh', 'en', 'mixed') if L.get(k, 0)) + (f'，我用{"中文" if pref == "zh" else "英文"}回答。' if pref else '，还不够判断。')
        if p.get('habits'):
            hr = self._ranges(self.usual_hours(name)); z = self.top_zone(name); ti = self.top_intents(name)
            if hr: s += '通常在 ' + '、'.join(f'{a}-{b} 点' for a, b in hr) + (f'出现在{ZONE_ZH.get(z, z)}' if z else '出现') + '。'
            if ti: s += '常常：' + '；'.join(INTENT_ZH[i] for i in ti) + '。'
            if p.get('items'): s += f'自己的东西：{len(p["items"])} 件。'
        elif kid: s += '小孩的资料我只留名字、语言，以及能不能认出他/她。'
        return s + f'保存的内容：{stored_zh}说“忘掉{name}”可以全部删除。'

    def export(self, name):
        """Everything kept about a person, as it is stored (for the person or the owner to read or take away)."""
        return json.loads(json.dumps(self.people.get(name))) if name in self.people else None
