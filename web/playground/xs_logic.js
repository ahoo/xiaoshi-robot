/* xs_logic.js: the decision rules of the real system, written again in JavaScript so that they can be played with in a web page.
 *
 *   Battery   power.py        charge profile, voltage, the reading of the battery board (state of charge plus a drift)
 *   Guard     people.Guard    the safety layer between the modes and the wheels (halt, slow down, arm pause, "please make way")
 *   Voice     voice.py        wake word, stop word, intent from the pinyin of what was said, who may ask for what, asking back, confirming
 *
 * None of it runs the physics, the grasping or the real perception. What it does run is checked against what the real code answers (web/playground/data/model.json "golden", see test_core.mjs
 * and the "对照" tab of the page). */
(function (G) {
'use strict';
const XS = G.XS = G.XS || {};
const PI = Math.PI;
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const hypot = Math.hypot;
XS.clamp = clamp;

// ---------------------------------------------------------------------------------------------------------------- battery (power.py)
class Battery {
    constructor(B, soc = 1.0, bias = null, rnd = Math.random) {
        this.B = B; this.soc = soc; this.drawn_wh = 0; this.charged_wh = 0; this.load_w = B.p_run_w; this.dead = false; this.t_charging = 0;
        if (bias === null) { const u = Math.max(rnd(), 1e-9), v = rnd(); bias = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * PI * v) * B.est_bias_sigma; }
        this.bias = bias;
    }
    step(dt, load_w, charge_w = 0) {
        const B = this.B; this.load_w = load_w;
        const eOut = load_w * dt / 3600, eIn = charge_w * dt / 3600;
        this.drawn_wh += eOut; this.charged_wh += eIn;
        this.soc = Math.min(1, Math.max(0, this.soc + (eIn - eOut) / B.capacity_wh));
        if (charge_w > 0) this.t_charging += dt;
        if (this.soc <= B.soc_cutoff) this.dead = true;
    }
    get est() { return clamp(this.soc + this.bias, 0, 1); }
    voltage(load_w) {
        const B = this.B, w = load_w === undefined ? this.load_w : load_w;
        const ocv = B.v_empty + (B.v_full - B.v_empty) * this.soc;
        return ocv - B.r_int * w / Math.max(ocv, 1.0);
    }
    chargePower(supplied, docked) {
        const B = this.B; if (supplied === undefined) supplied = B.charger_w; if (docked === undefined) docked = B.p_docked_w;
        let p = supplied - docked;
        if (this.soc > B.cc_until) p *= Math.exp(-(this.soc - B.cc_until) / B.cv_taper);
        return Math.max(p, 0);
    }
    /** one step of the charging fast-forward: `stepS` seconds of charging; false when the taper has reached nothing */
    chargeStep(stepS, docked, servoIdle) {
        const B = this.B; if (docked === undefined) docked = B.p_docked_w; if (servoIdle === undefined) servoIdle = B.servo_idle_w;
        const pIn = this.chargePower(undefined, docked + servoIdle);
        if (pIn < 0.5) return false;
        this.soc = Math.min(1, this.soc + pIn * stepS / 3600 / B.capacity_wh); this.charged_wh += pIn * stepS / 3600;
        this.load_w = docked + servoIdle; this.t_charging += stepS; return true;
    }
    chargeUntil(target, docked, servoIdle, stepS = 5, maxS = 8 * 3600) {
        let t = 0;
        while (this.soc < target && t < maxS) { if (!this.chargeStep(stepS, docked, servoIdle)) break; t += stepS; }
        return t;
    }
}
XS.Battery = Battery;

// ---------------------------------------------------------------------------------------------------------------- the safety layer (people.Guard)
/** What the guard needs from the robot, `S`: S.t, S.mode, S.state, S.armBusy, S.armDone(), S.pose() -> [x, y, th], S.zoneSkipT, S.tracks() -> [{cls, x, y, vx, vy, coast, predict(t)}],
 *  S.wheelTarget {vx, vy, wz} (body frame), S.setTwist(0, 0, 0), S.holdArm (written), S.dynZones / S.personZones (written), S.event(text), S.speak(text), S.onAsk(x, y). */
class Guard {
    constructor(cfg) {
        this.c = Object.assign({}, cfg); this.halted = false; this._clearT = 0; this.stats = { halt_s: 0, slow_s: 0, halts: 0, arm_pauses: 0, max_scale_cut: 0, blocked_s: 0 };
        this.why = ''; this.scale = 1; this._haltT = 0; this.mem = new Map(); this.episodes = []; this._zoneAskT = -99; this._lastSay = -99; this.dt = 0.05;
    }
    _people(S) {
        const out = [], [x, y, th] = S.pose(), c = this.c;
        for (const k of S.tracks()) {
            const kid = k.cls === 'child' || k.cls === 'unsure';
            const [px, py] = k.coast ? k.predict(S.t) : [k.x, k.y];
            const d = hypot(px - x, py - y), gap = d - c.robot_r - c.person_r;
            const clos = -((px - x) * k.vx + (py - y) * k.vy) / Math.max(d, 1e-3);
            out.push({ k, kid, gap, bearing: Math.atan2(py - y, px - x) - th, closing: clos });
        }
        return out;
    }
    pre(S) {
        const c = this.c;
        if (S.mode === 'RIGHT' || S.mode === 'FAULT' || S.mode === 'TELEOP') { this.halted = false; S.holdArm = false; S.personZones = []; S.dynZones = []; return false; }
        const ppl = this._people(S), armWork = ['抓取', '放置', '放下'].includes(S.state) || S.armBusy || !S.armDone(0.08), t = S.t;
        S.dynZones = ppl.filter(p => !p.k.coast).map(p => [p.k.x, p.k.y, p.kid ? 0.25 : 0.32]);
        for (const p of ppl) { const k = p.k; if (!k.coast) this.mem.set(k.id, [k.x, k.y, t, p.kid, hypot(k.vx, k.vy) < 0.25]); }
        for (const [id, m] of Array.from(this.mem)) if (!(t - m[2] < (m[4] ? c.mem_still : c.mem_moving))) this.mem.delete(id);
        S.personZones = Array.from(this.mem.values()).map(m => [m[0], m[1], m[3] ? c.zone_child : c.zone]);
        let trig = false; this.why = '';
        if (S.mode === 'TASK' && S.t - S.zoneSkipT < 1.5 && S.t - this._zoneAskT > 20.0) { this._zoneAskT = S.t; this._say(S, '请让一让，我想拿您旁边的玩具', 0.0); }
        for (const p of ppl) {
            const reach = p.kid ? c.arm_reach_child : c.arm_reach;
            if (p.gap < 0.2) { trig = true; this.why = `${p.k.cls} at ${p.gap.toFixed(2)} m`; }
            if (p.gap < reach && armWork) { trig = true; this.why = `arm, ${p.k.cls} ${p.gap.toFixed(2)} m`; }
            if (p.closing > c.close_speed && p.gap < (p.kid ? 1.2 : 0.9)) { trig = true; this.why = `${p.k.cls} coming fast ${p.closing.toFixed(1)} m/s`; }
        }
        if (trig) {
            if (!this.halted) { this.stats.halts += 1; this.stats.arm_pauses += armWork ? 1 : 0; this._say(S, `有人在附近（${this.why}），先停一下`); this.episodes.push([Math.round(S.t * 10) / 10, null, this.why]); }
            this.halted = true; this._clearT = S.t;
        } else if (this.halted && S.t - this._clearT > c.release) {
            this.halted = false;
            if (this.episodes.length && this.episodes[this.episodes.length - 1][1] === null) this.episodes[this.episodes.length - 1][1] = Math.round(S.t * 10) / 10;
        }
        S.holdArm = !!(this.halted && armWork);
        if (this.halted) {
            this.stats.halt_s += this.dt; S.setTwist(0, 0, 0); this._haltT += this.dt;
            if (this._haltT > c.ask_after) this._say(S, '请让一让，我要过去拿东西', 15.0);
        } else this._haltT = 0;
        return this.halted;
    }
    post(S) {
        const c = this.c; this.scale = 1.0;
        if (S.mode === 'RIGHT' || S.mode === 'FAULT' || this.halted) return;
        const w = S.wheelTarget, v = hypot(w.vx, w.vy);
        if (v < 1e-3) return;
        let blocked = false;
        for (const p of this._people(S)) {
            const slow = p.kid ? c.slow_child : c.slow, halt = p.kid ? c.halt_child : c.halt;
            if (p.gap > slow) continue;
            const toward = Math.cos(Math.atan2(w.vy, w.vx) - p.bearing);
            if (toward < 0.2) continue;
            if (p.gap <= halt) { this.scale = 0.0; blocked = true; break; }
            let vmax = p.kid ? c.v_crawl_child : c.v_crawl;
            vmax = Math.max(vmax, Math.min(v, Math.sqrt(2 * c.brake * Math.max(p.gap - halt, 0.0))));
            vmax = p.gap > halt + 0.25 ? Math.min(vmax, v) : Math.min(vmax, p.kid ? c.v_crawl_child : c.v_crawl);
            this.scale = Math.min(this.scale, vmax / v);
        }
        if (this.scale < 1.0) {
            S.wheelTarget = { vx: w.vx * this.scale, vy: w.vy * this.scale, wz: w.wz };
            this.stats.slow_s += this.dt; this.stats.max_scale_cut = Math.max(this.stats.max_scale_cut, 1 - this.scale);
            if (blocked) this.stats.blocked_s += this.dt;
        }
    }
    _say(S, text, every = 10.0) {
        if (S.t - this._lastSay < every) return;
        this._lastSay = S.t; S.event(text); if (S.speak) S.speak(text);
        if (S.onAsk && text.includes('请让一让')) { const p = S.pose(); S.onAsk(p[0], p[1]); }
    }
}
XS.Guard = Guard;

/** The size class of a person from a height and a shoulder width, as people.classify */
XS.classify = function (h, w, lowerBound, K) {
    if (lowerBound) return h >= K.adult_from ? 'adult' : 'unsure';
    if (h >= K.adult_from) return 'adult';
    if (h < K.child_below) return 'child';
    return (w !== null && w !== undefined && w >= K.adult_width) ? 'adult' : 'unsure';
};

// ---------------------------------------------------------------------------------------------------------------- voice (voice.py)
const INIT = [['zh', 'z'], ['ch', 'c'], ['sh', 's']];
const FIN = [['iang', 'ian'], ['uang', 'uan'], ['ang', 'an'], ['eng', 'en'], ['ing', 'in']];
function fz(s) {
    for (const [a, b] of INIT) if (s.startsWith(a)) { s = b + s.slice(a.length); break; }
    if (s.length > 1 && 'nf'.includes(s[0]) && 'aeiouüv'.includes(s[1])) s = (s[0] === 'n' ? 'l' : 'h') + s.slice(1);
    for (const [a, b] of FIN) if (s.endsWith(a) && s.length > a.length - 1) { s = s.slice(0, s.length - a.length) + b; break; }
    return s;
}
const INITRE = /^[bpmfdtlnghjqxzcsryw]*/;
function sd(a, b) {
    if (a === b) return 0.0;
    const ia = INITRE.exec(a)[0], ib = INITRE.exec(b)[0], fa = a.slice(ia.length), fb = b.slice(ib.length);
    return ((ia === ib && ia) || (fa === fb && fa)) ? 0.5 : 1.0;
}
/** Best approximate occurrence of the syllable list `pat` in `text`: [score 0..1, start, end] (voice.match) */
function match(pat, text, strict = false) {
    const m = pat.length, n = text.length;
    if (m === 0 || n === 0) return [0.0, 0, 0];
    const D = [], S = [];
    for (let i = 0; i <= m; i++) { D.push(new Float64Array(n + 1)); S.push(new Int32Array(n + 1)); D[i][0] = i; }
    for (let j = 0; j <= n; j++) S[0][j] = j;
    for (let i = 1; i <= m; i++) for (let j = 1; j <= n; j++) {
        const sub = strict ? (pat[i - 1] === text[j - 1] ? 0.0 : 1.0) : sd(pat[i - 1], text[j - 1]);
        const c0 = D[i - 1][j - 1] + sub, c1 = D[i - 1][j] + 1.0, c2 = D[i][j - 1] + 1.0;
        let k = 0, best = c0; if (c1 < best) { best = c1; k = 1; } if (c2 < best) { best = c2; k = 2; }
        D[i][j] = best; S[i][j] = k === 0 ? S[i - 1][j - 1] : (k === 1 ? S[i - 1][j] : S[i][j - 1]);
    }
    let j = 1; for (let jj = 2; jj <= n; jj++) if (D[m][jj] < D[m][j]) j = jj;
    return [Math.max(0.0, 1.0 - D[m][j] / m), S[m][j], j];
}
XS.fz = fz; XS.match = match;
const thr = n => n <= 2 ? 1.0 : (n === 3 ? 0.8 : (n === 4 ? 0.74 : 0.7));
const round = (v, d) => { const f = Math.pow(10, d); return Math.round(v * f) / f; };

class Voice {
    /** V: the "voice" part of model.json; py: its "pinyin" table; host: null (nothing connected: the language side alone, as the real code is tested) or the robot's hooks. */
    constructor(V, py, host) {
        this.V = V; this.host = host || null; this.ch = new Map();
        for (const [s, chars] of Object.entries(py)) for (const c of chars) this.ch.set(c, s);
        this.pinyinWords = new Set(Object.keys(py));
        this.trig = {};
        for (const [k, v] of Object.entries(V.trig)) this.trig[k] = v.map(([p, w]) => ({ p: this.fkey(p), phrase: p, w }));
        this.lex = [];
        for (const [k, ws] of Object.entries(V.items)) for (const w of ws) this.lex.push(['item', k, w, this.fkey(w)]);
        for (const [k, ws] of Object.entries(V.places)) for (const w of ws) this.lex.push(['place', k, w, this.fkey(w)]);
        this.wake = this.fkey(V.wake); this.wakeAlt = this.fkey(V.wake_alt); this.stopP = V.stop_words.map(w => this.fkey(w));
        this.yes = V.yes.map(w => [w, this.fkey(w)]); this.no = V.no.map(w => [w, this.fkey(w)]);
        this.t = 0; this.sessionUntil = -1; this.pending = null; this.ctx = { item: null, place: null, person: null };
        this.tts = []; this.log = []; this.actions = []; this.muted = false; this.resumeCmd = null; this.stats = { heard: 0, ignored: 0, reask: 0, confirm: 0, denied: 0, acted: 0, stop: 0 };
    }
    now() { return this.host ? this.host.now() : this.t; }
    advance(dt) { this.t += dt; }
    speak(text) { this.tts.push([round(this.now(), 1), text]); if (this.host && this.host.onSpeak) this.host.onSpeak(text); return text; }

    syl(text) { return Array.from(text).map(c => this.ch.get(c) || c); }
    fkey(text) { return this.syl(text).map(fz); }
    static clean(text) { return text.replace(/[\s，。！？、,.!?；;：:“”"'（）()…~～—-]/g, ''); }
    english(text) {
        return (text.match(/[A-Za-z][A-Za-z'\-]*/g) || []).filter(w => w.length >= 2 && !['hey', 'hi'].includes(w.toLowerCase()) && !this.isPinyinWord(w.toLowerCase()));
    }
    isPinyinWord(w) {                          // 'xiaoshi' is the robot's name spelled in pinyin, not English
        const n = w.length, ok = new Array(n + 1).fill(false); ok[0] = true;
        for (let i = 0; i < n; i++) if (ok[i]) for (let l = 1; l <= 6 && i + l <= n; l++) if (this.pinyinWords.has(w.slice(i, i + l))) ok[i + l] = true;
        return ok[n];
    }
    wakeAt(ft) {
        let best = [0.0, 0, 0];
        for (const w of [this.wake, this.wakeAlt]) { const r = match(w, ft.slice(0, 12), false); if (r[0] > best[0]) best = r; }
        return best;
    }
    hasWake(text) {
        const [sc, a] = this.wakeAt(this.fkey(Voice.clean(text)));
        return (sc >= 0.74 && a <= 4) || /\bhey[\s,]*(xiao|shao)\s*shi\b/i.test(text);
    }
    stripWake(text) {
        const t = Voice.clean(text), [sc, a, b] = this.wakeAt(this.fkey(t));
        if (sc >= 0.74 && a <= 4) return t.slice(b);
        const m = /\bhey[\s,]*(xiao|shao)\s*shi\b[\s,.!?，。]*/i.exec(text);
        return m ? text.slice(m.index + m[0].length) : t;
    }
    isStopWord(text) {
        const t = Voice.clean(text);
        if (t.length <= 7) {
            const ft = this.fkey(t);
            for (const p of this.stopP) { const [sc] = match(p, ft, true); if (sc >= 1.0 && (p.length >= 2 || ft.length <= 2)) return true; }
        }
        return /stop|halt/i.test(text) && text.trim().length <= 12 && /^(please\s+)?(stop|halt)\b/i.test(text.trim());
    }
    entities(text) {
        const ft = this.fkey(text), found = [];
        for (const [kind, ref, w, p] of this.lex) {
            const [sc, a, b] = match(p, ft, true);
            if (sc >= thr(p.length) && b - a >= Math.max(p.length - 1, 1)) found.push([kind, ref, w, sc, a, b]);
        }
        found.sort((f, g) => ((g[5] - g[4]) - (f[5] - f[4])) || (g[3] - f[3]));
        const out = [];
        for (const f of found) if (!out.some(o => !(f[5] <= o[4] || f[4] >= o[5]))) out.push(f);
        return out.sort((f, g) => f[4] - g[4]);
    }
    /** utterance (without the wake word) -> {intent, conf, slots, missing, alts, phrase} */
    parse(raw) {
        if (this.english(raw).length) return { intent: null, conf: 0, slots: {}, missing: [], alts: [], english: true, text: raw };
        const t = Voice.clean(raw), ft = this.fkey(t), ents = t ? this.entities(t) : [], slots = {};
        for (const [kind, ref, w] of ents) {
            if (kind === 'item' && !('item' in slots)) { slots.item = ref; slots.item_word = w; }
            else if (kind === 'place' && !('place' in slots)) slots.place = ref;
        }
        for (const w of ['它', '这个', '那个', '这个东西', '那个东西']) if (t.includes(w) && !('item' in slots)) { slots.anaphora = true; break; }
        const nm = /(?:叫做|叫它|叫|起名叫|取名叫|名字是|名字叫)([一-鿿A-Za-z0-9]{1,6})/.exec(t); if (nm) slots.newname = nm[1];
        const bestBy = {};
        for (const [intent, trigs] of Object.entries(this.trig)) {
            let best = [0.0, null];
            for (const { p, phrase, w } of trigs) {
                if (!ft.length) break;
                const [sc] = match(p, ft, false);
                let th = thr(p.length); if (intent === 'stop') th = Math.min(th, p.length > 1 ? 0.7 : 0.5);
                if (sc >= th) { const v = sc * w * (0.62 + 0.13 * Math.min(p.length, 3)); if (v > best[0]) best = [v, phrase]; }
            }
            bestBy[intent] = best;
        }
        let scored = [];
        for (const intent of Object.keys(this.trig)) {
            const best = bestBy[intent]; if (best[1] === null) continue;
            let conf = best[0]; const need = this.V.needs[intent] || [];
            const missing = need.filter(s => !((s in slots) || (s === 'item' && slots.anaphora)));
            if (intent === 'name_item' && !('newname' in slots) && !missing.includes('newname')) missing.push('newname');
            if (need.length && !missing.length) conf = Math.min(1.0, conf + 0.12);
            if (missing.length) conf *= 0.8;
            if (intent === 'go_to' && !('place' in slots)) conf *= 0.3;
            scored.push([conf, intent, best[1], missing]);
        }
        if (t.length <= 4) {
            let best = null;
            for (const [kind, words] of [['yes', this.yes], ['no', this.no]]) for (const [, p] of words) {
                const [sc, a, b] = match(p, ft, true);
                if (sc >= 1.0 && a === 0 && b >= ft.length && (best === null || p.length > best[0])) best = [p.length, kind];
            }
            if (best) scored.push([0.9, best[1], t, []]);
        }
        const personNoun = /(孩子|小孩|宝宝|女儿|儿子|妈妈|爸爸|爷爷|奶奶|哥哥|姐姐|弟弟|妹妹|宝贝|这个人|他|她)/.test(t);
        if (personNoun) slots.person_noun = true;
        scored = scored.map(s => [s[1] === 'name_item' && personNoun ? s[0] * 0.5 : s[0], s[1], s[2], s[3]]);
        if (slots.item_word && slots.newname && !personNoun && !scored.some(s => s[1] === 'name_item')) scored.push([0.85, 'name_item', '叫', []]);
        if (/^(别|不要|不用|先别)/.test(t) && scored.some(s => ['tidy', 'scan_home', 'go_to', 'charge', 'fetch_item'].includes(s[1]))) scored.push([0.9, 'stop', t, []]);
        if (scored.some(s => s[1] === 'set_lang')) slots.lang = /英/.test(t) ? 'en' : 'zh';
        scored.sort((a, b) => b[0] - a[0]);
        if (!scored.length) return { intent: null, conf: 0.0, slots, missing: [], alts: [], text: t };
        const top = scored[0];
        return { intent: top[1], conf: round(top[0], 3), slots, missing: top[3], alts: scored.slice(1, 4).map(s => [round(s[0], 2), s[1]]), text: t, phrase: top[2] };
    }

    // ---- one utterance
    hear(said, role = 'owner') {
        const now = this.now(), rec = { t: round(now, 1), said, role }; this.stats.heard++;
        const out = this._hear(said, role, rec, now); this.log.push(out); return out;
    }
    _hear(said, role, rec, now) {
        if (this.muted) { this.stats.ignored++; rec.action = 'ignored: muted'; return rec; }
        if (this.english(said).length && !/^\s*hey[\s,]*(xiao|shao)\s*shi\s*$/i.test(said)) {
            rec.action = 'english'; rec.outOfScope = true; rec.reply = this.speak('（英文和中英夹杂：真实系统听得懂，沙盘没做）'); return rec;
        }
        if (this.isStopWord(said)) { this._doStop(); this.stats.stop++; rec.intent = 'stop'; rec.action = 'stop'; rec.reply = this.speak('好的，我停下来了'); this.sessionUntil = now + this.V.session_s; return rec; }
        const woke = this.hasWake(said), live = now < this.sessionUntil || (this.pending !== null && now <= this.pending.until);
        if (!woke && !live) { this.stats.ignored++; rec.action = 'ignored: no wake word'; return rec; }
        if (woke) rec.woke = true;
        const body = woke ? this.stripWake(said) : Voice.clean(said);
        if (!Voice.clean(body)) { this.sessionUntil = now + this.V.session_s; rec.intent = 'wake'; rec.action = 'listening'; rec.reply = this.speak('我在'); return rec; }
        if (this.pending !== null && now > this.pending.until) this.pending = null;
        if (this.pending !== null) { if (this._answer(body, role, rec)) { this.sessionUntil = now + this.V.session_s; return rec; } }
        this._command(body, role, rec); this.sessionUntil = now + (this.pending ? this.V.pending_s : this.V.session_s); return rec;
    }
    intentCn(intent, s = {}) {
        const it = s.item_word || '';
        return ({ tidy: '收拾玩具', fetch_item: `把${it || '它'}放进筐里`, scan_home: '扫描一遍家里', go_to: `去${s.place || '那里'}`, charge: '去充电', where_is: `找${it || '东西'}`, resume: '继续刚才的事',
            enrol: '记住一个人', forget: '忘掉一个人', stop: '停下', status: '说说我在做什么', list_items: '数数地上的东西', who_here: '看看谁在', come_here: '过来找你', profile_show: '看看我对某人的了解', set_lang: '换种语言' })[intent] || intent;
    }
    _allowed(intent, role, rec) {
        const need = this.V.min_role[intent] || 'adult';
        if (this.V.role_rank[role] >= this.V.role_rank[need]) return true;
        this.stats.denied++; rec.action = 'denied';
        rec.reply = this.speak(({ child: '这个要大人来说哦', unknown: '我没看清是谁在说话，这个要大人在我面前说才行' })[role] || '这个需要主人来确认'); return false;
    }
    _command(body, role, rec) {
        const p = this.parse(body), conf = p.conf * (0.55 + 0.45 * this.V.asr_conf); let intent = p.intent; const slots = Object.assign({}, p.slots);
        if (intent === 'privacy' && ['静音', '别听', '关掉麦克风', '别看'].includes(p.phrase)) slots.mute = true;
        if (intent === 'forget' && /(所有|全部|大家|每个人)/.test(body)) slots.all = true;
        rec.intent = intent; rec.conf = round(conf, 2); rec.slots = Object.fromEntries(Object.entries(slots).filter(([k]) => k !== 'item_word'));
        if (intent === null || conf < 0.45) {
            this.stats.reask++; rec.action = 'reask';
            rec.reply = this.speak(intent === null ? '我没听清，请再说一遍' : `我没太听清，你是说“${this.intentCn(intent)}”吗？`);
            if (intent !== null) this.pending = { kind: 'confirm', intent, slots, role, until: this.now() + this.V.pending_s };
            return;
        }
        if (intent === 'yes' || intent === 'no') { rec.action = 'nothing to answer'; rec.reply = this.speak('嗯？'); return; }
        const needs = this.V.needs[intent] || [];
        if (needs.includes('item') && !('item' in slots) && slots.anaphora && this.ctx.item !== null) slots.item = this.ctx.item;
        const missing = needs.filter(s => !(s in slots) || slots[s] === null);
        if (!this._allowed(intent, role, rec)) return;
        if (intent === 'enrol' || intent === 'forget' || intent === 'profile_show' || intent === 'name_item') {
            rec.action = intent; rec.outOfScope = true;
            rec.reply = this.speak('（记住、忘掉、给东西起名：真实系统要走同意和确认的流程，见验收 E4、H1，沙盘没做）'); return;
        }
        if (missing.length) {
            const ask = { item: '要哪一个东西？', place: '要去哪里？', newname: '想叫它什么名字？' }[missing[0]];
            this.pending = { kind: 'slot', intent, slots, slot: missing[0], role, until: this.now() + this.V.pending_s }; rec.action = 'ask'; rec.reply = this.speak(ask); return;
        }
        if (this.V.motion.includes(intent) && conf < 0.7) {
            this.stats.confirm++; this.pending = { kind: 'confirm', intent, slots, role, until: this.now() + this.V.pending_s };
            rec.action = 'confirm'; rec.reply = this.speak(`你是要我${this.intentCn(intent, slots)}吗？`); return;
        }
        this._execute(intent, slots, role, rec);
    }
    _answer(body, role, rec) {
        const pd = this.pending, p = this.parse(body), yes = p.intent === 'yes', no = p.intent === 'no';
        if (pd.kind === 'confirm') {
            if (no) { this.pending = null; rec.intent = 'no'; rec.action = 'cancelled'; rec.reply = this.speak('好的，不做了'); return true; }
            if (yes) {
                if (this.V.role_rank[role] < this.V.role_rank[this.V.min_role[pd.intent] || 'adult']) { this._allowed(pd.intent, role, rec); return true; }
                this.pending = null; rec.intent = pd.intent; rec.via = 'confirmed'; this._execute(pd.intent, pd.slots, role, rec); return true;
            }
            this.pending = null; return false;
        }
        if (pd.kind === 'slot') {
            const slots = Object.assign({}, pd.slots), s = pd.slot, ents = this.entities(Voice.clean(body)), got = {};
            for (const e of ents) if (!(e[0] in got)) got[e[0]] = e[1];
            if (!(s in got)) { this.pending = null; return false; }
            slots[s] = got[s]; if (s === 'item') slots.item_word = (ents.find(e => e[0] === 'item') || [])[2] || '';
            this.pending = null; rec.intent = pd.intent; rec.via = 'answer'; this._execute(pd.intent, slots, role, rec); return true;
        }
        this.pending = null; return false;
    }
    _doStop() {
        this.actions.push(['stop']);
        if (this.host) {
            const m = this.host.mode();
            if (['TASK', 'NAV', 'SCAN', 'CHARGE', 'FETCH'].includes(m)) this.resumeCmd = { mode: m, kw: this.host.curKw ? this.host.curKw() : {} };
            this.host.idle();
        }
    }
    _execute(intent, slots, role, rec) {
        const kid = role === 'child' || role === 'unknown', H = this.host, V = this.V;
        this.stats.acted++; rec.action = intent; this.actions.push([intent, { item: slots.item, place: slots.place }]);
        const say = (a, c) => { rec.reply = this.speak(kid && c ? c : a); };
        const izh = slots.item_word ? (slots.item_word) : '它', hasItem = !!slots.item_word;
        const iname = hasItem ? (/[一-鿿]/.test(slots.item_word) ? slots.item_word : (V.items[slots.item] || [slots.item_word])[0]) : '它';
        switch (intent) {
        case 'stop': this._doStop(); say('好的，我停下来了'); break;
        case 'resume':
            if (this.resumeCmd && H) { const { mode, kw } = this.resumeCmd; H.requestMode(mode, mode === 'TASK' ? { resume: true } : kw); say(`好的，接着${V.mode_cn[mode] || '刚才的事'}`); this.resumeCmd = null; }
            else say('我没有要接着做的事');
            break;
        case 'cancel': this.pending = null; say('好的，不做了'); break;
        case 'tidy': if (H) H.startTask(); say('好的，我去收地上的玩具', '好的！我去收玩具啦'); break;
        case 'fetch_item': {
            let it = null;
            if (H) { const its = H.items(slots.item); it = its.find(i => i.status === 'on_floor' && H.canFetch(i)) || null; }
            if (H && it === null) { say(`我现在不知道${hasItem ? iname : '它'}在哪里，我先找一找`); H.explore(); }
            else { if (H) H.fetch(it); this.ctx.item = it ? it.id : slots.item; say(`好的，我去把${hasItem ? iname : '它'}放进筐里`, '好的！'); }
            break;
        }
        case 'where_is': {
            const its = H ? H.items(slots.item) : [];
            if (!H) say(`${hasItem ? iname : '它'}？我现在还没有物品记录`);
            else if (!its.length) say(`我没见过${hasItem ? iname : '这个东西'}`);
            else { this.ctx.item = its[0].id; say(its.slice(0, 2).map(i => H.where(i)).join('；')); }
            break;
        }
        case 'list_items': {
            if (!H) { say('我还没有物品记录'); break; }
            const fl = H.items(null).filter(i => i.status === 'on_floor');
            say(fl.length ? '地上现在有：' + fl.map(i => `${i.label}（${i.place || '地板上'}）`).join('、') : '地上现在没看到东西'); break;
        }
        case 'what_new': say(H ? '和上次一样，没什么变化' : '我还没有物品记录'); break;
        case 'scan_home': if (H) H.scanHome(); say('好的，我去把家里扫描一遍，大概要两分钟'); break;
        case 'go_to': {
            const name = slots.place;
            if (name === '充电座') return this._execute('charge', slots, role, rec);
            if (!H) { say(`好的，我去${name}那边`); break; }
            if (!H.placeKnown(name)) say(`我还不知道${name}在哪里，要先扫描一下家里`);
            else if (H.goToPlace(name)) { this.ctx.place = name; say(`好的，我去${name}那边`); }
            else say(`我去不了${name}那边，路被挡住了`);
            break;
        }
        case 'charge': if (H) H.chargeNow(); say('好的，我去充电'); break;
        case 'who_here': { const ds = H ? H.describePeople() : []; say(ds.length ? '我看到：' + ds.join('；') : '我现在没看到人'); break; }
        case 'status':
            if (!H) say('我在待命');
            else say(`我现在在${V.mode_cn[H.mode()] || H.mode()}，电量大约百分之${Math.round(100 * H.socEst())}`);
            break;
        case 'privacy':
            if (slots.mute) { this.muted = true; say('好的，我不听了。要我重新听，请按机身上的静音键'); }
            else say('我只在听到“小拾小拾”以后才处理你说的话，不保存录音，也不会把声音传出去。要让我完全不听，请按机身上的静音键');
            break;
        case 'greet': say('你好！我是小拾', '你好呀！'); break;
        case 'identity': say('我是小拾，家里的收纳小助手', '我是小拾！'); break;
        case 'help': say('我可以收拾地上的玩具、扫描家里、记住东西放在哪里、告诉你东西在哪儿，也可以去充电。', '我会收玩具，还会找东西！'); break;
        case 'come_here': {
            const loc = H && H.speakerLoc ? H.speakerLoc() : null;
            if (!loc) { say('我没听出你在哪里。请站到我前面再说一遍'); break; }
            if (H) H.comeHere(loc);
            say('好的，我过来', '来啦！'); break;
        }
        case 'set_lang': say(slots.lang === 'en' ? "OK, I'll speak English from now on.（沙盘不换语言，后面的话还是中文）" : '好的，我以后说中文。'); rec.outOfScope = slots.lang === 'en'; break;
        default: say('这个我还不会');
        }
    }
}
XS.Voice = Voice;
})(typeof window !== 'undefined' ? window : globalThis);
