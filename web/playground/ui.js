/* ui.js: the page of the playground: the drawing of the room, the buttons, the checks, the replay of recorded runs and the comparison with the real code. */
(function () {
'use strict';
const XS = window.XS, $ = s => document.querySelector(s), $$ = s => Array.from(document.querySelectorAll(s));
const hypot = Math.hypot, PI = Math.PI;
const MODE_CN = { IDLE: '待命', TASK: '收玩具', NAV: '自己走', SCAN: '扫描家里', FETCH: '捡一件', CHARGE: '充电', TELEOP: '被遥控', RIGHT: '爬起来', FAULT: '故障' };
const ROLE_CN = { owner: '主人', adult: '大人（不认识）', child: '孩子', unknown: '看不清是谁' };
const CLS_CN = { adult: '大人', child: '小孩', unsure: '拿不准' };
const app = { D: null, sim: null, speed: 4, paused: false, view: 'sim', tool: 'speaker', role: 'owner', tapes: {}, tape: null, tapeT: 0, tapePlay: true, tapeSpeed: 1, keys: new Set(), utter: [], passed: {}, seed: 4, acc: 0, colors: null, colorsAt: 0, lastHud: 0, teleOn: false, personSel: 'adult_m', replayName: null, say: '' };
const tests = {};

// ---------------------------------------------------------------------------------------------------------------- loading
function jsonURL(p) { return fetch(p).then(r => { if (!r.ok) throw new Error(p + ' ' + r.status); return r.json(); }); }
async function boot() {
    try { app.D = await jsonURL('data/model.json'); } catch (e) { $('#saybox').textContent = '数据没有加载出来（data/model.json）：' + e.message; return; }
    buildStatic(); newSim(app.seed); requestAnimationFrame(loop);
}
function newSim(seed, soc0) {
    app.seed = seed; app.script = []; app.sim = new XS.Sim(app.D, { seed, soc0: soc0 === undefined ? 0.9 : soc0 }); app.utter = []; app.sim.guardOn = $('#o-guard').checked; app.sim.failRight = $('#o-failright').checked; app.teleOn = false; app.acc = 0;
    app.sim.say('系统', '机器人停在充电座上，等你说话。试试：收拾一下玩具。'); setTele(false);
}

// ---------------------------------------------------------------------------------------------------------------- static parts of the page
function buildStatic() {
    const D = app.D, sp = $('#seg-speed');
    [1, 2, 4, 8, 16, 32].forEach(v => { const b = document.createElement('button'); b.type = 'button'; b.textContent = v + '×'; b.setAttribute('aria-pressed', String(v === app.speed)); b.onclick = () => { app.speed = v; $$('#seg-speed button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); }; sp.appendChild(b); });
    const sd = $('#sel-seed'); for (let i = 1; i <= 12; i++) { const o = document.createElement('option'); o.value = i; o.textContent = i; if (i === app.seed) o.selected = true; sd.appendChild(o); }
    sd.onchange = () => { newSim(+sd.value); setView('sim'); };
    $('#btn-reset').onclick = () => { newSim(app.seed); setView('sim'); };
    $('#btn-pause').onclick = e => { app.paused = !app.paused; e.target.textContent = app.paused ? '继续' : '暂停'; e.target.setAttribute('aria-pressed', String(app.paused)); };
    const rs = $('#seg-role'); Object.entries(ROLE_CN).forEach(([k, v]) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = v; b.style.fontFamily = 'var(--font)'; b.dataset.role = k; b.setAttribute('aria-pressed', String(k === app.role)); b.onclick = () => { app.role = k; $$('#seg-role button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); }; rs.appendChild(b); });
    const quick = [['收拾一下玩具', 0], ['把家里扫描一遍', 0], ['去充电', 0], ['停下', 0], ['继续', 0], ['去沙发那边', 1], ['对', 1], ['不是', 1], ['把小球放进筐里', 1], ['小球在哪', 1], ['地上有什么东西', 1], ['现在有谁', 1], ['你在干什么', 1], ['你好', 1], ['你是谁', 1], ['你会什么', 1], ['过来', 1]];
    const q = $('#quick-say'); quick.forEach(([t]) => { const b = document.createElement('button'); b.type = 'button'; b.className = 'btn small'; b.textContent = t; b.onclick = () => speak(t); q.appendChild(b); });
    $('#say-form').onsubmit = e => { e.preventDefault(); const i = $('#say-text'); if (i.value.trim()) { speak(i.value.trim()); i.value = ''; } };
    const pb = $('#people-btns');
    [['孩子冲过来（5 岁）', 'kid'], ['大人走过来', 'adult'], ['幼儿在筐边', 'toddler'], ['一家三口', 'family'], ['人都离开', 'leave']].forEach(([t, k]) => { const b = document.createElement('button'); b.type = 'button'; b.className = 'btn' + (k === 'kid' ? ' warn' : ''); b.textContent = t; b.onclick = () => { app.sim.person(k); app.sim.say('系统', { kid: '一个 5 岁的孩子从门口跑过来，停在离机器人很近的地方。', adult: '一个大人走到机器人旁边再离开。', toddler: '一个幼儿爬到收纳筐边玩。', family: '妈妈、8 岁的孩子和幼儿先后进来在房间里走动。', leave: '房间里的人都离开。' }[k]); }; pb.appendChild(b); });
    const mb = $('#mess-btns');
    [['推倒（左）', () => app.sim.push('roll_l')], ['推倒（前）', () => app.sim.push('pitch_f')], ['电量调到 15%', () => setSoc(0.15)], ['电量调到 8%（危急）', () => setSoc(0.08)], ['电量调满', () => setSoc(0.95)]].forEach(([t, f]) => { const b = document.createElement('button'); b.type = 'button'; b.className = 'btn'; b.textContent = t; b.onclick = f; mb.appendChild(b); });
    $('#o-guard').onchange = e => { app.sim.guardOn = e.target.checked; app.sim.say('系统', '安全层 ' + (e.target.checked ? '开' : '关了：现在它不会因为人靠近而减速、停下（对照用）')); };
    $('#o-failright').onchange = e => { app.sim.failRight = e.target.checked; };
    $('#btn-tele').onclick = () => setTele(!app.teleOn);
    $$('.keys button').forEach(b => { const k = b.dataset.key; const down = e => { e.preventDefault(); app.keys.add(k); }; const up = () => app.keys.delete(k); b.addEventListener('pointerdown', down); b.addEventListener('pointerup', up); b.addEventListener('pointerleave', up); b.addEventListener('pointercancel', up); });
    window.addEventListener('keydown', e => { if (/input|select|textarea/i.test(document.activeElement && document.activeElement.tagName)) return; const k = e.key.toLowerCase(); if ('wasdqe'.includes(k) && k.length === 1 && app.view === 'sim') { app.keys.add(k); if (document.activeElement === $('#room')) e.preventDefault(); } });
    window.addEventListener('keyup', e => app.keys.delete(e.key.toLowerCase()));
    window.addEventListener('blur', () => app.keys.clear());
    const st = $('#seg-tool');
    [['speaker', '我站这里'], ['goto', '让它去这里'], ['person', '放一个人'], ['move', '把它搬到这里']].forEach(([k, t]) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = t; b.style.fontFamily = 'var(--font)'; b.dataset.tool = k; b.setAttribute('aria-pressed', String(k === app.tool)); b.onclick = () => { app.tool = k; $$('#seg-tool button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); toolNote(); }; st.appendChild(b); });
    const ps = $('#sel-person'); D.people.forEach(p => { const o = document.createElement('option'); o.value = p.name; o.textContent = `${p.cn}（${p.h} m）`; ps.appendChild(o); }); ps.onchange = () => { app.personSel = ps.value; }; ps.value = app.personSel; toolNote();
    $('#tabs').addEventListener('click', e => { const b = e.target.closest('button[data-tab]'); if (b) showTab(b.dataset.tab); });
    $('#btn-clear-checks').onclick = () => { app.passed = {}; renderChecks(); };
    $('#btn-run-compare').onclick = runCompare;
    const room = $('#room'); room.addEventListener('click', onRoomClick); room.addEventListener('pointerdown', () => room.focus({ preventScroll: true }));
    $('#legend').innerHTML = '<span style="--c:var(--info)">大人</span><span style="--c:var(--warn)">小孩 / 拿不准（安全层按小孩办）</span><span style="--c:var(--good)">机器人</span><span style="--c:var(--bad)">安全层停住</span><span style="--c:#e8743b">地上的东西（淡的是还没看到）</span>';
    buildChecks(); buildAbout(); buildTapes();
    const ts = $('#seg-tspeed'); [1, 2, 4, 8].forEach(v => { const b = document.createElement('button'); b.type = 'button'; b.textContent = v + '×'; b.setAttribute('aria-pressed', String(v === 1)); b.onclick = () => { app.tapeSpeed = v; $$('#seg-tspeed button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); }; ts.appendChild(b); });
    $('#tape-play').onclick = e => { app.tapePlay = !app.tapePlay; e.target.textContent = app.tapePlay ? '暂停' : '播放'; };
    $('#tape-back').onclick = () => { setView('sim'); };
    $('#tape-scrub').oninput = e => { if (app.tape) app.tapeT = (+e.target.value / 100) * app.tape.frames[app.tape.frames.length - 1][0]; };
}
function toolNote() {
    $('#tool-note').textContent = { speaker: '点房间里一处：你站在那里（画面上出现“你”）。然后说“过来”，它会走到离你 1 米处。', goto: '点房间里一处：直接让它自己走过去（相当于导航模式 NAV）。人靠近时安全层照样管着。', person: '点房间里一处：把上面选的人放在那里，他站着看着机器人。', move: '点房间里一处：把机器人搬到那里（它会停在原地）。' }[app.tool];
    $('#sel-person').style.display = app.tool === 'person' ? '' : 'none';
}
function showTab(name) {
    $$('#tabs button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
    ['play', 'verify', 'replay', 'compare', 'about'].forEach(n => { $('#pane-' + n).hidden = n !== name; });
    if (name === 'verify') renderChecks();
}
function setSoc(v) { const s = app.sim; s.bat.soc = Math.min(Math.max(v - s.bat.bias, 0.02), 1); s.say('系统', `真实电量设成 ${Math.round(s.bat.soc * 100)}%（机器人自己读到的是 ${Math.round(s.bat.est * 100)}%，有偏差）。它读到低于 20% 就会把手里的放好再去充电，低于 10% 马上去。`); }
function setTele(on) { app.teleOn = on; const b = $('#btn-tele'); b.setAttribute('aria-pressed', String(on)); b.textContent = on ? '遥控 关' : '遥控 开'; if (app.sim) { if (on) { app.sim.teleop(); app.sim.say('系统', '遥控开了：W 前进 S 后退 A 左移 D 右移 Q 左转 E 右转（点一下画面再按）。松手它就停。人靠近时安全层照样限速（在“捣乱”里可以关）。'); } else if (app.sim.mode === 'TELEOP') app.sim.idle(); } }
function speak(text, role) { speakAs(text, role || app.role, $('#o-raw').checked); }
function speakAs(text, role, rawFlag) {
    const s = app.sim; if (app.view !== 'sim') setView('sim');
    const raw = rawFlag || /小拾|xiaoshi/i.test(text), full = raw ? text : '小拾小拾，' + text;
    s.say('你', (role === 'owner' ? '' : ROLE_CN[role] + '：') + text + (rawFlag ? '（没喊它）' : ''));
    const before = s.mode, rec = s.voice.hear(full, role); app.utter.push({ t: s.t, text, full, role, rec, modeBefore: before, raw: rawFlag });
    if (!rec.reply) s.say('系统', `（没有回应：${rec.action || '没听到唤醒词'}）`);
}
/** the commands of a recorded run ("2:/task; 40:/person kid; 60:/battery 0.15; 3:孩子：去充电") done in the sandbox at the same simulated seconds */
function parseScript(str) { return (str || '').split(';').map(x => x.trim()).filter(Boolean).map(x => { const i = x.indexOf(':'); return [parseFloat(x.slice(0, i)), x.slice(i + 1).trim()]; }).sort((a, b) => a[0] - b[0]); }
function runCmd(c) {
    const s = app.sim; s.say('脚本', c); const w = c.split(/\s+/);
    if (c.startsWith('/')) {
        const k = w[0].slice(1).toLowerCase(), a = w[1] || '';
        if (k === 'task') s.startTask(); else if (k === 'scan') s.scanHome(); else if (k === 'charge') s.chargeNow(); else if (k === 'person') s.person(a);
        else if (k === 'battery') setSoc(parseFloat(a)); else if (k === 'push') s.push({ left: 'roll_l', 右: 'roll_r', right: 'roll_r', fwd: 'pitch_f', back: 'pitch_b' }[a] || 'roll_l');
        else if (k === 'stop') s.idle();
        return;
    }
    let role = 'owner', text = c; for (const pre of ['孩子：', '孩子:']) if (text.startsWith(pre)) { role = 'child'; text = text.slice(pre.length).trim(); }
    speakAs(text, role, false);
}
function replayInSandbox(tp) {
    newSim(tp.seed, tp.battery0); app.script = parseScript(tp.do); setView('sim'); $('#viewtag').textContent = `沙盘重演：${tp.title}（同样的剧本，同一个房间布局）`;
    app.sim.say('系统', `沙盘重演“${tp.title}”：${tp.do.replace(/;/g, '；')}`);
}
function onRoomClick(e) {
    if (app.view !== 'sim') return; const r = e.currentTarget.getBoundingClientRect(), W = 4.24, sc = r.width / W; const x = (e.clientX - r.left) / sc - 0.12, y = (r.bottom - e.clientY) / sc - 0.12; const s = app.sim;
    if (x < 0.1 || y < 0.1 || x > 3.9 || y > 2.9) return;
    if (app.tool === 'speaker') { s.speakerAt = [x, y]; s.say('系统', `你站在 (${x.toFixed(1)}, ${y.toFixed(1)}) 了。现在说“过来”。`); }
    else if (app.tool === 'goto') { s.goTo(x, y); s.say('系统', `让它去 (${x.toFixed(1)}, ${y.toFixed(1)})。`); }
    else if (app.tool === 'person') { const i = s.people.index(app.personSel); s.people.scripts[i] = new XS.Script([['at', x, y, 0.0], ['face_robot']]); s.say('系统', `${s.people.specs[i].cn}站在 (${x.toFixed(1)}, ${y.toFixed(1)}) 了。`); }
    else if (app.tool === 'move') { if (s.world.clearance(x, y) > 0.14) { s.x = x; s.y = y; s.vx = s.vy = 0; s.path = null; s.say('系统', '把机器人搬到了新的位置。'); } }
}

// ---------------------------------------------------------------------------------------------------------------- views: the sandbox or a recorded run
function setView(v) {
    app.view = v; $('#viewtag').textContent = v === 'sim' ? '沙盘（行为模型）' : `回放：${app.tape ? app.tape.title : ''}（真实物理仿真录下来的）`; $('#viewtag').style.color = v === 'sim' ? '' : 'var(--accent)';
    $('#tape-detail').hidden = v !== 'tape'; $$('.tape').forEach(t => t.setAttribute('aria-current', String(v === 'tape' && t.dataset.name === app.replayName)));
}
function viewOfSim() {
    const s = app.sim, W = s.world;
    return {
        world: { room: [W.W, W.H], furn: W.furn, boxes: W.boxes, basket: W.basket, dock: s.dockPose, doors: s.D.world.doors }, t: s.t,
        robot: { x: s.x, y: s.y, th: s.th, tilt: s.tilt, mode: s.mode, state: s.state, held: s.held, charging: s.mode === 'CHARGE' && s.state === '充电', halted: s.guard.halted && s.guardOn, scale: s.guardOn ? s.guard.scale : 1, armOut: ['抓取', '放置'].includes(s.state) },
        objects: W.objects.map(o => ({ x: o.x, y: o.y, r: o.r, kind: o.kind, status: o.status, seen: o.seen })),
        people: s.people.state.filter(p => p.inside).map(p => ({ x: p.x, y: p.y, yaw: p.yaw, cls: p.cls, w: p.w, h: p.h, cn: p.cn, sit: p.sit })),
        tracks: s.tracks().map(k => ({ cls: k.cls, x: k.x, y: k.y, src: k.src, coast: k.coast, dist: k.dist, h: k.h })), path: s.path, trail: s.trail, speaker: s.speakerAt, guardWhy: s.guard.why, fov: true,
    };
}
function viewOfTape(tp, t) {
    const F = tp.frames; let i = Math.min(Math.max(0, Math.floor(t * tp.hz)), F.length - 2); const a = F[i], b = F[i + 1], k = b[0] > a[0] ? Math.min(Math.max((t - a[0]) / (b[0] - a[0]), 0), 1) : 0, L = (u, v) => u + (v - u) * k;
    const th = a[3] + ((((b[3] - a[3]) + PI) % (2 * PI) + 2 * PI) % (2 * PI) - PI) * k, W = tp.world, modes = tp.modes, states = tp.states;
    const objs = []; for (let j = 0; j < W.objects.length; j++) { const o = W.objects[j], z = a[14][3 * j + 2], x = a[14][3 * j], y = a[14][3 * j + 1]; const inB = Math.abs(x - W.basket.x) < W.basket.sx / 2 && Math.abs(y - W.basket.y) < W.basket.sy / 2 && z < 0.12; objs.push({ x, y, r: o.r, kind: o.kind, status: inB ? 'in_basket' : (z > 0.06 ? 'held' : 'on_floor'), seen: true }); }
    const ppl = []; for (const p of a[15]) { const q = b[15].find(r => r[0] === p[0]); const sp = tp.people[p[0]]; ppl.push({ x: q ? L(p[1], q[1]) : p[1], y: q ? L(p[2], q[2]) : p[2], yaw: p[3], cls: sp.cls, w: sp.w, h: sp.h, cn: sp.cn, sit: p[4] }); }
    const tracks = a[16].map(r => ({ cls: r[0] === 'a' ? 'adult' : (r[0] === 'c' ? 'child' : 'unsure'), x: r[1], y: r[2], src: r[0] === 'l' ? 'lidar' : 'cam', coast: false, dist: hypot(r[1] - a[1], r[2] - a[2]), h: 1.2 }));
    const trail = []; for (let j = Math.max(0, i - 400); j <= i; j += 10) trail.push([F[j][1], F[j][2]]);
    return {
        world: { room: W.room, furn: W.furniture.map(f => ({ name: f[0], cx: f[1], cy: f[2], sx: f[3], sy: f[4], h: f[5], rgba: f[6] })), boxes: W.boxes.map(b => ({ cx: b[0], cy: b[1], sx: b[2], sy: b[3] })), basket: W.basket, dock: W.dock, doors: app.D.world.doors }, t,
        robot: { x: L(a[1], b[1]), y: L(a[2], b[2]), th, tilt: a[8], mode: modes[a[4]], state: states[a[5]], held: a[11] ? 1 : null, charging: states[a[5]] === '充电', halted: !!a[9], scale: a[10], armOut: ['抓取', '放置'].includes(states[a[5]]) },
        objects: objs, people: ppl, tracks, path: null, trail, speaker: null, guardWhy: a[17], fov: true, frame: a,
    };
}

// ---------------------------------------------------------------------------------------------------------------- drawing
function colors() {
    const now = performance.now(); if (app.colors && now - app.colorsAt < 1000) return app.colors; const cs = getComputedStyle(document.documentElement), g = n => cs.getPropertyValue(n).trim();
    app.colorsAt = now; return app.colors = { floor: g('--floor'), grid: g('--grid'), wall: g('--wall'), ink: g('--ink'), muted: g('--muted'), accent: g('--accent'), good: g('--good'), warn: g('--warn'), bad: g('--bad'), info: g('--info'), panel: g('--panel') };
}
const MODE_COL = { IDLE: '#7a8b86', TASK: 'accent', NAV: 'info', SCAN: '#8a63c6', FETCH: 'accent', TELEOP: '#c2409a', CHARGE: 'good', RIGHT: 'warn', FAULT: 'bad' };
function rgba(s, a = 1) { const v = s.split(' ').map(Number); return `rgba(${Math.round(v[0] * 255)},${Math.round(v[1] * 255)},${Math.round(v[2] * 255)},${a})`; }
const KIND_COL = { ball: '#e8743b', cube: '#3a74d6', car: '#d64a4a', banana: '#e6c52e', lbrick: '#3fa66a', rock: '#8a8f8c', sock: '#8a63c6', plush: '#f08fb0' };
function draw(v) {
    const cv = $('#room'), box = cv.parentElement, dpr = window.devicePixelRatio || 1, cw = Math.round(box.clientWidth * dpr), ch = Math.round(box.clientHeight * dpr);
    if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
    const ctx = cv.getContext('2d'), C = colors(), sc = cw / 4.24, X = x => (x + 0.12) * sc, Y = y => ch - (y + 0.12) * sc, Lm = m => m * sc, Wd = v.world;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cw, ch); ctx.fillStyle = C.floor; ctx.fillRect(0, 0, cw, ch);
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.beginPath(); for (let x = 0; x <= 4.001; x += 0.5) { ctx.moveTo(X(x), Y(0)); ctx.lineTo(X(x), Y(3)); } for (let y = 0; y <= 3.001; y += 0.5) { ctx.moveTo(X(0), Y(y)); ctx.lineTo(X(4), Y(y)); } ctx.stroke();
    ctx.strokeStyle = C.wall; ctx.lineWidth = Math.max(2, sc * 0.03); ctx.strokeRect(X(0), Y(3), Lm(4), Lm(3));
    const txt = (s, x, y, size, col, al = 'center') => { ctx.font = `${Math.max(9, size * dpr)}px "Noto Sans SC", sans-serif`; ctx.fillStyle = col; ctx.textAlign = al; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y); };
    for (const d of Wd.doors) { ctx.strokeStyle = C.floor; ctx.lineWidth = Math.max(4, sc * 0.05); ctx.beginPath(); const horiz = d[1] > 2.5; if (horiz) { ctx.moveTo(X(d[0] - 0.25), Y(3)); ctx.lineTo(X(d[0] + 0.25), Y(3)); } else { ctx.moveTo(X(d[0] < 1 ? 0 : 4), Y(d[1] - 0.25)); ctx.lineTo(X(d[0] < 1 ? 0 : 4), Y(d[1] + 0.25)); } ctx.stroke(); }
    const FN = { sofa: '沙发', table: '桌子', shelf: '架子', plant: '盆栽', chair: '椅子', carton: '箱子' };
    for (const f of Wd.furn.concat(Wd.boxes)) {
        ctx.fillStyle = rgba(f.rgba || '0.79 0.64 0.45 1', 0.85); ctx.strokeStyle = 'rgba(0,0,0,.35)'; ctx.lineWidth = 1;
        if (f.sy === null || f.sy === undefined) { ctx.beginPath(); ctx.arc(X(f.cx), Y(f.cy), Lm(f.sx), 0, 2 * PI); ctx.fill(); ctx.stroke(); } else { ctx.fillRect(X(f.cx - f.sx / 2), Y(f.cy + f.sy / 2), Lm(f.sx), Lm(f.sy)); ctx.strokeRect(X(f.cx - f.sx / 2), Y(f.cy + f.sy / 2), Lm(f.sx), Lm(f.sy)); }
        txt(FN[f.name] || '箱子', X(f.cx), Y(f.cy), 11, 'rgba(40,30,20,.85)');
    }
    const dk = Wd.dock; ctx.fillStyle = v.robot.charging ? C.good : C.muted; ctx.fillRect(X(dk[0] - 0.12), Y(3.0), Lm(0.24), Lm(0.04)); txt('充电座', X(dk[0] + 0.32), Y(2.95), 10, C.muted);
    const b = Wd.basket; ctx.fillStyle = 'rgba(150,105,60,.55)'; ctx.strokeStyle = 'rgba(100,70,40,.9)'; ctx.lineWidth = 2; ctx.fillRect(X(b.x - b.sx / 2), Y(b.y + b.sy / 2), Lm(b.sx), Lm(b.sy)); ctx.strokeRect(X(b.x - b.sx / 2), Y(b.y + b.sy / 2), Lm(b.sx), Lm(b.sy)); txt('收纳筐', X(b.x), Y(b.y - b.sy / 2 - 0.09), 10, C.muted);
    // trail and path
    if ($('#o-path').checked) {
        if (v.trail && v.trail.length > 1) { ctx.strokeStyle = C.accent; ctx.globalAlpha = 0.25; ctx.lineWidth = 2; ctx.beginPath(); v.trail.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))); ctx.stroke(); ctx.globalAlpha = 1; }
        if (v.path && v.path.length > 1) { ctx.strokeStyle = C.accent; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5; ctx.beginPath(); v.path.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))); ctx.stroke(); ctx.setLineDash([]); }
    }
    // things on the floor
    let nb = 0; for (const o of v.objects) {
        if (o.status === 'held') continue; let x = o.x, y = o.y, k = 1; if (o.status === 'in_basket') { k = 0.7; x = b.x - b.sx / 2 + 0.06 + (nb % 5) * 0.055; y = b.y + (nb < 5 ? 0 : 0.05); nb++; }
        drawObj(ctx, o.kind, X(x), Y(y), Math.max(o.r, 0.04) * sc * k, KIND_COL[o.kind] || '#999', o.seen || o.status === 'in_basket' ? 1 : 0.4, !o.seen && o.status !== 'in_basket');
    }
    const R = v.robot, rr = 0.105;
    // guard rings and what the robot sees of the people
    const K = app.D.guard;
    if ($('#o-rings').checked) for (const t of v.tracks) { const kid = t.cls !== 'adult', base = K.robot_r + K.person_r; ctx.lineWidth = 1; ctx.globalAlpha = 0.7; ctx.setLineDash([2, 4]); ctx.strokeStyle = C.warn; ctx.beginPath(); ctx.arc(X(t.x), Y(t.y), Lm(base + (kid ? K.slow_child : K.slow)), 0, 2 * PI); ctx.stroke(); ctx.setLineDash([6, 3]); ctx.strokeStyle = C.bad; ctx.beginPath(); ctx.arc(X(t.x), Y(t.y), Lm(base + (kid ? K.halt_child : K.halt)), 0, 2 * PI); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1; }
    if (v.fov && $('#o-fov').checked && R.mode !== 'RIGHT') { ctx.fillStyle = C.accent; ctx.globalAlpha = 0.07; ctx.beginPath(); ctx.moveTo(X(R.x), Y(R.y)); ctx.arc(X(R.x), Y(R.y), Lm(2.4), -(R.th + 35 * PI / 180), -(R.th - 35 * PI / 180)); ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1; }
    // people (the truth)
    for (const p of v.people) {
        const kid = p.cls === 'child', col = kid ? C.warn : C.info, rad = Math.max(p.w / 2, 0.12);
        ctx.fillStyle = col; ctx.globalAlpha = 0.85; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), Lm(rad), 0, 2 * PI); ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = 'rgba(0,0,0,.45)'; ctx.lineWidth = 1; ctx.stroke();
        ctx.strokeStyle = 'rgba(255,255,255,.9)'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(p.x), Y(p.y)); ctx.lineTo(X(p.x + Math.cos(p.yaw) * rad), Y(p.y + Math.sin(p.yaw) * rad)); ctx.stroke();
        txt(`${p.cn} ${p.h.toFixed(2)} m`, X(p.x), Y(p.y) - Lm(rad) - 9 * dpr, 10.5, C.ink);
    }
    if ($('#o-truth').checked) for (const t of v.tracks) {
        ctx.lineWidth = 2; const col = t.src === 'lidar' ? C.muted : (t.cls === 'adult' ? C.info : C.warn); ctx.strokeStyle = col; ctx.setLineDash(t.src === 'lidar' ? [2, 3] : (t.cls === 'unsure' ? [5, 3] : [])); ctx.beginPath(); ctx.arc(X(t.x), Y(t.y), Lm(0.24), 0, 2 * PI); ctx.stroke(); ctx.setLineDash([]);
        txt(t.src === 'lidar' ? '雷达：有东西在动' : '看到：' + CLS_CN[t.cls], X(t.x), Y(t.y) + Lm(0.24) + 9 * dpr, 10.5, col);
    }
    // the robot
    const mc = MODE_COL[R.mode] || 'accent', col = mc === 'accent' ? C.accent : (C[mc] || mc), pulse = 0.5 + 0.5 * Math.sin(performance.now() / 160);
    if (R.halted) { ctx.strokeStyle = C.bad; ctx.lineWidth = 3; ctx.globalAlpha = 0.5 + 0.5 * pulse; ctx.beginPath(); ctx.arc(X(R.x), Y(R.y), Lm(rr + 0.07), 0, 2 * PI); ctx.stroke(); ctx.globalAlpha = 1; }
    if (R.charging) { ctx.strokeStyle = C.good; ctx.lineWidth = 3; ctx.globalAlpha = 0.4 + 0.5 * pulse; ctx.beginPath(); ctx.arc(X(R.x), Y(R.y), Lm(rr + 0.06), 0, 2 * PI); ctx.stroke(); ctx.globalAlpha = 1; }
    if (R.tilt > 20) {
        ctx.save(); ctx.translate(X(R.x), Y(R.y)); ctx.rotate(-R.th); ctx.fillStyle = col; ctx.strokeStyle = C.bad; ctx.lineWidth = 3; ctx.beginPath(); ctx.ellipse(0, 0, Lm(rr * 1.7), Lm(rr * 0.8), 0, 0, 2 * PI); ctx.fill(); ctx.stroke(); ctx.restore(); txt('倒了', X(R.x), Y(R.y) - Lm(0.2), 11, C.bad);
    } else {
        ctx.fillStyle = col; ctx.strokeStyle = 'rgba(0,0,0,.55)'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(R.x), Y(R.y), Lm(rr), 0, 2 * PI); ctx.fill(); ctx.stroke();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = Math.max(2, sc * 0.02); ctx.beginPath(); ctx.moveTo(X(R.x), Y(R.y)); ctx.lineTo(X(R.x + Math.cos(R.th) * rr * 1.25), Y(R.y + Math.sin(R.th) * rr * 1.25)); ctx.stroke();
        if (R.armOut) { ctx.strokeStyle = 'rgba(0,0,0,.7)'; ctx.lineWidth = Math.max(3, sc * 0.03); ctx.beginPath(); ctx.moveTo(X(R.x + Math.cos(R.th) * rr), Y(R.y + Math.sin(R.th) * rr)); ctx.lineTo(X(R.x + Math.cos(R.th) * (rr + 0.16)), Y(R.y + Math.sin(R.th) * (rr + 0.16))); ctx.stroke(); }
    }
    if (R.held !== null && R.held !== undefined) { const o = v.objects.find((q, i) => i === R.held) || v.objects.find(q => q.status === 'held'); if (o) drawObj(ctx, o.kind, X(R.x + Math.cos(R.th) * (rr + 0.12)), Y(R.y + Math.sin(R.th) * (rr + 0.12)), Math.max(o.r, 0.04) * sc, KIND_COL[o.kind] || '#999', 1, false); }
    txt(`${MODE_CN[R.mode] || R.mode}${R.state && R.state !== MODE_CN[R.mode] ? '·' + R.state : ''}`, X(R.x), Y(R.y) + Lm(rr) + 10 * dpr, 10.5, C.ink);
    if (v.speaker) { ctx.fillStyle = C.info; ctx.beginPath(); ctx.moveTo(X(v.speaker[0]), Y(v.speaker[1]) - 9 * dpr); ctx.lineTo(X(v.speaker[0]) + 7 * dpr, Y(v.speaker[1])); ctx.lineTo(X(v.speaker[0]), Y(v.speaker[1]) + 9 * dpr); ctx.lineTo(X(v.speaker[0]) - 7 * dpr, Y(v.speaker[1])); ctx.closePath(); ctx.fill(); txt('你', X(v.speaker[0]), Y(v.speaker[1]) - 16 * dpr, 11, C.info); }
}
function drawObj(ctx, kind, x, y, r, col, alpha, dashed) {
    ctx.save(); ctx.globalAlpha = alpha; ctx.fillStyle = col; ctx.strokeStyle = 'rgba(0,0,0,.5)'; ctx.lineWidth = 1;
    if (kind === 'ball') { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * PI); ctx.fill(); ctx.stroke(); }
    else if (kind === 'cube') { ctx.fillRect(x - r, y - r, 2 * r, 2 * r); ctx.strokeRect(x - r, y - r, 2 * r, 2 * r); }
    else if (kind === 'car') { ctx.fillRect(x - r * 1.3, y - r * 0.7, r * 2.6, r * 1.4); ctx.strokeRect(x - r * 1.3, y - r * 0.7, r * 2.6, r * 1.4); ctx.fillStyle = '#222'; ctx.fillRect(x - r, y - r * 0.9, r * 0.5, r * 0.3); ctx.fillRect(x + r * 0.5, y - r * 0.9, r * 0.5, r * 0.3); }
    else if (kind === 'banana') { ctx.strokeStyle = col; ctx.lineWidth = Math.max(3, r * 0.6); ctx.beginPath(); ctx.arc(x, y + r * 0.5, r * 1.1, PI * 1.15, PI * 1.85); ctx.stroke(); }
    else if (kind === 'lbrick') { ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x - r * 0.1, y - r); ctx.lineTo(x - r * 0.1, y + r * 0.1); ctx.lineTo(x + r, y + r * 0.1); ctx.lineTo(x + r, y + r); ctx.lineTo(x - r, y + r); ctx.closePath(); ctx.fill(); ctx.stroke(); }
    else if (kind === 'rock') { ctx.beginPath(); for (let i = 0; i < 7; i++) { const a = i / 7 * 2 * PI, rr = r * (0.8 + 0.25 * ((i * 37) % 5) / 5); i ? ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr) : ctx.moveTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr); } ctx.closePath(); ctx.fill(); ctx.stroke(); }
    else if (kind === 'sock') { ctx.beginPath(); ctx.moveTo(x - r * 0.5, y - r); ctx.lineTo(x + r * 0.5, y - r); ctx.lineTo(x + r * 0.5, y + r * 0.2); ctx.lineTo(x + r * 1.2, y + r); ctx.lineTo(x - r * 0.5, y + r); ctx.closePath(); ctx.fill(); ctx.stroke(); }
    else { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * PI); ctx.fill(); ctx.stroke(); ctx.beginPath(); ctx.arc(x - r * 0.7, y - r * 0.8, r * 0.38, 0, 2 * PI); ctx.arc(x + r * 0.7, y - r * 0.8, r * 0.38, 0, 2 * PI); ctx.fill(); }
    if (dashed) { ctx.setLineDash([3, 3]); ctx.strokeStyle = colors().muted; ctx.beginPath(); ctx.arc(x, y, r * 1.5, 0, 2 * PI); ctx.stroke(); }
    ctx.restore();
}

// ---------------------------------------------------------------------------------------------------------------- the readout
const fmtT = t => t < 90 ? `${t.toFixed(0)} s` : `${Math.floor(t / 60)} 分 ${Math.round(t % 60)} 秒`;
function chip(cls, html) { return `<span class="chip ${cls || ''}">${html}</span>`; }
function hudSim() {
    const s = app.sim, g = s.guard, soc = s.bat.est, tr = Math.round(s.bat.soc * 100), n = s.world.objects.length; let h = '';
    h += chip(s.mode === 'FAULT' ? 'bad' : (s.mode === 'IDLE' ? '' : 'info'), `模式 <b>${s.mode}</b> ${MODE_CN[s.mode] || ''}${s.state && s.state !== MODE_CN[s.mode] ? '· ' + s.state : ''}`);
    h += chip(soc < 0.1 ? 'bad' : (soc < 0.2 ? 'warn' : ''), `电量读数 <b>${Math.round(soc * 100)}%</b><span class="meter ${soc < 0.1 ? 'crit' : (soc < 0.2 ? 'low' : '')}"><i style="width:${Math.round(soc * 100)}%"></i></span><span class="note">真 ${tr}%</span>`);
    h += chip(s.delivered() === n ? 'good' : '', `送进筐里 <b>${s.delivered()}/${n}</b>`);
    if (!s.guardOn) h += chip('bad', '安全层 <b>关</b>'); else if (g.halted) h += chip('bad', `安全层 <b>停住</b> <span class="note">${g.why}</span>`); else if (g.scale < 0.999) h += chip('warn', `安全层 <b>减速到 ${Math.round(g.scale * 100)}%</b>`); else h += chip('good', '安全层 <b>正常</b>');
    const seen = s.tracks().length, real = s.people.state.filter(p => p.inside).length; if (real || seen) h += chip('', `人：看到 <b>${seen}</b> / 在场 <b>${real}</b>`);
    if (s.fault) h += chip('bad', `故障：${s.fault}`);
    $('#hud').innerHTML = h; $('#clock').textContent = `${fmtT(s.t)} · ${app.speed}×`;
    const last = s.chat[s.chat.length - 1]; $('#saybox').textContent = last ? `${last.who}：${last.text}` : '';
}
function hudTape(v) {
    const a = v.frame, tp = app.tape; let h = '';
    h += chip('info', `模式 <b>${v.robot.mode}</b> ${MODE_CN[v.robot.mode] || ''}· ${v.robot.state}`);
    h += chip(a[7] < 0.1 ? 'bad' : (a[7] < 0.2 ? 'warn' : ''), `电量读数 <b>${Math.round(a[7] * 100)}%</b><span class="meter ${a[7] < 0.1 ? 'crit' : (a[7] < 0.2 ? 'low' : '')}"><i style="width:${Math.round(a[7] * 100)}%"></i></span><span class="note">真 ${Math.round(a[6] * 100)}%</span>`);
    h += chip(a[12] === tp.world.objects.length ? 'good' : '', `送进筐里 <b>${a[12]}/${tp.world.objects.length}</b>`);
    h += a[9] ? chip('bad', `安全层 <b>停住</b> <span class="note">${a[17]}</span>`) : (a[10] < 0.999 ? chip('warn', `安全层 <b>减速到 ${Math.round(a[10] * 100)}%</b>`) : chip('good', '安全层 <b>正常</b>'));
    h += chip('', `人：看到 <b>${a[16].length}</b> / 在场 <b>${a[15].length}</b>`);
    $('#hud').innerHTML = h; $('#clock').textContent = `${fmtT(v.t)} / ${fmtT(tp.frames[tp.frames.length - 1][0])}`;
    const say = tp.say.filter(x => x[0] <= v.t && x[1] !== '脚本').slice(-1)[0], ev = tp.events.filter(x => x[0] <= v.t).slice(-1)[0]; $('#saybox').textContent = say && (!ev || say[0] >= ev[0]) ? `${say[1]}：${say[2]}` : (ev ? `${ev[2]}` : '');
}
function renderPanels() {
    const s = app.sim; if (app.view !== 'sim') return;
    const chat = $('#chat'), top0 = chat.scrollTop, atEnd = chat.scrollTop + chat.clientHeight >= chat.scrollHeight - 30; chat.innerHTML = s.chat.slice(-60).map(m => `<div><span class="t">${m.t.toFixed(0)}s</span><span class="w ${m.who === '机器人' ? 'r' : (m.who === '你' ? 'y' : 's')}">${m.who}</span><span>${esc(m.text)}</span></div>`).join(''); chat.scrollTop = atEnd ? chat.scrollHeight : top0;
    $('#evlog').innerHTML = s.logLines.slice(-14).reverse().map(l => `<div>${l[0].toFixed(0)}s ${esc(l[1])}</div>`).join('');
    const truth = s.people.state.filter(p => p.inside), seen = s.tracks();
    $('#peopletable').innerHTML = (truth.length || seen.length) ? `<tr><th>真正在房间里</th><th class="num">离机器人</th><th>机器人看到了吗</th></tr>` + truth.map(p => { const d = hypot(p.x - s.x, p.y - s.y), k = seen.find(t => t.truth === p.i); return `<tr><td>${p.cn} ${p.h.toFixed(2)} m</td><td class="num">${d.toFixed(1)} m</td><td>${k ? (k.src === 'lidar' ? '雷达看到有东西在动（拿不准，按小孩办）' : `看到了，判成<b>${CLS_CN[k.cls]}</b>`) : '没看到（不在相机视野里，或还不到 0.4 秒）'}</td></tr>`; }).join('') : '<tr><td class="note">房间里现在没有人。点上面“让人走进来”。</td></tr>';
    const mg = s.people.minGap.map((g, i) => g < 9 ? `${s.people.specs[i].cn} ${Math.max(g, 0).toFixed(2)} m` : null).filter(Boolean);
    $('#people-now').textContent = mg.length ? `离机器人最近（表面到表面，从开始到现在）：${mg.join('；')}；碰到人 ${s.stats.contacts} 次` : '';
}
const esc = t => String(t).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// ---------------------------------------------------------------------------------------------------------------- the checks
const CHECKS = [
    { id: 'tidy', title: '收玩具：5 件都送进筐里', how: '点“收拾一下玩具”，把速度调到 8×，等它做完。', expect: '它从充电座出来，先转一圈看，再一件件找、夹、搬到收纳筐、放下；“送进筐里”涨到 5/5，最后回去充电。', real: ['B1', 'A2'], test: a => a.sim.delivered() === a.sim.world.objects.length },
    { id: 'child_denied', title: '孩子不能让它去充电', how: '“谁在说”选“孩子”，点“去充电”。', expect: '它回“这个要大人来说哦”，不动。', real: ['F1', 'F3'], test: a => a.utter.some(u => u.role === 'child' && u.rec.intent === 'charge' && u.rec.action === 'denied') },
    { id: 'child_tidy', title: '孩子可以让它收玩具', how: '“谁在说”选“孩子”，说“收拾一下玩具”。', expect: '它回“好的！我去收玩具啦”（对孩子的回答语气更短），开始收。', real: ['F1'], test: a => a.utter.some(u => u.role === 'child' && u.rec.intent === 'tidy' && u.rec.action === 'tidy') },
    { id: 'no_wake', title: '没喊它的话，它不动', how: '勾上“不喊小拾小拾”，说“去充电”或“收拾一下玩具”。', expect: '没有回应，也不动（它只听喊了“小拾小拾”以后的话）。', real: ['F1'], test: a => a.utter.some(u => u.rec.action === 'ignored: no wake word' && /充电|收拾|扫描/.test(u.text)) },
    { id: 'stop', title: '“停下”随时有效，不用喊它', how: '先让它干活，再勾上“不喊小拾小拾”，说“停下”。', expect: '马上回“好的，我停下来了”，模式回到待命。', real: ['F1', 'F5'], test: a => a.utter.some(u => u.raw && u.rec.intent === 'stop' && u.rec.action === 'stop' && ['TASK', 'NAV', 'SCAN', 'CHARGE', 'FETCH'].includes(u.modeBefore)) },
    { id: 'confirm', title: '拿不准就先问，答“对”才动', how: '说“去沙发那边”，它问回来后再说“对”。（还没扫描过家里，它会说不知道沙发在哪）', expect: '先问“你是要我去沙发吗？”；说“对”之后才行动；说“不是”就取消。', real: ['F3'], test: a => a.utter.some(u => u.rec.via === 'confirmed') && a.utter.some(u => u.rec.action === 'confirm') },
    { id: 'scan_go', title: '扫描之后知道沙发在哪，能自己走过去', how: '说“把家里扫描一遍”（调到 8×），完了再说“去沙发那边”，再说“对”。', expect: '扫描结束日志里写出家具名；之后“去沙发那边”确认后它走到沙发旁 0.5 米处。', real: ['D1', 'C6'], test: a => a.sim.scanDone && a.sim.navResults.some(r => r.result === 'arrived') && a.utter.some(u => u.rec.action === 'go_to') },
    { id: 'kid_halt', title: '孩子冲过来，安全层停住，没碰到', how: '让它收玩具，点“孩子冲过来”。', expect: '“安全层”变红“停住”，日志写“有人在附近”；孩子走开后它接着做；整个过程碰到人 0 次（孩子走出房间后页面才打勾）。', real: ['E3', 'E1'], test: a => a.sim.guard.stats.halts >= 1 && a.sim.stats.contacts === 0 && a.sim.people.state.some((p, i) => p.cls === 'child' && a.sim.people.minGap[i] < 9) && !a.sim.people.state.some(p => p.inside) },
    { id: 'guard_off', title: '对照：关掉安全层，孩子冲过来', how: '取消勾选“安全层开着”，让它收玩具，点“孩子冲过来”。', expect: '它不再因为孩子停下或减速：日志里没有“有人在附近”，照常往前开；“离人最近”那一行能看到孩子凑到多近。说明刚才的停是安全层做的。', real: ['E3'], test: a => !a.sim.guardOn && a.sim.guard.stats.halts === 0 && a.sim.people.minGap.some((g, i) => a.sim.people.specs[i].cls === 'child' && g < 0.6) && a.sim.mode !== 'IDLE' },
    { id: 'low_soft', title: '电量低：放好手里的，去充电，回来接着干', how: '让它收玩具，点“电量调到 15%”（8×）。', expect: '它不丢下手里的东西：先放进筐，日志“电量低，停在两件事之间去充电”→ CHARGE → 充好电回到收玩具。', real: ['C4', 'C3'], test: a => a.sim.events.some(e => /停在两件事之间去充电/.test(e[2])) && a.sim.events.some(e => /→ TASK/.test(e[2]) && a.sim.events.some(f => /→ CHARGE/.test(f[2]) && f[0] < e[0])) && a.sim.stats.charges >= 1 },
    { id: 'low_crit', title: '电量危急（低于 10%）：马上去充', how: '让它收玩具，点“电量调到 8%（危急）”。', expect: '不等放好手里的，日志“低于紧急线，放下手里的事去充电”。', real: ['C4'], test: a => a.sim.events.some(e => /低于紧急线/.test(e[2])) },
    { id: 'fall', title: '被推倒，自己起来，接着干', how: '让它收玩具，点“推倒（左）”。', expect: '日志“摔倒了”→ RIGHT（爬起来，约半分钟）→ 回到收玩具；手里的东西放回地上，重新找。', real: ['C1', 'C2'], test: a => a.sim.stats.righted >= 1 && a.sim.mode !== 'RIGHT' },
    { id: 'teleop', title: '遥控：松手就停，人靠近安全层仍管着', how: '点“遥控 开”，点一下画面，按住 W，再松手。', expect: '它跟着动；松手 0.5 秒内停，日志“指令超时，停车（死人开关）”。', real: ['C5'], test: a => a.sim.events.some(e => /死人开关/.test(e[2])) },
    { id: 'who', title: '问“现在有谁”，说得出大人还是小孩、多远', how: '点“一家三口”或“孩子冲过来”，等人进了相机视野，说“现在有谁”。', expect: '“我看到：一个小孩，约 1.1 米高，在正前方 1.2 米…”', real: ['E1', 'F4'], test: a => a.utter.some(u => u.rec.intent === 'who_here' && /我看到/.test(u.rec.reply || '')) },
    { id: 'come', title: '“过来”：走到你面前 1 米', how: '“点画面”选“我站这里”，点房间里一处，再说“过来”。', expect: '回“好的，我过来”，它走到离你 1 米处，面朝你。', real: ['F4'], test: a => a.utter.some(u => u.rec.intent === 'come_here' && u.rec.reply === '好的，我过来') },
];
function buildChecks() { $('#verify-count').textContent = `0/${CHECKS.length}`; renderChecks(); }
function realChips(ids) {
    const acc = app.D.accept && app.D.accept.items || {}; return ids.map(id => { const r = acc[id]; if (!r) return `<span class="chip">${id}</span>`; const ok = r.verdict === 'PASS'; return `<span class="chip ${ok ? 'good' : 'warn'}" title="${esc(r.title)}${r.accepted ? '：' + esc(r.accepted) : ''}">真实验收 ${id} <b>${ok ? '通过' : (r.accepted ? '未通过，已接受' : '未通过')}</b></span>`; }).join('');
}
function renderChecks() {
    const el = $('#checks'); if (!el) return;
    el.innerHTML = CHECKS.map(c => {
        const ids = c.real, accl = ids.map(id => (app.D.accept.items || {})[id]).filter(Boolean)[0]; const lines = accl ? accl.lines.slice(0, 2).map(l => `${esc(l.label)} ${esc(l.measured)}（要求 ${esc(l.need)}）`).join('；') : '';
        return `<div class="chk ${app.passed[c.id] ? 'ok' : ''}"><div class="mark">${app.passed[c.id] ? '✓' : ''}</div><div><h4>${c.title}</h4><p><b>怎么做：</b>${c.how}</p><p><b>该看到：</b>${c.expect}</p></div><div class="real">${realChips(ids)}<span class="note">${esc(lines)}</span></div></div>`;
    }).join('');
    const n = CHECKS.filter(c => app.passed[c.id]).length; $('#verify-count').textContent = `${n}/${CHECKS.length}`; $('#verify-bar').style.width = (100 * n / CHECKS.length) + '%';
}
function runChecks() {
    if (app.view !== 'sim') return; let changed = false;
    for (const c of CHECKS) if (!app.passed[c.id]) { let ok = false; try { ok = !!c.test(app); } catch (e) { ok = false; } if (ok) { app.passed[c.id] = true; changed = true; app.sim.say('系统', `✓ 验证通过：${c.title}`); } }
    if (changed) renderChecks();
}

// ---------------------------------------------------------------------------------------------------------------- recorded runs
const TAPE_LIST = [['tidy', '收拾玩具'], ['kid', '孩子冲过来'], ['lowbat', '电量低'], ['fall', '被推倒'], ['family', '一家人在家'], ['scan', '扫描家里'], ['child_denied', '孩子要它去充电']];
function buildTapes() {
    $('#tapes').innerHTML = TAPE_LIST.map(([n, t]) => `<div class="tape" data-name="${n}" aria-current="false"><h4>${t}</h4><p class="note" id="tape-about-${n}">真实仿真录像</p><div class="row"><button class="btn primary small" data-play="${n}" type="button">放这一段（真实仿真）</button><button class="btn small" data-redo="${n}" type="button">用沙盘重演同样的剧本</button></div><span class="note" id="tape-meta-${n}"></span></div>`).join('');
    $('#tapes').addEventListener('click', async e => { const b = e.target.closest('button[data-play]'), r = e.target.closest('button[data-redo]'); if (b) playTape(b.dataset.play); if (r) { const tp = await fetchTape(r.dataset.redo); if (tp) replayInSandbox(tp); } });
    TAPE_LIST.forEach(([n]) => fetchTape(n).then(tp => { if (tp) { $('#tape-about-' + n).textContent = tp.about; $('#tape-meta-' + n).textContent = `${fmtT(tp.frames[tp.frames.length - 1][0])} · 种子 ${tp.seed} · 录的时候真实机器人要 ${fmtT(tp.result.wall_s)} 的电脑时间`; } }));
}
async function fetchTape(n) { if (app.tapes[n]) return app.tapes[n]; try { const tp = await jsonURL(`tapes/${n}.json`); app.tapes[n] = tp; return tp; } catch (e) { const m = $('#tape-meta-' + n); if (m) m.textContent = '（这一段没有加载出来）'; return null; } }
async function playTape(n) {
    const tp = await fetchTape(n); if (!tp) return; app.tape = tp; app.replayName = n; app.tapeT = 0; app.tapePlay = true; $('#tape-play').textContent = '暂停'; setView('tape');
    const T = tp.frames[tp.frames.length - 1][0]; $('#tape-scrub').max = 100; $('#tape-scrub').value = 0;
    const L = [].concat(tp.events.map(e => [e[0], `[${e[1]}] ${e[2]}`])).concat(tp.say.filter(x => x[1] === '你' || x[1] === '机器人').map(x => [x[0], `${x[1]}：${x[2]}`])).sort((a, b) => a[0] - b[0]);
    $('#tape-events').innerHTML = L.map(l => `<div data-t="${l[0]}">${l[0].toFixed(0)}s ${esc(l[1])}</div>`).join('');
    const room = $('#room'); room.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

// ---------------------------------------------------------------------------------------------------------------- compare with the real code
function runCompare() {
    const body = $('#cmp-body'); body.innerHTML = '<p class="note">正在算…</p>';
    setTimeout(() => {
        try { body.innerHTML = compareHTML(); } catch (e) { body.innerHTML = `<p class="note">出错了：${esc(e.message)}</p>`; console.error(e); }
    }, 30);
}
function cmpVoice() {
    const g = app.D.golden.voice, rows = { same: [], diff: [], skip: [] };
    for (const r of g.single) {
        const v = new XS.Voice(app.D.voice, app.D.pinyin, null), rec = v.hear(r.text, r.role);
        if (rec.outOfScope) { rows.skip.push([r, rec]); continue; }
        const ok = (rec.intent ?? null) === (r.intent ?? null) && (rec.action ?? null) === (r.action ?? null) && (rec.reply ?? null) === (r.reply ?? null);
        (ok ? rows.same : rows.diff).push([r, rec]);
    }
    let dsame = 0; const ddiff = [];
    for (const d of g.dialogues) {
        const v = new XS.Voice(app.D.voice, app.D.pinyin, null); let ok = true;
        for (const t of d.turns) { v.advance(t.wait); const rec = v.hear(t.text, t.role); if ((rec.intent ?? null) !== (t.intent ?? null) || (rec.action ?? null) !== (t.action ?? null) || (rec.reply ?? null) !== (t.reply ?? null)) ok = false; }
        if (ok) dsame++; else ddiff.push(d.name);
    }
    return { rows, dsame, dtotal: g.dialogues.length, ddiff, n: g.single.length };
}
function cmpGuard() {
    const K = app.D.guard; let same = 0; const bad = [];
    for (const r of app.D.golden.guard) {
        const gu = new XS.Guard(K), b = r.bearing * PI / 180, d = K.robot_r + K.person_r + r.gap, px = 2 + d * Math.cos(b), py = 1.5 + d * Math.sin(b);
        const k = { id: 1, cls: r.cls, x: px, y: py, vx: -r.closing * Math.cos(b), vy: -r.closing * Math.sin(b), coast: false, predict() { return [this.x, this.y]; } };
        const S = { t: 100, mode: 'TASK', state: '前往', armBusy: r.arm, armDone: () => true, pose: () => [2, 1.5, 0], zoneSkipT: -1e9, tracks: () => [k], wheelTarget: { vx: 0, vy: 0, wz: 0 }, setTwist() {}, holdArm: false, dynZones: [], personZones: [], event() {}, speak() {}, onAsk: null };
        const halted = !!gu.pre(S); S.wheelTarget = { vx: r.v, vy: 0, wz: 0 }; gu.post(S);
        if (halted === r.halted && Math.abs(gu.scale - r.scale) < 1e-4) same++; else bad.push(r);
    }
    return { same, n: app.D.golden.guard.length, bad };
}
function firstT(ev, re, after = -1) { const e = ev.find(x => re.test(x[2]) && x[0] > after); return e ? e[0] : null; }
function minutesOf(ev) { const e = ev.find(x => /充电 \d+% → \d+%，用了 (\d+) 分钟/.test(x[2])); return e ? +/用了 (\d+) 分钟/.exec(e[2])[1] : null; }
const seqOf = ev => ['IDLE'].concat(ev.filter(e => /^→ [A-Z]+/.test(e[2])).map(e => /^→ ([A-Z]+)/.exec(e[2])[1])).join('→');
async function cmpRuns() {
    const out = [], D = app.D, mk = () => new XS.Sim(D, { seed: 4, soc0: 0.9 }), say = (s, t) => s.voice.hear('小拾小拾，' + t, 'owner');
    const tapes = {}; for (const n of ['kid', 'fall', 'lowbat', 'tidy', 'scan']) tapes[n] = await fetchTape(n);
    const f = v => v === null || v === undefined ? '—' : v;
    if (tapes.kid) {
        const s = mk(); say(s, '收拾一下玩具'); s.run(40); s.person('kid'); s.run(130); const ev = s.events, re = tapes.kid.events;
        out.push(['孩子冲过来', '模式变化的先后', seqOf(re), seqOf(ev)], ['', '安全层停住的次数', tapes.kid.result.guard.halts, s.guard.stats.halts], ['', '第一次停住的时间', f(firstT(re, /有人在附近/)) + ' s', f(firstT(ev, /有人在附近/)) + ' s'], ['', '一共停了多久', tapes.kid.result.guard.halt_s.toFixed(1) + ' s', s.guard.stats.halt_s.toFixed(1) + ' s'], ['', '碰到人', 0, s.stats.contacts]);
    }
    if (tapes.fall) {
        const s = mk(); say(s, '收拾一下玩具'); s.run(70); s.push('roll_l'); s.run(190); const tf = firstT(tapes.fall.events, /摔倒了/), tb = firstT(tapes.fall.events, /→ TASK/, tf), t2 = firstT(s.events, /摔倒了/), tb2 = firstT(s.events, /→ TASK/, t2);
        out.push(['被推倒', '模式变化的先后', seqOf(tapes.fall.events), seqOf(s.events)], ['', '摔倒到回到收玩具', f(tb && tf ? (tb - tf).toFixed(1) : null) + ' s', f(tb2 && t2 ? (tb2 - t2).toFixed(1) : null) + ' s'], ['', '爬起来的次数', tapes.fall.result.sup_stats.righted, s.stats.righted]);
    }
    if (tapes.lowbat) {
        const s = mk(); say(s, '收拾一下玩具'); s.run(60); s.bat.soc = 0.15 - s.bat.bias; let t = 0; while (t < 1500 && !s.taskDone) { s.run(10); t += 10; }
        s.run(30); out.push(['电量低（15%）', '模式变化的先后', seqOf(tapes.lowbat.events), seqOf(s.events)], ['', '去充电的时间点', f(firstT(tapes.lowbat.events, /停在两件事之间去充电/)) + ' s', f(firstT(s.events, /停在两件事之间去充电/)) + ' s'], ['', '充电用时（快进）', f(minutesOf(tapes.lowbat.events)) + ' 分', f(minutesOf(s.events)) + ' 分'], ['', '5 件全送完的时间', f(firstT(tapes.lowbat.events, /任务结束/)) + ' s', f(firstT(s.events, /任务结束/)) + ' s'], ['', '送到', tapes.lowbat.result.delivered + '/5', s.delivered() + '/5']);
    }
    if (tapes.tidy) {
        const s = mk(); say(s, '收拾一下玩具'); let t = 0; while (t < 2000 && !s.taskDone) { s.run(10); t += 10; }
        const F = tapes.tidy.frames; let tLast = null; for (let i = 1; i < F.length; i++) if (F[i][12] > F[i - 1][12]) tLast = F[i][0];
        s.run(30); out.push(['整套收玩具（不打断）', '模式变化的先后（收完后去充电还是待命，看电量低于不低于 85%）', seqOf(tapes.tidy.events), seqOf(s.events)], ['', '第 5 件放进筐的时间', f(tLast) + ' s', f(s.deliveries && s.deliveries[s.deliveries.length - 1] !== undefined ? s.deliveries[s.deliveries.length - 1].toFixed(0) : null) + ' s'], ['', '任务结束的时间（含放完后满屋再巡一圈、回起点）', f(firstT(tapes.tidy.events, /任务结束/)) + ' s', f(firstT(s.events, /任务结束/)) + ' s'], ['', '送到', tapes.tidy.result.delivered + '/5', s.delivered() + '/5']);
    }
    if (tapes.scan) {
        const s = mk(); say(s, '把家里扫描一遍'); let t = 0; while (t < 900 && !s.scanDone) { s.run(5); t += 5; } const re = tapes.scan.events.find(x => /扫描完成/.test(x[2])), me = s.events.find(x => /扫描完成/.test(x[2]));
        const num = (e, r) => e ? (r.exec(e[2]) || [])[1] : null;
        s.run(3); out.push(['扫描家里', '模式变化的先后', seqOf(tapes.scan.events), seqOf(s.events)], ['', '用了多久', f(num(re, /，(\d+) 秒/)) + ' s', f(num(me, /，(\d+) 秒/)) + ' s'], ['', '停靠点', f(num(re, /：(\d+) 个停靠点/)), f(num(me, /：(\d+) 个停靠点/))]);
    }
    return out;
}
function compareHTML() {
    const V = cmpVoice(), G = cmpGuard(), B = app.D.golden.battery; let h = '';
    const nIn = V.rows.same.length + V.rows.diff.length;
    h += `<div class="sec"><h3>1　说的话（${V.n} 句 × 4 种说话的人，和 ${V.dtotal} 段多轮对话）</h3><p>${chip(V.rows.diff.length ? 'warn' : 'good', `沙盘做了的 ${nIn} 句里：回答和真代码完全一样 <b>${V.rows.same.length}</b>，不一样 <b>${V.rows.diff.length}</b>`)} ${chip('', `沙盘没做的 <b>${V.rows.skip.length}</b> 句`)} ${chip(V.ddiff.length ? 'warn' : 'good', `多轮对话一样 <b>${V.dsame}/${V.dtotal}</b>`)}</p><p class="note">“一样”是指听懂的意图、动作（直接做 / 先问 / 拒绝 / 追问）和说出来的话逐字相同。真代码按拼音的近似匹配听，沙盘用同一张拼音表和同一个匹配算法，所以打同音字也认。</p>`;
    h += `<details><summary>沙盘没做的 ${V.rows.skip.length} 句（真实系统要走同意和确认流程，或是英文）</summary><div class="scroll"><table><tr><th>谁</th><th>说</th><th>真代码</th></tr>${V.rows.skip.map(([r]) => `<tr><td>${ROLE_CN[r.role]}</td><td>${esc(r.text.replace('小拾小拾，', ''))}</td><td>${esc(r.action)} · ${esc(r.reply || '')}</td></tr>`).join('')}</table></div></details>`;
    if (V.rows.diff.length) h += `<details open><summary>不一样的 ${V.rows.diff.length} 句</summary><div class="scroll"><table><tr><th>谁</th><th>说</th><th>真代码</th><th>沙盘</th></tr>${V.rows.diff.map(([r, j]) => `<tr><td>${ROLE_CN[r.role]}</td><td>${esc(r.text.replace('小拾小拾，', ''))}</td><td>${esc(r.action)} · ${esc(r.reply || '')}</td><td>${esc(j.action)} · ${esc(j.reply || '')}</td></tr>`).join('')}</table></div></details>`;
    h += `<details><summary>看几句真代码的回答（前 24 句，主人 / 孩子）</summary><div class="scroll"><table><tr><th>说</th><th>主人</th><th>孩子</th></tr>${app.D.golden.voice.single.filter(r => r.role === 'owner').slice(0, 24).map(r => { const c = app.D.golden.voice.single.find(q => q.role === 'child' && q.text === r.text); return `<tr><td>${esc(r.text.replace('小拾小拾，', '') || '（只喊了名字）')}</td><td>${esc(r.reply || '（不回应）')}</td><td>${esc(c.reply || '（不回应）')}</td></tr>`; }).join('')}</table></div></details></div>`;
    h += `<div class="sec"><h3>2　安全层（${G.n} 种情形：人是大人 / 小孩 / 拿不准，在 4 个方向，离 5 cm 到 2 m，朝它走过来的速度 0 到 1.4 m/s，机器人 0.06 或 0.18 m/s，手臂干活或不干活）</h3><p>${chip(G.bad.length ? 'bad' : 'good', `“停住与否”和“减速到多少”一样 <b>${G.same}/${G.n}</b>`)}</p></div>`;
    h += `<div class="sec"><h3>3　电池</h3><div class="scroll"><table><tr><th>从</th><th class="num">真代码：充到 90% 要</th><th class="num">沙盘</th></tr>${B.charge.filter(r => r.from_soc < 0.89).map(r => { const b = new XS.Battery(app.D.battery, r.from_soc, 0); const m = b.chargeUntil(0.9) / 60; return `<tr><td>${Math.round(r.from_soc * 100)}%</td><td class="num">${r.minutes.toFixed(1)} 分钟</td><td class="num">${m.toFixed(1)} 分钟</td></tr>`; }).join('')}</table></div></div>`;
    h += `<div class="sec"><h3>4　同一个剧本，真实物理仿真 和 沙盘</h3><p class="note">这里规则相同，但移动、抓取、起身的用时是简化的，所以数字相近而不相同。差得多的地方就是沙盘不可信的地方。</p><div class="scroll" id="runs-tbl"><p class="note">正在跑…</p></div></div>`;
    setTimeout(async () => {
        const rows = await cmpRuns(); const t = $('#runs-tbl'); if (!t) return;
        t.innerHTML = `<table><tr><th>剧本</th><th>量</th><th class="num">真实仿真</th><th class="num">沙盘</th></tr>${rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td class="num">${r[2]}</td><td class="num">${r[3]}</td></tr>`).join('')}</table>`;
    }, 50);
    return h;
}

// ---------------------------------------------------------------------------------------------------------------- the notes page
function buildAbout() {
    const acc = app.D.accept || {}, items = acc.items || {}, vals = Object.values(items), pass = vals.filter(i => i.verdict === 'PASS').length;
    $('#about-body').innerHTML = `
    <div class="sec"><h3>什么是真的，什么是模型</h3><div class="scroll"><table>
      <tr><th>部分</th><th>这个页面里</th><th>怎么保证</th></tr>
      <tr><td>安全层（离人多远减速、停住、手臂在动时停下、孩子的距离更大）</td><td>真代码的规则，逐行重写</td><td>${app.D.golden.guard.length} 种情形，在“对照”页现场和真代码的答案比一遍</td></tr>
      <tr><td>听中文：唤醒词、停止词、拼音近似匹配、谁能让它做什么、拿不准先问</td><td>真代码的规则和同一张拼音表</td><td>${app.D.golden.voice.single.length} 句 × 4 种说话人和 ${app.D.golden.voice.dialogues.length} 段多轮对话，在“对照”页现场比（沙盘没做的会列出来）</td></tr>
      <tr><td>电池：充电曲线、读数的偏差、20% 和 10% 两条线</td><td>真代码的公式和常数</td><td>“对照”页比充电时间</td></tr>
      <tr><td>模式：收玩具、导航、扫描、遥控、充电、起身、待命、故障，谁能打断谁，打断后怎么接着做</td><td>真代码的逻辑，用同样结构的生成器重写</td><td>剧本对照（“对照”第 4 部分）；真实录像里的先后顺序一样</td></tr>
      <tr><td>机器人怎么走、怎么抓、怎么起身、功率</td><td><b>简化</b>：A* 路线 + 匀速；抓取是 9.6 秒计时 + 实测成功率；起身是录像里的计时；功率是录像里各状态的平均值</td><td>只能说数量级接近。准不准看“对照”第 4 部分</td></tr>
      <tr><td>它看到什么</td><td><b>简化</b>：相机 ±35°、5 米内、没有噪声；雷达看到在动的东西当“拿不准”</td><td>真实的深度图检测和分类在验收 E1、E2、E5 里测</td></tr>
      <tr><td>物理（摩擦、抓得住抓不住、摔倒的力学）、真实相机和雷达的数据</td><td><b>没有</b></td><td>看“回放”页的真实仿真录像，或在电脑上运行真的仿真</td></tr>
      <tr><td>英文、中英夹杂、记住人和忘掉人、给东西起名、声音从哪来、声纹、画像</td><td><b>没有</b>（说了会告诉你没做）</td><td>真实系统有，验收 E4、F2、G1–G4、H1</td></tr>
    </table></div></div>
    <div class="sec"><h3>真实验收结果</h3><p>真实物理仿真的 ${vals.length} 项验收：${pass} 项通过，${vals.length - pass} 项没通过（${vals.filter(i => i.verdict !== 'PASS').map(i => i.title ? Object.keys(items).find(k => items[k] === i) : '').join('、')}）。跑于 ${esc(acc.when || '')}，代码版本 ${esc(acc.code || '')}。逐项数字在验证页每一条的最后一行，完整的在仓库的 <span class="kbd">results/acceptance_sim.json</span>。没通过的 E3（人走近时刹得住）：安全层本身停得住、碰到人 0 次、走进 0.6 m 的场景它都停过；没过的是“5 件都送到”这一条：一家人的场景里，一个球被走过的人踢到沙发边上，机器人找不到能站的位置去夹它。后来试着做了“沿着家具面把它推开”的动作：在物理仿真里夹爪的手掌会碰到沙发，推的时候球又被卡在拨板和沙发之间，没成，所以这一条仍然没过（要换更窄的夹爪、带手腕偏转的夹爪，或者让人帮一把）。<b>已接受为已知限制</b>（2026-11-26，判定仍记“没通过”）。</p></div>
    <div class="sec"><h3>想看真的代码在跑</h3><p>这个页面不用装任何东西。想看真的 MuJoCo 仿真（有 3D 画面、真的控制器），要在电脑上运行：</p><div class="log" style="white-space:pre">cd sim\npython live.py --open</div><p class="note">在仓库里的 <span class="kbd">docs/playground.md</span> 里有安装步骤，和在 GitHub Codespaces（浏览器里的开发环境，不用装）里运行的办法（这一条我没有在 Codespaces 里试过）。</p></div>
    <div class="sec"><h3>这个页面怎么来的</h3><p class="note">数字、词表和真代码的答案由 <span class="kbd">sim/export_playground.py</span> 从仓库导出；回放由 <span class="kbd">sim/record_tapes.py</span> 在真实仿真里录；规则的 JavaScript 版在 <span class="kbd">web/playground/xs_logic.js</span> 和 <span class="kbd">xs_world.js</span>；<span class="kbd">node web/playground/test_core.mjs</span> 做“对照”页里的那些比较。</p></div>`;
}

// ---------------------------------------------------------------------------------------------------------------- the loop
let last = performance.now();
function loop(ts) {
    const dtw = Math.min((ts - last) / 1000, 0.1); last = ts;
    try {
        if (app.view === 'sim') {
            const s = app.sim;
            if (!app.paused) {
                app.acc += app.speed * dtw / s.dt; let n = Math.min(Math.floor(app.acc), 3000); app.acc -= Math.floor(app.acc); if (app.acc > 3000) app.acc = 0;
                for (let i = 0; i < n; i++) { while (app.script.length && app.script[0][0] <= s.t) runCmd(app.script.shift()[1]); if (app.keys.size && s.mode === 'TELEOP') { const k = app.keys; s.drive(((k.has('w') ? 1 : 0) - (k.has('s') ? 1 : 0)) * 0.18, ((k.has('a') ? 1 : 0) - (k.has('d') ? 1 : 0)) * 0.18, ((k.has('q') ? 1 : 0) - (k.has('e') ? 1 : 0)) * 1.0); } s.tick(); }
            }
            draw(viewOfSim());
            if (ts - app.lastHud > 120) { app.lastHud = ts; hudSim(); renderPanels(); runChecks(); }
        } else if (app.tape) {
            const T = app.tape.frames[app.tape.frames.length - 1][0];
            if (app.tapePlay) { app.tapeT += dtw * app.tapeSpeed; if (app.tapeT >= T) { app.tapeT = T - 0.01; app.tapePlay = false; $('#tape-play').textContent = '重放'; } }
            const v = viewOfTape(app.tape, app.tapeT); draw(v);
            if (ts - app.lastHud > 120) { app.lastHud = ts; hudTape(v); $('#tape-scrub').value = 100 * app.tapeT / T; $$('#tape-events div').forEach(d => d.style.color = +d.dataset.t <= app.tapeT ? 'var(--ink)' : ''); }
        }
    } catch (e) { console.error(e); const sb = $('#saybox'); if (sb) sb.textContent = '出错了：' + e.message; }
    requestAnimationFrame(loop);
}
window.XSApp = app;
boot();
})();
