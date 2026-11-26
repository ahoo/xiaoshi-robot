// Runs the scenarios of the playground in node, with no page. Run: node web/playground/test_world.mjs [name]
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
require('./xs_logic.js'); require('./xs_world.js');
const XS = globalThis.XS;
const M = JSON.parse(fs.readFileSync(path.join(here, 'data', 'model.json'), 'utf8'));
const only = process.argv[2];
let bad = 0;
const check = (ok, msg) => { console.log(ok ? '  ok  ' : '  FAIL', msg); if (!ok) bad++; };
const say = (s, text, role = 'owner') => s.voice.hear('小拾小拾，' + text, role);
const modes = s => s.events.filter(e => e[2].startsWith('→')).map(e => `${e[0]}:${e[2].slice(2).split(' ')[0]}`).join(' ');
const scenarios = {
  tidy() {
    const s = new XS.Sim(M, { seed: 4, soc0: 0.9 }); const r = say(s, '收拾一下玩具'); check(r.action === 'tidy', 'voice tidy → ' + r.reply);
    const t0 = Date.now(); let t = 0; while (t < 2000 && !s.taskDone) { s.run(10); t += 10; }
    check(s.taskDone, `task finished at ${s.t.toFixed(0)} s (wall ${Date.now() - t0} ms)`); check(s.delivered() === 5, `delivered ${s.delivered()}/5`);
    s.run(120); console.log('  modes:', modes(s), ' soc', s.bat.soc.toFixed(3), 'mode', s.mode, 'state', s.state);
    check(s.mode === 'IDLE' || s.mode === 'CHARGE', 'ends charging or idle');
  },
  kid() {
    const s = new XS.Sim(M, { seed: 4, soc0: 0.9 }); say(s, '收拾一下玩具'); s.run(40); s.person('kid'); let minGap = 9, halted = 0;
    for (let i = 0; i < 130 / s.dt; i++) { s.tick(); for (const p of s.people.state) if (p.inside) minGap = Math.min(minGap, Math.hypot(p.x - s.x, p.y - s.y) - 0.105 - p.w / 2); if (s.guard.halted) halted++; }
    console.log('  guard', JSON.stringify(s.guard.stats), 'episodes', JSON.stringify(s.guard.episodes), 'min gap', minGap.toFixed(2), 'contacts', s.stats.contacts);
    check(s.guard.stats.halts >= 1, 'the kid made it halt'); check(s.stats.contacts === 0, 'no contact');
    s.guardOn = false; console.log('  (off) skipped');
  },
  kidoff() {
    const s = new XS.Sim(M, { seed: 4, soc0: 0.9 }); say(s, '收拾一下玩具'); s.run(40); s.guardOn = false; s.person('kid'); s.run(60);
    console.log('  contacts with the guard off:', s.stats.contacts, 'min gap', Math.min(...s.people.minGap).toFixed(2)); check(s.stats.contacts >= 0, 'ran');
  },
  lowbat() {
    const s = new XS.Sim(M, { seed: 4, soc0: 0.9 }); say(s, '收拾一下玩具'); s.run(60); s.bat.soc = 0.15 - s.bat.bias; s.run(30);
    let t = 0; while (t < 1500 && !s.taskDone) { s.run(10); t += 10; }
    console.log('  modes:', modes(s)); console.log('  events:', s.events.filter(e => /电量|充电/.test(e[2])).map(e => `${e[0]} ${e[2]}`).join(' | '));
    check(s.stats.charges >= 1, 'charged'); check(s.delivered() === 5 || s.taskDone, `delivered ${s.delivered()}/5`);
  },
  fall() {
    const s = new XS.Sim(M, { seed: 4, soc0: 0.9 }); say(s, '收拾一下玩具'); s.run(70); s.push('roll_l'); s.run(60);
    console.log('  modes:', modes(s)); check(s.stats.falls === 1 && s.stats.righted === 1, `falls ${s.stats.falls} righted ${s.stats.righted}`); check(s.mode === 'TASK', 'back to TASK: ' + s.mode);
  },
  scan() {
    const s = new XS.Sim(M, { seed: 4, soc0: 0.9 }); let r = say(s, '去沙发那边'); console.log('  ', r.reply); r = say(s, '对'); console.log('  ', r.reply); check(/不知道/.test(r.reply), 'does not know the sofa before a scan');
    say(s, '把家里扫描一遍'); s.run(1); let t = 0; while (t < 600 && s.mode !== 'IDLE') { s.run(5); t += 5; } s.run(1);
    console.log('  scan done', s.scanDone, 'at', s.t.toFixed(0), 's'); check(s.scanDone, 'scan finished');
    r = say(s, '去沙发那边'); console.log('  ', r.reply); r = say(s, '对'); console.log('  ', r.reply);
    s.run(1); t = 0; while (t < 200 && s.mode === 'NAV') { s.run(2); t += 2; } console.log('  nav', JSON.stringify(s.navResults.slice(-1)), 'at', s.x.toFixed(2), s.y.toFixed(2)); check(s.navResults.some(r => r.result === 'arrived'), 'arrived next to the sofa');
  },
  voice() {
    const s = new XS.Sim(M, { seed: 4 }); const lines = ['你好', '你在干什么', '现在有谁', '孩子：去充电', '去充电', '把小球放进筐里', '小球在哪', '地上有什么东西'];
    for (const l of lines) { const role = l.startsWith('孩子：') ? 'child' : 'owner'; const r = say(s, l.replace('孩子：', ''), role); console.log('  ', role, l, '→', r.action, '«' + r.reply + '»'); s.run(2); }
  },
  teleop() {
    const s = new XS.Sim(M, { seed: 4, soc0: 0.9 }); s.start = 0; s.request('TELEOP'); s.run(1); check(s.mode === 'TELEOP', 'teleop on'); for (let i = 0; i < 100; i++) { s.drive(0.1, 0, 0); s.tick(); }
    console.log('  moved to', s.x.toFixed(2), s.y.toFixed(2)); s.run(2); const v = Math.hypot(s.vx, s.vy); check(v < 0.01, 'stopped after the hand is off the stick (deadman): ' + v.toFixed(3));
  },
};
for (const [n, f] of Object.entries(scenarios)) if (!only || only === n) { console.log('--', n); try { f(); } catch (e) { console.log('  EXCEPTION', e.stack); bad++; } }
console.log(bad ? `${bad} failed` : 'all ok'); process.exit(bad ? 1 : 0);
