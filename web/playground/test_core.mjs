// Checks the JavaScript rules against what the real code answered (data/model.json "golden"). Run: node web/playground/test_core.mjs
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
require('./xs_logic.js');
const XS = globalThis.XS;
const M = JSON.parse(fs.readFileSync(path.join(here, 'data', 'model.json'), 'utf8'));
let fails = 0;
const note = (ok, msg) => { if (!ok) { fails++; console.log('  DIFF', msg); } return ok; };

// ---- voice: single sentences
const g = M.golden.voice; let same = 0, diff = 0, skip = 0; const diffs = [];
for (const r of g.single) {
  const v = new XS.Voice(M.voice, M.pinyin, null);
  const rec = v.hear(r.text, r.role);
  const reply = rec.reply || null;
  if (rec.outOfScope) { skip++; continue; }
  const ok = (rec.intent ?? null) === (r.intent ?? null) && (rec.action ?? null) === (r.action ?? null) && reply === r.reply && (r.conf == null || rec.conf == null || Math.abs(rec.conf - r.conf) < 0.011);
  if (ok) same++; else { diff++; diffs.push([r, rec]); }
}
console.log(`voice sentences: ${same} same, ${diff} different, ${skip} not modelled (of ${g.single.length})`);
for (const [r, rec] of diffs.slice(0, 40)) console.log(`  ${r.role.padEnd(7)} ${r.text}  real: ${r.intent}/${r.action}/${r.conf} «${r.reply}»  js: ${rec.intent}/${rec.action}/${rec.conf} «${rec.reply}»`);

// ---- voice: dialogues
let dsame = 0, ddiff = 0;
for (const d of g.dialogues) {
  const v = new XS.Voice(M.voice, M.pinyin, null); let ok = true; const lines = [];
  for (const t of d.turns) {
    v.advance(t.wait); const rec = v.hear(t.text, t.role);
    const same = (rec.intent ?? null) === (t.intent ?? null) && (rec.action ?? null) === (t.action ?? null) && (rec.reply ?? null) === (t.reply ?? null);
    if (!same) { ok = false; lines.push(`    turn «${t.text}» real: ${t.intent}/${t.action} «${t.reply}»  js: ${rec.intent}/${rec.action} «${rec.reply}»`); }
  }
  if (ok) dsame++; else { ddiff++; console.log('  DIALOGUE DIFF', d.name); lines.forEach(l => console.log(l)); }
}
console.log(`voice dialogues: ${dsame} same, ${ddiff} different (of ${g.dialogues.length})`);

// ---- guard
let gsame = 0, gdiff = 0; const gd = [];
const K = M.guard;
for (const r of M.golden.guard) {
  const gu = new XS.Guard(K); const b = r.bearing * Math.PI / 180; const rr = K.robot_r + K.person_r, d = rr + r.gap;
  const px = 2 + d * Math.cos(b), py = 1.5 + d * Math.sin(b);
  const k = { id: 1, cls: r.cls, x: px, y: py, vx: -r.closing * Math.cos(b), vy: -r.closing * Math.sin(b), coast: false, predict() { return [this.x, this.y]; } };
  const S = { t: 100, mode: 'TASK', state: '前往', armBusy: r.arm, armDone: () => true, pose: () => [2, 1.5, 0], zoneSkipT: -1e9, tracks: () => [k], wheelTarget: { vx: 0, vy: 0, wz: 0 }, setTwist() {}, holdArm: false, dynZones: [], personZones: [], event() {}, speak() {}, onAsk: null };
  const halted = !!gu.pre(S); S.wheelTarget = { vx: r.v, vy: 0, wz: 0 }; gu.post(S);
  if (halted === r.halted && Math.abs(gu.scale - r.scale) < 1e-4) gsame++; else { gdiff++; gd.push([r, halted, gu.scale]); }
}
console.log(`guard situations: ${gsame} same, ${gdiff} different (of ${M.golden.guard.length})`);
for (const [r, h, s] of gd.slice(0, 12)) console.log('  ', JSON.stringify(r), 'js halted', h, 'scale', s.toFixed(4));

// ---- battery
for (const r of M.golden.battery.charge) {
  const b = new XS.Battery(M.battery, r.from_soc, 0); const mins = b.chargeUntil(0.9) / 60;
  note(Math.abs(mins - r.minutes) < 0.06 && Math.abs(b.soc - r.final) < 1e-3, `charge from ${r.from_soc}: real ${r.minutes} min, js ${mins.toFixed(2)}`);
}
for (const r of M.golden.battery.voltage) { const b = new XS.Battery(M.battery, r.soc, 0); note(Math.abs(b.voltage(M.battery.p_run_w) - r.volts_run) < 2e-3, `voltage at ${r.soc}`); }
console.log(fails ? `${fails} problems` : 'battery: same');
process.exit(0);
