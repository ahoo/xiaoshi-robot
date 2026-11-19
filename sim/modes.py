"""Operating modes and the supervisor that arbitrates between them.

  TASK    the autonomous fetch job (controller.Agent.run): find things, pick them up, take them to the basket
  NAV     autonomous navigation: to a point (optionally with a heading), through a list of points, or explore the room; nothing is picked up
  TELEOP  a person drives: velocity commands with a dead-man timeout, the arm by joint angle or by hand position; through the safety layer or raw
  CHARGE  to the dock, reverse in, charge (fast-forwarded), drive off
  RIGHT   get back on the wheels after a fall
  IDLE    standing still, powered, listening for requests
  FAULT   needs a person (could not get up, battery empty, dock unreachable, something stuck to the gripper): everything stopped

Who may interrupt whom (a "hard" switch closes the running generator; a "soft" stop is asked for and taken at the next safe point):
  fall            hard, beats everything (auto_right)
  battery empty   FAULT
  battery critical hard -> CHARGE, also over a person driving (crit_overrides_teleop)
  operator request hard (TELEOP, NAV, TASK, CHARGE, IDLE); not while getting up from a fall
  battery low     soft: TASK finishes the object in hand and stops between objects, NAV stops between goals; then CHARGE, then the job resumes
  job finished    CHARGE if the battery is not full, then IDLE

Everything runs in the simulator's control loop: Supervisor.tick() does what `next(gen); robot.step(); agent.after_step()` did, plus the monitors
(IMU for falls, battery) and the battery bookkeeping. Requests from another thread (the TCP server) go through a queue.
"""
import math
import queue

import numpy as np

import dock as dock_lib
from controller import V_MAX, W_MAX
from power import P_RUN_W
from robot import CTRL_DT, GRIP_OPEN, Q_CARRY
from model import DOCK_POSE
from skills import IK, carry_pose, wait

DEFAULTS = dict(
    soc_soft=0.20,            # battery board's state of charge below which the robot goes to charge at the next safe point
    soc_crit=0.10,            # below which it goes at once
    soc_full=0.90,            # true state of charge it charges to
    auto_charge=True, auto_right=True, crit_overrides_teleop=True,
    deadman=0.5,              # s: teleop commands older than this count as "stop"
    fall_tilt=45.0, fall_hold=0.3,   # IMU tilt (deg) held this long (s) is a fall: the static tipping angle of this robot is about 30
    nav_timeout=120.0, v_tele=V_MAX, w_tele=W_MAX, arm_speed_tele=1.0,
)
ARM_OUT = 0.25               # rad: an arm joint this far from the carry pose and the arm may be in front of the lidar


class TeleopInput:
    """The operator's latest commands. Anything may write here (a keyboard handler, a gamepad bridge, the TCP server); the supervisor reads it each tick."""
    def __init__(self):
        self.vx = self.vy = self.wz = 0.0; self.t = -1e9      # last drive command and when it came (simulator time)
        self.safety = True                                      # True: through the safety layer (stops short of obstacles); False: raw
        self.arm = None                                         # pending arm command: dict(q5=..., grip=...), taken by the supervisor
        self.messages = []


class Supervisor:
    def __init__(self, robot, agent, battery=None, cfg=None):
        self.r, self.a, self.bat = robot, agent, battery
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self.mode = 'IDLE'; self._kw = {}; self._gen = None; self._mode_t = robot.t
        self.q = queue.Queue(); self.tele = TeleopInput()
        self.events = []; self.stats = dict(charges=0, charge_min=0.0, falls=0, righted=0, right_failed=0, takeovers=0, nav_ok=0, nav_fail=0, dock_fail=0)
        self.task_done = False; self._task_started = False; self._looked = False; self._handback_needed = False; self.nav_explore = False
        self.resume = None                     # (mode, kwargs) to go back to after CHARGE
        self.resume_right = None               # same, after RIGHT
        self.nav_queue = []; self.nav_results = []
        self._running = False; self._deferred = None
        self._soft_req = False; self._fall_ticks = 0; self.tilt_imu = 0.0; self._last_j = robot.servo_j
        self.fault = None; self.warn = None; self.fall_log = []; self.charge_log = []
        self.right_fn = None                   # set by righting.attach(): generator function (supervisor) -> bool
        agent.pause = self._soft_stop

    # ------------------------------------------------------------------ public API (operator side)
    def request(self, mode, **kw):
        """Ask for a mode. Safe to call from another thread."""
        self.q.put((mode.upper(), kw))

    def start_task(self): self.request('TASK')
    def teleop(self): self.request('TELEOP')
    def idle(self): self.request('IDLE')
    def charge_now(self): self.request('CHARGE', reason='requested')

    def go_to(self, x, y, th=None, tol=0.07):
        """Navigate to (x, y) [and heading th]; tol is the position tolerance when no heading is given. Results land in nav_results."""
        self.request('NAV', goals=[dict(x=x, y=y, th=th, tol=tol)])

    def go_via(self, points, loops=1):
        """Visit a list of (x, y[, th]) in order, `loops` times (a patrol)."""
        g = [dict(x=p[0], y=p[1], th=(p[2] if len(p) > 2 else None), tol=0.07) for p in points] * loops
        self.request('NAV', goals=g)

    def explore(self): self.request('NAV', goals=[], explore=True)

    def drive(self, vx, vy, wz):
        """Teleop velocity command, body frame (m/s, m/s, rad/s). Must be repeated faster than the dead-man timeout."""
        self.tele.vx, self.tele.vy, self.tele.wz = float(vx), float(vy), float(wz); self.tele.t = self.r.t

    def arm_joints(self, q5=None, grip=None):
        self.tele.arm = dict(q5=None if q5 is None else np.asarray(q5, float), grip=grip)

    def arm_to(self, x, y, z, approach=(0, 0, -1), grip=None):
        """Put the gripper's pinch point at (x, y, z) in the chassis frame (z above the floor). Returns False if it cannot be reached."""
        q, err, pe = IK.solve([x, y, z], approach=approach)
        if pe > 0.01 or IK.hits_body(q): return False
        self.tele.arm = dict(q5=q, grip=grip); return True

    def set_safety(self, on): self.tele.safety = bool(on)

    def at_dock(self):
        """Does the robot think it is on the dock? (By its own pose estimate, within 12 cm of the docked pose; the charge-detect signal says so too.)"""
        if self.r.dock_pad < 0: return False
        x, y, _ = self.a.pose(); return self.r.dock_connected() or math.hypot(x - DOCK_POSE[0], y - DOCK_POSE[1]) < 0.12

    def status(self, truth=False):
        x, y, th = self.a.pose()
        s = dict(t=round(self.r.t, 1), mode=self.mode, state=self.a.state, pose=(round(x, 3), round(y, 3), round(th, 3)), tilt=round(self.tilt_imu, 1),
                 fault=self.fault, warn=self.warn)
        if self.bat is not None: s.update(soc=round(self.bat.soc_est, 3), volts=round(self.bat.voltage(), 2), dock=bool(self.r.dock_connected()))
        if truth: s['true_pose'] = tuple(round(v, 3) for v in self.r.true_pose()); s['soc_true'] = None if self.bat is None else round(self.bat.soc, 4)
        return s

    # ------------------------------------------------------------------ the loop
    def tick(self):
        """One 20 ms control period."""
        self._intake(); self._watch()
        if self._gen is not None:
            self._running = True; finished = False
            try: next(self._gen)
            except StopIteration: self._gen = None; finished = True
            finally: self._running = False
            if self._deferred is not None:                 # the mode asked for another mode while it was running: that wins over the default
                mode, kw = self._deferred; self._deferred = None
                if finished or self._gen is not None: self._switch(mode, **kw)
            elif finished: self._on_done()
        self.r.step(); self.a.after_step(); self._account()

    def run(self, t_end, on_step=None, until=None):
        while self.r.t < t_end:
            self.tick()
            if on_step: on_step(self.r, self.a)
            if until is not None and until(self): break

    # ------------------------------------------------------------------ bookkeeping
    def _event(self, text):
        self.events.append((round(self.r.t, 1), self.mode, text)); self.a.log(f'[{self.mode}] {text}')

    def _account(self):
        if self.bat is None: return
        dj = self.r.servo_j - self._last_j; self._last_j = self.r.servo_j
        self.bat.step(CTRL_DT, P_RUN_W + dj / CTRL_DT)

    def _soft_stop(self):
        """Asked by the autonomous jobs at their safe points (controller.Agent.pause)."""
        if self._soft_req: return True
        if self.bat is None or not self.cfg['auto_charge'] or self.mode not in ('TASK', 'NAV'): return False
        return self.bat.soc_est < self.cfg['soc_soft']

    # ------------------------------------------------------------------ switching
    def _switch(self, mode, **kw):
        if self._running: self._deferred = (mode, kw); return      # asked from inside the running mode: applied right after its current step
        if self._gen is not None: self._gen.close(); self._gen = None
        self.a.stop(); self.a.path = None
        self._event(f'→ {mode}' + (f' {kw}' if kw and mode != 'NAV' else ''))
        self.mode = mode; self._kw = kw; self._mode_t = self.r.t; self._soft_req = False
        self.a.arm_busy = False
        self._gen = getattr(self, '_gen_' + mode.lower())(**kw)

    def _intake(self):
        while True:
            try: mode, kw = self.q.get_nowait()
            except queue.Empty: return
            if self.mode in ('RIGHT', 'FAULT') and mode != 'IDLE':
                if self.mode == 'RIGHT': self.q.put((mode, kw)); return        # not while getting up: look at it next tick
                continue                                                          # FAULT: only a person's explicit reset (IDLE) gets out
            if mode == 'TELEOP' and self.mode != 'TELEOP': self.stats['takeovers'] += 1; self.resume = None
            if mode != 'TASK' and self.mode == 'TASK': self._handback_needed = True       # the job was cut off mid-way: tidy up when it comes back
            if mode == 'NAV':
                kw = dict(kw); self.nav_queue = list(kw.pop('goals', [])); self.nav_explore = bool(kw.get('explore', False))
            if mode == 'IDLE': self.fault = None
            self._switch(mode, **kw)

    def _watch(self):
        r, cfg = self.r, self.cfg
        up = r.imu(); self.tilt_imu = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
        frz = 2 if self.tilt_imu > 12.0 else (1 if self.mode == 'RIGHT' else 0)    # 2: lying or tipping, the sensors see the room at the wrong angle; 1: upright, pose unknown
        if frz and not self.a.freeze: self.a.pose_before_fall = self.a.loc.est.copy()
        self.a.freeze = frz
        self._fall_ticks = self._fall_ticks + 1 if self.tilt_imu > cfg['fall_tilt'] else 0
        if self._fall_ticks >= cfg['fall_hold'] / CTRL_DT and self.mode not in ('RIGHT', 'FAULT'):
            self.stats['falls'] += 1; self.fall_log.append(dict(t=round(r.t, 1), mode=self.mode, tilt=round(self.tilt_imu, 1)))
            self._event(f'摔倒了（IMU 倾角 {self.tilt_imu:.0f}°）')
            if cfg['auto_right'] and self.right_fn is not None:
                self.resume_right = (self.mode, dict(self._kw)); self._handback_needed = self._handback_needed or self.mode == 'TASK'; self._switch('RIGHT')
            else:
                self._fault('fall')
            return
        b = self.bat
        if b is None or self.mode in ('FAULT', 'RIGHT'): return
        if b.dead: self._fault('battery'); return
        e = b.soc_est
        self.warn = 'low battery' if e < cfg['soc_soft'] else None
        if e < cfg['soc_crit'] and self.mode not in ('CHARGE',) and cfg['auto_charge'] and (self.mode != 'TELEOP' or cfg['crit_overrides_teleop']):
            self.resume = ('TASK', dict(resume=True)) if self.mode == 'TASK' else (('NAV', dict(self._kw)) if self.mode == 'NAV' else None)
            if self.mode == 'TASK': self._handback_needed = True
            self._event(f'电量 {e:.0%}，低于紧急线，放下手里的事去充电'); self._switch('CHARGE', reason='critical')

    def _fault(self, why):
        self.fault = why; self._event(f'故障：{why}，需要人来处理'); self._switch('FAULT', why=why)

    def _on_done(self):
        mode, a = self.mode, self.a
        if mode == 'TASK':
            if a.state == '暂停':
                self.resume = ('TASK', dict(resume=True)); self._event('电量低，停在两件事之间去充电'); self._switch('CHARGE', reason='low battery')
            elif a.state == '求助': self._fault('payload stuck')
            else:
                self.task_done = True; self._event('任务结束')
                if self.bat is not None and self.cfg['auto_charge'] and self.bat.soc_est < self.cfg['soc_full'] - 0.05: self.resume = None; self._switch('CHARGE', reason='task done')
                else: self._switch('IDLE')
        elif mode == 'NAV':
            if (self.nav_queue or self.nav_explore) and self._soft_stop():
                self.resume = ('NAV', dict(explore=self.nav_explore)); self._event('电量低，停在两个目标之间去充电'); self._switch('CHARGE', reason='low battery')
            else: self._switch('IDLE')
        elif mode == 'CHARGE':
            if self.fault: return
            res, self.resume = self.resume, None
            if res is not None and not self._soft_stop(): self._switch(res[0], **res[1])
            else: self._switch('IDLE')
        elif mode == 'RIGHT':
            res, self.resume_right = self.resume_right, None
            if self.fault: return
            if res is None or res[0] in ('IDLE', 'RIGHT', 'FAULT'): self._switch('IDLE')
            else:
                kw = dict(res[1])
                if res[0] == 'TASK': kw.update(resume=True, handback=True)
                if res[0] == 'CHARGE': kw.setdefault('reason', 'after fall')
                self._switch(res[0], **kw)
        else:
            self._switch('IDLE')

    # ------------------------------------------------------------------ mode generators
    def _gen_idle(self):
        self.a.state = '待命'
        while True:
            self.a.stop()
            if self.bat is not None and self.cfg['auto_charge'] and self.bat.soc_est < self.cfg['soc_soft']:
                self.resume = None; self._event('待命时电量低，去充电'); self._switch('CHARGE', reason='low battery while idle'); return
            yield

    def _gen_fault(self, why=''):
        self.a.state = '求助'
        while True:
            self.a.stop(); yield

    def _handback(self):
        """Taking the robot back from an operator: whatever the gripper holds is set down in front, the arm goes to the carry pose, and what the job
        believed about 'in hand' is dropped (the camera finds the object again where it lies)."""
        a, r = self.a, self.r
        a.state = '交接'
        r.set_arm(grip=GRIP_OPEN, speed=1.0); yield from wait(r, 0.6)
        yield from carry_pose(r, speed=1.0); r.set_arm(grip=0.3); yield from wait(r, 0.3)
        a.arm_busy = False
        for k in [k for k, e in a.memory.items() if e['status'] == 'held']: del a.memory[k]
        a.held = False; a.target = None; a.priority = None; a.path = None; a.slip_flag = False
        self._handback_needed = False

    def _gen_task(self, resume=None, handback=None):
        a = self.a
        if self.at_dock(): yield from dock_lib.pull_out(a)
        if resume is None: resume = self._task_started or self._looked
        if handback or (handback is None and self._handback_needed): yield from self._handback()
        self._task_started = True; self._looked = True
        yield from a.run(resume=resume)

    def _gen_nav(self, explore=False):
        a, r, cfg = self.a, self.r, self.cfg
        if self.at_dock(): yield from dock_lib.pull_out(a)
        if not self._looked and not explore:                      # a fresh map: look round once first, as the job does at start-up
            a.state = '巡视'; self._event('第一次导航，先原地环顾建图'); yield from wait(r, 0.5); yield from a.spin(); self._looked = True
        a.state = '导航'
        if explore:
            self._event('自主巡视，只建图不捡东西')
            while True:
                if self._soft_stop(): a.stop(); return
                more = yield from a.explore_step(stop_on_find=False)
                if not more: break
            self.nav_explore = False; self._event('整个房间看过了'); self.nav_results.append(dict(t=round(r.t, 1), goal='explore', result='done')); return
        while self.nav_queue:
            if self._soft_stop(): return
            g = self.nav_queue[0]; x, y, th, tol = g['x'], g['y'], g['th'], g['tol']
            self._event(f'去 ({x:.2f}, {y:.2f})' + ('' if th is None else f' 朝向 {math.degrees(th):.0f}°'))
            a.state = '导航'; t0 = r.t
            goal = yield from a.goto(lambda D: [(x, y, th, tol if th is None else 0.0)], timeout=cfg['nav_timeout'])
            ex, ey, eth = a.pose()
            res = dict(t=round(r.t, 1), goal=(x, y, th), result='arrived' if goal is not None else a.abort, secs=round(r.t - t0, 1))
            self.nav_results.append(res); self.stats['nav_ok' if goal is not None else 'nav_fail'] += 1
            self._event(f'{"到了" if goal is not None else "没到：" + str(a.abort)}（{res["secs"]} 秒）')
            self.nav_queue.pop(0)

    def _gen_teleop(self):
        a, r, tc, cfg = self.a, self.r, self.tele, self.cfg
        a.state = '遥控'; a.path = None; tc.vx = tc.vy = tc.wz = 0.0
        self._event('遥控开始' + ('（带安全层）' if tc.safety else '（直连，无安全层）'))
        stale = True
        while True:
            if tc.arm is not None:
                cmd, tc.arm = tc.arm, None
                q5 = cmd['q5']
                if q5 is not None:
                    if IK.hits_body(q5): tc.messages.append('arm command refused: the arm would hit the chassis')
                    else: r.set_arm(q5=np.clip(q5, IK.lo, IK.hi), speed=cfg['arm_speed_tele'])
                if cmd['grip'] is not None: r.set_arm(grip=float(cmd['grip']))
            a.arm_busy = bool(np.abs(r.joints()[:4] - Q_CARRY[:4]).max() > ARM_OUT)
            live = r.t - tc.t < cfg['deadman']
            if live: stale = False
            elif not stale: stale = True; self._event('指令超时，停车（死人开关）')
            vx, vy, wz = (tc.vx, tc.vy, tc.wz) if live else (0.0, 0.0, 0.0)
            sp = math.hypot(vx, vy)
            if sp > cfg['v_tele']: vx, vy = vx * cfg['v_tele'] / sp, vy * cfg['v_tele'] / sp
            wz = float(np.clip(wz, -cfg['w_tele'], cfg['w_tele']))
            if math.degrees(r.tilt()) > 8.0: vx = vy = wz = 0.0            # tilted: no driving until it has settled
            if tc.safety: a.cmd(vx, vy, wz)
            else: r.set_twist(vx, vy, wz)
            yield

    def _gen_charge(self, reason=''):
        a, r, b = self.a, self.r, self.bat
        if b is None: return
        self._event(f'去充电（{reason}），电量 {b.soc_est:.0%}')
        if not r.dock_connected():
            ok = yield from dock_lib.dock(a, tries=3)
            if not ok:
                self.stats['dock_fail'] += 1; self._fault('dock unreachable'); return
        a.stop(); a.state = '充电'; yield from wait(r, 1.0)
        soc0 = b.soc; secs = b.charge_until(self.cfg['soc_full'])
        self.stats['charges'] += 1; self.stats['charge_min'] += secs / 60
        self.charge_log.append(dict(t=round(r.t, 1), reason=reason, soc_from=round(soc0, 3), soc_to=round(b.soc, 3), minutes=round(secs / 60, 1)))
        self._event(f'充电 {soc0:.0%} → {b.soc:.0%}，用了 {secs / 60:.0f} 分钟（仿真里快进，物理时间没走）')
        self.warn = None
        if self.resume is not None: yield from dock_lib.pull_out(a)       # something to go back to: off the dock; otherwise stay put, charged

    def _gen_right(self):
        yield from self.right_fn(self)
