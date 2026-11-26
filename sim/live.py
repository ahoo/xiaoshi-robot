"""A live, interactive view of the simulated robot, in a web browser on your own computer.

    cd sim
    python live.py                     # then open http://127.0.0.1:8000   (python live.py --open does that for you)
    python live.py --seed 7 --speed 2  # another layout of the room, twice real time
    python live.py --do "5:/task; 60:/person kid; 120:/push left"      # no clicking: do these at those simulated seconds
    python live.py --tcp 9000          # also the remote-control TCP port: in another terminal  python teleop_client.py 127.0.0.1 9000

What you see: the room from above and from behind the robot, the map the robot builds for itself, what it is doing (mode, state, battery, how many things are in the basket, who it sees and whether it
takes them for a child), and its own log. What you can do: type to the robot in Chinese (it goes through the real voice pipeline: wake word, who is allowed to ask for what, asking back before it
acts), press the buttons, bring a child running in, push the robot over, switch the safety layer off, drive it with the keys W A S D Q E (after pressing 遥控), make the battery low.

It is the same code that the acceptance runs use (controller, modes, people, voice, home), with one difference: the clock is slowed to real time (--speed) and the browser is looking over its shoulder.
Nothing here needs a window system: pictures are rendered off screen (MUJOCO_GL=egl is set for you on Linux; on macOS and Windows the default is used) and sent as a motion-JPEG stream.
It listens on 127.0.0.1 only (--host 0.0.0.0 to let other computers on your network in: nothing is protected, do not do that on a network you do not trust).

Commands (typed in the box; a line that does not start with / is said to the robot; "孩子：" in front says it as a child, to see what a child is not allowed to ask for):
    /task  /stop  /scan  /charge  /teleop on|off  /status  /help
    /push left|right|fwd|back      knock the robot over (it should get up by itself)
    /person kid|adult|toddler|family|leave   bring people in, walking about; they are real simulated bodies with the head camera and the guard looking at them
    /guard on|off                  the safety layer between the modes and the wheels
    /battery 0.15                  set the true state of charge (when the battery board reads below 0.20 it finishes the thing in its hand and goes to the dock by itself)
    /raw text                      say it as it is, without the wake word put in front (chatter that is not meant for the robot)
    /speed 0.5|1|2|4               simulated seconds per second
"""
import argparse
import io
import json
import os
import queue
import sys
import tempfile
import threading
import time

if sys.platform.startswith('linux') and 'MUJOCO_GL' not in os.environ: os.environ['MUJOCO_GL'] = 'egl'      # off-screen rendering that needs no display

import numpy as np
import mujoco
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FRAME = (1280, 600)
MAIN_SZ, SIDE_SZ = (880, 495), (400, 300)

WORDS_PEOPLE = {'kid': 'kid', '孩子': 'kid', '小孩': 'kid', '孩子冲过来': 'kid', 'adult': 'adult', '大人': 'adult', '大人走过来': 'adult', '成人': 'adult', 'toddler': 'toddler', '幼儿': 'toddler',
                '幼儿在筐边': 'toddler', 'family': 'family', '一家人': 'family', '一家三口': 'family', 'leave': 'leave', '离开': 'leave', '走': 'leave'}
WORDS_PUSH = {'left': 'roll_l', '左': 'roll_l', 'right': 'roll_r', '右': 'roll_r', 'fwd': 'pitch_f', '前': 'pitch_f', 'back': 'pitch_b', '后': 'pitch_b'}
HELP = ('直接打字就是对机器人说话（会自动加上“小拾小拾，”），例如：收拾一下玩具 / 把家里扫描一遍 / 去充电 / 停下 / 继续 / 现在有谁 / 记住我，我叫爸爸。'
        '“孩子：去充电”是以孩子的身份说（孩子没有这个权限，会被拒绝）。\n'
        '/task 开始捡东西  /stop 停  /scan 扫描家里  /charge 去充电  /teleop on|off 遥控（再用 W A S D Q E 开）  /status\n'
        '/push left|right|fwd|back 推倒机器人  /person kid|adult|toddler|family|leave 让人走进来  /guard on|off 安全层  /battery 0.15 设电量  /speed 0.5|1|2|4  /raw 话 原样说（不自动加唤醒词，试试闲聊它会不会乱动）')


class Shared:
    """What the web server and the simulation loop hand to each other."""
    def __init__(self):
        self.cond = threading.Condition(); self.frame = b''; self.frame_id = 0
        self.status = {}; self.cmds = queue.Queue(); self.chat = []; self.drive = (0.0, 0.0, 0.0, -9.0); self.lock = threading.Lock(); self.last_view = 0.0   # last_view: when somebody last asked for a picture

    def put_frame(self, jpg):
        with self.cond: self.frame = jpg; self.frame_id += 1; self.cond.notify_all()

    def say(self, who, text):
        with self.lock: self.chat.append(dict(who=who, text=text)); self.chat[:] = self.chat[-40:]


G = Shared()


# ---------------------------------------------------------------------------------------------------------------- what the browser is shown
def _draw_process(model_path, conn):
    """The picture-drawing process: it has its own copy of the model, is sent the robot's and the people's positions, and sends back a JPEG. (Drawing without a graphics card takes a quarter of a second
    a picture, and the python bindings hold the interpreter's lock while they draw: in a thread it would stop the simulation; a process of its own does not. It also keeps macOS happy: OpenGL
    wants the main thread there, and in this process drawing is all the main thread does.)"""
    import mujoco
    m = mujoco.MjModel.from_binary_path(model_path); d = mujoco.MjData(m); m.vis.quality.shadowsize = 1024
    main = mujoco.Renderer(m, MAIN_SZ[1], MAIN_SZ[0]); side = mujoco.Renderer(m, SIDE_SZ[1], SIDE_SZ[0])
    for r in (main, side): r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0; r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0                 # half the time, and hardly visible at this size
    bg = (15, 42, 37)
    while True:
        msg = conn.recv()
        if msg is None: break
        qpos, mpos, mquat, map_arr = msg
        d.qpos[:] = qpos; d.mocap_pos[:] = mpos; d.mocap_quat[:] = mquat; mujoco.mj_forward(m, d)
        main.update_scene(d, 'overview'); a = Image.fromarray(main.render()); side.update_scene(d, 'chase'); b = Image.fromarray(side.render())
        im = Image.new('RGB', FRAME, bg); im.paste(a, (0, 0)); im.paste(b, (880, 0)); im.paste(Image.fromarray(map_arr), (880, 300))
        buf = io.BytesIO(); im.save(buf, 'JPEG', quality=80); conn.send(buf.getvalue())


class Eye:
    """The picture side, as seen from the simulation: hands the state to the drawing process and passes on the JPEGs it sends back (one at a time: a picture is requested only when the last has come)."""
    def __init__(self, robot):
        import multiprocessing as mp
        import render_video as RV
        self.RV = RV; self.busy = False
        self.path = os.path.join(tempfile.gettempdir(), f'xiaoshi_live_{os.getpid()}.mjb'); mujoco.mj_saveModel(robot.m, self.path, None)
        ctx = mp.get_context('spawn'); self.conn, child = ctx.Pipe(); self.proc = ctx.Process(target=_draw_process, args=(self.path, child), daemon=True); self.proc.start()
        threading.Thread(target=self._receive, daemon=True).start()

    def request(self, robot, agent):
        if self.busy or not self.proc.is_alive(): return
        self.busy = True; d = robot.d
        self.conn.send((d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy(), np.asarray(self.RV.draw_map(agent, robot))))

    def _receive(self):
        while True:
            try: jpg = self.conn.recv()
            except (EOFError, OSError): break
            G.put_frame(jpg); self.busy = False

    def close(self):
        try: self.conn.send(None)
        except Exception: pass
        if os.path.exists(self.path): os.remove(self.path)


# ---------------------------------------------------------------------------------------------------------------- the simulation side
class Live:
    def __init__(self, a):
        self.a = a; self.speed = float(a.speed); self.sup = None; self.vs = None; self.eye = None; self.guard = None; self.falls = []; self.pacer = None
        self.last_wall_frame = 0.0; self.last_wall_status = 0.0; self.todo = sorted(self.parse_do(a.do)); self.names = None

    @staticmethod
    def parse_do(s):
        out = []
        for part in (s or '').split(';'):
            part = part.strip()
            if not part: continue
            t, _, c = part.partition(':'); out.append((float(t), c.strip()))
        return out

    # ---- building the world
    def run(self):
        import people as P
        from episode import run_supervised
        a = self.a; names = list(P.SPECS); self.names = names
        reg = os.path.join(tempfile.gettempdir(), f'xiaoshi_live_registry_{os.getpid()}.json')
        pe = dict(names=names, who=names, family=['A'] * len(names), guard=True, comply=[1.0, 1.0, 0.7, 0.7, 0.0], steps=[[] for _ in names], cam='forearm_up', registry_path=reg)
        try:
            run_supervised(a.seed, dock=True, soc0=a.battery, people=pe, start='idle', setup=self.setup, script=self.on_tick, on_step=self.on_step, t_end=1e9,
                           perturbation=dict(lidar_model='ld19cal'), until=lambda s: False)
        finally:
            if os.path.exists(reg): os.remove(reg)
            if self.eye is not None: self.eye.close()

    def setup(self, sup):
        from home import Home
        from voice import TextASR, VoiceSystem
        self.sup = sup; self.guard = sup.ppl.guard
        inv = os.path.join(tempfile.gettempdir(), f'xiaoshi_live_inventory_{os.getpid()}.json')
        if os.path.exists(inv): os.remove(inv)
        home = Home(sup.r, sup.a, inv_path=inv); sup.home = home
        self.vs = VoiceSystem(sup, home, sup.ppl, asr=TextASR(), seed=self.a.seed, setup_mode=True)
        if self.a.tcp: from teleop_server import TeleopServer; self.tcp = TeleopServer(sup, port=self.a.tcp); G.say('系统', f'遥控 TCP 口开在 127.0.0.1:{self.a.tcp}（python teleop_client.py 127.0.0.1 {self.a.tcp}）。')
        self.eye = Eye(sup.r); G.say('系统', '准备好了。机器人停在充电座上，等你说话。输入 /help 看能做什么。')

    # ---- every control tick (20 ms of simulated time)
    def on_tick(self, sup):
        r = sup.r
        while True:
            try: text = G.cmds.get_nowait()
            except queue.Empty: break
            self.handle(text)
        while self.todo and self.todo[0][0] <= r.t:
            t, c = self.todo.pop(0); G.say('脚本', f'{t:.0f} s：{c}'); self.handle(c)
        self.falls = [f for f in self.falls if r.t < f[1] + 0.35]                    # the push lasts 0.25 s; it is called a little longer so that it lets go
        for fn, t0 in self.falls: fn()
        vx, vy, wz, t = G.drive
        if sup.mode == 'TELEOP' and time.time() - t < 0.4 and (vx or vy or wz): sup.drive(vx, vy, wz)

    def on_step(self, robot, agent):
        sup = self.sup
        if self.pacer is None: self.pacer = (time.time(), robot.t)
        target = self.pacer[0] + (robot.t - self.pacer[1]) / max(self.speed, 1e-3); dt = target - time.time()
        if dt > 0: time.sleep(min(dt, 0.05))
        now = time.time()
        if now - self.last_wall_frame > 0.07 and (now - G.last_view < 3.0 or G.frame_id == 0):          # nobody watching: do not spend time drawing
            self.last_wall_frame = now; self.eye.request(robot, agent)
        if now - self.last_wall_status > 0.2:
            self.last_wall_status = now; G.status = self.snapshot(robot, agent)

    @staticmethod
    def delivered(robot, agent):
        n = len(robot.objects); return sum(bool(agent._in_basket_world(robot.obj_pos(i))) for i in range(n)), n

    def snapshot(self, robot, agent):
        sup = self.sup; s = sup.status(); done, n = self.delivered(robot, agent); ppl = sup.ppl; g = self.guard
        present = [dict(who=p['name'], cls=p['cls'], x=round(p['x'], 2), y=round(p['y'], 2)) for p in ppl.people.truth() if p['x'] > 0]
        tracks = [dict(cls=k.cls, dist=round(float(k.dist), 2), who=k.who, h=round(float(k.height[0]), 2)) for k in ppl.tracks()]
        gs = None
        if g is not None: gs = dict(halted=bool(g.halted), why=g.why, halts=g.stats['halts'], halt_s=round(g.stats['halt_s'], 1), slow_s=round(g.stats['slow_s'], 1), arm_pauses=g.stats['arm_pauses'])
        with G.lock: chat = list(G.chat)
        s.update(delivered=done, total=n, speed=self.speed, guard_on=sup.ppl.guard is not None, guard=gs, present=present, tracks=tracks, chat=chat,
                 log=[(round(t, 1), m) for t, m in agent.log_lines[-14:]][::-1], events=[(t, md, tx) for t, md, tx in sup.events[-6:]][::-1], tts=[x[1] for x in self.vs.tts[-3:]] if self.vs else [],
                 battery_true=None if sup.bat is None else round(sup.bat.soc, 3), tele=sup.mode == 'TELEOP')
        return s

    # ---- what the user asked for
    def handle(self, text):
        text = (text or '').strip()
        if not text: return
        G.say('你', text)
        try:
            if text.startswith('/') or text.startswith('／'): self.command(text.lstrip('/／').strip())
            else: self.speak(text)
        except Exception as e:                                                      # a typo must not kill the simulation
            G.say('系统', f'出错了：{type(e).__name__}: {e}')

    def speak(self, text, raw=False):
        role = 'owner'
        for pre in ('孩子：', '孩子:', 'child:', 'child：'):
            if text.startswith(pre): role = 'child'; text = text[len(pre):].strip()
        if not raw and '小拾' not in text and 'xiaoshi' not in text.lower(): text = '小拾小拾，' + text
        before = len(self.vs.tts); rec = self.vs.hear(text, role=role)
        said = [x[1] for x in self.vs.tts[before:]] or ([rec['reply']] if rec.get('reply') else [])
        G.say('机器人' if said else '系统', ' / '.join(said) if said else f'（没有回应：{rec.get("action") or "没听到唤醒词"}）')

    def command(self, c):
        sup = self.sup; w = c.split(); k = w[0].lower() if w else ''; arg = w[1] if len(w) > 1 else ''
        if k in ('help', '帮助', 'h', '?'): G.say('系统', HELP)
        elif k in ('task', '任务', '开始', '捡'): sup.start_task(); G.say('系统', '开始捡东西。')
        elif k in ('stop', '停', '停下'): sup.idle(); G.say('系统', '停下（直接让监督层回到待命）。想试语音的“停下”，直接打“停下”。')
        elif k in ('scan', '扫描'): sup.scan_home(); G.say('系统', '开始扫描家里（它会一个点一个点地走，建三维地图）。')
        elif k in ('charge', '充电'): sup.charge_now(); G.say('系统', '去充电座。')
        elif k in ('teleop', '遥控'):
            if arg in ('off', '关'): sup.idle(); G.say('系统', '遥控关了。')
            else: sup.teleop(); G.say('系统', '遥控开了：W 前进 S 后退 A 左移 D 右移 Q 左转 E 右转（点一下画面再按）。松手它就停；安全层仍然看着人。')
        elif k in ('push', '推倒', '推'):
            from test_modes import inject_fall
            kind = WORDS_PUSH.get(arg or '左');
            if kind is None: G.say('系统', '推哪边？left right fwd back（左 右 前 后）'); return
            self.falls.append((inject_fall(sup.r, kind, 1.6), sup.r.t)); G.say('系统', f'推了一下（{kind}）。看它是不是自己起来。')
        elif k == 'raw': self.speak(c[3:].strip(), raw=True)                                  # as it is, without the wake word put in front: chatter that is not meant for the robot
        elif k in ('person', '人', '来人'): self.person(WORDS_PEOPLE.get(arg or 'kid'))
        elif k in ('guard', '安全层'):
            on = arg not in ('off', '关')
            sup.ppl.guard = self.guard if on else None; G.say('系统', '安全层 ' + ('开' if on else '关了：现在它不会因为人靠近而减速、停下（对照用）'))
        elif k in ('battery', '电量'):
            v = float(arg or 0.2)
            if sup.bat is None: G.say('系统', '这个仿真没有电池模型'); return
            sup.bat.soc = min(max(v, 0.0), 1.0); G.say('系统', f'真实电量设成 {sup.bat.soc:.2f}（机器人自己读到的会有偏差）。它读到低于 0.20 就会把手里的放好、自己去充电。')
        elif k in ('speed', '速度'):
            self.speed = float(arg or 1.0); self.pacer = None; G.say('系统', f'速度 {self.speed:g}×')
        elif k in ('status', '状态'): G.say('系统', json.dumps(sup.status(truth=True), ensure_ascii=False))
        else: G.say('系统', f'不认识 /{c}。输入 /help')

    def person(self, kind):
        import people as P
        if kind is None: G.say('系统', 'kid adult toddler family leave'); return
        S = self.sup.ppl.people.scripts; idx = {n: i for i, n in enumerate(self.names)}
        if kind == 'kid': S[idx['child_5']].steps.extend([('enter',), ('chase', 1.4, 0.12, 15.0), ('face_robot',), ('wait', 6.0), ('exit', 0.9)]); G.say('系统', '一个 5 岁的孩子从门口跑过来，停在离机器人很近的地方。看它减速、停下。')
        elif kind == 'adult': S[idx['adult_m']].steps.extend([('enter',), ('to_robot', 0.5, 1.0), ('exit', 1.0)]); G.say('系统', '一个大人走到机器人旁边再离开。')
        elif kind == 'toddler': S[idx['toddler']].steps.extend([('enter',), ('walk', 3.2, 2.35, 0.4), ('wait', 25.0), ('walk', 3.05, 2.45, 0.3), ('wait', 15.0), ('exit', 0.4)]); G.say('系统', '一个幼儿爬到收纳筐边玩。')
        elif kind == 'family':
            S[idx['adult_f']].steps.extend([('enter',), ('walk', 2.0, 1.1, 0.8), ('wait', 10.0), ('walk', 3.6, 1.5, 0.8), ('wait', 20.0), ('walk', 1.0, 2.6, 0.8), ('wait', 20.0), ('exit', 0.8)])
            S[idx['child_8']].steps.extend([('wait', 10.0), ('enter',), ('walk', 2.2, 1.9, 1.1), ('walk', 1.0, 1.0, 1.1), ('wait', 5.0), ('walk', 3.0, 2.4, 1.4), ('wait', 5.0), ('walk', 3.6, 1.9, 1.3), ('wait', 5.0), ('exit', 1.0)])
            S[idx['toddler']].steps.extend([('wait', 20.0), ('enter',), ('walk', 2.6, 1.7, 0.3), ('wait', 8.0), ('walk', 3.4, 2.2, 0.3), ('wait', 15.0), ('exit', 0.4)]); G.say('系统', '妈妈、8 岁的孩子和幼儿先后进来，在房间里走动。')
        elif kind == 'leave':
            for i in range(len(self.names)): S[i].steps.append(('exit', 1.0))
            G.say('系统', '房间里的人都离开。')


# ---------------------------------------------------------------------------------------------------------------- the web side
def serve(host, port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    html = open(os.path.join(HERE, 'live_ui.html'), 'rb').read()

    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *a): pass

        def _send(self, code, body, ctype):
            self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            p = self.path.split('?')[0]
            if p in ('/', '/index.html'): self._send(200, html, 'text/html; charset=utf-8')
            elif p == '/status': self._send(200, json.dumps(G.status, ensure_ascii=False, default=str).encode(), 'application/json; charset=utf-8')
            elif p == '/frame.jpg': G.last_view = time.time(); self._send(200, G.frame or b'', 'image/jpeg')
            elif p == '/stream':
                self.send_response(200); self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame'); self.send_header('Cache-Control', 'no-store'); self.send_header('Connection', 'close'); self.end_headers()
                last = -1
                try:
                    while True:
                        with G.cond:
                            while G.frame_id == last or not G.frame: G.cond.wait(1.0)          # (not before the first picture exists: an empty part would break the stream)
                            jpg, last = G.frame, G.frame_id
                        G.last_view = time.time(); self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                except (BrokenPipeError, ConnectionResetError, OSError): pass
            else: self._send(404, b'not found', 'text/plain')

        def do_POST(self):
            # Only the page itself may drive the robot: a web page on another site can send a "simple" POST (text/plain, no preflight) to 127.0.0.1 from the user's browser; requiring JSON makes the browser
            # ask first, and the Origin has to be this server's own when there is one.
            origin = self.headers.get('Origin')
            if 'application/json' not in (self.headers.get('Content-Type') or '') or (origin and origin.split('://', 1)[-1] != self.headers.get('Host')):
                self._send(403, b'{"ok":false}', 'application/json'); return
            try:
                n = int(self.headers.get('Content-Length', 0) or 0); body = json.loads(self.rfile.read(n) or b'{}')
                if not isinstance(body, dict): raise ValueError('not an object')
                if self.path == '/drive': vx, vy, wz = (float(body.get(k, 0)) for k in ('vx', 'vy', 'wz'))
            except (ValueError, TypeError):
                self._send(400, b'{"ok":false}', 'application/json'); return
            if self.path == '/cmd': G.cmds.put(str(body.get('text', ''))); self._send(200, b'{"ok":true}', 'application/json')
            elif self.path == '/drive':
                G.drive = (vx, vy, wz, time.time()); self._send(200, b'{"ok":true}', 'application/json')
            else: self._send(404, b'not found', 'text/plain')

    class Srv(ThreadingHTTPServer): daemon_threads = True
    srv = Srv((host, port), H); threading.Thread(target=srv.serve_forever, daemon=True).start(); return srv


def main():
    ap = argparse.ArgumentParser(description='Live, interactive view of the simulated robot in a browser.')
    ap.add_argument('--seed', type=int, default=4, help='which random layout of the room and the five things on the floor'); ap.add_argument('--speed', type=float, default=1.0, help='simulated seconds per real second')
    ap.add_argument('--battery', type=float, default=0.9, help='true state of charge at the start (0..1)'); ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--tcp', type=int, default=0, help='also open the remote-control TCP server (teleop_server.py) on this port, for teleop_client.py'); ap.add_argument('--open', action='store_true', help='open the page in the default browser'); ap.add_argument('--do', default='', help='"seconds:command; seconds:command": things to do at those simulated times')
    a = ap.parse_args()
    srv = serve(a.host, a.port); url = f'http://{a.host if a.host != "0.0.0.0" else "127.0.0.1"}:{a.port}'
    print(f'打开 {url}   （Ctrl+C 结束）', flush=True)
    if a.open:
        import webbrowser; webbrowser.open(url)
    try: Live(a).run()
    except KeyboardInterrupt: pass
    finally: srv.shutdown()


if __name__ == '__main__':
    main()
