"""Remote control over TCP: one JSON object per line in, one JSON line back. For a keyboard, gamepad or phone app that talks to the robot's Pi.

  {"cmd": "drive", "vx": 0.1, "vy": 0.0, "wz": 0.0}        body-frame velocity (m/s, m/s, rad/s); repeat at 10-20 Hz, it stops 0.5 s after the last one
  {"cmd": "arm_joints", "q5": [0, -0.4, 0.7, 1.5, 0], "grip": 0.3}     joint angles (rad); either field may be left out
  {"cmd": "arm_to", "x": 0.25, "y": 0, "z": 0.06, "grip": null}       gripper point in the chassis frame; answers ok=false if unreachable
  {"cmd": "mode", "mode": "teleop" | "task" | "idle" | "charge"}
  {"cmd": "goto", "x": 2.0, "y": 1.5, "th": 1.57}          autonomous navigation to a point (th optional); {"cmd": "via", "points": [[x, y], ...], "loops": 1}; {"cmd": "explore"}
  {"cmd": "safety", "on": false}                            raw driving without the obstacle layer (the operator is then responsible)
  {"cmd": "status"}                                         mode, pose estimate, battery, tilt, warnings, last navigation results
Every reply has "ok". Commands are applied by the supervisor on its next control tick (queue), so the simulator and a real robot behave the same way.
"""
import json
import socketserver
import threading


def handle(sup, line):
    """Apply one command line to the supervisor; returns the reply dict. (Also the unit-testable core of the server.)"""
    try: c = json.loads(line)
    except Exception as e: return dict(ok=False, error=f'bad json: {e}')
    cmd = c.get('cmd')
    try:
        if cmd == 'drive': sup.drive(c.get('vx', 0.0), c.get('vy', 0.0), c.get('wz', 0.0))
        elif cmd == 'arm_joints': sup.arm_joints(c.get('q5'), c.get('grip'))
        elif cmd == 'arm_to':
            if not sup.arm_to(c['x'], c['y'], c['z'], grip=c.get('grip')): return dict(ok=False, error='unreachable or would hit the body')
        elif cmd == 'mode': sup.request(c['mode'])
        elif cmd == 'goto': sup.go_to(c['x'], c['y'], c.get('th'), c.get('tol', 0.07))
        elif cmd == 'via': sup.go_via([tuple(p) for p in c['points']], c.get('loops', 1))
        elif cmd == 'explore': sup.explore()
        elif cmd == 'safety': sup.set_safety(c.get('on', True))
        elif cmd == 'status':
            s = sup.status(); s['nav'] = sup.nav_results[-3:]; s['messages'] = sup.tele.messages[-3:]
            return dict(ok=True, **s)
        else: return dict(ok=False, error=f'unknown cmd {cmd!r}')
    except Exception as e:
        return dict(ok=False, error=f'{type(e).__name__}: {e}')
    return dict(ok=True)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            line = raw.decode('utf-8', 'replace').strip()
            if not line: continue
            self.wfile.write((json.dumps(handle(self.server.sup, line), ensure_ascii=False) + '\n').encode('utf-8')); self.wfile.flush()


class TeleopServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True; daemon_threads = True

    def __init__(self, sup, host='127.0.0.1', port=0):
        super().__init__((host, port), _Handler); self.sup = sup
        self.thread = threading.Thread(target=self.serve_forever, daemon=True); self.thread.start()

    @property
    def port(self): return self.server_address[1]

    def close(self): self.shutdown(); self.server_close()
