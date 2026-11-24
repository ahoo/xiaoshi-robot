"""Keyboard remote control for teleop_server.py, in a terminal. Run it on a computer that can reach the robot's Pi (or the simulator).

  python teleop_client.py [host] [port]

  w / s   forward / back          a / d   left / right (sideways)        q / e   turn left / right      space   stop
  1 2 3   arm: carry pose / reach down in front / open the gripper        4  close the gripper
  t       take over (teleop mode)  r  hand back to the job                 c  go and charge         x  idle         p  status         ESC  quit
A terminal reports key presses, not releases: a key keeps its command alive for 0.35 s, so the robot moves while a key is held (auto-repeat) and stops
a moment after you let go, and the robot's own dead-man timer (0.5 s) stops it if this program or the link dies.
"""
import json
import select
import socket
import sys
import time

DRIVE = {'w': (1, 0, 0), 's': (-1, 0, 0), 'a': (0, 1, 0), 'd': (0, -1, 0), 'q': (0, 0, 1), 'e': (0, 0, -1)}
V, W = 0.12, 0.6                       # m/s and rad/s per key
HOLD = 0.35                            # s a key press stays alive


def keys_to_cmd(pressed, now):
    """pressed: {key: time of its last press}. The drive command for this moment (sum of the live keys)."""
    vx = vy = wz = 0.0
    for k, t in pressed.items():
        if now - t < HOLD and k in DRIVE:
            a, b, c = DRIVE[k]; vx += a * V; vy += b * V; wz += c * W
    return dict(cmd='drive', vx=vx, vy=vy, wz=wz)


ONE_SHOT = {
    '1': dict(cmd='arm_joints', q5=[0.002, -0.393, 0.733, 1.523, 0.002], grip=0.3),
    '2': dict(cmd='arm_to', x=0.24, y=0.0, z=0.06),
    '3': dict(cmd='arm_joints', grip=1.1), '4': dict(cmd='arm_joints', grip=-0.17),
    't': dict(cmd='mode', mode='teleop'), 'r': dict(cmd='mode', mode='task'), 'c': dict(cmd='mode', mode='charge'), 'x': dict(cmd='mode', mode='idle'),
    'p': dict(cmd='status'),
}


def main(host='127.0.0.1', port=7700):
    import termios, tty
    s = socket.create_connection((host, port)); f = s.makefile('rw')
    ask = lambda d: (f.write(json.dumps(d) + '\n'), f.flush(), json.loads(f.readline()))[2]
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd); pressed = {}; last = 0.0
    try:
        tty.setcbreak(fd)
        while True:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                k = sys.stdin.read(1)
                if k == '\x1b': break
                if k == ' ': pressed.clear()
                elif k in DRIVE: pressed[k] = time.time()
                elif k in ONE_SHOT: print(ask(ONE_SHOT[k]), end='\r\n')
            if time.time() - last > 0.1: last = time.time(); ask(keys_to_cmd(pressed, last))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old); s.close()


if __name__ == '__main__':
    main(*(sys.argv[1:2] or ['127.0.0.1']), *([int(sys.argv[2])] if len(sys.argv) > 2 else []))
