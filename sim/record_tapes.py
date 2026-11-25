"""Record "tapes": a real run of the simulator (MuJoCo physics + the controller, modes, people, guard and voice code that the acceptance runs use) written down
at 5 frames a second, so that it can be played back in a web page without installing anything (docs/playground.md, the "回放" tab of web/playground).

    python record_tapes.py                 # all tapes, four at a time  -> ../web/playground/tapes/*.json
    python record_tapes.py kid fall        # only these
    python record_tapes.py --list

A tape holds the true position of the robot, the objects and the people, the robot's mode and what it is doing, the battery (true and what the battery board reports), the safety layer's
state and what the head camera's tracker believes about each person, plus the events the supervisor logged and what was said (to the robot and by it). Nothing is drawn: it is the
numbers of the run, not a video. The scenarios are the buttons of live.py, pressed at fixed simulated times (`do`), so a tape can be recorded again at any time and gives the
same run (the simulator is seeded).
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(os.path.dirname(HERE), 'web', 'playground', 'tapes')
HZ = 5.0

# name: title, what it shows, seed, starting battery, the schedule (simulated seconds: command, as typed into live.py), when to stop
TAPES = {
    'tidy': dict(title='收拾玩具', about='说“收拾一下玩具”：它从充电座出来，一个一个找、捡、送进收纳筐，收完回去充电。', seed=4, battery=0.9,
                 do='2:收拾一下玩具', t_end=1500.0, done='tidy'),
    'kid': dict(title='孩子冲过来', about='它在收玩具，一个 5 岁的孩子从门口跑过来停在它面前：看它减速、停下、等孩子走开再接着做。', seed=4, battery=0.9,
                do='2:/task; 40:/person kid', t_end=170.0),
    'lowbat': dict(title='电量低', about='收玩具时电量被调到 15%：它把手里这件放进筐里，停在两件事之间去充电，充完回来接着收。', seed=4, battery=0.9,
                   do='2:/task; 60:/battery 0.15', t_end=900.0, done='after_charge'),
    'fall': dict(title='被推倒', about='收玩具时被推倒：它自己判断摔倒，翻回来，再接着做。', seed=4, battery=0.9,
                 do='2:/task; 70:/push left', t_end=260.0),
    'family': dict(title='一家人在家', about='妈妈、8 岁的孩子和幼儿先后进来在房间里走：它一边收玩具，一边在有人靠近时让路、减速。', seed=4, battery=0.9,
                   do='2:/task; 15:/person family', t_end=420.0),
    'scan': dict(title='扫描家里', about='说“把家里扫描一遍”：它一个点一个点地走、转一圈，建三维地图，记下家具和地上的东西在哪里。', seed=4, battery=0.9,
                 do='2:把家里扫描一遍', t_end=900.0, done='scan'),
    'child_denied': dict(title='孩子要它去充电', about='孩子说“去充电”被拒绝，大人说同一句就去；孩子说“收拾玩具”可以。', seed=4, battery=0.9,
                         do='3:孩子：去充电; 9:孩子：收拾一下玩具; 14:停下; 19:去充电', t_end=60.0),
}


def snapshot_say(tape, live):
    from live import G
    orig = G.say
    def say(who, text):
        orig(who, text)
        t = live.sup.r.t if live.sup is not None else 0.0
        tape['say'].append([round(t, 1), who, text])
    G.say = say
    return orig


class Rec:
    """Built lazily so that importing this file (for --list) does not import MuJoCo."""
    pass


def record(name):
    import live as L
    import numpy as np
    from episode import make_layout
    from model import BASKET, DOCK_POSE, FURNITURE, HOME, ROOM_H, ROOM_W

    cfg = TAPES[name]
    modes, states = [], []
    tape = dict(name=name, title=cfg['title'], about=cfg['about'], seed=cfg['seed'], battery0=cfg['battery'], do=cfg['do'], hz=HZ, frames=[], events=[], say=[], modes=modes, states=states)
    objs0, boxes0 = make_layout(cfg['seed'])
    tape['world'] = dict(room=[ROOM_W, ROOM_H], furniture=[list(f) for f in FURNITURE], basket=BASKET, home=list(HOME), dock=list(DOCK_POSE), boxes=[list(b) for b in boxes0],
                         objects=[dict(kind=o[0], x=o[1], y=o[2], r=round(float(o[4]['radius']), 3)) for o in objs0])
    a = argparse.Namespace(seed=cfg['seed'], speed=1.0, battery=cfg['battery'], do=cfg['do'], tcp=0, host='127.0.0.1', port=0, open=False)

    class R(L.Live):
        def __init__(self):
            super().__init__(a); self.next_t = 0.0; self.t0_wall = time.time(); self.charged_once = False; self.seen_charge = False
            self.people_seen = {}

        def run(self):
            import people as P
            from episode import run_supervised
            names = list(P.SPECS); self.names = names; tape['people'] = [dict(i=i, name=n, cn=P.SPECS[n]['cn'], cls=P.SPECS[n]['cls'], h=P.SPECS[n]['h'], w=P.SPECS[n]['w']) for i, n in enumerate(names)]
            reg = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'xiaoshi_tape_registry_{os.getpid()}.json')
            pe = dict(names=names, who=names, family=['A'] * len(names), guard=True, comply=[1.0, 1.0, 0.7, 0.7, 0.0], steps=[[] for _ in names], cam='forearm_up', registry_path=reg)
            try:
                res, sup = run_supervised(cfg['seed'], dock=True, soc0=cfg['battery'], people=pe, start='idle', setup=self.setup, script=self.on_tick, on_step=self.on_step, t_end=cfg['t_end'],
                                          perturbation=dict(lidar_model='ld19cal'), until=self.until)
            finally:
                if os.path.exists(reg): os.remove(reg)
            tape['events'] = [[t, m, tx] for t, m, tx in sup.events]
            tape['result'] = dict(delivered=res['delivered'], total=res['total'], sim_time=res['sim_time'], mode=res['mode'], fault=res['fault'], sup_stats=res['sup_stats'], charge_log=res['charge_log'],
                                  fall_log=res['fall_log'], soc_end=res['soc_end'], guard=dict(self.guard.stats) if self.guard else None, wall_s=round(time.time() - self.t0_wall, 1))
            return res

        def setup(self, sup):
            from home import Home
            from voice import TextASR, VoiceSystem
            self.sup = sup; self.guard = sup.ppl.guard
            inv = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'xiaoshi_tape_inventory_{os.getpid()}.json')
            if os.path.exists(inv): os.remove(inv)
            home = Home(sup.r, sup.a, inv_path=inv); sup.home = home
            self.vs = VoiceSystem(sup, home, sup.ppl, asr=TextASR(), seed=cfg['seed'], setup_mode=True)
            snapshot_say(tape, self)

        def on_step(self, robot, agent):
            if robot.t + 1e-9 < self.next_t: return
            self.next_t += 1.0 / HZ
            sup = self.sup; g = sup.ppl.guard; ppl = sup.ppl
            if sup.mode not in modes: modes.append(sup.mode)
            if agent.state not in states: states.append(agent.state)
            x, y, th = robot.true_pose()
            n = len(robot.objects)
            ob = []
            for i in range(n):
                p = robot.obj_pos(i); ob += [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)]
            pe = []
            for p in ppl.people.truth():
                if p['inside']: pe.append([p['i'], round(p['x'], 3), round(p['y'], 3), round(p['yaw'], 2), round(p['sit'], 2)])
            tr = [[k.cls[0], round(float(k.x), 3), round(float(k.y), 3)] for k in ppl.tracks()]
            tr += [['l', round(float(k.x), 3), round(float(k.y), 3)] for k in ppl.movers.confirmed() if not any(math.hypot(k.x - c.x, k.y - c.y) < 0.6 for c in ppl.tracks())]       # what the lidar sees moving (the guard uses these too)
            delivered = sum(bool(agent._in_basket_world(robot.obj_pos(i))) for i in range(n))
            tape['frames'].append([round(robot.t, 1), round(x, 3), round(y, 3), round(th, 3), modes.index(sup.mode), states.index(agent.state), round(sup.bat.soc, 4) if sup.bat else None,
                                   round(sup.bat.soc_est, 4) if sup.bat else None, round(sup.tilt_imu, 1), int(bool(g.halted)) if g else 0, round(g.scale, 2) if g else 1.0, int(bool(agent.held)),
                                   delivered, int(bool(robot.dock_connected())), ob, pe, tr, (g.why if g and g.halted else '')])

        def until(self, sup):
            d = cfg.get('done')
            if d == 'tidy': return sup.task_done and sup.mode in ('IDLE', 'CHARGE') and self.sup.r.t > 30 and (sup.mode == 'IDLE' or sup.r.dock_connected())
            if d == 'after_charge':
                if sup.mode == 'CHARGE': self.seen_charge = True
                return self.seen_charge and sup.mode in ('IDLE',) or (self.seen_charge and sup.task_done)
            if d == 'scan': return sup.scan_done and sup.mode == 'IDLE'
            return False

    live = R()
    live.run()
    tape['frames'] = [f for f in tape['frames']]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'{name}.json')
    with open(path, 'w') as f: json.dump(tape, f, ensure_ascii=False, separators=(',', ':'))
    return name, os.path.getsize(path), len(tape['frames']), tape['result']['sim_time'], tape['result']['wall_s']


def main():
    ap = argparse.ArgumentParser(description='Record tapes of real simulator runs for the web playground.')
    ap.add_argument('names', nargs='*'); ap.add_argument('--list', action='store_true'); ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    if a.list:
        for k, v in TAPES.items(): print(f'{k:14s} {v["title"]}  ({v["do"]})')
        return
    names = a.names or list(TAPES)
    for n in names:
        if n not in TAPES: sys.exit(f'unknown tape {n}; --list shows them')
    with Pool(min(a.jobs, len(names))) as p:
        for name, size, nf, sim_t, wall in p.imap_unordered(record, names):
            print(f'{name:14s} {nf:5d} frames  {size / 1024:6.0f} KB  simulated {sim_t:6.0f} s  in {wall:5.0f} s of real time', flush=True)


if __name__ == '__main__':
    main()
