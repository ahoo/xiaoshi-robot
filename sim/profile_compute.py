"""How much computing the robot's own software needs: one full episode (standard seed 4) with a timer around every
algorithm the real robot would run, and nothing of the simulator (physics, rendering: on the robot the cameras and
the lidar do that themselves). -> results/compute_profile.json

  python profile_compute.py

The numbers are for this machine. To size the Raspberry Pi 5, compare one benchmark run on both (see the result's
'machine' field), or run the same timings there on recorded frames.
"""
import json
import platform
import time
from collections import defaultdict

import mujoco

import controller
import episode
import localization
import perception
import skills

T = defaultdict(float); N = defaultdict(int)


def timed(name, fn):
    def w(*a, **k):
        t0 = time.perf_counter(); r = fn(*a, **k); T[name] += time.perf_counter() - t0; N[name] += 1; return r
    return w


def timed_gen(name, fn):
    def w(*a, **k):
        g = fn(*a, **k)
        try:
            while True:
                t0 = time.perf_counter(); v = next(g); T[name] += time.perf_counter() - t0
                yield v
        except StopIteration as e:
            return e.value
    return w


# what the simulator does on the robot's behalf: excluded by timing it separately and subtracting
_render = mujoco.Renderer.render
mujoco.Renderer.render = timed('_render', _render)
_upd = mujoco.Renderer.update_scene
mujoco.Renderer.update_scene = timed('_render', _upd)
perception.FrontCamera.detect = timed('camera: segment and measure', perception.FrontCamera.detect)
perception.WristCamera.load = timed('wrist camera check', perception.WristCamera.load)
localization.Localizer.correct = timed('lidar scan matching', localization.Localizer.correct)
localization.Localizer.add_scan = timed('lidar map update', localization.Localizer.add_scan)
controller.Agent.costmap = timed('cost map', controller.Agent.costmap)
controller.Agent.astar = timed('path search (A*)', controller.Agent.astar)
controller.Agent.cmd = timed('safety layer', controller.Agent.cmd)
controller.Agent.grasp_dirs = timed('grasp side search', controller.Agent.grasp_dirs)
skills.IK.solve = timed('arm inverse kinematics', skills.IK.solve)


if __name__ == '__main__':
    t0 = time.perf_counter()
    res, agent, robot = episode.run_episode(4, 0.30)
    wall = time.perf_counter() - t0
    sim = res['sim_time']
    # rendering happened inside detect/load: take it out of those
    n_cam = N['camera: segment and measure']; n_w = N['wrist camera check']
    rend_per = T['_render'] / max(1, N['_render'])
    T['camera: segment and measure'] -= 4 * rend_per * n_cam          # seg + depth, scene update + render each
    T['wrist camera check'] -= 4 * rend_per * n_w
    out = {k: dict(calls=N[k], ms_per_call=round(1000 * T[k] / max(N[k], 1), 2), share_of_one_core=round(T[k] / sim, 4))
           for k in sorted(T) if not k.startswith('_')}
    total = sum(v['share_of_one_core'] for v in out.values())
    cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.processor())
    res = dict(machine=cpu, sim_time=sim, wall_time=round(wall, 1), modules=out, total_share_of_one_core=round(total, 3))
    json.dump(res, open('../results/compute_profile.json', 'w'), ensure_ascii=False, indent=1)
    print(cpu, f'sim {sim:.0f} s'); [print(f'  {k:<26} {v["calls"]:>6} calls  {v["ms_per_call"]:>7.2f} ms  {100 * v["share_of_one_core"]:5.1f} % of one core') for k, v in out.items()]
    print(f'  total {100 * total:.1f} % of one core of this machine')
