"""Render one episode to an MP4: overview camera, chase camera, the robot's own map, and its log.

  MUJOCO_GL=egl python render_video.py [seed] [clearance] [out.mp4]
  MUJOCO_GL=egl python render_video.py clear [seed] [out.mp4]       the clear plastic bottle alone in the room (layout of 'materials'
                                                                    seed); the chase view is replaced by the front depth image, red =
                                                                    no reading, the patch the robot finds the bottle by

Frames are taken every 0.25 s of simulated time and played at 20 fps, i.e. 5x speed.
"""
import glob
import math
import sys

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from controller import CS, GX, GY
from episode import run_episode

W, H = 1280, 720
MAIN = (880, 495); SIDE = (400, 300)
COL = dict(bg=(15, 42, 37), paper=(241, 238, 226), dim=(164, 189, 177), orange=(255, 107, 44), warn=(255, 209, 102), free=(38, 98, 79), unk=(13, 36, 32))


def _font(size):
    hits = (sorted(glob.glob('/usr/share/fonts/**/NotoSansCJK-Medium.ttc', recursive=True)) or sorted(glob.glob('/usr/share/fonts/**/NotoSansCJK-Regular.ttc', recursive=True))
            or sorted(glob.glob('/usr/share/fonts/**/NotoSansCJK*.ttc', recursive=True)))
    if hits: return ImageFont.truetype(hits[0], size, index=2)      # index 2 = Simplified Chinese face
    return ImageFont.load_default()


def draw_map(agent, robot):
    """The robot's occupancy map, drawn as seen from above with the same orientation as the overview camera
    (x grows to the left, y grows downwards)."""
    lo = agent.lo; img = np.zeros((GY, GX, 3), np.uint8)
    img[lo == 0] = COL['unk']; img[lo < 0] = COL['free']; img[lo > 4] = COL['paper']
    im = Image.fromarray(img[:, ::-1]).resize(SIDE, Image.NEAREST); d = ImageDraw.Draw(im); k = SIDE[0] / (GX * CS)
    X = lambda x: (GX * CS - x) * k
    if agent.scan is not None:
        lx, ly, a, rr = agent.scan
        for x, y in zip(lx + np.cos(a) * rr, ly + np.sin(a) * rr): d.point((X(x), y * k), fill=COL['warn'])
    b = agent.basket; d.rectangle([X(b['x'] + b['sx'] / 2), (b['y'] - b['sy'] / 2) * k, X(b['x'] - b['sx'] / 2), (b['y'] + b['sy'] / 2) * k], outline=COL['orange'], width=2)
    x, y, th = agent.pose()
    if agent.path: d.line([(X(x), y * k)] + [(X(p[0]), p[1] * k) for p in agent.path[1:]], fill=COL['orange'], width=2)
    for j, e in agent.memory.items():
        if e['status'] != 'pending': continue
        # the robot does not know what the thing is: it draws the outline it measured
        ol = [(X(p[0]), p[1] * k) for p in e['outline']]
        if len(ol) >= 3: d.polygon(ol, fill=COL['warn'], outline=COL['unk'])
        r = 2; d.ellipse([X(e['x']) - r, e['y'] * k - r, X(e['x']) + r, e['y'] * k + r], fill=COL['unk'])
        if j == agent.target: d.ellipse([X(e['x']) - 9, e['y'] * k - 9, X(e['x']) + 9, e['y'] * k + 9], outline=COL['orange'], width=2)
    r = 0.14 * k; d.ellipse([X(x) - r, y * k - r, X(x) + r, y * k + r], fill=COL['paper'])
    d.line([(X(x), y * k), (X(x + 0.2 * math.cos(th)), (y + 0.2 * math.sin(th)) * k)], fill=COL['orange'], width=3)
    return im


def depth_view(agent):
    """The front depth image as the robot has it, no-reading pixels red, scaled to the side panel."""
    d = agent.cam.depth
    if d is None: return Image.new('RGB', SIDE, COL['unk'])
    img = np.zeros(d.shape + (3,), np.uint8); g = (255 * (1 - np.clip(d / 2.5, 0, 1))).astype(np.uint8)
    img[..., 0] = img[..., 1] = img[..., 2] = g
    img[d < 0.02] = (235, 60, 50)
    return Image.fromarray(img).resize(SIDE, Image.NEAREST)


def run_clear_episode(seed, on_step):
    """The bottle of the 'materials' layout of this seed, alone in the room; stops 3 s after it is in the basket."""
    import controller as C
    import perturb
    from controller import Agent, T_MAX
    from episode import make_layout
    from robot import Robot
    C.CLEAR_CUE, C.SLIP_ON = True, True; perturb.CUR = {}
    objs, boxes = make_layout(seed, 0.30, materials=True)
    robot = Robot([[o for o in objs if o[0] == 'bottle'][0]], seed=seed, extra_boxes=boxes)
    agent = Agent(robot, seed=seed); gen = agent.run(); t_done = None
    try:
        while robot.t < T_MAX + 60:
            next(gen); robot.step(); agent.after_step(); on_step(robot, agent)
            if agent._in_basket_world(robot.obj_pos(0)) and robot.t > 20:
                t_done = t_done or robot.t
                if robot.t - t_done > 3: break
    except StopIteration:
        pass
    res = dict(seed=seed, clearance=0.30, objects={'bottle': dict(mass_g=round(1000 * float(robot.m.body_subtreemass[robot.obj_body[0]]), 1), size_cm=round(200 * robot.objects[0][4]['radius'], 1))},
               delivered=int(agent._in_basket_world(robot.obj_pos(0))), total=1, sim_time=round(robot.t, 1), grasp_attempts=agent.stats['grasp_attempts'], grasp_fail=agent.stats['grasp_fail'], collisions=agent.stats['collisions'])
    return res, agent, robot


def main(seed=4, clearance=0.30, out='../results/episode.mp4', clear=False):
    f_big, f_mid, f_small = _font(34), _font(19), _font(15)
    state = dict(ren_main=None, ren_side=None, n=0)
    writer = imageio.get_writer(out, fps=20, codec='libx264', quality=None, macro_block_size=None, ffmpeg_params=['-crf', '27', '-preset', 'medium', '-pix_fmt', 'yuv420p'])

    def on_step(robot, agent):
        state['n'] += 1
        if state['n'] % 12 != 1: return          # every 0.24 s
        if state['ren_main'] is None:
            robot.m.vis.quality.shadowsize = 1024
            state['ren_main'] = mujoco.Renderer(robot.m, MAIN[1], MAIN[0]); state['ren_side'] = mujoco.Renderer(robot.m, SIDE[1], SIDE[0])
        rm, rs = state['ren_main'], state['ren_side']
        rm.update_scene(robot.d, 'overview'); main_img = rm.render()
        frame = Image.new('RGB', (W, H), COL['bg'])
        if clear: side = depth_view(agent)
        else: rs.update_scene(robot.d, 'chase'); side = Image.fromarray(rs.render())
        frame.paste(Image.fromarray(main_img), (0, 0)); frame.paste(side, (MAIN[0], 0)); frame.paste(draw_map(agent, robot), (MAIN[0], SIDE[1]))
        d = ImageDraw.Draw(frame)
        d.text((MAIN[0] + 8, 6), '前置深度相机：红色 = 没有读数' if clear else '跟拍', font=f_small, fill=COL['paper'] if not clear else (255, 255, 255)); d.text((MAIN[0] + 8, SIDE[1] + 6), '机器人自己建的地图', font=f_small, fill=COL['paper'])
        done = sum(agent._in_basket_world(robot.obj_pos(i)) for i in range(len(robot.objects)))
        y0 = MAIN[1] + 14
        d.text((20, y0), agent.state, font=f_big, fill=COL['orange'])
        d.text((150, y0 + 10), f'仿真时间 {int(robot.t // 60):02d}:{int(robot.t % 60):02d}   已送达 {done}/{len(robot.objects)}   5 倍速   MuJoCo 物理仿真', font=f_mid, fill=COL['paper'])
        for i, (t, msg) in enumerate(agent.log_lines[-5:][::-1]):
            d.text((20, y0 + 56 + i * 28), f'{int(t // 60):02d}:{int(t % 60):02d}  {msg}', font=f_mid, fill=COL['paper'] if i == 0 else COL['dim'])
        d.text((MAIN[0] + 8, 2 * SIDE[1] + 12), '白：障碍  绿：可通行  黄点：雷达回波', font=f_small, fill=COL['dim'])
        d.text((MAIN[0] + 8, 2 * SIDE[1] + 36), '橙线：规划路径  黄块：相机量到的物体', font=f_small, fill=COL['dim'])
        d.text((MAIN[0] + 8, 2 * SIDE[1] + 68), 'LeKiwi 三全向轮底盘 + SO-101 机械臂', font=f_small, fill=COL['paper'])
        writer.append_data(np.asarray(frame))

    res, agent, robot = run_clear_episode(seed, on_step) if clear else run_episode(seed, clearance, on_step=on_step)
    writer.close()
    import json
    json.dump({k: res[k] for k in ('seed', 'clearance', 'objects', 'delivered', 'total', 'sim_time', 'grasp_attempts', 'grasp_fail', 'collisions')}, open(out.replace('.mp4', '.json'), 'w'))
    rd = imageio.get_reader(out); n = rd.count_frames(); imageio.imwrite(out.replace('.mp4', '.jpg'), rd.get_data(int(n * 0.3)), quality=82); rd.close()
    print(res['delivered'], '/', res['total'], 'sim time', res['sim_time'], '->', out)
    return res


if __name__ == '__main__':
    a = sys.argv
    if len(a) > 1 and a[1] == 'clear': main(int(a[2]) if len(a) > 2 else 301, 0.30, a[3] if len(a) > 3 else '../results/clear.mp4', clear=True)
    else: main(int(a[1]) if len(a) > 1 else 4, float(a[2]) if len(a) > 2 else 0.30, a[3] if len(a) > 3 else '../results/episode.mp4')
