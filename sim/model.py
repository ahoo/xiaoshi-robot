"""Builds the MuJoCo model: LeKiwi-style 3-omni-wheel base + SO-101 arm + lidar, in a furnished room.

Everything physical lives here so it can be checked against the real hardware:
wheel/roller geometry, masses, friction, servo torque limits.
"""
import math
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import perturb

import objects as objects_lib

HERE = os.path.dirname(os.path.abspath(__file__))
SO101_DIR = os.path.join(HERE, '..', 'third_party', 'so101')

# ---- hardware parameters (measure these on the real robot and write the values back) ----
P = dict(
    wheel_radius=0.0508,      # 4" omni wheel
    base_radius=0.125,        # wheel contact circle, as in LeRobot's LeKiwi kinematics
    wheel_angles=(60.0, 180.0, 300.0),  # front-left, back, front-right (deg from +x = forward)
    rollers_per_row=8,
    roller_b=0.009,           # roller radius
    roller_row_offset=0.0065,
    roller_mass=0.004,
    hub_mass=0.070,
    chassis_mass=1.295,       # plates, mounts, 3 servos, battery, Pi, board, camera, lidar (estimate)
    chassis_com=(-0.015, 0.0, 0.045),
    servo_stall=2.94,         # N*m, STS3215 12 V stall torque (30 kg*cm)
    gripper_torque=1.0,       # N*m software limit on the gripper servo
    arm_mount=(0.03, 0.0, 0.0862),   # arm base origin relative to the wheel-axle plane centre
    lidar_pos=(-0.07, 0.0, 0.146),   # scan plane: 0.197 m above the floor
    floor_friction=0.9,
    pad_friction=1.0,         # jaws with grip tape; bare PLA is about 0.35
)

ROOM_W, ROOM_H = 4.0, 3.0
# static furniture: (name, cx, cy, sx, sy, height, rgba); circles use sx as radius and sy=None
FURNITURE = [
    ('sofa', 0.95, 0.40, 1.5, 0.6, 0.40, '0.74 0.59 0.40 1'),
    ('table', 1.0, 1.45, 0.8, 0.45, 0.30, '0.85 0.74 0.57 1'),
    ('shelf', 2.6, 0.20, 0.9, 0.3, 0.62, '0.70 0.55 0.36 1'),
    ('plant', 3.55, 0.45, 0.17, None, 0.32, '0.85 0.54 0.35 1'),
    ('chair', 3.0, 1.90, 0.42, 0.42, 0.42, '0.79 0.64 0.45 1'),
    ('carton', 1.9, 2.30, 0.36, 0.30, 0.25, '0.79 0.64 0.45 1'),
]
BASKET = dict(x=3.45, y=2.55, sx=0.30, sy=0.22, h=0.09, t=0.008)
HOME = (0.45, 2.55, -math.pi / 2)
# Charging dock (built only when asked for: build_xml(dock=True)). Wall-mounted: a back plate on the +y wall with one wide contact strip
# at the height of the robot's rear "nose" (a charging probe on the rear of the chassis, above the rear wheel). The robot reverses
# into it; any lateral position within the strip's width (+-HALF_W) makes contact, so the alignment needed is a few cm, not a few mm.
DOCK = dict(x=0.45, wall_y=3.0, plate_half_w=0.12, plate_t=0.03, plate_h=0.30, pad_t=0.008, pad_z=0.122, pad_half_h=0.012, nose_len=0.175, nose_r=0.01, nose_z=0.122)
DOCK['pad_y'] = DOCK['wall_y'] - DOCK['plate_t'] - DOCK['pad_t']          # y of the strip's front face
DOCK_POSE = (DOCK['x'], DOCK['pad_y'] - DOCK['nose_len'], -math.pi / 2)    # chassis pose with the nose touching the strip

def _quat_z(a):
    return f'{math.cos(a / 2):.6f} 0 0 {math.sin(a / 2):.6f}'


def _wheel_xml(i, ang_deg):
    R, Rb, b, N = P['wheel_radius'], P['base_radius'], P['roller_b'], P['rollers_per_row']
    a = math.sqrt(R * b)  # ellipsoid half-length that makes the roller envelope a circle of radius R
    c = R - b
    al = math.radians(ang_deg)
    s = [f'<body name="wheel{i}" pos="{Rb * math.cos(al):.5f} {Rb * math.sin(al):.5f} 0" quat="{_quat_z(al)}">',
         f'<joint name="wheel{i}" class="sts3215" type="hinge" axis="1 0 0"/>',
         f'<geom class="rvis" type="cylinder" fromto="-0.012 0 0 0.012 0 0" size="0.034" rgba="0.15 0.16 0.17 1" mass="{P["hub_mass"]}"/>',
         '<geom class="rvis" type="box" size="0.013 0.03 0.005" rgba="1 0.42 0.17 1" mass="0"/>']
    for row in range(2):
        xo = (-1 if row == 0 else 1) * P['roller_row_offset']
        for j in range(N):
            ph = 2 * math.pi * j / N + row * math.pi / N
            s.append(f'<body name="w{i}r{row}_{j}" pos="{xo:.5f} {c * math.cos(ph):.5f} {c * math.sin(ph):.5f}" '
                     f'quat="{math.cos(ph / 2):.6f} {math.sin(ph / 2):.6f} 0 0">'
                     f'<joint class="roller" type="hinge" axis="0 0 1"/>'
                     f'<geom class="roller" type="ellipsoid" size="{b} {b} {a:.5f}" mass="{P["roller_mass"]}"/></body>')
    s.append('</body>')
    return '\n'.join(s)


def _arm_parts():
    """Read the official SO-101 MJCF and return (defaults, assets, body, actuators) as XML strings,
    with the gripper collision replaced by primitive pads (a convex hull of the jaw mesh fills the grip gap)."""
    root = ET.parse(os.path.join(SO101_DIR, 'so101_new_calib.xml')).getroot()
    defaults = ''.join(ET.tostring(d, encoding='unicode') for dd in root.findall('default') for d in dd)
    assets = ''.join(ET.tostring(a, encoding='unicode') for a in root.find('asset'))
    base = root.find('worldbody').find('body')
    base.set('pos', ' '.join(str(v) for v in P['arm_mount']))
    for body in base.iter('body'):
        for g in list(body.findall('geom')):
            if g.get('class') == 'collision':
                if body.get('name') in ('gripper', 'moving_jaw_so101_v1'):
                    body.remove(g)          # replaced by pads below
                else:
                    g.set('contype', '2'); g.set('conaffinity', '1')
    grip = next(b for b in base.iter('body') if b.get('name') == 'gripper')
    jaw = next(b for b in base.iter('body') if b.get('name') == 'moving_jaw_so101_v1')
    pad = f'contype="2" conaffinity="1" group="3" friction="{P["pad_friction"]} 0.02 0.0002" condim="4" priority="1" solref="0.006 1" rgba="0.1 0.1 0.1 1"'
    # fixed jaw: inner face at x = -0.009, finger spans z = -0.045 .. -0.104 in the gripper frame
    ET.SubElement(grip, 'geom', dict(name='pad_fixed', type='box', pos='-0.0135 0 -0.0745', size='0.0045 0.007 0.0295')).attrib.update(_attrs(pad))
    ET.SubElement(grip, 'geom', dict(name='palm', type='box', pos='-0.002 0 -0.02', size='0.031 0.024 0.02')).attrib.update(_attrs(pad))
    ET.SubElement(grip, 'site', dict(name='pinch', pos='-0.009 0 -0.086', size='0.003', rgba='1 0 0 1', group='4'))
    ET.SubElement(grip, 'camera', dict(name='wrist', pos='0.015 -0.08 -0.03', xyaxes='1 0 0 0 0.566 0.824', fovy='70'))
    # moving jaw pad, defined at q=0 in the gripper frame as box centre (0.0125, 0, -0.0745), then mapped into the jaw frame
    jp = np.array([float(v) for v in jaw.get('pos').split()]); jq = np.array([float(v) for v in jaw.get('quat').split()])
    Rj = np.zeros(9); mujoco.mju_quat2Mat(Rj, jq); Rj = Rj.reshape(3, 3)
    c = Rj.T @ (np.array([0.0125, 0, -0.0745]) - jp)
    qinv = np.zeros(4); mujoco.mju_negQuat(qinv, jq)
    ET.SubElement(jaw, 'geom', dict(name='pad_moving', type='box', pos=' '.join(f'{v:.5f}' for v in c), quat=' '.join(f'{v:.6f}' for v in qinv),
                                    size='0.0045 0.007 0.0295')).attrib.update(_attrs(pad))
    acts = []
    for a in root.find('actuator'):
        lim = P['gripper_torque'] if a.get('name') == 'gripper' else P['servo_stall']
        a.set('forcerange', f'{-lim} {lim}')
        acts.append(ET.tostring(a, encoding='unicode'))
    return defaults, assets, ET.tostring(base, encoding='unicode'), ''.join(acts)


def _attrs(s):
    out = {}
    for tok in s.replace('" ', '"|').split('|'):
        k, v = tok.split('=', 1); out[k.strip()] = v.strip().strip('"')
    return out


def build_xml(objects, robot_pose=HOME, extra_boxes=(), basket=BASKET, dock=False):
    """objects: list of (kind, x, y, yaw, spec) with spec from objects.sample(). Returns MJCF text. World frame: x right, y 'down' the room plan, z up."""
    defaults, assets, arm_body, arm_acts = _arm_parts()
    R = P['wheel_radius']
    lx, ly, lz = P['lidar_pos']
    furn = []
    for name, cx, cy, sx, sy, h, rgba in FURNITURE:
        if sy is None:
            furn.append(f'<geom name="{name}" type="cylinder" pos="{cx} {cy} {h / 2}" size="{sx} {h / 2}" rgba="{rgba}"/>')
        else:
            furn.append(f'<geom name="{name}" type="box" pos="{cx} {cy} {h / 2}" size="{sx / 2} {sy / 2} {h / 2}" rgba="{rgba}"/>')
    for k, (cx, cy, sx, sy, h) in enumerate(extra_boxes):
        furn.append(f'<geom name="xbox{k}" type="box" pos="{cx} {cy} {h / 2}" size="{sx / 2} {sy / 2} {h / 2}" rgba="0.79 0.64 0.45 1"/>')
    wt, wh = 0.03, 0.4
    walls = [(ROOM_W / 2, -wt / 2, ROOM_W / 2 + wt, wt / 2), (ROOM_W / 2, ROOM_H + wt / 2, ROOM_W / 2 + wt, wt / 2),
             (-wt / 2, ROOM_H / 2, wt / 2, ROOM_H / 2), (ROOM_W + wt / 2, ROOM_H / 2, wt / 2, ROOM_H / 2)]
    for k, (cx, cy, hx, hy) in enumerate(walls):
        furn.append(f'<geom name="wall{k}" type="box" pos="{cx} {cy} {wh / 2}" size="{hx} {hy} {wh / 2}" rgba="0.92 0.90 0.84 1"/>')
    b = basket; t = b['t']; hh = b['h'] / 2
    bk = [f'<geom name="basket_floor" type="box" pos="{b["x"]} {b["y"]} 0.004" size="{b["sx"] / 2} {b["sy"] / 2} 0.004" rgba="0.95 0.93 0.88 1"/>']
    for k, (dx, dy, hx, hy) in enumerate([(0, -(b['sy'] / 2 - t / 2), b['sx'] / 2, t / 2), (0, b['sy'] / 2 - t / 2, b['sx'] / 2, t / 2),
                                          (-(b['sx'] / 2 - t / 2), 0, t / 2, b['sy'] / 2), (b['sx'] / 2 - t / 2, 0, t / 2, b['sy'] / 2)]):
        bk.append(f'<geom name="basket_w{k}" type="box" pos="{b["x"] + dx} {b["y"] + dy} {hh}" size="{hx} {hy} {hh}" rgba="1 0.42 0.17 1"/>')
    nose = ''
    if dock:
        D = DOCK
        furn.append(f'<geom name="dock_plate" type="box" pos="{D["x"]} {D["wall_y"] - D["plate_t"] / 2} {D["plate_h"] / 2}" size="{D["plate_half_w"]} {D["plate_t"] / 2} {D["plate_h"] / 2}" rgba="0.25 0.28 0.32 1"/>')
        furn.append(f'<geom name="dock_pad" type="box" pos="{D["x"]} {D["wall_y"] - D["plate_t"] - D["pad_t"] / 2} {D["pad_z"]}" size="{D["plate_half_w"]} {D["pad_t"] / 2} {D["pad_half_h"]}" rgba="0.85 0.65 0.2 1" friction="0.3 0.005 0.0001"/>')
        zc = D['nose_z'] - (P['wheel_radius'] + 0.0005)          # nose height in the chassis frame
        nose = (f'<geom name="dock_nose" type="capsule" fromto="-0.10 0 {zc} {-(D["nose_len"] - D["nose_r"])} 0 {zc}" size="{D["nose_r"]}" contype="2" conaffinity="1" group="4" friction="0.3 0.005 0.0001" mass="0"/>'
                f'<geom class="rvis" type="capsule" fromto="-0.10 0 {zc} {-(D["nose_len"] - D["nose_r"])} 0 {zc}" size="{D["nose_r"]}" rgba="0.85 0.65 0.2 1" mass="0"/>')
    parts = [objects_lib.xml(f'obj{i}', spec, x, y, yaw) for i, (k, x, y, yaw, spec) in enumerate(objects)]
    objs = '\n'.join(b for _, b in parts); obj_assets = ''.join(a for a, _ in parts)
    wheels = '\n'.join(_wheel_xml(i, a) for i, a in enumerate(P['wheel_angles']))
    wheel_acts = '\n'.join(f'<velocity name="wheel{i}" joint="wheel{i}" kv="3" ctrlrange="-7 7" forcerange="{-P["servo_stall"]} {P["servo_stall"]}"/>' for i in range(3))
    rx, ry, rth = robot_pose
    cm = P['chassis_com']
    return f'''<mujoco model="xiaoshi">
  <compiler angle="radian" meshdir="{os.path.join(SO101_DIR, 'assets')}" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="4"/>
  <size memory="64M"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/><headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0 0 0"/><map znear="0.002" zfar="2"/></visual>
  <default>
    <geom contype="1" conaffinity="1" friction="0.6 0.005 0.0001"/>
    <default class="rvis"><geom contype="0" conaffinity="0" group="2"/></default>
    <default class="rcol"><geom contype="2" conaffinity="1" group="3" rgba="0.5 0.5 0.5 0.3"/></default>
    <default class="roller">
      <geom contype="2" conaffinity="1" group="1" condim="3" friction="{P['floor_friction']} 0.005 0.0001" priority="1" rgba="0.08 0.08 0.09 1" solref="0.004 1"/>
      <joint damping="0.00002" frictionloss="0.0003" armature="0.000003"/>
    </default>
    {defaults}
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.06 0.16 0.14" rgb2="0.03 0.08 0.07" width="64" height="384"/>
    <texture name="mat" type="2d" builtin="checker" rgb1="0.122 0.314 0.271" rgb2="0.135 0.335 0.29" mark="edge" markrgb="0.3 0.52 0.46" width="256" height="256"/>
    <material name="mat" texture="mat" texuniform="true" texrepeat="2 2" reflectance="0"/>
    {assets}
    {obj_assets}
  </asset>
  <worldbody>
    <light pos="1 1 4" dir="0.15 0.1 -1" directional="true" diffuse="0.55 0.55 0.52" castshadow="true"/>
    <geom name="floor" type="plane" pos="{ROOM_W / 2} {ROOM_H / 2} 0" size="{ROOM_W / 2 + 0.6} {ROOM_H / 2 + 0.6} 0.05" material="mat" friction="{P['floor_friction']} 0.005 0.0001"/>
    {''.join(furn)}
    {''.join(bk)}
    <camera name="overview" pos="2 5.6 4.2" xyaxes="-1 0 0 0 -0.6 0.8" fovy="42"/>
    <body name="chassis" pos="{rx} {ry} {R + 0.0005}" quat="{_quat_z(rth)}">
      <freejoint name="base"/>
      <inertial pos="{cm[0]} {cm[1]} {cm[2]}" mass="{P['chassis_mass'] * perturb.get('chassis_mass', 1.0)}" diaginertia="0.0040 0.0040 0.0066"/>
      <geom class="rcol" name="chassis_col" type="cylinder" pos="0 0 0.056" size="0.105 0.03"/>
      <geom class="rvis" type="cylinder" pos="0 0 0.029" size="0.105 0.003" rgba="0.95 0.93 0.88 1"/>
      <geom class="rvis" type="cylinder" pos="0 0 0.0832" size="0.105 0.003" rgba="0.95 0.93 0.88 1"/>
      <geom class="rvis" type="box" pos="-0.045 0 0.052" size="0.04 0.03 0.018" rgba="0.12 0.13 0.15 1"/>
      <geom class="rvis" type="box" pos="-0.06 0 0.1" size="0.045 0.032 0.012" rgba="0.2 0.5 0.3 1"/>
      <geom class="rvis" type="cylinder" pos="{lx} {ly} {lz - 0.004}" size="0.019 0.017" rgba="0.1 0.1 0.1 1"/>
      <geom class="rvis" type="cylinder" pos="{lx} {ly} {lz - 0.036}" size="0.004 0.016" rgba="0.7 0.7 0.7 1"/>
      <geom name="lidar_head" type="cylinder" pos="{lx} {ly} {lz - 0.004}" size="0.0195 0.017" contype="2" conaffinity="1" group="4" friction="0.5 0.005 0.0001" mass="0"/>
      <geom name="lidar_post" type="cylinder" pos="{lx} {ly} {lz - 0.0385}" size="0.008 0.0215" contype="2" conaffinity="1" group="4" friction="0.5 0.005 0.0001" mass="0"/>
      <geom class="rvis" type="box" pos="0.1 0 0.07" size="0.006 0.02 0.012" rgba="0.1 0.1 0.1 1"/>
      {nose}
      <site name="lidar" pos="{lx} {ly} {lz}" size="0.004" group="4"/>
      <site name="base_center" pos="0 0 0" size="0.004" group="4"/>
      <camera name="front" pos="0.108 0 0.075" xyaxes="0 -1 0 0.42 0 0.91" fovy="72"/>
      <camera name="chase" mode="trackcom" pos="-0.9 -0.9 0.9" xyaxes="0.707 -0.707 0 0.42 0.42 0.8" fovy="50"/>
      {wheels}
      {arm_body}
    </body>
    {objs}
  </worldbody>
  <actuator>
    {wheel_acts}
    {arm_acts}
  </actuator>
</mujoco>'''


def build(objects, **kw):
    xml = build_xml(objects, **kw)
    return mujoco.MjModel.from_xml_string(xml), xml


if __name__ == '__main__':
    m, xml = build([('block', 1.5, 2.2, 0.3)])
    d = mujoco.MjData(m)
    print('nq', m.nq, 'nv', m.nv, 'nu', m.nu, 'nbody', m.nbody, 'ngeom', m.ngeom, 'robot mass %.3f kg' % m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'chassis')])
