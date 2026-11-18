"""Library of things left on the floor: hard and soft, regular and irregular, with random sizes.

Every object is generated from a random draw, so no two runs see exactly the same one, and nothing in the robot's
perception or grasping is told what an object is or how big it is.

Soft objects are approximated the way robot simulators commonly do it: a few rigid pieces joined by springy hinges
(so the object bends and droops) with compliant contact (so its surface gives a few millimetres when squeezed).
That is not continuum deformation: a real sock or plush toy also changes cross-section when pinched.
"""
import math

import numpy as np

import perturb

TYPES = {
    'cube':   dict(cn='方块', soft=False, rgba='0.31 0.62 0.88 1'),
    'ball':   dict(cn='小球', soft=False, rgba='0.84 0.91 0.29 1'),
    'car':    dict(cn='玩具车', soft=False, rgba='0.95 0.69 0.18 1'),
    'banana': dict(cn='香蕉玩具', soft=False, rgba='0.98 0.86 0.30 1'),
    'lbrick': dict(cn='L 形积木', soft=False, rgba='0.36 0.75 0.45 1'),
    'rock':   dict(cn='石头', soft=False, rgba='0.55 0.53 0.50 1'),
    'sock':   dict(cn='袜子（软）', soft=True, rgba='0.89 0.28 0.24 1'),
    'plush':  dict(cn='毛绒玩具（软）', soft=True, rgba='0.80 0.55 0.72 1'),
}
KINDS = list(TYPES)
# compliant contact for soft things; priority 2 makes these parameters win over the gripper pads' stiff ones
# Soft surface, firmer core (stuffing packs down as it is squeezed): compliant for the first ~8 mm, then much stiffer.
# With a softer core a 10 g toy sank 16 mm into the finger pad at full grip and stayed wedged there after release.
SOFT = 'priority="2" condim="6" friction="0.8 0.02 0.02" solref="0.015 1" solimp="0.6 0.99 0.008"'
HINGE_SOFT = 'damping="0.0008" armature="0.000002"'


def sample(kind, rng):
    """Random size and density for one object of this kind. Returns a spec dict."""
    u = rng.uniform
    if kind == 'cube':
        p = dict(s=u(0.03, 0.05), density=u(400, 700))
        r, h = 0.71 * p['s'], p['s']
    elif kind == 'ball':
        p = dict(r=u(0.026, 0.034), density=u(250, 450))
        r, h = p['r'], 2 * p['r']
    elif kind == 'car':
        p = dict(L=u(0.08, 0.10), W=u(0.038, 0.046), wr=u(0.009, 0.012), density=u(250, 450))
        r, h = p['L'] / 2, 2 * p['wr'] + 0.026
    elif kind == 'banana':
        p = dict(R=u(0.07, 0.10), L=u(0.12, 0.16), rc=u(0.014, 0.018), density=u(250, 450))
        r, h = p['L'] / 2, 2 * p['rc']
    elif kind == 'lbrick':
        p = dict(la=u(0.05, 0.07), lb=u(0.04, 0.06), t=u(0.02, 0.028), h=u(0.018, 0.03), density=u(400, 700))
        r, h = 0.7 * max(p['la'], p['lb']), p['h']
    elif kind == 'rock':
        a, b, c = u(0.020, 0.030), u(0.016, 0.024), u(0.013, 0.020)
        d = rng.normal(size=(16, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        v = d * [a, b, c] * rng.uniform(0.8, 1.0, (16, 1))
        v[:, 2] -= v[:, 2].min()
        p = dict(v=v.round(4).tolist(), density=u(2000, 2700))
        r, h = max(a, b), float(v[:, 2].max())
    elif kind == 'sock':
        p = dict(ls=u(0.025, 0.035), rs=u(0.018, 0.023), density=u(150, 250))
        r, h = 1.5 * p['ls'] + p['rs'], 2 * p['rs']
    elif kind == 'plush':
        p = dict(rt=u(0.022, 0.028), lim=u(0.02, 0.03), rl=u(0.007, 0.009), density=u(80, 150))
        r, h = p['rt'] + p['lim'], 2.3 * p['rt']
    else:
        raise ValueError(kind)
    p['density'] *= perturb.get('obj_mass', 1.0)
    return dict(kind=kind, p=p, radius=float(r), height=float(h))


def xml(name, spec, x, y, yaw):
    """(asset xml, body xml) for one object, its lowest point resting on the floor at (x, y)."""
    k, p = spec['kind'], spec['p']; rgba = TYPES[k]['rgba']
    hard = 'condim="4" friction="0.5 0.01 0.0001"'
    dens = f'density="{p["density"]:.0f}"'
    q = f'{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}'
    asset = ''
    g = lambda i, rest: f'<geom name="{name}_g{i}" rgba="{rgba}" {dens} {rest}/>'
    if k == 'cube':
        s = p['s'] / 2; inner = g(0, f'type="box" pos="0 0 {s:.4f}" size="{s:.4f} {s:.4f} {s:.4f}" {hard}')
    elif k == 'ball':
        inner = g(0, f'type="sphere" pos="0 0 {p["r"]:.4f}" size="{p["r"]:.4f}" condim="6" friction="0.6 0.005 0.0015"')
    elif k == 'car':
        L, W, wr = p['L'], p['W'], p['wr']; zc = 2 * wr + 0.004
        inner = (g(0, f'type="box" pos="0 0 {zc:.4f}" size="{L / 2:.4f} {W / 2:.4f} 0.007" {hard}')
                 + g(1, f'type="box" pos="{-0.05 * L:.4f} 0 {zc + 0.013:.4f}" size="{0.22 * L:.4f} {0.42 * W:.4f} 0.006" {hard}'))
        for j, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            inner += (f'<geom name="{name}_w{j}" type="cylinder" pos="{sx * 0.32 * L:.4f} {sy * (W / 2 - 0.004):.4f} {wr:.4f}" '
                      f'quat="0.7071 0.7071 0 0" size="{wr:.4f} 0.004" rgba="0.15 0.15 0.15 1" {dens} condim="4" friction="0.8 0.01 0.0001"/>')
    elif k == 'banana':
        R, L, rc = p['R'], p['L'], p['rc']; A = L / R
        ang = np.linspace(-A / 2, A / 2, 5); pts = np.stack([R * np.sin(ang), R * (1 - np.cos(ang))], 1)
        pts -= (pts.min(0) + pts.max(0)) / 2
        inner = ''
        for j in range(4):
            rr = rc * (0.75 if j in (0, 3) else 1.0)
            (x0, y0), (x1, y1) = pts[j], pts[j + 1]
            inner += g(j, f'type="capsule" fromto="{x0:.4f} {y0:.4f} {rc:.4f} {x1:.4f} {y1:.4f} {rc:.4f}" size="{rr:.4f}" condim="4" friction="0.6 0.01 0.0001"')
    elif k == 'lbrick':
        la, lb, t, h = p['la'], p['lb'], p['t'], p['h']
        cx, cy = la / 2, lb / 2
        inner = (g(0, f'type="box" pos="{la / 2 - cx:.4f} {t / 2 - cy:.4f} {h / 2:.4f}" size="{la / 2:.4f} {t / 2:.4f} {h / 2:.4f}" {hard}')
                 + g(1, f'type="box" pos="{t / 2 - cx:.4f} {(lb + t) / 2 - cy:.4f} {h / 2:.4f}" size="{t / 2:.4f} {(lb - t) / 2:.4f} {h / 2:.4f}" {hard}'))
    elif k == 'rock':
        asset = f'<mesh name="{name}_mesh" vertex="{" ".join(f"{c:.4f}" for c in np.ravel(p["v"]))}"/>'
        inner = g(0, f'type="mesh" mesh="{name}_mesh" condim="4" friction="0.6 0.01 0.0001"')
    elif k == 'sock':
        ls, rs = p['ls'], p['rs']
        inner = g(0, f'type="capsule" fromto="{-ls / 2:.4f} 0 {rs:.4f} {ls / 2:.4f} 0 {rs:.4f}" size="{rs:.4f}" {SOFT}')
        for j, sg in enumerate((1, -1)):
            inner += (f'<body name="{name}_b{j}" pos="{sg * ls / 2:.4f} 0 {rs:.4f}">'
                      f'<joint type="hinge" axis="0 1 0" stiffness="0.012" range="-0.9 0.9" {HINGE_SOFT}/>'
                      f'<joint type="hinge" axis="0 0 1" stiffness="0.012" range="-0.9 0.9" {HINGE_SOFT}/>'
                      + g(10 + j, f'type="capsule" fromto="0 0 0 {sg * ls:.4f} 0 0" size="{0.9 * rs:.4f}" {SOFT}') + '</body>')
    elif k == 'plush':
        rt, lim, rl = p['rt'], p['lim'], p['rl']; rh = 0.72 * rt
        inner = g(0, f'type="sphere" pos="0 0 {rt:.4f}" size="{rt:.4f}" {SOFT}')
        inner += (f'<body name="{name}_head" pos="{rt + 0.7 * rh:.4f} 0 {1.15 * rt:.4f}">'
                  f'<joint type="hinge" axis="0 1 0" stiffness="0.006" range="-0.6 0.6" {HINGE_SOFT}/>'
                  f'<joint type="hinge" axis="0 0 1" stiffness="0.006" range="-0.6 0.6" {HINGE_SOFT}/>'
                  + g(1, f'type="sphere" size="{rh:.4f}" {SOFT}') + '</body>')
        for j, (sx, sy) in enumerate(((0.5, 1), (0.5, -1), (-0.6, 1), (-0.6, -1))):
            ax, ay, az = sx * 0.7 * rt, sy * 0.7 * rt, 0.6 * rt
            n = math.hypot(sx, sy); dx, dy = sx / n * lim, sy / n * lim
            inner += (f'<body name="{name}_l{j}" pos="{ax:.4f} {ay:.4f} {az:.4f}">'
                      f'<joint type="hinge" axis="{-dy / lim:.3f} {dx / lim:.3f} 0" stiffness="0.002" range="-0.8 0.8" {HINGE_SOFT}/>'
                      f'<joint type="hinge" axis="0 0 1" stiffness="0.002" range="-0.8 0.8" {HINGE_SOFT}/>'
                      + g(2 + j, f'type="capsule" fromto="0 0 0 {dx:.4f} {dy:.4f} {rl - az:.4f}" size="{rl:.4f}" {SOFT}') + '</body>')
    body = f'<body name="{name}" pos="{x:.4f} {y:.4f} 0.003" quat="{q}"><freejoint name="{name}"/>{inner}</body>'
    return asset, body
