"""Where on the camera body does the ground press it when the robot has fallen on its side or face and is righting itself? (camera box local frame: x along the 90 mm length, y the 25 mm height,
z the 25 mm depth behind the lens). Runs the first stored maneuver of each class with the camera box touching the floor (and, with ARM_CAM_GUARD=0, without the bumper) and prints the contacts.

  MUJOCO_GL=egl python cam_contact_probe.py
"""
import os, sys, json, math
os.environ['ARM_CAM'] = 'forearm_up'; os.environ['ARM_CAM_COLLIDE'] = '1'; os.environ.setdefault('ARM_CAM_GUARD', '0')           # without the bumper by default: where it would be hit
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, mujoco
import search_righting as S
lib = json.load(open(os.path.join(HERE, 'righting_maneuvers_armcam.json')))
print({k: (type(v).__name__, (len(v) if hasattr(v, '__len__') else v)) for k, v in lib['maneuvers'].items()})
for cls in ('side_l', 'side_r', 'fwd'):
    env = S.Env(cls); r = env.r; m = r.m
    ph = lib['maneuvers'][cls][0]['phases']
    best = None
    # same loop as Env.run_phases, but looking at where the camera is touched
    for trial in range(1):
        rng = np.random.default_rng(trial); env.drop(rng, jitter=3.0)
        pts = []; peak = 0.0; f6 = np.zeros(6)
        r.d.qpos[:] = env.qpos0; r.d.qvel[:] = 0
        r.arm_cmd[:] = env.arm0 + 0; r.arm_target[:] = r.arm_cmd; r.arm_vel[:] = 0; r.wheel_cmd[:] = 0; r.wheel_target[:] = 0
        r.d.qpos[r.arm_qadr] = r.arm_cmd + r.arm_offset; r.d.ctrl[3:9] = r.arm_cmd + r.arm_offset; r.arm_acc = env.arm_acc; mujoco.mj_forward(m, r.d)
        for p in [dict(q=np.array(x['q']), w=np.array(x['w']), T=x['T'], speed=x['speed']) for x in ph]:
            r.arm_target[:4] = p['q']; r.arm_target[4] = 0.0; r.arm_target[5] = S.GRIP_CLOSED; r.arm_speed = p['speed']; r.wheel_target[:] = p['w']
            for _ in range(int(p['T'] / 0.02)):
                r.step(); d = r.d
                for i in range(d.ncon):
                    c = d.contact[i]
                    if env.cam_geom in (c.geom1, c.geom2):
                        mujoco.mj_contactForce(m, d, i, f6); other = c.geom2 if c.geom1 == env.cam_geom else c.geom1
                        R = d.geom_xmat[env.cam_geom].reshape(3, 3); loc = R.T @ (np.array(c.pos) - d.geom_xpos[env.cam_geom])
                        pts.append((float(f6[0]), np.round(loc * 1000, 1).tolist(), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or f'g{other}'))
        print(cls, 'contacts', len(pts), 'peak', round(max([p_[0] for p_ in pts], default=0), 1), 'N')
        for f, loc, other in sorted(pts, reverse=True)[:6]: print('   %.1f N at local (mm) %s with %s' % (f, loc, other))
        if pts:
            L = np.array([p_[1] for p_ in pts]); print('   local x range %.0f..%.0f, y range %.0f..%.0f, z range %.0f..%.0f' % (L[:, 0].min(), L[:, 0].max(), L[:, 1].min(), L[:, 1].max(), L[:, 2].min(), L[:, 2].max()))
