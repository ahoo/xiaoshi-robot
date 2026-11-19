"""Low-level robot interface on top of MuJoCo: what the real robot's driver layer would provide.

Commands: body twist for the base, joint targets for the arm (servo position mode).
Sensors: wheel encoders, joint positions and loads, 2D lidar ranges.
Ground truth (pose of robot and objects) is exposed separately and only used by the sensor models and the scorer.
"""
import math

import mujoco
import numpy as np

import perturb

from model import P, build
from arm import ALL_JOINTS

WHEEL_ANG = [math.radians(a) for a in P['wheel_angles']]
MAX_WHEEL_SPEED = 3.6          # rad/s commanded, 75 % of the servo's no-load speed
CTRL_DT = 0.02                 # 50 Hz control
WHEEL_ACC = 8.0                # rad/s^2 wheel acceleration limit (0.4 m/s^2 at the rim)
LIDAR_RAYS = 240
LIDAR_RANGE = 6.0
LIDAR_MIN = 0.12               # returns closer than this to the sensor are the robot itself
LIDAR_NOISE = 0.01             # metres, 1 sigma
ODOM_SCALE = 0.015             # 1 sigma: effective radius error of each omni wheel (calibration, roller wear)
ODOM_SLIP = 0.03               # 1 sigma: per-step multiplicative slip noise on each wheel's travel
FOOTPRINT = (-0.05, 0.27, 0.10)  # lidar self-filter box in the chassis frame (folded arm plus payload): x min, x max, |y| max
IMU_NOISE = 0.004              # rad, 1 sigma per axis, of the fused attitude (gravity direction) the IMU driver reports
IMU_BIAS = 0.008               # rad, 1 sigma of the fixed mounting/calibration bias of that direction (0.5 degrees)
# Electrical model of the servos (estimates from the STS3215 datasheet: 12 V, 2.9 A stall at 2.94 N*m; see power.py and validation.md):
SERVO_IDLE_W = 0.35            # W per servo holding position with no load
SERVO_K_CU = 3.7               # W per (N*m)^2: copper loss, R/Kt^2 = (12/2.9)/(2.94/2.9)^2
SERVO_ETA = 0.55               # mechanical output / electrical input while doing positive work

Q_CARRY = np.array([0.002, -0.393, 0.733, 1.523, 0.002])   # pinch 0.16 m ahead, 0.19 m up; a held object clears the shoulder and the deck
GRIP_OPEN, GRIP_CLOSED = 1.1, -0.17


def twist_to_wheels(vx, vy, wz):
    """Kiwi-drive inverse kinematics. Body frame: x forward, y left. Returns wheel speeds (rad/s)."""
    w = np.array([-(-math.sin(a) * vx + math.cos(a) * vy + P['base_radius'] * wz) / P['wheel_radius'] for a in WHEEL_ANG])
    k = np.abs(w).max() / MAX_WHEEL_SPEED
    return w / k if k > 1 else w


def wheels_to_twist(w):
    """Forward kinematics (least squares), used for odometry from the wheel encoders."""
    A = np.array([[-(-math.sin(a)), -math.cos(a), -P['base_radius']] for a in WHEEL_ANG]) / P['wheel_radius']
    return np.linalg.solve(A, np.asarray(w))


class Robot:
    def __init__(self, objects, seed=0, **kw):
        self.m, self.xml = build(objects, **kw)
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        m = self.m
        nid = lambda t, n: mujoco.mj_name2id(m, t, n)
        self.chassis = nid(mujoco.mjtObj.mjOBJ_BODY, 'chassis')
        self.lidar_site = nid(mujoco.mjtObj.mjOBJ_SITE, 'lidar')
        self.pinch_site = nid(mujoco.mjtObj.mjOBJ_SITE, 'pinch')
        self.wheel_dof = [m.jnt_dofadr[nid(mujoco.mjtObj.mjOBJ_JOINT, f'wheel{i}')] for i in range(3)]
        self.wheel_qadr = [m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, f'wheel{i}')] for i in range(3)]
        self.arm_qadr = [m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ALL_JOINTS]
        self.arm_dof = [m.jnt_dofadr[nid(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ALL_JOINTS]
        self.arm_act = [nid(mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ALL_JOINTS]
        self.objects = objects
        self.obj_body = [nid(mujoco.mjtObj.mjOBJ_BODY, f'obj{i}') for i in range(len(objects))]
        self.arm_target = np.concatenate([Q_CARRY, [0.3]])
        self.arm_cmd = self.arm_target.copy(); self.arm_vel = np.zeros(6)
        self.wheel_target = np.zeros(3); self.wheel_cmd = np.zeros(3)
        self.arm_speed = 1.6                     # rad/s joint-space speed limit used when interpolating targets
        self.arm_acc = 8.0                       # rad/s^2 joint acceleration limit of the servo profile (the STS3215 register allows far more; self-righting raises it)
        # servo zero-point errors (perturb.py): each arm joint really sits this far from where it is commanded and read
        self.arm_offset = np.append(self.rng.normal(0, math.radians(perturb.get('arm_offset_deg', 0.0)), 5), 0.0) if perturb.get('arm_offset_deg', 0.0) else np.zeros(6)
        self.d.qpos[self.arm_qadr] = self.arm_cmd + self.arm_offset
        self.d.ctrl[3:9] = self.arm_cmd + self.arm_offset
        # lidar: rays in the sensor frame; geoms in groups 0 (world), 1 (rollers), 3 (robot collision) are visible to it
        ang = np.linspace(0, 2 * math.pi, LIDAR_RAYS, endpoint=False)
        self.ray_ang = ang
        self._ray_local = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1)
        self._geomgroup = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)
        self._ray_dist = np.zeros(LIDAR_RAYS); self._ray_geom = np.zeros(LIDAR_RAYS, dtype=np.int32)
        self.odom = np.zeros(3)                  # dead-reckoned pose from wheel encoders (starts at the true pose)
        self.wheel_scale = 1 + self.rng.normal(0, ODOM_SCALE * perturb.get('odom', 1.0), 3)   # each wheel's real radius differs from the nominal one
        self.peak_torque = np.zeros(9); self.torque_sq = np.zeros(9); self.n_steps = 0; self.sat_steps = np.zeros(9)
        self.hold_arm = False                    # set by people.Guard: the arm (and gripper) stays where it is while someone is within reach
        self._flim = self.m.actuator_forcerange[:9, 1].copy()
        self.servo_j = 0.0                       # electrical energy the nine servos have drawn so far (J)
        self.imu_rng = np.random.default_rng(seed + 4321)      # its own stream: reading the IMU never disturbs the lidar/odometry noise
        self.imu_bias = self.imu_rng.normal(0, IMU_BIAS, 3)
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.dock_nose, self.dock_pad, self.dock_plate = gid('dock_nose'), gid('dock_pad'), gid('dock_plate')      # -1 when the model has no dock
        self._dock_t = -9.0
        mujoco.mj_forward(m, self.d)
        x, y, th = self.true_pose(); self.odom[:] = [x, y, th]
        self._last_wheel_q = self.d.qpos[self.wheel_qadr].copy()

    # ---- ground truth ----
    def true_pose(self):
        q = self.d.qpos[3:7]
        return self.d.qpos[0], self.d.qpos[1], math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def tilt(self):
        return math.acos(max(-1.0, min(1.0, self.d.xmat[self.chassis][8])))

    # ---- IMU and charging contact ----
    def imu(self):
        """What the IMU driver reports: the direction of 'up' in the body frame (unit vector; z is cos(tilt)), from fused accelerometer
        and gyro, with noise and a fixed bias. At rest it is exact up to those; in a fall it follows the body."""
        up = self.d.xmat[self.chassis].reshape(3, 3)[2] + self.imu_bias + self.imu_rng.normal(0, IMU_NOISE, 3)
        return up / np.linalg.norm(up)

    def dock_connected(self, hold=0.3):
        """True while the charging probe on the rear of the chassis touches the dock's contact strip (the electrical contact; held for
        `hold` s so the flicker of a gentle touch does not count as losing it). False on a model without a dock."""
        if self.dock_pad < 0: return False
        d = self.d
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 == self.dock_nose and c.geom2 == self.dock_pad) or (c.geom2 == self.dock_nose and c.geom1 == self.dock_pad):
                if c.dist < 0.001: self._dock_t = d.time; break
        return d.time - self._dock_t <= hold

    def obj_pos(self, i):
        return self.d.xpos[self.obj_body[i]].copy()

    def obj_yaw(self, i):
        R = self.d.xmat[self.obj_body[i]].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    def pinch_pos(self):
        return self.d.site_xpos[self.pinch_site].copy()

    # ---- commands ----
    def set_twist(self, vx, vy, wz):
        self.wheel_target = twist_to_wheels(vx, vy, wz)

    def set_arm(self, q5=None, grip=None, speed=None):
        if q5 is not None: self.arm_target[:5] = q5
        if grip is not None: self.arm_target[5] = grip
        if speed is not None: self.arm_speed = speed

    def set_grip_torque(self, limit):
        """Torque limit of the gripper servo (N*m); the STS3215 has a register for this."""
        a = self.arm_act[5]; self.m.actuator_forcerange[a] = [-limit, limit]; self._flim[a] = limit

    def arm_done(self, tol=0.03):
        return np.abs(self.arm_cmd[:5] - self.arm_target[:5]).max() < 0.004 and np.abs(self.joints()[:5] - self.arm_target[:5]).max() < tol

    def joints(self):
        return self.d.qpos[self.arm_qadr] - self.arm_offset

    # ---- sensors ----
    def lidar(self):
        """One 360-degree scan. Returns (angles in the robot frame, ranges with noise; inf where nothing or the robot itself was hit)."""
        d = self.d
        R = d.xmat[self.chassis].reshape(3, 3)
        vec = (self._ray_local @ R.T).copy()
        mujoco.mj_multiRay(self.m, d, d.site_xpos[self.lidar_site], vec.reshape(-1), self._geomgroup, 1, -1, self._ray_geom, self._ray_dist, None, LIDAR_RAYS, LIDAR_RANGE)
        r = self._ray_dist.copy()
        r[(self._ray_geom < 0) | (r < LIDAR_MIN)] = np.inf
        # footprint filter, as a real lidar driver would be configured: returns that land inside a box around the robot
        # (arm reach and a carried object included) are discarded, because they are the robot itself or its payload
        px = P['lidar_pos'][0] + r * np.cos(self.ray_ang); py = r * np.sin(self.ray_ang)
        own = np.isfinite(r) & (px > FOOTPRINT[0]) & (px < FOOTPRINT[1]) & (np.abs(py) < FOOTPRINT[2])
        self.last_self_hits = int(own.sum())
        r[own] = np.inf
        self.last_ray_geom = np.where(np.isfinite(r), self._ray_geom, -1)
        self.last_raw = r.copy()
        model = perturb.get('lidar_model', None)
        if model is not None:
            if isinstance(model, str):
                import lidar_models
                model = perturb.CUR['lidar_model'] = lidar_models.MODELS[model]
            return self.ray_ang, model(r, self.last_ray_geom, self.rng)
        return self.ray_ang, r + self.rng.normal(0, LIDAR_NOISE * perturb.get('lidar_noise', 1.0), LIDAR_RAYS) * np.isfinite(r)

    def lidar_origin(self):
        x, y, th = self.true_pose()
        lx = P['lidar_pos'][0]
        return x + math.cos(th) * lx, y + math.sin(th) * lx

    # ---- stepping ----
    def step(self):
        """Advance one control period (20 ms)."""
        m, d = self.m, self.d
        # joint targets are approached with a velocity that ramps up (8 rad/s^2) and eases in, like the servo's own profile
        v_des = np.clip(5.0 * (self.arm_target - self.arm_cmd), -self.arm_speed, self.arm_speed)
        v_des[5] = np.clip(8.0 * (self.arm_target[5] - self.arm_cmd[5]), -3.0, 3.0)
        if self.hold_arm: v_des[:] = 0.0
        self.arm_vel += np.clip(v_des - self.arm_vel, -self.arm_acc * CTRL_DT, self.arm_acc * CTRL_DT)
        cmd0 = self.arm_cmd.copy()
        self.arm_cmd += self.arm_vel * CTRL_DT
        # wheel speed ramp (the servo's acceleration setting): without it a step command wheelies the chassis
        self.wheel_cmd += np.clip(self.wheel_target - self.wheel_cmd, -WHEEL_ACC * CTRL_DT, WHEEL_ACC * CTRL_DT)
        d.ctrl[:3] = self.wheel_cmd * (1 + 0.6 / 3)   # feed-forward for the servo's back-EMF damping
        n = int(round(CTRL_DT / m.opt.timestep))
        for k in range(n):
            d.ctrl[3:9] = cmd0 + (self.arm_cmd - cmd0) * (k + 1) / n + self.arm_offset     # the servo's own profile generator runs at kHz
            mujoco.mj_step(m, d)
            f = np.abs(d.actuator_force[:9]); self.peak_torque = np.maximum(self.peak_torque, f); self.torque_sq += f * f; self.n_steps += 1
            self.sat_steps += f > 0.8 * self._flim
            self.servo_j += (9 * SERVO_IDLE_W + SERVO_K_CU * float(f @ f) + float(np.maximum(d.actuator_force[:9] * d.actuator_velocity[:9], 0.0).sum()) / SERVO_ETA) * m.opt.timestep
        # odometry from wheel encoders
        wq = d.qpos[self.wheel_qadr]
        dw = (wq - self._last_wheel_q) * self.wheel_scale * (1 + self.rng.normal(0, ODOM_SLIP * perturb.get('odom', 1.0), 3)); self._last_wheel_q = wq.copy()
        tw = wheels_to_twist(dw / CTRL_DT)
        th = self.odom[2] + 0.5 * tw[2] * CTRL_DT
        self.odom[0] += (math.cos(th) * tw[0] - math.sin(th) * tw[1]) * CTRL_DT
        self.odom[1] += (math.sin(th) * tw[0] + math.cos(th) * tw[1]) * CTRL_DT
        self.odom[2] += tw[2] * CTRL_DT

    @property
    def t(self):
        return self.d.time
