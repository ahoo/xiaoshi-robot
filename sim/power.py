"""Battery, power draw and charger of the robot. Every number here is an estimate from datasheets or typical hardware; each one has a row in
docs/validation.md to measure on the real pack before the low-battery thresholds are trusted.

What the control software gets is what a battery-management board gives: a state of charge (coulomb counting, so it drifts: a
fixed error per run plus a little noise) and a voltage. The true state stays inside the simulator.

Power draw = a constant for the compute and sensors (Raspberry Pi 5, depth cameras, lidar, DC-DC loss) + what the nine servos draw (computed
every physics step in robot.py from the actuator torques and speeds: idle current, copper loss, mechanical output / efficiency).
Charging is constant current up to CC_UNTIL, then a tapering constant-voltage phase; the robot keeps drawing its (reduced) docked load meanwhile.
"""
import math

import numpy as np

CAPACITY_WH = 50.0          # LeKiwi kit: 12 V (3S Li-ion, 11.1 V nominal) 4500 mAh
P_RUN_W = 10.0              # compute + sensors while working (Pi 5 busy 7 W, camera 1.5 W, lidar 0.7 W, 10 % conversion loss)
P_DOCKED_W = 7.0            # the same, docked and waiting: lidar motor off, camera idle, arm torque off (servo idle is added on top)
CHARGER_W = 37.8            # dock supply: 12.6 V at 3 A, passed through the robot's charge controller
CC_UNTIL = 0.80             # state of charge where the constant-current phase ends
CV_TAPER = 0.07             # the charging power falls by 1/e every this much state of charge in the CV phase
V_EMPTY, V_FULL = 9.6, 12.6  # open-circuit voltage at 0 % and 100 % (3S Li-ion; the real curve is flatter in the middle)
R_INT = 0.12                # ohm: pack plus wiring, voltage sag under load
SOC_CUTOFF = 0.03           # the battery board cuts the output below this: the robot is off


class Battery:
    """True state of charge `soc` (0..1) plus what the battery board reports (`soc_est`, `voltage`)."""
    def __init__(self, soc=1.0, rng=None, est_bias=None):
        self.soc = float(soc); self.rng = rng if rng is not None else np.random.default_rng(0)
        self.est_bias = float(self.rng.normal(0, 0.03)) if est_bias is None else float(est_bias)   # coulomb-counter drift: 1 sigma 3 %
        self.drawn_wh = 0.0; self.charged_wh = 0.0; self.t_charging = 0.0; self.load_w = P_RUN_W
        self.dead = False

    def step(self, dt, load_w, charge_w=0.0):
        """Advance `dt` seconds with the robot drawing `load_w` and the dock supplying `charge_w` into the pack (after the robot's own load)."""
        self.load_w = load_w
        e_out = load_w * dt / 3600.0; e_in = charge_w * dt / 3600.0
        self.drawn_wh += e_out; self.charged_wh += e_in
        self.soc = min(1.0, max(0.0, self.soc + (e_in - e_out) / CAPACITY_WH))
        if charge_w > 0: self.t_charging += dt
        if self.soc <= SOC_CUTOFF: self.dead = True

    @property
    def soc_est(self):
        return min(1.0, max(0.0, self.soc + self.est_bias))

    def voltage(self, load_w=None):
        w = self.load_w if load_w is None else load_w
        ocv = V_EMPTY + (V_FULL - V_EMPTY) * self.soc
        return ocv - R_INT * w / max(ocv, 1.0)

    def charge_power(self, supplied_w=CHARGER_W, docked_load_w=P_DOCKED_W):
        """Power going into the pack at the present state of charge (W)."""
        p = supplied_w - docked_load_w
        if self.soc > CC_UNTIL: p *= math.exp(-(self.soc - CC_UNTIL) / CV_TAPER)
        return max(p, 0.0)

    def charge_until(self, target, docked_load_w=P_DOCKED_W, servo_idle_w=3.15, step_s=5.0, max_s=8 * 3600.0):
        """Fast-forward charging on the dock: integrate the CC/CV profile until the pack reaches `target` (true state of charge).
        Returns the real time it took, in seconds (the simulator does not step the physics for it)."""
        t = 0.0
        while self.soc < target and t < max_s:
            p_in = self.charge_power(docked_load_w=docked_load_w + servo_idle_w)      # what is left of the dock's supply after the robot's own load
            if p_in < 0.5: break                     # taper has reached nothing: done as far as it goes
            self.soc = min(1.0, self.soc + p_in * step_s / 3600.0 / CAPACITY_WH); self.charged_wh += p_in * step_s / 3600.0
            self.load_w = docked_load_w + servo_idle_w; t += step_s
        self.t_charging += t
        return t
