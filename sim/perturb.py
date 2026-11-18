"""Deliberate errors for sensitivity tests: each key makes one part of the simulated world worse than the nominal
model, so the trials show which real-world error the task is most sensitive to. Empty (the default) changes nothing.

  pad_friction   - set through run_episode(pad_friction=...)
  obj_mass       - multiplies every object's density
  arm_offset_deg - 1-sigma zero-point error of each arm servo: the joint sits that far from where the robot thinks
  cam_err        - (m, deg): 1-sigma error of the front camera's mounting position and angle as the robot believes it
  chassis_mass   - multiplies the chassis mass
  odom           - multiplies wheel-radius error and slip noise
  lidar_noise    - multiplies lidar range noise
  depth_noise    - multiplies depth-camera noise
"""
CUR = {}


def get(key, default):
    return CUR.get(key, default)
