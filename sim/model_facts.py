"""Writes results/model_facts.json: masses and lidar self-occlusion, read off the model itself."""
import json

import mujoco

from robot import LIDAR_RAYS, Robot

if __name__ == '__main__':
    r = Robot([])
    for _ in range(25): r.step()
    r.lidar()
    bid = lambda n: mujoco.mj_name2id(r.m, mujoco.mjtObj.mjOBJ_BODY, n)
    facts = dict(robot_mass=float(r.m.body_subtreemass[r.chassis]), arm_mass=float(r.m.body_subtreemass[bid('base')]),
                 wheel_mass=float(r.m.body_subtreemass[bid('wheel0')]), lidar_self_hits=int(r.last_self_hits), lidar_rays=LIDAR_RAYS)
    json.dump(facts, open('../results/model_facts.json', 'w'), indent=1); print(facts)
