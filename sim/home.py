"""Everything the robot knows about the home, in one place, and the hooks that keep it up to date.

  map      the 3D voxel map and how it was made (mapping3d.py); the SCAN mode drives it
  (the object records, the people and the voice join here: inventory.py, people.py, voice.py)
Attach with Home(robot, agent) and sup.home = home. Nothing here runs unless it is attached; a robot without a Home behaves exactly as before.
"""
import numpy as np

import mapping3d as M
from inventory import Inventory


class Home:
    def __init__(self, robot, agent, size=(4.0, 3.0, 1.0), colour=True, inventory=True, inv_path=None):
        self.r, self.a = robot, agent
        self.map = M.VoxelMap(size); self.scan = M.Scan3D(self.map, colour=colour); self.planner = M.ScanPlanner(self.map)
        self.mapping = False; self.scan_log = []; self.analysis = None
        self.inv = Inventory(inv_path) if inventory else None; self.recording = inventory
        if self.inv is not None: self.inv.start_session()
        agent.depth_hook = self._hook

    def _hook(self, agent, x, y, th):
        if self.mapping: self.scan.hook(agent, x, y, th)
        if self.inv is not None and self.recording: self.inv.observe(agent, x, y, th)

    def finish_scan(self):
        self.analysis = M.analyze(self.map)
        if self.inv is not None: self.inv.set_places(self.analysis['furniture'] + self._basket_box())
        return self.analysis

    def _basket_box(self):
        b = self.a.basket
        return [dict(id=100, centre=[b['x'], b['y']], length=b['sx'], width=b['sy'], yaw=0.0, height_seen=b['h'])]
