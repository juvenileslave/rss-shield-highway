"""SIMPLIFIED copy of the protocol traffic window (section 7.6) - replace with the real generator.
Protocol-like traffic window on top of rss_shield.env.RSSHighwayEnv.
4 lanes, D veh/km/lane in [-300; +700] m around ego, jittered spacing, IDM equilibrium speeds,
wrap-around of vehicles leaving the window (position jumps outside act/step)."""
import numpy as np
from highway_env.vehicle.behavior import IDMVehicle
from rss_shield.env import RSSHighwayEnv

V_EQ = {10: 23.6, 25: 17.0, 40: 9.8}


class WindowEnv(RSSHighwayEnv):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({"D": 25, "lanes_count": 4, "duration": 10000, "vehicles_count": 0,
                    "observation": {"type": "Kinematics", "vehicles_count": 15,
                                    "see_behind": True, "absolute": False, "normalize": True}})
        return cfg

    def _create_vehicles(self):
        D, road, rng = self.config["D"], self.road, self.np_random
        s = 1000.0 / D
        j = min(20.0, (s - 5 - 2) / 2)
        v_eq = V_EQ.get(D, 17.0)
        x_e, l_e = 1000.0, int(rng.integers(0, 4))
        lane_e = road.network.get_lane(("0", "1", l_e))
        ego = self.action_type.vehicle_class(road, lane_e.position(x_e, 0), lane_e.heading_at(x_e), v_eq)
        self.controlled_vehicles = [ego]
        road.vehicles.append(ego)
        for l in range(4):
            lane = road.network.get_lane(("0", "1", l))
            for k in range(D):
                x = x_e - 300 + (k + 0.5) * s + rng.uniform(-j, j)
                if l == l_e and abs(x - x_e) < 15:
                    continue
                v = IDMVehicle(road, lane.position(x, 0), lane.heading_at(x),
                               v_eq + rng.uniform(-1, 1), target_speed=rng.uniform(23, 27))
                road.vehicles.append(v)

    def wrap(self):
        x_e = self.vehicle.position[0]
        for v in self.road.vehicles:
            if v is self.vehicle:
                continue
            dx = v.position[0] - x_e
            if dx < -300 or dx > 700:
                new_x = v.position[0] + (1000 if dx < 0 else -1000)
                y = v.position[1]
                if all(abs(o.position[0] - new_x) > 10 or abs(o.position[1] - y) > 2
                       for o in self.road.vehicles if o is not v):
                    v.position = np.array([new_x, y], dtype=float)
                    v.on_state_update()

    def step(self, action):
        out = super().step(action)
        self.wrap()
        return out
