"""Integration with highway-env: the delayed / limited ego is used through the regular env.step().

HighwayEnv creates the ego with action_type.vehicle_class(...). RSSMetaAction returns
DelayedRSSVehicle there, so FIFO + limiter work inside the real _simulate() loop
(action_type.act() on frame 0 of a step, road.act() -> vehicle.act() on every frame).
env.step() accepts a policy action index 0..4 or a supervisory command string
("TAKEOVER", "ABORT", "FALLBACK") produced by the shield.
"""
import functools
from highway_env.envs.common.action import DiscreteMetaAction
from highway_env.envs.highway_env import HighwayEnv

from .params import TARGET_SPEEDS, DT, DT_DECISION
from .vehicle import DelayedRSSVehicle


class RSSMetaAction(DiscreteMetaAction):
    def __init__(self, env, **kwargs):
        kwargs.setdefault("target_speeds", TARGET_SPEEDS)
        super().__init__(env, **kwargs)

    @property
    def vehicle_class(self):
        return functools.partial(DelayedRSSVehicle, target_speeds=self.target_speeds)

    def act(self, action):
        cmd = action if isinstance(action, str) else self.actions[int(action)]
        self.controlled_vehicle.act(cmd)


class RSSHighwayEnv(HighwayEnv):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": int(round(1 / DT)),       # 10 Hz
            "policy_frequency": int(round(1 / DT_DECISION)),  # 5 Hz
            "action": {"type": "DiscreteMetaAction", "target_speeds": list(TARGET_SPEEDS)},
        })
        return cfg

    def define_spaces(self):
        super().define_spaces()
        kw = {k: v for k, v in self.config["action"].items() if k != "type"}
        self.action_type = RSSMetaAction(self, **kw)

    def _reset(self):
        assert self.config["simulation_frequency"] == int(round(1 / DT)), "DT of the package != env"
        assert self.config["policy_frequency"] == int(round(1 / DT_DECISION))
        super()._reset()


def make_env(**config):
    env = RSSHighwayEnv(config=config or None)
    return env
