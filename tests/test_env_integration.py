"""E1-E3: the same vehicle inside the real HighwayEnv.step() loop (no wrapper of step)."""
import numpy as np
import pytest
from rss_shield.env import make_env
from rss_shield.vehicle import DelayedRSSVehicle
from rss_shield.shield import RSSShield, ShieldConfig
from rss_shield.params import ACTION_INDEX, SUPERVISORY, A_MAX_ACCEL, A_MAX_BRAKE, A_MIN_BRAKE


def _env(n=0, seed=0):
    env = make_env(vehicles_count=n, duration=60)
    env.reset(seed=seed)
    ego = env.unwrapped.vehicle
    rec = []
    road = env.unwrapped.road
    orig = road.step

    def step(dt):
        v0 = ego.speed
        orig(dt)
        rec.append(dict(frame=ego.frame - 1, cmd=ego.active_cmd, takeover=ego.takeover,
                        a_ctrl=ego.a_ctrl, a=ego.a_applied, a_int=(ego.speed - v0) / dt,
                        v=ego.speed, crashed=ego.crashed))
    road.step = step
    return env, ego, rec


def test_E1_env_uses_delayed_vehicle_and_protocol_frequencies():
    env, ego, _ = _env()
    assert isinstance(ego, DelayedRSSVehicle)
    assert env.unwrapped.config["simulation_frequency"] == 10
    assert env.unwrapped.config["policy_frequency"] == 5
    assert list(ego.target_speeds) == [0, 5, 10, 15, 20, 25, 30]


def test_E2_delay_two_frames_and_limiter_survive_simulate_loop():
    env, ego, rec = _env()
    env.step(ACTION_INDEX["FASTER"])        # frames 0,1: FASTER queued, preset command acts
    env.step(ACTION_INDEX["IDLE"])          # frames 2,3: FASTER active
    env.step("TAKEOVER")                    # frames 4,5
    env.step(ACTION_INDEX["IDLE"])          # frames 6,7: TAKEOVER active, IDLE queued
    env.step(ACTION_INDEX["IDLE"])          # frames 8,9: IDLE active -> takeover released
    env.step(ACTION_INDEX["IDLE"])
    assert [r["frame"] for r in rec] == list(range(12))
    assert rec[0]["cmd"] is None and rec[1]["cmd"] is None
    assert rec[2]["cmd"] == "FASTER" and rec[2]["a_ctrl"] > A_MAX_ACCEL
    assert rec[2]["a"] == pytest.approx(A_MAX_ACCEL) and rec[3]["a"] == pytest.approx(A_MAX_ACCEL)
    assert not rec[5]["takeover"]
    for r in rec[6:8]:
        assert r["takeover"] and r["cmd"] == "TAKEOVER" and r["a"] == pytest.approx(-A_MIN_BRAKE)
    assert not rec[8]["takeover"] and rec[8]["cmd"] == "IDLE"
    for r in rec:
        assert abs(r["a_int"] - r["a"]) < 1e-9          # what was integrated = limited value


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_E3_shield_in_traffic_keeps_limits(seed):
    env, ego, rec = _env(n=30, seed=seed)
    sh = RSSShield(ego, ShieldConfig(rho=0.4))
    rng = np.random.RandomState(seed)
    names = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
    for _ in range(100):
        d = sh.decide(names[rng.randint(5)])
        cmd = d.command if d.command in SUPERVISORY else ACTION_INDEX[d.command]
        _, _, term, trunc, _ = env.step(cmd)
        if term or trunc:
            break
    for r in rec:
        if r["crashed"]:
            break
        assert -A_MAX_BRAKE - 1e-9 <= r["a"] <= A_MAX_ACCEL + 1e-9
        assert abs(r["a_int"] - r["a"]) < 1e-9
        assert r["v"] >= -1e-9
