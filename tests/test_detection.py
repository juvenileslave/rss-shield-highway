"""L8: scenario 1B, tick at which the shield sets A_safe = empty (R1.6).

Expected ticks follow from the mask definition in rss_shield/shield.py:
  long_prediction = "nominal"   : leader predicted at constant speed; at t = 0 the gap is
      d_min + 0.5 with equal speeds -> admissible; at t = 0.2 the leader is at 23.4 m/s, the
      required gap grew by (25^2 - 23.4^2)/16 = 4.84 m > 0.5 m margin -> tick 1 (worst phase).
  long_prediction = "worst_case": leader predicted with -8 m/s^2 on the 0.4 s horizon -> the
      required gap grows already in the prediction from t = 0 -> tick 0 (sync phase).
"""
import pytest
from rss_shield.scenarios import scenario1B

EXPECTED = {"nominal": 1, "worst_case": 0}


@pytest.mark.parametrize("rho", [0.1, 0.2, 0.4])
@pytest.mark.parametrize("model", ["nominal", "worst_case"])
def test_L8_detection_tick(rho, model):
    assert scenario1B(rho, model).result["detect_tick"] == EXPECTED[model]


def test_L8_no_false_alarm_without_braking_leader():
    """Control: same start, leader keeps 25 m/s -> nominal model never empties A_safe."""
    from rss_shield.sim import make_road, add_ego, add_scripted, Runner, shielded
    from rss_shield.shield import RSSShield, ShieldConfig
    from rss_shield.rss import d_min_long
    road = make_road()
    ego = add_ego(road, 1, 100.0, 25.0, target_speed=25.0, initial_active="IDLE")
    add_scripted(road, 1, 105.0 + d_min_long(25, 25, 0.4) + 0.5, 25.0, "L")
    sh = RSSShield(ego, ShieldConfig(rho=0.4))
    r = Runner(road, ego, shielded(sh, lambda k, t: "IDLE"), 0.4).run(3.0)
    assert all(d.command == "IDLE" for _, _, d in r.decisions)
