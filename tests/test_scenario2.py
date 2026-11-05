"""T2a, T2b-i, T2b-iii, T2b-occ: lane-change supervision (section 7.4, scenario 2)."""
import numpy as np
import pytest
from rss_shield.params import LENGTH, ACTION_INDEX, ABORT, FALLBACK, K_TAU
from rss_shield.rss import d_min_long
from rss_shield.scenarios import scenario2a, scenario2b

LL = ACTION_INDEX["LANE_LEFT"]
LR = ACTION_INDEX["LANE_RIGHT"]


def test_T2a_lane_change_forbidden_while_rear_gap_below_d_min():
    r = scenario2a(T=4.0)
    assert r.t_contact is None
    checked = 0
    passed_frame = None
    for row in r.rows:
        if passed_frame is None and row["R_x"] - row["x_end"] >= LENGTH:
            passed_frame = row["frame"]
        if not row["decision_tick"]:
            continue
        dx = row["x_end"] - row["R_x"]                      # state used at the next tick
        gap = abs(dx) - LENGTH
        need = d_min_long(row["R_v"], row["v_end"], 0.4) if dx > 0 else d_min_long(row["v_end"], row["R_v"], 0.4)
        if gap < need:
            checked += 1
    assert checked > 0
    for tick, t, d in r.decisions:
        if tick * 2 > passed_frame:
            break
        assert d.mask[LL] == 0 and d.a_exec == "IDLE", (t, d.mask)
    for row in r.rows[:passed_frame + 1]:
        assert row["y_end"] == pytest.approx(4.0, abs=1e-9)
    assert passed_frame is not None


def _cmds(r):
    return [(round(t, 1), d.command) for _, t, d in r.decisions if d.command not in ("IDLE",)]


def test_T2b_i_abort_returns_to_origin_lane():
    r = scenario2b("i", model="observed")
    cmds = _cmds(r)
    assert cmds[0] == (0.0, "LANE_RIGHT")
    assert cmds[1] == (0.4, ABORT)                  # first tick after t_b = 0.2 with N_vlat > 0.1
    assert all(c != FALLBACK for _, c in cmds)
    tl = [row["target_lane_index"] for row in r.rows]
    assert tl[2] == 2                               # LANE_RIGHT active from frame 2
    assert tl[2 * 2 + K_TAU] == 1                   # ABORT active at frame 6 (t = 0.6)
    assert all(x == 1 for x in tl[6:])
    assert r.t_contact is None
    assert abs(r.rows[-1]["y_end"] - 4.0) < 0.1
    assert max(row["lane_index"] for row in r.rows) == 1     # ego centre never left lane 1


def test_T2b_iii_fallback_when_return_and_completion_inadmissible():
    r = scenario2b("iii", model="observed", T=8.0)
    cmds = _cmds(r)
    assert cmds[0] == (0.0, "LANE_RIGHT")
    assert cmds[1] == (0.4, FALLBACK)
    d = [d for _, t, d in r.decisions if d.command == FALLBACK][0]
    assert d.violations["ABORT"] and d.violations["COMPLETE_BRAKE"]   # both inadmissible
    fb = [row for row in r.rows if row["fallback"]]
    assert fb and fb[0]["frame"] == 2 * 2 + K_TAU                     # flag in the log
    assert all(row["a"] <= -4.0 + 1e-9 for row in fb)
    assert r.t_contact is None                       # RSS-compliant rear vehicle R
    assert abs(r.rows[-1]["y_end"] - 4.0) < 0.1


def test_T2b_iii_nonreacting_rear_is_recorded_as_env_collision():
    """R keeps 28 m/s and ignores RSS: contact from behind is CR_env, not a shield failure."""
    r = scenario2b("iii", model="observed", T=10.0, rear_reactive=False)
    assert (0.4, FALLBACK) in _cmds(r)
    assert r.t_contact is not None
    last = r.rows[-1]
    assert last["R_x"] < last["x_end"]               # hit from behind


def test_T2b_occ_occupancy_model_forbids_lane_change_at_tick0():
    r = scenario2b("i", model="occupancy")
    first = r.decisions[0][2]
    assert first.mask[LR] == 0
    assert all(d.command == "IDLE" for _, _, d in r.decisions)
    assert all(row["y_end"] == pytest.approx(4.0, abs=1e-9) for row in r.rows)
