"""L1-L7: scenario 1A against the independent reference (tests/reference.py) and table 2 of v6."""
import pytest
from reference import table2_row, TABLE2_V6
from rss_shield.scenarios import scenario1A

ROWS = [(0.1, "sync", "L1"), (0.1, "worst", "L2"), (0.2, "sync", "L3"),
        (0.2, "worst", "L4"), (0.4, "sync", "L5"), (0.4, "worst", "L6")]


@pytest.mark.parametrize("rho,phase,code", ROWS)
def test_L0_reference_reproduces_table2_of_v6(rho, phase, code):
    ref, v6 = table2_row(rho, phase), TABLE2_V6[(rho, phase)]
    assert ref["d0"] == pytest.approx(v6["d0"], abs=1e-4)
    assert ref["crashed"] == v6["crashed"]
    if v6["crashed"]:
        assert ref["t_contact"] == pytest.approx(v6["t_contact"], abs=1e-9)
    else:
        assert ref["final_gap"] == pytest.approx(v6["final_gap"], abs=1e-3)


@pytest.mark.parametrize("rho,phase,code", ROWS)
def test_L1_L6_simulation_matches_reference(rho, phase, code):
    ref = table2_row(rho, phase)
    res = scenario1A(rho, phase).result
    assert res["crashed"] == ref["crashed"], code
    if ref["crashed"]:
        assert abs(res["t_contact"] - ref["t_contact"]) <= 0.1 + 1e-9, code
    else:
        assert abs(res["final_gap"] - ref["final_gap"]) <= 0.01, code


def test_L7_single_slower_instead_of_takeover_collides():
    """One SLOWER at the detection tick (rho = 0.2, sync, FASTER active, target 30 -> 25):
    the target equals the current speed, the controller does not brake -> contact.
    v6 draft value: t ~ 4.2 s."""
    res = scenario1A(0.2, "sync", slower_instead="once").result
    assert res["crashed"]
    assert abs(res["t_contact"] - 4.2) <= 0.1 + 1e-9


def test_L7b_slower_every_tick_is_recorded():
    """SLOWER on every tick lowers the target by 5 m/s per 0.2 s, the P controller saturates
    at the -8 m/s^2 limit and matches the leader: NO contact in this implementation.
    This contradicts the v6 wording "SLOWER на каждом такте ... контакт при t ~ 4.2 с";
    the test only fixes the behaviour, the text of 7.5 must say "однократный SLOWER"."""
    res = scenario1A(0.2, "sync", slower_instead="every_tick").result
    assert not res["crashed"]
    assert res["final_gap"] > 0
