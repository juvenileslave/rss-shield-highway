"""R0: RSS distance formulas (section 7.4) against hand-computed values."""
import pytest
from rss_shield.rss import d_min_long, d_min_lat


@pytest.mark.parametrize("v_r,v_f,rho,expected", [
    (25, 25, 0.1, 43.4638), (25, 25, 0.2, 47.9175), (25, 25, 0.4, 56.9825),
    (25, 25, 1.0, 85.4375),
    (28, 20, 0.4, 93.02),      # scenario 2, v_ego = 20 (v6 had 87.77 = value for 28/22)
    (28, 22, 0.4, 87.77),
])
def test_R0_d_min_long_values(v_r, v_f, rho, expected):
    assert d_min_long(v_r, v_f, rho) == pytest.approx(expected, abs=5e-3)


def test_R0_d_min_long_non_negative_and_monotone_in_rho():
    assert d_min_long(0, 30, 0.4) == 0.0
    vals = [d_min_long(25, 25, r) for r in (0.1, 0.2, 0.4, 0.6, 1.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("rho", [0.1, 0.2, 0.4])
def test_R0_d_min_lat_closed_form_at_rest(rho):
    # v1 = v2 = 0: 2 * (0.5*1.5*rho^2 + (1.5*rho)^2 / (2*3)) + mu
    expected = 0.5 + 2 * (0.75 * rho ** 2 + (1.5 * rho) ** 2 / 6.0)
    assert d_min_lat(0.0, 0.0, rho) == pytest.approx(expected, abs=1e-12)
    assert d_min_lat(0.0, 0.0, 0.4) == pytest.approx(0.86, abs=1e-9)


def test_R0_d_min_lat_approaching_neighbour_needs_more_room():
    # neighbour 2 moves towards ego 1 with 1 m/s (v2 = -1 in the convention of rss.d_min_lat)
    assert d_min_lat(0.0, -1.0, 0.4) > d_min_lat(0.0, 0.0, 0.4)
    assert d_min_lat(0.0, 1.0, 0.4) <= d_min_lat(0.0, 0.0, 0.4)
