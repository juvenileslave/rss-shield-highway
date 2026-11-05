"""P0: lateral profile of the standard highway-env lane change (descriptive, R1.2)."""
import pytest
from rss_shield.scenarios import profile_lane_change


@pytest.mark.parametrize("v", [15.0, 20.0, 25.0, 30.0])
def test_P0_profile_is_finite_and_completes(v):
    p = profile_lane_change(v)
    assert p["duration"] is not None and 1.0 <= p["duration"] <= 3.0
    assert p["vlat_peak"] > 1.5          # exceeds the RSS lateral assumptions (a_lat 1.5 m/s^2)
    assert p["alat_peak"] > 1.5
