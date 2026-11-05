"""RSS safe distances, formulas of section 7.4 of the protocol."""
from .params import (A_MAX_ACCEL, A_MIN_BRAKE, A_MAX_BRAKE,
                     A_LAT_MAX_ACCEL, A_LAT_MIN_BRAKE, MU_LAT)


def d_min_long(v_rear: float, v_front: float, rho: float,
               a_accel: float = A_MAX_ACCEL, b_min: float = A_MIN_BRAKE,
               b_max: float = A_MAX_BRAKE) -> float:
    """Minimal safe longitudinal bumper-to-bumper gap between a rear and a front vehicle."""
    v_r, v_f = max(0.0, v_rear), max(0.0, v_front)
    d = (v_r * rho + 0.5 * a_accel * rho ** 2
         + (v_r + rho * a_accel) ** 2 / (2.0 * b_min)
         - v_f ** 2 / (2.0 * b_max))
    return max(0.0, d)


def d_min_lat(v1: float, v2: float, rho: float, a_lat: float = A_LAT_MAX_ACCEL,
              b_lat: float = A_LAT_MIN_BRAKE, mu: float = MU_LAT) -> float:
    """Minimal safe lateral clearance (side to side).

    Sign convention: the axis points from vehicle 1 towards vehicle 2.
    v1 > 0 means vehicle 1 moves towards vehicle 2; v2 > 0 means vehicle 2 moves away from vehicle 1.
    """
    v1r = v1 + rho * a_lat
    v2r = v2 - rho * a_lat
    term = ((v1 + v1r) / 2.0 * rho + v1r ** 2 / (2.0 * b_lat)
            - ((v2 + v2r) / 2.0 * rho - v2r ** 2 / (2.0 * b_lat)))
    return mu + max(0.0, term)
