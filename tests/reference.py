"""Independent discrete reference of table 2 (protocol 7.5, scenario 1A).

Written WITHOUT highway-env and without rss_shield.vehicle: plain arrays, explicit Euler with
the position updated before the speed (as Vehicle.step of highway-env), FIFO of 2 frames,
P speed controller K_P = 1/0.6 (ControlledVehicle.TAU_ACC), limiter of section 7.3.
Do not "fix" this file to match the simulation: it is the yardstick.
"""
DT = 0.1
TAU_ACC = 0.6
L = 5.0


def d_min_long(v_r, v_f, rho, a=3.0, bmin=4.0, bmax=8.0):
    return max(0.0, v_r * rho + 0.5 * a * rho ** 2 + (v_r + rho * a) ** 2 / (2 * bmin) - v_f ** 2 / (2 * bmax))


def table2_row(rho, phase, T=12.0):
    """Returns dict(d0, crashed, t_contact, final_gap, ego_path, lead_path).

    t = 0: ego and leader at 25 m/s, gap d0 = d_min_long(25, 25, rho); ego has FASTER
    (target 30 m/s) active, so before takeover a = min(3, (30 - v)/0.6).
    Leader: -8 m/s^2 from frame 0, no reversing (a = -v/dt on the last frame).
    TAKEOVER is issued at tick 0 (sync) or tick 1 (worst), reaches the actuator
    2 frames later; in takeover a = clip(a_ctrl, -8, -4) with target speed unchanged."""
    d0 = d_min_long(25.0, 25.0, rho)
    issue_frame = 0 if phase == "sync" else 2
    active_from = issue_frame + 2
    xe, ve, xl, vl = 0.0, 25.0, L + d0, 25.0
    for k in range(int(round(T / DT))):
        a_ctrl = (30.0 - ve) / TAU_ACC
        if k >= active_from:
            a = min(max(a_ctrl, -8.0), -4.0)
        else:
            a = min(max(a_ctrl, -8.0), 3.0)
        if ve + a * DT < 0:
            a = -ve / DT
        al = -8.0 if vl - 8.0 * DT >= 0 else -vl / DT
        xe += ve * DT
        ve += a * DT
        xl += vl * DT
        vl += al * DT
        gap = xl - xe - L
        if gap <= 0.0:
            return dict(d0=d0, crashed=True, t_contact=round((k + 1) * DT, 10), final_gap=None)
        if ve <= 1e-12 and vl <= 1e-12:
            break
    return dict(d0=d0, crashed=False, t_contact=None, final_gap=xl - xe - L,
                ego_path=xe, lead_path=xl - L - d0)


# Values printed in table 2 of v6 (rounded); used to check the reference itself.
TABLE2_V6 = {
    (0.1, "sync"): dict(d0=43.4638, crashed=True, t_contact=5.2),
    (0.1, "worst"): dict(d0=43.4638, crashed=True, t_contact=4.5),
    (0.2, "sync"): dict(d0=47.9175, crashed=False, final_gap=0.007),
    (0.2, "worst"): dict(d0=47.9175, crashed=True, t_contact=4.9),
    (0.4, "sync"): dict(d0=56.9825, crashed=False, final_gap=9.072),
    (0.4, "worst"): dict(d0=56.9825, crashed=False, final_gap=0.002),
}
