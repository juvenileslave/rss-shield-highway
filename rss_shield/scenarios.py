"""Control scenarios of protocol section 7.5 (v6) with the fixes of the v7 instruction."""
import numpy as np
from .params import DT, LENGTH, TAKEOVER, A_MAX_BRAKE
from .rss import d_min_long
from .sim import make_road, add_ego, add_scripted, Runner, scripted, shielded
from .shield import RSSShield, ShieldConfig

X0 = 100.0


# ---------------------------------------------------------------- scenario 1A
def scenario1A(rho, phase, slower_instead=None, T=12.0):
    """Execution chain. Ego and leader at 25 m/s, d0 = d_min_long(25,25,rho), FASTER active
    before t=0 (target 30 m/s, controller 8.33 m/s^2 clipped to +3). Leader brakes -8 m/s^2
    from the first physics frame. Detection tick is scripted: sync -> tick 0, worst -> tick 1.
    slower_instead: None | "once" (one SLOWER at the detection tick, then IDLE) |
                    "every_tick" (SLOWER on every tick from the detection tick)."""
    road = make_road()
    d0 = d_min_long(25.0, 25.0, rho)
    ego = add_ego(road, 1, X0, 25.0, target_speed=30.0, initial_active="FASTER")
    lead = add_scripted(road, 1, X0 + LENGTH + d0, 25.0, "L", accel=lambda t: -A_MAX_BRAKE)
    det = 0 if phase == "sync" else 1
    if slower_instead is None:
        sched = lambda k, t: TAKEOVER if k >= det else "IDLE"
    elif slower_instead == "once":
        sched = lambda k, t: "SLOWER" if k == det else "IDLE"
    else:
        sched = lambda k, t: "SLOWER" if k >= det else "IDLE"
    r = Runner(road, ego, scripted(sched), rho,
               name=f"S1A_rho{rho}_{phase}" + (f"_SLOWER_{slower_instead}" if slower_instead else ""))
    r.run(T, stop=lambda rr: rr.ego.speed <= 1e-12 and lead.speed <= 1e-12)
    last = r.rows[-1]
    r.result = dict(rho=rho, phase=phase, d0=d0, detect_tick=det, crashed=bool(ego.crashed),
                    t_contact=r.t_contact,
                    final_gap=None if ego.crashed else float(lead.position[0] - ego.position[0] - LENGTH),
                    ego_path=float(ego.position[0] - X0), lead_path=float(lead.position[0] - (X0 + LENGTH + d0)))
    return r


# ---------------------------------------------------------------- scenario 1B
def scenario1B(rho, long_prediction, margin=0.5, T=1.2):
    """Detection logic. Equal speeds 25 m/s, ego IDLE (target 25), d0 = d_min_long + margin;
    leader brakes -8 m/s^2 from the first frame after t=0. Policy proposes IDLE every tick.
    Returns the first tick with A_safe = empty (TAKEOVER)."""
    road = make_road()
    d0 = d_min_long(25.0, 25.0, rho) + margin
    ego = add_ego(road, 1, X0, 25.0, target_speed=25.0, initial_active="IDLE")
    lead = add_scripted(road, 1, X0 + LENGTH + d0, 25.0, "L", accel=lambda t: -A_MAX_BRAKE)
    sh = RSSShield(ego, ShieldConfig(rho=rho, long_prediction=long_prediction))
    r = Runner(road, ego, shielded(sh, lambda k, t: "IDLE"), rho,
               name=f"S1B_rho{rho}_{long_prediction}")
    r.run(T)
    ticks = [tick for tick, t, d in r.decisions if d.command == TAKEOVER]
    r.result = dict(rho=rho, long_prediction=long_prediction, d0=d0,
                    detect_tick=ticks[0] if ticks else None)
    return r


# ---------------------------------------------------------------- stop / release
def scenario_stop(v0=10.0, T_brake=4.0, T_hold=2.0):
    """S1 braking to standstill in takeover, S2 holding v = 0 under takeover."""
    road = make_road()
    ego = add_ego(road, 1, X0, v0, target_speed=v0, initial_active="IDLE")
    r = Runner(road, ego, scripted(lambda k, t: TAKEOVER), 0.4, name="S1S2_stop_hold")
    r.run(T_brake + T_hold)
    return r


def scenario_release(v0=20.0, release_tick=5, then="IDLE", T=4.0, standstill=False):
    """S3 release of takeover: takeover in progress at t = 0 (target speed 30 m/s left from an
    earlier FASTER), TAKEOVER on ticks < release_tick, then `then` every tick."""
    road = make_road()
    ego = add_ego(road, 1, X0, 0.0 if standstill else v0, target_speed=30.0,
                  initial_active=TAKEOVER)
    r = Runner(road, ego, scripted(lambda k, t: TAKEOVER if k < release_tick else then), 0.4,
               name=f"S3_release_{'standstill' if standstill else 'moving'}_{then}")
    r.run(T)
    r.release_tick = release_tick
    return r


# ---------------------------------------------------------------- random limiter stress
def scenario_random(seed=0, T=60.0):
    rng = np.random.RandomState(seed)
    cmds = ["IDLE", "FASTER", "SLOWER", "LANE_LEFT", "LANE_RIGHT", TAKEOVER]
    p = [0.25, 0.25, 0.2, 0.05, 0.05, 0.2]
    road = make_road()
    ego = add_ego(road, 1, X0, 15.0, target_speed=15.0, initial_active="IDLE")
    seq = rng.choice(cmds, size=int(T / 0.2) + 1, p=p)
    r = Runner(road, ego, scripted(lambda k, t: str(seq[k])), 0.4, name=f"A1_random_seed{seed}")
    r.run(T)
    return r


# ---------------------------------------------------------------- scenario 2
EGO_Y_LANE = 1


def scenario2a(rho=0.4, T=3.0):
    """Ego lane 1, 20 m/s; rear vehicle R in target lane 0, 18 m behind (bumper gap), 28 m/s,
    constant speed (scripted). Policy asks LANE_LEFT every tick."""
    road = make_road()
    ego = add_ego(road, 1, X0, 20.0, target_speed=20.0, initial_active="IDLE")
    add_scripted(road, 0, X0 - LENGTH - 18.0, 28.0, "R")
    sh = RSSShield(ego, ShieldConfig(rho=rho))
    r = Runner(road, ego, shielded(sh, lambda k, t: "LANE_LEFT"), rho, name="S2a")
    r.run(T)
    return r


def scenario2b(variant="i", rho=0.4, t_b=0.2, model="observed", T=5.0, gap_n=40.0,
               rear_margin=1.0, rear_reactive=True):
    """Ego lane 1 -> lane 2 (LANE_RIGHT at tick 0) at 20 m/s. Neighbour N in lane 3, gap_n m
    ahead (bumper gap), 20 m/s; at t_b it starts a lane change into lane 2.
    variant "i":   origin lane empty -> return (ABORT) expected;
    variant "iii": rear vehicle R in origin lane 1 at 28 m/s, bumper gap
                   d_min_long(28, 20, rho) + rear_margin behind the ego (R complies with RSS at
                   t = 0). With worst-case prediction of R (+A_MAX_ACCEL) the return is
                   inadmissible, completion too -> FALLBACK expected. rear_reactive=True: R is an
                   RSS-compliant follower (brakes -A_MIN_BRAKE after rho when its gap < d_min);
                   False: R keeps 28 m/s (non-reacting, any contact is CR_env by the rear car).
    model: adjacent_lane_model of the shield ("observed" / "occupancy")."""
    road = make_road()
    ego = add_ego(road, 1, X0, 20.0, target_speed=20.0, initial_active="IDLE")
    add_scripted(road, 3, X0 + LENGTH + gap_n, 20.0, "N", lane_change=(t_b, ("0", "1", 2)))
    if variant == "iii":
        gap_r = d_min_long(28.0, 20.0, rho) + rear_margin
        add_scripted(road, 1, X0 - LENGTH - gap_r, 28.0, "R",
                     rss_follow=rho if rear_reactive else None)
    sh = RSSShield(ego, ShieldConfig(rho=rho, adjacent_lane_model=model))
    r = Runner(road, ego, shielded(sh, lambda k, t: "LANE_RIGHT" if k == 0 else "IDLE"), rho,
               name=f"S2b_{variant}_{model}")
    r.run(T, stop_after_crash=0.5)
    return r


# ---------------------------------------------------------------- P0 lane change profile
def profile_lane_change(v=20.0, T=4.0):
    """Lateral profile of the standard highway-env lane change (ControlledVehicle cascade,
    same controller as MOBIL lane changes of background traffic)."""
    road = make_road()
    veh = add_scripted(road, 1, X0, v, "P", lane_change=(0.0, ("0", "1", 2)))
    ys, vys = [], []
    for k in range(int(T / DT)):
        road.act(); road.step(DT)
        ys.append(float(veh.position[1])); vys.append(float(veh.speed * np.sin(veh.heading)))
    ys, vys = np.array(ys), np.array(vys)
    a_lat = np.diff(vys) / DT
    done = np.argmax(np.abs(ys - 8.0) < 0.1) if np.any(np.abs(ys - 8.0) < 0.1) else None
    return dict(v=v, duration=None if done is None else (done + 1) * DT,
                vlat_peak=float(np.max(np.abs(vys))), alat_peak=float(np.max(np.abs(a_lat))))
