"""RSS-distance safety filter (mask, TAKEOVER, lane-change supervision).

Definition of admissibility (protocol R3.1, to be written into section 7.2 of v7):
an action a chosen at decision tick t_k becomes active on [t_{k+1}, t_{k+2}) because of the
FIFO delay; on [t_k, t_{k+1}) the already queued command acts. a is admissible if on every
predicted physics frame of the horizon no RSS dangerous situation is created, i.e. there is
no neighbour for which BOTH
    longitudinal gap < d_min_long   and   lateral clearance < d_min_lat.
Vehicles ahead are always checked; vehicles behind only in lanes the ego is entering.
Horizon: 2 decision periods for lane keeping; 2 decision periods + LC_HORIZON for lane
changes and while a maneuver is in progress.

Neighbour prediction model (ShieldConfig):
  long_prediction = "nominal"    - vehicles in the ego's own lane keep their current speed.
                                   RSS reactive scheme: leader braking is covered by d_min,
                                   reaction comes after tau_eff = tau + dt_decision
                                   (this is the model behind table 2 of the protocol).
                  = "worst_case" - own-lane leader brakes with -A_MAX_BRAKE on the horizon.
  Vehicles relevant to a lane change (target lane and lanes adjacent to it) are always
  predicted with worst-case bounds: ahead -A_MAX_BRAKE, behind +A_MAX_ACCEL (v >= 0).
  adjacent_lane_model = "occupancy" - every vehicle in a lane adjacent to the target lane is
                                      treated as occupying the target lane (v6 text, 7.4);
                      = "observed"  - only if it already moves laterally towards the target
                                      lane (|v_lat| > V_LAT_OBS).

Supervision of an ongoing maneuver (hierarchy of section 7.4), checked at every tick:
  if some action is admissible -> normal choice;
  else (1) before crossing: ABORT if the return is admissible (origin lane leader AND rear);
       (2) TAKEOVER (brake, keep lateral target) if completion with braking is admissible;
       (4) FALLBACK: brake in [-8,-4] and hold the lane containing the ego centre.
In lane keeping an empty safe set gives TAKEOVER. Takeover is released at the first tick
with a non-empty safe set.
"""
from dataclasses import dataclass, field
import numpy as np

from .params import (DT, FRAMES_PER_DECISION, A_MAX_ACCEL, A_MAX_BRAKE, ACTIONS, ACTION_INDEX,
                     TAKEOVER, ABORT, FALLBACK, LENGTH, WIDTH, LANE_WIDTH)
from .rss import d_min_long, d_min_lat

V_LAT_OBS = 0.1   # m/s, lateral speed treated as an observed lane change


@dataclass
class ShieldConfig:
    rho: float
    long_prediction: str = "nominal"
    adjacent_lane_model: str = "occupancy"
    lc_horizon: float = 2.0
    zone: float = 150.0          # longitudinal zone of considered vehicles, m
    eps: float = 1e-9


@dataclass
class Decision:
    command: object
    a_pi: object
    a_exec: object
    mask: np.ndarray
    takeover: bool = False
    fallback: bool = False
    reason: str = ""
    violations: dict = field(default_factory=dict)


class _NeighbourPrediction:
    """Neighbour prediction: constant longitudinal acceleration a (v >= 0, explicit Euler,
    position first, as in highway-env) and lateral motion towards y_goal.

    y_mode = "const"  - keeps current y;
           = "vel"    - current lateral speed, stopped at the centre of the lane it moves to;
           = "jump"   - worst case of the occupancy model: already at the target lane centre.
    """

    def __init__(self, veh, accel, y_mode="vel", y_goal=None):
        self.v0 = float(veh.speed)
        self.x0 = float(veh.position[0])
        self.y0 = float(veh.position[1])
        self.vy = float(veh.speed * np.sin(veh.heading))
        self.a = accel
        self.y_mode = y_mode
        self.y_goal = y_goal

    def at(self, n):
        x, v = self.x0, self.v0
        for _ in range(n):
            x += v * DT
            v = max(0.0, v + self.a * DT)
        if self.y_mode == "jump":
            return x, self.y_goal, v, 0.0
        if self.y_mode == "const" or abs(self.vy) <= V_LAT_OBS:
            return x, self.y0, v, 0.0
        y = self.y0 + self.vy * n * DT
        goal = self.y_goal
        if goal is not None and (y - goal) * np.sign(self.vy) > 0:
            return x, goal, v, 0.0
        return x, y, v, self.vy


class RSSShield:
    def __init__(self, ego, cfg: ShieldConfig):
        self.ego = ego
        self.cfg = cfg

    # ------------------------------------------------------------- helpers
    @property
    def road(self):
        return self.ego.road

    def _lane_id(self, lane_index):
        return lane_index[2]

    def _lanes_overlapped(self, y):
        """Lane ids whose area intersects the ego body (straight road, lane i centred at 4*i)."""
        ids = set()
        n_lanes = len(self.road.network.graph[self.ego.lane_index[0]][self.ego.lane_index[1]])
        for i in range(n_lanes):
            if abs(y - LANE_WIDTH * i) < LANE_WIDTH / 2 + WIDTH / 2 - 1e-6:
                ids.add(i)
        return ids

    def _others(self):
        x0 = self.ego.position[0]
        return [v for v in self.road.vehicles
                if v is not self.ego and abs(v.position[0] - x0) < self.cfg.zone]

    def _simulate_ego(self, commands, n_frames):
        """Predict ego with the real vehicle code (FIFO + controller + limiter)."""
        g = self.ego.ghost()
        states = []
        for f in range(n_frames):
            if f % FRAMES_PER_DECISION == 0:
                k = f // FRAMES_PER_DECISION
                if k < len(commands) and commands[k] is not None:
                    g.act(commands[k])
            g.act()
            g.step(DT)
            states.append((float(g.position[0]), float(g.position[1]), float(g.speed),
                           float(g.speed * np.sin(g.heading)), self._lane_id(g.lane_index),
                           self._lane_id(g.target_lane_index)))
        return states

    # ------------------------------------------------------------- check
    def check_plan(self, commands, n_frames, lc_lanes):
        """Violations of the ego command sequence on n_frames (empty dict = admissible).

        A predicted frame violates RSS w.r.t. a neighbour when BOTH the longitudinal and the
        lateral distances are below d_min (RSS dangerous situation, protocol 7.7):
          * neighbour ahead  -> always checked (ego must keep d_min to the front);
          * neighbour behind -> checked only if it is in / predicted to occupy a lane of
            lc_lanes, i.e. a lane the ego is entering (lane change target or return lane);
            in lane keeping ego is not responsible for the vehicle behind it.
        lc_lanes - lanes of the lane change / return; vehicles occupying them are predicted
                   with worst-case bounds (ahead -A_MAX_BRAKE, behind +A_MAX_ACCEL).
        """
        cfg = self.cfg
        ego_states = self._simulate_ego(commands, n_frames)
        lc_lanes = set(lc_lanes)
        preds = []
        for o in self._others():
            lane_o = self._lane_id(o.lane_index)
            vy = o.speed * np.sin(o.heading)
            moving = abs(vy) > V_LAT_OBS
            ahead = o.position[0] >= self.ego.position[0]
            occ = {lane_o}
            y_mode, y_goal = "vel", None
            if moving:
                y_goal = LANE_WIDTH * round((o.position[1] + np.sign(vy) * (LANE_WIDTH / 2 - 1e-6)) / LANE_WIDTH)
                if abs(o.position[1] - y_goal) < 1e-6:
                    y_goal = o.position[1] + np.sign(vy) * LANE_WIDTH
            goal_lane = int(round(y_goal / LANE_WIDTH)) if moving else None
            if moving:
                occ.add(goal_lane)
            for tl in lc_lanes:
                if tl in occ:
                    continue
                if abs(lane_o - tl) == 1 and cfg.adjacent_lane_model == "occupancy" and not moving:
                    occ.add(tl)
                    y_mode, y_goal = "jump", LANE_WIDTH * tl
            if occ & lc_lanes:
                a = -A_MAX_BRAKE if ahead else A_MAX_ACCEL
            elif cfg.long_prediction == "worst_case" and ahead:
                a = -A_MAX_BRAKE
            else:
                a = 0.0
            preds.append((o, _NeighbourPrediction(o, a, y_mode, y_goal), occ, ahead))
        viol = {}
        for n, (xe, ye, ve, vye, _, _) in enumerate(ego_states, start=1):
            for o, pred, occ, ahead in preds:
                if not ahead and not (occ & lc_lanes):
                    continue
                xo, yo, vo, vyo = pred.at(n)
                name = getattr(o, "name", str(id(o)))
                dx = xo - xe
                if dx >= 0:
                    gap, need = dx - LENGTH, d_min_long(ve, vo, cfg.rho)
                else:
                    gap, need = -dx - LENGTH, d_min_long(vo, ve, cfg.rho)
                if gap >= need - cfg.eps:
                    continue
                side = abs(yo - ye) - WIDTH
                sgn = np.sign(yo - ye) or 1.0
                need_lat = d_min_lat(vye * sgn, vyo * sgn, cfg.rho)
                if side < need_lat - cfg.eps:
                    key = "front" if dx >= 0 else "rear"
                    viol.setdefault(key, dict(frame=n, vehicle=name, gap=round(gap, 3),
                                              d_min_long=round(need, 3), side=round(side, 3),
                                              d_min_lat=round(need_lat, 3)))
        return viol

    # ------------------------------------------------------------- mask
    def _lc_target(self, action):
        lane = self._lane_id(self.ego.target_lane_index)
        n_lanes = len(self.road.network.graph[self.ego.lane_index[0]][self.ego.lane_index[1]])
        if action == "LANE_LEFT":
            return lane - 1 if lane - 1 >= 0 else None
        if action == "LANE_RIGHT":
            return lane + 1 if lane + 1 < n_lanes else None
        return None

    def mask(self):
        ego = self.ego
        m = np.zeros(5, dtype=int)
        info = {}
        lc_frames = 2 * FRAMES_PER_DECISION + int(round(self.cfg.lc_horizon / DT))
        keep_frames = 2 * FRAMES_PER_DECISION
        man = ego.maneuver
        for idx, name in ACTIONS.items():
            if name in ("LANE_LEFT", "LANE_RIGHT"):
                if man is not None:
                    info[name] = {"busy": True}
                    continue
                tl = self._lc_target(name)
                if tl is None:
                    info[name] = {"no_lane": True}
                    continue
                v = self.check_plan([name], lc_frames, {tl})
            else:
                if man is not None:
                    tl = self._lane_id(man["target"])
                    v = self.check_plan([name], lc_frames, {tl})
                else:
                    v = self.check_plan([name], keep_frames, set())
            m[idx] = int(not v)
            info[name] = v
        return m, info

    def decide(self, a_pi, q_values=None):
        """a_pi: unmasked policy choice (str). q_values: optional dict name->Q for argmax over A_safe."""
        ego = self.ego
        m, info = self.mask()
        safe = [ACTIONS[i] for i in range(5) if m[i]]
        if safe:
            if a_pi in safe:
                a_exec = a_pi
            elif q_values:
                a_exec = max(safe, key=lambda n: q_values[n])
            else:
                a_exec = "IDLE" if "IDLE" in safe else safe[0]
            return Decision(a_exec, a_pi, a_exec, m, reason="safe", violations=info)
        man = ego.maneuver
        lc_frames = 2 * FRAMES_PER_DECISION + int(round(self.cfg.lc_horizon / DT))
        if man is not None and man["kind"] in ("lane_change", "return"):
            origin = self._lane_id(man["origin"])
            target = self._lane_id(man["target"])
            if not man["crossed"] and man["kind"] == "lane_change":
                v = self.check_plan([ABORT], lc_frames, {origin})
                info["ABORT"] = v
                if not v:
                    return Decision(ABORT, a_pi, None, m, takeover=True, reason="return",
                                    violations=info)
            v = self.check_plan([TAKEOVER], lc_frames, {target})
            info["COMPLETE_BRAKE"] = v
            if not v:
                return Decision(TAKEOVER, a_pi, None, m, takeover=True, reason="complete",
                                violations=info)
            return Decision(FALLBACK, a_pi, None, m, takeover=True, fallback=True,
                            reason="fallback", violations=info)
        return Decision(TAKEOVER, a_pi, None, m, takeover=True, reason="empty_safe_set",
                        violations=info)
