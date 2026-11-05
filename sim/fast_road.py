"""fast_road - bit-exact acceleration of highway-env 1.12.1 hot spots (O(N^2) per frame).

Patched:
  Road.act                - builds a per-frame cache: vehicle positions projected on lanes,
                            sorted by the longitudinal coordinate (vectorised).
  Road.neighbour_vehicles - inside Road.act only: walks the sorted cache from the query
                            position and evaluates the few candidates with the ORIGINAL code
                            (lane.local_coordinates, lane.on_lane(margin=1), same comparisons,
                            same tie-breaking by list order). Vehicles skipped by the walk
                            provably cannot change the original result (margin EPS_NV >> float
                            error of the vectorised projection).
  Road.step               - vectorised conservative pre-filters of vehicle pairs (spherical, as in
                            _is_colliding, then separating axes, as in are_polygons_intersecting);
                            candidate pairs go through the ORIGINAL handle_collisions in the
                            original (i, j) order. Skipped pairs provably get "no contact" from the
                            original code, which then has no side effects.
Falls back to the original code when: road.objects is not empty, neighbour_vehicles_connected_lanes
is enabled, a lane is not exactly StraightLane, a vehicle class overrides handle_collisions /
_is_colliding / polygon / direction / velocity, or neighbour_vehicles is called outside Road.act.

Usage: from sim import fast_road; fast_road.enable()   (fast_road.disable() restores the original)
"""
from __future__ import annotations

from bisect import bisect_left

import numpy as np
from highway_env.road.lane import StraightLane
from highway_env.road.road import Road
from highway_env.vehicle.kinematics import Vehicle as _KVehicle
from highway_env.vehicle.objects import RoadObject

# originals are stored on the class once, so a second import (e.g. as sim.fast_road and as
# fast_road) never captures already patched methods
if "_fast_road_orig" not in Road.__dict__:
    Road._fast_road_orig = {"act": Road.act, "step": Road.step,
                            "neighbour_vehicles": Road.neighbour_vehicles}
_ORIG = Road._fast_road_orig
EPS_NV = 1e-6    # m; float error of the vectorised projection is ~1e-11 m at |s| ~ 1e4 m
EPS_COL = 1e-3   # m; extra margin of the spherical collision pre-filter
EPS_SAT = 1e-4   # m; required gap of the separating-axis pre-filter (float error ~1e-9 m)
MARGIN = 1.0     # Road.neighbour_vehicles uses lane.on_lane(..., margin=1)
_BASE = dict(handle_collisions=RoadObject.handle_collisions, _is_colliding=RoadObject._is_colliding,
             polygon=RoadObject.polygon, direction=RoadObject.direction, velocity=_KVehicle.velocity)


class _FrameCache:
    __slots__ = ("vehicles", "pos", "index", "lanes")

    def __init__(self, vehicles):
        self.vehicles = vehicles
        self.pos = np.array([v.position for v in vehicles], dtype=float).reshape(len(vehicles), 2)
        self.index = {id(v): i for i, v in enumerate(vehicles)}
        self.lanes = {}

    def lane(self, lane_index, lane):
        entry = self.lanes.get(lane_index)
        if entry is None:
            d = self.pos - lane.start
            s = d @ lane.direction
            lat = d @ lane.direction_lateral
            ok = ((np.abs(lat) <= lane.width / 2 + MARGIN + EPS_NV)
                  & (s >= -lane.VEHICLE_LENGTH - EPS_NV)
                  & (s < lane.length + lane.VEHICLE_LENGTH + EPS_NV))
            idx = np.flatnonzero(ok)
            idx = idx[np.argsort(s[idx], kind="stable")]
            entry = (idx.tolist(), s[idx].tolist())
            self.lanes[lane_index] = entry
        return entry


def _act(self):
    if self.objects or getattr(self, "neighbour_vehicles_connected_lanes", False):
        return _ORIG["act"](self)
    self._fast_nv_cache = _FrameCache(self.vehicles)
    try:
        for vehicle in self.vehicles:
            vehicle.act()
    finally:
        self._fast_nv_cache = None


def _neighbour_vehicles(self, vehicle, lane_index=None):
    cache = self.__dict__.get("_fast_nv_cache")
    if cache is None:
        return _ORIG["neighbour_vehicles"](self, vehicle, lane_index)
    lane_index = lane_index or vehicle.lane_index
    if not lane_index:
        return None, None
    lane = self.network.get_lane(lane_index)
    if type(lane) is not StraightLane:
        return _ORIG["neighbour_vehicles"](self, vehicle, lane_index)
    s = lane.local_coordinates(vehicle.position)[0]
    idx, sa = cache.lane(lane_index, lane)
    vehicles = cache.vehicles
    me = cache.index.get(id(vehicle), -1)
    n = len(idx)

    # front: original keeps the LAST vehicle (list order) with minimal s_v among s_v >= s
    cands, best = [], None
    k = bisect_left(sa, s - EPS_NV)
    while k < n:
        if best is not None and sa[k] > best + 2 * EPS_NV:
            break
        i = idx[k]
        k += 1
        if i == me:
            continue
        v = vehicles[i]
        s_v, lat_v = lane.local_coordinates(v.position)
        if not lane.on_lane(v.position, s_v, lat_v, margin=1):
            continue
        if s <= s_v:
            cands.append((i, s_v, v))
            if best is None or s_v < best:
                best = s_v
    s_front = v_front = None
    for _, s_v, v in sorted(cands, key=lambda c: c[0]):
        if s_front is None or s_v <= s_front:
            s_front, v_front = s_v, v

    # rear: original keeps the FIRST vehicle (list order) with maximal s_v among s_v < s
    cands, best = [], None
    k = bisect_left(sa, s + EPS_NV) - 1
    while k >= 0:
        if best is not None and sa[k] < best - 2 * EPS_NV:
            break
        i = idx[k]
        k -= 1
        if i == me:
            continue
        v = vehicles[i]
        s_v, lat_v = lane.local_coordinates(v.position)
        if not lane.on_lane(v.position, s_v, lat_v, margin=1):
            continue
        if s_v < s:
            cands.append((i, s_v, v))
            if best is None or s_v > best:
                best = s_v
    s_rear = v_rear = None
    for _, s_v, v in sorted(cands, key=lambda c: c[0]):
        if s_rear is None or s_v > s_rear:
            s_rear, v_rear = s_v, v
    return v_front, v_rear


def _plain_types(vehicles):
    for t in {type(v) for v in vehicles}:
        if not issubclass(t, _KVehicle) or any(getattr(t, k) is not f for k, f in _BASE.items()):
            return False
    return True


def _separated(pos, c, s, hl, hw, step, ii, jj):
    """True where the pair is separated by > EPS_SAT along one of the rectangles' edge normals even
    after the relative displacement over dt: are_polygons_intersecting tests exactly these axes, so
    it returns (False, False, None) for such pairs and handle_collisions does nothing."""
    ci, si, cj, sj = c[ii], s[ii], c[jj], s[jj]
    dcx, dcy = pos[jj, 0] - pos[ii, 0], pos[jj, 1] - pos[ii, 1]
    rx, ry = step[ii] * ci - step[jj] * cj, step[ii] * si - step[jj] * sj
    sep = np.zeros(ii.size, dtype=bool)
    for ux, uy in ((ci, si), (-si, ci), (cj, sj), (-sj, cj)):
        ra = hl[ii] * np.abs(ci * ux + si * uy) + hw[ii] * np.abs(ci * uy - si * ux)
        rb = hl[jj] * np.abs(cj * ux + sj * uy) + hw[jj] * np.abs(cj * uy - sj * ux)
        sep |= np.abs(dcx * ux + dcy * uy) - ra - rb - np.abs(rx * ux + ry * uy) > EPS_SAT
    return sep


def _step(self, dt):
    vehicles = self.vehicles
    if self.objects or not _plain_types(vehicles):
        return _ORIG["step"](self, dt)
    for vehicle in vehicles:
        vehicle.step(dt)
    n = len(vehicles)
    if n < 2:
        return
    pos = np.array([v.position for v in vehicles], dtype=float)
    diag = np.array([v.diagonal for v in vehicles], dtype=float)
    reach = np.abs(np.array([v.speed for v in vehicles], dtype=float)) * abs(dt)
    dx = pos[:, 0, None] - pos[None, :, 0]
    dy = pos[:, 1, None] - pos[None, :, 1]
    thr = (diag[:, None] + diag[None, :]) * 0.5 + reach[:, None] + EPS_COL
    ii, jj = np.nonzero(np.triu(dx * dx + dy * dy <= thr * thr, 1))   # row-major = original order
    if ii.size:
        hd = np.array([v.heading for v in vehicles], dtype=float)
        sp = np.array([v.speed for v in vehicles], dtype=float)
        hl = np.array([v.LENGTH for v in vehicles], dtype=float) / 2
        hw = np.array([v.WIDTH for v in vehicles], dtype=float) / 2
        keep = ~_separated(pos, np.cos(hd), np.sin(hd), hl, hw, sp * dt, ii, jj)
        ii, jj = ii[keep], jj[keep]
    for i, j in zip(ii.tolist(), jj.tolist()):
        vehicles[i].handle_collisions(vehicles[j], dt)


def enable():
    """Patch Road (all existing and future instances). Call in every process before stepping envs."""
    import importlib.metadata
    ver = importlib.metadata.version("highway-env")
    if ver != "1.12.1":
        raise RuntimeError(f"fast_road is verified for highway-env 1.12.1 only, found {ver}")
    Road.act, Road.step, Road.neighbour_vehicles = _act, _step, _neighbour_vehicles


def disable():
    Road.act, Road.step, Road.neighbour_vehicles = _ORIG["act"], _ORIG["step"], _ORIG["neighbour_vehicles"]
