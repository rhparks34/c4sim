"""Candidate paths (targets) for one defender over the next second from his position at the snap: a leverage corridor on each
receiver (receiver-match options) and the path of each of the seven Cover-4 roles, driven by the coded rules. Verbatim from the
research code (w42 target builders): inputs are only the defender's snap position, the routes, the line of scrimmage and the
horizon; no assignment, label or realized defender motion is reachable."""
from __future__ import annotations

import math
import pathlib

import numpy as np

from src.prep.rules.route_concepts import build_receiver_routes
from src.prep.rules.cover4_rules import (
    DEEP_ROLE_ORDER,
    DefenderRole,
    DefenderSimState,
    ROLE_NAME_TO_IDX,
    ROLE_ORDER,
    SimParams,
    _assign_deep_lane_bounds,
    _blend_target,
    _clip_field_xy,
    _clip_role_xy,
    _deep_cluster_target_y,
    _protect_deep_over_top_x,
    _role_landmarks,
    _routes_by_side,
    assignment_trust_for_role,
    choose_responsibility_state,
    movement_anchor_for_decision,
    raw_target_for_decision,
    target_blend_for_decision,
)

DT = 0.1


Y_MID = 26.65


VEL_CLIP = 10.0           # yd/s magnitude clip on all target velocities


DEPTH_CLIP = (-2.0, 6.0)   # yd, receiver corridor depth offset


INSIDE_CLIP = (-3.0, 3.0)  # yd, receiver corridor inside offset


CENTER_EPS = 0.25          # yd, dead zone for the e_inside convention


REPO = pathlib.Path(__file__).resolve().parent


FROZEN_SIM_PARAMS_PATH = REPO / "frozen_params.json"       # the frozen rule constants (v1.18)


_SIM_PARAMS_CACHE: SimParams | None = None


def frozen_sim_params() -> SimParams:
    """The frozen legal rule-simulator constants (v1.18 surface params)."""
    global _SIM_PARAMS_CACHE
    if _SIM_PARAMS_CACHE is None:
        if FROZEN_SIM_PARAMS_PATH.exists():
            _SIM_PARAMS_CACHE = SimParams.from_json(FROZEN_SIM_PARAMS_PATH)
        else:
            _SIM_PARAMS_CACHE = SimParams()
    return _SIM_PARAMS_CACHE


def clip_speed(v: np.ndarray, cap: float = VEL_CLIP) -> np.ndarray:
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.where(speed > cap, cap / np.maximum(speed, 1e-12), 1.0)
    return v * scale


def route_velocity(pos: np.ndarray, dt: float = DT) -> np.ndarray:
    """Deterministic receiver-route velocity from stored positions [T, 2]
    (§1.1): five-point second-order local-polynomial (Savitzky-Golay)
    derivative where five points are available, central differences
    otherwise in the interior, one-sided differences at the endpoints,
    magnitude clipped to 10 yd/s."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    v = np.zeros_like(pos)
    if n == 1:
        return v
    v[0] = (pos[1] - pos[0]) / dt
    v[-1] = (pos[-1] - pos[-2]) / dt
    if n >= 3:
        v[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
    if n >= 5:
        # SG(5,2): (2(p[k+2]-p[k-2]) + (p[k+1]-p[k-1])) / (10 dt)
        v[2:-2] = (2.0 * (pos[4:] - pos[:-4]) + (pos[3:-1] - pos[1:-3])) / (10.0 * dt)
    return clip_speed(v)


def path_velocity_central(pos: np.ndarray, dt: float = DT) -> np.ndarray:
    """Zone-path derivative (§1.2): central differences in the interior,
    one-sided at the endpoints, magnitude clipped.  Applied ONLY to the
    completed hysteresis-smoothed role path, never to a raw state jump."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    v = np.zeros_like(pos)
    if n == 1:
        return v
    v[0] = (pos[1] - pos[0]) / dt
    v[-1] = (pos[-1] - pos[-2]) / dt
    if n >= 3:
        v[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
    return clip_speed(v)


def inside_unit(r0_y: float, p0_y: float) -> tuple[float, float]:
    """Deterministic flip-compatible e_inside convention: (sign on e_y,
    valid).  e_inside = sign * e_y, or invalid when both the receiver and
    the defender sit on field center."""
    if abs(r0_y - Y_MID) > CENTER_EPS:
        return float(np.sign(Y_MID - r0_y)), 1.0
    if abs(p0_y - Y_MID) > CENTER_EPS:
        return float(np.sign(Y_MID - p0_y)), 1.0
    return 0.0, 0.0


def build_receiver_target(
    p0: np.ndarray,           # [2] legal boundary position
    route_xy: np.ndarray,     # [T, 2] legal receiver route from frame 0
    n_steps: int,             # N = H/dt (target has N+1 samples)
) -> dict:
    """Leverage-preserving moving corridor around receiver j (§1.1):

        q_j(t) = r_j(t) + depth_offset * e_offense + inside_offset * e_inside_j
        depth_offset  = clip((p0-r_j(0)) . e_offense,  -2, 6)
        inside_offset = clip((p0-r_j(0)) . e_inside_j, -3, 3)

    Target velocity = receiver route velocity (offsets constant in t),
    estimated on the FULL stored route then sliced to the horizon."""
    p0 = np.asarray(p0, dtype=float)
    route_xy = np.asarray(route_xy, dtype=float)
    assert route_xy.shape[0] >= n_steps + 1, "route shorter than horizon"
    r0 = route_xy[0]
    depth = float(np.clip(p0[0] - r0[0], *DEPTH_CLIP))
    sgn, valid = inside_unit(float(r0[1]), float(p0[1]))
    inside = float(np.clip((p0[1] - r0[1]) * sgn, *INSIDE_CLIP)) if valid else 0.0

    q = route_xy[: n_steps + 1].copy()
    q[:, 0] += depth
    q[:, 1] += inside * sgn
    qdot = route_velocity(route_xy)[: n_steps + 1]
    return {
        "q": q,
        "qdot": qdot,
        "meta": {
            "depth_offset": depth,
            "inside_offset": inside,
            "inside_sign": sgn,
            "inside_valid": bool(valid),
        },
    }


def make_route_context(
    legal_routes: list[dict],   # [{"slot", "nfl_id", "position", "xy": [T,2]}]
    los_x: float,
    n_frames: int,
    params: SimParams | None = None,
) -> dict:
    """Build the frozen-simulator route objects (side/order/tags via
    `build_receiver_routes`) and route-conditioned role landmarks/lanes from
    legal inputs ONLY.  Deterministic route ordering: slot index, then
    nfl_id."""
    params = params or frozen_sim_params()
    ordered = sorted(legal_routes,
                     key=lambda r: (int(r["slot"]), str(r["nfl_id"])))
    n_off = len(ordered)
    offense_data = np.zeros((max(n_off, 1), n_frames, 2))
    for i, r in enumerate(ordered):
        xy = np.asarray(r["xy"], dtype=float)
        k = min(n_frames, xy.shape[0])
        offense_data[i, :k] = xy[:k]
        if k < n_frames:
            offense_data[i, k:] = xy[k - 1]
    play_dict = {
        "offense_data": offense_data,
        "offense_mask": np.ones((max(n_off, 1), n_frames), dtype=bool),
        "offense_player_ids": [str(r["nfl_id"]) for r in ordered] or ["none"],
        "offense_positions": [str(r.get("position", "UNKNOWN")) for r in ordered]
        or ["UNKNOWN"],
        "n_off": n_off,
        "n_frames": n_frames,
        "los_x": float(los_x),
    }
    routes = build_receiver_routes(play_dict, params=params)
    landmarks = _role_landmarks(play_dict, routes, params)

    # deep quarter lane bounds from the route-conditioned LANDMARK centers
    # (no realized defender snap enters the hypothetical universe): the four
    # deep hypothetical roles are anchored at their landmarks and the frozen
    # midpoint-split logic runs on those centers
    deep_stubs = [
        _hypothetical_role(name, np.array(landmarks[name]), landmarks)
        for name in DEEP_ROLE_ORDER
    ]
    _assign_deep_lane_bounds(deep_stubs, params, routes)
    lanes = {r.role_name: (r.lane_y_min, r.lane_y_max) for r in deep_stubs}
    return {
        "routes": routes,
        "routes_by_side": _routes_by_side(routes),
        "routes_by_id": {route.nfl_id: route for route in routes},
        "landmarks": landmarks,
        "lanes": lanes,
        "los_x": float(los_x),
        "params": params,
    }


def _hypothetical_role(role_name: str, snap_xy: np.ndarray,
                       landmarks: dict) -> DefenderRole:
    lx, ly = landmarks.get(role_name, (math.nan, math.nan))
    return DefenderRole(
        nfl_id="hypothetical",
        defender_index=-1,
        role_name=role_name,
        role_idx=ROLE_NAME_TO_IDX[role_name],
        position="DB",
        snap_x=float(snap_xy[0]),
        snap_y=float(snap_xy[1]),
        # hypothetical option trust is ONE: an infinite assignment margin
        # drives assignment_trust_for_role to its maximum; no Hungarian
        # confidence is inherited (verified by w42_c0_tests)
        assignment_confidence=1.0,
        assignment_margin=1e9,
        landmark_x=float(lx),
        landmark_y=float(ly),
        missing_flat_support=False,
    )


def build_role_target(
    role_index: int,
    p0: np.ndarray,            # [2] legal boundary position
    ctx: dict,                 # from make_route_context (legal inputs only)
    n_steps: int,
) -> dict:
    """Hypothetical role option (§1.2): what THIS defender could initiate
    if occupying role ``role_index`` — via the frozen rule-simulator
    responsibility logic, route ordering, dwell/hysteresis, and
    route-conditioned target generation.

    The hypothetical role is anchored at ``p0`` (snap := p0), starts in the
    explicit 'initial' state with age zero and no threats, and its target
    EMA is seeded at ``p0`` so the completed path starts at the defender
    (spec item 4) and never differentiates a raw responsibility-state jump.
    """
    p0 = np.asarray(p0, dtype=float)
    params: SimParams = ctx["params"]
    role_name = ROLE_ORDER[role_index]
    role = _hypothetical_role(role_name, p0, ctx["landmarks"])
    lane = ctx["lanes"].get(role_name)
    if lane is not None:
        role.lane_y_min, role.lane_y_max = lane

    trust = assignment_trust_for_role(role, params)
    routes_by_side = ctx["routes_by_side"]
    routes_by_id = ctx["routes_by_id"]
    los_x = ctx["los_x"]
    alpha = float(np.clip(params.target_smoothing_alpha, 0.0, 1.0))

    state = DefenderSimState(
        x=float(p0[0]), y=float(p0[1]), vx=0.0, vy=0.0,
        state_name="initial", state_age=0,
        primary_threat_id=None, secondary_threat_id=None)
    q = np.zeros((n_steps + 1, 2))
    q[0] = p0
    prev_target = (float(p0[0]), float(p0[1]))
    states = ["initial"]
    for t in range(1, n_steps + 1):
        decision = choose_responsibility_state(
            role=role, t=t, routes_by_side=routes_by_side,
            prev_state=state, params=params)
        raw_target = raw_target_for_decision(
            role, decision, routes_by_id, t, los_x, params)
        deep_cluster_y = _deep_cluster_target_y(
            role, decision, routes_by_id.values(), params)
        deep_cluster_active = deep_cluster_y is not None
        if deep_cluster_active:
            raw_target = (raw_target[0], float(deep_cluster_y))
        zone_anchor = movement_anchor_for_decision(role, decision, los_x, params)
        if decision.state_name in {"zone_quarter", "curl_flat_zone", "middle_hook_zone"}:
            if role_name in {"deep_right", "deep_center_right",
                             "deep_center_left", "deep_left"}:
                zone_blend = float(np.clip(params.deep_zone_state_anchor_blend, 0.0, 1.0))
            else:
                zone_blend = (float(np.clip(params.zone_state_anchor_blend, 0.0, 1.0))
                              * trust)
            blended = _blend_target((role.snap_x, role.snap_y), zone_anchor, zone_blend)
            blended = _clip_role_xy(role, blended[0], blended[1])
        else:
            blend = target_blend_for_decision(
                role, decision, params, zone_anchor, raw_target,
                deep_cluster_active)
            blended = _blend_target(zone_anchor, raw_target, blend)
            blended = _protect_deep_over_top_x(
                role, zone_anchor, raw_target, blended, params)
        smoothed = (alpha * blended[0] + (1.0 - alpha) * prev_target[0],
                    alpha * blended[1] + (1.0 - alpha) * prev_target[1])
        smoothed = _clip_field_xy(*smoothed)
        q[t] = smoothed
        prev_target = smoothed
        # frozen dwell/hysteresis bookkeeping (state age resets on switch)
        state = DefenderSimState(
            x=state.x, y=state.y, vx=state.vx, vy=state.vy,
            state_name=decision.state_name,
            state_age=(state.state_age + 1
                       if decision.state_name == state.state_name else 1),
            primary_threat_id=decision.primary_threat_id,
            secondary_threat_id=decision.secondary_threat_id)
        states.append(decision.state_name)

    qdot = path_velocity_central(q)
    return {
        "q": q,
        "qdot": qdot,
        "meta": {
            "role_name": role_name,
            "trust": float(trust),
            "landmark_x": role.landmark_x,
            "landmark_y": role.landmark_y,
            "lane_y_min": role.lane_y_min,
            "lane_y_max": role.lane_y_max,
            "initial_state": "initial",
            "final_state": states[-1],
            "n_state_switches": int(sum(1 for a, b in zip(states, states[1:])
                                        if a != b)),
        },
    }


N_MAX = 10


def pad_path(q: np.ndarray) -> np.ndarray:
    out = np.zeros((N_MAX + 1, 2), dtype=np.float32)
    n = q.shape[0]
    out[:n] = q
    if n <= N_MAX:
        out[n:] = q[-1]
    return out
