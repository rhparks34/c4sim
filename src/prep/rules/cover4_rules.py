"""The coded Cover-4 rules: the seven roles (four deep quarters, two flats, the hook), their landmarks, and how a defender in a
role chooses his responsibility from the routes frame by frame (state, dwell and switching), with the target and anchor that
responsibility implies. Verbatim from the research code (the deterministic Cover-4 simulator)."""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.prep.rules.movement import (
    carry_crosser_target as v2_carry_crosser_target,
    hook_wall_target as v2_hook_wall_target,
    middle_flat_target as v2_middle_flat_target,
    sideline_flat_target as v2_sideline_flat_target,
    sink_under_vertical_target as v2_sink_under_vertical_target,
    under_mirror_target as v2_under_mirror_target,
)

FIELD_WIDTH = 120.0


FIELD_HEIGHT = 53.3


FIELD_MID_Y = FIELD_HEIGHT / 2.0


RIGHT_SIDE = "right"  # low y after normalization


LEFT_SIDE = "left"    # high y after normalization


MIDDLE_SIDE = "middle"


ROLE_ORDER = [
    "deep_right",
    "deep_center_right",
    "deep_center_left",
    "deep_left",
    "flat_right",
    "hook_curl",
    "flat_left",
]


ROLE_NAME_TO_IDX = {name: idx for idx, name in enumerate(ROLE_ORDER)}


DEEP_ROLES = {"deep_right", "deep_center_right", "deep_center_left", "deep_left"}


OUTSIDE_DEEP_ROLES = {"deep_right", "deep_left"}


INSIDE_DEEP_ROLES = {"deep_center_right", "deep_center_left"}


FLAT_ROLES = {"flat_right", "flat_left"}


UNDERNEATH_ROLES = {"flat_right", "hook_curl", "flat_left"}


DEEP_ROLE_ORDER = ["deep_right", "deep_center_right", "deep_center_left", "deep_left"]


DEEP_ROLE_FALLBACK_Y = {
    "deep_right": 5.0,
    "deep_center_right": 18.0,
    "deep_center_left": 35.0,
    "deep_left": 48.0,
}


def outside_y_sign(side: str) -> float:
    """Return the sideline direction in y for a normalized side."""
    return -1.0 if side == RIGHT_SIDE else 1.0


def inside_y_sign(side: str) -> float:
    """Return the inside-field direction in y for a normalized side."""
    return -outside_y_sign(side)


def normalize_id(value: Any) -> str:
    """Normalize IDs consistently across BDB/Hudl artifacts."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def role_side(role_name: str) -> str | None:
    if role_name.endswith("_right"):
        return RIGHT_SIDE
    if role_name.endswith("_left"):
        return LEFT_SIDE
    return None


def _clip_field_xy(x: float, y: float) -> tuple[float, float]:
    return (
        float(np.clip(x, 0.0, FIELD_WIDTH)),
        float(np.clip(y, 0.0, FIELD_HEIGHT)),
    )


def _finite_lane_bound(value: float, fallback: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return out if math.isfinite(out) else float(fallback)


def _clip_deep_lane_y(role: DefenderRole, y: float) -> float:
    """Clamp a deep role to its static quarter lane when lane metadata exists."""
    if role.role_name not in DEEP_ROLES:
        return float(np.clip(y, 0.0, FIELD_HEIGHT))
    lane_min = _finite_lane_bound(role.lane_y_min, 0.0)
    lane_max = _finite_lane_bound(role.lane_y_max, FIELD_HEIGHT)
    if lane_min > lane_max:
        lane_min, lane_max = lane_max, lane_min
    return float(np.clip(y, max(0.0, lane_min), min(FIELD_HEIGHT, lane_max)))


def _clip_role_xy(role: DefenderRole, x: float, y: float) -> tuple[float, float]:
    x, y = _clip_field_xy(x, y)
    if role.role_name in DEEP_ROLES:
        y = _clip_deep_lane_y(role, y)
    return x, y


def _finite_or(default: float, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


@dataclass
class ReceiverRoute:
    nfl_id: str
    slot_index: int
    side: str
    order: int | None
    position: str
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    speed: np.ndarray
    mask: np.ndarray
    tags: dict[str, Any]


@dataclass
class DefenderRole:
    nfl_id: str
    defender_index: int
    role_name: str
    role_idx: int
    position: str
    snap_x: float
    snap_y: float
    assignment_confidence: float
    assignment_margin: float = math.nan
    assignment_cost: float = math.nan
    assignment_second_best_cost: float = math.nan
    assignment_cost_model_version: str = "v1"
    landmark_x: float = math.nan
    landmark_y: float = math.nan
    lane_y_min: float = math.nan
    lane_y_max: float = math.nan
    missing_flat_support: bool = False


@dataclass
class DefenderSimState:
    x: float
    y: float
    vx: float
    vy: float
    state_name: str
    state_age: int
    primary_threat_id: str | None
    secondary_threat_id: str | None


@dataclass
class SimFrameDecision:
    role_name: str
    state_name: str
    primary_threat_id: str | None
    secondary_threat_id: str | None
    target_x: float
    target_y: float
    target_weight: float


@dataclass
class SimParams:
    dt: float = 0.1
    reaction_delay_frames: int = 2
    min_state_dwell_frames: int = 3
    switch_margin: float = 0.15
    default_role_overrides_csv: str = ""
    movement_version: str = "v1"
    enable_v2_middle_flat: bool = False
    enable_v2_sink_under: bool = False
    enable_v2_carry_crosser: bool = False
    enable_v2_hook_wall: bool = False
    enable_v2_under_mirror: bool = False
    enable_v2_carry_mirror: bool = False
    enable_v2_carry_vertical_mirror: bool = False
    enable_v2_midpoint_mirror: bool = False
    enable_v2_wall_mirror: bool = False
    enable_v2_zone_react: bool = False
    # Stage-D coordinated transition policy.  The flag defaults off so every
    # historical params artifact follows the byte-identical legacy path.
    enable_transition_policy: bool = False
    transition_policy_path: str = ""
    transition_beam_width: int = 64
    transition_min_dwell_frames: int = 4
    transition_switch_margin: float = 0.0
    # Causal v2 temporal finite-state rules.  These are deliberately separate
    # from the v1 distilled-tree flag/artifact so both implementations remain
    # reproducible and all new behavior defaults off.
    enable_transition_policy_v2: bool = False
    transition_rules_v2_path: str = ""
    # native | relation_online | relation_physics.  The relation surfaces
    # preserve all seven relation states for the compiler/movement bridge.
    transition_movement_surface: str = "native"
    transition_relation_params_path: str = ""
    transition_relation_subtype_map_path: str = ""
    transition_relation_zone_predictions_path: str = ""
    transition_assignment_replay_path: str = ""
    transition_replay_bypass_resolver: bool = False

    max_speed_db: float = 8.5
    max_speed_lb: float = 7.5
    max_accel_db: float = 7.0
    max_accel_lb: float = 6.0
    tau_db: float = 0.35
    tau_lb: float = 0.45
    max_turn_rate_deg: float = 420.0
    target_smoothing_alpha: float = 0.35
    deep_target_blend: float = 0.60
    inside_deep_target_blend: float = 0.50
    flat_target_blend: float = 0.25
    flat_target_trust_floor: float = 0.20
    flat_middle_target_blend: float = 0.85
    flat_middle_target_trust_floor: float = 1.00
    flat_middle_projection_frames: int = 8
    flat_sideline_target_blend: float = 0.95
    flat_sideline_target_trust_floor: float = 1.00
    flat_sideline_outside_leverage_y: float = 0.75
    flat_sideline_min_outside_y: float = 0.0
    flat_sideline_current_outside_y: float = 0.0
    flat_sideline_max_depth_extra: float = 3.0
    underneath_target_blend: float = 0.25
    deep_lateral_target_min_blend: float = 0.0
    deep_lateral_target_min_delta_y: float = 2.0
    deep_cluster_lateral_min_y: float = 3.0
    deep_cluster_target_blend_floor: float = 0.25
    missing_flat_deep_target_blend: float = 0.80
    missing_flat_deep_target_trust_floor: float = 1.00
    missing_flat_deep_projection_frames: int = 8
    missing_flat_deep_snap_advance: float = 4.0
    missing_flat_deep_outside_leverage_y: float = 0.75

    vertical_min_depth: float = 8.0
    vertical_min_vx: float = 1.8
    vertical_min_total_depth: float = 10.0
    flat_max_depth: float = 6.0
    flat_min_lateral: float = 4.0
    crosser_min_lateral: float = 8.0
    curl_decel_threshold: float = -2.0

    deep_zone_depth: float = 14.0
    inside_deep_zone_depth: float = 12.0
    flat_zone_depth: float = 5.0
    hook_zone_depth: float = 8.0
    deep_anchor_snap_advance: float = 5.0
    flat_anchor_snap_advance: float = 3.0
    hook_anchor_snap_advance: float = 2.0
    hook_wall_two_snap_y_window: float = 4.0
    inside_deep_anchor_midfield_blend: float = 0.55
    flat_anchor_snap_y_blend: float = 0.35
    hook_anchor_midfield_blend: float = 0.25
    assignment_trust_min: float = 0.25
    assignment_trust_margin_offset: float = 1.0
    assignment_trust_margin_scale: float = 3.0
    assignment_version: str = "v1"
    assignment_y_weight: float = 0.95
    assignment_depth_weight: float = 0.45
    assignment_side_mismatch_penalty: float = 3.5
    assignment_deep_lb_penalty: float = 3.0
    assignment_deep_other_penalty: float = 1.5
    assignment_hook_db_penalty: float = 0.8
    assignment_hook_other_penalty: float = 1.5
    assignment_underneath_other_penalty: float = 1.5
    low_trust_anchor_snap_blend: float = 0.75
    zone_state_anchor_blend: float = 0.20
    deep_zone_state_anchor_blend: float = 0.85
    deep_target_min_x_blend: float = 1.00
    deep_threat_projection_frames: int = 8
    deep_lane_half_width_min: float = 0.0
    deep_lane_threat_expand_y: float = 2.0
    deep_lane_threat_margin_y: float = 1.0
    deep_empty_sideline_lane_margin_y: float = 1.0
    output_motion_scale_start: float = 0.20
    output_motion_scale_end: float = 1.05
    output_motion_scale_power: float = 2.0
    filter_blitz_candidates: bool = True
    blitz_cross_los_margin: float = 0.5

    deep_over_top_cushion: float = 4.0
    deep_inside_leverage_y: float = 1.0
    flat_inside_leverage_y: float = 1.5
    flat_depth_cushion: float = 0.5
    sink_under_depth: float = 10.0
    sink_under_receiver_cushion_x: float = 2.0
    sink_under_route_y_blend: float = 1.0
    sink_under_max_lateral_from_snap_y: float = 53.3
    # Dictionary under-help geometry (defaults = Stage-1 fitted help_under)
    under_mirror_vertical_gain: float = 0.49
    under_mirror_lateral_gain: float = 0.93
    under_mirror_cushion_x: float = 1.6
    under_mirror_min_depth: float = 1.0
    # Same geometry applied to carry_crosser (trail technique on crossers;
    # trigger-coverage extension, COVER4_GAP_PLAN.md 0.0.5.7 step 2)
    carry_mirror_vertical_gain: float = 0.49
    carry_mirror_lateral_gain: float = 0.93
    carry_mirror_cushion_x: float = 1.6
    carry_mirror_min_depth: float = 1.0
    # In-phase carry of verticals (carry_1/carry_2; mass map says these
    # defenders mostly play level/under the stem, not over the top)
    carry_vert_mirror_vertical_gain: float = 0.49
    carry_vert_mirror_lateral_gain: float = 0.93
    carry_vert_mirror_cushion_x: float = 1.6
    carry_vert_mirror_min_depth: float = 1.0
    # Two-threat midpoint played as a mirror of the midpoint displacement
    # (midpoint_1_2), replacing the over-top midpoint cushion
    midpoint_mirror_vertical_gain: float = 0.49
    midpoint_mirror_lateral_gain: float = 0.93
    midpoint_mirror_cushion_x: float = 1.6
    midpoint_mirror_min_depth: float = 1.0
    # Wall states retried as under-mirror (hook_wall's wall-at-a-landmark
    # geometry was rejected; the dictionary walls track the threat laterally
    # at level depth, downhill allowed)
    wall_mirror_vertical_gain: float = 0.49
    wall_mirror_lateral_gain: float = 0.93
    wall_mirror_cushion_x: float = 1.6
    wall_mirror_min_depth: float = 1.0
    # Relaxed carry trigger for deep roles stuck in the zone_quarter
    # fallback: a same-side stem past this depth (below the vertical-tag
    # bar) routes into the promoted carry_1/carry_2 mirror geometry
    zone_react_min_stem: float = 6.0
    zone_react_min_vx: float = 0.5
    flat_middle_depth_cushion: float = 0.5
    flat_middle_snap_advance: float = 3.0
    flat_middle_route_y_blend: float = 1.0
    hook_wall_depth: float = 12.0
    hook_wall_depth_cushion: float = 1.0
    hook_wall_route_y_blend: float = 1.0
    hook_crosser_depth: float = 9.0
    hook_crosser_depth_cushion: float = 0.5
    hook_crosser_route_y_blend: float = 0.70
    flat_crosser_depth_cushion: float = 0.5
    flat_crosser_inside_leverage_y: float = 0.75
    midpoint_over_top_cushion: float = 3.5

    VECTOR_FIELDS = (
        "reaction_delay_frames",
        "min_state_dwell_frames",
        "max_speed_db",
        "max_speed_lb",
        "max_accel_db",
        "max_accel_lb",
        "tau_db",
        "tau_lb",
        "target_smoothing_alpha",
        "deep_target_blend",
        "inside_deep_target_blend",
        "flat_target_blend",
        "flat_target_trust_floor",
        "flat_middle_target_blend",
        "flat_middle_target_trust_floor",
        "flat_middle_projection_frames",
        "underneath_target_blend",
        "vertical_min_depth",
        "vertical_min_vx",
        "flat_max_depth",
        "flat_min_lateral",
        "deep_zone_depth",
        "inside_deep_zone_depth",
        "flat_zone_depth",
        "hook_zone_depth",
        "deep_anchor_snap_advance",
        "flat_anchor_snap_advance",
        "hook_anchor_snap_advance",
        "inside_deep_anchor_midfield_blend",
        "flat_anchor_snap_y_blend",
        "hook_anchor_midfield_blend",
        "assignment_trust_min",
        "assignment_trust_margin_offset",
        "assignment_trust_margin_scale",
        "assignment_y_weight",
        "assignment_depth_weight",
        "assignment_side_mismatch_penalty",
        "assignment_deep_lb_penalty",
        "assignment_deep_other_penalty",
        "assignment_hook_db_penalty",
        "assignment_hook_other_penalty",
        "assignment_underneath_other_penalty",
        "low_trust_anchor_snap_blend",
        "zone_state_anchor_blend",
        "deep_zone_state_anchor_blend",
        "deep_target_min_x_blend",
        "deep_threat_projection_frames",
        "deep_lane_half_width_min",
        "output_motion_scale_start",
        "output_motion_scale_end",
        "output_motion_scale_power",
        "deep_over_top_cushion",
        "deep_inside_leverage_y",
        "flat_inside_leverage_y",
        "sink_under_depth",
        "midpoint_over_top_cushion",
    )

    INT_FIELDS = {
        "reaction_delay_frames",
        "min_state_dwell_frames",
        "deep_threat_projection_frames",
        "flat_middle_projection_frames",
        "transition_beam_width",
        "transition_min_dwell_frames",
    }

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "SimParams":
        with open(path, "r") as f:
            raw = json.load(f)
        valid_names = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in raw.items() if key in valid_names}
        return cls(**kwargs)

    def to_json(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
            f.write("\n")

    def to_vector(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in self.VECTOR_FIELDS], dtype=float)

    @classmethod
    def from_vector(cls, vector: Sequence[float], base: "SimParams | None" = None) -> "SimParams":
        params = asdict(base if base is not None else cls())
        for name, value in zip(cls.VECTOR_FIELDS, vector):
            params[name] = int(round(value)) if name in cls.INT_FIELDS else float(value)
        return cls(**params)

    @classmethod
    def bounds(cls) -> dict[str, tuple[float, float]]:
        return {
            "reaction_delay_frames": (0, 5),
            "min_state_dwell_frames": (1, 6),
            "max_speed_db": (7.0, 9.5),
            "max_speed_lb": (6.0, 8.5),
            "max_accel_db": (4.0, 9.0),
            "max_accel_lb": (3.5, 8.0),
            "tau_db": (0.20, 0.70),
            "tau_lb": (0.25, 0.85),
            "target_smoothing_alpha": (0.10, 0.80),
            "deep_target_blend": (0.0, 1.0),
            "inside_deep_target_blend": (0.0, 1.0),
            "flat_target_blend": (0.0, 1.0),
            "flat_target_trust_floor": (0.0, 1.0),
            "flat_middle_target_blend": (0.0, 1.0),
            "flat_middle_target_trust_floor": (0.0, 1.0),
            "flat_middle_projection_frames": (0, 12),
            "underneath_target_blend": (0.0, 1.0),
            "vertical_min_depth": (5.0, 11.0),
            "vertical_min_vx": (0.8, 3.0),
            "flat_max_depth": (3.0, 8.0),
            "flat_min_lateral": (2.0, 8.0),
            "deep_zone_depth": (10.0, 18.0),
            "inside_deep_zone_depth": (9.0, 16.0),
            "flat_zone_depth": (2.0, 8.0),
            "hook_zone_depth": (4.0, 12.0),
            "deep_anchor_snap_advance": (0.0, 10.0),
            "flat_anchor_snap_advance": (-1.0, 5.0),
            "hook_anchor_snap_advance": (-1.0, 4.0),
            "inside_deep_anchor_midfield_blend": (0.0, 1.0),
            "flat_anchor_snap_y_blend": (0.0, 1.0),
            "hook_anchor_midfield_blend": (0.0, 1.0),
            "assignment_trust_min": (0.0, 1.0),
            "assignment_trust_margin_offset": (-2.0, 4.0),
            "assignment_trust_margin_scale": (0.5, 8.0),
            "assignment_y_weight": (0.1, 2.0),
            "assignment_depth_weight": (0.0, 1.5),
            "assignment_side_mismatch_penalty": (0.0, 8.0),
            "assignment_deep_lb_penalty": (0.0, 8.0),
            "assignment_deep_other_penalty": (0.0, 8.0),
            "assignment_hook_db_penalty": (0.0, 4.0),
            "assignment_hook_other_penalty": (0.0, 8.0),
            "assignment_underneath_other_penalty": (0.0, 8.0),
            "low_trust_anchor_snap_blend": (0.0, 1.0),
            "zone_state_anchor_blend": (0.0, 1.0),
            "deep_zone_state_anchor_blend": (0.0, 1.0),
            "deep_target_min_x_blend": (0.0, 1.0),
            "deep_threat_projection_frames": (0, 8),
            "deep_lane_half_width_min": (0.0, 8.0),
            "output_motion_scale_start": (0.0, 1.25),
            "output_motion_scale_end": (0.0, 1.50),
            "output_motion_scale_power": (0.5, 3.0),
            "deep_over_top_cushion": (0.0, 5.0),
            "deep_inside_leverage_y": (-2.0, 3.0),
            "flat_inside_leverage_y": (-1.0, 4.0),
            "sink_under_depth": (6.0, 14.0),
            "midpoint_over_top_cushion": (0.0, 4.0),
        }


def _role_depth(role_name: str, params: SimParams) -> float:
    if role_name in OUTSIDE_DEEP_ROLES:
        return params.deep_zone_depth
    if role_name in INSIDE_DEEP_ROLES:
        return params.inside_deep_zone_depth
    if role_name in FLAT_ROLES:
        return params.flat_zone_depth
    return params.hook_zone_depth


def _route_valid_index(route: ReceiverRoute, t: int) -> int:
    if len(route.x) == 0:
        return 0
    t = int(np.clip(t, 0, len(route.x) - 1))
    if route.mask[t]:
        return t
    valid = np.flatnonzero(route.mask)
    if len(valid) == 0:
        return t
    return int(valid[np.argmin(np.abs(valid - t))])


def _route_xy(route: ReceiverRoute, t: int) -> tuple[float, float]:
    idx = _route_valid_index(route, t)
    return float(route.x[idx]), float(route.y[idx])


def _route_projected_x(route: ReceiverRoute, t: int, projection_frames: int) -> float:
    idx = _route_valid_index(route, int(t) + max(0, int(projection_frames)))
    return float(route.x[idx])


def _tag_frame(route: ReceiverRoute, name: str, t: int) -> bool:
    value = route.tags.get(f"{name}_frame")
    if isinstance(value, np.ndarray):
        if len(value) == 0:
            return False
        return bool(value[min(max(int(t), 0), len(value) - 1)])
    return bool(route.tags.get(name, False))


def _route_rule_side(route: ReceiverRoute) -> str:
    if route.side in (RIGHT_SIDE, LEFT_SIDE):
        return route.side
    side = route.tags.get("release_side") or route.side
    if side in (RIGHT_SIDE, LEFT_SIDE):
        return side
    if len(route.y) >= 2:
        return RIGHT_SIDE if route.y[-1] < FIELD_MID_Y else LEFT_SIDE
    return RIGHT_SIDE if (route.y[0] if len(route.y) else FIELD_MID_Y) < FIELD_MID_Y else LEFT_SIDE


def _routes_by_side(routes: Sequence[ReceiverRoute]) -> dict[str, list[ReceiverRoute]]:
    out: dict[str, list[ReceiverRoute]] = {RIGHT_SIDE: [], LEFT_SIDE: [], MIDDLE_SIDE: [], "all": []}
    for route in routes:
        out["all"].append(route)
        if route.side == MIDDLE_SIDE:
            out[MIDDLE_SIDE].append(route)
        side = _route_rule_side(route)
        if side in (RIGHT_SIDE, LEFT_SIDE):
            out[side].append(route)

    for side in (RIGHT_SIDE, LEFT_SIDE):
        reverse = side == LEFT_SIDE
        out[side].sort(
            key=lambda r: (
                99 if r.order is None else r.order,
                -float(r.y[0]) if reverse and len(r.y) else float(r.y[0]) if len(r.y) else FIELD_MID_Y,
            )
        )
    return out


def _find_order(routes: Sequence[ReceiverRoute], order: int) -> ReceiverRoute | None:
    for route in routes:
        if route.order == order:
            return route
    return None


def _route_y_bounds(route: ReceiverRoute, margin: float = 0.0) -> tuple[float, float] | None:
    y = np.asarray(route.y, dtype=float)
    mask = np.asarray(route.mask, dtype=bool)
    if len(y) == 0:
        return None
    valid = mask & np.isfinite(y)
    if not np.any(valid):
        return None
    lo = float(np.nanmin(y[valid])) - float(margin)
    hi = float(np.nanmax(y[valid])) + float(margin)
    return float(np.clip(lo, 0.0, FIELD_HEIGHT)), float(np.clip(hi, 0.0, FIELD_HEIGHT))


def _expand_lane_toward_bounds(
    lane_min: float,
    lane_max: float,
    threat_bounds: tuple[float, float] | None,
    max_expand: float,
) -> tuple[float, float]:
    if threat_bounds is None:
        return lane_min, lane_max
    threat_min, threat_max = threat_bounds
    max_expand = max(0.0, float(max_expand))
    if threat_min < lane_min:
        lane_min = max(float(threat_min), float(lane_min) - max_expand)
    if threat_max > lane_max:
        lane_max = min(float(threat_max), float(lane_max) + max_expand)
    return lane_min, lane_max


def _route_deep_lane_threat(route: ReceiverRoute | None) -> bool:
    if route is None:
        return False
    return bool(
        route.tags.get("vertical")
        or route.tags.get("seam")
        or route.tags.get("corner")
        or route.tags.get("post")
    )


def _route_inside_y_expansion_bounds(
    route: ReceiverRoute,
    side: str,
    margin: float,
) -> tuple[float, float] | None:
    bounds = _route_y_bounds(route, margin)
    if bounds is None:
        return None
    if side == RIGHT_SIDE:
        return FIELD_HEIGHT, bounds[1]
    if side == LEFT_SIDE:
        return bounds[0], 0.0
    return bounds


def _route_first_last_y(route: ReceiverRoute) -> tuple[float, float] | None:
    valid = np.flatnonzero(np.asarray(route.mask, dtype=bool))
    if len(valid) == 0:
        return None
    return float(route.y[int(valid[0])]), float(route.y[int(valid[-1])])


def _deep_directional_cluster_routes(
    routes: Iterable[ReceiverRoute],
    params: SimParams,
) -> tuple[str | None, list[tuple[ReceiverRoute, float]]] | None:
    min_lateral = max(0.0, float(params.deep_cluster_lateral_min_y))
    positive: list[tuple[ReceiverRoute, float]] = []
    negative: list[tuple[ReceiverRoute, float]] = []
    for route in routes:
        if not _route_deep_lane_threat(route):
            continue
        endpoints = _route_first_last_y(route)
        if endpoints is None:
            continue
        first_y, last_y = endpoints
        dy = last_y - first_y
        if abs(dy) < min_lateral:
            continue
        bucket = positive if dy > 0.0 else negative
        bucket.append((route, last_y))

    if len(positive) < 2 and len(negative) < 2:
        return None
    chosen = positive
    if len(negative) > len(positive):
        chosen = negative
    elif len(negative) == len(positive) and len(negative) >= 2:
        pos_mag = sum(abs(y - _route_first_last_y(route)[0]) for route, y in positive)  # type: ignore[index]
        neg_mag = sum(abs(y - _route_first_last_y(route)[0]) for route, y in negative)  # type: ignore[index]
        if neg_mag > pos_mag:
            chosen = negative

    side_counts = {
        RIGHT_SIDE: sum(1 for route, _ in chosen if _route_rule_side(route) == RIGHT_SIDE),
        LEFT_SIDE: sum(1 for route, _ in chosen if _route_rule_side(route) == LEFT_SIDE),
    }
    cluster_side = None
    if max(side_counts.values()) >= 2:
        cluster_side = RIGHT_SIDE if side_counts[RIGHT_SIDE] >= side_counts[LEFT_SIDE] else LEFT_SIDE
    chosen.sort(key=lambda item: item[1])
    return cluster_side, chosen


def _deep_cluster_target_y(
    role: DefenderRole,
    decision: SimFrameDecision,
    routes: Iterable[ReceiverRoute],
    params: SimParams,
) -> float | None:
    if role.role_name not in DEEP_ROLES:
        return None
    if decision.state_name not in {
        "carry_1",
        "carry_2",
        "squeeze_2",
        "inside_help_1",
        "poach_2",
        "poach_3",
        "midpoint_1_2",
    }:
        return None
    cluster = _deep_directional_cluster_routes(routes, params)
    if cluster is None:
        return None
    cluster_side, chosen = cluster
    cluster_ids = {route.nfl_id for route, _ in chosen}
    if decision.primary_threat_id not in cluster_ids and decision.secondary_threat_id not in cluster_ids:
        return None

    low_y = float(chosen[0][1])
    high_y = float(chosen[-1][1])
    mid_y = 0.5 * (low_y + high_y)
    if cluster_side == RIGHT_SIDE:
        role_targets = {
            "deep_right": low_y,
            "deep_center_right": mid_y,
            "deep_center_left": high_y,
            "deep_left": high_y,
        }
    elif cluster_side == LEFT_SIDE:
        role_targets = {
            "deep_right": low_y,
            "deep_center_right": low_y,
            "deep_center_left": mid_y,
            "deep_left": high_y,
        }
    else:
        role_targets = {
            "deep_right": low_y,
            "deep_center_right": mid_y,
            "deep_center_left": mid_y,
            "deep_left": high_y,
        }
    return role_targets.get(role.role_name)


def _role_landmarks(
    play: Mapping[str, Any],
    routes: Sequence[ReceiverRoute],
    params: SimParams,
) -> dict[str, tuple[float, float]]:
    los_x = _finite_or(0.0, play.get("los_x", 0.0))
    by_side = _routes_by_side(routes)
    split_y = FIELD_MID_Y
    if routes:
        split_y = _finite_or(FIELD_MID_Y, routes[0].tags.get("formation_split_y", FIELD_MID_Y))

    def snap_y(route: ReceiverRoute | None) -> float | None:
        if route is None or len(route.y) == 0:
            return None
        return float(route.y[_route_valid_index(route, 0)])

    def inside_deep_y(
        side: str,
        one: ReceiverRoute | None,
        two: ReceiverRoute | None,
        fallback: float,
    ) -> float:
        """Landmark for an inside-deep role.

        If #2 exists, midpoint #1/#2. If only #1 exists, place the inside-deep
        defender several yards inside #1, capped near the formation split. This
        avoids inverted deep-role landmarks on 3x1 and nub-side formations.
        """
        ya = snap_y(one)
        yb = snap_y(two)
        if ya is not None and yb is not None:
            return (ya + yb) / 2.0
        if yb is not None:
            return yb
        if ya is not None:
            candidate = ya + inside_y_sign(side) * 6.0
            if side == RIGHT_SIDE:
                candidate = min(candidate, split_y - 1.0)
            else:
                candidate = max(candidate, split_y + 1.0)
            return float(np.clip(candidate, 0.0, FIELD_HEIGHT))
        return fallback

    r1 = _find_order(by_side[RIGHT_SIDE], 1)
    r2 = _find_order(by_side[RIGHT_SIDE], 2)
    r3 = _find_order(by_side[RIGHT_SIDE], 3)
    l1 = _find_order(by_side[LEFT_SIDE], 1)
    l2 = _find_order(by_side[LEFT_SIDE], 2)
    l3 = _find_order(by_side[LEFT_SIDE], 3)

    r_flat = r2 or r3
    l_flat = l2 or l3
    for route in by_side[RIGHT_SIDE]:
        if route.tags.get("back_fast_flat"):
            r_flat = route
            break
    for route in by_side[LEFT_SIDE]:
        if route.tags.get("back_fast_flat"):
            l_flat = route
            break

    role_y = {
        "deep_right": (snap_y(r1) + outside_y_sign(RIGHT_SIDE) * 1.0) if snap_y(r1) is not None else 5.0,
        "deep_center_right": inside_deep_y(RIGHT_SIDE, r1, r2, 18.0),
        "deep_center_left": inside_deep_y(LEFT_SIDE, l1, l2, 35.0),
        "deep_left": (snap_y(l1) + outside_y_sign(LEFT_SIDE) * 1.0) if snap_y(l1) is not None else 48.0,
        "flat_right": snap_y(r_flat) if snap_y(r_flat) is not None else 10.0,
        "hook_curl": FIELD_MID_Y,
        "flat_left": snap_y(l_flat) if snap_y(l_flat) is not None else 43.0,
    }

    landmarks = {}
    for role_name in ROLE_ORDER:
        x, y = _clip_field_xy(los_x + _role_depth(role_name, params), role_y[role_name])
        landmarks[role_name] = (x, y)
    return landmarks


def _assign_deep_lane_bounds(
    roles: Sequence[DefenderRole],
    params: SimParams,
    routes: Sequence[ReceiverRoute] | None = None,
) -> None:
    """Set static lateral quarter-lane bounds for assigned deep roles.

    The bounds use the ordered deep defenders' snap centers, then split adjacent
    centers at their midpoints. This keeps each quarter defender in his assigned
    lane even when the receiver he is carrying crosses into another quarter.
    """
    role_by_name = {role.role_name: role for role in roles}
    if not any(name in role_by_name for name in DEEP_ROLE_ORDER):
        return

    centers = []
    for name in DEEP_ROLE_ORDER:
        role = role_by_name.get(name)
        if role is None:
            centers.append(DEEP_ROLE_FALLBACK_Y[name])
            continue
        center = _finite_or(math.nan, role.snap_y)
        if not math.isfinite(center):
            center = _finite_or(DEEP_ROLE_FALLBACK_Y[name], role.landmark_y)
        centers.append(float(np.clip(center, 0.0, FIELD_HEIGHT)))

    if any(centers[i] >= centers[i + 1] for i in range(len(centers) - 1)):
        centers = sorted(centers)
    for idx in range(1, len(centers)):
        if centers[idx] <= centers[idx - 1]:
            centers[idx] = min(FIELD_HEIGHT, centers[idx - 1] + 0.5)

    boundaries = [
        0.5 * (centers[0] + centers[1]),
        0.5 * (centers[1] + centers[2]),
        0.5 * (centers[2] + centers[3]),
    ]
    lane_bounds = {
        "deep_right": (0.0, boundaries[0]),
        "deep_center_right": (boundaries[0], boundaries[1]),
        "deep_center_left": (boundaries[1], boundaries[2]),
        "deep_left": (boundaries[2], FIELD_HEIGHT),
    }
    lane_expansions: dict[str, list[tuple[float, float]]] = {name: [] for name in DEEP_ROLE_ORDER}
    lane_min_limits: dict[str, float | None] = {name: None for name in DEEP_ROLE_ORDER}
    lane_max_limits: dict[str, float | None] = {name: None for name in DEEP_ROLE_ORDER}
    if routes:
        by_side = _routes_by_side(routes)
        margin = max(0.0, float(params.deep_lane_threat_margin_y))
        empty_sideline_margin = max(0.0, float(params.deep_empty_sideline_lane_margin_y))
        side_specs = {
            RIGHT_SIDE: ("deep_right", "deep_center_right", "deep_center_left"),
            LEFT_SIDE: ("deep_left", "deep_center_left", "deep_center_right"),
        }

        def limit_empty_sideline(role_name: str, side: str, threat_bounds: tuple[float, float] | None) -> None:
            role = role_by_name.get(role_name)
            if role is None or threat_bounds is None:
                return
            center = _finite_or(DEEP_ROLE_FALLBACK_Y[role_name], role.snap_y)
            if side == RIGHT_SIDE:
                floor = min(center, threat_bounds[0]) - empty_sideline_margin
                lane_min_limits[role_name] = max(lane_min_limits[role_name] or 0.0, floor)
            elif side == LEFT_SIDE:
                ceiling = max(center, threat_bounds[1]) + empty_sideline_margin
                current = lane_max_limits[role_name]
                lane_max_limits[role_name] = ceiling if current is None else min(current, ceiling)

        for side, (outside_role, inside_role, weak_inside_role) in side_specs.items():
            side_routes = by_side.get(side, [])
            one = _find_order(side_routes, 1)
            two = _find_order(side_routes, 2)
            three = _find_order(side_routes, 3)
            one_deep = _route_deep_lane_threat(one)
            two_deep = _route_deep_lane_threat(two)
            three_deep = _route_deep_lane_threat(three)

            # Outside quarters must be allowed to squeeze an inside vertical
            # when #1 is underneath. Otherwise strict static lanes pull the
            # defender away from the only real deep threat.
            if one_deep:
                bounds = _route_inside_y_expansion_bounds(one, side, margin)
                if bounds is not None:
                    lane_expansions[outside_role].append(bounds)
            if two_deep:
                bounds = _route_inside_y_expansion_bounds(two, side, margin)
                if bounds is not None:
                    lane_expansions[inside_role].append(bounds)
                    if not one_deep:
                        lane_expansions[outside_role].append(bounds)
                        route_bounds = _route_y_bounds(two, margin)
                        limit_empty_sideline(outside_role, side, route_bounds)
            if three_deep:
                bounds = _route_inside_y_expansion_bounds(three, side, margin)
                if bounds is not None:
                    lane_expansions[inside_role].append(bounds)
                    lane_expansions[weak_inside_role].append(bounds)

    min_half_width = max(0.0, float(params.deep_lane_half_width_min))
    max_threat_expand = max(0.0, float(params.deep_lane_threat_expand_y))
    for name, (lane_min, lane_max) in lane_bounds.items():
        role = role_by_name.get(name)
        if role is None:
            continue
        center = _finite_or(DEEP_ROLE_FALLBACK_Y[name], role.snap_y)
        lane_min = min(float(lane_min), center - min_half_width)
        lane_max = max(float(lane_max), center + min_half_width)
        for threat_bounds in lane_expansions.get(name, []):
            lane_min, lane_max = _expand_lane_toward_bounds(
                lane_min,
                lane_max,
                threat_bounds,
                max_threat_expand,
            )
        min_limit = lane_min_limits.get(name)
        max_limit = lane_max_limits.get(name)
        if min_limit is not None:
            lane_min = max(lane_min, float(min_limit))
        if max_limit is not None:
            lane_max = min(lane_max, float(max_limit))
        if lane_min > lane_max:
            center = float(np.clip(center, 0.0, FIELD_HEIGHT))
            lane_min = min(lane_min, center)
            lane_max = max(lane_max, center)
        role.lane_y_min = float(np.clip(lane_min, 0.0, FIELD_HEIGHT))
        role.lane_y_max = float(np.clip(lane_max, 0.0, FIELD_HEIGHT))


def _is_vertical_threat(route: ReceiverRoute | None, t: int) -> bool:
    if route is None:
        return False
    return bool(
        route.tags.get("vertical")
        and (
            _tag_frame(route, "vertical", t)
            or _finite_or(0.0, route.vx[_route_valid_index(route, t)]) > 0.5
            or route.tags.get("seam")
            or route.tags.get("corner")
            or route.tags.get("post")
        )
    )


def _is_flat_threat(route: ReceiverRoute | None, t: int) -> bool:
    if route is None:
        return False
    return bool(
        route.tags.get("back_fast_flat")
        or route.tags.get("flat")
        or route.tags.get("out")
        or _tag_frame(route, "flat", t)
    )


def _is_sideline_flat_threat(route: ReceiverRoute | None, t: int, side: str | None, params: SimParams) -> bool:
    if route is None or side not in (RIGHT_SIDE, LEFT_SIDE):
        return False
    if route.side != side and _route_rule_side(route) != side:
        return False
    if route.order != 1:
        return False
    if _route_deep_lane_threat(route):
        return False
    if not (
        route.tags.get("flat")
        or route.tags.get("out")
        or route.tags.get("shallow_cross")
        or _tag_frame(route, "flat", t)
    ):
        return False

    idx = _route_valid_index(route, t)
    first = _route_valid_index(route, 0)
    current_depth_gain = float(route.x[idx] - route.x[first]) if len(route.x) else 0.0
    current_outside = outside_y_sign(side) * float(route.y[idx] - route.y[first]) if len(route.y) else 0.0
    max_outside = _finite_or(0.0, route.tags.get("max_outside", 0.0))
    max_depth_gain = _finite_or(0.0, route.tags.get("max_depth_gain", 0.0))
    min_total_outside = max(
        2.0,
        0.60 * float(params.flat_min_lateral),
        float(params.flat_sideline_min_outside_y),
    )
    min_current_outside = max(0.0, float(params.flat_sideline_current_outside_y))
    max_depth_allowed = float(params.flat_max_depth) + float(params.flat_sideline_max_depth_extra)
    return bool(
        max(current_outside, max_outside) >= min_total_outside
        and current_outside >= min_current_outside
        and max(current_depth_gain, max_depth_gain) <= max_depth_allowed
    )


def _is_middle_flat_threat(route: ReceiverRoute | None, t: int, side: str | None) -> bool:
    if route is None or side not in (RIGHT_SIDE, LEFT_SIDE):
        return False
    if _route_rule_side(route) != side:
        return False
    if not (route.side == MIDDLE_SIDE or route.tags.get("backfield") or route.position.upper() in {"RUNNING_BACK", "RB"}):
        return False
    return _is_flat_threat(route, t)


def _route_finishes_on_side(route: ReceiverRoute, side: str, cushion_y: float = 4.0) -> bool:
    endpoints = _route_first_last_y(route)
    if endpoints is None:
        return False
    _, final_y = endpoints
    if side == RIGHT_SIDE:
        return final_y <= FIELD_MID_Y + float(cushion_y)
    if side == LEFT_SIDE:
        return final_y >= FIELD_MID_Y - float(cushion_y)
    return False


def _route_currently_on_side(route: ReceiverRoute, t: int, side: str, cushion_y: float = 4.0) -> bool:
    _, y = _route_xy(route, t)
    if side == RIGHT_SIDE:
        return y <= FIELD_MID_Y + float(cushion_y)
    if side == LEFT_SIDE:
        return y >= FIELD_MID_Y - float(cushion_y)
    return False


def _is_missing_flat_underneath_threat(
    route: ReceiverRoute | None,
    t: int,
    side: str | None,
    params: SimParams,
) -> bool:
    if route is None or side not in (RIGHT_SIDE, LEFT_SIDE):
        return False
    if _route_deep_lane_threat(route):
        return False
    if not (
        _is_flat_threat(route, t)
        or _is_shallow_or_under(route, t)
        or route.tags.get("out")
        or route.tags.get("shallow_cross")
    ):
        return False

    in_side_now = _route_currently_on_side(route, t, side)
    finishes_side = _route_finishes_on_side(route, side)
    release_side = route.tags.get("release_side")
    same_side_under = _route_rule_side(route) == side and (route.order == 1 or _is_flat_threat(route, t))
    crossing_to_side = bool(route.tags.get("shallow_cross") and finishes_side and (release_side == side or in_side_now))
    return bool(
        _is_sideline_flat_threat(route, t, side, params)
        or _is_middle_flat_threat(route, t, side)
        or crossing_to_side
        or (same_side_under and (in_side_now or finishes_side))
    )


def _missing_flat_underneath_threat(
    routes: Iterable[ReceiverRoute],
    t: int,
    side: str | None,
    params: SimParams,
) -> ReceiverRoute | None:
    if side not in (RIGHT_SIDE, LEFT_SIDE):
        return None
    candidates = [
        route
        for route in routes
        if _is_missing_flat_underneath_threat(route, t, side, params)
    ]
    if not candidates:
        return None

    projection = max(0, int(params.missing_flat_deep_projection_frames))

    def outside_score(route: ReceiverRoute) -> tuple[float, float, float]:
        _, projected_y = _route_xy(route, int(t) + projection)
        endpoints = _route_first_last_y(route)
        final_y = endpoints[1] if endpoints is not None else projected_y
        outside_y = outside_y_sign(side) * (0.65 * projected_y + 0.35 * final_y)
        order_bonus = 2.0 if route.order == 1 else 0.0
        release_bonus = 1.0 if route.tags.get("release_side") == side else 0.0
        return order_bonus + release_bonus + outside_y, order_bonus, outside_y

    return max(candidates, key=outside_score)


def _is_shallow_or_under(route: ReceiverRoute | None, t: int) -> bool:
    if route is None:
        return False
    return bool(
        route.tags.get("shallow_cross")
        or route.tags.get("flat")
        or route.tags.get("curl_sit")
        or _tag_frame(route, "shallow_cross", t)
    )


def _threat_id(route: ReceiverRoute | None) -> str | None:
    return route.nfl_id if route is not None else None


def _zone_react_threat(
    routes: list[ReceiverRoute | None],
    t: int,
    params: SimParams,
) -> ReceiverRoute | None:
    """Deepest same-side stem past the relaxed zone-react bar, if any."""
    best = None
    best_stem = -math.inf
    for route in routes:
        if route is None or _is_shallow_or_under(route, t):
            continue
        idx = _route_valid_index(route, t)
        stem = float(route.x[idx]) - float(route.x[0])
        vx = _finite_or(0.0, route.vx[idx])
        if stem >= float(params.zone_react_min_stem) and vx >= float(params.zone_react_min_vx):
            if stem > best_stem:
                best, best_stem = route, stem
    return best


def _state_score(state_name: str) -> float:
    if state_name in {"carry_1", "carry_2", "squeeze_2", "inside_help_1", "poach_2", "poach_3", "wall_2", "wall_3"}:
        return 0.90
    if state_name in {
        "midpoint_1_2",
        "missing_flat_underneath",
        "widen_flat",
        "widen_middle_flat",
        "widen_sideline_flat",
        "carry_crosser",
    }:
        return 0.82
    if state_name == "sink_under_vertical":
        return 0.72
    if state_name in {"curl_flat_zone", "middle_hook_zone", "zone_quarter"}:
        return 0.35
    return 0.0


def _candidate_decision(
    role: DefenderRole,
    state_name: str,
    primary: ReceiverRoute | None = None,
    secondary: ReceiverRoute | None = None,
    score: float | None = None,
) -> SimFrameDecision:
    return SimFrameDecision(
        role_name=role.role_name,
        state_name=state_name,
        primary_threat_id=_threat_id(primary),
        secondary_threat_id=_threat_id(secondary),
        target_x=math.nan,
        target_y=math.nan,
        target_weight=float(_state_score(state_name) if score is None else score),
    )


def _decision_from_previous(
    role: DefenderRole,
    prev_state: DefenderSimState,
    routes_by_id: Mapping[str, ReceiverRoute],
) -> SimFrameDecision:
    return SimFrameDecision(
        role_name=role.role_name,
        state_name=prev_state.state_name,
        primary_threat_id=prev_state.primary_threat_id,
        secondary_threat_id=prev_state.secondary_threat_id,
        target_x=math.nan,
        target_y=math.nan,
        target_weight=_state_score(prev_state.state_name),
    )


def _primary_still_valid(prev_state: DefenderSimState, routes_by_id: Mapping[str, ReceiverRoute], t: int) -> bool:
    if prev_state.primary_threat_id is None:
        return prev_state.state_name in {"zone_quarter", "curl_flat_zone", "middle_hook_zone"}
    route = routes_by_id.get(prev_state.primary_threat_id)
    if route is None:
        return False
    idx = _route_valid_index(route, t)
    return bool(route.mask[idx])


def choose_responsibility_state(
    role: DefenderRole,
    t: int,
    routes_by_side: Mapping[str, list[ReceiverRoute]],
    prev_state: DefenderSimState,
    params: SimParams,
    subcall_name: str = "base_quarters",
) -> SimFrameDecision:
    """Finite-state base-quarters responsibility rule engine."""
    if subcall_name != "base_quarters":
        subcall_name = "base_quarters"

    threat_t = max(0, int(t) - int(params.reaction_delay_frames))
    routes_by_id = {route.nfl_id: route for route in routes_by_side.get("all", [])}
    side = role_side(role.role_name)

    if role.role_name in OUTSIDE_DEEP_ROLES:
        side_routes = routes_by_side.get(side or RIGHT_SIDE, [])
        one = _find_order(side_routes, 1)
        two = _find_order(side_routes, 2)
        one_vert = _is_vertical_threat(one, threat_t) or bool(one and (one.tags.get("corner") or one.tags.get("post")))
        two_vert = _is_vertical_threat(two, threat_t)
        one_under = _is_shallow_or_under(one, threat_t)
        missing_flat_under = (
            _missing_flat_underneath_threat(routes_by_side.get("all", []), threat_t, side, params)
            if role.missing_flat_support
            else None
        )
        if one_vert and two_vert:
            break_frame = one.tags.get("route_break_frame") if one else None
            early_window = 12 if break_frame is None else max(8, int(break_frame) + 2)
            candidate = (
                _candidate_decision(role, "midpoint_1_2", one, two)
                if threat_t <= early_window
                else _candidate_decision(role, "carry_1", one)
            )
        elif one_under and two_vert:
            candidate = _candidate_decision(role, "squeeze_2", two, one)
        elif one_vert:
            candidate = _candidate_decision(role, "carry_1", one)
        elif two_vert:
            candidate = _candidate_decision(role, "squeeze_2", two)
        elif missing_flat_under is not None:
            candidate = _candidate_decision(role, "missing_flat_underneath", missing_flat_under)
        else:
            react = (
                _zone_react_threat([one, two], threat_t, params)
                if bool(params.enable_v2_zone_react)
                else None
            )
            if react is not None:
                candidate = _candidate_decision(role, "carry_1", react)
            else:
                candidate = _candidate_decision(role, "zone_quarter")

    elif role.role_name in INSIDE_DEEP_ROLES:
        side_routes = routes_by_side.get(side or RIGHT_SIDE, [])
        opposite_side = LEFT_SIDE if side == RIGHT_SIDE else RIGHT_SIDE
        opposite_routes = routes_by_side.get(opposite_side, [])
        one = _find_order(side_routes, 1)
        two = _find_order(side_routes, 2)
        opposite_two = _find_order(opposite_routes, 2)
        opposite_three = _find_order(opposite_routes, 3)
        one_vert = _is_vertical_threat(one, threat_t)
        two_vert = _is_vertical_threat(two, threat_t) or bool(two and two.tags.get("seam"))
        opposite_two_vert = _is_vertical_threat(opposite_two, threat_t) or bool(opposite_two and opposite_two.tags.get("seam"))
        opposite_three_vert = _is_vertical_threat(opposite_three, threat_t) or bool(
            opposite_three and opposite_three.tags.get("seam")
        )
        two_out = _is_flat_threat(two, threat_t)
        if one_vert and two_vert:
            candidate = _candidate_decision(role, "midpoint_1_2", one, two)
        elif two_vert:
            candidate = _candidate_decision(role, "carry_2", two)
        elif two_out and one_vert:
            candidate = _candidate_decision(role, "inside_help_1", one, two)
        elif one_vert:
            candidate = _candidate_decision(role, "inside_help_1", one)
        elif opposite_three_vert:
            candidate = _candidate_decision(role, "poach_3", opposite_three)
        elif opposite_two_vert:
            candidate = _candidate_decision(role, "poach_2", opposite_two)
        else:
            react = (
                _zone_react_threat([one, two], threat_t, params)
                if bool(params.enable_v2_zone_react)
                else None
            )
            if react is not None:
                candidate = _candidate_decision(role, "carry_2", react)
            else:
                candidate = _candidate_decision(role, "zone_quarter")

    elif role.role_name in FLAT_ROLES:
        side_routes = routes_by_side.get(side or RIGHT_SIDE, [])
        sideline_flat_threat = None
        for route in side_routes:
            if _is_sideline_flat_threat(route, threat_t, side, params):
                sideline_flat_threat = route
                break
        middle_flat_threat = None
        for route in side_routes:
            if _is_middle_flat_threat(route, threat_t, side):
                middle_flat_threat = route
                break
        flat_threat = None
        for route in side_routes:
            if _is_flat_threat(route, threat_t):
                flat_threat = route
                break
        crosser = None
        for route in routes_by_side.get("all", []):
            if route.tags.get("shallow_cross") and route.mask[_route_valid_index(route, threat_t)]:
                _, ry = _route_xy(route, threat_t)
                if (side == RIGHT_SIDE and ry <= FIELD_MID_Y + 4.0) or (side == LEFT_SIDE and ry >= FIELD_MID_Y - 4.0):
                    crosser = route
                    break
        vertical = next((route for route in side_routes if _is_vertical_threat(route, threat_t)), None)
        if sideline_flat_threat is not None:
            candidate = _candidate_decision(role, "widen_sideline_flat", sideline_flat_threat)
        elif middle_flat_threat is not None:
            candidate = _candidate_decision(role, "widen_middle_flat", middle_flat_threat)
        elif flat_threat is not None:
            candidate = _candidate_decision(role, "widen_flat", flat_threat)
        elif vertical is not None:
            candidate = _candidate_decision(role, "sink_under_vertical", vertical)
        elif crosser is not None:
            candidate = _candidate_decision(role, "carry_crosser", crosser)
        else:
            candidate = _candidate_decision(role, "curl_flat_zone")

    else:
        hook_side = RIGHT_SIDE if role.snap_y < FIELD_MID_Y else LEFT_SIDE
        side_two = _find_order(routes_by_side.get(hook_side, []), 2)
        wall_two = None
        if side_two is not None and (_is_vertical_threat(side_two, threat_t) or side_two.tags.get("seam")):
            _, two_snap_y = _route_xy(side_two, 0)
            if abs(float(role.snap_y) - two_snap_y) <= float(params.hook_wall_two_snap_y_window):
                wall_two = side_two
        third_or_back = None
        for route in routes_by_side.get("all", []):
            through_middle = abs(_route_xy(route, threat_t)[1] - FIELD_MID_Y) <= 7.0
            if (route.order == 3 or route.side == MIDDLE_SIDE or route.position.upper() in {"RUNNING_BACK", "RB"}) and through_middle:
                if _is_vertical_threat(route, threat_t) or route.tags.get("seam"):
                    third_or_back = route
                    break
        crosser = None
        for route in routes_by_side.get("all", []):
            if (route.tags.get("shallow_cross") or route.tags.get("dig_in")) and route.mask[_route_valid_index(route, threat_t)]:
                if abs(_route_xy(route, threat_t)[1] - FIELD_MID_Y) <= 10.0:
                    crosser = route
                    break
        if wall_two is not None:
            candidate = _candidate_decision(role, "wall_2", wall_two)
        elif third_or_back is not None:
            candidate = _candidate_decision(role, "wall_3", third_or_back)
        elif crosser is not None:
            candidate = _candidate_decision(role, "carry_crosser", crosser)
        else:
            candidate = _candidate_decision(role, "middle_hook_zone")

    if prev_state.state_name and prev_state.state_name != "initial" and prev_state.state_name != candidate.state_name:
        current_valid = _primary_still_valid(prev_state, routes_by_id, threat_t)
        current_score = _state_score(prev_state.state_name)
        if (
            current_valid
            and prev_state.state_age < params.min_state_dwell_frames
            and candidate.target_weight < current_score + params.switch_margin
        ):
            return _decision_from_previous(role, prev_state, routes_by_id)
        if current_valid and candidate.target_weight + 1e-9 < current_score + params.switch_margin:
            return _decision_from_previous(role, prev_state, routes_by_id)

    return candidate


def assignment_trust_for_role(role: DefenderRole, params: SimParams) -> float:
    """Return a smooth 0-1 trust score for a static role assignment."""
    margin = _finite_or(0.0, role.assignment_margin)
    scale = max(float(params.assignment_trust_margin_scale), 1e-6)
    raw = (margin + float(params.assignment_trust_margin_offset)) / scale
    trust = float(np.clip(raw, params.assignment_trust_min, 1.0))
    return trust


def zone_anchor_for_role(
    role: DefenderRole,
    los_x: float,
    params: SimParams,
) -> tuple[float, float]:
    """Return the movement zone anchor for a role.

    Assignment landmarks are allowed to be LOS-relative. Movement anchors need a
    depth floor from the defender's actual snap, otherwise already-deep safeties
    and apex defenders can be pulled backward toward a shallow nominal zone.
    """
    if role.role_name in OUTSIDE_DEEP_ROLES:
        x = max(los_x + params.deep_zone_depth, role.snap_x + params.deep_anchor_snap_advance)
        y = role.landmark_y if math.isfinite(role.landmark_y) else role.snap_y
    elif role.role_name in INSIDE_DEEP_ROLES:
        x = max(los_x + params.inside_deep_zone_depth, role.snap_x + params.deep_anchor_snap_advance)
        landmark_y = role.landmark_y if math.isfinite(role.landmark_y) else role.snap_y
        middle_blend = float(np.clip(params.inside_deep_anchor_midfield_blend, 0.0, 1.0))
        y = (1.0 - middle_blend) * landmark_y + middle_blend * FIELD_MID_Y
    elif role.role_name in FLAT_ROLES:
        x = max(los_x + params.flat_zone_depth, role.snap_x + params.flat_anchor_snap_advance)
        landmark_y = role.landmark_y if math.isfinite(role.landmark_y) else role.snap_y
        snap_y_blend = float(np.clip(params.flat_anchor_snap_y_blend, 0.0, 1.0))
        y = (1.0 - snap_y_blend) * landmark_y + snap_y_blend * role.snap_y
    else:
        x = max(los_x + params.hook_zone_depth, role.snap_x + params.hook_anchor_snap_advance)
        middle_blend = float(np.clip(params.hook_anchor_midfield_blend, 0.0, 1.0))
        y = (1.0 - middle_blend) * role.snap_y + middle_blend * FIELD_MID_Y

    trust = assignment_trust_for_role(role, params)
    snap_blend = (1.0 - trust) * float(np.clip(params.low_trust_anchor_snap_blend, 0.0, 1.0))
    if snap_blend > 0.0:
        if role.role_name not in DEEP_ROLES:
            x = (1.0 - snap_blend) * x + snap_blend * role.snap_x
        y = (1.0 - snap_blend) * y + snap_blend * role.snap_y
    return _clip_role_xy(role, x, y)


def movement_anchor_for_decision(
    role: DefenderRole,
    decision: SimFrameDecision,
    los_x: float,
    params: SimParams,
) -> tuple[float, float]:
    """Return the conservative anchor for a responsibility state."""
    if role.role_name in OUTSIDE_DEEP_ROLES and decision.state_name == "missing_flat_underneath":
        x = max(
            los_x + params.flat_zone_depth,
            role.snap_x + params.missing_flat_deep_snap_advance,
        )
        return _clip_role_xy(role, x, role.snap_y)
    return zone_anchor_for_role(role, los_x, params)


def raw_target_for_decision(
    role: DefenderRole,
    decision: SimFrameDecision,
    routes_by_id: Mapping[str, ReceiverRoute],
    t: int,
    los_x: float,
    params: SimParams,
) -> tuple[float, float]:
    """Convert a responsibility decision to an unblended on-field target point."""
    side = role_side(role.role_name)
    if side is None:
        side = RIGHT_SIDE if role.snap_y < FIELD_MID_Y else LEFT_SIDE

    primary = routes_by_id.get(decision.primary_threat_id or "")
    secondary = routes_by_id.get(decision.secondary_threat_id or "")
    state = decision.state_name
    use_v2_movement = str(getattr(params, "movement_version", "v1")).lower() == "v2"

    if primary is not None:
        px, py = _route_xy(primary, t)
        projection_frames = int(params.deep_threat_projection_frames) if role.role_name in DEEP_ROLES else 0
        projected_px = _route_projected_x(primary, t, projection_frames)
    else:
        px, py = float(role.landmark_x), float(role.landmark_y)
        if not math.isfinite(px):
            px = los_x + _role_depth(role.role_name, params)
        if not math.isfinite(py):
            py = role.snap_y if role.role_name != "hook_curl" else FIELD_MID_Y
        projected_px = px

    if (
        state in {"wall_2", "wall_3"}
        and role.role_name == "hook_curl"
        and use_v2_movement
        and bool(params.enable_v2_hook_wall)
    ):
        return v2_hook_wall_target(
            los_x=los_x,
            projected_route_x=projected_px,
            route_y=py,
            role_snap_x=role.snap_x,
            wall_depth=params.hook_wall_depth,
            depth_cushion=params.hook_wall_depth_cushion,
            route_y_blend=params.hook_wall_route_y_blend,
        )
    if state in {"carry_1", "carry_2", "wall_2", "wall_3"}:
        if (
            state in {"carry_1", "carry_2"}
            and use_v2_movement
            and bool(params.enable_v2_carry_vertical_mirror)
            and primary is not None
        ):
            route_snap_x, route_snap_y = _route_xy(primary, 0)
            return v2_under_mirror_target(
                los_x=los_x,
                route_snap_x=route_snap_x,
                route_snap_y=route_snap_y,
                route_x=px,
                route_y=py,
                role_snap_x=role.snap_x,
                role_snap_y=role.snap_y,
                vertical_gain=params.carry_vert_mirror_vertical_gain,
                lateral_gain=params.carry_vert_mirror_lateral_gain,
                under_cushion_x=params.carry_vert_mirror_cushion_x,
                min_depth=params.carry_vert_mirror_min_depth,
            )
        if (
            state in {"wall_2", "wall_3"}
            and use_v2_movement
            and bool(params.enable_v2_wall_mirror)
            and primary is not None
        ):
            route_snap_x, route_snap_y = _route_xy(primary, 0)
            return v2_under_mirror_target(
                los_x=los_x,
                route_snap_x=route_snap_x,
                route_snap_y=route_snap_y,
                route_x=px,
                route_y=py,
                role_snap_x=role.snap_x,
                role_snap_y=role.snap_y,
                vertical_gain=params.wall_mirror_vertical_gain,
                lateral_gain=params.wall_mirror_lateral_gain,
                under_cushion_x=params.wall_mirror_cushion_x,
                min_depth=params.wall_mirror_min_depth,
            )
        x = projected_px + params.deep_over_top_cushion
        y = py + inside_y_sign(side) * params.deep_inside_leverage_y
    elif state in {"squeeze_2", "inside_help_1", "poach_2", "poach_3"}:
        x = projected_px + params.deep_over_top_cushion
        y = py + inside_y_sign(side) * 0.5
    elif state == "midpoint_1_2" and secondary is not None:
        sx, sy = _route_xy(secondary, t)
        if use_v2_movement and bool(params.enable_v2_midpoint_mirror) and primary is not None:
            p_snap_x, p_snap_y = _route_xy(primary, 0)
            s_snap_x, s_snap_y = _route_xy(secondary, 0)
            return v2_under_mirror_target(
                los_x=los_x,
                route_snap_x=0.5 * (p_snap_x + s_snap_x),
                route_snap_y=0.5 * (p_snap_y + s_snap_y),
                route_x=0.5 * (px + sx),
                route_y=0.5 * (py + sy),
                role_snap_x=role.snap_x,
                role_snap_y=role.snap_y,
                vertical_gain=params.midpoint_mirror_vertical_gain,
                lateral_gain=params.midpoint_mirror_lateral_gain,
                under_cushion_x=params.midpoint_mirror_cushion_x,
                min_depth=params.midpoint_mirror_min_depth,
            )
        projection_frames = int(params.deep_threat_projection_frames) if role.role_name in DEEP_ROLES else 0
        projected_sx = _route_projected_x(secondary, t, projection_frames)
        x = 0.52 * projected_px + 0.48 * projected_sx + params.midpoint_over_top_cushion
        y = 0.50 * py + 0.50 * sy + inside_y_sign(side) * 0.25
    elif state == "missing_flat_underneath":
        if primary is not None:
            projection_t = int(t) + int(params.missing_flat_deep_projection_frames)
            px, py = _route_xy(primary, projection_t)
        x = max(
            role.snap_x + params.missing_flat_deep_snap_advance,
            px + params.deep_over_top_cushion,
        )
        y = py + outside_y_sign(side) * params.missing_flat_deep_outside_leverage_y
    elif state == "widen_sideline_flat":
        if use_v2_movement:
            return v2_sideline_flat_target(
                side=side,
                route_x=px,
                route_y=py,
                depth_cushion=params.flat_depth_cushion,
                outside_leverage_y=params.flat_sideline_outside_leverage_y,
            )
        else:
            x = px + params.flat_depth_cushion
            y = py + outside_y_sign(side) * params.flat_sideline_outside_leverage_y
    elif state == "widen_middle_flat":
        if primary is not None:
            _, py = _route_xy(primary, int(t) + int(params.flat_middle_projection_frames))
        if use_v2_movement and bool(params.enable_v2_middle_flat):
            return v2_middle_flat_target(
                side=side,
                los_x=los_x,
                route_x=px,
                projected_route_y=py,
                role_snap_x=role.snap_x,
                role_snap_y=role.snap_y,
                flat_zone_depth=params.flat_zone_depth,
                depth_cushion=params.flat_middle_depth_cushion,
                snap_advance=params.flat_middle_snap_advance,
                inside_leverage_y=params.flat_inside_leverage_y,
                route_y_blend=params.flat_middle_route_y_blend,
            )
        else:
            x = max(px + params.flat_depth_cushion, role.snap_x + params.flat_anchor_snap_advance)
            y = py + inside_y_sign(side) * params.flat_inside_leverage_y
            if side == RIGHT_SIDE:
                y = min(y, role.snap_y)
            elif side == LEFT_SIDE:
                y = max(y, role.snap_y)
    elif state == "widen_flat":
        x = px + params.flat_depth_cushion
        y = py + inside_y_sign(side) * params.flat_inside_leverage_y
    elif state == "sink_under_vertical":
        if use_v2_movement and bool(params.enable_v2_under_mirror):
            if primary is not None:
                route_snap_x, route_snap_y = _route_xy(primary, 0)
            else:
                route_snap_x, route_snap_y = px, py
            return v2_under_mirror_target(
                los_x=los_x,
                route_snap_x=route_snap_x,
                route_snap_y=route_snap_y,
                route_x=px,
                route_y=py,
                role_snap_x=role.snap_x,
                role_snap_y=role.snap_y,
                vertical_gain=params.under_mirror_vertical_gain,
                lateral_gain=params.under_mirror_lateral_gain,
                under_cushion_x=params.under_mirror_cushion_x,
                min_depth=params.under_mirror_min_depth,
            )
        elif use_v2_movement and bool(params.enable_v2_sink_under):
            return v2_sink_under_vertical_target(
                side=side,
                los_x=los_x,
                route_x=px,
                route_y=py,
                role_snap_x=role.snap_x,
                role_snap_y=role.snap_y,
                sink_depth=params.sink_under_depth,
                receiver_cushion_x=params.sink_under_receiver_cushion_x,
                inside_leverage_y=params.flat_inside_leverage_y,
                route_y_blend=params.sink_under_route_y_blend,
                max_lateral_from_snap_y=params.sink_under_max_lateral_from_snap_y,
            )
        else:
            x = min(los_x + params.sink_under_depth, px - 2.0)
            y = py + inside_y_sign(side) * params.flat_inside_leverage_y
    elif state == "carry_crosser":
        if use_v2_movement and bool(params.enable_v2_carry_mirror):
            if primary is not None:
                route_snap_x, route_snap_y = _route_xy(primary, 0)
            else:
                route_snap_x, route_snap_y = px, py
            return v2_under_mirror_target(
                los_x=los_x,
                route_snap_x=route_snap_x,
                route_snap_y=route_snap_y,
                route_x=px,
                route_y=py,
                role_snap_x=role.snap_x,
                role_snap_y=role.snap_y,
                vertical_gain=params.carry_mirror_vertical_gain,
                lateral_gain=params.carry_mirror_lateral_gain,
                under_cushion_x=params.carry_mirror_cushion_x,
                min_depth=params.carry_mirror_min_depth,
            )
        elif use_v2_movement and bool(params.enable_v2_carry_crosser):
            return v2_carry_crosser_target(
                role_name=role.role_name,
                side=side,
                los_x=los_x,
                route_x=px,
                route_y=py,
                role_snap_x=role.snap_x,
                hook_depth=params.hook_crosser_depth,
                hook_depth_cushion=params.hook_crosser_depth_cushion,
                hook_route_y_blend=params.hook_crosser_route_y_blend,
                flat_depth_cushion=params.flat_crosser_depth_cushion,
                flat_inside_leverage_y=params.flat_crosser_inside_leverage_y,
            )
        else:
            x = px + 0.5
            if role.role_name == "hook_curl":
                y = 0.7 * py + 0.3 * FIELD_MID_Y
            else:
                y = py + inside_y_sign(side) * 0.75
    elif state in {"zone_quarter", "curl_flat_zone", "middle_hook_zone"}:
        return zone_anchor_for_role(role, los_x, params)
    else:
        x = px
        y = py

    return _clip_field_xy(x, y)


def target_blend_for_decision(
    role: DefenderRole,
    decision: SimFrameDecision,
    params: SimParams,
    anchor_xy: tuple[float, float] | None = None,
    raw_xy: tuple[float, float] | None = None,
    deep_cluster_active: bool = False,
) -> float:
    """Return how far to move from the role zone anchor toward the raw target."""
    if decision.state_name in {"zone_quarter", "curl_flat_zone", "middle_hook_zone"}:
        return 0.0
    if role.role_name in INSIDE_DEEP_ROLES:
        base = float(np.clip(params.inside_deep_target_blend, 0.0, 1.0))
    elif role.role_name in OUTSIDE_DEEP_ROLES:
        base = float(np.clip(params.deep_target_blend, 0.0, 1.0))
    elif role.role_name in FLAT_ROLES:
        base = float(np.clip(params.flat_target_blend, 0.0, 1.0))
    elif role.role_name in UNDERNEATH_ROLES:
        base = float(np.clip(params.underneath_target_blend, 0.0, 1.0))
    else:
        base = 1.0
    trust = assignment_trust_for_role(role, params)
    if role.role_name in OUTSIDE_DEEP_ROLES and decision.state_name == "missing_flat_underneath":
        base = float(np.clip(params.missing_flat_deep_target_blend, 0.0, 1.0))
        trust = max(trust, float(np.clip(params.missing_flat_deep_target_trust_floor, 0.0, 1.0)))
    elif role.role_name in FLAT_ROLES and decision.state_name == "widen_sideline_flat":
        base = float(np.clip(params.flat_sideline_target_blend, 0.0, 1.0))
        trust = max(trust, float(np.clip(params.flat_sideline_target_trust_floor, 0.0, 1.0)))
    elif role.role_name in FLAT_ROLES and decision.state_name == "widen_middle_flat":
        base = float(np.clip(params.flat_middle_target_blend, 0.0, 1.0))
        trust = max(trust, float(np.clip(params.flat_middle_target_trust_floor, 0.0, 1.0)))
    elif role.role_name in FLAT_ROLES and decision.state_name == "widen_flat":
        trust = max(trust, float(np.clip(params.flat_target_trust_floor, 0.0, 1.0)))
    blend = float(np.clip(base * trust, 0.0, 1.0))
    if (
        role.role_name in DEEP_ROLES
        and decision.state_name in {"carry_1", "carry_2", "squeeze_2", "inside_help_1", "poach_2", "poach_3"}
        and anchor_xy is not None
        and raw_xy is not None
    ):
        side = role_side(role.role_name)
        if side is None:
            side = RIGHT_SIDE if role.snap_y < FIELD_MID_Y else LEFT_SIDE
        inside_delta = inside_y_sign(side) * (float(raw_xy[1]) - float(anchor_xy[1]))
        if inside_delta >= float(params.deep_lateral_target_min_delta_y):
            blend = max(blend, float(np.clip(params.deep_lateral_target_min_blend, 0.0, 1.0)))
    if deep_cluster_active:
        blend = max(blend, float(np.clip(params.deep_cluster_target_blend_floor, 0.0, 1.0)))
    return blend


def _blend_target(
    anchor_xy: tuple[float, float],
    raw_xy: tuple[float, float],
    blend: float,
) -> tuple[float, float]:
    blend = float(np.clip(blend, 0.0, 1.0))
    x = anchor_xy[0] + blend * (raw_xy[0] - anchor_xy[0])
    y = anchor_xy[1] + blend * (raw_xy[1] - anchor_xy[1])
    return _clip_field_xy(x, y)


def _protect_deep_over_top_x(
    role: DefenderRole,
    anchor_xy: tuple[float, float],
    raw_xy: tuple[float, float],
    blended_xy: tuple[float, float],
    params: SimParams,
) -> tuple[float, float]:
    """Keep deep-quarter targets from settling underneath vertical threats."""
    if role.role_name not in DEEP_ROLES:
        return blended_xy
    min_blend = float(np.clip(params.deep_target_min_x_blend, 0.0, 1.0))
    min_x = anchor_xy[0]
    if raw_xy[0] > anchor_xy[0]:
        min_x = anchor_xy[0] + min_blend * (raw_xy[0] - anchor_xy[0])
    x = max(blended_xy[0], min_x)
    return _clip_role_xy(role, x, blended_xy[1])
