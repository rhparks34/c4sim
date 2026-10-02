"""Movement targets of the coded zone rules: sideline and middle flat, hook wall, sink under a vertical, carry a crosser,
mirror under. Verbatim from the research code."""
from __future__ import annotations

import numpy as np

FIELD_WIDTH = 120.0


FIELD_HEIGHT = 53.3


FIELD_MID_Y = FIELD_HEIGHT / 2.0


RIGHT_SIDE = "right"


LEFT_SIDE = "left"


def outside_y_sign(side: str) -> float:
    return -1.0 if side == RIGHT_SIDE else 1.0


def inside_y_sign(side: str) -> float:
    return -outside_y_sign(side)


def clip_field_xy(x: float, y: float) -> tuple[float, float]:
    return float(np.clip(x, 0.0, FIELD_WIDTH)), float(np.clip(y, 0.0, FIELD_HEIGHT))


def sink_under_vertical_target(
    *,
    side: str,
    los_x: float,
    route_x: float,
    route_y: float,
    role_snap_x: float,
    role_snap_y: float,
    sink_depth: float,
    receiver_cushion_x: float,
    inside_leverage_y: float,
    route_y_blend: float,
    max_lateral_from_snap_y: float,
) -> tuple[float, float]:
    x = max(float(role_snap_x), min(float(los_x) + float(sink_depth), float(route_x) - float(receiver_cushion_x)))
    raw_y = float(route_y) + inside_y_sign(side) * float(inside_leverage_y)
    y_delta = float(np.clip(raw_y - float(role_snap_y), -float(max_lateral_from_snap_y), float(max_lateral_from_snap_y)))
    y = float(role_snap_y) + float(np.clip(route_y_blend, 0.0, 1.0)) * y_delta
    return clip_field_xy(x, y)


def under_mirror_target(
    *,
    los_x: float,
    route_snap_x: float,
    route_snap_y: float,
    route_x: float,
    route_y: float,
    role_snap_x: float,
    role_snap_y: float,
    vertical_gain: float,
    lateral_gain: float,
    under_cushion_x: float,
    min_depth: float,
) -> tuple[float, float]:
    """Dictionary under-help geometry (COVER4_GAP_PLAN.md §0.0.5.6 Stage 3).

    Proportional mirror of the threat's displacement from the defender's own
    snap point: partial vertical carry, near-full lateral tracking (the trail/
    rob shape fitted in Stage 1: 0.49 / 0.93), held UNDER the threat by a
    cushion that MAY pull the defender downhill below his snap depth (the
    drive-on-in-breakers behavior the old sink-under's max(snap_x, ...)
    forbade)."""
    x = float(role_snap_x) + float(vertical_gain) * (float(route_x) - float(route_snap_x))
    x = min(x, float(route_x) - float(under_cushion_x))
    x = max(x, float(los_x) + float(min_depth))
    y = float(role_snap_y) + float(lateral_gain) * (float(route_y) - float(route_snap_y))
    return clip_field_xy(x, y)


def middle_flat_target(
    *,
    side: str,
    los_x: float,
    route_x: float,
    projected_route_y: float,
    role_snap_x: float,
    role_snap_y: float,
    flat_zone_depth: float,
    depth_cushion: float,
    snap_advance: float,
    inside_leverage_y: float,
    route_y_blend: float,
) -> tuple[float, float]:
    x = max(
        float(los_x) + float(flat_zone_depth),
        float(role_snap_x) + float(snap_advance),
        float(route_x) + float(depth_cushion),
    )
    raw_y = float(projected_route_y) + inside_y_sign(side) * float(inside_leverage_y)
    y = float(role_snap_y) + float(np.clip(route_y_blend, 0.0, 1.0)) * (raw_y - float(role_snap_y))
    if side == RIGHT_SIDE:
        y = min(y, float(role_snap_y))
    elif side == LEFT_SIDE:
        y = max(y, float(role_snap_y))
    return clip_field_xy(x, y)


def sideline_flat_target(
    *,
    side: str,
    route_x: float,
    route_y: float,
    depth_cushion: float,
    outside_leverage_y: float,
) -> tuple[float, float]:
    x = float(route_x) + float(depth_cushion)
    y = float(route_y) + outside_y_sign(side) * float(outside_leverage_y)
    return clip_field_xy(x, y)


def hook_wall_target(
    *,
    los_x: float,
    projected_route_x: float,
    route_y: float,
    role_snap_x: float,
    wall_depth: float,
    depth_cushion: float,
    route_y_blend: float,
) -> tuple[float, float]:
    x = max(float(role_snap_x), min(float(los_x) + float(wall_depth), float(projected_route_x) + float(depth_cushion)))
    y = (1.0 - float(np.clip(route_y_blend, 0.0, 1.0))) * FIELD_MID_Y + float(
        np.clip(route_y_blend, 0.0, 1.0)
    ) * float(route_y)
    return clip_field_xy(x, y)


def carry_crosser_target(
    *,
    role_name: str,
    side: str,
    los_x: float,
    route_x: float,
    route_y: float,
    role_snap_x: float,
    hook_depth: float,
    hook_depth_cushion: float,
    hook_route_y_blend: float,
    flat_depth_cushion: float,
    flat_inside_leverage_y: float,
) -> tuple[float, float]:
    if role_name == "hook_curl":
        x = max(float(role_snap_x), min(float(los_x) + float(hook_depth), float(route_x) + float(hook_depth_cushion)))
        blend = float(np.clip(hook_route_y_blend, 0.0, 1.0))
        y = (1.0 - blend) * FIELD_MID_Y + blend * float(route_y)
    else:
        x = float(route_x) + float(flat_depth_cushion)
        y = float(route_y) + inside_y_sign(side) * float(flat_inside_leverage_y)
    return clip_field_xy(x, y)
