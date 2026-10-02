"""Route tags for the coded rules: each receiver's side order (#1, #2, #3 from the sideline), release side, depth, vertical and
break frame. Verbatim from the research code."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from src.prep.rules.cover4_rules import (
    FIELD_HEIGHT,
    FIELD_MID_Y,
    LEFT_SIDE,
    MIDDLE_SIDE,
    RIGHT_SIDE,
    ROLE_ORDER,
    ReceiverRoute,
    SimParams,
    inside_y_sign,
    normalize_id,
    outside_y_sign,
)

def _first_valid(mask: np.ndarray) -> int:
    valid = np.flatnonzero(mask)
    return int(valid[0]) if len(valid) else 0


def _safe_array(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    if out.ndim != 1:
        out = out.reshape(-1)
    if len(out) == 0:
        return out
    finite_mask = np.isfinite(out) & mask
    if not np.any(finite_mask):
        out[:] = 0.0
        return out
    valid_idx = np.flatnonzero(finite_mask)
    valid_vals = out[valid_idx]
    all_idx = np.arange(len(out))
    out = np.interp(all_idx, valid_idx, valid_vals)
    return out


def _smooth_bool(mask: np.ndarray, width: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if width <= 0 or len(out) == 0:
        return out
    base = out.copy()
    for shift in range(1, width + 1):
        out[shift:] |= base[:-shift]
        out[:-shift] |= base[shift:]
    return out


def _signed_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _position_is_rb(position: str) -> bool:
    return str(position).upper() in {"RUNNING_BACK", "RB", "HB", "FB"}


def assign_receiver_side_order(
    snap_y: np.ndarray,
    player_ids: Sequence[str],
    field_height: float = FIELD_HEIGHT,
    middle_band: float = 3.0,
    release_y: np.ndarray | None = None,
    positions: Sequence[str] | None = None,
    snap_x: np.ndarray | None = None,
    los_x: float | None = None,
    split_y: float | None = None,
) -> dict[str, tuple[str, int | None]]:
    """Assign snap-alignment side and outside-in order.

    ``side`` is intentionally the alignment side, not the release side. Receivers
    can cross or release away from their original side after the snap, but Cover 4
    #1/#2/#3 rules start from formation alignment. Release side is tagged later
    by ``tag_route`` for dynamic responsibility decisions.
    """
    snap_y = np.asarray(snap_y, dtype=float)
    if release_y is not None:
        release_y = np.asarray(release_y, dtype=float)
    if snap_x is not None:
        snap_x = np.asarray(snap_x, dtype=float)
    positions = list(positions or ["UNKNOWN"] * len(player_ids))
    if len(positions) < len(player_ids):
        positions.extend(["UNKNOWN"] * (len(player_ids) - len(positions)))

    mid_y = field_height / 2.0
    split = float(split_y) if split_y is not None and np.isfinite(split_y) else mid_y
    player_ids_norm = [normalize_id(pid) for pid in player_ids]
    idx_by_pid = {pid: idx for idx, pid in enumerate(player_ids_norm)}
    side_by_id: dict[str, str] = {}
    for idx, raw_id in enumerate(player_ids):
        pid = normalize_id(raw_id)
        y0 = float(snap_y[idx]) if idx < len(snap_y) and np.isfinite(snap_y[idx]) else split
        side = RIGHT_SIDE if y0 < split else LEFT_SIDE
        ambiguous = abs(y0 - split) <= middle_band
        is_backfield_middle = False
        if snap_x is not None and los_x is not None and idx < len(snap_x):
            is_backfield_middle = (
                float(snap_x[idx]) < float(los_x) - 1.0
                and abs(y0 - split) <= middle_band
            )

        if ambiguous:
            if is_backfield_middle or _position_is_rb(positions[idx]):
                side = MIDDLE_SIDE
            else:
                side = RIGHT_SIDE if y0 < split else LEFT_SIDE
        side_by_id[pid] = side

    assignments: dict[str, tuple[str, int | None]] = {}
    for side in (RIGHT_SIDE, LEFT_SIDE):
        ids_on_side = [
            normalize_id(pid)
            for idx, pid in enumerate(player_ids)
            if side_by_id.get(normalize_id(pid)) == side
        ]
        ids_on_side.sort(
            key=lambda pid: (
                -snap_y[idx_by_pid[pid]]
                if side == LEFT_SIDE
                else snap_y[idx_by_pid[pid]]
            )
        )
        for order_idx, pid in enumerate(ids_on_side, start=1):
            assignments[pid] = (side, order_idx)

    for raw_id in player_ids:
        pid = normalize_id(raw_id)
        if pid not in assignments:
            assignments[pid] = (MIDDLE_SIDE, None)
    return assignments


def _formation_split_y(play: Mapping[str, Any], snap_y: np.ndarray) -> float:
    explicit = play.get("formation_split_y")
    if explicit is not None:
        try:
            value = float(explicit)
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value):
            return float(np.clip(value, 0.0, FIELD_HEIGHT))

    qb_data = np.asarray(play.get("qb_data"), dtype=float)
    if qb_data.ndim == 3 and qb_data.shape[0] > 0 and qb_data.shape[1] > 0 and qb_data.shape[2] >= 2:
        qb_x = float(qb_data[0, 0, 0])
        qb_y = float(qb_data[0, 0, 1])
        if math.isfinite(qb_x) and math.isfinite(qb_y) and qb_x > 0.0 and 0.5 < qb_y < FIELD_HEIGHT - 0.5:
            return qb_y

    finite_snap_y = np.asarray(snap_y, dtype=float)
    finite_snap_y = finite_snap_y[np.isfinite(finite_snap_y)]
    if len(finite_snap_y):
        return float(np.clip(np.median(finite_snap_y), 0.0, FIELD_HEIGHT))
    return FIELD_MID_Y


def _infer_release_side(x: np.ndarray, y: np.ndarray, mask: np.ndarray, initial_side: str) -> str:
    first = _first_valid(mask)
    valid = np.flatnonzero(mask)
    if len(valid) <= 1:
        return RIGHT_SIDE if y[first] < FIELD_MID_Y else LEFT_SIDE
    end = int(valid[min(len(valid) - 1, 7)])
    dy = float(y[end] - y[first])
    if abs(dy) >= 0.75:
        return RIGHT_SIDE if dy < 0.0 else LEFT_SIDE
    if initial_side in (RIGHT_SIDE, LEFT_SIDE):
        return initial_side
    return RIGHT_SIDE if y[first] < FIELD_MID_Y else LEFT_SIDE


def build_receiver_routes(play: Mapping[str, Any], params: SimParams | None = None) -> list[ReceiverRoute]:
    """Convert play['offense_data'] into receiver routes with side/order/tags."""
    params = params or SimParams()
    offense_data = np.asarray(play.get("offense_data"), dtype=float)
    if offense_data.ndim != 3 or offense_data.shape[0] == 0:
        return []
    n_off = int(min(play.get("n_off", offense_data.shape[0]), offense_data.shape[0]))
    n_frames = int(min(play.get("n_frames", offense_data.shape[1]), offense_data.shape[1]))
    offense_mask = np.asarray(play.get("offense_mask", np.ones((n_off, n_frames))), dtype=bool)
    if offense_mask.ndim != 2:
        offense_mask = np.ones((n_off, n_frames), dtype=bool)

    player_ids = play.get("offense_player_ids", play.get("offense_ids", np.arange(n_off, dtype=object)))
    player_ids = [normalize_id(pid) for pid in list(player_ids)[:n_off]]
    positions = list(play.get("offense_positions", ["UNKNOWN"] * n_off))
    if len(positions) < n_off:
        positions.extend(["UNKNOWN"] * (n_off - len(positions)))
    positions = [str(pos) for pos in positions[:n_off]]

    snap_y = np.zeros(n_off, dtype=float)
    snap_x = np.zeros(n_off, dtype=float)
    release_y = np.zeros(n_off, dtype=float)
    release_frame = min(5, max(0, n_frames - 1))
    for idx in range(n_off):
        mask = offense_mask[idx, :n_frames].astype(bool)
        first = _first_valid(mask)
        rel = release_frame if mask[release_frame] else first
        snap_x[idx] = float(offense_data[idx, first, 0])
        snap_y[idx] = float(offense_data[idx, first, 1])
        release_y[idx] = float(offense_data[idx, rel, 1])

    los_x = float(play.get("los_x", np.nanmax(snap_x) if len(snap_x) else 0.0))
    formation_split_y = _formation_split_y(play, snap_y)
    side_order = assign_receiver_side_order(
        snap_y=snap_y,
        player_ids=player_ids,
        release_y=release_y,
        positions=positions,
        snap_x=snap_x,
        los_x=los_x,
        split_y=formation_split_y,
    )

    routes: list[ReceiverRoute] = []
    for idx, pid in enumerate(player_ids):
        mask = offense_mask[idx, :n_frames].astype(bool)
        x = _safe_array(offense_data[idx, :n_frames, 0], mask)
        y = _safe_array(offense_data[idx, :n_frames, 1], mask)
        speed_feat = _safe_array(offense_data[idx, :n_frames, 2], mask) if offense_data.shape[2] > 2 else np.zeros(n_frames)
        dir_feat = _safe_array(offense_data[idx, :n_frames, 4], mask) if offense_data.shape[2] > 4 else np.full(n_frames, np.nan)

        vx_fd = np.gradient(x, params.dt) if n_frames > 1 else np.zeros(n_frames)
        vy_fd = np.gradient(y, params.dt) if n_frames > 1 else np.zeros(n_frames)
        if np.nanmedian(speed_feat) > 0.1 and np.isfinite(dir_feat).any():
            rad = np.deg2rad(dir_feat)
            vx = np.sin(rad) * speed_feat
            vy = np.cos(rad) * speed_feat
        else:
            vx = vx_fd
            vy = vy_fd
        speed = np.sqrt(vx * vx + vy * vy)

        side, order = side_order[pid]
        route = ReceiverRoute(
            nfl_id=pid,
            slot_index=idx,
            side=side,
            order=order,
            position=positions[idx],
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            speed=speed,
            mask=mask,
            tags={},
        )
        route.tags = tag_route(route, los_x=los_x, params=params)
        route.tags["alignment_side"] = side
        route.tags["formation_split_y"] = formation_split_y
        routes.append(route)

    return routes


def tag_route(route: ReceiverRoute, los_x: float, params: SimParams) -> dict[str, Any]:
    """Return route-level and frame-level concept tags."""
    x = np.asarray(route.x, dtype=float)
    y = np.asarray(route.y, dtype=float)
    vx = np.asarray(route.vx, dtype=float)
    vy = np.asarray(route.vy, dtype=float)
    speed = np.asarray(route.speed, dtype=float)
    mask = np.asarray(route.mask, dtype=bool)
    if len(x) == 0:
        return {}

    first = _first_valid(mask)
    valid = np.flatnonzero(mask)
    last = int(valid[-1]) if len(valid) else first
    y0 = float(y[first])
    x0 = float(x[first])
    depth = x - float(los_x)
    depth_gain = x - x0
    max_depth = float(np.nanmax(depth[mask])) if np.any(mask) else 0.0
    max_depth_gain = float(np.nanmax(depth_gain[mask])) if np.any(mask) else 0.0

    release_side = _infer_release_side(x, y, mask, route.side)
    side_for_breaks = release_side if route.side == MIDDLE_SIDE else route.side
    outside_sign = outside_y_sign(side_for_breaks if side_for_breaks in (RIGHT_SIDE, LEFT_SIDE) else release_side)
    inside_sign = inside_y_sign(side_for_breaks if side_for_breaks in (RIGHT_SIDE, LEFT_SIDE) else release_side)
    signed_outside = (y - y0) * outside_sign
    signed_inside = (y - y0) * inside_sign
    max_outside = float(np.nanmax(signed_outside[mask])) if np.any(mask) else 0.0
    max_inside = float(np.nanmax(signed_inside[mask])) if np.any(mask) else 0.0
    total_lateral = abs(float(y[last] - y0))

    vertical_frame = mask & (
        ((depth >= params.vertical_min_depth) & (vx >= params.vertical_min_vx))
        | (depth_gain >= params.vertical_min_total_depth)
    )
    vertical_frame = _smooth_bool(vertical_frame, width=1)
    vertical = bool(
        max_depth >= params.vertical_min_total_depth
        or max_depth_gain >= params.vertical_min_total_depth
        or np.any(vertical_frame)
    )

    shallow_frame = mask & (depth <= 7.0) & (np.abs(y - y0) >= params.crosser_min_lateral * 0.45)
    shallow_cross = bool(
        np.any(shallow_frame)
        or (float(np.nanmax(depth[mask])) <= 7.5 if np.any(mask) else False)
        and total_lateral >= params.crosser_min_lateral
    )
    shallow_frame = _smooth_bool(shallow_frame, width=1)

    flat_frame = mask & (depth <= params.flat_max_depth + 1.0) & (signed_outside >= params.flat_min_lateral * 0.35)
    flat = bool(
        (float(np.nanmax(depth[mask])) <= params.flat_max_depth + 1.0 if np.any(mask) else False)
        and max_outside >= params.flat_min_lateral
    )
    flat_frame = _smooth_bool(flat_frame, width=1)

    depth_max = float(np.nanmax(depth[mask])) if np.any(mask) else 0.0
    out = bool(5.0 <= depth_max <= 12.5 and max_outside >= params.flat_min_lateral and not flat)
    dig_in = bool(8.0 <= depth_max <= 22.0 and max_inside >= 3.0 and vertical)
    corner = bool(vertical and depth_max >= 8.0 and max_outside >= 3.0)
    post = bool(vertical and depth_max >= 8.0 and max_inside >= 3.0)
    seam = bool(vertical and (route.order in {2, 3} or route.side == MIDDLE_SIDE))

    if len(speed) > 1:
        accel = np.gradient(speed, params.dt)
        heading = np.arctan2(vy, vx)
        heading_change = np.zeros_like(heading)
        heading_change[1:] = np.abs(_signed_angle_diff(heading[1:], heading[:-1]))
    else:
        accel = np.zeros_like(speed)
        heading_change = np.zeros_like(speed)

    break_candidates = np.flatnonzero(
        mask
        & (
            (heading_change >= math.radians(35.0))
            | (accel <= params.curl_decel_threshold)
        )
        & (np.arange(len(mask)) >= max(1, first + 1))
    )
    route_break_frame = int(break_candidates[0]) if len(break_candidates) else None
    route_break_frame_mask = np.zeros(len(mask), dtype=bool)
    if route_break_frame is not None:
        lo = max(0, route_break_frame - 1)
        hi = min(len(mask), route_break_frame + 2)
        route_break_frame_mask[lo:hi] = True

    late_valid = valid[max(0, int(len(valid) * 0.65)) :] if len(valid) else np.asarray([last])
    late_speed = float(np.nanmedian(speed[late_valid])) if len(late_valid) else float(speed[last])
    curl_sit = bool(
        5.0 <= depth_max <= 15.5
        and (
            np.any(accel[mask] <= params.curl_decel_threshold) if np.any(mask) else False
            or late_speed <= 1.6
        )
    )

    backfield = bool(
        x0 < float(los_x) - 1.0
        and (
            _position_is_rb(route.position)
            or route.side == MIDDLE_SIDE
            or abs(y0 - FIELD_MID_Y) <= 3.0
        )
    )
    early_end = min(len(mask), first + 8)
    back_fast_flat = bool(backfield and flat and np.any(flat_frame[first:early_end]))

    return {
        "release_side": release_side,
        "backfield": backfield,
        "vertical": vertical,
        "vertical_frame": vertical_frame,
        "seam": seam,
        "flat": flat,
        "flat_frame": flat_frame,
        "out": out,
        "shallow_cross": shallow_cross,
        "shallow_cross_frame": shallow_frame,
        "dig_in": dig_in,
        "curl_sit": curl_sit,
        "corner": corner,
        "post": post,
        "back_fast_flat": back_fast_flat,
        "route_break_frame": route_break_frame,
        "route_break_frame_mask": route_break_frame_mask,
        "max_depth": depth_max,
        "max_depth_gain": max_depth_gain,
        "max_outside": max_outside,
        "max_inside": max_inside,
        "total_lateral": total_lateral,
    }
