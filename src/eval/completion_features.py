"""Frame-level receiver features of the completion-probability engine (the feature set the engines were trained on).

Coordinates are raw Big Data Bowl 2021 tracking (10 Hz); velocities come from neighbouring frames or speed and direction.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

FIELD_WIDTH = 120.0
FIELD_HEIGHT = 53.3
ELIGIBLE_RECEIVER_GENERAL = {"WIDE_RECEIVER", "RUNNING_BACK", "TIGHT_END"}
PASS_ARRIVAL_EVENTS = {"pass_arrived", "pass_outcome_caught", "pass_outcome_incomplete", "pass_outcome_interception",
                       "pass_outcome_touchdown"}


def normalize_id(value) -> str:
    """Numeric-looking ids as stable strings (1004953.0 -> 1004953)."""
    if pd.isna(value):
        return ""
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def euclidean(x1, y1, x2, y2) -> float:
    if any(pd.isna(v) for v in (x1, y1, x2, y2)):
        return np.nan
    return float(math.hypot(float(x1) - float(x2), float(y1) - float(y2)))


def eligible_receiver_ids(play_df: pd.DataFrame, frame_id: int) -> list[str]:
    rows = play_df[play_df["frame_id"] == frame_id]
    rows = rows[(rows["player_side"] == "offense") & rows["general_position"].isin(ELIGIBLE_RECEIVER_GENERAL)]
    return [normalize_id(v) for v in rows["nfl_id"].tolist()]


def infer_target_receiver(play_df: pd.DataFrame) -> str:
    """Eligible receiver nearest the football at the first pass-arrival event ("" when none)."""
    ev = play_df["event"].fillna("").astype(str).str.lower()
    arrive = play_df.loc[ev.isin(PASS_ARRIVAL_EVENTS), "frame_id"]
    football = play_df[play_df["is_football"]]
    for fid in ([int(arrive.min())] if len(arrive) else []):
        ball = football[football["frame_id"] == fid]
        if ball.empty:
            continue
        bx, by = float(ball.iloc[0]["x"]), float(ball.iloc[0]["y"])
        recs = play_df[play_df["frame_id"] == fid]
        recs = recs[(recs["player_side"] == "offense") & recs["general_position"].isin(ELIGIBLE_RECEIVER_GENERAL)]
        if recs.empty:
            continue
        dist = np.hypot(recs["x"].astype(float) - bx, recs["y"].astype(float) - by)
        return normalize_id(recs.iloc[int(np.argmin(dist.values))]["nfl_id"])
    return ""


EPS = 1e-9
PRESSURE_SIGMAS = (1.0, 2.0, 3.0)
DEFAULT_PRESSURE_SIGMA = 2.0
LANE_SIGMA = 1.0


def first_non_null(df: pd.DataFrame, col: str, default=np.nan):
    if col not in df.columns:
        return default
    vals = df[col].dropna()
    return vals.iloc[0] if len(vals) else default


def player_row(frame_rows: pd.DataFrame, player_id: str) -> pd.Series | None:
    rows = frame_rows[frame_rows["nfl_id"] == normalize_id(player_id)]
    if rows.empty:
        return None
    return rows.iloc[0]


def nearest_frame_rows(play_df: pd.DataFrame, frame_id: int) -> pd.DataFrame:
    frames = np.asarray(sorted(play_df["frame_id"].dropna().unique()))
    if len(frames) == 0:
        return pd.DataFrame()
    nearest = int(frames[np.argmin(np.abs(frames - int(frame_id)))])
    rows = play_df[play_df["frame_id"] == nearest].copy()
    return rows.drop_duplicates("nfl_id", keep="first")


def offense_direction(play_direction) -> int:
    text = str(play_direction).strip().lower()
    return -1 if text == "left" else 1


def normalize_x(x, play_direction) -> float:
    if pd.isna(x):
        return np.nan
    x = float(x)
    return x if offense_direction(play_direction) == 1 else FIELD_WIDTH - x


def normalize_vx(vx, play_direction) -> float:
    if pd.isna(vx):
        return np.nan
    return float(vx) * offense_direction(play_direction)


def _numeric(row: pd.Series | None, col: str, default=np.nan) -> float:
    if row is None or col not in row.index or pd.isna(row[col]):
        return default
    return float(row[col])


def _point(row: pd.Series | None) -> tuple[float, float] | None:
    if row is None:
        return None
    x = _numeric(row, "x")
    y = _numeric(row, "y")
    if pd.isna(x) or pd.isna(y):
        return None
    return x, y


def _rows_by_side(frame_rows: pd.DataFrame, side: str) -> pd.DataFrame:
    if "player_side" not in frame_rows.columns:
        return pd.DataFrame()
    return frame_rows[frame_rows["player_side"].astype(str).str.lower().eq(side)].copy()


def quarterback_row(frame_rows: pd.DataFrame) -> pd.Series | None:
    if frame_rows.empty or "general_position" not in frame_rows.columns:
        return None
    offense = _rows_by_side(frame_rows, "offense")
    qb = offense[offense["general_position"].astype(str).str.upper().eq("QUARTERBACK")]
    if qb.empty and "position_code" in offense.columns:
        qb = offense[offense["position_code"].astype(str).str.upper().eq("QB")]
    if qb.empty:
        return None
    return qb.iloc[0]


def _velocity_from_dir(row: pd.Series | None) -> tuple[float, float]:
    if row is None:
        return np.nan, np.nan
    speed = _numeric(row, "s")
    direction = _numeric(row, "dir")
    if pd.isna(speed) or pd.isna(direction):
        return np.nan, np.nan
    radians = math.radians(direction)
    return float(speed * math.sin(radians)), float(speed * math.cos(radians))


def player_velocity(play_df: pd.DataFrame, player_id: str, frame_id: int, row: pd.Series | None = None) -> tuple[float, float]:
    """Velocity: the cached `_vx`/`_vy` when present, else central difference of adjacent frames, else speed/dir."""
    if row is not None and "_vx" in row.index and "_vy" in row.index:
        vx = row["_vx"]
        vy = row["_vy"]
        if not pd.isna(vx) and not pd.isna(vy):
            return float(vx), float(vy)

    player_id = normalize_id(player_id)
    p = play_df[play_df["nfl_id"].map(normalize_id).eq(player_id)].sort_values("frame_id")
    if len(p) >= 2:
        before = p[p["frame_id"] < frame_id].tail(1)
        after = p[p["frame_id"] > frame_id].head(1)
        if not before.empty and not after.empty:
            dt = float(after["timestamp"].iloc[0] - before["timestamp"].iloc[0])
            if abs(dt) > EPS:
                return (
                    float((after["x"].iloc[0] - before["x"].iloc[0]) / dt),
                    float((after["y"].iloc[0] - before["y"].iloc[0]) / dt),
                )
        prev = before
        current = p[p["frame_id"] == frame_id].head(1)
        if not prev.empty and not current.empty:
            dt = float(current["timestamp"].iloc[0] - prev["timestamp"].iloc[0])
            if abs(dt) > EPS:
                return (
                    float((current["x"].iloc[0] - prev["x"].iloc[0]) / dt),
                    float((current["y"].iloc[0] - prev["y"].iloc[0]) / dt),
                )
    return _velocity_from_dir(row)


def _unit_vector(dx: float, dy: float) -> tuple[float, float, float]:
    dist = math.hypot(dx, dy)
    if dist < EPS:
        return np.nan, np.nan, np.nan
    return dx / dist, dy / dist, dist


def _soft_pressure(distances: Iterable[float], sigma: float) -> float:
    vals = [d for d in distances if not pd.isna(d)]
    if not vals:
        return np.nan
    return float(np.sum(np.exp(-(np.asarray(vals) ** 2) / (2.0 * sigma**2))))


def _defender_distances(defense: pd.DataFrame, x: float, y: float) -> np.ndarray:
    if defense.empty or pd.isna(x) or pd.isna(y):
        return np.asarray([], dtype=float)
    return np.sqrt((defense["x"].astype(float).values - x) ** 2 + (defense["y"].astype(float).values - y) ** 2)


def _nearest_distances(defense: pd.DataFrame, x: float, y: float) -> tuple[float, float]:
    distances = np.sort(_defender_distances(defense, x, y))
    first = float(distances[0]) if len(distances) else np.nan
    second = float(distances[1]) if len(distances) > 1 else np.nan
    return first, second


def _nearest_defender_row(defense: pd.DataFrame, x: float, y: float) -> pd.Series | None:
    if defense.empty or pd.isna(x) or pd.isna(y):
        return None
    distances = _defender_distances(defense, x, y)
    if len(distances) == 0:
        return None
    return defense.iloc[int(np.argmin(distances))]


def _path_occlusion_features(
    defense: pd.DataFrame,
    qx: float,
    qy: float,
    rx: float,
    ry: float,
) -> dict:
    out = {
        "min_throw_lane_dist": np.nan,
        "throw_lane_pressure": np.nan,
        "occluding_defender_dist_to_receiver": np.nan,
        "occluding_defender_projection": np.nan,
    }
    if defense.empty or any(pd.isna(v) for v in (qx, qy, rx, ry)):
        return out

    vx = rx - qx
    vy = ry - qy
    denom = vx * vx + vy * vy
    if denom < EPS:
        return out

    dx = defense["x"].astype(float).values - qx
    dy = defense["y"].astype(float).values - qy
    projection = (dx * vx + dy * vy) / denom
    projection_clipped = np.clip(projection, 0.0, 1.0)
    closest_x = qx + projection_clipped * vx
    closest_y = qy + projection_clipped * vy
    perp_dist = np.sqrt((defense["x"].astype(float).values - closest_x) ** 2 + (defense["y"].astype(float).values - closest_y) ** 2)
    between = (projection >= 0.0) & (projection <= 1.0)

    if not np.any(between):
        return out

    lane_perp = perp_dist[between]
    lane_projection = projection[between]
    receiver_dist = np.sqrt((defense["x"].astype(float).values[between] - rx) ** 2 + (defense["y"].astype(float).values[between] - ry) ** 2)
    lane_pressure = np.exp(-(lane_perp**2) / (2.0 * LANE_SIGMA**2))
    best_idx = int(np.argmax(lane_pressure))

    out["min_throw_lane_dist"] = float(np.min(lane_perp))
    out["throw_lane_pressure"] = float(np.sum(lane_pressure))
    out["occluding_defender_dist_to_receiver"] = float(receiver_dist[best_idx])
    out["occluding_defender_projection"] = float(lane_projection[best_idx])
    return out


def build_frame_receiver_features(
    play_df: pd.DataFrame,
    frame_rows: pd.DataFrame,
    receiver_id: str,
    frame_id: int,
) -> dict | None:
    """Features of one candidate receiver at one frame (only information available at that frame)."""
    receiver_id = normalize_id(receiver_id)
    rec = player_row(frame_rows, receiver_id)
    if rec is None:
        return None

    play_direction = first_non_null(play_df, "play_direction", "")
    qb = quarterback_row(frame_rows)
    qb_point = _point(qb)
    if qb_point is None:
        qx = first_non_null(play_df, "x_event", np.nan)
        qy = first_non_null(play_df, "y_event", np.nan)
    else:
        qx, qy = qb_point

    rx = _numeric(rec, "x")
    ry = _numeric(rec, "y")
    if any(pd.isna(v) for v in (qx, qy, rx, ry)):
        return None

    defense = _rows_by_side(frame_rows, "defense")
    nearest_receiver, second_receiver = _nearest_distances(defense, rx, ry)
    nearest_passer, _ = _nearest_distances(defense, qx, qy)
    receiver_def = _nearest_defender_row(defense, rx, ry)
    passer_def = _nearest_defender_row(defense, qx, qy)
    receiver_def_distances = _defender_distances(defense, rx, ry)
    passer_def_distances = _defender_distances(defense, qx, qy)

    rec_vx, rec_vy = player_velocity(play_df, receiver_id, frame_id, rec)
    qb_id = normalize_id(qb["nfl_id"]) if qb is not None and "nfl_id" in qb.index else ""
    qb_vx, qb_vy = player_velocity(play_df, qb_id, frame_id, qb) if qb_id else (np.nan, np.nan)
    path_ux, path_uy, passer_receiver_dist = _unit_vector(rx - qx, ry - qy)

    rec_path_vel = np.nan
    rec_lateral_vel = np.nan
    qb_path_vel = np.nan
    qb_lateral_vel = np.nan
    qb_throw_path_angle_error = np.nan
    if not pd.isna(path_ux):
        rec_path_vel = rec_vx * path_ux + rec_vy * path_uy
        rec_lateral_vel = rec_vx * (-path_uy) + rec_vy * path_ux
        qb_path_vel = qb_vx * path_ux + qb_vy * path_uy
        qb_lateral_vel = qb_vx * (-path_uy) + qb_vy * path_ux
        qb_speed = math.hypot(qb_vx, qb_vy) if not pd.isna(qb_vx) and not pd.isna(qb_vy) else np.nan
        if qb_speed and not pd.isna(qb_speed) and qb_speed > EPS:
            cos_angle = float(np.clip(qb_path_vel / qb_speed, -1.0, 1.0))
            qb_throw_path_angle_error = math.degrees(math.acos(cos_angle))

    nearest_def_closing_speed = np.nan
    nearest_def_lateral_vel = np.nan
    if receiver_def is not None:
        def_id = normalize_id(receiver_def["nfl_id"])
        def_vx, def_vy = player_velocity(play_df, def_id, frame_id, receiver_def)
        def_to_rec_ux, def_to_rec_uy, _ = _unit_vector(rx - _numeric(receiver_def, "x"), ry - _numeric(receiver_def, "y"))
        if not pd.isna(def_to_rec_ux):
            rel_vx = def_vx - rec_vx
            rel_vy = def_vy - rec_vy
            nearest_def_closing_speed = rel_vx * def_to_rec_ux + rel_vy * def_to_rec_uy
            nearest_def_lateral_vel = rel_vx * (-def_to_rec_uy) + rel_vy * def_to_rec_ux

    nearest_rusher_closing_speed = np.nan
    if passer_def is not None and qb is not None:
        rush_id = normalize_id(passer_def["nfl_id"])
        rush_vx, rush_vy = player_velocity(play_df, rush_id, frame_id, passer_def)
        rush_to_qb_ux, rush_to_qb_uy, _ = _unit_vector(qx - _numeric(passer_def, "x"), qy - _numeric(passer_def, "y"))
        if not pd.isna(rush_to_qb_ux):
            nearest_rusher_closing_speed = (rush_vx - qb_vx) * rush_to_qb_ux + (rush_vy - qb_vy) * rush_to_qb_uy

    qx_dir = normalize_x(qx, play_direction)
    rx_dir = normalize_x(rx, play_direction)
    rec_vx_dir = normalize_vx(rec_vx, play_direction)
    qb_vx_dir = normalize_vx(qb_vx, play_direction)
    receiver_depth_from_qb = rx_dir - qx_dir if not pd.isna(qx_dir) and not pd.isna(rx_dir) else np.nan
    receiver_lateral_from_qb = ry - qy
    sideline_dist_receiver = min(ry, FIELD_HEIGHT - ry)
    endline_dist_receiver = FIELD_WIDTH - rx_dir if not pd.isna(rx_dir) else np.nan
    release_x_dir = qx_dir

    receiver_dist_before = np.nan   # needs the snap rows, which the engine never had (imputed)
    time_snap_to_frame = np.nan

    occlusion = _path_occlusion_features(defense, qx, qy, rx, ry)
    throw_lane_pressure = occlusion["throw_lane_pressure"]
    occlusion_weighted_separation = np.nan
    true_tight_window_score = np.nan
    if not pd.isna(nearest_receiver) and not pd.isna(throw_lane_pressure):
        occlusion_weighted_separation = nearest_receiver / (1.0 + throw_lane_pressure)
        close_window = math.exp(-(nearest_receiver**2) / (2.0 * DEFAULT_PRESSURE_SIGMA**2))
        true_tight_window_score = close_window * throw_lane_pressure

    features = {
        "time_snap_to_pass": time_snap_to_frame,
        "receiver_s_pass": _numeric(rec, "s"),
        "receiver_dir_pass": _numeric(rec, "dir"),
        "ball_receiver_x_pass": rx - qx,
        "ball_receiver_y_pass": ry - qy,
        "ball_receiver_dist_pass": passer_receiver_dist,
        "ball_def_pass_x_pass": _numeric(receiver_def, "x") - qx if receiver_def is not None else np.nan,
        "ball_def_pass_y_pass": _numeric(receiver_def, "y") - qy if receiver_def is not None else np.nan,
        "ball_def_pass_dist_pass": euclidean(_numeric(receiver_def, "x"), _numeric(receiver_def, "y"), qx, qy) if receiver_def is not None else np.nan,
        "separation_pass": nearest_receiver,
        "receiver_dist_before": receiver_dist_before,
        "passer_receiver_dist": passer_receiver_dist,
        "nearest_opponent_receiver": nearest_receiver,
        "second_nearest_opponent_receiver": second_receiver,
        "nearest_opponent_passer": nearest_passer,
        "sideline_dist_receiver": sideline_dist_receiver,
        "endline_dist_receiver": endline_dist_receiver,
        "release_x_dir": release_x_dir,
        "receiver_depth_from_qb": receiver_depth_from_qb,
        "receiver_lateral_from_qb": receiver_lateral_from_qb,
        "receiver_vx_dir": rec_vx_dir,
        "receiver_vy": rec_vy,
        "receiver_path_velocity": rec_path_vel,
        "receiver_lateral_velocity": rec_lateral_vel,
        "qb_s_pass": _numeric(qb, "s"),
        "qb_a_pass": _numeric(qb, "a"),
        "qb_vx_dir": qb_vx_dir,
        "qb_vy": qb_vy,
        "qb_path_velocity": qb_path_vel,
        "qb_lateral_velocity": qb_lateral_vel,
        "qb_throw_path_angle_error": qb_throw_path_angle_error,
        "nearest_defender_closing_speed": nearest_def_closing_speed,
        "nearest_defender_lateral_velocity": nearest_def_lateral_vel,
        "nearest_rusher_closing_speed": nearest_rusher_closing_speed,
        "defenders_within_2_receiver": float(np.sum(receiver_def_distances <= 2.0)) if len(receiver_def_distances) else np.nan,
        "defenders_within_3_receiver": float(np.sum(receiver_def_distances <= 3.0)) if len(receiver_def_distances) else np.nan,
        "defenders_within_5_receiver": float(np.sum(receiver_def_distances <= 5.0)) if len(receiver_def_distances) else np.nan,
        "pressure_x_distance": _soft_pressure(passer_def_distances, DEFAULT_PRESSURE_SIGMA) * passer_receiver_dist,
    }

    for sigma in PRESSURE_SIGMAS:
        suffix = str(sigma).replace(".", "_")
        features[f"soft_pressure_receiver_sigma_{suffix}"] = _soft_pressure(receiver_def_distances, sigma)
        features[f"soft_pressure_passer_sigma_{suffix}"] = _soft_pressure(passer_def_distances, sigma)
    features["soft_pressure_receiver"] = features["soft_pressure_receiver_sigma_2_0"]
    features["soft_pressure_passer"] = features["soft_pressure_passer_sigma_2_0"]

    features.update(occlusion)
    features["occlusion_weighted_separation"] = occlusion_weighted_separation
    features["true_tight_window_score"] = true_tight_window_score
    return features
