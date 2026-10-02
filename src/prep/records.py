"""Stage 7: one model record per play (the trainer's input format), cut to the play's window.

python -m src.prep.records

Per play: the window's frames of the route-set players, the QB and the modeled defenders (stage 5, ordered by id as text), the play's
coverage family and play action, each defender's coded-rule inputs and rule teacher (stage 6), and the model tensors for the
recorded and the mirrored (y -> 53.3 - y) view, 80 frames long:
  features [8, 80, 41]  per defender and frame: the five nearest receivers relative to his snap spot (offset, distance, velocity,
                        closing speed), the time fraction, his snap spot, the QB's offset and drop, his depth off the line, and four
                        play-level counts; target / target_abs [8, 80, 2] his displacement from the snap; masks; snap_xy [8, 2]
  route_context [5, 80, 4]  each route runner's depth past the line / 30, lateral offset from the middle / 26.65, velocity / 10
The feature code is verbatim from the research code (W55 compute_features_v4 / build_route_context_tensor / precompute).
-> $DATA_ROOT/prepared/records/records_NNN.pt (+ manifest.json with each shard's SHA-256)
"""
import hashlib
import json
import pickle

import numpy as np
import torch

from src.paths import PREPARED
from src.prep.defenders import read_defenders
from src.prep.plays import read_plays
from src.prep.rule_inputs import output_path as rule_inputs_path
from src.prep.tracks import read_tracks
from src.prep.windows import read_windows

FIELD_HEIGHT = 53.3
SRC_FEAT_IDX = {name: i for i, name in enumerate(['x', 'y', 's', 'a', 'dir'])}
MAX_RECEIVERS, MAX_DEFENDERS, MAX_ROUTE_RUNNERS = 5, 8, 5
N_PER_RECEIVER_FEATS, N_PLAY_STATE_FEATS, N_AGGREGATE_FEATS = 6, 7, 4
N_FEATURES = N_PER_RECEIVER_FEATS * MAX_RECEIVERS + N_PLAY_STATE_FEATS + N_AGGREGATE_FEATS
BASE_ROUTE_FEATS = 4
ROUTE_NORM_X, ROUTE_NORM_Y, ROUTE_NORM_V = 30.0, FIELD_HEIGHT / 2, 10.0
MAX_SEQ_LEN = 80
SHARD_PLAYS = 1000
# Coverage defenders are named by role: a defensive lineman who drops into coverage counts as a linebacker.
POSITION_NAMES = {'DB': 'DEFENSIVE_BACK', 'LB': 'LINEBACKER', 'DL': 'LINEBACKER'}


def records_dir():
    return PREPARED / 'records'


def compute_features_v4(
    offense_data, offense_mask,
    recv_vx, recv_vy, recv_accel,
    defender_snap_xy,
    qb_data, qb_snap_x, los_x,
    n_frames,
):
    """
    Compute per-frame features for one defender relative to their SNAP position.
    Early defender motion features are intentionally excluded for this BDB probe.
    """
    features = np.zeros((n_frames, N_FEATURES), dtype=np.float32)
    n_off = offense_data.shape[0]

    snap_x, snap_y = defender_snap_xy
    los_depth = snap_x - los_x

    for t in range(n_frames):
        feat_idx = 0

        # Group A: Per-receiver features (sorted by distance to snap position)
        recv_x = offense_data[:, t, SRC_FEAT_IDX["x"]]
        recv_y = offense_data[:, t, SRC_FEAT_IDX["y"]]
        recv_valid = offense_mask[:, t]

        dx_all = recv_x - snap_x
        dy_all = recv_y - snap_y
        dist_all = np.sqrt(dx_all**2 + dy_all**2)
        dist_all[recv_valid == 0] = 999.0

        sorted_indices = np.argsort(dist_all)

        for r in range(MAX_RECEIVERS):
            base = feat_idx + r * N_PER_RECEIVER_FEATS

            if r < n_off and recv_valid[sorted_indices[r]] > 0:
                idx = sorted_indices[r]
                dx = dx_all[idx]
                dy = dy_all[idx]
                dist = dist_all[idx]
                vx = recv_vx[idx, t]
                vy = recv_vy[idx, t]

                if dist > 0.1:
                    closing = -(dx * vx + dy * vy) / dist
                else:
                    closing = 0.0

                features[t, base + 0] = dx
                features[t, base + 1] = dy
                features[t, base + 2] = dist
                features[t, base + 3] = vx
                features[t, base + 4] = vy
                features[t, base + 5] = closing

        feat_idx += N_PER_RECEIVER_FEATS * MAX_RECEIVERS

        # Group C: Play state features (7)
        time_frac = t / max(n_frames - 1, 1)
        qb_x = float(qb_data[0, t, SRC_FEAT_IDX["x"]]) if not np.isnan(qb_data[0, t, 0]) else qb_snap_x
        qb_y = float(qb_data[0, t, SRC_FEAT_IDX["y"]]) if not np.isnan(qb_data[0, t, 1]) else FIELD_HEIGHT / 2
        qb_pocket_depth = qb_x - qb_snap_x

        features[t, feat_idx + 0] = time_frac
        features[t, feat_idx + 1] = snap_x
        features[t, feat_idx + 2] = snap_y
        features[t, feat_idx + 3] = qb_x - snap_x
        features[t, feat_idx + 4] = qb_y - snap_y
        features[t, feat_idx + 5] = qb_pocket_depth
        features[t, feat_idx + 6] = los_depth
        feat_idx += N_PLAY_STATE_FEATS

        # Group D: Aggregate threat features (4)
        valid_dists = dist_all[recv_valid > 0]
        n_nearby = int(np.sum(valid_dists < 12.0)) if len(valid_dists) > 0 else 0
        nearest_dist = float(np.min(valid_dists)) if len(valid_dists) > 0 else 30.0

        recv_depths = recv_x[recv_valid > 0] - los_x if np.sum(recv_valid) > 0 else np.array([0.0])
        mean_depth = float(np.mean(recv_depths))

        valid_accels = recv_accel[recv_valid > 0, t] if np.sum(recv_valid) > 0 else np.array([0.0])
        any_break = 1.0 if np.max(np.abs(valid_accels)) > 5.0 else 0.0

        features[t, feat_idx + 0] = n_nearby
        features[t, feat_idx + 1] = nearest_dist
        features[t, feat_idx + 2] = mean_depth
        features[t, feat_idx + 3] = any_break
        feat_idx += N_AGGREGATE_FEATS

    return features


def build_route_context_tensor(offense_data, offense_mask, route_vx, route_vy, los_x, max_seq_len, flip_y=False):
    """Route context for the route encoder: [MAX_ROUTE_RUNNERS, max_seq_len, 4] (depth, lateral, vx, vy; fixed normalization),
    receiver-slot mask [MAX_ROUTE_RUNNERS], per-frame validity [MAX_ROUTE_RUNNERS, max_seq_len]."""
    n_off = offense_data.shape[0]
    recv_x = offense_data[:, :, SRC_FEAT_IDX["x"]]
    recv_y = offense_data[:, :, SRC_FEAT_IDX["y"]]
    vx = route_vx.copy()
    vy = route_vy.copy()
    if flip_y:
        recv_y = FIELD_HEIGHT - recv_y
        vy = -vy
    route_depth = (recv_x - los_x) / ROUTE_NORM_X
    lateral_pos = (recv_y - FIELD_HEIGHT / 2) / ROUTE_NORM_Y
    vx_norm = vx / ROUTE_NORM_V
    vy_norm = vy / ROUTE_NORM_V
    route_feats = np.stack([route_depth, lateral_pos, vx_norm, vy_norm], axis=-1)  # [N_off, T, 4]
    for r in range(n_off):
        invalid = offense_mask[r] == 0
        route_feats[r, invalid] = 0.0
    T_sub = route_feats.shape[1]
    actual_len = min(T_sub, max_seq_len)
    n_routes = min(n_off, MAX_ROUTE_RUNNERS)
    route_padded = np.zeros((MAX_ROUTE_RUNNERS, max_seq_len, BASE_ROUTE_FEATS), dtype=np.float32)
    route_recv_mask = np.zeros(MAX_ROUTE_RUNNERS, dtype=np.float32)
    route_frame_mask = np.zeros((MAX_ROUTE_RUNNERS, max_seq_len), dtype=np.float32)
    route_padded[:n_routes, :actual_len] = route_feats[:n_routes, :actual_len]
    route_recv_mask[:n_routes] = 1.0
    route_frame_mask[:n_routes, :actual_len] = offense_mask[:n_routes, :actual_len]
    return np.nan_to_num(route_padded, nan=0.0), route_recv_mask, route_frame_mask


def _get_play_variant_arrays(play, flip=False):
    """Prepare base or y-flipped play arrays for feature precomputation."""
    if not flip:
        return (play["offense_data"], play["offense_mask"], play["defense_data"], play["defense_mask"], play["defense_snap_xy"],
                play["qb_data"], play["recv_vx"], play["recv_vy"], play["recv_accel"], play["route_vx"], play["route_vy"])
    offense_data = play["offense_data"].copy()
    defense_data = play["defense_data"].copy()
    defense_snap_xy = play["defense_snap_xy"].copy()
    qb_data = play["qb_data"].copy()
    offense_data[:, :, SRC_FEAT_IDX["y"]] = FIELD_HEIGHT - offense_data[:, :, SRC_FEAT_IDX["y"]]
    offense_data[:, :, SRC_FEAT_IDX["dir"]] = (180.0 - offense_data[:, :, SRC_FEAT_IDX["dir"]]) % 360.0
    dir_rad = np.deg2rad(offense_data[:, :, SRC_FEAT_IDX["dir"]])
    recv_vx = np.sin(dir_rad) * offense_data[:, :, SRC_FEAT_IDX["s"]]
    recv_vy = np.cos(dir_rad) * offense_data[:, :, SRC_FEAT_IDX["s"]]
    recv_accel = offense_data[:, :, SRC_FEAT_IDX["a"]]
    defense_data[:, :, SRC_FEAT_IDX["y"]] = FIELD_HEIGHT - defense_data[:, :, SRC_FEAT_IDX["y"]]
    defense_snap_xy[:, 1] = FIELD_HEIGHT - defense_snap_xy[:, 1]
    qb_data[:, :, SRC_FEAT_IDX["y"]] = FIELD_HEIGHT - qb_data[:, :, SRC_FEAT_IDX["y"]]
    return (offense_data, play["offense_mask"], defense_data, play["defense_mask"], defense_snap_xy, qb_data, recv_vx, recv_vy,
            recv_accel, recv_vx, recv_vy)


def _build_precomputed_variant(play, max_seq_len=MAX_SEQ_LEN, flip=False):
    """Build one cached variant (base or y-flipped) for a play."""
    (offense_data, offense_mask, defense_data, defense_mask, defense_snap_xy, qb_data, recv_vx, recv_vy, recv_accel, route_vx,
     route_vy) = _get_play_variant_arrays(play, flip=flip)
    n_def = play["n_def"]
    n_frames = play["n_frames"]
    actual_len = min(n_frames, max_seq_len)
    feat_padded = np.zeros((MAX_DEFENDERS, max_seq_len, N_FEATURES), dtype=np.float32)
    target_padded = np.zeros((MAX_DEFENDERS, max_seq_len, 2), dtype=np.float32)
    def_mask_padded = np.zeros(MAX_DEFENDERS, dtype=np.float32)
    frame_mask_padded = np.zeros(max_seq_len, dtype=np.float32)
    snap_xy_padded = np.zeros((MAX_DEFENDERS, 2), dtype=np.float32)
    def_frame_mask_padded = np.zeros((MAX_DEFENDERS, max_seq_len), dtype=np.float32)
    def_mask_padded[:n_def] = 1.0
    frame_mask_padded[:actual_len] = 1.0
    snap_xy_padded[:n_def] = defense_snap_xy[:n_def]
    defense_traj_sub = defense_data[:, :, :2]
    defense_mask_sub = defense_mask
    for d_idx in range(n_def):
        snap_xy = defense_snap_xy[d_idx]
        feats = compute_features_v4(offense_data, offense_mask, recv_vx, recv_vy, recv_accel, snap_xy, qb_data, play["qb_snap_x"],
                                    play["los_x"], n_frames)
        feat_padded[d_idx, :actual_len] = feats[:actual_len]
        target_padded[d_idx, :actual_len] = (defense_traj_sub[d_idx] - snap_xy)[:actual_len]
        valid_mask = defense_mask_sub[d_idx, :actual_len]
        def_frame_mask_padded[d_idx, :actual_len] = valid_mask
        invalid = valid_mask == 0
        feat_padded[d_idx, :actual_len][invalid] = 0.0
        target_padded[d_idx, :actual_len][invalid] = 0.0
    feat_padded = np.nan_to_num(feat_padded, nan=0.0)
    target_padded = np.nan_to_num(target_padded, nan=0.0)
    route_context, route_recv_mask, route_frame_mask = build_route_context_tensor(
        offense_data, offense_mask, route_vx, route_vy, play["los_x"], max_seq_len, flip_y=False)   # y-flip already applied
    return {"features": feat_padded, "target": target_padded, "target_abs": target_padded.copy(), "def_mask": def_mask_padded,
            "frame_mask": frame_mask_padded, "snap_xy": snap_xy_padded, "def_frame_mask": def_frame_mask_padded, "n_def": int(n_def),
            "n_frames": int(actual_len), "route_context": route_context, "route_recv_mask": route_recv_mask,
            "route_frame_mask": route_frame_mask}


def precompute_play_tensors(play, max_seq_len=MAX_SEQ_LEN):
    """Precompute base + y-flipped tensors used by the Dataset."""
    return {"precomputed_base": _build_precomputed_variant(play, max_seq_len=max_seq_len, flip=False),
            "precomputed_flip": _build_precomputed_variant(play, max_seq_len=max_seq_len, flip=True)}


def play_record(gp, entry, window, defenders, play, rule):
    """The record of one play (its tracks cut to the window, its modeled defenders in id order)."""
    w = window
    idx = [entry['defense_ids'].index(n) for n in defenders]
    offense_data = entry['offense_data'][:, :w]
    defense_data = entry['defense_data'][idx, :w]
    defense_mask = entry['defense_mask'][idx, :w]
    snap = np.zeros((len(idx), 2), dtype=np.float32)
    for i in range(len(idx)):
        valid = np.where(defense_mask[i] > 0)[0]
        if len(valid):
            snap[i] = defense_data[i, valid[0], :2]
    dir_rad = np.deg2rad(offense_data[:, :, SRC_FEAT_IDX["dir"]])
    recv_vx = np.sin(dir_rad) * offense_data[:, :, SRC_FEAT_IDX["s"]]
    recv_vy = np.cos(dir_rad) * offense_data[:, :, SRC_FEAT_IDX["s"]]
    game_id, play_id = gp.split('_')
    rec = dict(game_play_id=gp, game_id=game_id, play_id=play_id, frame_ids=entry['frame_ids'][:w], n_frames=int(w),
               offense_player_ids=np.asarray(entry['offense_ids'], dtype=object), offense_data=offense_data,
               offense_mask=entry['offense_mask'][:, :w], qb_data=entry['qb_data'][:, :w], qb_snap_x=entry['qb_snap_x'],
               los_x=entry['los_x'], defender_ids=np.asarray(defenders, dtype=object), defense_data=defense_data,
               defense_mask=defense_mask, defense_snap_xy=snap,
               defense_positions=[POSITION_NAMES[entry['defense_general'][j]] for j in idx],
               n_off=len(entry['offense_ids']), n_def=len(idx), recv_vx=recv_vx, recv_vy=recv_vy,
               recv_accel=offense_data[:, :, SRC_FEAT_IDX["a"]], route_vx=recv_vx, route_vy=recv_vy, n_route_feats=BASE_ROUTE_FEATS,
               coverage_family_idx=int(play.coverage_family_idx), play_action=int(play.play_action), pa_known=int(play.pa_known))
    rec.update(precompute_play_tensors(rec))
    rec.update(rule)
    return rec


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plays = read_plays().set_index('game_play_id')
    windows = read_windows().set_index('game_play_id')
    d = read_defenders(); d = d[d.modeled]
    modeled = {gp: sorted(map(str, g.nfl_id)) for gp, g in d.groupby('game_play_id')}
    with open(rule_inputs_path(), 'rb') as fh:
        rules = pickle.load(fh)
    out = records_dir(); out.mkdir(parents=True, exist_ok=True)
    shards, batch, n = [], {}, 0
    for season in (2018, 2021, 2022, 2023):
        for gp, entry in sorted(read_tracks(season).items()):
            if not windows.at[gp, 'kept'] or gp not in modeled:
                continue
            batch[gp] = play_record(gp, entry, int(windows.at[gp, 'window']), modeled[gp], plays.loc[gp], rules[gp])
            n += 1
            if len(batch) == SHARD_PLAYS:
                shards.append(_write(out, len(shards), batch)); batch = {}
    if batch:
        shards.append(_write(out, len(shards), batch))
    (out / 'manifest.json').write_text(json.dumps(dict(plays=n, max_seq_len=MAX_SEQ_LEN, shards=shards), indent=1))
    print(f'records: {n:,} plays in {len(shards)} shards', flush=True)


def _write(out, i, batch):
    path = out / f'records_{i:03d}.pt'
    torch.save(batch, path)
    return dict(filename=path.name, sha256=sha256(path), plays=len(batch))


if __name__ == '__main__':
    main()
