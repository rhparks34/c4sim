"""Stage 3: each play's tracks from the snap to the throw, as the arrays the play records are cut from.

python -m src.prep.tracks

Per play of the play table:
  frame_ids     every frame from the snap to the release where a back, receiver, quarterback or defensive back / linebacker is
                tracked (plays with fewer than 6 such frames are left out)
  offense       route-set players: the offense's WR / RB / FB / TE, ordered by id (as text)
  qb            the quarterback with the lowest id (all-NaN when none is tracked)
  defense       every defensive back, linebacker and lineman, ordered by id (stage 5 picks the modeled ones)
  each player:  data [T, 5] = x, y, s, a, dir in float32 (a frame missing after the first tracked one repeats the previous frame),
                mask [T]
  los_x         the furthest-forward route-set player at the snap (the line of scrimmage); qb_snap_x the QB's x at the snap
The 2021 and 2023 tracks are rounded to float32 before the field is turned (how the published runs' records were built).
-> $DATA_ROOT/prepared/tracks/{season}.pkl  ({game_play_id: entry})
"""
import pickle

import numpy as np
import pandas as pd

from src.paths import PREPARED
from src.prep.frames import oriented, read_frames
from src.prep.plays import read_plays

MIN_PLAY_FRAMES = 6
CHANNELS = ['x', 'y', 's', 'a', 'dir']


def general_position(code):
    """Roster code -> the coarse position the pipeline uses."""
    code = str(code).upper().strip()
    if code in {'CB', 'DB', 'FS', 'SS', 'S'}:
        return 'DB'
    if code in {'LB', 'ILB', 'MLB', 'OLB'}:
        return 'LB'
    if code in {'DE', 'DT', 'NT', 'DL', 'EDGE'}:
        return 'DL'
    if code in {'WR', 'RB', 'HB', 'FB', 'TE'}:
        return 'ROUTE'
    if code == 'QB':
        return 'QB'
    return 'OTHER'


def tracks_path(season):
    return PREPARED / 'tracks' / f'{season}.pkl'


def read_tracks(season):
    with open(tracks_path(season), 'rb') as f:
        return pickle.load(f)


def player_arrays(rows, frame_ids):
    """One player's rows -> data [T, 5] float32 (missing frames after the first tracked one repeat the previous frame), mask [T]."""
    T = len(frame_ids)
    data = np.full((T, 5), np.nan, dtype=np.float32)
    mask = np.zeros(T, dtype=np.float32)
    at = np.searchsorted(frame_ids, rows.frame_id.to_numpy())
    data[at] = rows[CHANNELS].to_numpy()
    mask[at] = 1.0
    last = np.maximum.accumulate(np.where(mask > 0, np.arange(T), -1))
    fill = (mask == 0) & (last >= 0)
    data[fill] = data[last[fill]]
    mask[fill] = 1.0
    return data, mask


def play_tracks(g):
    """g: one play's rows (sorted by id as text, then frame) -> the play's entry, or None when it has too few frames or no
    route-set player or no defensive back / linebacker."""
    core = g[g.general.isin(['DB', 'LB', 'ROUTE', 'QB'])]
    frame_ids = np.sort(core.frame_id.unique())
    if len(frame_ids) < MIN_PLAY_FRAMES:
        return None
    first = g.drop_duplicates('nfl_id')
    offense = first.nfl_id[first.side.eq('offense') & first.general.eq('ROUTE')].tolist()
    defense = first.nfl_id[first.side.eq('defense') & first.general.isin(['DB', 'LB', 'DL'])].tolist()
    qbs = first.nfl_id[first.side.eq('offense') & first.general.eq('QB')].tolist()
    if not offense or not first.general[first.nfl_id.isin(defense)].isin(['DB', 'LB']).any():
        return None
    g = g[g.frame_id.isin(frame_ids)]
    by_player = dict(tuple(g.groupby('nfl_id', sort=False)))

    def stack(ids):
        arrays = [player_arrays(by_player[i], frame_ids) for i in ids]
        return (np.stack([a for a, _ in arrays]) if arrays else np.zeros((0, len(frame_ids), 5), np.float32),
                np.stack([m for _, m in arrays]) if arrays else np.zeros((0, len(frame_ids)), np.float32))

    off_data, off_mask = stack(offense)
    def_data, def_mask = stack(defense)
    qb_data = stack(qbs[:1])[0] if qbs else np.full((1, len(frame_ids), 5), np.nan, dtype=np.float32)
    at_snap = g[g.nfl_id.isin(offense) & g.frame_id.eq(frame_ids[0])]
    los_x = float(at_snap.x.max()) if len(at_snap) else 0.0
    qb_snap_x = float(np.nanmean(qb_data[0, 0, 0])) if not np.isnan(qb_data[0, 0, 0]) else 0.0
    return dict(frame_ids=frame_ids.astype(np.int32), los_x=los_x, qb_snap_x=qb_snap_x,
                offense_ids=offense, offense_data=off_data, offense_mask=off_mask, qb_data=qb_data,
                defense_ids=defense, defense_data=def_data, defense_mask=def_mask,
                defense_general=first.set_index('nfl_id').general.reindex(defense).tolist(),
                defense_codes=first.set_index('nfl_id').position.reindex(defense).tolist())


def season_tracks(season, plays):
    p = plays[plays.season.eq(season)].set_index('game_play_id')
    f = read_frames(season, ['game_play_id', 'nfl_id', 'frame_id', 'side', 'position', 'going_left'] + CHANNELS, set(p.index))
    f = oriented(f.dropna(subset=['x', 'y']), float32_first=season in (2021, 2023))
    f['general'] = f.position.map(general_position)
    f = f[f.general.ne('OTHER')]
    f = f[(f.frame_id >= f.game_play_id.map(p.snap_frame)) & (f.frame_id <= f.game_play_id.map(p.release_frame))]
    f = f.sort_values(['game_play_id', 'nfl_id', 'frame_id'], kind='mergesort')
    out = {}
    for gp, g in f.groupby('game_play_id', sort=True):
        entry = play_tracks(g)
        if entry is not None:
            out[gp] = entry
    return out


def main():
    plays = read_plays()
    (PREPARED / 'tracks').mkdir(parents=True, exist_ok=True)
    for season in (2018, 2021, 2022, 2023):
        out = season_tracks(season, plays)
        with open(tracks_path(season), 'wb') as fh:
            pickle.dump(out, fh, protocol=4)
        wanted = int(plays.season.eq(season).sum())
        print(f'tracks {season}: {len(out):,} of {wanted:,} plays', flush=True)


if __name__ == '__main__':
    main()
