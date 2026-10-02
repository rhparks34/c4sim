"""Stage 1: every season's player tracking in one table.

python -m src.prep.frames [--season S]

One row per tracked player and frame (the football's rows are dropped), coordinates as published plus `going_left`; `oriented()`
turns a play so the offense moves toward +x (x -> 120 - x, y -> 53.3 - y, +180 degrees on `o` and `dir` for plays that go left).
Columns:
  game_play_id, nfl_id, frame_id, side ('offense' / 'defense'), position (the roster code, e.g. CB, WR), x, y, s, a, o, dir, going_left,
  event (lower case, '' when none; 2023 files have no events), route (2018 only: the route the tracking file names),
  role (2023 only: Passer / Targeted Receiver / Other Route Runner / Defensive Coverage)
2022: dropback plays only (`isDropback`); the other seasons: every play in the tracking files.
-> $DATA_ROOT/prepared/frames/{2018,2021,2022,2023}.parquet
"""
import argparse

import numpy as np
import pandas as pd

from src.paths import PREPARED
from src.prep import raw

SEASONS = (2018, 2021, 2022, 2023)
FIELD_LENGTH, FIELD_WIDTH = 120.0, 53.3
TRACKING = ['gameId', 'playId', 'nflId', 'frameId', 'team', 'club', 'playDirection', 'x', 'y', 's', 'a', 'o', 'dir', 'event',
            'position', 'route']
TRACKING_2023 = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'play_direction', 'player_position', 'player_side', 'player_role', 'x',
                 'y', 's', 'a', 'o', 'dir']
COLUMNS = ['game_play_id', 'nfl_id', 'frame_id', 'side', 'position', 'x', 'y', 's', 'a', 'o', 'dir', 'going_left', 'event', 'route',
           'role']


def frames_path(season):
    return PREPARED / 'frames' / f'{season}.parquet'


def read_frames(season, columns=None, plays=None):
    """The stage-1 table of one season (optionally only some columns / plays)."""
    filters = [('game_play_id', 'in', sorted(plays))] if plays is not None else None
    return pd.read_parquet(frames_path(season), columns=columns, filters=filters)


def oriented(df, float32_first=False):
    """A copy of frame rows turned so the offense moves toward +x. float32_first: values rounded to float32 before and after
    turning (how the stored 2021 and 2023 play records were built: float32 columns, turned in float64)."""
    df = df.copy()
    left = df['going_left'].to_numpy()
    if float32_first:
        for c in ('x', 'y', 's', 'a', 'o', 'dir'):
            if c in df:
                df[c] = df[c].astype(np.float32).astype(np.float64)
    if 'x' in df:
        df['x'] = np.where(left, FIELD_LENGTH - df['x'], df['x'])
    if 'y' in df:
        df['y'] = np.where(left, FIELD_WIDTH - df['y'], df['y'])
    for angle in ('o', 'dir'):
        if angle in df:
            df[angle] = np.where(left, (df[angle] + 180.0) % 360.0, df[angle])
    if float32_first:
        for c in ('x', 'y', 'o', 'dir'):
            if c in df:
                df[c] = df[c].astype(np.float32).astype(np.float64)
    return df


def _ids(df, game, play, player):
    return (df[game].astype('int64').astype(str) + '_' + df[play].astype('int64').astype(str),
            df[player].astype('int64').astype(str))


def season_frames(season):
    plays = raw.read_plays(season)
    if season == 2023:
        parts = []
        for path in raw.tracking_files(season):
            t = raw.read_tracking(path, TRACKING_2023)
            t['game_play_id'], t['nfl_id'] = _ids(t, 'game_id', 'play_id', 'nfl_id')
            t['frame_id'] = t['frame_id'].astype('int32')
            t['side'] = t['player_side'].str.lower()
            t['position'] = t['player_position']
            t['event'] = ''
            t['route'] = None
            t['role'] = t['player_role']
            t['going_left'] = t['play_direction'].eq('left')
            parts.append(t[COLUMNS])
        return pd.concat(parts, ignore_index=True)
    plays['game_play_id'] = plays['gameId'] + '_' + plays['playId']
    possession = plays.set_index('game_play_id')['possessionTeam']
    keep = set(plays.loc[plays['isDropback'].eq('TRUE'), 'game_play_id']) if season == 2022 else None
    games = raw.read_games(season).set_index('gameId')
    players = raw.read_players(season)
    roster = players.drop_duplicates('nflId').set_index('nflId')['officialPosition' if season == 2021 else 'position']
    parts = []
    for path in raw.tracking_files(season):
        t = raw.read_tracking(path, TRACKING)
        t = t[t['nflId'].notna()].copy()
        t['game_play_id'], t['nfl_id'] = _ids(t, 'gameId', 'playId', 'nflId')
        if keep is not None:
            t = t[t['game_play_id'].isin(keep)].copy()
        team = (t['club'] if season == 2022 else t['team']).astype(str)
        if season == 2018:   # 2018 names teams home / away
            game = t['gameId'].astype('int64').astype(str)
            team = np.where(team.eq('home'), game.map(games['homeTeamAbbr']), game.map(games['visitorTeamAbbr']))
        t['side'] = np.where(np.asarray(team) == t['game_play_id'].map(possession).to_numpy(), 'offense', 'defense')
        on_roster = t['nfl_id'].map(roster)
        t['position'] = t['position'].fillna(on_roster) if 'position' in t else on_roster
        t['frame_id'] = t['frameId'].astype('int32')
        t['event'] = t['event'].fillna('').astype(str).str.lower()
        if 'route' not in t:
            t['route'] = None
        t['role'] = None
        t['going_left'] = t['playDirection'].eq('left')
        parts.append(t[COLUMNS])
    return pd.concat(parts, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, choices=SEASONS)
    args = ap.parse_args(argv)
    (PREPARED / 'frames').mkdir(parents=True, exist_ok=True)
    for season in [args.season] if args.season else SEASONS:
        df = season_frames(season)
        df.to_parquet(frames_path(season), index=False)
        print(f'frames {season}: {len(df):,} rows, {df.game_play_id.nunique():,} plays', flush=True)


if __name__ == '__main__':
    main()
