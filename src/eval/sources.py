"""Every evaluation input the scorers read, one function per file, with where its data comes from.

All folds are 2018 games. The play-level files under DATA_ROOT/eval/plays and the completion play tables were built from the Big
Data Bowl 2021 tracking (2018 season); the coverage classifier and the completion models are evaluation-only assets. The baselines
are built per fold by src/eval/baselines.py, and the coverage defenders come from the prepared inputs (stage 5).
"""
import json

import numpy as np
import pandas as pd

from src.paths import EVAL, SCORING_MASK, fold_dir


def tracks_2018(columns):
    """2018 tracking after the snap: one row per player (and the ball) per frame; f = frames after the snap, side O / D / B."""
    # Kaggle: nfl-big-data-bowl-2021, week{N}.csv (https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data)
    return pd.read_parquet(EVAL / 'plays/tracks_2018.parquet', columns=columns)


def route_flags():
    """Per play and offensive player: found in tracking, ran a route, route name."""
    # Kaggle: nfl-big-data-bowl-2021, week{N}.csv `route` (https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data)
    rf = pd.read_parquet(EVAL / 'plays/route_flags.parquet'); rf['nfl_id'] = rf.nfl_id.astype(str)
    return rf


def play_meta():
    """Per play: frames in the window, snap frame, line of scrimmage, offensive player ids."""
    # Kaggle: nfl-big-data-bowl-2021, plays.csv and week{N}.csv (https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data)
    return pd.read_parquet(EVAL / 'plays/play_meta.parquet').set_index('game_play_id')


def coverage_defenders(plays):
    """{play: [coverage defender ids]}: the defenders the model predicts (stage 5's modeled defenders, in record order)."""
    from src.prep.defenders import read_defenders
    d = read_defenders(); d = d[d.modeled & d.game_play_id.isin(plays)].copy(); d['nfl_id'] = d.nfl_id.astype(str)
    return {gp: sorted(g.nfl_id) for gp, g in d.groupby('game_play_id')}


def reads_context():
    """Per Cover-4 play: field side, role -> defender id, #1/#2/#3 receivers' release classes and positions at 2.5 s."""
    # Kaggle: nfl-big-data-bowl-2021, week{N}.csv (https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data)
    return pd.read_parquet(EVAL / 'plays/reads_context.parquet')


def fold0_context_tracks():
    """Fold-0 tracking (offense toward +x) with events, team and position: the coverage classifier's input."""
    # Kaggle: nfl-big-data-bowl-2021, week{N}.csv, players.csv (https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data)
    ctx = pd.read_parquet(EVAL / 'plays/fold0_context_tracks.parquet')
    ctx['nfl_id_str'] = ctx.nflId.astype('Int64').astype(str)
    return ctx


def completion_play_tables(fold):
    """{play: tracking rows in the completion engine's schema (raw coordinates, 10 Hz)} for the fold's validation plays."""
    # Kaggle: nfl-big-data-bowl-2021, week{N}.csv, plays.csv, players.csv (https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data)
    table = pd.read_parquet(EVAL / 'completion' / f'play_tables_fold{fold}.parquet')
    return {gp: g.reset_index(drop=True) for gp, g in table.groupby('game_play_id', sort=False)}


def baselines(fold, columns=None):
    """Every defender-frame of the fold: real position and the static, constant-velocity, rule and analog baselines
    (src/eval/baselines.py)."""
    return pd.read_parquet(EVAL / 'baselines' / f'fold{fold}.parquet', columns=columns)


def validation_plays(fold):
    ids = pd.read_csv(fold_dir(fold) / 'play_ids.csv', dtype=str)
    return pd.Index(ids.game_play_id[ids.split.eq('validation')])


def score_windows():
    """game_play_id -> last scored frame_index for the 54 plays whose routes break down into a scramble."""
    return pd.read_csv(SCORING_MASK, dtype={'game_play_id': str}).set_index('game_play_id').score_window


def classifier_files(seed):
    """(state dict, temperature) of one classifier seed."""
    import torch
    state = torch.load(EVAL / f'classifier/seed{seed}.pt', map_location='cpu', weights_only=True)
    return state, json.loads((EVAL / f'classifier/seed{seed}_report.json').read_text())['temperature']


def quarters_plays(label_index):
    """Fold-0 plays whose real coverage label is Quarters (the classifier's population; label_index = Quarters' class index)."""
    labels = np.load(EVAL / 'classifier/seed0_fold0_posteriors.npz', allow_pickle=True)
    return set(labels['gp'][labels['y'] == label_index])
