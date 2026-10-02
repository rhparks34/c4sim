"""Modeled completion probability at the actual pass frame: the real defense vs each output, scored by a frozen engine.

Engine per fold (evaluation only; levels from different engines are never pooled): fold 0 = an engine trained without the fold-0
games; folds 1-3 = engines trained with that fold's games held out. Each engine is a gradient-boosted classifier on the frame-level
receiver features of `completion_features`.
Scene: offense, quarterback and ball stay real. The output's coverage defenders replace the real ones over the whole play window
(positions held at their first/last value outside the rows the output covers; a defender the output lacks takes the static
baseline position); every other defender stays real. A replaced defender's velocity = backward difference of his smoothed track
(lam 512, ramp 12). Receivers = true route runners. A play is read when its scored window contains the pass frame (first
pass_forward / pass_shovel). Per play, the mean over its receivers; the table reports the mean over plays and output - real in
probability points, with a game bootstrap (multinomial, 5,000 draws).
"""
import os
import warnings
from multiprocessing import Pool

import joblib
import numpy as np
import pandas as pd

from src.eval.common import B_DRAWS, KEYS, OUTPUTS, SEED, in_mask, prediction_path, round_ci
from src.eval.completion_features import (build_frame_receiver_features, eligible_receiver_ids, infer_target_receiver,
                                          nearest_frame_rows, normalize_id, player_velocity)
from src.eval.smoothing import smooth_track
from src.eval.sources import baselines, completion_play_tables, play_meta, route_flags, tracks_2018
from src.paths import EVAL

DT = 0.1
OUT_COLUMNS = ['game_play_id', 'frame_id', 'rank', 'rec_id', 'is_target', 'ehcp']
_G = {}


def engine_path(fold):
    return EVAL / 'completion' / f'engine_fold{fold}.joblib'


def play_direction_of(play_df):
    pdir = play_df['play_direction']
    return pdir.dropna().iloc[0] if pdir.notna().any() else ''


def flip_canonical(xy, play_direction):
    """Prediction frame <-> raw tracking frame (an involution): x -> 120 - x, y -> 53.3 - y on plays going left."""
    out = np.array(xy, dtype=float)
    if str(play_direction).strip().lower() == 'left':
        out[..., 0] = 120.0 - out[..., 0]
        out[..., 1] = 53.3 - out[..., 1]
    return out


def smoothed_fd_velocity(x, y):
    sx, sy = smooth_track(np.array(x, dtype=float), np.array(y, dtype=float))
    vx = np.zeros_like(sx); vy = np.zeros_like(sy)
    if len(sx) >= 2:
        vx[1:] = np.diff(sx) / DT; vy[1:] = np.diff(sy) / DT
        vx[0], vy[0] = vx[1], vy[1]
    return vx, vy


def defender_ids(play_df):
    side = play_df['player_side'].astype(str).str.lower()
    return sorted({normalize_id(v) for v in play_df.loc[side.eq('defense'), 'nfl_id']})


def real_tracks(play_df, frames, ids):
    """Real positions in the prediction frame, {nfl_id: (len(frames), 2)}."""
    pdir = play_direction_of(play_df)
    ids = [normalize_id(i) for i in ids]
    sub = play_df[play_df['nfl_id'].isin(ids) & play_df['frame_id'].isin(frames)]
    out = {}
    for nid in ids:
        g = sub[sub['nfl_id'] == nid].drop_duplicates('frame_id').set_index('frame_id').reindex(frames)
        if g[['x', 'y']].isna().any().any():
            raise ValueError(f'{nid}: no raw row for some window frames')
        out[nid] = flip_canonical(g[['x', 'y']].to_numpy(dtype=float), pdir)
    return out


class PlayScene:
    """One play's frame rows, receivers and real-player velocities at the scored frames; scores any number of scenes."""

    def __init__(self, play_df, frames, score_frames):
        frames = [int(f) for f in frames]
        self.play_df = play_df; self.gp = str(play_df['game_play_id'].iloc[0]); self.frames = frames
        self.index = {f: i for i, f in enumerate(frames)}
        self.play_direction = play_direction_of(play_df)
        self.target = normalize_id(infer_target_receiver(play_df))
        self.score_frames = sorted({int(f) for f in score_frames})
        self._base = {}
        for fid in self.score_frames:
            receivers = eligible_receiver_ids(play_df, fid)
            if not receivers:
                continue
            rows = nearest_frame_rows(play_df, fid)
            rows['_vx'] = np.nan; rows['_vy'] = np.nan
            ids = rows['nfl_id'].map(normalize_id)
            for idx, row in rows.iterrows():
                if bool(row.get('is_football', False)):
                    continue
                vx, vy = player_velocity(play_df, ids[idx], fid, row)
                rows.at[idx, '_vx'] = vx; rows.at[idx, '_vy'] = vy
            is_def = rows['player_side'].astype(str).str.lower().eq('defense')
            self._base[fid] = (rows, ids, is_def, receivers)

    def features(self, defender_tracks, keep_real):
        keep = {normalize_id(k) for k in keep_real}; states = {}
        for key, xy in defender_tracks.items():
            nid = normalize_id(key); xy = np.asarray(xy, dtype=float)
            if xy.shape != (len(self.frames), 2) or not np.isfinite(xy).all():
                raise ValueError(f'{nid}: track must be finite with shape ({len(self.frames)}, 2)')
            raw = flip_canonical(xy, self.play_direction); vx, vy = smoothed_fd_velocity(raw[:, 0], raw[:, 1])
            states[nid] = (raw, vx, vy)
        in_scene_ids = set(states) | keep
        feats, meta = [], []
        for fid in self.score_frames:
            if fid not in self._base:
                continue
            rows, ids, is_def, receivers = self._base[fid]
            missing = set(states) - set(ids[is_def])
            if missing:
                raise ValueError(f'frame {fid}: supplied defenders without a tracked row {sorted(missing)}')
            scene = ~is_def | ids.isin(in_scene_ids)
            ghost = rows[scene].copy(); gids = ids[scene]; i = self.index[fid]
            for nid, (raw, vx, vy) in states.items():
                mask = (gids == nid).values
                ghost.loc[mask, 'x'] = float(raw[i, 0]); ghost.loc[mask, 'y'] = float(raw[i, 1])
                ghost.loc[mask, '_vx'] = float(vx[i]); ghost.loc[mask, '_vy'] = float(vy[i])
            for rid in receivers:
                f = build_frame_receiver_features(self.play_df, ghost, rid, fid)
                if f is None:
                    continue
                f['play_direction'] = self.play_direction
                feats.append(f); meta.append((self.gp, fid, fid - self.frames[0] + 1, rid, rid == self.target))
        return feats, meta

    def score_many(self, scenes):
        feats, meta = [], []
        for name, (tracks, keep) in scenes.items():
            f, m = self.features(tracks, keep); feats += f; meta += [(name, *x) for x in m]
        pipeline, cols = _G['engine']
        out = pd.DataFrame(meta, columns=['scene', *OUT_COLUMNS[:-1]])
        if not feats:
            out['ehcp'] = pd.Series(dtype=float); return out
        X = pd.DataFrame(feats)
        for col in cols:
            if col not in X.columns:
                X[col] = np.nan
        with warnings.catch_warnings():
            # the engine's imputer has no statistic for the two snap-dependent features and drops them; the warning is inert
            warnings.filterwarnings('ignore', message='Skipping features without any observed values')
            out['ehcp'] = pipeline.predict_proba(X[cols])[:, 1]
        return out


def _init(fold):
    art = joblib.load(engine_path(fold))
    _G['engine'] = (art['pipeline'], art['numeric_features'] + art['categorical_features'])
    _G['tab'] = completion_play_tables(fold)
    _G['meta'] = play_meta()
    rf = route_flags()
    _G['true'] = set(zip(rf.game_play_id[rf.found & rf.ran_route], rf.nfl_id[rf.found & rf.ran_route]))


def _fill(a):
    return pd.DataFrame(a).ffill().bfill().to_numpy()


def _work(arg):
    """One play: completion probability of every true route runner at the pass frame, real scene and each output's scene."""
    gp, wide, names, pass_fid = arg
    play = _G['tab'][gp]
    n = int(_G['meta'].at[gp, 'n_frames']); f0 = int(_G['meta'].at[gp, 'snap_frame']); frames = list(range(f0, f0 + n))
    assert int(wide.frame_index.max()) == n, (gp, wide.frame_index.max(), n)
    cov = sorted(set(wide.nfl_id))
    real = real_tracks(play, frames, cov)
    cov = [c for c in cov if c in real]
    others = [d for d in defender_ids(play) if d not in cov]
    scenes = {'real': (real, others)}
    for s in names:
        tr = {}
        for c, x in wide.groupby('nfl_id'):
            if c in cov:
                a = np.full((n, 2), np.nan); a[x.frame_index.to_numpy() - 1] = x[[f'{s}_x', f'{s}_y']].to_numpy(float); tr[c] = _fill(a)
        scenes[s] = (tr, others)
    out = PlayScene(play, frames, score_frames=[pass_fid]).score_many(scenes)
    return out[[(gp, str(r)) in _G['true'] for r in out.rec_id]]


def surfaces(fold, root):
    """Every defender-frame of the fold: real position and each output's (static baseline where the output has no row)."""
    b = baselines(fold); b['nfl_id'] = b.nfl_id.astype(float).astype(np.int64).astype(str)
    w = b[KEYS + ['game_id', 'frame_id', 'actual_x', 'actual_y']].copy()
    for name in OUTPUTS:
        d = pd.read_parquet(prediction_path(fold, name, root), columns=KEYS + ['candidate_x', 'candidate_y'])
        d['nfl_id'] = d.nfl_id.astype(float).astype(np.int64).astype(str); d['frame_index'] = d.frame_index.astype(np.int64)
        m = w[KEYS].merge(d, on=KEYS, how='left'); assert len(m) == len(w), name
        miss = m.candidate_x.isna().to_numpy()
        w[name + '_x'] = np.where(miss, b.static_x, m.candidate_x); w[name + '_y'] = np.where(miss, b.static_y, m.candidate_y)
    return w


def at_pass_rows(fold, root, procs=6):
    """One row per true route runner at the pass frame of every play read at the pass: completion probability per source."""
    w = surfaces(fold, root)
    meta = play_meta()
    plays = sorted(w.game_play_id.unique())
    t = tracks_2018(['game_play_id', 'f', 'event']); t = t[t.game_play_id.isin(plays)]
    pf = t[t.event.isin(['pass_forward', 'pass_shovel'])].groupby('game_play_id').f.min().reindex(plays)
    pfi, pfid = pf + 1, meta.snap_frame.reindex(pf.index) + pf
    mend = w[in_mask(w)].groupby('game_play_id').frame_index.max().reindex(plays)
    read = pfi.notna() & (mend >= pfi)
    tasks = [(gp, g.reset_index(drop=True), list(OUTPUTS), int(pfid[gp])) for gp, g in w.groupby('game_play_id') if read[gp]]
    with Pool(procs, initializer=_init, initargs=(fold,)) as pool:
        res = pool.map(_work, tasks, chunksize=2)
    long = pd.concat([r for r in res if len(r)], ignore_index=True)
    wide = long.pivot_table(index=['game_play_id', 'frame_id', 'rec_id', 'is_target'], columns='scene', values='ehcp').reset_index()
    wide.columns.name = None
    return wide.dropna(subset=['real'] + list(OUTPUTS))


def compare(tab):
    """tab: one row per play. -> mean of each source and source - real in probability points, game bootstrap."""
    games = tab.game_play_id.str.split('_').str[0].to_numpy()
    g, gi = np.unique(games, return_inverse=True)
    rng = np.random.default_rng(SEED)
    Wr = rng.multinomial(len(g), np.full(len(g), 1.0 / len(g)), size=B_DRAWS).astype(float)[:, gi]
    m = lambda q: (Wr @ q) / Wr.sum(1)
    x = tab['real'].to_numpy(float)
    res = {'n': len(tab), 'real': {'mean': round_ci(x.mean(), m(x))}}
    for s in OUTPUTS:
        y = tab[s].to_numpy(float); d = y - x
        res[s] = {'mean': round_ci(y.mean(), m(y)), 'bias_pp': round_ci(100 * d.mean(), 100 * m(d), 2)}
    return res


def score(root, folds=(0, 1, 2, 3), procs=None):
    procs = procs or min(6, os.cpu_count() or 1)
    out = {}
    for k in folds:
        ap = at_pass_rows(k, root, procs)
        out[k] = compare(ap.groupby('game_play_id')[['real'] + list(OUTPUTS)].mean().reset_index())
    return out
