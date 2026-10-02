"""Receiver openness at the throw, and the tightest-covered receiver, for the real defense and each output.

Openness of a receiver = distance (yd) to the nearest coverage defender of the source at the pass frame (the first pass_forward or
pass_shovel event; frame_index = frames after the snap + 1). Receivers are the true route runners. A play is read when its scored
window (after the scoring mask) still contains the pass frame. Tightest receiver = the smallest openness on the play, per source.
Means are over receivers (tightest: over plays); intervals resample games within each fold (multinomial, 5,000 draws).
"""
import numpy as np
import pandas as pd

from src.eval.common import B_DRAWS, KEYS, OUTPUTS, SEED, apply_mask, prediction_path
from src.eval.sources import baselines, route_flags, tracks_2018

SRC = ['real'] + list(OUTPUTS)


def receiver_tracks(plays):
    """True route runners' tracked positions (frame_index = frames after the snap + 1) and each play's pass frame_index."""
    t = tracks_2018(['game_play_id', 'nflId', 'f', 'side', 'event', 'x', 'y'])
    t = t[t.game_play_id.isin(plays)]
    rf = route_flags(); rf = rf[rf.game_play_id.isin(plays) & rf.found & rf.ran_route]
    passf = t[t.event.isin(['pass_forward', 'pass_shovel'])].groupby('game_play_id').f.min()
    o = t[t.side.eq('O')].copy(); o['nfl_id'] = o.nflId.astype(np.int64).astype(str)
    o = o.merge(rf[['game_play_id', 'nfl_id']], on=['game_play_id', 'nfl_id'])
    o['frame_index'] = o.f + 1
    return o[['game_play_id', 'nfl_id', 'frame_index', 'x', 'y']], passf + 1


def at_pass_rows(fold, root):
    """One row per true route runner at the pass frame of every play read at the pass: openness under each source."""
    b = apply_mask(baselines(fold, KEYS + ['actual_x', 'actual_y']))
    S = {'real': b[KEYS + ['actual_x', 'actual_y']].rename(columns={'actual_x': 'x', 'actual_y': 'y'})}
    for name in OUTPUTS:
        d = pd.read_parquet(prediction_path(fold, name, root), columns=KEYS + ['candidate_x', 'candidate_y'])
        d['nfl_id'] = d.nfl_id.astype(float).astype(np.int64).astype(str)
        S[name] = d.merge(b[KEYS], on=KEYS).rename(columns={'candidate_x': 'x', 'candidate_y': 'y'})
        assert len(S[name]) == len(b), name
    plays = set(b.game_play_id)
    recv, passfi = receiver_tracks(plays)
    wend = b.groupby('game_play_id').frame_index.max()
    pf = passfi.reindex(list(plays))
    read = pf[pf.notna() & (wend.reindex(pf.index) >= pf)]
    recv = recv[recv.frame_index.eq(recv.game_play_id.map(read))]
    out = None
    for name, d in S.items():
        d = d[d.frame_index.eq(d.game_play_id.map(read))]
        dd = d.groupby('game_play_id'); rows = []
        for gp, r in recv.groupby('game_play_id'):
            if gp not in dd.groups:
                continue
            g = dd.get_group(gp)
            dist = np.sqrt((g.x.to_numpy()[None, :] - r.x.to_numpy()[:, None]) ** 2 + (g.y.to_numpy()[None, :] - r.y.to_numpy()[:, None]) ** 2)
            rows.append(pd.DataFrame({'game_play_id': gp, 'rid': r.nfl_id.to_numpy(), 'frame_index': r.frame_index.to_numpy(),
                                      name: np.nanmin(dist, axis=1)}))
        t = pd.concat(rows)
        out = t if out is None else out.merge(t, on=['game_play_id', 'rid', 'frame_index'], how='outer')
    out = out.dropna(subset=SRC)
    out = out.sort_values(['game_play_id', 'rid']); out['game'] = out.game_play_id.str.split('_').str[0]; out['fold'] = fold
    return out


def _stat(df, games_sorted, fold_games):
    """Mean of each source and source - real, game bootstrap stratified by fold; each value [est, lo, hi] rounded to 4 decimals."""
    rng = np.random.default_rng(SEED); gi = np.searchsorted(games_sorted, df.game.to_numpy())
    W = np.zeros((B_DRAWS, len(games_sorted)))
    for f in fold_games:
        idx = np.searchsorted(games_sorted, f); W[:, idx] = rng.multinomial(len(f), np.full(len(f), 1 / len(f)), size=B_DRAWS)
    n = np.bincount(gi, minlength=len(games_sorted)).astype(float); o = {}
    for s in SRC:
        sg = np.bincount(gi, weights=df[s].to_numpy(), minlength=len(games_sorted)); o[s] = (sg.sum() / n.sum(), (W @ sg) / (W @ n))
    ci = lambda e, b: [round(float(e), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
    r = {s: ci(*o[s]) for s in SRC}
    r.update({f'{s}_minus_real': ci(o[s][0] - o['real'][0], o[s][1] - o['real'][1]) for s in SRC[1:]})
    r['n'] = len(df); r['plays'] = int(df.game_play_id.nunique()); r['games'] = int(len(np.unique(df.game)))
    return r


def score(root, folds=(0, 1, 2, 3)):
    """-> {fold or 'pooled': {'runners': stats, 'tightest': stats}}."""
    rows = {k: at_pass_rows(k, root) for k in folds}
    res = {}
    for name, parts in [(k, [k]) for k in folds] + [('pooled', list(folds))]:
        d = pd.concat([rows[k] for k in parts], ignore_index=True)
        fg = [np.array(sorted(rows[k].game.unique())) for k in parts]
        order = np.concatenate(fg); assert len(set(order)) == len(order), 'a game in two folds'
        games_sorted = np.array(sorted(order)); fg = [games_sorted[np.isin(games_sorted, f)] for f in fg]
        ti = d.groupby(['fold', 'game', 'game_play_id'])[SRC].min().reset_index()
        res[name] = dict(runners=_stat(d, games_sorted, fg), tightest=_stat(ti, games_sorted, fg))
    return res
