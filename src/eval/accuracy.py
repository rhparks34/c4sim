"""Position accuracy inside the scoring mask: mean error, median play, and each output's paired difference from the standard output.

mean error        coordinate RMSE over every scored defender-frame, sqrt(SSE / (2 * rows)), yd
median play       median over plays of each play's coordinate RMSE, yd
median frame      median over defender-frames of the Euclidean error sqrt(dx^2 + dy^2), yd (only in the paired differences)
"""
import numpy as np
import pandas as pd

from src.eval.common import KEYS, OUTPUTS, draw_counts, est, gindex, in_mask, prediction_path, wmedian_boot

METRICS = ['mean_rmse', 'median_play_rmse', 'median_euclid']


def fold_rows(fold, root):
    """Scored rows of one fold: game, play, squared error of each output. Returns (masked rows, the fold's sorted game ids)."""
    first = OUTPUTS[0]
    ref = pd.read_parquet(prediction_path(fold, first, root), columns=KEYS + ['game_id', 'actual_x', 'actual_y', 'candidate_x', 'candidate_y'])
    ref['nfl_id'] = ref.nfl_id.astype(str); ref['game_id'] = ref.game_id.astype(str)
    assert not ref.duplicated(KEYS).any()
    df = ref.rename(columns={'candidate_x': f'{first}_x', 'candidate_y': f'{first}_y'})
    for name in OUTPUTS[1:]:
        s = pd.read_parquet(prediction_path(fold, name, root), columns=KEYS + ['candidate_x', 'candidate_y'])
        s['nfl_id'] = s.nfl_id.astype(str); s['game_play_id'] = s.game_play_id.astype(str); s['frame_index'] = s.frame_index.astype(np.int64)
        m = df[KEYS].merge(s, on=KEYS, how='left', validate='one_to_one')
        df[f'{name}_x'] = m.candidate_x.to_numpy(); df[f'{name}_y'] = m.candidate_y.to_numpy()
    out = df[['game_id', 'game_play_id', 'nfl_id', 'frame_index']].copy(); out['fold'] = fold
    for name in OUTPUTS:
        out[f'se_{name}'] = (df[f'{name}_x'] - df.actual_x) ** 2 + (df[f'{name}_y'] - df.actual_y) ** 2
    strata = np.sort(out.game_id.unique())
    return out[in_mask(out)].reset_index(drop=True), strata


def _stats(d, name, sel, gid_all, games, W):
    se = d[f'se_{name}'].to_numpy()[sel]
    assert np.isfinite(se).all()
    pc, pu = pd.factorize(d.game_play_id.to_numpy()[sel])
    sse_p = np.bincount(pc, weights=se); n_p = np.bincount(pc).astype(float); rmse_p = np.sqrt(sse_p / (2 * n_p))
    eu = np.sqrt(se)
    point = dict(mean_rmse=float(np.sqrt(se.sum() / (2 * len(se)))), median_play_rmse=float(np.median(rmse_p)), median_euclid=float(np.median(eu)))
    if W is None:
        return point, None
    gi = gid_all[sel]; G = len(games); pg = np.zeros(len(pu), np.int64); pg[pc] = gi
    sse_g = np.bincount(gi, weights=se, minlength=G); n_g = np.bincount(gi, minlength=G).astype(float)
    boot = dict(mean_rmse=np.sqrt((W @ sse_g) / (2 * (W @ n_g))), median_play_rmse=wmedian_boot(rmse_p, pg, W),
                median_euclid=wmedian_boot(eu, gi, W))
    return point, boot


def summarize(d, strata, intervals=False):
    """d: scored rows of one fold or of several folds stacked; strata: each fold's game ids.
    -> {'plays', 'rows', 'defined': {output: point metrics}, 'delta': {output: {metric: est + CI}} (with intervals only)}."""
    games, W = draw_counts(strata) if intervals else (None, None)
    gid_all = gindex(games, d.game_id.to_numpy()) if intervals else None
    fin = {s: np.isfinite(d[f'se_{s}'].to_numpy()) for s in OUTPUTS}
    res = dict(rows=len(d), plays=int(d.game_play_id.nunique()), games=int(d.game_id.nunique()), defined={}, delta={})
    for s in OUTPUTS:
        res['defined'][s] = _stats(d, s, fin[s], gid_all, games, None)[0]
    if intervals:
        base = OUTPUTS[0]
        for s in OUTPUTS[1:]:
            sel = fin[s] & fin[base]
            a, ab = _stats(d, s, sel, gid_all, games, W); p, pb = _stats(d, base, sel, gid_all, games, W)
            res['delta'][s] = dict(rows=int(sel.sum()), plays=int(d.game_play_id[sel].nunique()),
                                   **{m: est(a[m] - p[m], ab[m] - pb[m]) for m in METRICS})
    return res


def score(root, folds=(0, 1, 2, 3)):
    """-> {fold: summary, 'pooled': summary with paired intervals}."""
    parts = {k: fold_rows(k, root) for k in folds}
    out = {k: summarize(parts[k][0], [parts[k][1]]) for k in folds}
    pooled = pd.concat([parts[k][0] for k in folds], ignore_index=True)
    assert pooled.groupby('game_id').fold.nunique().max() == 1
    out['pooled'] = summarize(pooled, [parts[k][1] for k in folds], intervals=True)
    return out
