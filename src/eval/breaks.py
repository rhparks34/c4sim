"""Separation one second after a receiver's cut: how far the covering defender is from the receiver, real defense vs each output.

Cut (route break), from the real receiver track (10 Hz, heading = differences of 3-frame-smoothed positions):
  turn  speed >= 3 yd/s over the 4 frames before and frames 2-5 after frame tb, and the heading turns >= 40 degrees, or
  stop  speed >= 4 yd/s before and <= 2.5 yd/s after (hitch, curl, comeback).
Cuts are taken largest turn first, each >= 10 frames from every cut already taken, on true route runners with a complete track.
Covering defender = the coverage defender nearest the receiver (REAL positions) over the follow window tb..te, te = min(tb + 10,
last frame), if his mean distance is <= 6 yd and the source defines him over the whole window. Separation at te = distance from
the receiver; a "full" cut has a whole second to follow (te - tb = 10 frames), the table's population. The scoring mask is not
applied here: the cut and its covering defender come from tracking. Intervals: game bootstrap stratified by fold.
"""
import numpy as np
import pandas as pd

from src.eval.common import OUTPUTS, draw_counts, gindex, prediction_path, wmean_boot
from src.eval.sources import coverage_defenders, play_meta, route_flags, tracks_2018, validation_plays


def _arr(g, n, ids, xcol='x', ycol='y', fcol='f'):
    a = np.full((len(ids), n, 2), np.nan)
    for i, nid in enumerate(ids):
        t = g[g['_id'] == nid]
        ff = t[fcol].to_numpy().astype(int); ok = (ff >= 0) & (ff < n)
        a[i, ff[ok]] = t[[xcol, ycol]].to_numpy()[ok]
    return a


def _heading(p):
    """p [n, 2] -> velocity [n, 2]: central difference of 3-frame smoothed positions (one-sided at the ends)."""
    q = p.copy()
    for k in (0, 1):
        q[:, k] = pd.Series(p[:, k]).rolling(3, center=True, min_periods=1).mean().to_numpy()
    v = np.full_like(q, np.nan); v[1:-1] = (q[2:] - q[:-2]) / 0.2; v[0] = (q[1] - q[0]) / 0.1; v[-1] = (q[-1] - q[-2]) / 0.1
    return v


def _ang(a, b):
    return np.degrees(np.abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


class Fold:
    """Real tracks of one fold's validation plays: receivers (true route runners) and coverage defenders."""

    def __init__(self, fold):
        self.meta = play_meta()
        self.val = val = validation_plays(fold)
        self.cov = coverage_defenders(val)
        tr = tracks_2018(['game_play_id', 'nflId', 'f', 'side', 'position', 'x', 'y'])
        tr = tr[tr.game_play_id.isin(val)].drop_duplicates(['game_play_id', 'nflId', 'f'])
        rfl = route_flags(); rfl = rfl[rfl.game_play_id.isin(val)]
        self.true = set(zip(rfl.game_play_id[rfl.found & rfl.ran_route], rfl.nfl_id[rfl.found & rfl.ran_route]))
        self.trg = {gp: g.assign(_id=g.nflId) for gp, g in tr.groupby('game_play_id')}
        self.games = None

    def cuts(self, pred):
        """One row per cut the source covers: play, receiver, cut frame, follow frames, real and source separation at te."""
        pr = pred.copy(); pr['nfl_id'] = pr.nfl_id.astype(str)
        prg = {gp: g for gp, g in pr.groupby('game_play_id')}
        BK, games = [], set()
        for gp in self.val:
            n = int(self.meta.at[gp, 'n_frames']); cids = self.cov.get(gp, [])
            g = self.trg[gp]; p = prg.get(gp)
            rids = [r for r in self.meta.at[gp, 'offense_ids'] if (g._id == r).any() and (gp, r) in self.true]
            R = _arr(g, n, rids); Dr = _arr(g, n, cids)
            Dp = np.full_like(Dr, np.nan)
            if p is not None:
                pp = p.assign(_id=p.nfl_id, _f=p.frame_index - 1)
                have = [c for c in cids if (pp._id == c).any()]
                Dp_h = _arr(pp, n, have, 'candidate_x', 'candidate_y', '_f')
                for i, c in enumerate(cids):
                    if c in have:
                        Dp[i] = Dp_h[have.index(c)]
            if not len(rids) or not len(cids):
                continue
            dr = np.linalg.norm(R[:, None] - Dr[None], axis=-1)
            dr = np.where(np.isfinite(dr), dr, np.inf)
            if (np.isfinite(R[:, :, 0]) & np.isfinite(dr.min(1))).any():
                games.add(gp.split('_')[0])
            for j, r in enumerate(rids):
                P = R[j]
                if np.isnan(P).any():
                    continue
                v = _heading(P); sp = np.linalg.norm(v, axis=1)
                cand = []
                for tb in range(5, n - 8):
                    a0 = np.arctan2(v[tb - 4:tb, 1].mean(), v[tb - 4:tb, 0].mean()); a1 = np.arctan2(v[tb + 2:tb + 6, 1].mean(), v[tb + 2:tb + 6, 0].mean())
                    if sp[tb - 4:tb].min() >= 3 and sp[tb + 2:tb + 6].min() >= 3 and _ang(a0, a1) >= 40:
                        cand.append((_ang(a0, a1), tb, a1))
                    elif sp[tb - 4:tb].min() >= 4 and sp[tb + 2:tb + 6].max() <= 2.5:
                        a1s = a1 if sp[tb + 2:tb + 6].mean() >= 1 else a0 + np.pi
                        cand.append((90.0, tb, a1s))
                accepted = []
                for dth, tb, a1 in sorted(cand, reverse=True):
                    if any(abs(tb - b) < 10 for b in accepted):
                        continue
                    te = min(tb + 10, n - 1)
                    md = np.nanmean(dr[j, :, tb:te + 1], axis=1); md = np.where(np.isfinite(md), md, np.inf); di = int(np.argmin(md))
                    if md[di] > 6 or np.isnan(Dp[di, tb:te + 1]).any() or np.isnan(Dr[di, tb:te + 1]).any():
                        continue
                    accepted.append(tb)
                    BK.append((gp, r, tb + 1, te - tb, float(np.linalg.norm(Dr[di][te] - P[te])), float(np.linalg.norm(Dp[di][te] - P[te]))))
        self.games = np.sort(np.array(sorted(games), dtype=object))
        return pd.DataFrame(BK, columns=['game_play_id', 'rec_id', 'break_frame', 'follow', 'sep1_real', 'sep1_pred'])


def fold_cuts(fold, root):
    """Every cut of the fold with the real separation and each output's (NaN where the output does not cover it)."""
    F = Fold(fold); wide = None
    for s in OUTPUTS:
        pred = pd.read_parquet(prediction_path(fold, s, root), columns=['game_play_id', 'nfl_id', 'frame_index', 'candidate_x', 'candidate_y'])
        b = F.cuts(pred).rename(columns={'sep1_pred': f'sep1_{s}'})
        if wide is None:
            wide = b
        else:
            k = ['game_play_id', 'rec_id', 'break_frame']
            m = wide[k + ['follow', 'sep1_real']].merge(b[k + ['follow', 'sep1_real']], on=k)
            assert len(m) == len(b) and (m.sep1_real_x == m.sep1_real_y).all(), (fold, s)   # an output may lose cuts, never add them
            wide = wide.merge(b[k + [f'sep1_{s}']], on=k, how='left', validate='one_to_one')
    wide['game_id'] = wide.game_play_id.str.split('_').str[0]
    return wide[wide.follow.eq(10)].reset_index(drop=True), F.games


def summarize(wide, strata, intervals=False):
    """wide: full cuts; -> counts, mean real and per-output separation, and (with intervals) output - real with a 95% CI."""
    res = dict(n=len(wide), plays=int(wide.game_play_id.nunique()), games=int(wide.game_id.nunique()),
               real=float(np.mean(wide.sep1_real.to_numpy(float))), sources={}, minus_real={})
    if intervals:
        games, W = draw_counts(strata); gi = gindex(games, wide.game_id.to_numpy())
    for s in OUTPUTS:
        d = wide[f'sep1_{s}'].notna().to_numpy()
        res['sources'][s] = float(np.mean(wide[f'sep1_{s}'].to_numpy(float)[d]))
        if intervals:
            e = (wide[f'sep1_{s}'].to_numpy() - wide.sep1_real.to_numpy())[d]
            b = wmean_boot(e, gi[d], W); b = b[np.isfinite(b)]
            res['minus_real'][s] = [float(np.mean(e)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return res


def score(root, folds=(0, 1, 2, 3)):
    parts = {k: fold_cuts(k, root) for k in folds}
    out = {k: summarize(parts[k][0], [parts[k][1]]) for k in folds}
    out['pooled'] = summarize(pd.concat([parts[k][0] for k in folds], ignore_index=True), [parts[k][1] for k in folds], intervals=True)
    return out
