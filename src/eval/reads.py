"""Quarters reads at 2.5 s after the snap: does the defender move the way the coverage rule says, given what the receiver did?

Roles and receiver numbering come from the play table (`reads_context.parquet`: per Cover-4 play, the side of the field, each
role's defender id and the #1/#2/#3 receivers' positions 2.5 s after the snap). A receiver's release is classed from his
displacement over 2.5 s (dx = depth gained, du = width gained toward the sideline):
  #2 read   #2 receiver VERT (dx >= 12, |du| <= 4) or FLAT (dx <= 6, du >= 4); the read's safety (S) goes 5+ yd deep.
  Solo read on a three-receiver side, #3 VERT (dx >= 12, |du| <= 6) or SHORT (dx <= 6); the backside safety moves 2+ yd toward
            the trips side.
A defender's movement = his position at 2.5 s minus the REAL snap position. A play counts when every source places the read's
defenders at 2.5 s inside the scoring mask. Rate = share of plays with the move, per receiver class.
"""
import numpy as np
import pandas as pd

from src.eval.common import B_DRAWS, OUTPUTS, SEED, apply_mask, prediction_path, round_ci
from src.eval.sources import baselines, reads_context

TS = [5, 10, 15, 20, 25]; T = 25
ROLES = ['C', 'S', 'O', 'M', 'BS', 'BC', 'W']
FRAMES = [1] + [t + 1 for t in TS]
BASELINES = ['static', 'cv', 'rules', 'analogs']
READS = {  # id: (receiver class column, class a, class b, quantity, threshold); every read draws its own bootstrap, in this order
    'read_2': ('c2', 'VERT', 'FLAT', 'S_dep', 5.0),
    'read_2_outside_width': ('c2', 'VERT', 'FLAT', 'O_out', 3.0),
    'read_2_safety_width': ('c2', 'FLAT', 'VERT', 'S_out', 2.0),
    'solo_middle_depth': ('c3', 'VERT', 'SHORT', 'M_dep', 6.0),
    'read_solo': ('c3', 'VERT', 'SHORT', 'BS_out', 2.0),
}
TABLE_READS = ('read_2', 'read_solo')   # the two reads in the table
NEED = {'c2': ['S', 'O', 'C'], 'c3': ['M', 'BS']}


def sources(fold, root):
    """Positions at the snap and every 0.5 s to 2.5 s: real, the baselines (they fix which plays are read) and each output."""
    b = baselines(fold)
    b = b[b.frame_index.isin(FRAMES)]
    src = {'real': b[['game_play_id', 'nfl_id', 'frame_index', 'actual_x', 'actual_y']].rename(columns={'actual_x': 'x', 'actual_y': 'y'})}
    for c in BASELINES:
        src[c] = b[['game_play_id', 'nfl_id', 'frame_index', c + '_x', c + '_y']].rename(columns={c + '_x': 'x', c + '_y': 'y'}).dropna()
    for name in OUTPUTS:
        d = pd.read_parquet(prediction_path(fold, name, root), columns=['game_play_id', 'nfl_id', 'frame_index', 'candidate_x', 'candidate_y'])
        d['nfl_id'] = d.nfl_id.astype(float).astype(np.int64).astype(str)
        src[name] = d[d.frame_index.isin(FRAMES)].rename(columns={'candidate_x': 'x', 'candidate_y': 'y'})
    return {k: apply_mask(v) for k, v in src.items()}


def per_play(ctx, src_df, snap_df):
    """Per play: each role's depth and outward movement from the real snap spot at every 0.5 s, for one source."""
    pos = {(g, n, f): (x, y) for g, n, f, x, y in zip(src_df.game_play_id, src_df.nfl_id, src_df.frame_index, src_df.x, src_df.y)}
    snap = {(g, n): (x, y) for g, n, x, y in zip(snap_df.game_play_id, snap_df.nfl_id, snap_df.x, snap_df.y)}
    out = []
    for r in ctx.itertuples(index=False):
        row = {'gp': r.gp}
        for role in ROLES:
            nid = getattr(r, f'{role}_id', None)
            if not isinstance(nid, str) or (r.gp, nid) not in snap:
                continue
            x0, y0 = snap[(r.gp, nid)]
            for t in TS:
                v = pos.get((r.gp, nid, t + 1))
                if v is None or not np.isfinite(v[0]):
                    continue
                row[f'{role}_dep{t}'] = v[0] - x0; row[f'{role}_out{t}'] = r.side * (v[1] - y0)
        out.append(row)
    return pd.DataFrame(out)


def _gsum(df, games, col):
    g = df.groupby('game')[col].agg(['sum', 'count']).reindex(games).fillna(0.)
    return g['sum'].to_numpy(), g['count'].to_numpy()


def score_population(src):
    """src: {source: rows} for one fold or several stacked. -> {read: {'n_plays', 'n_games', source: {'a', 'b', 'diff'}}}."""
    rng = np.random.default_rng(SEED)
    ctx = reads_context()
    plays = set(src['real'].game_play_id); ctx = ctx[ctx.gp.isin(plays)].copy()
    snap = src['real'][src['real'].frame_index.eq(1)]
    pp = {k: per_play(ctx, v, snap).set_index('gp') for k, v in src.items()}
    res = {}
    for rid, (ccol, a, b_, q, thr) in READS.items():
        need = [f'{r}_dep{T}' for r in NEED[ccol]]
        base = ctx[ctx[ccol].isin([a, b_])].set_index('gp')
        ok = pd.Series(True, index=base.index)
        for k in pp:
            ok &= base.index.isin(pp[k].dropna(subset=[c for c in need if c in pp[k]]).index) & all(c in pp[k] for c in need)
        base = base[ok]
        games = np.array(sorted(base.game.unique())); G = len(games)
        W = rng.multinomial(G, np.full(G, 1.0 / G), size=B_DRAWS).astype(np.float64)
        if rid not in TABLE_READS:
            continue
        r = dict(n_plays={a: int((base[ccol] == a).sum()), b_: int((base[ccol] == b_).sum())}, n_games=int(len(games)), classes=(a, b_))
        for k in ['real'] + list(OUTPUTS):
            d = pp[k].reindex(base.index); d['game'] = base.game; d['cls'] = base[ccol]
            d['ev'] = (d[f'{q}{T}'] >= thr).astype(float)
            ra, rna = _gsum(d[d.cls == a], games, 'ev'); rb, rnb = _gsum(d[d.cls == b_], games, 'ev')
            ea, ba = ra.sum() / rna.sum(), (W @ ra) / (W @ rna); eb, bb = rb.sum() / rnb.sum(), (W @ rb) / (W @ rnb)
            r[k] = dict(a=round_ci(ea, ba[np.isfinite(ba)]), b=round_ci(eb, bb[np.isfinite(bb)]), diff=round_ci(ea - eb, (ba - bb)[np.isfinite(ba - bb)]))
        res[rid] = r
    return res


def score(root, folds=(0, 1, 2, 3)):
    src = {k: sources(k, root) for k in folds}
    out = {k: score_population(src[k]) for k in folds}
    out['pooled'] = score_population({s: pd.concat([src[k][s] for k in folds], ignore_index=True) for s in src[folds[0]]})
    return out
