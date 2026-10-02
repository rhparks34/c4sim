"""Pieces every scorer shares: where prediction files live, the route-breakdown scoring mask, and the game-cluster bootstrap.

Every interval in the table resamples whole games (a play's defenders and frames move together), 5,000 draws, seed 20260928.
"""
import numpy as np

from src.eval.sources import score_windows
from src.paths import OUT_ROOT

OUTPUTS = ('standard', 'route_break', 'midpoint')
KEYS = ['game_play_id', 'nfl_id', 'frame_index']
B_DRAWS, SEED = 5000, 20260928
PREDICTIONS = OUT_ROOT / 'predictions'


def prediction_path(fold, output, root=PREDICTIONS):
    return root / f'fold{fold}' / f'{output}.parquet'


def in_mask(df, frame_col='frame_index'):
    """True for rows that stay in the metric (frame_index is 1-based after the snap)."""
    lim = df.game_play_id.map(score_windows())
    return (lim.isna() | (df[frame_col] <= lim)).to_numpy()


def apply_mask(df, frame_col='frame_index'):
    return df[in_mask(df, frame_col)]


def draw_counts(strata, B=B_DRAWS, seed=SEED):
    """strata: one array of game ids per fold -> (games, W [B, G] resample counts); games are resampled within their fold."""
    rng = np.random.default_rng(seed)
    games = np.concatenate([np.asarray(s, dtype=object) for s in strata]); W = np.zeros((B, len(games)), np.int32); off = 0
    for s in strata:
        G = len(s); idx = rng.integers(0, G, size=(B, G)) + np.arange(B)[:, None] * G
        W[:, off:off + G] = np.bincount(idx.ravel(), minlength=B * G).reshape(B, G); off += G
    return games, W


def gindex(games, gid):
    pos = {g: i for i, g in enumerate(games)}
    return np.fromiter((pos[g] for g in gid), dtype=np.int64, count=len(gid))


def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def est(point, boot):
    return dict(est=float(point), ci=ci(boot))


def wmedian_boot(values, gidx, W, chunk=200):
    """Lower weighted median of `values` in every draw (a row's weight = how often its game was drawn)."""
    o = np.argsort(values, kind='stable'); v = values[o]; g = gidx[o]; out = np.empty(W.shape[0])
    for s in range(0, W.shape[0], chunk):
        c = np.cumsum(W[s:s + chunk][:, g], axis=1, dtype=np.int32)
        out[s:s + chunk] = v[(2 * c >= c[:, -1:]).argmax(1)]
    return out


def wmean_boot(values, gidx, W):
    """Mean of `values` in every draw: sum over drawn games of their row sums / their row counts."""
    G = W.shape[1]; num = np.bincount(gidx, weights=values, minlength=G); den = np.bincount(gidx, minlength=G).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (W @ num) / (W @ den)


def multinomial_draws(n_games, rng, B=B_DRAWS):
    """[B, n_games] multinomial resample counts (the second bootstrap form, used by the at-throw, read and completion scorers)."""
    return rng.multinomial(n_games, np.full(n_games, 1.0 / n_games), size=B)


def round_ci(est_value, boot, k=4):
    """[estimate, 2.5th, 97.5th percentile], each rounded to k decimals (the precision the table reads)."""
    return [round(float(est_value), k), round(float(np.percentile(boot, 2.5)), k), round(float(np.percentile(boot, 97.5)), k)]
