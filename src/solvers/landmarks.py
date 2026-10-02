"""Zone landmarks: for every defender and option (each receiver's spot at the end and the middle of the defender's window, and
the seven Cover-4 zone landmarks), the minimum time to reach it from the defender's snap state and the minimum energy to reach
it within the window. The solve and the feature transform are verbatim from the research code (W37 ocbasis).

Problem rows: game_play_id, defender_index, nfl_id, option_kind ('slot' / 'zone'), option_index, duration (s), p0x, p0y, vx0, vy0,
v0_valid, endx, endy, midx, midy (mid NaN for zone options). Solved in chunks of 16,384 rows in row order.
"""
import time

import numpy as np
import pandas as pd

from src.solvers.effort import solve_effort_constrained_batch, solve_min_time_batch

DT, AMAX, CHUNK = 0.1, 7.0, 16384
ZONE_DEPTH = np.array([15.0, 15.0, 15.0, 15.0, 3.0, 6.0, 3.0])          # yd past the line of scrimmage
ZONE_Y = np.array([6.7, 20.0, 33.3, 46.6, 6.7, 26.65, 46.6])             # yd across the field
N_SLOTS, N_ZONES, ZONE_START, N_FEATS = 5, 7, 5, 7
SOLVED = ['oc_t_min_end', 'oc_reach_ratio_end', 'oc_t_min_mid', 'oc_reach_ratio_mid', 'oc_energy_rate', 'oc_terminal_miss']


def solve(pr):
    P0 = pr[['p0x', 'p0y']].to_numpy(float); V0 = pr[['vx0', 'vy0']].to_numpy(float)
    END = pr[['endx', 'endy']].to_numpy(float); MID = pr[['midx', 'midy']].to_numpy(float); TF = pr.duration.to_numpy(float)
    vf = np.zeros_like(P0); is_slot = pr.option_kind.eq('slot').to_numpy()
    def mt(a, b, c, d):
        out = np.full(len(a), np.nan)
        for s in range(0, len(a), CHUNK):
            e = min(s + CHUNK, len(a)); out[s:e] = np.asarray(solve_min_time_batch(a[s:e], b[s:e], c[s:e], d[s:e], amax=AMAX)['t_seg'], float)
        return out
    t0 = time.time(); df = pr[['game_play_id', 'defender_index', 'nfl_id', 'option_kind', 'option_index', 'duration']].copy()
    df['oc_t_min_end'] = mt(P0, V0, END, vf); df['oc_reach_ratio_end'] = df.oc_t_min_end / df.duration
    tm = np.full(len(df), np.nan); tm[is_slot] = mt(P0[is_slot], V0[is_slot], MID[is_slot], vf[is_slot])
    df['oc_t_min_mid'] = tm; df['oc_reach_ratio_mid'] = df.oc_t_min_mid / (df.duration / 2.0)
    energy = np.full(len(df), np.nan); miss = np.full(len(df), np.nan)
    for s in range(0, len(df), CHUNK):
        e = min(s + CHUNK, len(df))
        out = solve_effort_constrained_batch(P0[s:e], V0[s:e], END[s:e], TF[s:e], e_max=TF[s:e] * 150.0, f_max=200.0, dt=DT,
                                             objective='min_energy_reach', energy_weight=0.01, adam_iters=400, lr=0.08, device='cpu')
        energy[s:e] = np.asarray(out['z_final'], float); miss[s:e] = np.asarray(out['terminal_dist'], float)
        print(f'  effort {e}/{len(df)} [{time.time() - t0:.0f}s]', flush=True)
    df['oc_energy_rate'] = energy / np.maximum(TF, 1e-6); df['oc_terminal_miss'] = miss
    return df


def features(df):
    """Solved rows -> {game_play_id: {nfl_id: [13, 7] float32}} (option rows 0-4 receivers, 5-11 zones, row 12 unused; columns
    valid, min time / 4, reach ratio, mid-window min time / 4, its reach ratio, log energy rate / 5, terminal miss / 10)."""
    feats = np.stack([np.ones(len(df), dtype=np.float32), df['oc_t_min_end'].to_numpy(np.float32) / 4.0,
                      np.minimum(df['oc_reach_ratio_end'].to_numpy(np.float32), 4.0) / 2.0,
                      np.nan_to_num(df['oc_t_min_mid'].to_numpy(np.float32)) / 4.0,
                      np.minimum(np.nan_to_num(df['oc_reach_ratio_mid'].to_numpy(np.float32)), 4.0) / 2.0,
                      np.log1p(np.maximum(df['oc_energy_rate'].to_numpy(np.float32), 0.0)) / 5.0,
                      np.minimum(df['oc_terminal_miss'].to_numpy(np.float32), 20.0) / 10.0], axis=1)
    feats[~np.isfinite(feats).all(axis=1)] = 0.0
    opt = df['option_index'].to_numpy(np.int64) + np.where((df['option_kind'] == 'zone').to_numpy(), ZONE_START, 0)
    lookup = {}
    for gp, nfl, o, f in zip(df['game_play_id'].astype(str), df['nfl_id'].astype(str), opt, feats):
        lookup.setdefault(gp, {}).setdefault(nfl, np.zeros((N_SLOTS + N_ZONES + 1, N_FEATS), np.float32))[o] = f
    return lookup
