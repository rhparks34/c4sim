"""Stage 5: which defenders each play models (the coverage defenders), from their movement over the play's window.

python -m src.prep.defenders

1. Route flags: did each route-set player (offensive WR / RB / FB / TE) run a route? 2018: the tracking file's `route`; 2021: PFF
   pff_role; 2022: wasRunningRoute; 2023: the player's role (the 2023 files track no blockers).
2. Movement features of every tracked defensive back, linebacker and lineman over the window: depth and distance to the QB at the snap
   and at the end, how far and when he crossed the line, his nearest route runner and nearest skill blocker, the player he guarded.
3. Initial role where charted: 2021 pff_role, 2022 wasInitialPassRusher; in 2023 every tracked defender is in coverage. 2018 has no
   chart: two LightGBM classifiers (seed 95) fit on the 2021-2022 charted roles (one with every receiver by position, one with true
   route runners and blockers) give each 2018 defender a rush probability; their average > 0.5 makes him an initial rusher.
4. Label: the end state at the last window frame overrides the initial role (ending >= 5 yd off the ball or carrying a route runner
   = coverage; at the QB after an early backfield entry = rush; engaged with a blocker = rush); categories (STANDARD, CROSS_TRACK,
   PEEL, ...). Modeled: defensive backs / linebackers / linemen whose category is scored, at most 8 per play (defensive backs
   first, then the deepest at the snap).
-> $DATA_ROOT/prepared/defenders.parquet (one row per tracked defensive player), route_flags.parquet
"""
import re
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.paths import PREPARED
from src.prep import raw
from src.prep.frames import oriented, read_frames
from src.prep.plays import read_plays
from src.prep.tracks import read_tracks
from src.prep.windows import read_windows

SEASONS = (2018, 2021, 2022, 2023)
DB, LB, DL = {'CB', 'DB', 'FS', 'SS', 'S'}, {'LB', 'ILB', 'MLB', 'OLB'}, {'DE', 'DT', 'NT', 'DL'}
LINEMEN = ('C', 'G', 'T', 'OT', 'OG', 'OL')
POST = 15                    # frames kept after the window (only to pick the QB)
PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=95)
ROUNDS = 400
FEATURES_RECEIVERS = ['depth0', 'lat0', 'abs_lat0', 'dqb0', 'near_rec0', 'pen_max', 'cross05', 'f_cross05_rel', 'cross2', 'frac_bf',
                      'dqb_min', 'f_dqb_min', 'dqb_end', 'dqb_1s', 'dqb_change_1s', 'dx_1s', 'depth_end', 'lat_end', 'abs_lat_end',
                      'v_end', 'cos_to_qb_end', 'maxdepth', 'near_rec_end', 'near_rec_last1s', 'near_rec_same_last1s', 'rec_depth_end',
                      'qb_lat_range', 'qb_y_corr', 'dqb_std', 'n', 'is_DB', 'is_LB', 'is_DL']
FEATURES_ROUTES = ['depth0', 'lat0', 'abs_lat0', 'dqb0', 'rr0', 'pen_max', 'cross05', 'f_cross05_rel', 'cross2', 'frac_bf', 'dqb_min',
                   'f_dqb_min', 'dqb_end', 'dqb_1s', 'dqb_change_1s', 'dx_1s', 'depth_end', 'lat_end', 'abs_lat_end', 'v_end',
                   'cos_to_qb_end', 'maxdepth', 'rr_end', 'rr_last1s', 'rr_same_last1s', 'rr_depth_end', 'qb_lat_range', 'qb_y_corr',
                   'dqb_std', 'n', 'is_DB', 'is_LB', 'is_DL', 'sb_end', 'sb_last1s', 'sb_min']
SCORED = ('STANDARD', 'CROSS_RECOVER', 'CROSS_TRACK', 'LATE_PURSUIT', 'RUSH_DROP', 'R2C_OTHER', 'BEHIND_LINE_PAIR', 'PEEL', 'CROSS_OTHER')
NOT_MODELED = ('CROSS_OTHER',)


def defenders_path():
    return PREPARED / 'defenders.parquet'


def route_flags_path():
    return PREPARED / 'route_flags.parquet'


def read_defenders():
    return pd.read_parquet(defenders_path())


def read_route_flags():
    return pd.read_parquet(route_flags_path())


def generalize(code):
    return 'DB' if code in DB else 'LB' if code in LB else 'DL' if code in DL else 'other'


def window_tracks(season, meta):
    """Every player's x, y (float32) at frames f = 0 .. window - 1 after the snap and up to 15 frames after the window (`post`; used
    only to pick the QB when two are tracked); side 'D' (defense) or 'O' (offense). Rows stay in the published files' order."""
    t = oriented(read_frames(season, ['game_play_id', 'nfl_id', 'frame_id', 'side', 'position', 'x', 'y', 'going_left', 'route', 'role'],
                             set(meta.index)))
    t['f'] = t.frame_id - t.game_play_id.map(meta.snap_frame)
    n = t.game_play_id.map(meta.n)
    keep = (t.f >= 0) & (t.f < n + POST)
    if season == 2022:                       # the 2022 tracking the labels were first built from ended at the throw
        keep &= t.frame_id <= t.game_play_id.map(meta.release_frame)
    t = t[keep].copy()
    t['post'] = (t.f >= t.game_play_id.map(meta.n)).to_numpy()
    t['side'] = np.where(t.side.eq('defense'), 'D', 'O')
    t['x'] = t.x.astype(np.float32); t['y'] = t.y.astype(np.float32)
    return t.drop_duplicates(['game_play_id', 'nfl_id', 'f'])


def quarterback(rows):
    """The QB rows of a play: when two are tracked, the one with more rows (ties: the first listed)."""
    q = rows[rows.side.eq('O') & rows.position.eq('QB')]
    if q.nfl_id.nunique() > 1:
        q = q[q.nfl_id.eq(q.nfl_id.value_counts().index[0])]
    return q


def route_flags(season, meta, t):
    """One row per (play, route-set player): position, route, ran_route."""
    rows = [(gp, o, i) for gp, offs in zip(meta.index, meta.offense_ids) for i, o in enumerate(offs)]
    g = pd.DataFrame(rows, columns=['game_play_id', 'nfl_id', 'slot'])
    k = g.game_play_id + ':' + g.nfl_id
    first = t[~t.post].sort_values(['game_play_id', 'nfl_id', 'f'], kind='mergesort').drop_duplicates(['game_play_id', 'nfl_id'])
    fk = first.game_play_id + ':' + first.nfl_id
    if season == 2018:
        on_file = read_frames(season, ['game_play_id', 'nfl_id', 'frame_id', 'position', 'route'], set(meta.index))
        on_file = on_file.sort_values(['game_play_id', 'nfl_id', 'frame_id'], kind='mergesort').drop_duplicates(['game_play_id', 'nfl_id'])
        ok = on_file.game_play_id + ':' + on_file.nfl_id
        g['position'] = k.map(dict(zip(ok, on_file.position))); g['route'] = k.map(dict(zip(ok, on_file.route)))
        g['ran_route'] = g.route.notna()
    elif season == 2021:
        s = raw.read_pff_scouting_2021(['gameId', 'playId', 'nflId', 'pff_role'])
        role = dict(zip(s.gameId + '_' + s.playId + ':' + s.nflId, s.pff_role))
        pos = raw.read_players(2021).drop_duplicates('nflId').set_index('nflId').officialPosition
        g['position'] = g.nfl_id.map(pos); g['ran_route'] = k.map(role).eq('Pass Route')
        g['route'] = np.where(g.ran_route, 'ROUTE', None)
    elif season == 2022:
        pp = raw.read_player_play_2022(['gameId', 'playId', 'nflId', 'wasRunningRoute', 'routeRan'])
        pk = pp.gameId + '_' + pp.playId + ':' + pp.nflId
        pos = raw.read_players(2022).drop_duplicates('nflId').set_index('nflId').position
        g['position'] = g.nfl_id.map(pos)
        g['ran_route'] = k.map(dict(zip(pk, pp.wasRunningRoute))).astype(str).str.upper().isin(['TRUE', '1', '1.0'])
        g['route'] = k.map(dict(zip(pk, pp.routeRan)))
    else:
        role = dict(zip(fk, first.role)); pos = dict(zip(fk, first.position))
        g['position'] = k.map(pos); g['ran_route'] = k.map(role).isin(['Targeted Receiver', 'Other Route Runner'])
        g['route'] = np.where(g.ran_route, 'ROUTE', None)
    return g


def _xy(rows, F):
    a = np.full((F, 2), np.nan, np.float32)
    ff = rows.f.to_numpy(); ok = ff < F
    a[ff[ok]] = rows[['x', 'y']].to_numpy()[ok]
    return a


def _nearest(D, P, n):
    """D [n, 2] defender, P [k, F, 2] players -> distances [k, n] (inf where missing)."""
    if len(P) == 0:
        return np.full((1, n), np.inf)
    d = np.hypot(*(P[:, :n] - D[None, :n]).transpose(2, 0, 1))
    return np.where(np.isfinite(d), d, np.inf)


def play_features(gp, g, m, ran, route_name):
    """Movement features of every tracked defensive player of one play (g: the play's window rows; m: its meta row)."""
    n = int(m.n); los = float(m.los_x); offs = list(m.offense_ids)
    q = quarterback(g)                                             # counted over the window and the 15 frames after it
    Q = _xy(q, n) if len(q) else np.full((n, 2), np.nan, np.float32)
    q5 = quarterback(g[~g.post])                                   # counted over the window (route-runner features)
    Q5 = _xy(q5, n) if len(q5) else np.full((n, 2), np.nan, np.float32)
    g = g[~g.post]
    piv = {nid: _xy(h, n) for nid, h in g[g.side.eq('O')].groupby('nfl_id')}
    R = [piv[r] for r in offs if r in piv]
    R = np.stack(R) if R else np.full((1, n, 2), np.nan, np.float32)
    rr = [o for o in offs if o in piv and ran.get((gp, o), True)]
    sb = [o for o in offs if o in piv and not ran.get((gp, o), True)]
    pos_o = g[g.side.eq('O')].drop_duplicates('nfl_id').set_index('nfl_id').position
    ol = [o for o in piv if o not in offs and str(pos_o.get(o)) in LINEMEN]
    RR = np.stack([piv[o] for o in rr]) if rr else np.zeros((0, n, 2)); SB = np.stack([piv[o] for o in sb]) if sb else np.zeros((0, n, 2))
    AB = np.stack([piv[o] for o in sb + ol]) if (sb or ol) else np.zeros((0, n, 2))
    sk_ids = [o for o in offs if o in piv]
    SK = np.stack([piv[o] for o in sk_ids]) if sk_ids else np.zeros((0, n, 2))
    rows = []
    for nid, d in g[g.side.eq('D')].groupby('nfl_id'):
        w = _xy(d, n); ok = np.isfinite(w[:, 0])
        if ok.sum() < 3 or not np.isfinite(w[0, 0]):
            continue
        pos = str(d.position.iloc[0]); dq = np.hypot(*(w - Q).T)
        dr = np.hypot(*(w[None] - R).transpose(2, 0, 1))
        dr_min = np.nanmin(np.where(np.isfinite(dr), dr, np.inf), 0); dr_arg = np.argmin(np.where(np.isfinite(dr), dr, np.inf), 0)
        last = np.flatnonzero(ok)[-1]; k1 = min(10, last)
        v = (w[last] - w[max(last - 3, 0)]) / (0.1 * max(last - max(last - 3, 0), 1))
        toq = Q[last] - w[last]; cq = float(np.dot(v, toq) / (np.linalg.norm(v) * np.linalg.norm(toq) + 1e-6))
        cross = np.flatnonzero(w[:, 0] < los - 0.5); cross2 = np.flatnonzero(w[:, 0] < los - 2.0)
        l10 = slice(max(0, last - 9), last + 1)
        same = float(np.mean(dr_arg[l10] == dr_arg[last])) if np.isfinite(dr_min[last]) else np.nan
        qy = Q[:n, 1]; yy = w[:, 1]; both = np.isfinite(qy) & np.isfinite(yy)
        qlat = float(np.nanmax(qy) - np.nanmin(qy)) if np.isfinite(qy).any() else np.nan
        corr = float(np.corrcoef(yy[both], qy[both])[0, 1]) if both.sum() > 5 and np.nanstd(qy[both]) > 0.3 and np.nanstd(yy[both]) > 0.05 else np.nan
        rec = dict(
            game_play_id=gp, nfl_id=nid, position=pos, gen=generalize(pos), n=n,
            depth0=float(w[0, 0] - los), lat0=float(w[0, 1] - Q[0, 1]), dqb0=float(dq[0]), near_rec0=float(dr_min[0]),
            pen_max=float(los - np.nanmin(w[:, 0])), cross05=bool(len(cross)), f_cross05=float(cross[0]) if len(cross) else np.nan,
            cross2=bool(len(cross2)), frac_bf=float(np.nanmean(w[:, 0] < los - 0.5)),
            dqb_min=float(np.nanmin(dq)), f_dqb_min=float(np.nanargmin(dq) / max(n - 1, 1)) if np.isfinite(dq).any() else np.nan,
            dqb_end=float(dq[last]), dqb_1s=float(dq[k1]), dqb_change_1s=float(dq[k1] - dq[0]), dx_1s=float(w[k1, 0] - w[0, 0]),
            depth_end=float(w[last, 0] - los), lat_end=float(w[last, 1] - Q[last, 1]) if np.isfinite(Q[last, 1]) else np.nan,
            v_end=float(np.linalg.norm(v)), cos_to_qb_end=cq, maxdepth=float(np.nanmax(w[:, 0]) - los),
            near_rec_end=float(dr_min[last]), near_rec_last1s=float(np.nanmean(dr_min[l10])), near_rec_same_last1s=same,
            rec_depth_end=float(R[dr_arg[last], last, 0] - los) if np.isfinite(dr_min[last]) else np.nan,
            qb_lat_range=qlat, qb_y_corr=corr, dqb_std=float(np.nanstd(dq)))
        # nearest true route runner, skill blocker, any blocker
        drr = _nearest(w, RR, n); drm = drr.min(0); dra = drr.argmin(0)
        rec.update(rr0=float(drm[0]), rr_end=float(drm[last]), rr_last1s=float(np.mean(drm[l10])),
                   rr_same_last1s=float(np.mean(dra[l10] == dra[last])) if np.isfinite(drm[last]) else np.nan,
                   rr_depth_end=float(RR[dra[last], last, 0] - los) if rr and np.isfinite(drm[last]) else np.nan,
                   rr_qb_end=float(np.hypot(*(RR[dra[last], last] - Q5[last]))) if rr and np.isfinite(drm[last]) else np.nan)
        dsm = _nearest(w, SB, n).min(0)
        rec.update(sb_end=float(dsm[last]), sb_last1s=float(np.mean(dsm[l10])), sb_min=float(dsm.min()))
        # the guarded player: the skill player closest on average from his first frame past the line (or the snap) to the end
        s0 = int(cross[0]) if len(cross) else 0
        dk = _nearest(w, SK, n)[:, s0:last + 1]
        if dk.size and np.isfinite(dk).any():
            md = np.nanmean(np.where(np.isfinite(dk), dk, np.nan), 1); j = int(np.nanargmin(md)); gid = sk_ids[j]
            rec.update(g_id=gid, g_mean_dist=float(md[j]), g_share_nearest=float(np.mean(dk.argmin(0) == j)),
                       g_ran_route=ran.get((gp, gid), True), g_route=route_name.get((gp, gid)),
                       g_maxdepth=float(np.nanmax(piv[gid][:n, 0]) - los), g_depth_end=float(piv[gid][:n][np.isfinite(piv[gid][:n, 0])][-1, 0] - los))
        rows.append(rec)
    return rows


def charted_roles(season, d):
    """rush_pff (1 = initial pass rusher, 0 = coverage, NaN = not charted), cov_assign (2022), role_2026 (2023)."""
    k = d.game_play_id + ':' + d.nfl_id
    d['rush_pff'] = np.nan; d['cov_assign'] = None; d['role_2026'] = None
    if season == 2021:
        s = raw.read_pff_scouting_2021(['gameId', 'playId', 'nflId', 'pff_role'])
        role = k.map(dict(zip(s.gameId + '_' + s.playId + ':' + s.nflId, s.pff_role)))
        d.loc[role.eq('Pass Rush'), 'rush_pff'] = 1.0; d.loc[role.eq('Coverage'), 'rush_pff'] = 0.0
    elif season == 2022:
        pp = raw.read_player_play_2022(['gameId', 'playId', 'nflId', 'wasInitialPassRusher', 'pff_defensiveCoverageAssignment'])
        pk = pp.gameId + '_' + pp.playId + ':' + pp.nflId
        d['rush_pff'] = k.map(dict(zip(pk, pp.wasInitialPassRusher.astype(float))))
        d['cov_assign'] = k.map(dict(zip(pk, pp.pff_defensiveCoverageAssignment)))
    elif season == 2023:
        d['role_2026'] = d.pop('first_role')
        d.loc[d.role_2026.eq('Defensive Coverage'), 'rush_pff'] = 0.0
    d.pop('first_role') if 'first_role' in d else None
    return d


def season_table(season):
    plays, windows = read_plays().set_index('game_play_id'), read_windows().set_index('game_play_id')
    tracks = read_tracks(season)
    keep = [gp for gp in sorted(tracks) if windows.at[gp, 'kept']]
    meta = pd.DataFrame({'snap_frame': [int(tracks[gp]['frame_ids'][0]) for gp in keep], 'n': [int(windows.at[gp, 'window']) for gp in keep],
                         'release_frame': [int(plays.at[gp, 'release_frame']) for gp in keep],
                         'los_x': [float(tracks[gp]['los_x']) for gp in keep],
                         'offense_ids': [list(tracks[gp]['offense_ids']) for gp in keep]}, index=pd.Index(keep, name='game_play_id'))
    t = window_tracks(season, meta)
    flags = route_flags(season, meta, t)
    ran = {(gp, n): bool(r) for gp, n, r in zip(flags.game_play_id, flags.nfl_id, flags.ran_route)}
    route_name = {(gp, n): r for gp, n, r in zip(flags.game_play_id, flags.nfl_id, flags.route)}
    rows = []
    for gp, g in t.groupby('game_play_id', sort=True):
        rows += play_features(gp, g, meta.loc[gp], ran, route_name)
    d = pd.DataFrame(rows)
    first = t[~t.post].sort_values(['game_play_id', 'nfl_id', 'f'], kind='mergesort').drop_duplicates(['game_play_id', 'nfl_id'])
    d['first_role'] = (d.game_play_id + ':' + d.nfl_id).map(dict(zip(first.game_play_id + ':' + first.nfl_id, first.role)))
    d = charted_roles(season, d)
    d['season'] = season
    return d, flags.assign(season=season)


def fit_and_score(X, y, mask):
    model = lgb.train(PARAMS, lgb.Dataset(X[mask], y[mask]), num_boost_round=ROUNDS)
    return model.predict(X)


def label(f):
    """Rush probabilities -> labels, categories and the modeled set (the agreed v5 universe)."""
    f['f_cross05_rel'] = (f.f_cross05 / (f.n - 1).clip(lower=1)).fillna(2.0)
    f['abs_lat0'] = f.lat0.abs(); f['abs_lat_end'] = f.lat_end.abs()
    for g in ('DB', 'LB', 'DL'):
        f[f'is_{g}'] = (f.gen == g).astype(float)
    for c in ('rr0', 'rr_end', 'rr_last1s', 'sb_end', 'sb_last1s', 'sb_min'):
        f[c] = f[c].replace(np.inf, 99.).fillna(99.).clip(upper=99.)
    f['rr_same_last1s'] = f.rr_same_last1s.fillna(0.); f['rr_depth_end'] = f.rr_depth_end.fillna(0.); f['rr_qb_end'] = f.rr_qb_end.fillna(99.)
    el = f.gen.isin(['DB', 'LB', 'DL'])
    charted = el & f.rush_pff.notna() & f.season.isin([2021, 2022])
    f['p_rush'] = np.nan; f['p_rush5'] = np.nan
    f.loc[el, 'p_rush'] = fit_and_score(f.loc[el, FEATURES_RECEIVERS].astype(float), f.loc[el, 'rush_pff'], charted[el])
    f.loc[el, 'p_rush5'] = fit_and_score(f.loc[el, FEATURES_ROUTES].astype(float), f.loc[el, 'rush_pff'], charted[el])
    end_rush0 = ((f.depth_end <= -1) & (f.dqb_end <= 8)) | ((f.depth_end <= 1) & (f.dqb_end <= 3))
    end_cover0 = (f.depth_end >= 3) | ((f.depth_end > 1) & (f.dqb_end >= 10))
    end_rush = end_rush0 & ((f.rr_last1s > 2) | (f.dqb_end <= 3))
    carry = (f.gen.isin(['DB', 'LB']) & (f.rr_last1s <= 3) & (f.dqb_end >= 6) & (f.depth_end > 0) & (f.cos_to_qb_end < 0.5)
             & (f.rr_qb_end >= 5))
    end_cover = (f.depth_end >= 5) | carry | (f.cov_assign.notna() & end_cover0)
    end_rush_blitz = end_rush & f.f_cross05.notna() & (f.f_cross05 <= f.n - 16)
    engaged = (f.sb_last1s <= 2.0) & (f.sb_last1s < f.rr_last1s) & (f.depth_end <= 1.0)
    init = pd.Series(np.nan, index=f.index)
    init[f.season.isin([2021, 2022])] = f.rush_pff[f.season.isin([2021, 2022])]
    init[f.season.eq(2023) & f.role_2026.eq('Defensive Coverage')] = 0.0
    init[f.season.eq(2018)] = (((f.p_rush + f.p_rush5) / 2)[f.season.eq(2018)] > 0.5).astype(float)
    f['init_rush'] = init
    bad = f.season.eq(2023) & f.role_2026.isin(['Other Route Runner', 'Targeted Receiver'])      # offensive players in a defense
    lab = np.where(bad, 'OFFENSE_DATA_FIX', np.where(engaged & ~end_cover, 'RUSH', np.where(end_rush_blitz, 'RUSH', np.where(end_cover, 'COVER',
                   np.where(init == 1, 'RUSH', np.where(init == 0, 'COVER', 'UNLABELED'))))))
    f['label'] = lab
    late_pursuit = end_rush & ~end_rush_blitz & f.label.eq('COVER')
    spy = (f.label.eq('COVER') & f.gen.isin(['DB', 'LB']) & (f.maxdepth <= 5) & (f.dqb_min >= 3) & (f.dqb_end <= 9) & (f.dqb0 <= 9)
           & (f.rr_last1s >= 4) & (f.qb_lat_range >= 3) & (f.qb_y_corr >= 0.8))
    crossed = f.pen_max > 0.5
    g_track = f.g_ran_route.fillna(False).astype(bool) & (f.g_mean_dist <= 6.0) & (f.g_share_nearest >= 0.7)
    r2c = (init == 1) & f.label.eq('COVER')
    cat = np.select(
        [f.label.eq('OFFENSE_DATA_FIX'), f.label.eq('UNLABELED'), f.label.eq('RUSH') & engaged & ~end_rush_blitz, f.label.eq('RUSH'),
         spy, late_pursuit, r2c & (f.rr_last1s <= 3), r2c & (f.depth_end >= 5), r2c, crossed & g_track, crossed, f.label.eq('COVER')],
        ['OFFENSE_DATA_FIX', 'UNLABELED', 'BLITZ_PICKED_UP', 'RUSH', 'SPY', 'LATE_PURSUIT', 'PEEL', 'RUSH_DROP', 'R2C_OTHER',
         'CROSS_TRACK', 'CROSS_OTHER', 'STANDARD'], 'OTHER')
    shallow = f.g_maxdepth <= 3.0
    blocker_guard = crossed & f.label.eq('COVER') & ~f.g_ran_route.fillna(True).astype(bool) & (f.g_share_nearest >= 0.7)
    track = cat == 'CROSS_TRACK'
    f['category'] = np.select([blocker_guard & np.isin(cat, ['CROSS_OTHER', 'LATE_PURSUIT', 'CROSS_TRACK', 'R2C_OTHER']),
                               track & shallow & (f.depth_end < 3), track & (f.depth_end >= 3), track, cat == 'PEEL'],
                              ['HUG', 'BEHIND_LINE_PAIR', 'CROSS_RECOVER', 'CROSS_TRACK', 'PEEL'], cat)
    candidate = el & f.category.isin(SCORED)
    c = f[candidate].assign(_db=(~f.gen.eq('DB')).astype(int), _d=-f.depth0).sort_values(['game_play_id', '_db', '_d'])
    rank = pd.Series(99, index=f.index); rank[c.index] = c.groupby('game_play_id').cumcount().to_numpy()
    f['modeled'] = candidate & (rank < 8) & ~f.category.isin(NOT_MODELED)
    return f.drop(columns=[c for c in f.columns if c.startswith('is_')])


def main():
    with Pool(len(SEASONS)) as pool:
        parts = pool.map(season_table, SEASONS)
    f = pd.concat([d for d, _ in parts], ignore_index=True).sort_values(['game_play_id', 'nfl_id'], kind='mergesort').reset_index(drop=True)
    f = label(f)
    f.to_parquet(defenders_path(), index=False)
    pd.concat([r for _, r in parts], ignore_index=True).to_parquet(route_flags_path(), index=False)
    m = f[f.modeled]
    print(f'defenders: {len(f):,} tracked, {len(m):,} modeled on {m.game_play_id.nunique():,} plays', m.groupby('season').size().to_dict(),
          flush=True)


if __name__ == '__main__':
    main()
