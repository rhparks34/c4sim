"""Stage 6: the coded-rule inputs of every modeled defender, solved by the physics solvers.

python -m src.prep.rule_inputs

For each modeled defender (stage 5) of each play, over his frames in the play's window:
  zone landmarks    (solvers/landmarks.py) the minimum time and energy to reach each receiver's spot at the end and the middle of the
                    window and each of the seven zone landmarks, from his snap position and pre-snap velocity
  rule options      the 12 candidate paths over the next second (rules/targets.py: a corridor on each of 5 receivers, the path of each
                    of the 7 Cover-4 roles), tracked by the solvers (solvers/bank.py) -> 68 token features and ten-step previews
  boundary history  his last 0.5 s before the snap and his pre-snap velocity (2018 and 2021: the only seasons whose files hold
                    pre-snap tracking; 2022 and 2023 get an explicit "no history" row and zero velocity)
  rule teacher      the c4 solve from his real first step minus the legal one (training-only targets)
Coordinates are the published float64 values turned to the offense frame. The 2018 Cover-4 plays' rule inputs are built as they were
first published (from the original rule simulator's export): coordinates rounded to float32 and no route-runner positions (so no
route counts as a running back's). Every solved row is kept, with its exact inputs, in
$DATA_ROOT/prepared/solver_cache/; a row is solved only when no cached row has the same inputs (within 1e-4 yd), in one pass
over all missing rows, in the solvers' 16,384-row chunks.
-> $DATA_ROOT/prepared/rule_inputs.pkl  {game_play_id: {nfl_id: arrays}}
"""
import pickle

import numpy as np
import pandas as pd

from src.paths import PREPARED
from src.prep.defenders import read_defenders, read_route_flags
from src.prep.frames import oriented, read_frames
from src.prep.plays import read_plays
from src.prep.rules.targets import build_receiver_target, build_role_target, make_route_context, pad_path
from src.prep.tracks import read_tracks
from src.prep.windows import read_windows
from src.solvers import bank, landmarks

DT, N_MAX, N_SLOTS, N_ZONES, MIN_H = 0.1, 10, 5, 7, 0.5
N_HIST, V_ORACLE_CAP = 5, 12.0
PRE_SNAP_SEASONS = (2018, 2021)
TOL = 1e-4
CACHE = PREPARED / 'solver_cache'
TOKEN_SPEC = [   # (column, scale): the record stores clip(value / scale, -4, 4)
    ('c4_ax0', 7.0), ('c4_ay0', 7.0), ('c4_f0', 200.0), ('c4_lam0', 1.0), ('c4_power0', 200.0), ('c4_energy_total', 150.0),
    ('c4_energy_early', 75.0), ('c4_e_viol_rel', 0.1), ('c4_path_len', 8.0), ('c4_excess_len', 4.0), ('c4_heading_change_total', 6.0),
    ('c4_term_miss', 10.0), ('c4_term_vel_miss', 5.0), ('c4_sat_f_frac', 1.0), ('c4_sat_lam_frac', 1.0), ('c4_dp_0p2_x', 1.0),
    ('c4_dp_0p2_y', 1.0), ('c4_dp_0p5_x', 2.5), ('c4_dp_0p5_y', 2.5), ('c4_dp_1p0_x', 6.0), ('c4_dp_1p0_y', 6.0),
    ('c4_pers_res_0p2', 0.5), ('c4_pers_res_0p5', 1.5), ('c4_pers_res_1p0', 4.0), ('c4_finite', 1.0), ('c4_converged', 1.0),
    ('c4_retried', 1.0), ('c4_ab_disagree', 0.1), ('c4_win_start', 2.0), ('c4_rel_impr_final25', 1e-3), ('c1_dp_0p2_x', 1.0),
    ('c1_dp_0p2_y', 1.0), ('c1_dp_0p5_x', 2.5), ('c1_dp_0p5_y', 2.5), ('c1_dp_1p0_x', 6.0), ('c1_dp_1p0_y', 6.0), ('c1_term_miss', 10.0),
    ('c1_energy_total', 150.0), ('c1_ax0', 7.0), ('c1_ay0', 7.0), ('an_areq_mag_0p3', 7.0), ('an_ax_0p3', 7.0), ('an_ay_0p3', 7.0),
    ('an_miss_0p3', 5.0), ('an_valid_0p3', 1.0), ('an_areq_mag_0p5', 7.0), ('an_ax_0p5', 7.0), ('an_ay_0p5', 7.0), ('an_miss_0p5', 5.0),
    ('an_valid_0p5', 1.0), ('an_areq_mag_1p0', 7.0), ('an_ax_1p0', 7.0), ('an_ay_1p0', 7.0), ('an_miss_1p0', 5.0), ('an_valid_1p0', 1.0),
    ('tgt_dp_0p2_x', 2.0), ('tgt_dp_0p2_y', 2.0), ('tgt_dp_0p5_x', 4.0), ('tgt_dp_0p5_y', 4.0), ('tgt_dp_1p0_x', 8.0),
    ('tgt_dp_1p0_y', 8.0), ('tgt_v0_x', 10.0), ('tgt_v0_y', 10.0), ('tgt_vN_x', 10.0), ('tgt_vN_y', 10.0), ('option_available', 1.0),
    ('v0_valid', 1.0), ('n_active', 10.0)]
BOUNDARY_SPEC = ([('valid', 1.0), ('n_hist', 5.0)] + [(f'hist_dx_{k}', 2.0) for k in range(1, 6)] + [(f'hist_dy_{k}', 2.0) for k in range(1, 6)]
                 + [(f'hist_valid_{k}', 1.0) for k in range(1, 6)]
                 + [('vx_fd', 8.0), ('vy_fd', 8.0), ('vx_smooth', 8.0), ('vy_smooth', 8.0), ('ax_fd', 10.0), ('ay_fd', 10.0),
                    ('speed_fd', 8.0), ('cos_heading', 1.0), ('sin_heading', 1.0), ('heading_change', np.pi), ('fd_noise', 0.25)])
BOUNDARY_SPEC_COLS = [c for c, _ in BOUNDARY_SPEC]
DETERMINISTIC = [c for c, _ in TOKEN_SPEC if c.startswith(('an_', 'tgt_')) or c in ('option_available', 'v0_valid', 'n_active')]


def output_path():
    return PREPARED / 'rule_inputs.pkl'


# ---- per-play geometry in float64 ---------------------------------------------------------------------------------------------
def play_geometry(season, tracks, windows, modeled, float32_plays=()):
    """{gp: dict(frames, offense ids / xy [n_off, T, 2], defenders (modeled, id order) / xy, window)} with NaN where untracked;
    plus the raw (unturned) pre-snap rows of the modeled defenders. float32_plays: plays whose turned positions are rounded to
    float32."""
    plays = sorted(gp for gp in tracks if gp in modeled)
    f = read_frames(season, ['game_play_id', 'nfl_id', 'frame_id', 'x', 'y', 'going_left'], set(plays))
    pre = f.copy()
    f = oriented(f)
    r32 = f.game_play_id.isin(set(float32_plays)).to_numpy()
    for c in ('x', 'y'):
        f.loc[r32, c] = f.loc[r32, c].astype(np.float32).astype(np.float64)
    out = {}
    for gp, g in f.groupby('game_play_id', sort=True):
        e = tracks[gp]; fid = e['frame_ids']; T = len(fid)
        at = {(n, fr): (x, y) for n, fr, x, y in zip(g.nfl_id, g.frame_id, g.x, g.y)}

        def xy(ids):
            a = np.full((len(ids), T, 2), np.nan)
            for i, nid in enumerate(ids):
                for t, fr in enumerate(fid):
                    v = at.get((nid, int(fr)))
                    if v is not None:
                        a[i, t] = v
            return a
        out[gp] = dict(frames=fid, window=int(windows[gp]), offense_ids=list(e['offense_ids']), offense_xy=xy(e['offense_ids']),
                       defenders=modeled[gp], defense_xy=xy(modeled[gp]), going_left=bool(g.going_left.iloc[0]))
    pre = pre[pre.nfl_id.isin({n for gp in plays for n in modeled[gp]})]
    return out, pre


def defender_window(g, i):
    """(index of his first tracked frame, number of frames nf) of modeled defender i within the window."""
    ok = np.isfinite(g['defense_xy'][i, :g['window'], 0])
    if not ok.any():
        return None, 0
    return int(np.flatnonzero(ok)[0]), int(np.flatnonzero(ok)[-1]) + 1


# ---- legal pre-snap velocity and boundary history ----------------------------------------------------------------------------
def finish_row(gp, nfl, pts):
    """pts: (x, y) positions in the offense frame, oldest .. the snap (>= 2). Verbatim from w42_boundary_history._finish_row."""
    arr = np.asarray(pts, dtype=np.float64)
    p0 = arr[-1]
    n_hist = len(arr) - 1
    row = {'game_play_id': gp, 'nfl_id': nfl, 'n_hist': n_hist, 'valid': 1}
    for k in range(1, N_HIST + 1):
        if k <= n_hist:
            rel = arr[-1 - k] - p0
            row[f'hist_dx_{k}'] = rel[0]; row[f'hist_dy_{k}'] = rel[1]; row[f'hist_valid_{k}'] = 1
        else:
            row[f'hist_dx_{k}'] = 0.0; row[f'hist_dy_{k}'] = 0.0; row[f'hist_valid_{k}'] = 0
    v = np.diff(arr, axis=0) / DT
    row['vx_fd'], row['vy_fd'] = v[-1]
    t = np.arange(len(arr)) * DT
    if len(arr) >= 3:
        cx = np.polyfit(t, arr[:, 0], 1); cy = np.polyfit(t, arr[:, 1], 1)
        row['vx_smooth'], row['vy_smooth'] = cx[0], cy[0]
        resid = np.concatenate([arr[:, 0] - np.polyval(cx, t), arr[:, 1] - np.polyval(cy, t)])
        row['fd_noise'] = float(np.std(resid))
    else:
        row['vx_smooth'], row['vy_smooth'] = v[-1]
        row['fd_noise'] = 0.0
    if len(v) >= 2:
        row['ax_fd'] = (v[-1, 0] - v[-2, 0]) / DT; row['ay_fd'] = (v[-1, 1] - v[-2, 1]) / DT
        hc = np.arctan2(v[-1, 1], v[-1, 0]) - np.arctan2(v[0, 1], v[0, 0])
        row['heading_change'] = float(np.arctan2(np.sin(hc), np.cos(hc)))
    else:
        row['ax_fd'] = row['ay_fd'] = row['heading_change'] = 0.0
    sp = float(np.hypot(*v[-1]))
    row['speed_fd'] = sp
    row['cos_heading'] = float(v[-1, 0] / sp) if sp > 1e-6 else 0.0
    row['sin_heading'] = float(v[-1, 1] / sp) if sp > 1e-6 else 0.0
    return row


def no_history_row(gp, nfl):
    row = {'game_play_id': gp, 'nfl_id': nfl, 'n_hist': 0, 'valid': 0}
    for k in range(1, N_HIST + 1):
        row[f'hist_dx_{k}'] = row[f'hist_dy_{k}'] = 0.0; row[f'hist_valid_{k}'] = 0
    for c in ('vx_fd', 'vy_fd', 'vx_smooth', 'vy_smooth', 'ax_fd', 'ay_fd', 'speed_fd', 'cos_heading', 'sin_heading', 'heading_change',
              'fd_noise'):
        row[c] = 0.0
    return row


def pre_snap(season, geometry, pre):
    """-> legal velocity {(gp, nfl): (vx, vy, valid)} and boundary rows, from the published positions before each defender's first
    frame (2018 and 2021 only)."""
    v0, rows = {}, []
    raw = {(gp, n, int(fr)): (x, y) for gp, n, fr, x, y in zip(pre.game_play_id, pre.nfl_id, pre.frame_id, pre.x, pre.y)}
    turn = lambda left: (lambda x, y: (120.0 - x, 53.3 - y)) if left else (lambda x, y: (x, y))
    for gp, g in geometry.items():
        for i, nid in enumerate(g['defenders']):
            first, nf = defender_window(g, i)
            if season not in PRE_SNAP_SEASONS or first is None:
                v0[(gp, nid)] = (0.0, 0.0, 0); rows.append(no_history_row(gp, nid)); continue
            f0 = int(g['frames'][first]); tf = turn(g['going_left']); sign = -1.0 if g['going_left'] else 1.0
            cur, prev = raw.get((gp, nid, f0)), raw.get((gp, nid, f0 - 1))
            v0[(gp, nid)] = ((sign * (cur[0] - prev[0]) / DT, sign * (cur[1] - prev[1]) / DT, 1) if cur is not None and prev is not None
                             else (0.0, 0.0, 0))
            pts = []
            for k in range(N_HIST, -1, -1):                      # oldest .. the snap; a gap restarts the window
                q = raw.get((gp, nid, f0 - k))
                if q is None:
                    pts = [] if pts else pts
                    continue
                pts.append(tf(*q))
            rows.append(finish_row(gp, nid, pts) if len(pts) >= 2 else no_history_row(gp, nid))
    return v0, rows


# ---- solver problems ---------------------------------------------------------------------------------------------------------
def landmark_problems(geometry, v0):
    rows = []
    for gp, g in geometry.items():
        off = g['offense_xy']; n_off = off.shape[0]
        for i, nid in enumerate(g['defenders']):
            first, nf = defender_window(g, i)
            if first is None or nf - first < 4:
                continue
            frames = np.arange(first, nf)
            p0 = g['defense_xy'][i, first]; n = len(frames); tf = (n - 1) * DT; mid_f = max(1, n // 2)
            starts = off[:min(N_SLOTS, n_off), first, 0]
            los_x = float(np.nanmax(starts)) if np.isfinite(starts).any() else 0.0
            vx, vy, ok = v0[(gp, nid)]
            base = dict(game_play_id=gp, defender_index=i, nfl_id=nid, duration=tf, p0x=float(p0[0]), p0y=float(p0[1]),
                        vx0=float(vx), vy0=float(vy), v0_valid=int(ok))
            for s in range(min(N_SLOTS, n_off)):
                track = off[s, frames]
                if not np.isfinite(track[:, 0]).any():
                    continue
                at = lambda k: track[int(np.flatnonzero(np.isfinite(track[:k + 1, 0]))[-1])] if np.isfinite(track[:k + 1, 0]).any() else track[k]
                e, m = at(n - 1), at(mid_f)
                rows.append(dict(base, option_kind='slot', option_index=s, endx=float(e[0]), endy=float(e[1]), midx=float(m[0]),
                                 midy=float(m[1])))
            for z in range(N_ZONES):
                rows.append(dict(base, option_kind='zone', option_index=z, endx=los_x + landmarks.ZONE_DEPTH[z],
                                 endy=float(landmarks.ZONE_Y[z]), midx=np.nan, midy=np.nan))
    return pd.DataFrame(rows)


def option_problems(geometry, v0, positions):
    """-> meta (one row per defender and option, canonical order), target paths q, velocities qd [n, 11, 2]."""
    meta, qs, qds = [], [], []
    for gp, g in geometry.items():
        off = g['offense_xy']
        for i, nid in enumerate(g['defenders']):
            first, nf = defender_window(g, i)
            if first is None:
                continue
            frames = np.arange(first, nf); n = len(frames)
            if (n - 1) * DT < MIN_H - 1e-9:
                continue
            n_steps = min(N_MAX, n - 1)
            p0 = g['defense_xy'][i, first].astype(float)
            routes = [dict(slot=s, nfl_id=oid, position=positions.get((gp, oid), 'UNKNOWN'), xy=off[s, frames])
                      for s, oid in enumerate(g['offense_ids'][:N_SLOTS]) if np.isfinite(off[s, frames]).all()]
            if not routes:
                continue
            los_x = max(float(r['xy'][0, 0]) for r in routes)
            ctx = make_route_context(routes, los_x, n_frames=n)
            by_slot = {r['slot']: r for r in routes}
            vx, vy, ok = v0[(gp, nid)]
            base = dict(game_play_id=gp, defender_index=i, nfl_id=nid, n_active=int(n_steps), p0x=p0[0], p0y=p0[1],
                        vx0_legal=float(vx), vy0_legal=float(vy), v0_valid=int(ok), los_x=los_x)
            for s in range(N_SLOTS):
                r = by_slot.get(s)
                if r is None or r['xy'].shape[0] < n_steps + 1:
                    meta.append(dict(base, option_kind='slot', option_index=s, option_available=0))
                    qs.append(np.zeros((N_MAX + 1, 2), np.float32)); qds.append(np.zeros((N_MAX + 1, 2), np.float32))
                    continue
                o = build_receiver_target(p0, r['xy'], n_steps)
                meta.append(dict(base, option_kind='slot', option_index=s, option_available=1))
                qs.append(pad_path(o['q'])); qds.append(pad_path(o['qdot']))
            for z in range(N_ZONES):
                o = build_role_target(z, p0, ctx, n_steps)
                meta.append(dict(base, option_kind='zone', option_index=z, option_available=1))
                qs.append(pad_path(o['q'])); qds.append(pad_path(o['qdot']))
    meta = pd.DataFrame(meta)
    return meta, np.stack(qs).astype(np.float32), np.stack(qds).astype(np.float32)


def oracle_velocity(geometry):
    """The defender's real first-step velocity (the teacher's start), NaN beyond the 12 yd/s physical guard."""
    out = {}
    for gp, g in geometry.items():
        for i, nid in enumerate(g['defenders']):
            first, nf = defender_window(g, i)
            if first is None or nf - first < 2:
                continue
            a, b = g['defense_xy'][i, first], g['defense_xy'][i, first + 1]
            v = (b - a) / DT
            out[(gp, nid)] = v if np.hypot(*v) <= V_ORACLE_CAP else np.array([np.nan, np.nan])
    return out


# ---- every defender's inputs ---------------------------------------------------------------------------------------------
def problems():
    """Stage-6 inputs of every modeled defender of every play: landmark problems, option meta / q / qd, boundary rows, oracle
    velocity; plus each defender's season."""
    plays = read_plays().set_index('game_play_id')
    windows = read_windows().set_index('game_play_id')
    d = read_defenders(); d = d[d.modeled]
    modeled = {gp: sorted(map(str, g.nfl_id)) for gp, g in d.groupby('game_play_id')}
    rf = read_route_flags(); positions = {(g, str(n)): str(p) for g, n, p in zip(rf.game_play_id, rf.nfl_id, rf.position)}
    lms, metas, qs, qds, rows, oracle = [], [], [], [], [], {}
    for season in (2018, 2021, 2022, 2023):
        tracks = read_tracks(season)
        mine = {gp: ids for gp, ids in modeled.items() if gp in tracks}
        win = {gp: int(windows.at[gp, 'window']) for gp in mine}
        cover4_2018 = set(plays.index[plays.season.eq(2018) & plays.coverage_family.eq('quarters')])
        geo, pre = play_geometry(season, tracks, win, mine, float32_plays=cover4_2018 & set(mine))
        v0, brows = pre_snap(season, geo, pre)
        lms.append(landmark_problems(geo, v0)); rows += brows
        m, q, qd = option_problems(geo, v0, {k: v for k, v in positions.items() if k[0] not in cover4_2018})
        metas.append(m); qs.append(q); qds.append(qd)
        oracle.update(oracle_velocity(geo))
        print(f'  problems {season}: {len(mine):,} plays', flush=True)
    meta = pd.concat(metas, ignore_index=True)
    meta.insert(0, 'row_id', np.arange(len(meta), dtype=np.int64))
    return dict(landmarks=pd.concat(lms, ignore_index=True), meta=meta, q=np.concatenate(qs), qd=np.concatenate(qds),
                boundary=pd.DataFrame(rows), oracle=oracle, plays=plays, windows=windows, defenders=modeled)


def normalized_tokens(cols, names):
    return np.stack([np.clip(np.nan_to_num(np.asarray(cols[c], np.float32)) / s, -4.0, 4.0) for c, s in TOKEN_SPEC if c in names], 1)


# ---- reuse of cached rows ----------------------------------------------------------------------------------------------------
def _close(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (np.abs(a - b) <= TOL) | (np.isnan(a) & np.isnan(b))


def match_landmarks(lp):
    """-> {(gp, nfl): source} for defenders whose every landmark row equals a cached source's inputs (earliest source wins)."""
    path = CACHE / 'landmarks.parquet'
    if not path.exists() or lp.empty:
        return {}
    cache = pd.read_parquet(path)
    m = lp.merge(cache, on=LANDMARK_KEY, suffixes=('', '_c'))
    ok = np.ones(len(m), bool)
    for c in LANDMARK_INPUTS:
        ok &= _close(m[c], m[c + '_c'])
    m['ok'] = ok
    per = m.groupby(['game_play_id', 'nfl_id', 'source', 'priority']).agg(rows=('ok', 'size'), ok=('ok', 'all')).reset_index()
    need = lp.groupby(['game_play_id', 'nfl_id']).size().rename('need').reset_index()
    per = per.merge(need, on=['game_play_id', 'nfl_id'])
    per = per[per.ok & per.rows.eq(per.need)].sort_values('priority').drop_duplicates(['game_play_id', 'nfl_id'])
    return dict(zip(zip(per.game_play_id, per.nfl_id), per.source))


def match_options(meta, q, qd):
    """-> {(gp, nfl): (source, [cache rows in option order])}: production rows match on every input and the full target paths;
    other sources (which stored no paths) on the token columns that are exact functions of the inputs."""
    path = CACHE / 'options.parquet'
    if not path.exists():
        return {}
    cache = pd.read_parquet(path); cq = np.load(CACHE / 'options_q.npy'); cqd = np.load(CACHE / 'options_qd.npy')
    cache['cache_row'] = np.arange(len(cache))
    det_cols = bank.deterministic_columns(meta, q, qd)
    mine = normalized_tokens(det_cols, DETERMINISTIC_CALC)
    m = meta[['row_id'] + OPTION_KEY + ['n_active', 'option_available', 'v0_valid', 'p0x', 'p0y', 'vx0_legal', 'vy0_legal']].merge(
        cache[OPTION_KEY + ['n_active', 'option_available', 'v0_valid', 'p0x', 'p0y', 'vx0_legal', 'vy0_legal', 'source', 'priority',
                            'cache_row'] + [c for c in DETERMINISTIC_CALC]], on=OPTION_KEY, suffixes=('', '_c'))
    ok = (m.n_active.to_numpy() == m.n_active_c.to_numpy()) & (m.option_available.to_numpy() == m.option_available_c.to_numpy()) \
        & (m.v0_valid.to_numpy() == m.v0_valid_c.to_numpy())
    prod = m.source.eq('production').to_numpy()
    for c in ('p0x', 'p0y', 'vx0_legal', 'vy0_legal'):
        ok &= ~prod | _close(m[c], m[c + '_c'])
    r, cr = m.row_id.to_numpy(), m.cache_row.to_numpy()
    path_ok = np.ones(len(m), bool)
    path_ok[prod] = (np.abs(q[r[prod]] - cq[cr[prod]]).reshape(prod.sum(), -1).max(1) <= TOL) \
        & (np.abs(qd[r[prod]] - cqd[cr[prod]]).reshape(prod.sum(), -1).max(1) <= TOL)
    theirs = normalized_tokens({c: m[c].to_numpy() for c in DETERMINISTIC_CALC}, DETERMINISTIC_CALC)
    tok_ok = (np.abs(mine[r] - theirs) <= TOL).all(1)
    m['ok'] = ok & np.where(prod, path_ok, tok_ok)
    per = m.groupby(['game_play_id', 'nfl_id', 'source', 'priority']).agg(rows=('ok', 'size'), ok=('ok', 'all')).reset_index()
    per = per[per.ok & per.rows.eq(12)].sort_values('priority').drop_duplicates(['game_play_id', 'nfl_id'])
    m = m.merge(per[['game_play_id', 'nfl_id', 'source']], on=['game_play_id', 'nfl_id', 'source'])
    return {k: (g.source.iloc[0], g.sort_values('row_id').cache_row.to_numpy()) for k, g in m.groupby(['game_play_id', 'nfl_id'])}


def match_teacher(option_match, oracle):
    """-> {(gp, nfl): teacher cache rows} where the defender's options matched production rows and his real first-step velocity
    equals the stored one (both beyond the guard counts as equal)."""
    path = CACHE / 'teacher.parquet'
    if not path.exists():
        return {}
    t = pd.read_parquet(path); t['teacher_row'] = np.arange(len(t))
    opt = pd.read_parquet(CACHE / 'options.parquet', columns=['production_row'])
    by_prow = t.groupby('production_row').teacher_row.first()
    out = {}
    for k, (src, rows) in option_match.items():
        if src != 'production':
            continue
        prows = opt.production_row.to_numpy()[rows]
        hit = by_prow.reindex(prows).dropna().astype(int).to_numpy()
        if not len(hit):
            continue
        tv = t.iloc[hit[0]]
        v = oracle.get(k, np.array([np.nan, np.nan]))
        same = (_close(v[0], tv.oracle_vx) and _close(v[1], tv.oracle_vy)) if int(tv.oracle_valid) == 1 else bool(np.isnan(v[0]))
        if same:
            out[k] = hit
    return out


def solve_status(P, lm_match, op_match, te_match):
    """One row per modeled defender: for landmarks, options and the teacher, 'reuse' (a cached row has the same inputs), 'solve'
    or 'none' (no such input: fewer than 4 frames for landmarks; a first step beyond the 12 yd/s guard for the teacher)."""
    defs = P['meta'].drop_duplicates(['game_play_id', 'nfl_id'])[['game_play_id', 'nfl_id']].reset_index(drop=True)
    keys = list(zip(defs.game_play_id, defs.nfl_id))
    has_lm = set(zip(P['landmarks'].game_play_id, P['landmarks'].nfl_id))
    has_v = {k for k, v in P['oracle'].items() if np.isfinite(v[0])}
    defs['landmarks'] = [('reuse' if k in lm_match else 'solve') if k in has_lm else 'none' for k in keys]
    defs['landmarks_source'] = [lm_match.get(k) for k in keys]
    defs['options'] = ['reuse' if k in op_match else 'solve' for k in keys]
    defs['options_source'] = [op_match[k][0] if k in op_match else None for k in keys]
    defs['teacher'] = ['reuse' if k in te_match else ('solve' if k in has_v else 'none') for k in keys]
    return defs


DETERMINISTIC_CALC = [c for c, _ in TOKEN_SPEC if c.startswith(('an_', 'tgt_'))]
OPTION_KEY = ['game_play_id', 'nfl_id', 'option_kind', 'option_index']
LANDMARK_KEY = ['game_play_id', 'nfl_id', 'option_kind', 'option_index']
LANDMARK_INPUTS = ['duration', 'p0x', 'p0y', 'vx0', 'vy0', 'v0_valid', 'endx', 'endy', 'midx', 'midy']


# ---- solving what the cache does not hold, and the per-defender arrays ---------------------------------------------------------
def top_up(n, chunk):
    """How many already-solved rows fill the last solver chunk (they run in the same batch and serve as the positive control)."""
    return (-n) % chunk


def solve_landmarks(P, lm_match):
    """Reuse matched landmark rows, solve the rest once -> ({gp: {nfl: [13, 7]}}, report)."""
    lp = P['landmarks']
    path = CACHE / 'landmarks.parquet'
    cache = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=LANDMARK_KEY + LANDMARK_INPUTS + landmarks.SOLVED + ['source', 'priority'])
    src = pd.DataFrame([(g, n, s) for (g, n), s in lm_match.items()], columns=['game_play_id', 'nfl_id', 'source'])
    reused = cache.merge(src, on=['game_play_id', 'nfl_id', 'source'])
    todo = lp[[k not in lm_match for k in zip(lp.game_play_id, lp.nfl_id)]].reset_index(drop=True)
    report = dict(reused_defenders=len(lm_match), solved_rows=len(todo), solved_defenders=int(todo.groupby(['game_play_id', 'nfl_id']).ngroups))
    rows = reused[LANDMARK_KEY + landmarks.SOLVED]
    if len(todo):
        control = reused[reused.source.eq('production')].head(top_up(len(todo), landmarks.CHUNK)).assign(defender_index=-1)
        solved = landmarks.solve(pd.concat([todo, control[todo.columns]], ignore_index=True))
        new = pd.concat([todo[LANDMARK_KEY + LANDMARK_INPUTS], solved.iloc[:len(todo)][landmarks.SOLVED].reset_index(drop=True)], axis=1)
        pd.concat([cache, new.assign(source='rebuild', priority=9)], ignore_index=True).to_parquet(path, index=False)
        ctl = solved.iloc[len(todo):].reset_index(drop=True)
        report['control'] = dict(rows=len(control), **{c: float(np.nanmax(np.abs(ctl[c].to_numpy() - control[c].to_numpy()))) if len(control) else None
                                                        for c in landmarks.SOLVED})
        rows = pd.concat([rows, new[LANDMARK_KEY + landmarks.SOLVED]], ignore_index=True)
    return landmarks.features(rows), report


def solve_options(P, op_match, te_match):
    """Reuse matched option rows, solve the rest once (c4, c1; then the teacher's priv_c4) -> (token columns, previews, teacher
    rows, report); token rows follow P['meta']."""
    meta, q, qd = P['meta'], P['q'], P['qd']
    path = CACHE / 'options.parquet'
    have = path.exists()
    cache = pd.read_parquet(path) if have else pd.DataFrame()
    cq = np.load(CACHE / 'options_q.npy') if have else np.zeros((0, 11, 2), np.float32)
    cqd = np.load(CACHE / 'options_qd.npy') if have else np.zeros((0, 11, 2), np.float32)
    cprev = np.load(CACHE / 'options_previews.npy') if have else np.zeros((0, 11, 11), np.float16)
    tok_cols = [c for c, _ in TOKEN_SPEC if c not in ('option_available', 'v0_valid', 'n_active')]
    tokens = pd.DataFrame(np.zeros((len(meta), len(tok_cols)), np.float32), columns=tok_cols)
    previews = np.zeros((len(meta), 11, 11), np.float16)
    rows_of = meta.groupby(['game_play_id', 'nfl_id'], sort=False).indices
    cache_row = np.full(len(meta), -1)
    for k, (_, rows) in op_match.items():
        cache_row[np.sort(rows_of[k])] = rows
    hit = cache_row >= 0
    tokens.loc[hit, tok_cols] = cache.iloc[cache_row[hit]][tok_cols].to_numpy(np.float32)
    previews[hit] = cprev[cache_row[hit]]
    todo = np.flatnonzero(~hit)
    report = dict(reused_defenders=len(op_match), solved_defenders=len(todo) // 12)
    if len(todo):
        n_avail = int(meta.option_available.to_numpy()[todo].sum())
        prod = [k for k, (src, _) in op_match.items() if src == 'production']
        pool = np.concatenate([np.sort(rows_of[k]) for k in prod]) if prod else np.zeros(0, int)
        pool = pool[meta.option_available.to_numpy()[pool] == 1][:top_up(n_avail, bank.CHUNK)]
        run = np.concatenate([todo, pool])
        sub = meta.iloc[run].reset_index(drop=True).assign(tag=''); sub['row_id'] = np.arange(len(sub))
        c4, c4t = bank.solve_bank(sub, q[run], qd[run], 'c4')
        c1, c1t = bank.solve_bank(sub, q[run], qd[run], 'c1')
        tk, pv = bank.option_tokens(sub, q[run], qd[run], c4, c4t, c1, c1t)
        tokens.iloc[todo] = tk.iloc[:len(todo)][tok_cols].to_numpy(np.float32)
        previews[todo] = pv[:len(todo)]
        if len(pool):                                   # positive control: stored production rows solved in this batch
            got = tk.iloc[len(todo):][tok_cols].to_numpy(np.float32); ref = cache.iloc[cache_row[pool]][tok_cols].to_numpy(np.float32)
            det = [tok_cols.index(c) for c in DETERMINISTIC_CALC]
            report['control'] = dict(rows=len(pool), deterministic_max_abs=float(np.abs(got[:, det] - ref[:, det]).max()),
                                     solver_share_over_0p1=float((np.abs(np.delete(got - ref, det, axis=1)) > 0.1).mean()))
        new = pd.concat([meta.iloc[todo][OPTION_KEY + ['n_active', 'option_available', 'v0_valid', 'p0x', 'p0y', 'vx0_legal', 'vy0_legal']]
                         .reset_index(drop=True), tk.iloc[:len(todo)][tok_cols].reset_index(drop=True)], axis=1)
        pd.concat([cache, new.assign(production_row=-1, source='rebuild', priority=9)], ignore_index=True).to_parquet(path, index=False)
        np.save(CACHE / 'options_q.npy', np.concatenate([cq, q[todo]])); np.save(CACHE / 'options_qd.npy', np.concatenate([cqd, qd[todo]]))
        np.save(CACHE / 'options_previews.npy', np.concatenate([cprev, pv[:len(todo)]]))
    # the rule teacher: c4 from the real first step minus the legal c4 (reused or just solved)
    tpath = CACHE / 'teacher.parquet'
    tcache = pd.read_parquet(tpath) if tpath.exists() else pd.DataFrame()
    reused = tcache.iloc[np.concatenate(list(te_match.values()))] if te_match else tcache.head(0)
    oracle = np.array([P['oracle'].get(k, (np.nan, np.nan)) for k in zip(meta.game_play_id, meta.nfl_id)], dtype=float)
    need = np.array([k not in te_match for k in zip(meta.game_play_id, meta.nfl_id)]) & np.isfinite(oracle[:, 0])
    t_todo = np.flatnonzero(need & (meta.option_available.to_numpy() == 1))
    new_teacher = tcache.head(0)
    if len(t_todo):
        # top-up: rows of defenders whose stored teacher is reused fill the last chunk and serve as the positive control
        pool = np.concatenate([np.sort(rows_of[k]) for k in te_match]) if te_match else np.zeros(0, int)
        pool = pool[(meta.option_available.to_numpy()[pool] == 1) & np.isfinite(oracle[pool, 0])][:top_up(len(t_todo), bank.CHUNK)]
        run = np.concatenate([t_todo, pool])
        sub = meta.iloc[run].reset_index(drop=True); sub['row_id'] = np.arange(len(sub))
        priv, _ = bank.solve_bank(sub, q[run], qd[run], 'priv_c4', v0_override=oracle[run])
        legal = tokens.iloc[run].reset_index(drop=True).assign(row_id=np.arange(len(sub)))
        solved = bank.teacher_rows(sub.assign(tag=''), legal, priv, oracle[run])
        if len(pool):
            got = solved[solved.row_id >= len(t_todo)]
            tkey = ['game_play_id', 'nfl_id', 'option_kind', 'option_index']
            ref = got[tkey].merge(reused, on=tkey, how='left', validate='one_to_one')
            dcols = ['d_ax0', 'd_ay0'] + [f'd_dp_{h}_{a}' for h in ('0p2', '0p5', '1p0') for a in ('x', 'y')]
            diff = np.abs(got[dcols].to_numpy(np.float64) - ref[dcols].to_numpy(np.float64))
            report['teacher_control'] = dict(rows=len(got), unmatched=int(ref.d_ax0.isna().sum()),
                                             d_v0_max_abs=float(np.nanmax(np.abs(got[['d_vx0', 'd_vy0']].to_numpy() - ref[['d_vx0', 'd_vy0']].to_numpy()))),
                                             share_over_0p1=float(np.nanmean(diff > 0.1)), median_abs=float(np.nanmedian(diff)))
        new_teacher = solved[solved.row_id < len(t_todo)].copy()
        new_teacher['oracle_vx'] = oracle[run][new_teacher.row_id.to_numpy(), 0]
        new_teacher['oracle_vy'] = oracle[run][new_teacher.row_id.to_numpy(), 1]
        new_teacher = new_teacher.drop(columns='row_id').assign(oracle_valid=1, production_row=-1, source='rebuild')
        pd.concat([tcache, new_teacher], ignore_index=True).to_parquet(tpath, index=False)
    report.update(reused_teacher_defenders=len(te_match),
                  solved_teacher_defenders=int(new_teacher.groupby(['game_play_id', 'nfl_id']).ngroups) if len(new_teacher) else 0)
    return tokens, previews, pd.concat([reused, new_teacher], ignore_index=True), report


def assemble(P, lm_lookup, tokens, previews, teacher):
    """{gp: {nfl: arrays}} -> per play arrays padded to 8 defenders in the record's defender order."""
    meta = P['meta']
    column = lambda c: tokens[c].to_numpy(np.float32) if c in tokens else meta[c].to_numpy(np.float32)
    feats = np.stack([np.clip(np.nan_to_num(column(c)) / s, -4.0, 4.0) for c, s in TOKEN_SPEC], 1)
    valid = meta.option_available.to_numpy(np.float32) * tokens['c4_finite'].to_numpy(np.float32)
    bh = P['boundary']
    bfeat = np.stack([np.clip(np.nan_to_num(bh[c].to_numpy(np.float32)) / s, -4.0, 4.0) for c, s in BOUNDARY_SPEC], 1)
    bvalid = bh['valid'].to_numpy(np.float32)
    bvpre = np.stack([np.nan_to_num(bh.vx_fd.to_numpy(np.float32)), np.nan_to_num(bh.vy_fd.to_numpy(np.float32))], 1) * bvalid[:, None]
    blk = {(g, n): i for i, (g, n) in enumerate(zip(bh.game_play_id, bh.nfl_id))}
    opt = meta.groupby(['game_play_id', 'nfl_id'], sort=False).indices
    tch = {}
    if len(teacher):
        o = teacher.option_index.to_numpy(np.int64) + np.where(teacher.option_kind.to_numpy() == 'zone', landmarks.ZONE_START, 0)
        for (g, n), idx in teacher.groupby(['game_play_id', 'nfl_id']).indices.items():
            e = [np.zeros((12, 2), np.float32), np.zeros((12, 6), np.float32), np.zeros(12, np.float32), np.zeros(2, np.float32), 0.0]
            r = teacher.iloc[idx]
            e[0][o[idx]] = np.stack([r.d_ax0, r.d_ay0], 1); e[1][o[idx]] = np.stack([r[f'd_dp_{h}_{a}'] for h in ('0p2', '0p5', '1p0') for a in ('x', 'y')], 1)
            e[2][o[idx]] = (r.priv_finite * r.legal_finite).to_numpy(np.float32); e[3][:] = (r.d_vx0.iloc[0], r.d_vy0.iloc[0]); e[4] = float(r.d_v0_valid.iloc[0])
            tch[(g, n)] = e
    out = {}
    for gp, ids in P['defenders'].items():
        a = dict(zone_landmarks=np.zeros((8, 13, 7), np.float32), rule_tokens=np.zeros((8, 12, 68), np.float32),
                 rule_previews=np.zeros((8, 12, 11, 11), np.float16), rule_option_valid=np.zeros((8, 12), np.float32),
                 rule_boundary=np.zeros((8, 28), np.float32), rule_velocity_before=np.zeros((8, 2), np.float32),
                 rule_boundary_valid=np.zeros(8, np.float32))
        teach = [np.zeros((8, 12, 2), np.float32), np.zeros((8, 12, 6), np.float32), np.zeros((8, 12), np.float32), np.zeros((8, 2), np.float32),
                 np.zeros(8, np.float32)]
        any_teacher = False
        for i, n in enumerate(ids[:8]):
            if gp in lm_lookup and n in lm_lookup[gp]:
                a['zone_landmarks'][i] = lm_lookup[gp][n]
            ix = opt.get((gp, n))
            if ix is not None:
                ix = np.sort(ix)
                a['rule_tokens'][i] = feats[ix]; a['rule_previews'][i] = previews[ix]; a['rule_option_valid'][i] = valid[ix]
            j = blk.get((gp, n))
            if j is not None:
                a['rule_boundary'][i] = bfeat[j]; a['rule_velocity_before'][i] = bvpre[j]; a['rule_boundary_valid'][i] = bvalid[j]
            e = tch.get((gp, n))
            if e is not None:
                any_teacher = True
                for k in range(4):
                    teach[k][i] = e[k]
                teach[4][i] = e[4]
        if any_teacher:
            a['rule_teacher'] = tuple(teach)
        out[gp] = a
    return out


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true', help='write the solve list (solve_list.parquet / .json) and stop')
    args = ap.parse_args(argv)
    P = problems()
    lm_match = match_landmarks(P['landmarks']); print('landmark defenders reused', len(lm_match), flush=True)
    op_match = match_options(P['meta'], P['q'], P['qd']); print('option defenders reused', len(op_match), flush=True)
    te_match = match_teacher(op_match, P['oracle']); print('teacher defenders reused', len(te_match), flush=True)
    defs = solve_status(P, lm_match, op_match, te_match)
    defs.to_parquet(PREPARED / 'solve_list.parquet', index=False)
    summary = {kind: defs[kind].value_counts().to_dict() for kind in ('landmarks', 'options', 'teacher')}
    (PREPARED / 'solve_list.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    if args.plan:
        return
    lm_lookup, lm_report = solve_landmarks(P, lm_match)
    tokens, previews, teacher, op_report = solve_options(P, op_match, te_match)
    rules = assemble(P, lm_lookup, tokens, previews, teacher)
    with open(output_path(), 'wb') as fh:
        pickle.dump(rules, fh, protocol=4)
    (PREPARED / 'rule_inputs_report.json').write_text(json.dumps(dict(landmarks=lm_report, options=op_report), indent=1, default=str))
    print('rule inputs:', len(rules), 'plays', json.dumps(dict(landmarks=lm_report, options=op_report), default=str), flush=True)


if __name__ == '__main__':
    main()
