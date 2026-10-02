"""2018 play-action flags (part of stage 2). The 2018 files chart no play action; 2021-2023 do.

A 2018 play is play action if its tracking carries a `play_action` event at or before the first pass or sack, OR a detector
trained on the 2021 and 2022 charted plays gives p >= 0.9. The detector is LightGBM on the offense's pre-throw movement (QB drop and
facing, the nearest backs' paths and closeness to the QB, receiver depth), frames 0-30 after the snap; the event itself is never
a feature. Its 2022 training plays are the charted dropbacks with a pass, cut at the pass (as first built).
-> {game_play_id: 0/1} for every 2018 play
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.prep import raw
from src.prep.frames import oriented, read_frames

SNAP = {'ball_snap', 'snap_direct', 'autoevent_ballsnap'}
END = {'pass_forward', 'pass_shovel', 'qb_sack', 'qb_strip_sack', 'autoevent_passforward'}
BACKS, RECEIVERS, HORIZON = {'RB', 'HB', 'FB'}, {'WR', 'TE'}, 30
THRESHOLD = .9
FEATURES = ['end_frames', 'qb_depth0', 'qb_dx5', 'qb_ady5', 'qb_dx10', 'qb_ady10', 'qb_dx15', 'qb_ady15', 'qb_dx20', 'qb_ady20', 'qb_dx25',
            'qb_ady25', 'qb_maxady15', 'qb_smax15', 'qb_smean10', 'qb_fx0', 'qb_fx_min20', 'qb_fx_argmin20', 'qb_fx_mean10',
            'qb_fx_mean20', 'qb_fyabs_max15', 'qb_turn20', 'qb_fx_at_end', 'n_backs', 'b_rel_x0', 'b_rel_ady0', 'b_qb_dmin20',
            'b_qb_argmin20', 'b_dx5', 'b_qb_d5', 'b_dx10', 'b_qb_d10', 'b_dx15', 'b_qb_d15', 'b_dx20', 'b_qb_d20', 'b_x15_los',
            'b_xmax20_los', 'b_cross15', 'b_vx_mean10', 'b_vx_max15', 'b_smax15', 'b_ady_qb15', 'any_back_dmin20', 'rec_depth15_mean',
            'rec_depth_end_mean', 'rec_depth_end_max', 'n_recv', 'b2_rel_x0', 'b2_rel_ady0', 'b2_qb_dmin20', 'b2_qb_argmin20']
PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=8)
ROUNDS = 600


def charted_flag(value):
    """A charted yes/no field -> 1/0 (missing counts as 0)."""
    return 1 if str(value).strip().lower() in ('1', 'true', '1.0') else 0


def play_events(frames):
    """(game_play_id, frame_id, event) once per event."""
    ev = frames.loc[frames.event.ne('') & frames.event.ne('none'), ['game_play_id', 'frame_id', 'event']]
    return ev.drop_duplicates()


def event_flags_2018(frames):
    """The tracking rule: a play_action event at or before the first pass_forward / pass_shovel / qb_sack / qb_strip_sack."""
    flags = dict.fromkeys(frames.game_play_id.unique(), 0)
    for gp, g in play_events(frames).groupby('game_play_id'):
        cut_at = g.loc[g.event.isin(END - {'autoevent_passforward'}), 'frame_id']
        cut = cut_at.min() if len(cut_at) else g.frame_id.max()
        flags[gp] = int(((g.event == 'play_action') & (g.frame_id <= cut)).any())
    return flags


def _filled(g, n):
    """One player's rows -> [n, 5] (x, y, s, o, dir) over frames 0..n-1 after the snap, gaps filled forward then back."""
    a = np.full((n, 5), np.nan)
    f = g.f.to_numpy(); ok = (f >= 0) & (f < n)
    a[f[ok]] = g[['x', 'y', 's', 'o', 'dir']].to_numpy(float)[ok]
    return pd.DataFrame(a).ffill().bfill().to_numpy()


def _unit(o):
    r = np.deg2rad(o)
    return np.sin(r), np.cos(r)


def play_features(w, los, end_f):
    """Offense movement features of one play (w: QB / back / receiver rows, f = frames after the snap)."""
    n = int(min(end_f, HORIZON)) + 1
    at = lambda k: min(k, n - 1)
    qbs = w[w.position.eq('QB')]
    if qbs.empty:
        return None
    x0 = qbs[qbs.f.eq(qbs.f.min())].groupby('nfl_id').x.first()
    Q = _filled(qbs[qbs.nfl_id.eq(x0.idxmin())].sort_values('f'), n)          # the deepest QB at the snap is the passer
    F = dict(end_frames=float(end_f), qb_depth0=Q[0, 0] - los)
    for k in (5, 10, 15, 20, 25):
        F[f'qb_dx{k}'] = Q[at(k), 0] - Q[0, 0]; F[f'qb_ady{k}'] = abs(Q[at(k), 1] - Q[0, 1])
    m15 = slice(0, at(15) + 1); m20 = slice(0, at(20) + 1)
    F['qb_maxady15'] = np.abs(Q[m15, 1] - Q[0, 1]).max(); F['qb_smax15'] = Q[m15, 2].max(); F['qb_smean10'] = Q[:at(10) + 1, 2].mean()
    fx, fy = _unit(Q[:, 3])
    F['qb_fx0'] = fx[0]; F['qb_fx_min20'] = fx[m20].min(); F['qb_fx_argmin20'] = float(np.argmin(fx[m20]))
    F['qb_fx_mean10'] = fx[:at(10) + 1].mean(); F['qb_fx_mean20'] = fx[m20].mean(); F['qb_fyabs_max15'] = np.abs(fy[m15]).max()
    turn = np.abs((np.diff(Q[m20, 3]) + 180) % 360 - 180); F['qb_turn20'] = turn.sum() if len(turn) else 0.
    F['qb_fx_at_end'] = fx[n - 1]
    backs = w[w.position.isin(BACKS)]
    ids = list(backs.nfl_id.unique()); F['n_backs'] = float(len(ids))
    B = []
    for b in ids:
        T = _filled(backs[backs.nfl_id.eq(b)].sort_values('f'), n)
        B.append((np.hypot(T[0, 0] - Q[0, 0], T[0, 1] - Q[0, 1]), T))
    B.sort(key=lambda t: t[0])                                                 # nearest back to the QB at the snap first
    for j, name in enumerate(('b', 'b2')):
        if j >= len(B):
            continue
        T = B[j][1]; d = np.hypot(T[:, 0] - Q[:, 0], T[:, 1] - Q[:, 1])
        F[f'{name}_rel_x0'] = T[0, 0] - Q[0, 0]; F[f'{name}_rel_ady0'] = abs(T[0, 1] - Q[0, 1])
        F[f'{name}_qb_dmin20'] = d[m20].min(); F[f'{name}_qb_argmin20'] = float(np.argmin(d[m20]))
        if name == 'b':
            for k in (5, 10, 15, 20):
                F[f'b_dx{k}'] = T[at(k), 0] - T[0, 0]; F[f'b_qb_d{k}'] = d[at(k)]
            F['b_x15_los'] = T[at(15), 0] - los; F['b_xmax20_los'] = T[m20, 0].max() - los
            rel = T[m15, 1] - Q[m15, 1]; F['b_cross15'] = float(np.any(np.sign(rel[1:]) != np.sign(rel[:-1]))) if len(rel) > 1 else 0.
            vx = T[:, 2] * np.sin(np.deg2rad(T[:, 4])); F['b_vx_mean10'] = vx[:at(10) + 1].mean(); F['b_vx_max15'] = vx[m15].max()
            F['b_smax15'] = T[m15, 2].max()
            F['b_ady_qb15'] = abs(T[at(15), 1] - Q[at(15), 1])
    if B:
        F['any_back_dmin20'] = min(np.hypot(T[:, 0] - Q[:, 0], T[:, 1] - Q[:, 1])[m20].min() for _, T in B)
    rc = w[w.position.isin(RECEIVERS)]
    if not rc.empty:
        k15 = rc[rc.f.eq(at(15))]; ke = rc[rc.f.eq(n - 1)]
        F['rec_depth15_mean'] = (k15.x - los).mean() if len(k15) else np.nan
        F['rec_depth_end_mean'] = (ke.x - los).mean() if len(ke) else np.nan
        F['rec_depth_end_max'] = (ke.x - los).max() if len(ke) else np.nan
        F['n_recv'] = float(rc.nfl_id.nunique())
    return F


def season_features(season, plays=None, cut_at_pass=False):
    """One row of detector features per play with a snap event (optionally only `plays`; optionally frames after the first pass
    dropped, as the 2022 training plays were built)."""
    frames = oriented(read_frames(season, ['game_play_id', 'nfl_id', 'frame_id', 'side', 'position', 'x', 'y', 's', 'o', 'dir', 'going_left',
                                           'event'], plays))
    ev = play_events(frames)
    snap = ev[ev.event.isin(SNAP)].groupby('game_play_id').frame_id.min()
    end = ev[ev.event.isin(END)].groupby('game_play_id').frame_id.min()
    w = frames[frames.side.eq('offense') & frames.position.isin(BACKS | RECEIVERS | {'QB'}) & frames.game_play_id.isin(snap.index)].copy()
    if cut_at_pass:
        first_pass = ev[ev.event.isin({'pass_forward', 'pass_shovel'})].groupby('game_play_id').frame_id.min()
        w = w[w.frame_id <= w.game_play_id.map(first_pass)]
    w['f'] = w.frame_id - w.game_play_id.map(snap)
    w = w[(w.f >= 0) & (w.f <= HORIZON)]
    rows = []
    for gp, g in w.groupby('game_play_id'):
        s0 = g[g.f.eq(0)]
        if s0.empty:
            continue
        los = s0.x.max()
        e = end.get(gp, np.nan)
        end_f = (e - snap[gp]) if pd.notna(e) else float(g.f.max())
        if end_f < 0:
            continue
        F = play_features(g, los, end_f)
        if F is not None:
            rows.append(dict(F, game_play_id=gp))
    return pd.DataFrame(rows).reindex(columns=['game_play_id'] + FEATURES)


def flags_2018():
    """{game_play_id: 0/1} for every 2018 play: the tracking event OR detector p >= 0.9."""
    train = []
    p21 = raw.read_plays(2021)
    label21 = dict(zip(p21.gameId + '_' + p21.playId, p21.pff_playAction.map(charted_flag)))
    f21 = season_features(2021)
    train.append(f21.assign(label=f21.game_play_id.map(label21)))
    p22 = raw.read_plays(2022)
    p22['game_play_id'] = p22.gameId + '_' + p22.playId
    charted = raw.read_player_play_2022(['gameId', 'playId', 'pff_defensiveCoverageAssignment'])
    charted = set((charted.gameId + '_' + charted.playId)[charted.pff_defensiveCoverageAssignment.notna()])
    passes = read_frames(2022, ['game_play_id', 'event'])
    passes = set(passes.game_play_id[passes.event.isin({'pass_forward', 'pass_shovel'})])
    f22 = season_features(2022, plays=charted & passes, cut_at_pass=True)
    label22 = dict(zip(p22.game_play_id, p22.playAction.map(charted_flag)))
    train.append(f22.assign(label=f22.game_play_id.map(label22)))
    train = pd.concat(train, ignore_index=True).dropna(subset=['label'])
    model = lgb.train(PARAMS, lgb.Dataset(train[FEATURES], train.label.astype(int)), num_boost_round=ROUNDS)
    f18 = season_features(2018)
    p = dict(zip(f18.game_play_id, model.predict(f18[FEATURES])))
    events = event_flags_2018(read_frames(2018, ['game_play_id', 'frame_id', 'event']))
    return {gp: int(ev == 1 or p.get(gp, 0.) >= THRESHOLD) for gp, ev in events.items()}
