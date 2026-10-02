"""Stage 8: each fold's split and side inputs.

python -m src.prep.folds [--fold K]

Fold K holds out the 62 scored 2018 games of metadata/folds.csv (column fold == K):
  play_ids.csv        validation = the 2018 Cover-4 plays in those games (minus the 62 training-only passes of metadata/plays.csv);
                      training = every play outside those games; the games' other plays are left out of fold K
  stats.json          mean and standard deviation of each of the 41 defender features over every valid defender-frame of the
                      training split (the model normalizes with them; its QB buffers are set from them)
  population.json     play counts and the loss normalizer: the mean number of valid defender-frames per training play
  retrieval.pt        route-matched analog plays (below), from the fold's training split only
  orientation.pt      body orientation: QB and route runners per frame, defenders at the snap (sin o, cos o, valid)
  position_blocker.pt each defender's roster position class; whether each route-set player ran no route (a blocker)

Analogs ("rules by example"): for each play the K nearest training plays of the same coverage family from another game, matched on
routes and alignment; each query defender gets the matched analog defender's displacement (Hungarian matching on snap alignment).
Descriptor (yards, relative to the line of scrimmage x and the QB's snap y): receivers grouped per side outside-in (#1/#2/#3 left
and right), each sampled at frames 0, 5, ..., 30 of its route; defender snap alignment sorted by lateral position; window length.
Each bank play also enters mirrored. Channels [8, 80, 8]: 0-1 weighted mean dx/10, dy/10; 2-3 weighted std; 4 fraction of analogs
valid; 5-6 mean snap offset /10; 7 mean matched snap distance /10. Under the loader's y-flip channels 1 and 6 change sign.
"""
import argparse
import io
import json
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from src.paths import REPO, fold_dir
from src.prep.defenders import read_defenders, read_route_flags
from src.prep.frames import read_frames
from src.prep.plays import read_plays
from src.prep.records import records_dir

warnings.filterwarnings('ignore', category=RuntimeWarning)
K = 64                      # analogs per play
FRAMES = [0, 5, 10, 15, 20, 25, 30]
W_PRESENT, W_ROUTE, W_DEF_PRESENT, W_TIME = 6.0, 1.0, 3.0, 2.0
W_DEF = 0.5
TAU = 6.0                   # similarity weight exp(-(d - d_min) / TAU)
POS_PENALTY = 5.0           # matching cost added when a linebacker is paired with a non-linebacker
MAXT = 80
MATCH_MAX = 10.0
N_CH = 8
FEATURE_NAMES = []
for _i in range(5):
    FEATURE_NAMES += [f'recv_dx_{_i}', f'recv_dy_{_i}', f'recv_dist_{_i}', f'recv_vx_{_i}', f'recv_vy_{_i}', f'recv_closing_speed_{_i}']
FEATURE_NAMES += ['time_frac', 'snap_x', 'snap_y', 'qb_dx', 'qb_dy', 'qb_pocket_depth', 'los_depth', 'n_receivers_nearby',
                  'nearest_recv_dist', 'mean_recv_depth', 'any_route_break']
POSITION_CLASS = {'CB': 1, 'FS': 2, 'SS': 2, 'S': 2, 'DB': 3, 'ILB': 4, 'MLB': 4, 'LB': 5, 'OLB': 6, 'DE': 7, 'DT': 8, 'NT': 8, 'DL': 9}


def pt_bytes(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def sha256(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def read_records():
    """Every record, compacted to what the fold layer needs (one pass over the shards)."""
    man = json.loads((records_dir() / 'manifest.json').read_text())
    out = {}
    for r in man['shards']:
        for gp, p in torch.load(records_dir() / r['filename'], map_location='cpu', weights_only=False).items():
            b = p['precomputed_base']
            rc = np.asarray(b['route_context'], np.float32)
            qy = np.asarray(p['qb_data'], np.float64)[0, :, 1]
            qy = qy[np.isfinite(qy)]
            dfm = np.asarray(b['def_frame_mask'], bool)
            out[gp] = dict(
                gp=gp, game=gp.split('_')[0], family=int(p['coverage_family_idx']), los=float(p['los_x']),
                yball=float(qy[0]) if len(qy) else 26.65, n_frames=int(b['n_frames']), def_mask=np.asarray(b['def_mask'], bool),
                snap=np.asarray(b['snap_xy'], np.float32), target=np.asarray(b['target'], np.float32), dfm=dfm,
                rx=rc[..., 0] * 30., ry=rc[..., 1] * 26.65 + 26.65, rmask=np.asarray(b['route_recv_mask'], bool),
                rfm=np.asarray(b['route_frame_mask'], bool),
                lb=np.array([str(x).upper().startswith('LINE') for x in p['defense_positions']] + [False] * (8 - len(p['defense_positions']))),
                features=np.asarray(b['features'], np.float32)[dfm], n_def=int(p['n_def']),
                frames=np.asarray(p['frame_ids'], int), defenders=[str(x) for x in p['defender_ids']],
                offense=[str(x) for x in p['offense_player_ids']], qb=np.asarray(p['qb_data'], np.float64)[0, :, :2])
    return out


def receiver_slots(r, mirror):
    """Six side slots (L1..L3, R1..R3), each (present, x(F), u(F)); u = lateral offset from the ball."""
    slots = [None] * 6
    valid = [i for i in range(5) if r['rmask'][i] and r['rfm'][i].any()]
    tracks = []
    for i in valid:
        last = int(np.flatnonzero(r['rfm'][i])[-1])
        idx = [min(f, last) for f in FRAMES]
        x = r['rx'][i, idx]
        u = r['ry'][i, idx] - r['yball']
        if mirror:
            u = -u
        tracks.append((u[0], x, u))
    left = sorted([t for t in tracks if t[0] < 0], key=lambda t: t[0])[:3]          # widest (most negative) first
    right = sorted([t for t in tracks if t[0] >= 0], key=lambda t: -t[0])[:3]       # widest (most positive) first
    for k, t in enumerate(left):
        slots[k] = t
    for k, t in enumerate(right):
        slots[3 + k] = t
    return slots


def descriptor(r, mirror):
    parts = []
    for s in receiver_slots(r, mirror):
        if s is None:
            parts.append(np.zeros(1 + 2 * len(FRAMES)))
        else:
            parts.append(np.concatenate([[W_PRESENT], W_ROUTE * s[1], W_ROUTE * s[2]]))
    dx = r['snap'][r['def_mask'], 0] - r['los']
    du = r['snap'][r['def_mask'], 1] - r['yball']
    if mirror:
        du = -du
    order = np.argsort(du)
    d = np.zeros((8, 3))
    for k, j in enumerate(order[:8]):
        d[k] = (W_DEF_PRESENT, W_DEF * dx[j], W_DEF * du[j])
    parts.append(d.ravel())
    parts.append([W_TIME * r['n_frames'] / 10.])
    return np.concatenate(parts)


def knn(recs):
    fam = np.array([r['family'] for r in recs])
    games = np.array([r['game'] for r in recs])
    train = np.array([r['split'] == 'train' for r in recs])
    D0 = np.stack([descriptor(r, False) for r in recs])
    D1 = np.stack([descriptor(r, True) for r in recs])
    assert np.isfinite(D0).all() and np.isfinite(D1).all()
    neighbors = [None] * len(recs)
    for f in sorted(set(fam)):
        q_idx = np.flatnonzero(fam == f)
        b_idx = np.flatnonzero((fam == f) & train)
        bank = np.concatenate([D0[b_idx], D1[b_idx]])
        bank_play = np.concatenate([b_idx, b_idx])
        bank_mirror = np.concatenate([np.zeros(len(b_idx), bool), np.ones(len(b_idx), bool)])
        bank_game = games[bank_play]
        bn = (bank ** 2).sum(1)
        for s in range(0, len(q_idx), 512):
            qi = q_idx[s:s + 512]
            Q = D0[qi]
            dist = (Q ** 2).sum(1)[:, None] + bn[None] - 2. * Q @ bank.T
            dist[games[qi][:, None] == bank_game[None]] = np.inf          # other games only (excludes self)
            top = np.argpartition(dist, K, axis=1)[:, :K]
            for row, q in enumerate(qi):
                t = top[row][np.argsort(dist[row, top[row]])]
                assert np.isfinite(dist[row, t]).all()
                neighbors[q] = [(int(bank_play[j]), bool(bank_mirror[j]), float(np.sqrt(max(dist[row, j], 0.)))) for j in t]
        print('family', f, 'queries', len(q_idx), 'bank plays', len(b_idx), flush=True)
    return neighbors


def aggregate(recs, neighbors):
    side, matched, mirrored = {}, [], 0
    for q, r in enumerate(recs):
        qd = np.flatnonzero(r['def_mask'])
        Qxy = np.stack([r['snap'][qd, 0] - r['los'], r['snap'][qd, 1] - r['yball']], 1)
        analog = np.full((8, K, MAXT, 2), np.nan, np.float32)
        match = np.full((8, K), np.nan, np.float32)
        offset = np.full((8, K, 2), np.nan, np.float32)
        dists = np.array([d for _, _, d in neighbors[q]])
        w = np.ones(K) if TAU <= 0 else np.exp(-(dists - dists.min()) / TAU)
        for k, (n, mirror, _) in enumerate(neighbors[q]):
            nr = recs[n]
            assert nr['split'] == 'train' and nr['game'] != r['game'] and nr['family'] == r['family']
            nd = np.flatnonzero(nr['def_mask'])
            Nxy = np.stack([nr['snap'][nd, 0] - nr['los'], nr['snap'][nd, 1] - nr['yball']], 1)
            if mirror:
                Nxy[:, 1] *= -1
            cost = np.sqrt(((Qxy[:, None] - Nxy[None]) ** 2).sum(-1))
            cost_match = cost + POS_PENALTY * (r['lb'][qd][:, None] != nr['lb'][nd][None])
            ri, ci = linear_sum_assignment(cost_match)
            for a, c in zip(ri, ci):
                if cost[a, c] > MATCH_MAX:
                    continue
                i, j = qd[a], nd[c]
                tr = nr['target'][j].copy()
                if mirror:
                    tr[:, 1] *= -1
                tr[~nr['dfm'][j]] = np.nan
                analog[i, k] = tr
                match[i, k] = cost[a, c]
                offset[i, k] = Nxy[c] - Qxy[a]
        valid = ~np.isnan(analog[..., 0])                                   # [8, K, 80]
        wk = w[None, :, None] * valid
        wsum = wk.sum(1)
        a0 = np.nan_to_num(analog)
        mean = (wk[..., None] * a0).sum(1) / np.maximum(wsum, 1e-9)[..., None]
        var = (wk[..., None] * (a0 - mean[:, None]) ** 2).sum(1) / np.maximum(wsum, 1e-9)[..., None]
        mean[wsum == 0] = 0.
        var[wsum == 0] = 0.
        ch = np.zeros((8, MAXT, N_CH), np.float32)
        ch[..., 0:2] = mean / 10.
        ch[..., 2:4] = np.sqrt(var) / 10.
        ch[..., 4] = valid.sum(1) / K
        okm = np.isfinite(match)
        mw = w[None] * okm
        msum = np.maximum(mw.sum(1), 1e-9)
        ch[..., 5] = ((mw * np.nan_to_num(offset[..., 0])).sum(1) / msum / 10.)[:, None]
        ch[..., 6] = ((mw * np.nan_to_num(offset[..., 1])).sum(1) / msum / 10.)[:, None]
        ch[..., 7] = ((mw * np.nan_to_num(match)).sum(1) / msum / 10.)[:, None]
        ch[~r['def_mask']] = 0.
        side[r['gp']] = torch.tensor(ch, dtype=torch.float16)
        matched.append(float(okm[qd].mean()))
        mirrored += sum(m for _, m, _ in neighbors[q])
        if q % 5000 == 0:
            print('aggregated', q, flush=True)
    return side, float(np.mean(matched)), mirrored / (K * len(recs))


def split(fold, records, plays):
    """-> play_ids DataFrame (game_play_id, game_id, n_def, split) of fold K, sorted by play."""
    folds = pd.read_csv(REPO / 'metadata' / 'folds.csv', dtype={'game_id': str})
    held = set(folds.game_id[folds.fold.eq(fold)])
    rows = []
    for gp in sorted(records):
        game = gp.split('_')[0]
        p = plays.loc[gp]
        if game in held:
            if p.season == 2018 and p.coverage_family == 'quarters' and p.kept_out_of_validation == 0:
                rows.append((gp, game, records[gp]['n_def'], 'validation'))
        else:
            rows.append((gp, game, records[gp]['n_def'], 'train'))
    return pd.DataFrame(rows, columns=['game_play_id', 'game_id', 'n_def', 'split'])


def statistics(records, train):
    x = np.concatenate([records[gp]['features'] for gp in train]).astype(np.float64)
    mean, std = x.mean(0), x.std(0)
    return dict(feature_mean=dict(zip(FEATURE_NAMES, map(float, mean))), feature_std=dict(zip(FEATURE_NAMES, map(float, np.maximum(std, 1e-6)))))


def facing(o, going_left):
    """(sin o, cos o) of the published angle, turned with the play (offense toward +x)."""
    s = -1.0 if going_left else 1.0
    r = np.deg2rad(o)
    return np.stack((s * np.sin(r), s * np.cos(r)), -1)


def orientation(records):
    """{gp: qb [80, 3], recv [5, 80, 3], dfn [8, 3]} from the published `o` at the record's frames."""
    out = {}
    for season in (2018, 2021, 2022, 2023):
        wanted = {gp for gp in records if plays_season(gp) == season}
        f = read_frames(season, ['game_play_id', 'nfl_id', 'frame_id', 'side', 'position', 'x', 'o', 'going_left'], wanted)
        for gp, g in f.groupby('game_play_id', sort=False):
            r = records[gp]; frames = r['frames']; T = min(len(frames), MAXT); left = bool(g.going_left.iloc[0])
            by = {}
            for nid, h in g.groupby('nfl_id'):
                h = h.drop_duplicates('frame_id').set_index('frame_id').reindex(frames)
                by[nid] = (h.o.to_numpy(np.float64), h.x.to_numpy(np.float64))
            qb = np.zeros((MAXT, 3), np.float32); recv = np.zeros((5, MAXT, 3), np.float32); dfn = np.zeros((8, 3), np.float32)
            for j, nid in enumerate(r['defenders'][:8]):
                if nid in by and np.isfinite(by[nid][0][0]):
                    dfn[j, :2] = facing(by[nid][0][0], left); dfn[j, 2] = 1.
            qbs = sorted(set(g.nfl_id[g.side.eq('offense') & g.position.eq('QB')]))
            if qbs and qbs[0] in by:
                o, x = by[qbs[0]]; ok = (np.isfinite(o) & np.isfinite(x))[:T]
                qb[:T][ok, :2] = facing(o[:T][ok], left); qb[:T][ok, 2] = 1.
            for k, nid in enumerate(r['offense'][:5]):
                if r['rmask'][k] and nid in by:
                    o, x = by[nid]; ok = (np.isfinite(o) & np.isfinite(x))[:T]
                    recv[k, :T][ok, :2] = facing(o[:T][ok], left); recv[k, :T][ok, 2] = 1.
            out[gp] = {k: torch.as_tensor(v, dtype=torch.float16) for k, v in dict(qb=qb, recv=recv, dfn=dfn).items()}
        print('orientation', season, len(out), flush=True)
    return out


def plays_season(gp):
    y = int(gp[:4])
    return 2023 if y == 2024 else y


def position_blocker(records):
    d = read_defenders()
    pos_of = {(g, str(n)): p for g, n, p in zip(d.game_play_id, d.nfl_id, d.position)}
    rf = read_route_flags()
    flag = {(g, str(n)): (int(s), bool(r)) for g, n, s, r in zip(rf.game_play_id, rf.nfl_id, rf.slot, rf.ran_route)}
    out = {}
    for gp, r in records.items():
        pos = torch.zeros(8, dtype=torch.int8); blk = torch.zeros(5, dtype=torch.float32)
        for i, nid in enumerate(r['defenders'][:8]):
            pos[i] = POSITION_CLASS.get(pos_of.get((gp, nid)), 0)
            assert pos[i] > 0, (gp, nid, pos_of.get((gp, nid)))
        for j, oid in enumerate(r['offense'][:5]):
            f = flag.get((gp, oid))
            if f is not None:
                assert f[0] == j, (gp, oid)
                blk[j] = float(not f[1])
        out[gp] = dict(pos=pos, blk=blk)
    return out


def write_side(fold, name, side, **info):
    raw = pt_bytes(side)
    (fold_dir(fold) / f'{name}.pt').write_bytes(raw)
    (fold_dir(fold) / f'{name}_manifest.json').write_text(json.dumps(dict(plays=len(side), side_sha256=sha256(raw), **info), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, choices=(0, 1, 2, 3))
    args = ap.parse_args(argv)
    plays = read_plays().set_index('game_play_id')
    records = read_records()
    print('records', len(records), flush=True)
    orient = orientation(records)
    posblk = position_blocker(records)
    for fold in [args.fold] if args.fold is not None else (0, 1, 2, 3):
        out = fold_dir(fold); out.mkdir(parents=True, exist_ok=True)
        ids = split(fold, records, plays)
        ids.to_csv(out / 'play_ids.csv', index=False)
        train = ids.game_play_id[ids.split.eq('train')].tolist()
        (out / 'stats.json').write_text(json.dumps(statistics(records, train), indent=2))
        n_rows = sum(int(records[gp]['dfm'].sum()) for gp in train)
        (out / 'population.json').write_text(json.dumps(dict(training_plays=len(train), validation_plays=int(ids.split.eq('validation').sum()),
                                                             training_rows=n_rows, training_mean_native_rows=n_rows / len(train), max_seq_len=MAXT,
                                                             fold=fold), indent=2))
        recs = [dict(records[gp], split=s) for gp, s in zip(ids.game_play_id, ids.split)]
        side, matched, mirrored = aggregate(recs, knn(recs))
        write_side(fold, 'retrieval', side, K=K, channels=N_CH, mean_matched_fraction=matched, mirrored_share=mirrored,
                   legality=f'bank = fold-{fold} training split, same coverage family, another game')
        write_side(fold, 'orientation', {gp: orient[gp] for gp in ids.game_play_id})
        write_side(fold, 'position_blocker', {gp: posblk[gp] for gp in ids.game_play_id})
        print(f'fold {fold}: {len(train):,} training / {int(ids.split.eq("validation").sum()):,} validation plays', flush=True)


if __name__ == '__main__':
    main()
