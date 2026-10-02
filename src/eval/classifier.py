"""Coverage classifier read: does a frozen coverage classifier still call the play Quarters when an output's defenders replace the
real ones?

The classifier (a frame x player transformer, two seeds, per-seed temperature, mirror averaging) was trained on 2018 plays outside
fold 0, so only fold 0 is read. Per play: the deepest <= 7 defenders at the real snap, <= 5 receivers widest first and the QB, from
the snap to the first pass (at most 40 frames), positions relative to the line of scrimmage and mid-field, each track smoothed
(lam 512, ramp 12) after substitution. A source replaces the selected defenders it predicts; frames after its last predicted frame
hold that position. Population: fold-0 plays whose real coverage label is Quarters (456 plays). Reported: share of plays whose
most likely class is Quarters.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.eval.common import OUTPUTS, prediction_path
from src.eval.smoothing import smooth_track
from src.eval.sources import baselines, classifier_files, fold0_context_tracks, quarters_plays

CLASSES = ['cover0', 'cover1', 'cover2', 'cover3', 'cover6', 'man2', 'quarters', 'other']
QI = CLASSES.index('quarters')
N_TOK, MAX_F, N_DEF, N_REC = 13, 40, 7, 5
ELIGIBLE_POS = {'WR', 'TE', 'RB', 'FB', 'HB'}
Y_MID = 53.3 / 2.0
KEY = ['game_play_id', 'nfl_id', 'frame_id']


class Block(nn.Module):
    """Attention over frames for each player, then over players at each frame, then a feed-forward layer."""

    def __init__(self, d, heads):
        super().__init__()
        self.t_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.s_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h, pmask_cls, fmask):
        B, P1, Fr, D = h.shape
        x = h.reshape(B * P1, Fr, D)
        kpm = (~fmask).unsqueeze(1).expand(B, P1, Fr).reshape(B * P1, Fr)
        xa = self.n1(x)
        a, _ = self.t_attn(xa, xa, xa, key_padding_mask=kpm)
        x = x + torch.nan_to_num(a)
        x = x.reshape(B, P1, Fr, D).permute(0, 2, 1, 3).reshape(B * Fr, P1, D)
        spm = (~pmask_cls).unsqueeze(1).expand(B, Fr, P1).reshape(B * Fr, P1)
        xa = self.n2(x)
        a, _ = self.s_attn(xa, xa, xa, key_padding_mask=spm)
        x = x + torch.nan_to_num(a)
        x = x + self.ff(self.n3(x))
        return x.reshape(B, Fr, P1, D).permute(0, 2, 1, 3)


class CoverageNet(nn.Module):
    """Players x frames transformer with a class token per frame; attention-pooled play logits (frame and man/zone heads unused)."""

    def __init__(self, d=256, heads=8, blocks=3, n_cls=8):
        super().__init__()
        self.proj = nn.Linear(5, d)
        self.role_emb = nn.Embedding(4, d)      # 0 defender, 1 receiver, 2 quarterback, 3 empty slot
        self.season_emb = nn.Embedding(4, d)
        self.frame_pos = nn.Parameter(torch.zeros(MAX_F, d))
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, 1, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(blocks)])
        self.frame_head = nn.Linear(d, n_cls)
        self.pool_q = nn.Linear(d, 1)
        self.play_head = nn.Linear(d, n_cls)
        self.mz_head = nn.Linear(d, 1)

    def forward(self, feat, pmask, fmask, roles, season):
        B = feat.shape[0]
        r = roles.clamp(min=0).long()
        r[roles < 0] = 3
        h = self.proj(feat) + self.role_emb(r).unsqueeze(2) + self.frame_pos
        h = h + self.season_emb(season).unsqueeze(1).unsqueeze(1)
        cls = self.cls_tok.expand(B, 1, MAX_F, -1)
        h = torch.cat([cls, h], dim=1)
        pmask_cls = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=pmask.device), pmask], dim=1)
        for blk in self.blocks:
            h = blk(h, pmask_cls, fmask)
        cls_seq = h[:, 0]
        w = self.pool_q(cls_seq).squeeze(-1)
        w = w.masked_fill(~fmask, -1e9).softmax(dim=1)
        pooled = (cls_seq * w.unsqueeze(-1)).sum(dim=1)
        return self.play_head(pooled)


def features(X, nf, pmask):
    """(13, 40, 2) positions -> (13, 40, 5) inputs: x / 50, y / 26.65, velocity / 10, speed / 10 (zero outside the window)."""
    X = X.astype(np.float32).copy()
    v = np.zeros_like(X)
    v[:, 1:nf] = (X[:, 1:nf] - X[:, :nf - 1]) / 0.1
    if nf > 1:
        v[:, 0] = v[:, 1]
    sp = np.linalg.norm(v, axis=2, keepdims=True)
    feat = np.concatenate([X[:, :, 0:1] / 50.0, X[:, :, 1:2] / 26.65, v / 10.0, sp / 10.0], axis=2)
    fmask = np.zeros(MAX_F, dtype=bool); fmask[:nf] = True
    feat[:, ~fmask] = 0.0
    feat[~pmask] = 0.0
    return feat, fmask


def load_surface(df, xc, yc):
    df = df[KEY + [xc, yc]].copy()
    df['nfl_id'] = df.nfl_id.astype(str).str.replace(r'\.0$', '', regex=True)
    df = df.rename(columns={xc: 'x', yc: 'y'})
    listed = df.groupby('game_play_id').nfl_id.apply(lambda s: set(s)).to_dict()
    df = df.dropna(subset=['x', 'y']).sort_values(KEY)
    by = {k: (g.frame_id.to_numpy(), g[['x', 'y']].to_numpy(float)) for k, g in df.groupby(['game_play_id', 'nfl_id'], sort=False)}
    return {'listed': listed, 'by': by}


def build(plays, surf, ctx_by_gp):
    """Model inputs per source: {source: {'X', 'pmask', 'roles', 'n_frames', 'gp'}}."""
    names = ['real'] + list(surf)
    data = {s: {k: [] for k in ['X', 'pmask', 'roles', 'n_frames', 'gp']} for s in names}
    for gp in plays:
        cp = ctx_by_gp[gp]
        ev = cp[cp.event.notna()]
        snap = int(ev[ev.event == 'ball_snap'].frameId.min())
        thr = ev[ev.event == 'pass_forward'].frameId.min()
        if pd.isna(thr):
            thr = ev[ev.event == 'pass_shovel'].frameId.min()
        frames = list(range(snap, min(int(thr), snap + MAX_F - 1) + 1))
        fset = {f: i for i, f in enumerate(frames)}
        F = len(frames)
        modeled = set().union(*[s['listed'].get(gp, set()) for s in surf.values()])
        def_team = cp[cp.nfl_id_str.isin(modeled)].team.mode().iloc[0]
        ball = cp[cp.is_ball & (cp.frameId == frames[0])]
        if not len(ball):
            continue
        los = float(ball.x.iloc[0])
        players = cp[~cp.is_ball & cp.frameId.isin(fset)]

        def raw_track(pid):
            g = players[players.nfl_id_str == pid]
            if len(g) < max(4, F * 0.9):
                return None
            P = np.full((F, 2), np.nan)
            ix = [fset[f] for f in g.frameId]
            P[ix, 0] = g.x.to_numpy() - los
            P[ix, 1] = g.y.to_numpy() - Y_MID
            if np.isnan(P).any():
                ok = ~np.isnan(P[:, 0])
                P[:, 0] = np.interp(np.arange(F), np.flatnonzero(ok), P[ok, 0])
                P[:, 1] = np.interp(np.arange(F), np.flatnonzero(ok), P[ok, 1])
            return P

        info = players.drop_duplicates('nfl_id_str')
        d_tracks = {p: t for p in info[info.team == def_team].nfl_id_str if (t := raw_track(p)) is not None}
        r_tracks = {p: t for p in info[(info.team != def_team) & info.position.isin(ELIGIBLE_POS)].nfl_id_str if (t := raw_track(p)) is not None}
        q_tracks = {p: t for p in info[(info.team != def_team) & (info.position == 'QB')].nfl_id_str if (t := raw_track(p)) is not None}
        if len(d_tracks) < 4 or len(r_tracks) < 2 or not q_tracks:
            continue
        d_ids = sorted(d_tracks, key=lambda p: -d_tracks[p][0, 0])[:N_DEF]
        r_ids = sorted(r_tracks, key=lambda p: -abs(r_tracks[p][0, 1]))[:N_REC]
        q_ids = list(q_tracks)[:1]
        for s in names:
            X = np.zeros((N_TOK, MAX_F, 2), dtype=np.float32)
            pmask = np.zeros(N_TOK, dtype=bool)
            roles = np.full(N_TOK, -1, dtype=np.int8)
            slots = ([(i, 0, d_tracks[p], p) for i, p in enumerate(d_ids)] + [(N_DEF + i, 1, r_tracks[p], p) for i, p in enumerate(r_ids)]
                     + [(N_DEF + N_REC, 2, q_tracks[p], p) for p in q_ids])
            for slot, role, P, pid in slots:
                P = P.copy()
                if role == 0 and s != 'real' and pid in surf[s]['listed'].get(gp, set()) and (gp, pid) in surf[s]['by']:
                    fr, xy = surf[s]['by'][(gp, pid)]
                    inwin = np.array([f in fset for f in fr])
                    fr, xy = fr[inwin], xy[inwin]
                    if len(fr):
                        ix = np.array([fset[f] for f in fr])
                        P[ix, 0] = xy[:, 0] - los
                        P[ix, 1] = xy[:, 1] - Y_MID
                        last = ix.max()
                        if last < F - 1:
                            P[last + 1:, 0] = xy[-1, 0] - los
                            P[last + 1:, 1] = xy[-1, 1] - Y_MID
                sx, sy = smooth_track(P[:, 0].copy(), P[:, 1].copy())
                X[slot, :F, 0] = sx[:MAX_F]
                X[slot, :F, 1] = sy[:MAX_F]
                pmask[slot] = True
                roles[slot] = role
            d = data[s]
            d['X'].append(X); d['pmask'].append(pmask); d['roles'].append(roles); d['n_frames'].append(F); d['gp'].append(gp)
    return data


@torch.no_grad()
def posteriors(models, temps, d):
    """Class probabilities per play: mirror-averaged softmax, per-seed temperature, averaged over seeds."""
    n = len(d['gp']); prob_sum = None
    for model, T in zip(models, temps):
        outs = []
        for s in range(0, n, 64):
            batch = [features(d['X'][j], int(d['n_frames'][j]), d['pmask'][j]) for j in range(s, min(s + 64, n))]
            feat = torch.from_numpy(np.stack([b[0] for b in batch])); fmask = torch.from_numpy(np.stack([b[1] for b in batch]))
            pmask = torch.from_numpy(np.stack(d['pmask'][s:s + 64])); roles = torch.from_numpy(np.stack(d['roles'][s:s + 64]).astype(np.int64))
            season = torch.zeros(len(batch), dtype=torch.int64)   # every fold-0 play is from 2018
            p = model(feat, pmask, fmask, roles, season).softmax(1)
            f2 = feat.clone(); f2[:, :, :, 1] *= -1.0; f2[:, :, :, 3] *= -1.0
            p = 0.5 * (p + model(f2, pmask, fmask, roles, season).softmax(1))
            outs.append(p)
        p = torch.log(torch.cat(outs).clamp(min=1e-9)).div(T).softmax(1)
        prob_sum = p if prob_sum is None else prob_sum + p
    return (prob_sum / len(models)).numpy()


def score(root, threads=4):
    """Fold 0 -> {source: share of Quarters-labelled plays called Quarters}; sources real, the outputs and the static baseline."""
    torch.set_num_threads(threads)
    surf = {s: load_surface(pd.read_parquet(prediction_path(0, s, root), columns=KEY + ['candidate_x', 'candidate_y']), 'candidate_x', 'candidate_y')
            for s in OUTPUTS}
    surf['static'] = load_surface(baselines(0, KEY + ['static_x', 'static_y']), 'static_x', 'static_y')
    ctx = fold0_context_tracks()
    plays = sorted(set().union(*[set(s['listed']) for s in surf.values()]))
    ctx_by_gp = {gp: g for gp, g in ctx[ctx.game_play_id.isin(plays)].groupby('game_play_id')}
    data = build(plays, surf, ctx_by_gp)
    models, temps = [], []
    for sd in (0, 1):
        state, temp = classifier_files(sd)
        m = CoverageNet(d=256, blocks=3); m.load_state_dict(state); m.eval(); models.append(m); temps.append(temp)
    quarters = quarters_plays(QI)
    in_all = set.intersection(*[set(s['listed']) for s in surf.values()])
    gps = data['real']['gp']; main = np.array([(g in quarters) and (g in in_all) for g in gps])
    out = {'plays': int(main.sum()), 'games': int(len({g.split('_')[0] for g in np.array(gps)[main]}))}
    for s in data:
        p = posteriors(models, temps, data[s])
        out[s] = float((p[main].argmax(1) == QI).mean())
    return out
