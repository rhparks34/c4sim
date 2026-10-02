"""The model: a scene transformer over defenders, route runners and the quarterback that refines every defender's path from the
snap in three rounds, plus a route-break readout (a detached fourth round with its own loss).

Shapes: batch B, defenders d (8), route runners R (5), frames t (<= 80), width 192. Positions are yards; outputs are displacements
from each defender's snap position.
"""
import torch
from torch import nn
from torch.nn import functional as F

from src.recipe import RECIPE

MAX_T = RECIPE.max_frames   # frames after the snap
WIDTH = 192
QB_FEATURES = (33, 34)      # columns of the defender features holding the QB's offset from the defender (qb_dx, qb_dy)
N_OPTIONS = 12              # coded Cover-4 rule options per defender (+ one learned "keep doing what you do" token)
RULE_DIM = 64
N_POSITIONS = 10            # roster position classes (0 = empty slot)
N_RECEIVER_SLOTS = 5        # nearest-receiver feature blocks per defender

# Per-batch side inputs (analogs, orientation, position, blocker flag, receiver-slot validity), stashed by the loader before each
# forward.
CURRENT = {'side': None}
SIDE_KEYS = ('analogs', 'orientation_qb', 'orientation_receiver', 'orientation_defender', 'position', 'blocker', 'receiver_slots')


def stash(batch):
    side = {k: batch[k] for k in batch if k in SIDE_KEYS}
    side['def_mask'] = batch['def_mask']; side['frame_mask'] = batch['frame_mask']
    CURRENT['side'] = side


def pair_geometry(positions):
    """[B,T,P,2] absolute yards -> [B,T,query,key,5]: offsets, distance and direction between every pair of players."""
    delta = positions[:, :, None, :, :] - positions[:, :, :, None, :]
    distance = (delta.square().sum(-1) + 1e-6).sqrt()
    return torch.stack((delta[..., 0] / 30., delta[..., 1] / 26.65,
                        distance / 40., delta[..., 0] / distance,
                        delta[..., 1] / distance), -1)


class SceneBlock(nn.Module):
    """Attention over time for each player, then over players at each frame with a bias and messages from their geometry."""

    def __init__(self, width=WIDTH, heads=4, dropout=.15):
        super().__init__()
        self.heads = heads
        self.norm_time = nn.LayerNorm(width)
        self.time_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_player = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.edge_bias = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, heads))
        self.edge_value = nn.Linear(5, width, bias=False)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid, edges):
        b, t, p, e = x.shape
        # Full-horizon attention over time per player; absent players get one dummy key and their outputs are discarded.
        z = self.norm_time(x).transpose(1, 2).reshape(b * p, t, e)
        tm = valid.transpose(1, 2).reshape(b * p, t).clone()
        tm[:, 0] |= ~tm.any(-1)
        z = self.time_attn(z, z, z, key_padding_mask=~tm, need_weights=False)[0]
        x = torch.where(valid[..., None], x + self.drop(z.reshape(b, p, t, e).transpose(1, 2)), 0.)
        qkv = self.qkv(self.norm_player(x)).reshape(b, t, p, 3, self.heads, e // self.heads)
        q, k, v = (qkv[:, :, :, j].permute(0, 1, 3, 2, 4) for j in range(3))
        logits = (q @ k.transpose(-1, -2)) * ((e // self.heads) ** -.5)
        logits = logits + self.edge_bias(edges).permute(0, 1, 4, 2, 3)
        weights = logits.masked_fill(~valid[:, :, None, None, :], -1e4).softmax(-1)
        weights = weights * valid[:, :, None, None, :]
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-12)
        context = (self.drop(weights) @ v).transpose(2, 3).reshape(b, t, p, e)
        # Displacement messages: reduced in the five geometric coordinates before the projection (linear, so equivalent).
        edge_message = torch.einsum('btqk,btqkf->btqf', weights.mean(2), edges)
        context = context + self.edge_value(edge_message)
        x = torch.where(valid[..., None], x + self.drop(self.out(context)), 0.)
        return torch.where(valid[..., None], x + self.drop(self.ff(x)), 0.)


class AnchorHead(nn.Module):
    """Receiver-anchored offsets: p = own + sum_r alpha_r * (receiver_r(t) - snap + offset_r - own), alpha = softmax over
    {no receiver, each route runner valid at t}; logits and cushion offsets from the defender token, the receiver token and
    their geometry."""

    def __init__(self, width=WIDTH, inner=64):
        super().__init__()
        self.d = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, inner))
        self.r = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, inner))
        self.g = nn.Linear(5, inner)
        self.pair = nn.Sequential(nn.GELU(), nn.Linear(inner, 3))
        self.none = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        with torch.no_grad():
            self.none[1].bias.fill_(6.)          # starts close to the plain head: each receiver weight ~0.0025

    def forward(self, h_def, h_recv, own, snap_xy, recv_xy, recv_valid):
        """h_def [B,d,t,E], h_recv [B,R,t,E], own [B,d,t,2], snap_xy [B,d,2], recv_xy [B,R,t,2], recv_valid [B,R,t]
        -> displacement [B,d,t,2] and receiver weights [B,d,R,t]."""
        pos = (snap_xy[:, :, None] + own.detach())[:, :, None]
        delta = recv_xy[:, None] - pos
        dist = (delta.square().sum(-1) + 1e-6).sqrt()
        geo = torch.stack((delta[..., 0] / 30., delta[..., 1] / 26.65, dist / 40., delta[..., 0] / dist, delta[..., 1] / dist), -1)
        pair = self.pair(self.d(h_def)[:, :, None] + self.r(h_recv)[:, None] + self.g(geo))
        logit = pair[..., 0].masked_fill(~recv_valid[:, None], -1e4)
        none = self.none(h_def)[..., 0][:, :, None]
        alpha = torch.cat((none, logit), 2).softmax(2)[:, :, 1:]
        anchored = recv_xy[:, None] - snap_xy[:, :, None, None] + pair[..., 1:3]
        return own + (alpha[..., None] * (anchored - own[:, :, None])).sum(2), alpha


class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear_gate = nn.Linear(in_dim, hidden_dim)
        self.linear_feat = nn.Linear(in_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear_out(F.silu(self.linear_gate(x)) * self.linear_feat(x)))


class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.ffn = SwiGLU(dim, hidden_dim, dim, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class RouteEncoder(nn.Module):
    """Bidirectional transformer over each route runner's whole tracked path (inputs: depth, lateral, vx, vy per frame)."""

    def __init__(self, n_route_feats=4, embed_dim=64, n_heads=4, n_layers=2, dropout=.15):
        super().__init__()
        self.input_proj = nn.Linear(n_route_feats, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, route_feats, route_frame_mask):
        B, R, T, F_ = route_feats.shape
        x = self.input_norm(self.input_proj(route_feats.reshape(B * R, T, F_)))
        pad = route_frame_mask.reshape(B * R, T) == 0
        # Attention returns NaN when every token is masked: empty route slots run unmasked and are zeroed afterwards.
        all_masked = pad.all(dim=1)
        if all_masked.any():
            pad = pad.clone()
            pad[all_masked] = False
        x = self.transformer(x, src_key_padding_mask=pad)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x * route_frame_mask.reshape(B * R, T).unsqueeze(-1).to(dtype=x.dtype)
        return x.reshape(B, R, T, -1)


def _unit(v):
    return v / (v.square().sum(-1, keepdim=True) + 1e-6).sqrt()


def _cos_sin(f, u):
    return torch.stack(((f * u).sum(-1), f[..., 0] * u[..., 1] - f[..., 1] * u[..., 0]), -1)


def orientation_channels(side, t, snap_xy, qb_xy, qb_valid, route_xy, rm, dm):
    """Body-orientation inputs -> defender [B,d,t,8], route runner [B,R,t,8], QB [B,t,3]; zero where the token is invalid.
    Defender: own facing at the snap (2), valid, QB facing at t (2), QB valid, cos/sin(QB facing, bearing QB -> own snap).
    Route runner: facing (2), valid, cos/sin(facing, bearing to QB) ("looking back"), cos/sin(QB facing, bearing QB -> him), QB valid."""
    dev = snap_xy.device; d = snap_xy.shape[1]; R = route_xy.shape[1]
    q = side['orientation_qb'].to(dev).float()[:, :t]
    rc = side['orientation_receiver'].to(dev).float()[:, :R, :t]
    df = side['orientation_defender'].to(dev).float()[:, :d]
    qv = (q[..., 2] > 0) & qb_valid
    qf = q[..., :2] * qv[..., None]
    rv = (rc[..., 2] > 0) & rm
    rf = rc[..., :2] * rv[..., None]
    qvf = qv.float()
    bearing = _unit(snap_xy[:, :, None] - qb_xy[:, None])
    cs = _cos_sin(qf[:, None].expand(-1, d, -1, -1), bearing) * qvf[:, None, :, None]
    def_ch = torch.cat((df[:, :, None, :2].expand(-1, -1, t, -1), df[:, :, None, 2:3].expand(-1, -1, t, -1),
                        qf[:, None].expand(-1, d, -1, -1), qvf[:, None, :, None].expand(-1, d, -1, -1), cs), -1)
    def_ch = torch.where(dm[..., None], def_ch, 0.)
    to_qb = _unit(qb_xy[:, None] - route_xy)
    back = _cos_sin(rf, to_qb) * rv[..., None]
    look = _cos_sin(qf[:, None].expand(-1, R, -1, -1), -to_qb) * (qv[:, None] & rm)[..., None]
    recv_ch = torch.cat((rf, rv[..., None].float(), back, look, (qv[:, None] & rm)[..., None].float()), -1)
    recv_ch = torch.where(rm[..., None], recv_ch, 0.)
    qb_ch = torch.cat((qf, qvf[..., None]), -1)
    return def_ch, recv_ch, qb_ch


def readout_head():
    return nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, 2))


def confidence_head():
    return nn.Sequential(nn.LayerNorm(WIDTH), ResidualBlock(WIDTH, 2 * WIDTH, .15), nn.Linear(WIDTH, 2))


class C4Sim(nn.Module):
    """Predicts every coverage defender's path after the snap (standard output) and the route-break readout.
    Construction order fixes the parameter order (and so the gradient-clipping reductions): keep it."""

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(51, WIDTH)
        self.input_norm = nn.LayerNorm(WIDTH)
        self.route_encoder = RouteEncoder()
        self.conf_head = confidence_head()
        # Coded-rule reaction encoder: each defender's boundary state queries his candidate rule options.
        self.rule_preview_encoder = nn.GRU(11, 32, batch_first=True)       # ten-step preview of each option -> 32
        self.rule_token_mlp = nn.Sequential(nn.Linear(68 + 32, RULE_DIM), nn.GELU(), nn.Linear(RULE_DIM, RULE_DIM))
        self.rule_query_mlp = nn.Sequential(nn.Linear(28 + WIDTH, RULE_DIM), nn.GELU(), nn.Linear(RULE_DIM, RULE_DIM))
        self.rule_attention = nn.MultiheadAttention(RULE_DIM, 4, dropout=.15, batch_first=True)
        self.rule_post = nn.Sequential(nn.LayerNorm(RULE_DIM), nn.Linear(RULE_DIM, RULE_DIM), nn.GELU())
        self.rule_velocity_head = nn.Linear(RULE_DIM, 20)           # velocity residuals, 10 steps x 2
        self.rule_displacement_head = nn.Linear(RULE_DIM, 6)         # displacement at 3 horizons x 2
        self.teacher_acceleration_head = nn.Sequential(nn.Linear(2 * RULE_DIM, 32), nn.GELU(), nn.Linear(32, 2))   # teacher targets (training)
        self.teacher_velocity_head = nn.Linear(RULE_DIM, 2)
        self.rule_option_embedding = nn.Parameter(torch.zeros(N_OPTIONS + 1, RULE_DIM))
        self.rule_broadcast = nn.Linear(RULE_DIM, WIDTH)
        self.rule_boundary_proj = nn.Linear(29, WIDTH)
        self.zone_landmark_proj = nn.Sequential(nn.Linear(13 * 7, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH))
        self.route_proj = nn.Linear(64, WIDTH)
        self.qb_proj = nn.Linear(2, WIDTH)
        self.time = nn.Embedding(MAX_T, WIDTH)
        self.kind = nn.Embedding(3, WIDTH)                     # 0 route runner, 1 defender, 2 quarterback
        self.proposal_proj = nn.ModuleList([nn.Linear(2, WIDTH) for _ in range(2)])
        self.layers = nn.ModuleList([SceneBlock() for _ in range(6)])
        self.heads = nn.ModuleList([readout_head() for _ in range(3)])
        self.register_buffer('qb_mean', torch.zeros(2))       # feature statistics of qb_dx / qb_dy (loaded with the weights)
        self.register_buffer('qb_std', torch.ones(2))
        self.analog_proj = nn.Linear(8, WIDTH)
        self.anchor = nn.ModuleList([AnchorHead() for _ in range(3)])
        self.orientation_defender_proj = nn.Linear(8, WIDTH)
        self.orientation_receiver_proj = nn.Linear(8, WIDTH)
        self.orientation_qb_proj = nn.Linear(3, WIDTH)
        self.position_embedding = nn.Embedding(N_POSITIONS, WIDTH)
        self.blocker_proj = nn.Linear(1, WIDTH, bias=False)
        # Route-break readout: a fourth round on the detached round-3 inputs.
        self.route_break_proposal_proj = nn.Linear(2, WIDTH)
        self.route_break_layers = nn.ModuleList([SceneBlock() for _ in range(2)])
        self.route_break_head = readout_head()
        self.route_break_anchor = AnchorHead()
        self.route_break_conf = confidence_head()
        # Which of the nearest-receiver feature blocks hold a real route runner at each frame (an empty block's zeros would
        # otherwise read as a receiver standing on the defender's snap spot).
        self.receiver_slot_proj = nn.Linear(N_RECEIVER_SLOTS, WIDTH, bias=False)
        self.last_stages = None
        self.route_break_out = None

    def side_input(self, def_mask, frame_mask, t):
        side = CURRENT['side']
        dev = def_mask.device
        torch.testing.assert_close(side['def_mask'].to(dev).float(), def_mask.float(), rtol=0, atol=0)
        torch.testing.assert_close(side['frame_mask'].to(dev).float(), frame_mask.float(), rtol=0, atol=0)
        return side['analogs'].to(dev).float()[:, :, :t]

    def defender_extra(self, def_ch, d):
        """Orientation channels + roster position class of each defender slot + the frame's receiver-slot validity."""
        out = self.orientation_defender_proj(def_ch)
        pos = CURRENT['side']['position'].to(out.device).long()
        out = out + self.position_embedding(pos[:, :d])[:, :, None]
        slots = CURRENT['side']['receiver_slots'].to(out.device, out.dtype)
        return out + self.receiver_slot_proj(slots[:, :out.shape[2]])[:, None]

    def rule_reaction(self, x, tokens, previews, opt_valid, boundary, def_mask):
        """Each defender's boundary state (+ his snap-frame hidden) attends over his rule options -> reaction latent z [B,d,64]."""
        B, D, T, E = x.shape
        O = N_OPTIONS
        _, h_n = self.rule_preview_encoder(previews.reshape(B * D * O, 11, 11))
        prev_emb = h_n.squeeze(0).reshape(B, D, O, 32)
        tok_emb = self.rule_token_mlp(torch.cat([tokens, prev_emb], dim=-1))
        tok_emb = tok_emb + self.rule_option_embedding[None, None, :O, :]
        pers = self.rule_option_embedding[O][None, None, None, :].expand(B, D, 1, tok_emb.shape[-1])
        keys = torch.cat([tok_emb, pers], dim=2)
        kv_valid = torch.cat([opt_valid, opt_valid.new_ones(B, D, 1)], dim=2)
        query = self.rule_query_mlp(torch.cat([boundary, x[:, :, 0, :]], dim=-1))
        q = query.reshape(B * D, 1, -1)
        k = keys.reshape(B * D, O + 1, -1)
        kpm = kv_valid.reshape(B * D, O + 1) == 0
        ctx, _ = self.rule_attention(q, k, k, key_padding_mask=kpm, need_weights=True)   # need_weights=True selects the math path used in training
        z = self.rule_post(ctx.squeeze(1)).reshape(B, D, -1)
        z = z * def_mask[..., None]
        # The heads run in this order on purpose: it fixes the order in which their gradients add up in z.
        out = dict(z=z, v_res=self.rule_velocity_head(z).reshape(B, D, 10, 2), cumdisp=self.rule_displacement_head(z).reshape(B, D, 3, 2))
        zexp = z.unsqueeze(2).expand(B, D, O, z.shape[-1])
        out['teacher_acceleration'] = self.teacher_acceleration_head(torch.cat([tok_emb, zexp], dim=-1))
        out['teacher_velocity'] = self.teacher_velocity_head(z)
        return out

    def forward(self, features, route_context, route_recv_mask=None, route_frame_mask=None,
                def_mask=None, frame_mask=None, def_frame_mask=None, snap_xy=None, los_x=None,
                zone_landmarks=None, rule_tokens=None, rule_previews=None, rule_option_valid=None,
                rule_boundary=None, rule_boundary_valid=None, return_reaction=False):
        b, d, t, _ = features.shape
        assert t <= MAX_T and features.shape[-1] == 51
        dm = def_mask.bool()[:, :, None] & frame_mask.bool()[:, None, :]
        if def_frame_mask is not None:
            dm = dm & def_frame_mask.bool()
        rm = route_recv_mask.bool()[:, :, None] & route_frame_mask.bool() & frame_mask.bool()[:, None, :]
        features = torch.where(dm[..., None], features, 0.)
        routes = torch.where(rm[..., None], route_context, 0.)
        side = torch.where(dm[..., None], self.side_input(def_mask, frame_mask, t), 0.)
        # The QB's position, rebuilt from each defender's measured offset to him and averaged over the valid defenders.
        qb_per_def = features[..., list(QB_FEATURES)] * self.qb_std + self.qb_mean + snap_xy[:, :, None]
        qb_xy = (qb_per_def * dm[..., None]).sum(1) / dm.sum(1).clamp(min=1)[..., None]
        qb_valid = dm.any(1) & frame_mask.bool()
        route_xy = torch.stack((routes[..., 0] * 30. + los_x[:, None, None],
                                routes[..., 1] * 26.65 + 26.65), -1)
        def_ch, recv_ch, qb_ch = orientation_channels(CURRENT['side'], t, snap_xy, qb_xy, qb_valid, route_xy, rm, dm)
        x = self.input_norm(self.input_proj(features) + self.analog_proj(side) + self.defender_extra(def_ch, d))
        x = x + self.rule_boundary_proj(torch.cat((rule_boundary, rule_boundary_valid[..., None]), -1))[:, :, None]
        x = x + self.zone_landmark_proj(zone_landmarks.flatten(2))[:, :, None]
        x = torch.where(dm[..., None], x, 0.)
        reaction = self.rule_reaction(x, rule_tokens, rule_previews, rule_option_valid, rule_boundary, def_mask)
        x = x + self.rule_broadcast(reaction['z'])[:, :, None]
        route_hidden = self.route_proj(self.route_encoder(routes, rm.to(routes.dtype)))
        n_routes = routes.shape[1]
        blk = CURRENT['side']['blocker'].to(features.device, route_hidden.dtype)
        route_hidden = route_hidden + (self.orientation_receiver_proj(recv_ch) + self.blocker_proj(blk[:, :n_routes, None])[:, :, None])
        qb_content = torch.stack(((qb_xy[..., 0] - los_x[:, None]) / 30.,
                                  (qb_xy[..., 1] - 26.65) / 26.65), -1)
        offense = torch.cat((route_hidden, (self.qb_proj(qb_content) + self.orientation_qb_proj(qb_ch))[:, None]), 1)
        off_valid = torch.cat((rm, qb_valid[:, None]), 1)
        offense_xy = torch.cat((route_xy, qb_xy[:, None]), 1)
        off = offense + self.kind.weight[0]
        off[:, -1] = offense[:, -1] + self.kind.weight[2]
        base = torch.cat((x + self.kind.weight[1], off), 1).transpose(1, 2)
        valid = torch.cat((dm, off_valid), 1).transpose(1, 2)
        base = base + self.time(torch.arange(t, device=features.device))[None, :, None]
        base = torch.where(valid[..., None], base, 0.)
        hidden = base
        proposal = features.new_zeros(b, d, t, 2)
        time_factor = (torch.arange(t, device=features.device, dtype=features.dtype) * .1 / 4.)[None, None, :, None]
        stages = []
        for round_idx in range(3):
            if round_idx == 2:
                trunk_hidden, trunk_proposal = hidden, proposal
            if round_idx:
                previous_content = self.proposal_proj[round_idx - 1](proposal / 10.)
                extra = torch.cat((previous_content, torch.zeros_like(off)), 1).transpose(1, 2)
                hidden = torch.where(valid[..., None], hidden + base + extra, 0.)
            # Geometry is measured from each defender's snap spot moved by the previous round's (detached) proposal.
            reference = snap_xy[:, :, None].expand(b, d, t, 2)
            if round_idx:
                reference = reference + proposal.detach()
            edges = pair_geometry(torch.cat((reference, offense_xy), 1).transpose(1, 2))
            for layer in self.layers[2 * round_idx:2 * round_idx + 2]:
                hidden = layer(hidden, valid, edges)
            defender_hidden = hidden[:, :, :d].transpose(1, 2)
            proposal = self.heads[round_idx](defender_hidden) * time_factor
            proposal, _ = self.anchor[round_idx](defender_hidden, hidden[:, :, d:d + n_routes].transpose(1, 2),
                                                 proposal, snap_xy, route_xy, rm)
            proposal = torch.where(dm[..., None], proposal, 0.)
            stages.append(proposal)
        self.last_stages = stages
        confidence = torch.where(dm[..., None], self.conf_head(defender_hidden), 0.)
        # Route-break readout: a second final round on the detached round-3 inputs (after the standard path, so the standard
        # output's dropout draws are unchanged).
        hs, ps0, bs = trunk_hidden.detach(), trunk_proposal.detach(), base.detach()
        extra_s = torch.cat((self.route_break_proposal_proj(ps0 / 10.), torch.zeros_like(off)), 1).transpose(1, 2)
        hs = torch.where(valid[..., None], hs + bs + extra_s, 0.)
        ref_s = snap_xy[:, :, None].expand(b, d, t, 2) + ps0
        edges_s = pair_geometry(torch.cat((ref_s, offense_xy), 1).transpose(1, 2))
        for layer in self.route_break_layers:
            hs = layer(hs, valid, edges_s)
        dh = hs[:, :, :d].transpose(1, 2)
        own_s = self.route_break_head(dh) * time_factor
        ps, alpha_s = self.route_break_anchor(dh, hs[:, :, d:d + n_routes].transpose(1, 2), own_s, snap_xy, route_xy, rm)
        ps = torch.where(dm[..., None], ps, 0.)
        cs = torch.where(dm[..., None], self.route_break_conf(dh), 0.)
        self.route_break_out = (ps, cs, alpha_s.detach())
        result = (proposal, confidence)
        return result + (reaction,) if return_reaction else result


class OutputView(nn.Module):
    """Evaluation view of the model: 'standard' (its return value), 'route_break' (the readout) or 'midpoint' ((standard +
    route_break) / 2 with the standard confidence). The standard pass runs the model and caches every output per call; the other
    views replay the cache in the same loader order."""

    def __init__(self, model):
        super().__init__()
        self.model = model; self.mode = 'standard'; self.cache = []; self.cursor = 0

    def forward(self, features, *args, **kwargs):
        if self.mode == 'standard':
            p, c = self.model(features, *args, **kwargs)
            ps, cs, _ = self.model.route_break_out
            self.cache.append((tuple(features.shape), p, c, ps, cs))
            return p, c
        shape, p, c, ps, cs = self.cache[self.cursor]; self.cursor += 1
        assert shape == tuple(features.shape), 'replay out of order'
        return (ps, cs) if self.mode == 'route_break' else ((p + ps) * .5, c)

    def reset(self, mode):
        if mode == 'standard':
            self.cache = []
        self.mode = mode; self.cursor = 0
