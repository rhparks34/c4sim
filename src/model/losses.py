"""Training losses: the late-weighted position loss, the loss on each refinement round, the rule-teacher losses and the openness
penalty of the route-break output, combined by `objective`.

Tensors: predictions and targets are displacements from each defender's snap position, [batch, defender, frame, 2] in yards.
"""
import torch
from torch.nn import functional as F

from src.recipe import RECIPE



def late_weighted_loss(pred, conf, target, def_frame_mask):
    """Gaussian negative log-likelihood (per-coordinate log-variance `conf`) plus a squared error whose frame weights grow
    exponentially along each defender's track, so late-play errors count more."""
    mask = def_frame_mask.to(pred.dtype)
    expanded = mask.unsqueeze(-1).expand_as(pred)
    squared = (pred - target).square()
    conf = torch.clamp(conf, min=-6.0, max=6.0)
    element_nll = 0.5 * (squared * torch.exp(-conf) + conf)
    base_nll = (element_nll * expanded).sum() / expanded.sum().clamp(min=1.0)

    rank = (mask.cumsum(dim=2) - 1.0).clamp(min=0.0)
    length = mask.sum(dim=2, keepdim=True)
    progress = torch.where(length > 1.0, rank / (length - 1.0).clamp(min=1.0), torch.zeros_like(rank))
    raw_weight = torch.exp(RECIPE.late_beta * (2.0 * progress - 1.0)) * mask
    track_mean = raw_weight.sum(dim=2, keepdim=True) / length.clamp(min=1.0)
    weight = torch.where(mask > 0, raw_weight / track_mean.clamp(min=1e-12),
                         torch.zeros_like(raw_weight)).unsqueeze(-1).expand_as(squared)
    weighted_mse = (squared * weight).sum() / weight.sum().clamp(min=1.0)
    return base_nll + RECIPE.position_weight * weighted_mse


def round_loss(stages, target, mask, mean_rows):
    """Squared error of the first two refinement rounds (weight 0.25 each), normalised by the training set's mean rows per play."""
    return sum(RECIPE.round_weight * ((s - target).square() * mask[..., None]).sum()
               / (2. * target.shape[0] * mean_rows) for s in stages[:2])


def rule_losses(reaction, batch, pred, lambdas):
    """Losses on the coded-rule reaction encoder. Segment: its 10-step velocity residuals and 3 cumulative displacements against
    the real early movement. Teacher (training only): privileged per-option acceleration and initial-velocity targets.
    -> (weighted total, {name: float})."""
    parts = {}
    total = pred.new_zeros(())
    target_abs = batch['target_abs']
    dfm = batch['def_frame_mask']
    def_mask = batch['def_mask']
    _, _, T, _ = target_abs.shape
    vpre = batch['rule_velocity_before']
    bh_valid = batch['rule_boundary_valid']
    w_def = torch.where(bh_valid > 0, torch.ones_like(bh_valid), torch.full_like(bh_valid, 0.25)) * def_mask
    dt = 0.1

    ns = min(10, T - 1)
    v_tgt = (target_abs[:, :, 1:ns + 1] - target_abs[:, :, :ns]) / dt
    v_res_tgt = v_tgt - vpre[:, :, None, :]
    step_mask = dfm[:, :, 1:ns + 1] * dfm[:, :, :ns]
    el = F.smooth_l1_loss(reaction['v_res'][:, :, :ns], v_res_tgt, reduction='none').mean(-1)
    denom = (step_mask * w_def[:, :, None]).sum().clamp(min=1.0)
    seg_v = (el * step_mask * w_def[:, :, None]).sum() / denom
    seg_d = pred.new_zeros(())
    n_d = 0
    for i, k in enumerate((2, 5, 10)):
        kk = min(k, T - 1)
        d_tgt = target_abs[:, :, kk] - vpre * (kk * dt)
        m = dfm[:, :, kk] * w_def
        el_d = F.smooth_l1_loss(reaction['cumdisp'][:, :, i], d_tgt, reduction='none').mean(-1)
        seg_d = seg_d + (el_d * m).sum() / m.sum().clamp(min=1.0)
        n_d += 1
    seg_d = seg_d / max(n_d, 1)
    l_seg = seg_v + 2.0 * seg_d
    total = total + lambdas['segment'] * l_seg
    parts['rule_segment_loss'] = float(l_seg.detach().cpu())

    if 'teacher_acceleration' in batch:
        pmask = batch['teacher_mask'] * def_mask[:, :, None]
        tgt = batch['teacher_acceleration'] / 7.0
        el = F.huber_loss(reaction['teacher_acceleration'], tgt, reduction='none').mean(-1)
        l_teacher = (el * pmask).sum() / pmask.sum().clamp(min=1.0)
        vmask = batch['teacher_velocity_valid'] * def_mask
        el_v = F.huber_loss(reaction['teacher_velocity'], batch['teacher_velocity'] / 0.5, reduction='none').mean(-1)
        l_teacher = l_teacher + (el_v * vmask).sum() / vmask.sum().clamp(min=1.0)
        total = total + lambdas['priv'] * l_teacher
        parts['rule_teacher_loss'] = float(l_teacher.detach().cpu())
    return total, parts


def receiver_xy(route_context, los_x, t):
    """Receiver positions from the route input, [batch, receiver, frame, 2] in absolute yards."""
    return torch.stack((route_context[..., 0] * 30. + los_x[:, None, None], route_context[..., 1] * 26.65 + 26.65), -1)[:, :, :t]


def openness(pred, target, snap_xy, route_context, los_x, def_frame_mask, route_recv_mask, route_frame_mask, frame_mask,
             tau=0.5, margin=0.5, cover_max=5.0, delta=1.0):
    """Penalty when a receiver the real defense covered closely (nearest real defender within cover_max yards) is left more open
    by the predicted defense: Huber(relu(open_pred - open_real - margin)). open_real = distance to the nearest real defender;
    open_pred = softmax(-d / tau)-weighted mean distance to the predicted defenders. -> (loss, rows, mean excess)."""
    t = pred.shape[2]
    ph = snap_xy[:, :, None] + pred; pr = snap_xy[:, :, None] + target
    rec = receiver_xy(route_context, los_x, t)
    rv = route_recv_mask.bool()[:, :, None] & route_frame_mask.bool()[:, :, :t] & frame_mask.bool()[:, None, :t]
    dv = def_frame_mask.bool()[:, None]
    dreal = ((rec[:, :, None] - pr[:, None]).square().sum(-1) + 1e-6).sqrt()
    dpred = ((rec[:, :, None] - ph[:, None]).square().sum(-1) + 1e-6).sqrt()
    oreal = torch.where(dv, dreal, torch.full_like(dreal, 1e3)).min(2).values
    w = torch.where(dv, -dpred / tau, torch.full_like(dpred, -1e4)).softmax(2)
    opred = (w * torch.where(dv, dpred, torch.zeros_like(dpred))).sum(2)
    rows = rv & dv.any(2) & (oreal <= cover_max)
    e = F.relu(opred - oreal - margin)
    h = torch.where(e < delta, .5 * e.square(), delta * (e - .5 * delta))
    n = rows.sum()
    return (h * rows).sum() / n.clamp(min=1), n, (e * rows).sum() / n.clamp(min=1)


def route_runner_mask(batch):
    """Receivers that ran a route: route-set players flagged as blockers are left out of the openness penalty."""
    rr = batch['route_recv_mask']
    return rr * (1. - batch['blocker'][:, :rr.shape[1]].to(rr.dtype))


def objective(model, batch, mean_rows, lambdas):
    """Standard output: late-weighted loss + round loss + rule losses. Route-break output (its own detached readout):
    late-weighted loss + RECIPE.openness_weight * openness penalty on route runners.
    -> (loss, parts, per-round squared error [3], rule-loss parts)."""
    p, c, reaction = model(**model_inputs(batch), return_reaction=True)
    mask = batch['def_frame_mask']
    main = late_weighted_loss(p, c, batch['target'], mask)
    intermediate = round_loss(model.last_stages, batch['target'], mask, mean_rows)
    teaching, teaching_parts = rule_losses(reaction, batch, p, lambdas)
    sse = torch.stack([((s.detach() - batch['target']).square() * mask[..., None]).sum() for s in model.last_stages])
    loss = main + intermediate + teaching
    ps, cs, _ = model.route_break_out
    main_s = late_weighted_loss(ps, cs, batch['target'], mask)
    runners = route_runner_mask(batch)
    cov, rows, excess = openness(ps, batch['target'], batch['snap_xy'], batch['route_context'], batch['los_x'],
                                 mask, runners, batch['route_frame_mask'], batch['frame_mask'])
    t = ps.shape[2]
    gone = ((batch['route_recv_mask'].bool() & ~runners.bool())[:, :, None] & batch['route_frame_mask'].bool()[:, :, :t]
            & batch['frame_mask'].bool()[:, None, :t])
    m = mask[..., None]
    parts = dict(main=main, round=intermediate, rule=teaching, route_break_main=main_s.detach(), route_break_openness=cov.detach(),
                 route_break_openness_excess=excess.detach(), route_break_openness_rows=rows.detach().float(),
                 route_break_blocker_frames=gone.sum().float(), route_break_sse=((ps.detach() - batch['target']).square() * m).sum())
    total = loss + main_s + RECIPE.openness_weight * cov
    return total, parts, sse, teaching_parts


MODEL_KEYS = ('features', 'route_context', 'route_recv_mask', 'route_frame_mask', 'def_mask', 'frame_mask', 'def_frame_mask',
              'snap_xy', 'los_x', 'zone_landmarks', 'rule_tokens', 'rule_previews', 'rule_option_valid', 'rule_boundary', 'rule_boundary_valid')


def model_inputs(batch):
    """The inputs the model may see (no targets, no training-only teacher tensors)."""
    return {k: batch[k] for k in MODEL_KEYS}
