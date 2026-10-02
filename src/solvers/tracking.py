"""Batched tracking solver: the control that keeps a defender on a moving target path over the next second under the same
force and power limits, from controlled starts, with the retry flags. Verbatim from the research code (w42_tracking_solver)."""
from __future__ import annotations

import math

import numpy as np

from src.solvers.effort import _effort_dynamics_step, make_body_params

DT = 0.1


F_MAX = 200.0


EMAX_RATE = 150.0            # Emax = 150 * H


QDOT_CLIP = 10.0             # yd/s, applied before the terminal term


TRACK_NORM = 4.0             # (2 yd)^2


TERMVEL_NORM = 16.0          # (4 yd/s)^2


TERMVEL_W = 0.25


ENERGY_W = 0.02


SMOOTH_W = 0.01


CEIL_W = 10.0


CONTROLLED_INITS = {
    "A": (0.50, 0.50),
    "B": (0.25, 0.20),
    "C": (0.75, 0.80),
}


PRODUCTION_INIT = "A"


def _record_iters(adam_iters: int) -> list[int]:
    lo = max(1, adam_iters - 100)
    pts = list(range(lo, adam_iters + 1, 25))
    if pts[-1] != adam_iters:
        pts.append(adam_iters)
    return pts


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def solve_tracking_batch(
    p0: np.ndarray,            # [B, 2]
    v0: np.ndarray,            # [B, 2]
    q: np.ndarray,             # [B, N_max+1, 2] target path q_0..q_N (padded)
    qdot: np.ndarray,          # [B, N_max+1, 2] target velocity (padded)
    n_active: np.ndarray,      # [B] int, active step count N_b (H_b = N_b*dt)
    terminal_mode: str = "target_velocity",   # or "free" | "zero"
    energy_mode: str = "ceiling",             # or "min_energy"
    f_max: float = F_MAX,
    dt: float = DT,
    emax_rate: float = EMAX_RATE,
    adam_iters: int = 300,
    lr: float = 0.08,
    init: str = PRODUCTION_INIT,
    device: str = "cpu",
    seed: int = 0,
):
    """One tracking solve per row.  Returns a dict of numpy arrays.

    terminal_mode:
      target_velocity — 0.25 ||v_N - qdot_N||^2 / 16 (qdot_N clipped to 10)
      free            — no terminal-velocity term (factorial candidate 3)
      zero            — soft vf=0 braking negative control (candidate 5)
    energy_mode:
      ceiling    — + 0.02 z_N/Emax + 10 relu(z_N/Emax - 1)^2
      min_energy — + 0.02 z_N/Emax only (candidate 6; no ceiling penalty)

    No output defender coordinate enters this function: p0/v0/q/qdot/n_active
    must all be inference-legal for the legal bank (enforced upstream).
    """
    import torch

    if init not in CONTROLLED_INITS:
        raise ValueError(f"unknown controlled init {init!r}")
    torch.manual_seed(seed)
    dev = torch.device(device)
    body = make_body_params()

    p0_t = torch.as_tensor(np.asarray(p0, dtype=np.float32), device=dev)
    v0_t = torch.as_tensor(np.asarray(v0, dtype=np.float32), device=dev)
    q_t = torch.as_tensor(np.asarray(q, dtype=np.float32), device=dev)
    qd_t = torch.as_tensor(np.asarray(qdot, dtype=np.float32), device=dev)
    n_act = torch.as_tensor(np.asarray(n_active, dtype=np.int64), device=dev)

    n_batch = p0_t.shape[0]
    n_max = q_t.shape[1] - 1
    assert qd_t.shape == q_t.shape
    assert int(n_act.max()) <= n_max and int(n_act.min()) >= 1

    ks = torch.arange(1, n_max + 1, device=dev)[None, :]          # [1, N_max]
    active = (ks <= n_act[:, None]).float()                        # [B, N_max]
    # ramped step weights w_k = 0.5 + 0.5 k / N_b on active steps
    w = (0.5 + 0.5 * ks.float() / n_act[:, None].float()) * active
    w_sum = w.sum(dim=1).clamp(min=1e-9)
    # control-difference mask: diffs (u_k - u_{k-1}) for k = 1 .. N_b-1
    diff_mask = (ks[:, : n_max - 1] <= (n_act - 1)[:, None]).float()
    diff_count = diff_mask.sum(dim=1).clamp(min=1.0)
    h_b = n_act.float() * dt
    emax = emax_rate * h_b                                         # [B]

    # terminal target velocity: qdot at the terminal active index, clipped
    idx = n_act[:, None, None].expand(-1, 1, 2)
    qdot_term = qd_t.gather(1, idx)[:, 0, :]                       # [B, 2]
    qdot_norm = qdot_term.norm(dim=1, keepdim=True).clamp(min=1e-9)
    qdot_term = qdot_term * (qdot_norm.clamp(max=QDOT_CLIP) / qdot_norm)
    if terminal_mode == "zero":
        qdot_term = torch.zeros_like(qdot_term)

    fr0, lam0 = CONTROLLED_INITS[init]
    raw = torch.full((n_batch, n_max, 2), 0.0, device=dev)
    raw[:, :, 0] = _logit(fr0)
    raw[:, :, 1] = _logit(lam0)
    raw.requires_grad_(True)

    rec_pts = set(_record_iters(adam_iters))
    hist: dict[int, np.ndarray] = {}
    grad_rms = None

    def components(states, vels, z):
        """Per-row objective components.  states/vels: [B, N_max+1, 2]."""
        d2 = ((states[:, 1:] - q_t[:, 1:]) ** 2).sum(dim=2)        # [B, N_max]
        track = (w * d2).sum(dim=1) / (w_sum * TRACK_NORM)
        v_term = vels.gather(1, idx)[:, 0, :]
        if terminal_mode == "free":
            term_vel = torch.zeros_like(track)
        else:
            term_vel = TERMVEL_W * ((v_term - qdot_term) ** 2).sum(dim=1) / TERMVEL_NORM
        energy = ENERGY_W * z / emax
        f_seq = torch.sigmoid(raw[:, :, 0])
        l_seq = torch.sigmoid(raw[:, :, 1])
        df = (f_seq[:, 1:] - f_seq[:, :-1]) * diff_mask
        dl = (l_seq[:, 1:] - l_seq[:, :-1]) * diff_mask
        smooth = SMOOTH_W * (df * df + dl * dl).sum(dim=1) / diff_count
        if energy_mode == "ceiling":
            ceil = CEIL_W * torch.relu(z / emax - 1.0) ** 2
        else:
            ceil = torch.zeros_like(track)
        total = track + term_vel + energy + smooth + ceil
        return {"track": track, "term_vel": term_vel, "energy": energy,
                "smooth": smooth, "ceiling": ceil, "total": total}

    def rollout(record_full: bool):
        f_ctrl = torch.sigmoid(raw[:, :, 0]) * f_max
        lam = torch.sigmoid(raw[:, :, 1])
        px, py = p0_t[:, 0], p0_t[:, 1]
        vx, vy = v0_t[:, 0], v0_t[:, 1]
        z = torch.zeros(n_batch, device=dev)
        states = [torch.stack([px, py], dim=1)]
        vels = [torch.stack([vx, vy], dim=1)]
        extras = {"accel": [], "power": []} if record_full else None
        for k in range(n_max):
            alive = active[:, k]
            ax, ay, p_val = _effort_dynamics_step(
                px, py, vx, vy, f_ctrl[:, k], lam[:, k],
                q_t[:, k + 1, 0], q_t[:, k + 1, 1], body,
            )
            vx = vx + alive * dt * ax
            vy = vy + alive * dt * ay
            px = px + alive * dt * vx
            py = py + alive * dt * vy
            z = z + alive * dt * p_val
            states.append(torch.stack([px, py], dim=1))
            vels.append(torch.stack([vx, vy], dim=1))
            if record_full:
                extras["accel"].append(torch.stack([alive * ax, alive * ay], dim=1))
                extras["power"].append(alive * p_val)
        states = torch.stack(states, dim=1)
        vels = torch.stack(vels, dim=1)
        return states, vels, z, f_ctrl, lam, extras

    opt = torch.optim.Adam([raw], lr=lr)
    for it in range(1, adam_iters + 1):
        opt.zero_grad()
        states, vels, z, _, _, _ = rollout(record_full=False)
        comp = components(states, vels, z)
        loss = comp["total"].mean()
        loss.backward()
        if it in rec_pts:
            hist[it] = comp["total"].detach().cpu().numpy().copy()
        if it == adam_iters:
            # batch-size-independent per-row gradient RMS on raw controls:
            # multiply the 1/B batch-mean factor back out, RMS over the
            # active control entries of each row
            g = raw.grad.detach() * n_batch
            g2 = (g ** 2).sum(dim=2)                               # [B, N_max]
            act_ctrl = (torch.arange(n_max, device=dev)[None, :]
                        < n_act[:, None]).float()
            grad_rms = torch.sqrt(
                (g2 * act_ctrl).sum(dim=1)
                / (2.0 * act_ctrl.sum(dim=1).clamp(min=1.0)))
        opt.step()

    with torch.no_grad():
        states, vels, z, f_ctrl, lam, extras = rollout(record_full=True)
        comp = components(states, vels, z)
        accel = torch.stack(extras["accel"], dim=1)                # [B, N_max, 2]
        power = torch.stack(extras["power"], dim=1)                # [B, N_max]
        p_term = states.gather(1, idx)[:, 0, :]
        v_term = vels.gather(1, idx)[:, 0, :]
        q_term = q_t.gather(1, idx)[:, 0, :]
        act_ctrl = (torch.arange(n_max, device=dev)[None, :] < n_act[:, None]).float()
        nc = act_ctrl.sum(dim=1).clamp(min=1.0)
        f_frac = f_ctrl / f_max
        sat_f = (((f_frac > 0.95) | (f_frac < 0.05)).float() * act_ctrl).sum(dim=1) / nc
        sat_l = (((lam > 0.95) | (lam < 0.05)).float() * act_ctrl).sum(dim=1) / nc
        finite = (
            torch.isfinite(states).all(dim=(1, 2))
            & torch.isfinite(vels).all(dim=(1, 2))
            & torch.isfinite(comp["total"])
            & torch.isfinite(z)
        )

    rec = sorted(hist)
    j_hist = np.stack([hist[i] for i in rec], axis=1) if rec else np.zeros((n_batch, 0))
    # relative improvement over the final 25 iterations
    if len(rec) >= 2:
        j_prev, j_last = hist[rec[-2]], hist[rec[-1]]
        rel_impr = (j_prev - j_last) / np.maximum(np.abs(j_prev), 1e-9)
    else:
        rel_impr = np.zeros(n_batch)

    z_np = z.cpu().numpy()
    emax_np = emax.cpu().numpy()
    return {
        "path": states.cpu().numpy(),
        "vel": vels.cpu().numpy(),
        "accel": accel.cpu().numpy(),
        "power": power.cpu().numpy(),
        "f_ctrl": f_ctrl.cpu().numpy(),
        "lam": lam.cpu().numpy(),
        "z_final": z_np,
        "emax": emax_np,
        "e_viol_abs": z_np - emax_np,
        "e_viol_rel": (z_np - emax_np) / np.maximum(emax_np, 1e-9),
        "p_term": p_term.cpu().numpy(),
        "v_term": v_term.cpu().numpy(),
        "q_term": q_term.cpu().numpy(),
        "qdot_term": qdot_term.cpu().numpy(),
        "loss_track": comp["track"].cpu().numpy(),
        "loss_term_vel": comp["term_vel"].cpu().numpy(),
        "loss_energy": comp["energy"].cpu().numpy(),
        "loss_smooth": comp["smooth"].cpu().numpy(),
        "loss_ceiling": comp["ceiling"].cpu().numpy(),
        "loss_total": comp["total"].cpu().numpy(),
        "grad_rms": grad_rms.cpu().numpy(),
        "sat_f_frac": sat_f.cpu().numpy(),
        "sat_lam_frac": sat_l.cpu().numpy(),
        "finite": finite.cpu().numpy(),
        "rel_impr_final25": rel_impr,
        "hist_iters": np.asarray(rec, dtype=np.int64),
        "hist_total": j_hist,
        "n_active": np.asarray(n_active, dtype=np.int64),
        "config": {
            "terminal_mode": terminal_mode, "energy_mode": energy_mode,
            "f_max": f_max, "dt": dt, "emax_rate": emax_rate,
            "adam_iters": adam_iters, "lr": lr, "init": init, "seed": seed,
        },
    }


RETRY_REL_IMPR = 1e-3


RETRY_E_VIOL_REL = 0.02


def retry_flags(out: dict, rel_impr_thresh: float | None = None,
                e_viol_thresh: float | None = None) -> np.ndarray:
    """Provisional selective-retry triggers: non-finite, still materially
    improving at the end, or relative energy-ceiling violation.  A high
    tracking loss alone is NOT failure.  Thresholds read the module
    attributes at call time so the C0 pinning step can adjust them."""
    if rel_impr_thresh is None:
        rel_impr_thresh = RETRY_REL_IMPR
    if e_viol_thresh is None:
        e_viol_thresh = RETRY_E_VIOL_REL
    bad = ~np.asarray(out["finite"], dtype=bool)
    bad |= out["rel_impr_final25"] > rel_impr_thresh
    if out["config"]["energy_mode"] == "ceiling":
        bad |= out["e_viol_rel"] > e_viol_thresh
    return bad
