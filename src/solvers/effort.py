"""Batched reach solvers: the minimum time to reach a point from a moving start (closed form), and the minimum-energy reach
under force and power limits (Adam over the control sequence). Verbatim from the research code (oc_batched_solvers)."""
from __future__ import annotations

import math

import numpy as np

def _second_difference_inverse(n_grid: int) -> np.ndarray:
    """inv(M) for the (N-1)x(N-1) second-difference matrix used by abcSfun."""
    m = (
        -2.0 * np.eye(n_grid - 1)
        + np.eye(n_grid - 1, k=1)
        + np.eye(n_grid - 1, k=-1)
    )
    return np.linalg.inv(m)


def _abcs_fun_batch(flc: np.ndarray, l_inv: np.ndarray, n_grid: int):
    """Batched abcSfun: canonical (0,0)->(1,0) path under a(t)=(At+B)/|At+B|.

    flc: [B, 4] = (a, b, c, S), d fixed at 1.  Returns boundary velocity
    components (alph1, bet1, alph2, bet2) each [B] and paths X, Y [B, N+1].
    """
    a = flc[:, 0:1]
    b = flc[:, 1:2]
    c = flc[:, 2:3]
    d = 1.0
    s = np.maximum(flc[:, 3:4], 1e-6)

    tau = np.linspace(0.0, 1.0, n_grid + 1)[None, :]  # [1, N+1]
    t = s * tau                                       # [B, N+1]
    denom = np.sqrt((a * t + b) ** 2 + (c * t + d) ** 2)
    denom = np.maximum(denom, 1e-12)
    h2 = (s / n_grid) ** 2
    f = h2 * (c * t + d) / denom                      # x-accel RHS
    g = h2 * (a * t + b) / denom                      # y-accel RHS
    # interior nodes are indices 1..N-1; boundary X(N)=1 moves into the RHS
    f_int = f[:, 1:n_grid].copy()
    f_int[:, -1] -= 1.0
    g_int = g[:, 1:n_grid]

    x = np.zeros((flc.shape[0], n_grid + 1))
    y = np.zeros_like(x)
    x[:, n_grid] = 1.0
    x[:, 1:n_grid] = f_int @ l_inv.T
    y[:, 1:n_grid] = g_int @ l_inv.T

    scale = n_grid / s[:, 0]
    # second-order boundary velocities: correct the one-sided difference with
    # the analytically known boundary acceleration (the MATLAB original is
    # first-order here, which biases matched boundary velocities by O(h))
    h = s[:, 0] / n_grid
    ax0 = (c[:, 0] * t[:, 0] + d) / denom[:, 0]
    ay0 = (a[:, 0] * t[:, 0] + b[:, 0]) / denom[:, 0]
    axf = (c[:, 0] * t[:, -1] + d) / denom[:, -1]
    ayf = (a[:, 0] * t[:, -1] + b[:, 0]) / denom[:, -1]
    alph1 = (x[:, 1] - x[:, 0]) * scale - 0.5 * h * ax0
    alph2 = (x[:, n_grid] - x[:, n_grid - 1]) * scale + 0.5 * h * axf
    bet1 = (y[:, 1] - y[:, 0]) * scale - 0.5 * h * ay0
    bet2 = (y[:, n_grid] - y[:, n_grid - 1]) * scale + 0.5 * h * ayf
    return alph1, bet1, alph2, bet2, x, y


def _residual_batch(flc, target, l_inv, n_grid):
    a1, b1, a2, b2 = _abcs_fun_batch(flc, l_inv, n_grid)[:4]
    achieved = np.stack([a1, b1, a2, b2], axis=1)
    return target - achieved  # [B, 4]


def _match_coords_batch(flc0, target, l_inv, n_grid, iters=30, fd_step=0.1):
    """Batched damped Gauss-Newton (LM) with per-problem backtracking."""
    flc = flc0.copy()
    n_batch = flc.shape[0]
    lam = np.full(n_batch, 1e-2)
    r = _residual_batch(flc, target, l_inv, n_grid)
    for _ in range(iters):
        err = np.linalg.norm(r, axis=1)
        active = err > 1e-9
        if not active.any():
            break
        # finite-difference Jacobian, batched: 8 residual evaluations
        jac = np.zeros((n_batch, 4, 4))
        for k in range(4):
            dp = np.zeros(4)
            dp[k] = fd_step
            r_plus = _residual_batch(flc + dp, target, l_inv, n_grid)
            r_minus = _residual_batch(flc - dp, target, l_inv, n_grid)
            # residual = target - achieved, so d(achieved)/dp = -(dr/dp)
            jac[:, :, k] = -(r_plus - r_minus) / (2.0 * fd_step)

        jtj = np.einsum("bik,bil->bkl", jac, jac)
        jtr = np.einsum("bik,bi->bk", jac, r)
        eye = np.eye(4)[None, :, :]
        # numpy>=2 requires stacked vector RHS to carry an explicit trailing
        # dim ((m,m),(m,n)->(m,n) gufunc); [..., None] + squeeze works on 1.x
        # and 2.x identically
        step = np.linalg.solve(
            jtj + lam[:, None, None] * eye, jtr[..., None])[..., 0]

        # backtracking line search, batched via masks
        accepted = np.zeros(n_batch, dtype=bool)
        t_ls = np.ones(n_batch)
        for _ in range(7):  # t down to 1/64 like the original
            pending = active & ~accepted
            if not pending.any():
                break
            trial = flc.copy()
            trial[pending] = flc[pending] + t_ls[pending, None] * step[pending]
            trial[:, 3] = np.maximum(trial[:, 3], 1e-6)
            r_trial = _residual_batch(trial, target, l_inv, n_grid)
            better = np.linalg.norm(r_trial, axis=1) < np.linalg.norm(r, axis=1)
            take = pending & better
            flc[take] = trial[take]
            r[take] = r_trial[take]
            accepted |= take
            t_ls[pending & ~better] *= 0.5
        lam = np.where(accepted, np.maximum(lam / 2.0, 1e-8), np.minimum(lam * 4.0, 1e3))
    return flc, r


DEFAULT_MULTISTART = np.array(
    [
        [0.5, 0.0, 0.5, 1.2],   # original MATLAB init
        [0.0, 0.0, -1.5, 2.0],  # single-switch (bang-bang-like) profiles
        [0.0, 0.0, 1.5, 0.8],
        [-0.5, 0.5, 0.0, 1.5],
    ]
)


def solve_min_time_batch(
    xi: np.ndarray,
    vi: np.ndarray,
    xf: np.ndarray,
    vf: np.ndarray,
    amax: float | np.ndarray = 7.0,
    n_grid: int = 100,
    iters: int = 30,
    inits: np.ndarray = DEFAULT_MULTISTART,
):
    """Batched min-time solve.  All of xi, vi, xf, vf are [B, 2] world units.

    amax is the acceleration bound in world units (scalar or [B]).  Returns a
    dict with world paths ``x, y`` [B, N+1], segment times ``t_seg`` [B],
    residual norms ``resid`` [B], and canonical parameters ``flc`` [B, 4].
    """
    xi = np.asarray(xi, dtype=float)
    vi = np.asarray(vi, dtype=float)
    xf = np.asarray(xf, dtype=float)
    vf = np.asarray(vf, dtype=float)
    n_batch = xi.shape[0]
    amax_arr = np.broadcast_to(np.asarray(amax, dtype=float), (n_batch,)).copy()

    e = xf - xi
    lseg = np.linalg.norm(e, axis=1)
    degenerate = lseg < 1e-9
    lseg_safe = np.where(degenerate, 1.0, lseg)
    ex = e / lseg_safe[:, None]
    # rows of R are ex and ey (CCW), matching the MATLAB frame
    rot = np.stack([ex, np.stack([-ex[:, 1], ex[:, 0]], axis=1)], axis=1)  # [B,2,2]

    # nondimensionalization: space by Lseg, time by tau = sqrt(Lseg/amax)
    tau = np.sqrt(lseg_safe / amax_arr)
    vel_scale = tau / lseg_safe  # world velocity -> canonical velocity
    vi_f = np.einsum("bij,bj->bi", rot, vi) * vel_scale[:, None]
    vf_f = np.einsum("bij,bj->bi", rot, vf) * vel_scale[:, None]
    target = np.concatenate([vi_f, vf_f], axis=1)  # [B, 4]

    l_inv = _second_difference_inverse(n_grid)

    best_flc = None
    best_resid = np.full(n_batch, np.inf)
    for init in inits:
        flc0 = np.tile(init[None, :], (n_batch, 1))
        flc, r = _match_coords_batch(flc0, target, l_inv, n_grid, iters=iters)
        resid = np.linalg.norm(r, axis=1)
        # among converged roots prefer the smallest time S (min-time problem
        # can admit multiple stationary boundary matches)
        if best_flc is None:
            best_flc, best_resid = flc.copy(), resid
        else:
            better = (resid < best_resid - 1e-9) | (
                (resid < 1e-6) & (best_resid < 1e-6) & (flc[:, 3] < best_flc[:, 3])
            )
            best_flc[better] = flc[better]
            best_resid[better] = resid[better]

    _, _, _, _, xc, yc = _abcs_fun_batch(best_flc, l_inv, n_grid)
    path_canon = np.stack([xc, yc], axis=2)  # [B, N+1, 2]
    path_world = xi[:, None, :] + lseg_safe[:, None, None] * np.einsum(
        "bnk,bkj->bnj", path_canon, rot
    )
    t_seg = best_flc[:, 3] * tau
    path_world[degenerate] = xi[degenerate][:, None, :]
    t_seg[degenerate] = 0.0
    return {
        "x": path_world[:, :, 0],
        "y": path_world[:, :, 1],
        "t_seg": t_seg,
        "resid": best_resid,
        "flc": best_flc,
        "degenerate": degenerate,
    }


def make_body_params(height_m: float = 1.8, weight_kg: float = 80.0, gamma: float = 0.2):
    """Drag and body constants exactly as in energyConstrainedPath.m."""
    rho, cd = 1.225, 0.9
    af = 0.0537985 * height_m**0.725 * weight_kg**0.425
    return {
        "m": weight_kg,
        "k": 0.5 * rho * af * cd,
        "gamma": gamma,
        "v_eps": 0.10,
        "eps_abs": 1e-2,
    }


def _power_radius(f_ctrl, ct, v2, m, k, gamma, eps_abs):
    """Solve P(r) = F for accel magnitude r along a direction with cosine ct
    to the velocity: P = sqrt((m*ct*r + k*v2)^2 + eps^2) + gamma*m*(1-ct^2)*r^2.

    Closed-form root of the unsmoothed model, then two Newton polish steps on
    the smoothed model.  Fully batched and differentiable (torch tensors).
    """
    import torch

    st2 = torch.clamp(1.0 - ct * ct, min=0.0)
    q = gamma * m * st2                      # quadratic coefficient
    lin = m * ct                             # linear coefficient inside |.|
    drag = k * v2

    # Branch on the sign of ct (which |tan_force| regime is active), never on
    # the size of q: the naive quadratic formula divides by 2q and explodes
    # for near-tangential directions.  The stable form 2c/(b + sqrt(b^2+4qc))
    # reduces smoothly to the linear root c/b as q -> 0.
    f_eff = torch.clamp(f_ctrl, min=0.0)
    # ct >= 0 (driving): q r^2 + lin r - (F - drag) = 0, lin >= 0
    c_pos = torch.clamp(f_eff - drag, min=0.0)
    lin_pos = torch.clamp(lin, min=0.0)
    # the 1e-12 inside each sqrt keeps its gradient finite when the argument
    # is exactly zero (e.g. the dead branch at ct = +/-1): sqrt'(0) = inf and
    # the where-mask would turn that into 0 * inf = NaN
    r_pos = 2.0 * c_pos / (
        lin_pos + torch.sqrt(lin_pos * lin_pos + 4.0 * q * c_pos + 1e-12) + 1e-9
    )
    # ct < 0 (braking, full-effort root): q r^2 + |lin| r - (F + drag) = 0
    c_neg = f_eff + drag
    lin_neg = torch.clamp(-lin, min=0.0)
    r_neg = 2.0 * c_neg / (
        lin_neg + torch.sqrt(lin_neg * lin_neg + 4.0 * q * c_neg + 1e-12) + 1e-9
    )
    r0 = torch.where(ct >= 0.0, r_pos, r_neg)
    r0 = torch.clamp(r0, min=0.0)

    # Newton polish on smoothed P
    r = r0
    for _ in range(2):
        tan_force = lin * r + drag
        p_tan = torch.sqrt(tan_force * tan_force + eps_abs * eps_abs)
        p = p_tan + q * r * r
        dp = (tan_force / p_tan) * lin + 2.0 * q * r
        # keep step finite; only move where derivative is well-signed
        step = (p - f_eff) / torch.where(dp.abs() > 1e-6, dp, torch.full_like(dp, 1e-6))
        r = torch.clamp(r - step, min=0.0)
    return r


def _effort_dynamics_step(px, py, vx, vy, f_ctrl, lam, tx, ty, body):
    """One dynamics evaluation (accelerations + power), batched torch."""
    import torch

    m, k, gamma = body["m"], body["k"], body["gamma"]
    v_eps, eps_abs = body["v_eps"], body["eps_abs"]

    v2 = vx * vx + vy * vy
    v_norm = torch.sqrt(v2 + v_eps * v_eps)
    gx, gy = tx - px, ty - py
    g_norm = torch.sqrt(gx * gx + gy * gy + 1e-12)
    ghx, ghy = gx / g_norm, gy / g_norm
    vhx, vhy = vx / v_norm, vy / v_norm
    # tiny speed: head toward the goal.  NOTE: test raw v2, not v_norm —
    # v_norm is floored at v_eps so a threshold on it can never fire (latent
    # bug in the MATLAB original: stationary starts produce a zero vHat and
    # a degenerate all-zero direction basis)
    tiny = v2 < 1e-8
    vhx = torch.where(tiny, ghx, vhx)
    vhy = torch.where(tiny, ghy, vhy)
    nx_, ny_ = -vhy, vhx
    c_v = ghx * vhx + ghy * vhy
    c_n = ghx * nx_ + ghy * ny_

    # momentum / brake direction
    sgn_v = torch.where(c_v >= 0.0, torch.ones_like(c_v), -torch.ones_like(c_v))
    ddx, ddy = sgn_v * vhx, sgn_v * vhy

    # max-turn direction: cancel drag tangentially, all remaining budget lateral
    sgn_n = torch.where(c_n >= 0.0, torch.ones_like(c_n), -torch.ones_like(c_n))
    a_tan_cancel = -(k / m) * v2
    a_norm_max = torch.sqrt(torch.clamp(f_ctrl, min=0.0) / (gamma * m) + 1e-8)
    avx = a_tan_cancel * vhx + sgn_n * a_norm_max * nx_
    avy = a_tan_cancel * vhy + sgn_n * a_norm_max * ny_
    av_norm = torch.sqrt(avx * avx + avy * avy + 1e-12)
    aligned = c_n.abs() < 1e-8
    dax = torch.where(aligned, ddx, avx / av_norm)
    day = torch.where(aligned, ddy, avy / av_norm)

    drx = (1.0 - lam) * ddx + lam * dax
    dry = (1.0 - lam) * ddy + lam * day
    # clamp keeps the normalization gradient bounded when the two candidate
    # directions nearly cancel
    dr_norm = torch.clamp(torch.sqrt(drx * drx + dry * dry + 1e-12), min=1e-3)
    adx, ady = drx / dr_norm, dry / dr_norm

    ct = adx * vhx + ady * vhy
    r = _power_radius(f_ctrl, ct, v2, m, k, gamma, eps_abs)
    ax, ay = r * adx, r * ady

    # realized power for the energy state
    a_tan = ax * vhx + ay * vhy
    anx, any_ = ax - a_tan * vhx, ay - a_tan * vhy
    tan_force = m * a_tan + k * v2
    p_val = torch.sqrt(tan_force * tan_force + eps_abs * eps_abs) + gamma * m * (
        anx * anx + any_ * any_
    )
    return ax, ay, p_val


def solve_effort_constrained_batch(
    x0: np.ndarray,
    v0: np.ndarray,
    target: np.ndarray,
    t_final: np.ndarray,
    e_max: np.ndarray,
    f_max: float = 200.0,
    dt: float = 0.1,
    height_m: float = 1.8,
    weight_kg: float = 80.0,
    gamma: float = 0.2,
    energy_margin: float = 0.10,
    adam_iters: int = 300,
    lr: float = 0.08,
    device: str = "cpu",
    penalty: float = 5.0,
    seed: int = 0,
    objective: str = "budget",
    energy_weight: float = 0.01,
):
    """Batched effort-constrained target reach.  x0, v0, target: [B, 2];
    t_final, e_max: [B].  Horizons may differ per problem; the roll-out uses
    the max horizon with a per-problem active mask (frozen after t_final).

    objective:
      - "budget": original energyConstrainedPath.m problem — minimize terminal
        distance subject to spending z_T in [Emin, Emax] (window penalty).
      - "min_energy_reach": minimize terminal distance plus
        ``energy_weight * z_T`` with no energy window — the minimum-effort
        physically feasible path that reaches the target at t_final.

    Returns dict with paths [B, T+1, 2] at dt resolution, energies z_T [B],
    terminal distances [B], and the control sequences.
    """
    import torch

    torch.manual_seed(seed)
    dev = torch.device(device)
    body = make_body_params(height_m, weight_kg, gamma)

    x0_t = torch.as_tensor(np.asarray(x0, dtype=np.float32), device=dev)
    v0_t = torch.as_tensor(np.asarray(v0, dtype=np.float32), device=dev)
    tgt = torch.as_tensor(np.asarray(target, dtype=np.float32), device=dev)
    tf = torch.as_tensor(np.asarray(t_final, dtype=np.float32), device=dev)
    emax = torch.as_tensor(np.asarray(e_max, dtype=np.float32), device=dev)
    emin = (1.0 - energy_margin) * emax

    n_batch = x0_t.shape[0]
    n_steps = int(math.ceil(float(tf.max().item()) / dt))
    step_alive = (
        torch.arange(n_steps, device=dev)[None, :] < (tf / dt - 1e-6)[:, None]
    ).float()  # [B, T]

    # raw controls -> F in (0, Fmax), lambda in (0, 1)
    raw = torch.zeros(n_batch, n_steps, 2, device=dev, requires_grad=True)
    with torch.no_grad():
        # init near F = Emax / Tfinal (the MATLAB initial guess), lambda = 0.5
        f_init = torch.clamp(emax / torch.clamp(tf, min=dt) / f_max, 1e-3, 1 - 1e-3)
        raw[:, :, 0] = torch.log(f_init / (1 - f_init))[:, None]

    opt = torch.optim.Adam([raw], lr=lr)
    for _ in range(adam_iters):
        opt.zero_grad()
        f_ctrl = torch.sigmoid(raw[:, :, 0]) * f_max
        lam = torch.sigmoid(raw[:, :, 1])
        px, py = x0_t[:, 0], x0_t[:, 1]
        vx, vy = v0_t[:, 0], v0_t[:, 1]
        z = torch.zeros(n_batch, device=dev)
        for t_idx in range(n_steps):
            alive = step_alive[:, t_idx]
            ax, ay, p_val = _effort_dynamics_step(
                px, py, vx, vy, f_ctrl[:, t_idx], lam[:, t_idx],
                tgt[:, 0], tgt[:, 1], body,
            )
            vx = vx + alive * dt * ax
            vy = vy + alive * dt * ay
            px = px + alive * dt * vx
            py = py + alive * dt * vy
            z = z + alive * dt * p_val
        term = (px - tgt[:, 0]) ** 2 + (py - tgt[:, 1]) ** 2
        if objective == "min_energy_reach":
            loss = (term + energy_weight * z).mean()
        else:
            e_viol = torch.relu(z - emax) ** 2 + torch.relu(emin - z) ** 2
            loss = (term + penalty * e_viol).mean()
        loss.backward()
        opt.step()

    # final roll-out recording the full paths
    with torch.no_grad():
        f_ctrl = torch.sigmoid(raw[:, :, 0]) * f_max
        lam = torch.sigmoid(raw[:, :, 1])
        px, py = x0_t[:, 0].clone(), x0_t[:, 1].clone()
        vx, vy = v0_t[:, 0].clone(), v0_t[:, 1].clone()
        z = torch.zeros(n_batch, device=dev)
        path = torch.zeros(n_batch, n_steps + 1, 2, device=dev)
        path[:, 0, 0], path[:, 0, 1] = px, py
        for t_idx in range(n_steps):
            alive = step_alive[:, t_idx]
            ax, ay, p_val = _effort_dynamics_step(
                px, py, vx, vy, f_ctrl[:, t_idx], lam[:, t_idx],
                tgt[:, 0], tgt[:, 1], body,
            )
            vx = vx + alive * dt * ax
            vy = vy + alive * dt * ay
            px = px + alive * dt * vx
            py = py + alive * dt * vy
            z = z + alive * dt * p_val
            path[:, t_idx + 1, 0], path[:, t_idx + 1, 1] = px, py
        term_dist = torch.sqrt(
            (px - tgt[:, 0]) ** 2 + (py - tgt[:, 1]) ** 2
        )
    return {
        "path": path.cpu().numpy(),
        "z_final": z.cpu().numpy(),
        "terminal_dist": term_dist.cpu().numpy(),
        "f_ctrl": f_ctrl.detach().cpu().numpy(),
        "lam": lam.detach().cpu().numpy(),
        "e_max": e_max,
    }
