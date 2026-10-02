"""The rule-option bank: for every (defender, option) the physics solvers track the option's target path over the next second.

  c4       continuation tracking of the target path (best of two controlled starts, each retried at 1,000 iterations on the pinned
           triggers, plus a third start where the two disagree by more than 0.1 yd) -> control summaries and a ten-step preview
  c1       the minimum-energy reach of the path's end point (400 iterations)
  priv_c4  c4 from the defender's real first-step velocity (training plays only; the rule teacher)
The solves run in chunks of 16,384 rows in the given row order; results depend slightly on which rows share a chunk.
Function bodies are verbatim from the research code (w42_bank_solver, w42_option_tokens); only the file I/O is replaced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.solvers import tracking as TS
from src.solvers.effort import solve_effort_constrained_batch
from src.solvers.tracking import retry_flags, solve_tracking_batch

TS.RETRY_E_VIOL_REL = 0.08    # the pinned production retry trigger (the module default 0.02 is provisional)
DT = 0.1
N_MAX = 10
CHUNK = 16384
RESCUE_DISAGREE = 0.10        # yd, A/B terminal disagreement triggering start C
RETRY_ITERS = 1000
AMAX = 7.0
HORIZONS = [(3, "0p3", 0.3), (5, "0p5", 0.5), (10, "1p0", 1.0)]
N_STEPS = 11
N_PREV_CH = 11
C4_COPY_COLS = [
    "ax0", "ay0", "f0", "lam0", "power0", "energy_total", "energy_early",
    "e_viol_rel", "path_len", "excess_len", "heading_change_total",
    "term_miss", "term_vel_miss", "sat_f_frac", "sat_lam_frac",
    "dp_0p2_x", "dp_0p2_y", "dp_0p5_x", "dp_0p5_y", "dp_1p0_x", "dp_1p0_y",
    "pers_res_0p2", "pers_res_0p5", "pers_res_1p0",
    "finite", "retried", "ab_disagree", "win_start", "rel_impr_final25",
]
C1_COPY_COLS = ["dp_0p2_x", "dp_0p2_y", "dp_0p5_x", "dp_0p5_y",
                "dp_1p0_x", "dp_1p0_y", "term_miss", "energy_total",
                "ax0", "ay0"]
ID_COLS = ("row_id", "tag", "game_play_id", "defender_index", "nfl_id", "option_kind", "option_index", "option_available",
           "v0_valid", "n_active")


def solve_start(p0, v0, q, qd, na, init: str) -> dict:
    """One controlled start with the pinned retry (1,000 iters on flags)."""
    out = solve_tracking_batch(p0, v0, q, qd, na, init=init)
    flags = retry_flags(out)
    if flags.any():
        sub = np.where(flags)[0]
        out2 = solve_tracking_batch(p0[sub], v0[sub], q[sub], qd[sub],
                                    na[sub], init=init,
                                    adam_iters=RETRY_ITERS)
        better = out2["loss_total"] < out["loss_total"][sub]
        take, ts = sub[better], np.where(better)[0]
        for key, val in out.items():
            if key in {"config", "hist_iters", "hist_total"}:
                continue
            if isinstance(val, np.ndarray) and val.shape[:1] == flags.shape:
                val[take] = out2[key][ts]
    out["retried"] = flags
    return out


def merge_best(runs: dict[str, dict]) -> tuple[dict, np.ndarray]:
    """Per-row best-objective merge across controlled starts."""
    names = list(runs)
    losses = np.stack([runs[n]["loss_total"] for n in names])
    win = np.argmin(np.where(np.isfinite(losses), losses, np.inf), axis=0)
    base = runs[names[0]]
    merged = {}
    for key, val in base.items():
        if key in {"config", "hist_iters", "hist_total", "retried"}:
            continue
        if isinstance(val, np.ndarray) and val.shape[:1] == win.shape:
            stack = np.stack([runs[n][key] for n in names])
            merged[key] = stack[win, np.arange(len(win))]
    merged["retried"] = np.stack(
        [runs[n]["retried"] for n in names]).any(axis=0)
    return merged, win


def solve_c4_chunk(p0, v0, q, qd, na) -> tuple[dict, dict]:
    a = solve_start(p0, v0, q, qd, na, "A")
    b = solve_start(p0, v0, q, qd, na, "B")
    dis = np.linalg.norm(a["p_term"] - b["p_term"], axis=1)
    runs = {"A": a, "B": b}
    n_res = 0
    if (dis > RESCUE_DISAGREE).any():
        sub = np.where(dis > RESCUE_DISAGREE)[0]
        n_res = len(sub)
        c = solve_start(p0[sub], v0[sub], q[sub], qd[sub], na[sub], "C")
        cfull = {}
        for key, val in a.items():
            if key in {"config", "hist_iters", "hist_total"}:
                continue
            if isinstance(val, np.ndarray) and val.shape[:1] == dis.shape:
                full = val.copy()
                if key == "loss_total":
                    full[:] = np.inf     # C competes only where solved
                full[sub] = c[key][np.arange(len(sub))]
                cfull[key] = full
        cfull["retried"] = np.zeros(len(dis), bool)
        cfull["retried"][sub] = c["retried"]
        runs["C"] = cfull
    merged, win = merge_best(runs)
    merged["ab_disagree"] = dis
    merged["win_start"] = win
    stats = {"n_rescued": int(n_res),
             "n_retried": int(merged["retried"].sum())}
    return merged, stats


def summarize(rows: pd.DataFrame, out: dict, p0, v0) -> pd.DataFrame:
    path, acc, power = out["path"], out["accel"], out["power"]
    na = out["n_active"]
    r = np.arange(len(path))
    disp = path - p0[:, None, :]
    act = (np.arange(N_MAX)[None, :] < na[:, None])
    seg = np.linalg.norm(np.diff(path, axis=1), axis=2)
    plen = (seg * act).sum(axis=1)
    chord = np.linalg.norm(path[r, na] - p0, axis=1)
    heads = np.arctan2(acc[:, :, 1], acc[:, :, 0])
    dhead = np.abs((np.diff(heads, axis=1) + np.pi) % (2 * np.pi) - np.pi) * act[:, 1:]
    df = pd.DataFrame({
        "row_id": rows["row_id"].to_numpy(),
        "ax0": acc[:, 0, 0], "ay0": acc[:, 0, 1],
        "a0_heading": heads[:, 0],
        "f0": out["f_ctrl"][:, 0], "lam0": out["lam"][:, 0],
        "power0": power[:, 0],
        "energy_total": out["z_final"],
        "energy_early": (power[:, :3] * act[:, :3]).sum(axis=1) * DT,
        "e_viol_rel": out["e_viol_rel"],
        "path_len": plen, "excess_len": plen - chord,
        "heading_change_total": dhead.sum(axis=1),
        "term_x": out["p_term"][:, 0], "term_y": out["p_term"][:, 1],
        "term_vx": out["v_term"][:, 0], "term_vy": out["v_term"][:, 1],
        "term_miss": np.linalg.norm(out["p_term"] - out["q_term"], axis=1),
        "term_vel_miss": np.linalg.norm(out["v_term"] - out["qdot_term"], axis=1),
        "loss_total": out["loss_total"], "loss_track": out["loss_track"],
        "grad_rms": out["grad_rms"],
        "sat_f_frac": out["sat_f_frac"], "sat_lam_frac": out["sat_lam_frac"],
        "finite": out["finite"].astype(np.int8),
        "rel_impr_final25": out["rel_impr_final25"],
        "retried": out["retried"].astype(np.int8),
        "ab_disagree": out["ab_disagree"],
        "win_start": out["win_start"].astype(np.int8),
        "n_active": na,
    })
    for k, name in [(2, "0p2"), (3, "0p3"), (5, "0p5"), (7, "0p7"), (10, "1p0")]:
        kk = np.minimum(k, na)
        dpx, dpy = disp[r, kk, 0].copy(), disp[r, kk, 1].copy()
        pres = np.linalg.norm(disp[r, kk] - v0 * (kk[:, None] * DT), axis=1)
        dpx[kk < k] = np.nan
        dpy[kk < k] = np.nan
        pres[kk < k] = np.nan
        df[f"dp_{name}_x"] = dpx
        df[f"dp_{name}_y"] = dpy
        df[f"pers_res_{name}"] = pres
    return df.astype({c: np.float32 for c in df.columns
                      if df[c].dtype == np.float64})



def solve_bank(meta, q, qd, formulation, v0_override=None):
    """Solve every available option row of `meta` (columns row_id, p0x, p0y, vx0_legal, vy0_legal, n_active, option_available;
    q / qd [n, 11, 2] target paths and velocities) -> (summary DataFrame by row_id, step tensors). v0_override [n, 2]: the start
    velocity for priv_c4 (rows where it is NaN are skipped)."""
    avail = meta["option_available"].to_numpy() == 1
    if v0_override is not None:
        avail &= np.isfinite(v0_override[:, 0])
    idx = np.where(avail)[0]
    rows = meta.iloc[idx].reset_index(drop=True)
    p0 = meta[["p0x", "p0y"]].to_numpy(np.float32)
    v_legal = meta[["vx0_legal", "vy0_legal"]].to_numpy(np.float32)
    na_all = meta["n_active"].to_numpy(np.int64)
    v0 = np.nan_to_num(v0_override, nan=0.0).astype(np.float32) if v0_override is not None else v_legal
    frames, tens = [], []
    for s in range(0, len(idx), CHUNK):
        e = min(s + CHUNK, len(idx))
        sl = idx[s:e]
        if formulation == "c1":
            tgt = q[sl, na_all[sl]]
            tf = na_all[sl] * DT
            o = solve_effort_constrained_batch(
                p0[sl], v0[sl], tgt, tf, e_max=tf * 150.0, f_max=200.0,
                dt=DT, objective="min_energy_reach", energy_weight=0.01,
                adam_iters=400, lr=0.08, device="cpu")
            path = o["path"]
            disp = path - p0[sl][:, None, :]
            r = np.arange(len(sl))
            sub = pd.DataFrame({
                "row_id": rows["row_id"].to_numpy()[s:e],
                "term_miss": o["terminal_dist"],
                "energy_total": o["z_final"],
                "f0": o["f_ctrl"][:, 0], "lam0": o["lam"][:, 0],
                "ax0": (path[:, 1, 0] - p0[sl, 0] - v0[sl, 0] * DT) / DT**2,
                "ay0": (path[:, 1, 1] - p0[sl, 1] - v0[sl, 1] * DT) / DT**2,
                "finite": np.isfinite(path).all(axis=(1, 2)).astype(np.int8),
                "n_active": na_all[sl],
            })
            for k, name in [(2, "0p2"), (5, "0p5"), (10, "1p0")]:
                kj = np.minimum(k, na_all[sl])
                dpx, dpy = disp[r, kj, 0].copy(), disp[r, kj, 1].copy()
                dpx[kj < k] = np.nan
                dpy[kj < k] = np.nan
                sub[f"dp_{name}_x"] = dpx
                sub[f"dp_{name}_y"] = dpy
            frames.append(sub.astype(
                {c: np.float32 for c in sub.columns
                 if sub[c].dtype == np.float64}))
            tens.append({"path16": (path - p0[sl][:, None, :]).astype(np.float16)})
        else:
            merged, st = solve_c4_chunk(p0[sl], v0[sl], q[sl], qd[sl],
                                        na_all[sl])
            frames.append(summarize(rows.iloc[s:e], merged, p0[sl], v0[sl]))
            tens.append({
                "disp": (merged["path"] - p0[sl][:, None, :]).astype(np.float16),
                "vel": merged["vel"].astype(np.float16),
                "accel": merged["accel"].astype(np.float16),
                "f_frac": (merged["f_ctrl"] / 200.0).astype(np.float16),
                "lam": merged["lam"].astype(np.float16),
                "power_frac": (merged["power"] / 200.0).astype(np.float16),
            })
        print(f"  [{formulation}] {e:,}/{len(idx):,} rows", flush=True)
    df = pd.concat(frames, ignore_index=True)
    return df, {k: np.concatenate([t[k] for t in tens]) for k in tens[0]}


def scatter_cols(meta_n, bank_df, cols, prefix):
    """Return {out_name: full-universe float32 col} scattered by row_id."""
    rid = bank_df["row_id"].to_numpy()
    out = {}
    for c in cols:
        full = np.zeros(meta_n, dtype=np.float32)
        full[rid] = np.nan_to_num(
            bank_df[c].to_numpy(dtype=np.float32), nan=0.0)
        out[f"{prefix}_{c}"] = full
    return out


def deterministic_columns(meta, q, qd):
    """The token columns that are exact functions of the solver inputs: the clipped analytic required acceleration to reach the
    path at 0.3 / 0.5 / 1.0 s and the target geometry (path offsets, start and end velocity)."""
    cols = {}
    n = len(meta)
    p0 = meta[["p0x", "p0y"]].to_numpy()
    v0 = meta[["vx0_legal", "vy0_legal"]].to_numpy()
    na = meta["n_active"].to_numpy()
    r = np.arange(n)
    for k, name, h in HORIZONS:
        kk = np.minimum(k, na)
        qh = q[r, kk]
        areq = 2.0 * (qh - p0 - v0 * h) / (h * h)
        mag = np.linalg.norm(areq, axis=1)
        scale = np.minimum(1.0, AMAX / np.maximum(mag, 1e-9))
        aclip = areq * scale[:, None]
        dp = v0 * h + 0.5 * aclip * h * h
        miss = np.linalg.norm(p0 + dp - qh, axis=1)
        cols[f"an_areq_mag_{name}"] = mag.astype(np.float32)
        cols[f"an_ax_{name}"] = aclip[:, 0].astype(np.float32)
        cols[f"an_ay_{name}"] = aclip[:, 1].astype(np.float32)
        cols[f"an_miss_{name}"] = miss.astype(np.float32)
        cols[f"an_valid_{name}"] = (kk == k).astype(np.float32)
    for k, name, _h in HORIZONS[1:] + [(2, "0p2", 0.2)]:
        kk = np.minimum(k, na)
        rel = q[r, kk] - p0
        cols[f"tgt_dp_{name}_x"] = rel[:, 0].astype(np.float32)
        cols[f"tgt_dp_{name}_y"] = rel[:, 1].astype(np.float32)
    cols["tgt_v0_x"] = qd[:, 0, 0].astype(np.float32)
    cols["tgt_v0_y"] = qd[:, 0, 1].astype(np.float32)
    vN = qd[r, na]
    cols["tgt_vN_x"] = vN[:, 0].astype(np.float32)
    cols["tgt_vN_y"] = vN[:, 1].astype(np.float32)
    return cols


def option_tokens(meta, q, qd, c4, c4_tensors, c1, c1_tensors):
    """Per (defender, option): the solver summaries, the clipped analytic required acceleration (AMAX 7, at 0.3 / 0.5 / 1.0 s), the
    target geometry, availability masking -> (tokens DataFrame in meta order, previews float16 [n, 11, 11]: dx dy vx vy ax ay
    f_frac lam power_frac c1_dx c1_dy). meta.row_id must be 0..n-1."""
    n = len(meta)
    cols = {}
    cols.update(scatter_cols(n, c4, C4_COPY_COLS, "c4"))
    cols.update(scatter_cols(n, c1, C1_COPY_COLS, "c1"))
    conv = np.zeros(n, dtype=np.float32)
    conv[c4["row_id"].to_numpy()] = (
        (c4["rel_impr_final25"].to_numpy() <= 1e-3)
        & (c4["finite"].to_numpy() > 0)).astype(np.float32)
    cols["c4_converged"] = conv
    cols.update(deterministic_columns(meta, q, qd))
    out = meta[list(ID_COLS)].copy()
    for name in sorted(cols):
        out[name] = np.nan_to_num(cols[name], nan=0.0, posinf=0.0, neginf=0.0)
    feat_cols = [c for c in out.columns if c not in ID_COLS]
    out.loc[out["option_available"].to_numpy() == 0, feat_cols] = 0.0
    prev = np.zeros((n, N_STEPS, N_PREV_CH), dtype=np.float16)
    rid4 = c4["row_id"].to_numpy()
    disp = np.nan_to_num(c4_tensors["disp"].astype(np.float32), nan=0.0)
    vel = np.nan_to_num(c4_tensors["vel"].astype(np.float32), nan=0.0)
    acc = np.nan_to_num(c4_tensors["accel"].astype(np.float32), nan=0.0)
    prev[rid4, :, 0] = disp[:, :, 0].astype(np.float16)
    prev[rid4, :, 1] = disp[:, :, 1].astype(np.float16)
    prev[rid4, :, 2] = vel[:, :, 0].astype(np.float16)
    prev[rid4, :, 3] = vel[:, :, 1].astype(np.float16)
    for ch, key in ((4, None), (5, None), (6, "f_frac"), (7, "lam"), (8, "power_frac")):
        src = acc[:, :, ch - 4] if ch in (4, 5) else np.nan_to_num(c4_tensors[key].astype(np.float32), nan=0.0)
        prev[rid4, :, ch] = np.concatenate([src, src[:, -1:]], axis=1).astype(np.float16)   # 10 -> 11 steps
    rid1 = c1["row_id"].to_numpy()
    path16 = np.nan_to_num(c1_tensors["path16"].astype(np.float32), nan=0.0)
    prev[rid1, :, 9] = path16[:, :, 0].astype(np.float16)
    prev[rid1, :, 10] = path16[:, :, 1].astype(np.float16)
    return out, prev


def teacher_rows(meta, tokens, priv, oracle_v0):
    """The rule teacher: per option the privileged-minus-legal c4 initial control and displacements; per defender the real
    first-step velocity minus the legal one (oracle_v0 [n, 2], NaN where not valid)."""
    prid = priv["row_id"].to_numpy()
    pv_meta = meta.iloc[prid]
    legal_sub = tokens.set_index("row_id").loc[prid]
    teacher = pv_meta[["tag", "game_play_id", "defender_index", "nfl_id", "option_kind", "option_index"]].reset_index(drop=True)
    teacher["row_id"] = prid
    for c, name in (("ax0", "d_ax0"), ("ay0", "d_ay0")):
        teacher[name] = (priv[c].to_numpy(dtype=np.float32) - legal_sub[f"c4_{c}"].to_numpy(dtype=np.float32))
    for h in ("0p2", "0p5", "1p0"):
        for ax in ("x", "y"):
            teacher[f"d_dp_{h}_{ax}"] = (priv[f"dp_{h}_{ax}"].to_numpy(dtype=np.float32)
                                         - legal_sub[f"c4_dp_{h}_{ax}"].to_numpy(dtype=np.float32))
    teacher["priv_finite"] = priv["finite"].to_numpy(dtype=np.float32)
    teacher["priv_converged"] = ((priv["rel_impr_final25"].to_numpy() <= 1e-3) & (priv["finite"].to_numpy() > 0)).astype(np.float32)
    teacher["legal_finite"] = legal_sub["c4_finite"].to_numpy(dtype=np.float32)
    vo = oracle_v0[prid]
    dv = vo - pv_meta[["vx0_legal", "vy0_legal"]].to_numpy()
    teacher["d_vx0"] = np.nan_to_num(dv[:, 0], nan=0.0).astype(np.float32)
    teacher["d_vy0"] = np.nan_to_num(dv[:, 1], nan=0.0).astype(np.float32)
    teacher["d_v0_valid"] = np.isfinite(vo[:, 0]).astype(np.float32)
    teacher = teacher.replace([np.inf, -np.inf], 0.0)
    for c in teacher.columns:
        if teacher[c].dtype == np.float32:
            teacher[c] = np.nan_to_num(teacher[c].to_numpy(), nan=0.0)
    return teacher
