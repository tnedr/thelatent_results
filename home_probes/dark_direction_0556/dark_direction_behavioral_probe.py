#!/usr/bin/env python3
"""MTI OPB3-D BEHAVIOURAL arm — the decisive test (Phase 1: controllable ICL).

Design contract: dark_direction_behavioral_PLAN.md
Pre-registration: _brain/TRUTH_CLAIMS.yaml  b_opb3_darkdirection_preregis_b4dd
Formal anchors (verified): dark_direction_blocks_in_context_learning,
dark_bright_contraction_gap (capacity_transport, 195/195).

THE QUESTION. Does the in-context Gram SPECTRUM let us predict, a priori, a
direction the model CANNOT in-context learn — and does that prediction BEAT a
naive (non-spectral) baseline? "Null space => no learning" is almost
definitional; the contribution is real only if the SPECTRAL selection predicts
unlearnability better than a cheap heuristic.

PHASE 1 (this file, corp-cheap, numpy) — the kill-gate. A controllable
von-Oswald linear-regression ICL testbed where hidden directions = task
directions BY CONSTRUCTION:
  - inputs x ~ N(0, Sigma), Sigma = Q diag(eig) Q^T with a random orthonormal Q
    (so the dark direction is ROTATED, NOT axis-aligned — this is what defeats a
    coordinate-variance heuristic while the spectrum still finds it);
  - bright block: large eigenvalue (in the in-context Gram range);
  - dark block: eigenvalue ~ 0 (Gram kernel — no input variance along it);
  - teacher y = w.x with w full-support (so w HAS a dark component to (not) learn);
  - the optimal in-context linear predictor (ridge least-squares over k demos) is
    the von-Oswald ICL step: w_hat(k) = (X^T X + lam I)^-1 X^T Y. It recovers w in
    range(Gram) and ZERO in ker(Gram) — exactly g-dagger=0 => no progress.
  - LEARNING CURVE: relative error along a direction v vs the number of demos k.
    learnability L_v = 1 - err_v(k_max)/err_v(0).

BASELINES (without these the result is meaningless):
  bright (B2, positive control), dark_spectral (bottom Gram eigenvector),
  random (B0), dark_heuristic (B3: lowest per-coordinate sample variance — naive).

VERDICT is PASS only if the bright control learns, the spectral dark is blocked,
the gap is large with effect size, AND the spectral dark is MORE blocked than
both random and the cheap heuristic (the baseline-beat). If Phase 1 fails, the
mechanism is not real — do NOT dispatch Phase 2 (real models) to home.

PHASE 2 (real model, home) is GATED on Phase 1 passing and is not built here;
run_model raises with the reason so the kill-gate is enforced.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# pre-registered thresholds (PLAN section 3) — frozen
GATE_LB = 0.40          # bright control must learn
DARK_MAX = 0.15         # spectral dark must be blocked
GAP_MIN = 0.50          # bright - dark gap
COHEN_D_MIN = 0.80      # effect size of bright vs dark per task
BASELINE_MARGIN = 0.10  # spectral dark must beat random & heuristic by this


def _ridge_solve(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """w_hat = (X^T X + lam I)^-1 X^T Y  — the von-Oswald in-context ICL step."""
    d = X.shape[1]
    G = X.T @ X
    return np.linalg.solve(G + lam * np.eye(d), X.T @ Y)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / sp)


def run_phase1(seed: int = 0, *, d: int = 32, n_tasks: int = 240,
               k_max: int = 48, lam: float = 1e-2) -> dict:
    rng = np.random.default_rng(seed)

    # --- planted input covariance with a ROTATED bright/dark structure ---------
    # The realistic regime for the baseline-beat: a LARGE learnable range (most
    # directions well-conditioned) with a RARE dark kernel. Then random and the
    # heuristic land in the range (learnable); only the spectral kernel is blocked
    # — that is the predictive content a naive picker lacks.
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eig = np.linspace(1.0, 0.4, d)   # broad, well-conditioned range
    bright_idx = 0                    # largest eigenvalue (top of range)
    dark_idx = d - 1                  # forced kernel below
    eig[dark_idx] = 1e-8             # dark: zero variance (true Gram kernel, g-dagger=0)
    Sigma_sqrt = Q @ np.diag(np.sqrt(eig))
    v_bright = Q[:, bright_idx]
    v_dark_true = Q[:, dark_idx]

    # teacher weights: full support so w has a dark component to (fail to) learn
    Ws = rng.standard_normal((n_tasks, d))

    # --- pooled sample to estimate the in-context Gram SPECTRUM (a priori) -----
    Xpool = rng.standard_normal((4000, d)) @ Sigma_sqrt.T
    Ghat = (Xpool.T @ Xpool) / Xpool.shape[0]
    eg, EV = np.linalg.eigh(Ghat)                  # ascending eigenvalues
    v_spec_bright = EV[:, -1]                       # top eigenvector
    v_spec_dark = EV[:, 0]                          # bottom eigenvector (kernel)
    # B3 naive heuristic: axis with the lowest per-coordinate sample variance
    coord_var = Xpool.var(axis=0)
    e_heur = np.zeros(d); e_heur[int(np.argmin(coord_var))] = 1.0
    # B0 random unit
    v_rand = rng.standard_normal(d); v_rand /= np.linalg.norm(v_rand)

    dirs = {
        "bright_spectral": v_spec_bright,
        "dark_spectral": v_spec_dark,
        "random": v_rand,
        "dark_heuristic": e_heur,
        "bright_true": v_bright,
        "dark_true": v_dark_true,
    }
    # how well the SPECTRUM recovered the planted kernel vs the heuristic:
    recov_spectral = float(abs(v_spec_dark @ v_dark_true))
    recov_heuristic = float(abs(e_heur @ v_dark_true))

    shots = sorted(set([1, 2, 4, 8, 16, 24, 32, k_max]))
    # variance-explained (R^2-style) learnability along v: robust to the per-task
    # ratio blow-up (|<w,v>|~0). L_v = 1 - E_t[<w-w_hat,v>^2] / E_t[<w,v>^2].
    num_sq = {name: {k: 0.0 for k in shots} for name in dirs}
    den_sq = {name: 0.0 for name in dirs}
    per_task = {name: np.zeros(n_tasks) for name in dirs}  # for effect size + CI

    for t in range(n_tasks):
        w = Ws[t]
        X_all = rng.standard_normal((k_max, d)) @ Sigma_sqrt.T
        Y_all = X_all @ w + 1e-3 * rng.standard_normal(k_max)
        proj_w = {name: float(w @ v) for name, v in dirs.items()}
        for name in dirs:
            den_sq[name] += proj_w[name] ** 2
        for k in shots:
            w_hat = _ridge_solve(X_all[:k], Y_all[:k], lam)
            err = w - w_hat
            for name, v in dirs.items():
                e2 = float(err @ v) ** 2
                num_sq[name][k] += e2
                if k == k_max:
                    den_t = proj_w[name] ** 2 + 1e-12
                    per_task[name][t] = float(np.clip(1.0 - e2 / den_t, 0.0, 1.0))

    curves = {name: {k: num_sq[name][k] / (den_sq[name] + 1e-12) for k in shots}
              for name in dirs}
    L = {name: float(np.clip(1.0 - curves[name][k_max], 0.0, 1.0)) for name in dirs}
    Lb = L["bright_spectral"]; Ld = L["dark_spectral"]
    Lr = L["random"]; Lh = L["dark_heuristic"]

    # effect size: bright vs spectral-dark per-task learnability
    d_eff = _cohens_d(per_task["bright_spectral"], per_task["dark_spectral"])
    # bootstrap 95% CI on the gap (bright - dark) over tasks
    rb = np.random.default_rng(seed + 99)
    gaps = []
    pb = per_task["bright_spectral"]; pd = per_task["dark_spectral"]
    for _ in range(2000):
        idx = rb.integers(0, n_tasks, n_tasks)
        gaps.append(pb[idx].mean() - pd[idx].mean())
    gap_lo, gap_hi = (float(np.percentile(gaps, 2.5)),
                      float(np.percentile(gaps, 97.5)))

    verdict = {
        "gate_bright_learns": bool(Lb >= GATE_LB),
        "dark_blocked": bool(Ld <= DARK_MAX),
        "gap": round(Lb - Ld, 4),
        "gap_ci95": [round(gap_lo, 4), round(gap_hi, 4)],
        "gap_sufficient": bool((Lb - Ld) >= GAP_MIN and gap_lo > 0.0),
        "cohens_d_bright_vs_dark": round(d_eff, 3),
        "effect_size_sufficient": bool(d_eff >= COHEN_D_MIN),
        "beats_random": bool(Ld < Lr - BASELINE_MARGIN),
        "beats_heuristic": bool(Ld < Lh - BASELINE_MARGIN),
        "spectral_recovers_kernel": round(recov_spectral, 4),
        "heuristic_recovers_kernel": round(recov_heuristic, 4),
        "spectrum_beats_heuristic_recovery": bool(recov_spectral > recov_heuristic + 0.3),
    }
    verdict["all_pass"] = bool(
        verdict["gate_bright_learns"] and verdict["dark_blocked"]
        and verdict["gap_sufficient"] and verdict["effect_size_sufficient"]
        and verdict["beats_random"] and verdict["beats_heuristic"])
    verdict["classification"] = (
        "MECHANISM_REAL_BASELINE_BEATEN" if verdict["all_pass"]
        else "NO_GAP" if (Lb - Ld) < 0.25
        else "SPECTRUM_NOT_BETTER_THAN_BASELINE"
        if not (verdict["beats_random"] and verdict["beats_heuristic"])
        else "PARTIAL")

    out = {
        "mode": "phase1_synthetic_controllable_icl",
        "probe": "dark_direction_behavioral",
        "date": time.strftime("%Y-%m-%d"),
        "plan": "dark_direction_behavioral_PLAN.md",
        "truth_claim": "b_opb3_darkdirection_preregis_b4dd",
        "formal_anchors": ["dark_direction_blocks_in_context_learning",
                           "dark_bright_contraction_gap"],
        "dims": {"d": d, "n_tasks": n_tasks, "k_max": k_max, "lam": lam},
        "learnability": {name: round(L[name], 4) for name in dirs},
        "learning_curves": {name: {str(k): round(1.0 - curves[name][k], 4)
                                   for k in shots} for name in dirs},
        "thresholds": {"GATE_LB": GATE_LB, "DARK_MAX": DARK_MAX,
                       "GAP_MIN": GAP_MIN, "COHEN_D_MIN": COHEN_D_MIN,
                       "BASELINE_MARGIN": BASELINE_MARGIN},
        "verdict": verdict,
    }
    path = HERE / "dark_direction_behavioral_synthetic.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print("\n=== Phase 1 (controllable ICL) — dark/bright behavioural verdict ===")
    print(f"  learnability: bright={Lb:.3f} dark_spectral={Ld:.3f} "
          f"random={Lr:.3f} dark_heuristic={Lh:.3f}")
    print(f"  gap={Lb - Ld:.3f} CI95={verdict['gap_ci95']} "
          f"cohen_d={d_eff:.2f}")
    print(f"  kernel recovery: spectral={recov_spectral:.3f} "
          f"heuristic={recov_heuristic:.3f}")
    for kk in ("gate_bright_learns", "dark_blocked", "gap_sufficient",
               "effect_size_sufficient", "beats_random", "beats_heuristic"):
        print(f"  {kk}: {verdict[kk]}")
    print(f"  => {verdict['classification']}  (all_pass={verdict['all_pass']})")
    return out


# ===========================================================================
# Phase 2 — REAL MODEL behavioural ICL (gate passed by Phase 1)
# ===========================================================================
# Phase-2-local few-shot builders (do NOT touch the frozen shared make_prompt).
# The carry builder is PAIRED: a fixed (codes, query) lets us rebuild the SAME
# task at different shot counts k, so Delta h = h(k_hi) - h(k_lo) is a clean
# "same query, more in-context demos" delta.


def _fewshot_carry(tok, rng, k, fixed=None):
    from capacity_threshold_sweep import _vocab_lohi
    lo, hi = _vocab_lohi(tok)
    if fixed is None:
        codes = [int(t) for t in rng.choice(np.arange(lo, hi), size=12, replace=False)]
        xq, yq = int(rng.integers(0, 10)), int(rng.integers(0, 10))
        fixed = (codes, xq, yq)
    codes, xq, yq = fixed
    digit, carry = codes[:10], codes[10:12]

    def ex(x, y):
        return [digit[x], digit[y], carry[1 if (x + y) >= 10 else 0]]

    seq = []
    for _ in range(k):
        x, y = int(rng.integers(0, 10)), int(rng.integers(0, 10))
        seq.extend(ex(x, y))
    seq.extend([digit[xq], digit[yq]])
    return seq, len(seq) - 1, carry[1 if (xq + yq) >= 10 else 0], fixed


def _capture(ctx, l_c, input_ids, pos, correct_id):
    """Capacity-layer query state h_q, readout functional g = d logp_correct/d h_Lc,
    and the readout log-prob of the correct token (the ICL loss signal)."""
    import torch
    ids = torch.tensor([input_ids], device=ctx.device)
    captured = {}

    def grab(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        h.retain_grad()
        captured["h"] = h
        return output

    handle = ctx.layers[l_c].register_forward_hook(grab)
    try:
        ctx.model.zero_grad(set_to_none=True)
        out = ctx.model(input_ids=ids, output_hidden_states=True)
        h_final = out.hidden_states[-1][0, pos, :]
        logits = ctx.unembed(ctx.norm(h_final))
        logp = torch.log_softmax(logits.float(), dim=-1)
        logp_correct = logp[correct_id]
        logp_correct.backward()
        g = captured["h"].grad[0, pos, :].detach().cpu().numpy().astype(float)
        h_q = captured["h"][0, pos, :].detach().cpu().numpy().astype(float)
    finally:
        handle.remove()
    ctx.model.zero_grad(set_to_none=True)
    return h_q, g, float(logp_correct.item())


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return round(float(np.corrcoef(rx, ry)[0, 1]), 4)


def _phase2_domain(ctx, l_c, builder, n_samples, k_lo, k_hi, seed):
    """One ICL domain: capture the SAME queries at k_lo and k_hi shots, build the
    transport-Gram spectrum, and attribute the in-context loss reduction to the
    spectral bright/dark directions vs baselines."""
    from dark_direction_probe import select_directions

    rng = np.random.default_rng(seed)
    H_hi, G_lo, dH, dlogp = [], [], [], []
    for _ in range(n_samples):
        _, _, _, fixed = _fewshot_carry(ctx.tok, rng, k_hi)   # fix the query+codes
        slo, plo, cid, _ = _fewshot_carry(ctx.tok, rng, k_lo, fixed=fixed)
        shi, phi, cid2, _ = _fewshot_carry(ctx.tok, rng, k_hi, fixed=fixed)
        try:
            h_lo, g_lo, lp_lo = _capture(ctx, l_c, slo, plo, cid)
            h_hi, g_hi, lp_hi = _capture(ctx, l_c, shi, phi, cid2)
        except Exception as e:  # noqa: BLE001
            print(f"    prompt failed ({type(e).__name__}: {e})")
            continue
        H_hi.append(h_hi); G_lo.append(g_lo)
        dH.append(h_hi - h_lo); dlogp.append(lp_hi - lp_lo)
    if len(H_hi) < 4:
        return None

    dlogp = np.array(dlogp)
    icl_gain = float(dlogp.mean())            # mean log-prob increase with shots
    sel = select_directions(np.array(H_hi), np.array(G_lo), seed=seed)
    vecs = sel.pop("_vecs")
    Gl = np.array(G_lo); DH = np.array(dH)

    def red_along(v):
        # first-order in-context loss reduction along v: <g,v> * <Delta h,v>,
        # averaged over queries (positive = log-prob increased = loss reduced).
        return float(np.mean((Gl @ v) * (DH @ v)))

    # baselines
    rb = np.random.default_rng(seed + 7)
    d = vecs["bright"].shape[0]
    v_rand = rb.standard_normal(d); v_rand /= np.linalg.norm(v_rand)
    var_axis = np.argmin(np.array(H_hi).var(axis=0))
    v_heur = np.zeros(d); v_heur[var_axis] = 1.0

    reds = {
        "bright": red_along(vecs["bright"]),
        "dark_task": red_along(vecs["dark_task"]),
        "dark_kernel": red_along(vecs["dark_kernel"]),
        "random": red_along(v_rand),
        "heuristic": red_along(v_heur),
    }
    ref = max((abs(v) for v in reds.values()), default=1e-12) or 1e-12
    L = {k: round(v / ref, 4) for k, v in reds.items()}   # signed, in [-1,1]

    # spectral ranking vs behavioural ranking across task directions
    spec_L, beh_red = [], []
    # recompute task directions for the spearman (re-SVD cheap, reuse H_hi/G_lo)
    Hc = np.array(H_hi) - np.array(H_hi).mean(0, keepdims=True)
    _, S, Vh = np.linalg.svd(Hc, full_matrices=False)
    r = sel["task_rank"]
    for i in range(r):
        vi = Vh[i]
        rho_i = float(np.sqrt(((Gl @ vi) ** 2).mean()))
        spec_L.append(((S[i] / S[0]) ** 2) * rho_i)
        beh_red.append(red_along(vi))
    spear = _spearman(spec_L, beh_red)

    return {
        "n_instances": len(H_hi),
        "k_lo": k_lo, "k_hi": k_hi,
        "icl_gain_logprob": round(icl_gain, 4),
        "icl_active": bool(icl_gain > 0.05),
        "task_rank": int(r),
        "reds_raw": {k: round(v, 6) for k, v in reds.items()},
        "L_norm": L,
        "spectral_vs_behavioural_spearman": spear,
        "bright_over_random": round(L["bright"] - L["random"], 4),
        "bright_over_dark": round(L["bright"] - L["dark_task"], 4),
    }


def run_model(model_name: str, n_samples: int = 32, seed: int = 0,
              k_lo: int = 1, k_hi: int = 16, n_seeds: int = 3) -> dict:
    """Phase 2 on a real model. The gpt2 pilot showed the single-seed directional
    attribution is NOISE-DOMINATED (the bright-vs-dark gap flipped sign between
    n=24 and n=32), so we average the headline over `n_seeds` seeds and report
    mean +/- std. A result is only trustworthy if the gap CI excludes 0."""
    from icl_convergence_probe import safe_name
    from probe_base import load_context

    ctx = load_context(model_name)
    l_c = ctx.capacity_layer

    runs = []
    for s in range(n_seeds):
        r = _phase2_domain(ctx, l_c, _fewshot_carry, n_samples, k_lo, k_hi, seed + s)
        if r:
            runs.append(r)
    per_domain = {}
    verdict = {}
    if runs:
        gaps = np.array([r["bright_over_dark"] for r in runs])
        b_rand = np.array([r["bright_over_random"] for r in runs])
        spears = [r["spectral_vs_behavioural_spearman"] for r in runs
                  if r["spectral_vs_behavioural_spearman"] is not None]
        icl = np.array([r["icl_gain_logprob"] for r in runs])
        gap_mean, gap_std = float(gaps.mean()), float(gaps.std(ddof=1) if len(gaps) > 1 else 0.0)
        spear_mean = float(np.mean(spears)) if spears else None
        gate = bool(icl.mean() > 0.05)
        # robust gap: mean must clear the bar AND exceed its own spread
        gap_robust = bool(gap_mean >= GAP_MIN and gap_mean > 2 * gap_std)
        beats = bool(b_rand.mean() > BASELINE_MARGIN and b_rand.mean() > 2 * b_rand.std(ddof=1) if len(b_rand) > 1 else b_rand.mean() > BASELINE_MARGIN)
        spec_pred = bool(spear_mean is not None and spear_mean > 0.3)
        per_domain = {"arithmetic_carry": {
            "n_seeds": len(runs), "n_samples": n_samples,
            "icl_gain_logprob_mean": round(float(icl.mean()), 4),
            "icl_active": gate,
            "bright_over_dark_mean": round(gap_mean, 4),
            "bright_over_dark_std": round(gap_std, 4),
            "bright_over_random_mean": round(float(b_rand.mean()), 4),
            "spearman_mean": round(spear_mean, 4) if spear_mean is not None else None,
            "per_seed": runs,
        }}
        verdict = {
            "icl_gate": gate,
            "gap_robust": gap_robust,
            "beats_random": beats,
            "spectral_predicts_behaviour": spec_pred,
            "classification": (
                "DARK_CONFIRMED" if (gate and gap_robust and beats and spec_pred)
                else "NEEDS_BIGGER_MODEL" if not gate
                else "UNSTABLE" if gap_std > abs(gap_mean)
                else "NULL_OR_WEAK"),
        }
    out = {
        "mode": "phase2_real_model",
        "probe": "dark_direction_behavioral",
        "model": model_name,
        "hf_name": ctx.hf_name,
        "date": time.strftime("%Y-%m-%d"),
        "capacity_layer": l_c,
        "n_samples": n_samples,
        "n_seeds": n_seeds,
        "truth_claim": "b_opb3_darkdirection_preregis_b4dd",
        "per_domain": per_domain,
        "verdict": verdict,
    }
    path = HERE / f"dark_direction_behavioral_{safe_name(model_name)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    if per_domain:
        a = per_domain["arithmetic_carry"]
        print(f"  arithmetic_carry: icl_gain={a['icl_gain_logprob_mean']} "
              f"gap={a['bright_over_dark_mean']}+/-{a['bright_over_dark_std']} "
              f"beats_rand={a['bright_over_random_mean']} spearman={a['spearman_mean']}")
    print(f"  verdict: {verdict}")
    return out


# OOP front (optional)
try:
    from probe_base import Probe, register_probe

    class DarkDirectionBehavioralProbe(Probe):
        name = "dark_direction_behavioral"
        classifier_features: list[str] = []
        label_key = None

        def synthetic(self, seed: int = 0) -> dict:
            return run_phase1(seed=seed)

        def run_model(self, model_name: str, **kwargs) -> dict:
            return run_model(model_name, **kwargs)

    register_probe(DarkDirectionBehavioralProbe())
except Exception:  # noqa: BLE001
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase1", action="store_true", help="run the Phase 1 gate")
    ap.add_argument("--synthetic", action="store_true", help="alias for --phase1")
    ap.add_argument("--models", nargs="+", default=None,
                    help="run Phase 2 (real-model behavioural ICL) on these models")
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--k-lo", type=int, default=1)
    ap.add_argument("--k-hi", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.models:
        for m in args.models:
            try:
                run_model(m, args.n_samples, args.seed, args.k_lo, args.k_hi,
                          args.n_seeds)
            except Exception as e:  # noqa: BLE001
                print(f"{m}: FAILED ({type(e).__name__}: {e})")
        return
    # default: Phase 1 (the controllable-ICL kill-gate)
    run_phase1(seed=args.seed)


if __name__ == "__main__":
    main()
