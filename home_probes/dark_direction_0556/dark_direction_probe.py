#!/usr/bin/env python3
"""MTI OPB3 — the DARK-DIRECTION ICL bridge (pre-registered falsifiable test).

Formal seed (verified): elysium/fields/capacity_transport/capacity_transport_proof.py
  - dark_direction_blocks_in_context_learning  (g-dagger = 0  =>  contraction = 1)
  - dark_bright_contraction_gap                (bright sigma^2 g-dagger = 1, dark = 0
                                                =>  contraction gap = 1, maximal)
Pre-registration: _brain/TRUTH_CLAIMS.yaml  id b_opb3_darkdirection_preregis_b4dd.

THE CLAIM (weight-derived, before any behavioural fit). From a transformer's
transport-Gram spectrum at a capacity layer ALONE, a DARK direction (in the
Gram kernel, g-dagger = 0) is in-context-UNLEARNABLE: the one-step in-context
contraction multiplier sigma_i^2 * g-dagger_i is 0, so adding in-context
structure cannot reduce loss along it (contraction = 1, depth-INDEPENDENT). A
BRIGHT control (top of the transport range, sigma^2 g-dagger = 1) reaches the
opposite end. The prediction is a MAXIMAL differential: ~0 learnability on the
dark direction vs ~full on the bright control.

WHAT IS MEASURED (per capacity layer L_c, per task domain):
  query states h_q across prompt instances span the task transport subspace.
    Hc = centered query states;  SVD  Hc = U diag(sigma) V^T.
    sigma_i = task transport singular values;  V_i = task directions.
  readout coupling rho_i = RMS_p |<g_p, V_i>|   (g_p = d logit_correct / d h_Lc).

  per-direction in-context LEARNABILITY (empirical contraction-multiplier proxy):
    L_i = (sigma_i / sigma_1)^2 * (rho_i / rho_max).
    The (sigma/sigma_1)^2 factor is the task curvature a_i; the rho factor is the
    readout's ability to convert capacity-layer motion into a logit change. Their
    product is the empirical analog of sigma_i^2 g-dagger_i. Normalized so the
    brightest task direction is ~1.

  SELECTION (weight-derived):
    bright      = argmax_i L_i                          (top of the range)
    dark_task   = smallest in-window task sigma         (range edge)
    dark_kernel = a unit direction ORTHOGONAL to the task subspace (Gram kernel,
                  sigma ~ 0 by construction; sigma is still MEASURED from the
                  H-projection so L_kernel is a measurement, not a definition).

  HEADLINE differential (analog of the proven gap = 1):
    dark_bright_gap = L_bright(=1) - L_dark_kernel       (predict ~1)

  DYNAMIC cross-check (real models): the per-prompt transport operator T applied
  to v_bright vs v_dark_kernel vs random (||T v||, matrix-free JVP), reported as
  a supplementary operator-gain ratio (the headline stays the spectrum-derived L,
  which carries the task-curvature gate the raw operator norm lacks).

VERDICT: DARK_CONFIRMED if the kernel direction is unlearnable (L_dark_kernel
small) and the dark/bright gap is near maximal across domains; BRIGHT_LEAKS if a
kernel/edge direction shows substantial learnability (would weigh AGAINST the
pre-registered claim). The synthetic gate plants a readout-coupled-but-curvature-
less direction and checks it comes out DARK — i.e. the curvature gate (g-dagger=0)
dominates readout coupling, which is the theorem's content.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

DOMAINS = ["relation_composition", "arithmetic_carry", "induction"]
DEFAULT_MODELS = ["gpt2"]

# learnability is "small" (a direction is effectively dark) below this
DARK_THRESH = 0.10
# the dark/bright gap is "near maximal" above this
GAP_THRESH = 0.80
# how many prompts to use for the dynamic operator-gain cross-check
OPGAIN_PROMPTS = 4


# ---------------------------------------------------------------------------
# Core selection — pure numpy (shared by synthetic gate and real models)
# ---------------------------------------------------------------------------


def _task_rank(sigma: np.ndarray, n_inst: int, *, rel_tol: float = 1e-6,
               cap: int = 24) -> int:
    """Number of task directions estimable from the query-state SVD: singular
    values above rel_tol * sigma_1, capped by the instance budget."""
    if sigma.size == 0 or sigma[0] <= 0:
        return 0
    keep = int(np.sum(sigma >= rel_tol * sigma[0]))
    return max(1, min(keep, n_inst - 1, cap))


def _kernel_direction(V_task: np.ndarray, d: int, seed: int) -> np.ndarray:
    """A unit vector orthogonal to the task subspace span(V_task) — the Gram
    kernel (no task curvature). V_task: (d, r)."""
    rng = np.random.default_rng(seed + 13)
    v = rng.standard_normal(d)
    if V_task.size:
        v = v - V_task @ (V_task.T @ v)        # project out the task subspace
    nrm = np.linalg.norm(v)
    if nrm < 1e-9:                              # degenerate (full-rank task); retry deterministically
        v = rng.standard_normal(d)
        v = v - V_task @ (V_task.T @ v)
        nrm = np.linalg.norm(v) + 1e-12
    return v / nrm


def select_directions(H: np.ndarray, G: np.ndarray, *, seed: int = 0) -> dict:
    """From query states H (N,d) and readout functionals G (N,d), build the task
    transport spectrum, the per-direction learnability profile L_i, and select
    the bright / dark_task / dark_kernel directions. Returns a metrics dict plus
    the chosen unit direction vectors under `_vecs` (stripped before JSON)."""
    H = np.asarray(H, float)
    G = np.asarray(G, float)
    N, d = H.shape
    Hc = H - H.mean(axis=0, keepdims=True)
    U, S, Vh = np.linalg.svd(Hc, full_matrices=False)
    sigma = S
    V = Vh.T                                    # (d, min(N,d)) columns = directions
    r = _task_rank(sigma, N)
    Vr = V[:, :r]
    sig_r = sigma[:r]

    # readout coupling per task direction: RMS over prompts of |<g_p, V_i>|
    proj = G @ Vr                               # (N, r)
    rho = np.sqrt((proj ** 2).mean(axis=0))     # (r,)
    rho_max = float(rho.max()) if rho.size else 0.0

    # per-direction learnability L_i = (sigma_i/sigma_1)^2 * (rho_i/rho_max)
    curv = (sig_r / sig_r[0]) ** 2 if sig_r[0] > 0 else np.zeros(r)
    read = rho / rho_max if rho_max > 0 else np.zeros(r)
    L = curv * read                             # (r,)  empirical contraction-multiplier proxy
    L_max = float(L.max()) if L.size else 0.0
    Ln = L / L_max if L_max > 0 else L          # normalize so brightest task dir ~1

    bright_idx = int(np.argmax(L)) if L.size else 0
    dark_task_idx = int(r - 1)                  # smallest in-window task sigma

    # dark_kernel: orthogonal to the task subspace -> Gram kernel (sigma ~ 0)
    v_kernel = _kernel_direction(Vr, d, seed)
    sigma_kernel = float(np.std(Hc @ v_kernel))             # MEASURED curvature ~ 0
    rho_kernel = float(np.sqrt(((G @ v_kernel) ** 2).mean()))
    curv_kernel = (sigma_kernel / sig_r[0]) ** 2 if sig_r[0] > 0 else 0.0
    read_kernel = rho_kernel / rho_max if rho_max > 0 else 0.0
    L_kernel = curv_kernel * read_kernel
    Ln_kernel = float(L_kernel / L_max) if L_max > 0 else 0.0

    Ln_bright = float(Ln[bright_idx]) if Ln.size else 0.0
    Ln_dark_task = float(Ln[dark_task_idx]) if Ln.size else 0.0

    return {
        "n_instances": int(N),
        "d_model": int(d),
        "task_rank": int(r),
        "sigma_top": round(float(sig_r[0]), 6) if sig_r.size else None,
        "sigma_dark_task": round(float(sig_r[dark_task_idx]), 6) if sig_r.size else None,
        "sigma_kernel_measured": round(sigma_kernel, 6),
        "rho_bright": round(float(rho[bright_idx]), 6) if rho.size else None,
        "rho_dark_task": round(float(rho[dark_task_idx]), 6) if rho.size else None,
        "rho_kernel": round(rho_kernel, 6),
        "L_bright_norm": round(Ln_bright, 6),
        "L_dark_task_norm": round(Ln_dark_task, 6),
        "L_dark_kernel_norm": round(Ln_kernel, 6),
        # headline differential — analog of the proven dark_bright_contraction_gap
        "dark_bright_gap_kernel": round(Ln_bright - Ln_kernel, 6),
        "dark_bright_gap_task": round(Ln_bright - Ln_dark_task, 6),
        "dark_kernel_is_dark": bool(Ln_kernel <= DARK_THRESH),
        "_vecs": {
            "bright": Vr[:, bright_idx].copy(),
            "dark_task": Vr[:, dark_task_idx].copy(),
            "dark_kernel": v_kernel,
        },
    }


# ---------------------------------------------------------------------------
# Synthetic ground-truth gate (numpy only — no torch, no model)
# ---------------------------------------------------------------------------


def run_synthetic(seed: int = 0) -> dict:
    """Plant a transport-Gram with a known bright direction, a known dark task
    edge, AND a readout-coupled-but-curvatureless direction. The curvature gate
    (g-dagger = 0 => dark) must make that direction DARK despite strong readout
    coupling — that is the content of dark_direction_blocks_in_context_learning."""
    rng = np.random.default_rng(seed)
    d, N = 48, 240
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    # task subspace = Q[:, :4] with a decaying spectrum (bright e0 ... dark-edge e3)
    task_dirs = Q[:, :4]
    sig_profile = np.array([4.0, 2.0, 1.0, 0.3])
    coeffs = rng.standard_normal((N, 4)) * sig_profile
    H = coeffs @ task_dirs.T + 0.01 * rng.standard_normal((N, d))

    # readout g reads the BRIGHT direction e0 strongly AND the CURVATURELESS
    # kernel direction Q[:, 6] strongly (the trap: readout coupling without
    # curvature must still be dark).
    e_bright = Q[:, 0]
    e_trap = Q[:, 6]                            # orthogonal to the task subspace
    G = (3.0 * np.outer(rng.standard_normal(N), e_bright)
         + 3.0 * np.outer(rng.standard_normal(N), e_trap)
         + 0.05 * rng.standard_normal((N, d)))

    sel = select_directions(H, G, seed=seed)
    vb = sel["_vecs"]["bright"]
    overlap_bright = float(abs(vb @ e_bright))   # recovered bright ~ e0 ?
    # the trap direction's learnability, measured directly through the selector's
    # curvature gate: build L for e_trap the same way select_directions does
    Hc = H - H.mean(0, keepdims=True)
    sig_top = np.linalg.svd(Hc, compute_uv=False)[0]
    sigma_trap = float(np.std(Hc @ e_trap))
    rho_trap = float(np.sqrt(((G @ e_trap) ** 2).mean()))
    # rho normalizer ~ bright readout coupling
    rho_bright = float(np.sqrt(((G @ e_bright) ** 2).mean()))
    L_trap = ((sigma_trap / sig_top) ** 2) * (rho_trap / (rho_bright + 1e-12))

    verdict = {
        "recovers_bright_direction": bool(overlap_bright > 0.9),
        "bright_overlap": round(overlap_bright, 4),
        "kernel_is_dark": bool(sel["L_dark_kernel_norm"] <= DARK_THRESH),
        "dark_task_below_bright": bool(
            sel["L_dark_task_norm"] < sel["L_bright_norm"]),
        "gap_near_maximal": bool(sel["dark_bright_gap_kernel"] >= GAP_THRESH),
        # the trap: readout-coupled but curvatureless -> must be ~unlearnable
        "curvature_gate_dominates_readout": bool(L_trap <= DARK_THRESH),
        "trap_learnability": round(float(L_trap), 6),
    }
    verdict["all_pass"] = bool(
        verdict["recovers_bright_direction"]
        and verdict["kernel_is_dark"]
        and verdict["dark_task_below_bright"]
        and verdict["gap_near_maximal"]
        and verdict["curvature_gate_dominates_readout"])

    sel.pop("_vecs", None)
    out = {
        "mode": "synthetic",
        "probe": "dark_direction",
        "date": time.strftime("%Y-%m-%d"),
        "spec_anchor": "OPERATOR_LIFT_SPEC.md (OPB3-D dark-direction ICL bridge)",
        "formal_anchors": ["dark_direction_blocks_in_context_learning",
                           "dark_bright_contraction_gap"],
        "truth_claim": "b_opb3_darkdirection_preregis_b4dd",
        "dims": {"d": d, "N": N, "task_subspace": 4},
        "planted": {"sigma_profile": sig_profile.tolist(),
                    "trap": "readout-coupled, curvatureless (g-dagger=0)"},
        "selection": sel,
        "verdict": verdict,
    }
    path = HERE / "dark_direction_synthetic.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print("\n=== synthetic dark-direction verdict ===")
    for kk, vv in verdict.items():
        print(f"  {kk}: {vv}")
    return out


# ---------------------------------------------------------------------------
# Real model — capture query states, select, dynamic operator-gain cross-check
# ---------------------------------------------------------------------------


def _operator_gain_crosscheck(ctx, l_c, prompts, vecs, seed) -> dict:
    """Dynamic supplement: apply the per-prompt transport operator T to the
    bright / dark_kernel / random directions and report ||T v|| ratios. The
    headline stays the spectrum-derived L (this lacks the task-curvature gate),
    but a bright >> dark dynamic gain corroborates the static differential."""
    from jacobian_product_spectrum_probe import _operator_gains

    rng = np.random.default_rng(seed + 31)
    d = vecs["bright"].shape[0]
    v_rand = rng.standard_normal(d)
    v_rand /= (np.linalg.norm(v_rand) + 1e-12)
    cols = np.stack([vecs["bright"], vecs["dark_kernel"], v_rand], axis=1)  # (d,3)

    gains = []
    for (seq, pos) in prompts[:OPGAIN_PROMPTS]:
        try:
            gains.append(_operator_gains(ctx, ctx.layers, l_c, seq, pos, cols))
        except Exception:  # noqa: BLE001
            continue
    if len(gains) < 2:
        return {"available": False}
    GG = np.stack(gains, axis=0).mean(axis=0)              # (3,)
    g_bright, g_dark, g_rand = float(GG[0]), float(GG[1]), float(GG[2])
    return {
        "available": True,
        "n_prompts": len(gains),
        "gain_bright": round(g_bright, 6),
        "gain_dark_kernel": round(g_dark, 6),
        "gain_random": round(g_rand, 6),
        "bright_over_random": round(g_bright / (g_rand + 1e-12), 4),
        "dark_over_random": round(g_dark / (g_rand + 1e-12), 4),
        "bright_over_dark": round(g_bright / (g_dark + 1e-12), 4),
    }


def _model_verdict(per_domain: dict) -> dict:
    def med(key):
        vals = [v[key] for v in per_domain.values() if v.get(key) is not None]
        return float(np.median(vals)) if vals else None

    m_gap = med("dark_bright_gap_kernel")
    m_dark = med("L_dark_kernel_norm")
    n_dom = len(per_domain)
    n_dark = sum(1 for v in per_domain.values() if v.get("dark_kernel_is_dark"))
    confirmed = bool(
        m_gap is not None and m_gap >= GAP_THRESH
        and m_dark is not None and m_dark <= DARK_THRESH)
    leaks = bool(m_dark is not None and m_dark > DARK_THRESH)
    return {
        "median_dark_bright_gap_kernel": round(m_gap, 4) if m_gap is not None else None,
        "median_L_dark_kernel_norm": round(m_dark, 4) if m_dark is not None else None,
        "domains_with_dark_kernel": f"{n_dark}/{n_dom}",
        "classification": ("DARK_CONFIRMED" if confirmed
                           else "BRIGHT_LEAKS" if leaks else "INCONCLUSIVE"),
        "supports_preregistered_claim": confirmed,
    }


def run_model(model_name, n_samples=24, seed=0) -> dict:
    from icl_convergence_probe import safe_name
    from singular_spectrum_probe import capture_query_state
    from capacity_threshold_sweep import make_prompt
    from routing_selection_probe import build_induction
    from probe_base import load_context

    ctx = load_context(model_name)
    l_c = ctx.capacity_layer

    per_domain = {}
    for domain in DOMAINS:
        rng = np.random.default_rng(seed)
        H, G, prompts = [], [], []
        for _ in range(n_samples):
            if domain == "induction":
                seq, pos, correct_id = build_induction(ctx.tok, rng, block_len=8)
            else:
                seq, pos, correct_id = make_prompt(domain, ctx.tok, rng)
            try:
                h_q, g = capture_query_state(
                    ctx.model, ctx.layers, l_c, ctx.norm, ctx.unembed,
                    seq, pos, correct_id, ctx.device)
            except Exception as e:  # noqa: BLE001
                print(f"  {domain}: prompt failed ({type(e).__name__}: {e})")
                continue
            H.append(h_q)
            G.append(g)
            prompts.append((seq, pos))
        if len(H) < 4:
            continue
        sel = select_directions(np.array(H), np.array(G), seed=seed)
        vecs = sel.pop("_vecs")
        sel["operator_gain"] = _operator_gain_crosscheck(ctx, l_c, prompts, vecs, seed)
        per_domain[domain] = sel

    verdict = _model_verdict(per_domain)
    out = {
        "mode": "real_model",
        "probe": "dark_direction",
        "model": model_name,
        "hf_name": ctx.hf_name,
        "date": time.strftime("%Y-%m-%d"),
        "n_layers": len(ctx.layers),
        "capacity_layer": l_c,
        "d_model": ctx.model.config.hidden_size,
        "n_samples": n_samples,
        "formal_anchors": ["dark_direction_blocks_in_context_learning",
                           "dark_bright_contraction_gap"],
        "truth_claim": "b_opb3_darkdirection_preregis_b4dd",
        "per_domain": per_domain,
        "verdict": verdict,
    }
    path = HERE / f"dark_direction_{safe_name(model_name)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    for dom, a in per_domain.items():
        og = a.get("operator_gain", {})
        print(f"  {dom:>22}: gap={a['dark_bright_gap_kernel']} "
              f"L_dark_kernel={a['L_dark_kernel_norm']} "
              f"(dark={a['dark_kernel_is_dark']})  "
              f"op b/d={og.get('bright_over_dark')}")
    print(f"  verdict: {verdict}")
    return out


# ---------------------------------------------------------------------------
# OOP front (probe_base.Probe) + registration
# ---------------------------------------------------------------------------
try:
    from probe_base import Probe, register_probe

    class DarkDirectionProbe(Probe):
        name = "dark_direction"
        classifier_features: list[str] = []   # spectrum measurement, no classifier
        label_key = None

        def synthetic(self, seed: int = 0) -> dict:
            return run_synthetic(seed=seed)

        def run_model(self, model_name: str, *, n_samples: int = 24,
                      seed: int = 0, **_) -> dict:
            return run_model(model_name, n_samples, seed)

    register_probe(DarkDirectionProbe())
except Exception:  # noqa: BLE001  (probe_base optional at import time)
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic(seed=args.seed)
        return
    for m in (args.models or DEFAULT_MODELS):
        try:
            run_model(m, args.n_samples, args.seed)
        except Exception as e:  # noqa: BLE001
            print(f"{m}: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
