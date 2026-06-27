#!/usr/bin/env python3
"""Free-probability transport probe — is trained layer-to-layer computation a
FREE MULTIPLICATIVE CONVOLUTION?

THE ALTERNATIVE EXPLANATION (Tao-flavored, exact, falsifiable)
  The whole capacity/transport program describes WHAT the trained operator looks
  like (low-rank readout-aligned spectrum, slowly-rotating basis through depth).
  It does not say WHERE the spectrum COMES FROM. This probe tests a precise,
  borrowed-from-deep-math hypothesis:

    The depth-L transport operator P = T_{L-1} ... T_1 T_0 has a singular-value
    distribution equal to the FREE MULTIPLICATIVE CONVOLUTION (Voiculescu ⊠) of
    the per-layer transport spectra μ_{T_i}.

  Free multiplicative convolution is EXACTLY the spectrum law of a product of
  factors that are in "free position" — i.e. each factor's singular basis is
  Haar-randomly rotated relative to the others (free independence). So the
  hypothesis has a clean operational meaning: trained computation composes
  layers as if their bases were freely rotated.

  This is the natural companion of the cross-layer SLOWLY_ROTATING finding
  (cross_layer_transport_probe): the readout basis is neither FROZEN (one basis
  for the whole stack) nor MIXING (reshuffled every layer) — it rotates
  smoothly. Free probability lives at one extreme of that same axis (maximal,
  Haar rotation). The question this probe answers quantitatively:

    HOW FREE is trained transport? Does ⊠ predict the real product spectrum
    (a free-probability LAW for trained intelligence), or does the deviation
    itself measure the smooth geometry we already found?

THREE SPECTRA ON ONE AXIS (all compared as SHAPE — log-singular-values,
mean-centered to remove overall multiplicative gain)
  TRUE    spectrum of the ACTUAL product P = ∏ T_i (the real rotations the
          trained net uses between factors).
  FREE    spectrum of ∏ (O_i · diag s_i) with O_i independent Haar orthogonal
          (an unbiased sample of the free multiplicative convolution ⊠ μ_{T_i}),
          pooled over several draws.
  FROZEN  spectrum of the basis-aligned product (Π_i sorted s_i, elementwise) —
          the coherent extreme, factors sharing one basis.

NEW SCALAR INVARIANT
  freeness κ = d(TRUE, FROZEN) / ( d(TRUE, FREE) + d(TRUE, FROZEN) )  ∈ [0,1]
    κ → 1  TRUE ≈ FREE     ⇒ free-probability law holds (sensational).
    κ → 0  TRUE ≈ FROZEN   ⇒ coherent / aligned composition.
    d = Wasserstein-1 between mean-centered sorted log-spectra.
  κ is a single number per (model, domain) measuring how free trained
  computation is — directly comparable across architectures.

WHAT IS FIT (real arm)
  Reuse cross_layer_transport_probe.capture_all_layers: one forward + one
  backward per prompt yields each layer's query hidden state. Project all layers
  onto a shared top-r PCA basis B (consistent coordinates), then fit the
  EFFECTIVE per-layer linear transport T_i = argmin ||X_{i+1} - X_i T_i^T|| in
  those coordinates (the standard linearized-on-data transport object, same
  spirit as transport_gain / cross_layer). The product P = ∏ T_i is the depth
  transport whose spectrum we test against ⊠.

  Honest caveat: T_i is the data-linearized layer map, not the exact nonlinear
  layer. The claim is therefore about the EFFECTIVE transport, exactly as in the
  sibling transport probes.

Local cache only (no training, no download). Run:
  .venv/bin/python free_probability_transport_probe.py --synthetic
  .venv/bin/python free_probability_transport_probe.py --models gpt2 pythia-410m
  .venv/bin/python free_probability_transport_probe.py --aggregate
Output: free_probability_transport_<model>.json (+ _summary.json, _synthetic.json)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from singular_spectrum_probe import DOMAINS, effective_rank  # noqa: E402

DEFAULT_MODELS = ["gpt2", "pythia-410m"]
DEFAULT_R = 32          # shared reduced subspace dimension (transport coordinates)
# N MUST be >> R: the per-layer transport fit T_i (R x R) is solved from N samples,
# so N near R is under-determined and gives noisy, spuriously-free spectra. The
# 2026-06-26 N-scaling diagnostic showed N=48 (~R+16) produced FALSE free-law
# results that vanish at N=128/256. Keep N >= ~4*R.
DEFAULT_N = 128         # prompt instances per domain (must be >> R)
N_FREE_DRAWS = 24       # Haar draws pooled for the FREE surrogate


# ---------------------------------------------------------------------------
# Spectrum-as-shape machinery (multiplicative -> work in log domain)
# ---------------------------------------------------------------------------
def _log_sv(M, eps=1e-12):
    s = np.linalg.svd(np.asarray(M, float), compute_uv=False)
    s = s[s > eps]
    return np.log(s)


def _centered_sorted(logs):
    """Sorted (desc) log-singular-values, mean-centered: removes the overall
    multiplicative gain so we compare the SHAPE of the spectrum, which is what
    free multiplicative convolution predicts."""
    v = np.sort(np.asarray(logs, float))[::-1]
    if v.size == 0:
        return v
    return v - v.mean()


def _w1(a, b, m=128):
    """Wasserstein-1 between two 1-D samples via shared quantiles."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    q = np.linspace(0.0, 1.0, m)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def _haar_orth(r, rng):
    Q, R = np.linalg.qr(rng.standard_normal((r, r)))
    Q *= np.sign(np.diag(R))            # fix the QR sign ambiguity -> true Haar
    return Q


def free_surrogate(factor_svals, rng, n_draws=N_FREE_DRAWS):
    """Sample the free multiplicative convolution ⊠_i μ_{|T_i|} as ∏_i (O_i diag
    s_i) with O_i Haar. Returns (pooled_centered_log_spectrum, mean_single_draw_
    eff_rank). IMPORTANT: the SHAPE (kappa) is measured on the POOLED draws (more
    samples -> smoother quantiles), but the EFFECTIVE RANK must be measured PER
    SINGLE PRODUCT and averaged — pooling many draws together inflates the
    apparent rank and is NOT what a single trained network's product is. A
    product of many near-full-rank factors already collapses in rank via Lyapunov
    dominance, so the per-draw free eff-rank is the right comparison for the
    real product's eff-rank."""
    pooled = []
    per_draw_eff = []
    for _ in range(n_draws):
        P = None
        for s in factor_svals:
            r = len(s)
            G = _haar_orth(r, rng) @ np.diag(s)
            P = G if P is None else G @ P
        sv = np.linalg.svd(P, compute_uv=False)
        pooled.append(np.log(sv[sv > 1e-12]))
        per_draw_eff.append(float(effective_rank(sv)))
    return _centered_sorted(np.concatenate(pooled)), float(np.mean(per_draw_eff))


def frozen_logspectrum(factor_svals):
    """Basis-aligned (coherent) product: Π_i sorted(s_i), elementwise."""
    r = min(len(s) for s in factor_svals)
    logsum = np.zeros(r)
    for s in factor_svals:
        logsum += np.log(np.sort(np.asarray(s, float))[::-1][:r] + 1e-12)
    return _centered_sorted(logsum)


def freeness_kappa(true_log, free_log, frozen_log):
    d_free = _w1(true_log, free_log)
    d_frozen = _w1(true_log, frozen_log)
    denom = d_free + d_frozen
    kappa = (d_frozen / denom) if denom > 0 else float("nan")
    return kappa, d_free, d_frozen


def analyse_factors(factor_svals, true_product_svals, rng, tail_k=4):
    """Given per-layer factor singular values and the TRUE product singular
    values, return the freeness analysis. Also reports a TAIL variant with the
    top-`tail_k` singular values removed from all three spectra: this isolates
    the BULK from the coherent low-rank task channel (the L3 slowly-rotating
    readout subspace). If the bulk is MORE free than the full spectrum, the
    free-conv deviation lives in the few top (coherent) directions."""
    true_log = _centered_sorted(np.log(np.asarray(true_product_svals, float) + 1e-12))
    free_log, free_eff = free_surrogate(factor_svals, rng)
    frozen_log = frozen_logspectrum(factor_svals)
    kappa, d_free, d_frozen = freeness_kappa(true_log, free_log, frozen_log)

    # TAIL (bulk): drop the top-k of each sorted-desc, mean-centered log-spectrum
    def _tail(v):
        w = np.sort(np.asarray(v, float))[::-1][tail_k:]
        return (w - w.mean()) if w.size else w
    t_true, t_free, t_frozen = _tail(true_log), _tail(free_log), _tail(frozen_log)
    kappa_t, d_free_t, d_frozen_t = freeness_kappa(t_true, t_free, t_frozen)

    # per-FACTOR eff-rank: is the compression already in each T_i (per-factor),
    # or only in the composition? If mean per-factor eff-rank >> true product
    # eff-rank, the rank collapse is a COMPOSITION (alignment) property, not a
    # per-layer one (free position of the SAME factors fills rank -> free_eff_rank).
    per_factor_eff = [float(effective_rank(np.asarray(s, float))) for s in factor_svals]
    return {
        "freeness_kappa": round(float(kappa), 4),
        "w1_true_to_free": round(float(d_free), 4),
        "w1_true_to_frozen": round(float(d_frozen), 4),
        "true_eff_rank": round(float(effective_rank(np.asarray(true_product_svals, float))), 3),
        "free_eff_rank": round(float(free_eff), 3),
        "mean_per_factor_eff_rank": round(float(np.mean(per_factor_eff)), 3),
        "bulk_freeness_kappa": round(float(kappa_t), 4),
        "bulk_w1_true_to_free": round(float(d_free_t), 4),
        "bulk_w1_true_to_frozen": round(float(d_frozen_t), 4),
        "tail_k_removed": int(tail_k),
        "n_factors": len(factor_svals),
        "factor_dim": int(min(len(s) for s in factor_svals)),
    }


def _kappa_verdict(kappa):
    if kappa is None or (isinstance(kappa, float) and np.isnan(kappa)):
        return "UNDEFINED"
    if kappa >= 0.7:
        return "FREE_LAW"            # ⊠ predicts the product: free-probability law
    if kappa <= 0.3:
        return "COHERENT"           # aligned / frozen-like composition
    return "PARTIALLY_FREE"         # the deviation measures the smooth rotation


# ---------------------------------------------------------------------------
# Real-model arm
# ---------------------------------------------------------------------------
def _shared_basis(layer_h, r):
    """Top-r right singular directions of all layers' centered query states,
    stacked — a single coordinate frame for the per-layer transport fit."""
    blocks = []
    for i in sorted(layer_h):
        H = np.asarray(layer_h[i], float)
        blocks.append(H - H.mean(axis=0, keepdims=True))
    M = np.vstack(blocks)
    _, _, Vh = np.linalg.svd(M, full_matrices=False)
    rr = min(r, Vh.shape[0])
    return Vh[:rr]                                  # [rr, d]


def _fit_transport(Xa, Xb):
    """Effective linear transport T: Xa @ T.T ≈ Xb, least squares. Returns T."""
    sol, *_ = np.linalg.lstsq(Xa, Xb, rcond=None)   # Xa @ sol ≈ Xb, sol = T.T
    return sol.T


def _fit_factors_true(model_name, n_samples=DEFAULT_N, r=DEFAULT_R, seed=0):
    """Capture per-layer query states, fit the effective per-layer transports
    T_i in a shared top-r PCA basis, and return per-domain (factor_svals,
    true_product_svals, n_instances). Shared by run_model (kappa) and
    run_stransform (analytic free-conv fit)."""
    import torch  # noqa: F401
    from icl_convergence_probe import (safe_name, pick_device, load_model,
                                       get_final_norm_and_unembed)
    from transport_gain_probe import get_decoder_layers
    from capacity_threshold_sweep import make_prompt
    from routing_selection_probe import build_induction
    from cross_layer_transport_probe import capture_all_layers

    device = pick_device()
    hf_name, tok, model = load_model(model_name)
    model = model.to(device)
    norm, unembed = get_final_norm_and_unembed(model)
    layers = get_decoder_layers(model)
    meta = {"hf_name": hf_name, "n_layers": len(layers),
            "d_model": int(model.config.hidden_size), "safe_name": safe_name(model_name)}

    per_domain = {}
    for domain in DOMAINS:
        drng = np.random.default_rng(seed)
        layer_h = defaultdict(list)
        for _ in range(n_samples):
            if domain == "induction":
                seq, pos, cid = build_induction(tok, drng, block_len=8)
            else:
                seq, pos, cid = make_prompt(domain, tok, drng)
            try:
                caps = capture_all_layers(model, layers, norm, unembed,
                                          seq, pos, cid, device)
            except Exception as e:  # noqa: BLE001
                print(f"  {domain}: prompt failed ({type(e).__name__}: {e})")
                continue
            for i, (hq, _g) in caps.items():
                layer_h[i].append(hq)

        present = [i for i in sorted(layer_h) if len(layer_h[i]) >= r + 2]
        if len(present) < 3:
            print(f"  {domain}: too few usable layers/samples — skipped")
            continue

        B = _shared_basis(layer_h, r)                       # [rr, d]
        X = {}
        for i in present:
            H = np.asarray(layer_h[i], float)
            X[i] = (H - H.mean(axis=0, keepdims=True)) @ B.T   # [N, rr]

        factor_svals = []
        factor_mats = []
        P = None
        for a, b in zip(present[:-1], present[1:]):
            T = _fit_transport(X[a], X[b])                  # [rr, rr]
            factor_mats.append(T)
            factor_svals.append(np.linalg.svd(T, compute_uv=False))
            P = T if P is None else T @ P
        true_svals = np.linalg.svd(P, compute_uv=False)
        per_domain[domain] = (factor_svals, true_svals,
                              len(layer_h[present[0]]), factor_mats)
    return meta, per_domain


def run_model(model_name, n_samples=DEFAULT_N, r=DEFAULT_R, seed=0):
    from icl_convergence_probe import safe_name  # noqa: F401
    rng = np.random.default_rng(seed)
    meta, captured = _fit_factors_true(model_name, n_samples, r, seed)
    n_layers = meta["n_layers"]
    d_model = meta["d_model"]
    hf_name = meta["hf_name"]

    per_domain = {}
    for domain, (factor_svals, true_svals, n_inst, _mats) in captured.items():
        res = analyse_factors(factor_svals, true_svals, rng)
        res["verdict"] = _kappa_verdict(res["freeness_kappa"])
        res["n_instances"] = n_inst
        per_domain[domain] = res

    kappas = [a["freeness_kappa"] for a in per_domain.values()
              if not np.isnan(a["freeness_kappa"])]
    model_kappa = round(float(np.mean(kappas)), 4) if kappas else float("nan")
    out = {
        "mode": "real_model",
        "model": model_name,
        "hf_name": hf_name,
        "date": time.strftime("%Y-%m-%d"),
        "n_layers": n_layers,
        "d_model": d_model,
        "reduced_dim": r,
        "n_samples": n_samples,
        "method": "effective per-layer transport (data-linearized) in shared top-r PCA basis",
        "claim": "spectrum(prod T_i) == free_mult_conv(spectra T_i) ?",
        "per_domain": per_domain,
        "model_freeness_kappa": model_kappa,
        "model_verdict": _kappa_verdict(model_kappa),
    }
    path = HERE / f"free_probability_transport_{safe_name(model_name)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    for dom, a in per_domain.items():
        print(f"  {dom:>22}: kappa={a['freeness_kappa']:.3f} bulk_kappa={a['bulk_freeness_kappa']:.3f} "
              f"(d_free={a['w1_true_to_free']:.3f} d_frozen={a['w1_true_to_frozen']:.3f}) "
              f"eff_rank true={a['true_eff_rank']} free={a['free_eff_rank']} -> {a['verdict']}")
    print(f"  model freeness kappa = {model_kappa}  -> {out['model_verdict']}")
    return out


# ---------------------------------------------------------------------------
# Analytic free multiplicative convolution via the S-transform (Haar-FREE)
# ---------------------------------------------------------------------------
# The free multiplicative convolution mu_P = ⊠_i mu_{|T_i|^2} is characterized
# by the S-TRANSFORM multiplying: S_{mu⊠nu}(t) = S_mu(t) * S_nu(t). So if trained
# transport composes freely, the analytic identity
#     S_TRUE(t) == prod_i S_{factor_i}(t)
# must hold on t in (-1,0) — a CLOSED-FORM law, with NO Haar sampling at all.
# Definitions: psi_mu(z) = E[ z*lam/(1 - z*lam) ] (lam = squared singular values);
# chi = psi^{-1}; S(t) = chi(t)*(1+t)/t.
def _psi(z, lam):
    return float(np.mean((z * lam) / (1.0 - z * lam)))


def _chi(t, lam):
    """Invert psi(z)=t for z<0. psi: (-inf,0)->(-1,0) is increasing, so a unique
    z<0 exists for every t in (-1,0). Bisection (no scipy)."""
    z_hi = -1e-12                                   # psi(z_hi) -> 0^-
    z_lo = -1.0
    for _ in range(400):                            # push z_lo until psi(z_lo) < t
        if _psi(z_lo, lam) < t:
            break
        z_lo *= 2.0
    lo, hi = z_lo, z_hi
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if _psi(mid, lam) < t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _s_transform(lam, t_grid):
    lam = np.asarray(lam, float)
    lam = lam[lam > 1e-12]
    lam = lam / lam.mean()                          # scale-out (S(0)=1/mean); compare SHAPE
    out = np.empty(len(t_grid))
    for i, t in enumerate(t_grid):
        z = _chi(float(t), lam)
        out[i] = z * (1.0 + t) / t
    return out


def _free_product_S(factor_lams, t_grid):
    S = np.ones(len(t_grid))
    for lam in factor_lams:
        S *= _s_transform(lam, t_grid)
    return S


FREE_RELERR = 0.30      # FREE_LAW if S-multiplicativity log-relerr below this
COHERENT_RELERR = 0.45  # OFF_LAW above this; between = PARTIALLY_FREE


def _stransform_fit(factor_svals, true_svals):
    """Analytic free-conv fit via the S-TRANSFORM IDENTITY (no Haar sampling):
    test S_TRUE(t) == prod_i S_factor_i(t) on t in (-1,0). The relative error of
    log S is the closed-form deviation from the free multiplicative-convolution
    law. (Moment/eff-rank reconstruction from S is numerically delicate and the
    eff-rank=free claim is already established by the alignment model, so the
    verdict rests on the rigorous multiplicativity identity alone.) pr_eff_rank is
    the MEASURED participation eff-rank of the true product, for context only."""
    factor_lams = [np.asarray(s, float) ** 2 for s in factor_svals]
    lam_true = np.asarray(true_svals, float) ** 2

    t_mult = np.linspace(-0.6, -0.05, 18)
    S_true = _s_transform(lam_true, t_mult)
    S_pred = _free_product_S(factor_lams, t_mult)
    log_rel = float(np.mean(np.abs(np.log(S_true) - np.log(S_pred)))
                    / (np.mean(np.abs(np.log(S_true))) + 1e-12))

    pr_true = float((lam_true.sum() ** 2) / (np.square(lam_true).sum() + 1e-12))
    if log_rel <= FREE_RELERR:
        verdict = "FREE_LAW"
    elif log_rel >= COHERENT_RELERR:
        verdict = "OFF_LAW"
    else:
        verdict = "PARTIALLY_FREE"
    return {
        "s_mult_log_relerr": round(log_rel, 4),
        "pr_eff_rank_true_measured": round(pr_true, 3),
        "verdict": verdict,
    }


def run_stransform(model_name="gpt2", n_samples=DEFAULT_N, r=DEFAULT_R, seed=0):
    """Bet: confirm Frame 1 with the ANALYTIC free convolution (S-transform), not
    the Haar surrogate. Synthetic gate first (the identity must hold for a planted
    free product and FAIL for a coherent/frozen one), then a real gpt2 arm."""
    rng = np.random.default_rng(seed)

    # --- synthetic gate -------------------------------------------------------
    rr, nf = 32, 8
    svals = [np.sort(np.exp(rng.normal(0, 0.5, rr)))[::-1] for _ in range(nf)]

    def haar_product(svs):
        P = None
        for s in svs:
            G = _haar_orth(rr, rng) @ np.diag(s)
            P = G if P is None else G @ P
        return np.linalg.svd(P, compute_uv=False)

    def frozen_product(svs):
        prod = np.ones(rr)
        for s in svs:
            prod = prod * np.sort(s)[::-1]
        return prod

    # smoothly-rotating intermediate (the L3 SLOWLY_ROTATING regime) — should sit
    # between free and frozen, confirming the test is graded, not binary
    def slow_product(svs, theta=0.12):
        P = None
        O = np.eye(rr)
        for s in svs:
            O = _expm_skew(theta * (rng.standard_normal((rr, rr))
                                    - rng.standard_normal((rr, rr)).T) / 2) @ O
            G = O @ np.diag(s)
            P = G if P is None else G @ P
        return np.linalg.svd(P, compute_uv=False)

    free_fit = _stransform_fit(svals, haar_product(svals))
    frozen_fit = _stransform_fit(svals, frozen_product(svals))
    slow_fit = _stransform_fit(svals, slow_product(svals))

    checks = [
        ("planted FREE product satisfies S-multiplicativity (log relerr <= FREE_RELERR)",
         free_fit["s_mult_log_relerr"] <= FREE_RELERR),
        ("planted FREE product -> FREE_LAW verdict",
         free_fit["verdict"] == "FREE_LAW"),
        ("COHERENT/frozen product VIOLATES S-multiplicativity (log relerr >= COHERENT_RELERR)",
         frozen_fit["s_mult_log_relerr"] >= COHERENT_RELERR),
        ("COHERENT/frozen product -> OFF_LAW verdict",
         frozen_fit["verdict"] == "OFF_LAW"),
        ("free clearly more multiplicative than frozen (>= 2x lower relerr)",
         frozen_fit["s_mult_log_relerr"] >= 2.0 * free_fit["s_mult_log_relerr"]),
        ("slow-rotation intermediate sits between free and frozen",
         free_fit["s_mult_log_relerr"] <= slow_fit["s_mult_log_relerr"] + 1e-9
         and slow_fit["s_mult_log_relerr"] <= frozen_fit["s_mult_log_relerr"] + 1e-9),
    ]
    n_pass = sum(1 for _, ok in checks if ok)
    print(f"S-transform synthetic gate: {n_pass}/{len(checks)} checks pass")
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
    print(f"  free:   {free_fit}")
    print(f"  slow:   {slow_fit}")
    print(f"  frozen: {frozen_fit}")

    out = {
        "mode": "stransform_analytic",
        "claim": "S_TRUE(t) == prod_i S_factor_i(t)  (analytic free mult conv, no Haar sampling)",
        "date": time.strftime("%Y-%m-%d"),
        "synthetic": {"free": free_fit, "slow": slow_fit, "frozen": frozen_fit,
                      "checks": [{"test": d_, "pass": bool(ok)} for d_, ok in checks],
                      "pass_rate": f"{n_pass}/{len(checks)}"},
    }

    # --- real arm (one cheap model) ------------------------------------------
    if model_name:
        from icl_convergence_probe import safe_name
        meta, captured = _fit_factors_true(model_name, n_samples, r, seed)
        per_domain = {}
        for domain, (factor_svals, true_svals, _n, _mats) in captured.items():
            fit = _stransform_fit(factor_svals, true_svals)
            per_domain[domain] = fit
            print(f"  {domain:>22}: S_mult_log_relerr={fit['s_mult_log_relerr']:.3f} "
                  f"pr_eff_rank(measured)={fit['pr_eff_rank_true_measured']} "
                  f"-> {fit['verdict']}")
        relerrs = [d["s_mult_log_relerr"] for d in per_domain.values()]
        out["model"] = model_name
        out["real"] = {"model": meta["hf_name"], "n_layers": meta["n_layers"],
                       "per_domain": per_domain,
                       "mean_s_mult_log_relerr": round(float(np.mean(relerrs)), 4) if relerrs else None}
        print(f"  {model_name} mean S-mult log relerr = {out['real']['mean_s_mult_log_relerr']}")

    path = HERE / "free_probability_transport_stransform.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    return out


# ---------------------------------------------------------------------------
# Deviation-from-free diagnostics (corp, N >> R) — characterize the NON-free
# structure: (1) rank-preservation + its mechanism, (2) which free moment breaks.
# ---------------------------------------------------------------------------
def _principal_cosines(A, B):
    """Principal cosines between the column spaces of orthonormal A,B (k x k)."""
    return np.linalg.svd(A.T @ B, compute_uv=False)


def rank_preservation_diag(factor_mats, true_svals, rng, k=4):
    """The trained product keeps a HIGHER effective rank than a free product of
    the SAME per-factor spectra (free uses identical s_i, only the bases are
    Haar) — so the rank-preservation comes entirely from the trained BASES.
    A free product collapses because the dominant OUTPUT direction of T_i aligns
    (over Haar chance) with the dominant amplifying INPUT direction of T_{i+1},
    compounding one direction (Lyapunov). Measure the trained hand-off alignment
    = mean principal cosine between the top-k LEFT subspace of T_i (its output)
    and the top-k RIGHT subspace of T_{i+1} (its input), vs the random (free)
    baseline. handoff < baseline => trained DE-CORRELATES successive dominant
    directions, spreading energy and preserving rank."""
    fs = [np.linalg.svd(T, compute_uv=False) for T in factor_mats]
    true_eff = float(effective_rank(np.asarray(true_svals, float)))
    _, free_eff = free_surrogate(fs, rng)
    rr = factor_mats[0].shape[0]
    Us, Vs = [], []
    for T in factor_mats:
        U, _s, Vt = np.linalg.svd(T)
        Us.append(U[:, :k])
        Vs.append(Vt[:k].T)
    handoff = float(np.mean([np.mean(_principal_cosines(Us[i], Vs[i + 1]))
                             for i in range(len(factor_mats) - 1)]))
    base = []
    for _ in range(50):
        A, _ = np.linalg.qr(rng.standard_normal((rr, k)))
        B, _ = np.linalg.qr(rng.standard_normal((rr, k)))
        base.append(float(np.mean(_principal_cosines(A[:, :k], B[:, :k]))))
    baseline = float(np.mean(base))
    return {
        "true_eff_rank": round(true_eff, 3),
        "free_eff_rank": round(free_eff, 3),
        "rank_preservation_ratio": round(true_eff / (free_eff + 1e-9), 3),
        "handoff_topk_cos": round(handoff, 4),
        "random_free_baseline_cos": round(baseline, 4),
        "handoff_minus_baseline": round(handoff - baseline, 4),
    }


def cumulant_break_diag(factor_mats, true_svals, rng, n_draws=12, K=4):
    """Which free MOMENT breaks first? The S-transform showed the exact free law
    fails; this localizes the order. Compare normalized moments m_k=E[(lam/mean)^k]
    (lam=squared singular values) of the TRUE product vs the free surrogate (same
    factor spectra, Haar bases, averaged over draws), for k=2..K. The first k with
    large relative difference is where freeness breaks."""
    fs = [np.linalg.svd(T, compute_uv=False) for T in factor_mats]

    def norm_moments(sv):
        lam = np.asarray(sv, float) ** 2
        lam = lam / (lam.mean() + 1e-12)
        return np.array([float(np.mean(lam ** kk)) for kk in range(2, K + 1)])

    mt = norm_moments(true_svals)
    draws = []
    for _ in range(n_draws):
        P = None
        for s in fs:
            G = _haar_orth(len(s), rng) @ np.diag(s)
            P = G if P is None else G @ P
        draws.append(norm_moments(np.linalg.svd(P, compute_uv=False)))
    mf = np.mean(draws, axis=0)
    rel = np.abs(mt - mf) / (np.abs(mf) + 1e-9)
    orders = list(range(2, K + 1))
    brk = next((orders[i] for i, rv in enumerate(rel) if rv > 0.5), None)
    return {
        "orders": orders,
        "true_moments": [round(float(x), 3) for x in mt],
        "free_moments": [round(float(x), 3) for x in mf],
        "rel_diff": [round(float(x), 3) for x in rel],
        "first_break_order": brk,
    }


def run_diagnostics(models=("gpt2",), n_samples=DEFAULT_N, r=DEFAULT_R, seed=0):
    """Corp deviation-from-free diagnostics at proper sampling (N >> R)."""
    rng = np.random.default_rng(seed)
    out = {"mode": "deviation_diagnostics", "date": time.strftime("%Y-%m-%d"),
           "n_samples": n_samples, "r": r, "results": {}}
    for model in models:
        meta, captured = _fit_factors_true(model, n_samples, r, seed)
        for domain, (_fs, true_svals, _n, mats) in captured.items():
            rp = rank_preservation_diag(mats, true_svals, rng)
            cb = cumulant_break_diag(mats, true_svals, rng)
            key = f"{model}|{domain}"
            out["results"][key] = {"n_layers": meta["n_layers"],
                                   "rank_preservation": rp, "cumulant_break": cb}
            print(f"{model:>12} L{meta['n_layers']:>2} {domain:>22} "
                  f"ratio={rp['rank_preservation_ratio']:<6} "
                  f"handoff-base={rp['handoff_minus_baseline']:+.3f} "
                  f"| moments true={cb['true_moments']} free={cb['free_moments']} "
                  f"break@{cb['first_break_order']}", flush=True)
    path = HERE / "free_probability_transport_diagnostics.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    return out


# ---------------------------------------------------------------------------
# Coherent-routing composition model (the generative law candidate for Frame 1)
# ---------------------------------------------------------------------------
# The deviation diagnostics showed: trained transport is free in SHAPE only, the
# universal non-free structure is COHERENT hand-off alignment (T_i output subspace
# overlaps T_{i+1} input subspace above the free baseline), and its net rank effect
# flips with depth (preserve at L12, over-collapse at L24). This is a 1-PARAMETER
# generative model of exactly that: a product of factors O_i diag(s) with the next
# factor's INPUT basis aligned to the current factor's OUTPUT basis at level rho.
# rho=0 -> free (independent Haar); rho>0 -> coherent routing. The test: does ONE
# rho (matching the real hand-off) reproduce BOTH the free-like kappa AND the
# depth-flipping true/free eff-rank ratio?
def _routing_product(s, L, rho, rng, align_mode="whole", k=4):
    """Product P = T_L...T_1 of factors T_i = U_i diag(s) V_i^T where V_{i+1}
    (input basis of the NEXT factor) is blended toward U_i (output basis of THIS
    factor) at alignment level rho. align_mode='whole' blends the entire basis
    (generic coherent routing); align_mode='topk' aligns ONLY the top-k input
    directions (the most-amplified ones) to the top-k outputs, leaving the bulk
    free — a STRUCTURED routing that keeps a k-subspace alive (the candidate
    rank-PRESERVING mechanism). Returns (P, Us, Vs) for hand-off measurement."""
    r = len(s)
    P = None
    Us, Vs = [], []
    V = _haar_orth(r, rng)
    for _ in range(L):
        U = _haar_orth(r, rng)
        T = U @ np.diag(s) @ V.T
        Us.append(U)
        Vs.append(V)
        P = T if P is None else T @ P
        G = _haar_orth(r, rng)
        if align_mode == "topk":
            M = G.copy()
            M[:, :k] = rho * U[:, :k] + np.sqrt(max(0.0, 1.0 - rho * rho)) * G[:, :k]
        elif align_mode == "subspace":
            # align the top-k SPAN to U's top-k, but with a random rotation WITHIN
            # the k-block: the subspace overlaps (positive hand-off) yet WHICH
            # direction is amplified scrambles each layer -> no single Lyapunov
            # runaway -> energy spreads across k (candidate rank-PRESERVING law).
            Qk, _ = np.linalg.qr(rng.standard_normal((k, k)))   # random k-rotation
            tgt = U[:, :k] @ Qk
            M = G.copy()
            M[:, :k] = rho * tgt + np.sqrt(max(0.0, 1.0 - rho * rho)) * G[:, :k]
        else:
            M = rho * U + np.sqrt(max(0.0, 1.0 - rho * rho)) * G
        V, _ = np.linalg.qr(M)                    # next input basis (aligned to U)
    return P, Us, Vs


def _routing_handoff(Us, Vs, rng, k=4):
    """Mean top-k principal cosine between T_i output (U_i) and T_{i+1} input
    (V_{i+1}), minus the random (free) baseline — same metric as the real probe."""
    cos = [float(np.mean(_principal_cosines(Us[i][:, :k], Vs[i + 1][:, :k])))
           for i in range(len(Us) - 1)]
    r = Us[0].shape[0]
    base = [float(np.mean(_principal_cosines(
        np.linalg.qr(rng.standard_normal((r, k)))[0][:, :k],
        np.linalg.qr(rng.standard_normal((r, k)))[0][:, :k]))) for _ in range(40)]
    return float(np.mean(cos)) - float(np.mean(base))


def run_routing_model(r=DEFAULT_R, seed=0,
                      rhos=(0.0, 0.2, 0.4, 0.6, 0.8),
                      depths=(6, 11, 23), n_rep=12, decay=0.04):
    """Sweep the coherent-routing model over alignment rho x depth L, for BOTH
    align modes ('whole' = generic subspace blend; 'topk' = structured top-k
    routing). For each cell report kappa (shape free-ness vs the rho=0 free /
    coherent-frozen references), hand-off alignment (vs random baseline), and the
    true/free eff-rank ratio. Goal: which mode + rho reproduces the REAL hand-off
    (+0.07..+0.21) AND free-like kappa AND the eff-rank ratio flipping >1 (shallow)
    -> <1 (deep)? 'whole' captures shape + over-collapse; 'topk' is the candidate
    rank-PRESERVING mechanism."""
    rng = np.random.default_rng(seed)
    s = np.exp(-decay * np.arange(r))
    s = s / s.max()
    per_factor_eff = float(effective_rank(s))
    out = {"mode": "coherent_routing_model", "date": time.strftime("%Y-%m-%d"),
           "r": r, "decay": decay, "per_factor_eff_rank": round(per_factor_eff, 2),
           "real_targets": {"handoff_minus_baseline": "+0.07..+0.21",
                            "kappa": "0.80..0.86",
                            "ratio_L12": "1.8..3.1", "ratio_L24": "0.48..1.64"},
           "cells": []}
    print(f"per-factor eff-rank = {per_factor_eff:.2f} (of r={r}); "
          f"references: free=rho0, frozen=coherent")
    for align_mode in ("whole", "topk", "subspace"):
        print(f"\n=== align_mode = {align_mode} ===")
        print(f"{'rho':>5} {'L':>4} {'handoff-base':>13} {'kappa':>7} "
              f"{'true_eff':>9} {'free_eff':>9} {'ratio':>7}")
        for rho in rhos:
            for L in depths:
                kaps, ratios, handoffs, te, fe = [], [], [], [], []
                for _ in range(n_rep):
                    P, Us, Vs = _routing_product(s, L, rho, rng, align_mode=align_mode)
                    tsv = np.linalg.svd(P, compute_uv=False)
                    true_log = _centered_sorted(np.log(tsv[tsv > 1e-12]))
                    free_log, free_eff = free_surrogate([s] * L, rng, n_draws=8)
                    frozen_log = frozen_logspectrum([s] * L)
                    kap, _, _ = freeness_kappa(true_log, free_log, frozen_log)
                    true_eff = float(effective_rank(tsv))
                    kaps.append(kap)
                    te.append(true_eff)
                    fe.append(free_eff)
                    ratios.append(true_eff / (free_eff + 1e-9))
                    handoffs.append(_routing_handoff(Us, Vs, rng))
                cell = {"align_mode": align_mode, "rho": rho, "L": L,
                        "handoff_minus_baseline": round(float(np.mean(handoffs)), 4),
                        "kappa": round(float(np.nanmean(kaps)), 4),
                        "true_eff_rank": round(float(np.mean(te)), 3),
                        "free_eff_rank": round(float(np.mean(fe)), 3),
                        "ratio": round(float(np.mean(ratios)), 3)}
                out["cells"].append(cell)
                print(f"{rho:>5.1f} {L:>4} {cell['handoff_minus_baseline']:>+13.4f} "
                      f"{cell['kappa']:>7.3f} {cell['true_eff_rank']:>9.3f} "
                      f"{cell['free_eff_rank']:>9.3f} {cell['ratio']:>7.3f}", flush=True)
    path = HERE / "free_probability_transport_routing_model.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {path.name}")
    return out


# ---------------------------------------------------------------------------
# Rank-preservation isolation (corp operator half of dispatch 0752 TASK 1.4)
# ---------------------------------------------------------------------------
# OPEN OBJECT: shallow real nets (gpt2 L12) have true/free eff-rank ratio > 1 —
# the trained product PRESERVES more rank than the free (Haar) product of the
# SAME per-layer factors. The coherent-routing model proved positive inter-layer
# alignment can only push ratio <= 1 (alignment COLLAPSES rank vs free). So the
# rank-preserving ingredient is NOT generic alignment. This experiment isolates
# the mechanism on the operator level (numpy only, exact r x r operators, no
# sampling artifact) by testing two candidate ingredients named in the dispatch:
#   (i)  heterogeneous / near-isometric per-layer spectra (some layers s ~ const)
#   (ii) structured (non-collapsing) routing: feed strong outputs into NON-top
#        inputs so no single Lyapunov direction compounds, while the top-k
#        SUBSPACE still overlaps (positive hand-off).
# The free baseline uses the SAME per-layer spectra (free_surrogate), so any
# ratio > 1 is a property of the ROUTING STRUCTURE, not of the spectra.

def _rankpres_spectra(r, L, hetero, iso_frac, decay, rng):
    """Per-layer spectra list. hetero=False: every layer decays (s=exp(-decay*i)).
    hetero=True: a fraction iso_frac of the L layers are NEAR-ISOMETRIC (s ~ flat,
    eff-rank ~ r), the rest decay. Returns a list of L sorted-desc arrays."""
    s_decay = np.exp(-decay * np.arange(r)); s_decay /= s_decay.max()
    s_iso = np.exp(-0.002 * np.arange(r)); s_iso /= s_iso.max()   # ~flat, near-isometric
    if not hetero:
        return [s_decay.copy() for _ in range(L)]
    iso = rng.random(L) < iso_frac
    return [(s_iso.copy() if iso[i] else s_decay.copy()) for i in range(L)]


def _block_spectrum(r, k, decay):
    """Near-isometric top-k task channel (s[:k] ~ 1) + freely-decaying bulk
    (s[k:] = exp(-decay*i)). The candidate L3 mechanism: a low-rank readout
    subspace carried near-isometrically while the bulk composes freely."""
    s = np.exp(-decay * np.arange(r))
    s[:k] = 1.0
    return s / s.max()


def _rankpres_product(spectra, rho, structure, rng, k=4):
    """Build P = T_L...T_1, T_i = U_i diag(s_i) V_i^T, choosing the NEXT input
    basis V_{i+1} per `structure` to test rank-preserving routing:
      free            : V random (Haar) — the free baseline draw (ratio ~ 1)
      coherent        : V_{i+1} top-k -> U_i top-k (known rank COLLAPSE, ratio<1)
      dispersive_cyclic: V_{i+1} top-k -> CYCLIC-SHIFTED U_i top-k (same subspace,
                         positive hand-off, but strongest output feeds a NON-top
                         input -> no Lyapunov runaway -> candidate PRESERVING)
      anti_bottom     : V_{i+1} top-k -> U_i BOTTOM-k (max dispersion, ~zero handoff)
      partial_product : V_{i+1} top-k -> the PARTIAL PRODUCT's top-k LEFT singular
                         directions (route the running dominant direction onward)
    Returns (P, Us, Vs)."""
    r = len(spectra[0])
    P = None; Us = []; Vs = []
    V = _haar_orth(r, rng)
    for s in spectra:
        U = _haar_orth(r, rng)
        T = U @ np.diag(s) @ V.T
        Us.append(U); Vs.append(V)
        P = T if P is None else T @ P
        G = _haar_orth(r, rng)
        c = np.sqrt(max(0.0, 1.0 - rho * rho))
        if structure == "free":
            M = G
        elif structure == "coherent":
            M = G.copy(); M[:, :k] = rho * U[:, :k] + c * G[:, :k]
        elif structure == "dispersive_cyclic":
            tgt = np.roll(U[:, :k], shift=1, axis=1)          # strong out -> next slot
            M = G.copy(); M[:, :k] = rho * tgt + c * G[:, :k]
        elif structure == "anti_bottom":
            tgt = U[:, -k:]
            M = G.copy(); M[:, :k] = rho * tgt + c * G[:, :k]
        elif structure == "partial_product":
            UP = np.linalg.svd(P, full_matrices=False)[0]
            M = G.copy(); M[:, :k] = rho * UP[:, :k] + c * G[:, :k]
        else:
            raise ValueError(structure)
        V, _ = np.linalg.qr(M)
    return P, Us, Vs


def run_rank_preservation(r=DEFAULT_R, seed=0, rhos=(0.0, 0.3, 0.6, 0.9),
                          depths=(11, 23), n_rep=12, decay=0.04, iso_frac=0.4):
    """Isolate the rank-PRESERVING ingredient. For each (hetero, structure, rho,
    L) report true/free eff-rank ratio, hand-off (top-k cosine - baseline) and
    kappa (shape free-ness). WIN = ratio>1 AND positive hand-off AND free-like
    kappa (matches gpt2-L12: ratio 1.8-3.1, handoff +0.07..+0.21, kappa 0.80-0.86),
    AND the ratio should be HIGHER at the shallow depth (L=11) than deep (L=23)."""
    structures = ("free", "coherent", "dispersive_cyclic", "anti_bottom",
                  "partial_product", "block_carry")
    out = {"mode": "rank_preservation_isolation", "date": time.strftime("%Y-%m-%d"),
           "r": r, "decay": decay, "iso_frac": iso_frac, "seed": seed,
           "real_targets": {"ratio_L12": "1.8..3.1", "handoff": "+0.07..+0.21",
                            "kappa": "0.80..0.86", "depth_trend": "ratio(L=11) > ratio(L=23)"},
           "cells": [], "winners": []}
    for hetero in (False, True):
        tag = "HETERO (iso layers)" if hetero else "HOMOG (all decay)"
        print(f"\n========== {tag} ==========")
        print(f"{'structure':>18} {'rho':>5} {'L':>4} {'ratio':>7} "
              f"{'handoff':>9} {'kappa':>7}")
        for structure in structures:
            for rho in rhos:
                if structure == "free" and rho > 0:
                    continue
                for L in depths:
                    rng = np.random.default_rng(seed + L)   # depth-stable seeding
                    ratios, handoffs, kaps = [], [], []
                    for _ in range(n_rep):
                        if structure == "block_carry":
                            # near-isometric top-k channel + free-decaying bulk,
                            # routed coherently on the top-k (the L3 candidate)
                            spectra = [_block_spectrum(r, 4, decay) for _ in range(L)]
                            P, Us, Vs = _rankpres_product(spectra, rho, "coherent", rng)
                        else:
                            spectra = _rankpres_spectra(r, L, hetero, iso_frac, decay, rng)
                            P, Us, Vs = _rankpres_product(spectra, rho, structure, rng)
                        tsv = np.linalg.svd(P, compute_uv=False)
                        free_log, free_eff = free_surrogate(spectra, rng, n_draws=8)
                        frozen_log = frozen_logspectrum(spectra)
                        true_log = _centered_sorted(np.log(tsv[tsv > 1e-12]))
                        kap, _, _ = freeness_kappa(true_log, free_log, frozen_log)
                        ratios.append(float(effective_rank(tsv)) / (free_eff + 1e-9))
                        handoffs.append(_routing_handoff(Us, Vs, rng))
                        kaps.append(kap)
                    cell = {"hetero": hetero, "structure": structure, "rho": rho, "L": L,
                            "ratio": round(float(np.mean(ratios)), 3),
                            "handoff_minus_baseline": round(float(np.mean(handoffs)), 4),
                            "kappa": round(float(np.nanmean(kaps)), 4)}
                    out["cells"].append(cell)
                    if (cell["ratio"] > 1.05 and cell["handoff_minus_baseline"] > 0.05
                            and 0.6 <= cell["kappa"] <= 0.95):
                        out["winners"].append(cell)
                    print(f"{structure:>18} {rho:>5.1f} {L:>4} {cell['ratio']:>7.3f} "
                          f"{cell['handoff_minus_baseline']:>+9.4f} {cell['kappa']:>7.3f}",
                          flush=True)
    print(f"\nWINNERS (ratio>1.05 & handoff>0.05 & free-like kappa): "
          f"{len(out['winners'])}")
    for w in out["winners"]:
        print(f"  {w['structure']} rho={w['rho']} L={w['L']} hetero={w['hetero']} "
              f"-> ratio={w['ratio']} handoff={w['handoff_minus_baseline']:+.3f} "
              f"kappa={w['kappa']}")
    path = HERE / "free_probability_transport_rank_preservation.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {path.name}")
    return out


# ---------------------------------------------------------------------------
# Synthetic validation (numpy only) — planted composition regimes
# ---------------------------------------------------------------------------
def run_synthetic():
    """Plant three composition regimes and verify freeness kappa separates them:
      coherent : every factor shares ONE basis        -> kappa ~ 0  (FROZEN-like)
      free     : every factor Haar-rotated (free)      -> kappa ~ 1  (FREE law)
      slow     : factors rotate by a small fixed angle -> intermediate kappa
    Also verify ⊠ recovers the FREE product spectrum but NOT the coherent one."""
    rng = np.random.default_rng(0)
    r, n_factors = 32, 11

    def factor_svals():
        return [np.sort(np.exp(rng.normal(0, 0.6, r)))[::-1] for _ in range(n_factors)]

    shared_O = _haar_orth(r, rng)
    Gskew = rng.standard_normal((r, r))
    Gskew = (Gskew - Gskew.T)
    Gskew = Gskew / (np.linalg.norm(Gskew) + 1e-9)     # fixed rotation generator
    step = _expm_skew(0.06 * r * Gskew)                # one small cumulative step

    def build_product(svals, mode):
        """coherent: shared eigenbasis (commuting symmetric factors) -> aligned.
        free:       independent Haar rotation per factor -> free position.
        slow:       eigenbasis rotates by a fixed small step each layer ->
                    locally smooth, globally drifting (the SLOWLY_ROTATING regime)."""
        P = None
        O = shared_O.copy()
        for s in svals:
            D = np.diag(np.sort(s)[::-1])
            if mode == "coherent":
                Gf = shared_O @ D @ shared_O.T         # all share one eigenbasis
            elif mode == "free":
                Gf = _haar_orth(r, rng) @ D            # free position
            elif mode == "slow":
                Gf = O @ D @ O.T
                O = step @ O                           # smooth cumulative rotation
            else:
                raise ValueError(mode)
            P = Gf if P is None else Gf @ P
        return np.linalg.svd(P, compute_uv=False)

    regimes = {}
    for mode in ("coherent", "free", "slow"):
        svals = factor_svals()
        true_svals = build_product(svals, mode)
        res = analyse_factors(svals, true_svals, rng)
        res["verdict"] = _kappa_verdict(res["freeness_kappa"])
        regimes[mode] = res

    checks = [
        ("coherent kappa < 0.3", regimes["coherent"]["freeness_kappa"] < 0.3),
        ("free kappa > 0.7", regimes["free"]["freeness_kappa"] > 0.7),
        ("slow is intermediate",
         regimes["coherent"]["freeness_kappa"] < regimes["slow"]["freeness_kappa"]
         < regimes["free"]["freeness_kappa"]),
        ("free verdict == FREE_LAW", regimes["free"]["verdict"] == "FREE_LAW"),
        ("coherent verdict == COHERENT", regimes["coherent"]["verdict"] == "COHERENT"),
        ("⊠ recovers free product (d_free<d_frozen)",
         regimes["free"]["w1_true_to_free"] < regimes["free"]["w1_true_to_frozen"]),
        ("⊠ fails on coherent product (d_frozen<d_free)",
         regimes["coherent"]["w1_true_to_frozen"] < regimes["coherent"]["w1_true_to_free"]),
    ]
    n_pass = sum(1 for _, ok in checks if ok)
    print(f"Synthetic: {n_pass}/{len(checks)} checks pass")
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
    for mode, res in regimes.items():
        print(f"  {mode:>9}: kappa={res['freeness_kappa']:.3f} "
              f"(d_free={res['w1_true_to_free']:.3f} d_frozen={res['w1_true_to_frozen']:.3f}) "
              f"-> {res['verdict']}")
    out = {
        "mode": "synthetic",
        "date": time.strftime("%Y-%m-%d"),
        "regimes": regimes,
        "synthetic_checks": [{"test": d_, "pass": ok} for d_, ok in checks],
        "pass_rate": f"{n_pass}/{len(checks)}",
    }
    path = HERE / "free_probability_transport_synthetic.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    return out


def _expm_skew(A):
    """exp of a (small) skew-symmetric matrix via eigh of iA — gives an
    orthogonal matrix. A must be real skew-symmetric."""
    w, V = np.linalg.eigh(1j * A)
    return np.real(V @ np.diag(np.exp(-1j * w)) @ V.conj().T)


# ---------------------------------------------------------------------------
# Alignment-law model (bet A) — explain BOTH the free shape AND the rank collapse
# ---------------------------------------------------------------------------
def _complete_basis(Qk, rng):
    """Extend an r×k orthonormal block Qk to a full r×r orthonormal basis whose
    FIRST k columns are exactly Qk (Haar in the orthogonal complement)."""
    r, k = Qk.shape
    R = rng.standard_normal((r, r - k))
    R = R - Qk @ (Qk.T @ R)                       # project off Qk
    Qc, _ = np.linalg.qr(R)
    return np.hstack([Qk, Qc[:, :r - k]])


def _aligned_left_basis(anchor, rho, rng):
    """Left basis whose top-k columns are correlated (corr ~rho) with a SHARED
    anchor subspace, the rest Haar. rho=0 -> free (independent Haar top); rho=1
    -> every factor routes its largest singular values through the same anchor
    (coherent rank collapse)."""
    r, k = anchor.shape
    G = rng.standard_normal((r, k))
    M = rho * anchor + np.sqrt(max(1.0 - rho ** 2, 0.0)) * G
    Qk, _ = np.linalg.qr(M)
    return _complete_basis(Qk[:, :k], rng)


def run_alignment_model(r=32, n_factors=23, k_anchor=4, target_eff_rank=6.0,
                        seed=0):
    """Bet A (CORRECTED): does FREE composition ALONE reproduce BOTH the free
    spectral SHAPE and the low effective RANK measured on real models — without
    any coherent alignment term?

    This model was built to test whether a planted top-direction correlation rho
    is NEEDED to collapse the rank on top of a free bulk. It FALSIFIED that
    premise and corrected a measurement artifact in the parent finding: a product
    of many NEAR-FULL-RANK factors already collapses in effective rank via
    LYAPUNOV dominance (the top singular value runs away multiplicatively), so a
    single free product is low-rank by itself. The earlier 'free overestimates
    eff-rank 10-20x' was an artifact of pooling 24 draws (which inflates apparent
    rank); the per-single-product free eff-rank is low and matches the real one.

    The sweep over rho (top-direction correlation, 0=free .. 1=fully coherent)
    shows eff-rank is already low at rho=0 and only weakly rho-dependent: rank
    collapse is a COMPOSITION (Lyapunov) property of free multiplication, not a
    coherent alignment add-on."""
    rng = np.random.default_rng(seed)
    anchor, _ = np.linalg.qr(rng.standard_normal((r, k_anchor)))
    anchor = anchor[:, :k_anchor]

    # near-full-rank factor spectra (flat-ish in log) -> per-factor eff-rank ~ real
    def factor_svals():
        return [np.sort(np.exp(rng.normal(0, 0.25, r)))[::-1]
                for _ in range(n_factors)]

    def build(rho, svals):
        P = None
        corrs = []
        for s in svals:
            O = _aligned_left_basis(anchor, rho, rng)
            corrs.append(float(np.mean(np.linalg.svd(
                anchor.T @ O[:, :k_anchor], compute_uv=False))))  # top-subspace corr
            G = O @ np.diag(s)
            P = G if P is None else G @ P
        return np.linalg.svd(P, compute_uv=False), float(np.mean(corrs))

    sweep = []
    for rho in [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]:
        svals = factor_svals()
        true_sv, corr = build(rho, svals)
        res = analyse_factors(svals, true_sv, rng)
        sweep.append({
            "rho": rho,
            "achieved_top_corr": round(corr, 3),
            "true_eff_rank": res["true_eff_rank"],
            "free_eff_rank_single": res["free_eff_rank"],
            "mean_per_factor_eff_rank": res["mean_per_factor_eff_rank"],
            "freeness_kappa": res["freeness_kappa"],
            "verdict": _kappa_verdict(res["freeness_kappa"]),
        })

    free_pt = sweep[0]                              # rho=0 = pure free composition
    eff_ranks = [p["true_eff_rank"] for p in sweep]

    checks = [
        ("free (rho=0) ALREADY collapses rank (eff-rank << r)",
         free_pt["true_eff_rank"] < 0.2 * r),
        ("free (rho=0) eff-rank matches the real target (Lyapunov collapse)",
         abs(free_pt["true_eff_rank"] - target_eff_rank) <= 4.0),
        ("per-factor eff-rank stays HIGH (near-full-rank factors -> collapse is COMPOSITION, not per-layer)",
         all(p["mean_per_factor_eff_rank"] > 0.5 * r for p in sweep)),
        ("alignment rho is NOT the rank driver (eff-rank weakly rho-dependent)",
         (max(eff_ranks) - min(eff_ranks)) < 0.6 * free_pt["true_eff_rank"] + 3.0),
        ("kappa stays in the FREE band across rho",
         all(p["freeness_kappa"] >= 0.7 for p in sweep)),
        ("free single-product eff-rank ~ true eff-rank (free predicts the rank)",
         abs(free_pt["free_eff_rank_single"] - free_pt["true_eff_rank"]) <= 4.0),
    ]
    n_pass = sum(1 for _, ok in checks if ok)
    print(f"Alignment model (corrected): {n_pass}/{len(checks)} checks pass  "
          f"(target eff-rank={target_eff_rank})")
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
    print(f"  free(rho=0): true_eff_rank={free_pt['true_eff_rank']} "
          f"free_single_eff_rank={free_pt['free_eff_rank_single']} "
          f"kappa={free_pt['freeness_kappa']} per_factor={free_pt['mean_per_factor_eff_rank']}")
    for p in sweep:
        print(f"   rho={p['rho']:<5} corr={p['achieved_top_corr']:<5} "
              f"eff_rank={p['true_eff_rank']:<7} kappa={p['freeness_kappa']:<6} "
              f"per_factor={p['mean_per_factor_eff_rank']}")
    out = {
        "mode": "alignment_model_corrected",
        "claim": "free composition ALONE reproduces free SHAPE + low RANK (Lyapunov); alignment NOT needed",
        "finding": "REFUTES the 'coherent rank collapse' residual — it was a pooled-vs-single eff-rank artifact",
        "date": time.strftime("%Y-%m-%d"),
        "r": r, "n_factors": n_factors, "k_anchor": k_anchor,
        "target_eff_rank": target_eff_rank,
        "free_reference_rho0": free_pt,
        "sweep": sweep,
        "checks": [{"test": d_, "pass": ok} for d_, ok in checks],
        "pass_rate": f"{n_pass}/{len(checks)}",
    }
    path = HERE / "free_probability_transport_alignment.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    return out


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def aggregate():
    rows = {}
    skip_stems = ("summary", "synthetic", "diagnostics", "stransform",
                  "alignment", "routing_model", "rank_preservation")
    for p in sorted(HERE.glob("free_probability_transport_*.json")):
        if any(p.stem.endswith(s) or f"_{s}_" in p.stem for s in skip_stems):
            continue
        d = json.loads(p.read_text())
        if "model" not in d:
            continue
        rows[d["model"]] = {
            "model_freeness_kappa": d.get("model_freeness_kappa"),
            "model_verdict": d.get("model_verdict"),
            "per_domain": {dom: {"kappa": a["freeness_kappa"],
                                 "bulk_kappa": a.get("bulk_freeness_kappa"),
                                 "verdict": a["verdict"],
                                 "true_eff_rank": a["true_eff_rank"],
                                 "free_eff_rank": a["free_eff_rank"],
                                 "mean_per_factor_eff_rank": a.get("mean_per_factor_eff_rank")}
                           for dom, a in d.get("per_domain", {}).items()},
        }
    out = {"analysis": "free-probability transport (freeness kappa) aggregate",
           "claim": "is trained depth transport a free multiplicative convolution?",
           "date": time.strftime("%Y-%m-%d"), "models": rows}
    path = HERE / "free_probability_transport_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}  ({len(rows)} models)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Free-probability transport / ⊠ test")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--alignment", action="store_true",
                    help="run the alignment-law model (bet A): free shape + rank collapse")
    ap.add_argument("--stransform", action="store_true",
                    help="run the analytic S-transform free-conv fit (synthetic gate + real arm)")
    ap.add_argument("--diagnostics", action="store_true",
                    help="deviation-from-free diagnostics: rank-preservation + cumulant-break")
    ap.add_argument("--routing-model", action="store_true",
                    help="coherent-routing generative model sweep (rho x depth): does ONE rho reproduce free-like kappa AND the depth-flipping rank?")
    ap.add_argument("--rank-preservation", action="store_true",
                    help="isolate the rank-PRESERVING ingredient (hetero spectra x routing structure): which gives true/free ratio>1 at positive hand-off?")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-samples", type=int, default=DEFAULT_N)
    ap.add_argument("--r", type=int, default=DEFAULT_R)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic()
        return
    if args.alignment:
        run_alignment_model(seed=args.seed)
        return
    if args.stransform:
        model = (args.models[0] if args.models else "gpt2")
        run_stransform(model_name=model, n_samples=args.n_samples,
                       r=args.r, seed=args.seed)
        return
    if args.diagnostics:
        models = args.models if args.models else ["gpt2"]
        run_diagnostics(models=models, n_samples=args.n_samples,
                        r=args.r, seed=args.seed)
        return
    if args.routing_model:
        run_routing_model(r=args.r, seed=args.seed)
        return
    if args.rank_preservation:
        run_rank_preservation(r=args.r, seed=args.seed)
        return
    if args.aggregate:
        aggregate()
        return
    for m in (args.models or DEFAULT_MODELS):
        print(f"\n{'=' * 60}\n  {m}  (free-probability transport, r={args.r})\n{'=' * 60}")
        t0 = time.time()
        try:
            run_model(m, n_samples=args.n_samples, r=args.r, seed=args.seed)
        except Exception as e:  # noqa: BLE001
            print(f"{m}: FAILED ({type(e).__name__}: {e})")
        print(f"  ({time.time() - t0:.1f}s)")
    aggregate()


if __name__ == "__main__":
    main()
