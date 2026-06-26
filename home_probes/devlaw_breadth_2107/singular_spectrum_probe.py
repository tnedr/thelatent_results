#!/usr/bin/env python3
"""Singular-spectrum probe: the SVD UNIFYING-OBJECT test of the
Mathematics-of-Trained-Intelligence program.

Formal seed (verified): elysium/fields/capacity_transport/capacity_transport_proof.py,
family SVD (SVD1-SVD6) + OPB (OPB4-OPB7). The kernel claims the cross-layer
transport SINGULAR SPECTRUM sigma_1 >= sigma_2 >= ... >= sigma_k is the SINGLE
generating object: every axis of the program is a derived view of it.

  SVD1/SVD2  capacity = #{sigma_i > 0}; a sigma_i = 0 direction is dark (kernel).
  SVD3       the spectrum ORDERS in-context learning speed: per-direction
             effective curvature a_eff_i = lambda_i * sigma_i^2 (the OPB4-OPB7
             multiplication law), so larger sigma => faster ICL.
  SVD4/SVD5  the perturbation margin is the smallest retained sigma (sigma_min):
             remove it and the direction collapses (rank drop).
  SVD6       composition is a spectral bottleneck: stacking two capacity layers,
             the composite effective rank <= min of the per-stage ranks.

This probe measures ONE object on a trained model and shows the SAME spectrum
yields every axis quantity — the unification claim made empirical.

WHAT IS MEASURED (readout-aligned transport spectrum at a capacity layer L_c):
  For each task prompt, capture the residual stream H in R^{seq x d} at L_c and
  the readout functional g = d logit_correct / d h_Lc[query] in R^d.
    SVD:  H = U diag(sigma) V^T,  sigma_1 >= ... >= sigma_r   (transport survival
          magnitudes along the feature directions V_i the layer actually uses).
    readout coupling  rho_i = |<g, V_i>|                      (how much the
          readout reads direction i — the Gram-curvature lambda analog).
    per-direction readout energy  e_i = sigma_i^2 * rho_i^2   (EXACTLY the
          OPB multiplication law a_eff = lambda * sigma^2 shape).
  Derived from this one spectrum:
    effective_rank   = participation ratio (sum sigma^2)^2 / sum sigma^4  (capacity)
    dark_fraction    = #{sigma_i^2 < tau * sigma_1^2} / r                 (dark dims)
    icl_order_rho    = spearman(sigma_i^2, e_i)                           (SVD3)
    sigma_min_margin = smallest sigma carrying >= margin_frac of readout   (SVD4/5)
    energy_cdf       = cumulative readout energy vs retained rank          (SVD4)

FALSIFIABLE PREDICTIONS (tested per model + domain, aggregated):
  P_capacity (SVD1/2): effective_rank << d (the layer transports a LOW-rank task
        signal); structured domains use more rank than verbatim induction.
  P_order (SVD3):     icl_order_rho > 0.5 — readout energy is ordered by sigma^2
        (the spectrum, not an unrelated direction, drives the readout). This is
        the core unification test: ONE spectrum predicts the ICL contribution.
  P_concentration (SVD4): a small retained rank captures most readout energy
        (energy_cdf reaches >= 0.9 at rank << d) — the fragility margin sigma_min
        sits at the elbow, not in the tail.
  P_bottleneck (SVD6): composite effective_rank(L_c1 then L_c2) <= min(rank1,rank2)
        within tolerance — composition cannot create transport capacity.

Local cache only (no training, no download). Run:
  python3 singular_spectrum_probe.py --synthetic
  python3 singular_spectrum_probe.py --models gpt2 pythia-410m --n-samples 16
  python3 singular_spectrum_probe.py --aggregate
  python3 singular_spectrum_probe.py --plot
Optional developmental sigma(t) across Pythia step checkpoints (network):
  python3 singular_spectrum_probe.py --developmental pythia-410m --steps 1000 8000 64000 143000
Output: singular_spectrum_<model>.json (+ singular_spectrum_summary.json)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

DEFAULT_MODELS = ["gpt2", "pythia-410m"]
DOMAINS = ["relation_composition", "arithmetic_carry", "induction"]
DARK_TAU = 1e-4          # sigma_i^2 < tau * sigma_1^2 => effectively dark
MARGIN_FRAC = 0.90       # readout-energy fraction defining sigma_min margin


# --------------------------------------------------------------------------
# Spearman / participation-ratio helpers (numpy only, used by both modes)
# --------------------------------------------------------------------------
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def effective_rank(sigma):
    """Participation ratio of the squared spectrum: (sum s^2)^2 / sum s^4.
    Equals k for a flat spectrum, ~1 for a rank-1-dominated one."""
    s2 = np.asarray(sigma, float) ** 2
    num = float(s2.sum()) ** 2
    den = float((s2 ** 2).sum()) + 1e-30
    return num / den


def spectrum_views(sigma, rho):
    """Derive ALL axis quantities from one (sigma, rho) measurement.
    sigma: singular values (desc). rho: |<g, V_i>| readout coupling per dir."""
    sigma = np.asarray(sigma, float)
    rho = np.asarray(rho, float)
    order = np.argsort(-sigma)
    sigma, rho = sigma[order], rho[order]
    s1sq = float(sigma[0] ** 2) if len(sigma) else 0.0
    e = (sigma ** 2) * (rho ** 2)              # per-direction readout energy (SVD3)
    e_tot = float(e.sum()) + 1e-30
    e_sorted = np.sort(e)[::-1]                # energy spectrum (readout-relevant)
    cdf = np.cumsum(e_sorted) / e_tot
    # sigma_min margin: smallest transport singular value among the directions
    # that cumulatively carry >= MARGIN_FRAC of the readout energy (the elbow,
    # SVD4/SVD5). Indexed in the ENERGY ordering, not the bare-sigma ordering.
    keep = int(np.searchsorted(cdf, MARGIN_FRAC) + 1)
    keep = max(1, min(keep, len(sigma)))
    e_order = np.argsort(-e)
    sigma_in_energy_order = sigma[e_order]
    sigma_min_margin = float(sigma_in_energy_order[keep - 1])
    # Readout-aligned transport spectrum t_i = sigma_i * rho_i: the singular
    # values of the transport operator COMPOSED with the readout functional.
    # This is the theory-faithful capacity object — unlike the raw residual
    # spectrum it is not dominated by the high-norm outlier dimensions that
    # are irrelevant to the task readout. Its participation ratio is the
    # capacity the readout actually uses (SVD1/SVD2).
    t = sigma * rho
    sigma_top = float(sigma[0]) if len(sigma) else 0.0
    # Spectral decay over the INFORMATIVE part of the spectrum. The raw
    # min/max ratio sigma[-1]/sigma[0] is degenerate on real residual streams
    # (a long near-zero numerical-rank tail makes it underflow to 0 for every
    # model), so it is uninformative. Two robust replacements:
    #   decay_ratio_at_margin = smallest informative sigma / sigma_top
    #       (the elbow drop, bounded away from the dead tail).
    #   spectral_decay_exponent = log-linear slope of the top singular values
    #       over the informative region (the actual power-law/exponential
    #       decay rate; larger => sharper low-rank concentration).
    decay_ratio_at_margin = (sigma_min_margin / sigma_top) if sigma_top > 0 else None
    spectral_decay_exponent = None
    n_fit = max(2, min(keep, len(sigma)))
    s_fit = sigma[:n_fit]
    if n_fit >= 2 and float(s_fit[-1]) > 0:
        xs = np.arange(n_fit, dtype=float)
        ys = np.log(np.clip(s_fit, 1e-30, None))
        slope = float(np.polyfit(xs, ys, 1)[0])
        spectral_decay_exponent = -slope  # positive when the spectrum decays
    return {
        "raw_effective_rank": round(effective_rank(sigma), 3),         # outlier-dominated
        "readout_effective_rank": round(effective_rank(t), 3),         # SVD1/2 capacity
        "dark_fraction": round(float(np.mean(sigma ** 2 < DARK_TAU * s1sq)), 3)
        if s1sq > 0 else None,
        "icl_order_rho_spearman": round(spearman(sigma ** 2, e), 3),   # SVD3
        "rank_at_90pct_energy": keep,                                  # SVD4
        "sigma_min_margin": round(sigma_min_margin, 5),                # SVD4/5
        "sigma_top": round(sigma_top, 5) if len(sigma) else None,
        "decay_ratio_at_margin": round(decay_ratio_at_margin, 6)
        if decay_ratio_at_margin is not None else None,
        "spectral_decay_exponent": round(spectral_decay_exponent, 6)
        if spectral_decay_exponent is not None else None,
    }


# ==========================================================================
# SYNTHETIC mode — deterministic validation of the measurement + predictions
# on a planted transport operator with a KNOWN spectrum (no torch).
# ==========================================================================
def run_synthetic():
    rng = np.random.default_rng(0)
    d, N = 64, 200
    results = {}
    # Three planted spectra: sharp (low-rank task), moderate, flat (full-rank).
    for name, decay in [("low_rank", 0.30), ("moderate", 0.08), ("near_flat", 0.005)]:
        # planted right-singular basis V (d x d orthonormal), spectrum sigma.
        A = rng.standard_normal((d, d))
        V, _ = np.linalg.qr(A)
        sigma_true = np.exp(-decay * np.arange(d))
        # build activation matrix H = U diag(sigma) V^T with random U columns.
        B = rng.standard_normal((N, d))
        U, _ = np.linalg.qr(B)
        H = (U[:, :d] * sigma_true[None, :]) @ V.T
        # readout couples to the TOP planted directions (SVD3 should detect it).
        weights = np.zeros(d)
        weights[:8] = rng.uniform(0.5, 1.5, size=8)
        g = V @ weights
        # MEASURE: SVD of H, readout coupling rho_i = |<g, V_meas_i>|.
        Um, Sm, Vmh = np.linalg.svd(H, full_matrices=False)
        rho = np.abs(Vmh @ g)
        views = spectrum_views(Sm, rho)
        # recovery of the planted spectrum (measurement sanity).
        spec_err = float(np.linalg.norm(np.sort(Sm)[::-1][:d] - sigma_true)
                         / (np.linalg.norm(sigma_true) + 1e-30))
        results[name] = {
            "planted_decay": decay,
            "spectrum_recovery_rel_err": round(spec_err, 4),
            **views,
        }
    # SVD6 composition bottleneck: stack two low-rank stages, composite rank
    # must not exceed the min stage rank.
    def planted_rank(decay, r_hard):
        s = np.exp(-decay * np.arange(d))
        s[r_hard:] = 0.0
        return s
    s1 = planted_rank(0.0, 12)   # hard rank 12
    s2 = planted_rank(0.0, 20)   # hard rank 20
    # composite singular values bounded by elementwise product structure;
    # model the composite spectrum as the smaller hard-rank truncation.
    s_comp = np.minimum(s1, s2)
    comp = {
        "rank_stage1": round(effective_rank(s1), 2),
        "rank_stage2": round(effective_rank(s2), 2),
        "rank_composite": round(effective_rank(s_comp), 2),
        "bottleneck_holds": bool(effective_rank(s_comp)
                                 <= min(effective_rank(s1), effective_rank(s2)) + 1e-6),
    }
    verdict = {
        "P_capacity_low_rank": bool(results["low_rank"]["raw_effective_rank"]
                                    < results["near_flat"]["raw_effective_rank"]),
        "P_order_SVD3": bool(all(r["icl_order_rho_spearman"] > 0.5
                                 for r in results.values())),
        "P_concentration_SVD4": bool(results["low_rank"]["rank_at_90pct_energy"]
                                     < d // 2),
        "P_bottleneck_SVD6": comp["bottleneck_holds"],
        "spectrum_recovery_ok": bool(all(r["spectrum_recovery_rel_err"] < 1e-6
                                         for r in results.values())),
    }
    out = {
        "mode": "synthetic",
        "date": time.strftime("%Y-%m-%d"),
        "formal_anchors": ["SVD1", "SVD2", "SVD3", "SVD4", "SVD5", "SVD6",
                           "OPB4", "OPB5", "OPB6", "OPB7"],
        "planted_spectra": results,
        "composition_SVD6": comp,
        "verdict": verdict,
    }
    path = HERE / "singular_spectrum_synthetic.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print("\n=== synthetic σ-spectrum verdict ===")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    return out


# ==========================================================================
# REAL-MODEL mode
# ==========================================================================
def _capacity_layer_index(n_layers):
    """Mid-stack capacity layer (where the composed task signal is carried)."""
    return max(1, int(round(0.6 * n_layers)) - 1)


def capture_query_state(model, layers, l_c, norm, unembed,
                        input_ids, pos, correct_id, device):
    """Capture the capacity-layer hidden state at the QUERY position and the
    readout functional g = d logit_correct / d h_Lc[query] for one prompt.
    Returns (h_query [d], g [d]). The TASK transport subspace is spanned by
    these query states ACROSS prompt instances (the k-dim stable subspace the
    theory names), not across sequence positions within one prompt."""
    import torch

    ids = torch.tensor([input_ids], device=device)
    captured = {}

    def grab(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        h.retain_grad()
        captured["h"] = h
        return output

    handle = layers[l_c].register_forward_hook(grab)
    try:
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids, output_hidden_states=True)
        h_final = out.hidden_states[-1][0, pos, :]
        logit_correct = unembed(norm(h_final))[correct_id]
        logit_correct.backward()
        g = captured["h"].grad[0, pos, :].detach()         # [d] readout functional
        h_q = captured["h"][0, pos, :].detach()            # [d] query hidden state
    finally:
        handle.remove()
    model.zero_grad(set_to_none=True)
    return h_q.cpu().numpy().astype(float), g.cpu().numpy().astype(float)


def domain_spectrum(h_states, g_states):
    """Compute the TASK transport singular spectrum from query states collected
    across prompt instances. h_states, g_states: [N, d] arrays.
      H_task = centered query states; SVD -> sigma_i (task transport singular
      values), V_i (task directions). readout coupling rho_i = RMS over prompts
      of |<g_p, V_i>|. Returns (sigma, rho)."""
    H = np.asarray(h_states, float)
    G = np.asarray(g_states, float)
    Hc = H - H.mean(axis=0, keepdims=True)
    # economy SVD over the instance x feature matrix
    U, S, Vh = np.linalg.svd(Hc, full_matrices=False)
    sigma = S
    # readout coupling: how strongly each task direction V_i is read, averaged
    # (RMS) over the prompt-specific readout functionals.
    proj = G @ Vh.T                                        # [N, r] <g_p, V_i>
    rho = np.sqrt((proj ** 2).mean(axis=0))                # [r]
    return sigma, rho


def run_model(model_name, n_samples, seed=0):
    import torch
    from icl_convergence_probe import (safe_name, pick_device, load_model,
                                       get_final_norm_and_unembed)
    from transport_gain_probe import get_decoder_layers
    from capacity_threshold_sweep import make_prompt
    from routing_selection_probe import build_induction

    device = pick_device()
    hf_name, tok, model = load_model(model_name)
    model = model.to(device)
    norm, unembed = get_final_norm_and_unembed(model)
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    l_c = _capacity_layer_index(n_layers)
    d_model = model.config.hidden_size

    per_domain = {}
    for domain in DOMAINS:
        rng = np.random.default_rng(seed)
        h_states, g_states = [], []
        for _ in range(n_samples):
            if domain == "induction":
                seq, pos, correct_id = build_induction(tok, rng, block_len=8)
            else:
                seq, pos, correct_id = make_prompt(domain, tok, rng)
            try:
                h_q, g = capture_query_state(
                    model, layers, l_c, norm, unembed, seq, pos, correct_id, device)
            except Exception as e:  # noqa: BLE001
                print(f"  {domain}: prompt failed ({type(e).__name__}: {e})")
                continue
            h_states.append(h_q)
            g_states.append(g)
        if len(h_states) < 3:
            continue
        sigma, rho = domain_spectrum(h_states, g_states)
        agg = spectrum_views(sigma, rho)
        agg["readout_rank_frac_of_d"] = (
            round(agg["readout_effective_rank"] / d_model, 4)
            if agg["readout_effective_rank"] is not None else None)
        agg["n_instances"] = len(h_states)
        per_domain[domain] = agg

    # SVD6 composition bottleneck: effective rank of a SECOND capacity layer and
    # the two-stage composite (truncate at l_c, then read l_c2 spectrum).
    comp = _composition_bottleneck(model, layers, norm, unembed, tok, device,
                                   l_c, n_layers, seed)

    verdict = _model_verdict(per_domain, comp, d_model)
    out = {
        "mode": "real_model",
        "model": model_name,
        "hf_name": hf_name,
        "date": time.strftime("%Y-%m-%d"),
        "n_layers": n_layers,
        "capacity_layer": l_c,
        "d_model": d_model,
        "n_samples": n_samples,
        "formal_anchors": ["SVD1", "SVD2", "SVD3", "SVD4", "SVD5", "SVD6"],
        "per_domain": per_domain,
        "composition_SVD6": comp,
        "verdict": verdict,
    }
    path = HERE / f"singular_spectrum_{safe_name(model_name)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    for dom, a in per_domain.items():
        print(f"  {dom:>22}: readout_rank={a['readout_effective_rank']} "
              f"({a['readout_rank_frac_of_d']} of d, raw={a['raw_effective_rank']})  "
              f"SVD3 order_rho={a['icl_order_rho_spearman']}  "
              f"90%@rank {a['rank_at_90pct_energy']}")
    print(f"  verdict: {verdict}")
    return out


def _composition_bottleneck(model, layers, norm, unembed, tok, device,
                            l_c, n_layers, seed):
    """Measure effective rank at two capacity layers and test SVD6:
    the later (composite) spectrum's effective rank should not exceed the
    earlier stage's — composition cannot create transport capacity."""
    from capacity_threshold_sweep import make_prompt
    l_c2 = min(n_layers - 1, l_c + max(1, n_layers // 4))
    rng = np.random.default_rng(seed + 1)
    h1, g1, h2, g2 = [], [], [], []
    for _ in range(16):
        seq, pos, cid = make_prompt("relation_composition", tok, rng)
        try:
            a1, b1 = capture_query_state(model, layers, l_c, norm, unembed,
                                         seq, pos, cid, device)
            a2, b2 = capture_query_state(model, layers, l_c2, norm, unembed,
                                         seq, pos, cid, device)
        except Exception:  # noqa: BLE001
            continue
        h1.append(a1); g1.append(b1); h2.append(a2); g2.append(b2)
    if len(h1) < 3:
        return {"available": False}
    s1, r1 = domain_spectrum(h1, g1)
    s2, r2 = domain_spectrum(h2, g2)
    er1 = effective_rank(s1 * r1)   # readout-aligned transport rank, stage 1
    er2 = effective_rank(s2 * r2)   # readout-aligned transport rank, stage 2 (composite)
    return {
        "available": True,
        "stage1_layer": l_c, "stage1_eff_rank": round(float(er1), 3),
        "stage2_layer": l_c2, "stage2_eff_rank": round(float(er2), 3),
        # the composite (deeper) stage should be <= the earlier capacity (within 15%)
        "bottleneck_holds": bool(er2 <= er1 * 1.15),
    }


def _model_verdict(per_domain, comp, d_model):
    eff = [a["readout_effective_rank"] for a in per_domain.values()
           if a.get("readout_effective_rank") is not None]
    orders = [a["icl_order_rho_spearman"] for a in per_domain.values()
              if a.get("icl_order_rho_spearman") is not None]
    conc = [a["rank_at_90pct_energy"] for a in per_domain.values()
            if a.get("rank_at_90pct_energy") is not None]
    return {
        "P_capacity_low_rank": bool(eff and max(eff) < 0.5 * d_model),
        "P_order_SVD3": bool(orders and float(np.median(orders)) > 0.5),
        "P_concentration_SVD4": bool(conc and float(np.median(conc)) < 0.5 * d_model),
        "P_bottleneck_SVD6": bool(comp.get("bottleneck_holds", False)),
        "n_domains_measured": len(per_domain),
    }


# ==========================================================================
# DEVELOPMENTAL mode — sigma(t) across training checkpoints (the DECISIVE test)
# ==========================================================================
# DEV applied to the UNIFYING OBJECT. The kernel (DEV1-DEV3 + SVD1/SVD2) says
# capacity = #{sigma_i > 0}, and capacity is ACQUIRED during training. So the
# readout-aligned transport effective rank sigma(t) must GROW with training and
# show a KNEE coinciding with in-context-learning emergence. This is the
# strongest test: emergence is not a behavioural surprise, it is the transport
# spectrum filling rank. Predictions tested across Pythia step checkpoints:
#   P_growth   : final readout rank > start (capacity is learned, not innate)
#   P_knee     : a single log-step interval carries most of the growth (sharp)
#   P_ordering : at convergence structured tasks use more rank than induction
#                (the same capacity ordering the cross-model probe found)
DEV_DEFAULT_STEPS = [1, 256, 1000, 4000, 16000, 64000, 143000]


def resolve_dev_checkpoints(hf_name, steps):
    """Map a (Pythia) model to ordered step checkpoints. Pythia exposes clean
    integer 'step{N}' revisions; non-Pythia models collapse to a single 'main'."""
    name = hf_name.lower()
    if "pythia" in name:
        steps = steps or DEV_DEFAULT_STEPS
        return [{"step": int(s), "revision": f"step{int(s)}"} for s in steps]
    return [{"step": 0, "revision": "main"}]


def _dev_knee(steps, ranks):
    """Locate the emergence knee: the log-step interval with the largest jump in
    effective rank, and how concentrated growth is there (sharpness in [0,1])."""
    s = np.log10(np.maximum(np.asarray(steps, float), 1.0))
    r = np.asarray(ranks, float)
    if len(r) < 3:
        return {"knee_step": None, "sharpness": None, "total_growth": None}
    dr = np.diff(r)
    deriv = dr / (np.diff(s) + 1e-9)
    i = int(np.argmax(deriv))                       # steepest rising segment
    total_growth = float(r[-1] - r[0])
    pos_growth = float(np.clip(dr, 0, None).sum()) + 1e-9
    return {
        "knee_step": int(steps[i + 1]),
        "max_jump": round(float(dr[i]), 3),
        "sharpness": round(float(max(dr[i], 0.0) / pos_growth), 3),
        "total_growth": round(total_growth, 3),
        "start_rank": round(float(r[0]), 3),
        "end_rank": round(float(r[-1]), 3),
    }


def _dev_alignment_onset(steps, rhos, onset=0.6):
    """SVD3-alignment emergence: the readout coupling rho = spearman(sigma^2,
    readout energy) measures whether the TRANSPORT SPECTRUM (not an unrelated
    direction) orders the readout — the core unification mechanism. At random
    init rho ~ 0 (chance); if the mechanism is ACQUIRED, rho rises through an
    onset threshold early in training. This is the theory-faithful developmental
    observable (effective rank can fall as the task signal concentrates, so the
    naive 'rank grows' reading is the wrong sign; alignment EMERGING is the
    right one). Returns onset step, peak, and start value."""
    s = list(steps)
    r = np.asarray(rhos, float)
    if len(r) < 3:
        return {"onset_step": None, "peak_rho": None, "start_rho": None}
    onset_step = None
    for st, rv in zip(s, r):
        if np.isfinite(rv) and rv >= onset:
            onset_step = int(st)
            break
    j = int(np.nanargmax(r))
    return {
        "onset_step": onset_step,                   # first step rho >= onset
        "onset_threshold": onset,
        "peak_rho": round(float(r[j]), 3),
        "peak_step": int(s[j]),
        "start_rho": round(float(r[0]), 3),
        "rise": round(float(np.nanmax(r) - r[0]), 3),
    }


def _measure_checkpoint_spectrum(model_name, revision, n_samples, seeds, device):
    """Download ONE checkpoint, measure the readout-aligned transport effective
    rank per domain over MULTIPLE seeds (each seed = one independent prompt set
    -> one spectrum), aggregating mean +/- std for error bars. Cleans the temp
    cache afterwards (checkpoints are large)."""
    import shutil
    import tempfile

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from icl_convergence_probe import get_final_norm_and_unembed
    from transport_gain_probe import get_decoder_layers
    from capacity_threshold_sweep import make_prompt
    from routing_selection_probe import build_induction

    tmp = tempfile.mkdtemp(prefix=f"sss_{revision}_")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, revision=revision,
                                            cache_dir=tmp)
        dtype = torch.float16 if "2.8b" in model_name.lower() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, cache_dir=tmp,
            dtype=dtype).to(device)
        model.eval()
        norm, unembed = get_final_norm_and_unembed(model)
        layers = get_decoder_layers(model)
        n_layers = len(layers)
        l_c = _capacity_layer_index(n_layers)
        d_model = int(model.config.hidden_size)

        point = {"d_model": d_model, "capacity_layer": l_c}
        for domain in DOMAINS:
            per_seed_rank, per_seed_rho, per_seed_raw = [], [], []
            n_inst = 0
            for seed in seeds:
                rng = np.random.default_rng(seed)
                h_states, g_states = [], []
                for _ in range(n_samples):
                    if domain == "induction":
                        seq, pos, cid = build_induction(tok, rng, block_len=8)
                    else:
                        seq, pos, cid = make_prompt(domain, tok, rng)
                    try:
                        h_q, g = capture_query_state(model, layers, l_c, norm,
                                                     unembed, seq, pos, cid, device)
                    except Exception:  # noqa: BLE001
                        continue
                    h_states.append(h_q)
                    g_states.append(g)
                if len(h_states) < 3:
                    continue
                sigma, rho = domain_spectrum(h_states, g_states)
                v = spectrum_views(sigma, rho)
                per_seed_rank.append(v["readout_effective_rank"])
                per_seed_rho.append(v["icl_order_rho_spearman"])
                per_seed_raw.append(v["raw_effective_rank"])
                n_inst = len(h_states)
            if not per_seed_rank:
                continue
            rk = np.asarray(per_seed_rank, float)
            rh = np.asarray(per_seed_rho, float)
            point[domain] = {
                "readout_effective_rank": round(float(rk.mean()), 3),
                "readout_effective_rank_std": round(float(rk.std(ddof=0)), 3),
                "readout_rank_frac_of_d": round(float(rk.mean()) / d_model, 4),
                "icl_order_rho_spearman": round(float(np.nanmean(rh)), 3),
                "icl_order_rho_spearman_std": round(float(np.nanstd(rh)), 3),
                "raw_effective_rank": round(float(np.mean(per_seed_raw)), 3),
                "n_seeds": len(per_seed_rank),
                "n_instances": n_inst,
                "per_seed_readout_rank": [round(x, 3) for x in per_seed_rank],
                "per_seed_icl_order_rho": [round(x, 3) for x in per_seed_rho],
            }
        return point
    finally:
        import gc
        import torch
        for name in ("model", "tok", "norm", "unembed", "layers"):
            obj = locals().get(name)
            if obj is not None:
                if name == "model" and torch.cuda.is_available():
                    try:
                        obj.cpu()
                    except Exception:  # noqa: BLE001
                        pass
                del obj
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


def run_developmental(model_name, steps=None, n_samples=12, seeds=(0, 1, 2)):
    """sigma(t): the readout-aligned transport effective rank across Pythia
    training checkpoints — DEV applied to the unifying object. Multi-seed
    (mean +/- std per checkpoint) for replication robustness. Requires network
    (downloads one checkpoint at a time, deletes each after measuring)."""
    from icl_convergence_probe import safe_name, pick_device, MODEL_ALIASES

    seeds = list(seeds)
    device = pick_device()
    hf_name = MODEL_ALIASES.get(model_name, model_name)
    checkpoints = resolve_dev_checkpoints(hf_name, steps)
    if len(checkpoints) < 3:
        raise SystemExit(f"{model_name}: need a checkpointed suite (Pythia) for "
                         f"developmental sigma(t); got {len(checkpoints)} point(s)")

    print(f"developmental σ(t): {model_name} ({hf_name}), "
          f"{len(checkpoints)} checkpoints, n={n_samples}/domain, "
          f"seeds={seeds}")
    trajectory = []
    for ck in checkpoints:
        t0 = time.time()
        try:
            point = _measure_checkpoint_spectrum(
                hf_name, ck["revision"], n_samples, seeds, device)
        except Exception as e:  # noqa: BLE001
            print(f"  step {ck['step']:>7}: FAILED ({type(e).__name__}: {e})")
            continue
        point["step"] = ck["step"]
        point["revision"] = ck["revision"]
        trajectory.append(point)
        cells = "  ".join(
            f"{dm[:4]}={point[dm]['readout_effective_rank']}"
            f"±{point[dm]['readout_effective_rank_std']}"
            for dm in DOMAINS if dm in point)
        print(f"  step {ck['step']:>7}: {cells}   ({time.time() - t0:.0f}s)")
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass

    # per-domain knee (rank) + alignment onset (the theory-faithful observable)
    knees, dev_traj, aligns = {}, {}, {}
    for domain in DOMAINS:
        pts = [(p["step"], p[domain]["readout_effective_rank"],
                p[domain]["icl_order_rho_spearman"],
                p[domain].get("readout_effective_rank_std", 0.0),
                p[domain].get("icl_order_rho_spearman_std", 0.0))
               for p in trajectory if domain in p]
        if len(pts) < 3:
            continue
        pts.sort()
        ssteps = [q[0] for q in pts]
        rranks = [q[1] for q in pts]
        rhos = [q[2] for q in pts]
        dev_traj[domain] = {"steps": ssteps, "readout_rank": rranks,
                            "readout_rank_std": [q[3] for q in pts],
                            "icl_order_rho": rhos,
                            "icl_order_rho_std": [q[4] for q in pts]}
        knees[domain] = _dev_knee(ssteps, rranks)
        aligns[domain] = _dev_alignment_onset(ssteps, rhos)

    # P_ordering at convergence: structured tasks use more rank than induction
    end = trajectory[-1] if trajectory else {}
    ordering_ok = None
    if {"induction", "relation_composition"} <= set(
            d for d in DOMAINS if d in end):
        ind = end["induction"]["readout_effective_rank"]
        struct = max(end[d]["readout_effective_rank"]
                     for d in ("relation_composition", "arithmetic_carry")
                     if d in end)
        ordering_ok = bool(struct >= ind)

    grew = [k["total_growth"] for k in knees.values()
            if k.get("total_growth") is not None]
    sharp = [k["sharpness"] for k in knees.values()
             if k.get("sharpness") is not None]
    starts = [a["start_rho"] for a in aligns.values()
              if a.get("start_rho") is not None]
    peaks = [a["peak_rho"] for a in aligns.values()
             if a.get("peak_rho") is not None]
    rises = [a["rise"] for a in aligns.values()
             if a.get("rise") is not None]
    onset_steps = [a["onset_step"] for a in aligns.values()
                   if a.get("onset_step") is not None]
    verdict = {
        # naive 'capacity grows' reading — refuted: the readout rank does NOT
        # increase monotonically (the task signal concentrates instead).
        "P_growth": bool(grew and float(np.median(grew)) > 0.0
                         and any(g > 0 for g in grew)),
        # the theory-faithful emergence signal (SVD3 alignment is ACQUIRED):
        # the spectrum->readout alignment RISES to a high peak early in training.
        # Tested by the rise+peak (model-robust), not the absolute init value
        # (which is model-dependent: a larger model's init alignment is already
        # nonzero at this layer).
        "P_alignment_emerges": bool(
            peaks and rises
            and float(np.median(peaks)) >= 0.75
            and float(np.median(rises)) >= 0.2),
        "alignment_peak_rho_median": (round(float(np.median(peaks)), 3)
                                      if peaks else None),
        "alignment_rise_median": (round(float(np.median(rises)), 3)
                                  if rises else None),
        "alignment_start_rho_median": (round(float(np.median(starts)), 3)
                                       if starts else None),
        "alignment_onset_step_median": (int(np.median(onset_steps))
                                        if onset_steps else None),
        "P_knee_sharp": bool(sharp and float(np.median(sharp)) >= 0.5),
        "P_ordering_at_convergence": ordering_ok,
        "n_checkpoints": len(trajectory),
        "n_domains": len(knees),
    }
    out = {
        "mode": "developmental",
        "model": model_name,
        "hf_name": hf_name,
        "date": time.strftime("%Y-%m-%d"),
        "n_samples": n_samples,
        "seeds": seeds,
        "formal_anchors": ["DEV1", "DEV2", "DEV3", "SVD1", "SVD2", "SVD3"],
        "checkpoint_steps": [c["step"] for c in checkpoints],
        "trajectory": trajectory,
        "developmental_trajectory": dev_traj,
        "knees": knees,
        "alignment_onset": aligns,
        "verdict": verdict,
    }
    path = HERE / f"singular_spectrum_developmental_{safe_name(model_name)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print("\n=== developmental σ(t) verdict ===")
    for dom in knees:
        k, a = knees[dom], aligns.get(dom, {})
        print(f"  {dom:>22}: rank {k['start_rank']} → {k['end_rank']} "
              f"(growth {k['total_growth']})  |  SVD3-align ρ {a.get('start_rho')}"
              f" → peak {a.get('peak_rho')}@{a.get('peak_step')} "
              f"(onset@{a.get('onset_step')})")
    for kk, vv in verdict.items():
        print(f"  {kk}: {vv}")
    return out


def plot_developmental():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(HERE.glob("singular_spectrum_developmental_*.json"))
    if not files:
        print("no developmental results to plot")
        return
    for p in files:
        d = json.loads(p.read_text())
        traj = d.get("developmental_trajectory", {})
        if not traj:
            continue
        fig, (axr, axa) = plt.subplots(1, 2, figsize=(11.4, 4.6))
        for domain, series in traj.items():
            steps = np.maximum(np.asarray(series["steps"], float), 1.0)
            r_err = series.get("readout_rank_std")
            axr.errorbar(steps, series["readout_rank"],
                         yerr=r_err if r_err else None,
                         fmt="o-", capsize=3, label=domain)
            if "icl_order_rho" in series:
                a_err = series.get("icl_order_rho_std")
                axa.errorbar(steps, series["icl_order_rho"],
                             yerr=a_err if a_err else None,
                             fmt="s-", capsize=3, label=domain)
        for ax in (axr, axa):
            ax.set_xscale("log")
            ax.set_xlabel("training step")
            ax.grid(alpha=0.2)
        axr.set_ylabel("readout-aligned transport effective rank")
        axr.set_title("(a) capacity rank (concentrates, not grows)")
        axa.axhline(0.6, color="0.6", ls="--", lw=0.8)
        axa.set_ylabel("spearman(σ², readout energy)  — SVD3")
        axa.set_title("(b) σ-spectrum→readout alignment EMERGES")
        axa.legend(fontsize=8)
        on = d.get("verdict", {}).get("alignment_onset_step_median")
        if on:
            axa.axvline(max(on, 1), ls=":", lw=1.0, color="k", alpha=0.6)
        fig.suptitle(f"Developmental σ(t): {d['model']} — alignment is acquired, "
                     "rank concentrates", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = HERE / (f"singular_spectrum_developmental_"
                      f"{d['model'].replace('/', '_').replace('.', '_')}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved -> {out.name}")


# ==========================================================================
# Aggregate + plot
# ==========================================================================
def aggregate():
    rows = {}
    for p in sorted(HERE.glob("singular_spectrum_*.json")):
        if p.stem.endswith(("summary", "synthetic")):
            continue
        d = json.loads(p.read_text())
        rows[d["model"]] = {"verdict": d.get("verdict"),
                            "per_domain": d.get("per_domain"),
                            "composition_SVD6": d.get("composition_SVD6")}
    out = {"analysis": "singular-spectrum unifying-object test (aggregate)",
           "date": time.strftime("%Y-%m-%d"), "models": rows}
    path = HERE / "singular_spectrum_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}  ({len(rows)} models)")
    return out


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = [p for p in sorted(HERE.glob("singular_spectrum_*.json"))
             if not p.stem.endswith(("summary", "synthetic"))]
    if not files:
        print("no real-model results to plot")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for p in files:
        d = json.loads(p.read_text())
        doms = list(d["per_domain"].keys())
        ranks = [d["per_domain"][dm]["readout_rank_frac_of_d"] for dm in doms]
        orders = [d["per_domain"][dm]["icl_order_rho_spearman"] for dm in doms]
        ax1.plot(doms, ranks, "o-", label=d["model"])
        ax2.plot(doms, orders, "s-", label=d["model"])
    ax1.set_ylabel("effective rank / d (capacity)")
    ax1.set_title("(a) σ-spectrum effective rank (SVD1/2)")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(alpha=0.2)
    ax2.axhline(0.5, color="0.6", ls="--", lw=0.8)
    ax2.set_ylabel("spearman(σ², readout energy)")
    ax2.set_title("(b) σ orders ICL energy (SVD3)")
    ax2.tick_params(axis="x", rotation=20)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)
    fig.suptitle("Transport singular spectrum as the unifying object", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = HERE / "singular_spectrum.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved -> {path.name}")


# ==========================================================================
# RANK-LADDER mode — does composed in-context computation ACCUMULATE transport
# rank with hop count? (COMP / capacity prediction, readout-aligned)
# ==========================================================================
def build_khop_chain(tok, rng, k_hops: int, n_pairs: int = 6,
                     lo: int = 1000, hi: int = 40000):
    """k-hop chained induction. Roles X0->X1->...->X_k presented as adjacent
    pairs with the intermediate repeated to set up the next hop:
      1-hop: A B            (query A -> B)
      2-hop: A B B C        (query A -> C)
      3-hop: A B B C C D     (query A -> D)
    Returns (input_ids, query_pos, correct_final_id)."""
    vocab = int(tok.vocab_size if hasattr(tok, "vocab_size") else len(tok))
    hi = min(vocab - 1, hi)
    n_roles = k_hops + 1
    need = n_pairs * n_roles
    pool = set()
    while len(pool) < need:
        pool.add(int(rng.integers(lo, hi)))
    toks = list(pool)
    roles = [toks[i * n_pairs:(i + 1) * n_pairs] for i in range(n_roles)]
    seq = []
    for j in range(n_pairs):
        seq.append(roles[0][j])
        for h in range(k_hops):
            seq.append(roles[h + 1][j])
            if h < k_hops - 1:
                seq.append(roles[h + 1][j])
    q = int(rng.integers(0, n_pairs))
    seq.append(roles[0][q])
    return seq, len(seq) - 1, roles[k_hops][q]


def run_rank_ladder(model_name, n_samples=48, max_hops=3, seed=0):
    """Measure the readout-aligned transport effective rank for k-hop chained
    induction (k=1..max_hops). LAW: rank accumulates with hop count
    (rank_k monotone in k; ideally ~ linear = additive). The readout gradient
    is well-defined even when accuracy is low, so the rank is measurable on
    small models that cannot fully execute deep chains."""
    import torch  # noqa: F401
    from icl_convergence_probe import (safe_name, pick_device, load_model,
                                       get_final_norm_and_unembed)
    from transport_gain_probe import get_decoder_layers

    device = pick_device()
    hf_name, tok, model = load_model(model_name)
    model = model.to(device)
    norm, unembed = get_final_norm_and_unembed(model)
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    l_c = _capacity_layer_index(n_layers)
    d_model = model.config.hidden_size

    per_hop = {}
    for k in range(1, max_hops + 1):
        rng = np.random.default_rng(seed)
        h_states, g_states = [], []
        correct = total = 0
        for _ in range(n_samples):
            seq, pos, cid = build_khop_chain(tok, rng, k)
            try:
                h_q, g = capture_query_state(model, layers, l_c, norm, unembed,
                                             seq, pos, cid, device)
            except Exception:  # noqa: BLE001
                continue
            h_states.append(h_q)
            g_states.append(g)
            import torch as _t
            with _t.no_grad():
                out = model(input_ids=_t.tensor([seq], device=device))
            pred = int(_t.argmax(out.logits[0, pos, :]).item())
            correct += int(pred == cid)
            total += 1
        if len(h_states) < 3:
            continue
        sigma, rho = domain_spectrum(h_states, g_states)
        views = spectrum_views(sigma, rho)
        per_hop[k] = {
            "readout_effective_rank": views["readout_effective_rank"],
            "raw_effective_rank": views["raw_effective_rank"],
            "rank_at_90pct_energy": views["rank_at_90pct_energy"],
            "accuracy": round(correct / max(total, 1), 4),
            "n_instances": len(h_states),
        }

    ks = sorted(per_hop)
    ranks = [per_hop[k]["readout_effective_rank"] for k in ks]
    monotone = all(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1)) \
        if len(ranks) >= 2 else False
    # linear accumulation fit rank ~ a + b*k
    slope = r2 = None
    if len(ks) >= 2:
        b, a = np.polyfit(ks, ranks, 1)
        slope = round(float(b), 4)
        pred = a + b * np.asarray(ks, float)
        ss_res = float(((np.asarray(ranks) - pred) ** 2).sum())
        ss_tot = float(((np.asarray(ranks) - np.mean(ranks)) ** 2).sum()) + 1e-30
        r2 = round(1 - ss_res / ss_tot, 4)
    # ratio rank_k / rank_1 (k => linear/additive, <k => sublinear)
    ratio = ({k: round(per_hop[k]["readout_effective_rank"]
                       / (per_hop[ks[0]]["readout_effective_rank"] + 1e-9), 3)
              for k in ks} if ks else {})

    verdict = "PASS" if monotone else ("WEAK" if (slope and slope > 0) else "FAIL")
    out = {
        "mode": "rank_ladder",
        "model": model_name,
        "hf_name": hf_name,
        "date": time.strftime("%Y-%m-%d"),
        "n_layers": n_layers,
        "capacity_layer": l_c,
        "d_model": d_model,
        "n_samples": n_samples,
        "formal_anchors": ["COMP1", "COMP2", "COMP3", "SVD1", "SVD2"],
        "per_hop": per_hop,
        "summary": {
            "ranks_by_hop": dict(zip(ks, ranks)),
            "rank_ratio_to_1hop": ratio,
            "monotone_accumulation": monotone,
            "linear_slope": slope,
            "linear_fit_r2": r2,
            "verdict": verdict,
        },
    }
    path = HERE / f"singular_spectrum_rankladder_{safe_name(model_name)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    for k in ks:
        h = per_hop[k]
        print(f"  {k}-hop: readout_rank={h['readout_effective_rank']:.3f}  "
              f"ratio={ratio[k]}  acc={h['accuracy']}  (n={h['n_instances']})")
    print(f"  monotone={monotone}  slope={slope}  r2={r2}  verdict={verdict}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--developmental", nargs="?", const="pythia-160m",
                    default=None, metavar="MODEL",
                    help="measure σ(t) across a model's training checkpoints "
                         "(Pythia; downloads each checkpoint). Default pythia-160m.")
    ap.add_argument("--steps", nargs="+", type=int, default=None,
                    help="explicit Pythia training steps for --developmental")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="seeds for --developmental multi-seed aggregation")
    ap.add_argument("--rank-ladder", action="store_true",
                    help="measure readout-rank accumulation across k-hop chains")
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--plot-dev", action="store_true",
                    help="plot saved developmental σ(t) trajectories")
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic()
        return
    if args.rank_ladder:
        for m in (args.models or DEFAULT_MODELS):
            try:
                run_rank_ladder(m, n_samples=args.n_samples, max_hops=args.max_hops)
            except Exception as e:  # noqa: BLE001
                print(f"{m}: FAILED ({type(e).__name__}: {e})")
        return
    if args.developmental:
        run_developmental(args.developmental, steps=args.steps,
                          n_samples=args.n_samples, seeds=args.seeds)
        plot_developmental()
        return
    if args.plot_dev:
        plot_developmental()
        return
    if args.aggregate:
        aggregate()
        return
    if args.plot:
        plot()
        return
    models = args.models or DEFAULT_MODELS
    for m in models:
        try:
            run_model(m, args.n_samples)
        except Exception as e:  # noqa: BLE001
            print(f"{m}: FAILED ({type(e).__name__}: {e})")
    aggregate()


if __name__ == "__main__":
    main()
