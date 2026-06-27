#!/usr/bin/env python3
"""Structure-vs-randomness transport probe — does the trained transport operator
admit an EXACT STRUCTURE + RANDOMNESS split (a Szemeredi/RMT-flavored dichotomy)?

THE ALTERNATIVE EXPLANATION (Tao-flavored, exact, falsifiable)
  The capacity/transport program DESCRIBES the trained operator as a low-rank,
  readout-aligned object. The free-probability sibling (free_probability_transport)
  asked WHERE the spectrum comes from (a free multiplicative convolution) and
  found the SHAPE is free but the RANK is not. This probe asks a different,
  borrowed-from-deep-math question, the STRUCTURE-vs-RANDOMNESS dichotomy that
  underlies the Szemeredi regularity lemma and the Tao-Green decomposition:

    Does the effective transport operator T at the capacity layer split EXACTLY
    into a STRUCTURED part S and a RANDOM part R,

        T = S + R,

    where S is LOW-RANK and carries (almost) all the READOUT energy (the task —
    the "structured complexity"), and R is a PSEUDORANDOM bulk whose singular
    spectrum matches a RANDOM-MATRIX law (Marchenko-Pastur for the operator's
    aspect ratio), i.e. R is "structureless"?

  This is the spiked-covariance reading of trained computation: SPIKES = the task
  (structure), MP BULK = noise (randomness). It is the natural complement of the
  free-probability finding — there the open residual was the rank concentration;
  here we test directly whether that concentration is an exact low-rank-spikes /
  random-bulk separation.

THE OPERATOR (real arm)
  At the mid-stack capacity layer L_c, capture per prompt the query-position
  hidden state h and the readout functional g = d logit_correct / d h[query]
  (reuses cross_layer_transport_probe.capture_all_layers — one forward + one
  backward yields every layer; we read L_c). Stacking N prompts gives the
  centered query-state matrix H_c in R^{N x d} — the same transport object the
  singular_spectrum probe measures. Its SVD H_c = U diag(sigma) V^T is the
  transport spectrum; readout coupling rho_i = RMS_p |<g_p, V_i>| and per-direction
  readout energy e_i = sigma_i^2 rho_i^2 (the OPB multiplication law) define the
  TASK. Honest caveat: H_c is the data sample at L_c, so the spectrum is the
  EFFECTIVE transport spectrum (same spirit as the sibling transport probes), and
  on small local-cache models N is small => the bulk aspect ratio c = d/N is large
  and the MP test is underpowered (an honest regime bound, not a defect).

TWO SCALARS (the falsifiable readings)
  structured_complexity = smallest rank r such that the top-r singular directions
        carry >= 90% of the READOUT energy (ordered by singular value). Small r
        relative to min(N,d) => the task is genuinely low-rank (structured).
  bulk_mp_ks            = Kolmogorov-Smirnov distance of the RESIDUAL spectrum
        (the bulk eigenvalues lam_i = sigma_i^2/N after removing the spikes that
        stick out above the MP edge) to the best-fit Marchenko-Pastur law (aspect
        c = d/N known; noise variance sigma^2 fit from the bulk mean). Small KS
        => the bulk is structureless (a random matrix).

  STRUCTURED  : structured_complexity LOW, readout energy concentrates in the
                spikes (energy_in_spikes high), AND bulk passes MP (KS small)
                -> a clean structure/randomness split; "structured complexity"
                = rank(S) = the number of MP spikes.
  MIXED       : readout energy spread into the bulk (structured_complexity high),
                OR the bulk fails the MP fit (KS large) -> no clean split.
  UNDERPOWERED: too few bulk modes / degenerate fit (the honest small-N corp case).

Local cache only (no training, no download). Run:
  .venv/bin/python structure_randomness_transport_probe.py --synthetic
  .venv/bin/python structure_randomness_transport_probe.py --models gpt2 pythia-160m
  .venv/bin/python structure_randomness_transport_probe.py --aggregate
Output: structure_randomness_transport_<model>.json (+ _summary.json, _synthetic.json)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from singular_spectrum_probe import (  # noqa: E402
    DOMAINS,
    domain_spectrum,
    effective_rank,
    _capacity_layer_index,
)

DEFAULT_MODELS = ["gpt2", "pythia-160m"]
DEFAULT_N = 32              # prompt instances per domain (modest, corp CPU)
ENERGY_FRAC = 0.90          # readout-energy fraction defining structured_complexity
KS_PASS = 0.20             # bulk_mp_ks <= this => bulk looks Marchenko-Pastur
ENERGY_IN_S_PASS = 0.70    # spike directions must carry this much readout energy
LOWRANK_FRAC = 0.25        # structured_complexity <= LOWRANK_FRAC*min(N,d) => low-rank
MIN_BULK_MODES = 20         # fewer bulk modes than this => UNDERPOWERED


# ---------------------------------------------------------------------------
# Marchenko-Pastur machinery (numpy only) — the RANDOMNESS reference
# ---------------------------------------------------------------------------
def _mp_edges(c, sigma2):
    """Support edges of the MP eigenvalue law for aspect ratio c and noise var
    sigma2: [sigma2 (1-sqrt c)^2, sigma2 (1+sqrt c)^2]."""
    sq = np.sqrt(c)
    return sigma2 * (1.0 - sq) ** 2, sigma2 * (1.0 + sq) ** 2


def _mp_cdf_grid(c, sigma2, m=4096):
    """Numerically integrated CDF of the MP CONTINUOUS part (renormalized to mass
    1 over [a,b]), so it can be compared to the empirical bulk eigenvalue CDF for
    any c (the c>1 zero-mass is excluded — we test only the nonzero eigenvalues)."""
    a, b = _mp_edges(c, sigma2)
    a = max(a, 1e-12)
    if b <= a:
        return None, None
    xs = np.linspace(a, b, m)
    # MP continuous density shape sqrt((b-x)(x-a)) / x (constants drop on renorm).
    dens = np.sqrt(np.clip((b - xs) * (xs - a), 0.0, None)) / xs
    area = float(np.trapezoid(dens, xs))
    if area <= 0:
        return None, None
    dens = dens / area
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(xs))])
    cdf = cdf / cdf[-1]
    return xs, cdf


def fit_mp_bulk(eigs, c, max_iter=12):
    """Fit the MP law to the bulk of a nonzero-eigenvalue sample: iteratively
    estimate the noise variance sigma2 from the bulk mean (the nonzero eigenvalues
    of an iid matrix have mean sigma2*max(c,1)) and peel off SPIKES that exceed the
    upper edge sigma2 (1+sqrt c)^2. Returns (sigma2, bulk_eigs, n_spikes, (a,b))."""
    eigs = np.sort(np.asarray(eigs, float))[::-1]
    if eigs.size == 0:
        return float("nan"), eigs, 0, (float("nan"), float("nan"))
    norm = max(c, 1.0)
    sigma2 = float(eigs.mean()) / norm
    bulk = eigs
    a, b = _mp_edges(c, sigma2)
    for _ in range(max_iter):
        a, b = _mp_edges(c, sigma2)
        new_bulk = eigs[eigs <= b]
        if new_bulk.size < 2:
            break
        sigma2_new = float(new_bulk.mean()) / norm
        bulk = new_bulk
        if abs(sigma2_new - sigma2) <= 1e-9 * max(sigma2, 1e-12):
            sigma2 = sigma2_new
            break
        sigma2 = sigma2_new
    a, b = _mp_edges(c, sigma2)
    n_spikes = int((eigs > b).sum())
    return sigma2, bulk, n_spikes, (a, b)


def mp_ks_distance(bulk_eigs, c, sigma2):
    """Two-sided KS distance between the empirical bulk eigenvalue CDF and the
    best-fit MP continuous CDF."""
    bulk = np.sort(np.asarray(bulk_eigs, float))
    n = bulk.size
    if n < 5:
        return float("nan")
    xs, cdf = _mp_cdf_grid(c, sigma2)
    if xs is None:
        return float("nan")
    fmp = np.interp(bulk, xs, cdf, left=0.0, right=1.0)
    f_hi = np.arange(1, n + 1) / n
    f_lo = np.arange(0, n) / n
    return float(np.max(np.maximum(np.abs(f_hi - fmp), np.abs(f_lo - fmp))))


# ---------------------------------------------------------------------------
# The structure + randomness decomposition (the core measurement)
# ---------------------------------------------------------------------------
def _verdict(structured_complexity, energy_in_spikes, bulk_mp_ks,
             n_bulk_modes, n_spikes, min_nd):
    if n_bulk_modes < MIN_BULK_MODES or not np.isfinite(bulk_mp_ks):
        return "UNDERPOWERED"
    low_rank = structured_complexity <= max(2, LOWRANK_FRAC * min_nd)
    mp_ok = bulk_mp_ks <= KS_PASS
    task_in_spikes = (n_spikes >= 1) and (energy_in_spikes >= ENERGY_IN_S_PASS)
    if low_rank and mp_ok and task_in_spikes:
        return "STRUCTURED"
    return "MIXED"


def structure_randomness_split(sigma, rho, n_samples, d_model,
                               energy_frac=ENERGY_FRAC):
    """Decompose the transport spectrum (sigma, readout coupling rho) into a
    low-rank readout-aligned STRUCTURE S and a Marchenko-Pastur RANDOM bulk R.

      structured_complexity : smallest r s.t. top-r (by sigma) carries >= 90% of
                              the readout energy e = sigma^2 rho^2.
      MP bulk               : eigenvalues lam = sigma^2 / N, aspect c = d/N; spikes
                              peeled above the MP edge, sigma2 fit from the bulk.
      energy_in_spikes      : readout-energy fraction carried by the spike dirs
                              (the cross-check that the RMT structure IS the task).
      bulk_mp_ks            : KS(residual bulk eigenvalues, best-fit MP)."""
    sigma = np.asarray(sigma, float)
    rho = np.asarray(rho, float)
    order = np.argsort(-sigma)
    sigma, rho = sigma[order], rho[order]
    e = (sigma ** 2) * (rho ** 2)
    e_tot = float(e.sum()) + 1e-30
    cum = np.cumsum(e) / e_tot
    structured_complexity = int(np.searchsorted(cum, energy_frac) + 1)
    structured_complexity = max(1, min(structured_complexity, len(sigma)))

    c = float(d_model) / float(n_samples)
    eigs = (sigma ** 2) / float(n_samples)
    smax = float(eigs.max()) if eigs.size else 0.0
    eigs = eigs[eigs > 1e-12 * smax] if smax > 0 else eigs   # drop numerical zeros
    sigma2, bulk, n_spikes, (a, b) = fit_mp_bulk(eigs, c)
    bulk_mp_ks = mp_ks_distance(bulk, c, sigma2)

    n_spk = max(int(n_spikes), 0)
    # spikes are the top-by-eigenvalue dirs == top-by-sigma dirs (same order)
    energy_in_spikes = float(e[:n_spk].sum() / e_tot) if n_spk >= 1 else 0.0
    energy_in_struct = float(cum[structured_complexity - 1])

    min_nd = min(int(n_samples) - 1, int(d_model))
    verdict = _verdict(structured_complexity, energy_in_spikes, bulk_mp_ks,
                       bulk.size, n_spikes, min_nd)
    return {
        "structured_complexity": int(structured_complexity),
        "bulk_mp_ks": round(float(bulk_mp_ks), 4) if np.isfinite(bulk_mp_ks) else None,
        "n_spikes": int(n_spikes),
        "energy_in_spikes": round(energy_in_spikes, 4),
        "energy_in_structured_cut": round(energy_in_struct, 4),
        "mp_sigma2": round(float(sigma2), 6) if np.isfinite(sigma2) else None,
        "mp_edge_lo": round(float(a), 6) if np.isfinite(a) else None,
        "mp_edge_hi": round(float(b), 6) if np.isfinite(b) else None,
        "aspect_ratio_c": round(c, 4),
        "n_bulk_modes": int(bulk.size),
        "n_eigs": int(eigs.size),
        "raw_effective_rank": round(float(effective_rank(sigma)), 3),
        "readout_effective_rank": round(float(effective_rank(sigma * rho)), 3),
        "structured_rank_frac_of_min": round(structured_complexity / max(min_nd, 1), 4),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Real-model arm
# ---------------------------------------------------------------------------
def run_model(model_name, n_samples=DEFAULT_N, seed=0):
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
    n_layers = len(layers)
    l_c = _capacity_layer_index(n_layers)
    d_model = int(model.config.hidden_size)

    per_domain = {}
    for domain in DOMAINS:
        rng = np.random.default_rng(seed)
        h_states, g_states = [], []
        for _ in range(n_samples):
            if domain == "induction":
                seq, pos, cid = build_induction(tok, rng, block_len=8)
            else:
                seq, pos, cid = make_prompt(domain, tok, rng)
            try:
                caps = capture_all_layers(model, layers, norm, unembed,
                                          seq, pos, cid, device)
            except Exception as e:  # noqa: BLE001
                print(f"  {domain}: prompt failed ({type(e).__name__}: {e})")
                continue
            # capacity layer (or the nearest captured layer) — one operator at L_c
            if l_c in caps:
                hq, g = caps[l_c]
            elif caps:
                near = min(caps, key=lambda i: abs(i - l_c))
                hq, g = caps[near]
            else:
                continue
            h_states.append(hq)
            g_states.append(g)
        if len(h_states) < 5:
            print(f"  {domain}: too few usable prompts — skipped")
            continue
        sigma, rho = domain_spectrum(h_states, g_states)
        res = structure_randomness_split(sigma, rho, len(h_states), d_model)
        res["n_instances"] = len(h_states)
        per_domain[domain] = res

    verdicts = [a["verdict"] for a in per_domain.values()]
    clean = sum(1 for v in verdicts if v == "STRUCTURED")
    model_verdict = ("STRUCTURED" if clean and clean == len(verdicts)
                     else "MIXED" if any(v in ("STRUCTURED", "MIXED") for v in verdicts)
                     else "UNDERPOWERED")
    out = {
        "mode": "real_model",
        "model": model_name,
        "hf_name": hf_name,
        "date": time.strftime("%Y-%m-%d"),
        "n_layers": n_layers,
        "capacity_layer": l_c,
        "d_model": d_model,
        "n_samples": n_samples,
        "method": "structure+randomness split of the capacity-layer query-state "
                  "matrix H_c (N x d): low-rank readout-aligned spikes S vs "
                  "Marchenko-Pastur bulk R",
        "claim": "T = S(low-rank, readout-aligned) + R(MP-random bulk)? "
                 "clean split iff structured_complexity low AND bulk passes MP",
        "thresholds": {"energy_frac": ENERGY_FRAC, "ks_pass": KS_PASS,
                       "energy_in_s_pass": ENERGY_IN_S_PASS,
                       "lowrank_frac": LOWRANK_FRAC, "min_bulk_modes": MIN_BULK_MODES},
        "per_domain": per_domain,
        "model_verdict": model_verdict,
        "n_domains_structured": clean,
    }
    path = HERE / f"structure_randomness_transport_{safe_name(model_name)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    for dom, a in per_domain.items():
        ks = a["bulk_mp_ks"]
        print(f"  {dom:>22}: struct_complexity={a['structured_complexity']} "
              f"(spikes={a['n_spikes']}, frac={a['structured_rank_frac_of_min']}) "
              f"bulk_mp_ks={ks} energy_in_spikes={a['energy_in_spikes']} "
              f"(c={a['aspect_ratio_c']}, bulk_modes={a['n_bulk_modes']}) -> {a['verdict']}")
    print(f"  model verdict = {model_verdict}  ({clean}/{len(per_domain)} structured)")
    return out


# ---------------------------------------------------------------------------
# Synthetic validation (numpy only) — planted T = S + R
# ---------------------------------------------------------------------------
def _planted_case(rng, N, d, r0, a_struct, bulk="gaussian", sval_bad=None):
    """Build H = S + R and a readout matrix G coupling to S's directions.
      S : rank-r0, right dirs u_k, left scores ell_k, singular values a_struct.
      R : 'gaussian' iid N(0,1) bulk (MP) | 'flat' all-equal singular values
          (a delta spectrum, blatantly non-MP) | 'none' (no bulk, pure noise floor).
    Returns (H, G)."""
    Vd, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Ln, _ = np.linalg.qr(rng.standard_normal((N, N)))
    u = Vd[:, :r0]                                   # d x r0 right dirs
    ell = Ln[:, :r0]                                 # N x r0 left scores
    S_mat = (ell * np.asarray(a_struct, float)) @ u.T  # N x d, rank r0
    if bulk == "gaussian":
        R_mat = rng.standard_normal((N, d))
    elif bulk == "flat":
        m = min(N, d)
        Ub, _ = np.linalg.qr(rng.standard_normal((N, N)))
        Vb, _ = np.linalg.qr(rng.standard_normal((d, d)))
        R_mat = (Ub[:, :m] * float(sval_bad)) @ Vb[:, :m].T   # all svals == sval_bad
    elif bulk == "none":
        R_mat = 0.05 * rng.standard_normal((N, d))
    else:
        raise ValueError(bulk)
    H = S_mat + R_mat
    # readout couples to the structured directions (prompt-dependent weights)
    C = rng.uniform(0.5, 1.5, (N, r0))
    G = C @ u.T + 0.01 * rng.standard_normal((N, d))
    return H, G


def run_synthetic():
    """Plant T = S + R with a KNOWN structured rank and a KNOWN bulk law, and
    verify the probe (a) recovers rank(S), (b) finds the readout energy in S,
    (c) the Gaussian bulk passes MP, (d) a NON-MP planted bulk is rejected, and
    (e) a no-structure spread case reads MIXED. Thresholds are pre-registered
    (module constants); they are NOT tuned to force a pass."""
    rng = np.random.default_rng(0)
    N, d, r0 = 200, 400, 3
    a_struct = [100.0, 80.0, 60.0]               # spike singular values >> MP edge

    # (1) clean structure + Gaussian (MP) bulk
    Hc, Gc = _planted_case(rng, N, d, r0, a_struct, bulk="gaussian")
    sig_c, rho_c = domain_spectrum(Hc, Gc)
    clean = structure_randomness_split(sig_c, rho_c, N, d)

    # (2) structure + NON-MP (flat / delta-spectrum) bulk
    Hb, Gb = _planted_case(rng, N, d, r0, a_struct, bulk="flat", sval_bad=20.0)
    sig_b, rho_b = domain_spectrum(Hb, Gb)
    bad = structure_randomness_split(sig_b, rho_b, N, d)

    # (3) no structure, readout spread (random) over a Gaussian bulk
    Hs = rng.standard_normal((N, d))
    Gs = rng.standard_normal((N, d))             # readout couples to everything
    sig_s, rho_s = domain_spectrum(Hs, Gs)
    spread = structure_randomness_split(sig_s, rho_s, N, d)

    min_nd = min(N - 1, d)
    checks = [
        ("recovers structured rank (struct_complexity in [r0, r0+2])",
         r0 <= clean["structured_complexity"] <= r0 + 2),
        ("RMT recovers planted rank (|n_spikes - r0| <= 1)",
         abs(clean["n_spikes"] - r0) <= 1),
        ("readout energy concentrates in S (energy_in_spikes >= 0.9)",
         clean["energy_in_spikes"] >= 0.90),
        ("Gaussian bulk passes MP (bulk_mp_ks < 0.15)",
         clean["bulk_mp_ks"] is not None and clean["bulk_mp_ks"] < 0.15),
        ("MP fit recovers noise variance (0.7 < sigma2 < 1.3)",
         clean["mp_sigma2"] is not None and 0.7 < clean["mp_sigma2"] < 1.3),
        ("clean case verdict == STRUCTURED", clean["verdict"] == "STRUCTURED"),
        ("NON-MP bulk is rejected (ks_bad > 0.3 and > 2x ks_good)",
         bad["bulk_mp_ks"] is not None and clean["bulk_mp_ks"] is not None
         and bad["bulk_mp_ks"] > 0.30
         and bad["bulk_mp_ks"] > 2.0 * clean["bulk_mp_ks"]),
        ("NON-MP bulk verdict == MIXED", bad["verdict"] == "MIXED"),
        ("no-structure spread reads MIXED (not low-rank)",
         spread["verdict"] == "MIXED"
         and spread["structured_complexity"] > LOWRANK_FRAC * min_nd),
    ]
    n_pass = sum(1 for _, ok in checks if ok)
    all_pass = (n_pass == len(checks))
    print(f"Synthetic: {n_pass}/{len(checks)} checks pass")
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
    print(f"  clean : struct_complexity={clean['structured_complexity']} "
          f"spikes={clean['n_spikes']} ks={clean['bulk_mp_ks']} "
          f"sigma2={clean['mp_sigma2']} energy_in_spikes={clean['energy_in_spikes']} "
          f"-> {clean['verdict']}")
    print(f"  non-MP: struct_complexity={bad['structured_complexity']} "
          f"spikes={bad['n_spikes']} ks={bad['bulk_mp_ks']} -> {bad['verdict']}")
    print(f"  spread: struct_complexity={spread['structured_complexity']} "
          f"spikes={spread['n_spikes']} ks={spread['bulk_mp_ks']} -> {spread['verdict']}")
    out = {
        "mode": "synthetic",
        "date": time.strftime("%Y-%m-%d"),
        "planted": {"N": N, "d": d, "r0": r0, "a_struct": a_struct,
                    "bulk_noise_sigma": 1.0},
        "cases": {"clean_gaussian_bulk": clean, "nonmp_flat_bulk": bad,
                  "no_structure_spread": spread},
        "synthetic_checks": [{"test": d_, "pass": bool(ok)} for d_, ok in checks],
        "pass_rate": f"{n_pass}/{len(checks)}",
        "all_pass": all_pass,
    }
    path = HERE / "structure_randomness_transport_synthetic.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    return out, all_pass


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def aggregate():
    rows = {}
    for p in sorted(HERE.glob("structure_randomness_transport_*.json")):
        if p.stem.endswith(("summary", "synthetic")):
            continue
        d = json.loads(p.read_text())
        rows[d["model"]] = {
            "model_verdict": d.get("model_verdict"),
            "n_domains_structured": d.get("n_domains_structured"),
            "per_domain": {dom: {"structured_complexity": a["structured_complexity"],
                                 "bulk_mp_ks": a["bulk_mp_ks"],
                                 "n_spikes": a["n_spikes"],
                                 "energy_in_spikes": a["energy_in_spikes"],
                                 "aspect_ratio_c": a["aspect_ratio_c"],
                                 "n_bulk_modes": a["n_bulk_modes"],
                                 "verdict": a["verdict"]}
                           for dom, a in d.get("per_domain", {}).items()},
        }
    out = {"analysis": "structure-vs-randomness transport split (aggregate)",
           "claim": "does trained transport split into low-rank task spikes + MP bulk?",
           "date": time.strftime("%Y-%m-%d"), "models": rows}
    path = HERE / "structure_randomness_transport_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}  ({len(rows)} models)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Structure-vs-randomness transport split")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-samples", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        _, all_pass = run_synthetic()
        sys.exit(0 if all_pass else 1)
    if args.aggregate:
        aggregate()
        return
    for m in (args.models or DEFAULT_MODELS):
        print(f"\n{'=' * 60}\n  {m}  (structure-vs-randomness transport)\n{'=' * 60}")
        t0 = time.time()
        try:
            run_model(m, n_samples=args.n_samples, seed=args.seed)
        except Exception as e:  # noqa: BLE001
            print(f"{m}: FAILED ({type(e).__name__}: {e})")
        print(f"  ({time.time() - t0:.1f}s)")
    aggregate()


if __name__ == "__main__":
    main()
