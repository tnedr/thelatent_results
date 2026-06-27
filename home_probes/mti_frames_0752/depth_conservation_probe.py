#!/usr/bin/env python3
"""Depth-conservation probe — is there a CONSERVATION LAW ALONG DEPTH (a Noether
theorem for the moving frame)?

THE ALTERNATIVE EXPLANATION (Tao-flavored, exact, falsifiable)
  The capacity/transport program DESCRIBES the trained per-layer operator (low-rank
  readout-aligned spectrum, a slowly-rotating frame through depth — the L3
  MOVING_FRAME finding). The free-probability probe asks whether the depth product
  composes FREELY (Haar position). This probe asks a complementary, deeper question:

    Does the per-layer transport have a DEPTH-INVARIANT integral of motion — a
    scalar Q computed from each layer's effective transport T_i that stays
    APPROXIMATELY CONSTANT across depth (an emergent conservation law)?

  If such a Q exists, the trained computation is a genuine GEOMETRIC FLOW: the
  layers are not 23 unrelated maps but discrete steps of one flow that conserves
  Q, exactly as a Hamiltonian flow conserves energy (Noether). This is the
  natural depth-direction companion of the moving-frame geometry already found.

WHAT IS FIT (real arm — identical to free_probability_transport_probe)
  Reuse cross_layer_transport_probe.capture_all_layers (one forward + one backward
  per prompt yields every layer's query hidden state). Project all layers onto a
  shared top-r PCA basis B (r=32), then fit the EFFECTIVE per-layer linear
  transport T_i = argmin ||X_{i+1} - X_i T_i^T|| in those coordinates (the standard
  data-linearized transport object; reuses _shared_basis + _fit_transport). The
  ordered sequence T_0, T_1, ... is the discrete depth flow we test for an integral
  of motion.

  Honest caveat: T_i is the data-linearized layer map, not the exact nonlinear
  layer — the claim is about the EFFECTIVE transport, as in the sibling probes.

CANDIDATE INVARIANTS (a small panel, per layer i)
  (a) log_volume_rate  = sum_j log sigma_j(T_i)  (log|det| on the r-dim subspace)
        per-step volume change — constant => the flow is volume-rate-preserving.
  (b) energy_ratio     = ||T_i||_F^2 / r  (mean squared gain per direction).
  (c) spectral_entropy = effective_rank of T_i's singular values (participation
        ratio — how spread the per-step gain is).
  (d) holonomy_rotation_rate = mean principal angle between the top-k right-
        singular subspace of T_i and T_{i+1} (the per-step rotation of the moving
        frame; constant rate = constant angular velocity = a geometric flow). This
        is the only adjacency-coupled candidate and the one the moving-frame
        picture most directly predicts.

TEST + THE CRUCIAL NULL
  For each candidate we measure the COEFFICIENT OF VARIATION CV = std/|mean| across
  the INFORMATIVE (second-half) layers — the region where the task signal is
  formed. Low CV = a conservation-law candidate. But "low CV" is only meaningful
  relative to chance, so we compare against a LAYER-SHUFFLED NULL: randomly permute
  WHICH T_i sits at WHICH depth, recompute the candidate series, and take the CV of
  the (now random) second-half window. Because the window is the second HALF, a
  shuffle changes which factors land in it (and, for the rotation candidate, which
  factors are adjacent) — so the null is non-trivial. A real depth conservation law
  has CV_real significantly BELOW the shuffled-null CV distribution (a permutation
  test, Bonferroni across candidates).

    conservation_score = 1 - CV_real / mean(CV_shuffled)   (>0 => more conserved
                                                            than chance)
    p_value            = fraction of shuffles with CV <= CV_real (small => real)

FALSIFIABLE READINGS (per model + domain)
  CONSERVED     at least one candidate has CV_real significantly below the null
                (Bonferroni p < 0.05) with positive conservation_score => a
                depth-invariant integral of motion exists.
  NO_INVARIANT  no candidate beats the shuffled null => no conservation law; the
                per-layer quantities are no more constant than a random reordering.
                This is a VALID finding and is reported as such.

Local cache only (no training, no download). Run:
  .venv/bin/python depth_conservation_probe.py --synthetic
  .venv/bin/python depth_conservation_probe.py --models gpt2 pythia-160m
  .venv/bin/python depth_conservation_probe.py --aggregate
Output: depth_conservation_<model>.json (+ _summary.json, _synthetic.json)
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
# reuse the EXACT transport-fitting machinery of the free-probability probe
from free_probability_transport_probe import (  # noqa: E402
    _shared_basis,
    _fit_transport,
)

DEFAULT_MODELS = ["gpt2", "pythia-410m"]
DEFAULT_R = 32          # shared reduced subspace dimension (transport coordinates)
DEFAULT_N = 40          # prompt instances per domain (>= r+2 for a well-posed fit)
K_ROT = 4               # top-k right-singular subspace tracked for the holonomy rate
N_SHUFFLE = 300         # layer-shuffled null draws
MIN_LAYERS = 6          # need enough layers for a meaningful second half + null

CANDIDATES = [
    "log_volume_rate",
    "energy_ratio",
    "spectral_entropy",
    "holonomy_rotation_rate",
]


# ---------------------------------------------------------------------------
# Per-factor candidate quantities + the layer-shuffled conservation test
# ---------------------------------------------------------------------------
def _factor_scalars(T, eps=1e-12):
    """The three intrinsic per-layer candidate invariants of one transport T."""
    s = np.linalg.svd(np.asarray(T, float), compute_uv=False)
    s_pos = np.clip(s, eps, None)
    return {
        "log_volume_rate": float(np.sum(np.log(s_pos))),       # log|det| on r-dim
        "energy_ratio": float(np.sum(s ** 2) / max(len(s), 1)),  # mean sq gain
        "spectral_entropy": float(effective_rank(s)),          # participation ratio
    }


def _topk_right(T, k):
    """Top-k right-singular vectors of T (orthonormal rows, sigma-desc order)."""
    _, _, Vh = np.linalg.svd(np.asarray(T, float), full_matrices=False)
    return Vh[:min(k, Vh.shape[0])]


def _principal_rotation(Va, Vb):
    """Mean principal angle (radians) between two orthonormal k-subspaces — the
    per-step rotation magnitude of the moving frame."""
    c = np.linalg.svd(Va @ Vb.T, compute_uv=False)
    return float(np.mean(np.arccos(np.clip(c, -1.0, 1.0))))


def _cv(vals, eps=1e-9):
    """Coefficient of variation std/|mean| over finite values; nan if undefined
    (fewer than 2 values, or a near-zero mean that would blow the ratio up)."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size < 2:
        return float("nan")
    m = float(v.mean())
    if abs(m) < eps:
        return float("nan")
    return float(v.std(ddof=0) / abs(m))


def _series_for_order(order, scalars, vtopk, name):
    """Candidate series along a given factor ORDER. Scalar candidates reorder
    their per-factor value; the rotation candidate recomputes adjacent angles."""
    if name == "holonomy_rotation_rate":
        return [_principal_rotation(vtopk[order[j]], vtopk[order[j + 1]])
                for j in range(len(order) - 1)]
    return [scalars[i][name] for i in order]


def _windowed_cv(order, scalars, vtopk):
    """CV of each candidate over the SECOND-HALF window of the given order."""
    n = len(order)
    out = {}
    for name in CANDIDATES:
        ser = _series_for_order(order, scalars, vtopk, name)
        win = ser[len(ser) // 2:]            # informative / second-half region
        out[name] = _cv(win)
    return out


def conservation_analysis(factor_Ts, rng, k=K_ROT, n_shuffle=N_SHUFFLE,
                          alpha=0.05):
    """Layer-shuffled conservation test over an ordered list of transports.

    Returns per-candidate {cv_real, cv_shuffled, conservation_score, p_value,
    significant} plus the model-domain verdict, the best candidate, and its
    numbers. A candidate is SIGNIFICANT when its real second-half CV is below the
    shuffled-null CV distribution at a Bonferroni-corrected level AND its
    conservation_score is positive."""
    n = len(factor_Ts)
    if n < MIN_LAYERS:
        return {"verdict": "INSUFFICIENT", "n_factors": n,
                "candidates": {}, "best_candidate": None}

    scalars = [_factor_scalars(T) for T in factor_Ts]
    vtopk = [_topk_right(T, k) for T in factor_Ts]

    cv_real = _windowed_cv(list(range(n)), scalars, vtopk)
    shuffled = {c: [] for c in CANDIDATES}
    for _ in range(n_shuffle):
        perm = list(rng.permutation(n))
        cvs = _windowed_cv(perm, scalars, vtopk)
        for c in CANDIDATES:
            if np.isfinite(cvs[c]):
                shuffled[c].append(cvs[c])

    cands = {}
    for c in CANDIDATES:
        arr = np.asarray(shuffled[c], float)
        cvr = cv_real[c]
        if not np.isfinite(cvr) or arr.size < max(10, n_shuffle // 10):
            cands[c] = {"cv_real": None, "cv_shuffled": None,
                        "conservation_score": None, "p_value": None,
                        "significant": False}
            continue
        cv_sh = float(arr.mean())
        # permutation p: how often a random reordering is at least as flat as real
        p = float((np.sum(arr <= cvr) + 1) / (arr.size + 1))
        score = float(1.0 - cvr / cv_sh) if cv_sh > 0 else float("nan")
        cands[c] = {
            "cv_real": round(cvr, 5),
            "cv_shuffled": round(cv_sh, 5),
            "conservation_score": round(score, 4) if np.isfinite(score) else None,
            "p_value": round(p, 5),
            "n_shuffle_valid": int(arr.size),
        }

    n_tested = sum(1 for c in CANDIDATES if cands[c].get("p_value") is not None)
    for c in CANDIDATES:
        v = cands[c]
        sig = (v.get("p_value") is not None
               and v["p_value"] * max(n_tested, 1) < alpha          # Bonferroni
               and v.get("conservation_score") is not None
               and v["conservation_score"] > 0.0)
        v["significant"] = bool(sig)

    sig_names = [c for c in CANDIDATES if cands[c]["significant"]]
    if sig_names:
        best = max(sig_names, key=lambda c: cands[c]["conservation_score"])
        verdict = "CONSERVED"
    else:
        verdict = "NO_INVARIANT"
        scored = [c for c in CANDIDATES
                  if cands[c].get("conservation_score") is not None]
        best = (max(scored, key=lambda c: cands[c]["conservation_score"])
                if scored else None)

    return {
        "verdict": verdict,
        "n_factors": n,
        "k_rot": k,
        "n_shuffle": n_shuffle,
        "candidates": cands,
        "best_candidate": best,
        "best_conservation_score": (cands[best]["conservation_score"]
                                    if best else None),
        "best_p_value": cands[best]["p_value"] if best else None,
    }


# ---------------------------------------------------------------------------
# Real-model arm
# ---------------------------------------------------------------------------
def _domain_factors(layer_h, r):
    """Build the ordered effective-transport sequence T_0..T_{L-2} in a shared
    top-r PCA basis from captured per-layer query states (identical fit to the
    free-probability probe)."""
    present = [i for i in sorted(layer_h) if len(layer_h[i]) >= r + 2]
    if len(present) < MIN_LAYERS + 1:
        return None, present
    B = _shared_basis(layer_h, r)
    X = {}
    for i in present:
        H = np.asarray(layer_h[i], float)
        X[i] = (H - H.mean(axis=0, keepdims=True)) @ B.T
    factors = [_fit_transport(X[a], X[b])
               for a, b in zip(present[:-1], present[1:])]
    return factors, present


def run_model(model_name, n_samples=DEFAULT_N, r=DEFAULT_R, seed=0):
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
    d_model = int(model.config.hidden_size)

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

        factors, present = _domain_factors(layer_h, r)
        if factors is None:
            print(f"  {domain}: too few usable layers/samples — skipped")
            continue
        anrng = np.random.default_rng(seed)
        res = conservation_analysis(factors, anrng)
        res["n_instances"] = len(layer_h[present[0]])
        res["layers_used"] = len(present)
        per_domain[domain] = res

    verdicts = [a["verdict"] for a in per_domain.values()]
    any_conserved = any(v == "CONSERVED" for v in verdicts)
    # model-level best candidate = the most frequently conserved candidate
    best_per_dom = [a.get("best_candidate") for a in per_domain.values()
                    if a.get("verdict") == "CONSERVED"]
    model_best = (max(set(best_per_dom), key=best_per_dom.count)
                  if best_per_dom else None)
    out = {
        "mode": "real_model",
        "model": model_name,
        "hf_name": hf_name,
        "date": time.strftime("%Y-%m-%d"),
        "n_layers": n_layers,
        "d_model": d_model,
        "reduced_dim": r,
        "n_samples": n_samples,
        "method": "depth-invariant search over effective per-layer transport "
                  "(data-linearized) in a shared top-r PCA basis; CV vs "
                  "layer-shuffled null",
        "claim": "is there a depth-invariant integral of motion of the per-layer "
                 "transport (a conservation law along depth)?",
        "candidates": CANDIDATES,
        "per_domain": per_domain,
        "model_verdict": "CONSERVED" if any_conserved else "NO_INVARIANT",
        "model_best_candidate": model_best,
    }
    path = HERE / f"depth_conservation_{safe_name(model_name)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    for dom, a in per_domain.items():
        bc = a.get("best_candidate")
        sc = a.get("best_conservation_score")
        pv = a.get("best_p_value")
        print(f"  {dom:>22}: {a['verdict']:<12} best={bc} "
              f"score={sc} p={pv} (layers={a.get('layers_used')})")
    print(f"  model verdict = {out['model_verdict']}  "
          f"best_candidate = {model_best}")
    return out


# ---------------------------------------------------------------------------
# Synthetic validation (numpy only) — planted depth processes
# ---------------------------------------------------------------------------
def _expm_skew(A):
    """Orthogonal exp of a real skew-symmetric matrix via eigh of iA."""
    w, V = np.linalg.eigh(1j * A)
    return np.real(V @ np.diag(np.exp(-1j * w)) @ V.conj().T)


def _rand_skew(rng, r):
    G = rng.standard_normal((r, r))
    G = G - G.T
    return G / (np.linalg.norm(G) + 1e-9)


def _factor_from(svals_sorted, basis):
    """T = diag(s) @ B with B orthogonal => sigma = s (sorted), right-singular
    vectors = rows of B (so the top-k right subspace is B[:k] — directly
    controllable for the rotation candidate)."""
    return np.diag(svals_sorted) @ basis


def run_synthetic():
    """Plant three depth processes and verify the conservation test separates
    them WITHOUT tuning thresholds:
      A (log-det conserved): in the second half the per-step log-volume (log|det|)
        is held CONSTANT while the spectrum shape (hence energy/entropy) varies;
        the first half has a wildly varying log-volume. The probe must report
        CONSERVED with best candidate = log_volume_rate.
      C (rotation conserved): the top-k right-singular frame rotates at a CONSTANT
        angular velocity in the second half (varying rate in the first half),
        spectra i.i.d. everywhere. The probe must report CONSERVED with best
        candidate = holonomy_rotation_rate.
      B (null): every layer i.i.d. (no second-half structure). The probe must
        report NO_INVARIANT — no candidate beats the shuffled null."""
    rng = np.random.default_rng(0)
    r, n_layers, k = 32, 12, K_ROT
    half = n_layers // 2

    def rand_svals():
        return np.sort(np.exp(rng.normal(0.0, 0.6, r)))[::-1]

    def set_logdet(s, target):
        """Rescale s (keep SHAPE) so sum(log s) == target."""
        cur = float(np.sum(np.log(np.clip(s, 1e-12, None))))
        return s * np.exp((target - cur) / r)

    # --- Process A: constant log-det in the second half, wild in the first ---
    A_factors = []
    for i in range(n_layers):
        s = rand_svals()
        if i >= half:
            s = set_logdet(s, target=8.0)                 # held constant
        else:
            s = set_logdet(s, target=float(rng.normal(0.0, 12.0)))  # wild
        A_factors.append(_factor_from(s, np.eye(r)))      # no rotation
    A = conservation_analysis(A_factors, np.random.default_rng(1))

    # --- Process C: constant rotation rate of the top-k frame in the second half.
    # Build factors whose top-k right-singular subspace rigidly rotates in a fixed
    # 2k-dim block (V0 -> comp), at a CONSTANT per-pair angle in the second half
    # and a varying angle in the first half; spectra are i.i.d. everywhere (so no
    # scalar candidate carries the signal). The per-pair principal angle then
    # EQUALS the planted increment, so the rotation rate is conserved late. ---
    Q, _ = np.linalg.qr(rng.standard_normal((r, r)))
    V0, comp, C0 = Q[:k], Q[k:2 * k], Q[2 * k:]
    incr = [(0.30 if j >= (n_layers - 1) // 2 else rng.uniform(0.03, 0.6))
            for j in range(n_layers - 1)]
    thetas = np.concatenate([[0.0], np.cumsum(incr)])

    def frame(theta):
        Vb = np.cos(theta) * V0 + np.sin(theta) * comp     # rigid 2k-block rotation
        Vperp = -np.sin(theta) * V0 + np.cos(theta) * comp  # its in-block complement
        return np.vstack([Vb, Vperp, C0])                  # orthonormal r x r

    C_factors = []
    for i in range(n_layers):
        s = np.sort(np.concatenate([                       # top-k dominant -> the
            3.0 + rng.uniform(0.0, 1.0, k),                # top-k right subspace
            rng.uniform(0.0, 1.0, r - k)]))[::-1]          # is exactly Vb
        C_factors.append(_factor_from(s, frame(thetas[i])))
    C = conservation_analysis(C_factors, np.random.default_rng(2))

    # --- Process B: fully i.i.d. layers (genuine null) ---
    B_factors = []
    for _ in range(n_layers):
        s = rand_svals()
        Bb = _expm_skew(rng.uniform(0.05, 0.30) * r * _rand_skew(rng, r))
        B_factors.append(_factor_from(s, Bb))
    Bnull = conservation_analysis(B_factors, np.random.default_rng(3))

    checks = [
        ("A verdict == CONSERVED", A["verdict"] == "CONSERVED"),
        ("A best candidate == log_volume_rate",
         A["best_candidate"] == "log_volume_rate"),
        ("A log_volume_rate beats null (significant)",
         A["candidates"].get("log_volume_rate", {}).get("significant", False)),
        ("C verdict == CONSERVED", C["verdict"] == "CONSERVED"),
        ("C best candidate == holonomy_rotation_rate",
         C["best_candidate"] == "holonomy_rotation_rate"),
        ("C rotation beats null (significant)",
         C["candidates"].get("holonomy_rotation_rate", {})
         .get("significant", False)),
        ("B verdict == NO_INVARIANT", Bnull["verdict"] == "NO_INVARIANT"),
        ("B: no candidate significant",
         not any(v.get("significant") for v in Bnull["candidates"].values())),
    ]
    n_pass = sum(1 for _, ok in checks if ok)
    print(f"Synthetic: {n_pass}/{len(checks)} checks pass")
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
    for name, res in [("A_logdet", A), ("C_rotation", C), ("B_null", Bnull)]:
        bc, sc = res["best_candidate"], res.get("best_conservation_score")
        print(f"  {name:>11}: {res['verdict']:<12} best={bc} score={sc} "
              f"p={res.get('best_p_value')}")
    out = {
        "mode": "synthetic",
        "date": time.strftime("%Y-%m-%d"),
        "k_rot": k,
        "n_layers": n_layers,
        "processes": {"A_logdet_conserved": A,
                      "C_rotation_conserved": C,
                      "B_null": Bnull},
        "synthetic_checks": [{"test": d_, "pass": ok} for d_, ok in checks],
        "pass_rate": f"{n_pass}/{len(checks)}",
    }
    path = HERE / "depth_conservation_synthetic.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    return out


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def aggregate():
    rows = {}
    for p in sorted(HERE.glob("depth_conservation_*.json")):
        if p.stem.endswith(("summary", "synthetic")):
            continue
        d = json.loads(p.read_text())
        rows[d["model"]] = {
            "model_verdict": d.get("model_verdict"),
            "model_best_candidate": d.get("model_best_candidate"),
            "per_domain": {dom: {"verdict": a["verdict"],
                                 "best_candidate": a.get("best_candidate"),
                                 "best_conservation_score":
                                     a.get("best_conservation_score"),
                                 "best_p_value": a.get("best_p_value")}
                           for dom, a in d.get("per_domain", {}).items()},
        }
    out = {"analysis": "depth-conservation (integral of motion) aggregate",
           "claim": "is there a conservation law along depth in trained transport?",
           "date": time.strftime("%Y-%m-%d"), "models": rows}
    path = HERE / "depth_conservation_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}  ({len(rows)} models)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Depth-conservation / integral-of-"
                                             "motion test for trained transport")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-samples", type=int, default=DEFAULT_N)
    ap.add_argument("--r", type=int, default=DEFAULT_R)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic()
        return
    if args.aggregate:
        aggregate()
        return
    for m in (args.models or DEFAULT_MODELS):
        print(f"\n{'=' * 60}\n  {m}  (depth conservation, r={args.r})\n{'=' * 60}")
        t0 = time.time()
        try:
            run_model(m, n_samples=args.n_samples, r=args.r, seed=args.seed)
        except Exception as e:  # noqa: BLE001
            print(f"{m}: FAILED ({type(e).__name__}: {e})")
        print(f"  ({time.time() - t0:.1f}s)")
    aggregate()


if __name__ == "__main__":
    main()
