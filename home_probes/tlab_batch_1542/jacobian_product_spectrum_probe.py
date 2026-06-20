#!/usr/bin/env python3
"""Jacobian-product spectrum probe — the F0 GROUNDING check of the operator lift.

Spec: topics/ml_mathematics_of_trained_intelligence/OPERATOR_LIFT_SPEC.md (Part F0).

THE QUESTION (decisive, cheap, do-first). The MTI program names its central
object the *transport operator* T = prod_l J_l (product of per-layer Jacobians),
restricted to the task subspace, and claims its singular spectrum sigma_i drives
capacity / ICL speed / dark knowledge. But every existing probe MEASURES sigma_i
as a PROXY: the SVD of the residual-stream activation covariance across prompts
(singular_spectrum_probe.domain_spectrum). Those are two different mathematical
objects:

    PROXY      sigma_i(H)  : singular values of the across-prompt query-state
                            matrix H in R^{N x d} — the directions the layer's
                            state VARIES along across tasks.
    OPERATOR   sigma_i(T)  : singular values of the per-prompt Jacobian
                            T = d h_final[query] / d h_Lc[query] in R^{d x d} —
                            the SENSITIVITY map the theory actually talks about.

If these agree (spectrum aligned, leading subspaces overlapping), the operator
identification is sound and Parts A-E of the spec have empirical ground. If they
disagree, the program's sigma-claims are about activation covariance, not the
Jacobian product, and the operator lift is on sand. This probe measures BOTH and
reports their relationship. It is the gate that decides whether to develop the
operator math.

HOW (white-box; matrix-free, no d x d materialization):
  - Inject the query-position hidden state h at the capacity layer L_c via a
    forward hook, continue the network, read final-norm hidden at the query
    position. cont(h) is then a pure function whose Jacobian at h = h_query is
    the transport operator T (other positions' L_c states held fixed, exactly as
    the readout functional g is defined in singular_spectrum_probe).
  - JVP (T v) and VJP (T^T u) via torch.autograd.functional; top-k singular
    values by block subspace iteration + Rayleigh-Ritz. No d x d Jacobian.
  - Compare to the PROXY spectrum from the same prompts (reuse domain_spectrum).

SYNTHETIC gate (numpy only, no torch): planted L linear layers J_l with KNOWN
product T = prod J_l; the same matrix-free subspace iteration must recover
sigma_i(T) to high precision (estimator correctness), AND a planted case where
activation covariance and operator spectrum deliberately DIFFER must be flagged
as a mismatch (the probe can tell proxy from operator).

Run:
  python3 jacobian_product_spectrum_probe.py --synthetic
  python3 jacobian_product_spectrum_probe.py --models gpt2 --n-samples 16
Output: jacobian_product_spectrum_<model>.json (+ _synthetic.json)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import probe_base
from probe_base import Probe, load_context, register_probe

HERE = Path(__file__).resolve().parent

DEFAULT_MODELS = ["gpt2"]
DOMAINS = ["relation_composition", "arithmetic_carry", "induction"]
DEFAULT_K = 8          # number of leading singular values to estimate
DEFAULT_ITERS = 12     # subspace-iteration sweeps
DEFAULT_TAU = 1e-4     # (kept for signature parity with sibling probes)


# ==========================================================================
# Matrix-free top-k SVD by block subspace iteration.
# `apply` computes T @ V (n_out x k from n_in x k); `apply_t` computes T^T @ U.
# Both are plain callables, so the SAME routine serves numpy (synthetic) and
# torch JVP/VJP (real model) — one tested estimator, two backends.
# ==========================================================================
def topk_singular_values(apply, apply_t, n_in, k, *, iters, seed=0, want_vecs=False):
    """Top-k singular values (and optional right-singular vectors) of a linear
    operator given only its action T@V and T^T@U. Block subspace iteration on
    T^T T followed by a Rayleigh-Ritz step for accuracy."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n_in, k))
    V, _ = np.linalg.qr(V)
    for _ in range(iters):
        Z = apply_t(apply(V))            # (T^T T) V  -> (n_in x k)
        V, _ = np.linalg.qr(Z)
    # Rayleigh-Ritz: B = T V (n_out x k); its SVD gives the leading sigma and
    # the Ritz right vectors V @ Wr.
    B = apply(V)                          # (n_out x k)
    Ub, Sb, Wt = np.linalg.svd(B, full_matrices=False)
    order = np.argsort(-Sb)
    sigma = Sb[order]
    if want_vecs:
        Vr = V @ Wt.T[:, order]          # Ritz right-singular vectors (n_in x k)
        return sigma, Vr
    return sigma, None


def _spectrum_alignment(sig_op, sig_proxy, V_op=None, V_proxy=None):
    """Relationship between the operator spectrum and the proxy spectrum.
    Reports log-spectrum correlation, top-1 scale ratio, normalized spectral
    shape distance, and (if vectors given) leading-subspace overlap."""
    a = np.asarray(sig_op, float)
    b = np.asarray(sig_proxy, float)
    k = min(len(a), len(b))
    a, b = a[:k], b[:k]
    out = {"k": int(k)}
    if k >= 3 and a.min() > 0 and b.min() > 0:
        la, lb = np.log(a), np.log(b)
        if la.std() > 1e-9 and lb.std() > 1e-9:
            out["log_spectrum_pearson"] = round(float(np.corrcoef(la, lb)[0, 1]), 4)
        # shape distance after matching scale (normalize each to unit top value)
        an, bn = a / a[0], b / b[0]
        out["shape_l2_dist"] = round(float(np.linalg.norm(an - bn) / np.sqrt(k)), 4)
    out["top1_ratio_op_over_proxy"] = round(float(a[0] / (b[0] + 1e-30)), 4)
    if V_op is not None and V_proxy is not None:
        r = min(V_op.shape[1], V_proxy.shape[1], 4)
        # principal-subspace overlap in [0,1]: ||Vp^T Vo||_F^2 / r
        M = V_proxy[:, :r].T @ V_op[:, :r]
        out["leading_subspace_overlap"] = round(float((M ** 2).sum() / r), 4)
        out["subspace_dim"] = int(r)
    return out


# ==========================================================================
# SYNTHETIC mode (numpy only): planted Jacobian product with KNOWN spectrum.
# ==========================================================================
def run_synthetic(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d, L, k = 48, 6, 8

    # (1a) ESTIMATOR CORRECTNESS: explicit T with a WELL-SEPARATED (geometric)
    # spectrum, so the top-k singular subspace is unambiguous and the matrix-free
    # routine must recover sigma_i to high precision against np.linalg.svd.
    Ua, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Va, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s_geo = 0.7 ** np.arange(d)
    T = (Ua * s_geo) @ Va.T
    sigma_exact = s_geo[:k]
    Vop = Va[:, :k]                              # true right-singular vectors

    def apply(V):
        return T @ V

    def apply_t(U):
        return T.T @ U

    sigma_est, _ = topk_singular_values(apply, apply_t, d, k,
                                        iters=DEFAULT_ITERS, seed=1)
    est_rel_err = float(np.linalg.norm(sigma_est - sigma_exact)
                        / (np.linalg.norm(sigma_exact) + 1e-30))

    # (1b) PRODUCT-ACTION CORRECTNESS: a genuine layer product T_p = prod J_l,
    # and the matrix-free product action must equal the explicit product exactly
    # (this is the path the real-model JVP/VJP backend mirrors).
    Js = [np.eye(d) + 0.12 * rng.standard_normal((d, d)) / np.sqrt(d)
          for _ in range(L)]
    T_p = np.eye(d)
    for J in Js:
        T_p = J @ T_p

    def apply_p(V):
        X = V
        for J in Js:
            X = J @ X
        return X

    probe_vecs = rng.standard_normal((d, k))
    product_action_err = float(
        np.linalg.norm(apply_p(probe_vecs) - T_p @ probe_vecs)
        / (np.linalg.norm(T_p @ probe_vecs) + 1e-30))

    # (2) PROXY-vs-OPERATOR DISCRIMINATION (negative control): activations whose
    # covariance lives in a subspace rotated AWAY from the operator's top
    # directions — the alignment metric must flag the mismatch (low overlap).
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    task_dirs = Q[:, :4]
    N = 200
    coeffs = rng.standard_normal((N, 4)) * np.array([3.0, 2.0, 1.0, 0.5])
    Hc = coeffs @ task_dirs.T
    Hc = Hc - Hc.mean(0, keepdims=True)
    _, Sproxy, Vproxyh = np.linalg.svd(Hc, full_matrices=False)
    mismatch = _spectrum_alignment(sigma_exact, Sproxy[:k], Vop, Vproxyh.T[:, :k])

    # (3) PROXY-vs-OPERATOR MATCH (positive control): activations whose
    # covariance is generated ALONG the operator's top directions with the
    # operator's own spectrum — overlap ~1 and scale-free shape must match.
    coeffs2 = rng.standard_normal((N, k)) * sigma_exact
    H2c = coeffs2 @ Vop.T
    H2c = H2c - H2c.mean(0, keepdims=True)
    _, Sproxy2, Vproxyh2 = np.linalg.svd(H2c, full_matrices=False)
    match = _spectrum_alignment(sigma_exact, Sproxy2[:k], Vop, Vproxyh2.T[:, :k])

    verdict = {
        "estimator_recovers_spectrum": bool(est_rel_err < 1e-4),
        "estimator_rel_err": round(est_rel_err, 8),
        "product_action_correct": bool(product_action_err < 1e-10),
        "product_action_rel_err": round(product_action_err, 12),
        "discriminates_mismatch": bool(
            mismatch.get("leading_subspace_overlap", 1.0) < 0.4),
        "confirms_match": bool(
            match.get("leading_subspace_overlap", 0.0) > 0.9
            and match.get("shape_l2_dist", 1.0) < 0.1),
    }
    verdict["all_pass"] = bool(
        verdict["estimator_recovers_spectrum"]
        and verdict["product_action_correct"]
        and verdict["discriminates_mismatch"]
        and verdict["confirms_match"])
    out = {
        "mode": "synthetic",
        "probe": "jacobian_product_spectrum",
        "date": time.strftime("%Y-%m-%d"),
        "spec_anchor": "OPERATOR_LIFT_SPEC.md#part-f-empirical-contract (F0)",
        "dims": {"d": d, "L": L, "k": k, "iters": DEFAULT_ITERS},
        "sigma_exact": [round(float(x), 6) for x in sigma_exact],
        "sigma_estimated": [round(float(x), 6) for x in sigma_est],
        "negative_control_mismatch": mismatch,
        "positive_control_match": match,
        "verdict": verdict,
    }
    path = HERE / "jacobian_product_spectrum_synthetic.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print("\n=== synthetic Jacobian-product spectrum verdict ===")
    for kk, vv in verdict.items():
        print(f"  {kk}: {vv}")
    return out


# ==========================================================================
# REAL-MODEL mode (white-box): per-prompt transport operator T via JVP/VJP.
# ==========================================================================
def _transport_continuation(ctx, layers, l_c, input_ids, pos):
    """Build cont(h): inject hidden state h at the query position of layer L_c,
    continue the network, return final-norm hidden at the query position (R^d).
    Its Jacobian at h = h_query is the transport operator T = d h_final / d h_Lc."""
    import torch

    device = ctx.device
    norm, unembed = ctx.norm, ctx.unembed  # noqa: F841 (unembed available if needed)
    ids = torch.tensor([input_ids], device=device)

    # capture the real query-position L_c state (the linearization point)
    grabbed = {}

    def grab(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        grabbed["h"] = h[0, pos, :].detach().clone()
        return output

    hh = layers[l_c].register_forward_hook(grab)
    try:
        with torch.no_grad():
            ctx.model(input_ids=ids)
    finally:
        hh.remove()
    h_query = grabbed["h"]                       # (d,)

    def cont(h_vec):
        def inject(_m, _i, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            h = h.clone()
            h[0, pos, :] = h_vec
            return (h,) + tuple(output[1:]) if is_tuple else h
        handle = layers[l_c].register_forward_hook(inject)
        try:
            out = ctx.model(input_ids=ids, output_hidden_states=True)
            h_final = out.hidden_states[-1][0, pos, :]
            return ctx.norm(h_final)            # (d,) final-norm readout side
        finally:
            handle.remove()

    return cont, h_query


def _operator_spectrum(ctx, layers, l_c, input_ids, pos, k, iters):
    """Top-k singular values + right-singular vectors of the per-prompt transport
    operator, matrix-free via torch JVP/VJP."""
    import torch
    from torch.autograd.functional import jvp, vjp

    cont, h_query = _transport_continuation(ctx, layers, l_c, input_ids, pos)
    d = h_query.shape[0]

    def apply(Vnp):                              # T @ V  (d x k)
        cols = []
        for j in range(Vnp.shape[1]):
            v = torch.tensor(Vnp[:, j], dtype=h_query.dtype, device=ctx.device)
            _, jv = jvp(cont, (h_query,), (v,))
            cols.append(jv.detach().cpu().numpy().astype(float))
        return np.stack(cols, axis=1)

    def apply_t(Unp):                            # T^T @ U  (d x k)
        cols = []
        for j in range(Unp.shape[1]):
            u = torch.tensor(Unp[:, j], dtype=h_query.dtype, device=ctx.device)
            _, vt = vjp(cont, h_query, u)
            cols.append(vt.detach().cpu().numpy().astype(float))
        return np.stack(cols, axis=1)

    sigma, Vr = topk_singular_values(apply, apply_t, int(d), k,
                                     iters=iters, seed=0, want_vecs=True)
    return sigma, Vr


def _operator_gains(ctx, layers, l_c, input_ids, pos, cols):
    """Apply the per-prompt transport operator to each column of `cols` (d x m)
    and return the output norms ||T c_j|| (m,). The F0' task-relevance test: does
    the operator AMPLIFY the proxy's task directions more than random ones?"""
    import torch
    from torch.autograd.functional import jvp

    cont, h_query = _transport_continuation(ctx, layers, l_c, input_ids, pos)
    gains = []
    for j in range(cols.shape[1]):
        v = torch.tensor(cols[:, j], dtype=h_query.dtype, device=ctx.device)
        v = v / (v.norm() + 1e-12)
        _, jv = jvp(cont, (h_query,), (v,))
        gains.append(float(jv.detach().norm().cpu()))
    return np.asarray(gains, float)


def _task_gain_test(ctx, layers, l_c, prompts, V_proxy, sigma_proxy,
                    d_model, seed, r=4):
    """F0': for each prompt apply the operator to the proxy task directions and to
    random directions; report (i) gain_ratio = mean||T v_task|| / mean||T v_rand||
    (>1 => the task subspace is operator-relevant) and (ii) the correlation
    between the operator's per-direction gain and the proxy's sigma_i (does the
    operator amplify task direction i in proportion to its proxy magnitude?)."""
    r = int(min(r, V_proxy.shape[1], d_model))
    rng = np.random.default_rng(seed + 777)
    Vt = V_proxy[:, :r]
    Rrand, _ = np.linalg.qr(rng.standard_normal((d_model, r)))
    task_gains, rand_gains = [], []
    for (seq, pos) in prompts:
        try:
            task_gains.append(_operator_gains(ctx, layers, l_c, seq, pos, Vt))
            rand_gains.append(_operator_gains(ctx, layers, l_c, seq, pos, Rrand))
        except Exception:  # noqa: BLE001
            continue
    if len(task_gains) < 3:
        return {"task_gain_ratio": None, "task_gain_vs_sigma_pearson": None}
    TG = np.stack(task_gains, axis=0)            # (n, r)
    RG = np.stack(rand_gains, axis=0)            # (n, r)
    gain_ratio = float(TG.mean() / (RG.mean() + 1e-30))
    mean_gain_per_dir = TG.mean(axis=0)          # (r,)
    sig = np.asarray(sigma_proxy[:r], float)
    pear = None
    if r >= 3 and mean_gain_per_dir.std() > 1e-9 and sig.std() > 1e-9:
        pear = round(float(np.corrcoef(mean_gain_per_dir, sig)[0, 1]), 4)
    return {
        "task_gain_ratio": round(gain_ratio, 4),
        "task_gain_vs_sigma_pearson": pear,
        "task_gain_dim": r,
    }


def run_with_context(ctx, n_samples=16, seed=42, tau=DEFAULT_TAU,
                     k=DEFAULT_K, iters=DEFAULT_ITERS) -> dict:
    import torch  # noqa: F401
    from icl_convergence_probe import safe_name
    from capacity_threshold_sweep import make_prompt
    from routing_selection_probe import build_induction
    from singular_spectrum_probe import (capture_query_state, domain_spectrum,
                                         _capacity_layer_index)

    layers = ctx.layers
    n_layers = len(layers)
    l_c = _capacity_layer_index(n_layers)
    d_model = int(ctx.model.config.hidden_size)
    k = min(k, d_model)

    per_domain = {}
    for domain in DOMAINS:
        rng = np.random.default_rng(seed)
        op_specs, op_vecs, prompts = [], [], []
        h_states, g_states = [], []
        for _ in range(n_samples):
            if domain == "induction":
                seq, pos, cid = build_induction(ctx.tok, rng, block_len=8)
            else:
                seq, pos, cid = make_prompt(domain, ctx.tok, rng)
            try:
                # PROXY ingredients (across-prompt covariance), reused verbatim
                h_q, g = capture_query_state(ctx.model, layers, l_c, ctx.norm,
                                             ctx.unembed, seq, pos, cid, ctx.device)
                # OPERATOR spectrum (per-prompt Jacobian product)
                sig_op, V_op = _operator_spectrum(ctx, layers, l_c, seq, pos,
                                                  k, iters)
            except Exception as e:  # noqa: BLE001
                print(f"  {domain}: prompt failed ({type(e).__name__}: {e})")
                continue
            h_states.append(h_q)
            g_states.append(g)
            op_specs.append(sig_op)
            op_vecs.append(V_op)
            prompts.append((seq, pos))
        if len(op_specs) < 3:
            continue
        # PROXY spectrum (cross-prompt task subspace) + its right vectors
        sigma_proxy, rho = domain_spectrum(h_states, g_states)
        Hc = np.asarray(h_states, float) - np.mean(h_states, axis=0, keepdims=True)
        _, _, Vproxyh = np.linalg.svd(Hc, full_matrices=False)
        V_proxy = Vproxyh.T[:, :k]
        # OPERATOR spectrum: mean over prompts (sorted, top-k), and mean overlap
        op_mat = np.stack([s[:k] for s in op_specs], axis=0)
        sigma_op_mean = op_mat.mean(axis=0)
        # per-prompt alignment of operator right-subspace to the proxy subspace
        aligns = [_spectrum_alignment(s[:k], sigma_proxy[:k], V, V_proxy)
                  for s, V in zip(op_specs, op_vecs)]
        overlaps = [a["leading_subspace_overlap"] for a in aligns
                    if "leading_subspace_overlap" in a]
        agg = _spectrum_alignment(sigma_op_mean, sigma_proxy[:k])
        agg["leading_subspace_overlap_mean"] = (
            round(float(np.mean(overlaps)), 4) if overlaps else None)
        agg["leading_subspace_overlap_random_baseline"] = round(
            float(min(k, d_model) / d_model), 4)  # E[overlap] for random subspaces
        agg["sigma_op_mean_top"] = round(float(sigma_op_mean[0]), 5)
        agg["sigma_proxy_top"] = round(float(sigma_proxy[0]), 5)
        agg["n_instances"] = len(op_specs)
        # ---- F0': task-subspace gain test ----------------------------------
        # Does the per-prompt operator AMPLIFY the proxy task directions more
        # than random directions? (the task-relevant grounding the global top-σ
        # overlap cannot see, because the global top-σ are dark high-norm modes.)
        agg.update(_task_gain_test(ctx, layers, l_c, prompts, V_proxy,
                                   sigma_proxy, d_model, seed))
        per_domain[domain] = agg

    verdict = _verdict(per_domain)
    out = {
        "mode": "real_model",
        "probe": "jacobian_product_spectrum",
        "model": ctx.name,
        "hf_name": ctx.hf_name,
        "date": time.strftime("%Y-%m-%d"),
        "spec_anchor": "OPERATOR_LIFT_SPEC.md#part-f-empirical-contract (F0)",
        "n_layers": n_layers,
        "capacity_layer": l_c,
        "d_model": d_model,
        "k": k,
        "iters": iters,
        "n_samples": n_samples,
        "per_domain": per_domain,
        "verdict": verdict,
    }
    path = HERE / f"jacobian_product_spectrum_{safe_name(ctx.name)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    for dom, a in per_domain.items():
        print(f"  {dom:>22}: log_pearson={a.get('log_spectrum_pearson')} "
              f"overlap={a.get('leading_subspace_overlap_mean')} "
              f"(rand {a.get('leading_subspace_overlap_random_baseline')})  "
              f"F0' gain_ratio={a.get('task_gain_ratio')} "
              f"gain~σ={a.get('task_gain_vs_sigma_pearson')}")
    print(f"  verdict: {verdict}")
    return out


def _verdict(per_domain: dict) -> dict:
    def med(key):
        vals = [a[key] for a in per_domain.values() if a.get(key) is not None]
        return float(np.median(vals)) if vals else None

    m_pear = med("log_spectrum_pearson")          # shape grounding
    m_ov = med("leading_subspace_overlap_mean")   # global top-σ direction overlap
    m_base = med("leading_subspace_overlap_random_baseline")
    m_gain = med("task_gain_ratio")               # F0' task-relevance
    m_gsig = med("task_gain_vs_sigma_pearson")    # F0' per-direction agreement

    shape_grounded = bool(m_pear is not None and m_pear > 0.5)
    # F0': the task subspace is operator-relevant if the operator amplifies the
    # proxy task directions clearly above random AND the per-direction gain tracks
    # the proxy spectrum. This is the grounding the global top-σ overlap (which is
    # dominated by dark high-norm modes) cannot establish.
    task_grounded = bool(m_gain is not None and m_gain > 1.5
                         and m_gsig is not None and m_gsig > 0.4)

    if shape_grounded and task_grounded:
        interp = ("GROUNDED: spectrum shape tracks the proxy AND the operator "
                  "amplifies the proxy task subspace above random (F0') — the "
                  "operator identification holds; Parts A-E have empirical ground.")
    elif shape_grounded:
        interp = ("SHAPE GROUNDED, TASK-RELEVANCE WEAK: σ magnitudes track the "
                  "proxy but the operator does not preferentially amplify the "
                  "proxy task subspace (F0') — the operator and the proxy share a "
                  "spectral envelope but not the task-relevant directions.")
    else:
        interp = ("PROXY ARTIFACT RISK: operator spectrum diverges from the "
                  "measured proxy — the lift needs the operator measured directly.")

    return {
        "shape_grounded": shape_grounded,
        "task_relevance_grounded_F0prime": task_grounded,
        "operator_matches_proxy": bool(shape_grounded and task_grounded),
        "median_log_spectrum_pearson": (round(m_pear, 4) if m_pear is not None else None),
        "median_global_subspace_overlap": (round(m_ov, 4) if m_ov is not None else None),
        "random_overlap_baseline": (round(m_base, 4) if m_base is not None else None),
        "median_task_gain_ratio_F0prime": (round(m_gain, 4) if m_gain is not None else None),
        "median_task_gain_vs_sigma_pearson": (round(m_gsig, 4) if m_gsig is not None else None),
        "n_domains_measured": len(per_domain),
        "interpretation": interp,
    }


def run_model(model_name, n_samples=16, seed=42, tau=DEFAULT_TAU,
              k=DEFAULT_K, iters=DEFAULT_ITERS) -> dict:
    """Load the model, then compute. Use run_with_context to share a load."""
    return run_with_context(load_context(model_name), n_samples, seed, tau, k, iters)


# ==========================================================================
# OOP front (probe_base.Probe) + registration
# ==========================================================================
class JacobianProductSpectrumProbe(Probe):
    name = "jacobian_product_spectrum"
    classifier_features: list[str] = []   # spectrum measurement, no classifier
    label_key = None

    def synthetic(self, seed: int = 0) -> dict:
        return run_synthetic(seed=seed)

    def run_model(self, model_name: str, *, n_samples: int = 16, seed: int = 42,
                  tau: float = DEFAULT_TAU, k: int = DEFAULT_K,
                  iters: int = DEFAULT_ITERS, **_) -> dict:
        return run_model(model_name, n_samples, seed, tau, k, iters)

    def run_on(self, ctx, *, n_samples: int = 16, seed: int = 42,
               tau: float = DEFAULT_TAU, k: int = DEFAULT_K,
               iters: int = DEFAULT_ITERS, **_) -> dict:
        return run_with_context(ctx, n_samples, seed, tau, k, iters)


register_probe(JacobianProductSpectrumProbe())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic()
        return
    for m in (args.models or DEFAULT_MODELS):
        try:
            run_model(m, args.n_samples, k=args.k, iters=args.iters)
        except Exception as e:  # noqa: BLE001
            print(f"{m}: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
