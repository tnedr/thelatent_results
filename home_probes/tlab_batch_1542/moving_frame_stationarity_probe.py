#!/usr/bin/env python3
"""Moving-frame stationarity probe — the F1 check that decides the B2 branch.

Spec: topics/ml_mathematics_of_trained_intelligence/OPERATOR_LIFT_SPEC.md
      (Def. 2.3 moving frame; Part D connection; §13.3 B2 whitening vs imperfect).

THE QUESTION. The B2 kernel result (capacity_transport_proof.py) has two branches:

  * whitening_preconditioner_zeroes_contraction — the EXACT depth-Newton step,
    valid when the transport singular basis is STATIONARY across depth (the
    multi-layer contributions sum in a FIXED frame, so a fixed-basis
    preconditioner can whiten the spectrum perfectly, delta = 0);
  * imperfect_preconditioning_near_newton + depth_refines_preconditioner — the
    near-Newton bound contraction <= delta^2, valid when the frame ROTATES with
    depth, so the cross-layer coupling delta > 0 (and shrinks with depth).

Which one governs real models is an EMPIRICAL question about the moving frame
(Def. 2.3): the flow V_{l_c+1}, ..., V_L of the top-k right-singular subspace of
the partial transport T_{l_c -> l} = d h_l / d h_{l_c}, living in R^d at the
capacity layer. If that subspace is stationary (consecutive overlap ~ 1, the
Procrustes connection Q_l ~ I), exact whitening is the right model. If it rotates
slowly (overlap well above the random baseline k/d but Q_l != I, low effective
rotation dim), the imperfect-preconditioning bound governs and delta ~ the frame
drift. If it is incoherent (overlap ~ k/d), there is no stable transport frame
and neither branch applies cleanly.

This probe MEASURES the frame flow and returns the branch verdict + a delta proxy
(1 - total-drift overlap) that the kernel's imperfect-preconditioning delta maps
to. It does not train; it measures, with a synthetic ground-truth gate.

HOW (white-box; matrix-free, no d x d materialization):
  - inject the query-position hidden state at the capacity layer L_c via a hook,
    continue, and read the hidden state at an END layer cut e (no final norm —
    the transport Jacobian d h_e / d h_Lc lives in hidden space);
  - its top-k right-singular subspace V_e (in R^d at L_c) is found matrix-free by
    JVP/VJP + subspace iteration (reused from jacobian_product_spectrum_probe);
  - sweep e over several cuts to get the frame flow; measure consecutive subspace
    overlap, the Procrustes connection Q_e and its deviation from I, the per-step
    principal angles, the effective rotation dimension, and the total drift.

SYNTHETIC gate (numpy only, no torch): planted frame flows with KNOWN geometry —
(1) a fixed in-subspace rotation by angle phi the Procrustes step must recover;
(2) a STATIONARY flow (overlap 1, Q = I); (3) a SLOWLY-ROTATING flow (high
overlap, low effective rotation dim) distinguished from (4) a RANDOM flow
(overlap ~ k/d). The probe must recover phi and classify all four regimes.

Run:
  python3 moving_frame_stationarity_probe.py --synthetic
  python3 moving_frame_stationarity_probe.py --models gpt2 --n-samples 6
Output: moving_frame_stationarity_<model>.json (+ _synthetic.json)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import probe_base
from probe_base import Probe, load_context, register_probe
from jacobian_product_spectrum_probe import topk_singular_values

HERE = Path(__file__).resolve().parent

DEFAULT_MODELS = ["gpt2"]
DOMAINS = ["relation_composition", "arithmetic_carry", "induction"]
DEFAULT_K = 6          # task-subspace dimension (frame rank)
DEFAULT_ITERS = 8      # subspace-iteration sweeps per cut
DEFAULT_CUTS = 5       # number of end-layer cuts in the frame flow
DEFAULT_N = 6          # prompts per domain (frame flow is per-prompt, expensive)

# Branch thresholds (pre-registered so the verdict is not fit to the data).
STATIONARY_DRIFT = 0.97        # end-to-end drift overlap above this = stationary
STATIONARY_ANGLE = 0.05        # mean per-step principal angle (rad) below = no twist
COHERENT_RATIO = 5.0           # consecutive overlap must beat random baseline by this


# ==========================================================================
# Frame geometry primitives (numpy; tested by the synthetic gate).
# Va, Vb are d x k matrices with orthonormal columns (subspace bases in R^d).
# ==========================================================================
def frame_step(Va: np.ndarray, Vb: np.ndarray) -> dict:
    """One step of the frame flow: subspace overlap, the Procrustes connection
    Q = argmin_{Q in O(k)} ||Vb - Va Q||_F, its deviation from I, the principal
    angles between span(Va) and span(Vb), and the effective rotation dimension."""
    k = Va.shape[1]
    M = Va.T @ Vb                                  # (k x k), M_ij = <a_i, b_j>
    A, c, Bt = np.linalg.svd(M)
    c = np.clip(c, -1.0, 1.0)                      # cos of principal angles
    overlap = float((M ** 2).sum() / k)           # ||M||_F^2 / k in [0, 1]
    Q = A @ Bt                                     # orthogonal Procrustes solution
    rot_dev = float(np.linalg.norm(Q - np.eye(k)) / np.sqrt(2.0 * k))
    angles = np.arccos(c)
    mean_angle = float(angles.mean())
    eff_rot_dim = float((np.sin(angles) ** 2).sum())   # # of rotating directions
    return {
        "overlap": overlap,
        "procrustes_dev": rot_dev,
        "mean_principal_angle": mean_angle,
        "effective_rotation_dim": eff_rot_dim,
        "Q": Q,
    }


def frame_flow_summary(frames: list[np.ndarray], d: int) -> dict:
    """Summarize a frame flow V_0, ..., V_T: mean consecutive overlap, mean
    connection deviation, mean per-step principal angle, mean effective rotation
    dim, the total drift overlap (first vs last) and a delta proxy (1 - drift)."""
    k = frames[0].shape[1]
    steps = [frame_step(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    ov = float(np.mean([s["overlap"] for s in steps]))
    dev = float(np.mean([s["procrustes_dev"] for s in steps]))
    ang = float(np.mean([s["mean_principal_angle"] for s in steps]))
    erd = float(np.mean([s["effective_rotation_dim"] for s in steps]))
    drift = frame_step(frames[0], frames[-1])
    baseline = float(k) / float(d)                 # E[overlap] for random subspaces
    return {
        "consecutive_overlap": round(ov, 4),
        "consecutive_procrustes_dev": round(dev, 4),
        "consecutive_mean_angle": round(ang, 4),
        "effective_rotation_dim": round(erd, 3),
        "total_drift_overlap": round(float(drift["overlap"]), 4),
        "delta_proxy": round(float(1.0 - drift["overlap"]), 4),
        "random_overlap_baseline": round(baseline, 4),
        "overlap_over_baseline": round(ov / (baseline + 1e-30), 2),
        "n_steps": len(steps),
    }


def classify_frame(summary: dict) -> dict:
    """Map a frame-flow summary to the B2 branch it supports.

    Two axes: COHERENCE (consecutive overlap vs the random k/d baseline — is there
    a frame at all?) and DRIFT (end-to-end total-drift overlap — does that frame
    move?). A frame can be coherent step-to-step yet drift substantially over depth
    as small rotations accumulate; that accumulated drift, not the per-step
    overlap, is what separates an exactly-whitenable stationary basis from a
    slowly-rotating one."""
    ang = summary["consecutive_mean_angle"]
    ratio = summary["overlap_over_baseline"]
    drift = summary.get("total_drift_overlap")
    if drift is None:
        drift = 1.0 - summary.get("delta_proxy", 0.0)
    if ratio < COHERENT_RATIO:
        regime = "INCOHERENT"
        branch = "neither (transport frame not stable across depth)"
        note = ("consecutive overlap is near the random baseline — there is no "
                "stable transport frame, so the B2 preconditioner picture does "
                "not apply in a fixed or slowly-moving basis.")
        return {"regime": regime, "b2_branch": branch, "note": note}
    if drift >= STATIONARY_DRIFT and ang <= STATIONARY_ANGLE:
        regime = "STATIONARY"
        branch = "whitening_preconditioner_zeroes_contraction (EXACT)"
        note = ("the transport singular basis is stationary in depth — a "
                "fixed-basis preconditioner can whiten the spectrum exactly; "
                "the exact depth-Newton step is the right model.")
    else:
        regime = "SLOWLY_ROTATING"
        branch = "imperfect_preconditioning_near_newton + depth_refines_preconditioner"
        note = ("the frame is coherent but rotates with depth — exact whitening "
                "fails, the near-Newton bound contraction <= delta^2 governs with "
                "delta ~ delta_proxy, and depth refines (shrinks) it.")
    return {"regime": regime, "b2_branch": branch, "note": note}


# ==========================================================================
# SYNTHETIC mode (numpy only): planted frame flows with KNOWN geometry.
# ==========================================================================
def _rotation_in_plane(k: int, i: int, j: int, phi: float) -> np.ndarray:
    R = np.eye(k)
    c, s = np.cos(phi), np.sin(phi)
    R[i, i], R[j, j] = c, c
    R[i, j], R[j, i] = -s, s
    return R


def run_synthetic(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d, k, T = 64, 6, 6

    base, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V0 = base[:, :k]

    # (1) ESTIMATOR: a single in-subspace rotation by a KNOWN angle phi; the
    # Procrustes step must recover R (Q == R), so it measures the connection.
    phi = 0.4
    R = _rotation_in_plane(k, 0, 1, phi)
    step = frame_step(V0, V0 @ R)
    recover_err = float(np.linalg.norm(step["Q"] - R) / np.sqrt(2.0 * k))
    # the recovered rotation angle (planted only in the (0,1) plane)
    ang_recovered = float(np.arccos(np.clip(step["Q"][0, 0], -1.0, 1.0)))

    # (2) STATIONARY flow: V_l = V0 for all l -> overlap 1, connection 0.
    stat = frame_flow_summary([V0.copy() for _ in range(T)], d)

    # (3) SLOWLY-ROTATING flow: cumulatively tilt ONE column toward a fixed
    # out-of-span direction by a small per-step angle eps (effective rotation
    # dim ~ 1). Each step's overlap stays high (coherent), but the small tilts
    # ACCUMULATE so the end-to-end drift is well below 1 — the signature the
    # classifier uses to separate this from a stationary frame.
    eps = 0.12
    out_dir = base[:, k]                            # fixed direction outside span
    frames_rot = [V0]
    for t in range(1, T):
        ang_t = eps * t
        nxt = V0.copy()
        nxt[:, 0] = np.cos(ang_t) * V0[:, 0] + np.sin(ang_t) * out_dir
        nxt, _ = np.linalg.qr(nxt)                  # re-orthonormalize
        frames_rot.append(nxt)
    rot = frame_flow_summary(frames_rot, d)

    # (4) RANDOM flow: independent random k-subspaces -> overlap ~ k/d.
    frames_rand = []
    for _ in range(T):
        Qr, _ = np.linalg.qr(rng.standard_normal((d, d)))
        frames_rand.append(Qr[:, :k])
    rand = frame_flow_summary(frames_rand, d)

    verdict = {
        "estimator_recovers_rotation": bool(recover_err < 1e-6),
        "estimator_recover_err": round(recover_err, 10),
        "estimator_angle_planted": round(phi, 4),
        "estimator_angle_recovered": round(ang_recovered, 4),
        "stationary_classified": bool(classify_frame(stat)["regime"] == "STATIONARY"),
        "slowly_rotating_classified": bool(
            classify_frame(rot)["regime"] == "SLOWLY_ROTATING"),
        "random_classified": bool(classify_frame(rand)["regime"] == "INCOHERENT"),
        "rotating_low_eff_dim": bool(rot["effective_rotation_dim"] < 2.5),
    }
    verdict["all_pass"] = bool(
        verdict["estimator_recovers_rotation"]
        and abs(verdict["estimator_angle_recovered"]
                - verdict["estimator_angle_planted"]) < 1e-4
        and verdict["stationary_classified"]
        and verdict["slowly_rotating_classified"]
        and verdict["random_classified"]
        and verdict["rotating_low_eff_dim"])

    out = {
        "mode": "synthetic",
        "probe": "moving_frame_stationarity",
        "date": time.strftime("%Y-%m-%d"),
        "spec_anchor": "OPERATOR_LIFT_SPEC.md#part-d (Def 2.3) / B2 branch (F1)",
        "dims": {"d": d, "k": k, "T": T},
        "stationary_control": stat,
        "slowly_rotating_control": rot,
        "random_control": rand,
        "verdict": verdict,
    }
    path = HERE / "moving_frame_stationarity_synthetic.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"saved -> {path.name}")
    print("\n=== synthetic moving-frame verdict ===")
    for kk, vv in verdict.items():
        print(f"  {kk}: {vv}")
    return out


# ==========================================================================
# REAL-MODEL mode (white-box): the partial-transport frame flow per prompt.
# ==========================================================================
def _partial_transport_continuation(ctx, layers, l_c, end_idx, input_ids, pos):
    """cont(h): inject hidden state h at the query position of layer L_c, continue,
    and return the hidden state at END layer cut `end_idx` (hidden_states index, no
    final norm). Its Jacobian at h = h_query is the partial transport
    T_{Lc->end} = d h_end / d h_Lc, whose right-singular subspace is the frame V."""
    import torch

    device = ctx.device
    ids = torch.tensor([input_ids], device=device)
    grabbed = {}

    def grab(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        grabbed["h"] = h[0, pos, :].detach().clone()
        return output

    hh = layers[l_c].register_forward_hook(grab)
    try:
        with torch.no_grad():
            ctx.model(input_ids=ids, output_hidden_states=True)
    finally:
        hh.remove()
    h_query = grabbed["h"]

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
            return out.hidden_states[end_idx][0, pos, :]    # hidden-space readout
        finally:
            handle.remove()

    return cont, h_query


def _transport_subspace(ctx, layers, l_c, end_idx, input_ids, pos, k, iters):
    """Top-k right-singular subspace (in R^d at L_c) of the partial transport
    T_{Lc->end_idx}, matrix-free via torch JVP/VJP + subspace iteration."""
    import torch
    from torch.autograd.functional import jvp, vjp

    cont, h_query = _partial_transport_continuation(
        ctx, layers, l_c, end_idx, input_ids, pos)
    d = h_query.shape[0]

    def apply(Vnp):
        cols = []
        for j in range(Vnp.shape[1]):
            v = torch.tensor(Vnp[:, j], dtype=h_query.dtype, device=ctx.device)
            _, jv = jvp(cont, (h_query,), (v,))
            cols.append(jv.detach().cpu().numpy().astype(float))
        return np.stack(cols, axis=1)

    def apply_t(Unp):
        cols = []
        for j in range(Unp.shape[1]):
            u = torch.tensor(Unp[:, j], dtype=h_query.dtype, device=ctx.device)
            _, vt = vjp(cont, h_query, u)
            cols.append(vt.detach().cpu().numpy().astype(float))
        return np.stack(cols, axis=1)

    _, Vr = topk_singular_values(apply, apply_t, int(d), k,
                                 iters=iters, seed=0, want_vecs=True)
    return Vr


def run_with_context(ctx, n_samples=DEFAULT_N, seed=42, k=DEFAULT_K,
                     iters=DEFAULT_ITERS, cuts=DEFAULT_CUTS) -> dict:
    import torch  # noqa: F401
    from icl_convergence_probe import safe_name
    from capacity_threshold_sweep import make_prompt
    from routing_selection_probe import build_induction
    from singular_spectrum_probe import _capacity_layer_index

    layers = ctx.layers
    n_layers = len(layers)
    l_c = _capacity_layer_index(n_layers)
    d_model = int(ctx.model.config.hidden_size)
    k = min(k, d_model)

    # END-layer cuts in hidden_states index space: from just past L_c to the last
    # layer. hidden_states[i] is the state after i layers; transport L_c -> i needs
    # i >= l_c + 2 (at least one block applied past the injection point).
    first_cut = l_c + 2
    last_cut = n_layers
    if last_cut - first_cut + 1 < 3:
        first_cut = max(2, l_c)                     # tiny models: widen the window
    cut_idxs = sorted(set(
        int(round(x)) for x in np.linspace(first_cut, last_cut,
                                           min(cuts, last_cut - first_cut + 1))))

    per_domain = {}
    for domain in DOMAINS:
        rng = np.random.default_rng(seed)
        flow_summaries = []
        for _ in range(n_samples):
            if domain == "induction":
                seq, pos, _cid = build_induction(ctx.tok, rng, block_len=8)
            else:
                seq, pos, _cid = make_prompt(domain, ctx.tok, rng)
            try:
                frames = [_transport_subspace(ctx, layers, l_c, e, seq, pos, k, iters)
                          for e in cut_idxs]
            except Exception as e:  # noqa: BLE001
                print(f"  {domain}: prompt failed ({type(e).__name__}: {e})")
                continue
            if len(frames) < 3:
                continue
            flow_summaries.append(frame_flow_summary(frames, d_model))
        if len(flow_summaries) < 3:
            continue

        def med(key):
            return round(float(np.median([s[key] for s in flow_summaries])), 4)

        agg = {key: med(key) for key in (
            "consecutive_overlap", "consecutive_procrustes_dev",
            "consecutive_mean_angle", "effective_rotation_dim",
            "total_drift_overlap", "delta_proxy", "overlap_over_baseline")}
        agg["random_overlap_baseline"] = flow_summaries[0]["random_overlap_baseline"]
        agg["n_instances"] = len(flow_summaries)
        agg.update(classify_frame(agg))
        per_domain[domain] = agg

    verdict = _verdict(per_domain)
    out = {
        "mode": "real_model",
        "probe": "moving_frame_stationarity",
        "model": ctx.name,
        "hf_name": ctx.hf_name,
        "date": time.strftime("%Y-%m-%d"),
        "spec_anchor": "OPERATOR_LIFT_SPEC.md#part-d (Def 2.3) / B2 branch (F1)",
        "n_layers": n_layers,
        "capacity_layer": l_c,
        "d_model": d_model,
        "k": k,
        "iters": iters,
        "cut_idxs": cut_idxs,
        "n_samples": n_samples,
        "per_domain": per_domain,
        "verdict": verdict,
    }
    path = HERE / f"moving_frame_stationarity_{safe_name(ctx.name)}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"saved -> {path.name}")
    for dom, a in per_domain.items():
        print(f"  {dom:>22}: overlap={a['consecutive_overlap']} "
              f"(x{a['overlap_over_baseline']} rand) angle={a['consecutive_mean_angle']} "
              f"eff_dim={a['effective_rotation_dim']} delta={a['delta_proxy']} "
              f"-> {a['regime']}")
    print(f"  verdict: {verdict['regime']} | {verdict['interpretation']}")
    return out


def _verdict(per_domain: dict) -> dict:
    def med(key):
        vals = [a[key] for a in per_domain.values() if a.get(key) is not None]
        return float(np.median(vals)) if vals else None

    m_ov = med("consecutive_overlap")
    m_ang = med("consecutive_mean_angle")
    m_ratio = med("overlap_over_baseline")
    m_delta = med("delta_proxy")
    m_erd = med("effective_rotation_dim")

    if m_ov is None:
        return {"regime": "NO_DATA", "interpretation": "no domains measured"}

    summary = {
        "consecutive_overlap": round(m_ov, 4),
        "consecutive_mean_angle": round(m_ang, 4) if m_ang is not None else 0.0,
        "overlap_over_baseline": round(m_ratio, 2) if m_ratio is not None else 0.0,
        "total_drift_overlap": round(1.0 - m_delta, 4) if m_delta is not None else 1.0,
    }
    cls = classify_frame(summary)
    regimes = [a["regime"] for a in per_domain.values() if "regime" in a]
    agree = len(set(regimes)) == 1

    interp = {
        "STATIONARY": ("Frame stationary across depth -> the EXACT whitening "
                       "branch (whitening_preconditioner_zeroes_contraction) is "
                       "the right model of B2; depth implements a fixed-basis "
                       "transported-Gram preconditioner."),
        "SLOWLY_ROTATING": ("Frame coherent but slowly rotating -> the IMPERFECT "
                            "branch governs (imperfect_preconditioning_near_newton, "
                            "contraction <= delta^2 with delta ~ delta_proxy), and "
                            "depth_refines_preconditioner predicts the residual "
                            "shrinks with depth. Exact whitening does NOT hold; the "
                            "B2 operator identity is approximate, bounded by the "
                            "cross-layer coupling. Consistent with F0' (directional "
                            "relevance weak)."),
        "INCOHERENT": ("No stable transport frame -> the B2 preconditioner picture "
                       "does not apply in a fixed or slowly-moving basis; the "
                       "operator lift needs a per-layer (non-stationary) treatment."),
    }[cls["regime"]]

    return {
        "regime": cls["regime"],
        "b2_branch": cls["b2_branch"],
        "median_consecutive_overlap": round(m_ov, 4),
        "median_mean_principal_angle": round(m_ang, 4) if m_ang is not None else None,
        "median_overlap_over_baseline": round(m_ratio, 2) if m_ratio is not None else None,
        "median_delta_proxy": round(m_delta, 4) if m_delta is not None else None,
        "median_effective_rotation_dim": round(m_erd, 3) if m_erd is not None else None,
        "domains_agree": agree,
        "n_domains_measured": len(per_domain),
        "interpretation": interp,
    }


def run_model(model_name, n_samples=DEFAULT_N, seed=42, k=DEFAULT_K,
              iters=DEFAULT_ITERS, cuts=DEFAULT_CUTS) -> dict:
    return run_with_context(load_context(model_name), n_samples, seed, k, iters, cuts)


# ==========================================================================
# OOP front (probe_base.Probe) + registration
# ==========================================================================
class MovingFrameStationarityProbe(Probe):
    name = "moving_frame_stationarity"
    classifier_features: list[str] = []   # geometry measurement, no classifier
    label_key = None

    def synthetic(self, seed: int = 0) -> dict:
        return run_synthetic(seed=seed)

    def run_model(self, model_name: str, *, n_samples: int = DEFAULT_N,
                  seed: int = 42, k: int = DEFAULT_K, iters: int = DEFAULT_ITERS,
                  cuts: int = DEFAULT_CUTS, **_) -> dict:
        return run_model(model_name, n_samples, seed, k, iters, cuts)

    def run_on(self, ctx, *, n_samples: int = DEFAULT_N, seed: int = 42,
               k: int = DEFAULT_K, iters: int = DEFAULT_ITERS,
               cuts: int = DEFAULT_CUTS, **_) -> dict:
        return run_with_context(ctx, n_samples, seed, k, iters, cuts)


register_probe(MovingFrameStationarityProbe())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-samples", type=int, default=DEFAULT_N)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--cuts", type=int, default=DEFAULT_CUTS)
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic()
        return
    for m in (args.models or DEFAULT_MODELS):
        try:
            run_model(m, args.n_samples, k=args.k, iters=args.iters, cuts=args.cuts)
        except Exception as e:  # noqa: BLE001
            print(f"{m}: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
