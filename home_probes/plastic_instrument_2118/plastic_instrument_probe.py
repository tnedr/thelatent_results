#!/usr/bin/env python3
"""Plastic instrument probe — using a PLASTIC OPERATOR as the MEASURING TOOL for a
transformer, instead of a fixed probe.

Concept: topics/ml_mathematics_of_trained_intelligence/RESEARCH_DIRECTION.md
         (the plastic operator, Tamas 2026-06-21). This is the DUAL of
         plastic_operator_probe.py:
           - plastic_operator_probe : the transformer IS a plastic operator (object)
           - plastic_instrument_probe: a plastic operator is the INSTRUMENT we probe
             the transformer WITH (to find what it really does and how).

WHY (today's lesson). The developmental_lyapunov home sweep got NO_LAMBDA_CROSSING
on all three Pythia scales: the GLOBAL top-sigma of the transport operator is
dominated by bulk forward expansion and is NOT the task direction — a FIXED,
task-blind instrument missed the structure. The right object (the task-direction
lambda) requires the instrument to FIND the task subspace, not assume it. A fixed
instrument on a plastic object sees a blur; the instrument's plasticity must match
the object's. This probe reproduces that failure in a box and fixes it.

THE INSTRUMENT (interaction-only — never opens the model). It may call three
oracles a real LM also exposes: apply(v)=T v (forward perturbation along v),
task_obs(z)=P_task z (read the task-relevant part of the response), and
apply_adj(u)=T^T u (a VJP / backprop). The PLASTIC probe reshapes its grip v
toward the direction the object is most TASK-sensitive in:

    v <- normalize( (1-alpha) v + alpha * normalize( T^T P_task T v ) )

i.e. online power iteration on the TASK-conditioned sensitivity operator
M = T^T P_task T, with a plasticity rate alpha (alpha<1 retains state -> can TRACK
a moving target; alpha=1 is plain power iteration). It converges to the input
direction that most drives the TASK output — the task subspace — discovered by
interaction, not assumed.

THE FIXED instruments it is compared against:
  - global (task-blind): top right-singular vector of T (power iteration on T^T T)
    — the analogue of today's global top-sigma. Picks the bulk direction.
  - task-aware snapshot: top eigvec of M_0 computed once, then HELD (task-aware but
    NOT plastic) — isolates the value of plasticity for a MOVING target.

TWO controlled demonstrations:
  (1) TASK-BLINDNESS (static): a planted transport operator with a high-gain
      task-IRRELEVANT bulk direction u_bulk and a moderate-gain task-RELEVANT
      direction u_task. The global probe aligns to u_bulk (misses u_task); the
      plastic probe recovers u_task. This is NO_LAMBDA_CROSSING in miniature.
  (2) PLASTICITY/TRACKING (moving): u_task(t) rotates across the interaction. The
      task-aware SNAPSHOT decays as the target moves; the plastic probe TRACKS it
      (moving camera for a moving subject).

SYNTHETIC gate (numpy, closed-form planted structure): fixed-global alignment to
u_task < 0.2 (misses); plastic alignment > 0.95 (recovers); on the moving target
the plastic mean alignment beats the snapshot by a clear margin and stays > 0.9.

Run:
  python3 plastic_instrument_probe.py --synthetic
  python3 plastic_instrument_probe.py
Output: plastic_instrument.json (+ plastic_instrument_synthetic.json)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# ==========================================================================
# Planted transport operator: a high-gain task-IRRELEVANT bulk + a moderate-gain
# task-RELEVANT direction. The minimal model of today's global-vs-task mismatch.
# ==========================================================================
def build_planted(rng, d, m, *, g_bulk=5.0, g_task=1.0, small=0.05, u_task=None):
    Q_in, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Q_out, _ = np.linalg.qr(rng.standard_normal((d, d)))
    u_bulk = Q_in[:, 0]
    if u_task is None:
        u_task = Q_in[:, 1]                       # task-relevant INPUT direction
    o_bulk = Q_out[:, 0]                          # task-IRRELEVANT output direction
    o_taskcols = Q_out[:, 1:1 + m]               # task output subspace (dim m)
    o_task0 = o_taskcols[:, 0]
    T = (g_bulk * np.outer(o_bulk, u_bulk)
         + g_task * np.outer(o_task0, u_task)
         + small * rng.standard_normal((d, d)))
    P_task = o_taskcols @ o_taskcols.T            # projection onto task output space
    return T, P_task, u_task, u_bulk


def _align(a, b):
    return abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# ==========================================================================
# Instruments (interaction-only: apply / task_obs / apply_adj).
# ==========================================================================
def fixed_global_probe(T, *, iters=100, seed=0):
    """Top right-singular vector of T (task-blind) — the analogue of global top-sigma."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(T.shape[1]); v /= np.linalg.norm(v)
    for _ in range(iters):
        v = T.T @ (T @ v)
        v /= (np.linalg.norm(v) + 1e-30)
    return v


def plastic_probe(T, P_task, *, iters=100, alpha=1.0, v0=None, seed=0):
    """Online power iteration on M = T^T P_task T, with plasticity rate alpha.
    alpha=1 -> plain power iteration (static recovery); alpha<1 -> retains grip
    (tracks a moving target). Interaction-only: uses apply (T@), task_obs (P_task@),
    apply_adj (T.T@)."""
    rng = np.random.default_rng(seed)
    v = v0.copy() if v0 is not None else rng.standard_normal(T.shape[1])
    v /= (np.linalg.norm(v) + 1e-30)
    for _ in range(iters):
        Mv = T.T @ (P_task @ (T @ v))            # apply_adj(task_obs(apply(v)))
        g = Mv / (np.linalg.norm(Mv) + 1e-30)
        v = (1.0 - alpha) * v + alpha * g
        v /= (np.linalg.norm(v) + 1e-30)
    return v


# ==========================================================================
# SYNTHETIC gate.
# ==========================================================================
def run_synthetic(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d, m = 24, 4

    # (1) static: global misses the task, plastic recovers it
    T, P_task, u_task, u_bulk = build_planted(rng, d, m)
    v_global = fixed_global_probe(T, iters=200, seed=1)
    v_plastic = plastic_probe(T, P_task, iters=200, alpha=1.0, seed=1)
    align_global = _align(v_global, u_task)
    align_plastic = _align(v_plastic, u_task)
    align_global_to_bulk = _align(v_global, u_bulk)

    # (2) moving target: snapshot decays, plastic tracks
    a, b = u_task, np.linalg.qr(rng.standard_normal((d, d)))[0][:, 2]
    b = b - (a @ b) * a; b /= np.linalg.norm(b)
    Tsteps = 30
    thetas = np.linspace(0.0, np.pi / 2, Tsteps)
    # snapshot: task-aware probe fixed at t=0
    T0, P0, ut0, _ = build_planted(rng, d, m, u_task=a)
    v_snap = plastic_probe(T0, P0, iters=200, alpha=1.0, seed=2)
    v_track = v_snap.copy()
    snap_al, track_al = [], []
    for th in thetas:
        ut = np.cos(th) * a + np.sin(th) * b
        Tt, Pt, _, _ = build_planted(rng, d, m, u_task=ut)
        v_track = plastic_probe(Tt, Pt, iters=3, alpha=0.4, v0=v_track, seed=3)
        snap_al.append(_align(v_snap, ut))
        track_al.append(_align(v_track, ut))
    snap_mean, track_mean = float(np.mean(snap_al)), float(np.mean(track_al))

    verdict = {
        "fixed_global_misses_task": bool(align_global < 0.2),
        "global_locks_onto_bulk": bool(align_global_to_bulk > 0.9),
        "plastic_recovers_task": bool(align_plastic > 0.95),
        "plastic_tracks_moving_target": bool(track_mean > snap_mean + 0.2 and track_mean > 0.9),
        "align_global_to_task": round(align_global, 4),
        "align_global_to_bulk": round(align_global_to_bulk, 4),
        "align_plastic_to_task": round(align_plastic, 4),
        "moving_snapshot_mean_align": round(snap_mean, 4),
        "moving_plastic_mean_align": round(track_mean, 4),
    }
    verdict["all_pass"] = bool(
        verdict["fixed_global_misses_task"] and verdict["global_locks_onto_bulk"]
        and verdict["plastic_recovers_task"] and verdict["plastic_tracks_moving_target"])

    out = {
        "mode": "synthetic", "probe": "plastic_instrument",
        "date": time.strftime("%Y-%m-%d"),
        "concept_anchor": "RESEARCH_DIRECTION.md plastic operator (instrument dual)",
        "dims": {"d": d, "m_task": m, "moving_steps": Tsteps}, "verdict": verdict,
    }
    path = HERE / "plastic_instrument_synthetic.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print("\n=== synthetic plastic-instrument verdict ===")
    for kk, vv in verdict.items():
        print(f"  {kk}: {vv}")
    return out


# ==========================================================================
# Full experiment: averaged recovery + tracking; report the failure-and-fix.
# ==========================================================================
def run_experiment(seed: int = 0, *, d=24, m=4, n_trials=60) -> dict:
    rng = np.random.default_rng(seed)
    ag, ap, ab = [], [], []
    snap_means, track_means = [], []

    for _ in range(n_trials):
        T, P_task, u_task, u_bulk = build_planted(rng, d, m)
        v_global = fixed_global_probe(T, iters=150, seed=int(rng.integers(1e6)))
        v_plastic = plastic_probe(T, P_task, iters=150, alpha=1.0,
                                  seed=int(rng.integers(1e6)))
        ag.append(_align(v_global, u_task))
        ap.append(_align(v_plastic, u_task))
        ab.append(_align(v_global, u_bulk))

        # moving target tracking
        a = u_task
        b = np.linalg.qr(rng.standard_normal((d, d)))[0][:, 2]
        b = b - (a @ b) * a; b /= np.linalg.norm(b)
        T0, P0, _, _ = build_planted(rng, d, m, u_task=a)
        v_snap = plastic_probe(T0, P0, iters=150, alpha=1.0, seed=int(rng.integers(1e6)))
        v_track = v_snap.copy()
        s_al, t_al = [], []
        for th in np.linspace(0.0, np.pi / 2, 24):
            ut = np.cos(th) * a + np.sin(th) * b
            Tt, Pt, _, _ = build_planted(rng, d, m, u_task=ut)
            v_track = plastic_probe(Tt, Pt, iters=3, alpha=0.4, v0=v_track,
                                    seed=int(rng.integers(1e6)))
            s_al.append(_align(v_snap, ut)); t_al.append(_align(v_track, ut))
        snap_means.append(float(np.mean(s_al))); track_means.append(float(np.mean(t_al)))

    def m_(a):
        return round(float(np.mean(a)), 4)

    ag_m, ap_m, ab_m = m_(ag), m_(ap), m_(ab)
    snap_m, track_m = m_(snap_means), m_(track_means)

    task_blind_fix = bool(ag_m < 0.25 and ap_m > 0.9 and ab_m > 0.85)
    tracking_win = bool(track_m > snap_m + 0.2 and track_m > 0.85)
    instrument_confirmed = bool(task_blind_fix and tracking_win)

    if instrument_confirmed:
        interp = (
            "PLASTIC_INSTRUMENT_CONFIRMED: a plastic (adaptive) instrument measures "
            "the object where a fixed one fails. TASK-BLINDNESS (today's "
            f"NO_LAMBDA_CROSSING in a box): the fixed global probe locks onto the "
            f"task-IRRELEVANT bulk (align-to-bulk {ab_m:.3f}) and MISSES the task "
            f"direction (align-to-task {ag_m:.3f}); the plastic, task-conditioned "
            f"probe RECOVERS it by interaction (align {ap_m:.3f}) — without being told "
            f"where it is. PLASTICITY/TRACKING: on a moving task direction the "
            f"task-aware SNAPSHOT decays (mean align {snap_m:.3f}) while the plastic "
            f"probe TRACKS it (mean align {track_m:.3f}). The instrument's plasticity "
            "must match the object's: a fixed instrument on a plastic object blurs; a "
            "plastic instrument co-moves and resolves. Interaction-only "
            "(apply/task_obs/apply_adj), never opens the model.")
    elif task_blind_fix:
        interp = (f"PARTIAL: task-blindness fixed (global {ag_m:.3f} -> plastic "
                  f"{ap_m:.3f}) but tracking margin weak (snapshot {snap_m:.3f}, "
                  f"plastic {track_m:.3f}).")
    else:
        interp = (f"WEAK: global {ag_m:.3f}, plastic {ap_m:.3f}, bulk {ab_m:.3f}, "
                  f"snapshot {snap_m:.3f}, track {track_m:.3f}.")

    verdict = {
        "instrument_confirmed": instrument_confirmed,
        "task_blindness_fixed": task_blind_fix,
        "tracking_win": tracking_win,
        "align_global_to_task": ag_m,
        "align_global_to_bulk": ab_m,
        "align_plastic_to_task": ap_m,
        "moving_snapshot_mean_align": snap_m,
        "moving_plastic_mean_align": track_m,
        "interpretation": interp,
    }

    out = {
        "mode": "experiment", "probe": "plastic_instrument",
        "date": time.strftime("%Y-%m-%d"),
        "concept_anchor": "RESEARCH_DIRECTION.md plastic operator (instrument dual)",
        "params": {"d": d, "m_task": m, "n_trials": n_trials},
        "verdict": verdict,
    }
    path = HERE / "plastic_instrument.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print("\n=== plastic-instrument experiment ===")
    print(f"  TASK-BLINDNESS: global->task {ag_m:.3f} (->bulk {ab_m:.3f}) | "
          f"plastic->task {ap_m:.3f}")
    print(f"  TRACKING (moving target): snapshot {snap_m:.3f} | plastic {track_m:.3f}")
    print(f"  verdict: {interp}")
    return out


# ==========================================================================
# REAL model — plastic instrument on Pythia (dispatch 2118).
# ==========================================================================
DEFAULT_BLOCK = 8
DEFAULT_EPS_FRACS = (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2)
DEFAULT_REAL_ITERS = 50
DEFAULT_REAL_PROMPTS = 24


def _task_direction_readout(ctx, input_ids, pos, target):
    """g = normalized grad of target logit w.r.t. final-norm readout repr."""
    import torch

    device = ctx.device
    ids = torch.tensor([input_ids], device=device)
    out = ctx.model(input_ids=ids, output_hidden_states=True)
    h_raw = out.hidden_states[-1][0, pos, :]
    z = ctx.norm(h_raw)
    z = z.detach().clone().requires_grad_(True)
    logits = ctx.unembed(z.unsqueeze(0)).squeeze(0)
    logits[target].backward()
    g = z.grad.detach().cpu().numpy().astype(float)
    n = float(np.linalg.norm(g))
    return (None if n < 1e-30 else g / n)


def _logprob_at_capacity(ctx, layers, l_c, input_ids, pos, target, h_vec):
    """Target-token log-prob with capacity-layer state set to h_vec."""
    import torch

    device = ctx.device
    ids = torch.tensor([input_ids], device=device)

    def inject(_m, _i, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h = h.clone()
        h[0, pos, :] = h_vec
        return (h,) + tuple(output[1:]) if is_tuple else h

    handle = layers[l_c].register_forward_hook(inject)
    try:
        with torch.no_grad():
            logits = ctx.model(input_ids=ids).logits[0, pos, :]
            lp = torch.log_softmax(logits.float(), dim=-1)[target]
        return float(lp.cpu())
    finally:
        handle.remove()


def _plastic_real_step(apply, apply_t, g, v, *, alpha=1.0):
    """One plastic iteration: v <- norm((1-a)v + a norm(T^T P_task T v))."""
    Tv = apply(v.reshape(-1, 1))[:, 0]
    coeff = float(g @ Tv)
    Pz = coeff * g
    Mv = apply_t(Pz.reshape(-1, 1))[:, 0]
    gdir = Mv / (np.linalg.norm(Mv) + 1e-30)
    v_new = (1.0 - alpha) * v + alpha * gdir
    return v_new / (np.linalg.norm(v_new) + 1e-30)


def _global_direction(apply, apply_t, d, iters):
    from jacobian_product_spectrum_probe import topk_singular_values

    _sigma, Vr = topk_singular_values(
        apply, apply_t, int(d), 1, iters=iters, seed=0, want_vecs=True)
    v = Vr[:, 0].astype(float)
    return v / (np.linalg.norm(v) + 1e-30)


def _causal_curve(ctx, layers, l_c, ids, pos, target, h_query, v_dir, eps_fracs):
    import torch

    base = _logprob_at_capacity(ctx, layers, l_c, ids, pos, target, h_query)
    v = v_dir / (np.linalg.norm(v_dir) + 1e-30)
    h_norm = float(h_query.detach().norm().cpu())
    curve = []
    for ef in eps_fracs:
        h_vec = h_query + (ef * h_norm) * torch.tensor(
            v, dtype=h_query.dtype, device=ctx.device)
        lp = _logprob_at_capacity(ctx, layers, l_c, ids, pos, target, h_vec)
        curve.append({"eps_frac": float(ef), "delta_logprob": round(lp - base, 6)})
    max_abs = max(abs(c["delta_logprob"]) for c in curve) if curve else 0.0
    return curve, round(base, 6), float(max_abs)


def _analyze_prompt(ctx, layers, l_c, ids, pos, target, *, iters, alpha, v0, eps_fracs):
    from jacobian_product_spectrum_probe import transport_matvecs

    g = _task_direction_readout(ctx, ids, pos, target)
    if g is None:
        return None
    apply, apply_t, h_query, d = transport_matvecs(ctx, layers, l_c, ids, pos)
    v_global = _global_direction(apply, apply_t, d, iters)
    v = (v0.copy() if v0 is not None
         else np.random.default_rng(pos + target).standard_normal(d))
    v = v / (np.linalg.norm(v) + 1e-30)
    for _ in range(iters):
        v = _plastic_real_step(apply, apply_t, g, v, alpha=alpha)
    if v @ v_global < 0:
        v = -v
    Tv_p = apply(v.reshape(-1, 1))[:, 0]
    Tv_g = apply(v_global.reshape(-1, 1))[:, 0]
    task_gain_pl = abs(float(g @ Tv_p))
    task_gain_gl = abs(float(g @ Tv_g))
    curve_p, base_lp, max_p = _causal_curve(
        ctx, layers, l_c, ids, pos, target, h_query, v, eps_fracs)
    curve_g, _, max_g = _causal_curve(
        ctx, layers, l_c, ids, pos, target, h_query, v_global, eps_fracs)
    return {
        "align_plastic_global": round(_align(v, v_global), 4),
        "task_gain_plastic": round(task_gain_pl, 6),
        "task_gain_global": round(task_gain_gl, 6),
        "task_gain_ratio": round(task_gain_pl / (task_gain_gl + 1e-30), 4),
        "causal_max_abs_plastic": round(max_p, 6),
        "causal_max_abs_global": round(max_g, 6),
        "causal_ratio": round(max_p / (max_g + 1e-30), 4),
        "baseline_logprob": base_lp,
        "eps_curve_plastic": curve_p,
        "eps_curve_global": curve_g,
        "v_plastic": v,
    }


def _real_verdict(agg):
    align = agg["align_plastic_global_mean"]
    causal_r = agg["causal_ratio_mean"]
    if causal_r > 3.0 and align < 0.5:
        interp = (
            "INSTRUMENT_REDEEMS: plastic direction is distinct from global "
            f"(align {align:.3f}) and causally drives induction "
            f"(|delta_plastic|/|delta_global| {causal_r:.2f} > 3).")
    elif align < 0.5 and causal_r < 1.5:
        interp = (
            "ALIGNED_BUT_NOT_CAUSAL: plastic found a different direction "
            f"(align {align:.3f}) but causal ratio {causal_r:.2f} ~ 1.")
    elif align >= 0.5:
        interp = (
            f"GLOBAL_WAS_FINE: plastic aligns with global top-sigma "
            f"(align {align:.3f}); 1015 failure may be normalization/sign.")
    else:
        interp = (f"MIXED: align {align:.3f}, causal ratio {causal_r:.2f}, "
                  f"task gain ratio {agg['task_gain_ratio_mean']:.2f}.")
    return {"interpretation": interp, "align_plastic_global_mean": align,
            "task_gain_ratio_mean": agg["task_gain_ratio_mean"],
            "causal_ratio_mean": causal_r}


def run_real_static(base, revision, *, n_prompts=DEFAULT_REAL_PROMPTS,
                    iters=DEFAULT_REAL_ITERS, k=8, seed=42,
                    block_len=DEFAULT_BLOCK, eps_fracs=DEFAULT_EPS_FRACS) -> dict:
    from icl_convergence_probe import safe_name
    from probe_base import load_context
    from routing_selection_probe import build_induction
    from singular_spectrum_probe import _capacity_layer_index

    spec = f"{base}@{revision}" if revision else base
    print(f"\n=== plastic instrument REAL static {spec} ===")
    ctx = load_context(spec)
    layers = ctx.layers
    l_c = _capacity_layer_index(len(layers))
    rng = np.random.default_rng(seed)
    per_prompt = []
    for i in range(n_prompts):
        ids, pos, target = build_induction(ctx.tok, rng, block_len)
        try:
            rec = _analyze_prompt(
                ctx, layers, l_c, ids, pos, target,
                iters=iters, alpha=1.0, v0=None, eps_fracs=eps_fracs)
        except Exception as e:  # noqa: BLE001
            print(f"  prompt {i} failed ({type(e).__name__}: {e})")
            continue
        if rec is None:
            continue
        per_prompt.append({k_: rec[k_] for k_ in rec if k_ != "v_plastic"})
        print(f"  prompt {i}: align={rec['align_plastic_global']:.3f} "
              f"task_gain_ratio={rec['task_gain_ratio']:.2f} "
              f"causal_ratio={rec['causal_ratio']:.2f}")
    if len(per_prompt) < 3:
        verdict = {"interpretation": "INSUFFICIENT_PROMPTS", "n_prompts": len(per_prompt)}
    else:
        def m(key):
            return round(float(np.mean([p[key] for p in per_prompt])), 4)
        agg = {"align_plastic_global_mean": m("align_plastic_global"),
               "task_gain_ratio_mean": m("task_gain_ratio"),
               "causal_ratio_mean": m("causal_ratio"),
               "task_gain_plastic_mean": m("task_gain_plastic"),
               "task_gain_global_mean": m("task_gain_global"),
               "n_prompts": len(per_prompt)}
        verdict = _real_verdict(agg)
        verdict.update(agg)
    out = {
        "mode": "real_static", "probe": "plastic_instrument",
        "base": base, "revision": revision, "date": time.strftime("%Y-%m-%d"),
        "dispatch": "dispatch_20260621_2118_home_plastic_instrument_real",
        "params": {"n_prompts": n_prompts, "iters": iters, "k": k,
                   "block_len": block_len, "eps_fracs": list(eps_fracs)},
        "per_prompt": per_prompt, "verdict": verdict,
    }
    path = HERE / f"plastic_instrument_real_{safe_name(base)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print(f"  verdict: {verdict.get('interpretation')}")
    return out


def run_real_track(base, revisions, *, n_prompts=DEFAULT_REAL_PROMPTS,
                   iters=30, k=8, seed=42, block_len=DEFAULT_BLOCK,
                   alpha=0.4, eps_fracs=DEFAULT_EPS_FRACS,
                   ref_revision="step143000") -> dict:
    from developmental_lyapunov_probe import _induction_accuracy, _step_of
    from icl_convergence_probe import safe_name
    from probe_base import load_context
    from routing_selection_probe import build_induction
    from singular_spectrum_probe import _capacity_layer_index

    ref_spec = f"{base}@{ref_revision}"
    ctx_ref = load_context(ref_spec)
    layers_ref = ctx_ref.layers
    l_c_ref = _capacity_layer_index(len(layers_ref))
    rng = np.random.default_rng(seed)
    prompts = [build_induction(ctx_ref.tok, rng, block_len) for _ in range(n_prompts)]
    v_refs = []
    for ids, pos, target in prompts:
        try:
            rec = _analyze_prompt(
                ctx_ref, layers_ref, l_c_ref, ids, pos, target,
                iters=iters, alpha=1.0, v0=None, eps_fracs=eps_fracs)
            v_refs.append(rec["v_plastic"] if rec else None)
        except Exception:  # noqa: BLE001
            v_refs.append(None)

    traj = []
    for rev in revisions:
        spec = f"{base}@{rev}"
        print(f"\n── track {spec} ──")
        ctx = load_context(spec)
        layers = ctx.layers
        l_c = _capacity_layer_index(len(layers))
        v_carry = None
        aligns, causal_d, task_ratios, stabs = [], [], [], []
        for i, (ids, pos, target) in enumerate(prompts):
            try:
                rec = _analyze_prompt(
                    ctx, layers, l_c, ids, pos, target,
                    iters=iters, alpha=alpha, v0=v_carry, eps_fracs=eps_fracs)
            except Exception:  # noqa: BLE001
                continue
            if rec is None:
                continue
            v_carry = rec["v_plastic"]
            v_ref = v_refs[i] if i < len(v_refs) else None
            if v_ref is not None:
                stabs.append(_align(v_carry, v_ref))
            aligns.append(rec["align_plastic_global"])
            causal_d.append(rec["causal_max_abs_plastic"])
            task_ratios.append(rec["task_gain_ratio"])
        acc = _induction_accuracy(ctx, n_prompts, seed, block_len)
        rec_ckpt = {
            "revision": rev, "step": _step_of(rev),
            "induction_acc": acc.get("induction_acc"),
            "align_plastic_global_mean": round(float(np.mean(aligns)), 4) if aligns else None,
            "causal_max_abs_plastic_mean": round(float(np.mean(causal_d)), 6) if causal_d else None,
            "task_gain_ratio_mean": round(float(np.mean(task_ratios)), 4) if task_ratios else None,
            "align_to_final_plastic_mean": round(float(np.mean(stabs)), 4) if stabs else None,
        }
        traj.append(rec_ckpt)
        print(f"  step={rec_ckpt['step']:.0f} acc={rec_ckpt['induction_acc']} "
              f"causal_pl={rec_ckpt['causal_max_abs_plastic_mean']} "
              f"stab={rec_ckpt['align_to_final_plastic_mean']}")
    out = {
        "mode": "real_track", "probe": "plastic_instrument",
        "base": base, "revisions": list(revisions),
        "date": time.strftime("%Y-%m-%d"),
        "dispatch": "dispatch_20260621_2118_home_plastic_instrument_real",
        "params": {"n_prompts": n_prompts, "iters": iters, "alpha": alpha, "k": k},
        "trajectory": traj,
    }
    path = HERE / f"plastic_instrument_real_{safe_name(base)}_track.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", action="store_true",
                    help="real Pythia plastic instrument (dispatch 2118)")
    ap.add_argument("--base", default="EleutherAI/pythia-410m")
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--revisions", nargs="+", default=None)
    ap.add_argument("--track", action="store_true",
                    help="carry plastic v across training checkpoints")
    ap.add_argument("--n-prompts", type=int, default=DEFAULT_REAL_PROMPTS)
    ap.add_argument("--iters", type=int, default=DEFAULT_REAL_ITERS)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--block-len", type=int, default=DEFAULT_BLOCK)
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic(seed=args.seed)
        return
    if args.real:
        if args.track:
            revs = args.revisions or ["step1000", "step2000", "step8000", "step143000"]
            run_real_track(args.base, revs, n_prompts=args.n_prompts,
                           iters=args.iters, k=args.k, seed=args.seed,
                           block_len=args.block_len)
        else:
            run_real_static(args.base, args.revision, n_prompts=args.n_prompts,
                            iters=args.iters, k=args.k, seed=args.seed,
                            block_len=args.block_len)
        return
    run_experiment(seed=args.seed, d=args.d, m=args.m, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
