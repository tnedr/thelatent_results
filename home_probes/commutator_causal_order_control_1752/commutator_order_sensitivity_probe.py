#!/usr/bin/env python3
"""Commutator -> ICL prompt-order sensitivity bridge (dispatch 1000).

Links the verified plastic_operator H2 commutator (1839 task-vector probe) to
behavioral order-sensitivity: does operator non-commutativity predict how much
the model's output changes when demo order is swapped (AB vs BA)?
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("ICL_ATTN_IMPLEMENTATION", "eager")

HERE = Path(__file__).resolve().parent
DISPATCH = "dispatch_20260625_1000_home_commutator_order_sensitivity"

from plastic_curvature_redesign_probe import (  # noqa: E402
    _commutator_norm,
    _ordered_pair_prompts,
    _single_demo_prompt,
    _spearman,
    _spread,
)


def _partial_spearman(x, y, z):
    """Spearman(x, y) residualized against rank(z)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    if len(x) < 8:
        return None
    rz = np.argsort(np.argsort(z))
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    # rank-linear residualize
    def _resid(a, b):
        b1 = np.polyfit(b, a, 1)
        return a - np.polyval(b1, b)
    rx2 = _resid(rx.astype(float), rz.astype(float))
    ry2 = _resid(ry.astype(float), rz.astype(float))
    if float(np.std(rx2)) < 1e-12 or float(np.std(ry2)) < 1e-12:
        return None
    return round(float(np.corrcoef(rx2, ry2)[0, 1]), 4)


def _permutation_z(x, y, *, n_rep=200, seed=0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 8:
        return None
    real = float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_rep):
        perm = rng.permutation(y)
        null.append(float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(perm)))[0, 1]))
    null = np.asarray(null)
    sd = float(null.std(ddof=1))
    if sd < 1e-12:
        return None
    return round((real - float(null.mean())) / sd, 3)


def _symmetric_kl(p, q, eps=1e-12):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    kl_pq = float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))
    kl_qp = float(np.sum(q * (np.log(q + eps) - np.log(p + eps))))
    return 0.5 * (kl_pq + kl_qp)


def _query_logits(ctx, input_ids, pos):
    import torch

    device = ctx.device
    ids = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        out = ctx.model(input_ids=ids)
        logits = out.logits[0, pos, :].float().cpu().numpy()
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return logits, probs


def _behavioral_order_gap(ctx, ids_ab, ids_ba, pos_q, query):
    logits_ab, probs_ab = _query_logits(ctx, ids_ab, pos_q)
    logits_ba, probs_ba = _query_logits(ctx, ids_ba, pos_q)
    sym_kl = _symmetric_kl(probs_ab, probs_ba)
    logit_diff = abs(float(logits_ab[query] - logits_ba[query]))
    pred_ab = int(np.argmax(logits_ab))
    pred_ba = int(np.argmax(logits_ba))
    acc_flip = int(pred_ab != pred_ba)
    return {
        "symmetric_kl": sym_kl,
        "logit_diff": logit_diff,
        "acc_flip": acc_flip,
        "primary": sym_kl,
    }


def _commutator_for_pair(ctx, layers, l_c, pair, rng):
    from jacobian_product_spectrum_probe import transport_matvecs
    from plastic_operator_partb_real import _task_vec_layer

    fa, fb, bl_a, bl_b, ids_ab, ids_ba, pos_q, query, _da, _db = pair
    sa = _single_demo_prompt(fa, ctx.tok, rng, bl_a, query=query)
    sb = _single_demo_prompt(fb, ctx.tok, rng, bl_b, query=query)
    if sa is None or sb is None:
        return None
    w_a = _task_vec_layer(ctx, layers, l_c, sa[0], sa[1], query)
    w_b = _task_vec_layer(ctx, layers, l_c, sb[0], sb[1], query)
    if w_a is None or w_b is None:
        return None
    v_ab = _task_vec_layer(ctx, layers, l_c, ids_ab, pos_q, query)
    v_ba = _task_vec_layer(ctx, layers, l_c, ids_ba, pos_q, query)
    if v_ab is None or v_ba is None:
        return None
    order_gap = v_ab - v_ba
    og_norm = float(np.linalg.norm(order_gap))
    if og_norm < 1e-10:
        return None
    apply, _apply_t, _, d = transport_matvecs(ctx, layers, l_c, ids_ab, pos_q)
    if len(w_a) != d or len(w_b) != d:
        return None
    comm = _commutator_norm(apply, w_a, w_b, order_gap)
    mag = float(np.linalg.norm(w_a) + np.linalg.norm(w_b))
    return {
        "commutator": comm,
        "internal_order_gap": og_norm,
        "magnitude_w_sum": mag,
        "families": [fa, fb],
    }


def _bridge_verdict(rho, z, rho_partial, n, beh_std):
    if rho is None or n < 8:
        return "still-degenerate (too few pairs)"
    if beh_std is not None and beh_std < 1e-8:
        return "still-degenerate (zero behavioral spread)"
    if z is not None and abs(z) >= 2.0 and rho is not None and abs(rho) >= 0.2:
        if rho_partial is not None and abs(rho_partial) >= 0.15:
            return "PASS (commutator predicts behavioral order-gap)"
        if rho_partial is not None and abs(rho_partial) <= 0.1:
            return "NULL (magnitude control absorbs correlation)"
    if rho is not None and abs(rho) <= 0.12:
        return "NULL (no bridge correlation)"
    return "weak/mixed"


def run_model(model_name, *, n_pairs=80, seed=0, revision=None):
    from icl_convergence_probe import safe_name
    from probe_base import load_context
    from singular_spectrum_probe import _capacity_layer_index

    spec = model_name
    if revision and "@" not in model_name and "pythia" in model_name:
        spec = f"{model_name}@{revision}"
    print(f"\n=== commutator order sensitivity {spec} seed={seed} ===")
    ctx = load_context(spec)
    layers = ctx.layers
    l_c = _capacity_layer_index(len(layers))
    rng = np.random.default_rng(seed)

    comms, beh_primary, beh_kl, beh_logit, beh_flip = [], [], [], [], []
    mags, int_gaps = [], []
    tries = 0

    while len(comms) < n_pairs and tries < n_pairs * 50:
        tries += 1
        pair = _ordered_pair_prompts(ctx, ctx.tok, rng)
        if pair is None:
            continue
        _fa, _fb, _bla, _blb, ids_ab, ids_ba, pos_q, query, _da, _db = pair
        try:
            beh = _behavioral_order_gap(ctx, ids_ab, ids_ba, pos_q, query)
            if beh["primary"] < 1e-12:
                continue
            cinfo = _commutator_for_pair(ctx, layers, l_c, pair, rng)
            if cinfo is None:
                continue
            comms.append(cinfo["commutator"])
            beh_primary.append(beh["primary"])
            beh_kl.append(beh["symmetric_kl"])
            beh_logit.append(beh["logit_diff"])
            beh_flip.append(beh["acc_flip"])
            mags.append(cinfo["magnitude_w_sum"])
            int_gaps.append(cinfo["internal_order_gap"])
        except Exception:  # noqa: BLE001
            continue

    rho, reason = _spearman(comms, beh_primary)
    rho_kl, _ = _spearman(comms, beh_kl)
    rho_logit, _ = _spearman(comms, beh_logit)
    rho_partial = _partial_spearman(comms, beh_primary, mags)
    z_perm = _permutation_z(comms, beh_primary, seed=seed)
    b_min, b_max, b_std = _spread(beh_primary)
    c_min, c_max, c_std = _spread(comms)

    bridge = {
        "spearman_comm_vs_behavioral_kl": rho,
        "spearman_comm_vs_symmetric_kl": rho_kl,
        "spearman_comm_vs_logit_diff": rho_logit,
        "partial_spearman_after_magnitude": rho_partial,
        "permutation_z": z_perm,
        "reason": reason,
        "n_pairs": len(comms),
        "behavioral_kl_min": b_min,
        "behavioral_kl_max": b_max,
        "behavioral_kl_std": b_std,
        "commutator_min": c_min,
        "commutator_max": c_max,
        "commutator_std": c_std,
        "mean_acc_flip_rate": round(float(np.mean(beh_flip)), 4) if beh_flip else None,
        "verdict": _bridge_verdict(rho, z_perm, rho_partial, len(comms), b_std),
    }

    out = {
        "mode": "commutator_order_sensitivity",
        "probe": "commutator_order_sensitivity_probe",
        "model": model_name,
        "hf_spec": spec,
        "revision": revision,
        "dispatch": DISPATCH,
        "date": time.strftime("%Y-%m-%d"),
        "params": {"n_pairs": n_pairs, "seed": seed, "capacity_layer": l_c},
        "bridge": bridge,
        "h2_internal": {
            "spearman_internal_gap_vs_comm": _spearman(int_gaps, comms)[0],
            "n": len(comms),
        },
    }
    safe = safe_name(model_name)
    path = HERE / f"commutator_order_sensitivity_{safe}_seed{seed}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print(f"  bridge: {bridge['verdict']}  rho={rho} z={z_perm} partial={rho_partial} n={len(comms)}")
    return out


def aggregate():
    rows = []
    for p in sorted(HERE.glob("commutator_order_sensitivity_*_seed*.json")):
        d = json.loads(p.read_text())
        if d.get("mode") != "commutator_order_sensitivity":
            continue
        b = d.get("bridge") or {}
        rows.append({
            "model": d["model"],
            "seed": d["params"]["seed"],
            "n_pairs": b.get("n_pairs"),
            "spearman": b.get("spearman_comm_vs_behavioral_kl"),
            "partial_spearman": b.get("partial_spearman_after_magnitude"),
            "permutation_z": b.get("permutation_z"),
            "verdict": b.get("verdict"),
        })
    out = {"mode": "summary", "date": time.strftime("%Y-%m-%d"), "runs": rows}
    path = HERE / "commutator_order_sensitivity_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}  ({len(rows)} runs)")
    for r in rows:
        print(f"  {r['model']:16s} seed={r['seed']} rho={r['spearman']} z={r['permutation_z']} -> {r['verdict']}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Commutator order sensitivity (dispatch 1000)")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-pairs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="gpt2 only, few pairs")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
        return
    if args.smoke:
        run_model("gpt2", n_pairs=12, seed=0)
        return
    models = args.models or ["gpt2", "pythia-410m", "pythia-1b", "pythia-2.8b"]
    rev = None if "gpt2" in models and len(models) == 1 else args.revision
    for m in models:
        run_model(m, n_pairs=args.n_pairs, seed=args.seed,
                  revision=None if m == "gpt2" else args.revision)


if __name__ == "__main__":
    main()
