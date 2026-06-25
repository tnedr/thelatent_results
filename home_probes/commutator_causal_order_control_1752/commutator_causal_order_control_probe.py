#!/usr/bin/env python3
"""Causal control of ICL order-sensitivity via commutator edit (dispatch 1752).

Steers the capacity-layer residual along the H2 commutator direction and measures
whether behavioral prompt-order gap (sym KL) follows a monotone dose-response.
Reuses the 1000 probe's pair construction and order-gap metric.
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
DISPATCH = "dispatch_20260625_1752_home_commutator_causal_order_control"
DEFAULT_ALPHAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
EDIT_TYPES = ("commutator", "random", "magnitude")

from commutator_order_sensitivity_probe import (  # noqa: E402
    _behavioral_order_gap,
    _commutator_norm,
    _permutation_z,
    _spearman,
)
from plastic_curvature_redesign_probe import (  # noqa: E402
    _ordered_pair_prompts,
    _single_demo_prompt,
)


def _commutator_vector(apply, w_a, w_b, order_gap):
    Ta = apply(w_a.reshape(-1, 1))[:, 0]
    Tb = apply(w_b.reshape(-1, 1))[:, 0]
    probe = order_gap / (np.linalg.norm(order_gap) + 1e-30)
    comm = Ta * float(w_b @ probe) - Tb * float(w_a @ probe)
    n = float(np.linalg.norm(comm))
    if n < 1e-12:
        return None
    return comm / n


def _pair_directions(ctx, layers, l_c, pair, rng):
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
    if float(np.linalg.norm(order_gap)) < 1e-10:
        return None
    apply, _apply_t, _, d = transport_matvecs(ctx, layers, l_c, ids_ab, pos_q)
    if len(w_a) != d:
        return None
    c_dir = _commutator_vector(apply, w_a, w_b, order_gap)
    if c_dir is None:
        return None
    sym = w_a + w_b
    sn = float(np.linalg.norm(sym))
    m_dir = sym / sn if sn > 1e-12 else None
    r_dir = rng.standard_normal(d)
    r_dir = r_dir / (np.linalg.norm(r_dir) + 1e-30)
    return {
        "ids_ab": ids_ab,
        "ids_ba": ids_ba,
        "pos_q": pos_q,
        "query": query,
        "commutator_dir": c_dir,
        "magnitude_dir": m_dir,
        "random_dir": r_dir,
        "commutator_norm": _commutator_norm(apply, w_a, w_b, order_gap),
    }


def _steer_hook(direction, alpha, pos_q, scale_frac):
    import torch

    d_np = np.asarray(direction, dtype=float)

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h_new = h.clone()
        row = h_new[0, pos_q, :].float()
        step = float(alpha) * float(scale_frac) * float(torch.norm(row).item())
        edit = torch.tensor(d_np * step, dtype=row.dtype, device=row.device)
        h_new[0, pos_q, :] = row + edit
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return hook


def _behavioral_with_edit(ctx, layers, l_c, pair_dirs, direction, alpha, *, scale_frac=0.1):
    import torch

    ids_ab = pair_dirs["ids_ab"]
    ids_ba = pair_dirs["ids_ba"]
    pos_q = pair_dirs["pos_q"]
    query = pair_dirs["query"]
    target = layers[l_c]
    handle = None
    try:
        if alpha != 0.0 and direction is not None:
            handle = target.register_forward_hook(
                _steer_hook(direction, alpha, pos_q, scale_frac))
        beh = _behavioral_order_gap(ctx, ids_ab, ids_ba, pos_q, query)
        device = ctx.device
        with torch.no_grad():
            logits_ab = ctx.model(
                input_ids=torch.tensor([ids_ab], device=device)).logits[0, pos_q, :]
            logit_correct = float(logits_ab[query].cpu())
            acc = int(int(torch.argmax(logits_ab).item()) == query)
        beh["logit_correct"] = logit_correct
        beh["acc"] = acc
        return beh
    finally:
        if handle is not None:
            handle.remove()


def _collect_pairs(ctx, layers, l_c, rng, n_pairs):
    pairs = []
    tries = 0
    while len(pairs) < n_pairs and tries < n_pairs * 50:
        tries += 1
        pair = _ordered_pair_prompts(ctx, ctx.tok, rng)
        if pair is None:
            continue
        pd = _pair_directions(ctx, layers, l_c, pair, rng)
        if pd is None:
            continue
        pairs.append(pd)
    return pairs


def _dose_response(ctx, layers, l_c, pairs, alphas, *, scale_frac=0.1):
    out = {et: {"alphas": list(alphas), "mean_order_gap": [], "mean_logit_correct": [],
                "mean_acc": []} for et in EDIT_TYPES}
    for alpha in alphas:
        for et in EDIT_TYPES:
            gaps, lcs, accs = [], [], []
            for pd in pairs:
                if et == "commutator":
                    direction = pd["commutator_dir"]
                elif et == "magnitude":
                    direction = pd["magnitude_dir"]
                    if direction is None:
                        continue
                else:
                    direction = pd["random_dir"]
                beh = _behavioral_with_edit(
                    ctx, layers, l_c, pd, direction, alpha, scale_frac=scale_frac)
                gaps.append(beh["primary"])
                lcs.append(beh["logit_correct"])
                accs.append(beh["acc"])
            out[et]["mean_order_gap"].append(round(float(np.mean(gaps)), 6) if gaps else None)
            out[et]["mean_logit_correct"].append(round(float(np.mean(lcs)), 6) if lcs else None)
            out[et]["mean_acc"].append(round(float(np.mean(accs)), 6) if accs else None)
    return out


def _causal_verdict(dr, alphas):
    a = np.asarray(alphas, float)
    comm = np.asarray(dr["commutator"]["mean_order_gap"], float)
    rand = np.asarray(dr["random"]["mean_order_gap"], float)
    mag = np.asarray(dr["magnitude"]["mean_order_gap"], float)
    if len(comm) < 4 or not np.all(np.isfinite(comm)):
        return "still-degenerate"
    rho_c, _ = _spearman(a, comm)
    rho_r, _ = _spearman(a, rand)
    rho_m, _ = _spearman(a, mag)
    z_c = _permutation_z(a, comm, seed=0)
    acc0 = dr["commutator"]["mean_acc"][list(alphas).index(0.0)]
    acc_min = min(x for x in dr["commutator"]["mean_acc"] if x is not None)
    acc_ok = acc_min is not None and acc0 is not None and acc_min >= max(0.0, acc0 - 0.25)
    gap0_idx = list(alphas).index(0.0)
    gap0 = comm[gap0_idx]
    neg_idx = int(np.argmin(a))
    reduction = float(gap0 - comm[neg_idx]) if np.isfinite(gap0) else None
    if (rho_c is not None and rho_c >= 0.35 and z_c is not None and abs(z_c) >= 2.0
            and acc_ok and (rho_r is None or abs(rho_r) < 0.25)):
        return "PASS (causal dose-response)"
    if rho_c is not None and abs(rho_c) <= 0.15:
        return "NULL (no causal effect)"
    return "weak/mixed"


def run_model(model_name, *, n_pairs=50, seed=0, revision=None, alphas=None, scale_frac=0.1):
    from icl_convergence_probe import safe_name
    from probe_base import load_context
    from singular_spectrum_probe import _capacity_layer_index

    alphas = tuple(alphas or DEFAULT_ALPHAS)
    spec = model_name
    if revision and "@" not in model_name and "pythia" in model_name:
        spec = f"{model_name}@{revision}"
    print(f"\n=== commutator causal order control {spec} seed={seed} ===")
    ctx = load_context(spec)
    layers = ctx.layers
    l_c = _capacity_layer_index(len(layers))
    rng = np.random.default_rng(seed)

    pairs = _collect_pairs(ctx, layers, l_c, rng, n_pairs)
    if len(pairs) < 8:
        print(f"  only {len(pairs)} pairs — abort")
        return None
    dr = _dose_response(ctx, layers, l_c, pairs, alphas, scale_frac=scale_frac)
    rho_c, _ = _spearman(list(alphas), dr["commutator"]["mean_order_gap"])
    rho_r, _ = _spearman(list(alphas), dr["random"]["mean_order_gap"])
    z_c = _permutation_z(list(alphas), dr["commutator"]["mean_order_gap"], seed=seed)
    verdict = _causal_verdict(dr, alphas)

    out = {
        "mode": "commutator_causal_order_control",
        "probe": "commutator_causal_order_control_probe",
        "model": model_name,
        "hf_spec": spec,
        "dispatch": DISPATCH,
        "date": time.strftime("%Y-%m-%d"),
        "params": {
            "n_pairs": len(pairs),
            "seed": seed,
            "capacity_layer": l_c,
            "alphas": list(alphas),
            "scale_frac": scale_frac,
        },
        "dose_response": dr,
        "causal": {
            "spearman_alpha_vs_order_gap_comm": rho_c,
            "spearman_alpha_vs_order_gap_random": rho_r,
            "permutation_z_comm": z_c,
            "verdict": verdict,
        },
    }
    safe = safe_name(model_name)
    path = HERE / f"commutator_causal_order_control_{safe}_seed{seed}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print(f"  causal: {verdict}  rho_comm={rho_c} rho_rand={rho_r} z={z_c} n={len(pairs)}")
    return out


def aggregate():
    rows = []
    for p in sorted(HERE.glob("commutator_causal_order_control_*_seed*.json")):
        d = json.loads(p.read_text())
        if d.get("mode") != "commutator_causal_order_control":
            continue
        c = d.get("causal") or {}
        rows.append({
            "model": d["model"],
            "seed": d["params"]["seed"],
            "n_pairs": d["params"]["n_pairs"],
            "spearman_comm": c.get("spearman_alpha_vs_order_gap_comm"),
            "spearman_random": c.get("spearman_alpha_vs_order_gap_random"),
            "z_comm": c.get("permutation_z_comm"),
            "verdict": c.get("verdict"),
        })
    out = {"mode": "summary", "date": time.strftime("%Y-%m-%d"), "runs": rows}
    path = HERE / "commutator_causal_order_control_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}  ({len(rows)} runs)")
    for r in rows:
        print(f"  {r['model']:16s} seed={r['seed']} rho={r['spearman_comm']} -> {r['verdict']}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Commutator causal order control (dispatch 1752)")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-pairs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
        return
    if args.smoke:
        run_model("pythia-410m", n_pairs=10, seed=0, revision=args.revision,
                  alphas=(-1.0, 0.0, 1.0))
        return
    models = args.models or ["pythia-410m", "pythia-1b", "pythia-2.8b", "gpt2"]
    for m in models:
        run_model(m, n_pairs=args.n_pairs, seed=args.seed,
                  revision=None if m == "gpt2" else args.revision)


if __name__ == "__main__":
    main()
