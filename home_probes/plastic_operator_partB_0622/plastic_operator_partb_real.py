#!/usr/bin/env python3
"""Part B real-model geometry — layer clay map + H2/B3/B2 (dispatch 0758 / fix 0835).

Object-side clay_index per layer (T3) and holonomy-lift tests (H2, B3, B2) on
real Pythia models. Uses Hendel-style task vectors and transport_matvecs JVP.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DISPATCH_PARTB = "dispatch_20260622_0758_home_plastic_operator_partB"
DISPATCH_FIX = "dispatch_20260624_0835_home_plastic_geometry_fix"


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 4 or len(x) != len(y):
        return None, "insufficient_n"
    if float(np.std(x)) < 1e-12:
        return None, "zero_variance_x"
    if float(np.std(y)) < 1e-12:
        return None, "zero_variance_y"
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return round(float(np.corrcoef(rx, ry)[0, 1]), 4), None


def _verdict_h2(rho, n, b2_pass):
    if not b2_pass:
        return "invalid (B2 gate failed)"
    if rho is None or n < 8:
        return "invalid (too few paired points)"
    if rho >= 0.35:
        return "signal (order-gap tracks commutator)"
    if abs(rho) <= 0.15:
        return "null (no association)"
    return "weak/mixed"


def _verdict_b3(rho, n, b2_pass):
    if not b2_pass:
        return "invalid (B2 gate failed)"
    if rho is None or n < 8:
        return "invalid (too few points or zero variance)"
    if rho >= 0.35:
        return "signal (clay ~ 1/gap^2)"
    if abs(rho) <= 0.15:
        return "null (curvature law not resolved)"
    return "weak/mixed"


def _task_vec_layer(ctx, layers, layer_idx, input_ids, pos, target):
    import torch

    device = ctx.device
    ids = torch.tensor([input_ids], device=device)
    leaf_holder: dict = {}

    def hook(_m, _i, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h_new = h.clone()
        leaf = h_new[0, pos, :].detach().clone().requires_grad_(True)
        h_new[0, pos, :] = leaf
        leaf_holder["leaf"] = leaf
        return (h_new,) + tuple(output[1:]) if is_tuple else h_new

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        out = ctx.model(input_ids=ids)
        logits = out.logits[0, pos, :]
        ctx.model.zero_grad(set_to_none=True)
        logits[target].backward()
        leaf = leaf_holder.get("leaf")
        if leaf is None or leaf.grad is None:
            return None
        g = leaf.grad.detach().cpu().numpy().astype(float)
    finally:
        handle.remove()
    n = float(np.linalg.norm(g))
    return None if n < 1e-30 else g / n


def _clay_from_vectors(vecs: list[np.ndarray]) -> float:
    W = np.stack(vecs)
    w_mean = W.mean(axis=0)
    spread = float(np.mean([np.linalg.norm(w - w_mean) for w in W]))
    return spread / (float(np.linalg.norm(w_mean)) + 1e-30)


def _commutator_norm(apply, v_a, v_b, probe):
    """Leading-order commutator of segment transport directions on probe vector."""
    Ta = apply(v_a.reshape(-1, 1))[:, 0]
    Tb = apply(v_b.reshape(-1, 1))[:, 0]
    comm = Ta * float(v_b @ probe) - Tb * float(v_a @ probe)
    return float(np.linalg.norm(comm))


def _b2_gauge_test(ctx, layers, l_c, rng, *, gauge_n=24, block_len=8):
    """Flip SVD frame signs; reported clay_proxy and inv_gap2 must be invariant."""
    from jacobian_product_spectrum_probe import transport_matvecs, topk_singular_values
    from plastic_instrument_probe import _make_prompt, _task_direction_readout
    from singular_spectrum_probe import _capacity_layer_index

    _ = _capacity_layer_index(len(layers))
    for _ in range(40):
        ids, pos, target = _make_prompt("induction", ctx.tok, rng, block_len)
        if pos < 6:
            continue
        mid = pos // 2
        seg_a, seg_b = ids[:mid], ids[mid:pos]
        ids_ab = seg_a + seg_b + [ids[pos]]
        pos_q = len(ids_ab) - 1
        g_ab = _task_direction_readout(ctx, ids_ab, pos_q, target)
        g_ba = _task_direction_readout(ctx, list(seg_b) + list(seg_a) + [ids[pos]], pos_q, target)
        if g_ab is None or g_ba is None:
            continue
        order_gap = g_ab - g_ba
        try:
            apply, apply_t, _, d = transport_matvecs(ctx, layers, l_c, ids_ab, pos_q)
            _sigma, Vr = topk_singular_values(
                apply, apply_t, int(d), min(3, d), iters=60, seed=0, want_vecs=True)
        except Exception:  # noqa: BLE001
            continue
        if len(_sigma) < 1:
            continue
        gap = float(_sigma[0] - _sigma[1]) if len(_sigma) > 1 else float(_sigma[0])
        if gap < 1e-8:
            continue
        base_clay = float(np.linalg.norm(order_gap))
        base_inv = 1.0 / (gap * gap)
        base_proj = float(np.linalg.norm(Vr.T @ order_gap))
        devs = []
        for _ in range(gauge_n):
            signs = rng.choice([-1.0, 1.0], size=Vr.shape[1])
            Vr2 = Vr * signs
            clay = float(np.linalg.norm(order_gap))
            inv_g2 = 1.0 / (gap * gap)
            proj = float(np.linalg.norm(Vr2.T @ order_gap))
            devs.append(max(
                abs(clay - base_clay),
                abs(inv_g2 - base_inv),
                abs(proj - base_proj),
            ))
        return devs
    return []


def run_layer_clay(base, revision, *, n_prompts=8, n_orders=8, block_len=8, seed=42,
                   dispatch=DISPATCH_PARTB):
    from icl_convergence_probe import safe_name
    from plastic_instrument_probe import _make_prompt
    from probe_base import load_context

    spec = f"{base}@{revision}" if revision else base
    print(f"\n=== layer clay map {spec} ===")
    ctx = load_context(spec)
    layers = ctx.layers
    n_layers = len(layers)
    rng = np.random.default_rng(seed)
    layer_clay = {str(l): [] for l in range(n_layers)}

    for _ in range(n_prompts):
        ids, pos, target = _make_prompt("induction", ctx.tok, rng, block_len)
        demo_len = pos
        if demo_len < 4:
            continue
        idx = np.arange(demo_len)
        for _ in range(n_orders):
            perm = rng.permutation(idx)
            shuffled = [ids[i] for i in perm] + ids[demo_len:]
            shuffled_pos = len(shuffled) - 1
            for l in range(n_layers):
                g = _task_vec_layer(ctx, layers, l, shuffled, shuffled_pos, target)
                if g is not None:
                    layer_clay[str(l)].append(g)

    clay_map = {}
    for l in range(n_layers):
        vecs = layer_clay[str(l)]
        clay_map[str(l)] = round(_clay_from_vectors(vecs), 6) if len(vecs) >= 4 else None

    out = {
        "mode": "layer_clay_map", "probe": "plastic_operator_partb",
        "base": base, "revision": revision, "dispatch": dispatch,
        "date": time.strftime("%Y-%m-%d"),
        "params": {"n_prompts": n_prompts, "n_orders": n_orders, "block_len": block_len},
        "clay_index_per_layer": clay_map,
    }
    path = HERE / f"plastic_operator_layer_clay_{safe_name(base)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    return out


def run_geometry_h2b3b2(base, revision, *, n_pairs=80, block_len=8, seed=42,
                          gauge_n=24, dispatch=DISPATCH_FIX):
    from icl_convergence_probe import safe_name
    from jacobian_product_spectrum_probe import transport_matvecs, topk_singular_values
    from plastic_instrument_probe import _make_prompt, _task_direction_readout
    from probe_base import load_context
    from singular_spectrum_probe import _capacity_layer_index

    spec = f"{base}@{revision}" if revision else base
    print(f"\n=== geometry H2/B3/B2 {spec} ===")
    ctx = load_context(spec)
    layers = ctx.layers
    l_c = _capacity_layer_index(len(layers))
    rng = np.random.default_rng(seed)

    h2_gaps, h2_comms = [], []
    b3_log_clay, b3_log_inv = [], []

    for _ in range(n_pairs):
        ids, pos, target = _make_prompt("induction", ctx.tok, rng, block_len)
        if pos < 6:
            continue
        mid = pos // 2
        seg_a = np.asarray(ids[:mid], dtype=float)
        seg_b = np.asarray(ids[mid:pos], dtype=float)
        ids_ab = ids[:mid] + ids[mid:pos] + [ids[pos]]
        ids_ba = ids[mid:pos] + ids[:mid] + [ids[pos]]
        pos_q = len(ids_ab) - 1

        try:
            g_ab = _task_direction_readout(ctx, ids_ab, pos_q, target)
            g_ba = _task_direction_readout(ctx, ids_ba, pos_q, target)
            if g_ab is None or g_ba is None:
                continue
            order_gap = g_ab - g_ba
            gap_norm = float(np.linalg.norm(order_gap))

            apply, apply_t, _, d = transport_matvecs(ctx, layers, l_c, ids_ab, pos_q)
            va = np.zeros(d)
            vb = np.zeros(d)
            va[: min(len(seg_a), d)] = seg_a[: min(len(seg_a), d)]
            vb[: min(len(seg_b), d)] = seg_b[: min(len(seg_b), d)]
            na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
            if na < 1e-12 or nb < 1e-12:
                continue
            va /= na
            vb /= nb
            comm_norm = _commutator_norm(apply, va, vb, order_gap)

            h2_gaps.append(gap_norm)
            h2_comms.append(comm_norm)

            sigma, _Vr = topk_singular_values(
                apply, apply_t, int(d), min(3, d), iters=60, seed=0, want_vecs=True)
            gap = float(sigma[0] - sigma[1]) if len(sigma) > 1 else float(sigma[0])
            if gap > 1e-8 and gap_norm > 1e-12:
                b3_log_clay.append(math.log(gap_norm))
                b3_log_inv.append(math.log(1.0 / (gap * gap)))
        except Exception:  # noqa: BLE001
            continue

    h2_rho, h2_reason = _spearman(h2_gaps, h2_comms)
    b3_rho, b3_reason = _spearman(b3_log_clay, b3_log_inv)

    b2_devs = _b2_gauge_test(ctx, layers, l_c, rng, gauge_n=gauge_n, block_len=block_len)
    b2_max = round(float(max(b2_devs)), 6) if b2_devs else None
    b2_pass = bool(b2_devs and max(b2_devs) < 1e-5)

    out = {
        "mode": "geometry_h2b3b2", "probe": "plastic_operator_partb",
        "base": base, "revision": revision, "dispatch": dispatch,
        "date": time.strftime("%Y-%m-%d"),
        "params": {"n_pairs": n_pairs, "gauge_n": gauge_n, "block_len": block_len},
        "H2": {
            "spearman_order_gap_vs_commutator": h2_rho,
            "reason": h2_reason,
            "n_paired": len(h2_gaps),
            "verdict": _verdict_h2(h2_rho, len(h2_gaps), b2_pass),
        },
        "B3": {
            "spearman_log_clay_vs_log_inv_gap2": b3_rho,
            "reason": b3_reason,
            "n_points": len(b3_log_clay),
            "verdict": _verdict_b3(b3_rho, len(b3_log_clay), b2_pass),
        },
        "B2": {
            "gauge_invariance_max_dev": b2_max,
            "pass": b2_pass,
            "n_flips": len(b2_devs),
            "verdict": "pass" if b2_pass else "fail (probe invalid)",
        },
    }
    path = HERE / f"plastic_operator_geometry_{safe_name(base)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print(f"  B2 pass={b2_pass}  H2={out['H2']['verdict']}  B3={out['B3']['verdict']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="EleutherAI/pythia-160m")
    ap.add_argument("--models", nargs="+", default=None,
                    help="geometry-only batch (0835): multiple bases")
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--layer-clay", action="store_true")
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--mode", choices=["geometry", "layer_clay", "all"], default=None)
    ap.add_argument("--n-pairs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dispatch-id", default=DISPATCH_FIX)
    args = ap.parse_args()

    mode = args.mode
    if mode is None:
        if args.geometry and not args.layer_clay:
            mode = "geometry"
        elif args.layer_clay and not args.geometry:
            mode = "layer_clay"
        elif args.layer_clay or args.geometry:
            mode = "all"
        else:
            mode = "all"

    bases = args.models if args.models else [args.base]
    for base in bases:
        if mode in ("layer_clay", "all"):
            run_layer_clay(base, args.revision, seed=args.seed, dispatch=args.dispatch_id)
        if mode in ("geometry", "all"):
            run_geometry_h2b3b2(
                base, args.revision, n_pairs=args.n_pairs, seed=args.seed,
                dispatch=args.dispatch_id)


if __name__ == "__main__":
    main()
