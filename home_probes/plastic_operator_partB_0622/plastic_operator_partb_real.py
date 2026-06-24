#!/usr/bin/env python3
"""Part B real-model geometry — layer clay map + H2/B3/B2 (dispatch 0758).

Object-side clay_index per layer (T3) and holonomy-lift tests (H2, B3, B2) on
real Pythia models. Uses Hendel-style task vectors (grad of target logit w.r.t.
layer readout) and the plastic instrument transport oracles where needed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DISPATCH = "dispatch_20260622_0758_home_plastic_operator_partB"


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


def run_layer_clay(base, revision, *, n_prompts=8, n_orders=8, block_len=8, seed=42):
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
        "base": base, "revision": revision, "dispatch": DISPATCH,
        "date": time.strftime("%Y-%m-%d"),
        "params": {"n_prompts": n_prompts, "n_orders": n_orders, "block_len": block_len},
        "clay_index_per_layer": clay_map,
    }
    path = HERE / f"plastic_operator_layer_clay_{safe_name(base)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    return out


def run_geometry_h2b3b2(base, revision, *, n_pairs=40, block_len=8, seed=42, gauge_n=20):
    from icl_convergence_probe import safe_name
    from jacobian_product_spectrum_probe import transport_matvecs, topk_singular_values
    from plastic_instrument_probe import _make_prompt, _task_direction_readout
    from probe_base import load_context
    from routing_selection_probe import _vocab_lohi
    from singular_spectrum_probe import _capacity_layer_index

    spec = f"{base}@{revision}" if revision else base
    print(f"\n=== geometry H2/B3/B2 {spec} ===")
    ctx = load_context(spec)
    layers = ctx.layers
    l_c = _capacity_layer_index(len(layers))
    rng = np.random.default_rng(seed)
    lo, hi = _vocab_lohi(ctx.tok)

    h2_gaps, h2_comms = [], []
    b3_clay, b3_inv_gap2 = [], []
    b2_devs = []

    for _ in range(n_pairs):
        ids, pos, target = _make_prompt("induction", ctx.tok, rng, block_len)
        if pos < 6:
            continue
        mid = pos // 2
        seg_a = ids[:mid]
        seg_b = ids[mid:pos]
        ids_ab = seg_a + seg_b + [ids[pos]]
        ids_ba = seg_b + seg_a + [ids[pos]]
        pos_q = len(ids_ab) - 1

        g_ab = _task_direction_readout(ctx, ids_ab, pos_q, target)
        g_ba = _task_direction_readout(ctx, ids_ba, pos_q, target)
        if g_ab is None or g_ba is None:
            continue
        order_gap = g_ab - g_ba
        h2_gaps.append(float(np.linalg.norm(order_gap)))

        try:
            apply, apply_t, _, d = transport_matvecs(ctx, layers, l_c, ids_ab, pos_q)
            v_a = np.zeros(d)
            v_a[: min(len(seg_a), d)] = 1.0
            v_a /= np.linalg.norm(v_a) + 1e-30
            v_b = np.zeros(d)
            v_b[: min(len(seg_b), d)] = 1.0
            v_b /= np.linalg.norm(v_b) + 1e-30
            Ta = apply(v_a.reshape(-1, 1))[:, 0]
            Tb = apply(v_b.reshape(-1, 1))[:, 0]
            comm = Ta * (Tb @ v_b) - Tb * (Ta @ v_a) if d > 1 else Ta - Tb
            h2_comms.append(float(np.linalg.norm(comm[: min(8, d)])))
        except Exception:  # noqa: BLE001
            pass

        g = _task_direction_readout(ctx, ids_ab, pos_q, target)
        if g is not None:
            apply, apply_t, _, d = transport_matvecs(ctx, layers, l_c, ids_ab, pos_q)
            _sigma, Vr = topk_singular_values(
                apply, apply_t, int(d), min(3, d), iters=30, seed=0, want_vecs=True)
            gap = float(_sigma[0] - _sigma[1]) if len(_sigma) > 1 else float(_sigma[0])
            clay_proxy = float(np.linalg.norm(order_gap))
            if gap > 1e-8:
                b3_clay.append(clay_proxy)
                b3_inv_gap2.append(1.0 / (gap * gap))

    corr_h2 = float(np.corrcoef(h2_gaps, h2_comms)[0, 1]) if len(h2_gaps) > 3 else None
    corr_b3 = float(np.corrcoef(b3_clay, b3_inv_gap2)[0, 1]) if len(b3_clay) > 3 else None

    for _ in range(gauge_n):
        X = rng.standard_normal((8, 8))
        _, s, Vt = np.linalg.svd(X, full_matrices=False)
        signs = rng.choice([-1.0, 1.0], size=Vt.shape[0])
        Vt2 = Vt * signs[:, None]
        recon1 = (s[:, None] * Vt).sum(axis=0)
        recon2 = (s[:, None] * Vt2).sum(axis=0)
        b2_devs.append(float(np.linalg.norm(recon1 - recon2)))

    out = {
        "mode": "geometry_h2b3b2", "probe": "plastic_operator_partb",
        "base": base, "revision": revision, "dispatch": DISPATCH,
        "date": time.strftime("%Y-%m-%d"),
        "H2": {"order_gap_vs_commutator_corr": round(corr_h2, 4) if corr_h2 else None,
               "n_pairs": len(h2_gaps)},
        "B3": {"clay_proxy_vs_inv_gap2_corr": round(corr_b3, 4) if corr_b3 else None,
               "n_points": len(b3_clay)},
        "B2": {"gauge_invariance_max_dev": round(float(max(b2_devs)), 6) if b2_devs else None,
               "pass": bool(max(b2_devs) < 1e-5) if b2_devs else None},
    }
    path = HERE / f"plastic_operator_geometry_{safe_name(base)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="EleutherAI/pythia-160m")
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--layer-clay", action="store_true")
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.layer_clay:
        run_layer_clay(args.base, args.revision, seed=args.seed)
    if args.geometry:
        run_geometry_h2b3b2(args.base, args.revision, seed=args.seed)
    if not args.layer_clay and not args.geometry:
        run_layer_clay(args.base, args.revision, seed=args.seed)
        run_geometry_h2b3b2(args.base, args.revision, seed=args.seed)


if __name__ == "__main__":
    main()
