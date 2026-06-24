#!/usr/bin/env python3
"""Plastic curvature redesign — non-degenerate H2/B3 on real models (dispatch 1839).

Part 1 of dispatch_20260624_1839_home_plastic_curvature_redesign. Replaces the
degenerate synthetic-basis commutator in run_geometry_h2b3b2 with:

  1a. Task-vector H2: Hendel-style readout gradients on WIDELY varied demo pairs
      (induction / recall / factual / multi_induction), commutator from single-demo
      task directions through the capacity-layer transport.

  1b. Plaquette B3: four-corner context loop (single-A, single-B, AB, BA) with
      top readout-direction holonomy vs spectral gap at the loop centre.

Reports gap-axis spread (min/max), n used, Spearman, explicit verdict.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DISPATCH = "dispatch_20260624_1839_home_plastic_curvature_redesign"
H2_FAMILIES = ("induction", "recall", "factual", "multi_induction")


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


def _spread(xs):
    xs = [float(x) for x in xs if np.isfinite(x)]
    if not xs:
        return None, None, None
    return float(min(xs)), float(max(xs)), float(np.std(xs))


def _verdict_h2(rho, n, x_spread):
    if rho is None or n < 8:
        return "still-degenerate (too few points)"
    if x_spread is not None and x_spread < 1e-6:
        return "still-degenerate (zero order-gap variance)"
    if rho >= 0.35:
        return "signal (order-gap tracks commutator)"
    if abs(rho) <= 0.15:
        return "null (no association)"
    return "weak/mixed"


def _verdict_b3(rho, n, gap_spread):
    if rho is None or n < 8:
        return "still-degenerate (too few points)"
    if gap_spread is not None and gap_spread < 1e-6:
        return "still-degenerate (zero gap variance)"
    if rho >= 0.35:
        return "signal (holonomy ~ 1/gap^2)"
    if abs(rho) <= 0.15:
        return "null (curvature law not resolved)"
    return "weak/mixed"


def _commutator_norm(apply, v_a, v_b, probe):
    Ta = apply(v_a.reshape(-1, 1))[:, 0]
    Tb = apply(v_b.reshape(-1, 1))[:, 0]
    comm = Ta * float(v_b @ probe) - Tb * float(v_a @ probe)
    return float(np.linalg.norm(comm))


def _line_angle(u, v):
    c = float(np.clip(abs(float(u @ v)), 0.0, 1.0))
    return float(np.arccos(c))


def _make_varied_prompt(task_family, tok, rng, block_len, n_blocks=1):
    from plastic_instrument_probe import _make_prompt

    if task_family == "multi_induction":
        return _make_prompt(task_family, tok, rng, block_len, n_blocks=max(2, n_blocks))
    return _make_prompt(task_family, tok, rng, block_len)


def _single_demo_prompt(task_family, tok, rng, block_len):
    """One demonstration block before the query token."""
    ids, pos, target = _make_varied_prompt(task_family, tok, rng, block_len, n_blocks=1)
    if pos < 2:
        return None
    demo_end = max(2, pos // 2)
    short = ids[:demo_end] + [ids[pos]]
    return short, len(short) - 1, target


def _run_h2(ctx, layers, l_c, rng, *, n_pairs=120, block_len_range=(6, 14)):
    from jacobian_product_spectrum_probe import transport_matvecs
    from plastic_instrument_probe import _task_direction_readout
    from singular_spectrum_probe import _capacity_layer_index

    _ = _capacity_layer_index(len(layers))
    order_gaps, comms = [], []
    tries = 0
    while len(order_gaps) < n_pairs and tries < n_pairs * 40:
        tries += 1
        fa = H2_FAMILIES[int(rng.integers(0, len(H2_FAMILIES)))]
        fb = H2_FAMILIES[int(rng.integers(0, len(H2_FAMILIES)))]
        bl_a = int(rng.integers(block_len_range[0], block_len_range[1] + 1))
        bl_b = int(rng.integers(block_len_range[0], block_len_range[1] + 1))
        nb_a = int(rng.integers(2, 4)) if fa == "multi_induction" else 1
        nb_b = int(rng.integers(2, 4)) if fb == "multi_induction" else 1

        pa = _make_varied_prompt(fa, ctx.tok, rng, bl_a, n_blocks=nb_a)
        pb = _make_varied_prompt(fb, ctx.tok, rng, bl_b, n_blocks=nb_b)
        if pa is None or pb is None:
            continue
        ids_a, pos_a, tgt_a = pa
        ids_b, pos_b, tgt_b = pb
        if pos_a < 4 or pos_b < 4:
            continue

        demo_a = ids_a[:pos_a]
        demo_b = ids_b[:pos_b]
        ids_ab = demo_a + demo_b + [ids_a[pos_a]]
        ids_ba = demo_b + demo_a + [ids_b[pos_b]]
        pos_q = len(ids_ab) - 1
        target = ids_a[pos_a]

        sa = _single_demo_prompt(fa, ctx.tok, rng, bl_a)
        sb = _single_demo_prompt(fb, ctx.tok, rng, bl_b)
        if sa is None or sb is None:
            continue
        w_a = _task_direction_readout(ctx, sa[0], sa[1], sa[2])
        w_b = _task_direction_readout(ctx, sb[0], sb[1], sb[2])
        if w_a is None or w_b is None:
            continue

        try:
            g_ab = _task_direction_readout(ctx, ids_ab, pos_q, target)
            g_ba = _task_direction_readout(ctx, ids_ba, pos_q, target)
            if g_ab is None or g_ba is None:
                continue
            order_gap = g_ab - g_ba
            og_norm = float(np.linalg.norm(order_gap))
            if og_norm < 1e-12:
                continue
            apply, _apply_t, _, d = transport_matvecs(ctx, layers, l_c, ids_ab, pos_q)
            if len(w_a) != d or len(w_b) != d:
                continue
            comm = _commutator_norm(apply, w_a, w_b, order_gap)
            order_gaps.append(og_norm)
            comms.append(comm)
        except Exception:  # noqa: BLE001
            continue

    rho, reason = _spearman(order_gaps, comms)
    og_min, og_max, og_std = _spread(order_gaps)
    return {
        "spearman_order_gap_vs_commutator": rho,
        "reason": reason,
        "n_paired": len(order_gaps),
        "order_gap_min": og_min,
        "order_gap_max": og_max,
        "order_gap_std": og_std,
        "verdict": _verdict_h2(rho, len(order_gaps), og_std),
    }


def _top_direction_and_gap(ctx, layers, l_idx, input_ids, pos, target):
    from cross_layer_transport_probe import capture_all_layers, layer_readout_subspace

    caps = capture_all_layers(ctx.model, layers, ctx.norm, ctx.unembed,
                              input_ids, pos, target, ctx.device)
    if l_idx not in caps:
        return None, None
    hq, g = caps[l_idx]
    _S, rho, V = layer_readout_subspace([hq], [g], max(2, 2))
    e = (np.asarray(_S) ** 2) * (np.asarray(rho) ** 2)
    if len(e) < 2 or e[0] <= 0:
        return None, None
    gap = float((e[0] - e[1]) / e[0])
    v = V[0] / (np.linalg.norm(V[0]) + 1e-30)
    return v, gap


def _loop_holonomy(angles):
    """Signed holonomy magnitude from four principal angles (radians)."""
    if len(angles) < 4:
        return None
    total = float(sum(angles))
    return abs(math.atan2(math.sin(total), math.cos(total)))


def _run_b3_plaquette(ctx, layers, l_c, rng, *, n_loops=100, block_len_range=(6, 14)):
    from singular_spectrum_probe import _capacity_layer_index

    _ = _capacity_layer_index(len(layers))
    log_hol, log_inv_g2 = [], []
    gaps_raw = []
    tries = 0
    layer_offsets = (0, 1, -1)

    while len(log_hol) < n_loops and tries < n_loops * 50:
        tries += 1
        fa = H2_FAMILIES[int(rng.integers(0, len(H2_FAMILIES)))]
        fb = H2_FAMILIES[int(rng.integers(0, len(H2_FAMILIES)))]
        bl_a = int(rng.integers(block_len_range[0], block_len_range[1] + 1))
        bl_b = int(rng.integers(block_len_range[0], block_len_range[1] + 1))

        pa = _make_varied_prompt(fa, ctx.tok, rng, bl_a, n_blocks=1)
        pb = _make_varied_prompt(fb, ctx.tok, rng, bl_b, n_blocks=1)
        sa = _single_demo_prompt(fa, ctx.tok, rng, bl_a)
        sb = _single_demo_prompt(fb, ctx.tok, rng, bl_b)
        if pa is None or pb is None or sa is None or sb is None:
            continue

        ids_a, pos_a, tgt_a = pa
        ids_b, pos_b, tgt_b = pb
        demo_a, demo_b = ids_a[:pos_a], ids_b[:pos_b]
        ids_ab = demo_a + demo_b + [ids_a[pos_a]]
        ids_ba = demo_b + demo_a + [ids_b[pos_b]]
        pos_ab = len(ids_ab) - 1
        target = ids_a[pos_a]

        l_use = int(np.clip(l_c + layer_offsets[int(rng.integers(0, len(layer_offsets)))],
                            1, len(layers) - 2))
        corners = [
            (sa[0], sa[1], sa[2]),
            (sb[0], sb[1], sb[2]),
            (ids_ab, pos_ab, target),
            (ids_ba, pos_ab, target),
        ]
        vecs, gaps_c = [], []
        ok = True
        for ids, pos, tgt in corners:
            v, g = _top_direction_and_gap(ctx, layers, l_use, ids, pos, tgt)
            if v is None:
                ok = False
                break
            vecs.append(v)
            gaps_c.append(g)
        if not ok or len(vecs) != 4:
            continue

        angles = [_line_angle(vecs[i], vecs[(i + 1) % 4]) for i in range(4)]
        hol = _loop_holonomy(angles)
        gap = float(gaps_c[2])  # centre = AB context
        if hol is None or hol < 1e-12 or gap < 1e-8:
            continue
        log_hol.append(math.log(hol))
        log_inv_g2.append(math.log(1.0 / (gap * gap)))
        gaps_raw.append(gap)

    rho, reason = _spearman(log_inv_g2, log_hol)
    g_min, g_max, g_std = _spread(gaps_raw)
    return {
        "spearman_log_inv_gap2_vs_log_holonomy": rho,
        "reason": reason,
        "n_points": len(log_hol),
        "gap_min": g_min,
        "gap_max": g_max,
        "gap_std": g_std,
        "verdict": _verdict_b3(rho, len(log_hol), g_std),
    }


def run_model(base, revision, *, n_pairs=120, n_loops=100, seed=42):
    from icl_convergence_probe import safe_name
    from probe_base import load_context
    from singular_spectrum_probe import _capacity_layer_index

    spec = f"{base}@{revision}" if revision else base
    print(f"\n=== plastic curvature redesign {spec} ===")
    ctx = load_context(spec)
    layers = ctx.layers
    l_c = _capacity_layer_index(len(layers))
    rng = np.random.default_rng(seed)

    h2 = _run_h2(ctx, layers, l_c, rng, n_pairs=n_pairs)
    b3 = _run_b3_plaquette(ctx, layers, l_c, rng, n_loops=n_loops)

    out = {
        "mode": "plastic_curvature_redesign",
        "probe": "plastic_curvature_redesign_probe",
        "base": base,
        "revision": revision,
        "dispatch": DISPATCH,
        "date": time.strftime("%Y-%m-%d"),
        "params": {"n_pairs": n_pairs, "n_loops": n_loops, "seed": seed, "capacity_layer": l_c},
        "H2_task_vector": h2,
        "B3_plaquette": b3,
    }
    path = HERE / f"plastic_curvature_redesign_{safe_name(base)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print(f"  H2: {h2['verdict']}  (n={h2['n_paired']}, og_std={h2.get('order_gap_std')})")
    print(f"  B3: {b3['verdict']}  (n={b3['n_points']}, gap=[{b3.get('gap_min')},{b3.get('gap_max')}])")
    return out


def main():
    ap = argparse.ArgumentParser(description="Plastic curvature redesign probe (dispatch 1839)")
    ap.add_argument("--base", default="EleutherAI/pythia-160m")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--n-pairs", type=int, default=120)
    ap.add_argument("--n-loops", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    bases = args.models if args.models else [args.base]
    for base in bases:
        run_model(base, args.revision, n_pairs=args.n_pairs, n_loops=args.n_loops, seed=args.seed)


if __name__ == "__main__":
    main()
