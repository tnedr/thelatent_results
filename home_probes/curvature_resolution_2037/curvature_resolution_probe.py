#!/usr/bin/env python3
"""Curvature resolution — gap regime + coupling-weighted curvature (dispatch 2037).

Decides H-genuine vs H-probe for the 1/gap^2 holonomy mystery on real models.
Reuses 1839 pair construction, transport_matvecs, and O2-style spectral curvature.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("ICL_ATTN_IMPLEMENTATION", "eager")

HERE = Path(__file__).resolve().parent
DISPATCH = "dispatch_20260625_2037_home_curvature_resolution"

from plastic_curvature_redesign_probe import (  # noqa: E402
    _commutator_norm,
    _ordered_pair_prompts,
    _single_demo_prompt,
    _spearman,
    _spread,
)
from plastic_operator_probe import _berry_fhs, _berry_spectral  # noqa: E402


def _transport_gram_eigs(apply, apply_t, d, *, k=4, iters=15):
    from jacobian_product_spectrum_probe import topk_singular_values

    sigma, vr = topk_singular_values(
        apply, apply_t, int(d), min(k, d), iters=iters, seed=0, want_vecs=True)
    if len(sigma) < 2 or sigma[0] < 1e-12:
        return None
    gap = float((sigma[0] - sigma[1]) / sigma[0])
    u0 = vr[:, 0] / (np.linalg.norm(vr[:, 0]) + 1e-30)
    u1 = vr[:, 1] / (np.linalg.norm(vr[:, 1]) + 1e-30)
    return {
        "sigma": sigma.tolist(),
        "gap": gap,
        "u0": u0,
        "u1": u1,
        "gram_eigs": (sigma ** 2).tolist(),
    }


def _coupling(u0, u1, w_diff):
    w = np.asarray(w_diff, float)
    n = float(np.linalg.norm(w))
    if n < 1e-12:
        return 0.0
    w = w / n
    return float(abs(np.dot(u0, w)) * abs(np.dot(u1, w)))


def _weighted_curvature(gap, coupling):
    g = max(float(gap), 1e-8)
    return float(coupling ** 2 / (g * g))


def _subspace_curvature(u0, u1, w_a, w_b):
    """O2-style 2-level curvature in the (u0,u1) subspace with task perturbation."""
    w_a = np.asarray(w_a, float)
    w_b = np.asarray(w_b, float)
    wa = w_a / (np.linalg.norm(w_a) + 1e-30)
    wb = w_b / (np.linalg.norm(w_b) + 1e-30)
    A2 = np.array([[np.dot(u0, wa), np.dot(u0, wb)],
                   [np.dot(u1, wa), np.dot(u1, wb)]], float)
    B2 = np.array([[np.dot(u0, wb), np.dot(u0, wa)],
                   [np.dot(u1, wb), np.dot(u1, wa)]], float)
    H0 = np.diag([1.0, 0.5]).astype(complex)
    try:
        f_spec = abs(_berry_spectral(H0, A2, B2, n=0))
        f_fhs = abs(_berry_fhs(H0, A2, B2, n=0, h=1e-2))
    except Exception:  # noqa: BLE001
        return None, None
    return float(f_spec), float(f_fhs)


def _pair_task_vectors(ctx, layers, l_c, pair, rng):
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
    return {
        "ids_ab": ids_ab,
        "ids_ba": ids_ba,
        "pos_q": pos_q,
        "query": query,
        "demo_a": _da,
        "demo_b": _db,
        "w_a": w_a,
        "w_b": w_b,
        "fa": fa,
        "fb": fb,
    }


def _interp_ids(demo_a, demo_b, query, alpha, rng):
    """Coupled forced-degeneracy family: vary demo mix before shared query."""
    n_a = max(1, int(round(float(alpha) * 4)))
    n_b = max(1, 4 - n_a)
    rng.shuffle(demo_a)
    rng.shuffle(demo_b)
    chunk_a = demo_a[: max(2, len(demo_a) * n_a // 4)]
    chunk_b = demo_b[: max(2, len(demo_b) * n_b // 4)]
    ids = list(chunk_a) + list(chunk_b) + [query]
    return ids, len(ids) - 1


def _sample_natural(ctx, layers, l_c, rng, *, n_target, block_len_range=(6, 14), max_tries=None):
    from jacobian_product_spectrum_probe import transport_matvecs

    rows = []
    tries = 0
    cap = max_tries if max_tries is not None else n_target * 50
    t0 = time.time()
    while len(rows) < n_target and tries < cap:
        tries += 1
        pair = _ordered_pair_prompts(ctx, ctx.tok, rng, block_len_range)
        if pair is None:
            if tries % 10 == 0:
                print(f"  part_A tries={tries} ok=0/{n_target} ({time.time()-t0:.0f}s)",
                      flush=True)
            continue
        info = _pair_task_vectors(ctx, layers, l_c, pair, rng)
        if info is None:
            continue
        try:
            apply, apply_t, _, d = transport_matvecs(
                ctx, layers, l_c, info["ids_ab"], info["pos_q"])
            spec = _transport_gram_eigs(apply, apply_t, d)
            if spec is None:
                continue
            w_diff = info["w_b"] - info["w_a"]
            coup = _coupling(spec["u0"], spec["u1"], w_diff)
            wc = _weighted_curvature(spec["gap"], coup)
            f_spec, f_fhs = _subspace_curvature(
                spec["u0"], spec["u1"], info["w_a"], info["w_b"])
            comm = _commutator_norm(apply, info["w_a"], info["w_b"], w_diff)
            rows.append({
                "family_a": info["fa"],
                "family_b": info["fb"],
                "gap": spec["gap"],
                "coupling": coup,
                "weighted_curvature": wc,
                "spectral_curvature": f_spec,
                "fhs_curvature": f_fhs,
                "commutator": comm,
                "log_inv_gap2": math.log(1.0 / (spec["gap"] ** 2)),
                "log_weighted": math.log(max(wc, 1e-30)),
            })
            if len(rows) % 5 == 0 or len(rows) == n_target or len(rows) == 1:
                print(f"  part_A {len(rows)}/{n_target}  tries={tries}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        except Exception:  # noqa: BLE001
            continue
    return rows


def _forced_degeneracy_curve(ctx, layers, l_c, rng, *, n_pairs=20):
    from jacobian_product_spectrum_probe import transport_matvecs

    alphas = [0.0, 0.5, 1.0]
    curves = []
    tries = 0
    while len(curves) < n_pairs and tries < n_pairs * 40:
        tries += 1
        pair = _ordered_pair_prompts(ctx, ctx.tok, rng)
        if pair is None:
            continue
        info = _pair_task_vectors(ctx, layers, l_c, pair, rng)
        if info is None:
            continue
        pts = []
        for alpha in alphas:
            ids, pos = _interp_ids(
                list(info["demo_a"]), list(info["demo_b"]), info["query"], alpha, rng)
            if pos < 2:
                continue
            try:
                apply, apply_t, _, d = transport_matvecs(ctx, layers, l_c, ids, pos)
                spec = _transport_gram_eigs(apply, apply_t, d)
                if spec is None:
                    break
                w_diff = info["w_b"] - info["w_a"]
                coup = _coupling(spec["u0"], spec["u1"], w_diff)
                wc = _weighted_curvature(spec["gap"], coup)
                pts.append({
                    "alpha": alpha,
                    "gap": spec["gap"],
                    "coupling": coup,
                    "weighted_curvature": wc,
                })
            except Exception:  # noqa: BLE001
                break
        if len(pts) >= 3:
            curves.append({"alphas": alphas, "points": pts})
    # pool across curves for slope
    gaps, wcs = [], []
    for c in curves:
        for p in c["points"]:
            if p["gap"] > 1e-8 and p["weighted_curvature"] > 0:
                gaps.append(p["gap"])
                wcs.append(p["weighted_curvature"])
    slope = None
    if len(gaps) >= 8:
        slope = float(np.polyfit(np.log(gaps), np.log(wcs), 1)[0])
    return {"curves": curves, "n_curves": len(curves), "log_gap_vs_wc_slope": slope}


def _gap_census(rows):
    gaps = [r["gap"] for r in rows]
    coups = [r["coupling"] for r in rows]
    wcs = [r["weighted_curvature"] for r in rows]
    g_min, g_max, g_std = _spread(gaps)
    med = float(np.median(gaps)) if gaps else None
    ratio = (g_min / med) if (g_min is not None and med and med > 1e-12) else None
    coup_at_small = [
        r["coupling"] for r in rows
        if med is not None and r["gap"] <= med * 1.5]
    return {
        "n": len(rows),
        "gap_min": g_min,
        "gap_max": g_max,
        "gap_median": med,
        "gap_min_over_median": round(ratio, 4) if ratio is not None else None,
        "coupling_median": round(float(np.median(coups)), 4) if coups else None,
        "coupling_at_small_gap_median": (
            round(float(np.median(coup_at_small)), 4) if coup_at_small else None),
        "weighted_curvature_median": round(float(np.median(wcs)), 4) if wcs else None,
    }


def _verdict(census, forced, part_c):
    slope = forced.get("log_gap_vs_wc_slope")
    gmin_ratio = census.get("gap_min_over_median")
    rho_comm_wc = part_c.get("spearman_commutator_vs_weighted")
    bounded_gap = gmin_ratio is not None and gmin_ratio > 0.05
    flat_forced = slope is None or slope >= -0.5
    spike_forced = slope is not None and slope <= -1.5
    comm_tracks = rho_comm_wc is not None and rho_comm_wc >= 0.25

    if spike_forced:
        return "H-probe (forced degeneracy recovers 1/gap^2 pole)"
    if bounded_gap and flat_forced and comm_tracks:
        return "H-genuine (bounded coupling-weighted curvature; commutator real)"
    if bounded_gap and flat_forced:
        return "H-genuine (bounded curvature; commutator link weak)"
    if slope is None:
        return "HONEST NULL (estimator noise / insufficient forced-gap range)"
    return "weak/mixed"


def run_model(model_name, *, revision="step143000", n_samples=80, seed=0, smoke=False):
    from icl_convergence_probe import MODEL_ALIASES, safe_name
    from probe_base import load_context
    from singular_spectrum_probe import _capacity_layer_index

    base = MODEL_ALIASES.get(model_name, model_name)
    spec = f"{base}@{revision}" if revision and "pythia" in base.lower() else base
    if smoke:
        n_samples = min(n_samples, 4)
    print(f"\n=== curvature resolution {model_name} ({spec}) seed={seed} ===")
    ctx = load_context(spec)
    layers = ctx.layers
    l_c = _capacity_layer_index(len(layers))
    rng = np.random.default_rng(seed)

    try_cap = 30 if smoke else None
    part_a_rows = _sample_natural(
        ctx, layers, l_c, rng, n_target=n_samples, max_tries=try_cap)
    print(f"  part_A done n={len(part_a_rows)}", flush=True)
    census = _gap_census(part_a_rows)
    print("  part_B forced degeneracy...", flush=True)
    forced = _forced_degeneracy_curve(
        ctx, layers, l_c, rng, n_pairs=max(3, n_samples // 3 if smoke else n_samples // 4))

    comms = [r["commutator"] for r in part_a_rows]
    wcs = [r["weighted_curvature"] for r in part_a_rows]
    fhs = [r["fhs_curvature"] for r in part_a_rows if r["fhs_curvature"] is not None]
    rho_cw, _ = _spearman(comms, wcs)
    log_g2 = [r["log_inv_gap2"] for r in part_a_rows]
    log_wc = [r["log_weighted"] for r in part_a_rows]
    rho_gw, _ = _spearman(log_g2, log_wc)
    part_c = {
        "spearman_commutator_vs_weighted": rho_cw,
        "spearman_log_inv_gap2_vs_weighted": rho_gw,
        "n": len(part_a_rows),
    }

    verdict = _verdict(census, forced, part_c)
    out = {
        "mode": "curvature_resolution",
        "probe": "curvature_resolution_probe",
        "model": model_name,
        "hf_spec": spec,
        "dispatch": DISPATCH,
        "date": time.strftime("%Y-%m-%d"),
        "params": {"n_samples": n_samples, "seed": seed, "capacity_layer": l_c},
        "part_A_census": census,
        "part_A_rows": part_a_rows if smoke else None,
        "part_B_forced_degeneracy": forced,
        "part_C_reconcile": part_c,
        "verdict": verdict,
    }
    path = HERE / f"curvature_resolution_{safe_name(model_name)}_seed{seed}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print(f"  census n={census['n']} gap_min/med={census.get('gap_min_over_median')} "
          f"forced_slope={forced.get('log_gap_vs_wc_slope')}")
    print(f"  part_C rho(comm,wc)={rho_cw}  verdict: {verdict}")
    return out


def aggregate():
    files = sorted(HERE.glob("curvature_resolution_*_seed*.json"))
    rows = []
    for p in files:
        d = json.loads(p.read_text())
        rows.append({
            "model": d["model"],
            "seed": d["params"]["seed"],
            "verdict": d["verdict"],
            "gap_min_over_median": d["part_A_census"].get("gap_min_over_median"),
            "forced_slope": d["part_B_forced_degeneracy"].get("log_gap_vs_wc_slope"),
            "rho_comm_wc": d["part_C_reconcile"].get("spearman_commutator_vs_weighted"),
        })
    summary = {"runs": rows, "n": len(rows), "date": time.strftime("%Y-%m-%d")}
    out = HERE / "curvature_resolution_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"saved -> {out.name} ({len(rows)} runs)")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Curvature resolution probe (dispatch 2037)")
    ap.add_argument("--models", nargs="+", default=["pythia-410m"])
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--n-samples", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
        return
    for m in args.models:
        rev = None if m == "gpt2" else args.revision
        run_model(m, revision=rev, n_samples=args.n_samples,
                  seed=args.seed, smoke=args.smoke)


if __name__ == "__main__":
    main()
