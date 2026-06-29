#!/usr/bin/env python3
"""Next-token CE contraction via Activation Digest (dispatch 0534).

Scan once (digest cache), then logit-lens CE + Fisher curvature from cached (H,G).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASKS = ("induction", "relation_composition")
DEFAULT_MODELS = ("pythia-410m", "pythia-1.4b", "gpt2-medium")
DEFAULT_ETA = 1.0
DEFAULT_K = 4


def _spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _ce_trajectory_from_digest(dctx, task: str) -> list[float]:
    import torch

    dg = dctx.digest
    ctx = dctx.model_ctx
    if ctx is None:
        raise RuntimeError("DigestContext needs model_ctx for logit-lens CE")
    cids = (dg.meta.get("correct_ids") or {}).get(task)
    layers = dg.layers(task)
    if not layers or not cids:
        raise RuntimeError(f"digest missing layers or correct_ids for {task}")
    n = int(dg.H[task][layers[0]].shape[0])
    if len(cids) != n:
        raise RuntimeError(f"correct_ids len {len(cids)} != digest rows {n} for {task}")

    norm, unembed = ctx.norm, ctx.unembed
    device = ctx.device
    ce_sum = np.zeros(len(layers), dtype=float)
    with torch.no_grad():
        for li, L in enumerate(layers):
            H = dg.H[task][L]
            for j in range(n):
                h = torch.as_tensor(H[j], device=device, dtype=torch.float32)
                logits = unembed(norm(h))
                logp = torch.log_softmax(logits.float(), dim=-1)
                ce_sum[li] += float(-logp[int(cids[j])].item())
    return (ce_sum / n).tolist()


def _curvature_a(H: np.ndarray, G: np.ndarray, k: int) -> float:
    from cross_layer_transport_probe import layer_readout_subspace

    S, rho, _V = layer_readout_subspace(H, G, k)
    e = (np.asarray(S, float) ** 2) * (np.asarray(rho, float) ** 2)
    return float(np.max(e)) if e.size else float("nan")


def run_cell_from_digest(dg, dctx, domain: str, eta: float, k: int) -> dict:
    ce_traj = _ce_trajectory_from_digest(dctx, domain)
    layers = dg.layers(domain)
    a_by_layer = {}
    for L in layers:
        H, G = dg.HG(domain, L)
        if H.shape[0] >= 3:
            a_by_layer[L] = _curvature_a(H, G, k)

    max_a = max(a_by_layer.values()) if a_by_layer else 1.0
    theoretical, empirical, layer_pairs = [], [], []
    for i in sorted(a_by_layer)[:-1]:
        lj = i + 1
        if lj not in a_by_layer:
            continue
        li_idx = layers.index(i)
        lj_idx = layers.index(lj)
        a_norm = a_by_layer[i] / (max_a + 1e-12)
        rho_th = float(np.clip(1.0 - eta * a_norm, 0.0, 1.0))
        e_i, e_j = ce_traj[li_idx], ce_traj[lj_idx]
        rho_em = float(e_j / (e_i + 1e-12))
        delta = float(e_j - e_i)
        theoretical.append(rho_th)
        empirical.append(rho_em)
        layer_pairs.append({
            "layer_i": i, "layer_j": lj,
            "ce_i": round(e_i, 4), "ce_j": round(e_j, 4),
            "delta_ce": round(delta, 4),
        })

    delta_ce = [ce_traj[j + 1] - ce_traj[j] for j in range(len(ce_traj) - 1)]
    corr = _spearman(theoretical, empirical)
    mono = float(np.mean(np.diff(ce_traj) <= 1e-6)) if len(ce_traj) > 1 else float("nan")

    return {
        "model": dg.spec.model_name,
        "hf_name": dg.meta.get("hf_name"),
        "domain": domain,
        "digest_key": dg.spec.key(),
        "n_samples": int(dg.H[domain][layers[0]].shape[0]),
        "n_layers": int(dg.meta.get("n_layers", len(layers))),
        "eta": eta,
        "k": k,
        "ce_trajectory": [round(x, 4) for x in ce_traj],
        "a_by_decoder_layer": {str(ki): round(v, 6) for ki, v in a_by_layer.items()},
        "theoretical_rho_trajectory": [round(x, 4) for x in theoretical],
        "empirical_rho_trajectory": [round(x, 4) for x in empirical],
        "layer_pairs": layer_pairs,
        "delta_ce": [round(x, 4) for x in delta_ce],
        "corr_theoretical_empirical_rho": round(corr, 4) if math.isfinite(corr) else None,
        "ce_monotone_frac": round(mono, 4),
        "ce_final": round(ce_traj[-1], 4),
        "ce_drop_total": round(ce_traj[0] - ce_traj[-1], 4),
    }


def run_cell(model_name: str, domain: str, n_samples: int, seed: int,
             eta: float, k: int, rebuild: bool = False) -> dict:
    import activation_digest as ad
    from probe_base import load_context

    dg = ad.load_or_build_digest(
        model_name, tasks=(domain,), n_samples=n_samples, seed=seed,
        depth="ct", rebuild=rebuild)
    ctx = load_context(model_name)
    dctx = ad.DigestContext(dg, ctx)
    return run_cell_from_digest(dg, dctx, domain, eta, k)


def _verdict(cells: list[dict]) -> str:
    corrs = [c["corr_theoretical_empirical_rho"] for c in cells
             if c.get("corr_theoretical_empirical_rho") is not None]
    if not corrs:
        return "INCONCLUSIVE — no valid correlations."
    mean_c = float(np.mean(corrs))
    pos = sum(1 for c in corrs if c is not None and c > 0.3)
    if mean_c >= 0.4 and pos >= len(cells) * 0.6:
        return (f"SUPPORTED — mean Spearman={mean_c:.3f} "
                f"({pos}/{len(cells)} cells > 0.3).")
    if mean_c >= 0.15:
        return f"WEAK — mean correlation {mean_c:.3f}; partial alignment only."
    return f"NOT_SUPPORTED — mean correlation {mean_c:.3f}."


def run(models=DEFAULT_MODELS, n_samples=32, seed=0, eta=DEFAULT_ETA, k=DEFAULT_K,
        rebuild: bool = False):
    import activation_digest as ad
    from probe_base import load_context

    cells = []
    ce_trajectories = {}
    theoretical_rho = {}
    empirical_rho = {}

    for model in models:
        print(f"\n=== {model} (digest) ===", flush=True)
        t_model = time.time()
        dg = ad.load_or_build_digest(
            model, tasks=TASKS, n_samples=n_samples, seed=seed,
            depth="ct", rebuild=rebuild)
        ctx = load_context(model)
        dctx = ad.DigestContext(dg, ctx)
        ce_trajectories[model] = {}
        theoretical_rho[model] = {}
        empirical_rho[model] = {}
        for domain in TASKS:
            print(f"  {domain} (n={n_samples})...", flush=True)
            t0 = time.time()
            cell = run_cell_from_digest(dg, dctx, domain, eta, k)
            cell["wall_s"] = round(time.time() - t0, 1)
            cells.append(cell)
            ce_trajectories[model][domain] = cell["ce_trajectory"]
            theoretical_rho[model][domain] = cell["theoretical_rho_trajectory"]
            empirical_rho[model][domain] = cell["empirical_rho_trajectory"]
            print(f"    corr={cell['corr_theoretical_empirical_rho']} "
                  f"ce {cell['ce_trajectory'][0]:.2f}->{cell['ce_final']:.2f} "
                  f"({cell['wall_s']}s)", flush=True)
        print(f"  model total {time.time() - t_model:.1f}s (incl. digest)", flush=True)

    out = {
        "mode": "next_token_ce_contraction_digest",
        "date": time.strftime("%Y-%m-%d"),
        "claim": "Per-layer next-token CE descent vs Fisher-local rho_i = 1 - eta * a_i "
                 "(Activation Digest engine).",
        "models": list(models),
        "tasks": list(TASKS),
        "n_samples": n_samples,
        "eta": eta,
        "k": k,
        "ce_trajectories": ce_trajectories,
        "theoretical_rho_trajectories": theoretical_rho,
        "empirical_rho_trajectories": empirical_rho,
        "cells": cells,
        "verdict": _verdict(cells),
    }
    path = HERE / "next_token_ce_contraction.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {out['verdict']}", flush=True)
    print(f"saved -> {path.name}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eta", type=float, default=DEFAULT_ETA)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    run(models=tuple(args.models), n_samples=args.n_samples, seed=args.seed,
        eta=args.eta, k=args.k, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
