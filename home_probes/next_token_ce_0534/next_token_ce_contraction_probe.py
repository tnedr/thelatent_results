#!/usr/bin/env python3
"""Next-token cross-entropy contraction — does Fisher-preconditioned local GD
describe per-layer CE descent in real LLMs? (dispatch 0534)"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
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


def _curvature_a(h_list, g_list, k: int) -> float:
    from cross_layer_transport_probe import layer_readout_subspace

    S, rho, _V = layer_readout_subspace(h_list, g_list, k)
    e = (np.asarray(S, float) ** 2) * (np.asarray(rho, float) ** 2)
    return float(np.max(e)) if e.size else float("nan")


def _logit_lens_ce(model, norm, unembed, input_ids, pos, correct_id, device):
    """Per-layer surprisal of the correct next token (logit lens)."""
    import torch

    ids = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    ce = []
    for h in out.hidden_states:
        logits = unembed(norm(h[0, pos, :]))
        logp = torch.log_softmax(logits.float(), dim=-1)
        ce.append(float(-logp[correct_id].item()))
    return ce


def run_cell(model_name: str, domain: str, n_samples: int, seed: int,
             eta: float, k: int) -> dict:
    import torch
    from icl_convergence_probe import (load_model, pick_device, safe_name,
                                       get_final_norm_and_unembed)
    from transport_gain_probe import get_decoder_layers
    from capacity_threshold_sweep import make_prompt
    from routing_selection_probe import build_induction
    from cross_layer_transport_probe import capture_all_layers

    device = pick_device()
    hf_name, tok, model = load_model(model_name)
    model.to(device)
    norm, unembed = get_final_norm_and_unembed(model)
    layers = get_decoder_layers(model)
    n_layers = len(layers)

    rng = np.random.default_rng(seed)
    layer_h: dict[int, list] = defaultdict(list)
    layer_g: dict[int, list] = defaultdict(list)
    ce_sum = None
    n_ok = 0

    for _ in range(n_samples):
        if domain == "induction":
            seq, pos, cid = build_induction(tok, rng, block_len=8)
        else:
            seq, pos, cid = make_prompt(domain, tok, rng)
        try:
            ce = _logit_lens_ce(model, norm, unembed, seq, pos, cid, device)
            caps = capture_all_layers(model, layers, norm, unembed, seq, pos, cid, device)
        except Exception as e:  # noqa: BLE001
            print(f"  skip sample ({type(e).__name__}: {e})", flush=True)
            continue
        if ce_sum is None:
            ce_sum = np.zeros(len(ce), dtype=float)
        ce_sum += np.asarray(ce, float)
        for i, (hq, g) in caps.items():
            layer_h[i].append(hq)
            layer_g[i].append(g)
        n_ok += 1

    if n_ok < 3:
        raise RuntimeError(f"{domain}: only {n_ok} usable samples")

    ce_traj = (ce_sum / n_ok).tolist()
    # Align decoder-layer indices 0..L-1 with hidden_states[1..L] CE steps
    a_by_layer = {}
    for i in sorted(layer_h):
        if len(layer_h[i]) >= 3:
            a_by_layer[i] = _curvature_a(layer_h[i], layer_g[i], k)

    max_a = max(a_by_layer.values()) if a_by_layer else 1.0
    theoretical = []
    empirical = []
    layer_pairs = []
    for i in sorted(a_by_layer)[:-1]:
        li, lj = i, i + 1
        if lj >= len(ce_traj):
            break
        a_norm = a_by_layer[li] / (max_a + 1e-12)
        rho_th = 1.0 - eta * a_norm
        rho_th = float(np.clip(rho_th, 0.0, 1.0))
        e_i, e_j = ce_traj[li + 1], ce_traj[lj + 1]  # +1: hidden_states offset
        rho_em = float(e_j / (e_i + 1e-12))
        theoretical.append(rho_th)
        empirical.append(rho_em)
        layer_pairs.append({"layer_i": li, "layer_j": lj,
                            "ce_i": round(e_i, 4), "ce_j": round(e_j, 4)})

    delta_ce = [ce_traj[i + 1] - ce_traj[i] for i in range(len(ce_traj) - 1)]
    corr = _spearman(theoretical, empirical)
    mono = float(np.mean(np.diff(ce_traj) <= 1e-6)) if len(ce_traj) > 1 else float("nan")

    return {
        "model": model_name,
        "hf_name": hf_name,
        "domain": domain,
        "n_samples": n_ok,
        "n_layers": n_layers,
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


def _verdict(cells: list[dict]) -> str:
    corrs = [c["corr_theoretical_empirical_rho"] for c in cells
             if c.get("corr_theoretical_empirical_rho") is not None]
    if not corrs:
        return "INCONCLUSIVE — no valid correlations."
    mean_c = float(np.mean(corrs))
    pos = sum(1 for c in corrs if c is not None and c > 0.3)
    if mean_c >= 0.4 and pos >= len(cells) * 0.6:
        return (f"SUPPORTED — mean Spearman(theoretical_rho, empirical_rho)={mean_c:.3f} "
                f"({pos}/{len(cells)} cells > 0.3); Fisher-local step tracks CE descent.")
    if mean_c >= 0.15:
        return (f"WEAK — mean correlation {mean_c:.3f}; partial alignment only.")
    return f"NOT_SUPPORTED — mean correlation {mean_c:.3f}; local Fisher step poor CE predictor."


def run(models=DEFAULT_MODELS, n_samples=32, seed=0, eta=DEFAULT_ETA, k=DEFAULT_K):
    cells = []
    for model in models:
        print(f"\n=== {model} ===", flush=True)
        for domain in TASKS:
            print(f"  {domain} (n={n_samples})...", flush=True)
            t0 = time.time()
            cell = run_cell(model, domain, n_samples, seed, eta, k)
            cell["wall_s"] = round(time.time() - t0, 1)
            cells.append(cell)
            print(f"    corr={cell['corr_theoretical_empirical_rho']} "
                  f"ce {cell['ce_trajectory'][0]:.2f}->{cell['ce_final']:.2f} "
                  f"({cell['wall_s']}s)", flush=True)

    out = {
        "mode": "next_token_ce_contraction",
        "date": time.strftime("%Y-%m-%d"),
        "claim": "Per-layer next-token CE descent aligns with Fisher-preconditioned "
                 "local GD surrogate rho_i = 1 - eta * a_i.",
        "models": list(models),
        "tasks": list(TASKS),
        "n_samples": n_samples,
        "eta": eta,
        "k": k,
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
    args = ap.parse_args()
    run(models=tuple(args.models), n_samples=args.n_samples, seed=args.seed,
        eta=args.eta, k=args.k)


if __name__ == "__main__":
    main()
