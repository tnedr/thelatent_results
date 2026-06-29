#!/usr/bin/env python3
"""Thermodynamic specific heat of attention across depth (dispatch 0740).

Uses activation_digest for the task panel + ModelContext (one model load).
Pre-softmax energies: e_j = log p_j from causal attention at the query row.
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
PEAK_SPREAD_MAX = 0.20
NULL_MARGIN = 2.0


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def _thermo_from_log_energies(e: np.ndarray) -> tuple[float, float]:
    """Entropy H and specific heat C at beta=1 (C = Var_p(e))."""
    p = _softmax(e)
    H = float(-np.sum(p * np.log(p + 1e-12)))
    mu = float(np.sum(p * e))
    mu2 = float(np.sum(p * e**2))
    C = mu2 - mu * mu
    return H, C


def _null_C(e: np.ndarray, rng: np.random.Generator) -> float:
    perm = rng.permutation(len(e))
    return _thermo_from_log_energies(e[perm])[1]


def _prompts_for_task(ctx, task: str, n_samples: int, seed: int) -> list[tuple]:
    from capacity_threshold_sweep import make_prompt
    from routing_selection_probe import build_induction

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_samples):
        if task == "induction":
            seq, pos, cid = build_induction(ctx.tok, rng, block_len=8)
        else:
            seq, pos, cid = make_prompt(task, ctx.tok, rng)
        out.append((seq, pos, int(cid)))
    return out


def _layer_thermo(model, input_ids: list[int], query_pos: int, device, rng) -> tuple[list, list, list]:
    """Return per-layer (H, C, C_null) averaged over heads."""
    import torch

    ids = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    if not out.attentions:
        raise RuntimeError("output_attentions empty — set attn_implementation=eager")

    Hs, Cs, Cnulls = [], [], []
    for attn in out.attentions:
        # [batch, heads, q, k]
        ah = attn[0, :, query_pos, : query_pos + 1].float().cpu().numpy()
        head_H, head_C, head_Cn = [], [], []
        for h in range(ah.shape[0]):
            p = ah[h]
            p = p / (p.sum() + 1e-12)
            e = np.log(p + 1e-12)
            H, C = _thermo_from_log_energies(e)
            head_H.append(H)
            head_C.append(C)
            head_Cn.append(_null_C(e, rng))
        Hs.append(float(np.mean(head_H)))
        Cs.append(float(np.mean(head_C)))
        Cnulls.append(float(np.mean(head_Cn)))
    return Hs, Cs, Cnulls


def _peak_stats(C: np.ndarray, n_layers: int) -> dict:
    rel = np.array([i / max(n_layers - 1, 1) for i in range(len(C))], float)
    peak_i = int(np.argmax(C))
    return {
        "peak_layer": peak_i,
        "peak_rel_depth": round(float(rel[peak_i]), 4),
        "peak_C": round(float(C[peak_i]), 6),
    }


def _verdict_cell(peak_rels: list[float], peak_C: float, null_C: float) -> str:
    spread = float(np.max(peak_rels) - np.min(peak_rels)) if len(peak_rels) > 1 else 0.0
    ratio = peak_C / (null_C + 1e-12)
    if spread < PEAK_SPREAD_MAX and ratio >= NULL_MARGIN:
        return "phase-transition-like"
    if ratio < NULL_MARGIN:
        return "smooth"
    return "inconsistent"


def _verdict_global(cells: list[dict]) -> str:
    labels = [c["verdict"] for c in cells]
    if all(v == "phase-transition-like" for v in labels):
        return "PHASE_TRANSITION_LIKE — localized C(L) peak consistent across cells and >=2x null."
    if sum(1 for v in labels if v == "phase-transition-like") >= len(labels) * 0.5:
        return "PARTIAL — some cells show phase-transition-like specific-heat peak."
    if all(v == "smooth" for v in labels):
        return "SMOOTH — C(L) peaks do not exceed shuffled-logit null by 2x."
    return "INCONSISTENT — peak depth or null margin fails across cells."


def run_cell(model_name: str, task: str, n_samples: int, seed: int) -> dict:
    import activation_digest as ad
    from icl_convergence_probe import safe_name
    from probe_base import load_context

    # Digest engine: shared task panel + cached (H,G); model ctx for attention readout.
    ad.load_or_build_digest(model_name, tasks=(task,), n_samples=n_samples, seed=seed, depth="ct")
    ctx = load_context(model_name)
    rng = np.random.default_rng(seed)
    n_layers = len(ctx.layers)

    prompts = _prompts_for_task(ctx, task, n_samples, seed)
    sum_H = np.zeros(n_layers, float)
    sum_C = np.zeros(n_layers, float)
    sum_Cn = np.zeros(n_layers, float)
    peak_rels = []
    n_ok = 0

    for seq, pos, _cid in prompts:
        try:
            Hs, Cs, Cns = _layer_thermo(ctx.model, seq, pos, ctx.device, rng)
        except Exception as e:  # noqa: BLE001
            print(f"  skip ({type(e).__name__}: {e})", flush=True)
            continue
        nL = min(len(Hs), n_layers)
        sum_H[:nL] += np.asarray(Hs[:nL], float)
        sum_C[:nL] += np.asarray(Cs[:nL], float)
        sum_Cn[:nL] += np.asarray(Cns[:nL], float)
        peak_rels.append(_peak_stats(np.asarray(Cs[:nL]), n_layers)["peak_rel_depth"])
        n_ok += 1

    if n_ok < 3:
        raise RuntimeError(f"{task}: only {n_ok} prompts ok")

    mean_H = (sum_H / n_ok).tolist()
    mean_C = (sum_C / n_ok).tolist()
    mean_Cn = (sum_Cn / n_ok).tolist()
    rel_depth = [round(i / max(n_layers - 1, 1), 4) for i in range(n_layers)]
    peak = _peak_stats(np.asarray(mean_C), n_layers)
    null_level = float(np.mean(mean_Cn))
    peak_rels_mean = float(np.mean(peak_rels))
    spread = float(np.max(peak_rels) - np.min(peak_rels)) if peak_rels else float("nan")
    verdict = _verdict_cell(peak_rels, peak["peak_C"], null_level)

    return {
        "model": model_name,
        "hf_name": ctx.hf_name,
        "domain": task,
        "n_samples": n_ok,
        "n_layers": n_layers,
        "rel_depth": rel_depth,
        "entropy_H": [round(x, 4) for x in mean_H],
        "specific_heat_C": [round(x, 6) for x in mean_C],
        "null_C": [round(x, 6) for x in mean_Cn],
        "null_level_mean": round(null_level, 6),
        "peak": peak,
        "peak_rel_depth_per_prompt": [round(x, 4) for x in peak_rels],
        "peak_rel_depth_spread": round(spread, 4),
        "peak_over_null": round(peak["peak_C"] / (null_level + 1e-12), 4),
        "verdict": verdict,
    }


def _plot_all(cells: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for cell in cells:
        label = f"{cell['model']}/{cell['domain']}"
        x = cell["rel_depth"]
        ax.plot(x, cell["specific_heat_C"], label=label, linewidth=1.5)
        ax.plot(x, cell["null_C"], linestyle=":", alpha=0.35, linewidth=1.0)
    ax.set_xlabel("relative depth")
    ax.set_ylabel("specific heat C(L)  (dotted = shuffled-logit null)")
    ax.set_title("Attention specific heat across depth")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(models=DEFAULT_MODELS, n_samples: int = 32, seed: int = 0):
    from icl_convergence_probe import safe_name

    cells = []
    per_model = {}
    for model in models:
        print(f"\n=== {model} ===", flush=True)
        per_model[model] = {}
        for task in TASKS:
            print(f"  {task} (n={n_samples})...", flush=True)
            t0 = time.time()
            cell = run_cell(model, task, n_samples, seed)
            cell["wall_s"] = round(time.time() - t0, 1)
            cells.append(cell)
            per_model[model][task] = cell
            print(f"    verdict={cell['verdict']} peak/null={cell['peak_over_null']} "
                  f"spread={cell['peak_rel_depth_spread']} ({cell['wall_s']}s)", flush=True)

        mpath = HERE / f"thermo_specific_heat_{safe_name(model)}.json"
        mout = {
            "mode": "thermo_specific_heat",
            "model": model,
            "cells": [c for c in cells if c["model"] == model],
            "verdict": _verdict_global([c for c in cells if c["model"] == model]),
        }
        mpath.write_text(json.dumps(mout, indent=2), encoding="utf-8")
        print(f"  saved -> {mpath.name}", flush=True)

    summary = {
        "mode": "thermo_specific_heat",
        "date": time.strftime("%Y-%m-%d"),
        "models": list(models),
        "tasks": list(TASKS),
        "n_samples": n_samples,
        "seed": seed,
        "peak_spread_max": PEAK_SPREAD_MAX,
        "null_margin": NULL_MARGIN,
        "cells": cells,
        "per_model": per_model,
        "verdict": _verdict_global(cells),
    }
    spath = HERE / "thermo_specific_heat_summary.json"
    spath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_all(cells, HERE / "thermo_specific_heat.png")
    print(f"\nVERDICT: {summary['verdict']}", flush=True)
    print("saved -> thermo_specific_heat_summary.json, thermo_specific_heat.png", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(models=tuple(args.models), n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()
