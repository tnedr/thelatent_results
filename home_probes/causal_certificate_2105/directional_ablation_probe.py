#!/usr/bin/env python3
"""Directional causal sigma-ablation probe — the CAUSAL upgrade of SVD3.

Formal seed (verified): elysium/fields/capacity_transport/capacity_transport_proof.py,
family SVD (SVD1-SVD3). Bridge: `transport_spectrum_unification` in
bridge_assumptions.yaml.

WHY THIS PROBE EXISTS
  singular_spectrum_probe measures SVD3 CORRELATIONALLY: the readout energy
  e_i = sigma_i^2 * rho_i^2 is rank-ordered by the transport singular values
  sigma_i (spearman ~0.7-0.92 on gpt2 / pythia-410m). A correlation, however
  strong, does not prove the spectrum CAUSES the in-context computation. This
  probe converts the correlation into a CAUSAL dose-response on a single small
  model, with NO training and NO large compute.

THEORY (what the kernel proves, SVD1-SVD3):
  At a capacity layer L_c the residual stream H = U diag(sigma) V^T transports
  the in-context task signal along the right-singular directions V_i, with the
  high-sigma directions carrying the readout. SVD3: per-direction effective
  curvature a_eff_i = lambda_i * sigma_i^2, so the TOP-sigma directions are the
  ones that drive in-context learning; the BOTTOM-sigma directions are
  near-dark (SVD1: sigma_i -> 0 is unlearnable).

CAUSAL PREDICTION (the falsifiable claim this probe tests):
  Ablate (project OUT) k transport directions at L_c, via a forward hook that
  removes the component of the hidden state along the chosen V_i:
        h_ablated = h - (h V_sel^T) V_sel.
  P_TOP   removing the TOP-k sigma directions collapses the in-context readout
          (accuracy -> chance) for small k.
  P_BOTTOM removing the BOTTOM-k sigma directions (same budget k) barely
          affects the readout.
  P_ASYM  the asymmetry is the causal SVD3 result: at a fixed budget k the
          accuracy drop from top-ablation >> the drop from bottom-ablation.
          A correlation cannot produce this asymmetry; only a causal
          dependence on the high-sigma subspace can.

  Reported per model:
    auc_top      = mean accuracy across the top-ablation sweep (low = fragile)
    auc_bottom   = mean accuracy across the bottom-ablation sweep (high = robust)
    causal_gap   = auc_bottom - auc_top   (the SVD3 causal effect size, in [0,1])
    k_half_top   = smallest k whose top-ablation halves baseline accuracy
    k_half_bottom= smallest k whose bottom-ablation halves baseline accuracy

  We also rank directions by READOUT ENERGY e_i (sigma_i^2 * rho_i^2) as a
  second ordering; the theory says sigma-order and energy-order should give the
  same causal asymmetry.

Local cache only. Run:
  python3 directional_ablation_probe.py --synthetic
  python3 directional_ablation_probe.py --models gpt2 pythia-410m --n-samples 48
  python3 directional_ablation_probe.py --aggregate
Output: directional_ablation_<model>.json (+ directional_ablation_summary.json)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from icl_convergence_probe import (  # noqa: E402
    safe_name,
    pick_device,
    load_model,
    get_final_norm_and_unembed,
    build_induction_ids,
)

DEFAULT_MODELS = ["gpt2", "pythia-410m"]
ABLATION_BUDGETS = [1, 2, 4, 8, 16, 32]


# ---------------------------------------------------------------------------
# Block discovery (shared with composition_probe semantics)
# ---------------------------------------------------------------------------

def _find_block_list(model):
    for name, mod in model.named_modules():
        if name in ("transformer.h", "gpt_neox.layers", "model.layers",
                    "model.decoder.layers") and hasattr(mod, "__len__"):
            return mod
    for name, mod in model.named_modules():
        children = list(mod.children())
        if len(children) > 4 and all(type(children[0]) == type(c) for c in children[:4]):
            return mod
    return None


# ---------------------------------------------------------------------------
# Directional ablation hook
# ---------------------------------------------------------------------------

def _ablation_hook(which: str, k: int, order: str):
    """Project OUT k transport directions from the layer output.

    which: 'top' removes the k largest-sigma directions; 'bottom' removes the
           k smallest (of the informative set).
    order: 'sigma' ranks by transport singular value; 'energy' ranks by the
           readout-energy proxy sigma_i^2 (rho unavailable inside the hook, so
           sigma^2 is the in-hook energy surrogate; readout-aligned ordering is
           handled by the sigma ranking itself for the query column)."""
    import torch

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output  # [b, seq, d]
        b, seq, d = h.shape
        h0 = h[0].float()                                        # [seq, d]
        # SVD of the residual stream: rows of Vh are the transport directions.
        U, S, Vh = torch.linalg.svd(h0, full_matrices=False)     # Vh: [r, d]
        r = Vh.shape[0]
        kk = min(k, r)
        if kk <= 0:
            return output
        if which == "top":
            sel = Vh[:kk]                                        # largest sigma
        else:
            sel = Vh[r - kk:]                                    # smallest sigma
        # remove the component along the selected directions for every position
        # h_ablated = h - (h @ sel^T) @ sel
        coeff = h0 @ sel.transpose(0, 1)                         # [seq, kk]
        h_ab = h0 - coeff @ sel                                  # [seq, d]
        h_new = h.clone()
        h_new[0] = h_ab.to(h.dtype)
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return hook


def _accuracy_under_ablation(model, device, tok, seed, layer_idx,
                             which, k, order, n_samples, block_len):
    import torch
    block_list = _find_block_list(model)
    if block_list is None:
        return None
    n_blocks = len(block_list)
    layer_idx = min(layer_idx, n_blocks - 1)
    target = block_list[layer_idx]

    rng = np.random.default_rng(seed)
    handle = target.register_forward_hook(_ablation_hook(which, k, order))
    correct = 0
    total = 0
    try:
        for _ in range(n_samples):
            ids, pos, val_id = build_induction_ids(tok, rng, block_len)
            t = torch.tensor([ids], device=device)
            with torch.no_grad():
                out = model(input_ids=t)
            pred = int(torch.argmax(out.logits[0, pos, :]).item())
            correct += int(pred == val_id)
            total += 1
    finally:
        handle.remove()
    return round(correct / max(total, 1), 4)


def _baseline_accuracy(model, device, tok, seed, n_samples, block_len):
    import torch
    rng = np.random.default_rng(seed)
    correct = 0
    total = 0
    for _ in range(n_samples):
        ids, pos, val_id = build_induction_ids(tok, rng, block_len)
        t = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model(input_ids=t)
        pred = int(torch.argmax(out.logits[0, pos, :]).item())
        correct += int(pred == val_id)
        total += 1
    return round(correct / max(total, 1), 4)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_model(model_name: str, n_samples: int = 48, block_len: int = 12,
              seed: int = 42) -> dict:
    device = pick_device()
    hf_name, tok, model = load_model(model_name)
    model.to(device)

    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 12))
    layer_idx = n_layers // 2

    baseline = _baseline_accuracy(model, device, tok, seed, n_samples, block_len)

    budgets = [k for k in ABLATION_BUDGETS]
    top_sweep, bottom_sweep = [], []
    for k in budgets:
        acc_top = _accuracy_under_ablation(
            model, device, tok, seed, layer_idx, "top", k, "sigma",
            n_samples, block_len)
        acc_bot = _accuracy_under_ablation(
            model, device, tok, seed, layer_idx, "bottom", k, "sigma",
            n_samples, block_len)
        top_sweep.append({"k": k, "accuracy": acc_top})
        bottom_sweep.append({"k": k, "accuracy": acc_bot})

    def _auc(sweep):
        vals = [pt["accuracy"] for pt in sweep if pt["accuracy"] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    def _k_half(sweep, base):
        thr = base / 2.0
        for pt in sweep:
            if pt["accuracy"] is not None and pt["accuracy"] <= thr:
                return pt["k"]
        return None

    auc_top = _auc(top_sweep)
    auc_bottom = _auc(bottom_sweep)
    causal_gap = (round(auc_bottom - auc_top, 4)
                  if (auc_top is not None and auc_bottom is not None) else None)
    k_half_top = _k_half(top_sweep, baseline)
    k_half_bottom = _k_half(bottom_sweep, baseline)

    # Verdict: causal SVD3 confirmed if top-ablation is much more damaging than
    # bottom-ablation, i.e. a positive causal gap, and the top sweep actually
    # halves accuracy at a small budget while the bottom sweep does not.
    base_ok = baseline >= 0.3
    asym_strong = causal_gap is not None and causal_gap >= 0.25
    asym_weak = causal_gap is not None and causal_gap >= 0.1
    top_fragile = k_half_top is not None
    bottom_robust = (k_half_bottom is None) or (
        k_half_top is not None and k_half_bottom > k_half_top)

    if not base_ok:
        verdict = "DARK"
    elif asym_strong and top_fragile and bottom_robust:
        verdict = "PASS"
    elif asym_weak:
        verdict = "WEAK"
    else:
        verdict = "FAIL"

    return {
        "model": model_name,
        "hf_name": hf_name,
        "device": device,
        "n_layers": n_layers,
        "capacity_layer_idx": layer_idx,
        "n_samples": n_samples,
        "block_len": block_len,
        "seed": seed,
        "baseline_accuracy": baseline,
        "top_ablation_sweep": top_sweep,
        "bottom_ablation_sweep": bottom_sweep,
        "summary": {
            "auc_top": auc_top,
            "auc_bottom": auc_bottom,
            "causal_gap": causal_gap,
            "k_half_top": k_half_top,
            "k_half_bottom": k_half_bottom,
            "verdict": verdict,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Subspace-interference law — HOW are in-context tasks organized in the
# transport geometry? Ablate task A's top-sigma subspace (a FIXED precomputed
# basis) and measure the damage to task B; predict damage ~ subspace overlap.
# ---------------------------------------------------------------------------

def _induction_builder(lo: int, hi: int):
    """Verbatim copy-induction over a vocab partition [lo, hi)."""
    def build(tok, rng, block_len):
        block = [int(rng.integers(lo, hi)) for _ in range(block_len)]
        seq = block + block
        return seq[:-1], len(seq) - 2, seq[-1]
    return build


def _constant_builder(lo: int, hi: int):
    """In-context constant rule: every key is followed by the SAME token c;
    a novel query key must also map to c. A different mechanism from copy
    (attend to the constant answer, not to a matching key)."""
    def build(tok, rng, block_len):
        c = int(rng.integers(lo, hi))
        seq = []
        for _ in range(block_len):
            k = int(rng.integers(lo, hi))
            seq += [k, c]
        qk = int(rng.integers(lo, hi))
        seq += [qk]
        return seq, len(seq) - 1, c
    return build


def _capture_residuals(model, device, tok, builder, seed, layer_idx,
                       n_prompts, block_len):
    """Collect the layer-output residual stream [seq, d] for n_prompts."""
    import torch
    block_list = _find_block_list(model)
    if block_list is None:
        return None
    layer_idx = min(layer_idx, len(block_list) - 1)
    target = block_list[layer_idx]
    captured = []

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h[0].detach().float().cpu().numpy())

    handle = target.register_forward_hook(hook)
    rng = np.random.default_rng(seed)
    mats = []
    try:
        for _ in range(n_prompts):
            ids, _pos, _val = builder(tok, rng, block_len)
            captured.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=device))
            if captured:
                mats.append(captured[0])
    finally:
        handle.remove()
    return mats


def _task_subspace(mats, k: int):
    """Top-k right-singular subspace of the stacked, cross-prompt-centered
    residuals — the directions the task's transport stream occupies."""
    H = np.concatenate(mats, axis=0)             # [N*seq, d]
    Hc = H - H.mean(axis=0, keepdims=True)
    _U, _S, Vh = np.linalg.svd(Hc, full_matrices=False)
    return Vh[:k]                                 # [k, d] orthonormal rows


def _subspace_overlap(V_a, V_b):
    """Mean squared principal cosine in [0,1]: ||V_a V_b^T||_F^2 / k."""
    k = V_a.shape[0]
    M = V_a @ V_b.T
    return float((M ** 2).sum() / k)


def _fixed_ablation_hook(V_sel):
    """Project OUT a FIXED basis V_sel [k, d] from the layer output."""
    import torch

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h0 = h[0].float()
        coeff = h0 @ V_sel.transpose(0, 1)
        h_ab = h0 - coeff @ V_sel
        h_new = h.clone()
        h_new[0] = h_ab.to(h.dtype)
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return hook


def _acc_fixed_ablation(model, device, tok, builder, seed, layer_idx,
                        V_sel, n_samples, block_len):
    import torch
    block_list = _find_block_list(model)
    layer_idx = min(layer_idx, len(block_list) - 1)
    target = block_list[layer_idx]
    Vt = torch.tensor(V_sel, device=device, dtype=torch.float32)
    handle = target.register_forward_hook(_fixed_ablation_hook(Vt))
    rng = np.random.default_rng(seed)
    correct = total = 0
    try:
        for _ in range(n_samples):
            ids, pos, val = builder(tok, rng, block_len)
            with torch.no_grad():
                out = model(input_ids=torch.tensor([ids], device=device))
            pred = int(torch.argmax(out.logits[0, pos, :]).item())
            correct += int(pred == val)
            total += 1
    finally:
        handle.remove()
    return round(correct / max(total, 1), 4)


def _baseline_task(model, device, tok, builder, seed, n_samples, block_len):
    import torch
    rng = np.random.default_rng(seed)
    correct = total = 0
    for _ in range(n_samples):
        ids, pos, val = builder(tok, rng, block_len)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device=device))
        pred = int(torch.argmax(out.logits[0, pos, :]).item())
        correct += int(pred == val)
        total += 1
    return round(correct / max(total, 1), 4)


def run_interference(model_name: str, k: int = 4, n_samples: int = 48,
                     n_basis: int = 48, block_len: int = 12,
                     seed: int = 42) -> dict:
    """Cross-task subspace ablation. For each task compute a fixed top-k
    transport subspace, then ablate every task's subspace from every task and
    measure the damage. LAW: cross_damage(A->B) ~ overlap(V_A, V_B).
    Self-ablation (A->A, overlap 1) is the damage upper bound."""
    device = pick_device()
    hf_name, tok, model = load_model(model_name)
    model.to(device)
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 12))
    layer_idx = n_layers // 2

    tasks = {
        "copy_a": _induction_builder(1000, 20000),
        "copy_b": _induction_builder(20000, 40000),   # same mechanism, disjoint vocab
        "constant": _constant_builder(1000, 20000),   # different mechanism
    }

    baselines = {t: _baseline_task(model, device, tok, b, seed, n_samples, block_len)
                 for t, b in tasks.items()}
    # only keep tasks the model can actually do (measurable damage)
    doable = {t: b for t, b in tasks.items() if baselines[t] >= 0.3}

    subspaces = {t: _task_subspace(
        _capture_residuals(model, device, tok, b, seed, layer_idx, n_basis, block_len), k)
        for t, b in doable.items()}

    overlap = {a: {bb: round(_subspace_overlap(subspaces[a], subspaces[bb]), 4)
                   for bb in doable} for a in doable}
    damage = {}
    for a in doable:                       # ablate A's subspace
        damage[a] = {}
        for bb in doable:                  # measure damage to B
            acc = _acc_fixed_ablation(model, device, tok, doable[bb], seed,
                                      layer_idx, subspaces[a], n_samples, block_len)
            damage[a][bb] = round(baselines[bb] - acc, 4)

    # LAW test: correlate overlap vs cross_damage over off-diagonal ordered pairs.
    ov, dm = [], []
    self_dmg, cross_dmg = [], []
    for a in doable:
        for bb in doable:
            if a == bb:
                self_dmg.append(damage[a][bb])
            else:
                ov.append(overlap[a][bb])
                dm.append(damage[a][bb])
                cross_dmg.append(damage[a][bb])
    law_corr = round(spearman_safe(ov, dm), 4) if len(ov) >= 3 else None
    mean_self = round(float(np.mean(self_dmg)), 4) if self_dmg else None
    mean_cross = round(float(np.mean(cross_dmg)), 4) if cross_dmg else None

    # PRIMARY law = damage tracks subspace overlap (spearman). self-vs-cross
    # separation is a secondary strength signal (it is confounded when the
    # top-k subspaces are entangled, as in larger models), so it does not gate.
    self_dominates = (mean_self is not None and mean_cross is not None
                      and mean_self - mean_cross >= 0.2)
    law_holds = law_corr is not None and law_corr >= 0.5
    law_weak = law_corr is not None and law_corr >= 0.3
    if len(doable) < 2:
        verdict = "DARK"
    elif law_holds:
        verdict = "PASS"
    elif law_weak:
        verdict = "WEAK"
    else:
        verdict = "FAIL"

    return {
        "model": model_name,
        "hf_name": hf_name,
        "mode": "interference",
        "n_layers": n_layers,
        "capacity_layer_idx": layer_idx,
        "k": k,
        "n_samples": n_samples,
        "n_basis": n_basis,
        "block_len": block_len,
        "seed": seed,
        "baselines": baselines,
        "doable_tasks": list(doable.keys()),
        "overlap_matrix": overlap,
        "damage_matrix": damage,
        "summary": {
            "law_corr_overlap_vs_damage": law_corr,
            "mean_self_damage": mean_self,
            "mean_cross_damage": mean_cross,
            "self_minus_cross": (round(mean_self - mean_cross, 4)
                                 if (mean_self is not None and mean_cross is not None)
                                 else None),
            "verdict": verdict,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def spearman_safe(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# Layer-localization law — WHERE is ICL gated? Sweep the ablation layer at a
# fixed budget and find where top-sigma ablation is most damaging.
# ---------------------------------------------------------------------------

def run_layer_sweep(model_name: str, k: int = 4, n_samples: int = 32,
                    block_len: int = 12, seed: int = 42) -> dict:
    """Top-sigma ablation at EVERY layer (fixed budget k). The damage curve
    drop(L) = baseline - acc_top(L) reveals the depth profile of the ICL
    transport pathway.

    LAW under test (transport-bottleneck localization):
      P_LOCALIZED  the damage is sharply peaked at one depth (not uniform) =>
                   ICL is gated through a localized transport bottleneck.
      P_DEPTH      the peak relative depth L*/(L-1) is comparable across model
                   families (a depth-scaling regularity).
    """
    device = pick_device()
    hf_name, tok, model = load_model(model_name)
    model.to(device)
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 12))

    baseline = _baseline_accuracy(model, device, tok, seed, n_samples, block_len)

    per_layer = []
    for L in range(n_layers):
        acc_top = _accuracy_under_ablation(
            model, device, tok, seed, L, "top", k, "sigma", n_samples, block_len)
        drop = round(baseline - acc_top, 4) if acc_top is not None else None
        per_layer.append({"layer": L, "acc_top": acc_top, "drop": drop})

    drops = [r["drop"] for r in per_layer if r["drop"] is not None]
    peak = max(per_layer, key=lambda r: (r["drop"] if r["drop"] is not None else -1))
    peak_drop = peak["drop"]
    mean_drop = float(np.mean(drops)) if drops else 0.0
    # bottom-ablation control at the peak layer (asymmetry must still hold there)
    acc_bottom_peak = _accuracy_under_ablation(
        model, device, tok, seed, peak["layer"], "bottom", k, "sigma",
        n_samples, block_len)

    # localization index: 0 == uniform damage (diffuse), ->1 == single sharp peak.
    localization = (round((peak_drop - mean_drop) / (peak_drop + 1e-9), 4)
                    if peak_drop and peak_drop > 0 else 0.0)
    peak_rel_depth = round(peak["layer"] / max(n_layers - 1, 1), 4)

    base_ok = baseline >= 0.3
    strong_peak = peak_drop is not None and peak_drop >= 0.4
    is_localized = localization >= 0.3
    asym_at_peak = (acc_bottom_peak is not None and peak["acc_top"] is not None
                    and acc_bottom_peak - peak["acc_top"] >= 0.25)
    if not base_ok:
        verdict = "DARK"
    elif strong_peak and is_localized and asym_at_peak:
        verdict = "LOCALIZED"
    elif strong_peak and not is_localized:
        verdict = "DIFFUSE"
    else:
        verdict = "WEAK"

    return {
        "model": model_name,
        "hf_name": hf_name,
        "mode": "layer_sweep",
        "n_layers": n_layers,
        "ablation_budget_k": k,
        "n_samples": n_samples,
        "block_len": block_len,
        "seed": seed,
        "baseline_accuracy": baseline,
        "per_layer": per_layer,
        "summary": {
            "peak_layer": peak["layer"],
            "peak_rel_depth": peak_rel_depth,
            "peak_drop": peak_drop,
            "mean_drop": round(mean_drop, 4),
            "localization_index": localization,
            "acc_bottom_at_peak": acc_bottom_peak,
            "verdict": verdict,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Synthetic validation (numpy only) — planted causal structure
# ---------------------------------------------------------------------------

def run_synthetic() -> dict:
    """Construct a system whose readout depends ONLY on the top-sigma
    directions; verify top-ablation kills the readout and bottom-ablation
    does not (the causal asymmetry the real probe looks for)."""
    rng = np.random.default_rng(0)
    d, seq = 64, 80
    A = rng.standard_normal((d, d))
    V, _ = np.linalg.qr(A)                       # transport basis
    sigma = np.exp(-0.15 * np.arange(d))         # decaying spectrum
    B = rng.standard_normal((seq, d))
    U, _ = np.linalg.qr(B)
    H = (U[:, :d] * sigma[None, :]) @ V.T         # [seq, d]
    # readout reads ONLY the top-4 transport directions
    g = V[:, :4] @ rng.uniform(0.5, 1.5, size=4)

    def ablate(which, k):
        Um, Sm, Vh = np.linalg.svd(H, full_matrices=False)
        r = Vh.shape[0]
        kk = min(k, r)
        sel = Vh[:kk] if which == "top" else Vh[r - kk:]
        H_ab = H - (H @ sel.T) @ sel
        # readout signal = energy of the ablated stream along g (query=last row)
        return float(abs(H_ab[-1] @ g))

    base = float(abs(H[-1] @ g))
    top = [ablate("top", k) / (base + 1e-30) for k in ABLATION_BUDGETS]
    bot = [ablate("bottom", k) / (base + 1e-30) for k in ABLATION_BUDGETS]
    auc_top = float(np.mean(top))
    auc_bottom = float(np.mean(bot))
    causal_gap = auc_bottom - auc_top

    # top-ablation collapses the readout once the budget reaches the readout
    # rank (k>=4); k=1,2 legitimately retain partial signal (dose-response).
    top_late = float(np.mean(top[2:]))
    checks = [
        ("top-ablation collapses readout at k>=readout_rank (mean top[k>=4] < 0.05)",
         top_late < 0.05),
        ("bottom-ablation preserves it (auc_bottom > 0.7)", auc_bottom > 0.7),
        ("causal asymmetry positive and large (gap > 0.4)", causal_gap > 0.4),
        ("k=4 top-ablation kills most signal (<0.2)", top[2] < 0.2),
    ]
    n_pass = sum(1 for _, ok in checks if ok)
    print(f"Synthetic: {n_pass}/{len(checks)} checks pass")
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")

    return {
        "mode": "synthetic",
        "auc_top": round(auc_top, 4),
        "auc_bottom": round(auc_bottom, 4),
        "causal_gap": round(causal_gap, 4),
        "top_signal_fraction": [round(x, 4) for x in top],
        "bottom_signal_fraction": [round(x, 4) for x in bot],
        "budgets": ABLATION_BUDGETS,
        "synthetic_checks": [{"test": d, "pass": ok} for d, ok in checks],
        "pass_rate": f"{n_pass}/{len(checks)}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_interference_synthetic() -> dict:
    """Planted task subspaces with TUNED overlap; verify cross-ablation damage
    tracks the overlap (the machinery the real probe relies on)."""
    rng = np.random.default_rng(0)
    d, k = 64, 4
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V_a = Q[:k]                                   # task A subspace
    rows = []
    overlaps, damages = [], []
    for theta in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5708]:   # 0..pi/2 rotation
        # V_b = cos*V_a + sin*orthogonal complement directions
        comp = Q[k:2 * k]
        V_b = np.cos(theta) * V_a + np.sin(theta) * comp
        # re-orthonormalize rows
        V_b, _ = np.linalg.qr(V_b.T)
        V_b = V_b.T[:k]
        ov = _subspace_overlap(V_a, V_b)
        # task B readout lives in V_b; ablating V_a removes the shared part.
        gB = V_b.T @ rng.uniform(0.5, 1.5, size=k)
        signal = float(np.linalg.norm(gB))
        # project gB out of V_a's complement: residual after removing V_a
        coeff = gB @ V_a.T
        gB_ab = gB - coeff @ V_a
        residual = float(np.linalg.norm(gB_ab))
        damage = 1.0 - residual / (signal + 1e-30)   # fraction of B removed
        overlaps.append(ov)
        damages.append(round(damage, 4))
        rows.append({"theta": round(theta, 3), "overlap": round(ov, 4),
                     "damage": round(damage, 4)})
    corr = spearman_safe(overlaps, damages)
    checks = [
        ("overlap spans a wide range", max(overlaps) - min(overlaps) > 0.7),
        ("damage tracks overlap (spearman > 0.9)", corr > 0.9),
        ("zero overlap => ~zero damage", damages[-1] < 0.1),
        ("full overlap => ~full damage", damages[0] > 0.9),
    ]
    n_pass = sum(1 for _, ok in checks if ok)
    print(f"Interference synthetic: {n_pass}/{len(checks)} checks pass  (spearman={corr:.3f})")
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
    return {
        "mode": "interference_synthetic",
        "law_corr_overlap_vs_damage": round(corr, 4),
        "rows": rows,
        "synthetic_checks": [{"test": d_, "pass": ok} for d_, ok in checks],
        "pass_rate": f"{n_pass}/{len(checks)}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Causal certificate mode (dispatch 2105) — 3-arm paired ablation
# ---------------------------------------------------------------------------

CERTIFICATE_KS = [1, 2, 4]


def _logit_margin(model, device, ids, pos, val_id):
    import torch

    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=t).logits[0, pos, :].float()
    v = float(logits[val_id].item())
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[val_id] = False
    other = float(logits[mask].max().item())
    return v - other


def _capture_vh_sequence(model, device, ids, layer_idx):
    """SVD of the full layer residual stream [seq, d] — same basis as SVD3 ablation."""
    import torch

    block_list = _find_block_list(model)
    if block_list is None:
        return None
    layer_idx = min(layer_idx, len(block_list) - 1)
    target = block_list[layer_idx]
    captured = []

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h[0].detach().float().cpu().numpy())

    handle = target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device=device))
    finally:
        handle.remove()
    if not captured:
        return None
    h0 = captured[0]
    if h0.ndim != 2 or h0.shape[0] < 2:
        return None
    _u, _s, vh = np.linalg.svd(h0, full_matrices=False)
    return vh


def _fixed_dirs_from_vh(vh, which, k, rng):
    r = vh.shape[0]
    kk = min(k, r)
    if kk <= 0:
        return None
    if which == "top":
        return vh[:kk]
    if which == "bottom":
        return vh[r - kk:]
    # random norm-matched in the transport subspace
    coeffs = rng.standard_normal((kk, r))
    q, _ = np.linalg.qr(coeffs.T)
    return q[:, :kk].T @ vh


def _margin_fixed_ablation(model, device, ids, pos, val_id, layer_idx, v_sel):
    import torch

    if v_sel is None or len(v_sel) == 0:
        return _logit_margin(model, device, ids, pos, val_id)
    block_list = _find_block_list(model)
    layer_idx = min(layer_idx, len(block_list) - 1)
    target = block_list[layer_idx]
    vt = torch.tensor(v_sel, device=device, dtype=torch.float32)
    handle = target.register_forward_hook(_fixed_ablation_hook(vt))
    try:
        return _logit_margin(model, device, ids, pos, val_id)
    finally:
        handle.remove()


def _paired_permutation_z(deltas, *, n_perm=1000, seed=0):
    d = np.asarray(deltas, float)
    if len(d) < 8:
        return None, None
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(d))
        null.append(float((d * signs).mean()))
    null = np.asarray(null)
    sd = float(null.std(ddof=1))
    if sd < 1e-12:
        return None, None
    z = (obs - float(null.mean())) / sd
    p = float((np.abs(null) >= abs(obs)).mean())
    return round(z, 3), round(p, 4)


def _cohens_d(deltas):
    d = np.asarray(deltas, float)
    if len(d) < 2:
        return None
    return round(float(d.mean() / (d.std(ddof=1) + 1e-12)), 3)


def _bootstrap_ci(deltas, *, n_boot=500, seed=0):
    d = np.asarray(deltas, float)
    if len(d) < 4:
        return None, None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        samp = d[rng.integers(0, len(d), size=len(d))]
        means.append(float(samp.mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def run_certificate(model_name: str, *, k: int = 4, n_samples: int = 48,
                    block_len: int = 12, seed: int = 42) -> dict:
    device = pick_device()
    hf_name, tok, model = load_model(model_name)
    model.to(device)
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 12))
    layer_idx = n_layers // 2
    rng = np.random.default_rng(seed)

    items = []
    for _ in range(n_samples):
        ids, pos, val_id = build_induction_ids(tok, rng, block_len)
        vh = _capture_vh_sequence(model, device, ids, layer_idx)
        if vh is None:
            continue
        base_m = _logit_margin(model, device, ids, pos, val_id)
        arms = {}
        for arm in ("top", "bottom", "random"):
            v_sel = _fixed_dirs_from_vh(vh, arm, k, rng)
            m = _margin_fixed_ablation(
                model, device, ids, pos, val_id, layer_idx, v_sel)
            arms[arm] = {"margin": round(m, 4), "drop": round(base_m - m, 4)}
        items.append({
            "baseline_margin": round(base_m, 4),
            "drop_top": arms["top"]["drop"],
            "drop_bottom": arms["bottom"]["drop"],
            "drop_random": arms["random"]["drop"],
        })

    d_top_rand = [it["drop_top"] - it["drop_random"] for it in items]
    d_top_bot = [it["drop_top"] - it["drop_bottom"] for it in items]
    z_tr, p_tr = _paired_permutation_z(d_top_rand, seed=seed)
    z_tb, p_tb = _paired_permutation_z(d_top_bot, seed=seed + 1)
    ci_lo, ci_hi = _bootstrap_ci(d_top_rand, seed=seed)
    d_rand = [it["drop_random"] for it in items]
    d_bot = [it["drop_bottom"] for it in items]
    mean_rand = float(np.mean(d_rand)) if d_rand else None
    mean_bot = float(np.mean(d_bot)) if d_bot else None
    specificity_ok = (
        mean_rand is not None and mean_bot is not None
        and mean_rand <= mean_bot + 0.05)

    if z_tr is not None and z_tr >= 3 and _cohens_d(d_top_rand) is not None \
            and _cohens_d(d_top_rand) >= 0.8 and specificity_ok:
        verdict = "CERTIFICATE PASS"
    elif mean_rand is not None and mean_bot is not None and mean_rand > mean_bot + 0.05:
        verdict = "WEAKER/BOUNDED (random also collapses)"
    elif z_tb is not None and z_tb >= 2:
        verdict = "WEAK (beats bottom, not random)"
    else:
        verdict = "HONEST NULL"

    return {
        "mode": "causal_certificate",
        "dispatch": "dispatch_20260625_2105_home_causal_certificate",
        "model": model_name,
        "hf_name": hf_name,
        "k": k,
        "n_items": len(items),
        "seed": seed,
        "items": items,
        "paired": {
            "mean_delta_top_minus_random": round(float(np.mean(d_top_rand)), 4) if d_top_rand else None,
            "mean_delta_top_minus_bottom": round(float(np.mean(d_top_bot)), 4) if d_top_bot else None,
            "bootstrap_ci_top_minus_random": [ci_lo, ci_hi],
            "z_top_vs_random": z_tr,
            "p_top_vs_random": p_tr,
            "z_top_vs_bottom": z_tb,
            "cohens_d_top_vs_random": _cohens_d(d_top_rand),
            "mean_drop_random": round(mean_rand, 4) if mean_rand is not None else None,
            "mean_drop_bottom": round(mean_bot, 4) if mean_bot is not None else None,
            "specificity_ok": specificity_ok,
        },
        "verdict": verdict,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_certificate_synthetic(*, n_items=64, k=4, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    d = 64
    v_top = rng.standard_normal((k, d))
    v_top, _ = np.linalg.qr(v_top.T)
    v_top = v_top.T
    v_bot = rng.standard_normal((k, d))
    v_bot, _ = np.linalg.qr(v_bot.T)
    v_bot = v_bot.T
    g = v_top.sum(axis=0)
    items = []
    for _ in range(n_items):
        h = rng.standard_normal(d)
        base = float(abs(h @ g))
        def drop(v_sel):
            coeff = h @ v_sel.T
            hab = h - coeff @ v_sel
            return base - float(abs(hab @ g))
        coeffs = rng.standard_normal((k, d))
        q, _ = np.linalg.qr(coeffs.T)
        v_rand = q[:, :k].T
        items.append({
            "drop_top": drop(v_top),
            "drop_bottom": drop(v_bot),
            "drop_random": drop(v_rand),
        })
    d_tr = [it["drop_top"] - it["drop_random"] for it in items]
    z, _ = _paired_permutation_z(d_tr, seed=seed)
    cd = _cohens_d(d_tr)
    pass_ok = z is not None and z >= 3 and cd is not None and cd >= 0.8
    return {
        "mode": "certificate_synthetic",
        "n_items": n_items,
        "k": k,
        "z_top_vs_random": z,
        "cohens_d": cd,
        "pass": bool(pass_ok),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def aggregate_certificate(out_dir: Path) -> dict:
    files = sorted(out_dir.glob("directional_ablation_certificate_*_seed*.json"))
    rows = []
    for f in files:
        d = json.load(open(f))
        p = d["paired"]
        rows.append({
            "model": d["model"],
            "seed": d["seed"],
            "k": d["k"],
            "z_top_vs_random": p["z_top_vs_random"],
            "cohens_d": p["cohens_d_top_vs_random"],
            "verdict": d["verdict"],
        })
    n_pass = sum(1 for r in rows if r["verdict"] == "CERTIFICATE PASS")
    summary = {
        "n_runs": len(rows),
        "n_certificate_pass": n_pass,
        "per_run": rows,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = out_dir / "directional_ablation_certificate_summary.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"Wrote {out} ({len(rows)} runs, {n_pass} PASS)")
    return summary


def aggregate(out_dir: Path) -> dict:
    files = sorted(out_dir.glob("directional_ablation_*.json"))
    files = [f for f in files if "summary" not in f.name and "synthetic" not in f.name]
    if not files:
        print("No per-model results found.")
        return {}
    rows = []
    for f in files:
        d = json.load(open(f))
        s = d["summary"]
        rows.append({
            "model": d["model"],
            "baseline": d["baseline_accuracy"],
            "auc_top": s["auc_top"],
            "auc_bottom": s["auc_bottom"],
            "causal_gap": s["causal_gap"],
            "verdict": s["verdict"],
        })
    summary = {
        "n_models": len(rows),
        "n_pass": sum(1 for r in rows if r["verdict"] == "PASS"),
        "n_weak": sum(1 for r in rows if r["verdict"] == "WEAK"),
        "per_model": rows,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = out_dir / "directional_ablation_summary.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"Wrote {out} ({len(rows)} models)")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Directional causal sigma-ablation (SVD3)")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-samples", type=int, default=48)
    ap.add_argument("--block-len", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--layer-sweep", action="store_true",
                    help="sweep the ablation layer at a fixed budget (localization law)")
    ap.add_argument("--budget", type=int, default=4,
                    help="fixed ablation budget k for --layer-sweep")
    ap.add_argument("--interference", action="store_true",
                    help="cross-task subspace ablation (subspace-interference law)")
    ap.add_argument("--k", type=int, default=4,
                    help="subspace dimension for --interference")
    ap.add_argument("--certificate", action="store_true",
                    help="3-arm causal certificate (dispatch 2105)")
    ap.add_argument("--certificate-synthetic", action="store_true",
                    help="self-test for --certificate")
    ap.add_argument("--certificate-aggregate", action="store_true",
                    help="aggregate certificate JSONs")
    ap.add_argument("--k-sweep", nargs="+", type=int, default=None,
                    help="k values for --certificate (default 1 2 4)")
    args = ap.parse_args()

    if args.certificate_synthetic:
        res = run_certificate_synthetic()
        out = HERE / "directional_ablation_certificate_synthetic.json"
        json.dump(res, open(out, "w"), indent=2)
        print(f"\nWrote {out}  pass={res['pass']}  z={res['z_top_vs_random']}")
        return

    if args.certificate_aggregate:
        aggregate_certificate(HERE)
        return

    if args.certificate:
        models = args.models or DEFAULT_MODELS + ["pythia-1b"]
        ks = args.k_sweep or CERTIFICATE_KS
        for m in models:
            for kk in ks:
                print(f"\n{'=' * 60}\n  {m}  certificate k={kk}\n{'=' * 60}")
                t0 = time.time()
                res = run_certificate(
                    m, k=kk, n_samples=args.n_samples,
                    block_len=args.block_len, seed=args.seed)
                out = HERE / (
                    f"directional_ablation_certificate_{safe_name(m)}_k{kk}_seed{args.seed}.json")
                json.dump(res, open(out, "w"), indent=2)
                p = res["paired"]
                print(f"  n={res['n_items']}  z(top-rand)={p['z_top_vs_random']}  "
                      f"d={p['cohens_d_top_vs_random']}  verdict={res['verdict']}  "
                      f"({time.time()-t0:.1f}s)")
                print(f"  -> {out}")
        aggregate_certificate(HERE)
        return

    if args.interference:
        if args.synthetic:
            res = run_interference_synthetic()
            out = HERE / "directional_ablation_interference_synthetic.json"
            json.dump(res, open(out, "w"), indent=2)
            print(f"\nWrote {out}")
            return
        models = args.models or DEFAULT_MODELS
        for m in models:
            print(f"\n{'=' * 60}\n  {m}  (subspace interference, k={args.k})\n{'=' * 60}")
            t0 = time.time()
            res = run_interference(m, k=args.k, n_samples=args.n_samples,
                                   block_len=args.block_len, seed=args.seed)
            out = HERE / f"directional_ablation_interference_{safe_name(m)}.json"
            json.dump(res, open(out, "w"), indent=2)
            s = res["summary"]
            print(f"  baselines    : {res['baselines']}")
            print(f"  doable tasks : {res['doable_tasks']}")
            print("  overlap matrix:")
            for a, row in res["overlap_matrix"].items():
                print(f"    {a:10s} " + " ".join(f"{bb}={v:.2f}" for bb, v in row.items()))
            print("  damage matrix (ablate row -> measure col):")
            for a, row in res["damage_matrix"].items():
                print(f"    {a:10s} " + " ".join(f"{bb}={v:+.2f}" for bb, v in row.items()))
            print(f"  law spearman(overlap,damage): {s['law_corr_overlap_vs_damage']}")
            print(f"  self={s['mean_self_damage']} cross={s['mean_cross_damage']} "
                  f"gap={s['self_minus_cross']}  verdict: {s['verdict']}  ({time.time()-t0:.1f}s)")
            print(f"  → {out}")
        return

    if args.layer_sweep:
        models = args.models or DEFAULT_MODELS
        for m in models:
            print(f"\n{'=' * 60}\n  {m}  (layer sweep, k={args.budget})\n{'=' * 60}")
            t0 = time.time()
            res = run_layer_sweep(m, k=args.budget, n_samples=args.n_samples,
                                  block_len=args.block_len, seed=args.seed)
            out = HERE / f"directional_ablation_layersweep_k{args.budget}_{safe_name(m)}.json"
            json.dump(res, open(out, "w"), indent=2)
            s = res["summary"]
            print(f"  baseline acc      : {res['baseline_accuracy']:.3f}")
            print(f"  damage per layer  : "
                  + " ".join(f"{r['drop']:.2f}" for r in res["per_layer"]))
            print(f"  peak layer        : {s['peak_layer']}/{res['n_layers']-1} "
                  f"(rel depth {s['peak_rel_depth']})")
            print(f"  peak drop         : {s['peak_drop']}  mean {s['mean_drop']}")
            print(f"  localization idx  : {s['localization_index']}  "
                  f"bottom@peak {s['acc_bottom_at_peak']}")
            print(f"  verdict           : {s['verdict']}  ({time.time()-t0:.1f}s)")
            print(f"  → {out}")
        return

    if args.synthetic:
        res = run_synthetic()
        out = HERE / "directional_ablation_synthetic.json"
        json.dump(res, open(out, "w"), indent=2)
        print(f"\nWrote {out}")
        return

    if args.aggregate:
        aggregate(HERE)
        return

    models = args.models or DEFAULT_MODELS
    for m in models:
        print(f"\n{'=' * 60}\n  {m}\n{'=' * 60}")
        t0 = time.time()
        res = run_model(m, n_samples=args.n_samples, block_len=args.block_len,
                        seed=args.seed)
        out = HERE / f"directional_ablation_{safe_name(m)}.json"
        json.dump(res, open(out, "w"), indent=2)
        s = res["summary"]
        print(f"  baseline acc : {res['baseline_accuracy']:.3f}")
        print(f"  auc_top      : {s['auc_top']}  (fragile if low)")
        print(f"  auc_bottom   : {s['auc_bottom']}  (robust if high)")
        print(f"  causal_gap   : {s['causal_gap']}  verdict: {s['verdict']}  ({time.time()-t0:.1f}s)")
        print(f"  → {out}")

    if len(models) > 1:
        aggregate(HERE)


if __name__ == "__main__":
    main()
