#!/usr/bin/env python3
"""Confident-error geometry probe (MTI direction #2, v2 — the "coherent false belief" test).

Question (sharpened from the hallucination_risk v1.1 result): a hallucination is a
CONFIDENT bump of the model's coupled probability field that is false. Among the
CONFIDENT predictions, can internal transport-geometry *consistency* separate the
confident-WRONG (coherent confabulations) from the confident-CORRECT — i.e. exactly
where surface confidence is held fixed by construction and therefore cannot help?

This isolates the genuine open question: does a true belief have a thicker / more
self-consistent internal structure than a self-consistent false belief, given the
model has no truth oracle? We test it by restricting to {confidence >= tau} and
asking whether consistency geometry predicts correctness there.

Consistency features (all reuse existing machinery; novel ones are forward-only):
  - layer_pred_consistency   fraction of late layers whose logit-lens top-1 equals
                             the final top-1 (a coherent belief is stable in depth)
  - margin_monotonicity      fraction of non-decreasing steps of the emitted-token
                             logit across late layers (a coherent belief builds up)
  - margin_traj_stability    1 - normalized late-layer dispersion of that logit
  - support_participation    participation ratio of per-layer logit increments
                             (thick distributed support vs one dominating layer)
  - residual_noise_energy    off-readout-subspace mass of the query state (low=clean)
  - shadow_error             readout gradient energy invisible to the subspace

Baseline by construction: within the confident set, surface confidence barely
separates (narrow high band), so geometry_auc clearly above the within-confident
confidence AUC is the real signal.

Run (corp, no torch needed for the gate):
  uv run --no-project --with numpy --with scipy python confident_error_geometry_probe.py --synthetic
  python3 confident_error_geometry_probe.py --models gpt2-large pythia-410m pythia-1.4b --n-samples 128
  python3 confident_error_geometry_probe.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE))

from factual_recall_tasks import FACTUAL_DOMAINS, make_factual_prompt  # noqa: E402
from probe_base import (  # noqa: E402
    kfold_auc as _kfold_auc,
    point_biserial as _point_biserial,
    roc_auc as _roc_auc,
    leak_guard,
    load_context,
    Probe,
    register_probe,
)
from icl_convergence_probe import safe_name  # noqa: E402
from singular_spectrum_probe import capture_query_state  # noqa: E402
from cross_layer_transport_probe import layer_readout_subspace  # noqa: E402
# geometry feature helper stays with the probe that defines it
from hallucination_risk_probe import _subspace_features  # noqa: E402

DEFAULT_MODELS = ["gpt2-large", "pythia-410m", "pythia-1.4b"]
CONSISTENCY_FEATURES = [
    "layer_pred_consistency",
    "margin_monotonicity",
    "margin_traj_stability",
    "support_participation",
    "residual_noise_energy",
    "shadow_error",
]
DEFAULT_K = 4
DEFAULT_TAU = 0.5
LATE_FRAC = 0.5          # last half of layers define "late" depth
MIN_CLASS = 12           # min confident-wrong and confident-correct to evaluate


def _verdict(geometry_auc, lift_over_confidence, n_cw, n_cc) -> str:
    if n_cw < MIN_CLASS or n_cc < MIN_CLASS:
        return "INSUFFICIENT_LABELS"
    if geometry_auc is None or lift_over_confidence is None:
        return "GEOMETRY_NULL"
    if geometry_auc >= 0.65 and lift_over_confidence >= 0.05:
        return "GEOMETRY_SEPARATES_COHERENT_FALSE"
    if geometry_auc >= 0.58 and lift_over_confidence >= 0.02:
        return "GEOMETRY_WEAK"
    return "GEOMETRY_NULL"


def _forward_consistency(model, norm, unembed, input_ids, pos, device, tau):
    """One forward pass: final-token label + per-layer logit-lens consistency."""
    import torch

    ids = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states                       # len = n_blocks + 1 (incl. embed)
    n = len(hs)
    late0 = max(1, int(round(n * (1.0 - LATE_FRAC))))

    final_logits = unembed(norm(hs[-1][0, pos, :])).float()
    probs = torch.softmax(final_logits, dim=-1)
    top_id = int(torch.argmax(final_logits).item())
    confidence = float(probs[top_id].item())

    # per-layer logit-lens top-1 and the emitted-token logit trajectory
    same_top = []
    traj = []
    for li in range(n):
        logits = unembed(norm(hs[li][0, pos, :])).float()
        traj.append(float(logits[top_id].item()))
        if li >= late0:
            same_top.append(int(int(torch.argmax(logits).item()) == top_id))
    traj = np.asarray(traj, dtype=float)

    layer_pred_consistency = float(np.mean(same_top)) if same_top else 0.0

    late = traj[late0:]
    if late.size >= 2:
        steps = np.diff(late)
        margin_monotonicity = float(np.mean(steps >= 0.0))
        rng = float(late.max() - late.min())
        margin_traj_stability = float(1.0 / (1.0 + (np.std(late) / (rng + 1e-9))))
    else:
        margin_monotonicity = 0.0
        margin_traj_stability = 0.0

    deltas = np.diff(traj)
    s1 = float(np.sum(np.abs(deltas)))
    s2 = float(np.sum(deltas ** 2))
    support_participation = float((s1 * s1) / (len(deltas) * s2 + 1e-12)) if s2 > 0 else 0.0

    return {
        "top_id": top_id,
        "confidence": round(confidence, 4),
        "layer_pred_consistency": round(layer_pred_consistency, 4),
        "margin_monotonicity": round(margin_monotonicity, 4),
        "margin_traj_stability": round(margin_traj_stability, 4),
        "support_participation": round(support_participation, 4),
    }


def _eval_confident(records, tau=DEFAULT_TAU, n_folds=5, seed=0) -> dict:
    """Restrict to the confident set; positive class = confident-WRONG."""
    conf_all = np.array([r["confidence"] for r in records], dtype=float)
    confident = [r for r in records if r["confidence"] >= tau]
    n_cw = sum(1 for r in confident if r["correct"] == 0)
    n_cc = sum(1 for r in confident if r["correct"] == 1)

    base = {
        "n_total": len(records),
        "n_confident": len(confident),
        "n_confident_wrong": n_cw,
        "n_confident_correct": n_cc,
        "tau": tau,
        "confident_rate": round(len(confident) / max(1, len(records)), 4),
    }
    if n_cw < MIN_CLASS or n_cc < MIN_CLASS:
        base.update({
            "univariate": {}, "geometry_auc_kfold": None,
            "within_confident_confidence_auc": None,
            "geometry_lift_over_confidence": None,
            "verdict": _verdict(None, None, n_cw, n_cc),
        })
        return base

    y = np.array([1.0 if r["correct"] == 0 else 0.0 for r in confident])  # CW = 1
    conf = np.array([r["confidence"] for r in confident], dtype=float)

    univariate = {}
    for name in CONSISTENCY_FEATURES:
        col = np.array([r[name] for r in confident], dtype=float)
        univariate[name] = {
            "auc": _roc_auc(y, col),
            "point_biserial": _point_biserial(y, col),
        }

    # confidence baseline WITHIN the confident band (should be ~chance by design)
    within_conf_auc = _roc_auc(y, conf)
    geom_raw = np.array(
        [[r[f] for f in CONSISTENCY_FEATURES] for r in confident], dtype=float)
    geometry_auc = _kfold_auc(geom_raw, y, n_folds, seed)
    ref = within_conf_auc if within_conf_auc is not None else 0.5
    # confidence can separate in either direction; only its distance from chance counts
    ref = max(ref, 1.0 - ref)
    lift = (round(geometry_auc - ref, 4)
            if geometry_auc is not None else None)

    # Leak guard: no consistency feature may be a near-deterministic restatement
    # of the confident-wrong label.
    guard_records = [{**r, "_cw": 1.0 if r["correct"] == 0 else 0.0} for r in confident]
    guard = leak_guard(guard_records, "_cw", CONSISTENCY_FEATURES)
    verdict = _verdict(geometry_auc, lift, n_cw, n_cc)
    if guard["leak_detected"]:
        verdict = "LEAK_GUARD_FAIL"

    base.update({
        "consistency_features": CONSISTENCY_FEATURES,
        "univariate": univariate,
        "geometry_auc_kfold": geometry_auc,
        "within_confident_confidence_auc": within_conf_auc,
        "geometry_lift_over_confidence": lift,
        "global_confidence_auc_all": _roc_auc(
            np.array([1.0 if r["correct"] == 0 else 0.0 for r in records]), -conf_all),
        "leak_guard": guard,
        "verdict": verdict,
    })
    return base


def run_model(model_name, n_samples=128, seed=42, tau=DEFAULT_TAU, k=DEFAULT_K) -> dict:
    """Load the model, then compute. Use run_with_context to share a load."""
    return run_with_context(load_context(model_name), n_samples, seed, tau, k)


def run_with_context(ctx, n_samples=128, seed=42, tau=DEFAULT_TAU, k=DEFAULT_K) -> dict:
    model_name = ctx.name
    hf_name, tok, model = ctx.hf_name, ctx.tok, ctx.model
    device, norm, unembed = ctx.device, ctx.norm, ctx.unembed
    layers, l_c, n_params = ctx.layers, ctx.capacity_layer, ctx.n_params

    raw = []
    for domain in FACTUAL_DOMAINS:
        rng = np.random.default_rng(seed + hash(domain) % 10000)
        for _ in range(n_samples):
            try:
                seq, pos, cid = make_factual_prompt(domain, tok, rng)
                fc = _forward_consistency(model, norm, unembed, seq, pos, device, tau)
                h_q, g = capture_query_state(
                    model, layers, l_c, norm, unembed, seq, pos, cid, device)
                raw.append({
                    "domain": domain,
                    "correct": int(fc["top_id"] == cid),
                    **{kk: fc[kk] for kk in (
                        "confidence", "layer_pred_consistency", "margin_monotonicity",
                        "margin_traj_stability", "support_participation")},
                    "h": h_q, "g": g,
                })
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {domain}: {exc}", flush=True)

    if len(raw) < 4 * MIN_CLASS:
        raise RuntimeError(f"too few prompts for {model_name}: {len(raw)}")

    H = np.stack([r["h"] for r in raw])
    G = np.stack([r["g"] for r in raw])
    _sigma, _rho, V_lc = layer_readout_subspace(H, G, k)

    records = []
    for r in raw:
        sub = _subspace_features(r["h"], r["g"], V_lc)
        records.append({
            "domain": r["domain"], "correct": r["correct"],
            "confidence": r["confidence"],
            "layer_pred_consistency": r["layer_pred_consistency"],
            "margin_monotonicity": r["margin_monotonicity"],
            "margin_traj_stability": r["margin_traj_stability"],
            "support_participation": r["support_participation"],
            **sub,
        })

    metrics = _eval_confident(records, tau=tau, seed=seed)
    return {
        "model": model_name, "hf_name": hf_name, "device": device,
        "n_params": n_params, "capacity_layer": l_c, "readout_k": k,
        "n_samples_per_domain": n_samples, "seed": seed, "tau": tau,
        **metrics,
        "records": records,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_synthetic(n: int = 1200, seed: int = 0) -> dict:
    """Plant CW/CC separable by consistency features, with confidence held in a
    narrow high band (so within-confident confidence CANNOT separate). Verify the
    geometry classifier recovers it and the confidence baseline does not."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)                  # 1 = confident-wrong (coherent-false)
    # coherent-false has LOWER consistency than true (the planted, detectable tell)
    w = np.array([1.2, 1.0, 1.1, 0.9, 1.0, 0.8])
    base = rng.normal(size=(n, len(CONSISTENCY_FEATURES)))
    signal = base - (2 * y - 1)[:, None] * w[None, :] * 0.9   # CW shifted down
    conf = np.clip(0.80 + 0.10 * rng.random(n), 0.5, 0.999)   # all confident, label-free

    records = []
    for i in range(n):
        records.append({
            "domain": "synthetic",
            "correct": int(1 - y[i]),
            "confidence": round(float(conf[i]), 4),
            **{CONSISTENCY_FEATURES[j]: round(float(signal[i, j]), 4)
               for j in range(len(CONSISTENCY_FEATURES))},
        })

    m = _eval_confident(records, seed=seed)
    checks = {
        "geometry_auc_ge_0.75": bool(
            m["geometry_auc_kfold"] is not None and m["geometry_auc_kfold"] >= 0.75),
        "verdict_separates": m["verdict"] == "GEOMETRY_SEPARATES_COHERENT_FALSE",
        "lift_ge_0.05": bool(
            m["geometry_lift_over_confidence"] is not None
            and m["geometry_lift_over_confidence"] >= 0.05),
        "confidence_cannot_separate": bool(
            m["within_confident_confidence_auc"] is not None
            and abs(m["within_confident_confidence_auc"] - 0.5) <= 0.1),
    }
    return {
        "mode": "synthetic", "n": n, "seed": seed,
        "validation_checks": checks, "all_pass": all(checks.values()),
        **m, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def aggregate(out_dir: Path) -> dict:
    rows = []
    for p in sorted(out_dir.glob("confident_error_geometry_*.json")):
        if p.name in ("confident_error_geometry_summary.json",
                      "confident_error_geometry_synthetic.json"):
            continue
        d = json.loads(p.read_text())
        if d.get("mode") == "synthetic" or "verdict" not in d:
            continue
        rows.append(d)
    summary = {"n_models": len(rows), "models": [r["model"] for r in rows],
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if rows:
        summary["per_model"] = [{
            "model": r["model"], "n_params_M": round(r["n_params"] / 1e6, 1),
            "verdict": r["verdict"], "geometry_auc": r["geometry_auc_kfold"],
            "within_confident_confidence_auc": r["within_confident_confidence_auc"],
            "geometry_lift": r["geometry_lift_over_confidence"],
            "n_confident_wrong": r["n_confident_wrong"],
            "n_confident_correct": r["n_confident_correct"],
        } for r in rows]
        lifts = [r["geometry_lift_over_confidence"] for r in rows
                 if r["geometry_lift_over_confidence"] is not None]
        summary["n_separates"] = sum(
            r["verdict"] == "GEOMETRY_SEPARATES_COHERENT_FALSE" for r in rows)
        summary["mean_geometry_lift"] = round(float(np.mean(lifts)), 4) if lifts else None
    return summary


class ConfidentErrorGeometryProbe(Probe):
    """OOP front (delegates to the module functions; output schema unchanged).

    The leak-guard runs INTERNALLY in _eval_confident (it must be restricted to the
    confident subset and scored against the confident-wrong label, which is not a
    plain record field), so the generic Probe.leak_check is left not-applicable and
    the real guard block ships inside the result JSON.
    """

    name = "confident_error_geometry"
    classifier_features: list[str] = []   # guarded internally on the confident subset
    label_key = None

    def synthetic(self, seed: int = 0) -> dict:
        return run_synthetic(seed=seed)

    def run_model(self, model_name: str, *, n_samples: int = 128, seed: int = 42,
                  tau: float = DEFAULT_TAU, k: int = DEFAULT_K, **_) -> dict:
        return run_model(model_name, n_samples, seed, tau, k)

    def run_on(self, ctx, *, n_samples: int = 128, seed: int = 42,
               tau: float = DEFAULT_TAU, k: int = DEFAULT_K, **_) -> dict:
        return run_with_context(ctx, n_samples, seed, tau, k)


register_probe(ConfidentErrorGeometryProbe())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        res = run_synthetic(seed=args.seed)
        (HERE / "confident_error_geometry_synthetic.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        print("\nSYNTHETIC VALIDATION:", "ALL PASS" if res["all_pass"] else "FAIL")
        if not res["all_pass"]:
            raise SystemExit(1)
        return

    if args.aggregate:
        summary = aggregate(HERE)
        (HERE / "confident_error_geometry_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    for m in (args.models or DEFAULT_MODELS):
        print(f"=== {m} ===", flush=True)
        try:
            res = run_model(m, args.n_samples, args.seed, args.tau, args.k)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {m}: {exc}", flush=True)
            continue
        out = HERE / f"confident_error_geometry_{safe_name(m)}.json"
        out.write_text(json.dumps(res, indent=2))
        print(f"  saved -> {out.name}")
        print(f"  verdict={res['verdict']} geom_auc={res['geometry_auc_kfold']} "
              f"within_conf_auc={res['within_confident_confidence_auc']} "
              f"lift={res['geometry_lift_over_confidence']} "
              f"(CW={res['n_confident_wrong']} CC={res['n_confident_correct']})", flush=True)


if __name__ == "__main__":
    main()
