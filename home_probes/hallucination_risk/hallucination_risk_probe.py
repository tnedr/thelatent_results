#!/usr/bin/env python3
"""Geometry-aware hallucination risk probe (MTI program direction #2).

Predicts confident factual wrong answers (hallucinations) from internal transport
geometry features, vs a confidence-only baseline.

Run:
  python3 hallucination_risk_probe.py --synthetic
  python3 hallucination_risk_probe.py --models gpt2 gpt2-large pythia-410m pythia-1.4b --n-samples 64
  python3 hallucination_risk_probe.py --aggregate

Output: hallucination_risk_<model>.json, hallucination_risk_synthetic.json,
        hallucination_risk_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from factual_recall_tasks import FACTUAL_DOMAINS, make_factual_prompt  # noqa: E402
from icl_convergence_probe import (  # noqa: E402
    get_final_norm_and_unembed,
    load_model,
    pick_device,
    safe_name,
)
from routing_selection_probe import contraction_metrics  # noqa: E402
from singular_spectrum_probe import (  # noqa: E402
    _capacity_layer_index,
    capture_query_state,
)
from cross_layer_transport_probe import layer_readout_subspace  # noqa: E402
from transport_gain_probe import get_decoder_layers  # noqa: E402

DEFAULT_MODELS = ["gpt2", "gpt2-large", "pythia-410m", "pythia-1.4b"]
FEATURES = [
    "transport_gain",
    "calibration_gap",
    "residual_noise_energy",
    "shadow_error",
]
DEFAULT_K = 4
DEFAULT_TAU = 0.5


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    if len(y) < 4 or y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(-s)
    y = y[order]
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    tpr = np.cumsum(y) / n_pos
    fpr = np.cumsum(1.0 - y) / n_neg
    return float(np.trapezoid(tpr, fpr))


def _point_biserial(y: np.ndarray, x: np.ndarray) -> float | None:
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if y.std() < 1e-12 or x.std() < 1e-12:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 4)


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _fit_logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    n, p = X.shape
    X1 = np.hstack([np.ones((n, 1)), X])

    def nll(beta: np.ndarray) -> float:
        z = np.clip(X1 @ beta, -30.0, 30.0)
        p_hat = 1.0 / (1.0 + np.exp(-z))
        return float(-np.sum(
            y * np.log(p_hat + 1e-12) + (1.0 - y) * np.log(1.0 - p_hat + 1e-12)
        ))

    res = minimize(nll, np.zeros(p + 1), method="L-BFGS-B")
    return res.x


def _predict_logistic(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X1 = np.hstack([np.ones((len(X), 1)), X])
    z = np.clip(X1 @ beta, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _verdict(geometry_auc: float | None, auc_lift: float | None) -> str:
    if geometry_auc is None or auc_lift is None:
        return "GEOMETRY_NULL"
    if geometry_auc >= 0.65 and auc_lift >= 0.03:
        return "GEOMETRY_PREDICTS_HALLUCINATION"
    if geometry_auc >= 0.55 or auc_lift > 0.0:
        return "GEOMETRY_WEAK"
    return "GEOMETRY_NULL"


def _eval_records(
    records: list[dict],
    tau: float = DEFAULT_TAU,
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    y = np.array([r["hallucination"] for r in records], dtype=float)
    conf = np.array([r["confidence"] for r in records], dtype=float)
    X_raw = np.array([[r[f] for f in FEATURES] for r in records], dtype=float)
    X, _mu, _sd = _standardize(X_raw)

    univariate = {}
    for j, name in enumerate(FEATURES):
        univariate[name] = {
            "auc": _roc_auc(y, X_raw[:, j]),
            "point_biserial": _point_biserial(y, X_raw[:, j]),
        }

    baseline_auc = _roc_auc(y, conf)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, n_folds)
    geom_scores = np.zeros(len(y))
    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        beta = _fit_logistic(X[train_idx], y[train_idx])
        geom_scores[test_idx] = _predict_logistic(X[test_idx], beta)

    geometry_auc = _roc_auc(y, geom_scores)
    auc_lift = (
        round(geometry_auc - baseline_auc, 4)
        if geometry_auc is not None and baseline_auc is not None
        else None
    )

    n_correct = int(sum(r["correct"] for r in records))
    n_hall = int(y.sum())
    by_domain: dict[str, dict] = {}
    for dom in sorted({r["domain"] for r in records}):
        sub = [r for r in records if r["domain"] == dom]
        yd = np.array([r["hallucination"] for r in sub], dtype=float)
        by_domain[dom] = {
            "n": len(sub),
            "n_hallucination": int(yd.sum()),
            "hallucination_rate": round(float(yd.mean()), 4),
        }

    return {
        "n_queries": len(records),
        "tau": tau,
        "label_counts": {
            "correct": n_correct,
            "wrong": len(records) - n_correct,
            "hallucination": n_hall,
            "abstention_wrong": (len(records) - n_correct) - n_hall,
        },
        "univariate": univariate,
        "baseline_confidence_auc": baseline_auc,
        "geometry_auc_kfold": geometry_auc,
        "auc_lift_over_confidence": auc_lift,
        "verdict": _verdict(geometry_auc, auc_lift),
        "by_domain": by_domain,
    }


def _forward_labels_and_margin(
    model, norm, unembed, input_ids, pos, correct_id, device, tau: float,
) -> dict:
    import torch

    ids = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states
    margins = []
    for li in range(len(hs)):
        h = hs[li][0, pos, :]
        logits = unembed(norm(h)).float()
        correct_logit = float(logits[correct_id].item())
        masked = logits.clone()
        masked[correct_id] = float("-inf")
        other_max = float(masked.max().item())
        margins.append(correct_logit - other_max)
    margins = np.asarray(margins, dtype=float)
    e_ra = np.log1p(np.exp(-np.clip(margins, -60.0, 60.0)))
    rho_ra = contraction_metrics(e_ra)["rho"]

    final_logits = unembed(norm(hs[-1][0, pos, :])).float()
    probs = torch.softmax(final_logits, dim=-1)
    top_id = int(torch.argmax(final_logits).item())
    confidence = float(probs[top_id].item())
    correct = int(top_id == correct_id)
    hallucination = int((not correct) and (confidence >= tau))

    return {
        "correct": correct,
        "confidence": round(confidence, 4),
        "hallucination": hallucination,
        "transport_gain": round(float(rho_ra), 4),
        "calibration_gap": round(confidence - float(correct), 4),
    }


def _subspace_features(h_q: np.ndarray, g: np.ndarray, V_lc: np.ndarray) -> dict:
    h_norm2 = float(np.dot(h_q, h_q)) + 1e-12
    g_norm2 = float(np.dot(g, g)) + 1e-12
    if V_lc is None or V_lc.shape[0] == 0:
        return {
            "residual_noise_energy": 1.0,
            "shadow_error": 1.0,
        }
    proj_h = V_lc @ h_q
    in_sub = float(np.dot(proj_h, proj_h))
    residual_noise_energy = max(0.0, min(1.0, (h_norm2 - in_sub) / h_norm2))
    g_proj = V_lc.T @ (V_lc @ g)
    g_perp = g - g_proj
    shadow_error = max(0.0, min(1.0, float(np.dot(g_perp, g_perp)) / g_norm2))
    return {
        "residual_noise_energy": round(residual_noise_energy, 4),
        "shadow_error": round(shadow_error, 4),
    }


def run_model(
    model_name: str,
    n_samples: int = 64,
    seed: int = 42,
    tau: float = DEFAULT_TAU,
    k: int = DEFAULT_K,
) -> dict:
    device = pick_device()
    hf_name, tok, model = load_model(model_name)
    model.to(device)
    norm, unembed = get_final_norm_and_unembed(model)
    layers = get_decoder_layers(model)
    l_c = _capacity_layer_index(len(layers))
    n_params = sum(p.numel() for p in model.parameters())

    raw_rows: list[dict] = []
    for domain in FACTUAL_DOMAINS:
        rng = np.random.default_rng(seed + hash(domain) % 10000)
        for _ in range(n_samples):
            try:
                seq, pos, cid = make_factual_prompt(domain, tok, rng)
                base = _forward_labels_and_margin(
                    model, norm, unembed, seq, pos, cid, device, tau)
                h_q, g = capture_query_state(
                    model, layers, l_c, norm, unembed, seq, pos, cid, device)
                raw_rows.append({
                    "domain": domain,
                    **base,
                    "h": h_q,
                    "g": g,
                })
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {domain}: {exc}", flush=True)

    if len(raw_rows) < 12:
        raise RuntimeError(f"too few factual prompts for {model_name}: {len(raw_rows)}")

    H = np.stack([r["h"] for r in raw_rows])
    G = np.stack([r["g"] for r in raw_rows])
    _sigma, _rho, V_lc = layer_readout_subspace(H, G, k)

    records = []
    for row in raw_rows:
        sub = _subspace_features(row["h"], row["g"], V_lc)
        records.append({
            "domain": row["domain"],
            "correct": row["correct"],
            "confidence": row["confidence"],
            "hallucination": row["hallucination"],
            "transport_gain": row["transport_gain"],
            "calibration_gap": row["calibration_gap"],
            **sub,
        })

    metrics = _eval_records(records, tau=tau, seed=seed)
    return {
        "model": model_name,
        "hf_name": hf_name,
        "device": device,
        "n_params": n_params,
        "capacity_layer": l_c,
        "readout_k": k,
        "n_samples_per_domain": n_samples,
        "seed": seed,
        "tau": tau,
        **metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_synthetic(n: int = 1000, seed: int = 0) -> dict:
    """Plant hallucination labels from geometry features + noise; verify recovery."""
    rng = np.random.default_rng(seed)
    w = np.array([0.9, 1.4, 1.1, 1.2])
    X_geom = rng.normal(size=(n, 4))
    logits = X_geom @ w + rng.normal(scale=0.35, size=n)
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = (prob >= 0.5).astype(int)
    correct = 1 - y
    # Weak confidence baseline (not the planted geometry signal)
    conf = np.clip(0.35 + 0.15 * rng.random(n), 0.0, 1.0)

    records = []
    for i in range(n):
        records.append({
            "domain": "synthetic",
            "correct": int(correct[i]),
            "confidence": round(float(conf[i]), 4),
            "hallucination": int(y[i]),
            "transport_gain": round(float(X_geom[i, 0]), 4),
            "calibration_gap": round(float(conf[i] - float(correct[i])), 4),
            "residual_noise_energy": round(float(abs(X_geom[i, 2])), 4),
            "shadow_error": round(float(abs(X_geom[i, 3])), 4),
        })

    metrics = _eval_records(records, seed=seed)
    planted_auc = _roc_auc(y, prob)
    checks = {
        "geometry_auc_ge_0.75": bool(
            metrics["geometry_auc_kfold"] is not None
            and metrics["geometry_auc_kfold"] >= 0.75),
        "verdict_predicts": metrics["verdict"] == "GEOMETRY_PREDICTS_HALLUCINATION",
        "lift_ge_0.03": bool(
            metrics["auc_lift_over_confidence"] is not None
            and metrics["auc_lift_over_confidence"] >= 0.03),
        "planted_prob_auc_ge_0.80": bool(
            planted_auc is not None and planted_auc >= 0.80),
    }
    return {
        "mode": "synthetic",
        "n": n,
        "seed": seed,
        "planted_oracle_auc": planted_auc,
        "validation_checks": checks,
        "all_pass": all(checks.values()),
        **metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def aggregate(out_dir: Path) -> dict:
    rows = []
    for p in sorted(out_dir.glob("hallucination_risk_*.json")):
        if p.name in ("hallucination_risk_summary.json", "hallucination_risk_synthetic.json"):
            continue
        d = json.loads(p.read_text())
        if d.get("mode") == "synthetic" or "verdict" not in d:
            continue
        rows.append(d)
    summary = {
        "n_models": len(rows),
        "models": [r["model"] for r in rows],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if rows:
        summary["per_model"] = [
            {
                "model": r["model"],
                "n_params_M": round(r["n_params"] / 1e6, 1),
                "verdict": r["verdict"],
                "geometry_auc": r["geometry_auc_kfold"],
                "baseline_auc": r["baseline_confidence_auc"],
                "auc_lift": r["auc_lift_over_confidence"],
                "n_hallucination": r["label_counts"]["hallucination"],
            }
            for r in rows
        ]
        verdicts = [r["verdict"] for r in rows]
        lifts = [r["auc_lift_over_confidence"] for r in rows
                 if r["auc_lift_over_confidence"] is not None]
        summary["n_predicts"] = sum(v == "GEOMETRY_PREDICTS_HALLUCINATION" for v in verdicts)
        summary["mean_auc_lift"] = round(float(np.mean(lifts)), 4) if lifts else None
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    out_dir = HERE
    if args.synthetic:
        res = run_synthetic(seed=args.seed)
        path = out_dir / "hallucination_risk_synthetic.json"
        path.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        print("\nSYNTHETIC VALIDATION:", "ALL PASS" if res["all_pass"] else "FAIL")
        if not res["all_pass"]:
            raise SystemExit(1)
        return

    if args.aggregate:
        summary = aggregate(out_dir)
        (out_dir / "hallucination_risk_summary.json").write_text(
            json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    models = args.models or DEFAULT_MODELS
    for m in models:
        print(f"=== {m} ===", flush=True)
        try:
            res = run_model(m, args.n_samples, args.seed, args.tau, args.k)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {m}: {exc}", flush=True)
            continue
        out = out_dir / f"hallucination_risk_{safe_name(m)}.json"
        out.write_text(json.dumps(res, indent=2))
        print(f"  saved -> {out.name}")
        print(f"  verdict={res['verdict']}  geometry_auc={res['geometry_auc_kfold']}  "
              f"lift={res['auc_lift_over_confidence']}", flush=True)


if __name__ == "__main__":
    main()
