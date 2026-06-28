#!/usr/bin/env python3
"""Home dispatch 1015 — confirm carry-exception mechanism on deep stacks.

Harvests depth_link cells from an existing JSON (0959/0752 panel); optionally
re-captures factor matrices for P5/P6 on deep models only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASKS = ("arithmetic_carry", "relation_composition", "induction")
DEEP_MODELS = ("gpt2-large", "gpt2-xl", "pythia-1b", "pythia-1.4b")
REF_24L = ("gpt2-medium", "pythia-410m")


def _load_depth_link(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell(results: dict, model: str, task: str) -> dict | None:
    return results.get(f"{model}|{task}")


def _second_half_slope(traj: list[float]) -> float:
    body = traj[:-1] if len(traj) > 4 else list(traj)
    h = body[len(body) // 2 :]
    if len(h) < 2:
        return float("nan")
    x = np.arange(len(h), dtype=float)
    return float(np.polyfit(x, np.asarray(h, float), 1)[0])


def _end_gap(r_true: list[float], r_free: list[float]) -> float:
    return float(r_true[-1] - r_free[-1])


def _table_row(rec: dict) -> dict:
    return {
        "lambda_true": rec.get("lambda_true"),
        "over_collapse_logratio": rec.get("over_collapse_logratio"),
        "rank_preservation_ratio": rec.get("rank_preservation_ratio"),
        "gap_slope": rec.get("gap_slope"),
        "handoff_minus_baseline": rec.get("handoff_minus_baseline"),
        "second_half_slope_r_true": round(_second_half_slope(rec["r_true_trajectory"]), 4),
        "end_r_true_minus_r_free": round(_end_gap(rec["r_true_trajectory"], rec["r_free_trajectory"]), 4),
        "n_layers": rec.get("n_layers"),
        "L_factors": rec.get("L_factors"),
    }


def _carry_slowest_lambda(rows: dict[str, dict]) -> bool:
    lam = {t: rows[t]["lambda_true"] for t in TASKS if t in rows}
    if len(lam) < 3:
        return False
    return lam["arithmetic_carry"] == max(lam.values())


def _exception_pattern(rows: dict[str, dict]) -> bool:
    c = rows.get("arithmetic_carry")
    if not c:
        return False
    carry_ok = c["rank_preservation_ratio"] >= 0.95 and c["over_collapse_logratio"] <= 0.05
    others = [rows[t] for t in ("relation_composition", "induction") if t in rows]
    if len(others) < 2:
        return False
    others_oc = all(o["over_collapse_logratio"] > 0.05 for o in others)
    others_rp = all(o["rank_preservation_ratio"] < 0.95 for o in others)
    return carry_ok and others_oc and others_rp


def _spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _late_gain_spectrum_anticorr(mats: list[np.ndarray]) -> float:
    """Mean Spearman(log s_acc, per-direction gain norm) on late body layers."""
    corrs = []
    L = len(mats)
    for i in range(L // 2, L - 1):
        P = mats[0]
        for j in range(1, i):
            P = mats[j] @ P
        U, s_acc, _ = np.linalg.svd(P, full_matrices=False)
        M = U.T @ mats[i]
        gains = np.linalg.norm(M, axis=1)
        n = min(len(gains), len(s_acc))
        corrs.append(_spearman(np.log(s_acc[:n] + 1e-12), gains[:n]))
    return float(np.nanmean(corrs)) if corrs else float("nan")


def _late_operator_coupling(mats: list[np.ndarray]) -> float:
    late = mats[len(mats) // 2 : -1]
    if len(late) < 2:
        return float("nan")
    cors = []
    for a, b in zip(late[:-1], late[1:]):
        fa, fb = a.ravel(), b.ravel()
        if np.std(fa) < 1e-12 or np.std(fb) < 1e-12:
            continue
        cors.append(float(np.corrcoef(fa, fb)[0, 1]))
    return float(np.mean(cors)) if cors else float("nan")


def _participation_gap_pattern(rec: dict, kappa: float | None) -> bool:
    rp = rec.get("rank_preservation_ratio", 1.0)
    oc = rec.get("over_collapse_logratio", 0.0)
    low_rank = rp < 0.85
    over = oc > 0.05
    free_shape = kappa is None or kappa >= 0.85
    return bool(low_rank and over and free_shape)


def harvest(depth: dict, models: tuple[str, ...]) -> dict:
    results = depth.get("results", {})
    per_model = {}
    for model in models:
        rows = {}
        for task in TASKS:
            rec = _cell(results, model, task)
            if rec:
                rows[task] = _table_row(rec)
        if rows:
            per_model[model] = {
                "tasks": rows,
                "P1_carry_slowest_lambda": _carry_slowest_lambda(rows),
                "P2_depth_gated_exception": _exception_pattern(rows),
            }
    return per_model


def _p1_verdict(per_model: dict) -> str:
    hits = sum(1 for m in per_model if per_model[m]["P1_carry_slowest_lambda"])
    n = len(per_model)
    need = max(3, (len(DEEP_MODELS) * 3) // 4)
    if hits >= need:
        return f"CONFIRMED — carry slowest lambda_true in {hits}/{n} deep models (need >={need})."
    return f"REFUTED/PARTIAL — carry slowest in only {hits}/{n} deep models (need >={need})."


def _p2_verdict(per_model: dict, ref: dict) -> str:
    deep_hits = sum(1 for m in per_model if per_model[m]["P2_depth_gated_exception"])
    ref_hits = sum(1 for m in ref if ref[m]["P2_depth_gated_exception"])
    widen = []
    for model in DEEP_MODELS:
        c = per_model.get(model, {}).get("tasks", {}).get("arithmetic_carry")
        if c:
            widen.append(c["end_r_true_minus_r_free"])
    ref_widen = []
    for model in REF_24L:
        c = ref.get(model, {}).get("tasks", {}).get("arithmetic_carry")
        if c:
            ref_widen.append(c["end_r_true_minus_r_free"])
    mean_deep = float(np.mean(widen)) if widen else float("nan")
    mean_ref = float(np.mean(ref_widen)) if ref_widen else float("nan")
    if deep_hits >= 3 and mean_deep >= mean_ref:
        return (f"CONFIRMED — exception in {deep_hits}/{len(per_model)} deep stacks; "
                f"carry end gap mean {mean_deep:.3f} vs 24L ref {mean_ref:.3f}.")
    return (f"PARTIAL — deep exception {deep_hits}/{len(per_model)}; "
            f"carry end gap deep {mean_deep:.3f} vs 24L {mean_ref:.3f}.")


def _p3_verdict(per_model: dict) -> str:
    hits = 0
    for m in per_model:
        rows = per_model[m]["tasks"]
        slopes = {t: rows[t]["second_half_slope_r_true"] for t in TASKS if t in rows}
        if len(slopes) == 3 and slopes["arithmetic_carry"] == max(slopes.values()):
            hits += 1
    need = 3
    if hits >= need:
        return f"CONFIRMED — carry least-negative 2nd-half product slope in {hits}/{len(per_model)} deep models."
    return f"PARTIAL — carry best plateau slope in {hits}/{len(per_model)} deep models (need >={need})."


def _p4_verdict(per_model: dict, ref: dict) -> str:
    deep_slopes = []
    ref_slopes = []
    for m in per_model:
        c = per_model[m]["tasks"].get("arithmetic_carry")
        if c:
            deep_slopes.append(c["second_half_slope_r_true"])
    for m in ref:
        c = ref[m]["tasks"].get("arithmetic_carry")
        if c:
            ref_slopes.append(c["second_half_slope_r_true"])
    if not deep_slopes or not ref_slopes:
        return "INCONCLUSIVE — missing carry slopes."
    if float(np.mean(deep_slopes)) > float(np.mean(ref_slopes)):
        return (f"CONFIRMED — deeper carry plateau stronger (mean 2nd-half slope "
                f"{np.mean(deep_slopes):+.3f} vs 24L {np.mean(ref_slopes):+.3f}).")
    return (f"PARTIAL — deep carry slope {np.mean(deep_slopes):+.3f} vs 24L "
            f"{np.mean(ref_slopes):+.3f}.")


def capture_p5_p6(models: tuple[str, ...], n_samples: int, r: int, seed: int) -> dict:
    from free_probability_transport_probe import _fit_factors_true, analyse_factors

    rng = np.random.default_rng(seed)
    out = {}
    for model in models:
        print(f"\n=== capture {model} ===", flush=True)
        _meta, captured = _fit_factors_true(model, n_samples=n_samples, r=r, seed=seed)
        model_cells = {}
        for task, (factor_svals, true_svals, _n, mats) in captured.items():
            anti = _late_gain_spectrum_anticorr(mats)
            coup = _late_operator_coupling(mats)
            kap = analyse_factors(factor_svals, true_svals, rng)["freeness_kappa"]
            model_cells[task] = {
                "late_gain_spectrum_spearman": round(anti, 4),
                "late_operator_coupling": round(coup, 4),
                "freeness_kappa": round(float(kap), 4),
            }
            print(f"  {task}: anti_corr={anti:+.3f} coupling={coup:+.3f} kappa={kap:.3f}",
                  flush=True)
        out[model] = model_cells
    return out


def _p5_verdict(capture: dict) -> str:
    hits = 0
    for model in DEEP_MODELS:
        cells = capture.get(model, {})
        if len(cells) < 3:
            continue
        anti = {t: cells[t]["late_gain_spectrum_spearman"] for t in TASKS}
        if anti["arithmetic_carry"] == min(anti.values()):
            hits += 1
    if hits >= 3:
        return f"CONFIRMED — carry most anti-correlated late gains in {hits}/{len(DEEP_MODELS)} deep models."
    return f"PARTIAL — carry most anti-correlated in {hits}/{len(DEEP_MODELS)} deep models."


def _p6_verdict(capture: dict, harvested: dict) -> str:
    gap_hits = 0
    for model in DEEP_MODELS:
        hrows = harvested.get(model, {}).get("tasks", {})
        crows = capture.get(model, {})
        for task in ("relation_composition", "induction"):
            if task not in hrows or task not in crows:
                continue
            if _participation_gap_pattern(hrows[task], crows[task].get("freeness_kappa")):
                gap_hits += 1
    carry_coup = [capture[m]["arithmetic_carry"]["late_operator_coupling"]
                  for m in DEEP_MODELS if m in capture and "arithmetic_carry" in capture[m]]
    other_coup = [capture[m][t]["late_operator_coupling"]
                  for m in DEEP_MODELS if m in capture
                  for t in ("relation_composition", "induction") if t in capture[m]]
    sep = (float(np.mean(carry_coup)) < float(np.mean(other_coup))
           if carry_coup and other_coup else False)
    if gap_hits >= 4 and sep:
        return (f"CONFIRMED — participation-gap on {gap_hits}/8 rel/ind cells; "
                f"carry lower late op coupling ({np.mean(carry_coup):.3f} vs {np.mean(other_coup):.3f}).")
    return f"PARTIAL — participation-gap {gap_hits}/8; coupling separation={sep}."


def run(depth_path: Path, capture: bool, n_samples: int, r: int, seed: int) -> dict:
    depth = _load_depth_link(depth_path)
    deep = harvest(depth, DEEP_MODELS)
    ref = harvest(depth, REF_24L)

    out = {
        "mode": "carry_exception_deepstack",
        "date": time.strftime("%Y-%m-%d"),
        "machine": "home",
        "source_depth_link": str(depth_path),
        "deep_models": list(DEEP_MODELS),
        "reference_24L": list(REF_24L),
        "per_model": deep,
        "reference_24L_tables": ref,
        "verdicts": {
            "P1_carry_slowest_lambda": _p1_verdict(deep),
            "P2_depth_gated_exception": _p2_verdict(deep, ref),
            "P3_carry_plateau_slope": _p3_verdict(deep),
            "P4_depth_strengthens_plateau": _p4_verdict(deep, ref),
        },
    }

    if capture:
        cap = capture_p5_p6(DEEP_MODELS, n_samples=n_samples, r=r, seed=seed)
        out["capture_p5_p6"] = cap
        out["verdicts"]["P5_late_gain_anticorr"] = _p5_verdict(cap)
        out["verdicts"]["P6_participation_gap_joint_corr"] = _p6_verdict(cap, deep)

    path = HERE / "carry_exception_deepstack.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== VERDICTS ===", flush=True)
    for k, v in out["verdicts"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"saved -> {path.name}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-link", type=Path,
                    default=HERE / "free_probability_transport_depth_link.json")
    ap.add_argument("--capture", action="store_true",
                    help="GPU re-capture deep models for P5/P6")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.depth_link, capture=args.capture, n_samples=args.n_samples,
        r=args.r, seed=args.seed)


if __name__ == "__main__":
    main()
