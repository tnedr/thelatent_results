#!/usr/bin/env python3
"""Development-law breadth — co-occurrence summary (dispatch 2107).

Aggregates singular_spectrum_probe --developmental JSONs and tests whether
eff-rank sharpest concentration co-occurs with SVD3 alignment onset.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DISPATCH = "dispatch_20260625_2107_home_devlaw_breadth"
DOMAINS = ("induction", "relation_composition", "arithmetic_carry", "factual_recall")


def _sharpest_rank_step(dev_traj, domain):
    series = dev_traj.get(domain)
    if not series:
        return None
    steps = series["steps"]
    ranks = series["readout_rank"]
    if len(ranks) < 3:
        return None
    dr = np.diff(np.asarray(ranks, float))
    i = int(np.argmax(-dr))  # largest drop
    return int(steps[i + 1])


def _co_occur(knee_step, onset_step, *, tol_log2=True):
    if knee_step is None or onset_step is None:
        return None
    if tol_log2:
        la = np.log2(max(knee_step, 1))
        lb = np.log2(max(onset_step, 1))
        return bool(abs(la - lb) <= 1.0)
    return bool(knee_step == onset_step)


def _verdict(rate, n_cases):
    if n_cases < 4:
        return "HONEST NULL (too few cases)"
    if rate >= 0.8:
        return "LAW (promote)"
    if rate >= 0.5:
        return "REGIME-BOUND"
    return "HONEST NULL (co-occurrence fails on breadth panel)"


def aggregate_breadth():
    files = sorted(HERE.glob("singular_spectrum_developmental_*.json"))
    if not files:
        print("No developmental JSONs found.")
        return {}
    cases = []
    per_model = {}
    for p in files:
        d = json.loads(p.read_text())
        model = d["model"]
        dev_traj = d.get("developmental_trajectory", {})
        knees = d.get("knees", {})
        aligns = d.get("alignment_onset", {})
        model_cases = []
        for dom in DOMAINS:
            if dom not in knees or dom not in aligns:
                continue
            knee_step = knees[dom].get("knee_step")
            if knee_step is None:
                knee_step = _sharpest_rank_step(dev_traj, dom)
            onset_step = aligns[dom].get("onset_step")
            co = _co_occur(knee_step, onset_step)
            row = {
                "model": model,
                "domain": dom,
                "knee_step": knee_step,
                "onset_step": onset_step,
                "co_occur": co,
            }
            cases.append(row)
            model_cases.append(row)
        per_model[model] = model_cases

    scored = [c for c in cases if c["co_occur"] is not None]
    n_hit = sum(1 for c in scored if c["co_occur"])
    rate = float(n_hit / len(scored)) if scored else 0.0
    verdict = _verdict(rate, len(scored))
    summary = {
        "mode": "devlaw_breadth",
        "dispatch": DISPATCH,
        "date": time.strftime("%Y-%m-%d"),
        "n_models": len(per_model),
        "n_cases": len(scored),
        "co_occurrence_rate": round(rate, 4),
        "verdict": verdict,
        "cases": cases,
        "per_model": per_model,
    }
    out = HERE / "devlaw_breadth_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"saved -> {out.name}")
    print(f"  co-occur {n_hit}/{len(scored)} ({rate:.1%})  verdict: {verdict}")
    for m, rows in per_model.items():
        hits = sum(1 for r in rows if r.get("co_occur"))
        print(f"  {m}: {hits}/{len(rows)}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Dev-law breadth aggregate (dispatch 2107)")
    ap.add_argument("--aggregate", action="store_true", help="co-occurrence from developmental JSONs")
    args = ap.parse_args()
    if args.aggregate:
        aggregate_breadth()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
