#!/usr/bin/env python3
"""Aggregate berry_gap multi-seed runs into per-config slope mean +/- CI."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_SEED_RE = re.compile(r"^berry_gap_(.+)_seed(\d+)\.json$")


def _parse_name(stem: str):
    if stem.endswith("_factual"):
        return stem[: -len("_factual")], "factual"
    return stem, "icl"


def _ci95(slopes):
    s = np.asarray(slopes, float)
    if len(s) < 2:
        return None, None
    mu = float(s.mean())
    se = float(s.std(ddof=1) / np.sqrt(len(s)))
    return round(mu, 4), round(1.96 * se, 4)


def aggregate(pattern_dir: Path | None = None):
    root = pattern_dir or HERE
    groups: dict[tuple[str, str], list[float]] = {}
    for p in sorted(root.glob("berry_gap_*_seed*.json")):
        m = _SEED_RE.match(p.name)
        if not m:
            continue
        d = json.loads(p.read_text())
        if d.get("mode") != "real_model":
            continue
        model, family = _parse_name(m.group(1))
        fit = (d.get("connection_law") or {}).get("fit") or {}
        slope = fit.get("slope")
        if slope is None:
            continue
        groups.setdefault((model, family), []).append(float(slope))

    rows = []
    for (model, family), slopes in sorted(groups.items()):
        mu, ci = _ci95(slopes)
        rows.append({
            "model": model,
            "task_family": family,
            "n_seeds": len(slopes),
            "connection_slope_mean": mu,
            "connection_slope_ci95": ci,
            "connection_slopes": [round(s, 4) for s in slopes],
            "near_minus_one": bool(mu is not None and abs(mu + 1.0) <= 0.35),
        })

    out = {"mode": "multiseed_ci", "date": time.strftime("%Y-%m-%d"), "configs": rows}
    path = root / "berry_gap_multiseed_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}  ({len(rows)} configs)")
    for r in rows:
        print(f"  {r['model']:16s} {r['task_family']:8s} "
              f"slope={r['connection_slope_mean']}+/-{r['connection_slope_ci95']} "
              f"(n={r['n_seeds']})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--dir", type=Path, default=None)
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.dir)


if __name__ == "__main__":
    main()
