#!/usr/bin/env python3
"""Berry-curvature / spectral-gap probe — real-model test of the plastic-operator laws.

This is the bridge P1: it tests two KERNEL-VERIFIED plastic_operator laws
(elysium/fields/plastic_operator/plastic_operator_proof.py) on REAL transformer
activations, joining the verified holonomy/Berry theory to the empirically-scaling
transport geometry (the L3 moving frame, frame_geometry_probe.py).

The tracked object is the readout-relevant latent frame: at each layer l the
right-singular directions of the centred query activations, ranked by readout
energy e = sigma^2 * rho^2 (exactly the frame in cross_layer_transport_probe).
Tracking the top direction across depth is a discrete adiabatic transport of an
eigenvector; the "spectral gap" is the readout-energy gap that, by perturbation
theory, controls how fast that eigenvector turns.

Two verified laws, two predictions:
  * T5/Berry CONNECTION (T5c: tracking error ~ C*omega/gap):
        eigenvector rotation rate ~ 1/gap        -> log-log slope ~ -1
  * B3 Berry CURVATURE (F ~ 1/gap^2 at a degeneracy):
        task-bundle holonomy ||H-I|| ~ 1/gap^2   -> log-log slope ~ -2

The 1/gap^2 curvature has the SAME functional shape as the ICL paper's "raw
curvature law", which is NULL at scale. So this is decisive either way:
  * a clean negative slope  -> the verified plasticity laws ARE a real-model
    signature (and the transport geometry gets its formal Berry core);
  * a null/flat slope       -> the 1/gap(^2) law is synthetic-only, not a
    real transformer signature (an honest, important negative).

Each slope is scored against a PERMUTATION NULL (shuffle the gap<->rotation
pairing): a real law must beat the shuffled slope (|z| >= 2).

Local cache only (no training, no download beyond the HF model cache). Run:
  python3 berry_gap_probe.py --models gpt2 --n-samples 24
  python3 berry_gap_probe.py --models gpt2 gpt2-medium pythia-410m --n-samples 24
  python3 berry_gap_probe.py --aggregate
Output: berry_gap_<model>.json (+ berry_gap_summary.json on --aggregate)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = ["gpt2", "pythia-410m"]
DEFAULT_K = 4

from cross_layer_transport_probe import (  # noqa: E402
    capture_all_layers,
    layer_readout_subspace,
)
from frame_geometry_probe import _holonomy_metrics, _procrustes_q  # noqa: E402
from rotation_operator_probe import _domains_for_family, _make_family_prompt  # noqa: E402


def _line_angle(u, v):
    """Angle in [0, pi/2] between two unit directions (sign-free; a line in R^d)."""
    c = float(np.clip(abs(float(u @ v)), 0.0, 1.0))
    return float(np.arccos(c))


def _berry_coupling(u, v):
    """Saturation-correct Berry-connection magnitude: tan(angle) = |sin|/|cos|.

    The line angle in [0, pi/2] saturates at 90 deg (a line cannot turn further),
    which artificially flattens the 1/gap slope near a degeneracy. tan(angle) is
    the unbounded off-diagonal/diagonal overlap ratio |<v_j|dv_i>|/<v_i|v_i>, i.e.
    exactly the first-order perturbation coupling that diverges as 1/gap."""
    c = float(np.clip(abs(float(u @ v)), 1e-9, 1.0))
    s = float(np.sqrt(max(0.0, 1.0 - c * c)))
    return s / c


def _loglog_fit(gaps, ys):
    """Slope of log y vs log gap, with Spearman and n. Filters non-finite/positive."""
    g = np.asarray(gaps, float)
    y = np.asarray(ys, float)
    m = np.isfinite(g) & np.isfinite(y) & (g > 0) & (y > 0)
    g, y = g[m], y[m]
    if len(g) < 5 or np.std(np.log(g)) < 1e-9:
        return None
    slope = float(np.polyfit(np.log(g), np.log(y), 1)[0])
    rg = np.argsort(np.argsort(g))
    ry = np.argsort(np.argsort(y))
    spearman = float(np.corrcoef(rg, ry)[0, 1])
    return {"slope": round(slope, 3), "spearman": round(spearman, 3), "n": int(len(g))}


def _permutation_null(gaps, ys, *, n_rep=200, seed=0):
    """Null slope distribution from shuffling the gap<->y pairing. Returns the
    z-score of the real slope against the shuffled-slope distribution."""
    g = np.asarray(gaps, float)
    y = np.asarray(ys, float)
    m = np.isfinite(g) & np.isfinite(y) & (g > 0) & (y > 0)
    g, y = g[m], y[m]
    if len(g) < 5:
        return None
    lg, ly = np.log(g), np.log(y)
    real = float(np.polyfit(lg, ly, 1)[0])
    rng = np.random.default_rng(seed)
    null = np.array([np.polyfit(lg, rng.permutation(ly), 1)[0] for _ in range(n_rep)])
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    z = float((real - mu) / sd) if sd > 1e-12 else None
    return {"null_slope_mean": round(mu, 3), "null_slope_std": round(sd, 3),
            "z_score": round(z, 3) if z is not None else None, "n_rep": n_rep}


def _collect(model_name, n_samples, k, seed, task_family="icl"):
    """Per-domain per-layer readout frame + readout-energy spectrum."""
    from icl_convergence_probe import (safe_name, pick_device, load_model,
                                       get_final_norm_and_unembed)
    from transport_gain_probe import get_decoder_layers

    device = pick_device()
    hf_name, tok, model = load_model(model_name)
    model = model.to(device)
    norm, unembed = get_final_norm_and_unembed(model)
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    d_model = int(model.config.hidden_size)

    per_domain = {}
    for domain in _domains_for_family(task_family):
        rng = np.random.default_rng(seed)
        layer_h, layer_g = defaultdict(list), defaultdict(list)
        for _ in range(n_samples):
            seq, pos, cid = _make_family_prompt(task_family, domain, tok, rng)
            try:
                caps = capture_all_layers(model, layers, norm, unembed, seq, pos, cid, device)
            except Exception as e:  # noqa: BLE001
                print(f"  {domain}: prompt failed ({type(e).__name__}: {e})")
                continue
            for i, (hq, g) in caps.items():
                layer_h[i].append(hq)
                layer_g[i].append(g)
        present = [i for i in sorted(layer_h) if len(layer_h[i]) >= 3]
        frames = {}
        for i in present:
            S, rho, V = layer_readout_subspace(layer_h[i], layer_g[i], max(k, 2))
            e = (np.asarray(S) ** 2) * (np.asarray(rho) ** 2)   # readout energy per dir
            order = np.argsort(-e)
            e_sorted = e[order]
            if e_sorted[0] <= 0 or len(e_sorted) < 2:
                continue
            frames[i] = {
                "v_top": V[0],                                  # tracked top direction
                "v_k": V,                                       # top-k readout subspace
                "gap_top": float((e_sorted[0] - e_sorted[1]) / e_sorted[0]),
                "gap_sub": float((e_sorted[k - 1] - e_sorted[k]) / e_sorted[0])
                if len(e_sorted) > k else float(e_sorted[k - 1] / e_sorted[0]),
            }
        if len(frames) >= 3:
            per_domain[domain] = frames
    return hf_name, safe_name(model_name), n_layers, d_model, per_domain


def _connection_pairs(per_domain):
    """(gap, rotation_deg, berry_coupling) over adjacent layers for the tracked
    top direction. rotation_deg is the bounded line angle (saturates at 90 deg);
    berry_coupling = tan(angle) is the saturation-correct ~1/gap quantity."""
    gaps, rots, couplings = [], [], []
    for frames in per_domain.values():
        ls = sorted(frames)
        for a, b in zip(ls[:-1], ls[1:]):
            ang = _line_angle(frames[a]["v_top"], frames[b]["v_top"])
            gap = 0.5 * (frames[a]["gap_top"] + frames[b]["gap_top"])
            gaps.append(gap)
            rots.append(float(np.degrees(ang)))
            couplings.append(_berry_coupling(frames[a]["v_top"], frames[b]["v_top"]))
    return gaps, rots, couplings


def _curvature_pairs(per_domain):
    """(gap, {curvature measures}) over task-pair x adjacent-layer plaquettes.

    Three curvature measures so the 1/gap^2 verdict is not a measurement-ceiling
    artifact (the lesson the connection law taught): holonomy_fro and the trace
    angle both SATURATE (orthogonal H), so we also report the un-bounded
    commutator_norm (= the H-family curvature [transport, task-change], comm =
    kappa*theta_A*theta_B) and tan(trace_angle). Rank significance (Spearman) is
    measure-invariant, so if all three stay flat the null is genuine."""
    gaps = []
    hols, comms, tantr = [], [], []
    doms = sorted(per_domain)
    for i, a_dom in enumerate(doms):
        for b_dom in doms[i + 1:]:
            af, bf = per_domain[a_dom], per_domain[b_dom]
            common = sorted(set(af) & set(bf))
            for l0, l1 in zip(common[:-1], common[1:]):
                q_a = _procrustes_q(af[l0]["v_k"], af[l1]["v_k"])
                q_b = _procrustes_q(bf[l0]["v_k"], bf[l1]["v_k"])
                s_l = _procrustes_q(af[l0]["v_k"], bf[l0]["v_k"])
                s_next = _procrustes_q(af[l1]["v_k"], bf[l1]["v_k"])
                met = _holonomy_metrics(q_a, q_b, s_l, s_next)
                gap = 0.25 * (af[l0]["gap_sub"] + af[l1]["gap_sub"]
                              + bf[l0]["gap_sub"] + bf[l1]["gap_sub"])
                gaps.append(gap)
                hols.append(met["holonomy_fro"])
                comms.append(met["commutator_norm"])
                tantr.append(float(np.tan(np.clip(np.radians(met["trace_angle_deg"]), 0, 1.55))))
    return gaps, {"holonomy_fro": hols, "commutator_norm": comms,
                  "tan_trace_angle": tantr}


def _verdict(fit, null, predicted):
    """A law holds if the slope is near the prediction AND beats the permutation null."""
    if fit is None or null is None or null.get("z_score") is None:
        return "DARK"
    near = abs(fit["slope"] - predicted) <= 0.5
    sig = abs(null["z_score"]) >= 2.0 and fit["slope"] < 0
    if near and sig:
        return "LAW_HOLDS"
    if sig:
        return f"NEGATIVE_BUT_OFF_PREDICTION(slope={fit['slope']})"
    return "NULL_INDISTINGUISHABLE"


def analyze_model(model_name, n_samples=24, k=DEFAULT_K, seed=0, task_family="icl"):
    hf_name, safe, n_layers, d_model, per_domain = _collect(
        model_name, n_samples, k, seed, task_family)
    if not per_domain:
        print(f"{model_name}: no frames collected")
        return None

    g_c, r_c, b_c = _connection_pairs(per_domain)
    fit_b = _loglog_fit(g_c, b_c)          # primary: saturation-correct tan(angle)
    null_b = _permutation_null(g_c, b_c, seed=seed)
    fit_raw = _loglog_fit(g_c, r_c)        # secondary: bounded line angle (saturates)

    g_k, measures_k = _curvature_pairs(per_domain)
    curv_measures = {name: {"fit": _loglog_fit(g_k, ys),
                            "permutation_null": _permutation_null(g_k, ys, seed=seed)}
                     for name, ys in measures_k.items()}
    # Primary = the un-bounded commutator_norm (the H-family curvature object);
    # holonomy_fro/tan_trace are cross-checks against measurement saturation.
    fit_k = curv_measures["commutator_norm"]["fit"]
    null_k = curv_measures["commutator_norm"]["permutation_null"]

    out = {
        "mode": "real_model",
        "model": model_name,
        "hf_name": hf_name,
        "date": time.strftime("%Y-%m-%d"),
        "n_layers": n_layers,
        "d_model": d_model,
        "k_subspace": k,
        "n_samples": n_samples,
        "task_family": task_family,
        "n_domains": len(per_domain),
        "concept_anchor": "plastic_operator_proof.py  T5/Berry connection (~1/gap) + B3 Berry curvature (~1/gap^2)",
        "connection_law": {
            "prediction": "Berry coupling tan(angle) ~ 1/gap  (slope -1)",
            "predicted_slope": -1.0,
            "measure": "tan(line_angle) — saturation-correct off-diagonal coupling",
            "fit": fit_b,
            "permutation_null": null_b,
            "raw_angle_fit": fit_raw,
            "raw_angle_note": "bounded line angle saturates at 90 deg, flattening the slope",
            "verdict": _verdict(fit_b, null_b, -1.0),
        },
        "curvature_law": {
            "prediction": "curvature ~ 1/gap^2  (slope -2)",
            "predicted_slope": -2.0,
            "measure": "commutator_norm (un-bounded H-family curvature [transport,task])",
            "fit": fit_k,
            "permutation_null": null_k,
            "measures": curv_measures,
            "measures_note": "holonomy_fro and trace angle saturate (orthogonal H); "
                             "commutator_norm and tan_trace_angle are un-bounded. "
                             "Spearman is measure-invariant: all flat => genuine null.",
            "verdict": _verdict(fit_k, null_k, -2.0),
        },
    }
    safe_out = safe if task_family == "icl" else f"{safe}_{task_family}"
    path = HERE / f"berry_gap_{safe_out}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    c, cv = out["connection_law"], out["curvature_law"]
    print(f"  CONNECTION rotation~1/gap: slope={c['fit']['slope'] if c['fit'] else None} "
          f"(pred -1) spearman={c['fit']['spearman'] if c['fit'] else None} "
          f"z={c['permutation_null']['z_score'] if c['permutation_null'] else None} -> {c['verdict']}")
    print(f"  CURVATURE holonomy~1/gap^2: slope={cv['fit']['slope'] if cv['fit'] else None} "
          f"(pred -2) spearman={cv['fit']['spearman'] if cv['fit'] else None} "
          f"z={cv['permutation_null']['z_score'] if cv['permutation_null'] else None} -> {cv['verdict']}")
    return out


def aggregate():
    files = sorted(p for p in HERE.glob("berry_gap_*.json")
                   if not p.name.endswith("summary.json"))
    rows = []
    for p in files:
        d = json.loads(p.read_text())
        if d.get("mode") != "real_model":
            continue
        rows.append({
            "model": d["model"], "task_family": d.get("task_family", "icl"),
            "connection_slope": (d["connection_law"]["fit"] or {}).get("slope"),
            "connection_verdict": d["connection_law"]["verdict"],
            "curvature_slope": (d["curvature_law"]["fit"] or {}).get("slope"),
            "curvature_verdict": d["curvature_law"]["verdict"],
        })
    out = {"mode": "summary", "date": time.strftime("%Y-%m-%d"), "models": rows}
    (HERE / "berry_gap_summary.json").write_text(json.dumps(out, indent=2))
    print(f"saved -> berry_gap_summary.json  ({len(rows)} models)")
    for r in rows:
        print(f"  {r['model']:16s} {r['task_family']:8s} "
              f"conn slope={r['connection_slope']} [{r['connection_verdict']}]  "
              f"curv slope={r['curvature_slope']} [{r['curvature_verdict']}]")
    return out


def main():
    ap = argparse.ArgumentParser(description="Berry-curvature / spectral-gap real-model probe")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-family", choices=["icl", "routing", "factual"], default="icl")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
        return
    for m in args.models or DEFAULT_MODELS:
        print(f"\n{'=' * 60}\n  {m}  (berry-gap, family={args.task_family}, k={args.k})\n{'=' * 60}")
        try:
            analyze_model(m, n_samples=args.n_samples, k=args.k, seed=args.seed,
                          task_family=args.task_family)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"{m}: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
