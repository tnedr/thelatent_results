#!/usr/bin/env python3
"""Developmental Lyapunov probe — emergence = Lyapunov zero-crossing during training.

Spec: topics/ml_mathematics_of_trained_intelligence/OPERATOR_LIFT_SPEC.md
      (Part G5 / G7; kernel theorem `lyapunov_zero_crossing_emergence`).
Direction: RESEARCH_DIRECTION.md "Big-Bang Candidates" #1.

THE CLAIM (falsifiable, pre-registered). The transport operator's leading
Lyapunov exponent for a task direction,

    lambda_max = (1/D) * log sigma_top(T),    T = prod_l J_l  (capacity layer -> end),
    D = transport depth (#layers traversed),

is a DYNAMICAL order parameter for capability. A task direction is "dark"
(contracting, no in-context capacity) when lambda_max < 0  (sigma_top < 1) and
"bright" (expanding/preserving) when lambda_max >= 0  (sigma_top >= 1). The
emergence claim is the developmental statement of the kernel theorem
`lyapunov_zero_crossing_emergence`:

    During training, lambda_max for a task direction crosses 0 (negative ->
    non-negative) at the SAME step the task becomes in-context learnable.

THE TEST (canonical, with known ground truth). The cleanest emergent capability
in open checkpointed models is the INDUCTION head (Olsson et al. 2022): verbatim
copy of a repeated block, which turns on abruptly mid-training in the Pythia
family (the "induction bump", ~step 1-10k). So we sweep Pythia TRAINING
CHECKPOINTS and, at each, measure BOTH:

  (1) lambda_max on the induction task subspace  (F0 operator spectrum), and
  (2) induction in-context accuracy  (argmax next-token == the copy target).

If the claim holds, the lambda_max zero-crossing step ALIGNS with the
induction-accuracy onset step (and lambda_max co-rises with accuracy across
checkpoints). If lambda_max crosses much before/after the capability turns on,
or does not co-vary with it, the dynamical-order-parameter claim is FALSE.

Checkpoint loading: load_context("EleutherAI/pythia-410m@step1000") — the
"name@revision" syntax added to icl_convergence_probe.load_model. Requires
ICL_ALLOW_DOWNLOAD=1 (home/hetzner; corp is local-cache-only by OpSec).

SYNTHETIC gate (numpy only, no torch): planted lambda(step) that rises through 0
at a known step t0 and a planted capability sigmoid turning on at t_cap; the
crossing/onset detectors must recover t0 and t_cap, the alignment verdict must
fire when t0 == t_cap and NOT fire when they are far apart, and a never-crossing
control must report no crossing.

Run:
  python3 developmental_lyapunov_probe.py --synthetic
  ICL_ALLOW_DOWNLOAD=1 python3 developmental_lyapunov_probe.py \
      --base EleutherAI/pythia-410m \
      --revisions step0 step512 step1000 step2000 step4000 step8000 \
                  step16000 step32000 step64000 step143000
Output: developmental_lyapunov_<base>.json (+ _synthetic.json)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

import probe_base
from probe_base import Probe, load_context, register_probe

HERE = Path(__file__).resolve().parent

DEFAULT_BASE = "EleutherAI/pythia-410m"
# Log-spaced checkpoints bracketing the Pythia induction phase change.
DEFAULT_REVISIONS = [
    "step0", "step512", "step1000", "step2000", "step4000",
    "step8000", "step16000", "step32000", "step64000", "step143000",
]
DEFAULT_K = 8           # leading operator singular values
DEFAULT_ITERS = 12      # subspace-iteration sweeps
DEFAULT_OP_SAMPLES = 10   # induction prompts for the operator spectrum (expensive)
DEFAULT_ACC_SAMPLES = 96  # induction prompts for accuracy (cheap forward only)
DEFAULT_BLOCK = 8         # induction block length

ALIGN_FACTOR = 3.0      # primary: crossing within this factor of onset on step axis
SPEARMAN_MIN = 0.7      # secondary: lambda_max co-rises with accuracy


# ==========================================================================
# Trajectory analysis (shared by synthetic + real; numpy only).
# ==========================================================================
def _step_of(revision: str) -> float:
    m = re.search(r"(\d+)", revision)
    return float(m.group(1)) if m else 0.0


def _log_step(s: float) -> float:
    return float(np.log10(max(float(s), 1.0)))


def zero_crossing_step(steps, values, level=0.0):
    """First step (log-step-interpolated) where `values` rises through `level`.
    Returns a float step, or None if it never rises through `level`."""
    s = np.asarray(steps, float)
    v = np.asarray(values, float)
    for i in range(1, len(v)):
        if v[i - 1] < level <= v[i]:
            x0, x1 = _log_step(s[i - 1]), _log_step(s[i])
            t = (level - v[i - 1]) / (v[i] - v[i - 1] + 1e-30)
            return float(10 ** (x0 + t * (x1 - x0)))
    return None


def _spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return None
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def crossing_alignment(traj: list[dict]) -> dict:
    """Given per-checkpoint records with `step`, `lambda_max`, `induction_acc`,
    locate the lambda zero-crossing and the accuracy onset and test alignment."""
    traj = sorted(traj, key=lambda r: r["step"])
    steps = [r["step"] for r in traj]
    lam = [r["lambda_max"] for r in traj if r.get("lambda_max") is not None]
    acc = [r["induction_acc"] for r in traj if r.get("induction_acc") is not None]
    out: dict = {"n_checkpoints": len(traj)}
    if len(lam) < 3 or len(acc) < 3:
        out["interpretation"] = "INSUFFICIENT_CHECKPOINTS"
        return out

    lam_arr = [r["lambda_max"] for r in traj]
    acc_arr = [r["induction_acc"] for r in traj]
    lam_cross = zero_crossing_step(steps, lam_arr, 0.0)

    acc_rise = float(max(acc_arr) - min(acc_arr))
    acc_level = float(min(acc_arr) + 0.5 * acc_rise)
    acc_onset = (zero_crossing_step(steps, acc_arr, acc_level)
                 if acc_rise > 0.15 else None)

    spear = _spearman(lam_arr, acc_arr)
    log_gap = None
    aligned = False
    if lam_cross is not None and acc_onset is not None:
        log_gap = abs(_log_step(lam_cross) - _log_step(acc_onset))
        aligned = log_gap < float(np.log10(ALIGN_FACTOR))

    co_rises = bool(spear is not None and spear > SPEARMAN_MIN)
    emergence_tracked = bool(aligned and co_rises)

    if lam_cross is None:
        interp = ("NO_LAMBDA_CROSSING: lambda_max never crosses 0 across the "
                  "swept checkpoints (always dark, or always bright) — the "
                  "dynamical order-parameter picture does not apply in this range.")
    elif acc_onset is None:
        interp = ("NO_CAPABILITY_ONSET: induction accuracy does not show a clear "
                  "rise (>0.15) across checkpoints — capability did not emerge in "
                  "this range, cannot test alignment.")
    elif emergence_tracked:
        interp = (f"EMERGENCE_TRACKED: lambda_max zero-crossing (~step {lam_cross:.0f}) "
                  f"aligns with induction onset (~step {acc_onset:.0f}); they co-rise "
                  f"(Spearman {spear}). The leading Lyapunov exponent is a "
                  f"developmental order parameter for this capability.")
    elif aligned:
        interp = (f"ALIGNED_NOT_MONOTONE: crossing (~{lam_cross:.0f}) near onset "
                  f"(~{acc_onset:.0f}) but lambda_max does not co-rise monotonically "
                  f"(Spearman {spear}).")
    else:
        interp = (f"MISALIGNED: lambda_max crossing (~step {lam_cross:.0f}) is far "
                  f"from induction onset (~step {acc_onset:.0f}, log-gap "
                  f"{log_gap:.2f}) — the crossing does not time the capability.")

    out.update({
        "lambda_zero_crossing_step": (round(lam_cross, 1) if lam_cross else None),
        "induction_onset_step": (round(acc_onset, 1) if acc_onset else None),
        "onset_level": round(acc_level, 4),
        "accuracy_rise": round(acc_rise, 4),
        "log_step_gap": (round(log_gap, 4) if log_gap is not None else None),
        "align_factor": ALIGN_FACTOR,
        "aligned": aligned,
        "spearman_lambda_acc": spear,
        "co_rises": co_rises,
        "emergence_tracked": emergence_tracked,
        "interpretation": interp,
    })
    return out


# ==========================================================================
# SYNTHETIC gate (numpy only): planted trajectories with known crossing/onset.
# ==========================================================================
def run_synthetic(seed: int = 0) -> dict:
    steps = [0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
    xs = np.array([_log_step(s) for s in steps])

    # (A) ALIGNED case: lambda rises linearly in log-step, crossing 0 at the same
    #     log-step where the capability sigmoid hits its half-rise (planted t* near
    #     step 4000). Detectors must recover the crossing/onset and fire `aligned`.
    x_star = _log_step(4000)
    lam_aligned = 0.4 * (xs - x_star)                       # crosses 0 at x_star
    acc_aligned = 0.05 + 0.9 / (1.0 + np.exp(-3.0 * (xs - x_star)))
    traj_a = [{"step": s, "lambda_max": float(l), "induction_acc": float(a)}
              for s, l, a in zip(steps, lam_aligned, acc_aligned)]
    res_a = crossing_alignment(traj_a)

    # (B) MISALIGNED control: lambda crosses early (~step 256-512) but capability
    #     onset is late (~step 32000). Detector must NOT fire `aligned`.
    lam_early = 0.4 * (xs - _log_step(400))
    acc_late = 0.05 + 0.9 / (1.0 + np.exp(-3.0 * (xs - _log_step(32000))))
    traj_b = [{"step": s, "lambda_max": float(l), "induction_acc": float(a)}
              for s, l, a in zip(steps, lam_early, acc_late)]
    res_b = crossing_alignment(traj_b)

    # (C) NO-CROSSING control: lambda stays negative throughout (always dark);
    #     capability never turns on. Detector must report no crossing/onset.
    lam_neg = -0.5 - 0.1 * np.arange(len(xs))
    acc_flat = 0.05 + 0.01 * np.arange(len(xs)) / len(xs)
    traj_c = [{"step": s, "lambda_max": float(l), "induction_acc": float(a)}
              for s, l, a in zip(steps, lam_neg, acc_flat)]
    res_c = crossing_alignment(traj_c)

    # crossing-step accuracy: planted crossing is exactly x_star -> step 4000.
    cross_a = res_a.get("lambda_zero_crossing_step")
    onset_a = res_a.get("induction_onset_step")
    cross_err = (abs(_log_step(cross_a) - x_star) if cross_a else 9.9)
    onset_err = (abs(_log_step(onset_a) - x_star) if onset_a else 9.9)

    verdict = {
        "detects_crossing": bool(cross_a is not None and cross_err < 0.15),
        "detects_onset": bool(onset_a is not None and onset_err < 0.15),
        "aligned_fires_when_aligned": bool(res_a.get("aligned") is True),
        "aligned_silent_when_misaligned": bool(res_b.get("aligned") is False),
        "handles_no_crossing": bool(res_c.get("lambda_zero_crossing_step") is None),
        "crossing_log_err": round(float(cross_err), 4),
        "onset_log_err": round(float(onset_err), 4),
    }
    verdict["all_pass"] = bool(
        verdict["detects_crossing"] and verdict["detects_onset"]
        and verdict["aligned_fires_when_aligned"]
        and verdict["aligned_silent_when_misaligned"]
        and verdict["handles_no_crossing"])

    out = {
        "mode": "synthetic",
        "probe": "developmental_lyapunov",
        "date": time.strftime("%Y-%m-%d"),
        "spec_anchor": "OPERATOR_LIFT_SPEC.md G5/G7; lyapunov_zero_crossing_emergence",
        "aligned_case": res_a,
        "misaligned_control": res_b,
        "no_crossing_control": res_c,
        "verdict": verdict,
    }
    path = HERE / "developmental_lyapunov_synthetic.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    print("\n=== synthetic developmental-Lyapunov verdict ===")
    for kk, vv in verdict.items():
        print(f"  {kk}: {vv}")
    return out


# ==========================================================================
# REAL mode: per-checkpoint lambda_max (F0 operator spectrum) + induction acc.
# ==========================================================================
def _induction_accuracy(ctx, n_samples, seed, block_len):
    """Induction in-context accuracy: fraction of repeated-block prompts whose
    argmax next-token at the copy position equals the (known) copy target, plus
    the mean target log-prob (a smooth capability signal)."""
    import torch
    from routing_selection_probe import build_induction

    rng = np.random.default_rng(seed + 1234)
    correct, logps, n = 0, [], 0
    for _ in range(n_samples):
        ids, pos, target = build_induction(ctx.tok, rng, block_len)
        x = torch.tensor([ids], device=ctx.device)
        with torch.no_grad():
            logits = ctx.model(input_ids=x).logits[0, pos, :]
            lp = torch.log_softmax(logits.float(), dim=-1)
        pred = int(torch.argmax(logits))
        correct += int(pred == target)
        logps.append(float(lp[target].cpu()))
        n += 1
    return {
        "induction_acc": (correct / n if n else None),
        "induction_logprob_mean": (round(float(np.mean(logps)), 4) if logps else None),
        "n_acc": n,
    }


def _lambda_max(ctx, n_samples, seed, k, iters, block_len):
    """Leading Lyapunov exponent lambda_max = (1/D) log sigma_top(T) on the
    induction task, the full top-k lambda spectrum, and the dark/bright partition
    (count of lambda_i >= 0). T is the per-prompt transport operator from the
    capacity layer to the final-norm readout; D is the traversed depth."""
    from jacobian_product_spectrum_probe import _operator_spectrum
    from routing_selection_probe import build_induction
    from singular_spectrum_probe import _capacity_layer_index

    layers = ctx.layers
    n_layers = len(layers)
    l_c = _capacity_layer_index(n_layers)
    depth = max(n_layers - l_c, 1)        # layers traversed by T
    d_model = int(ctx.model.config.hidden_size)
    k = min(k, d_model)

    rng = np.random.default_rng(seed)
    sig_tops, spectra = [], []
    for _ in range(n_samples):
        ids, pos, _t = build_induction(ctx.tok, rng, block_len)
        try:
            sig_op, _V = _operator_spectrum(ctx, layers, l_c, ids, pos, k, iters)
        except Exception as e:  # noqa: BLE001
            print(f"    operator spectrum failed ({type(e).__name__}: {e})")
            continue
        sig_tops.append(float(sig_op[0]))
        spectra.append(np.asarray(sig_op[:k], float))
    if len(sig_tops) < 3:
        return {"lambda_max": None, "n_op": len(sig_tops)}

    sigma_top = float(np.median(sig_tops))                 # robust top sigma
    lam_max = float(np.log(max(sigma_top, 1e-12)) / depth)
    sig_mean = np.mean(np.stack(spectra, axis=0), axis=0)  # mean top-k spectrum
    lam_spectrum = [float(np.log(max(s, 1e-12)) / depth) for s in sig_mean]
    n_positive = int(sum(1 for x in lam_spectrum if x >= 0.0))
    return {
        "lambda_max": round(lam_max, 6),
        "sigma_top": round(sigma_top, 5),
        "depth": depth,
        "lambda_spectrum": [round(x, 6) for x in lam_spectrum],
        "n_positive_lambda": n_positive,
        "n_op": len(sig_tops),
    }


def measure_checkpoint(ctx, *, n_op_samples, n_acc_samples, seed, k, iters,
                       block_len) -> dict:
    rec = {}
    rec.update(_lambda_max(ctx, n_op_samples, seed, k, iters, block_len))
    rec.update(_induction_accuracy(ctx, n_acc_samples, seed, block_len))
    return rec


def run_developmental(base=DEFAULT_BASE, revisions=None, *, n_op_samples=DEFAULT_OP_SAMPLES,
                      n_acc_samples=DEFAULT_ACC_SAMPLES, seed=42, k=DEFAULT_K,
                      iters=DEFAULT_ITERS, block_len=DEFAULT_BLOCK) -> dict:
    revisions = revisions or DEFAULT_REVISIONS
    from icl_convergence_probe import safe_name

    traj = []
    for rev in revisions:
        spec = f"{base}@{rev}"
        print(f"\n── checkpoint {spec} ──")
        try:
            ctx = load_context(spec)
        except Exception as e:  # noqa: BLE001
            print(f"  load failed ({type(e).__name__}: {e}) — skipping")
            continue
        rec = measure_checkpoint(ctx, n_op_samples=n_op_samples,
                                 n_acc_samples=n_acc_samples, seed=seed, k=k,
                                 iters=iters, block_len=block_len)
        rec["revision"] = rev
        rec["step"] = _step_of(rev)
        traj.append(rec)
        print(f"  step={rec['step']:.0f}  lambda_max={rec.get('lambda_max')}  "
              f"sigma_top={rec.get('sigma_top')}  "
              f"n_pos_lambda={rec.get('n_positive_lambda')}  "
              f"induction_acc={rec.get('induction_acc')}  "
              f"logp={rec.get('induction_logprob_mean')}")
        # free model memory between checkpoints
        try:
            import torch
            del ctx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    verdict = crossing_alignment(traj)
    n_layers = None
    out = {
        "mode": "real_model",
        "probe": "developmental_lyapunov",
        "base": base,
        "revisions": revisions,
        "date": time.strftime("%Y-%m-%d"),
        "spec_anchor": "OPERATOR_LIFT_SPEC.md G5/G7; lyapunov_zero_crossing_emergence",
        "params": {"n_op_samples": n_op_samples, "n_acc_samples": n_acc_samples,
                   "k": k, "iters": iters, "block_len": block_len, "seed": seed},
        "trajectory": traj,
        "verdict": verdict,
    }
    path = HERE / f"developmental_lyapunov_{safe_name(base)}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {path.name}")
    print(f"  verdict: {verdict.get('interpretation')}")
    return out


# ==========================================================================
# OOP front + registration
# ==========================================================================
class DevelopmentalLyapunovProbe(Probe):
    name = "developmental_lyapunov"
    classifier_features: list[str] = []   # trajectory measurement, no classifier
    label_key = None

    def synthetic(self, seed: int = 0) -> dict:
        return run_synthetic(seed=seed)

    def run_model(self, model_name: str, *, revisions=None,
                  n_op_samples: int = DEFAULT_OP_SAMPLES,
                  n_acc_samples: int = DEFAULT_ACC_SAMPLES, seed: int = 42,
                  k: int = DEFAULT_K, iters: int = DEFAULT_ITERS,
                  block_len: int = DEFAULT_BLOCK, **_) -> dict:
        # `model_name` is the BASE (e.g. "EleutherAI/pythia-410m"); the sweep is
        # over `revisions` (training checkpoints).
        return run_developmental(model_name, revisions, n_op_samples=n_op_samples,
                                 n_acc_samples=n_acc_samples, seed=seed, k=k,
                                 iters=iters, block_len=block_len)


register_probe(DevelopmentalLyapunovProbe())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--revisions", nargs="+", default=None)
    ap.add_argument("--n-op-samples", type=int, default=DEFAULT_OP_SAMPLES)
    ap.add_argument("--n-acc-samples", type=int, default=DEFAULT_ACC_SAMPLES)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--block-len", type=int, default=DEFAULT_BLOCK)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic()
        return
    run_developmental(args.base, args.revisions, n_op_samples=args.n_op_samples,
                      n_acc_samples=args.n_acc_samples, seed=args.seed, k=args.k,
                      iters=args.iters, block_len=args.block_len)


if __name__ == "__main__":
    main()
