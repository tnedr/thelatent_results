#!/usr/bin/env python3
"""Representation-theoretic capacity probe — is the readout effective rank the
IRREP-DIMENSION BUDGET of the task's symmetry group?

THE ALTERNATIVE EXPLANATION (Tao-flavored, exact, falsifiable)
  The capacity program measures a low readout effective rank at grok and treats
  it as an emergent fact. This probe tests a precise REASON for that number,
  borrowed from representation theory:

    A grokking model on (a op b) mod p learns an approximate set of IRREDUCIBLE
    REPRESENTATIONS of the task's symmetry group Z_p. The irreps of the cyclic
    group Z_p are its p Fourier modes; each non-trivial mode is a 2-DIMENSIONAL
    real representation (a cos/sin rotation pair). If the model uses k Fourier
    frequencies, it spends 2k representation dimensions, and the CAPACITY LAW is:

        readout effective rank at grok  ==  2 * (number of ACTIVE frequencies)

  i.e. the measured rank is not arbitrary — it equals the irrep-dimension budget
  the network actually allocates.

NOVELTY BOUNDARY (read this — be honest about what is new)
  KNOWN PRIOR WORK (Nanda et al. 2023, "Progress measures for grokking via
  mechanistic interpretability"): mod-addition grokking learns a Fourier /
  "clock" circuit built from a few key frequencies. We DO NOT claim the Fourier
  structure as novel. The probe MEASURES that known structure as an input.

  The NEW, falsifiable claim is the CAPACITY LAW itself —
    (i)  readout effective rank == 2 * active-frequency count, and
    (ii) its OPERATION-GENERALITY: does the same law hold for * and -, with the
         ACTIVE-FREQUENCY SET differing by operation while the 2k==rank relation
         persists?
  A failure (rank not explained by 2*active-irrep-count) is a valid NULL finding.

WHAT IS MEASURED
  Train a tiny decoder-only transformer to grok (a op b) mod p (REUSING the
  grokking_dynamics_modadd machinery — same model class, dataset, optimizer, and
  the readout-eff-rank measurement measure_a / layer_readout_subspace). At the
  grok checkpoint:
    k      = number of ACTIVE non-trivial Z_p frequencies in the number-token
             embedding power spectrum (above the uniform-share mean level), with
             a frequency-domain participation ratio to decide whether the
             embedding is irrep-structured at all (the random control fails this).
    rank   = readout effective rank from measure_a (the feature-Gram readout
             energy participation ratio — the quantity the capacity program
             already tracks).
  Report (2k)/rank per operation. The law predicts ~1.

VERDICT (per the spec)
  IRREP_LAW if (2k)/rank in [0.7, 1.4] across >= 2 operations; NULL otherwise.

Corp-native: tiny from-scratch CPU models, NO download, NO GPU, NO pretrained LM.
Run:
  .venv/bin/python irrep_capacity_probe.py --synthetic
  .venv/bin/python irrep_capacity_probe.py --ops + *
  .venv/bin/python irrep_capacity_probe.py --aggregate
Output: irrep_capacity_synthetic.json, irrep_capacity_<op>.json,
        irrep_capacity_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# the grokking trainer lives in the sibling capacity-scaling topic
_CAP_EXP = HERE.parent.parent / "ml_icl_capacity_scaling" / "experiments"
if str(_CAP_EXP) not in sys.path:
    sys.path.insert(0, str(_CAP_EXP))

from singular_spectrum_probe import effective_rank  # noqa: E402  (numpy-only)

DEFAULT_P = 97
DEFAULT_OPS = ["+", "*", "-"]
RATIO_LO, RATIO_HI = 0.7, 1.4      # capacity-law acceptance band (per spec)
SPARSITY_MIN = 0.5                 # irrep-structured iff freq spectrum is sparse
MAX_THREADS = int(os.environ.get("IRREP_THREADS", "2"))  # CPU-light default; set IRREP_THREADS=1 to run many cells in parallel
OP_TAG = {"+": "add", "*": "mul", "/": "div", "-": "sub"}
# Which group's irreps the operation's symmetry lives in:
#   +, - : the ADDITIVE group Z_p   (irreps = ordinary Fourier modes)
#   *, / : the MULTIPLICATIVE group Z_p* (irreps = multiplicative characters =
#          additive Fourier modes in the DISCRETE-LOG coordinate over Z_{p-1})
CORRECT_BASIS = {"+": "additive", "-": "additive", "*": "multiplicative",
                 "/": "multiplicative"}


# ---------------------------------------------------------------------------
# Z_p Fourier (irrep) analysis of a [p, d] matrix of per-number row vectors
# ---------------------------------------------------------------------------
def fourier_power_spectrum(M, p):
    """Power carried by each non-trivial Z_p frequency f = 1..floor(p/2) in the
    rows of M ([p, d]). The non-trivial irreps of the cyclic group Z_p are the
    2-D rotation reps (cos_f, sin_f); f and p-f are conjugate, so 1..floor(p/2)
    enumerates each real irrep once. The DC term (f = 0, the trivial irrep) is
    removed by mean-centering over the p rows."""
    M = np.asarray(M, float)
    if M.shape[0] != p:
        raise ValueError(f"expected {p} rows, got {M.shape[0]}")
    Mc = M - M.mean(axis=0, keepdims=True)
    a = np.arange(p)
    freqs = np.arange(1, p // 2 + 1)
    power = np.empty(len(freqs), float)
    for i, f in enumerate(freqs):
        c = np.cos(2.0 * np.pi * f * a / p)
        s = np.sin(2.0 * np.pi * f * a / p)
        c = c / (np.linalg.norm(c) + 1e-12)
        s = s / (np.linalg.norm(s) + 1e-12)
        power[i] = float(np.sum((c @ Mc) ** 2) + np.sum((s @ Mc) ** 2))
    return freqs, power


def primitive_root(p):
    """Smallest primitive root g of the prime p (a generator of Z_p*)."""
    if p == 2:
        return 1
    factors = _prime_factors(p - 1)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    raise ValueError(f"no primitive root found for p={p}")


def _prime_factors(n):
    factors, d = set(), 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def discrete_log_table(p, g):
    """dlog[x] = i such that g^i == x (mod p), for x in 1..p-1 (Z_p* indices)."""
    dlog = {}
    cur = 1
    for i in range(p - 1):
        dlog[cur] = i
        cur = (cur * g) % p
    return dlog


def multiplicative_power_spectrum(E, p):
    """Multiplicative-character power spectrum of the number-token rows E ([p, d]).
    The irreps of the MULTIPLICATIVE group Z_p* are the characters
    chi_j(x) = exp(2*pi*i*j*dlog(x)/(p-1)) — i.e. ordinary Fourier modes in the
    DISCRETE-LOG coordinate over Z_{p-1}. So we reindex the nonzero rows x -> dlog(x)
    and take the additive Fourier power spectrum over Z_{p-1}. The x = 0 row carries
    no multiplicative character and is EXCLUDED."""
    E = np.asarray(E, float)
    g = primitive_root(p)
    dlog = discrete_log_table(p, g)
    m = p - 1
    M = np.empty((m, E.shape[1]), float)
    for x in range(1, p):
        M[dlog[x]] = E[x]
    freqs, power = fourier_power_spectrum(M, m)
    return freqs, power, g, M


def analyze_spectrum(freqs, power):
    """From a Z_p frequency power spectrum:
      k_active       = frequencies above the uniform-share mean (a hard count;
                       over-counts a spread tail, kept for transparency)
      k_eff          = freq_pr = frequency-domain participation ratio
                       (sum p)^2 / sum p^2; the ENERGY-WEIGHTED effective number
                       of active frequencies. This is the apples-to-apples dual of
                       the readout effective rank (also a participation ratio), so
                       it is the PRIMARY active-irrep count for the capacity law.
                       ~ k for a clean k-mode signal, ~ n_freq for a flat spectrum.
      sparsity       = 1 - k_eff / n_freq in [0,1]; high => irrep-structured,
                       ~0 => spread/non-Fourier (the random control)
      active_energy_frac = energy fraction carried by the above-mean frequencies."""
    power = np.asarray(power, float)
    n = len(power)
    total = float(power.sum())
    if total <= 0 or n == 0:
        return {"k_active": 0, "active_freqs": [], "k_eff": float("nan"),
                "sparsity": float("nan"), "active_energy_frac": 0.0}
    mean = total / n
    mask = power > mean
    k_eff = float(total ** 2 / (float(np.sum(power ** 2)) + 1e-30))
    return {
        "k_active": int(mask.sum()),
        "active_freqs": [int(f) for f in np.asarray(freqs)[mask]],
        "k_eff": round(k_eff, 3),
        "sparsity": round(float(1.0 - k_eff / n), 4),
        "active_energy_frac": round(float(power[mask].sum() / total), 4),
    }


def capacity_verdict(k, eff_rank, sparsity, sparsity_min=SPARSITY_MIN):
    """Compare the irrep-dimension budget 2k to the (readout) effective rank.
    `k` is the effective active-irrep count (k_eff, a participation ratio — the
    consistent dual of the readout participation ratio). Returns (verdict, ratio).
    The sparsity gate first decides whether the matrix is irrep-structured at all
    (random readouts are not)."""
    if eff_rank is None or eff_rank <= 0 or k is None or k <= 0:
        return "DARK", None
    ratio = (2.0 * k) / float(eff_rank)
    if not (isinstance(sparsity, float) and sparsity >= sparsity_min):
        return "NOT_IRREP", round(ratio, 3)
    if RATIO_LO <= ratio <= RATIO_HI:
        return "IRREP", round(ratio, 3)
    return "STRUCTURED_OFF_LAW", round(ratio, 3)


# ---------------------------------------------------------------------------
# Synthetic validation (numpy only) — planted vs random readouts
# ---------------------------------------------------------------------------
def planted_embedding(p, active, d, rng):
    """A [p, d] matrix whose rows are built from EXACTLY the chosen k Fourier
    modes of Z_p, with orthonormal coefficient directions and equal energy.
    By construction: power concentrates on exactly `active`, and the matrix has
    2k equal singular values, so its effective rank is exactly 2k."""
    a = np.arange(p)
    cols = []
    for f in active:
        c = np.cos(2.0 * np.pi * f * a / p)
        s = np.sin(2.0 * np.pi * f * a / p)
        cols.append(c / np.linalg.norm(c))
        cols.append(s / np.linalg.norm(s))
    C = np.stack(cols, axis=1)                      # [p, 2k], orthonormal columns
    k2 = C.shape[1]
    Q, _ = np.linalg.qr(rng.standard_normal((d, k2)))  # [d, 2k] orthonormal cols
    W = Q.T                                          # [2k, d] orthonormal rows
    return C @ W                                     # [p, d], 2k equal sing. vals


def planted_mult_embedding(p, active_chars, d, rng):
    """A [p, d] embedding whose NONZERO rows carry EXACTLY the chosen k
    multiplicative characters of Z_p* (placed via the discrete-log map), with
    the x = 0 row set to zero. Built so the multiplicative-spectrum path should
    recover exactly `active_chars` and find 2k spectral eff-rank over Z_{p-1}."""
    g = primitive_root(p)
    dlog = discrete_log_table(p, g)
    m = p - 1
    i = np.arange(m)
    cols = []
    for f in active_chars:
        c = np.cos(2.0 * np.pi * f * i / m)
        s = np.sin(2.0 * np.pi * f * i / m)
        cols.append(c / np.linalg.norm(c))
        cols.append(s / np.linalg.norm(s))
    C = np.stack(cols, axis=1)                      # [m, 2k] orthonormal columns
    Q, _ = np.linalg.qr(rng.standard_normal((d, C.shape[1])))
    M = C @ Q.T                                     # [m, d], 2k equal sing. vals
    E = np.zeros((p, d))
    for x in range(1, p):
        E[x] = M[dlog[x]]                           # row x carries character on dlog(x)
    return E, g


def _spectral_eff_rank(M):
    Mc = np.asarray(M, float)
    Mc = Mc - Mc.mean(axis=0, keepdims=True)
    return float(effective_rank(np.linalg.svd(Mc, compute_uv=False)))


def run_synthetic(seed=0):
    """Plant readouts from KNOWN Fourier modes and verify the probe (a) recovers
    exactly the planted active frequencies, (b) finds 2k == spectral eff-rank,
    and (c) flags a non-Fourier random readout as NOT irrep-structured. No
    threshold tuning: the separations below are order-of-magnitude clean."""
    rng = np.random.default_rng(seed)
    p, d = DEFAULT_P, 64

    planted5 = sorted(rng.choice(np.arange(1, p // 2 + 1), size=5, replace=False))
    planted3 = sorted(rng.choice(np.arange(1, p // 2 + 1), size=3, replace=False))

    E5 = planted_embedding(p, planted5, d, rng)
    E3 = planted_embedding(p, planted3, d, rng)
    Erand = rng.standard_normal((p, d))

    def measure(E):
        freqs, power = fourier_power_spectrum(E, p)
        a = analyze_spectrum(freqs, power)
        a["spectral_eff_rank"] = round(_spectral_eff_rank(E), 3)
        a["verdict"], a["ratio_2k_over_rank"] = capacity_verdict(
            a["k_eff"], a["spectral_eff_rank"], a["sparsity"])
        return a

    r5, r3, rr = measure(E5), measure(E3), measure(Erand)

    # --- multiplicative-character basis (Z_p* via discrete log over Z_{p-1}) ---
    m = p - 1
    mult_chars = sorted(rng.choice(np.arange(1, m // 2), size=4, replace=False))
    Emult, g_used = planted_mult_embedding(p, mult_chars, d, rng)

    def measure_mult(E):
        freqs, power, g, M = multiplicative_power_spectrum(E, p)
        a = analyze_spectrum(freqs, power)
        a["spectral_eff_rank"] = round(_spectral_eff_rank(M), 3)
        a["verdict"], a["ratio_2k_over_rank"] = capacity_verdict(
            a["k_eff"], a["spectral_eff_rank"], a["sparsity"])
        a["primitive_root"] = int(g)
        return a

    rmult = measure_mult(Emult)
    # cross-basis control: a multiplicative signal is NOT additive-Fourier-sparse
    rmult_in_add = measure(Emult)

    checks = [
        ("planted-5: recovers exactly the 5 planted frequencies",
         r5["active_freqs"] == list(planted5)),
        ("planted-5: effective freq count k_eff == 5",
         abs(r5["k_eff"] - 5.0) < 0.5),
        ("planted-5: spectral eff-rank == 2k = 10",
         abs(r5["spectral_eff_rank"] - 10.0) < 0.5),
        ("planted-5: (2*k_eff)/eff_rank in [0.7,1.4]",
         RATIO_LO <= r5["ratio_2k_over_rank"] <= RATIO_HI),
        ("planted-3: recovers exactly the 3 planted frequencies",
         r3["active_freqs"] == list(planted3)),
        ("planted-3: (2*k_eff)/eff_rank in [0.7,1.4]",
         RATIO_LO <= r3["ratio_2k_over_rank"] <= RATIO_HI),
        ("planted verdicts == IRREP",
         r5["verdict"] == "IRREP" and r3["verdict"] == "IRREP"),
        ("random readout flagged NOT_IRREP",
         rr["verdict"] == "NOT_IRREP"),
        ("random sparsity << planted sparsity",
         rr["sparsity"] < 0.5 < r5["sparsity"]),
        ("random k_active >> any planted k (flat spectrum)",
         rr["k_active"] > 2 * 5),
        ("mult-char: multiplicative spectrum recovers exactly the 4 planted chars",
         rmult["active_freqs"] == list(mult_chars)),
        ("mult-char: effective char count k_eff == 4",
         abs(rmult["k_eff"] - 4.0) < 0.5),
        ("mult-char: spectral eff-rank == 2k = 8",
         abs(rmult["spectral_eff_rank"] - 8.0) < 0.5),
        ("mult-char: (2*k_eff)/eff_rank in [0.7,1.4] and verdict IRREP",
         rmult["verdict"] == "IRREP"
         and RATIO_LO <= rmult["ratio_2k_over_rank"] <= RATIO_HI),
        ("BASIS MATTERS: the mult signal is NOT additive-sparse (NOT_IRREP in add)",
         rmult_in_add["sparsity"] < 0.5
         and rmult["sparsity"] > rmult_in_add["sparsity"]),
    ]
    n_pass = sum(1 for _, ok in checks if ok)
    print(f"Synthetic: {n_pass}/{len(checks)} checks pass")
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
    for tag, r in (("planted5", r5), ("planted3", r3), ("random", rr),
                   ("mult4", rmult), ("mult-in-add", rmult_in_add)):
        print(f"  {tag:>11}: k_eff={r['k_eff']} k_active={r['k_active']} "
              f"freqs={r['active_freqs']} eff_rank={r['spectral_eff_rank']} "
              f"sparsity={r['sparsity']} ratio={r['ratio_2k_over_rank']} "
              f"-> {r['verdict']}")

    out = {
        "mode": "synthetic",
        "date": time.strftime("%Y-%m-%d"),
        "planted5_freqs": [int(f) for f in planted5],
        "planted3_freqs": [int(f) for f in planted3],
        "planted_mult_chars": [int(f) for f in mult_chars],
        "primitive_root": int(g_used),
        "planted5": r5, "planted3": r3, "random": rr,
        "mult_chars": rmult, "mult_in_additive_basis": rmult_in_add,
        "synthetic_checks": [{"test": d_, "pass": bool(ok)} for d_, ok in checks],
        "pass_rate": f"{n_pass}/{len(checks)}",
        "all_pass": bool(n_pass == len(checks)),
    }
    path = HERE / "irrep_capacity_synthetic.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path.name}")
    return out


# ---------------------------------------------------------------------------
# Real arm — train a tiny model to grok, then measure k and readout eff-rank
# ---------------------------------------------------------------------------
def _make_dataset(p, op, train_frac, seed):
    """Thin wrapper over grokking_dynamics_modadd.make_dataset that adds the
    SUBTRACTION op (additive, like +). The grokking dataset supports +, *, /;
    we mirror its exact token scheme ([a, OP, b, EQ] -> answer) for - locally so
    we do not modify the sibling file."""
    from grokking_dynamics_modadd import make_dataset
    if op != "-":
        return make_dataset(p, op, train_frac, seed)
    import torch
    OP, EQ = p, p + 1
    vocab = p + 2
    rows = [(a, OP, b, EQ, (a - b) % p) for a in range(p) for b in range(p)]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    n_tr = int(len(rows) * train_frac)
    arr = np.array(rows, dtype=np.int64)
    X = torch.tensor(arr[:, :4])
    Y = torch.tensor(arr[:, 4])
    return (X[:n_tr], Y[:n_tr]), (X[n_tr:], Y[n_tr:]), vocab, 4


def _train_grok_cleanup(p, op, seed, max_steps, grok_acc, eval_every=200,
                        train_frac=0.4, wd=1.0, lr=1e-3):
    """REUSE the grokking_dynamics_modadd model class, dataset, optimizer config,
    and the readout-eff-rank measurement. Trains on CPU for `max_steps`, recording
    the GROK step (first checkpoint clearing `grok_acc`). It deliberately trains
    PAST grok into the cleanup phase, because the clean Fourier/clock circuit (the
    irrep decomposition we measure) crystallizes during cleanup, not at the grok
    step itself. Returns the trained model + a summary measured at the final
    (cleaned-up) checkpoint."""
    import torch
    from grokking_dynamics_modadd import build_model, measure_a, accuracy

    torch.manual_seed(seed)
    torch.set_num_threads(MAX_THREADS)
    device = os.environ.get("IRREP_DEVICE", "cpu")  # set mps/cuda on a GPU box
    (Xtr, Ytr), (Xte, Yte), vocab, seq_len = _make_dataset(p, op, train_frac, seed)
    model = build_model(vocab, seq_len, 128, 4, 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                            betas=(0.9, 0.98))
    lossf = torch.nn.CrossEntropyLoss()

    Xtr_d, Ytr_d = Xtr.to(device), Ytr.to(device)
    Xa, Ya = Xtr[:1500], Ytr[:1500]            # capped eval batch for measure_a
    Xte_e, Yte_e = Xte[:2000], Yte[:2000]
    Xtr_e, Ytr_e = Xtr[:2000], Ytr[:2000]

    t0 = time.time()
    grok_step, grokked = None, False
    for step in range(1, max_steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(Xtr_d)
        loss = lossf(logits[:, -1, :], Ytr_d)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == max_steps:
            model.eval()
            te = accuracy(model, Xte_e, Yte_e, device)
            if te >= grok_acc and not grokked:
                grok_step, grokked = step, True
    model.eval()
    tr = accuracy(model, Xtr_e, Ytr_e, device)
    te = accuracy(model, Xte_e, Yte_e, device)
    _a_t, eff_rank, _gap, _svd3 = measure_a(model, Xa, Ya, device, k=4)
    info = {
        "op": op, "grokked": bool(grokked), "grok_step": grok_step,
        "steps_trained": int(max_steps), "max_steps": int(max_steps),
        "train_acc": round(tr, 4), "test_acc": round(te, 4),
        "readout_eff_rank": round(float(eff_rank), 4) if eff_rank else None,
        "wall_s": round(time.time() - t0, 1),
    }
    return model, info


def run_model(op, p=DEFAULT_P, seed=0, max_steps=6000, grok_acc=0.85):
    print(f"\n{'=' * 60}\n  op '{op}' mod {p} (seed {seed}) — train to grok+cleanup\n{'=' * 60}")
    model, info = _train_grok_cleanup(p, op, seed, max_steps, grok_acc)
    print(f"  grokked={info['grokked']} grok_step={info['grok_step']} "
          f"(trained to {info['steps_trained']}) train={info['train_acc']} "
          f"test={info['test_acc']} readout_eff_rank={info['readout_eff_rank']} "
          f"({info['wall_s']}s)")

    eff_rank = info["readout_eff_rank"]
    E = model.tok.weight.detach().cpu().numpy()[:p]   # number-token embedding

    # BOTH group bases are always computed for transparency; the verdict uses the
    # CORRECT basis for the operation (additive for +,-; multiplicative for *,/).
    af, ap = fourier_power_spectrum(E, p)
    add = analyze_spectrum(af, ap)
    add["spectral_eff_rank"] = round(_spectral_eff_rank(E), 3)

    mf, mp_, g, M = multiplicative_power_spectrum(E, p)
    mul = analyze_spectrum(mf, mp_)
    mul["spectral_eff_rank"] = round(_spectral_eff_rank(M), 3)
    mul["primitive_root"] = int(g)

    basis = CORRECT_BASIS.get(op, "additive")
    chosen = mul if basis == "multiplicative" else add
    k_eff = chosen["k_eff"]
    verdict, ratio = capacity_verdict(k_eff, eff_rank, chosen["sparsity"])

    def _spectrum_block(s):
        return {
            "k_eff": s["k_eff"], "k_active": s["k_active"],
            "active_freqs": s["active_freqs"], "sparsity": s["sparsity"],
            "active_energy_frac": s["active_energy_frac"],
            "spectral_eff_rank": s["spectral_eff_rank"],
            "ratio_2k_over_readout_rank": (round(2 * s["k_eff"] / eff_rank, 3)
                                           if s["k_eff"] and eff_rank else None),
        }

    res = {
        "mode": "real_model",
        "op": op, "p": p, "seed": seed,
        "date": time.strftime("%Y-%m-%d"),
        "training": info,
        "correct_basis": basis,
        "readout_eff_rank": eff_rank,
        "k_eff": k_eff,
        "active_freqs": chosen["active_freqs"],
        "embedding_sparsity": chosen["sparsity"],
        "irrep_dim_budget_2k_eff": round(2 * k_eff, 3) if k_eff else None,
        "ratio_2k_over_readout_rank": ratio,
        "verdict": verdict,
        "additive_basis": _spectrum_block(add),
        "multiplicative_basis": _spectrum_block(mul),
        "method": "tiny from-scratch grokking transformer (reuses "
                  "grokking_dynamics_modadd, trained through cleanup). The verdict "
                  "uses the operation's CORRECT group basis: additive Fourier (Z_p) "
                  "for +,-; multiplicative characters (Z_p* via discrete log over "
                  "Z_{p-1}) for *,/. k_eff = participation-ratio effective count of "
                  "active irreps (dual of the readout participation ratio); readout "
                  "eff-rank via measure_a/layer_readout_subspace.",
        "novelty_note": "Fourier/clock circuit itself is KNOWN (Nanda et al. 2023). "
                        "NEW claim = readout eff-rank == 2*active-irrep-count "
                        "(capacity law) + operation-generality in the correct basis.",
    }
    path = HERE / f"irrep_capacity_{OP_TAG.get(op, op)}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"  [{basis} basis] k_eff={k_eff} sparsity={chosen['sparsity']} "
          f"active_freqs={chosen['active_freqs']}")
    print(f"  (additive k_eff={add['k_eff']} sp={add['sparsity']} | "
          f"multiplicative k_eff={mul['k_eff']} sp={mul['sparsity']})")
    print(f"  2*k_eff={round(2 * k_eff, 2) if k_eff else None}  "
          f"readout_eff_rank={eff_rank}  ratio=(2k_eff)/rank={ratio}  -> {verdict}")
    print(f"  saved -> {path.name}")
    return res


def run(ops, p=DEFAULT_P, seed=0, max_steps=8000):
    results = {}
    for op in ops:
        try:
            results[op] = run_model(op, p=p, seed=seed, max_steps=max_steps)
        except Exception as e:  # noqa: BLE001
            print(f"  op '{op}' FAILED ({type(e).__name__}: {e})")
    return aggregate()


# ---------------------------------------------------------------------------
# Multi-seed BUDGET-INEQUALITY test (the open IRREPCAP next-step)
# ---------------------------------------------------------------------------
# The single-seed BET B found the strict equality 2k_eff == readout-rank holds
# only for + (ratio 1.32) while *,- are STRUCTURED_OFF_LAW (1.67, 1.87) — the
# data look like a LOOSE UPPER BUDGET 2k_eff >= readout-rank (+ near-tight), not
# an identity. This sweep tests that reframing with error bars:
#   (1) BUDGET holds: is ratio = 2k_eff/rank >= 1 (i.e. 2k_eff >= readout-rank)
#       in (almost) every (op, seed) cell?
#   (2) FILL factor: readout-rank / (2k_eff) in (0,1] — the op-dependent fraction
#       of the irrep budget actually used. Is + reliably the tightest (highest
#       fill, closest to saturating its budget) across seeds?
def run_multiseed(ops=("+", "*", "-"), seeds=(0, 1, 2), p=DEFAULT_P,
                  max_steps=6000):
    per_op = {op: [] for op in ops}
    for seed in seeds:
        for op in ops:
            try:
                res = run_model(op, p=p, seed=seed, max_steps=max_steps)
            except Exception as e:  # noqa: BLE001
                print(f"  op '{op}' seed {seed} FAILED ({type(e).__name__}: {e})")
                continue
            # per-(op,seed) provenance (run_model's default file is clobbered each seed)
            (HERE / f"irrep_capacity_{OP_TAG.get(op, op)}_seed{seed}.json"
             ).write_text(json.dumps(res, indent=2))
            per_op[op].append(res)

    def _stats(xs):
        a = np.asarray([x for x in xs if x is not None], float)
        if a.size == 0:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        return {"n": int(a.size), "mean": round(float(a.mean()), 3),
                "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 3),
                "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}

    summary = {}
    for op, reslist in per_op.items():
        ratios = [r.get("ratio_2k_over_readout_rank") for r in reslist]   # 2k/rank
        fills = [(1.0 / r) if r else None for r in ratios]               # rank/2k
        keff = [r.get("k_eff") for r in reslist]
        ranks = [r.get("readout_eff_rank") for r in reslist]
        spars = [r.get("embedding_sparsity") for r in reslist]
        groksteps = [r.get("training", {}).get("grok_step") for r in reslist]
        # budget holds if 2k_eff >= readout-rank, i.e. ratio >= 1 (small tol)
        budget = [(rt is not None and rt >= 0.95) for rt in ratios]
        verdicts = [r.get("verdict") for r in reslist]
        summary[op] = {
            "correct_basis": CORRECT_BASIS.get(op, "additive"),
            "ratio_2k_over_rank": _stats(ratios),
            "fill_factor_rank_over_2k": _stats(fills),
            "k_eff": _stats(keff), "readout_eff_rank": _stats(ranks),
            "embedding_sparsity": _stats(spars),
            "budget_holds_frac": round(float(np.mean(budget)), 3) if budget else None,
            "grok_steps": groksteps, "verdicts": verdicts,
            "per_seed_ratio": [round(x, 3) if x else None for x in ratios],
        }

    # program-level reframed reading
    all_cells = [(op, rt) for op in ops
                 for rt in [r.get("ratio_2k_over_readout_rank") for r in per_op[op]]
                 if rt is not None]
    budget_cells = [rt for _, rt in all_cells if rt >= 0.95]
    fills_by_op = {op: summary[op]["fill_factor_rank_over_2k"]["mean"] for op in ops
                   if summary[op]["fill_factor_rank_over_2k"]["mean"] is not None}
    tightest_op = (max(fills_by_op, key=fills_by_op.get) if fills_by_op else None)
    out = {
        "mode": "multiseed_budget", "date": time.strftime("%Y-%m-%d"),
        "p": p, "seeds": list(seeds), "max_steps": max_steps,
        "claim_reframed": "2*k_eff is an UPPER BUDGET on the readout eff-rank "
                          "(2k_eff >= rank), with an op-dependent fill factor < 1; "
                          "the strict equality is a special (near-saturating) case.",
        "per_op": summary,
        "budget_inequality_holds_frac": (round(len(budget_cells) / len(all_cells), 3)
                                         if all_cells else None),
        "tightest_op_by_fill": tightest_op,
        "fill_by_op": fills_by_op,
    }
    path = HERE / "irrep_capacity_budget_multiseed.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n{'=' * 64}\nMULTI-SEED BUDGET SUMMARY (seeds {list(seeds)})\n{'=' * 64}")
    print(f"{'op':>3} {'basis':>14} {'ratio 2k/rank':>22} {'fill rank/2k':>20} "
          f"{'budget%':>8}")
    for op in ops:
        s = summary[op]
        r, f = s["ratio_2k_over_rank"], s["fill_factor_rank_over_2k"]
        print(f"{op:>3} {s['correct_basis']:>14} "
              f"{str(r['mean'])+'+-'+str(r['std']):>14} ({r['min']}-{r['max']}) "
              f"{str(f['mean'])+'+-'+str(f['std']):>12} ({f['min']}-{f['max']}) "
              f"{s['budget_holds_frac']:>8}")
    print(f"\nbudget inequality (2k_eff >= rank) holds in "
          f"{out['budget_inequality_holds_frac']} of cells")
    print(f"tightest op (highest fill factor): {tightest_op}  fills={fills_by_op}")
    print(f"saved -> {path.name}")
    return out


# ---------------------------------------------------------------------------
# Readout attribution (the open IRREPCAP next-step 6) — WHY does the readout
# discard ~30% of the embedding's allocated irrep dimensions?
# ---------------------------------------------------------------------------
# The budget-decomposition found the embedding ALLOCATES the full 2*k_eff irrep
# budget (embedding spectral rank ~= 2*k_eff) but the readout uses only ~0.7 of
# it. This probe asks per-frequency: of the active embedding frequencies, which
# ones does the READOUT actually USE? For each active frequency f we ABLATE its
# 2-D irrep component (cos_f, sin_f, in the op's CORRECT basis) from the
# number-token embedding and measure the resulting change in the model's output
# logits — a causal readout-importance imp[f]. Then:
#   k_readout = participation ratio of the per-frequency importance ENERGY
#             = effective number of frequencies the readout actually uses.
# THE RESCUE HYPOTHESIS: the strict capacity law holds for the USED frequencies,
#   2 * k_readout ~= readout_eff_rank  (and k_readout ~= 0.7 * k_eff),
# i.e. the readout rank counts the irreps the readout USES, not the ones the
# embedding ALLOCATES. A NULL is also informative (importance spread over all
# active freqs => no 'allocated but unused' set; the gap is elsewhere).
def _basis_vectors(op, f, p):
    """Length-p token-space (cos_f, sin_f) for frequency f in the op's CORRECT
    group basis (additive Fourier for +,-; multiplicative character via discrete
    log for *,/). Normalized; the x=0 row is 0 in the multiplicative basis."""
    if CORRECT_BASIS.get(op, "additive") == "multiplicative":
        g = primitive_root(p)
        dlog = discrete_log_table(p, g)
        m = p - 1
        c = np.zeros(p); s = np.zeros(p)
        for x in range(1, p):
            ang = 2.0 * np.pi * f * dlog[x] / m
            c[x] = np.cos(ang); s[x] = np.sin(ang)
    else:
        a = np.arange(p)
        c = np.cos(2.0 * np.pi * f * a / p)
        s = np.sin(2.0 * np.pi * f * a / p)
    c = c / (np.linalg.norm(c) + 1e-12)
    s = s / (np.linalg.norm(s) + 1e-12)
    return c, s


def run_readout_attribution(ops=("+", "*", "-"), seeds=(0, 1, 2), p=DEFAULT_P,
                            max_steps=6000, n_grid=2000):
    import torch
    from grokking_dynamics_modadd import build_model, measure_a, accuracy  # noqa: F401
    results = {}
    for op in ops:
        for seed in seeds:
            print(f"\n{'=' * 60}\n  readout-attribution op '{op}' seed {seed}\n{'=' * 60}")
            model, info = _train_grok_cleanup(p, op, seed, max_steps, grok_acc=0.85)
            print(f"  grokked={info['grokked']} test={info['test_acc']} "
                  f"readout_eff_rank={info['readout_eff_rank']} ({info['wall_s']}s)")
            if not info["grokked"]:
                print("  (skipped: did not grok)")
                continue
            device = os.environ.get("IRREP_DEVICE", "cpu")
            OP, EQ = p, p + 1
            rng = np.random.default_rng(1000 + seed)
            pairs = rng.integers(0, p, size=(n_grid, 2))
            X = torch.tensor([[int(a), OP, int(b), EQ] for a, b in pairs],
                             dtype=torch.long, device=device)
            Y = torch.tensor([int((a + b) % p) if op == "+"
                              else int((a * b) % p) if op == "*"
                              else int((a - b) % p) for a, b in pairs],
                             dtype=torch.long, device=device)
            with torch.no_grad():
                L0 = model(X)[:, -1, :p].cpu().numpy()           # [n, p] answer logits
            L0c = L0 - L0.mean(axis=1, keepdims=True)            # center over classes
            base_norm = float(np.linalg.norm(L0c)) + 1e-12
            base_acc = float((L0.argmax(1) == Y.cpu().numpy()).mean())

            E0 = model.tok.weight.detach().cpu().numpy().copy()  # [vocab, d]
            En = E0[:p]                                          # number-token rows
            # active frequencies in the correct basis
            if CORRECT_BASIS.get(op, "additive") == "multiplicative":
                freqs, power, _g, _M = multiplicative_power_spectrum(En, p)
            else:
                freqs, power = fourier_power_spectrum(En, p)
            an = analyze_spectrum(freqs, power)
            active = an["active_freqs"]
            imp = {}
            accs = {}
            for f in active:
                c, s = _basis_vectors(op, f, p)               # [p]
                # remove the rank-2 freq component from every embedding column
                E_f = En - np.outer(c, c @ En) - np.outer(s, s @ En)
                W = model.tok.weight.detach().clone()
                W[:p] = torch.tensor(E_f, dtype=W.dtype)
                with torch.no_grad():
                    model.tok.weight.copy_(W)
                    Lf = model(X)[:, -1, :p].cpu().numpy()
                    model.tok.weight.copy_(torch.tensor(E0, dtype=W.dtype))
                Lfc = Lf - Lf.mean(axis=1, keepdims=True)
                imp[f] = float(np.linalg.norm(Lfc - L0c) / base_norm)  # relative logit change
                accs[f] = float((Lf.argmax(1) == Y.cpu().numpy()).mean())
            # readout-usage participation ratio over importance ENERGY
            e = np.array([imp[f] ** 2 for f in active], float)
            k_readout = float(e.sum() ** 2 / (float((e ** 2).sum()) + 1e-30)) if e.size else 0.0
            keff = an["k_eff"]
            ro = info["readout_eff_rank"]
            # used = importance >= 10% of the max (and >1% acc drop is a stronger gate)
            mx = max(imp.values()) if imp else 0.0
            used = [f for f in active if mx > 0 and imp[f] >= 0.10 * mx]
            unused = [f for f in active if f not in used]
            res = {
                "op": op, "seed": seed, "p": p,
                "correct_basis": CORRECT_BASIS.get(op, "additive"),
                "base_test_acc": round(base_acc, 4),
                "readout_eff_rank": ro,
                "k_eff_embedding": keff, "k_active_embedding": an["k_active"],
                "k_readout_participation": round(k_readout, 3),
                "n_active": len(active), "n_used_freqs": len(used),
                "n_unused_freqs": len(unused),
                "unused_fraction": round(len(unused) / len(active), 3) if active else None,
                "k_readout_over_k_eff": round(k_readout / keff, 3) if keff else None,
                "two_k_readout_over_readout_rank": (round(2 * k_readout / ro, 3)
                                                    if ro else None),
                "active_freqs": active,
                "importance_per_freq": {int(f): round(imp[f], 4) for f in active},
                "acc_after_ablate_per_freq": {int(f): round(accs[f], 4) for f in active},
                "rescue_hypothesis": "2*k_readout ~= readout_eff_rank (the law counts "
                                     "the irreps the readout USES, not those the "
                                     "embedding allocates)",
            }
            path = HERE / f"irrep_readout_attr_{OP_TAG.get(op, op)}_seed{seed}.json"
            path.write_text(json.dumps(res, indent=2))
            print(f"  k_eff(emb)={keff} k_readout(PR)={round(k_readout,2)} "
                  f"k_readout/k_eff={res['k_readout_over_k_eff']} "
                  f"unused_frac={res['unused_fraction']} "
                  f"2k_readout/rank={res['two_k_readout_over_readout_rank']}")
            results[(op, seed)] = res
    return results


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def aggregate():
    rows = {}
    for path in sorted(HERE.glob("irrep_capacity_*.json")):
        if path.stem.endswith(("synthetic", "summary")):
            continue
        d = json.loads(path.read_text())
        rows[d["op"]] = {
            "correct_basis": d.get("correct_basis"),
            "k_eff": d.get("k_eff"),
            "active_freqs": d.get("active_freqs"),
            "embedding_sparsity": d.get("embedding_sparsity"),
            "readout_eff_rank": d.get("readout_eff_rank"),
            "ratio_2k_over_readout_rank": d.get("ratio_2k_over_readout_rank"),
            "verdict": d.get("verdict"),
            "grokked": d.get("training", {}).get("grokked"),
            "grok_step": d.get("training", {}).get("grok_step"),
        }
    in_band = [op for op, r in rows.items()
               if r["verdict"] == "IRREP"
               and r["ratio_2k_over_readout_rank"] is not None
               and RATIO_LO <= r["ratio_2k_over_readout_rank"] <= RATIO_HI]
    # operation-generality: do the active-frequency SETS differ across ops?
    freq_sets = {op: set(r["active_freqs"] or []) for op, r in rows.items()}
    distinct_sets = len({frozenset(s) for s in freq_sets.values()})
    # OP-GENERAL only if EVERY measured op lands in band (in its correct basis).
    program_verdict = ("IRREP_LAW" if len(rows) >= 2 and len(in_band) == len(rows)
                       else "NULL")

    out = {
        "analysis": "representation-theoretic capacity (irrep-dimension budget)",
        "claim": "readout effective rank at grok == 2 * active-irrep count "
                 "(operation-general)",
        "novelty_boundary": "Fourier/clock circuit KNOWN (Nanda et al. 2023); "
                            "the 2k==rank capacity law + op-generality is the new test",
        "date": time.strftime("%Y-%m-%d"),
        "ratio_band": [RATIO_LO, RATIO_HI],
        "per_op": rows,
        "ops_in_band": in_band,
        "distinct_active_frequency_sets": distinct_sets,
        "program_verdict": program_verdict,
    }
    path = HERE / "irrep_capacity_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {path.name}  ({len(rows)} ops)")
    for op, r in rows.items():
        print(f"  op '{op}' [{r['correct_basis']}]: k_eff={r['k_eff']} "
              f"sparsity={r['embedding_sparsity']} "
              f"2k/rank={r['ratio_2k_over_readout_rank']} "
              f"grok_step={r['grok_step']} -> {r['verdict']}")
    print(f"  PROGRAM VERDICT: {program_verdict} "
          f"({len(in_band)}/{len(rows)} op(s) in band; "
          f"{distinct_sets} distinct active-frequency set(s))")
    return out


def main():
    ap = argparse.ArgumentParser(description="Irrep-dimension capacity law test")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--ops", nargs="+", default=None,
                    help="operations to train+measure (default: + * -)")
    ap.add_argument("--p", type=int, default=DEFAULT_P)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--multiseed", action="store_true",
                    help="multi-seed budget-inequality test (2k_eff >= rank?) with CIs")
    ap.add_argument("--readout-attribution", action="store_true",
                    help="per-frequency readout ablation: which active embedding "
                         "irreps does the readout USE? (2k_readout ~= readout rank?)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="seeds for --multiseed / --readout-attribution (default 0 1 2)")
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic(seed=args.seed)
        return
    if args.aggregate:
        aggregate()
        return
    if args.multiseed:
        run_multiseed(ops=tuple(args.ops or DEFAULT_OPS), seeds=tuple(args.seeds),
                      p=args.p, max_steps=args.max_steps)
        return
    if args.readout_attribution:
        run_readout_attribution(ops=tuple(args.ops or DEFAULT_OPS),
                                seeds=tuple(args.seeds), p=args.p,
                                max_steps=args.max_steps)
        return
    run(args.ops or DEFAULT_OPS, p=args.p, seed=args.seed, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
