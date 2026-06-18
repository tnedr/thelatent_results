# Hetzner transformer-lab — lessons learned

Operational notes from CCX43 runs that produced artifacts in this repo.
Canonical home tooling docs: `thelatent_home/tools/lab_home/LESSONS_LEARNED.md`.

---

## Run #3 (2026-06-18)

| Field | Value |
|-------|--------|
| Box | `transformer-lab`, Hetzner id `142594043`, IP `178.104.77.197` |
| Type | CCX43, nbg1 (CPU-only, no GPU) |
| Batch | `batch_hetzner_lab.sh` |
| Results partial pull | commit `cb96fbf` — phases 1–2 (grid 2.8b/12b + `icl_convergence_*`) |
| Phase 4 stall | `singular_spectrum` ran ~9h under leaked `ICL_DTYPE=float16`; restarted float32 |

### Artifacts in this repo (as of partial pull)

- Root: `icl_gd_emergence_pythia-2.8b.json`, `icl_gd_emergence_pythia-12b.json`, `grid_progress.json`
- `hetzner_probes/`: `icl_convergence_pythia-6_9b.json`, `icl_convergence_pythia-12b.json`, `icl_convergence_OLMo-2-7B.json`, panel summary
- `MANIFEST_partial.json` — phases 1–2 only; capacity / singular / composition may arrive later

---

## Critical: `ICL_DTYPE=float16` breaks `singular_spectrum_probe` on CPU

### Symptom

- `singular_spectrum_probe.py --models pythia-2.8b --n-samples 64` runs **8+ hours**, **no output JSON**.
- Log stops after `Loading weights: 100%`; process stays at ~100% CPU — looks hung but is not.

### Cause

Phase 2 of `batch_hetzner_lab.sh` sets `export ICL_DTYPE=float16` for large **forward-only** probes (`icl_convergence`). Unless cleared, that env leaks into phase 4.

`singular_spectrum_probe` loads the model via `icl_convergence_probe.load_model()` (respects `ICL_DTYPE`) and runs a **full backward pass** per prompt (`capture_query_state`).

On CCX43 CPU, fp16 backward is ~50× slower than fp32:

| `ICL_DTYPE` | 1 backward pass (pythia-2.8b) | Full singular run (~224 passes) |
|-------------|------------------------------|----------------------------------|
| unset (float32) | ~3.3 s | ~12–15 min / model |
| `float16` | ~162 s | ~10 h / model |

**Do not** extrapolate home GPU timings (3080: ~20 s/model for the same probe) to Hetzner CPU.

### Fix

Before phase 4 in the batch script:

```bash
unset ICL_DTYPE
```

Re-export `ICL_DTYPE=float16` before phase 5 (`composition_probe`) if desired.

### Recovery (mid-batch)

```bash
kill <singular_spectrum_pid>
rm -f .../experiments/singular_spectrum_*.json   # partial files trigger SKIP
/usr/bin/tmux new-session -d -s lab 'bash /opt/lab/continue_phase4_float32.sh'
```

### Dtype per probe on Hetzner CPU

| Probe | `ICL_DTYPE` | Notes |
|-------|-------------|--------|
| `icl_convergence_probe` | `float16` OK | Forward-heavy |
| `capacity_threshold_sweep` | either | `torch.no_grad` |
| **`singular_spectrum_probe`** | **unset (float32)** | **full backward** |
| `composition_probe` | `float16` OK | Forward-heavy |

Always `unset` or re-`export` at phase boundaries — bash does not reset env between loop phases.

### Quick diagnosis on box

```bash
tr '\0' '\n' < /proc/<pid>/environ | grep ICL_DTYPE
py-spy dump --pid <pid>   # expect capture_query_state → backward
```

---

## Other issues affecting Hetzner grid quality

### Pythia early-revision `pad_token` (phases 1 / grid)

Early Pythia checkpoints (`step1`, …) have `pad_token=None` and `eos_token=None`. Batched tokenization crashes → incomplete JSONs (e.g. 2.8b with only 2 checkpoints).

**Fix:** `_ensure_pad_token()` in `icl_gd_emergence.py` (patch on box until corp drips). See home `patch_pad_token.py`.

### Hugging Face download stalls

`RemoteProtocolError` on large shards (12b). Mitigations: `HF_HUB_DISABLE_XET=1`, `HF_TOKEN`, in-script retries. Incomplete runs stay `INCOMPLETE` and are not pushed.

### Provisioning (Windows → box)

- CRLF in SSH heredocs breaks bash (`set -\r`, stray `EOF` in `config.env`).
- Poll SSH after API create (30–90 s lag); verify `tmux ls` + `batch_hetzner_lab.log` after `-StartBatch`.
- Recycled IP → `ssh-keygen -R <IP>`.

---

## Batch phases (`batch_hetzner_lab.sh`)

1. `run_grid.sh --rerun=2.8b,12b`
2. `icl_convergence` — 6.9b, 12b, OLMo-7B (`ICL_DTYPE=float16`)
3. `capacity_threshold` — 2.8b, 6.9b, OLMo-7B
4. `singular_spectrum` — same (**`unset ICL_DTYPE`**)
5. `composition_probe` — 6.9b, 12b, OLMo-7B
6. Push → `hetzner_probes/`

**Billing:** delete `transformer-lab` when done; powered-off boxes still cost.

---

*Last updated: 2026-06-18 (run #3, singular float16 stall + float32 restart).*
