# Transformer Lab Results

Raw experiment JSONs from home GPU and Hetzner `transformer-lab` boxes.

| Path | Contents |
|------|----------|
| `*.json` (root) | ICL-GD Pythia/OLMo grid sweeps |
| `home_probes/` | Home RTX 3080 probe outputs (capacity, routing, composition, …) |
| `hetzner_probes/` | Large-model probes from ephemeral CCX43 runs |
| `hetzner_probes/LESSONS_LEARNED.md` | Hetzner runbook: dtype leaks, stalls, recovery (run #3+) |
