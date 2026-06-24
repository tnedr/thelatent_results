# Plastic operator Part B — home summary

Dispatch: `dispatch_20260622_0758_home_plastic_operator_partB`
Machine: home RTX 3080 | `ICL_ATTN_IMPLEMENTATION=eager`

## T2 — INSTRUMENT_REDEEMS across scale?

**Yes** on 1b, 2.8b, 410m (2118); 160m MIXED (causal 2.2x still >1).
See `PANEL.jsonl` for panel + random baseline.

## T5 — frozen vs tracker on recall (410m)

`plastic_instrument_real_EleutherAI_pythia-410m_recall_frozen_vs_track.json`
Frozen alpha=0: align_to_final ~0.02 flat. Tracker alpha=0.4: 0.16 -> 1.0.

## T3 + geometry (H2/B3/B2)

`plastic_operator_layer_clay_*.json`, `plastic_operator_geometry_*.json`
