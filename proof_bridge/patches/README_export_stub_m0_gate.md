# export_goal_stub Real-bridge compile gate (M0)

The full `export_goal_stub` module includes Real-bridge lemmas that reference
`nlinarith` / `linarith` without guaranteed tactic imports, and some Mathlib
`.olean` targets are missing on Windows (e.g. `Mathlib.Tactic.Nlinarith` source
file absent in the pinned mathlib checkout).

**Symptom on home (2026-06-21):** M0 `stub_only_sorry_warning` fails with
`unknown tactic` on bridge lines 77–108; target `theorem leaf_nonneg_square := by sorry`
is well-formed at line 137.

**Workaround on home:** M2 hand-roundtrip uses a minimal standalone proof file;
M3 seal records provenance against the export stub hash while the sealed `.lean`
is the kernel-verified proof.

**Corp promotion:** add default `extra_imports` for tactic modules in
`export_goal_stub` callers and ensure Real-bridge lemmas compile under
`lake env lean` before M0 gate is required.
