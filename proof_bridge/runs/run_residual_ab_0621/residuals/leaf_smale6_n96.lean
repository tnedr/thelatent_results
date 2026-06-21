import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.ByContra
import Mathlib.Analysis.SpecialFunctions.Pow.Real

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false
set_option maxHeartbeats 400000
set_option linter.style.longLine false

noncomputable section

/- Platonic Real operations bridged to Lean 4 standard typeclass instances.
   This makes linarith/nlinarith work on exported proof terms. -/
abbrev Real.zero : Real := 0
abbrev Real.one : Real := 1
abbrev Real.add (a b : Real) : Real := a + b
abbrev Real.mul (a b : Real) : Real := a * b
abbrev Real.sub (a b : Real) : Real := a - b
abbrev Real.neg (a : Real) : Real := -a
abbrev Real.div (a b : Real) : Real := a / b
abbrev Real.lt (a b : Real) : Prop := a < b
abbrev Real.le (a b : Real) : Prop := a ≤ b
abbrev Real.ge (a b : Real) : Prop := a ≥ b
abbrev Real.gt (a b : Real) : Prop := a > b
noncomputable abbrev Real.ofNat (n : Nat) : Real := (n : Real)
noncomputable abbrev Real.ofNat_2  : Real := 2
noncomputable abbrev Real.ofNat_3  : Real := 3
noncomputable abbrev Real.ofNat_4  : Real := 4
noncomputable abbrev Real.ofNat_5  : Real := 5
noncomputable abbrev Real.ofNat_6  : Real := 6
noncomputable abbrev Real.ofNat_7  : Real := 7
noncomputable abbrev Real.ofNat_8  : Real := 8
noncomputable abbrev Real.ofNat_9  : Real := 9
noncomputable abbrev Real.ofNat_10 : Real := 10
abbrev Real.pow (a : Real) (n : Nat) : Real := a ^ n
-- Nat-exponent power recursion (Monoid.npow). The kernel axioms Real.pow_zero /
-- Real.pow_succ are suppressed in mathlib mode (mathlib_mappings._MATHLIB_REFS);
-- these bridges supply the real declarations. Keep in sync with
-- mathlib_mappings._BRIDGE_LEMMAS.
theorem Real.pow_zero (x : Real) : Real.pow x Nat.zero = Real.one := by simp
theorem Real.pow_succ (x : Real) (n : Nat) : Real.pow x (Nat.succ n) = Real.mul (Real.pow x n) x := by simpa [Nat.succ_eq_add_one] using _root_.pow_succ x n
noncomputable def Real.max (a b : Real) : Real := Max.max a b
noncomputable def Real.min (a b : Real) : Real := Min.min a b
noncomputable def Real.abs (a : Real) : Real := |a|
/- Order lemmas the term-mode exporter may reference directly (e.g. a
   transitivity proof collapses to `@Real.le_trans a b c h1 h2`). In
   mathlib mode the corresponding bootstrap axioms are skipped, so we
   provide explicit-argument bridges to the standard Mathlib lemmas. -/
theorem Real.le_refl (a : Real) : a ≤ a := _root_.le_refl a
theorem Real.le_trans (a b c : Real) (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c := _root_.le_trans h1 h2
theorem Real.lt_trans (a b c : Real) (h1 : a < b) (h2 : b < c) : a < c := _root_.lt_trans h1 h2
theorem Real.le_of_lt (a b : Real) (h : a < b) : a ≤ b := _root_.le_of_lt h
theorem Real.lt_of_le_of_lt (a b c : Real) (h1 : a ≤ b) (h2 : b < c) : a < c := _root_.lt_of_le_of_lt h1 h2
theorem Real.lt_of_lt_of_le (a b c : Real) (h1 : a < b) (h2 : b ≤ c) : a < c := _root_.lt_of_lt_of_le h1 h2
theorem Real.le_antisymm (a b : Real) (h1 : a ≤ b) (h2 : b ≤ a) : a = b := _root_.le_antisymm h1 h2
/- Commutative-ring algebra lemmas the term-mode exporter references for
   `ring`-style equalities (e.g. the degree-<=2 polynomial identities). In
   mathlib mode the Base.lean axioms that would otherwise provide these are
   skipped, so we bridge them to the standard unprefixed Mathlib lemmas. -/
theorem Real.mul_comm (a b : Real) : a * b = b * a := _root_.mul_comm a b
theorem Real.mul_assoc (a b c : Real) : a * b * c = a * (b * c) := _root_.mul_assoc a b c
theorem Real.add_comm (a b : Real) : a + b = b + a := _root_.add_comm a b
theorem Real.add_assoc (a b c : Real) : a + b + c = a + (b + c) := _root_.add_assoc a b c
theorem Real.add_left_comm (a b c : Real) : a + (b + c) = b + (a + c) := _root_.add_left_comm a b c
theorem Real.add_zero (a : Real) : a + 0 = a := _root_.add_zero a
theorem Real.zero_add (a : Real) : 0 + a = a := _root_.zero_add a
theorem Real.mul_one (a : Real) : a * 1 = a := _root_.mul_one a
theorem Real.one_mul (a : Real) : 1 * a = a := _root_.one_mul a
theorem Real.sub_eq_add_neg (a b : Real) : a - b = a + -b := _root_.sub_eq_add_neg a b
theorem Real.mul_pos (a b : Real) (ha : 0 < a) (hb : 0 < b) : 0 < a * b := _root_.mul_pos ha hb
theorem Real.mul_nonneg (a b : Real) (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a * b := _root_.mul_nonneg ha hb
theorem Real.mul_self_nonneg (a : Real) : 0 ≤ a * a := _root_.mul_self_nonneg a
theorem Real.add_pos (a b : Real) (ha : 0 < a) (hb : 0 < b) : 0 < a + b := _root_.add_pos ha hb
theorem Real.add_nonneg (a b : Real) (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a + b := _root_.add_nonneg ha hb
theorem Real.div_pos (a b : Real) (ha : 0 < a) (hb : 0 < b) : 0 < a / b := _root_.div_pos ha hb
-- NB: the Platonic kernel axiom `Real.div_nonneg` requires STRICT positivity of
-- the denominator (`0 < b`, see bootstrap/foundations.py) so the native
-- sign-prover can discharge it from strict denominator positivity. This bridge
-- MUST mirror that signature exactly (weakening to `0 ≤ b` internally via
-- `le_of_lt`), or native `@Real.div_nonneg …` proof terms fail to elaborate.
theorem Real.div_nonneg (a b : Real) (ha : 0 ≤ a) (hb : 0 < b) : 0 ≤ a / b := _root_.div_nonneg ha (_root_.le_of_lt hb)
theorem Real.nonneg_of_mul_nonneg_left (a b : Real) (h : 0 ≤ a * b) (ha : 0 < a) : 0 ≤ b := by by_contra hnb; push_neg at hnb; nlinarith [mul_neg_of_pos_of_neg ha hnb]
-- Division comparison bridges (mirror the kernel axioms added for the native
-- `_try_div_comparison` tactic). Each maps to the `.mpr` direction of the
-- corresponding Mathlib `div_*_iff₀` biconditional. Keep in sync with
-- mathlib_mappings._BRIDGE_LEMMAS.
theorem Real.div_le_div_of_cross (a b c d : Real) (hb : 0 < b) (hd : 0 < d) (h : a * d ≤ c * b) : a / b ≤ c / d := (_root_.div_le_div_iff₀ hb hd).mpr h
theorem Real.div_lt_div_of_cross (a b c d : Real) (hb : 0 < b) (hd : 0 < d) (h : a * d < c * b) : a / b < c / d := (_root_.div_lt_div_iff₀ hb hd).mpr h
theorem Real.div_le_of_le_mul (a b c : Real) (hb : 0 < b) (h : a ≤ c * b) : a / b ≤ c := (_root_.div_le_iff₀ hb).mpr h
theorem Real.div_lt_of_lt_mul (a b c : Real) (hb : 0 < b) (h : a < c * b) : a / b < c := (_root_.div_lt_iff₀ hb).mpr h
theorem Real.le_div_of_mul_le (a c d : Real) (hd : 0 < d) (h : a * d ≤ c) : a ≤ c / d := (_root_.le_div_iff₀ hd).mpr h
theorem Real.lt_div_of_mul_lt (a c d : Real) (hd : 0 < d) (h : a * d < c) : a < c / d := (_root_.lt_div_iff₀ hd).mpr h
-- Numerator-positivity and bounded-square bridges (mirror the kernel axioms
-- added for cosmic_inflation amplitude_constraint / dbi_non_gaussianity).
-- Keep in sync with mathlib_mappings._BRIDGE_LEMMAS.
theorem Real.pos_of_div_pos (a b : Real) (h : 0 < a / b) (hb : 0 < b) : 0 < a := by
  have h2 := _root_.mul_pos h hb
  rwa [_root_.div_mul_cancel₀ a (_root_.ne_of_gt hb)] at h2
theorem Real.mul_self_lt_one (a : Real) (h0 : 0 < a) (h1 : a < 1) : a * a < 1 := by
  nlinarith [h0, h1]
theorem Real.sub_nonneg (a b : Real) (h : a ≤ b) : 0 ≤ b - a := _root_.sub_nonneg.mpr h
theorem Real.sub_pos (a b : Real) (h : a < b) : 0 < b - a := _root_.sub_pos.mpr h
-- Reverse directions of sub_nonneg / sub_pos, used by the native Farkas
-- assembler to discharge `≤` / `<` goals (fold hyps into `0 ≤/< R − L`, then
-- reduce the goal). Kept in sync with mathlib_mappings._BRIDGE_LEMMAS.
theorem Real.le_of_sub_nonneg (a b : Real) (h : 0 ≤ b - a) : a ≤ b := _root_.le_of_sub_nonneg h
theorem Real.lt_of_sub_pos (a b : Real) (h : 0 < b - a) : a < b := _root_.sub_pos.mp h
theorem Real.lt_add_one (a : Real) : a < a + 1 := _root_.lt_add_one a
theorem Real.add_le_add_right (a b c : Real) (h : a ≤ b) : a + c ≤ b + c := by linarith
theorem Real.add_le_add_left (a b c : Real) (h : a ≤ b) : c + a ≤ c + b := by linarith
theorem Real.add_le_add (a b c d : Real) (h1 : a ≤ b) (h2 : c ≤ d) : a + c ≤ b + d := by linarith
theorem Real.sub_self (a : Real) : a - a = 0 := by linarith
theorem Real.neg_lt_neg (a b : Real) (h : a < b) : -b < -a := by linarith
theorem Real.add_self (a : Real) : a + a = Real.ofNat 2 * a := by
  show a + a = ((2 : Nat) : Real) * a
  push_cast; ring
/- Remaining algebra/order lemmas the term-mode exporter references. These were
   listed in `_MATHLIB_PROVIDED` but had no preamble bridge, so any proof term
   that used one resolved to an `Unknown constant` → `sorryAx` while still
   `lake build`-ing green (see axiom_audit.py --deep-check). Every bridge below
   was compile-verified against the pinned Mathlib before being added. -/
theorem Real.mul_zero (a : Real) : a * 0 = 0 := by ring
theorem Real.zero_mul (a : Real) : 0 * a = 0 := by ring
theorem Real.neg_neg (a : Real) : - -a = a := _root_.neg_neg a
theorem Real.neg_add (a b : Real) : -(a + b) = -a + -b := _root_.neg_add a b
theorem Real.neg_mul (a b : Real) : -a * b = -(a * b) := _root_.neg_mul a b
theorem Real.mul_neg (a b : Real) : a * -b = -(a * b) := _root_.mul_neg a b
theorem Real.left_distrib (a b c : Real) : a * (b + c) = a * b + a * c := _root_.left_distrib a b c
theorem Real.right_distrib (a b c : Real) : (a + b) * c = a * c + b * c := _root_.right_distrib a b c
theorem Real.lt_irrefl (a : Real) : ¬ a < a := _root_.lt_irrefl a
theorem Real.neg_le_neg (a b : Real) (h : a ≤ b) : -b ≤ -a := _root_.neg_le_neg h
theorem Real.sub_le_sub_right (a b c : Real) (h : a ≤ b) : a - c ≤ b - c := _root_.sub_le_sub_right h c
theorem Real.mul_le_mul_of_nonneg_left (a b c : Real) (h : b ≤ c) (hc : 0 ≤ a) : a * b ≤ a * c := _root_.mul_le_mul_of_nonneg_left h hc
theorem Real.add_neg_cancel (a : Real) : a + -a = 0 := by ring
theorem Real.gt_iff_lt (a b : Real) : a > b ↔ b < a := Iff.rfl
theorem Real.ge_iff_le (a b : Real) : a ≥ b ↔ b ≤ a := Iff.rfl
theorem Real.ofNat_zero : Real.ofNat 0 = 0 := by simp [Real.ofNat]
theorem Real.ofNat_one : Real.ofNat 1 = 1 := by simp [Real.ofNat]
theorem Real.ofNat_nonneg (k : Nat) : (0 : Real) ≤ Real.ofNat k := by show (0 : Real) ≤ ((k : Nat) : Real); positivity
theorem Real.ofNat_pos (k : Nat) : (0 : Real) < Real.ofNat (Nat.succ k) := by show (0 : Real) < ((Nat.succ k : Nat) : Real); exact_mod_cast Nat.succ_pos k

-- [sorry: auto_solve failed (timeout=5000ms)]
theorem leaf_smale6_n96 (t : @Real) (h_0 : (0 : ℝ) ≤ t) (h_1 : t ≤ (@Real.ofNat 1)) : (0 : ℝ) ≤ (@Real.ofNat 1) - (@Real.rpow t ((@Real.ofNat 18089) / (@Real.ofNat 49171))) := by sorry

end