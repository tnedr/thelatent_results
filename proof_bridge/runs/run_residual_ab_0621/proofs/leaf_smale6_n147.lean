import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.ByContra
import Mathlib.Analysis.SpecialFunctions.Pow.Real

theorem lean_lemma_10328 (x : ℝ) : 0 ≤ x → x ≤ 1 → 1 ≤ x → x = 1 := by
  intro hx0 hx1 hx2
  have h₁ : x = 1 := by
    -- Use the fact that x is non-negative and x ≤ 1 to show x = 1
    have h₂ : x ≤ 1 := hx1
    have h₃ : 1 ≤ x := hx2
    -- Since x ≤ 1 and 1 ≤ x, it follows that x = 1
    linarith
  exact h₁
