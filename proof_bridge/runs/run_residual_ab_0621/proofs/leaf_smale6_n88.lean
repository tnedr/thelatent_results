import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.ByContra
import Mathlib.Analysis.SpecialFunctions.Pow.Real

theorem lean_workbook_plus_51346 (x : ℝ) (h₀ : x ≥ 0) (h₁ : x ^ 2 ≤ 1) (h₂ : x ≥ 0) : x ≤ 1 := by
  have h₃ : x ≤ 1 := by
    by_contra h
    -- Assume for contradiction that x > 1
    have h₄ : x > 1 := by linarith
    -- Since x > 1, we have x^2 > 1
    have h₅ : x ^ 2 > 1 := by
      nlinarith
    -- This contradicts the given condition x^2 ≤ 1
    linarith
  exact h₃
