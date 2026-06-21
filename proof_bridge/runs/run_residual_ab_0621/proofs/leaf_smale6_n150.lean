import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.ByContra
import Mathlib.Analysis.SpecialFunctions.Pow.Real

theorem lean_workbook_plus_15070  (n : ℕ)
  (h₁ : n ≥ 1)
  (h₂ : (n : ℕ) ≥ 1) :
  n ^ 2 ≥ n := by
  have h₃ : n ^ 2 ≥ n := by
    have h₄ : n ≥ 1 := h₁
    have h₅ : n ^ 2 ≥ n := by
      -- Use the fact that n ≥ 1 to prove n^2 ≥ n
      have h₆ : n ≤ n ^ 2 := by
        -- Prove that n ≤ n^2 for n ≥ 1
        nlinarith
      -- Since n ≤ n^2, we have n^2 ≥ n
      linarith
    exact h₅
  exact h₃
