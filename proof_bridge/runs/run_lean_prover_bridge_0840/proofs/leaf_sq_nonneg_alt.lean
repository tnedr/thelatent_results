import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.ByContra

theorem leaf_sq_nonneg_alt (x : @Real) : (0 : ℝ) ≤ x * x := by
  have h : 0 ≤ x * x := by
    -- Use the fact that the square of any real number is non-negative.
    have h₁ : 0 ≤ x * x := by
      -- Use the `nlinarith` tactic to prove the inequality.
      nlinarith [sq_nonneg x]
    -- The result follows directly from the above statement.
    exact h₁
  -- The final result is already derived, so we just use it.
  exact h
