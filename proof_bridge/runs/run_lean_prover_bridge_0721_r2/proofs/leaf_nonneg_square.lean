import Mathlib.Data.Real.Basic

theorem leaf_nonneg_square (t : ℝ) : 0 ≤ t * t := by
  have h : 0 ≤ t * t := by
    -- Use the fact that the square of any real number is non-negative
    have h₁ : 0 ≤ t * t := by
      -- Use the `nlinarith` tactic to prove the inequality
      exact mul_self_nonneg t
    -- The result follows directly from the above step
    exact h₁
  -- The final result is already obtained in `h`
  exact h
