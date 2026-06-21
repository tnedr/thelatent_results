import Mathlib.Data.Real.Basic

theorem leaf_mul_comm_nonneg (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a * b := by
  exact mul_nonneg ha hb
