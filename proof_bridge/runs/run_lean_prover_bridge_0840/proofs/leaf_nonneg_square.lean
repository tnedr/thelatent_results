import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.ByContra

theorem leaf_nonneg_square (t : @Real) : (0 : ℝ) ≤ t * t := by
  exact mul_self_nonneg t
