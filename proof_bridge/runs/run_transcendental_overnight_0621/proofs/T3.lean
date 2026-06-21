import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Pow.Real

theorem T3 (x : ℝ) (hx : 0 < x) : 0 ≤ Real.sqrt x + Real.exp x := by
  have h₁ : 0 ≤ Real.sqrt x := Real.sqrt_nonneg x
  have h₂ : 0 < Real.exp x := Real.exp_pos x
  have h₃ : 0 ≤ Real.exp x := by linarith
  linarith
