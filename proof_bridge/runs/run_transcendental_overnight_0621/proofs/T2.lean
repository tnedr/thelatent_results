import Mathlib.Analysis.SpecialFunctions.Exp

theorem T2 (x : ℝ) (hx : 0 ≤ x) (hx2 : x ≤ 2) : 0 ≤ Real.exp x - 1 := by
  have h₁ : Real.exp x ≥ 1 := by
    -- Use the fact that `Real.exp x ≥ 1` for all real `x` because `exp` is always positive and `exp(0) = 1`.
    have h₂ : Real.exp x ≥ 1 := by
      linarith [Real.add_one_le_exp x]
    exact h₂
  -- Since `Real.exp x ≥ 1`, it follows that `Real.exp x - 1 ≥ 0`.
  linarith
