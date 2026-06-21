import Mathlib.Analysis.SpecialFunctions.Exp

theorem T5 (c t : ℝ) (hc : 0 ≤ c) (ht : 0 ≤ t) : Real.exp (-(c * t)) ≤ 1 := by
  have h₁ : -(c * t) ≤ 0 := by
    -- Prove that -(c * t) ≤ 0 using the fact that c * t ≥ 0
    have h₂ : c * t ≥ 0 := by
      nlinarith
    -- Use the fact that c * t ≥ 0 to show -(c * t) ≤ 0
    linarith
  
  -- Use the property of the exponential function that exp(x) ≤ 1 when x ≤ 0
  have h₂ : Real.exp (-(c * t)) ≤ 1 := by
    apply Real.exp_le_one_iff.mpr
    exact h₁
  
  exact h₂
