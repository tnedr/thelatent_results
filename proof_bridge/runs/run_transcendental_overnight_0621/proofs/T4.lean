import Mathlib.Analysis.SpecialFunctions.Pow.Real

theorem T4 (a b : ℝ) (h : a < b) : (2 : ℝ) ^ a < (2 : ℝ) ^ b := by
  have h₁ : (2 : ℝ) > 1 := by norm_num
  -- Use the property of real power functions with base > 1
  have h₂ : (2 : ℝ) ^ a < (2 : ℝ) ^ b := by
    -- Apply the lemma `Real.rpow_lt_rpow_of_exponent_lt` which states that if `b > a` and the base is `> 1`, then `b^a < b^b`
    apply Real.rpow_lt_rpow_of_exponent_lt
    · linarith -- Prove that `2 > 1` using `linarith`
    · linarith -- Prove that `a < b` using `linarith`
  -- The result follows directly from the above steps
  exact h₂
