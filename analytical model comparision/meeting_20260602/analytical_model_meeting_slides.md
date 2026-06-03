# Slide 1: Objective And Scope

## Goal of this meeting

- Summarize my current understanding of the analytical coating models
- Explain how the analytical-model predicted data are generated
- Compare analytical predictions against the experimental solvent datasets
- Identify what is still missing before the evaporation-based models can be fully evaluated

## Data used

- Hexane: [hexane comparison](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/hexane/results/comparison_20260521_124637)
- Toluene: [toluene comparison](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/toluene/results/comparison_20260524_163245)
- Ethyl acetate: [ethyl acetate comparison](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/ethyl%20acetate/results/comparison_20260524_163245)

## Main result so far

- The Bayesian-optimized RF remains much more accurate than the currently implemented analytical models
- The evaporation-dependent analytical models are kept symbolic because `E` is not yet known experimentally

---

# Slide 2: Analytical Model Set

## Models implemented

1. `Bonded-Layer Adsorption`
   - Form: `h_B(t) = h_eq [1 - exp(-t/tau)]`
2. `Concentration-Dependent Adsorption Time`
   - Form: `tau(C) = 1 / (k1_eff C + k2)`
   - Prediction: `h_B(t) = y0 + A0 [1 - exp(-t/tau(C))]`
3. `Landau-Levich Wet/Mobile Layer`
   - Form: `h_LL = 0.94 (eta U)^(2/3) / [gamma^(1/6) (rho g)^(1/2)]`
   - Then converted to a bonded-thickness proxy
4. `Capillarity / Evaporation Regime`
   - Symbolic only: `h_cap = k_i E / (L U)`
5. `Combined Capillarity + Landau-Levich`
   - Symbolic only: `h_f = k_i E / (L U) + D U^(2/3)`

## Current constants

- Dwell time: `20 s`
- Withdrawal speed: `1.0 mm/s`
- Film width: `0.065 m`
- Density in LL term: coating-solution density, approximated by solvent
- Relative evaporation reference for hexane: USDA, `BuAc = 1 -> 9`

---

# Slide 3: How Predicted Data Are Generated

## Experimental input columns

- `Concentration (g/mL)`
- `Uncoated Layer (nm)`
- `Total Thickness (nm)`
- `Bonded Thickness (nm)` as the evaluation target

## Prediction workflow

1. Load the solvent-specific experimental dataset
2. For the RF model:
   - add solvent constants: polarity, viscosity, boiling point, surface tension
   - apply the saved Bayesian-optimized RF artifact
3. For adsorption analytical models:
   - use concentration from the dataset
   - use fixed lab constants for dwell time and withdrawal speed
   - compute bonded-thickness prediction row by row
4. For the Landau-Levich model:
   - compute wet-film thickness from `eta`, `U`, `gamma`, `rho`
   - convert the wet-film result to a bonded-thickness proxy
5. Compare each prediction series against measured `Bonded Thickness (nm)` using `R2`, `RMSE`, and `MAE`

## Important limitation

- The evaporation-dependent models cannot be converted into numeric predicted data without an effective experimental `E`

---

# Slide 4: Current Comparison Results

## Summary figure

![Metrics summary](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/meeting_20260602/solvent_metrics_summary.png)

## Key numbers

- Hexane RF: `R2 = 0.9811`, `RMSE = 0.0124`
- Toluene RF: `R2 = 0.9611`, `RMSE = 0.0269`
- Ethyl acetate RF: `R2 = 0.9988`, `RMSE = 0.00909`

## Analytical model behavior

- `Concentration-Dependent Adsorption Time` is the strongest of the currently scored analytical models
- `Bonded-Layer Adsorption` is weaker because it predicts nearly the same value pattern when only time is fixed
- `Landau-Levich Wet/Mobile Layer` is least aligned because it starts from wet-film physics and only approximates bonded thickness

---

# Slide 5: Discussion Points For Lei

## What I think the graphs mean

- The ML model captures solvent-specific nonlinear structure better than the current analytical approximations
- The adsorption models are at least operating in the right bonded-thickness space
- The LL-based model likely needs a better wet-to-bonded conversion or a more defensible mechanistic link
- The evaporation-based models should not be judged numerically until `E` is defined or measured

## What I want to discuss

1. Is the current bonded-thickness proxy from the LL model acceptable, or should it be reformulated?
2. Should `k_i`, `D`, or adsorption constants be solvent-specific fitted parameters?
3. Do we want to estimate `E` from lab measurements, fit `E` from data, or leave those models symbolic in the paper?
4. Which graph should be emphasized in the next presentation or manuscript revision?
