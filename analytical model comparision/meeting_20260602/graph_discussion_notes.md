# Graph Discussion Notes

## Core message

The graph should be discussed as a comparison between:

- measured bonded thickness from each solvent dataset
- predicted bonded thickness from the saved RF model
- predicted bonded thickness from the analytical models that can currently be evaluated numerically
- symbolic evaporation-based models that still require `E`

## How the predicted data are obtained

### Bayesian-optimized RF

1. Start from the solvent dataset CSV
2. Add solvent properties used by the 7-feature RF
3. Reorder the features to match the saved model artifact
4. Run inference with `best_estimator.pkl`
5. Compare against measured bonded thickness

### Bonded-layer adsorption

1. Treat dwell time as the experiment constant `20 s`
2. Use the exponential bonded-layer growth form
3. Compute one predicted bonded-thickness value per row

### Concentration-dependent adsorption time

1. Read concentration directly from the solvent dataset
2. Convert concentration into a characteristic time `tau(C)`
3. Insert `tau(C)` into the adsorption-growth equation
4. Compute one predicted bonded-thickness value per row

### Landau-Levich wet/mobile layer

1. Use solvent viscosity, surface tension, and density
2. Use the confirmed withdrawal speed
3. Compute entrained wet-film thickness
4. Convert that wet-film result into a bonded-thickness proxy

### Evaporation-based models

1. Keep the formulas in terms of `E`
2. Do not assign a numeric prediction until `E` is available

## What to say if asked why the analytical graphs underperform

- They use simplified mechanistic forms with fixed constants
- They do not currently include a validated effective evaporation rate
- They are trying to predict bonded thickness, while some formulas originate from wet-film physics
- The ML model absorbs nonlinear solvent-property interactions directly from the expanded training set

## Recommended graph emphasis

- Focus on the solvent-specific scatter plots and the metrics summary
- Emphasize the prediction-generation chain, not only the final error values
- Make clear that the evaporation-based analytical models are incomplete numerically because `E` is missing
