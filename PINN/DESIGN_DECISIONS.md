# Initial design and feature decisions

Date: 2026-10-11. Choices below are starting hypotheses, not optimized conclusions.

## Scientific scope

Predict measured bonded thickness conditional on concentration, uncoated layer, and measured total thickness. This is useful for inference from measured film state. Because total thickness is a post-coating measurement, the model is **not** a recipe-only process predictor or a causal inverse-design tool. Confirm the timing/meaning of the uncoated-layer measurement before deployment. No target-derived mobile-layer feature (`total - bonded`) is included: that would leak the target.

The supplied proposal suggests boundedness, smoothness, monotonic trends, and hydrodynamic scaling. The slide explicitly makes monotonic trends candidates to test. We enforce the robust thickness inequality and enable configurable regularization; absent process measurements are not fabricated from experiment-level defaults in older analytical code.

## Feature register

| Feature | Initial treatment | Reason and limitation |
|---|---|---|
| Concentration (g/mL) | Natural logarithm, then training-only standardization | Recorded values span 0.02 to 10 (500-fold); logarithm reduces domination by large values while preserving ordering. The g/mL label and especially large values require lab confirmation; we retain recorded units rather than guess a conversion. |
| Uncoated Layer (nm) | Training-only standardization | Included in proposal and measured workbook. Narrow range (1.42-1.64 nm) makes sensitivity estimates fragile. |
| Total Thickness (nm) | Standardized network input; raw value also defines output bound | Captures measured film state and enforces bonded <= total. Available at inference only if measured; it can encode downstream information. |
| Bonded Thickness (nm) | Sole target, never an input | Explicit column mapping avoids older automatic selectors that treat every thickness as a target. |
| Withdrawal velocity | Excluded until measured | No column in the default workbook. Smoothness in velocity cannot be tested or claimed. |
| Viscosity, surface tension, density, solvent identity, dwell time | Excluded initially | Not present in the four-column experimental workbook. Synthetic or solvent-level constants cannot establish within-solvent effects. Add measured covariates and experimental grouping when available. |

No interactions, polynomial features, or synthetic augmentation initially: a smooth network already learns nonlinear interactions and 58 measurements provide limited independent evidence. No residual based on fitted analytical coefficients is imported: fitting those coefficients on all rows could leak test information.

## Network and loss

Three inputs -> 16 tanh units -> 16 tanh units -> one logit, totaling 353 trainable parameters. Width/depth are intentionally modest given sample count but still exceed the number of rows; regularization and independent validation remain necessary. Tanh supports meaningful second derivatives; a ReLU curvature penalty would be zero almost everywhere.

`bonded_hat = total_measured * sigmoid(network(x))` guarantees `0 <= bonded_hat <= total_measured` for a nonnegative envelope. With finite logits the endpoints are approached asymptotically; exact saturated labels may be harder to fit. For numerical training/inference we require positive measured total thickness. The architecture's zero-total boundary behavior is tested separately.

`L = mean(((bonded_hat - bonded)/s_y)^2) + 0.001*L_smooth + lambda_mono*L_mono`

Here `s_y` is the training population standard deviation in nm, floored at 1e-6. Smoothness is the mean squared **diagonal second derivative** of `bonded_hat/s_y` with respect to standardized transformed inputs, at training rows. It is a curvature prior, not a conservation law or full Hessian norm. Differentiation includes the total-thickness envelope (so total sensitivity is not accidentally held fixed). No collocation points are generated across arbitrary combinations of correlated film measurements.

Monotonic penalties are disabled (`monotonic_weight = 0`, empty sign map). To test an independently justified positive concentration trend, set `monotonic` to `{"Concentration (g/mL)": 1}` and give `monotonic_weight` a nonzero value in a separate configuration. Negative trends use -1. Penalties operate on local partial derivatives at training observations; they do not guarantee global monotonicity. Standardization and positive log transforms preserve derivative sign. Test a trend using training data and external physical evidence before enabling it; pooled correlations can confound solvent/batch effects. Do not choose signs using test outcomes.

## Optimization and comparisons

Adam learning rate 1e-3, weight decay 1e-4, full-batch CPU training, maximum 1500 epochs, gradient norm clipping at 10, and validation patience 150 are conservative reproducible starting values. Xavier initialization, fixed seed 42, one CPU thread, and deterministic PyTorch operations reduce run-to-run noise. Checkpoints minimize validation data loss; test data never select epochs. No loss annealing or learned weights initially: fixed, normalized terms make the first ablation interpretable. Logged data/smoothness/monotonic terms can reveal domination and motivate later balancing.

The bounded-only NN isolates the curvature prior from the hard bound. The standard NN uses the same hidden sizes, seed, preprocessing, normalized data loss, optimizer, and early stopping but has an unconstrained linear output. RF (300 trees, minimum leaf size 2) and GB (100 shallow depth-2 trees, learning rate .05) are untuned initial baselines. Identical partitions matter more than comparing to historical scores obtained with different or synthetic splits. RF/GB do not use validation for selection because their settings are fixed before the run.

## Evaluation and limits

Default splitting uses concentration groups: 20% of levels are reserved for test, then 20% of remaining levels for validation. Fractions apply to levels, not rows; resulting row fractions vary. Extrapolation reserves the largest concentration levels. This tests unseen concentrations, not unseen solvents, future batches, or spatial PDE generalization. Replicates must ultimately be grouped by actual independent experiment IDs. One seed and one small holdout cannot substantiate superiority or confidence intervals.

Sensitivity outputs are local partial derivatives in recorded input units, including chain-rule correction for log concentration and the total envelope. They are associations, not causal effects; changing total while holding other film measurements fixed may leave the physical data manifold. No calibrated uncertainty is claimed. Multi-seed ensembles or Bayesian extensions should follow a noise/replicate audit.

## Hydrodynamics and next measurements

Landau-Levich-Derjaguin scaling describes entrained wet-film behavior under specified geometry and flow assumptions. The repository separately compares it with the mobile layer; bonded adsorption is a different mechanism. The current model has **no LLD loss**, because velocity, solution density, verified viscosity/surface tension, wet-versus-dry conversion, and regime validity are unavailable. A dry mobile-layer residual would additionally need measured concentration units, polymer density, evaporation/drying assumptions and an appropriate conversion. Do not simply equate wet LLD thickness to dry bonded thickness.

Next steps: confirm units and thickness definitions; record batch/specimen IDs, velocity, dwell time, solvent/temperature, solution rheology and measurement uncertainty; establish an adsorption/transport equation with the lab; fit any unknown physical parameters using training data only and examine identifiability. Then add a named physical residual with explicit units, regime masks and ablations. Extend the feature/configuration register and change log whenever these choices change.
