# Initial validation results

Run date: 2026-10-11. These are exploratory single-seed baselines with the original configuration, not optimized results or evidence that the PINN is superior. No changes were selected using these test scores.

Input: local experimental `data/agg.data.xlsx`, 58 rows, 16 concentration levels. SHA256: `e6f1aee13956864b544868dedd0bb239731271177eb67948eb712fd4706ff11e`. Environment: Python 3.12.5, torch 2.7.1+cpu, numpy 2.3.3, pandas 2.2.3, scikit-learn 1.7.2. Configuration: `config.json`, seed 42.

## Concentration-group holdout

28 training, 12 validation, 18 test rows. Reproduce with `python PINN/train.py --output PINN/results/reproduce_grouped`.

| Model | MAE (nm) | RMSE (nm) | R2 |
|---|---:|---:|---:|
| PINN | 0.111693 | 0.122251 | -0.391624 |
| Bounded NN | 0.111957 | 0.122414 | -0.395331 |
| Standard NN | 0.173946 | 0.211947 | -3.182834 |
| Random forest | 0.056032 | 0.066980 | 0.582261 |
| Gradient boosting | 0.055622 | 0.064933 | 0.607406 |

## Highest-concentration extrapolation

36 training, 12 validation, 10 test rows. Reproduce with `python PINN/train.py --split extrapolation --output PINN/results/reproduce_extrapolation`. Highest four concentration levels are held out completely; validation uses separate lower-concentration groups.

| Model | MAE (nm) | RMSE (nm) | R2 |
|---|---:|---:|---:|
| PINN | 0.430493 | 0.616773 | -759.816923 |
| Bounded NN | 0.427829 | 0.611489 | -746.838691 |
| Standard NN | 0.091286 | 0.104239 | -20.731518 |
| Random forest | 0.025139 | 0.029366 | -0.724719 |
| Gradient boosting | 0.023617 | 0.029637 | -0.756752 |

All five models had zero observed thickness-bound violations on both test sets. Only the bounded models guarantee this property for valid nonnegative totals. Zero observed violations alone does not distinguish models on these small samples.

The PINN and bounded NN perform similarly: the initial curvature weight contributes little benefit here. The learned fraction is multiplied by total thickness, so at large unseen total thickness the bound can remain satisfied while bonded-thickness error becomes large. This is a plausible failure mechanism, not a proven causal explanation. Extrapolation labels have little variance, making R2 particularly unstable; report absolute errors alongside it. Even the trees have negative extrapolation R2, so none of these runs establishes reliable extrapolation.

Keep this configuration as a transparent baseline. Evaluate candidate saturation/adsorption-informed parameterizations only after physical evidence and using training/validation data; then obtain a fresh experimental test. Concentration/total-thickness correlation, measurement timing, and sample independence require lab review. Historical proposal scores use other protocols and are not directly comparable.

## Integrity checks

Five unit tests passed: algebraic bounds and zero-total boundary; independent grouped/extrapolation assignments; training-only scaling and restoration; curvature backpropagation plus raw-unit finite-difference sensitivities; short training and checkpoint equivalence. The command-line inference path was also run on all 58 workbook rows, with finite predictions satisfying the bounds. This validates the implementation, not the scientific adequacy of the learned model.

Generated histories, predictions, checkpoints, sensitivities, and manifests remain under the ignored local `results/` folder. Source and this report are reviewable separately from generated artifacts.
