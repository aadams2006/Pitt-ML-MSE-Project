# Li Lab research follow-up: November 11, 2026

## Outcome and scope

Executed 24 recorded runs (120 model fits) on 126 existing measured records spanning 36 solvent/concentration conditions. These are computational experiments, not newly collected laboratory measurements. Campaign: `20261111T134229Z_measured_followup`. Runtime UTC and New York timestamps, exact source, configuration, environment, checkpoints, splits and artifact hashes are retained in each run under `PINN/runs/`.

The October 14 update (`10-14_li.pptx`, slides 22-23) requested dataset alignment, separate loss comparisons, repeated/grouped validation and data-efficiency tests. This work prepares and evaluates the available measured data. Additional velocity and hydrodynamic penalties remain blocked by missing sample-linked, validated physics inputs in this checkout. The prior statement that the values are available is preserved; the missing step is bringing those values into a traceable table, not asserting they do not exist.

## Data preparation and exclusions

Reused the reviewed `near_term/run_analysis.py` canonical loader rather than inventing sample matches. Combined 142 source records: hexane 58, toluene 51, ethyl acetate 33. Excluded 15 potentially overlapping toluene summaries and one physically inconsistent toluene record (0.86 nm bonded > 0.85 nm total). Primary cohort: hexane 58, toluene 35, ethyl acetate 33. No clipping, imputation or synthetic augmentation. Raw files and old results were preserved.

Concentration is g/L, consistent with the earlier workbook-cell audit; legacy g/mL headings were erroneous. Values are unchanged. Source path/row, stable record ID and condition group are retained in the prepared snapshot. `data_provenance.json`, `canonical_all.csv`, `excluded_records.csv` and `workbook_inventory.json` preserve this audit. `physics_cell_evidence.json` records the targeted workbook search (no matching cells). Existing solvent descriptors and assumed 1 mm/s velocity from the analytical scripts are not promoted to measured solution properties. `physics_measurement_intake.csv` provides 126 pre-keyed rows for matching the remaining measurements.

## Changes

- `train.py`: deterministic five-fold grouped outer validation with a fixed training/validation split, independent of network initialization seed; opt-in whole-condition training subsampling with nested subsets.
- `core.py`: configurable smoothness columns so derivatives of solvent indicator columns are excluded. Existing default behavior is unchanged.
- `run_tracking.py`: model version 0.3.0; existing run snapshots remain immutable.
- `run_followup.py`: preparation, fixed experiment plan, immutable timestamped runs and incremental completion record.
- `summarize_followup.py`: prediction aggregation, condition-weighted scores, figures and this report.
- `test_followup.py`: group separation, full outer-test coverage, seed-independent splits and invalid smoothness-column checks. Tracking test follows the declared version.

## Experiment design

Inputs: concentration (natural log), measured uncoated thickness, measured total thickness, and three solvent indicators. Target: bonded thickness in nm. All scaling uses training rows only. This is conditional film-state inference, not prediction from a coating recipe alone. The six-input network has two 16-unit tanh layers. Fixed settings: at most 1,500 epochs, validation patience 150, learning rate 0.001, weight decay 0.0001, curvature coefficient 0.001; no tuning on outer-test results.

Five models share each split: bounded network with smoothness (PINN), same bounded network without the penalty, unconstrained network, 300-tree random forest and 100-stage gradient boosting. PINN versus bounded NN isolates the added curvature prior. Physical compliance is reported separately from accuracy. Smoothness applies only to continuous concentration/uncoated/total inputs, not solvent indicators. Velocity, LLD and monotonicity weights remain zero.

Five outer folds each hold out entire solvent/concentration conditions. Every record receives one out-of-fold prediction per seed. Initialization seeds 11, 42 and 73 use the same outer and inner splits: 15 runs. Nine additional runs use 25%, 50% and 75% of fold-1 training conditions at each seed; the full-data fold-1 runs supply the 100% endpoint. Ceil rounding and a minimum two conditions determine the actual subsets; exact membership is in splits.csv. Validation and test membership are unchanged across fractions. No synthetic parent can cross a split because no synthetic records are used.

## Results

Primary error gives each condition equal weight: average squared error within each condition, average across conditions, then square root. Mean and SD below summarize three initialization seeds on fixed folds, not confidence intervals or independent replications.

| Model | Condition RMSE mean (nm) | Seed SD (nm) | Row RMSE mean (nm) | Bounds violations |
|---|---:|---:|---:|---:|
| Bound + smoothness | 0.1061 | 0.0133 | 0.1022 | 0.0% |
| Bound only | 0.1065 | 0.0138 | 0.1033 | 0.0% |
| Unconstrained NN | 0.1230 | 0.0184 | 0.1234 | 1.3% |
| Random forest | 0.1018 | 0.0009 | 0.1005 | 0.0% |
| Gradient boosting | 0.1097 | 0.0007 | 0.1090 | 0.0% |

Adding smoothness improved condition-weighted fold RMSE in 9 of 15 paired fold/seed comparisons; mean paired difference was -0.0003 nm (negative favors smoothness). These comparisons share data and are not 15 independent experiments. See `paired_curvature_effect.csv` for all differences. Do not equate compliance with accuracy or claim a mechanistic law from these fits.

## Figures and tables

`figures/` contains PNG and vector PDF versions of measured-data coverage, model comparison, seed-42 parity, solvent-resolved errors, paired curvature effects, fixed-test learning curves and all 24 run validation histories. Parity seed 42 was fixed for illustration, not chosen for best results. Machine-readable scores and all source-linked predictions accompany the figures. Solvent breakdowns evaluate represented solvents, not leave-one-solvent-out transfer.

## Remaining dependencies and decisions

1. Match measured withdrawal speed, solution viscosity/surface tension/density, independent wet-to-dry conversion and LLD regime justification to record/experiment IDs using the intake. Record sources and units; do not derive the conversion from bonded target labels. Then run velocity-only, LLD-only and combined ablations on the same splits.
2. Obtain actual batch/specimen IDs. Current solvent/concentration groups are conservative proxies and cannot prove independence between concentrations prepared together.
3. Reserve genuinely new laboratory measurements for a final independent test. This campaign reuses existing data and cannot satisfy that requirement.
4. Adsorption/saturation priors need justified boundary/background definitions and identifiable parameters. No unverified adsorption law was imposed.
5. Synthetic augmentation remains deferred until parent lineage and generator validity are established. The completed measured-only learning curves establish the reference comparison.

## Reproduction

From repository root: `python -m unittest discover -s PINN -p "test*.py" -v`, then `python PINN/run_followup.py` (creates a new timestamped campaign; does not overwrite this one), then `python PINN/summarize_followup.py PINN/campaigns/<new-campaign>`. `python PINN/verify_runs.py` verifies all archived run hashes and scores. Individual runs can be reproduced using their archived config with `PINN/train.py`, the prepared dataset and a new output directory, including `--train-fraction` for reduced-data runs.

The local Windows test environment required TEMP/TMP to point to a writable temporary directory; the initial two temp-file failures were environmental; the rerun passed all 12 tests. See `tests.log` and `verification.log` for final validation.

## Complete run inventory

Each linked folder contains results for all five models; error below is row-weighted PINN RMSE for that individual test fold, not the pooled condition-weighted summary above.

| Run | Fold | Seed | Training fraction | Training rows | Training conditions | PINN RMSE (nm) |
|---|---:|---:|---:|---:|---:|---:|
| [20261111T134229Z_grouped_cv_fold1_seed11_fraction1](../../runs/20261111T134229Z_grouped_cv_fold1_seed11_fraction1/run_record.json) | 1 | 11 | 100% | 75 | 23 | 0.0860 |
| [20261111T134229Z_grouped_cv_fold2_seed11_fraction1](../../runs/20261111T134229Z_grouped_cv_fold2_seed11_fraction1/run_record.json) | 2 | 11 | 100% | 78 | 23 | 0.0611 |
| [20261111T134229Z_grouped_cv_fold3_seed11_fraction1](../../runs/20261111T134229Z_grouped_cv_fold3_seed11_fraction1/run_record.json) | 3 | 11 | 100% | 77 | 22 | 0.0772 |
| [20261111T134229Z_grouped_cv_fold4_seed11_fraction1](../../runs/20261111T134229Z_grouped_cv_fold4_seed11_fraction1/run_record.json) | 4 | 11 | 100% | 79 | 23 | 0.1293 |
| [20261111T134229Z_grouped_cv_fold5_seed11_fraction1](../../runs/20261111T134229Z_grouped_cv_fold5_seed11_fraction1/run_record.json) | 5 | 11 | 100% | 79 | 23 | 0.0955 |
| [20261111T134229Z_learning_curve_fold1_seed11_fraction0.25](../../runs/20261111T134229Z_learning_curve_fold1_seed11_fraction0.25/run_record.json) | 1 | 11 | 25% | 17 | 6 | 0.3122 |
| [20261111T134229Z_learning_curve_fold1_seed11_fraction0.5](../../runs/20261111T134229Z_learning_curve_fold1_seed11_fraction0.5/run_record.json) | 1 | 11 | 50% | 41 | 12 | 0.1302 |
| [20261111T134229Z_learning_curve_fold1_seed11_fraction0.75](../../runs/20261111T134229Z_learning_curve_fold1_seed11_fraction0.75/run_record.json) | 1 | 11 | 75% | 60 | 18 | 0.0898 |
| [20261111T134229Z_grouped_cv_fold1_seed42_fraction1](../../runs/20261111T134229Z_grouped_cv_fold1_seed42_fraction1/run_record.json) | 1 | 42 | 100% | 75 | 23 | 0.0875 |
| [20261111T134229Z_grouped_cv_fold2_seed42_fraction1](../../runs/20261111T134229Z_grouped_cv_fold2_seed42_fraction1/run_record.json) | 2 | 42 | 100% | 78 | 23 | 0.1402 |
| [20261111T134229Z_grouped_cv_fold3_seed42_fraction1](../../runs/20261111T134229Z_grouped_cv_fold3_seed42_fraction1/run_record.json) | 3 | 42 | 100% | 77 | 22 | 0.1070 |
| [20261111T134229Z_grouped_cv_fold4_seed42_fraction1](../../runs/20261111T134229Z_grouped_cv_fold4_seed42_fraction1/run_record.json) | 4 | 42 | 100% | 79 | 23 | 0.1109 |
| [20261111T134229Z_grouped_cv_fold5_seed42_fraction1](../../runs/20261111T134229Z_grouped_cv_fold5_seed42_fraction1/run_record.json) | 5 | 42 | 100% | 79 | 23 | 0.1171 |
| [20261111T134229Z_learning_curve_fold1_seed42_fraction0.25](../../runs/20261111T134229Z_learning_curve_fold1_seed42_fraction0.25/run_record.json) | 1 | 42 | 25% | 17 | 6 | 0.4200 |
| [20261111T134229Z_learning_curve_fold1_seed42_fraction0.5](../../runs/20261111T134229Z_learning_curve_fold1_seed42_fraction0.5/run_record.json) | 1 | 42 | 50% | 41 | 12 | 0.1275 |
| [20261111T134229Z_learning_curve_fold1_seed42_fraction0.75](../../runs/20261111T134229Z_learning_curve_fold1_seed42_fraction0.75/run_record.json) | 1 | 42 | 75% | 60 | 18 | 0.0862 |
| [20261111T134229Z_grouped_cv_fold1_seed73_fraction1](../../runs/20261111T134229Z_grouped_cv_fold1_seed73_fraction1/run_record.json) | 1 | 73 | 100% | 75 | 23 | 0.0985 |
| [20261111T134229Z_grouped_cv_fold2_seed73_fraction1](../../runs/20261111T134229Z_grouped_cv_fold2_seed73_fraction1/run_record.json) | 2 | 73 | 100% | 78 | 23 | 0.0985 |
| [20261111T134229Z_grouped_cv_fold3_seed73_fraction1](../../runs/20261111T134229Z_grouped_cv_fold3_seed73_fraction1/run_record.json) | 3 | 73 | 100% | 77 | 22 | 0.0769 |
| [20261111T134229Z_grouped_cv_fold4_seed73_fraction1](../../runs/20261111T134229Z_grouped_cv_fold4_seed73_fraction1/run_record.json) | 4 | 73 | 100% | 79 | 23 | 0.1195 |
| [20261111T134229Z_grouped_cv_fold5_seed73_fraction1](../../runs/20261111T134229Z_grouped_cv_fold5_seed73_fraction1/run_record.json) | 5 | 73 | 100% | 79 | 23 | 0.1075 |
| [20261111T134229Z_learning_curve_fold1_seed73_fraction0.25](../../runs/20261111T134229Z_learning_curve_fold1_seed73_fraction0.25/run_record.json) | 1 | 73 | 25% | 17 | 6 | 0.3123 |
| [20261111T134229Z_learning_curve_fold1_seed73_fraction0.5](../../runs/20261111T134229Z_learning_curve_fold1_seed73_fraction0.5/run_record.json) | 1 | 73 | 50% | 41 | 12 | 0.1304 |
| [20261111T134229Z_learning_curve_fold1_seed73_fraction0.75](../../runs/20261111T134229Z_learning_curve_fold1_seed73_fraction0.75/run_record.json) | 1 | 73 | 75% | 60 | 18 | 0.1071 |
