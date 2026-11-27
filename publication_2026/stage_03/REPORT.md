# Stage 3: nested recipe-only model comparison

Completed November 27, 2026. [Run artifacts](../runs/20261127T154651Z_A_stage03_nested/). [Frozen search space](search_space.json). [Handoff](HANDOFF.md).

## Result and decision

**The limited tree search produced a valid null result, not a general accuracy improvement.** The inner-selected RF/GB procedure has grouped-CV bonded/mobile/total condition RMSE of 0.10808 / 0.98902 / 0.96961 nm versus the fixed balanced pooled RF at 0.10482 / 0.90534 / 0.88937 nm. The tree selector worsens grouped-CV mobile error by about 9.24%. Nested RF alone gives only about 0.4% lower bonded error, below the 5% engineering threshold, and does not improve mobile error. Stop the search; do not expand the grid or choose settings after examining outer tests.

The simple/analytical mobile selector reaches 0.32211 nm grouped-CV mobile RMSE and 2.74081 nm high-concentration RMSE. It is better than the RF on those comparisons but worse than the already-available fixed local linear curve on grouped CV (0.22898 nm), and worse on low-concentration error (0.14179 versus 0.13426 nm). Most importantly, its leave-solvent-out error is 12.12920 nm versus 3.15154 nm for fixed RF. It is an exploratory known-solvent procedure, not a transfer solution. No candidate meets a universal improvement/no-degradation claim.

For the next uncertainty stage retain at most two Task A procedures: **fixed balanced pooled RF** (simpler reference retained under the prespecified null-result stop rule) and **simple_selected** (exactly the evaluated inner-selection procedure, with severe transfer limitations). Do not substitute a different curve per solvent based on outer results. The tree-selected route was evaluated as planned but is not promoted. Task B remains unchanged; carry its fixed RF and bounded NN as the two references for later constraint/uncertainty work. No PINN retraining or new loss activation was part of Stage 3.

## Scope and recovery

The original cloud-only Stage 1–2 commit `66c67d5` is unavailable here because its push was blocked and that workspace became unavailable. Stages 1–2 were regenerated from the unchanged source tables and archived predictions in `20261127T154337Z_AB_stage02_regenerated`. Reported Stage 2 scores reproduce to displayed precision. This is a new timestamped Windows run, not a recovered commit. All original tracked files remain byte-identical. Git commits are currently blocked by the environment's read-only `.git` permission; the analysis files are saved locally and remain uncommitted.

## Prespecified method

- Same 126 measured readings / 36 solvent-concentration groups and saved record-ID partitions as Stage 2. No synthetic data and no new observations.
- Five grouped outer folds, all three solvent holdouts and low/high numeric concentration holdouts. Within each development set use three disjoint grouped inner folds. No outer-test outcomes are used for preprocessing, fitting, configuration selection or tie-breaking.
- Exactly 12 RF configurations per target: descriptor versus training-fitted solvent identity encoding, raw versus log1p concentration, and three leaf/depth pairs (1/unlimited, 2/6, 4/3). Exactly 8 GB configurations per target: the same encoding/transform choices and two regularization settings. Every model has 100 trees; this fixed compute choice differs from the 200-tree Stage 2 RF reference and is disclosed rather than called a pure leaf-size ablation.
- Search seed 42. Aggregate all inner OOF reading errors with equal condition weight; do not average fold RMSEs. Ties follow the declared candidate order. Refit selected settings with seeds 11/42/73 on full outer-development records. The same configuration is used for all final seeds; seeds do not trigger a new search.
- `tuned_rf` and `tuned_gb` are separately evaluated family-selection procedures. `tree_selected` chooses between all 20 settings per target inside each outer development set. `simple_selected` uses that same selected bonded tree and selects its mobile predictor among local linear, local scaled LLD, local saturation and one fixed residual hybrid using inner errors only.
- The hybrid first fits local scaled LLD to inner-training condition means, then fits a condition-weighted Ridge residual (alpha 10) on log1p concentration, fixed solvent descriptors and their interactions. Both components are fitted anew inside every inner fold. Residuals of training observations are allowed for training; validation labels never fit either component. Full selected procedures are refit on outer development. No extra residual hyperparameter grid was added.
- Identity categories come only from training solvents; unseen solvents encode as all-zero indicators. These cases are explicitly flagged in predictions. This defines executable behavior, not demonstrated chemical transfer. Descriptor and identity configurations are labeled individually in predictions and selection tables; no causal descriptor interpretation is warranted from three solvents.
- All ML uses inverse-condition-frequency training weights normalized to mean one; analytical curves use condition means. Local fitting requires at least two conditions; otherwise use pooled training. Task A features exclude measured total, uncoated and target-derived fields. Components are projected nonnegative and total computed by adding paired predictions. Raw outputs and violation metrics are retained.

The fixed search space SHA256 is `644d98a4ca540b6d1ee20d29d5ec34ccabc1612367a1573bec67e550c685f3fb`. The run evaluated 1320 inner fits and 150 distinct outer fits in 97.08 seconds on CPU. No GPU, dependency upgrade or full neural rerun was needed. Extra stress protocols use the same frozen search space, not additional settings.

## Grouped-CV comparison

Condition RMSE (nm), mean over final seeds for new procedures; fixed references have the original single seed or are deterministic. Missing component/total entries mean a curve only predicts mobile. Seed SD appears in `seed_summary.csv`; it is training variation, not a confidence interval.

| procedure | bonded_nm | mobile_nm | total_nm |
| --- | --- | --- | --- |
| linear_local | 0.14263 | 0.22898 | 0.20663 |
| rf_balanced_local | 0.10910 | 0.87930 | 0.87894 |
| rf_balanced_pooled | 0.10482 | 0.90534 | 0.88937 |
| ridge_balanced | 0.16416 | 2.54694 | 2.55796 |
| saturating_mobile_local | nan | 0.41629 | nan |
| scaled_lld_local | nan | 0.36438 | nan |
| simple_selected | 0.10808 | 0.32211 | 0.30725 |
| tree_selected | 0.10808 | 0.98902 | 0.96961 |
| tuned_gb | 0.10542 | 0.95290 | 0.93609 |
| tuned_rf | 0.10440 | 0.92548 | 0.90678 |

## Mobile stress tests

All errors are nm. Each protocol uses its own held-out conditions. Do not compare their difficulty as if sample support were identical.

| procedure | grouped_cv | leave_one_solvent_out | ood_high_concentration | ood_low_concentration |
| --- | --- | --- | --- | --- |
| linear_local | 0.22898 | 9.23129 | 12.48907 | 0.13426 |
| rf_balanced_local | 0.87930 | 3.15154 | 4.02121 | 0.17122 |
| rf_balanced_pooled | 0.90534 | 3.15154 | 4.22412 | 0.15412 |
| ridge_balanced | 2.54694 | 7.19769 | 14.72041 | 2.19486 |
| saturating_mobile_local | 0.41629 | 5.32914 | 2.74081 | 0.24271 |
| scaled_lld_local | 0.36438 | 12.02479 | 33.12921 | 0.28688 |
| simple_selected | 0.32211 | 12.12920 | 2.74081 | 0.14179 |
| tree_selected | 0.98902 | 2.83301 | 4.06003 | 0.15202 |
| tuned_gb | 0.95290 | 2.84227 | 4.06003 | 0.31015 |
| tuned_rf | 0.92548 | 3.12387 | 4.20044 | 0.15202 |

## Solvent-resolved grouped-CV errors

| procedure | target | ethyl_acetate | hexane | toluene |
| --- | --- | --- | --- | --- |
| rf_balanced_pooled | bonded_nm | 0.13439 | 0.08285 | 0.10582 |
| rf_balanced_pooled | mobile_nm | 0.39074 | 1.30024 | 0.31383 |
| rf_balanced_pooled | total_nm | 0.32533 | 1.28263 | 0.33029 |
| simple_selected | bonded_nm | 0.14826 | 0.08819 | 0.09447 |
| simple_selected | mobile_nm | 0.51746 | 0.27532 | 0.10113 |
| simple_selected | total_nm | 0.47519 | 0.27265 | 0.12678 |
| tree_selected | bonded_nm | 0.14826 | 0.08819 | 0.09447 |
| tree_selected | mobile_nm | 0.35815 | 1.43458 | 0.31884 |
| tree_selected | total_nm | 0.31910 | 1.40671 | 0.33777 |

## Inner choices and stability

These are selections before each corresponding outer test is evaluated. The full RF/GB/simpler selection frequency is in `selection_frequency.csv`; all candidates' three inner-fold scores and OOF predictions are retained.

| fold | target | tree_selected | simple_selected |
| --- | --- | --- | --- |
| 1 | bonded_nm | rf_descriptors_log_leaf4_depth3 | rf_descriptors_log_leaf4_depth3 |
| 1 | mobile_nm | rf_descriptors_raw_leaf1_depthNone | linear_local |
| 2 | bonded_nm | rf_identity_log_leaf1_depthNone | rf_identity_log_leaf1_depthNone |
| 2 | mobile_nm | rf_descriptors_log_leaf1_depthNone | scaled_lld_plus_ridge_residual |
| 3 | bonded_nm | gb_descriptors_log_leaf2_depth2 | gb_descriptors_log_leaf2_depth2 |
| 3 | mobile_nm | rf_descriptors_raw_leaf1_depthNone | saturating_mobile_local |
| 4 | bonded_nm | rf_descriptors_raw_leaf4_depth3 | rf_descriptors_raw_leaf4_depth3 |
| 4 | mobile_nm | rf_descriptors_log_leaf1_depthNone | saturating_mobile_local |
| 5 | bonded_nm | rf_identity_log_leaf1_depthNone | rf_identity_log_leaf1_depthNone |
| 5 | mobile_nm | gb_descriptors_raw_leaf2_depth2 | saturating_mobile_local |

For mobile grouped CV, the selector chooses saturation in three folds, linear in one and the residual hybrid in one. It chooses the hybrid also for the toluene solvent holdout and low-concentration holdout. The hybrid is not a uniform improvement, and its inclusion does not validate the LLD mechanism. Across all stored fits, 14 parameter-bound events occur in analytical local/fallback components (some are shared/repeated across seeds). Inspect `fits.json` for context before interpreting amplitudes/scales. They are phenomenological fitting parameters.

`training_diagnostics.csv` records per-solvent, per-target raw/log concentration association and repeated-condition spread using only each outer development set. It identifies singletons explicitly, records their unavailable within-condition SD as zero only for the descriptive summary, and does not estimate causal solvent effects. These diagnostics did not enlarge the frozen search or justify physics constraints.

## Engineering thresholds and paired evidence

Changes below are percentages of mean condition RMSE relative to the fixed balanced pooled RF; positive means worse. The full table includes every solvent and all declared fixed comparators in `decision_thresholds.csv`. The >10% flag is disclosure, not a significance test. Bonded high-concentration error rises about 10.5%; simple-selected solvent-transfer mobile error rises about 285%. No degradation is hidden by the overall grouped score.

| protocol | procedure | target | change_percent | degradation_over_10_percent |
| --- | --- | --- | --- | --- |
| grouped_cv | simple_selected | bonded_nm | 3.10535 | False |
| grouped_cv | simple_selected | mobile_nm | -64.42124 | False |
| grouped_cv | simple_selected | total_nm | -65.45254 | False |
| grouped_cv | tree_selected | bonded_nm | 3.10535 | False |
| grouped_cv | tree_selected | mobile_nm | 9.24226 | False |
| grouped_cv | tree_selected | total_nm | 9.02263 | False |
| leave_one_solvent_out | simple_selected | bonded_nm | -15.05093 | False |
| leave_one_solvent_out | simple_selected | mobile_nm | 284.86537 | True |
| leave_one_solvent_out | simple_selected | total_nm | 292.89395 | True |
| leave_one_solvent_out | tree_selected | bonded_nm | -15.05093 | False |
| leave_one_solvent_out | tree_selected | mobile_nm | -10.10712 | False |
| leave_one_solvent_out | tree_selected | total_nm | -9.73359 | False |
| ood_high_concentration | simple_selected | bonded_nm | 10.51959 | True |
| ood_high_concentration | simple_selected | mobile_nm | -35.11525 | False |
| ood_high_concentration | simple_selected | total_nm | -34.66148 | False |
| ood_high_concentration | tree_selected | bonded_nm | 10.51959 | True |
| ood_high_concentration | tree_selected | mobile_nm | -3.88461 | False |
| ood_high_concentration | tree_selected | total_nm | -3.69316 | False |
| ood_low_concentration | simple_selected | bonded_nm | 8.54545 | False |
| ood_low_concentration | simple_selected | mobile_nm | -7.99737 | False |
| ood_low_concentration | simple_selected | total_nm | 28.98560 | True |
| ood_low_concentration | tree_selected | bonded_nm | 8.54545 | False |
| ood_low_concentration | tree_selected | mobile_nm | -1.36036 | False |
| ood_low_concentration | tree_selected | total_nm | -4.57018 | False |

`paired_condition_differences.csv` contains each condition's new minus fixed-reference MSE for every final seed and applicable reference. Negative means improvement. Differences are paired on identical conditions and measured targets, not independent samples. No uncertainty/significance interval is claimed in this stage; that remains Stage 5. Baseline stochastic models retain their historical seed 42, so comparisons are against that fixed run, not a three-seed baseline distribution.

## Validation and limitations

All 36 tests pass: 25 existing tests plus 11 publication-stage tests. The new checks verify input/artifact hashes, frozen budgets, all inner/outer fit memberships, exact candidate selection from saved inner OOF errors, numeric concentration holdouts, preprocessing fitted on training data, unknown-category behavior, feature poisoning with test outcomes, hybrid fitting isolation, complete OOF coverage, paired totals, independent metrics and paired-difference reconstruction, and historical prediction preservation. All 29 archived run records and the 24-run campaign also verify. Detailed logs are retained.

The physical-property intake still contains estimates, not measurements. No hydrodynamic/velocity/viscosity-monotonicity loss is enabled. Unknown sample/batch IDs and only three solvents limit claims; grouped folds cannot establish batch independence. Outer results have been seen historically and during development, so subsequent finalist decisions are not an untouched-test confirmation. Negative R2 on low-variance subsets remains visible in CSVs. No universal accuracy improvement or publication readiness is claimed.
