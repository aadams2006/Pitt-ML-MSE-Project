# Near-term experimental validation

This is the current validation entry point. Earlier scripts and published result
directories are historical, exploratory workflows; they retain legacy units,
synthetic dependence, and measured-thickness features. They must not be used as
evidence of independent process forecasting. Raw files are preserved unchanged.

## Reproduce

From the repository root, using Python 3.12:

```sh
python -m pip install -r near_term/requirements.txt
python near_term/run_analysis.py
python -m unittest discover -s near_term/tests -v
python near_term/build_report.py
```

The manuscript PDF can be rebuilt with Tectonic 0.17.0 from the repository root:

```sh
tectonic research_paper.tex --outdir near_term/results --keep-logs
```

Tectonic downloads its standard TeX bundle on first use. The committed PDF is
`near_term/results/research_paper.pdf`; build and test logs accompany it.

The analysis takes several minutes on a CPU. Results go to `near_term/results/`.
`--primary-only` omits the summary-inclusion sensitivity study. The manifest records
input hashes, source commit, runner hash, versions, seed, and estimator settings.
The source commit is the checkout at execution, before committing generated results.
No saved legacy estimator is loaded. No network is required for the analysis.

## Data and units

The three legacy tables contain 142 rows. The canonical data use
`concentration_g_l` with the original numeric values, and an explicit
`concentration_g_ml = concentration_g_l / 1000`. This is a schema correction,
not a 1000-fold change in the source experiments. Exact workbook unit-evidence
cells and file hashes are recorded in `workbook_inventory.json`. The descriptor
constants are inherited from the three existing analytical comparison runners;
they are fixed solvent approximations, not new measurements of solutions.

Every canonical row keeps its file path, original row number, solvent, and stable
record ID. One toluene record has total 0.85 nm and bonded 0.86 nm. It is
quarantined for both targets, without clipping or asserting a corrected value.
The first 15 toluene CSV records reproduce the workbook's Sheet2 summaries.
They may overlap later readings and have inconsistent uncoated values. All 15
are excluded from the primary analysis. This conservatively removes four
concentration conditions without individual readings as well. A separate
sensitivity cohort includes all physically valid records, with identical split
rules. It is not an independent replication of the primary study.

The primary cohort has 126 rows. The experimental unit and batch IDs are absent
from the legacy tables. `solvent + concentration` is therefore a conservative
proxy group: all readings at that condition stay together. It cannot establish
independence between different concentrations prepared in the same batch.
These are existing experimental records, not newly collected experiments.

The legacy synthetic audit flags 192 rows in total: 113 have negative mobile
differences, 182 have negative concentrations, and six have negative thicknesses.
These categories overlap and must not be added. Their source rows and values are
saved in `legacy_synthetic_exclusions.csv`.

## Validation design

- Five-fold GroupKFold estimates transfer to unobserved concentration conditions
  among represented solvents. Conditions at the extremes can also occur in a fold.
- Leave-one-solvent-out fits on two solvents and tests on the third. Each solvent's
  scores are in `fold_metrics.csv`; pooled R2 can hide failures on individual solvents.
- Low/high OOD splits hold out the lowest/highest ceil(20%) of distinct concentrations
  within each solvent. Boundaries depend only on concentration, never on outcomes.
- All split assignments, individual predictions and fold metrics are committed.
  No synthetic row, generator, scaling fit, or calibration label crosses a split.

Point predictions use the complete outer training partition. Fixed, untuned models
are a training-mean baseline, Ridge (alpha 10), and a 200-tree random forest (leaf
minimum 2). Both learned models are compared with raw features and with
log1p(concentration) plus its interactions with the four solvent descriptors.
StandardScaler fits inside each Ridge pipeline on training data only. Predictions
are projected to nonnegative thickness for every model. Bonded and mobile are
modeled separately; predicted total can be reconstructed as their nonnegative sum.
No total, bonded, mobile, or uncoated thickness is a predictor. Log1p uses the
dimensionless numerical ratio concentration/(1 g/L). Transform choices are
exploratory, fixed before execution; no winning model is selected on test scores
for a deployment claim. There is no tuning and hence no inner hyperparameter search.

The primary comparison is group-weighted RMSE: mean squared error per condition,
then mean across conditions and square root. Row-weighted RMSE, MAE and R2 are
also saved. Descriptive 95% percentile ranges resample condition errors 2000 times
with a fixed seed; they condition on fitted folds and do not capture model training,
batch, or solvent-population uncertainty. In particular, three solvents are not
enough to estimate reliable uncertainty for generalization to arbitrary solvents.

## Prediction uncertainty

An additional estimator uses only a proper-training subset; 35% of outer-training
groups are reserved for calibration. The calibration score is each group's maximum
absolute residual. The finite-sample 80% split-conformal radius is order statistic
ceil((G+1)*0.8); an unavailable finite order statistic gives infinity, never an
optimistically capped quantile. Bounds are centered on this separate estimator,
stored as `interval_prediction`, not the full-training `prediction`. Nonnegative
lower bounds preserve coverage for physically nonnegative targets.

Report both row coverage and simultaneous coverage of all readings in each test
condition, together with interval width. Nominal coverage requires exchangeable
groups. Batch confounding and deliberately shifted OOD/solvent tests violate or
question that assumption, so their observed coverage is a stress test, not a guarantee.

## Physical synthetic generation

`constrained_bootstrap` resamples only supplied training records, preserves parent
record and condition IDs, perturbs nonnegative bonded/mobile components, and
reconstructs total as their sum. Its 300-row saved audit draws only from the first
grouped fold's proper-training partition. It is not a new experiment and is never
used in model validation. The legacy 3000-row synthetic dataset is audited but
not overwritten. This constraint ensures admissibility, not synthetic realism.

## Work remaining outside computation

See `EXTERNAL_EXPERIMENT_PROTOCOL.md`. No independently collected, previously
unseen dataset has been supplied. Source workbook sheets contain reused solutions,
repeated washes, copied summaries and incomplete pairing, so they cannot be silently
promoted to new independent experiments. The paper is updated as a research draft;
external validation and scientific review remain necessary for final claims.
