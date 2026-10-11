# Physics-informed bonded-thickness model

All PINN-specific implementation, configuration, literature, checks, and generated runs live here. Existing model pipelines and datasets are unchanged.

This initial model is **physics-constrained tabular regression**, using a hard thickness bound and a smoothness prior. It is not yet a governing-PDE PINN or an identified adsorption mechanism. The proposal and attached slide motivate the experiment; they do not establish an applicable differential equation.

## Run from the repository root

```powershell
python -m pip install -r PINN/requirements.txt
python -m unittest discover -s PINN -p test_pinn.py -v
python PINN/train.py --output PINN/results/my_grouped_run
python PINN/train.py --split extrapolation --output PINN/results/my_extrapolation_run
python PINN/predict.py --checkpoint PINN/results/my_grouped_run/pinn.pt --data data/agg.data.xlsx --output PINN/results/my_grouped_run/inference.csv
```

Use a new output directory for each training run. Paths for default input/config/output are anchored to this folder, so the default experiment also works from other working directories. Custom relative paths resolve from your current directory. CPU PyTorch is sufficient. The dependency file declares supported ranges; each run records actual versions, rather than claiming these ranges are an exact environment lock.

The default input is the local experimental `data/agg.data.xlsx`: 58 rows, 16 concentration levels, three features and one target. The root repository ignores `data/`, so a fresh clone needs the experimental workbook placed there or supplied with `--data`. No synthetic holdout files are used by default. CSV and XLSX are supported; missing columns/values and physically inconsistent target labels fail explicitly rather than being silently imputed or clipped.

## Outputs

Each run writes metrics (MAE/RMSE in nm, R2, physical violation rate and magnitude), shared test predictions, source row split assignments, training histories, three neural checkpoints, raw-unit PINN sensitivities, and a manifest with configuration, dataset hash, split sizes, and library versions. Generated results/checkpoints are locally ignored; the initial findings are preserved in `INITIAL_RESULTS.md`.

Five models share the same train/validation/test assignment: regularized bounded PINN, bounded NN without physical regularization, standard unconstrained NN, random forest, and gradient boosting. Preprocessing is fitted only on training rows; validation selects neural checkpoints; test rows are used only for final reporting. Baseline predictions remain unclipped so violations remain visible. The saved checkpoints are the validation-selected research models trained on the training partition, not full-dataset deployment fits.

`--split grouped` holds out concentration levels (default). `--split extrapolation` holds out the largest 20% of concentration levels. `--split random` is an exploratory interpolation comparison with greater replicate leakage risk. If batch/specimen identifiers become available, replace the concentration group with the appropriate experimental unit. Concentration is a provisional grouping proxy, not proof of independent batches.

For data efficiency, repeat the same seed/split with `--train-fraction 0.25`, `0.5`, `0.75`, and `1.0`, each in a new output directory. Validation/test partitions remain fixed; only training rows are subsampled. These are row-level learning curves, not independent experiment-level curves. Use multiple seeds (`--seed`) before drawing conclusions.

Read `DESIGN_DECISIONS.md` for the feature and architecture rationale, `LITERATURE_REVIEW.md` for architecture comparisons and sources, and `CHANGELOG.md` for the implemented changes.
