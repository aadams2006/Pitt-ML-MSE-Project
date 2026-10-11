# Physics-informed bonded-thickness model

All PINN-specific implementation, configuration, literature, checks, and generated runs live here. Existing model pipelines and datasets are unchanged.

This initial model is **physics-constrained tabular regression**, using a hard thickness bound and a smoothness prior. It is not yet a governing-PDE PINN or an identified adsorption mechanism. The proposal and attached slide motivate the experiment; they do not establish an applicable differential equation.

## Run from the repository root

```powershell
python -m pip install -r PINN/requirements.txt
python -m unittest discover -s PINN -p 'test_*.py' -v
python PINN/verify_runs.py
python PINN/train.py
python PINN/train.py --split extrapolation
python PINN/train.py --data PINN/datasets/synthetic_3000.csv --data-kind synthetic --invalid-policy exclude --config PINN/config_synthetic.json
python PINN/train.py --data PINN/datasets/synthetic_3000.csv --data-kind synthetic --invalid-policy exclude --config PINN/config_synthetic_monotonic.json
python PINN/predict.py --checkpoint PINN/runs/YOUR_RUN/pinn.pt --data PINN/datasets/experimental_58.csv --output PINN/runs/YOUR_RUN/inference.csv
```

Each invocation creates a timestamped directory under `PINN/runs/` automatically. For a named run use `--output PINN/runs/my_run`; directories must be new. Defaults are anchored to this folder; custom relative paths resolve from your current directory. CPU PyTorch is sufficient. Dependencies declare supported ranges; runs record exact versions rather than treating the ranges as a lock.

The default input is the committed `PINN/datasets/experimental_58.csv`, exported from the measured workbook: 58 rows, 16 concentration levels, three inputs and one target. The committed `synthetic_3000.csv` preserves the complete eight-column synthetic source, including defects, and has a separate seven-input configuration. `datasets/provenance.json` records source and snapshot hashes. A fresh clone can reproduce both experiments without the root's ignored data folder. CSV/XLSX are supported; missing/nonfinite values fail. Physical defects fail by default; explicit synthetic-only exclusion records original row IDs and counts. Inputs/targets are never silently clipped or imputed.

## Outputs

Each run writes metrics (MAE/RMSE in nm, R2, physical violation rate/magnitude), shared test predictions, original source-row split IDs, training histories, three neural checkpoints, inspectable tree structures/parameters, raw-unit PINN sensitivities, training-only trend diagnostics, an exclusion audit, and a manifest. `run_record.json` adds UTC/New York times, semantic model version, Git revision/dirty state, source hash, per-model start/end/selected epoch/results, and artifact hashes. `source/` archives exact runnable code plus effective configuration. Failed runs also get records. **Runs and checkpoints under `runs/` are tracked in Git**, alongside the human-readable `RUN_LOG.md`. Every future run is locally recorded automatically; commit/push it to publish its record.

The two original runs are imported with source version 0.1.0 and their actual saved results. Start/end times were not previously captured, so these fields are null; artifact modification timestamps are labeled approximate. Version 0.2.0 captures true execution timestamps prospectively. Their exact original workbook is represented by the committed CSV export; the original workbook hash is preserved for provenance.

`verify_runs.py` checks archived sources/artifacts, available dataset hashes, score agreement and split/prediction IDs. `.gitattributes` preserves exact bytes so SHA256 records remain valid after a clone. New runs during development record both the Git base and archived source hash; a source snapshot is authoritative when the worktree was dirty. The first three v0.2 records annotate a corrected Git-status path-scope reporting error; scientific code snapshots and results remain unchanged.

Five models share the same train/validation/test assignment: regularized bounded PINN, bounded NN without physical regularization, standard unconstrained NN, random forest, and gradient boosting. Preprocessing is fitted only on training rows; validation selects neural checkpoints; test rows are used only for final reporting. Baseline predictions remain unclipped so violations remain visible. The saved checkpoints are the validation-selected research models trained on the training partition, not full-dataset deployment fits.

`--split grouped` holds out concentration levels (default). `--split extrapolation` holds out the largest 20% of concentration levels. `--split random` is an exploratory interpolation comparison with greater replicate leakage risk. If batch/specimen identifiers become available, replace the concentration group with the appropriate experimental unit. Concentration is a provisional grouping proxy, not proof of independent batches.

For data efficiency, repeat the same seed/split with `--train-fraction 0.25`, `0.5`, `0.75`, and `1.0`, each in a new output directory. Validation/test partitions remain fixed; only training rows are subsampled. These are row-level learning curves, not independent experiment-level curves. Use multiple seeds (`--seed`) before drawing conclusions.

Read `DESIGN_DECISIONS.md` for the initial rationale, `PHYSICS_CONSTRAINTS.md` for all four slide constraints, `SYNTHETIC_BENCHMARK.md` for synthetic interpretation, `LITERATURE_REVIEW.md` for architecture comparisons, `RUN_LOG.md` for individual runs, and `CHANGELOG.md` for changes.
