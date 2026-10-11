# Why separate experimental and synthetic runs?

The initial experiment used the 58 measured rows to assess the model against observations. The existing 3,000-row source is now explicitly included in a separate seven-input benchmark. Synthetic rows can help optimization and test architecture behavior, but they do not supply 3,000 independent laboratory experiments.

Repository scripts generate synthetic rows with bootstrap/correlated-noise methods and solvent-level constants; however, the exact combined 3k file's full generating history and parent-row IDs are not available. We do not claim to reconstruct its provenance or prove independent splits. If source experiments contributed to generation before splitting, synthetic siblings or the generator's fitted statistics may leak between partitions. Grouping by exact concentration reduces some repetition but cannot recover lost parent IDs, especially for jittered concentration values. These scores measure synthetic-distribution behavior, not independent experimental generalization.

## Source audit and explicit filtering

`datasets/synthetic_3000.csv` is a byte-for-byte snapshot of the original combined source. Column whitespace is normalized only while loading. It includes all seven predictors: polarity, viscosity, boiling point, surface tension, concentration, uncoated layer and total thickness, with bonded thickness as target. Added solvent properties can distinguish solvent-associated patterns; their independent causal effects remain unidentifiable from three solvent-property combinations.

- Input rows: 3,000.
- Nonpositive concentration: 182.
- Nonpositive total thickness: 6.
- Bonded greater than total: 113.
- Unique invalid rows across overlapping checks: 192.
- Retained rows: 2,808.

Use `--data-kind synthetic --invalid-policy exclude` to request this policy. Each run saves `data_audit.json` with overlapping reason counts and original excluded row IDs; `splits.csv` and predictions retain original source IDs. No negative concentration is logged, no target is clipped, and the original dataset is preserved. The larger exclusion count differs from the repository's mobile-layer RF workflow (113), because this PINN also requires positive concentration for its log transform.

Separate default and concentration-monotonicity configurations hold architecture, seed, split and other weights fixed. With seven features the two-16-layer MLP has 417 trainable parameters. The second run activates a local positive concentration penalty at weight .01 as an exploratory ablation. Viscosity monotonicity, velocity smoothness and LLD remain inactive for the evidentiary reasons in `PHYSICS_CONSTRAINTS.md`.

Full dated results and exact source/config/data identities are in `RUN_LOG.md` and each run's record, not substituted for measured-data scores. A defensible augmentation study would first split independent measured experiments, generate synthetic rows only from training parents, and evaluate exclusively on untouched measured validation/test experiments. The precomputed 3k file cannot establish this protocol retrospectively.

## Recorded v0.2.0 results

Both synthetic runs use seed 42, 1,959 training / 407 validation / 442 test retained rows, identical splits and a maximum of 1,500 epochs. All three neural variants reached this maximum; performance may still depend on optimization budget, which should be studied using validation data.

| Model | Synthetic default RMSE (nm) | Synthetic default R2 | Concentration-monotonic ablation RMSE (nm) |
|---|---:|---:|---:|
| PINN | 0.055718 | 0.905199 | 0.056009 |
| Bounded NN | 0.055137 | 0.907167 | 0.055137 |
| Standard NN | 0.054133 | 0.910518 | 0.054133 |
| Random forest | 0.035903 | 0.960637 | 0.035903 |
| Gradient boosting | 0.074783 | 0.829227 | 0.074783 |

All observed test-bound violation fractions were zero. The concentration penalty slightly worsened test RMSE in this one ablation. It reduced the training fraction of negative local concentration derivatives from 43.44% to 40.99%, but clearly did not establish global monotonicity. In the viscosity=.377 cohort the training Spearman association is only .100; this cautions against imposing a uniform positive trend across solvent regimes. Do not tune the monotonicity weight from these test scores or declare improvement.

The v0.2 measured-data rerun retained all 58 rows, reproduced the earlier grouped results, and again favored the tree models. Ten unit checks passed; checkpoint inference reproduced the synthetic holdout predictions, and the raw invalid table was correctly rejected for inference. `python PINN/verify_runs.py` verified all five archived/current runs' source/artifact/dataset identities where available, reported scores, and split/prediction alignment. The archive is scientific provenance, not a claim of independent generator provenance.
