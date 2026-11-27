# Stage 3 handoff — analysis complete, commit blocked

Completed November 27, 2026. Read [REPORT.md](REPORT.md), [search_space.json](search_space.json), [Stage 2 handoff](../stage_02/HANDOFF.md), then Stage 4 of the unchanged original roadmap. Only Stages 1–3 were executed; no Stage 4 neural/physics implementation has started.

## Authoritative artifacts

- Regenerated baseline: `publication_2026/runs/20261127T154337Z_AB_stage02_regenerated/`.
- Stage 3: `publication_2026/runs/20261127T154651Z_A_stage03_nested/`.
- `search_space.json` freezes the 12 RF / 8 GB settings per target and the four simple mobile candidates (including exactly one hybrid). The same bytes were saved before fitting in the run; SHA256 is in `started.json` and `run_record.json`.
- `splits.csv` and `inner_splits.csv` are copied from the regenerated baseline, preserving historical grouped-fold assignments and numeric endpoint holdouts. Read them by stable record ID. Roles train + validation form Stage 3 development; all inner fitting is restricted to its saved train memberships.
- `inner_scores.csv`, `inner_predictions.csv`, `selected_configs_by_fold.json`, `selection_frequency.csv` retain every candidate's fold scores, inner predictions and choices. Candidate labels encode descriptor/identity features and raw/log transforms. No outer labels choose configurations.
- `fits.json` records all 1,320 inner and 150 distinct outer fits: exact training IDs, weights, configurations, seeds, learned category lists, scaler parameters, local fit memberships and analytical/residual parameters. Different reported procedures may legitimately reuse one selected fit. Predictions are sufficient to audit the results; no unnecessary binary estimator serialization was added.
- `predictions.csv` contains final outer predictions for seeds 11/42/73, selected configuration, unseen-solvent flag, raw/projected components and paired total. Total rows have combined fit IDs; their configuration/encoding fields are intentionally blank because the two heads may differ.
- `metrics.csv`, `seed_summary.csv`, `comparison_metrics.csv`, `paired_condition_differences.csv` are the scientific outputs. All metrics are reconstructed by tests. Uncertainty intervals are not supplied here.
- `training_diagnostics.csv` contains outer-training-only concentration associations and repeated-condition spread. `inputs.json`, `feature_contract.json`, `environment.json`, `run_record.json` freeze provenance and compute details.
- This stage directory contains the report, per-solvent engineering-threshold disclosures, figures in PNG/PDF, execution/test logs and report manifest. Figures were visually inspected for legibility.

## Conclusions and retained procedures

1. Nested tree selection is a null result. Grouped-CV mobile error rises from fixed balanced RF 0.90534 nm to tree selector 0.98902 nm; total rises from 0.88937 to 0.96961. Do not launch another grid or promote the selected tree as an accuracy gain.
2. The analytical selector (`simple_selected`) has mobile RMSE 0.32211 nm in grouped CV and 2.74081 nm at high concentrations, but 12.12920 nm for unseen-solvent transfer. Fixed local linear is better on grouped CV (0.22898), but much worse at high concentrations (12.48907). There is no universal winner.
3. Carry **fixed balanced pooled RF** and **simple_selected** as at most two Task A procedures for later uncertainty evaluation. Returning to the fixed RF applies the prespecified simpler-model stop rule after unstable gains; it does not rewrite the evaluated search. Simple-selected remains an exploratory known-solvent method with explicit transfer failure. Preserve its bonded head and mobile-selection rule exactly if reusing its totals.
4. For Task B retain historical **RF** and **bounded NN** as two references; the later Stage 4 may compare bounded-plus-curvature against these. Stage 3 did not retrain conditional networks or establish a PINN advantage.
5. Do not select a different model for each solvent after looking at outer scores. No frozen outer set is newly untouched; claims remain development evidence. The >5% improvement / >10% degradation rules are engineering thresholds, not significance tests. Every target/solvent degradation is retained in `decision_thresholds.csv`.
6. The updated physics columns remain assumptions, not additional measurements. Keep LLD/velocity/viscosity-monotonicity losses off. Unknown sample/batch IDs remain a limitation. No synthetic training or validation was introduced.

## Reproduction

Use Python 3.12 and the exact package versions in the run environment (this machine already had them; no packages were upgraded). From repository root:

```powershell
python -m unittest discover -s publication_2026/tests -v
python -m publication_2026.report_stage3
# Optional only: creates a NEW timestamped search run; do not rerun for a status check.
python -m publication_2026.stage3
```

Tests use `stage_02/latest_run.txt` and `stage_03/latest_run.txt`; this handoff pins the completed runs above. Full reproduction creates new output directories and updates only the relevant latest-run pointer. Any input or source change must be recorded as a new run; do not silently edit a frozen result. The report generator derives figures/tables from saved results and does not refit models.

## Verification and compute

36 tests pass: 12 PINN, 13 near-term, 5 regenerated-baseline integrity, 6 nested-search integrity. Checks include complete prediction coverage, inner-only selection, unknown-category treatment, no measured-thickness features, analytical/residual isolation, training-only scaling, actual fit IDs/weights, independent metric and paired-difference reconstruction, all artifact/source/input hashes and preservation of every original tracked file. All 29 historical run records and 24 campaign runs verify separately.

The nested run used one CPU worker for RF, search seed 42 and three final seeds. Measured runtime: 97.08 seconds, 1,320 inner fits plus 150 outer fits. The regenerated Stage 2 took 19.13 seconds. These exclude environment checks, tests and writing. The roadmap budget was 10 human hours; human labor time was not recorded. No GPU was required.

## Commit status and remaining deadline

All new files are under `publication_2026/` in the user's existing repository. Original tracked files remain byte-identical. The earlier cloud-only commit `66c67d5` was not available, so it is neither cherry-picked nor claimed recovered. This session regenerated its prerequisite scientific work with new provenance and completed Stage 3.

**No new Git commit or push succeeded.** `git add`/`git commit` failed because `.git/index.lock` cannot be created: the environment grants only read access to `.git`. A permission-escalation attempt was rejected because sandbox approvals are disabled. Do not work around this by redirecting Git metadata or pushing through another service. Once using a terminal/environment permitted to write this repository's Git metadata, review and commit the additive directory:

```powershell
git status --short
git add publication_2026
git diff --cached --stat
git commit -m "Complete publication stage 3 with regenerated audited baselines"
```

No public push is included in those commands. The earlier public-disclosure approval block and this local metadata permission restriction are different issues.

Only one week remains until December 4 at the current date. Next authorized work should use this handoff to bound Stage 4, reuse existing neural runs, and preserve time for Stage 5 validation and the figure/manuscript package. Do not spend the remaining week expanding the null tree search. A valid null result satisfies Stage 3; publication readiness is not yet established.
