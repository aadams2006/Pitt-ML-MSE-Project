# Stage 2: regenerated benchmark handoff

November 27, 2026. Authoritative run is in `latest_run.txt`. This is a regenerated Windows benchmark because the cloud-only commit was unavailable. All previously reported grouped-CV mobile scores reproduce to displayed precision: local linear 0.228981 nm, local saturation 0.416295, local scaled LLD 0.364382, balanced pooled RF 0.905344, balanced local RF 0.879300, reused historical RF 0.944757. Keep the high-concentration and solvent-transfer failures visible; good CV does not imply reliable extrapolation.

The run contains the data audit, exclusions, updated measured snapshot, complete outer and inner split IDs, 14 Task A baselines, the five archived three-seed Task B comparisons plus a training-only constant-fraction baseline, prediction-level raw/projected values, paired component totals, fit IDs/training membership/weights, configuration and environment manifests, and all per-solvent metrics. Historical RF and Task B predictions are reused exactly. Other cheap baselines were refit without replacing any historical result.

A uses full outer-development records and condition-balanced weights (or analytical condition means); B uses the archived train subset, with validation held aside for neural checkpoints and the same subset for trees and the constant-fraction reference. Local models require two local training conditions; otherwise use pooled training. No feature/parameter search was performed in this stage. Fixed model settings are in `benchmark_spec.json` and fitted records in `fits.json`.

The linear mobile reference performs well on CV but high-concentration error is about 12.4891 nm; saturation is about 2.7408 nm there. Thus preserve both interpolation and stress-test evidence. Task B RF remains about 0.101755 nm versus PINN 0.106119; the proportional baseline is about 0.509375. Do not infer a PINN accuracy advantage.

Start Stage 3 from the saved splits and raw measured data; do not rerun the old neural campaign. Search choices must use inner grouped validation only. Save every candidate score, parameter choice, final seed, fit membership and paired total. Keep no more than two finalist procedures. New physical assumptions stay off as constraints. No new calibrated intervals are supplied in Stage 2.

Commands: `python -m publication_2026.recover_stage12` creates an additive run; `python -m unittest discover -s publication_2026/tests -v` verifies available stage artifacts. Python/package versions and CPU time are in the run. The regenerated run took about 19 seconds; human-hours are not recorded.
