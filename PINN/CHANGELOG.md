# Change and decision log

## 0.2.0 - 2026-10-11

- Added committed raw synthetic and experimental CSV snapshots with source hashes; the default measured experiment now works from a fresh clone. Separate seven-feature synthetic and concentration-monotonicity configs retain the measured-data baseline.
- Added explicit synthetic-only invalid-row exclusion and source-row audit. Preserved all 3,000 raw rows; 2,808 pass log-input/physical-label checks. No experimental measurements are automatically excluded.
- Implemented measured-velocity curvature and gated LLD dry-mobile residual with unit conversions, lab-supplied regime/geometry flag, independent wet-to-dry conversion, and fail-fast missing/out-of-regime checks. Both remain off for current data. Documented all four slide constraints and their actual evidence status.
- Added a positive concentration monotonicity ablation and training-only sign/sensitivity diagnostics, including viscosity cohorts. Left viscosity sign unconstrained because solvent confounds cannot be resolved here.
- Limited derivative evaluation to up to 256 training rows per epoch for synthetic scale. Fixed seed preserves deterministic selection; test observations never enter the penalties.
- Added model version, run lifecycle times (UTC/New York), Git/source/data identity, exact archived source/configuration, per-model settings/timings/selected checkpoints/results, exclusion audits, artifact hashes and a run register. Failed runs are recorded too.
- Preserved earlier v0.1 runs with their actual artifacts and archived source; explicitly marked missing execution times rather than reconstructing them from file dates. New `runs/` artifacts, including checkpoints, are version controlled.
- Added checks for LLD units/scaling and regime masking, missing physics, velocity curvature and monotonic gradients, audited exclusion IDs, and failed-run provenance.
- Added artifact/source/data hash verification and Git byte-preservation attributes. Corrected a Git-status path-scope metadata bug and explicitly annotated the affected run records; archived source identities and scientific results were unchanged.
- Prepared a GitHub branch/PR with source, literature, data snapshots and all recorded runs. Publishing is separate from automatically recording future local runs.

## 0.1.0 - 2026-10-11

- Created dedicated `PINN/` folder. Kept all existing model/data workflows intact and isolated the new PyTorch dependency file.
- Added explicit bonded target and three measured inputs; log concentration and training-only scaling replace heuristic column detection. This prevents target leakage and unstable input magnitudes.
- Added two 16-unit tanh layers, sigmoid total-thickness envelope, normalized MSE, curvature regularization, optional local monotonic penalties, deterministic training and validation-selected checkpoints. Small size and smooth activations suit the initial sparse-data study.
- Kept monotonic penalties off and omitted velocity/LLD residuals because necessary measurements and validated regime assumptions are absent. No fabricated PDE, constants, or mechanism parameters were introduced.
- Added bounded-only and standard NN ablations plus RF/GB comparisons on shared experimental splits. This separates constraint effects from curvature effects and avoids synthetic holdout evidence.
- Added concentration-group holdout and highest-concentration extrapolation, fixed-test learning-curve controls, physical violation metrics, raw-unit sensitivities and reproducibility manifests.
- Added safe inference with preprocessing restoration and checks against overwriting existing inputs/outputs.
- Added scientific integrity tests for bounds, independent concentration splits, train-only preprocessing, derivative backpropagation/finite differences, and checkpoint equivalence.
- Added design register, architecture bibliography and initial results report. Documented unit uncertainty, conditional prediction scope, sample limitations, and requirements for future governing-equation inference.

Future entries should record date, changed features/units/architecture/loss, physical evidence, validation protocol and results. Keep run configurations/manifests alongside results; changing test-driven choices requires a fresh final test.
