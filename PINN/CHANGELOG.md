# Change and decision log

## 0.1 - 2026-10-11

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
