# Follow-up verification — 2026-09-24

The Next Steps follow-up was present locally but had not been committed or pushed.
It was reviewed against the slide and the previously pushed near-term work. The
completed RF/Ridge runs were reused; no earlier model training was duplicated.

- All 13 unit tests pass, including the six solvent-analysis checks. They verify
  legacy LLD agreement, physical scaling and zero concentration, algebraic
  inversions, known-parameter recovery, group isolation and unchanged prior forecasts.
- `verify_solvent_results.py` independently recomputes all 288 saved metric rows
  from 6048 predictions. Counts, row RMSE/MAE/R2 and equal-condition RMSE agree.
  Predictions are finite, nonnegative and unique per cohort/protocol/fold/model/record.
- All four input/script hashes in the analysis manifest match the files used.
  The two recorded saturation-bound hits are hexane grouped fold 2, one in each
  cohort, consistent with the manuscript limitation. Machine-readable verification
  results and the verifier hash are saved in `verification.json`.
- The existing Tectonic build completed successfully and produced a 33-page PDF.
  The build emitted a Fontconfig warning but no missing-character, unresolved
  reference or overfull-box diagnostics. PDF text contains no unresolved `??` markers.
- Updated pages 27–29 and the conclusion on page 32 were rendered with Poppler
  and visually checked. The tables and solvent figure are readable and unclipped;
  fitted curves and hypothetical sweep ranges are labeled as descriptive.
- The two background references in `SOLVENT_METHODS.md` were checked against
  accessible primary-source abstracts. They motivate the target/measurement
  distinction; they do not identify this dataset's physical mechanism.

The new results establish a solvent-dependent discrepancy in magnitude and
concentration response. They do not establish a universal model improvement,
physical causation, independent external validation or deployment readiness.
Raw observations, earlier model results and the previous manuscript PDF are preserved.
