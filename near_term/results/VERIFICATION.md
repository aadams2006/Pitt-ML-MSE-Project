# Verification record

- Executed the full primary and summary-inclusion sensitivity analysis with seed 42.
  Both cohorts completed all ten outer splits and five fixed model/feature variants
  for two targets. `run.log` and `run_manifest.json` record execution and versions.
- Seven tests passed. They check concentration conversion and lineage, physical
  exclusions (including nonfinite/negative cases), absence of response features,
  grouped and solvent isolation, strict low/high OOD boundaries, finite-sample
  conformal behavior, synthetic constraints and parentage, calibration isolation,
  uniqueness of predictions, and independently recomputed saved RMSE values.
- Verified the three primary source SHA-256 hashes and runner hash against the
  executed manifest. Original experimental files and historical results are unchanged.
- Compiled `research_paper.tex` with Tectonic 0.17.0 to the committed 30-page PDF.
  No unresolved references, missing-character or overfull-box diagnostics were found.
  Tectonic emitted a Fontconfig configuration warning; it still embedded fonts and
  completed successfully. Build logs are retained.
- Rendered the updated abstract, methods, tables, figures, limitations and conclusion
  with Poppler and visually inspected them for clipping, overlap and readability.
  Confirmed the new tables match saved metrics. Removed the duplicate References
  heading in the source and clarified that the historical 2887-row mobile subset
  only enforced nonnegative mobile thickness, not all physical constraints.

These checks establish code/output consistency, not independent scientific
replication. Batch provenance is incomplete, only three solvents are represented,
and new external experimental measurements remain unavailable.
