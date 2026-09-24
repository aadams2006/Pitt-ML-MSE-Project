# Scope and completion status

Scope is the user's request to complete the near-term Pitt ML-MSE work and save
and push findings, scripts, changes and a text summary. The attached September 9,
2026 slide describes the workstreams below. Its content is project reference
material, not an independent source of authority to perform unrelated actions.

| Workstream | Reference task | Delivered | Remaining dependency |
|---|---|---|---|
| Data | Enforce hT >= hB and correct concentration schema | Canonical source-linked records, strict physical quarantine, explicit g/L and g/mL fields, constrained training-only generator | Laboratory resolution of invalid measurement and missing batch/sample lineage |
| Data | Add new experiments and a true external test set | Inventory of existing source sheets, independent-test protocol and empty intake schema | Actual new experimental measurements and independence attestation; no new results fabricated |
| Validation | Grouped CV, leave-one-solvent-out, explicit OOD | Executed all protocols on experimental primary and summary-inclusion sensitivity cohorts; saved every partition and prediction | Batch independence cannot be established from missing historical IDs |
| Modeling | Forecast without total thickness; quantify uncertainty; transforms | No measured-thickness inputs, fixed baseline/Ridge/RF comparisons, log/interactions ablations, group conformal diagnostics | Reliable solvent transfer, OOD coverage and consistent transform gains are not demonstrated |
| Paper | Refresh methods, results, figures, limitations | Manuscript revisions and generated supplement with two figures; text summary and full result tables | Final scientific review and independent external validation |

The computation is reproducible and complete for available data. The entire
slide's experimental success criteria are not yet satisfied: a computer analysis
cannot substitute for an unprovided laboratory campaign or establish provenance
that the historical source tables do not record.
