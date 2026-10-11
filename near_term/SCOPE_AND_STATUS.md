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

## Follow-up slide: Next Steps

| Priority | Prior completion | Follow-up disposition |
|---|---|---|
| Refine mobile-layer prediction beyond the optimized RF | Thickness-free RF/Ridge, transform tests, uncertainty and shifted validation completed | Reused existing predictions; added fixed/scaled LLD, local mean and exploratory saturation tests |
| Investigate solvent-specific analytical behavior | Limitations had been identified, but not quantified mechanistically | Completed amplitude/shape diagnosis, condition ratios, algebraic parameter inversions and a hypothetical parameter sweep; actual mechanism remains unidentifiable without measurements |
| Revisit prior model performance while separating targets | Bonded and mobile forecast evaluations completed separately | Preserved prior fits and added solvent-resolved mobile comparison on identical test rows; no bonded target used for fitting mobile analytical curves |
| Continue paper revision | First robustness revision completed | Added reproducible solvent section, tables, figure, conclusion and follow-up summary |

See `SOLVENT_METHODS.md`, `solvent_results/`, and `../NEXT_STEPS_SUMMARY.txt`.
The PINN transition depicted on the slide is future context. No PINN project was
started. New physical experiments and independent external validation are still
required to establish mechanisms or deployment readiness.
