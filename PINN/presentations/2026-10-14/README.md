# October 14, 2026 PINN research update

The presentation in `output/` uses the supplied research overview's 4:3 Pitt format, original banner/seal, gold titles, Arial content and Georgia date/page footers. The meeting date is October 14, 2026. Results cover the five archived runs available through October 11, 2026.

The 22-slide presentation covers the implementation chronology, audited data, feature decisions, network architecture and parameter counts, losses and coefficients, the four proposed constraints, LLD activation requirements, optimization and shared evaluation splits. It includes measured/synthetic comparisons, the measured extrapolation failure, parity plots, concentration-monotonicity results, learning histories, architecture literature, run provenance, limitations and next steps.

Eight native editable charts contain embedded workbook snapshots of their numerical values. Chart data use 12 significant digits for Excel compatibility, while the original full-precision CSVs remain in the run archives. The training-history charts show log10 normalized MSE and sample every fifth measured epoch / 25th synthetic epoch plus endpoints. Both parity charts show all 442 synthetic test rows. Table metrics display appropriately rounded archived values. Source URLs and interpretation details are in the speaker notes.

The presentation distinguishes synthetic-distribution testing from independent experimental validation. Only the hard thickness bound and empirical curvature prior are active in default runs. Concentration monotonicity has a separate exploratory ablation. Velocity curvature and the hydrodynamic residual await appropriate measured inputs and validated applicability. Legacy execution times remain unknown rather than inferred from file dates.

No new model training occurred while preparing this update. The results are pinned to the archived implementation/evidence at Git commit `c771a486fe82b1ea13e7a70e156f64f01db03128`.
