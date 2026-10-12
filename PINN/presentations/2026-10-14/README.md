# October 14, 2026 PINN research update

The presentation in `output/` uses the supplied research overview's 4:3 Pitt format, original banner/seal, gold titles, Arial content and Georgia date/page footers. The meeting date is October 14, 2026. Results cover the five archived runs available through October 11, 2026.

The latest presentation is `output/Alex_Adams_PINN_Research_Update_2026-10-14_LaTeX_v2.pptx`. The original 22-slide release remains available. The revised 24-slide deck adds an introductory PINN training diagram for engineers without an ML background and a slide with the exact derivative-loss definitions. It covers the implementation chronology, audited data, feature decisions, network architecture and parameter counts, losses and coefficients, the four proposed constraints, LLD activation requirements, optimization and shared evaluation splits. It includes measured/synthetic comparisons, the measured extrapolation failure, parity plots, concentration-monotonicity results, learning histories, architecture literature, run provenance, limitations and next steps.

Eight native editable charts contain embedded workbook snapshots of their numerical values. Chart data use 12 significant digits for Excel compatibility, while the original full-precision CSVs remain in the run archives. The training-history charts show log10 normalized MSE and sample every fifth measured epoch / 25th synthetic epoch plus endpoints. Both parity charts show all 442 synthetic test rows. Table metrics display appropriately rounded archived values. Source URLs and interpretation details are in the speaker notes.

The presentation distinguishes synthetic-distribution testing from independent experimental validation. Only the hard thickness bound and empirical curvature prior are active in default runs. Concentration monotonicity has a separate exploratory ablation. Velocity curvature and the hydrodynamic residual await appropriate measured inputs and validated applicability. Legacy execution times remain unknown rather than inferred from file dates.

No new model training occurred while preparing this update. The results are pinned to the archived implementation/evidence at Git commit `c771a486fe82b1ea13e7a70e156f64f01db03128`.

## Equation and diagram assets

Formulas are LaTeX-formatted equation graphics with high-resolution white backgrounds, typeset using Matplotlib's mathtext engine. Their editable LaTeX source is `assets/equations.tex`, with a named mapping in `assets/equations.json`. Equation graphics are images in PowerPoint, while surrounding text, original architecture diagram, tables and charts remain editable. The sources include the sigmoid definition as an additional reference. All displayed equations use the implementation's standardized-input coordinates and normalized target scale. LLD averages only over explicitly eligible rows.

The introductory image is `assets/pinn_training_diagram.png`, created with the built-in image-generation tool. It is a conceptual training schematic, not an exact count of this model's layers or units. The image source and generation prompt are documented in `assets/diagram_provenance.md`. The introduction cites the original PINN authors' project page and distinguishes classical PDE PINNs from the current constrained regression implementation.
