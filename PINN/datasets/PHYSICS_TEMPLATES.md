# PINN physics data templates

Created November 11, 2026 (New York). These CSVs combine existing data with blank physics intake columns. No values have been estimated, imputed, or taken from outside sources, and no models have been trained.

| File | Rows | Source |
|---|---:|---|
| `pinn_synthetic_physics_template.csv` | 3,000 | Existing `synthetic_3000.csv` snapshot |
| `pinn_experimental_physics_template.csv` | 126 | Latest audited `20261111T134229Z_measured_followup_measured.csv` snapshot: 58 hexane, 35 toluene, 33 ethyl acetate |

The experimental export uses the current eligible measured set, rather than only the older 58-row hexane set. Previously excluded records are not reintroduced. Existing identifiers, eligibility flags, grouping and source references are retained. The synthetic export preserves all 3,000 rows, including known defects: the existing PINN audit excludes 192 rows, leaving 2,808 for its baseline configuration. No rows are silently corrected or removed here.

## Blank physics columns

| Column | Units / purpose |
|---|---|
| `Withdrawal Velocity (mm/s)` | Speed for velocity smoothness and hydrodynamic loss |
| `Viscosity (cP)` | Validated solution viscosity for hydrodynamic loss |
| `Surface Tension (mN/m)` | Validated solution surface tension for hydrodynamic loss |
| `Solution Density (kg/m3)` | Solution density for hydrodynamic loss |
| `Wet-to-Dry Mobile Factor` | Dimensionless independent dry-mobile conversion, in (0,1] |
| `LLD Applicable` | Explicit 0/1 applicability decision; blank means unknown |
| `Experiment ID` | Recoverable experimental unit identity |
| `measurement_source`, `unit_confirmation`, `regime_evidence` | Evidence supporting supplied values, units and applicability |

The first six names match `PINN/physics.py`. Gravity is already fixed at 9.81 m/s2 in that implementation; it is not a missing measurement. Thickness bounds and the existing concentration curvature prior need no additional static constants. Monotonicity signs and loss weights belong in configuration, not invented per-row measurements. Additional adsorption kinetics or mass-conservation laws have no established implementation/data contract yet, so speculative parameters have not been added.

## Preserve the distinction between descriptors and physics measurements

The synthetic source's existing `Viscosity (cP)` and `Surface Tension (mN/m)` values are preserved as `Legacy Descriptor Viscosity (cP)` and `Legacy Descriptor Surface Tension (mN/m)`. This frees the exact physics field names for blank, validated solution properties. Other source headers are only whitespace-trimmed; source cell strings and row order are preserved exactly. Experimental legacy descriptors already have distinct names (`viscosity_cp`, `surface_tension_mn_m`) and remain unchanged.

These are intake templates, not drop-in replacements for every existing training configuration. To reproduce a synthetic baseline using this export, map its two legacy descriptor names in the feature list; do not select blank physical properties as features. For experimental baseline inputs use the existing canonical names (`concentration_g_l`, `uncoated_nm`, `total_nm`, solvent indicators; target `bonded_nm`). Keep velocity and hydrodynamic weights zero until the physics table and configuration have been validated. The current LLD validator requires finite fields on all supplied rows, including inactive rows; it does not mask missing values. Never fill missing values with zero merely to pass validation.

Some physical properties may be shared within a validated condition, but solution properties may vary with concentration and temperature. Constant velocity cannot support learning velocity derivatives. Synthetic rows have no recoverable measured parent identifiers; assigning physics values would be an explicitly documented modeling assumption, not a new measurement. The synthetic concentration header `Concentration (g/mL)` is retained verbatim and is not reconciled to experimental `concentration_g_l`; confirm units before combining datasets. Do not use these precomputed synthetic rows to claim independent experimental validation.

Regenerate with `python PINN/datasets/build_physics_templates.py` only before filling the templates; regeneration replaces the outputs. `physics_templates_provenance.json` records source/output hashes, mappings, blank-column definitions, and preservation checks. `template_source_csv_line` is the one-based physical source CSV line (header is line 1), distinct from existing workbook source-row identifiers.
