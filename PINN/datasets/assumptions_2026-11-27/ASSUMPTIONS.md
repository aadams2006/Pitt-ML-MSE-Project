# Provisional PINN physics-property datasets

Date: 2026-11-27. Source branch: `codex/pinn-physics-and-run-tracking`; input commit: `74c8d23c70e7cc6276a8d65a2df7a978633b2822`.

## Decision and intended use

These are populated **assumption scenarios**, not newly measured solution properties. Original templates and experimental values remain unchanged. Both exports retain every source row and identifier. No training, tuning, target-derived parameter estimation, or automatic LLD activation was performed.

The requested next step is possible as an exploratory calculation, but the available evidence cannot establish a statistically justified tolerance interval for viscosity, surface tension, or mobile-layer conversion. The provided min/max columns are **ranges across explicitly selected models and analogue endpoints**, not confidence intervals, measurement error bars, or guaranteed physical bounds. Actual solution properties and dry-film behavior can fall outside these ranges.

- `pinn_experimental_physics_assumed.csv`: 126 measured-label rows; estimates supplied on all 126.
- `pinn_synthetic_physics_assumed.csv`: 3,000 rows; concentration-dependent estimates on 2,818. The 182 nonpositive-concentration rows retain blanks in those fields. The existing thickness/concentration checks leave 2,808 baseline-label-eligible rows. A valid property calculation does not make an invalid thickness label valid.
- `property_summary.csv`: ranges by solvent and dataset.
- `provenance.json`: source/output hashes and checks.
- `build_estimates.py`: reproducible calculations, using only Python's standard library.

## Evidence supplied by the lab

Fajer's email, pasted by Alex in this conversation, reports 60 mm/min withdrawal and 20 s dwell for most experiments, additional hexane speed experiments, 65 cSt for neat PDMS, no measured solution viscosity/surface tension/density, and no established wet-to-dry mobile conversion. Coated-substrate contact angles do not supply the solution surface tension. Her reply also leaves LLD applicability unconfirmed because evaporation and adsorption may matter.

The scope of the 1 mm/s protocol is therefore an assumption for these particular existing rows, not a verified row-level measurement. The additional speed spreadsheets have not been provided here. No variable-speed records have been invented. Experiment IDs remain blank rather than fabricated; existing record IDs and source rows are retained.

## Assumption register and equations

### A1. Speed and dwell

Populate speed with 1 mm/s (= 60/60 mm/s) and add dwell = 20 s. These are the reported standard protocol. Each row labels the assignment as unverified at sample level. Synthetic rows inherit the same assumed protocol; they are not new observations. Keep velocity smoothness disabled for this constant-speed export.

### A2. Concentration basis and solvent identity

Use `concentration_g_l` for measured rows. For synthetic rows, interpret the numeric values under legacy `Concentration (g/mL)` as g/L, matching the repository's explicit renaming in `near_term/run_analysis.py`. This is supported by `near_term/results/workbook_inventory.json`, including hexane C35/C47 on sheet `11-(7,13)-2025`, ethyl acetate B36 on `11-10-2025`, and numerous toluene g/L labels through 50 g/L. The synthetic generators preserve source column names and numerical units; exact synthetic parentage is unavailable, so this remains an inherited interpretation. Preserve the misleading original header and add `physics_concentration_g_l` rather than silently rewriting the source.

Synthetic solvent identity is mapped only from the exact existing polarity/viscosity/surface-tension tuples, all of which match the repository's solvent descriptors. No nearest-neighbor guessing is used. Existing boiling-point values are preserved but not used; the synthetic toluene boiling-point range differs from the canonical solvent table.

Treat c as polymer mass per final solution volume; g/L and kg/m3 have the same numerical value. If the preparation used mass per starting solvent volume instead, use phi = c/(rho_p+c), not c/rho_p. At 50 g/L this alternative lowers phi by about 5%; preparation-volume ambiguity is additional uncertainty and is not included in the model min/max columns.

### A3. Ambient reference conditions, not known laboratory temperature

Use the existing solvent descriptors without rewriting their provenance. Use literature reference densities and nearby PDMS grade properties. Actual preparation/coating temperature and the exact source temperatures of the legacy descriptors are not established. No measured temperature or numerical temperature correction is fabricated. This is a mixed-reference ambient approximation, not a rigorously isothermal property table. Temperature, contamination, evaporation, and concentration-preparation errors are outside the scenario intervals.

### A4. Neat PDMS analogue properties

The manufacturer Gelest lists 50 cSt and 100 cSt conventional trimethylsiloxy-terminated PDMS fluids with densities 0.960 and 0.966 g/mL, and surface tensions 20.8 and 20.9 mN/m [S1]. Assume that this fluid family is a useful analogue for the reported 65 cSt PDMS; the actual product and end groups remain unknown.

Linear interpolation in nominal kinematic viscosity between those adjacent grades gives rho_p = 961.8 kg/m3 and gamma_p = 20.83 mN/m. Interpolation is chosen only as a transparent nearby-grade approximation. Numerical precision reflects the calculation, not experimental accuracy. Density endpoints 960 and 966 kg/m3 are used for an analogue sensitivity range; they do not bound an unidentified product.

The reported 65 cSt is kinematic viscosity of neat PDMS. Convert using eta_p[cP] = nu_p[cSt] * rho_p[g/mL] = 62.517 cP. Never insert 65 cSt directly as solution viscosity or confuse cSt with cP. Molecular weight is not inferred or populated from this grade approximation.

### A5. Solution density: additive volumes

Use solvent densities rho_s = 659 (hexane), 867 (toluene), and 902 (ethyl acetate) kg/m3 from the Sigma-Aldrich reference table [S2]. Hexane is modeled as n-hexane, not an unspecified commercial hexanes mixture; the source distinguishes them.

For one unit volume of solution, polymer occupies phi = c/rho_p and solvent occupies 1-phi. Hence:

`rho_solution = (1-phi)*rho_s + phi*rho_p = rho_s + c*(1-rho_s/rho_p)`.

This follows from mass balance under additive volumes, with no excess mixing volume. It is more grounded than fitting density to thickness. Recalculate using both PDMS density endpoints for the exported density range. Solvent-density, temperature and excess-volume uncertainty are not captured by that narrow range.

### A6. Solution viscosity: provisional interpolation, not a validated polymer law

Use a geometric interpolation as the working scenario:

`eta_solution = exp((1-phi)*ln(eta_s) + phi*ln(eta_p))`.

This requires no unknown fitted coefficients, stays positive, returns the solvent limit at zero polymer, and introduces concentration dependence. Those mathematical properties justify it as a screening assumption; they do NOT establish it as a true PDMS solution rheology model. Polymer coil effects and solvent quality can invalidate it, even at modest volume fractions.

For sensitivity, also calculate (i) eta = eta_s, the zero-polymer limiting approximation, and (ii) eta = (1-phi)*eta_s + phi*eta_p, arithmetic interpolation. Store the smallest/largest across these three models. Neither endpoint is a proven physical bound. Do not call all rows infinitely dilute: the measured concentration reaches 50 g/L, about 5.2% polymer volume under this model.

A PDMS/ethyl-acetate viscosity paper exists [S4], but the accessible abstract covers specified molecular weights and temperatures. The full quantitative model could not be retrieved in this session, and the user's polymer molecular weight is unknown. No coefficients were borrowed or invented. Huggins/Mark-Houwink fitting likewise requires missing polymer/solvent/temperature-specific information. The current geometric estimate is explicitly weaker than a validated correlation.

### A7. Solution surface tension: bulk interpolation scenario

Use `gamma_solution = (1-phi)*gamma_s + phi*gamma_p`, retaining existing pure-solvent values as gamma_s. This is an explicit composition-dependent baseline that recovers the component endpoints. Surface composition need not equal bulk composition, so it is NOT a validated surface-tension mixing law.

The sensitivity range spans the pure-solvent, bulk-interpolated, and neat-PDMS analogue values. The last is a surface-enrichment stress case, not a claim that the whole solution has pure-PDMS surface tension. Endpoint bracketing is not guaranteed for real mixtures. No coated-substrate contact-angle value is converted to liquid surface tension.

### A8. Wet-to-dry mobile factor: limiting scenario, not identified parameter

An ideal entrained layer containing polymer volume fraction phi would produce phi*h_wet of dense retained polymer if its area stays fixed, the solvent evaporates completely, and no polymer is lost or added. The cited coating study [S3] investigates volume-fraction conversion but does not validate it for these PDMS data.

Introduce q, the fraction of that entrained polymer remaining in the measured mobile layer:

`h_mobile_scenario = q*phi*h_wet`, with `0 <= q <= 1` **within this entrainment-only model**.

Populate `Wet-to-Dry Mobile Factor = phi` ONLY as the q=1 limiting scenario (all entrained polymer mobile). This deliberately avoids an unjustified central q value. The CSV labels that status on every row. It is not the recommended factor for training. q=0 and q=0.5 outputs are also supplied; 0.5 is an illustrative midpoint, not an estimate or probability. No q is fitted from total-minus-bonded labels.

The q interval covers complete removal/binding through full mobile retention of the assumed entrained mass. It does not cover evaporation-fed deposition from the meniscus, changed surface area, porous films, additional adsorption, or other transport mechanisms. Therefore the mobile model maximum is not a bound on actual observed mobile thickness. q cannot be independently identified from the information currently available.

### A9. LLD calculation and applicability

Use the repository implementation's flat-plate expression:

`Ca = eta*U/gamma`

`h_wet = 0.9458 * sqrt(gamma/(rho*9.81)) * Ca^(2/3)`.

Convert cP to Pa s, mm/s to m/s, and mN/m to N/m with 1e-3; convert resulting meters to nm with 1e9. Compute a wet-thickness range across all combinations of the selected viscosity, surface-tension and density scenarios. The mobile envelope is 0 to the largest wet scenario times the largest phi analogue. This Cartesian envelope is conservative relative to the selected combinations only; it is not calibrated uncertainty. Thickness sensitivity is proportional to eta^(2/3), U^(2/3), gamma^(-1/6), rho^(-1/2), and q*phi.

Set `LLD Applicable = 0` as an operational exclusion flag, with explicit reason **unvalidated**, not as evidence that LLD is physically false. Both hydrodynamic and velocity-smoothness weights remain zero. Low capillary number alone does not establish negligible evaporation, suitable wetting, Newtonian behavior, or relevance to a dry mobile layer. Constants and filled columns cannot resolve these structural assumptions.

## Diagnostics and next decision

Across the 126 experimental records, the q=1 working scenario divided by measured mobile thickness ranges from approximately 0.0265 to 723, with a median near 0.558. This is an in-sample descriptive check, not held-out predictive validation, and no parameters were changed in response. It shows why an exact equality penalty would presently be poorly justified. No attempt was made to force the ratio to one by deriving q from each label.

Use these files to inspect sensitivity and identify missing information. Before an LLD training experiment: confirm sample-level speeds and concentration preparation; identify the PDMS product and temperature; obtain a supported solution correlation or measurements; and establish a justified mobile-layer conversion and coating regime. If calibration is later attempted, fit parameters only inside training folds and keep evaluation labels untouched. The requested thickness constraints can continue independently; velocity smoothness remains excluded from this constant-speed dataset.

The assumption CSVs intentionally preserve original schemas plus metadata. They are not drop-in feature tables: use only an explicit approved feature list. Do not feed diagnostic thickness predictions, eligibility flags, or label-derived mobile columns as recipe-only inputs. Existing synthetic configurations use the original descriptor names, whereas templates renamed two descriptors to `Legacy Descriptor ...`; adapt that mapping explicitly if using a template. The existing LLD validator rejects missing values and requires at least one eligible row, so these exports cannot accidentally enable the loss unchanged.

## Reproduction and checks

From repository root: `python PINN/datasets/assumptions_2026-11-27/build_estimates.py`.

Inputs are the two unchanged template CSVs in the parent directory. Regeneration replaces only files in this dated output directory and refuses sources whose intake fields have since been filled. Checks verify source cell preservation, all row counts, disabled LLD flags, the zero-concentration solvent limit, U^(2/3) scaling, cSt-to-cP units, and rejection of nonpositive concentrations. Estimates are computed in a function whose arguments exclude all thickness labels.

## Sources and their scope

- **S1, manufacturer analogue data:** [Gelest conventional silicone fluids](https://technical.gelest.com/brochures/silicone-fluids/conventional-silicone-fluids/). Supports nearby-grade PDMS density/surface-tension endpoints, not identification of the user's bottle or solution mixing equations.
- **S2, solvent densities:** [Sigma-Aldrich Physical Properties of Solvents](https://www.sigmaaldrich.com/deepweb/assets/sigmaaldrich/marketing/global/documents/614/456/labbasics_pg144.pdf). Supports reference densities. Existing solvent viscosity/surface-tension descriptors are preserved from the user's dataset, not replaced by this table.
- **S3, coating model limitations:** [Zhang, Peng & Kornev (2022), The Thickness and Structure of Dip-Coated Polymer Films in the Liquid and Solid States](https://pmc.ncbi.nlm.nih.gov/articles/PMC9318159/), DOI 10.3390/mi13070982. Different polymer/solvent system; supports treating wet-to-dry conversion as a hypothesis requiring evaluation.
- **S4, relevant correlation lead only:** [Agasty et al. (2020), Scaling equation for viscosity of polydimethylsiloxane in ethyl acetate](https://doi.org/10.1016/j.polymer.2020.122779). Abstract inspected; full model not retrieved or implemented. No coefficients taken from this source.
- **Repository evidence:** `near_term/run_analysis.py`, `near_term/results/workbook_inventory.json`, `PINN/physics.py`, and the solvent-specific generator scripts, inspected at the input commit.
- **Protocol evidence:** Fajer's email supplied in the conversation; no unprovided spreadsheet or row-specific measurement is claimed.
