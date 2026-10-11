# The four slide constraints: implementation and evidence

Version 0.2.0, 2026-10-11. The slide proposes four candidate constraints; support in code must be distinguished from evidence sufficient to activate them.

| Slide constraint | Implementation | Current data / activation |
|---|---|---|
| Bonded thickness cannot exceed total | `total * sigmoid(logit)` hard bound; also nonnegative bonded/mobile thickness | Active in every PINN and bounded-NN run. Applies to valid measured totals; defective synthetic labels are explicitly excluded. |
| Smooth response to withdrawal velocity | Separate squared second derivative of normalized bonded prediction with respect to standardized measured velocity | Implemented via `velocity_smooth_weight`. Requires varying positive velocity as a network input; rejects missing/constant velocity. Current experimental and 3k synthetic tables have no velocity, so this is inactive in their runs. |
| Monotonic viscosity and concentration relationships can be tested before enforcement | Configurable derivative-sign penalties plus training-only Spearman and partial-derivative diagnostics, including within-viscosity cohorts for concentration | Default off. Separate synthetic ablation tests a positive concentration hypothesis with weight .01; not promoted to a physical law. Viscosity has just three solvent-associated values, so a pooled derivative/correlation cannot isolate viscosity from solvent chemistry. Its sign remains off. |
| Hydrodynamic scaling where applicable | Gated dry-mobile-layer LLD residual: `(total - bonded_hat - dry_factor*h_LLD)/s_y` | Implemented via `hydrodynamic_weight`, explicit per-row validity flag, measured fluid properties/velocity/density and supplied dry conversion. Missing measurements or out-of-regime active rows reject the run. Current files lack these measurements, so this is inactive. |

The initial concentration/film-state curvature prior is still present (`smooth_weight = .001`). It encourages a smooth empirical response but does **not** count as observed velocity smoothness. The bounded architecture also guarantees a nonnegative inferred mobile layer through `mobile = total - bonded_hat`; this is a consequence of the thickness bound, not an independent fourth mechanism.

## LLD residual and measurement contract

For a clean Newtonian liquid entrained by a smooth flat plate at low capillary number:

`Ca = eta*U/gamma`

`h_LLD_wet = 0.9458 * sqrt(gamma/(rho*9.81)) * Ca^(2/3)`

Convert cP to Pa*s (`1e-3`), mm/s to m/s (`1e-3`), mN/m to N/m (`1e-3`) and meters to nm (`1e9`). The [JFM two-liquid coating paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/dipcoating-flow-in-the-presence-of-two-immiscible-liquids/CB484EB73B1903B5933AB087847C642E) gives this classical plate normalization. Surface/interface effects can alter scaling; see [surfactant dynamics in JFM](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/surfactant-dynamics-hidden-variables-controlling-fluid-flows/7303EA9FF056B9B0D1EE12EE4D6D3A53). Applying this relation to a dry mobile layer is a project-specific conditional prior that additionally needs a validated conversion; it is not a bonded adsorption law.

Required columns: `Withdrawal Velocity (mm/s)`, `Viscosity (cP)`, `Surface Tension (mN/m)`, `Solution Density (kg/m3)`, `Wet-to-Dry Mobile Factor`, and `LLD Applicable` (explicit 0/1). Active measurements must be positive, conversion in (0,1], and `Ca < lld_max_capillary_number` (conservative starting threshold .01, subject to lab validation). The flag asserts geometry, rheology, wetting, evaporation and regime checks made outside the model; the code cannot infer those from a scalar table. The dry conversion must come from independent measurements/model assumptions, never `(total - bonded)/h_LLD`, which would leak the target. Disabled rows do not contribute to the loss. At least one eligible training row is required if the loss is enabled.

`config_measured_physics.example.json` shows how to enable measured-velocity curvature and the LLD residual with an experiment-ID split. It intentionally will not run on the current tables, and contains no fabricated physical values. When supplying a suitable table, run it as a distinct experiment and record each assumed regime/conversion externally. Check constraint ablations and residual units before interpreting fitted behavior.

## Monotonicity and additional constraints

The concentration-only ablation is explicitly a hypothesis test. Both positive and negative signs can be configured; local soft penalties do not guarantee global monotonicity. Diagnostics based only on the training split prevent using test outcomes to select signs. Synthetic correlations are properties of the generator, not proof of a viscosity/concentration law in the lab. A measured zero-solute boundary, adsorption saturation/kinetics, or mass-conservation residual could be stronger future constraints, but would need background-thickness definitions, time/coverage data, and dry-film mass/density measurements respectively. They are not imposed without those measurements.

For computational cost, derivative penalties sample up to 256 training observations per epoch, deterministically under the run seed. This changes v0.1's all-row derivative evaluation, especially for 3k data; the effective sample count is recorded in configuration. It samples only observed valid rows and does not create unverified combinations of correlated measurements.
