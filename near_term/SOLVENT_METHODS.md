# Follow-up to the Next Steps slide

The first pass already completed thickness-free mobile forecasting, grouped and
shifted validation, prior-model comparisons, target separation and paper revisions.
Those runs are preserved and reused. This follow-up completes the quantitative
solvent-specific analytical investigation and adds its implications to the paper.
The slide's future PINN transition is context, not a request to begin a PINN project.

## Reproduce

With the dependencies in `near_term/requirements.txt`, from the repository root:

```sh
python near_term/solvent_analysis.py
python near_term/build_solvent_report.py
python -m unittest discover -s near_term/tests -v
python near_term/verify_solvent_results.py
tectonic research_paper.tex --outdir near_term/solvent_results --keep-logs
```

The prior `near_term/results/` artifacts are inputs, not rerun or overwritten. The
new paper PDF and all new diagnostics go to `near_term/solvent_results/`.
`NEXT_STEPS_SUMMARY.txt` summarizes only this follow-up. The original canonical
data audit and experimental limitations still apply.

## Fixed analytical prediction and diagnostic calculations

The mobile-layer comparator is the existing uncalibrated approximation:

`h = 0.94 (mu U)^(2/3) / [gamma^(1/6) (rho g)^(1/2)] * c/(c+965)`.

The liquid thickness is converted from meters to nm. The concentration `c` is in
g/L, PDMS density is 965 g/L, speed is 1 mm/s, and viscosity/surface tension use
the canonical per-solvent constants. Bath densities (kg/m3) are 655, 867 and 902
for hexane, toluene and ethyl acetate, inherited from the existing implementations.
These constants are assumptions, not newly measured solution properties. Tests
compare all three new fixed predictions with the original analytical functions.

The fixed-property model has local log-log concentration slope `965/(965+c)`,
close to one on the observed range. Condition-mean observed log-log slopes and
Spearman correlations are descriptive, equally weighting conditions. They are not
causal estimates, significance tests, or an independent test of a prespecified
mechanism. The concentration/solution preparation and washing protocols can be
confounded with the missing batch history.

The ratio `r = measured condition mean / fixed prediction` would require
speed or viscosity multiplier `r^(3/2)`, surface tension multiplier `r^(-6)`, or
density multiplier `r^(-2)` if that single factor were solely responsible. These
are algebraic equivalences, not inferred physical properties. They demonstrate
non-identifiability; a thickness residual alone cannot distinguish these factors,
retention, drainage, washing, or measurement/protocol differences.

An 81-combination sweep independently varies speed, viscosity, tension and density
by factors 0.5, 1 and 2. Its fixed-concentration multiplier ranges from 0.25 to 4.
This is a hypothetical stress range, not a measured uncertainty interval or a claim
that these variations are equally plausible. Any concentration-independent choice
still changes only amplitude, not curve shape. Treating c as per total solution
volume instead of per solvent volume changes the fraction from `c/(c+965)` to
`c/965`; the relative change is `c/965`, at most 5.18% here. This arithmetic cannot
identify the actual preparation convention, but its magnitude is insufficient to
remove the large toluene divergence.

## Leakage-controlled exploratory comparators

The same previous grouped-CV, solvent holdout and low/high concentration splits
are reused for both primary and summary-inclusion sensitivity cohorts. A fitted
comparator uses condition means from the training portion of the same solvent.
For an unseen solvent, it uses only pooled other-solvent training conditions.
Every fitting membership and parameter is saved. Test responses never enter fits.

- `fixed_lld`: no fitted response parameters.
- `local_mean`: mean of training condition means. Diagnostic baseline only; its
  nonzero zero-concentration extrapolation is not a physical deposition law.
- `scaled_lld`: nonnegative, least-squares scalar times the fixed prediction.
  The scalar is not constrained to one or less and must not be interpreted as a
  measured retention fraction.
- `saturating_mobile`: `A*c/(K+c)`, fitted by bounded nonlinear least squares in
  log parameters. Seven starting K values derive only from training concentrations.
  Bounds are A in [1e-6, 1e4] nm and K in [1e-6, 1e5] g/L. All successful starts
  compete on training loss only. Bound hits and training objectives are recorded;
  failure of all optimization starts raises an error.
  This is a phenomenological mobile comparator, not a bonded adsorption model or
  evidence that mobile deposition obeys adsorption kinetics.

These fits weight conditions equally and are solvent-specific when the solvent is
seen. The prior RF/Ridge models are pooled and fitted to individual readings; their
saved predictions are reused on identical test rows. Thus comparison is useful for
diagnosis but does not isolate algorithm choice from weighting and fitting scope.
All model metrics are included rather than selecting a winning curve per solvent
after inspecting test outcomes. No newly selected hybrid or deployment model is
claimed. The comparators were proposed after earlier results were inspected, so
their scores are exploratory rather than untouched confirmation. Earlier RF
conformal intervals do not apply to these new comparators.

Plot dashed curves are full-data descriptive fits, explicitly separated from the
outer-test predictions and scores. Some fold fits hit a bound; parameters should
not be interpreted mechanistically, even when curve error is small.

## Interpretation and literature boundary

For functional lubricant films, Merzlikine et al. distinguish adsorption-related
bonded material from mobile material produced by viscous flow. That study concerns
different lubricants and substrates and motivates target separation; it does not
establish the cause of this PDMS dataset's solvent differences.
[Publisher abstract](https://link.springer.com/article/10.1007/s11249-004-2753-8).

Zhang et al. examine liquid and solid polymer-film thickness separately, supporting
the need to test the conversion from wet entrainment to final material instead of
assuming proportional shrinkage is universally accurate. This is background, not
a direct experimental test of the current PDMS films.
[Article record](https://pubmed.ncbi.nlm.nih.gov/35888799/),
[DOI](https://doi.org/10.3390/mi13070982).

Both references already appear in the manuscript. Source abstracts were checked
for this follow-up; no inaccessible full-text claims are attributed to them.
The data-supported result is a solvent-dependent amplitude/shape mismatch. The
specific physical cause remains unresolved without controlled measurements.
Vary withdrawal speed within each solvent/concentration batch, measure solution
viscosity, tension and density, and record matched wet, dried, washed and bonded
thicknesses plus dwell and elapsed drying times to distinguish the hypotheses.
