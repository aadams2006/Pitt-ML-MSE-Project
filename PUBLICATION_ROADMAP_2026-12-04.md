# Pitt Li Lab: publication-readiness roadmap through December 4, 2026

Prepared November 11, 2026 (New York), from the current working repository, including uncommitted November 11 results. **Planning only: no model, data, experiment, or manuscript changes were made in preparing this roadmap.**

## 1. Objective, budget, and realistic endpoint

Use the existing measurements to establish the strongest defensible predictions of bonded, mobile, and total dry-film thickness. Compare conventional ML, simple physical/phenomenological models, and physically constrained neural modeling without mixing their input-information budgets. Deliver a reproducible, PI-reviewable manuscript package by December 4. Acceptance, submission readiness after scientific review, improved accuracy, and generalization to new solvents cannot be guaranteed.

There are approximately 3.3 weeks from November 12 through December 4: about **49–66 available hours at 15–20 hours/week**. Plan **45 hours of core work plus 5 hours contingency**. Up to 10 additional hours are optional, only after core milestones are secure. These are hands-on research/engineering hours, including interpretation and documentation, not just model runtime. Avoid scheduling critical work on Thanksgiving, November 26.

The strongest attainable paper is an honest small-data comparison with measured-data validation, physical admissibility, and explicitly demonstrated domain limits. A negative result for additional PINN penalties or augmentation is a useful outcome. Do not make a physics-informed model outperform RF a condition of completion.

**No new experimental data are assumed or required by any core stage.** Missing physical measurements and historical batch identities become documented limitations. Previously used observations cannot become an independent external test by relabeling or resplitting them.

## 2. Repository assessment and evidence

Paths below are relative to repository root. `CAMP` in this document means `PINN/campaigns/20261111T134229Z_measured_followup/`.

### 2.1 Data available now

The reviewed canonical loader is `near_term/run_analysis.py:load_data`. Its source files are under `models/feature engineering v1/data FE-V1/`: `agg.data.xlsx`, `toluene+pdms.csv`, and `ethyl acetate+pdms.csv`. Raw workbook evidence also exists in root `PDMS with Toluene 2.xlsx`, root `Ethyl Acetate+PDMS.xlsx`, and `Hexane+PDMS-new.xlsx` under the feature-engineering data folder. Copies elsewhere in the repository must not be counted as additional experiments.

| Primary measured cohort | Rows | Solvent/concentration conditions | Concentration range, g/L | Bonded range, nm | Mobile range, nm |
|---|---:|---:|---:|---:|---:|
| Hexane | 58 | 16 | 0.02–10 | 0.34–0.69 | 0.04–12.67 |
| Toluene | 35 | 11 | 0.0025–50 | 0.34–0.74 | 0.09–0.39 |
| Ethyl acetate | 33 | 9 | 0.1–10 | 0.25–1.08 | 0.26–3.18 |
| Total | **126** | **36** | Solvent dependent | | |

Counts and ranges were checked against `CAMP/canonical_all.csv`. Of 142 source rows, 15 potentially overlapping toluene summaries and one invalid toluene row (bonded 0.86 nm > total 0.85 nm) are excluded. Preserve these exclusions and the separate summary-inclusion sensitivity cohort. Do not silently correct or clip the invalid observation. Concentration numbers are **g/L**, despite legacy g/mL column labels; the canonical schema has already corrected this using workbook evidence.

The groups are condition proxies, not established independent batches. Repeated washes, reused solutions, summaries, and missing sample lineage limit independence claims. A within-condition spread estimate is not necessarily instrument noise or an irreducible error floor.

Synthetic resources include the 3,000-row combined file in the feature-engineering data folder, solvent-specific synthetic files and generators, earlier single-solvent files under `data/`, and the byte-preserved `PINN/datasets/synthetic_3000.csv`. `PINN/datasets/experimental_58.csv` is the earlier hexane-only snapshot; it is not the current three-solvent cohort. `PINN/datasets/provenance.json` records snapshot identities.

The combined synthetic corpus has **192 uniquely invalid rows**, with overlapping counts of 182 negative concentrations, 113 negative mobile differences, and six negative thicknesses. The PINN benchmark retains 2,808 rows, whereas the historical mobile RF retained 2,887 because its exclusion policy differed. Exact synthetic parent lineage is unavailable. Existing generators bootstrap measurements and fit noise/covariance from their input dataset; splitting their generated output afterward cannot establish independence from measured holdouts. The 300-row `near_term/results/constrained_synthetic_audit.csv` demonstrates physical admissibility and training-parent tracking, but is an audit, not an augmentation efficacy result.

### 2.2 Distinguish the prediction tasks

| Task | Allowed inputs | Outputs | Existing implementation | Intended claim |
|---|---|---|---|---|
| A: recipe-only forecasting, primary | Concentration and known solvent identity/descriptors; no thickness measurements | Bonded and mobile; total reconstructed as their sum | `near_term/run_analysis.py`, `near_term/solvent_analysis.py` | Forecast within the measured process domain; quantify transfer failure |
| B: conditional film-state inference, secondary | Concentration, solvent, measured uncoated and total thickness | Bonded; mobile = measured total − predicted bonded | `PINN/run_followup.py`, `PINN/train.py`, `PINN/core.py` | Infer layer partition after total thickness is known |

Task B is legitimate if total thickness is available at inference, but it cannot demonstrate recipe-only forecasting. Do not feed measured total through a bound, loss, normalization, feature, or postprocessing step in Task A. For Task A a valid constraint is predicted bonded >= 0, predicted mobile >= 0, predicted total = their sum. Comparing predicted bonded with *observed* total is a diagnostic, not an available inference-time constraint.

### 2.3 Results worth building on

| Evidence | Recorded outcome | Implication |
|---|---|---|
| `near_term/results/summary_metrics.csv`, primary grouped RF | Bonded condition RMSE 0.1063 nm, row R² 0.5678; mobile condition RMSE 0.9448 nm, row R² 0.8790 | Useful recipe-only measured-data reference |
| Same RF, leave-one-solvent-out | Bonded condition RMSE 0.2103 nm, row R² −0.6054; mobile 2.9277 nm, row R² −0.4555 | Cross-solvent generalization is not established |
| Same RF, high-concentration OOD | Mobile condition RMSE 4.1572 nm; simultaneous condition coverage 11.1% for nominal 80% intervals | Extrapolation and uncertainty failures must remain visible |
| `near_term/solvent_results/mobile_metrics.csv`, primary grouped | Scaled LLD 0.3644 nm; saturating mobile curve 0.4163 nm; fixed LLD 15.2077 nm condition RMSE | Calibrated low-complexity models deserve priority over large searches |
| `CAMP/cv_summary.csv`, conditional bonded inference | RF 0.1018; bound + smoothness 0.1061; bound only 0.1065; unconstrained NN 0.1230; GB 0.1097 nm mean condition RMSE | RF is currently strongest; bounding is useful, smoothness advantage is negligible |
| `CAMP/paired_curvature_effect.csv` and report | Smoothness better in 9/15 fold/seed pairs; mean fold difference about −0.0003 nm | Not evidence of a meaningful accuracy gain or 15 independent experiments |
| Historical mobile Bayesian RF, August 26 | Synthetic OOF R² 0.999597, RMSE 0.036338 nm; measured total is an input and part of the response | Development diagnostic, not an experimental recipe-forecast benchmark |

The analytical comparisons already fit on training conditions, but use solvent-specific condition means while the RF uses pooled individual readings. Thus their apparent advantage mixes model family, fitting scope, and training weights. Resolve this before claiming a fair algorithm comparison. With fixed solvent properties, scaled LLD is effectively a calibrated concentration curve within each solvent, not validation of hydrodynamics.

The November 11 PINN campaign already completed **24 runs / 120 model fits**: five outer folds × three initialization seeds, plus three smaller training fractions × three seeds on fold 1. Its 126-row prepared dataset, fixed splits, exact source snapshots, histories, checkpoints, audit, figures, and verification logs are present. Learning curves already retain whole conditions with fixed validation/test partitions. Do not repeat this campaign merely to create more results. Three seeds on fixed folds measure initialization variability, not split uncertainty; the learning curve represents one test fold only.

### 2.4 Physical validity and code limitations

- `PINN/core.py` implements `bonded = measured_total * sigmoid(network_output)`. This guarantees the stated bounds for valid totals; it is constrained regression, not a PDE solver. The default curvature penalty is an empirical smoothness prior in standardized coordinates.
- Velocity smoothness and LLD penalties exist but are inactive. `CAMP/data_provenance.json` and `physics_cell_evidence.json` report no sample-linked validated velocity, solution properties, independent wet-to-dry conversion, and regime evidence in this checkout. Do not replace these with the analytical scripts' assumed 1 mm/s or solvent constants and call them measured physics.
- Solvent descriptors take only three combinations. Their separate causal effects and generalization to arbitrary solvents are not identifiable. Pooled feature importance is not a mechanistic explanation.
- Current PINN training uses row-average MSE and row-average validation error, although reporting prioritizes condition-weighted error. Condition-balanced training/selection is a testable improvement.
- `PINN/train.py` uses the same training subset for neural and tree comparisons, reserving validation groups. The earlier recipe-only point models use the entire outer-training partition. Cross-workflow scores therefore differ in both task and training budget.
- The generic PINN `extrapolation` branch sorts `group_column`. With string IDs such as `hexane:10`, this is not a safe numeric, within-solvent concentration OOD protocol. Reuse the explicit numeric split logic in `near_term/run_analysis.py`; do not simply apply `--split extrapolation` to the current multi-solvent group IDs.
- Historical Bayesian optimization selects parameters and then reports OOF predictions on reused folds. This is not nested model-selection validation. New tuning must occur inside outer training groups.
- `near_term/run_analysis.py` and solvent/report builders use fixed output locations. Running the README's reproduction commands without adaptation can replace historical artifacts. New work needs separate timestamped destinations.
- `PINN/verify_followup.py` contains campaign-specific assumptions, including 126 rows and seeds 11/42/73. Reuse it for that campaign; generalize checks separately for new experiments.

### 2.5 Manuscript and reproducibility status

`research_paper.tex` already includes the experimental and solvent supplements, but its title, abstract, and much of the historical narrative still emphasize Bayesian optimization and synthetic performance. The latest conditional PINN campaign is not yet integrated as a coherent second task. Existing `paper_figures/` and Overleaf bundles predominantly serve the older narrative. Preserve them, then create a focused new paper package.

The worktree was already dirty at assessment: PINN source/docs/requirements/tests modified, November 11 runs/campaign/presentation untracked, and two older presentation files marked deleted. Git HEAD was `6975448`. **Do not reset, clean, restore, delete, or blindly commit these changes.** Preserve current files and baseline status; archived run source hashes are more precise provenance than HEAD alone.

This assessment inspected current code, data snapshots, result CSV/JSON files, reports, verification logs, and manuscript text. It did not rerun model training, regenerate datasets, or independently repeat all historical tests. Stage 1 performs the current integrity checks. No new literature review was performed here; verify primary references during manuscript preparation rather than treating repository citations as newly checked.

## 3. Common execution contract for every stage

Each stage below is intended for a separate session. At the start, read this roadmap, the prior stage handoff, and current Git status. Use repository-relative paths and the actual current source, including authorized uncommitted work.

Proposed new root: `publication_2026/`. Create it only when Stage 1 is authorized. Keep a small `STATUS.md` linking completed stage reports and the active cohort/split/config manifests. Put new experiments in `publication_2026/runs/<UTC timestamp>_<task>_<experiment>/`, with `stage_01/` through `stage_07/` for decisions and handoffs. Future locations in this roadmap are planned, not existing deliverables.

Every completed stage must save `HANDOFF.md` containing:

1. What was completed, exact commands/configs, source and dataset hashes, and artifact paths.
2. Tasks/features/targets, cohort IDs, split IDs, seeds, exclusions, fit/calibration membership, and any deviations.
3. Results and decisions, including null/failed runs and why any candidate was dropped.
4. Test/verification results, time spent, remaining budget, and the next stage's concrete starting point.

All new runs must save raw predictions and condition-level errors, hyperparameter/selection records, actual feature lists, elapsed runtime, environment versions, and source identity. Preserve failed attempts. Never overwrite completed runs, source measurements, existing result folders, manuscript bundles, or earlier checkpoints. Capture the initial dirty-file state without treating it as disposable.

**Primary score:** equal-condition RMSE = sqrt(mean over conditions of mean squared reading error within each condition). Also report condition MAE, row RMSE/MAE/R², solvent-resolved errors, sample/group counts, and physical violations. Total error must be calculated from paired bonded/mobile predictions for the same held-out rows, not by adding component RMSEs.

Use measured rows exclusively for validation, calibration, and test. Fit transforms, feature selection, analytical calibration, augmentation, and tuning only within the relevant training partition. Historical test outcomes have already informed research choices: report new work as retrospective internal validation, even with nested selection. No new split makes these data pristine.

## 4. Schedule and priorities

| Stage | Target dates | Priority | Hands-on hours | Depends on | Principal deliverable |
|---|---|---|---:|---|---|
| 1. Freeze evidence and scope | Nov 12–13 | P0 | 4 | None | Audited input/claim contract and preserved baseline |
| 2. Shared validation and fair baselines | Nov 13–17 | P0 | 7 | 1 | Common measured-data benchmark |
| 3. Focused conventional/analytical improvement | Nov 17–20 | P1 | 10 | 2 | Limited nested comparison and selected procedures |
| 4. Physical modeling and neural ablations | Nov 20–24 | P1 | 7 | 2; stage 3 for final comparisons | Recipe-constrained model and conditional ablations |
| 5. Robustness and uncertainty | Nov 24–27 | P0 | 5 | 3–4 | Generalization/uncertainty evidence and domain limits |
| 6. Publication figures and synthesis | Nov 27–30 | P0 | 5 | 5 | Final tables, figures, and claim ledger |
| 7. Manuscript and reproducible handoff | Dec 1–4 | P0 | 7 | 6 | PI-reviewable draft and reproducibility package |
| Contingency | Across stages | P0 | 5 | As needed | Debugging, missing provenance, editorial fixes |
| **Core plus contingency** | | | **50** | | |

Suggested weekly allocation: Nov 12–18 about 15 h; Nov 19–25 about 17 h; Nov 26–Dec 2 about 15 h; Dec 3–4 about 3 h. Protect writing time. If behind, reduce candidate families/optional experiments, not leakage checks or the final evidence package. Freeze new model proposals November 24 and final numerical results November 30.

## Stage 1. Freeze evidence, data, and claims — 4 hours

**Read first:** `near_term/README.md`, `near_term/SOLVENT_METHODS.md`, `CAMP/RESEARCH_UPDATE.md`, `PINN/PHYSICS_CONSTRAINTS.md`, this roadmap.

**Tasks and time:**

- 0.75 h: Record Git status/HEAD, source hashes, and inventory of existing campaign artifacts. Preserve untracked work; establish additive output directories. Do not repair existing presentation deletions as part of this work.
- 1.25 h: Reconcile the three source tables, prepared PINN snapshot, and canonical records by stable record ID. Confirm units, exclusions, duplicate copies, primary 126/36 counts, and which columns are available before coating versus after measurement. Check existing metadata only for batch/sample lineage; time-box this search to 30 minutes. If unavailable, retain the condition proxy and record the limitation.
- 1 h: Run read-only verification of historical run hashes/scores and the November 11 campaign. Inspect discrepancies before any reproduction. Record dependency versions from the working environment, resolving the broad PINN versus pinned near-term requirements without upgrading blindly. Unit tests may use a writable temporary directory; do not invoke training/report builders that overwrite old folders.
- 1 h: Freeze Task A/B feature contracts, targets, split policy, primary metric, candidate limits, and claim categories. Record legacy synthetic lineage as unverified unless direct evidence establishes it.

**Deliver:** `stage_01/data_contract.md`, `input_manifest.json`, `baseline_status.txt`, `verification.log`, and `HANDOFF.md`.

**Success:** Every retained row has a source ID and exclusion rationale; the two tasks are distinct; existing results remain unchanged; verification passes or each discrepancy is documented and blocks reuse of the affected artifact. No unresolved provenance is silently assumed valid.

## Stage 2. Build one fair measured-data benchmark — 7 hours

**Read first:** Stage 1 handoff, `near_term/run_analysis.py`, `near_term/solvent_analysis.py`, `PINN/train.py`, `CAMP/all_splits.csv`.

**Tasks and time:**

- 2 h: Save explicit outer partitions for five-fold condition CV, all three solvent holdouts, and numeric per-solvent low/high 20% concentration holdouts. Check consistency with existing record-ID assignments; reuse compatible predictions instead of rerunning them. Include internal grouped validation for checkpoint selection and later tuning. Confirm every eligible row receives one grouped OOF prediction per procedure/seed.
- 2 h: Establish Task A references: global and same-solvent training means, Ridge, raw RF, GB, fixed/scaled LLD and saturating mobile curve. Add simple linear concentration fits to expose what scaled LLD contributes beyond calibration. Use identical test rows. Compare pooled versus solvent-specific fitting for the strongest RF and analytical mobile comparator, with condition-balanced training (weights 1/readings-per-condition, normalized) or condition-mean fitting documented explicitly. Avoid a full factorial search.
- 1 h: Add total-thickness evaluation from paired Task A component predictions. For Task B, reuse the five-model November 11 benchmark and add a simple training-only constant bonded fraction of total as an inexpensive reference. Keep the same train/validation allocation for causal ablations; distinguish any full outer-training tree refit from that comparison.
- 1 h: Add integrity checks for no group overlap, training-only transforms/fit membership, feature contracts, numeric OOD ordering, and exact metric reconstruction. Where solvent-specific training has too few groups, use a documented pooled fallback rather than a test-selected choice.
- 1 h: Summarize reference scores and discrepancies between old/new protocols. Treat negative R² on low-variance solvent subsets with care; retain absolute errors.

**Deliver:** frozen `splits.csv`, `benchmark_spec.json`, predictions/fit membership, `baseline_metrics.csv`, benchmark tests, and `stage_02/HANDOFF.md`.

**Success:** No cross-task leaderboard; every compared procedure has known input information and training scope; analytical and ML improvements can be evaluated on identical measured conditions. Physical compliance, weighting, and clipping/projection are explicit. The original published numbers remain available as historical references.

## Stage 3. Improve the most promising recipe-only models — 10 hours

**Read first:** Stage 2 handoff/specification and benchmark results; historical RF optimization source only as a reference, not as validation code to copy unmodified.

**Tasks and time:**

- 1.5 h: Diagnose solvent-resolved errors and repeated-condition spread using outer-training data for model choices. Examine concentration versus log-concentration, identity versus fixed descriptor encoding, and whether pooled weighting explains the analytical advantage. Do not infer causal descriptor effects from three solvents.
- 4 h: Implement a small **nested grouped** search. Maximum 12 RF settings and 8 GB settings per target, three inner group folds, five outer folds; fix the search space before running. Prioritize leaf size/depth and regularization, not thousands of trees. Use the existing fixed Ridge and analytical references. Feature encoding/transform choices count within these budgets, not extra unrestricted searches. For held-solvent evaluation, an unseen category must be supported without using its labels; descriptor-based transfer and identity-only interpolation remain separately labeled.
- 2 h: Test one low-complexity analytical-plus-residual candidate for mobile thickness. Fit the analytical component and residual learner inside each inner training fold, score only its validation groups, then refit the selected procedure on outer training. Never fit the analytical part on validation labels. Keep parameters phenomenological and report bound hits. If simple calibrated curves remain best, retain them.
- 1.5 h: Evaluate final procedures on outer tests, with paired condition error differences versus fixed references. Save all candidates' inner scores and selection frequency. Use one tree seed for search; reserve 11/42/73 for final stochastic comparisons rather than multiplying the search by seeds.
- 1 h: Choose at most two finalists per task for Stage 5, based on inner-selection procedures, interpretability, and stability. Report all prespecified outer model comparisons; do not select a different winner for each solvent after viewing test outcomes.

**Deliver:** `search_space.json`, inner/outer membership and scores, `selected_configs_by_fold.json`, fitted analytical parameters, OOF predictions, comparison report, and `stage_03/HANDOFF.md`.

**Success:** No test-driven parameter selection; no measured-thickness input in Task A; all new accuracy claims use measured outer tests. Aspirational improvement is >=5% lower condition RMSE than a matched strong reference, with no >10% degradation on another prespecified target/solvent without disclosure. These percentages are engineering decision thresholds, not significance tests or guaranteed outcomes. If gains are smaller/unstable, stop searching and prefer the simpler model. A valid null result passes this stage.

## Stage 4. Physical constraints and neural modeling — 7 hours

**Read first:** Stage 2/3 handoffs, `PINN/core.py`, `PINN/physics.py`, `PINN/PHYSICS_CONSTRAINTS.md`, `CAMP/paired_curvature_effect.csv`.

**Tasks and time:**

- 2.5 h: Add a separate minimal Task A neural path; do not break the existing conditional checkpoint architecture. Use small nonnegative bonded/mobile output heads and derive total by addition. Train on existing component observations with training-only target scaling so mobile error does not dominate solely through its larger numerical range. Do not treat total = bonded + mobile as a new independent measurement or claim the algebraic identity proves a physical mechanism. If this implementation exceeds the time box, retain validated Task A analytical models and complete Task B ablations instead; document the deferred recipe network.
- 1.5 h: Compare Task A unconstrained and nonnegative heads on the same inputs/splits/architecture, with three seeds for final settings. For Task B reuse existing standard/bounded/bounded-plus-smoothness results and test condition-balanced loss/checkpoint selection. Add a clipped RF diagnostic to separate neural architectural benefits from simple admissibility postprocessing; show both raw and constrained errors.
- 1.5 h: Limit curvature-weight selection to {0, 0.0001, 0.001, 0.01}, inside grouped outer-training validation only, with fixed small architecture. Count zero as a valid winner. Diagnose data loss and regularizer loss separately. Do not expand architecture searches or choose derivative signs from outer test trends.
- 0.75 h: Quantify bound violation rate/magnitude and behavior along concentration sweeps restricted to each solvent's supported range. Keep total/uncoated inputs in observed plausible combinations for Task B. Check component/total consistency and numerical stability. Label off-support sweeps as stress tests, not validated predictions.
- 0.75 h: Record what is guaranteed, what is an empirical regularizer, and what is unsupported. Leave velocity, viscosity monotonicity, and hydrodynamic residual weights off unless already-existing sample-linked evidence was verified in Stage 1; their activation is optional work, not a core dependency.

**Deliver:** task-specific configs and prediction artifacts, physics ablation table, domain/constraint diagnostics, meaningful bound/input-isolation tests, and `stage_04/HANDOFF.md`.

**Success:** Constrained outputs obey nonnegativity and the appropriate task-specific relationship within numerical tolerance; bounds do not use unavailable Task A measurements; gains are compared against bound-only and simple projection controls. If smoothness does not help, report that directly. Describe the method as physically constrained/regularized regression; reserve stronger mechanistic claims for evidence actually present.

## Stage 5. Validate robustness, uncertainty, and synthetic-data limits — 5 hours

**Read first:** Stage 3/4 handoffs and `near_term/README.md` uncertainty methodology.

**Tasks and time:**

- 1.5 h: Run the locked finalists on three solvent holdouts and numeric low/high concentration holdouts. Any tuning for these protocols must use their training groups only. Save per-solvent errors and failure cases, not just pooled scores. If compute/time is constrained, evaluate fixed finalist procedures rather than restarting large searches.
- 1.25 h: Extend group split-conformal diagnostics to the actual finalists, using separate proper-training/calibration groups within outer training. Reuse the existing nominal 80% method and finite-sample quantile rule; do not attach old RF intervals to new models. Report calibration group count, group/row coverage, width, and infinity where required. Interval centers belong to their calibration model, not automatically the full-data point model. For Task B, intersect intervals with [0, measured total] only under the stated valid-total contract.
- 1 h: Produce paired condition-bootstrap descriptive ranges for model differences using the same sampled conditions across models. Average seed-level errors within condition or retain the hierarchy; never treat 15 fold/seed pairs or repeated predictions as independent samples. Explain that these ranges condition on fitted folds and omit unknown batch dependence. Report all three solvent holdouts separately rather than a confident population CI from three solvents.
- 0.75 h: Check summary-inclusion sensitivity for finalists and inspect input/measurement sensitivity. Use observed replicate spread only as a labeled empirical proxy; do not invent instrument accuracy. Mark the single invalid row excluded and unresolved. Compare major conclusions with and without summaries, not the best-looking cohort.
- 0.5 h: Finish the synthetic-data decision: keep legacy 3k scores as generator-distribution diagnostics unless valid parent separation is demonstrable. Do not train on the full legacy synthetic set and test on measured parents. If augmentation cannot be validly isolated, document why it was not used rather than claiming it failed empirically.

**Deliver:** `robustness_metrics.csv`, interval predictions/coverage/width, paired error ranges, `synthetic_use_decision.md`, a model applicability statement, and `stage_05/HANDOFF.md`.

**Success:** Complete measured-data evaluation for supported use cases; honest solvent/OOD limits; no unsupported coverage guarantee under distribution shift; conclusions survive sensitivity checks or are explicitly qualified. Failure to generalize is reported, not hidden or repaired through test-driven retuning.

## Stage 6. Build publication-quality evidence — 5 hours

**Read first:** Stage 5 handoff, all selected run manifests, `research_paper.tex`, current generated supplements.

**Tasks and time:**

- 2 h: Generate a compact figure set from saved predictions: (1) cohort/exclusion and solvent/concentration coverage, (2) Task A parity/residuals and per-solvent errors for bonded/mobile/derived total, (3) matched model comparison including analytical baselines, (4) separate Task B bounds/smoothness ablations, (5) OOD/solvent-transfer errors with coverage and width. Reuse the existing learning curves in the supplement, labeled one-fold sensitivity rather than a universal data requirement.
- 1 h: Produce main tables for data/features, grouped measured performance, stress-test performance, and physical violations. Label units, task, prediction inputs, weighting, sample/group counts, and uncertainty type. Export vector PDF plus publication-resolution PNG; save exact plotting data and scripts.
- 1 h: Write a claim-to-evidence ledger linking each proposed statement to prediction/metric artifacts, distinguishing results, assumptions, and hypotheses. Keep legacy synthetic achievements in a historical/supplementary section.
- 1 h: Check all figure/table values against machine-readable results, ensure labels are readable, and write concise captions explaining the validation population and limits.

**Deliver:** `publication_2026/figures/`, `tables/`, `claims.md`, figure manifest with source hashes, and `stage_06/HANDOFF.md`.

**Success:** Every headline number is traceable and reproducible; tasks are visually separate; no chart implies seed SD is a confidence interval or synthetic rows are independent measurements. Main figures communicate the scientific conclusion without requiring the reader to reconstruct experiment history.

## Stage 7. Assemble manuscript and final handoff — 7 hours

**Read first:** Stage 6 handoff, claims ledger, current manuscript and supplements.

**Tasks and time:**

- 3 h: Create a new manuscript version under `publication_2026/manuscript/`, preserving the existing root manuscript. Rewrite title/abstract/results around the measured-data findings; integrate Task B as a separate section. Include data provenance, target definitions, input availability, nested selection, physical constraints, validation protocols, failed transfer, uncertainty limitations, and missing batch/measurement information. Treat no-new-data limitations as study boundaries, not unfinished promises.
- 1 h: Verify the small set of primary literature references needed for LLD applicability, dry-versus-wet thickness, bonded/mobile distinction, and the stated regularization method. Include only supported claims. Prepare PI questions on scientific interpretation, authorship, target venue, and whether existing metadata support a stronger claim; do not block the draft on a reply.
- 1.5 h: Save an environment lock/version record, reproduction guide, inference example, data dictionary, run manifest, and final selected-model procedure. If full-data final fits are made, label them untested full-data artifacts and preserve OOF research models separately. Distinguish deterministic replay from stochastic numerical tolerance. Do not execute fixed-output legacy commands against historical folders.
- 1 h: Run appropriate tests, hash/metric verification, and one representative end-to-end smoke reproduction into a new directory. Compile and visually inspect the manuscript PDF if the available TeX environment permits it; record any remaining build issue with source intact. Check artifact portability and all relative links.
- 0.5 h: Write `FINAL_HANDOFF.md` with completed stages, strongest defensible findings, exact artifact locations, unresolved limitations, and prioritized work after funding ends. Update `STATUS.md` with time spent and any deliberately deferred items.

**Deliver:** manuscript source/PDF where buildable, final figures/tables, reproducibility/inference guide, version manifest, verified model artifacts, and `FINAL_HANDOFF.md`.

**Success by December 4:** PI-reviewable manuscript with consistent claims and traceable measured-data results; reproducible code/artifacts; clear forecast/inference domains; no outstanding core leakage or provenance ambiguity hidden in the reported evidence. Submission remains subject to PI scientific review, authorship/venue decisions, and any required revisions.

## 5. Optional work, only if the core is ahead of schedule

Spend at most **10 extra hours**, in this order. These replace neither Stage 5 nor manuscript time.

1. **Training-only augmentation, 4 h.** Use resampling/perturbation derived solely from existing outer-training measurements; no new experimental information is implied. Reuse `constrained_bootstrap` with explicit parent IDs, generated anew inside each inner training partition. Compare measured-only, duplicate-bootstrap control, and constrained perturbation for one finalist; cap synthetic training weight and pick it only on inner validation. Test on measured data only. The saved audit file is not automatically valid for every fold. Without recoverable legacy parent lineage, do not use the precomputed 3k file for this experiment. Success is a valid measured-data comparison, including a null result.
2. **Split sensitivity or wider learning curves, 3 h.** Repeat one finalist/ablation under a small prespecified set of condition splits, or extend current learning curves beyond fold 1. Choose one, not both. Keep seeds separate from split variability and do not tune from these results.
3. **One targeted small-data model or measured-physics ablation, 3 h.** Choose either a modest regularized spline/Gaussian-process comparator if simpler curves leave systematic residuals, or velocity/LLD ablations only if all necessary pre-existing sample-linked inputs and validity evidence are already verified. Do not spend the final week searching for missing measurements, implementing a new simulator, or building new deep architectures.

**Not scheduled:** new laboratory experiments, external-test acquisition, unverified synthetic expansion, large architecture/optimizer sweeps, causal solvent-property discovery, reverse-design optimization outside supported ranges, new generative models, automatic submission/publication, or wholesale repository reorganization.

## 6. Compute budget and stopping rules

The saved November 11 per-model timestamps imply approximately **9.5 minutes total model-fit time for its 120 fits**, excluding preparation/plotting/verification. Per-fit averages were about 16.6 s for PINN, 4.2 s for bounded NN, 2.3 s for standard NN, 0.56 s for RF, and 0.10 s for GB. These are historical observations from that environment, not runtime guarantees for this machine or new settings.

CPU execution should be sufficient for this dataset and small networks; no GPU dependency is justified. The main constraint is research/engineering time. Benchmark one representative new fit first. Stage 3's 20 tree configurations × 3 inner folds × 5 outer folds × 2 targets is about 600 fits before refits; NN sweeps add hundreds of short fits. Reserve roughly 2–6 unattended CPU hours across the new campaigns, revise from the pilot, and cap optional search before it threatens writing time. Do not multiply every candidate by three seeds, every split, and both cohorts.

- If a search exceeds its predefined candidate or hands-on budget, finish the logged candidates and use the best valid inner-selected procedure; do not expand to chase a headline score.
- If a source-integrity or leakage issue appears, stop only the affected experiment, preserve outputs, fix the protocol, and issue a new run ID. Do not overwrite the evidence.
- If there is no >=5% stable accuracy improvement, retain the simpler model and invest remaining time in uncertainty, physical validity, and writing.
- If solvent holdouts remain worse than the training-mean reference, state that transfer is unsupported; do not market the model for unseen solvents.
- If fewer than 45 hours remain, omit the optional Task A NN implementation and extensive tuning, retain existing neural ablations, and preserve fair baselines, stress tests, and the paper package.
- Use the five-hour contingency for genuine blockers. At November 30, freeze numerical results unless correcting an error; December 1–4 is reserved for scientific synthesis and handoff.

## 7. Prompt for executing a stage in a later session

> Read `PUBLICATION_ROADMAP_2026-12-04.md`, `publication_2026/STATUS.md` if present, and the preceding stage's `HANDOFF.md`. Execute Stage [N] only, within its stated time/candidate budget and using the current repository state. Preserve all existing results and working code. Save new work in additive timestamped locations, verify its scientific and software integrity, and finish with the required handoff, exact artifact paths, remaining budget, and any deviations. Do not start the following stage automatically. If a prerequisite is missing, complete only the necessary prerequisite or document a bounded fallback; do not invent data or silently change the scientific question.

For Stage 1, no preceding handoff is required. The next authorized action should be Stage 1, not a full rerun of the repository.
