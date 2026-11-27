# Stage 1: regenerated evidence handoff

November 27, 2026. The earlier cloud-only commit `66c67d5` was not available in this Windows checkout and its public push had been blocked. This directory is a new, independently verified regeneration, not a claim that the inaccessible commit was recovered. No original tracked code or result was changed.

The source commit is `0ebeb04`. `input_manifest.json` records initial SHA256 hashes. All 29 archived runs verify, the 24-run campaign passes follow-up checks, and 12 PINN plus 13 near-term tests pass; see the logs in this directory. Windows supports the stored artifact separators natively, so no verifier patch is needed here.

The authoritative source tables reconcile exactly with canonical and prepared data: 142 original rows, excluding 15 toluene summaries and one physically invalid target, leaving 126 readings in 36 solvent/concentration conditions. Concentration is g/L despite legacy headers; thickness is nm. No exact duplicate complete measured-value rows occur. Raw input IDs, exclusions and updated assumed data are preserved in the Stage 2 run.

The updated experimental intake has unchanged measured inputs/targets and 126 estimated-property rows. Sample/batch IDs remain absent; solvent/concentration groups are a proxy, not proof of independence. The synthetic intake has 3,000 rows and unverified disjoint parent lineage; it is excluded from measured validation and training. Recorded source/output/builder hashes and all non-intake string columns are checked. New properties remain assumptions; LLD applicability is zero and withdrawal velocity constant. No hydrodynamic, velocity or viscosity-monotonicity loss is activated.

Task A uses only concentration, solvent descriptors or solvent identity to forecast bonded/mobile, with nonnegative projection and total as their sum. Measured total/uncoated and target-derived columns are prohibited. Task B uses measured total/uncoated and recipe inputs to infer bonded, with mobile derived from total. Never rank the tasks together or score B's supplied total as a prediction.

Primary error is sqrt(mean across conditions of mean reading squared error). Report condition MAE, row errors/R2, per-solvent results and violation rates too. Five grouped folds, three solvent holdouts, and low/high numeric 20% concentration holdouts are frozen. Inner groups are distinct from test groups. These are development tests already inspected historically, not newly untouched confirmation. Existing near-perfect synthetic scores are not primary experimental evidence.

Next: `../stage_02/HANDOFF.md`. Reproduce with `python -m publication_2026.recover_stage12`; each run gets a new timestamp and leaves old runs unchanged. Human labor hours were not tracked; measured execution time is recorded per run.
