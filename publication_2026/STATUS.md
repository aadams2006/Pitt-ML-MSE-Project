# Publication preparation status

Updated November 27, 2026. **Stages 1–3 analyses complete locally; changes are not committed. Stages 4–7 not started.**

- [Stage 1 evidence handoff](stage_01/HANDOFF.md)
- [Stage 2 regenerated baseline report](stage_02/REPORT.md) and [handoff](stage_02/HANDOFF.md)
- [Stage 3 nested-search report](stage_03/REPORT.md) and [next-session handoff](stage_03/HANDOFF.md)
- [Original roadmap](../PUBLICATION_ROADMAP_2026-12-04.md), unchanged

The cloud-only Stage 1–2 commit was unavailable in this checkout. Its scientific prerequisites were regenerated from unchanged sources and verified, with new run provenance. The Stage 2 reported scores reproduce to displayed precision.

Stage 3 evaluated the frozen 12-setting RF and 8-setting GB search per target, three inner group folds, ten outer/stress partitions and final seeds 11/42/73, plus one analytical/residual candidate. Tree tuning did not provide a robust gain. The simple-model selector is useful on known-solvent CV/high-end tests but fails unseen-solvent transfer. Full null and negative results are retained.

36 tests pass; all 29 archived run records and 24 campaign runs verify. All original tracked files are preserved. No new data, synthetic augmentation, neural retraining or new physics losses were introduced. All outputs, predictions, fit memberships, parameters, metrics and figures are saved in this repository.

**Commit blocker:** `.git` is read-only in the execution environment, and escalation is disabled. No commit/push succeeded. See the Stage 3 handoff for exact local commit commands once Git metadata is writable. Do not confuse this with the separate earlier block on publishing research artifacts to a public remote.
