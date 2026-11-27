"""Build reviewable Stage 3 tables and figures from immutable saved predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from publication_2026.common import ROOT,sha,dump

def table(d,index=True):
 if index:d=d.reset_index()
 def fmt(v):return f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v)
 return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))

def run():
 out=ROOT/'publication_2026/stage_03';r=ROOT/(out/'latest_run.txt').read_text().strip();base=ROOT/(ROOT/'publication_2026/stage_02/latest_run.txt').read_text().strip()
 m=pd.read_csv(r/'comparison_metrics.csv');ss=m.groupby(['protocol','procedure','target','solvent']).condition_rmse_nm.agg(['mean','std']).reset_index()
 b=pd.read_csv(base/'baseline_metrics.csv');choices=json.loads((r/'selected_configs_by_fold.json').read_text());fits=json.loads((r/'fits.json').read_text());record=json.loads((r/'run_record.json').read_text())
 selected=ss[ss.procedure.isin(['tree_selected','simple_selected'])]
 thresholds=[]
 for row in selected.itertuples():
  refs=b[(b.task=='A')&(b.protocol==row.protocol)&(b.target==row.target)&(b.solvent==row.solvent)&b.procedure.isin(['rf_balanced_pooled','rf_balanced_local','linear_local','saturating_mobile_local','scaled_lld_local'])]
  for ref in refs.itertuples():
   change=100*(row.mean/ref.condition_rmse_nm-1)
   thresholds.append(dict(protocol=row.protocol,procedure=row.procedure,target=row.target,solvent=row.solvent,reference=ref.procedure,new_mean_condition_rmse_nm=row.mean,reference_condition_rmse_nm=ref.condition_rmse_nm,change_percent=change,improvement_at_least_5_percent=bool(change<=-5),degradation_over_10_percent=bool(change>10)))
 threshold=pd.DataFrame(thresholds);threshold.to_csv(out/'decision_thresholds.csv',index=False)
 grouped=ss[(ss.protocol=='grouped_cv')&(ss.solvent=='all')].pivot(index='procedure',columns='target',values='mean')
 stress=ss[(ss.solvent=='all')&(ss.target=='mobile_nm')].pivot(index='procedure',columns='protocol',values='mean')
 per=ss[(ss.protocol=='grouped_cv')&(ss.solvent!='all')&ss.procedure.isin(['rf_balanced_pooled','tree_selected','simple_selected'])].pivot(index=['procedure','target'],columns='solvent',values='mean')
 selectors=[]
 for c in choices:
  if c['protocol']=='grouped_cv':selectors.append(dict(fold=c['fold'],target=c['target'],tree_selected=c['selected']['tree_selected'],simple_selected=c['selected']['simple_selected']))
 show=threshold[(threshold.reference=='rf_balanced_pooled')&(threshold.solvent=='all')][['protocol','procedure','target','change_percent','degradation_over_10_percent']]
 def bounds(obj):
  if isinstance(obj,dict):return int(obj.get('bound_hit',False))+sum(bounds(v) for k,v in obj.items() if k!='bound_hit')
  if isinstance(obj,list):return sum(bounds(v) for v in obj)
  return 0
 hits=sum(bounds(f['parameters']) for f in fits)
 report=f'''# Stage 3: nested recipe-only model comparison

Completed November 27, 2026. [Run artifacts](../runs/{r.name}/). [Frozen search space](search_space.json). [Handoff](HANDOFF.md).

## Result and decision

**The limited tree search produced a valid null result, not a general accuracy improvement.** The inner-selected RF/GB procedure has grouped-CV bonded/mobile/total condition RMSE of 0.10808 / 0.98902 / 0.96961 nm versus the fixed balanced pooled RF at 0.10482 / 0.90534 / 0.88937 nm. The tree selector worsens grouped-CV mobile error by about 9.24%. Nested RF alone gives only about 0.4% lower bonded error, below the 5% engineering threshold, and does not improve mobile error. Stop the search; do not expand the grid or choose settings after examining outer tests.

The simple/analytical mobile selector reaches 0.32211 nm grouped-CV mobile RMSE and 2.74081 nm high-concentration RMSE. It is better than the RF on those comparisons but worse than the already-available fixed local linear curve on grouped CV (0.22898 nm), and worse on low-concentration error (0.14179 versus 0.13426 nm). Most importantly, its leave-solvent-out error is 12.12920 nm versus 3.15154 nm for fixed RF. It is an exploratory known-solvent procedure, not a transfer solution. No candidate meets a universal improvement/no-degradation claim.

For the next uncertainty stage retain at most two Task A procedures: **fixed balanced pooled RF** (simpler reference retained under the prespecified null-result stop rule) and **simple_selected** (exactly the evaluated inner-selection procedure, with severe transfer limitations). Do not substitute a different curve per solvent based on outer results. The tree-selected route was evaluated as planned but is not promoted. Task B remains unchanged; carry its fixed RF and bounded NN as the two references for later constraint/uncertainty work. No PINN retraining or new loss activation was part of Stage 3.

## Scope and recovery

The original cloud-only Stage 1–2 commit `66c67d5` is unavailable here because its push was blocked and that workspace became unavailable. Stages 1–2 were regenerated from the unchanged source tables and archived predictions in `{base.name}`. Reported Stage 2 scores reproduce to displayed precision. This is a new timestamped Windows run, not a recovered commit. All original tracked files remain byte-identical. Git commits are currently blocked by the environment's read-only `.git` permission; the analysis files are saved locally and remain uncommitted.

## Prespecified method

- Same 126 measured readings / 36 solvent-concentration groups and saved record-ID partitions as Stage 2. No synthetic data and no new observations.
- Five grouped outer folds, all three solvent holdouts and low/high numeric concentration holdouts. Within each development set use three disjoint grouped inner folds. No outer-test outcomes are used for preprocessing, fitting, configuration selection or tie-breaking.
- Exactly 12 RF configurations per target: descriptor versus training-fitted solvent identity encoding, raw versus log1p concentration, and three leaf/depth pairs (1/unlimited, 2/6, 4/3). Exactly 8 GB configurations per target: the same encoding/transform choices and two regularization settings. Every model has 100 trees; this fixed compute choice differs from the 200-tree Stage 2 RF reference and is disclosed rather than called a pure leaf-size ablation.
- Search seed 42. Aggregate all inner OOF reading errors with equal condition weight; do not average fold RMSEs. Ties follow the declared candidate order. Refit selected settings with seeds 11/42/73 on full outer-development records. The same configuration is used for all final seeds; seeds do not trigger a new search.
- `tuned_rf` and `tuned_gb` are separately evaluated family-selection procedures. `tree_selected` chooses between all 20 settings per target inside each outer development set. `simple_selected` uses that same selected bonded tree and selects its mobile predictor among local linear, local scaled LLD, local saturation and one fixed residual hybrid using inner errors only.
- The hybrid first fits local scaled LLD to inner-training condition means, then fits a condition-weighted Ridge residual (alpha 10) on log1p concentration, fixed solvent descriptors and their interactions. Both components are fitted anew inside every inner fold. Residuals of training observations are allowed for training; validation labels never fit either component. Full selected procedures are refit on outer development. No extra residual hyperparameter grid was added.
- Identity categories come only from training solvents; unseen solvents encode as all-zero indicators. These cases are explicitly flagged in predictions. This defines executable behavior, not demonstrated chemical transfer. Descriptor and identity configurations are labeled individually in predictions and selection tables; no causal descriptor interpretation is warranted from three solvents.
- All ML uses inverse-condition-frequency training weights normalized to mean one; analytical curves use condition means. Local fitting requires at least two conditions; otherwise use pooled training. Task A features exclude measured total, uncoated and target-derived fields. Components are projected nonnegative and total computed by adding paired predictions. Raw outputs and violation metrics are retained.

The fixed search space SHA256 is `{record['search_space_sha256']}`. The run evaluated {record['inner_fits']} inner fits and {record['outer_fits']} distinct outer fits in {record['elapsed_seconds']:.2f} seconds on CPU. No GPU, dependency upgrade or full neural rerun was needed. Extra stress protocols use the same frozen search space, not additional settings.

## Grouped-CV comparison

Condition RMSE (nm), mean over final seeds for new procedures; fixed references have the original single seed or are deterministic. Missing component/total entries mean a curve only predicts mobile. Seed SD appears in `seed_summary.csv`; it is training variation, not a confidence interval.

{table(grouped)}

## Mobile stress tests

All errors are nm. Each protocol uses its own held-out conditions. Do not compare their difficulty as if sample support were identical.

{table(stress)}

## Solvent-resolved grouped-CV errors

{table(per)}

## Inner choices and stability

These are selections before each corresponding outer test is evaluated. The full RF/GB/simpler selection frequency is in `selection_frequency.csv`; all candidates' three inner-fold scores and OOF predictions are retained.

{table(pd.DataFrame(selectors),False)}

For mobile grouped CV, the selector chooses saturation in three folds, linear in one and the residual hybrid in one. It chooses the hybrid also for the toluene solvent holdout and low-concentration holdout. The hybrid is not a uniform improvement, and its inclusion does not validate the LLD mechanism. Across all stored fits, {hits} parameter-bound events occur in analytical local/fallback components (some are shared/repeated across seeds). Inspect `fits.json` for context before interpreting amplitudes/scales. They are phenomenological fitting parameters.

`training_diagnostics.csv` records per-solvent, per-target raw/log concentration association and repeated-condition spread using only each outer development set. It identifies singletons explicitly, records their unavailable within-condition SD as zero only for the descriptive summary, and does not estimate causal solvent effects. These diagnostics did not enlarge the frozen search or justify physics constraints.

## Engineering thresholds and paired evidence

Changes below are percentages of mean condition RMSE relative to the fixed balanced pooled RF; positive means worse. The full table includes every solvent and all declared fixed comparators in `decision_thresholds.csv`. The >10% flag is disclosure, not a significance test. Bonded high-concentration error rises about 10.5%; simple-selected solvent-transfer mobile error rises about 285%. No degradation is hidden by the overall grouped score.

{table(show,False)}

`paired_condition_differences.csv` contains each condition's new minus fixed-reference MSE for every final seed and applicable reference. Negative means improvement. Differences are paired on identical conditions and measured targets, not independent samples. No uncertainty/significance interval is claimed in this stage; that remains Stage 5. Baseline stochastic models retain their historical seed 42, so comparisons are against that fixed run, not a three-seed baseline distribution.

## Validation and limitations

All 36 tests pass: 25 existing tests plus 11 publication-stage tests. The new checks verify input/artifact hashes, frozen budgets, all inner/outer fit memberships, exact candidate selection from saved inner OOF errors, numeric concentration holdouts, preprocessing fitted on training data, unknown-category behavior, feature poisoning with test outcomes, hybrid fitting isolation, complete OOF coverage, paired totals, independent metrics and paired-difference reconstruction, and historical prediction preservation. All 29 archived run records and the 24-run campaign also verify. Detailed logs are retained.

The physical-property intake still contains estimates, not measurements. No hydrodynamic/velocity/viscosity-monotonicity loss is enabled. Unknown sample/batch IDs and only three solvents limit claims; grouped folds cannot establish batch independence. Outer results have been seen historically and during development, so subsequent finalist decisions are not an untouched-test confirmation. Negative R2 on low-variance subsets remains visible in CSVs. No universal accuracy improvement or publication readiness is claimed.
'''
 (out/'REPORT.md').write_text(report,encoding='utf-8')
 # Compact figures from saved metrics; no refitting.
 figs=out/'figures';figs.mkdir(exist_ok=True)
 names=['rf_balanced_pooled','tuned_rf','tuned_gb','tree_selected','simple_selected','linear_local']
 fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for ax,target,title in zip(axes,['bonded_nm','mobile_nm'],['Recipe-only bonded','Recipe-only mobile']):
  z=ss[(ss.protocol=='grouped_cv')&(ss.solvent=='all')&(ss.target==target)].set_index('procedure').loc[names]
  ax.barh(z.index,z['mean'],xerr=z['std'].fillna(0),color=['#003594']*4+['#997500','#777777']);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Condition RMSE (nm); seed SD for new procedures');ax.grid(axis='x',alpha=.2)
 for suffix in ['png','pdf']:fig.savefig(figs/f'grouped_comparison.{suffix}',dpi=180)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5),layout='constrained');protocols=['leave_one_solvent_out','ood_low_concentration','ood_high_concentration']
 for name in ['rf_balanced_pooled','tree_selected','simple_selected','linear_local','saturating_mobile_local']:
  z=ss[(ss.procedure==name)&(ss.solvent=='all')&(ss.target=='mobile_nm')].set_index('protocol').loc[protocols]
  ax.plot(range(3),z['mean'],marker='o',label=name)
 ax.set_xticks(range(3),['Unseen solvent','Low concentration','High concentration']);ax.set_yscale('log');ax.set_ylabel('Mobile condition RMSE (nm, log scale)');ax.set_title('Stress tests expose limited transfer');ax.legend(fontsize=8);ax.grid(alpha=.2)
 for suffix in ['png','pdf']:fig.savefig(figs/f'mobile_stress_tests.{suffix}',dpi=180)
 plt.close(fig)
 old=b.query('task=="A" and solvent=="all" and target=="mobile_nm"').pivot(index='procedure',columns='protocol',values='condition_rmse_nm')
 (ROOT/'publication_2026/stage_02/REPORT.md').write_text('# Regenerated Stage 2 results\n\nSee HANDOFF.md for provenance and regeneration limitations. All values below are mobile condition RMSE in nm. Source run: `'+base.name+'`.\n\n'+table(old)+'\n',encoding='utf-8')
 dump(out/'report_manifest.json',dict(source=Path(__file__).relative_to(ROOT).as_posix(),source_sha256=sha(__file__),run=r.relative_to(ROOT).as_posix(),artifacts={p.relative_to(out).as_posix():sha(p) for p in [out/'REPORT.md',out/'decision_thresholds.csv']+list(figs.iterdir())}))
 print(out/'REPORT.md')
if __name__=='__main__':run()
