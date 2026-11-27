"""Build traceable tables, scientific figures and Markdown from a completed campaign."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
MODELS=['pinn','bounded_nn','standard_nn','random_forest','gradient_boosting']
LABELS=['Bound + smoothness','Bound only','Unconstrained NN','Random forest','Gradient boosting']

def main(out):
    entries=json.loads((out/'completed_runs.json').read_text())
    plan=json.loads((out/'plan.json').read_text())
    if len(entries)!=len(plan['experiments']): raise ValueError('Campaign incomplete')
    prov=json.loads((out/'data_provenance.json').read_text())
    df=pd.read_csv(HERE.parent/prov['dataset'])
    figdir=out/'figures'; figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(name):
        plt.tight_layout()
        for ext in ['png','pdf']: plt.savefig(figdir/f'{name}.{ext}',dpi=200,bbox_inches='tight')
        plt.close()
    predictions=[]; runstats=[]; splits=[]
    for e in entries:
        folder=HERE/'runs'/e['run_id']
        p=pd.read_csv(folder/'predictions.csv')
        p=p.merge(df[['record_id','group_id','solvent']].rename_axis('source_row').reset_index(),on='source_row',validate='one_to_one')
        for key in ['seed','fold','fraction','label','run_id']: p[key]=e[key]
        predictions.append(p)
        sp=pd.read_csv(folder/'splits.csv'); sp['run_id']=e['run_id']; splits.append(sp)
        manifest=json.loads((folder/'manifest.json').read_text())
        groupcounts={f'n_{role}_conditions':df.iloc[g.source_row].group_id.nunique() for role,g in sp.groupby('split')}
        for m in e['metrics']: runstats.append({**{k:v for k,v in e.items() if k!='metrics'},**m,**{'n_'+k:v for k,v in manifest['rows'].items()},**groupcounts})
        plt.figure(figsize=(7,4))
        for m,label in zip(MODELS[:3],LABELS[:3]):
            h=pd.read_csv(folder/f'{m}_history.csv')
            plt.plot(h.epoch,h.val_loss,label=label)
        plt.yscale('log'); plt.xlabel('Training epoch'); plt.ylabel('Validation normalized squared error'); plt.legend(fontsize=8)
        save(e['run_id']+'_validation')
    pred=pd.concat(predictions,ignore_index=True); pred.to_csv(out/'all_predictions.csv',index=False)
    pd.DataFrame(runstats).to_csv(out/'run_metrics.csv',index=False)
    pd.concat(splits).to_csv(out/'all_splits.csv',index=False)
    cv=pred[pred.label=='grouped_cv']
    scores=[]
    for seed,p in cv.groupby('seed'):
        assert len(p)==len(df) and p.record_id.nunique()==len(df)
        for m in MODELS:
            err=(p[m]-p.observed_nm)**2
            group_mse=err.groupby(p.group_id).mean()
            scores.append(dict(seed=seed,model=m,group_rmse_nm=np.sqrt(group_mse.mean()),rmse_nm=np.sqrt(err.mean()),
                violation_fraction=((p[m]<-1e-6)|(p[m]>p.total_nm+1e-6)).mean()))
    scores=pd.DataFrame(scores); scores.to_csv(out/'cv_seed_metrics.csv',index=False)
    summary=scores.groupby('model').agg(group_rmse_mean=('group_rmse_nm','mean'),group_rmse_sd=('group_rmse_nm','std'),row_rmse_mean=('rmse_nm','mean'),violation_fraction=('violation_fraction','mean')).reindex(MODELS)
    summary.to_csv(out/'cv_summary.csv')
    plt.figure(figsize=(9,4.5)); x=np.arange(5)
    plt.bar(x,summary.group_rmse_mean,yerr=summary.group_rmse_sd,capsize=4,color=['#B49A45','#6F8FAE','#999999','#002B5E','#507A61'])
    plt.xticks(x,LABELS,rotation=12,ha='right');plt.ylabel('Condition-weighted RMSE (nm)');plt.title('Five grouped folds; mean and SD across 3 initialization seeds')
    save('cv_model_comparison')
    p=cv[cv.seed==42]
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,m,label in zip(axes,['pinn','gradient_boosting'],[LABELS[0],LABELS[4]]):
        for solvent,g in p.groupby('solvent'): ax.scatter(g.observed_nm,g[m],label=solvent.replace('_',' '),s=20,alpha=.7)
        limits=[min(p.observed_nm.min(),p[m].min()),max(p.observed_nm.max(),p[m].max())]
        ax.plot(limits,limits,'k--',lw=1);ax.set(xlabel='Measured bonded thickness (nm)',ylabel='Predicted bonded thickness (nm)',title=label);ax.legend(fontsize=8)
    save('parity_seed42')
    solvent_scores=[]
    for (seed,solvent),p in cv.groupby(['seed','solvent']):
        for m in MODELS: solvent_scores.append(dict(seed=seed,solvent=solvent,model=m,group_rmse_nm=np.sqrt(((p[m]-p.observed_nm)**2).groupby(p.group_id).mean().mean())))
    solvent_scores=pd.DataFrame(solvent_scores);solvent_scores.to_csv(out/'solvent_metrics.csv',index=False)
    solvent_scores.groupby(['solvent','model']).group_rmse_nm.mean().unstack()[MODELS].rename(columns=dict(zip(MODELS,LABELS))).plot.bar(figsize=(10,5))
    plt.ylabel('Condition-weighted RMSE (nm)');plt.xlabel('Solvent represented during training');plt.xticks(rotation=0);plt.legend(fontsize=8);save('solvent_errors')
    learning=pred[(pred.label=='learning_curve')|((pred.label=='grouped_cv')&(pred.fold==0))]
    ls=[]
    for (seed,fraction),p in learning.groupby(['seed','fraction']):
        for m in MODELS: ls.append(dict(seed=seed,fraction=fraction,model=m,group_rmse_nm=np.sqrt(((p[m]-p.observed_nm)**2).groupby(p.group_id).mean().mean())))
    ls=pd.DataFrame(ls);ls.to_csv(out/'learning_curve_metrics.csv',index=False)
    plt.figure(figsize=(8,4.5))
    for m,label in zip(MODELS,LABELS):
        s=ls[ls.model==m].groupby('fraction').group_rmse_nm.agg(['mean','std']);plt.errorbar(s.index*100,s['mean'],yerr=s['std'],marker='o',capsize=3,label=label)
    plt.xlabel('Requested fraction of training conditions (%)');plt.ylabel('Condition-weighted RMSE (nm)');plt.legend(fontsize=8);plt.title('Same fold-1 test and validation conditions at every fraction');save('learning_curve')
    paired=[]
    for (seed,fold),p in cv.groupby(['seed','fold']):
        vals={m:np.sqrt(((p[m]-p.observed_nm)**2).groupby(p.group_id).mean().mean()) for m in ['pinn','bounded_nn']}
        paired.append(dict(seed=seed,fold=fold+1,delta_rmse_nm=vals['pinn']-vals['bounded_nn']))
    paired=pd.DataFrame(paired);paired.to_csv(out/'paired_curvature_effect.csv',index=False)
    (out/'presentation_data.json').write_text(json.dumps(dict(summary=summary.reset_index().to_dict('records'),
        seeds=scores.to_dict('records'),paired=paired.to_dict('records'),
        learning=ls.to_dict('records'),provenance=prov,campaign=out.name),indent=2))
    plt.figure(figsize=(8,4.5))
    for seed,g in paired.groupby('seed'):plt.plot(g.fold,g.delta_rmse_nm,'o-',label=f'Seed {seed}')
    plt.axhline(0,color='black',lw=1);plt.xlabel('Outer fold');plt.ylabel('RMSE: smoothness minus bound only (nm)');plt.legend();plt.title('Below zero favors adding smoothness');save('curvature_effect')
    plt.figure(figsize=(8,4.5))
    for solvent,g in df.groupby('solvent'):plt.scatter(g.concentration_g_l,g.bonded_nm,label=solvent.replace('_',' '),alpha=.65)
    plt.xscale('log');plt.xlabel('Concentration (g/L)');plt.ylabel('Measured bonded thickness (nm)');plt.legend();save('measured_data')
    body=f'''# Li Lab research follow-up: November 11, 2026

## Outcome and scope

Executed {len(entries)} recorded runs ({len(entries)*5} model fits) on 126 existing measured records spanning 36 solvent/concentration conditions. These are computational experiments, not newly collected laboratory measurements. Campaign: `{out.name}`. Runtime UTC and New York timestamps, exact source, configuration, environment, checkpoints, splits and artifact hashes are retained in each run under `PINN/runs/`.

The October 14 update (`10-14_li.pptx`, slides 22-23) requested dataset alignment, separate loss comparisons, repeated/grouped validation and data-efficiency tests. This work prepares and evaluates the available measured data. Additional velocity and hydrodynamic penalties remain blocked by missing sample-linked, validated physics inputs in this checkout. The prior statement that the values are available is preserved; the missing step is bringing those values into a traceable table, not asserting they do not exist.

## Data preparation and exclusions

Reused the reviewed `near_term/run_analysis.py` canonical loader rather than inventing sample matches. Combined 142 source records: hexane 58, toluene 51, ethyl acetate 33. Excluded 15 potentially overlapping toluene summaries and one physically inconsistent toluene record (0.86 nm bonded > 0.85 nm total). Primary cohort: hexane 58, toluene 35, ethyl acetate 33. No clipping, imputation or synthetic augmentation. Raw files and old results were preserved.

Concentration is g/L, consistent with the earlier workbook-cell audit; legacy g/mL headings were erroneous. Values are unchanged. Source path/row, stable record ID and condition group are retained in the prepared snapshot. `data_provenance.json`, `canonical_all.csv`, `excluded_records.csv` and `workbook_inventory.json` preserve this audit. `physics_cell_evidence.json` records the targeted workbook search (no matching cells). Existing solvent descriptors and assumed 1 mm/s velocity from the analytical scripts are not promoted to measured solution properties. `physics_measurement_intake.csv` provides 126 pre-keyed rows for matching the remaining measurements.

## Changes

- `train.py`: deterministic five-fold grouped outer validation with a fixed training/validation split, independent of network initialization seed; opt-in whole-condition training subsampling with nested subsets.
- `core.py`: configurable smoothness columns so derivatives of solvent indicator columns are excluded. Existing default behavior is unchanged.
- `run_tracking.py`: model version 0.3.0; existing run snapshots remain immutable.
- `run_followup.py`: preparation, fixed experiment plan, immutable timestamped runs and incremental completion record.
- `summarize_followup.py`: prediction aggregation, condition-weighted scores, figures and this report.
- `test_followup.py`: group separation, full outer-test coverage, seed-independent splits and invalid smoothness-column checks. Tracking test follows the declared version.

## Experiment design

Inputs: concentration (natural log), measured uncoated thickness, measured total thickness, and three solvent indicators. Target: bonded thickness in nm. All scaling uses training rows only. This is conditional film-state inference, not prediction from a coating recipe alone. The six-input network has two 16-unit tanh layers. Fixed settings: at most 1,500 epochs, validation patience 150, learning rate 0.001, weight decay 0.0001, curvature coefficient 0.001; no tuning on outer-test results.

Five models share each split: bounded network with smoothness (PINN), same bounded network without the penalty, unconstrained network, 300-tree random forest and 100-stage gradient boosting. PINN versus bounded NN isolates the added curvature prior. Physical compliance is reported separately from accuracy. Smoothness applies only to continuous concentration/uncoated/total inputs, not solvent indicators. Velocity, LLD and monotonicity weights remain zero.

Five outer folds each hold out entire solvent/concentration conditions. Every record receives one out-of-fold prediction per seed. Initialization seeds 11, 42 and 73 use the same outer and inner splits: 15 runs. Nine additional runs use 25%, 50% and 75% of fold-1 training conditions at each seed; the full-data fold-1 runs supply the 100% endpoint. Ceil rounding and a minimum two conditions determine the actual subsets; exact membership is in splits.csv. Validation and test membership are unchanged across fractions. No synthetic parent can cross a split because no synthetic records are used.

## Results

Primary error gives each condition equal weight: average squared error within each condition, average across conditions, then square root. Mean and SD below summarize three initialization seeds on fixed folds, not confidence intervals or independent replications.

| Model | Condition RMSE mean (nm) | Seed SD (nm) | Row RMSE mean (nm) | Bounds violations |
|---|---:|---:|---:|---:|
'''
    for m,label in zip(MODELS,LABELS):
        r=summary.loc[m];body+=f'| {label} | {r.group_rmse_mean:.4f} | {r.group_rmse_sd:.4f} | {r.row_rmse_mean:.4f} | {r.violation_fraction:.1%} |\n'
    body+=f'''
Adding smoothness improved condition-weighted fold RMSE in {int((paired.delta_rmse_nm<0).sum())} of 15 paired fold/seed comparisons; mean paired difference was {paired.delta_rmse_nm.mean():+.4f} nm (negative favors smoothness). These comparisons share data and are not 15 independent experiments. See `paired_curvature_effect.csv` for all differences. Do not equate compliance with accuracy or claim a mechanistic law from these fits.

## Figures and tables

`figures/` contains PNG and vector PDF versions of measured-data coverage, model comparison, seed-42 parity, solvent-resolved errors, paired curvature effects, fixed-test learning curves and all 24 run validation histories. Parity seed 42 was fixed for illustration, not chosen for best results. Machine-readable scores and all source-linked predictions accompany the figures. Solvent breakdowns evaluate represented solvents, not leave-one-solvent-out transfer.

## Remaining dependencies and decisions

1. Match measured withdrawal speed, solution viscosity/surface tension/density, independent wet-to-dry conversion and LLD regime justification to record/experiment IDs using the intake. Record sources and units; do not derive the conversion from bonded target labels. Then run velocity-only, LLD-only and combined ablations on the same splits.
2. Obtain actual batch/specimen IDs. Current solvent/concentration groups are conservative proxies and cannot prove independence between concentrations prepared together.
3. Reserve genuinely new laboratory measurements for a final independent test. This campaign reuses existing data and cannot satisfy that requirement.
4. Adsorption/saturation priors need justified boundary/background definitions and identifiable parameters. No unverified adsorption law was imposed.
5. Synthetic augmentation remains deferred until parent lineage and generator validity are established. The completed measured-only learning curves establish the reference comparison.

## Reproduction

From repository root: `python -m unittest discover -s PINN -p "test*.py" -v`, then `python PINN/run_followup.py` (creates a new timestamped campaign; does not overwrite this one), then `python PINN/summarize_followup.py PINN/campaigns/<new-campaign>`. `python PINN/verify_runs.py` verifies all archived run hashes and scores. Individual runs can be reproduced using their archived config with `PINN/train.py`, the prepared dataset and a new output directory, including `--train-fraction` for reduced-data runs.

The local Windows test environment required TEMP/TMP to point to a writable temporary directory; the initial two temp-file failures were environmental and are recorded separately from scientific test outcomes. See `tests.log` and `verification.log` for final validation.
'''
    body+='\n## Complete run inventory\n\nEach linked folder contains results for all five models; error below is row-weighted PINN RMSE for that individual test fold, not the pooled condition-weighted summary above.\n\n| Run | Fold | Seed | Training fraction | Training rows | Training conditions | PINN RMSE (nm) |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in runstats:
        if r['model']=='pinn':
            body+=f"| [{r['run_id']}](../../runs/{r['run_id']}/run_record.json) | {r['fold']+1} | {r['seed']} | {r['fraction']:.0%} | {r['n_train']} | {r['n_train_conditions']} | {r['RMSE_nm']:.4f} |\n"
    (out/'RESEARCH_UPDATE.md').write_text(body,encoding='utf-8')
    print(summary.to_string());print('Paired curvature:',paired.delta_rmse_nm.mean())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('campaign',type=Path);main(parser.parse_args().campaign)
