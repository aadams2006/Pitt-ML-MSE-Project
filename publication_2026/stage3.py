"""Stage 3: prespecified nested grouped search on recipe-only inputs.

Run python -m publication_2026.stage3. Outputs are additive. No neural training.
"""
import time,json,shutil
import numpy as np
import pandas as pd
from publication_2026.common import *

def choose(scores,order):
    # Every validation condition gets equal weight, irrespective of inner fold size.
    return min(order,key=lambda cid:(scores[cid],order.index(cid)))

def diagnostics(train,protocol,fold):
    result=[]
    for solvent,g in train.groupby('solvent'):
        for target in ['bonded_nm','mobile_nm']:
            condition=g.groupby('group_id').agg(c=('concentration_g_l','first'),y=(target,'mean'),sd=(target,'std'),n=(target,'size'))
            for transform,x in [('raw',condition.c),('log1p',np.log1p(condition.c))]:
                corr=float(np.corrcoef(x,condition.y)[0,1]) if x.std()>0 and condition.y.std()>0 else None
                result.append(dict(protocol=protocol,fold=fold,solvent=solvent,target=target,transform=transform,training_rows=len(g),training_conditions=len(condition),
                    mean_within_condition_sd_nm=float(condition.sd.fillna(0).mean()),max_within_condition_sd_nm=float(condition.sd.fillna(0).max()),
                    singleton_conditions=int((condition.n==1).sum()),train_condition_correlation=corr,
                    interpretation='Descriptive outer-training association, not causal or a physical-law test.'))
    return result

def run():
    start=time.monotonic();out=new_run('A_stage03_nested');specpath=ROOT/'publication_2026/stage_03/search_space.json';spec=json.loads(specpath.read_text(encoding='utf-8-sig'))
    shutil.copyfile(specpath,out/'search_space.json');dump(out/'started.json',dict(spec_sha256=sha(specpath),started_utc=datetime.now(timezone.utc).isoformat()))
    base=ROOT/(ROOT/'publication_2026/stage_02/latest_run.txt').read_text().strip()
    d=pd.read_csv(base/'measured_snapshot.csv');splits=pd.read_csv(base/'splits.csv',dtype={'fold':str});inner=pd.read_csv(base/'inner_splits.csv',dtype={'fold':str})
    shutil.copyfile(base/'splits.csv',out/'splits.csv');shutil.copyfile(base/'inner_splits.csv',out/'inner_splits.csv')
    dump(out/'inputs.json',{str(p.relative_to(ROOT)):sha(p) for p in [base/'measured_snapshot.csv',base/'predictions.csv',base/'splits.csv',base/'inner_splits.csv',specpath]})
    dump(out/'feature_contract.json',dict(inputs='concentration_g_l; fixed solvent descriptors OR training-only solvent one-hot categories',
        forbidden=['total_nm','bonded_nm','mobile_nm','uncoated_nm','physical_valid','primary_eligible','assumed physics/target diagnostics'],
        unseen_identity='all-zero solvent indicators, no new category fitted on test records',parameters='Every fitted object uses inner-train or outer-development records only.'))
    choices=[];score_rows=[];inner_preds=[];outer_preds=[];fit_records=[];train_diagnostics=[]
    configs={c['id']:c for c in spec['candidates']+spec['simple_mobile']}
    rf=[c['id'] for c in spec['candidates'] if c['family']=='rf'];gb=[c['id'] for c in spec['candidates'] if c['family']=='gb'];simple=[c['id'] for c in spec['simple_mobile']]
    for (protocol,fold),partition in splits.groupby(['protocol','fold'],sort=False):
        print('START',protocol,fold,flush=True)
        dev=d[d.record_id.isin(partition[partition.role!='test'].record_id)];test=d[d.record_id.isin(partition[partition.role=='test'].record_id)]
        train_diagnostics.extend(diagnostics(dev,protocol,fold));selected={}
        for target in spec['targets']:
            active=rf+gb+(simple if target=='mobile_nm' else []);scores={}
            for cid in active:
                parts=[];cfg=configs[cid]
                for k,ip in inner[(inner.protocol==protocol)&(inner.fold==fold)].groupby('inner_fold'):
                    train=d[d.record_id.isin(ip[ip.role=='train'].record_id)];valid=d[d.record_id.isin(ip[ip.role=='validation'].record_id)]
                    assert set(train.group_id).isdisjoint(set(valid.group_id)|set(test.group_id))
                    f=Fitted(train,target,cfg,42);fid=f'inner_{protocol}_{fold}_{k}_{target}_{cid}'
                    fit_records.append(dict(fit_id=fid,role='inner',protocol=protocol,fold=fold,inner_fold=int(k),**f.record()))
                    p=rows(valid,f.predict(valid),f.predict_raw(valid),'A',protocol,fold,cid,target,42,fid);p['inner_fold']=k
                    parts.append(p);inner_preds.append(p)
                    score_rows.append(dict(protocol=protocol,fold=fold,target=target,candidate=cid,inner_fold=int(k),**metric(p)))
                allp=pd.concat(parts,ignore_index=True);scores[cid]=metric(allp)['condition_rmse_nm']**2
            selected[target]={'tuned_rf':choose(scores,rf),'tuned_gb':choose(scores,gb),'tree_selected':choose(scores,rf+gb)}
            selected[target]['simple_selected']=choose(scores,simple) if target=='mobile_nm' else selected[target]['tree_selected']
            choices.append(dict(protocol=protocol,fold=fold,target=target,selected=selected[target],inner_condition_mse=scores))
            print('SELECT',target,selected[target],flush=True)
        for seed in spec['final_seeds']:
            for target in spec['targets']:
                cache={}
                for procedure,cid in selected[target].items():
                    if cid not in cache:
                        f=Fitted(dev,target,configs[cid],seed);fid=f'outer_{protocol}_{fold}_{target}_{cid}_{seed}'
                        fit_records.append(dict(fit_id=fid,role='outer',protocol=protocol,fold=fold,inner_fold=None,**f.record()));cache[cid]=(f,fid)
                    f,fid=cache[cid];p=rows(test,f.predict(test),f.predict_raw(test),'A',protocol,fold,procedure,target,seed,fid)
                    p['selected_config']=cid;p['encoding']=configs[cid].get('encoding','analytical_or_descriptor_residual')
                    p['unseen_solvent']=~test.solvent.isin(dev.solvent.unique()).to_numpy();outer_preds.append(p)
        # Preserve completed folds if interrupted; no historical outputs are touched.
        dump(out/'progress.json',dict(completed_protocol=protocol,completed_fold=fold,elapsed_seconds=time.monotonic()-start))
    p=paired_totals(pd.concat(outer_preds,ignore_index=True));p.to_csv(out/'predictions.csv',index=False)
    metrics=summarize(p);metrics.to_csv(out/'metrics.csv',index=False)
    pd.concat(inner_preds,ignore_index=True).to_csv(out/'inner_predictions.csv',index=False)
    pd.DataFrame(score_rows).to_csv(out/'inner_scores.csv',index=False);dump(out/'selected_configs_by_fold.json',choices);dump(out/'fits.json',fit_records)
    pd.DataFrame(train_diagnostics).to_csv(out/'training_diagnostics.csv',index=False)
    freq=[]
    for c in choices:
        for procedure,cid in c['selected'].items():freq.append(dict(protocol=c['protocol'],target=c['target'],procedure=procedure,candidate=cid))
    pd.DataFrame(freq).groupby(['protocol','target','procedure','candidate']).size().rename('fold_count').reset_index().to_csv(out/'selection_frequency.csv',index=False)
    # Every prespecified fixed reference is reported; no test-selected solvent winner.
    basep=pd.read_csv(base/'predictions.csv',dtype={'fold':str});basep=basep[(basep.task=='A')&basep.procedure.isin(spec['fixed_references'])]
    combined=pd.concat([p,basep],ignore_index=True);summarize(combined).to_csv(out/'comparison_metrics.csv',index=False)
    newcond=p.assign(squared=(p.actual-p.prediction)**2).groupby(['protocol','procedure','target','seed','group_id','solvent']).squared.mean().rename('new_condition_mse').reset_index()
    refcond=basep.assign(squared=(basep.actual-basep.prediction)**2).groupby(['protocol','procedure','target','group_id','solvent']).squared.mean().rename('reference_condition_mse').reset_index().rename(columns={'procedure':'reference'})
    newcond.merge(refcond,on=['protocol','target','group_id','solvent']).assign(delta_condition_mse=lambda x:x.new_condition_mse-x.reference_condition_mse).to_csv(out/'paired_condition_differences.csv',index=False)
    metrics.groupby(['protocol','procedure','target','solvent']).condition_rmse_nm.agg(['mean','std']).reset_index().to_csv(out/'seed_summary.csv',index=False)
    finish(out,start,dict(search_space_sha256=sha(specpath),baseline_run=base.relative_to(ROOT).as_posix(),inner_fits=sum(f['role']=='inner' for f in fit_records),outer_fits=sum(f['role']=='outer' for f in fit_records)))
    (ROOT/'publication_2026/stage_03/latest_run.txt').write_text(out.relative_to(ROOT).as_posix()+'\n');print('DONE',out,flush=True)
if __name__=='__main__':run()
