"""Regenerate unavailable Stage 1/2 deliverables, preserving all tracked history."""
import time,json
import numpy as np
import pandas as pd
from publication_2026.common import *
BASELINES={
 'mean_pooled':{'family':'mean'},'mean_local':{'family':'mean','scope':'local'},
 'ridge_balanced':{'family':'ridge'},'rf_balanced_pooled':{'family':'rf'},
 'rf_balanced_local':{'family':'rf','scope':'local'},'gb_balanced':{'family':'gb'},
 'linear_pooled':{'family':'linear'},'linear_local':{'family':'linear','scope':'local'},
 'fixed_lld':{'family':'fixed_lld'},'scaled_lld_pooled':{'family':'scaled_lld'},
 'scaled_lld_local':{'family':'scaled_lld','scope':'local'},
 'saturating_mobile_pooled':{'family':'saturating_mobile'},'saturating_mobile_local':{'family':'saturating_mobile','scope':'local'}}

def run():
 start=time.monotonic();out=new_run('AB_stage02_regenerated');d=load_audit(out);save_splits(d,out);pred=[];fits=[]
 dump(out/'benchmark_spec.json',dict(recovery='Original cloud commit 66c67d5 unavailable; this is an independently regenerated Windows run, not that commit or byte-identical outputs.',
   candidates=BASELINES,features=FEATURES,seed_A=42,seeds_B=[11,42,73],weighting='Inverse condition frequency normalized to mean one; analytical condition means. Historical RF and B use row weighting.',
   training='A full development set; B archived train only, validation for neural checkpoints.',fallback='Local requires >=2 conditions; otherwise pooled.',projection='A nonnegative components, total from sum; B archived outputs unchanged.',synthetic='Excluded'))
 old=pd.read_csv(ROOT/'near_term/results/predictions.csv');old=old[(old.cohort=='primary')&(old.model=='rf_raw')]
 for protocol,fold,tr,te,_,_ in partitions(d):
  print('baseline',protocol,fold,flush=True);train,test=d.iloc[tr],d.iloc[te]
  for name,cfg in BASELINES.items():
   for target in (['mobile_nm'] if cfg['family'] in ['fixed_lld','scaled_lld','saturating_mobile'] else ['bonded_nm','mobile_nm']):
    f=Fitted(train,target,cfg);fid=f'A_{protocol}_{fold}_{name}_{target}'
    fits.append(dict(fit_id=fid,protocol=protocol,fold=fold,**f.record()))
    pred.append(rows(test,f.predict(test),f.predict_raw(test),'A',protocol,fold,name,target,42,fid))
  for target in ['bonded_nm','mobile_nm']:
   p=old[(old.protocol==protocol)&(old.fold.astype(str)==fold)&(old.target==target)].set_index('record_id').loc[test.record_id]
   np.testing.assert_allclose(p.actual,test[target],atol=1e-12);fid=f'A_{protocol}_{fold}_rf_raw_reused_{target}'
   pred.append(rows(test,p.prediction.to_numpy(),p.prediction.to_numpy(),'A',protocol,fold,'rf_raw_reused',target,42,fid))
   fits.append(dict(fit_id=fid,protocol=protocol,fold=fold,target=target,train_ids=list(train.record_id),weights=[1.]*len(train),source='near_term/results/predictions.csv'))
 old=pd.read_csv(CAMP/'all_predictions.csv');old=old[old.label=='grouped_cv'];sp=pd.read_csv(CAMP/'all_splits.csv')
 for (fold,seed),g in old.groupby(['fold','seed']):
  rid=g.run_id.iloc[0];s=sp[sp.run_id==rid];train=d.iloc[s.loc[s['split']=='train','source_row'].to_numpy(int)];test=d.set_index('record_id',drop=False).loc[g.record_id]
  np.testing.assert_allclose(g.observed_nm,test.bonded_nm,atol=1e-12)
  alpha=float(np.clip(np.dot(train.total_nm,train.bonded_nm)/np.dot(train.total_nm,train.total_nm),0,1))
  for name in ['pinn','bounded_nn','standard_nn','random_forest','gradient_boosting','constant_fraction']:
   p=g[name].to_numpy() if name!='constant_fraction' else alpha*test.total_nm.to_numpy();fid=f'B_{fold+1}_{seed}_{name}'
   fits.append(dict(fit_id=fid,protocol='grouped_cv',fold=str(fold+1),target='bonded_nm',train_ids=list(train.record_id),weights=[1.]*len(train),source=rid,alpha=alpha if name=='constant_fraction' else None))
   pred.append(rows(test,p,p,'B','grouped_cv',fold+1,name,'bonded_nm',seed,fid))
   q=test.total_nm.to_numpy()-p;pred.append(rows(test,q,q,'B','grouped_cv',fold+1,name,'mobile_nm',seed,fid))
 p=paired_totals(pd.concat(pred,ignore_index=True));p.to_csv(out/'predictions.csv',index=False);summarize(p).to_csv(out/'baseline_metrics.csv',index=False);dump(out/'fits.json',fits)
 finish(out,start);(ROOT/'publication_2026/stage_02/latest_run.txt').write_text(out.relative_to(ROOT).as_posix()+'\n');print(out,flush=True)
if __name__=='__main__':run()
