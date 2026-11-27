"""Shared measured-data contracts; no historical files are modified."""
from pathlib import Path
import sys,json,hashlib,subprocess,platform,importlib.metadata as metadata
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'near_term'))
import run_analysis as legacy
import solvent_analysis as analytical
CAMP=ROOT/'PINN/campaigns/20261111T134229Z_measured_followup'
ASSUMED=ROOT/'PINN/datasets/assumptions_2026-11-27'
FEATURES=['concentration_g_l']+legacy.DESCRIPTORS

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def new_run(label):
    out=ROOT/'publication_2026/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+label)
    out.mkdir(parents=True,exist_ok=False);return out

def metric(p):
    e=p.actual-p.prediction
    g=pd.DataFrame({'group_id':p.group_id,'squared':e**2,'absolute':abs(e)}).groupby('group_id').mean()
    denominator=np.sum((p.actual-p.actual.mean())**2)
    return dict(condition_rmse_nm=float(np.sqrt(g.squared.mean())),condition_mae_nm=float(g.absolute.mean()),
        row_rmse_nm=float(np.sqrt(np.mean(e**2))),row_mae_nm=float(np.mean(abs(e))),
        r2=float(1-np.sum(e**2)/denominator) if denominator>0 else np.nan,n=len(p),conditions=len(g))

def summarize(p):
    rows=[]
    for key,g in p.groupby(['task','protocol','procedure','target','seed']):
        for solvent,part in [('all',g)]+list(g.groupby('solvent')):
            v=np.maximum(0,-part.raw_prediction)
            if key[0]=='B':v=np.maximum(v,part.raw_prediction-part.total_observed)
            rows.append(dict(zip(['task','protocol','procedure','target','seed'],key),solvent=solvent,**metric(part),
                raw_violation_fraction=float(np.mean(v>1e-6)),raw_max_violation_nm=float(max(0,v.max()))))
    return pd.DataFrame(rows)

def weights(d):
    w=1/d.groupby('group_id').record_id.transform('size').to_numpy(float);return w/w.mean()

def make_features(d,config,categories=None):
    x=pd.DataFrame(index=d.index)
    c=d.concentration_g_l.to_numpy(float)
    x['concentration']=np.log1p(c) if config.get('transform')=='log' else c
    if config.get('encoding','descriptors')=='identity':
        categories=sorted(d.solvent.unique()) if categories is None else categories
        for s in categories:x['solvent_'+s]=(d.solvent==s).astype(float)
    else:
        for f in legacy.DESCRIPTORS:x[f]=d[f].to_numpy(float)
    return x

class Fitted:
    """Fitted procedure; prediction reads recipe inputs only."""
    def __init__(self,train,target,config,seed=42):
        self.config=dict(config);self.target=target;self.seed=seed
        self.ids=list(train.record_id);self.categories=sorted(train.solvent.unique());self.parts={};self.params={}
        self.w=weights(train);family=config['family']
        if config.get('scope')=='local':
            cfg=dict(config,scope='pooled')
            self.fallback=Fitted(train,target,cfg,seed)
            for solvent,local in train.groupby('solvent'):
                if local.group_id.nunique()>=2:self.parts[solvent]=Fitted(local,target,cfg,seed)
            self.params={'local':{s:m.params for s,m in self.parts.items()},'pooled_fallback':self.fallback.params}
            return
        if family in ['scaled_lld','saturating_mobile','fixed_lld']:
            assert target=='mobile_nm';self.params=analytical.fit_comparator(train,family);return
        if family=='hybrid':
            assert target=='mobile_nm'
            # Both components see only the same training partition.
            self.base=Fitted(train,target,{'family':'scaled_lld','scope':'local'},seed)
            residual=train.copy();residual[target]=train[target]-self.base.predict_raw(train)
            self.residual=Fitted(residual,target,{'family':'ridge','encoding':'descriptors','transform':'log','alpha':10.,'interactions':True},seed)
            self.params={'analytical':self.base.params,'residual':self.residual.params};return
        y=train[target].to_numpy(float)
        if family=='mean':self.params={'mean':float(np.average(y,weights=self.w))};return
        x=self.x(train)
        if family=='ridge':
            self.scaler=StandardScaler().fit(x,sample_weight=self.w)
            self.params={'scaler_mean':self.scaler.mean_.tolist(),'scaler_scale':self.scaler.scale_.tolist()}
            x=self.scaler.transform(x);model=Ridge(alpha=config.get('alpha',10.))
        elif family=='linear':model=LinearRegression()
        elif family=='rf':model=RandomForestRegressor(n_estimators=config.get('trees',200),min_samples_leaf=config.get('leaf',2),max_depth=config.get('depth'),max_features=1.,random_state=seed,n_jobs=1)
        elif family=='gb':model=GradientBoostingRegressor(n_estimators=config.get('trees',100),max_depth=config.get('depth',2),min_samples_leaf=config.get('leaf',1),learning_rate=config.get('lr',.05),random_state=seed)
        else:raise ValueError(family)
        self.model=model.fit(x,y,sample_weight=self.w)
        if hasattr(model,'coef_'):self.params.update(coef=np.asarray(model.coef_).tolist(),intercept=float(model.intercept_))
    def x(self,d):
        x=make_features(d,self.config,self.categories)
        if self.config['family']=='linear':return x[['concentration']]
        if self.config.get('interactions'):
            for f in legacy.DESCRIPTORS:x['concentration_x_'+f]=x.concentration*x[f]
        return x
    def predict_raw(self,d):
        family=self.config['family']
        if self.config.get('scope')=='local':
            p=np.empty(len(d))
            for solvent in d.solvent.unique():
                mask=d.solvent.to_numpy()==solvent;p[mask]=self.parts.get(solvent,self.fallback).predict_raw(d.iloc[np.flatnonzero(mask)])
            return p
        if family in ['scaled_lld','saturating_mobile','fixed_lld']:return np.asarray(analytical.predict(d,family,self.params))
        if family=='hybrid':return self.base.predict_raw(d)+self.residual.predict_raw(d)
        if family=='mean':return np.full(len(d),self.params['mean'])
        x=self.x(d)
        if family=='ridge':x=self.scaler.transform(x)
        return self.model.predict(x)
    def predict(self,d):return np.maximum(0,self.predict_raw(d))
    def record(self):return dict(config=self.config,seed=self.seed,target=self.target,train_ids=self.ids,weights=self.w.tolist(),categories=self.categories,parameters=self.params,
        local_train_ids={s:m.ids for s,m in self.parts.items()})

def load_audit(out):
    allrows,sources=legacy.load_data();d=allrows[allrows.primary_eligible].reset_index(drop=True)
    canonical=pd.read_csv(ROOT/'near_term/results/canonical_records.csv').fillna('')
    pd.testing.assert_frame_equal(allrows.fillna(''),canonical[allrows.columns],check_dtype=False,atol=1e-12)
    assert len(d)==126 and d.group_id.nunique()==36 and d.record_id.is_unique
    for path in [ROOT/'PINN/datasets/20261111T134229Z_measured_followup_measured.csv',ASSUMED/'pinn_experimental_physics_assumed.csv']:
        other=pd.read_csv(path);assert list(d.record_id)==list(other.record_id)
        for c in FEATURES+['total_nm','bonded_nm','mobile_nm','uncoated_nm']:np.testing.assert_allclose(d[c],other[c],atol=1e-12)
    intake=pd.read_csv(ASSUMED/'pinn_experimental_physics_assumed.csv')
    assert intake['Experiment ID'].isna().all() and intake['Withdrawal Velocity (mm/s)'].nunique()==1
    prov=json.loads((ASSUMED/'provenance.json').read_text())
    checks=[]
    for ds in prov['datasets']:
        for path,expected in [(ASSUMED/ds['output'],ds['output_sha256']),(ASSUMED.parent/ds['source'],ds['source_sha256'])]:
            checks.append(check_hash(path,expected))
        a=pd.read_csv(ASSUMED.parent/ds['source'],dtype=str,keep_default_na=False);b=pd.read_csv(ASSUMED/ds['output'],dtype=str,keep_default_na=False)
        mutable=['Withdrawal Velocity (mm/s)','Viscosity (cP)','Surface Tension (mN/m)','Solution Density (kg/m3)','Wet-to-Dry Mobile Factor','LLD Applicable','measurement_source','unit_confirmation','regime_evidence']
        cols=[c for c in a if c not in mutable];pd.testing.assert_frame_equal(a[cols],b[cols]);assert (b['LLD Applicable']=='0').all()
    checks.append(check_hash(ASSUMED/'build_estimates.py',prov['script_sha256']))
    allrows.to_csv(out/'canonical_records.csv',index=False);intake.to_csv(out/'measured_snapshot.csv',index=False)
    allrows[~allrows.primary_eligible].to_csv(out/'excluded_records.csv',index=False)
    dump(out/'audit.json',dict(rows=len(d),conditions=d.group_id.nunique(),sources=sources,hash_checks=checks,
        excluded=allrows[~allrows.primary_eligible].exclusion_reason.value_counts().to_dict(),
        units='Concentration g/L despite legacy g/mL headers; thickness nm.',
        duplicate_value_rows=int(d.duplicated(['solvent','concentration_g_l','total_nm','bonded_nm','uncoated_nm'],keep=False).sum()),
        experimental_ids='Absent; conditions are a proxy, not proof of batch independence.',physics='Estimated properties, not measurements; hydrodynamic and velocity weights remain zero.',synthetic='Excluded; no verified disjoint parent lineage.'))
    return d

def check_hash(path,expected):
    b=Path(path).read_bytes();variants={'exact':b,'LF':b.replace(b'\r\n',b'\n'),'CRLF':b.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')}
    match=next((k for k,v in variants.items() if hashlib.sha256(v).hexdigest()==expected),None)
    assert match is not None,str(path)
    return dict(path=str(Path(path).relative_to(ROOT)),expected=expected,actual=sha(path),match=match)

def partitions(d):
    old=pd.read_csv(CAMP/'all_predictions.csv');sp=pd.read_csv(CAMP/'all_splits.csv')
    for protocol,fold,tr,te in legacy.splits(d):
        if protocol=='grouped_cv':
            g=old[(old.label=='grouped_cv')&(old.seed==11)&(old.fold==int(fold)-1)]
            assert set(d.iloc[te].record_id)==set(g.record_id)
            s=sp[sp.run_id==g.run_id.iloc[0]];a=s.loc[s['split']=='train','source_row'].to_numpy(int);b=s.loc[s['split']=='validation','source_row'].to_numpy(int)
            assert set(a)|set(b)==set(tr)
        else:
            i,j=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(tr,groups=d.iloc[tr].group_id));a,b=tr[i],tr[j]
        yield protocol,str(fold),tr,te,a,b

def save_splits(d,out):
    outer=[];inner=[]
    for protocol,fold,tr,te,a,b in partitions(d):
        for role,ix in [('train',a),('validation',b),('test',te)]:
            for r in d.iloc[ix].itertuples():outer.append(dict(protocol=protocol,fold=fold,role=role,record_id=r.record_id,group_id=r.group_id))
        for k,(i,j) in enumerate(GroupKFold(3).split(tr,groups=d.iloc[tr].group_id),1):
            for role,ix in [('train',tr[i]),('validation',tr[j])]:
                for r in d.iloc[ix].itertuples():inner.append(dict(protocol=protocol,fold=fold,inner_fold=k,role=role,record_id=r.record_id,group_id=r.group_id))
    pd.DataFrame(outer).to_csv(out/'splits.csv',index=False);pd.DataFrame(inner).to_csv(out/'inner_splits.csv',index=False)

def rows(test,pred,raw,task,protocol,fold,procedure,target,seed,fit_id):
    p=test[['record_id','group_id','solvent']].copy();p['actual']=test[target].to_numpy();p['total_observed']=test.total_nm.to_numpy()
    return p.assign(prediction=pred,raw_prediction=raw,task=task,protocol=protocol,fold=str(fold),procedure=procedure,target=target,seed=seed,fit_id=fit_id)

def paired_totals(p):
    a=p[(p.task=='A')&(p.target=='bonded_nm')];m=p[(p.task=='A')&(p.target=='mobile_nm')]
    keys=['protocol','fold','procedure','seed','record_id'];j=a.merge(m,on=keys,suffixes=('_b','_m'),validate='one_to_one')
    z=j[keys].copy()
    for col in ['group_id','solvent','total_observed']:z[col]=j[col+'_b']
    z['actual']=j.total_observed_b;z['prediction']=j.prediction_b+j.prediction_m;z['raw_prediction']=j.raw_prediction_b+j.raw_prediction_m
    z['task']='A';z['target']='total_nm';z['fit_id']=j.fit_id_b+'|'+j.fit_id_m
    return pd.concat([p,z],ignore_index=True)

def finish(out,start,extra=None):
    import time
    sources=[Path(__file__),ROOT/'near_term/run_analysis.py',ROOT/'near_term/solvent_analysis.py']+list((ROOT/'publication_2026').glob('*.py'))
    dump(out/'environment.json',dict(python=sys.version,platform=platform.platform(),packages={n:metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch','matplotlib','openpyxl']}))
    dump(out/'run_record.json',dict(elapsed_seconds=time.monotonic()-start,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_files={p.relative_to(ROOT).as_posix():sha(p) for p in sources},artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()},**(extra or {})))
