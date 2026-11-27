"""Independent integrity checks for nested selection and final predictions."""
import unittest,json
import numpy as np
import pandas as pd
from publication_2026.common import *
from publication_2026.stage3 import choose
class NestedIntegrity(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.out=ROOT/(ROOT/'publication_2026/stage_03/latest_run.txt').read_text().strip()
  cls.base=ROOT/(ROOT/'publication_2026/stage_02/latest_run.txt').read_text().strip()
  cls.d=pd.read_csv(cls.base/'measured_snapshot.csv');cls.s=pd.read_csv(cls.out/'splits.csv',dtype={'fold':str});cls.inner=pd.read_csv(cls.out/'inner_splits.csv',dtype={'fold':str})
  cls.p=pd.read_csv(cls.out/'predictions.csv',dtype={'fold':str});cls.ip=pd.read_csv(cls.out/'inner_predictions.csv',dtype={'fold':str})
  cls.spec=json.loads((cls.out/'search_space.json').read_text(encoding='utf-8-sig'));cls.fits=json.loads((cls.out/'fits.json').read_text());cls.choices=json.loads((cls.out/'selected_configs_by_fold.json').read_text())
 def test_hashes_and_frozen_spec(self):
  r=json.loads((self.out/'run_record.json').read_text());self.assertEqual(r['search_space_sha256'],sha(self.out/'search_space.json'))
  for p,h in r['artifacts'].items():self.assertEqual(sha(self.out/p),h,p)
  for p,h in r['source_files'].items():self.assertEqual(sha(ROOT/p),h,p)
  for p,h in json.loads((self.out/'inputs.json').read_text()).items():self.assertEqual(sha(ROOT/p),h,p)
  self.assertEqual(sum(c['family']=='rf' for c in self.spec['candidates']),12);self.assertEqual(sum(c['family']=='gb' for c in self.spec['candidates']),8)
  self.assertEqual(sum(c['family']=='hybrid' for c in self.spec['simple_mobile']),1)
 def test_all_fit_membership_and_preprocessing(self):
  byid=self.d.set_index('record_id',drop=False)
  for f in self.fits:
   s=self.s[(self.s.protocol==f['protocol'])&(self.s.fold==f['fold'])]
   if f['role']=='inner':
    z=self.inner[(self.inner.protocol==f['protocol'])&(self.inner.fold==f['fold'])&(self.inner.inner_fold==f['inner_fold'])];allowed=z[z.role=='train'];valid=z[z.role=='validation']
    self.assertTrue(set(allowed.group_id).isdisjoint(set(valid.group_id)))
   else:allowed=s[s.role!='test']
   self.assertEqual(set(f['train_ids']),set(allowed.record_id));self.assertEqual(len(f['train_ids']),len(set(f['train_ids'])))
   self.assertTrue(set(allowed.group_id).isdisjoint(set(s[s.role=='test'].group_id)))
   train=byid.loc[f['train_ids']];self.assertEqual(f['categories'],sorted(train.solvent.unique()))
   w=pd.DataFrame({'group_id':train.group_id,'w':f['weights']}).groupby('group_id').w.sum();np.testing.assert_allclose(w,w.iloc[0],atol=1e-12)
   for solvent,ids in f.get('local_train_ids',{}).items():self.assertEqual(set(ids),set(train[train.solvent==solvent].record_id))
   if f['config']['family']=='hybrid':
    cfg=dict(family='ridge',encoding='descriptors',transform='log',interactions=True)
    x=make_features(train,cfg,sorted(train.solvent.unique()))
    for col in legacy.DESCRIPTORS:x['concentration_x_'+col]=x.concentration*x[col]
    np.testing.assert_allclose(f['parameters']['residual']['scaler_mean'],np.average(x,axis=0,weights=f['weights']),atol=1e-12)
 def test_inner_scores_selection_and_no_missing_candidates(self):
  scores=pd.read_csv(self.out/'inner_scores.csv',dtype={'fold':str})
  rf=[c['id'] for c in self.spec['candidates'] if c['family']=='rf'];gb=[c['id'] for c in self.spec['candidates'] if c['family']=='gb'];simple=[c['id'] for c in self.spec['simple_mobile']]
  for c in self.choices:
   ip=self.ip[(self.ip.protocol==c['protocol'])&(self.ip.fold==c['fold'])&(self.ip.target==c['target'])]
   expected=rf+gb+(simple if c['target']=='mobile_nm' else []);self.assertEqual(set(ip.procedure),set(expected));computed={}
   for cid,g in ip.groupby('procedure'):
    self.assertFalse(g.record_id.duplicated().any())
    dev=self.s[(self.s.protocol==c['protocol'])&(self.s.fold==c['fold'])&(self.s.role!='test')]
    self.assertEqual(set(g.record_id),set(dev.record_id))
    e=g.actual-g.prediction;mse=g.assign(sq=e**2).groupby('group_id').sq.mean().mean();computed[cid]=float(mse)
    self.assertAlmostEqual(mse,c['inner_condition_mse'][cid],places=11)
   self.assertEqual(c['selected']['tuned_rf'],choose(computed,rf));self.assertEqual(c['selected']['tuned_gb'],choose(computed,gb))
   self.assertEqual(c['selected']['tree_selected'],choose(computed,rf+gb))
   self.assertEqual(c['selected']['simple_selected'],choose(computed,simple) if c['target']=='mobile_nm' else choose(computed,rf+gb))
  for row in scores.itertuples():
   g=self.ip[(self.ip.protocol==row.protocol)&(self.ip.fold==row.fold)&(self.ip.target==row.target)&(self.ip.procedure==row.candidate)&(self.ip.inner_fold==row.inner_fold)]
   e=g.actual-g.prediction;v=np.sqrt(g.assign(sq=e**2).groupby('group_id').sq.mean().mean());self.assertAlmostEqual(v,row.condition_rmse_nm,places=11)
 def test_outer_coverage_and_selected_configs(self):
  p=self.p;self.assertFalse(p.duplicated(['protocol','fold','procedure','target','seed','record_id']).any())
  choices={(c['protocol'],c['fold'],c['target']):c['selected'] for c in self.choices}
  for (protocol,procedure,target,seed),g in p.groupby(['protocol','procedure','target','seed']):
   test=self.s[(self.s.protocol==protocol)&(self.s.role=='test')];self.assertEqual(set(zip(g.fold,g.record_id)),set(zip(test.fold,test.record_id)))
   if protocol=='grouped_cv':self.assertEqual(len(g),126)
   if target!='total_nm':
    for fold,part in g.groupby('fold'):self.assertEqual(set(part.selected_config),{choices[protocol,fold,target][procedure]})
  np.testing.assert_allclose(p.actual,[self.d.set_index('record_id').loc[r.record_id,r.target] for r in p.itertuples()],atol=1e-12)
 def test_totals_metrics_and_paired_differences(self):
  p=self.p;q=p.pivot(index=['protocol','fold','procedure','seed','record_id'],columns='target',values='prediction')
  np.testing.assert_allclose(q.total_nm,q.bonded_nm+q.mobile_nm,atol=1e-12);self.assertTrue((q>=0).all().all())
  m=pd.read_csv(self.out/'metrics.csv')
  for r in m.itertuples():
   g=p[(p.protocol==r.protocol)&(p.procedure==r.procedure)&(p.target==r.target)&(p.seed==r.seed)]
   if r.solvent!='all':g=g[g.solvent==r.solvent]
   e=g.actual-g.prediction;gm=g.assign(sq=e**2,ae=abs(e)).groupby('group_id')[['sq','ae']].mean()
   np.testing.assert_allclose([np.sqrt(gm.sq.mean()),gm.ae.mean(),np.sqrt(np.mean(e**2)),np.mean(abs(e))],[r.condition_rmse_nm,r.condition_mae_nm,r.row_rmse_nm,r.row_mae_nm],atol=1e-12)
  delta=pd.read_csv(self.out/'paired_condition_differences.csv');np.testing.assert_allclose(delta.delta_condition_mse,delta.new_condition_mse-delta.reference_condition_mse,atol=1e-12)
  ref=pd.read_csv(self.base/'predictions.csv');ref=ref[ref.task=='A']
  for (protocol,procedure,target,seed,reference),g in delta.groupby(['protocol','procedure','target','seed','reference']):
   a=p[(p.protocol==protocol)&(p.procedure==procedure)&(p.target==target)&(p.seed==seed)];b=ref[(ref.protocol==protocol)&(ref.procedure==reference)&(ref.target==target)]
   ma=a.assign(sq=(a.actual-a.prediction)**2).groupby('group_id').sq.mean();mb=b.assign(sq=(b.actual-b.prediction)**2).groupby('group_id').sq.mean()
   np.testing.assert_allclose(g.delta_condition_mse,ma.loc[g.group_id].to_numpy()-mb.loc[g.group_id].to_numpy(),atol=1e-12)
 def test_feature_poison_unknown_categories_and_hybrid_refit(self):
  d=self.d;train=d[d.solvent!='hexane'];test=d[d.solvent=='hexane'].copy();poison=test.copy()
  for col in ['bonded_nm','mobile_nm','total_nm','uncoated_nm','LLD Applicable','mobile_q1_scenario_nm']:poison[col]=1e9
  for cfg in [dict(family='rf',encoding='identity',transform='log',trees=10),dict(family='gb',encoding='descriptors',transform='raw',trees=10),dict(family='hybrid')]:
   f=Fitted(train,'mobile_nm',cfg,42);np.testing.assert_allclose(f.predict(test),f.predict(poison),atol=1e-12)
   self.assertEqual(set(f.ids),set(train.record_id))
   if cfg['family']=='hybrid':
    self.assertEqual(set(f.base.ids),set(train.record_id));self.assertEqual(set(f.residual.ids),set(train.record_id))
   if cfg.get('encoding')=='identity':
    x=f.x(test);self.assertEqual(list(x.columns),['concentration']+['solvent_'+s for s in sorted(train.solvent.unique())]);self.assertTrue((x.iloc[:,1:]==0).all().all())
if __name__=='__main__':unittest.main()
