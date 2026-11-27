import unittest,json
import numpy as np
import pandas as pd
from publication_2026.common import *
class BaselineIntegrity(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.out=ROOT/(ROOT/'publication_2026/stage_02/latest_run.txt').read_text().strip();cls.d=pd.read_csv(cls.out/'measured_snapshot.csv');cls.p=pd.read_csv(cls.out/'predictions.csv',dtype={'fold':str});cls.s=pd.read_csv(cls.out/'splits.csv',dtype={'fold':str})
 def test_hashes(self):
  r=json.loads((self.out/'run_record.json').read_text())
  for p,h in r['artifacts'].items():self.assertEqual(sha(self.out/p),h)
  for p,h in r['source_files'].items():self.assertEqual(sha(ROOT/p),h)
 def test_partitions_and_numeric_holdouts(self):
  for (protocol,fold),g in self.s.groupby(['protocol','fold']):
   self.assertEqual(len(g),126);self.assertEqual(g.record_id.nunique(),126);self.assertEqual(g.groupby('group_id').role.nunique().max(),1)
   if protocol.startswith('ood_'):
    z=g.merge(self.d[['record_id','concentration_g_l','solvent']],on='record_id')
    for _,s in z.groupby('solvent'):
     a=s[s.role!='test'].concentration_g_l;b=s[s.role=='test'].concentration_g_l
     self.assertTrue(b.max()<a.min() if 'low' in protocol else b.min()>a.max())
 def test_coverage_and_metrics(self):
  p=self.p;self.assertFalse(p.duplicated(['task','protocol','fold','procedure','target','seed','record_id']).any())
  for key,g in p.groupby(['task','protocol','procedure','target','seed']):
   s=self.s[(self.s.protocol==key[1])&(self.s.role=='test')];self.assertEqual(set(zip(g.fold,g.record_id)),set(zip(s.fold,s.record_id)))
  saved=pd.read_csv(self.out/'baseline_metrics.csv');calc=summarize(p);pd.testing.assert_frame_equal(saved,calc,check_dtype=False,atol=1e-12)
 def test_training_budget_and_weights(self):
  for f in json.loads((self.out/'fits.json').read_text()):
   s=self.s[(self.s.protocol==f['protocol'])&(self.s.fold==f['fold'])];allowed=s[s.role=='train'] if f['fit_id'].startswith('B_') else s[s.role!='test']
   self.assertTrue(set(f['train_ids'])<=set(allowed.record_id))
   if 'config' in f:
    train=self.d.set_index('record_id').loc[f['train_ids']];w=pd.DataFrame({'g':train.group_id,'w':f['weights']}).groupby('g').w.sum();np.testing.assert_allclose(w,w.iloc[0])
 def test_preservation_and_task_B_reuse(self):
  initial=json.loads((ROOT/'publication_2026/stage_01/input_manifest.json').read_text())
  for p,h in initial['files'].items():self.assertEqual(sha(ROOT/p),h,p)
  old=pd.read_csv(CAMP/'all_predictions.csv');old=old[old.label=='grouped_cv']
  for model in ['pinn','bounded_nn','standard_nn','random_forest','gradient_boosting']:
   p=self.p[(self.p.task=='B')&(self.p.procedure==model)&(self.p.target=='bonded_nm')].merge(old,on=['record_id','seed'],validate='one_to_one')
   np.testing.assert_allclose(p.prediction,p[model],rtol=0,atol=1e-15)
if __name__=='__main__':unittest.main()
