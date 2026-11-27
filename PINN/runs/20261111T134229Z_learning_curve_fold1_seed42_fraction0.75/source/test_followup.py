import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from train import split_indices
from core import validate_config

class FollowupTests(unittest.TestCase):
    def test_outer_coverage_and_seed_independence(self):
        cfg=json.loads((Path(__file__).parent/'config.json').read_text())
        df=pd.DataFrame({cfg['group_column']:np.repeat(np.arange(20),3)})
        tests=[]
        for fold in range(5):
            a=dict(cfg,outer_fold=fold,outer_folds=5,split_seed=42,seed=11)
            first=split_indices(df,a,'grouped')
            second=split_indices(df,dict(a,seed=73),'grouped')
            for x,y in zip(first,second): np.testing.assert_array_equal(x,y)
            sets=[set(df.iloc[x][cfg['group_column']]) for x in first]
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            tests.extend(first[2])
        self.assertEqual(sorted(tests),list(range(len(df))))

    def test_unknown_smooth_feature_rejected(self):
        cfg=json.loads((Path(__file__).parent/'config.json').read_text())
        cfg['smooth_features']=['invented']
        with self.assertRaises(ValueError): validate_config(cfg)

if __name__=='__main__': unittest.main()
