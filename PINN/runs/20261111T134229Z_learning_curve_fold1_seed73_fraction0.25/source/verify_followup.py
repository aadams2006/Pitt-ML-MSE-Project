"""Scientific integrity checks on completed follow-up partitions and predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from train import evaluate

HERE=Path(__file__).resolve().parent

def verify(out):
    entries=json.loads((out/'completed_runs.json').read_text())
    assert len(entries)==len(json.loads((out/'plan.json').read_text())['experiments'])
    prov=json.loads((out/'data_provenance.json').read_text())
    df=pd.read_csv(HERE.parent/prov['dataset'])
    assert df.record_id.is_unique and len(df)==126
    assert np.all(df.bonded_nm<=df.total_nm)
    partitions={}
    for entry in entries:
        folder=HERE/'runs'/entry['run_id']
        sp=pd.read_csv(folder/'splits.csv')
        by={role:set(g.source_row) for role,g in sp.groupby('split')}
        groups={role:set(df.iloc[sorted(ids)].group_id) for role,ids in by.items()}
        assert not (groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['test']&groups['validation'])
        partitions[(entry['seed'],entry['fold'],entry['fraction'])]=by
        p=pd.read_csv(folder/'predictions.csv')
        assert set(p.source_row)==by['test']
        for score in entry['metrics']:
            actual=evaluate(p.observed_nm,p[score['model']],p.total_nm)
            for key,value in actual.items():np.testing.assert_allclose(value,score[key],rtol=1e-6,atol=1e-8)
        for name in ['pinn','bounded_nn']:
            assert ((p[name]>=-1e-6)&(p[name]<=p.total_nm+1e-6)).all()
    for seed in [11,42,73]:
        alltests=[]
        for fold in range(5):
            p=partitions[seed,fold,1.]
            assert p==partitions[11,fold,1.]
            alltests.extend(p['test'])
        assert sorted(alltests)==list(range(len(df)))
        last=set()
        for fraction in [.25,.5,.75,1.]:
            p=partitions[seed,0,fraction]; full=partitions[seed,0,1.]
            assert p['test']==full['test'] and p['validation']==full['validation']
            assert last<=p['train'];last=p['train']
            # Every selected condition retains all its readings.
            gs=set(df.iloc[sorted(last)].group_id)
            assert set(df.index[df.group_id.isin(gs)])==last
    print(f'PASS: {len(entries)} runs; group separation, fixed partitions, nested whole-condition subsets, full CV coverage, recomputed scores and bounded predictions.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);verify(p.parse_args().campaign)
