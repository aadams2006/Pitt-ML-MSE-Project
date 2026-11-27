"""Audited November follow-up. New campaign/run directories; no historical overwrites."""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl
from train import run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(out):
    sys.path.insert(0, str(ROOT/'near_term'))
    from run_analysis import load_data, workbook_inventory
    canonical, sources = load_data()
    canonical.to_csv(out/'canonical_all.csv', index=False)
    canonical.loc[~canonical.primary_eligible].to_csv(out/'excluded_records.csv', index=False)
    df = canonical.loc[canonical.primary_eligible].copy().reset_index(drop=True)
    for solvent in sorted(df.solvent.unique()):
        df['solvent_'+solvent] = (df.solvent == solvent).astype(int)
    # Preserve true units and source identity. Descriptors remain metadata, not measured solution inputs.
    target = HERE/'datasets'/f'{out.name}_measured.csv'
    if target.exists():
        raise ValueError('Refusing to overwrite a prepared snapshot.')
    df.to_csv(target, index=False)
    inventory = workbook_inventory()
    hits = []
    paths = sorted({ROOT/r['path'] for r in inventory} | set((ROOT/'data').glob('*.xlsx')))
    for p in paths:
        for ws in openpyxl.load_workbook(p, data_only=True, read_only=True):
            for row in ws:
                for cell in row:
                    val = str(cell.value or '')
                    if any(k in val.lower() for k in ('mm/s', 'withdraw', 'velocity', 'viscos', 'density', 'dry factor', 'speed')):
                        hits.append(dict(path=p.relative_to(ROOT).as_posix(), sheet=ws.title, cell=cell.coordinate, text=val))
    (out/'workbook_inventory.json').write_text(json.dumps(inventory, indent=2))
    (out/'physics_cell_evidence.json').write_text(json.dumps(hits, indent=2))
    missing = ['Withdrawal Velocity (mm/s)', 'Viscosity (cP)', 'Surface Tension (mN/m)',
               'Solution Density (kg/m3)', 'Wet-to-Dry Mobile Factor', 'LLD Applicable']
    intake = df[['record_id','source_path','source_row','solvent','group_id']].copy()
    for col in missing + ['Experiment ID', 'measurement_source', 'unit_confirmation', 'regime_evidence']:
        intake[col] = ''
    intake.to_csv(out/'physics_measurement_intake.csv', index=False)
    provenance = dict(sources=sources, canonical_builder_sha256=digest(ROOT/'near_term/run_analysis.py'),
        dataset=target.relative_to(ROOT).as_posix(), dataset_sha256=digest(target),
        rows=len(df), groups=df.group_id.nunique(), counts=df.groupby('solvent').size().to_dict(),
        missing_matched_fields=missing, physics_status='No sample-linked validated table found; intake unfilled. Legacy scalar assumptions are not measurements.',
        concentration_units='g/L; numeric values preserved under previously audited canonical schema',
        scope='Conditional bonded inference using measured total and uncoated thickness; not recipe-only prediction.')
    (out/'data_provenance.json').write_text(json.dumps(provenance, indent=2))
    shutil.copy2(ROOT/'near_term/run_analysis.py', out/'canonical_builder.py')
    return target, df

def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = HERE/'campaigns'/f'{stamp}_measured_followup'
    out.mkdir(parents=True, exist_ok=False)
    data, df = prepare(out)
    cfg = json.loads((HERE/'config.json').read_text())
    continuous = ['concentration_g_l','uncoated_nm','total_nm']
    cfg.update(features=continuous+[c for c in df if c.startswith('solvent_')],
        target='bonded_nm',total_column='total_nm', log_features=['concentration_g_l'],
        group_column='group_id', smooth_features=continuous, split_seed=42,
        subsample_unit='group',subsample_seed=42, outer_folds=5)
    plan = []
    for seed in [11,42,73]:
        for fold in range(5):
            plan.append(dict(seed=seed,fold=fold,fraction=1.,label='grouped_cv'))
        for fraction in [.25,.5,.75]:
            plan.append(dict(seed=seed,fold=0,fraction=fraction,label='learning_curve'))
    # 24 invocations / 120 fits, including paired curvature, bound-only and unbounded comparisons.
    (out/'plan.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        seeds=[11,42,73],config=cfg,experiments=plan,
        decisions='Fixed settings before runs; no test-based tuning. Five outer folds; same inner split across initialization seeds. Learning curves use fold 1 only. No synthetic augmentation without vetted parent generation.'),indent=2))
    shutil.copy2(__file__,out/'campaign_runner.py')
    entries = []
    for item in plan:
        current = dict(cfg,seed=item['seed'],outer_fold=item['fold'])
        runid = f"{stamp}_{item['label']}_fold{item['fold']+1}_seed{item['seed']}_fraction{item['fraction']:g}"
        destination = HERE/'runs'/runid
        print(f'RUN {runid}',flush=True)
        metrics = run(data,current,destination,'grouped',item['fraction'])
        entries.append(dict(**item,run_id=runid,metrics=metrics))
        (out/'completed_runs.json').write_text(json.dumps(entries,indent=2))
    (out/'completed_utc.txt').write_text(datetime.now(timezone.utc).isoformat())
    print(f'CAMPAIGN {out}',flush=True)

if __name__ == '__main__':
    main()
