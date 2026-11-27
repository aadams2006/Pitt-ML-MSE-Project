"""Preserve source snapshots and import honestly timed legacy run records."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from run_tracking import refresh_log, source_hash

HERE = Path(__file__).resolve().parent
INITIAL_COMMIT = '9714ee5'


def prepare():
    dest = HERE/'datasets'
    dest.mkdir(exist_ok=True)
    sources = [(HERE.parent/'data'/'agg.data.xlsx', dest/'experimental_58.csv', 'experimental'),
               (HERE.parent/'Holdout Validation'/'Data'/'synthetic_data_improved.csv', dest/'synthetic_3000.csv', 'synthetic')]
    provenance = []
    for source, target, kind in sources:
        if target.exists():
            raise ValueError(f'Dataset snapshot already exists: {target}')
        if kind == 'experimental':
            df = pd.read_excel(source)
            df.to_csv(target, index=False)
        else:
            target.write_bytes(source.read_bytes())
            df = pd.read_csv(source)
        provenance.append({'kind': kind, 'source_path': str(source.relative_to(HERE.parent)),
                           'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                           'snapshot': target.name, 'snapshot_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                           'rows': len(df), 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                           'note': 'Raw source snapshot; no filtering or target clipping. Synthetic parent-row provenance unavailable.' if kind == 'synthetic' else 'CSV export of measured workbook; numeric values retained.'})
    (dest/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    for folder in (HERE/'runs').glob('legacy_*'):
        manifest = json.loads((folder/'manifest.json').read_text())
        metrics = pd.read_csv(folder/'metrics.csv').drop(columns=['epochs'], errors='ignore').to_dict('records')
        source_dir = folder/'source'
        source_dir.mkdir(exist_ok=True)
        for filename in ('core.py', 'train.py', 'predict.py', 'test_pinn.py'):
            content = subprocess.check_output(['git', 'show', f'{INITIAL_COMMIT}:PINN/{filename}'], cwd=HERE)
            (source_dir/filename).write_bytes(content)
        (source_dir/'config.json').write_text(json.dumps(manifest['config'], indent=2), encoding='utf-8')
        last_write = max(p.stat().st_mtime for p in folder.glob('*') if p.is_file())
        record = {'run_id': folder.name, 'status': 'completed', 'model_version': '0.1.0',
                  'started_at_utc': None, 'started_at_new_york': None, 'finished_at_utc': None,
                  'artifact_last_modified_at_utc': datetime.fromtimestamp(last_write, timezone.utc).isoformat(),
                  'timestamp_provenance': 'Imported legacy run. Start/end not recorded; filesystem artifact modification is approximate, not an execution timestamp.',
                  'source_sha256': source_hash(list(source_dir.glob('*.py'))),
                  'git': {'archived_source_commit': INITIAL_COMMIT, 'note': 'Source archived retrospectively; run predates this commit.'},
                  'data_kind': 'experimental', 'data_sha256': manifest['data_sha256'],
                  'data_snapshot': '../../datasets/experimental_58.csv', 'config': manifest['config'],
                  'split_mode': manifest['split_mode'], 'rows': manifest['rows'], 'results': metrics,
                  'models': [{'model': row['model'], 'version': '0.1.0' if 'nn' in row['model'] or row['model']=='pinn' else manifest['versions']['sklearn'],
                              'started_at_utc': None, 'finished_at_utc': None, 'results': row} for row in metrics]}
        record['artifacts'] = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in folder.rglob('*') if p.is_file() and p.name != 'run_record.json'}
        (folder/'run_record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    refresh_log()


if __name__ == '__main__':
    prepare()
