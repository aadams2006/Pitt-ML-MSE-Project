"""Verify published run provenance, source/artifact hashes and reported metrics."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from run_tracking import source_hash

HERE = Path(__file__).resolve().parent


def verify():
    records = sorted((HERE/'runs').glob('*/run_record.json'))
    if not records:
        raise ValueError('No recorded experiments found.')
    for file in records:
        record = json.loads(file.read_text(encoding='utf-8'))
        folder = file.parent
        if record['status'] not in ('completed', 'failed'):
            raise ValueError(f'Incomplete run: {folder.name}')
        if source_hash(list((folder/'source').glob('*.py'))) != record['source_sha256']:
            raise ValueError(f'Source identity mismatch: {folder.name}')
        for artifact, expected in record['artifacts'].items():
            if hashlib.sha256((folder/artifact).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Artifact changed: {folder.name}/{artifact}')
        if record['started_at_utc']:
            if datetime.fromisoformat(record['finished_at_utc']) < datetime.fromisoformat(record['started_at_utc']):
                raise ValueError('Run finished before start.')
            name = record['data_path'].replace('\\', '/').split('/')[-1]
            dataset = HERE/'datasets'/name
            if hashlib.sha256(dataset.read_bytes()).hexdigest() != record['data_sha256']:
                raise ValueError(f'Dataset identity mismatch: {folder.name}')
        if record['status'] == 'completed':
            csv = pd.read_csv(folder/'metrics.csv').set_index('model')
            if len(record['models']) != len(csv):
                raise ValueError('Missing per-model record.')
            for score in record['results']:
                for key in ('MAE_nm', 'RMSE_nm', 'R2', 'violation_fraction', 'max_violation_nm'):
                    np.testing.assert_allclose(csv.loc[score['model'], key], score[key], rtol=1e-9, atol=1e-12)
            splits = pd.read_csv(folder/'splits.csv')
            if splits.source_row.duplicated().any():
                raise ValueError(f'Duplicate source row across splits: {folder.name}')
            predictions = pd.read_csv(folder/'predictions.csv')
            if set(predictions.source_row) != set(splits.loc[splits['split']=='test', 'source_row']):
                raise ValueError(f'Prediction split mismatch: {folder.name}')
        print(f'Verified {folder.name}: {record["status"]}, v{record["model_version"]}')


if __name__ == '__main__':
    verify()
