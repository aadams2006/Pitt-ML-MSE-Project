"""Run identical held-out evaluations for PINN, NN ablations, RF and GB."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from core import load_data, fit_network, predict, sensitivity, Preprocessor

HERE = Path(__file__).resolve().parent


def split_indices(df, cfg, mode):
    indices = np.arange(len(df))
    group = cfg['group_column']
    if mode == 'random':
        dev, test = train_test_split(indices, test_size=cfg['test_size'], random_state=cfg['seed'])
        train, val = train_test_split(dev, test_size=cfg['validation_size'], random_state=cfg['seed'])
    else:
        if group not in df:
            raise ValueError(f'Missing split group: {group}')
        if df[group].nunique() < 5:
            raise ValueError('Need at least five groups for independent train/validation/test.')
        if mode == 'extrapolation':
            groups = sorted(df[group].unique())
            held = groups[-max(1, int(np.ceil(len(groups) * cfg['test_size']))):]
            test = indices[df[group].isin(held)]
            dev = indices[~df[group].isin(held)]
        else:
            dev, test = next(GroupShuffleSplit(n_splits=1, test_size=cfg['test_size'], random_state=cfg['seed']).split(df, groups=df[group]))
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=cfg['validation_size'], random_state=cfg['seed']).split(df.iloc[dev], groups=df.iloc[dev][group]))
        train, val = dev[a], dev[b]
    return train, val, test


def evaluate(y, pred, total):
    return {'MAE_nm': float(mean_absolute_error(y, pred)),
            'RMSE_nm': float(np.sqrt(mean_squared_error(y, pred))),
            'R2': float(r2_score(y, pred)) if len(y) > 1 else None,
            'violation_fraction': float(np.mean((pred < -1e-6) | (pred > total + 1e-6))),
            'max_violation_nm': float(np.maximum(np.maximum(-pred, pred-total), 0).max())}


def run(data_path, cfg, out, mode='grouped', fraction=1.):
    if not 0 < fraction <= 1:
        raise ValueError('Train fraction must lie in (0,1].')
    df = load_data(data_path, cfg)
    tr, va, te = split_indices(df, cfg, mode)
    if fraction < 1:
        # Training-only subsampling; validation and test stay identical across fractions.
        count = max(2, int(len(tr) * fraction))
        tr = np.sort(np.random.default_rng(cfg['seed']).choice(tr, count, replace=False))
    train, val, test = df.iloc[tr], df.iloc[va], df.iloc[te]
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is nonempty; choose a new run directory.')
    out.mkdir(parents=True, exist_ok=True)
    split = pd.DataFrame({'source_row': np.concatenate([tr, va, te]),
                          'split': ['train']*len(tr)+['validation']*len(va)+['test']*len(te)})
    split.to_csv(out/'splits.csv', index=False)
    predictions = pd.DataFrame({'source_row': te, 'observed_nm': test[cfg['target']].to_numpy(),
                                'total_nm': test[cfg['total_column']].to_numpy()})
    metrics = []
    for name, bounded, regularized in [('pinn', True, True), ('bounded_nn', True, False), ('standard_nn', False, False)]:
        model, prep, history = fit_network(train, val, cfg, bounded, regularized)
        pred = predict(model, prep, test)
        predictions[name] = pred
        metrics.append({'model': name, **evaluate(predictions.observed_nm, pred, predictions.total_nm), 'epochs': len(history)})
        pd.DataFrame(history).to_csv(out/f'{name}_history.csv', index=False)
        torch.save({'state_dict': model.state_dict(), 'config': cfg, 'preprocessing': prep.export(), 'bounded': bounded}, out/f'{name}.pt')
        if name == 'pinn':
            gradients = sensitivity(model, prep, test)
            gradients.insert(0, 'source_row', te)
            gradients.to_csv(out/'pinn_sensitivity.csv', index=False)
    prep = Preprocessor(cfg).fit(train)
    for name, model in [('random_forest', RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=cfg['seed'], n_jobs=1)),
                        ('gradient_boosting', GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=.05, random_state=cfg['seed']))]:
        model.fit(prep.transform(train), train[cfg['target']])
        pred = model.predict(prep.transform(test))
        predictions[name] = pred
        metrics.append({'model': name, **evaluate(predictions.observed_nm, pred, predictions.total_nm)})
    predictions.to_csv(out/'predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    manifest = {'config': cfg, 'data_path': str(Path(data_path).resolve()),
                'data_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                'split_mode': mode, 'train_fraction': fraction,
                'rows': {'train': len(tr), 'validation': len(va), 'test': len(te)},
                'versions': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
                'note': 'Constrained regression, not a PDE solver. Metrics are exploratory, not tuned benchmark claims.'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False))
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=HERE.parent/'data'/'agg.data.xlsx')
    p.add_argument('--config', type=Path, default=HERE/'config.json')
    p.add_argument('--output', type=Path, default=HERE/'results'/'initial_grouped')
    p.add_argument('--split', choices=['grouped', 'extrapolation', 'random'], default='grouped')
    p.add_argument('--epochs', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--train-fraction', type=float, default=1.)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    for key in ('epochs', 'seed'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    run(args.data, cfg, args.output, args.split, args.train_fraction)
