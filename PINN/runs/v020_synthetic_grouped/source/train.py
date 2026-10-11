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
from run_tracking import start_run, finish_run, save_record, now, MODEL_VERSION
from physics import validate_physics

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


def _run(data_path, cfg, out, record, mode='grouped', fraction=1., invalid_policy='error'):
    if not 0 < fraction <= 1:
        raise ValueError('Train fraction must lie in (0,1].')
    df, audit = load_data(data_path, cfg, invalid_policy=invalid_policy, return_audit=True)
    (out/'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    record['data_audit'] = audit
    tr, va, te = split_indices(df, cfg, mode)
    if fraction < 1:
        # Training-only subsampling; validation and test stay identical across fractions.
        count = max(2, int(len(tr) * fraction))
        tr = np.sort(np.random.default_rng(cfg['seed']).choice(tr, count, replace=False))
    train, val, test = df.iloc[tr], df.iloc[va], df.iloc[te]
    split = pd.DataFrame({'source_row': df.index[np.concatenate([tr, va, te])],
                          'split': ['train']*len(tr)+['validation']*len(va)+['test']*len(te)})
    split.to_csv(out/'splits.csv', index=False)
    predictions = pd.DataFrame({'source_row': test.index, 'observed_nm': test[cfg['target']].to_numpy(),
                                'total_nm': test[cfg['total_column']].to_numpy()})
    metrics = []
    for name, bounded, regularized in [('pinn', True, True), ('bounded_nn', True, False), ('standard_nn', False, False)]:
        model_record = {'model': name, 'version': MODEL_VERSION, 'started_at_utc': now(),
                        'bounded': bounded, 'regularized': regularized, 'hidden': cfg['hidden']}
        record['models'].append(model_record)
        save_record(out, record)
        model, prep, history = fit_network(train, val, cfg, bounded, regularized)
        pred = predict(model, prep, test)
        predictions[name] = pred
        score = {'model': name, **evaluate(predictions.observed_nm, pred, predictions.total_nm), 'epochs': len(history)}
        metrics.append(score)
        model_record.update({'finished_at_utc': now(), 'epochs_run': len(history),
                             'selected_epoch': next(h['epoch'] for h in reversed(history) if h['checkpoint_updated']), 'results': score})
        save_record(out, record)
        pd.DataFrame(history).to_csv(out/f'{name}_history.csv', index=False)
        torch.save({'state_dict': model.state_dict(), 'config': cfg, 'preprocessing': prep.export(), 'bounded': bounded,
                    'model_version': MODEL_VERSION, 'run_id': record['run_id'], 'source_sha256': record['source_sha256']}, out/f'{name}.pt')
        if name == 'pinn':
            gradients = sensitivity(model, prep, test)
            gradients.insert(0, 'source_row', test.index)
            gradients.to_csv(out/'pinn_sensitivity.csv', index=False)
            train_gradients = sensitivity(model, prep, train)
            # Association diagnostics, not evidence of a causal monotonic law.
            diagnostics = []
            for col in cfg['features']:
                if col == cfg['total_column']:
                    continue
                contexts = [('pooled', train)]
                if 'Viscosity (cP)' in train and col != 'Viscosity (cP)':
                    contexts += [(f'viscosity={level}', subset) for level, subset in train.groupby('Viscosity (cP)')]
                for context, subset in contexts:
                    diagnostics.append({'feature': col, 'context': context, 'n': len(subset),
                                        'spearman_training': subset[col].corr(subset[cfg['target']], method='spearman') if subset[col].nunique()>1 else None,
                                        'negative_partial_fraction': float((train_gradients.loc[subset.index, col] < 0).mean()),
                                        'note': 'Training-only association; not justification for causal sign enforcement.'})
            pd.DataFrame(diagnostics).to_csv(out/'training_trend_diagnostics.csv', index=False)
    prep = Preprocessor(cfg).fit(train)
    for name, model in [('random_forest', RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=cfg['seed'], n_jobs=1)),
                        ('gradient_boosting', GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=.05, random_state=cfg['seed']))]:
        model_record = {'model': name, 'version': sklearn.__version__, 'started_at_utc': now(), 'parameters': model.get_params()}
        record['models'].append(model_record)
        save_record(out, record)
        model.fit(prep.transform(train), train[cfg['target']])
        pred = model.predict(prep.transform(test))
        predictions[name] = pred
        score = {'model': name, **evaluate(predictions.observed_nm, pred, predictions.total_nm)}
        metrics.append(score)
        model_record.update({'finished_at_utc': now(), 'results': score})
        save_record(out, record)
        # Portable inspectable tree representation; no unsafe pickle required.
        trees = list(model.estimators_) if name == 'random_forest' else list(model.estimators_.ravel())
        np.savez_compressed(out/f'{name}_trees.npz', **{f'tree_{i}': tree.tree_.__getstate__()['nodes'] for i, tree in enumerate(trees)},
                            **{f'values_{i}': tree.tree_.value for i, tree in enumerate(trees)})
        (out/f'{name}_parameters.json').write_text(json.dumps({'parameters': model.get_params(), 'preprocessing': prep.export(),
            'note': 'Tree structure for inspection; refit using source/config/splits for inference.',
            'initial_prediction': float(model.init_.constant_.ravel()[0]) if name == 'gradient_boosting' else None}, indent=2), encoding='utf-8')
    predictions.to_csv(out/'predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    manifest = {'config': cfg, 'model_version': MODEL_VERSION, 'run_id': record['run_id'], 'data_kind': record['data_kind'],
                'data_audit': audit, 'data_path': str(Path(data_path).resolve()),
                'data_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                'split_mode': mode, 'train_fraction': fraction,
                'rows': {'train': len(tr), 'validation': len(va), 'test': len(te)},
                'versions': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
                'note': 'Constrained regression, not a PDE solver. Metrics are exploratory, not tuned benchmark claims.'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False))
    return metrics


def run(data_path, cfg, out=None, mode='grouped', fraction=1., data_kind='experimental', invalid_policy='error'):
    if data_kind not in ('experimental', 'synthetic'):
        raise ValueError('Data kind must be experimental or synthetic.')
    if invalid_policy == 'exclude' and data_kind != 'synthetic':
        raise ValueError('Experimental exclusions require a reviewed corrected dataset; this automatic policy is synthetic-only.')
    out, record = start_run(out, cfg, data_path, mode, fraction, data_kind, invalid_policy)
    try:
        metrics = _run(data_path, cfg, out, record, mode, fraction, invalid_policy)
    except Exception as error:
        finish_run(out, record, error=error)
        raise
    finish_run(out, record, metrics)
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=HERE/'datasets'/'experimental_58.csv')
    p.add_argument('--config', type=Path, default=HERE/'config.json')
    p.add_argument('--output', type=Path)
    p.add_argument('--data-kind', choices=['experimental', 'synthetic'], default='experimental')
    p.add_argument('--invalid-policy', choices=['error', 'exclude'], default='error')
    p.add_argument('--split', choices=['grouped', 'extrapolation', 'random'], default='grouped')
    p.add_argument('--epochs', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--train-fraction', type=float, default=1.)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    for key in ('epochs', 'seed'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    run(args.data, cfg, args.output, args.split, args.train_fraction, args.data_kind, args.invalid_policy)
