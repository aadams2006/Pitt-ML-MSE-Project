"""Small-data constrained regression; no unverified PDE is invented here."""
from __future__ import annotations

import copy
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def validate_config(cfg):
    features = cfg['features']
    if not features or len(set(features)) != len(features):
        raise ValueError('Features must be nonempty and unique.')
    if cfg['target'] in features:
        raise ValueError('Target cannot be an input feature.')
    if cfg['total_column'] not in features:
        raise ValueError('Measured total thickness is required for the bounded model.')
    if not set(cfg['log_features']).issubset(features):
        raise ValueError('Log features must be inputs.')
    if not set(cfg['monotonic']).issubset(features):
        raise ValueError('Monotonic constraints must reference inputs.')
    if any(v not in (-1, 1) for v in cfg['monotonic'].values()):
        raise ValueError('Monotonic signs must be +1 or -1.')
    if cfg['epochs'] < 1 or cfg['patience'] < 1:
        raise ValueError('Epochs and patience must be positive.')
    if any(cfg[k] < 0 for k in ('smooth_weight', 'monotonic_weight', 'weight_decay')):
        raise ValueError('Loss weights must be nonnegative.')
    if cfg['learning_rate'] <= 0 or not cfg['hidden'] or any(h < 1 for h in cfg['hidden']):
        raise ValueError('Learning rate and hidden widths must be positive.')
    if not all(0 < cfg[k] < 1 for k in ('test_size', 'validation_size')):
        raise ValueError('Split fractions must lie between zero and one.')


def load_data(path, cfg, require_target=True):
    validate_config(cfg)
    df = pd.read_excel(path) if str(path).lower().endswith('.xlsx') else pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = list(cfg['features']) + ([cfg['target']] if require_target else [])
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    values = df[required].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('Missing or nonfinite values; resolve explicitly before training.')
    df[required] = values
    if (df[cfg['total_column']] <= 0).any():
        raise ValueError('Total thickness must be positive.')
    for col in cfg['log_features']:
        if (df[col] <= 0).any():
            raise ValueError(f'{col} must be positive for log transform.')
    for col in cfg['features']:
        if ('Thickness' in col or 'Layer' in col) and (df[col] < 0).any():
            raise ValueError(f'{col} cannot be negative.')
    if require_target:
        y, total = df[cfg['target']], df[cfg['total_column']]
        if ((y < 0) | (y > total)).any():
            raise ValueError('Observed bonded thickness violates [0, total]; review measurement uncertainty.')
    return df


class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler = StandardScaler()

    def raw(self, df):
        values = df[self.cfg['features']].to_numpy(dtype=float).copy()
        for col in self.cfg['log_features']:
            values[:, self.cfg['features'].index(col)] = np.log(values[:, self.cfg['features'].index(col)])
        return values

    def fit(self, df):
        self.scaler.fit(self.raw(df))
        return self

    def transform(self, df):
        return self.scaler.transform(self.raw(df)).astype(np.float32)

    def tensors(self, df):
        return (torch.tensor(self.transform(df)),
                torch.tensor(df[self.cfg['total_column']].to_numpy(np.float32)).reshape(-1, 1))

    def export(self):
        return {'mean': self.scaler.mean_.tolist(), 'scale': self.scaler.scale_.tolist()}

    @classmethod
    def restore(cls, cfg, state):
        obj = cls(cfg)
        obj.scaler.mean_ = np.array(state['mean'])
        obj.scaler.scale_ = np.array(state['scale'])
        obj.scaler.var_ = obj.scaler.scale_ ** 2
        obj.scaler.n_features_in_ = len(cfg['features'])
        return obj


class ThicknessNet(nn.Module):
    def __init__(self, cfg, bounded=True):
        super().__init__()
        widths = [len(cfg['features']), *cfg['hidden'], 1]
        layers = []
        for i in range(len(widths) - 1):
            layer = nn.Linear(widths[i], widths[i+1])
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if i < len(widths) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
        self.bounded = bounded

    def forward(self, x, total):
        output = self.network(x)
        return total * torch.sigmoid(output) if self.bounded else output


def regularizers(model, x, total, prep, output_scale, cfg):
    """Diagonal Hessian penalty in standardized coordinates, at measured train rows.

    Differentiate the physical output, including the total-thickness envelope.
    No Cartesian collocation between correlated measurements is assumed valid.
    """
    x = x.detach().clone().requires_grad_(True)
    j = cfg['features'].index(cfg['total_column'])
    total_from_x = x[:, j:j+1] * float(prep.scaler.scale_[j]) + float(prep.scaler.mean_[j])
    if cfg['total_column'] in cfg['log_features']:
        total_from_x = torch.exp(total_from_x)
    pred = model(x, total_from_x) / output_scale
    grad = torch.autograd.grad(pred.sum(), x, create_graph=True)[0]
    curvature = []
    for k in range(x.shape[1]):
        second = torch.autograd.grad(grad[:, k].sum(), x, create_graph=True, retain_graph=True)[0][:, k]
        curvature.append(second.square().mean())
    smooth = torch.stack(curvature).mean()
    mono = x.new_zeros(())
    for col, sign in cfg['monotonic'].items():
        mono = mono + torch.relu(-sign * grad[:, cfg['features'].index(col)]).square().mean()
    return smooth, mono


def fit_network(train, val, cfg, bounded=True, regularized=True):
    seed_all(cfg['seed'])
    prep = Preprocessor(cfg).fit(train)
    x, total = prep.tensors(train)
    xv, tv = prep.tensors(val)
    y = torch.tensor(train[cfg['target']].to_numpy(np.float32)).reshape(-1, 1)
    yv = torch.tensor(val[cfg['target']].to_numpy(np.float32)).reshape(-1, 1)
    scale = max(float(y.std(unbiased=False)), 1e-6)
    model = ThicknessNet(cfg, bounded)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    best, best_state, stale, history = float('inf'), None, 0, []
    for epoch in range(cfg['epochs']):
        model.train()
        optimizer.zero_grad()
        data = ((model(x, total) - y) / scale).square().mean()
        smooth, mono = x.new_zeros(()), x.new_zeros(())
        if regularized and (cfg['smooth_weight'] or cfg['monotonic_weight']):
            smooth, mono = regularizers(model, x, total, prep, scale, cfg)
        loss = data + (cfg['smooth_weight'] * smooth + cfg['monotonic_weight'] * mono if regularized else 0)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss; inspect scaling/configuration.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float(((model(xv, tv) - yv) / scale).square().mean())
        history.append({'epoch': epoch+1, 'data_loss': float(data.detach()), 'smooth_loss': float(smooth.detach()),
                        'monotonic_loss': float(mono.detach()), 'val_loss': validation})
        if validation < best - 1e-7:
            best, best_state, stale = validation, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= cfg['patience']:
            break
    model.load_state_dict(best_state)
    return model, prep, history


def predict(model, prep, df):
    model.eval()
    with torch.no_grad():
        return model(*prep.tensors(df)).numpy().ravel()


def sensitivity(model, prep, df):
    x, total = prep.tensors(df)
    x.requires_grad_(True)
    j = prep.cfg['features'].index(prep.cfg['total_column'])
    reconstructed = x[:, j:j+1] * float(prep.scaler.scale_[j]) + float(prep.scaler.mean_[j])
    if prep.cfg['total_column'] in prep.cfg['log_features']:
        reconstructed = reconstructed.exp()
    grad = torch.autograd.grad(model(x, reconstructed).sum(), x)[0].detach().numpy()
    grad /= prep.scaler.scale_
    for col in prep.cfg['log_features']:
        k = prep.cfg['features'].index(col)
        grad[:, k] /= df[col].to_numpy()
    return pd.DataFrame(grad, columns=prep.cfg['features'], index=df.index)
