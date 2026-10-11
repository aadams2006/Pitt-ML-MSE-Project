"""Scientific integrity checks: bounds, gradients, leakage and checkpoint fidelity."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from core import ThicknessNet, Preprocessor, load_data, regularizers, predict, fit_network, sensitivity
from train import split_indices

HERE = Path(__file__).resolve().parent


class PinnTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((HERE/'config.json').read_text())
        c = np.repeat(np.geomspace(.02, 10., 10), 3)
        self.df = pd.DataFrame({self.cfg['features'][0]: c, self.cfg['features'][1]: 1.4+.01*c,
                                self.cfg['total_column']: 1+c, self.cfg['target']: .3+.01*c})

    def test_bounds_and_zero_total(self):
        model = ThicknessNet(self.cfg)
        x = torch.randn(100, 3)*10
        total = torch.rand(100, 1)*20
        y = model(x, total)
        self.assertTrue(bool(((y >= 0) & (y <= total)).all()))
        self.assertTrue(bool((model(x, torch.zeros_like(total)) == 0).all()))

    def test_group_disjoint_and_extrapolation(self):
        for mode in ('grouped', 'extrapolation'):
            tr, va, te = split_indices(self.df, self.cfg, mode)
            group = self.cfg['group_column']
            sets = [set(self.df.iloc[a][group]) for a in (tr, va, te)]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertEqual(len(set(tr) | set(va) | set(te)), len(self.df))
            if mode == 'extrapolation':
                self.assertGreater(min(sets[2]), max(sets[0] | sets[1]))

    def test_scaler_training_only_and_restore(self):
        prep = Preprocessor(self.cfg).fit(self.df.iloc[:10])
        np.testing.assert_allclose(prep.scaler.mean_, prep.raw(self.df.iloc[:10]).mean(axis=0))
        restored = Preprocessor.restore(self.cfg, prep.export())
        np.testing.assert_array_equal(prep.transform(self.df), restored.transform(self.df))

    def test_regularizer_backprop_and_sensitivity(self):
        prep = Preprocessor(self.cfg).fit(self.df)
        model = ThicknessNet(self.cfg)
        s, m = regularizers(model, *prep.tensors(self.df), prep, 1., self.cfg)
        s.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None), 0.)
        grads = sensitivity(model, prep, self.df)
        eps = 1e-3
        for col in self.cfg['features']:
            plus, minus = self.df.copy(), self.df.copy()
            plus[col] += eps
            minus[col] -= eps
            numeric = (predict(model, prep, plus)-predict(model, prep, minus))/(2*eps)
            np.testing.assert_allclose(grads[col], numeric, atol=2e-3, rtol=.04)

    def test_training_checkpoint_fidelity(self):
        cfg = {**self.cfg, 'epochs': 12, 'patience': 5}
        model, prep, history = fit_network(self.df.iloc[:20], self.df.iloc[20:], cfg)
        self.assertTrue(np.isfinite([h['val_loss'] for h in history]).all())
        clone = ThicknessNet(cfg)
        clone.load_state_dict(model.state_dict())
        np.testing.assert_array_equal(predict(model, prep, self.df), predict(clone, Preprocessor.restore(cfg, prep.export()), self.df))


if __name__ == '__main__':
    unittest.main()
