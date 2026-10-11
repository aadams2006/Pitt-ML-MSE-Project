"""Checks for physical units, invalid-data policy and experiment provenance."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from core import load_data, Preprocessor, ThicknessNet, regularizers
from physics import mobile_lld_nm, hydrodynamic_loss, validate_physics
from run_tracking import start_run, finish_run

HERE = Path(__file__).resolve().parent


class PhysicsTrackingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((HERE/'config.json').read_text())

    def test_synthetic_exclusion_preserves_source_rows(self):
        df = pd.DataFrame({'Concentration (g/mL)': [.1, -.1, .2, .3],
                           'Uncoated Layer (nm)': [1., 1., 1., 1.],
                           'Total Thickness (nm)': [1., 1., 1., 1.],
                           'Bonded Thickness (nm)': [.2, .2, 2., .4]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'data.csv'
            df.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_data(path, self.cfg)
            clean, audit = load_data(path, self.cfg, invalid_policy='exclude', return_audit=True)
            self.assertEqual(clean.index.tolist(), [0, 3])
            self.assertEqual(audit['excluded_source_rows'], [1, 2])

    def fixture_physics(self):
        return pd.DataFrame({'Withdrawal Velocity (mm/s)': [1., 2.], 'Viscosity (cP)': [1., 1.],
                             'Surface Tension (mN/m)': [20., 20.], 'Solution Density (kg/m3)': [1000., 1000.],
                             'Wet-to-Dry Mobile Factor': [.01, .01], 'LLD Applicable': [1, 1]})

    def test_lld_units_scaling_and_target_separation(self):
        df = self.fixture_physics()
        expected = .9458*np.sqrt(.02/(1000*9.81))*(.001*.001/.02)**(2/3)*1e9*.01
        values = mobile_lld_nm(df)
        self.assertAlmostEqual(values[0], expected, places=6)
        self.assertAlmostEqual(values[1]/values[0], 2**(2/3), places=6)
        cfg = {**self.cfg, 'hydrodynamic_weight': .01}
        validate_physics(df, cfg)
        total = torch.tensor(values+1, dtype=torch.float64).reshape(-1, 1)
        pred = torch.ones_like(total, requires_grad=True)
        self.assertLess(float(hydrodynamic_loss(pred, total, df, 1., cfg)), 1e-10)
        df.loc[1, 'LLD Applicable'] = 0
        total[1] = 9999
        self.assertLess(float(hydrodynamic_loss(pred, total, df, 1., cfg)), 1e-10)

    def test_missing_physics_and_high_ca_rejected(self):
        with self.assertRaises(ValueError):
            validate_physics(pd.DataFrame(), {**self.cfg, 'hydrodynamic_weight': 1.})
        df = self.fixture_physics()
        df['Withdrawal Velocity (mm/s)'] = 1e6
        with self.assertRaises(ValueError):
            validate_physics(df, {**self.cfg, 'hydrodynamic_weight': 1.})
        with self.assertRaises(ValueError):
            validate_physics(pd.DataFrame(), {**self.cfg, 'velocity_smooth_weight': 1.})

    def test_velocity_curvature_and_monotonic_penalty(self):
        cfg = {**self.cfg, 'features': self.cfg['features']+['Withdrawal Velocity (mm/s)'],
               'velocity_smooth_weight': .1, 'monotonic': {'Concentration (g/mL)': 1}, 'monotonic_weight': .1}
        df = pd.DataFrame({'Concentration (g/mL)': [.1, .2, .4], 'Uncoated Layer (nm)': [1., 1.1, 1.2],
                           'Total Thickness (nm)': [1., 2., 3.], 'Withdrawal Velocity (mm/s)': [1., 2., 3.]})
        prep = Preprocessor(cfg).fit(df)
        model = ThicknessNet(cfg)
        with torch.no_grad():
            for layer in (model.network[0], model.network[2], model.network[4]):
                layer.weight.zero_()
                layer.bias.zero_()
            model.network[0].weight[0, 0] = -1.
            model.network[0].weight[0, 3] = .5
            model.network[2].weight[0, 0] = 1.
            model.network[4].weight[0, 0] = 1.
        smooth, mono, velocity = regularizers(model, *prep.tensors(df), prep, 1., cfg, return_velocity=True)
        self.assertGreater(float(mono.detach()), 0.)
        self.assertGreater(float(velocity.detach()), 0.)
        (smooth+mono+velocity).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_failed_run_has_timestamp_version_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'input.csv'
            path.write_text('x\n1\n')
            out, record = start_run(Path(folder)/'run', self.cfg, path, 'grouped', 1., 'experimental', 'error')
            self.assertEqual(record['status'], 'running')
            self.assertTrue(record['started_at_new_york'])
            self.assertTrue((out/'source'/'core.py').is_file())
            finish_run(out, record, error=ValueError('test failure'))
            saved = json.loads((out/'run_record.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertIn('test failure', saved['error'])
            self.assertEqual(saved['model_version'], '0.2.0')
            self.assertGreaterEqual(saved['duration_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
