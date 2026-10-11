import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True  # Preserve tracked legacy bytecode during imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import solvent_analysis as s


class SolventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(s.PRIOR / "canonical_records.csv")
        cls.data = cls.data[cls.data.primary_eligible]

    def test_agreement_with_all_legacy_analytical_implementations(self):
        for solvent, frame in self.data.groupby("solvent"):
            path = s.ROOT / "analytical model comparision" / solvent.replace("_", " ") / "src/analytical_models.py"
            name = f"legacy_{solvent}"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            legacy = frame.rename(columns={"concentration_g_l": "Concentration (g/mL)"})
            np.testing.assert_allclose(s.lld(frame), module.landau_levich_wet_mobile_layer_model(legacy), rtol=1e-12)

    def test_scaling_and_zero_boundary(self):
        base = s.lld(self.data)
        np.testing.assert_allclose(s.lld(self.data, speed_mm_s=8), base*4)
        np.testing.assert_allclose(s.lld(self.data, density_factor=4), base/2)
        zero = self.data.iloc[:2].copy()
        zero["concentration_g_l"] = 0.
        np.testing.assert_array_equal(s.lld(zero), [0., 0.])
        with self.assertRaises(ValueError):
            s.lld(zero, viscosity_factor=0)

    def test_inverse_diagnostics(self):
        group = pd.read_csv(s.OUT / "condition_diagnostics.csv")
        ratio = group.observed_over_lld.to_numpy()
        np.testing.assert_allclose(group.equivalent_speed_or_viscosity_factor**(2/3), ratio)
        np.testing.assert_allclose(group.equivalent_density_factor**(-.5), ratio)
        np.testing.assert_allclose(group.equivalent_tension_factor**(-1/6), ratio)

    def test_condition_weighting_and_saturation_recovery(self):
        data = self.data[self.data.solvent == "hexane"].copy()
        data["mobile_nm"] = 2.5*s.lld(data)
        self.assertAlmostEqual(s.fit_comparator(data, "scaled_lld")["scale"], 2.5)
        extra = pd.concat([data, data[data.group_id == data.group_id.iloc[0]]]*2)
        self.assertAlmostEqual(s.fit_comparator(extra, "scaled_lld")["scale"], 2.5)
        data["mobile_nm"] = s.saturation(data.concentration_g_l, 2., .3)
        fit = s.fit_comparator(data, "saturating_mobile")
        self.assertAlmostEqual(fit["amplitude_nm"], 2., places=5)
        self.assertAlmostEqual(fit["half_concentration_g_l"], .3, places=5)

    def test_saved_fit_isolation(self):
        a = pd.read_csv(s.OUT / "fit_membership.csv")
        for (cohort, protocol, fold, solvent), part in a.groupby(["cohort", "protocol", "fold", "evaluation_solvent"]):
            tr, te = part[part.role == "train"], part[part.role == "test"]
            self.assertFalse(set(tr.record_id) & set(te.record_id))
            self.assertFalse(set(tr.group_id) & set(te.group_id))
            if protocol == "leave_one_solvent_out":
                self.assertTrue(all(not g.startswith(solvent + ":") for g in tr.group_id))

    def test_reused_predictions_and_identical_test_rows(self):
        new = pd.read_csv(s.OUT / "mobile_predictions.csv")
        old = pd.read_csv(s.PRIOR / "predictions.csv")
        old = old[old.target == "mobile_nm"]
        keys = ["cohort", "protocol", "fold", "model", "record_id"]
        match = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
        self.assertEqual(len(match), len(old))
        # CSV float parsing/serialization may round the last binary digit.
        np.testing.assert_allclose(match.prediction_old, match.prediction_new, rtol=1e-14, atol=1e-15)
        for _, part in new.groupby(["cohort", "protocol", "fold"]):
            expected = set(part[part.model == "rf_raw"].record_id)
            for _, model in part.groupby("model"):
                self.assertEqual(set(model.record_id), expected)
        self.assertTrue(np.isfinite(new.prediction).all())
        self.assertTrue((new.prediction >= 0).all())


if __name__ == "__main__":
    unittest.main()
