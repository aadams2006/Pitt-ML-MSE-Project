"""Scientific invariants, split isolation, and independently recomputed scores."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_analysis as a


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all, _ = a.load_data()
        cls.df = cls.all[cls.all.primary_eligible].reset_index(drop=True)

    def test_units_and_lineage(self):
        row = self.all[self.all.record_id == "toluene:2"].iloc[0]
        self.assertEqual(row.concentration_g_l, .0025)
        self.assertEqual(row.concentration_g_ml, .0000025)
        self.assertEqual(len(self.all), 142)
        self.assertEqual(len(self.df), 126)
        self.assertTrue(self.all.record_id.is_unique)

    def test_physical_failures_not_clipped(self):
        bad = self.all[~self.all.physical_valid]
        self.assertEqual(len(bad), 1)
        self.assertAlmostEqual(bad.mobile_nm.iloc[0], -.01)
        self.assertTrue((self.df.mobile_nm >= 0).all())
        example = self.df.iloc[:4].copy()
        example.loc[example.index[0], "total_nm"] = np.inf
        example.loc[example.index[1], "bonded_nm"] = -1
        example.loc[example.index[2], "concentration_g_l"] = np.nan
        self.assertEqual(a.physical_mask(example).tolist(), [False, False, False, True])

    def test_no_response_features(self):
        for transform in [False, True]:
            before = a.feature_frame(self.df, transform)
            altered = self.df.copy()
            for col in ["total_nm", "bonded_nm", "mobile_nm", "uncoated_nm"]:
                altered[col] = 9999
            pd.testing.assert_frame_equal(before, a.feature_frame(altered, transform))

    def test_split_isolation_and_ood_direction(self):
        seen = []
        for protocol, fold, train, test in a.splits(self.df):
            tr, te = self.df.iloc[train], self.df.iloc[test]
            self.assertFalse(set(tr.group_id) & set(te.group_id))
            if protocol == "grouped_cv":
                seen.extend(test)
            elif protocol == "leave_one_solvent_out":
                self.assertEqual(set(te.solvent), {fold})
                self.assertNotIn(fold, set(tr.solvent))
            else:
                for solvent, t in te.groupby("solvent"):
                    r = tr[tr.solvent == solvent]
                    if protocol == "ood_low_concentration":
                        self.assertLess(t.concentration_g_l.max(), r.concentration_g_l.min())
                    else:
                        self.assertGreater(t.concentration_g_l.min(), r.concentration_g_l.max())
        self.assertEqual(sorted(seen), list(range(len(self.df))))

    def test_conformal_small_samples(self):
        self.assertTrue(np.isinf(a.conformal_radius([1, 2], alpha=.1)))
        self.assertEqual(a.conformal_radius([1, 2, 3, 4], alpha=.2), 4)

    def test_generator_preserves_constraints_and_parentage(self):
        tr = self.df.iloc[:20]
        synthetic = a.constrained_bootstrap(tr, n=1000)
        self.assertTrue(a.physical_mask(synthetic).all())
        self.assertTrue(set(synthetic.parent_record_id) <= set(tr.record_id))
        np.testing.assert_allclose(synthetic.total_nm, synthetic.bonded_nm + synthetic.mobile_nm)
        pd.testing.assert_frame_equal(synthetic, a.constrained_bootstrap(tr, n=1000))
        with self.assertRaises(ValueError):
            a.constrained_bootstrap(self.all)

    def test_saved_scores_and_calibration_isolation(self):
        p = pd.read_csv(a.OUT / "predictions.csv")
        s = pd.read_csv(a.OUT / "summary_metrics.csv")
        self.assertFalse(p.duplicated(["cohort", "protocol", "model", "target", "record_id"]).any())
        self.assertTrue(np.isfinite(p[["actual", "prediction", "lower_80", "upper_80"]]).all().all())
        self.assertTrue((p.lower_80 <= p.upper_80).all())
        for row in s.itertuples():
            subset = p[(p.cohort == row.cohort) & (p.protocol == row.protocol)
                       & (p.model == row.model) & (p.target == row.target)]
            self.assertAlmostEqual(np.sqrt(np.mean((subset.actual-subset.prediction)**2)), row.rmse_nm)
        assignments = pd.read_csv(a.OUT / "split_assignments.csv")
        for _, split in assignments.groupby(["cohort", "protocol", "fold"]):
            groups = {role: set(part.group_id) for role, part in split.groupby("role")}
            for left, right in [("proper_train", "calibration"), ("proper_train", "test"), ("calibration", "test")]:
                self.assertFalse(groups[left] & groups[right])
        first = assignments[(assignments.cohort == "primary") & (assignments.protocol == "grouped_cv")
                            & (assignments.fold.astype(str) == "1") & (assignments.role == "proper_train")]
        generated = pd.read_csv(a.OUT / "constrained_synthetic_audit.csv")
        self.assertTrue(set(generated.parent_record_id) <= set(first.record_id))


if __name__ == "__main__":
    unittest.main()
