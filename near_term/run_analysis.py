"""Audited experimental-only validation. Run from any directory; no legacy pickles.

Historical artifacts remain immutable. Outputs replace only near_term/results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "models/feature engineering v1/data FE-V1"
OUT = ROOT / "near_term/results"
SEED = 42
TARGETS = ["bonded_nm", "mobile_nm"]
DESCRIPTORS = ["polarity", "viscosity_cp", "boiling_point_k", "surface_tension_mn_m"]
PROPERTIES = {"hexane": [3.9, .377, 342.039, 17.89],
              "toluene": [2.7, .68, 383.75, 29.46],
              "ethyl_acetate": [.7, .423, 350.372, 24.]}
RENAME = {"Concentration (g/mL)": "concentration_g_l",
          "Uncoated Layer (nm)": "uncoated_nm", "Total Thickness (nm)": "total_nm",
          "Bonded Thickness (nm)": "bonded_nm"}


def physical_mask(df):
    cols = ["concentration_g_l", "total_nm", "bonded_nm"]
    return (np.isfinite(df[cols]).all(axis=1) & (df[cols] >= 0).all(axis=1)
            & (df.total_nm >= df.bonded_nm))


def load_data():
    frames, sources = [], []
    for solvent, filename in [("hexane", "agg.data.xlsx"),
                              ("toluene", "toluene+pdms.csv"),
                              ("ethyl_acetate", "ethyl acetate+pdms.csv")]:
        path = DATA / filename
        df = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df = df.rename(columns=RENAME)
        df["source_path"] = path.relative_to(ROOT).as_posix()
        df["source_row"] = np.arange(len(df)) + 2
        df["record_id"] = [f"{solvent}:{i}" for i in df.source_row]
        df["solvent"] = solvent
        df["record_type"] = "legacy_individual_reading"
        if solvent == "toluene":
            df.loc[df.source_row <= 16, "record_type"] = "workbook_summary"
        df["concentration_g_ml"] = df.concentration_g_l / 1000
        df["mobile_nm"] = df.total_nm - df.bonded_nm
        df["group_id"] = [f"{solvent}:{c:.12g}" for c in df.concentration_g_l]
        for col, value in zip(DESCRIPTORS, PROPERTIES[solvent]):
            df[col] = value
        df["physical_valid"] = physical_mask(df)
        df["primary_eligible"] = df.physical_valid & (df.record_type != "workbook_summary")
        df["exclusion_reason"] = np.where(~df.physical_valid, "invalid_physical_target",
                                  np.where(df.record_type == "workbook_summary",
                                           "summary_may_overlap_individual_readings", ""))
        frames.append(df)
        sources.append({"path": path.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(df)})
    return pd.concat(frames, ignore_index=True), sources


def workbook_inventory():
    """Record exact unit-evidence cells and sheets without inferring new experiments."""
    rows = []
    for path in [ROOT / "PDMS with Toluene 2.xlsx", ROOT / "Ethyl Acetate+PDMS.xlsx",
                 DATA / "Hexane+PDMS-new.xlsx"]:
        wb = openpyxl.load_workbook(path, data_only=True)
        for sheet in wb:
            evidence = [{"cell": c.coordinate, "text": str(c.value)} for row in sheet for c in row
                        if c.value is not None and "g/L" in str(c.value)]
            rows.append({"path": path.relative_to(ROOT).as_posix(), "sheet": sheet.title,
                         "rows": sheet.max_row, "columns": sheet.max_column,
                         "unit_evidence": evidence,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "external_status": "not_independent_existing_repository_source"})
    return rows


def feature_frame(df, transformed=False):
    # No measured thickness, target-derived quantity, or solvent category in X.
    x = df[["concentration_g_l"] + DESCRIPTORS].copy()
    if transformed:
        c = np.log1p(x.pop("concentration_g_l"))
        x["log1p_concentration_g_l"] = c
        for col in DESCRIPTORS:
            x[f"log_concentration_x_{col}"] = c * x[col]
    return x


def model_specs():
    return {"mean": (DummyRegressor(), False),
            "ridge_raw": (make_pipeline(StandardScaler(), Ridge(alpha=10)), False),
            "ridge_log_interactions": (make_pipeline(StandardScaler(), Ridge(alpha=10)), True),
            "rf_raw": (RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                             max_features=1.0, random_state=SEED, n_jobs=1), False),
            "rf_log_interactions": (RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                          max_features=1.0, random_state=SEED, n_jobs=1), True)}


def splits(df):
    g = df.group_id.to_numpy()
    for fold, (tr, te) in enumerate(GroupKFold(5).split(df, groups=g), 1):
        yield "grouped_cv", str(fold), tr, te
    for solvent in sorted(df.solvent.unique()):
        te = np.flatnonzero(df.solvent.to_numpy() == solvent)
        tr = np.flatnonzero(df.solvent.to_numpy() != solvent)
        yield "leave_one_solvent_out", solvent, tr, te
    # Deterministic per-solvent concentration extremes; boundaries use X only.
    for side in ["low", "high"]:
        test_groups = []
        for _, s in df.groupby("solvent"):
            values = np.sort(s.concentration_g_l.unique())
            n = max(1, math.ceil(.2 * len(values)))
            chosen = values[:n] if side == "low" else values[-n:]
            test_groups.extend(s.loc[s.concentration_g_l.isin(chosen), "group_id"])
        test = df.group_id.isin(test_groups).to_numpy()
        yield f"ood_{side}_concentration", side, np.flatnonzero(~test), np.flatnonzero(test)


def conformal_radius(scores, alpha=.2):
    """Finite-sample group split-conformal quantile; never cap an infinite quantile."""
    scores = np.sort(np.asarray(scores))
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    return float(scores[k - 1]) if 0 < k <= len(scores) else float("inf")


def metrics(actual, pred):
    return {"rmse_nm": float(np.sqrt(mean_squared_error(actual, pred))),
            "mae_nm": float(mean_absolute_error(actual, pred)),
            "r2": float(r2_score(actual, pred)) if len(actual) > 1 and np.var(actual) > 0 else np.nan}


def evaluate(df, cohort):
    predictions, assignments, fold_metrics = [], [], []
    for protocol, fold, train, test in splits(df):
        train_groups, test_groups = set(df.iloc[train].group_id), set(df.iloc[test].group_id)
        assert train_groups.isdisjoint(test_groups)
        proper, calibration = next(GroupShuffleSplit(n_splits=1, test_size=.35,
                                      random_state=SEED).split(train, groups=df.iloc[train].group_id))
        fit, cal = train[proper], train[calibration]
        for role, idx in [("proper_train", fit), ("calibration", cal), ("test", test)]:
            for row in df.iloc[idx].itertuples():
                assignments.append(dict(cohort=cohort, protocol=protocol, fold=fold, role=role,
                                        record_id=row.record_id, group_id=row.group_id))
        for name, (estimator, transformed) in model_specs().items():
            x = feature_frame(df, transformed)
            for target in TARGETS:
                y = df[target].to_numpy()
                # Point scores use all outer-training rows; intervals use a separate
                # proper-training estimator, never recalibrated on outer-test labels.
                point = clone(estimator).fit(x.iloc[train], y[train])
                pred = np.maximum(0, point.predict(x.iloc[test]))
                interval_model = clone(estimator).fit(x.iloc[fit], y[fit])
                cal_pred = np.maximum(0, interval_model.predict(x.iloc[cal]))
                scores = pd.DataFrame({"group": df.iloc[cal].group_id.to_numpy(),
                                       "error": abs(y[cal] - cal_pred)}).groupby("group").error.max()
                radius = conformal_radius(scores)
                center = np.maximum(0, interval_model.predict(x.iloc[test]))
                lower, upper = np.maximum(0, center - radius), center + radius
                rows = df.iloc[test][["record_id", "group_id", "solvent", "concentration_g_l"]].copy()
                rows = rows.assign(cohort=cohort, protocol=protocol, fold=fold, model=name,
                                   target=target, actual=y[test], prediction=pred,
                                   interval_prediction=center, lower_80=lower, upper_80=upper,
                                   calibration_groups=len(scores), radius=radius)
                predictions.append(rows)
                fold_metrics.append(dict(cohort=cohort, protocol=protocol, fold=fold, model=name,
                    target=target, n_train=len(train), n_test=len(test),
                    n_train_groups=len(train_groups), n_test_groups=len(test_groups),
                    coverage_80=float(np.mean((y[test] >= lower) & (y[test] <= upper))),
                    mean_width_nm=float(np.mean(upper-lower)), **metrics(y[test], pred)))
        print(cohort, protocol, fold, "complete", flush=True)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(assignments), pd.DataFrame(fold_metrics)


def aggregate(predictions):
    rows = []
    for key, p in predictions.groupby(["cohort", "protocol", "model", "target"]):
        group_errors = p.assign(sq=(p.actual-p.prediction)**2).groupby("group_id").sq.mean().to_numpy()
        rng = np.random.default_rng(SEED)
        boot = np.sqrt(rng.choice(group_errors, (2000, len(group_errors)), replace=True).mean(axis=1))
        covered = p.assign(covered=(p.actual >= p.lower_80) & (p.actual <= p.upper_80))
        rows.append(dict(zip(["cohort", "protocol", "model", "target"], key),
                         n=len(p), groups=p.group_id.nunique(), **metrics(p.actual, p.prediction),
                         group_weighted_rmse_nm=float(np.sqrt(group_errors.mean())),
                         descriptive_rmse_p025=float(np.quantile(boot, .025)),
                         descriptive_rmse_p975=float(np.quantile(boot, .975)),
                         row_coverage_80=float(covered.covered.mean()),
                         group_simultaneous_coverage_80=float(covered.groupby("group_id").covered.all().mean()),
                         mean_width_nm=float((p.upper_80-p.lower_80).mean())))
    return pd.DataFrame(rows)


def constrained_bootstrap(df, n=300, seed=SEED):
    """Optional TRAINING-ONLY augmentation preserving parent IDs and hT >= hB.

    Called only for a generator audit here; synthetic rows never enter validation.
    Concentration and solvent descriptors are inherited; perturb nonnegative
    bonded and mobile components multiplicatively, then reconstruct total.
    """
    if not physical_mask(df).all():
        raise ValueError("Augmentation requires physically valid training rows")
    rng = np.random.default_rng(seed)
    out = df.iloc[rng.integers(0, len(df), n)].copy().reset_index(drop=True)
    out["parent_record_id"] = out.record_id
    out["record_id"] = [f"synthetic:{seed}:{i}" for i in range(n)]
    for target in TARGETS:
        out[target] *= rng.lognormal(mean=-.5*.05**2, sigma=.05, size=n)
    out["total_nm"] = out.bonded_nm + out.mobile_nm
    out["record_type"] = "synthetic_audit_only"
    assert physical_mask(out).all()
    return out


def plots(summary, predictions):
    protocols = ["grouped_cv", "leave_one_solvent_out", "ood_low_concentration", "ood_high_concentration"]
    labels = ["Grouped CV", "Unseen solvent", "Low concentration", "High concentration"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, target in zip(axes, TARGETS):
        s = summary[(summary.cohort == "primary") & (summary.target == target)]
        for name in model_specs():
            d = s[s.model == name].set_index("protocol").loc[protocols]
            ax.plot(labels, d.group_weighted_rmse_nm, marker="o", label=name)
        ax.set(title=target.replace("_nm", " thickness"), ylabel="Group-weighted RMSE (nm)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Experimental-only forecasting: no measured thickness inputs")
    fig.tight_layout()
    fig.savefig(OUT / "validation_rmse.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, target in zip(axes, TARGETS):
        p = predictions[(predictions.cohort == "primary") & (predictions.protocol == "grouped_cv")
                        & (predictions.model == "rf_raw") & (predictions.target == target)]
        for solvent, s in p.groupby("solvent"):
            ax.scatter(s.actual, s.prediction, label=solvent, alpha=.65, s=20)
        bounds = [0, max(p.actual.max(), p.prediction.max())*1.05]
        ax.plot(bounds, bounds, "k--", lw=1)
        ax.set(title=target, xlabel="Measured (nm)", ylabel="Outer-fold prediction (nm)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Fixed random forest, grouped cross-validation")
    fig.tight_layout()
    fig.savefig(OUT / "grouped_parity.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-only", action="store_true", help="Skip summary-inclusion sensitivity")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data, sources = load_data()
    data.to_csv(OUT / "canonical_records.csv", index=False)
    data.loc[~data.primary_eligible].to_csv(OUT / "excluded_records.csv", index=False)
    (OUT / "workbook_inventory.json").write_text(json.dumps(workbook_inventory(), indent=2), encoding="utf-8")
    legacy = pd.read_csv(DATA / "synthetic_data_improved.csv")
    legacy.columns = legacy.columns.str.strip()
    legacy = legacy.rename(columns=RENAME)
    legacy["source_row"] = np.arange(len(legacy)) + 2
    legacy["negative_mobile"] = legacy.total_nm < legacy.bonded_nm
    legacy["negative_concentration"] = legacy.concentration_g_l < 0
    legacy["negative_thickness"] = (legacy[["total_nm", "bonded_nm"]] < 0).any(axis=1)
    legacy.loc[~physical_mask(legacy)].to_csv(OUT / "legacy_synthetic_exclusions.csv", index=False)
    audit = {"input_rows": len(data), "primary_rows": int(data.primary_eligible.sum()),
             "physically_invalid_rows": int((~data.physical_valid).sum()),
             "summary_rows_excluded": int((data.record_type == "workbook_summary").sum()),
             "legacy_synthetic_rows": len(legacy),
             "legacy_synthetic_invalid": int((~physical_mask(legacy)).sum()),
             "legacy_synthetic_negative_mobile": int(legacy.negative_mobile.sum()),
             "legacy_synthetic_negative_concentration": int(legacy.negative_concentration.sum()),
             "legacy_synthetic_negative_thickness": int(legacy.negative_thickness.sum()),
             "external_test_status": "unavailable: no independently collected new experiments supplied"}
    predictions, assignments, folds = [], [], []
    cohorts = {"primary": data[data.primary_eligible]}
    if not args.primary_only:
        cohorts["including_summaries_sensitivity"] = data[data.physical_valid]
    for name, df in cohorts.items():
        p, a, f = evaluate(df.reset_index(drop=True), name)
        predictions.append(p); assignments.append(a); folds.append(f)
    predictions = pd.concat(predictions, ignore_index=True)
    summary = aggregate(predictions)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    pd.concat(assignments).to_csv(OUT / "split_assignments.csv", index=False)
    pd.concat(folds).to_csv(OUT / "fold_metrics.csv", index=False)
    summary.to_csv(OUT / "summary_metrics.csv", index=False)
    # Demonstration subset belongs to first grouped fold's proper-training set only.
    primary = cohorts["primary"].reset_index(drop=True)
    _, _, tr, _ = next(splits(primary))
    proper, _ = next(GroupShuffleSplit(n_splits=1, test_size=.35, random_state=SEED).split(
        tr, groups=primary.iloc[tr].group_id))
    synthetic = constrained_bootstrap(primary.iloc[tr[proper]])
    synthetic.to_csv(OUT / "constrained_synthetic_audit.csv", index=False)
    audit["constrained_synthetic_rows"] = len(synthetic)
    audit["constrained_synthetic_invalid"] = int((~physical_mask(synthetic)).sum())
    (OUT / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    manifest = {"seed": SEED, "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "sklearn": sklearn.__version__, "sources": sources,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "features": list(feature_frame(data)), "model_parameters": {k: str(v[0]) for k,v in model_specs().items()},
                "interval_nominal_coverage": .8, "calibration_group_fraction": .35,
                "external_test_used": False, "synthetic_used_in_validation": False,
                "cohorts": list(cohorts)}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plots(summary, predictions)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
