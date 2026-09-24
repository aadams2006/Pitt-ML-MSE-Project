"""Solvent-specific mobile-layer diagnosis on the already audited experimental data.

Reuses the previous outer splits; does not rerun the completed RF/Ridge study.
Calibrated comparators are exploratory and never fitted on outer-test outcomes.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from run_analysis import ROOT, OUT as PRIOR, splits, metrics

OUT = ROOT / "near_term/solvent_results"
DENSITY = {"hexane": 655., "toluene": 867., "ethyl_acetate": 902.}
MODELS = ["fixed_lld", "local_mean", "scaled_lld", "saturating_mobile"]


def lld(df, speed_mm_s=1., viscosity_factor=1., tension_factor=1., density_factor=1.):
    """Legacy LLD + volume-fraction approximation, with explicit canonical units."""
    values = np.asarray([speed_mm_s, viscosity_factor, tension_factor, density_factor])
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Positive finite physical parameters required")
    c = df.concentration_g_l.to_numpy(dtype=float)
    if not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("Finite nonnegative concentration required")
    mu = df.viscosity_cp.to_numpy() * 1e-3 * viscosity_factor
    gamma = df.surface_tension_mn_m.to_numpy() * 1e-3 * tension_factor
    rho = df.solvent.map(DENSITY).to_numpy() * density_factor
    if not np.isfinite(rho).all():
        raise ValueError("Unrecognized solvent")
    wet_nm = .94 * (mu * speed_mm_s * 1e-3)**(2/3) / (gamma**(1/6) * np.sqrt(rho*9.81)) * 1e9
    return wet_nm * c / (c + 965.)


def saturation(c, amplitude, half_concentration):
    return amplitude * np.asarray(c) / (half_concentration + np.asarray(c))


def fit_comparator(train, model):
    """Equal weight to each solvent/concentration condition, not each reading."""
    g = train.assign(lld_nm=lld(train)).groupby("group_id").agg(
        concentration_g_l=("concentration_g_l", "first"),
        actual=("mobile_nm", "mean"), lld_nm=("lld_nm", "first"))
    if model == "fixed_lld":
        return {"scale": 1.}
    if model == "local_mean":
        return {"mean_mobile_nm": float(g.actual.mean())}
    if model == "scaled_lld":
        x, y = g.lld_nm.to_numpy(), g.actual.to_numpy()
        return {"scale": float(max(0., np.dot(x, y)/np.dot(x, x)))}
    if model != "saturating_mobile":
        raise ValueError(model)
    c, y = g.concentration_g_l.to_numpy(), g.actual.to_numpy()
    bounds = (np.log([1e-6, 1e-6]), np.log([1e4, 1e5]))
    best = None
    # Fixed multistart rule; training concentrations only, no test selection.
    for k in np.geomspace(max(1e-5, c.min()/10), max(1e-4, c.max()*10), 7):
        amp = max(1e-5, np.max(y))
        init = np.clip(np.log([amp, k]), bounds[0]+1e-9, bounds[1]-1e-9)
        fit = least_squares(lambda p: saturation(c, *np.exp(p))-y, init,
                            bounds=bounds, max_nfev=2000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if fit.success and (best is None or fit.cost < best.cost):
            best = fit
    if best is None:
        raise RuntimeError("All saturation optimization starts failed")
    amp, k = np.exp(best.x)
    return {"amplitude_nm": float(amp), "half_concentration_g_l": float(k),
            "bound_hit": bool(np.any(best.x-bounds[0] < 1e-3) or np.any(bounds[1]-best.x < 1e-3)),
            "training_condition_mse": float(2*best.cost / len(c))}


def predict(df, model, params):
    if model == "local_mean":
        return np.full(len(df), params["mean_mobile_nm"])
    if model in ["fixed_lld", "scaled_lld"]:
        return lld(df)*params["scale"]
    return saturation(df.concentration_g_l, params["amplitude_nm"], params["half_concentration_g_l"])


def evaluate(df, cohort):
    predictions, parameters, membership = [], [], []
    for protocol, fold, tr, te in splits(df):
        train, test = df.iloc[tr], df.iloc[te]
        assert not set(train.group_id) & set(test.group_id)
        for solvent, test_part in test.groupby("solvent"):
            local = train[train.solvent == solvent]
            scope = "same_solvent" if len(local) else "pooled_other_solvents"
            fit_data = local if len(local) else train
            for role, part in [("train", fit_data), ("test", test_part)]:
                for r in part.itertuples():
                    membership.append(dict(cohort=cohort, protocol=protocol, fold=fold,
                        evaluation_solvent=solvent, role=role, record_id=r.record_id, group_id=r.group_id))
            for model in MODELS:
                params = fit_comparator(fit_data, model)
                parameters.append(dict(cohort=cohort, protocol=protocol, fold=fold, solvent=solvent,
                    model=model, fit_scope=scope, training_groups=fit_data.group_id.nunique(), **params))
                p = test_part[["record_id", "group_id", "solvent", "concentration_g_l"]].copy()
                p = p.assign(cohort=cohort, protocol=protocol, fold=str(fold), model=model,
                             actual=test_part.mobile_nm, prediction=predict(test_part, model, params))
                predictions.append(p)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(parameters), pd.DataFrame(membership)


def summarize(predictions):
    rows = []
    keys = ["cohort", "protocol", "model"]
    for key, p in predictions.groupby(keys):
        for solvent, part in [("all", p)] + list(p.groupby("solvent")):
            e = part.assign(squared=(part.actual-part.prediction)**2).groupby("group_id").squared.mean()
            rows.append(dict(zip(keys, key), solvent=solvent, n=len(part), groups=len(e),
                             group_weighted_rmse_nm=float(np.sqrt(e.mean())), **metrics(part.actual, part.prediction)))
    return pd.DataFrame(rows)


def diagnose(df):
    g = df.assign(lld_nm=lld(df)).groupby(["solvent", "group_id"]).agg(
        concentration_g_l=("concentration_g_l", "first"), n=("mobile_nm", "size"),
        mobile_mean_nm=("mobile_nm", "mean"), mobile_min_nm=("mobile_nm", "min"),
        mobile_max_nm=("mobile_nm", "max"), lld_nm=("lld_nm", "first")).reset_index()
    ratio = g.mobile_mean_nm / g.lld_nm
    g["observed_over_lld"] = ratio
    # One-factor inversions, not estimates of actual physical properties.
    g["equivalent_speed_or_viscosity_factor"] = ratio**1.5
    g["equivalent_tension_factor"] = np.where(ratio > 0, ratio**(-6), np.nan)
    g["equivalent_density_factor"] = np.where(ratio > 0, ratio**(-2), np.nan)
    g["dilute_vs_additive_relative_difference"] = g.concentration_g_l / 965.
    summaries = []
    for solvent, s in g.groupby("solvent"):
        slope = np.polyfit(np.log(s.concentration_g_l), np.log(s.mobile_mean_nm), 1)[0]
        summaries.append(dict(solvent=solvent, groups=len(s),
            concentration_min_g_l=s.concentration_g_l.min(), concentration_max_g_l=s.concentration_g_l.max(),
            descriptive_loglog_slope=slope,
            descriptive_spearman=float(spearmanr(s.concentration_g_l, s.mobile_mean_nm).statistic),
            ratio_min=s.observed_over_lld.min(), ratio_max=s.observed_over_lld.max(),
            ratio_span=s.observed_over_lld.max()/s.observed_over_lld.min(),
            minimum_lld_loglog_slope=float(965/(965+s.concentration_g_l.max())),
            max_volume_conversion_difference_percent=float(100*s.concentration_g_l.max()/965)))
    return g, pd.DataFrame(summaries)


def figures(groups, data):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    full_params = []
    for i, solvent in enumerate(sorted(data.solvent.unique())):
        s = groups[groups.solvent == solvent]
        local = data[data.solvent == solvent]
        grid = local.iloc[np.zeros(150, dtype=int)].copy().reset_index(drop=True)
        grid["concentration_g_l"] = np.geomspace(s.concentration_g_l.min(), s.concentration_g_l.max(), 150)
        ax = axes[0, i]
        ax.scatter(s.concentration_g_l, s.mobile_mean_nm, color="black", s=22, label="Measured condition mean")
        for model in MODELS:
            params = fit_comparator(local, model)
            full_params.append(dict(solvent=solvent, model=model, purpose="descriptive_full_data_fit_only", **params))
            ax.plot(grid.concentration_g_l, predict(grid, model, params),
                    linestyle="-" if model == "fixed_lld" else "--", label=model.replace("_", " "))
        ax.set(xscale="log", yscale="log", title=solvent.replace("_", " "), ylabel="Mobile thickness (nm)")
        ax.grid(alpha=.2)
        ax = axes[1, i]
        ax.scatter(s.concentration_g_l, s.observed_over_lld, color="#174b80", s=22)
        ax.axhline(1, color="black", linestyle="--")
        ax.axhspan(.25, 4, alpha=.12, color="#b39835", label="Hypothetical 0.5–2x parameter sweep")
        ax.set(xscale="log", yscale="log", xlabel="Concentration (g/L)", ylabel="Measured / fixed LLD")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Solvent-specific mobile-layer diagnostics (dashed fits are descriptive)")
    fig.tight_layout()
    fig.savefig(OUT / "solvent_diagnostics.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(full_params).to_csv(OUT / "descriptive_fit_parameters.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = PRIOR / "canonical_records.csv"
    data = pd.read_csv(source)
    primary = data[data.primary_eligible].reset_index(drop=True)
    frames, params, assignments = [], [], []
    for cohort, df in [("primary", primary), ("including_summaries_sensitivity", data[data.physical_valid].reset_index(drop=True))]:
        p, f, a = evaluate(df, cohort)
        frames.append(p); params.append(f); assignments.append(a)
    new_predictions = pd.concat(frames, ignore_index=True)
    # Reuse completed model results without refitting or changing their forecasts.
    old = pd.read_csv(PRIOR / "predictions.csv")
    old = old[old.target == "mobile_nm"].copy()
    old["fold"] = old.fold.astype(str)
    shared_cols = list(new_predictions.columns)
    predictions = pd.concat([new_predictions, old[shared_cols]], ignore_index=True)
    predictions.to_csv(OUT / "mobile_predictions.csv", index=False)
    pd.concat(params, ignore_index=True).to_csv(OUT / "outer_fit_parameters.csv", index=False)
    pd.concat(assignments, ignore_index=True).to_csv(OUT / "fit_membership.csv", index=False)
    summarize(predictions).to_csv(OUT / "mobile_metrics.csv", index=False)
    groups, diagnostic_summary = diagnose(primary)
    groups.to_csv(OUT / "condition_diagnostics.csv", index=False)
    diagnostic_summary.to_csv(OUT / "solvent_diagnostics.csv", index=False)
    sensitivity = []
    for u, mu, gamma, rho in itertools.product([.5, 1., 2.], repeat=4):
        pred = lld(primary, u, mu, gamma, rho)
        for solvent, s in primary.assign(pred=pred).groupby("solvent"):
            sensitivity.append(dict(solvent=solvent, speed_factor=u, viscosity_factor=mu,
                tension_factor=gamma, density_factor=rho,
                multiplier=u**(2/3)*mu**(2/3)*gamma**(-1/6)*rho**(-.5),
                **metrics(s.mobile_nm, s.pred)))
    pd.DataFrame(sensitivity).to_csv(OUT / "hypothetical_parameter_sweep.csv", index=False)
    figures(groups, primary)
    paths = [Path(__file__), ROOT / "near_term/run_analysis.py", source, PRIOR / "predictions.csv"]
    manifest = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "inputs": [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
                "target": "mobile_nm", "completed_ml_results_reused": True,
                "external_data": False, "split_policy": "unchanged near_term.run_analysis.splits",
                "density_kg_m3": DENSITY, "default_speed_mm_s": 1., "pdms_density_g_ml": .965,
                "fit_weighting": "equal condition means; per-solvent if available, otherwise pooled training solvents",
                "sweep_interpretation": "hypothetical stress range, not measured uncertainties",
                "selection_warning": "post-hoc exploratory comparators, not a new independent validation dataset"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(diagnostic_summary.to_string(index=False))


if __name__ == "__main__":
    main()
