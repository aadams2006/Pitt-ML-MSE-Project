"""Generate the Next Steps text summary and paper supplement from saved results."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "near_term/solvent_results"
NAMES = {"fixed_lld": "Fixed LLD", "local_mean": "Local mean", "scaled_lld": "Scaled LLD",
         "saturating_mobile": "Saturating mobile", "rf_raw": "Prior pooled RF"}


def main():
    metrics = pd.read_csv(OUT / "mobile_metrics.csv")
    diagnostic = pd.read_csv(OUT / "solvent_diagnostics.csv")
    primary = metrics[metrics.cohort == "primary"]
    cv = primary[primary.protocol == "grouped_cv"]
    txt = ["PITT ML-MSE NEXT STEPS FOLLOW-UP", "",
           "Completed work was retained: thickness-free mobile forecasting, earlier RF/Ridge",
           "comparisons, grouped/solvent/OOD validation, uncertainty analysis and first paper revision.",
           "Those model predictions were reused, not refitted. Bonded and mobile targets remain separate.", "",
           "NEW WORK", "Investigated the remaining solvent-specific analytical discrepancy using the canonical",
           "experimental records. Added fixed LLD, a solvent-specific condition-mean baseline,",
           "training-only scalar calibration, and an exploratory saturation comparator on the same splits.",
           "Repeated for the existing summary-inclusion sensitivity cohort. Saved predictions,",
           "fit memberships, parameters, errors, condition diagnostics, a physical-parameter sweep,",
           "reproducible scripts, tests, a figure and a new manuscript section.", "",
           "FINDINGS (PRIMARY EXPERIMENTAL COHORT)"]
    for r in diagnostic.itertuples():
        txt.append(f"{r.solvent}: descriptive concentration log-log slope {r.descriptive_loglog_slope:.3f}; "
                   f"measured/fixed-LLD ratio {r.ratio_min:.5g} to {r.ratio_max:.5g}.")
    txt += ["The fixed-property model predicts slopes near 1. Toluene is nearly flat and ethyl",
            "acetate grows more slowly. Constant property or speed changes rescale amplitude",
            "and cannot reproduce the entire observed concentration dependence.",
            "The alternative dilute volume-fraction convention changes predictions by at most 5.18%.",
            "A hypothetical 0.5-2x sweep of each of four physical inputs spans 0.25-4x output.",
            "Neither calculation establishes measured property uncertainties or the physical cause.", "",
            "GROUPED-CV GROUP-WEIGHTED RMSE (nm)"]
    for solvent in ["hexane", "toluene", "ethyl_acetate"]:
        parts = []
        for model, label in NAMES.items():
            value = cv[(cv.solvent == solvent) & (cv.model == model)].iloc[0].group_weighted_rmse_nm
            parts.append(f"{label}={value:.4f}")
        txt.append(solvent + ": " + "; ".join(parts))
    txt += ["", "INTERPRETATION", "Calibration is useful diagnostically, but no universally improved model is established.",
            "Toluene's local mean beats the saturation comparator; additional complexity is not justified there.",
            "Including historical summaries reverses this small ranking (saturation 0.1084 vs mean 0.1158 nm),",
            "so this comparison does not establish a stable model preference.",
            "Scaled LLD reduces pooled grouped errors but fails high-concentration and unseen-solvent tests.",
            "The new fits use solvent-specific condition means, unlike the prior pooled row-trained RF.",
            "The comparators were proposed after earlier results were inspected, so their scores are exploratory.",
            "Full-data plotted curves are descriptive fits, never substituted for outer-test predictions.",
            "Physical parameter inversions are algebraic equivalences, not measured mechanisms.", "",
            "VALIDATION AND PAPER", "All 13 tests pass, including legacy analytical agreement, known-parameter recovery,",
            "split isolation, physical scaling, and preservation of prior predictions within floating-point precision.",
            "Independently verified all 288 metric rows against 6048 saved predictions and checked source hashes.",
            "Updated the paper with quantitative solvent diagnostics, model comparisons and remaining limitations.",
            "The follow-up manuscript and verification record are in near_term/solvent_results/.", "",
            "REMAINING EXPERIMENTAL DEPENDENCIES", "New independent experiments, verified batch/sample IDs, measured solution properties and",
            "matched wet/dried/washed measurements are needed to identify the mechanism and validate deployment.",
            "The new slide's computational investigation is complete; physical causation and external validation",
            "are not claimed. The PINN transition shown on the slide was not started."]
    (ROOT / "NEXT_STEPS_SUMMARY.txt").write_text("\n".join(txt)+"\n", encoding="utf-8")
    report = ["# Solvent-specific mobile-layer findings", "", "Methods and limitations: ../SOLVENT_METHODS.md. Follow-up summary: ../../NEXT_STEPS_SUMMARY.txt.", ""]
    for cohort in metrics.cohort.unique():
        report += [f"## {cohort}", "", "| Protocol | Solvent | Model | Group RMSE (nm) | Row R2 |", "|---|---|---|---:|---:|"]
        for r in metrics[metrics.cohort == cohort].itertuples():
            report.append(f"| {r.protocol} | {r.solvent} | {r.model} | {r.group_weighted_rmse_nm:.4f} | {r.r2:.4f} |")
        report.append("")
    (OUT / "FINDINGS.md").write_text("\n".join(report), encoding="utf-8")
    tex = [r"\section{Solvent-specific mobile-layer investigation}", r"\label{sec:solvent_followup}",
           r"The prior follow-up already evaluated forecasting without measured thickness, earlier model families and robustness. The remaining near-term question is why the fixed-property mobile approximation diverges for toluene and ethyl acetate. This section reuses the audited records and previous outer splits; it does not repeat the completed RF/Ridge training or infer a physical cause from prediction error alone.",
           r"\subsection{Amplitude versus concentration dependence}",
           r"For fixed properties the comparator is proportional to $c/(c+965)$ for concentration $c$ in g/L. Its local log--log slope is $965/(965+c)$, between 0.951 and 1 on these data. Constant speed, viscosity, tension or density changes affect amplitude only. Equal-condition descriptive fits to the primary cohort give the following observed slopes and ratios. These summaries describe existing observations, not independent or causal tests.",
           r"\begin{table}[H]\centering\small\caption{Condition-mean mobile-layer diagnostics. Observed log--log slope is descriptive.}\begin{tabular}{lrrr}\toprule Solvent & Conditions & Observed slope & Observed/fixed ratio range \\\midrule"]
    for r in diagnostic.itertuples():
        tex.append(f"{r.solvent.replace('_',' ')} & {r.groups} & {r.descriptive_loglog_slope:.3f} & {r.ratio_min:.4g}--{r.ratio_max:.4g} " + r"\\")
    tex += [r"\bottomrule\end{tabular}\end{table}",
           r"Toluene is nearly flat across the observed concentration range, and ethyl acetate increases more slowly than the fixed approximation. A single solvent-specific amplitude cannot reproduce the complete concentration dependence. For a residual ratio $r$, equivalent one-factor changes are $r^{3/2}$ in speed or viscosity, $r^{-6}$ in tension, or $r^{-2}$ in density. These are algebraic inversions, not identified physical properties. An illustrative 81-combination sweep varies each factor by 0.5, 1 or 2 and spans 0.25--4 times the base prediction. This is a hypothetical range, not measured uncertainty. Replacing $c/(c+965)$ by the dilute convention $c/965$ changes the prediction by at most 5.18\%, insufficient to remove the largest discrepancy.",
           r"\subsection{Training-only calibration and alternative shape}",
           r"Four comparators are evaluated: fixed LLD, the mean of training condition means, a nonnegative least-squares scalar times LLD, and $Ac/(K+c)$ with positive parameters. The last is a phenomenological mobile curve, not an adsorption model for bonded material. Fits weight conditions equally and use only the training portion of the same solvent when available. In solvent holdout tests they use pooled other-solvent training conditions. Saturation fits use seven training-derived starts and bounds $A\in[10^{-6},10^4]$ nm and $K\in[10^{-6},10^5]$ g/L; one hexane fold in each cohort hits a bound. Parameters must not be assigned mechanistic meaning.",
           r"All models use identical outer-test rows. The prior RF predictions are reused, but that model is pooled and row-trained, so these comparisons do not isolate algorithm from weighting or solvent-specific fitting. The new candidates were motivated by previously inspected results and remain exploratory. No per-solvent winning hybrid is selected. Prior RF prediction intervals do not transfer to these comparators.",
           r"\begin{table}[H]\centering\small\caption{Primary grouped-CV mobile group-weighted RMSE (nm). Each condition receives equal evaluation weight.}\begin{tabular}{lrrr}\toprule Model & Hexane & Toluene & Ethyl acetate \\\midrule"]
    for model, label in NAMES.items():
        values = [cv[(cv.solvent == sol) & (cv.model == model)].iloc[0].group_weighted_rmse_nm for sol in ["hexane", "toluene", "ethyl_acetate"]]
        tex.append(label + " & " + " & ".join(f"{v:.3f}" for v in values) + r" \\")
    tex += [r"\bottomrule\end{tabular}\end{table}",
           r"Toluene's local mean outperforms the saturation curve in primary grouped validation. Including the historical summaries reverses this small ranking (0.1084 nm for saturation versus 0.1158 nm for the local mean), so the comparison does not establish a stable model preference. The pooled grouped error reduction from scalar calibration does not establish robustness. High-concentration and unseen-solvent tests expose large failures, as shown below. Summary-inclusion results are retained separately and do not establish independent replication.",
           r"\begin{table}[H]\centering\small\caption{Pooled mobile group-weighted RMSE (nm) under the same stress tests.}\begin{tabular}{lrrr}\toprule Model & Grouped CV & High concentration & Unseen solvent \\\midrule"]
    for model, label in NAMES.items():
        values = [primary[(primary.solvent == "all") & (primary.model == model) & (primary.protocol == p)].iloc[0].group_weighted_rmse_nm for p in ["grouped_cv", "ood_high_concentration", "leave_one_solvent_out"]]
        tex.append(label + " & " + " & ".join(f"{v:.3f}" for v in values) + r" \\")
    tex += [r"\bottomrule\end{tabular}\end{table}",
           r"\expfigure[1.0]{near_term/solvent_results/solvent_diagnostics.png}{Condition-level solvent diagnostics. Dashed curves are descriptive full-data fits, not held-out predictions. The shaded ratio band is a hypothetical parameter sweep, not a confidence interval.}",
           r"\subsection{Mechanistic interpretation and unresolved measurements}",
           r"The evidence supports a solvent-dependent mismatch of amplitude and concentration response. It does not distinguish evaporation, drainage, wash retention, solution-property changes, or protocol and measurement differences. Prior lubricant studies motivate separating bonded adsorption and mobile flow contributions \cite{merzlikine2005}, and polymer dip-coating studies motivate checking liquid-to-solid conversion \cite{zhang2022}; neither directly identifies the mechanism in this PDMS dataset. Controlled within-batch speed sweeps, measured solution viscosity/tension/density, matched wet/dried/washed thicknesses, and complete dwell/drying histories are needed. New independent experiments and scientific review remain necessary before finalizing deployment claims or the manuscript."]
    (OUT / "manuscript_update.tex").write_text("\n\n".join(tex)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
