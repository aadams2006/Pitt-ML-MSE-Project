"""Generate the manuscript supplement and plain-text summary from saved metrics."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "near_term/results"
LABELS = {"grouped_cv": "Grouped CV", "leave_one_solvent_out": "Unseen solvent",
          "ood_low_concentration": "Low concentration", "ood_high_concentration": "High concentration"}


def main():
    summary = pd.read_csv(OUT / "summary_metrics.csv")
    folds = pd.read_csv(OUT / "fold_metrics.csv")
    audit = json.loads((OUT / "data_audit.json").read_text())
    primary = summary[summary.cohort == "primary"]
    rf = primary[primary.model == "rf_raw"]
    text = ["PITT ML-MSE NEAR-TERM WORK SUMMARY", "", "Scope: September 9, 2026 near-term-work slide.",
            "Computational work is complete on available repository data. New laboratory",
            "experiments and a true external test set remain unavailable and are not claimed.", "",
            "DATA", f"Audited {audit['input_rows']} original-table rows; primary analysis uses {audit['primary_rows']} rows in 36 solvent/concentration groups.",
            "Corrected concentration schema to g/L; explicit g/mL conversion divides by 1000.",
            "Quarantined one hT < hB record; excluded 15 potentially overlapping toluene summaries.",
            "Saved source-row provenance, hashes, workbook inventory and exclusion reasons.",
            f"The broader physical audit flags {audit['legacy_synthetic_invalid']} of 3000 legacy synthetic rows.",
            "A training-only constrained bootstrap audit generated 300 rows with zero invalid targets.", "",
            "VALIDATION AND MODELING", "Evaluated mean, Ridge and fixed random-forest models on identical experimental-only splits.",
            "No measured thickness is a feature. Tested raw features and log-concentration interactions.",
            "Ran grouped five-fold CV, three leave-one-solvent-out tests, low/high concentration OOD tests,",
            "group split-conformal intervals and a sensitivity analysis including the historical summaries.",
            "Saved all predictions, partitions, fold scores, aggregate scores, interval coverage and plots.", "",
            "Fixed raw-feature random forest results (point scores use all outer training rows):"]
    for row in rf.itertuples():
        text.append(f"{LABELS[row.protocol]}, {row.target}: R2={row.r2:.4f}; row RMSE={row.rmse_nm:.4f} nm; "
                    f"group-weighted RMSE={row.group_weighted_rmse_nm:.4f} nm; interval row coverage={row.row_coverage_80:.1%}.")
    text += ["", "INTERPRETATION", "Unseen-solvent forecasting is not established. Both pooled RF R2 values are negative.",
             "Concentration transforms do not improve RF grouped-CV errors; modest low-concentration",
             "improvements do not generalize across protocols. No universal accuracy gain is claimed.",
             "High-concentration mobile coverage collapses to 14.8% at the nominal 80% level.",
             "Intervals belong to a separate proper-training estimator, not the full-training point model.",
             "Condition groups are proxy groups, not verified independent batches. Bootstrap ranges are descriptive.",
             "Historical synthetic scores are development diagnostics, not independent deployment validation.", "",
             "PAPER AND REPRODUCIBILITY", "Updated manuscript abstract, interpretation and conclusion; added generated methods/results supplement",
             "and two figures. All historical outputs and raw observations are preserved.",
             "Added pinned analysis requirements, invariant/leakage tests, reproducible runners,",
             "an empty external-experiment intake template, and a prospective measurement/test protocol.", "",
             "REMAINING LABORATORY WORK", "Collect and attest independent new batches, recover missing sample/batch pairing, resolve",
             "the invalid measurement, and evaluate a frozen model once on the sealed external set.",
             "The manuscript remains a research draft pending that evidence and scientific review."]
    (ROOT / "NEAR_TERM_SUMMARY.txt").write_text("\n".join(text) + "\n", encoding="utf-8")

    latex = [r"\section{Experimental-only robustness follow-up}", r"\label{sec:robustness}",
        r"This analysis supersedes deployment interpretations of the preceding historical synthetic benchmarks; it does not replace their archived numerical results. All new scores use existing experimental records only. No independent new laboratory campaign has been supplied.",
        r"\subsection{Audited data and leakage controls}",
        r"The source tables contain 142 records. One toluene record has $h_T=0.85$ nm and $h_B=0.86$ nm and is quarantined without clipping. The first 15 toluene rows reproduce workbook summary values and may overlap individual readings; excluding them gives 126 primary records across 36 solvent--concentration conditions. A sensitivity analysis retains all 141 physically valid records. The original concentration numbers are in g/L, as documented by the workbook unit labels; the canonical schema names them accordingly and provides $c_{\mathrm{g/mL}}=c_{\mathrm{g/L}}/1000$. Raw files are preserved, and source rows, unit-evidence cells and file hashes are archived.",
        r"The broader physical audit finds 192 of 3000 historical synthetic rows violating finite, nonnegative concentration/thickness or $h_T\ge h_B$ requirements. This is a broader criterion than the previously reported 113 negative mobile differences. A replacement training-only bootstrap perturbs nonnegative bonded and mobile components and reconstructs total as their sum. Its 300-row audit contains no invalid records and retains parent IDs. These synthetic rows are excluded from every validation experiment.",
        r"\subsection{Methods}",
        r"Bonded and mobile thickness are separate responses. Features are concentration and the four historical solvent descriptors. No measured thickness, including uncoated thickness, is used. Fixed models are the training mean, standardized Ridge regression ($\alpha=10$), and a 200-tree random forest with minimum leaf size two. Raw features are compared with $\log(1+c/(1\,\mathrm{g/L}))$ and its interactions with solvent descriptors. Models are untuned; nonnegative projection is applied to predictions. Scaling is fitted only within training partitions.",
        r"Five-fold grouped cross-validation keeps each solvent--concentration condition intact. Leave-one-solvent-out tests train on two solvents and test on the third. Explicit low/high concentration tests reserve the lowest/highest $\lceil0.2K_s\rceil$ distinct concentrations in each solvent, where $K_s$ is its number of represented conditions. These thresholds use features only. Batch and wafer IDs are unavailable, so condition groups are proxies and cannot rule out dependence across conditions prepared in the same batch. Grouped CV can include edge conditions and is not exclusively an interpolation experiment.",
        r"Point estimates use all outer-training rows. Separate interval estimators reserve 35\% of outer-training groups for calibration and fit on the remainder. Each calibration score is the group's maximum absolute residual. The nominal 80\% split-conformal radius is order statistic $\lceil(G+1)0.8\rceil$ of the $G$ group scores; an unavailable finite statistic produces an infinite radius. Interval centers therefore differ from full-training point predictions. Coverage relies on exchangeability and is not guaranteed under the deliberate distribution shifts evaluated here.",
        r"Group-weighted RMSE first averages squared residuals within each condition, then across conditions. Row-weighted RMSE, MAE and $R^2$ are retained for comparison. Descriptive percentile ranges resample condition errors 2000 times and condition on the fitted folds; they do not measure retraining or solvent-population uncertainty. All fold assignments and individual predictions are committed.",
        r"\subsection{Results}",
        r"\begin{table}[H]\centering\small",
        r"\caption{Fixed raw-feature random forest on experimental records. RMSE is row-weighted; G-RMSE gives equal weight to each condition. Coverage is observed row coverage for separately calibrated nominal 80\% intervals.}",
        r"\begin{tabular}{llrrrr}\toprule",
        r"Protocol & Target & $R^2$ & RMSE & G-RMSE & Coverage \\\midrule"]
    for protocol in LABELS:
        for target in ["bonded_nm", "mobile_nm"]:
            row = rf[(rf.protocol == protocol) & (rf.target == target)].iloc[0]
            latex.append(f"{LABELS[protocol]} & {target.split('_')[0].title()} & {row.r2:.3f} & {row.rmse_nm:.3f} & {row.group_weighted_rmse_nm:.3f} & {100*row.row_coverage_80:.1f}\\% " + r"\\")
    latex += [r"\bottomrule\end{tabular}\end{table}",
        r"The RF grouped-CV scores are materially below the historical synthetic benchmark. Both pooled leave-one-solvent-out $R^2$ values are negative. At high concentrations, mobile-layer row coverage falls to 14.8\% and simultaneous condition coverage to 11.1\%, despite the nominal 80\% level. These results do not support reliable extrapolation or an exchangeability-based coverage guarantee in deployment.",
        r"\begin{table}[H]\centering\small",
        r"\caption{Leave-one-solvent-out raw RF point scores. Small within-solvent target variance can make $R^2$ extreme, so absolute errors must also be considered.}",
        r"\begin{tabular}{llrr}\toprule Solvent & Target & $R^2$ & RMSE (nm) \\\midrule"]
    f = folds[(folds.cohort == "primary") & (folds.model == "rf_raw") & (folds.protocol == "leave_one_solvent_out")]
    for row in f.itertuples():
        latex.append(f"{row.fold.replace('_', ' ')} & {row.target.split('_')[0]} & {row.r2:.3f} & {row.rmse_nm:.3f} " + r"\\")
    latex += [r"\bottomrule\end{tabular}\end{table}",
        r"\expfigure[1.0]{near_term/results/validation_rmse.png}{Experimental-only comparison across deployment scenarios. All fixed candidates, including the training-mean baseline, are shown.}",
        r"\expfigure[0.95]{near_term/results/grouped_parity.png}{Grouped out-of-fold raw RF predictions. No measured thickness is a feature.}",
        r"The log/interactions RF has grouped-CV group-weighted RMSE of 0.136 nm for bonded and 1.401 nm for mobile thickness, versus 0.106 and 0.945 nm for raw RF. Transformations modestly reduce low-concentration errors but do not yield a consistent improvement across scenarios. They remain exploratory choices, not validated universal gains.",
        r"Summary-inclusion sensitivity results and all model-level scores are available in the committed CSVs. This sensitivity changes both condition coverage and fold composition, so its differences cannot be attributed solely to duplicated summary weighting. It is not independent validation.",
        r"\subsection{Remaining limitations and external test requirement}",
        r"The three solvents, incomplete batch lineage, summary/reading ambiguity, and fixed approximate solvent descriptors limit inference. Solution properties, withdrawal speed, dwell time and protocol variation are not resolved by concentration-only process information. The workbook inventory is historical material and cannot establish an untouched external test set. A prospective protocol and empty intake schema are provided; they require newly collected independent batches, matched measurements and uncertainty, source-cell provenance, and confirmation that neither outcomes nor samples informed prior model or generator development. The final model and scoring plan must be frozen before opening that test set. Consequently the manuscript remains a research draft, and external-validation completion is not claimed."]
    (OUT / "manuscript_update.tex").write_text("\n\n".join(latex) + "\n", encoding="utf-8")
    # Human-readable comprehensive result tables, including the sensitivity cohort.
    report = ["# Near-term findings", "", "See the root NEAR_TERM_SUMMARY.txt and near_term/README.md for interpretation and methods.", ""]
    for cohort in summary.cohort.unique():
        report += [f"## {cohort}", "", "| Protocol | Model | Target | Row R2 | Group RMSE (nm) | Group coverage |", "|---|---|---|---:|---:|---:|"]
        for row in summary[summary.cohort == cohort].itertuples():
            report.append(f"| {row.protocol} | {row.model} | {row.target} | {row.r2:.4f} | {row.group_weighted_rmse_nm:.4f} | {row.group_simultaneous_coverage_80:.1%} |")
        report.append("")
    (OUT / "FINDINGS.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
