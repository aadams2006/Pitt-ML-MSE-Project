"""Independently verify saved follow-up metrics without repeating completed fits."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "near_term/solvent_results"


def main():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    hashes = {}
    for source in manifest["inputs"]:
        actual = hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
        assert actual == source["sha256"], source["path"]
        hashes[source["path"]] = actual
    predictions = pd.read_csv(OUT / "mobile_predictions.csv")
    metrics = pd.read_csv(OUT / "mobile_metrics.csv")
    keys = ["cohort", "protocol", "fold", "model", "record_id"]
    assert not predictions.duplicated(keys).any()
    assert np.isfinite(predictions[["actual", "prediction"]]).all().all()
    assert (predictions.prediction >= 0).all()
    for row in metrics.itertuples():
        part = predictions[(predictions.cohort == row.cohort)
                           & (predictions.protocol == row.protocol)
                           & (predictions.model == row.model)]
        if row.solvent != "all":
            part = part[part.solvent == row.solvent]
        residual = part.actual - part.prediction
        group_mse = part.assign(squared=residual**2).groupby("group_id").squared.mean()
        assert len(part) == row.n and len(group_mse) == row.groups
        expected = [np.sqrt(np.mean(residual**2)), np.mean(abs(residual)),
                    np.sqrt(group_mse.mean()),
                    1 - np.sum(residual**2)/np.sum((part.actual-part.actual.mean())**2)]
        np.testing.assert_allclose(expected,
            [row.rmse_nm, row.mae_nm, row.group_weighted_rmse_nm, row.r2], rtol=1e-10, atol=1e-12)
    params = pd.read_csv(OUT / "outer_fit_parameters.csv")
    bound_hits = params[params.bound_hit.eq(True)]
    result = {"status": "passed", "python": platform.python_version(),
              "numpy": np.__version__, "pandas": pd.__version__,
              "source_hashes": hashes, "metric_rows_recomputed": len(metrics),
              "prediction_rows_checked": len(predictions),
              "bound_hits": bound_hits[["cohort", "protocol", "fold", "solvent", "model"]].to_dict("records"),
              "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
