# Ethyl Acetate Analytical Comparison

This folder runs the same experimental comparison workflow as the hexane folder, but against the local [ethyl acetate+pdms.csv](/c:/Users/alexg/Downloads/Pitt-ML-MSE-Project/analytical%20model%20comparision/ethyl%20acetate/ethyl%20acetate+pdms.csv) dataset.

- The best Bayesian-optimized RF artifact is reused from the project-wide best model.
- Ethyl acetate solvent constants are injected for the ML feature frame.
- The evaporation-dependent analytical models remain symbolic in terms of `E`.

Run:

```bash
python "analytical model comparision/ethyl acetate/src/run_comparison.py"
```
