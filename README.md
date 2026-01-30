# Pitt ML MSE Project

This repository is organized as a lightweight, experimental machine-learning workspace. It provides a consistent structure for experimenting with multiple model families while sharing a common placeholder dataset schema for thin-film processing.

## Repository layout

- `data/`: placeholder datasets and data documentation.
- `models/`: model-specific experiment folders with starter pipelines.
  - `bpnn/`: Backpropagation Neural Networks (BPNNs).
  - `knn/`: Known Nearest Neighbors (KNN).
  - `rfr/`: Random Forest Regression (RFR).
  - `linear_regression/`: Linear Regression.

## Placeholder feature + target schema

**Feature inputs**
- Withdrawal Speed
- Dwell Time (s)
- Substrate Type
- PDMS Concentration
- Solvent Type
- Etc.

**Target outputs**
- Total Film Thickness (μm or nm)
- Bonded Film Thickness (μm or nm)

Each model folder includes a `pipeline.py` with the same schema and a small helper for loading the placeholder CSV data.
