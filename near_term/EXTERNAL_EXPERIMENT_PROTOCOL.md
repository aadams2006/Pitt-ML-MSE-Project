# Independent experiments and test-set protocol

Status: pending actual new measurements and provenance confirmation. No external
test result is claimed in this release. Additional sheets in the existing workbooks
are historical source material, not automatically new or independent experiments.

1. Before measuring outcomes, register sample IDs, solution-preparation batch IDs,
   collection dates, operator, wafer/substrate, PDMS grade, solvent, concentration
   and units, dwell time, withdrawal speed, temperature, humidity, wash protocol,
   and measurement method. Record measured solution viscosity, surface tension,
   density, and evaporation conditions where available. Distinguish repeat readings
   on one wafer from separately prepared experimental replicates.
2. For a first campaign, plan (not actual data) three independently prepared batches
   per solvent, at five concentration conditions spanning each observed range and
   selected low/high extrapolation conditions, with separate wafers per condition.
   This is a proposed 45-condition campaign, not a power calculation. The PI must
   choose practical concentration limits and replication after examining measurement
   precision and the substantial solvent-transfer errors reported here.
3. Collect matched total and bonded measurements with uncertainties and sample IDs.
   Do not infer paired samples from row order. Resolve hT < hB with the experimental
   owner; retain original values and quarantine unresolved records. Never convert
   missing measurements into zero or silently clip invalid differences.
4. Confirm in writing that held-out batches/wafer measurements never contributed to
   generator fitting, feature choices, hyperparameter selection or previous model
   development. Record that attestation and source hashes in the repository.
5. Freeze this code, estimator parameters, training sources, features, and scoring
   plan before accessing test outcomes. Train on the existing valid primary cohort.
   Keep all new batches completely separate. Record one-shot predictions and metrics
   for each solvent and batch, including calibration coverage and interval width.
6. Run the saved fixed baselines and transforms on identical test rows. Report failed
   transfer as well as successful interpolation. If results prompt redesign, retire
   that test set into development data and obtain a fresh sealed test set.

`external_experiments_template.csv` is an empty intake schema, not generated evidence.
Fill concentration in g/L and thickness in nm. Record protocol changes explicitly.
The current model accepts only the three recorded solvents; additional solvent
descriptors need reviewed provenance and a prospectively specified evaluation.

For additional historical workbook records, first curate sample-to-reading pairing,
batch identity and protocol in a source-cell map. Check against canonical source rows
and summary means for duplication. Missing batch identity cannot be reconstructed
reliably from a sheet date; several sheets explicitly reuse earlier solutions.
