"""Export current data and blank, sample-linked physics intake fields (no training)."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIELDS = {
    'Withdrawal Velocity (mm/s)': 'Measured withdrawal speed; velocity smoothness and LLD. Must vary for velocity smoothness.',
    'Viscosity (cP)': 'Validated solution viscosity at the relevant concentration and temperature; LLD.',
    'Surface Tension (mN/m)': 'Validated solution surface tension at the relevant conditions; LLD.',
    'Solution Density (kg/m3)': 'Validated solution density; LLD.',
    'Wet-to-Dry Mobile Factor': 'Dimensionless independently justified wet-to-dry conversion in (0,1]; LLD. Never derive from target thickness.',
    'LLD Applicable': 'Explicit lab-validated 0/1 regime flag; blank means unknown, not zero.',
    'Experiment ID': 'Actual experimental unit identifier, if recoverable; do not invent synthetic parent IDs.',
    'measurement_source': 'Source/citation or measurement identifier for supplied physics values.',
    'unit_confirmation': 'Record checked units and applicable measurement conditions.',
    'regime_evidence': 'Evidence for geometry, rheology, wetting, evaporation and low-capillary-number applicability.',
}


def read(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = []
    for kind, source_name in [
        ('synthetic', 'synthetic_3000.csv'),
        ('experimental', '20261111T134229Z_measured_followup_measured.csv'),
    ]:
        source = HERE / source_name
        original_columns, rows = read(source)
        mapping = {c: c.strip() for c in original_columns}
        if kind == 'synthetic':
            for column in original_columns:
                if column.strip() in ('Viscosity (cP)', 'Surface Tension (mN/m)'):
                    mapping[column] = 'Legacy Descriptor ' + column.strip()
        columns = list(mapping.values())
        assert not set(columns) & set(FIELDS)
        extra = ['template_source_file', 'template_source_csv_line', 'dataset_kind']
        output_rows = []
        for line, row in enumerate(rows, start=2):
            output = {mapping[c]: row[c] for c in original_columns}
            output.update(template_source_file=source.relative_to(ROOT).as_posix(),
                          template_source_csv_line=str(line), dataset_kind=kind)
            output.update({c: '' for c in FIELDS})
            output_rows.append(output)
        target = HERE / f'pinn_{kind}_physics_template.csv'
        with target.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns + extra + list(FIELDS), lineterminator='\n')
            writer.writeheader()
            writer.writerows(output_rows)
        _, verified = read(target)
        assert len(verified) == len(rows)
        for before, after in zip(rows, verified):
            assert all(before[c] == after[mapping[c]] for c in original_columns)
            assert all(after[c] == '' for c in FIELDS)
        records.append(dict(kind=kind, source=source.relative_to(ROOT).as_posix(),
                            source_sha256=digest(source), output=target.name,
                            output_sha256=digest(target), rows=len(rows),
                            source_to_output_columns=mapping,
                            validation='All source cell strings preserved, row order/count preserved, all intake cells blank.'))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), datasets=records,
                    blank_fields=FIELDS, existing_sources_modified=False, training_performed=False)
    (HERE / 'physics_templates_provenance.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({r['output']: r['rows'] for r in records}, indent=2))


if __name__ == '__main__':
    main()
