# Kinematic Validation

Reads structural pseudo-label products and MaNGIA 2D Pipe3D maps, then writes
catalog-level coherence reports.

```bash
python -m validation.run_kinematic_validation \
  --matched-units /home/aespinola/Documents/pythonprojects/datacubes/matched_assets/matched_units.csv \
  --labels-dir /media/nuevo/structural_labels \
  --outdir /media/nuevo/structural_validations/kinematic \
  --dominant-class-threshold 0.50 \
  --min-spaxels-for-test 10 \
  --min-spaxels-test-b 10 \
  --continue-on-error
```

By default Test A uses `--rotation-test contrast --test-a-reference central`:
median `V/sigma` in disk-dominated spaxels must exceed the median in central
valid spaxels by `--disk-vsigma-ratio-min` (default `1.10`). The older global
Spearman test remains available with `--rotation-test spearman`.

`--min-spaxels-for-test` controls Test A, C, and D. Test B has a separate
minimum, `--min-spaxels-test-b` (default `10`), because bulge-dominated
regions are often much smaller than disk or central regions.

For Test A contrast, `--test-a-reference` selects the comparison population:

- `central` is the primary validation mode for the 500-unit run.
- `bulge` compares only against bulge-dominated spaxels.
- `bulge_other` keeps the original strict diagnostic comparison against bulge or other.

Example diagnostic reruns:

```bash
python -m validation.run_kinematic_validation \
  --matched-units /home/aespinola/Documents/pythonprojects/datacubes/matched_assets/matched_units.csv \
  --labels-dir /media/nuevo/structural_labels \
  --outdir /media/nuevo/structural_validations/kinematic_bulge_ref \
  --label-mode soft_mass \
  --rotation-test contrast \
  --test-a-reference bulge \
  --dominant-class-threshold 0.50 \
  --min-spaxels-for-test 30 \
  --min-spaxels-test-b 10 \
  --continue-on-error

python -m validation.run_kinematic_validation \
  --matched-units /home/aespinola/Documents/pythonprojects/datacubes/matched_assets/matched_units.csv \
  --labels-dir /media/nuevo/structural_labels \
  --outdir /media/nuevo/structural_validations/kinematic_central_ref \
  --label-mode soft_mass \
  --rotation-test contrast \
  --test-a-reference central \
  --central-reference-radius-fraction 0.25 \
  --dominant-class-threshold 0.50 \
  --min-spaxels-for-test 30 \
  --min-spaxels-test-b 10 \
  --continue-on-error
```

To compare Test A thresholds in one run, use:

```bash
python -m validation.run_test_a_sensitivity \
  --matched-units /home/aespinola/Documents/pythonprojects/datacubes/matched_assets/matched_units.csv \
  --labels-dir /media/nuevo/structural_labels \
  --outdir /media/nuevo/structural_validations/kinematic_central_sensitivity \
  --label-mode soft_mass \
  --thresholds 1.00,1.10,1.20 \
  --test-a-reference central \
  --central-reference-radius-fraction 0.25 \
  --dominant-class-threshold 0.50 \
  --min-spaxels-for-test 30 \
  --min-spaxels-test-b 10 \
  --continue-on-error
```

This writes each full validation into `ratio_1p00/`, `ratio_1p10/`, and
`ratio_1p20/`, plus combined summaries:

- `test_a_sensitivity_summary.csv`
- `test_a_sensitivity_summary.md`

Optional h3/h4 products from `kinematic_moments` can be enabled with:

```bash
  --kinematics-dir /media/nuevo/kinematics_ppxf
```

Outputs:

- `kinematic_validation_units.csv`
- `kinematic_validation_report.json`
- `kinematic_validation_report.md`
- `coherence_score_histogram.png`
- `test_a_diagnostics.csv`
- `test_a_summary_by_view.csv`
- `test_a_summary_by_sample.csv`
- `test_a_summary_by_global_vsigma.csv`
- `test_a_extreme_pass_fail.md`
