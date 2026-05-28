# Kinematic Validation

Reads structural pseudo-label products and MaNGIA 2D Pipe3D maps, then writes
catalog-level coherence reports.

```bash
python -m validation.run_kinematic_validation \
  --matched-units /home/aespinola/matched_assets/matched_units.csv \
  --labels-dir /media/nuevo/structural_labels \
  --outdir /media/nuevo/structural_validations/kinematic \
  --dominant-class-threshold 0.50 \
  --min-spaxels-for-test 10 \
  --continue-on-error
```

By default Test A uses `--rotation-test contrast`: median `V/sigma` in
disk-dominated spaxels must exceed the median in bulge/other-dominated
spaxels by `--disk-vsigma-ratio-min` (default `1.10`). The older global
Spearman test remains available with `--rotation-test spearman`.

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
