# Kinematic Validation

Reads structural pseudo-label products and MaNGIA 2D Pipe3D maps, then writes
catalog-level coherence reports.

```bash
python -m validation.run_kinematic_validation \
  --matched-units /home/aespinola/matched_assets/matched_units.csv \
  --labels-dir /media/nuevo/structural_labels \
  --outdir /media/nuevo/structural_validations/kinematic \
  --continue-on-error
```

Optional h3/h4 products from `kinematic_moments` can be enabled with:

```bash
  --kinematics-dir /media/nuevo/kinematics_ppxf
```

Outputs:

- `kinematic_validation_units.csv`
- `kinematic_validation_report.json`
- `kinematic_validation_report.md`
- `coherence_score_histogram.png`
