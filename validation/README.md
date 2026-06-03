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

## Baseline epsilon vs GMM-4D

After generating `/media/nuevo/epsilon_labels` with
`structural_labeling/run_epsilon_baseline.py`, run the same kinematic
validation on the hard-epsilon baseline:

```bash
cd /home/aespinola/Documents/pythonprojects/datacubes

python -m validation.run_kinematic_validation \
  --matched-units /home/aespinola/Documents/pythonprojects/datacubes/matched_assets/matched_units.csv \
  --labels-dir /media/nuevo/epsilon_labels \
  --outdir /media/nuevo/structural_validations/kinematic_epsilon_a10_b10 \
  --label-mode soft_mass \
  --dominant-class-threshold 0.50 \
  --min-spaxels-for-test 10 \
  --min-spaxels-test-b 10 \
  --continue-on-error
```

For the projection-IoU side of the same baseline:

```bash
python -m orientation_projection_validation.run_projection_validation \
  --manifest orientation_projection_validation/data/projection_manifest_matched.csv \
  --cache /media/nuevo/tng_cutouts \
  --morphology-catalog /media/nuevo/tng_cutouts/morphology/morphs_kinematic_bars.hdf5 \
  --config orientation_projection_validation/default_config.json \
  --outdir /media/nuevo/orientation_projection_validation/outputs_matched_epsilon \
  --label-model epsilon \
  --epsilon-threshold 0.70 \
  --continue-on-error

python orientation_projection_validation/summarize_metrics.py \
  --metrics-glob "/media/nuevo/orientation_projection_validation/outputs_matched_epsilon/*/metrics.json" \
  --out /media/nuevo/orientation_projection_validation/catalog_interorientation_summary_epsilon.csv \
  --report /media/nuevo/orientation_projection_validation/catalog_interorientation_summary_epsilon.md
```

Then compare the GMM-4D and epsilon reports in one publication-style table:

```bash
python -m validation.compare_baseline_reports \
  --gmm-kinematic /media/nuevo/structural_validations/kinematic_central_a10_b10/kinematic_validation_report.json \
  --epsilon-kinematic /media/nuevo/structural_validations/kinematic_epsilon_a10_b10/kinematic_validation_report.json \
  --gmm-orientation /media/nuevo/orientation_projection_validation/catalog_interorientation_summary_matched.csv \
  --epsilon-orientation /media/nuevo/orientation_projection_validation/catalog_interorientation_summary_epsilon.csv \
  --out /media/nuevo/structural_validations/gmm_vs_epsilon_baseline.md
```

If the epsilon labels are still running, the whole follow-up can be launched as
a single waiting pipeline:

```bash
cd /home/aespinola/Documents/pythonprojects/datacubes

python -m validation.run_epsilon_followup_pipeline
```

This waits until every row in
`/home/aespinola/Documents/pythonprojects/datacubes/matched_assets/matched_units.csv`
has an epsilon `*.labels.npz` and `*.summary.json`, then runs kinematic
validation, projection-IoU validation, IoU summarization, and the final
GMM-4D-vs-epsilon comparison. Progress is written to:

- `/media/nuevo/structural_validations/epsilon_followup_pipeline_state.json`
- `/media/nuevo/structural_validations/epsilon_followup_pipeline.log`

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

## Representative segmentation figures

To produce publication-ready colored examples of segmented galaxies:

```bash
cd /home/aespinola/Documents/pythonprojects/datacubes

python -m validation.make_segmentation_examples \
  --matched-units /home/aespinola/Documents/pythonprojects/datacubes/matched_assets/matched_units.csv \
  --labels-dir /media/nuevo/structural_labels \
  --kinematic-units /media/nuevo/structural_validations/kinematic_central_a10_b10/kinematic_validation_units.csv \
  --outdir /media/nuevo/structural_validations/segmentation_examples \
  --n-examples 4 \
  --label-mode soft_mass \
  --dominant-threshold 0.50 \
  --min-component-fraction 0.80
```

This writes:

- `segmentation_examples_montage.png`
- one `*.segmentation.png` per selected galaxy
- `selected_segmentation_examples.csv`
- `segmentation_examples_report.md`

Each figure contains the unsegmented light image next to a single colored
segmentation map with a class legend. For publication figures the script keeps
only the central connected component of the valid mask and, by default,
requires it to contain at least 80% of the valid pixels. This avoids examples
where disconnected edge islands appear after a blank gap. If this is too
strict for a small run, lower `--min-component-fraction`.
