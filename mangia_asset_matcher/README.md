# MaNGIA Asset Matcher

Independent matcher for reconstructed MaNGIA cubes, local TNG cutouts, and
MaNGIA 2D maps.

It only inventories local files and writes manifests. It does not download,
reconstruct, or modify scientific products.

## Run

```bash
python -m mangia_asset_matcher.match_assets \
  --catalog MaNGIA_catalog.fits \
  --cube-root /media/nuevo/output_cubos \
  --tng-cache /media/nuevo/tng_cutouts \
  --maps2d-root /home/aespinola/Documents/pythonprojects/datacubes/maps2D \
  --limit 500 \
  --require-count 500 \
  --outdir matched_assets
```

Use `--limit 0` to keep all strict matches. Use `--dry-run` to print the
summary without writing output files.

## Outputs

- `matched_units.csv`: selected strict matches after `--limit`.
- `matched_units_all.csv`: all strict matches.
- `asset_inventory.csv`: catalog-level inventory with availability flags.
- `unmatched_report.json` and `unmatched_report.md`: exclusion counts and scan
  summary.

The matching unit is `(snapshot, subhalo_id, view)`.
