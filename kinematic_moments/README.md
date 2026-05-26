# kinematic_moments

Extracts Gauss-Hermite stellar kinematic moments `h3` and `h4` from official
MaNGIA `*.cube.fits.gz` products using pPXF.

This stage intentionally reads the official MaNGIA cube only:

- `PRIMARY`: stellar flux cube
- `ERROR`: flux uncertainty cube
- `MASK > 0`: valid MaNGIA coverage
- `REDSHIFT` and `CRVAL3/CRPIX3/CDELT3`: spectral rest-frame conversion

It does not use `cube_val`, `74x74` harmonized cubes, LOGCUBE-like products, or
Pipe3D maps. Pipe3D cross-checks are a later validation stage.

Run the pilot:

```bash
python kinematic_moments/run_kinematics.py \
  --cube data/TNG50-87-141934-0-127.cube.fits.gz \
  --outdir kinematic_moments/output \
  --max-spaxels 10
```

For full runs, pass the MaStar template explicitly or set
`KINEMATICS_TEMPLATE_PATH`:

```bash
python kinematic_moments/run_kinematics.py \
  --cube-glob "data/*.cube.fits.gz" \
  --template-path /path/to/MaStar_CB19.slog_1_5.fits.gz \
  --outdir kinematic_moments/output \
  --n-workers 8
```

Portable batch options:

- `--n-workers N`: run up to `N` cubes concurrently using worker processes.
- `--limit N`: process only the first `N` galaxies after deterministic input ordering.
- `--progress-every N`: print a progress line every `N` completed galaxies. The default is 10.
- `--manifest manifest.csv`: read cube paths from a `cube_path` column instead of a glob.

Example for a first 100-galaxy batch on another machine:

```bash
export KINEMATICS_TEMPLATE_PATH=/path/to/MaStar_CB19.slog_1_5.fits.gz
python kinematic_moments/run_kinematics.py \
  --cube-glob "/data/mangia/*.cube.fits.gz" \
  --outdir /data/kinematics_ppxf \
  --n-workers 8 \
  --limit 100 \
  --progress-every 10
```
