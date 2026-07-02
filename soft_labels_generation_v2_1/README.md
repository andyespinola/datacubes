# aperturenet_labels — pipeline v2.1

Generación de etiquetas estructurales por spaxel para el entrenamiento de
**ApertureNet-S3** sobre cubos IFU MaNGIA.

Esta carpeta ya contiene un paquete ejecutable inicial. El objetivo de esta
implementación es validar el flujo end-to-end y los contratos con dos
galaxias locales completas en `../data/`.

## Quickstart local

```bash
cd /home/andy/pythonprojects/datacubes/soft_labels_generation_v2_1

# Verificar assets locales
PYTHONPATH=src /home/andy/pythonprojects/datacubes/.venv/bin/python \
  -m aperturenet_labels.cli validate-data

# Tests
/home/andy/pythonprojects/datacubes/.venv/bin/python -m pytest -q

# Smoke liviano
PYTHONPATH=src /home/andy/pythonprojects/datacubes/.venv/bin/python \
  -m aperturenet_labels.cli run \
  --galaxy-id TNG50-87-192324 \
  --max-particles 5000 \
  --no-copy-cube \
  --outdir outputs_smoke \
  --overwrite
```

Validación visual sobre ambas galaxias:

```bash
PYTHONPATH=src /home/andy/pythonprojects/datacubes/.venv/bin/python \
  -m aperturenet_labels.cli run \
  --all-local \
  --outdir outputs_visual \
  --max-particles 250000 \
  --no-copy-cube \
  --overwrite

/home/andy/pythonprojects/datacubes/.venv/bin/python \
  scripts/make_visual_validation.py --outputs-dir outputs_visual

/home/andy/pythonprojects/datacubes/.venv/bin/python \
  scripts/make_segmentation_validation.py --outputs-dir outputs_visual
```

Figuras generadas:

```text
outputs_visual/figures/TNG50-87-155298_visual_validation.png
outputs_visual/figures/TNG50-87-192324_visual_validation.png
outputs_visual/figures/fraction_summary.png
outputs_visual/figures/visual_qa_summary.csv
outputs_visual/figures/visual_validation_manifest.json
outputs_visual/figures/segmentation/TNG50-87-155298_segmentation_validation.png
outputs_visual/figures/segmentation/TNG50-87-192324_segmentation_validation.png
outputs_visual/figures/segmentation/segmentation_validation_report.md
outputs_visual/figures/segmentation/segmentation_validation_summary.csv
```

Manifest único de galaxias MaNGIA/TNG para planificar el escalado:

```bash
/home/andy/pythonprojects/datacubes/.venv/bin/python \
  scripts/build_unique_mangia_manifest.py
```

Instalación editable opcional:

```bash
/home/andy/pythonprojects/datacubes/.venv/bin/python -m pip install -e .
aperturenet-labels validate-data
aperturenet-labels run --all-local
```

## Documentación

- **Empezar por aquí**: [`CLAUDE.md`](./CLAUDE.md) — visión general del proyecto
- **Plan de implementación**: [`specs/00_ROADMAP.md`](./specs/00_ROADMAP.md)
- **Diseño detallado**: [`docs/`](./docs/)
- **Specs por módulo**: [`specs/`](./specs/)
- **Inventario local validado**: [`docs/05_data_inventory.md`](./docs/05_data_inventory.md)
- **Notas de implementación actual**: [`docs/06_implementation_notes.md`](./docs/06_implementation_notes.md)

## Comandos CLI principales

```bash
# Validar data/
aperturenet-labels validate-data

# Pipeline sobre una galaxia local
aperturenet-labels run --galaxy-id TNG50-87-155298
aperturenet-labels run --galaxy-id TNG50-87-192324

# Procesar las dos galaxias configuradas
aperturenet-labels run --all-local
```

## Para Claude Code

Este repo está diseñado para ser desarrollado iterativamente por Claude Code siguiendo el roadmap. La estrategia es:

1. **Hito 0**: setup del repo
2. **Hito 1**: skeleton end-to-end con módulos triviales
3. **Hito 2-4**: implementación real módulo por módulo
4. **Hito 5**: validación cruzada con pipeline v1
5. **Hito 6**: escalado a la muestra completa

Cada hito tiene criterios de aceptación claros en `specs/00_ROADMAP.md`.

## Galaxias locales disponibles

```
TNG50-87-155298
TNG50-87-192324
```

Los datos están en `../data/`. Cada galaxia tiene cutout TNG, metadata JSON,
cubo IFU, `cube_val` y `cube_maps.fits`.

También quedaron descargados los insumos de Hito 2:

```
../data/TNG50-87-155298.cutout_phase2.hdf5
../data/TNG50-87-192324.cutout_phase2.hdf5
../data/stellar_circs.hdf5
```

Los `cutout_phase2` agregan `ParticleIDs`, `GFM_InitialMass` y DM mínimo
(`PartType1/Coordinates`, `PartType1/ParticleIDs`). `Potential` no está
disponible por la ruta de subhalo cutout API para `PartType4` en estos
snapshots, aunque la documentación oficial lo lista en snapshots completos.
La extracción desde snapshot completo está implementada como cache externo.

## Estado científico actual

La implementación actual es un end-to-end funcional documentado. Produce
probabilidades normalizadas y empaqueta `dataset_entry.h5`. El extractor ya
usa `Potential` cacheado para `E = 0.5 v^2 + Phi` y estima
`epsilon = j_z / j_c(E)` por bins de energía en las dos galaxias locales.
Si se procesa una galaxia sin cache, cae de forma explícita al fallback
`epsilon = j_z / |j|` con energía cinética negativa.

El clasificador sigue siendo un baseline heurístico con prior morfológico,
no todavía el GMM `paper4d` completo del Hito 2. BarDetector y ArmDetector
redistribuyen probabilidad desde disco usando señal geométrica/morfológica;
la galaxia barrada local ya recupera una barra central suave en la validación
visual.

La validación visual actual es útil para chequear geometría, máscara,
proyección y patologías gruesas. No debe interpretarse aún como validación
física de bulbo/disco/barra/brazos.

La validación visual de segmentación confirma que el producto debe consumirse
como tensor soft en esta fase. En `TNG50-87-155298` el argmax duro cae todo
en `halo` aunque las probabilidades soft contienen bulge/disk; en
`TNG50-87-192324` aparece estructura bulge/disk y barra central suave, pero
con márgenes top1-top2 bajos y alta ambigüedad. Esto apunta al siguiente
trabajo científico: GMM `paper4d` y salida compatible con las clases del
artículo.

La proyección 2D corrige el roll de la grilla con una alineación D4
conservadora contra `pipe3d_maps/stellar_mass_density_log10`. En la muestra
local ambas galaxias aplican `rot90_k=3` sin flips.

Para preparar ε canónica se agregó:

```bash
/home/andy/pythonprojects/datacubes/.venv/bin/python \
  scripts/extract_stellar_potential_cache.py --all-local --preflight
```

El script extrae `PartType4/Potential` desde chunks de snapshot completo y
lo cachea por galaxia:

```bash
/home/andy/pythonprojects/datacubes/.venv/bin/python \
  scripts/extract_stellar_potential_cache.py --all-local --overwrite
```

Estado actual: `offsets.87.hdf5` y los chunks necesarios quedaron
descargados. Los caches finales están en:

```text
../data/potential_cache/TNG50-87-155298.stellar_potential.hdf5
../data/potential_cache/TNG50-87-192324.stellar_potential.hdf5
```

Chunks usados:

```text
TNG50-87-155298 -> snapshot-87.{171,172,173}.hdf5
TNG50-87-192324 -> snapshot-87.{216,217,218}.hdf5
```
