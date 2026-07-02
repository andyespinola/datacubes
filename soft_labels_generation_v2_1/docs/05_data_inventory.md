# Inventario de datos locales v2.1

Este inventario fue validado contra la carpeta `../data/` del workspace.
Reemplaza el piloto anterior `TNG50-87-141934` como muestra local por dos
galaxias con todos los insumos necesarios para el end-to-end inicial.

## Galaxias disponibles

| galaxy_id | cutout | cubo IFU | mapas pyPipe3D | catálogo morfológico |
|---|---:|---:|---:|---:|
| `TNG50-87-155298` | sí | sí | sí | sí |
| `TNG50-87-192324` | sí | sí | sí | sí |

Archivos esperados por galaxia:

```text
data/{galaxy_id}.cutout.hdf5
data/{galaxy_id}.subhalo.json
data/{galaxy_id}-0-127.cube.fits.gz
data/{galaxy_id}-0-127.cube_val.fits.gz
data/{galaxy_id}-0-127.cube_maps.fits
```

Catálogos compartidos:

```text
data/morphs_kinematic_bars.hdf5
data/stellar_circs.hdf5
```

Caches auxiliares:

```text
data/tng_snapshot_cache/offsets/offsets.87.hdf5
data/tng_snapshot_cache/snapshot_087/snapshot-87.{171,172,173}.hdf5
data/tng_snapshot_cache/snapshot_087/snapshot-87.{216,217,218}.hdf5
data/potential_cache/TNG50-87-155298.stellar_potential.hdf5
data/potential_cache/TNG50-87-192324.stellar_potential.hdf5
```

Estos caches contienen `PartType4/Potential` extraído desde snapshots
completos TNG y reordenado al orden local de `cutout_phase2` usando
`ParticleIDs`.

## Resumen de contenido

| galaxy_id | estrellas en cutout | gas en cutout | shape cubo | shape maps |
|---|---:|---:|---|---|
| `TNG50-87-155298` | 3,710,124 | 551,798 | `(6603, 69, 69)` | `(69, 69)` |
| `TNG50-87-192324` | 3,923,066 | 1,443,785 | `(6603, 69, 69)` | `(69, 69)` |

Los cutouts base contienen `PartType4` con `Coordinates`, `Velocities`,
`Masses`, `GFM_StellarFormationTime` y `GFM_Metallicity`. También contienen
`PartType0` con gas completo.

Además se descargaron assets de Fase 2 desde la API de TNG:

```text
data/{galaxy_id}.cutout_phase2.hdf5
data/stellar_circs.hdf5
```

Los cutouts `cutout_phase2` contienen `PartType4` con `Coordinates`,
`Velocities`, `Masses`, `GFM_StellarFormationTime`, `GFM_Metallicity`,
`GFM_InitialMass` y `ParticleIDs`; también contienen `PartType1` con
`Coordinates` y `ParticleIDs`. No contienen `PartType0`; para gas se conserva
el cutout base.

| galaxy_id | estrellas phase2 | DM phase2 | archivo phase2 |
|---|---:|---:|---:|
| `TNG50-87-155298` | 3,710,124 | 3,382,162 | 316 MiB |
| `TNG50-87-192324` | 3,923,066 | 5,827,665 | 403 MiB |

La API de subhalo cutouts rechazó `PartType4/Potential`, `PartType0/Potential`
y `PartType1/Potential` con `HTTP 400 Invalid input`. La documentación
oficial lista `Potential` para snapshots completos. La ruta implementada usa
`stellar_circs.hdf5` como validación agregada y puede extraer `Potential`
desde chunks del snapshot completo con offsets; si esa descarga sigue
fallando, queda abierto calcular un potencial local aproximado con
estrellas+gas+DM.

## Morfología del catálogo

| galaxy_id | disk | bulge | halo | barred |
|---|---:|---:|---:|---|
| `TNG50-87-155298` | 0.246 | 0.191 | 0.563 | `False` |
| `TNG50-87-192324` | 0.324 | 0.676 | 0.000 | `True` |

Estas fracciones se usan como prior suave en el clasificador baseline. No
se reescalan las probabilidades finales a esas fracciones.

## Circularidades oficiales

`data/stellar_circs.hdf5` cubre `Snapshot_87` y contiene las entradas de ambas
galaxias:

| galaxy_id | CircAbove07Frac | CircAbove07MinusBelowNeg07Frac | CircTwiceBelow0Frac | SpecificAngMom |
|---|---:|---:|---:|---:|
| `TNG50-87-155298` | 0.1406 | 0.1073 | 0.7359 | 958.351 |
| `TNG50-87-192324` | 0.3224 | 0.3141 | 0.3382 | 2633.802 |

Este catálogo entrega propiedades por subhalo, no etiquetas por partícula.
Sirve para calibrar/validar fracciones globales del clasificador.

## Manifest de escalado

`scripts/build_unique_mangia_manifest.py` genera un manifest operativo desde
`MaNGIA_catalog.fits`:

```text
soft_labels_generation_v2_1/outputs_catalog/mangia_unique_galaxies.csv
soft_labels_generation_v2_1/outputs_catalog/mangia_unique_galaxies_summary.json
```

El resumen actual contiene `7,522` galaxias únicas a partir de `10,051`
filas de catálogo, distribuidas entre snapshots `87` a `98`.

## Nota de IFU

`MaNGIA_catalog.fits` reporta `manga_ifu_dsn=61` para ambas galaxias, pero
los archivos locales están nombrados como `...-0-127`. La implementación
usa rutas explícitas y registra ambos valores (`file_ifu_design=127`,
`catalog_ifu_design=61`) en los metadatos del producto final.
