# Notas de implementación v2.1

## Qué se implementó

Se creó un paquete ejecutable en `soft_labels_generation_v2_1/` con:

- `pyproject.toml` y entry point `aperturenet-labels`.
- Configuración YAML en `configs/default.yaml` y `configs/pilot.yaml`.
- Paquete Python en `src/aperturenet_labels/`.
- Tests en `tests/`.

El pipeline ya corre de punta a punta sobre las dos galaxias locales:

```text
TNG50-87-155298
TNG50-87-192324
```

Productos por galaxia:

```text
particle_features.h5
particle_labels_initial.h5
particle_labels_with_bar.h5
particle_labels_final.h5
labels2d_v0.npz
M_valid_v0.npz
qa_report_v0.json
dataset_entry_{galaxy_id}_v0.h5
```

## Qué se portó del v1

Se portaron o reimplementaron de forma equivalente las piezas de
fontanería que `MIGRATION.md` marcaba como reutilizables:

- conversión de unidades TNG (`ckpc/h`, masas, velocidades);
- geometría de vistas MaNGIA (`view_vector_from_index`, proyección al plano);
- lectura de cutouts HDF5 y metadata JSON;
- lectura de cubos FITS y mapas `SSP_pyPipe3D_REC`;
- pesos de luz por partícula usando la grilla SSP MaStar.

No se portó `decomposition.py` ni los parches científicos del pipeline v1.

## Estado científico actual

La Fase A actual produce probabilidades normalizadas y ya deja conectados los
insumos principales de Hito 2, pero todavía no es el GMM completo del spec 11:

- si existe `data/potential_cache/{galaxy_id}.stellar_potential.hdf5`,
  `epsilon` usa `j_z / j_c(E)` con `E = 0.5 v^2 + Phi`;
- si no existe cache de potencial, el extractor registra el fallback y usa
  `epsilon = j_z / |j|` con energía cinética negativa;
- el clasificador combina heurística cinemática/espacial con prior
  morfológico suave;
- BarDetector redistribuye probabilidad desde `disk` usando señal Fourier
  m=2, fase de barra y prior de catálogo cuando la galaxia está marcada como
  barrada;
- ArmDetector redistribuye probabilidad desde `disk` usando exceso residual
  sobre un perfil radial suavizado.

Esto permite probar contratos, I/O, proyección, máscara, QA y empaquetado con
una salida físicamente interpretable como baseline, pero no sustituye todavía
la validación fina contra un catálogo externo por partícula.

## Assets descargados para Hito 2

Se usó la API de TNG50-1 con la key local para descargar insumos adicionales
sin reemplazar los archivos base:

```text
data/TNG50-87-155298.cutout_phase2.hdf5
data/TNG50-87-192324.cutout_phase2.hdf5
data/stellar_circs.hdf5
```

Cada `cutout_phase2` contiene:

- `PartType4`: `Coordinates`, `Velocities`, `Masses`,
  `GFM_StellarFormationTime`, `GFM_Metallicity`, `GFM_InitialMass`,
  `ParticleIDs`.
- `PartType1`: `Coordinates`, `ParticleIDs`.

Se validó que los IDs son únicos, los conteos coinciden con metadata local y
los campos numéricos inspeccionados son finitos.

`PartType4/Potential`, `PartType0/Potential` y `PartType1/Potential` no se
pudieron descargar vía subhalo cutout: la API respondió `HTTP 400 Invalid
input`. La documentación oficial de campos TNG lista `Potential` para
snapshots completos, pero la ruta de cutout lo rechaza para estos tipos en
`Snapshot_87`. Por eso se implementó una ruta adicional:

- usar `stellar_circs.hdf5` como calibración/validación agregada de
  circularidad;
- extraer `Potential` desde los chunks del snapshot completo usando offsets y
  guardarlo como cache por galaxia;
- calcular un potencial local aproximado usando estrellas del cutout phase2,
  gas del cutout base y DM del cutout phase2 si el acceso a snapshots
  completos falla de forma persistente.

## Cómo correr

Desde `soft_labels_generation_v2_1/`, sin instalar:

```bash
PYTHONPATH=src /home/andy/pythonprojects/datacubes/.venv/bin/python \
  -m aperturenet_labels.cli validate-data

PYTHONPATH=src /home/andy/pythonprojects/datacubes/.venv/bin/python \
  -m aperturenet_labels.cli run --galaxy-id TNG50-87-155298

PYTHONPATH=src /home/andy/pythonprojects/datacubes/.venv/bin/python \
  -m aperturenet_labels.cli run --galaxy-id TNG50-87-192324

PYTHONPATH=src /home/andy/pythonprojects/datacubes/.venv/bin/python \
  -m aperturenet_labels.cli run --all-local
```

Para un smoke liviano:

```bash
PYTHONPATH=src /home/andy/pythonprojects/datacubes/.venv/bin/python \
  -m aperturenet_labels.cli run \
  --galaxy-id TNG50-87-192324 \
  --max-particles 5000 \
  --no-copy-cube \
  --outdir outputs_smoke \
  --overwrite
```

Si se instala en editable:

```bash
/home/andy/pythonprojects/datacubes/.venv/bin/python -m pip install -e .
aperturenet-labels validate-data
aperturenet-labels run --all-local
```

## Tests

```bash
/home/andy/pythonprojects/datacubes/.venv/bin/python -m pytest -q
```

La suite valida:

- assets completos para ambas galaxias;
- campos mínimos en `PartType4` y `PartType0`;
- cobertura morfológica en `Snapshot_87`;
- cobertura de `stellar_circs.hdf5` para ambas galaxias;
- lectura de cubos y mapas pyPipe3D;
- escritura de `dataset_entry.h5` con grupos requeridos;
- utilidades de alineación D4 y cache de potencial.

## Validación visual

Se generó una corrida de inspección en `outputs_visual/` usando 250k
partículas por galaxia y sin copiar el cubo IFU completo al HDF5:

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

Productos:

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
outputs_visual/figures/segmentation/segmentation_validation_summary.json
```

Lectura rápida:

- `TNG50-87-155298`: status QA `pass`; domina `halo` en el argmax duro,
  aunque la salida soft conserva fracciones bulge/disk.
- `TNG50-87-192324`: status QA `pass`; mezcla bulge/disk visible y se
  recupera una barra central suave (`bar_mass_fraction` alrededor de 2-3%
  en masa válida) usando señal m=2 más prior de catálogo.

Estas figuras validan geometría, máscara, proyección y patologías gruesas.
No validan todavía la separación física fina de componentes.

La validación específica de segmentación añade mapas de clase dominante,
mezcla soft RGB, confianza `max(P)`, margen `top1 - top2`, entropía
normalizada y regiones de baja confianza/ambigüedad. Lectura actual:

- `TNG50-87-155298`: `mean max P=0.626`, baja fracción ambigua (`0.036`),
  pero el argmax duro queda casi completamente en `halo`. Las probabilidades
  soft sí contienen bulge/disk, por lo que esta salida debe consumirse como
  tensor soft, no como segmentación dura.
- `TNG50-87-192324`: `mean max P=0.397`, baja confianza en `0.941` de los
  spaxels válidos y ambigüedad `0.980` con el umbral de margen `0.15`.
  Visualmente aparecen regiones bulge/disk y una barra central suave, pero
  la segmentación dura no es robusta todavía.

La inspección visual mostró que la proyección inicial tenía un roll de grilla
distinto al cubo/mapa MaNGIA. Para corregirlo, `phase_b.label_projection`
estima una alineación D4 exacta (rotaciones de 90 grados y flips, sin
interpolación) entre la masa estelar proyectada y
`stellar_mass_density_log10` de pyPipe3D. La misma transformación se aplica a
todos los mapas crudos antes de normalizar y aplicar PSF. En las dos galaxias
locales el estimador eligió `rot90_k=3`, sin flips.

La metadata de cada `labels2d_v0.npz` registra:

```text
sky_alignment_enabled
sky_alignment_rot90_k
sky_alignment_degrees_ccw
sky_alignment_flip_x
sky_alignment_flip_y
sky_alignment_score
sky_alignment_reference
```

## Pendiente para Hito 2

- Sustituir el estimador inicial de `j_c(E)` por la receta definitiva del
  spec 11 si se adopta una implementación más fiel al artículo.
- Sustituir el clasificador heurístico por GMM `paper4d`.
- Ajustar la salida final a las clases del artículo; la clase `halo` actual
  debe quedar como diagnóstico interno o mapearse antes de exportar.
- Añadir tests de comparación contra MORDOR/Rodriguez-Gomez cuando esté
  disponible el catálogo correspondiente.
- Decidir la estrategia de producción para las 7.522 galaxias únicas del
  catálogo MaNGIA/TNG local: snapshot chunks compartidos, cache centralizado
  de offsets y política de limpieza de chunks grandes.

## Cache de potencial estelar

Se añadió un extractor para recuperar `PartType4/Potential` desde archivos
de snapshot completo sin bajar cutouts nuevos:

```bash
/home/andy/pythonprojects/datacubes/.venv/bin/python \
  scripts/extract_stellar_potential_cache.py \
  --all-local \
  --preflight
```

La extracción real se ejecuta con:

```bash
/home/andy/pythonprojects/datacubes/.venv/bin/python \
  scripts/extract_stellar_potential_cache.py \
  --all-local \
  --overwrite
```

Para endpoints intermitentes se puede aumentar la paciencia del downloader:

```bash
/home/andy/pythonprojects/datacubes/.venv/bin/python \
  scripts/extract_stellar_potential_cache.py \
  --galaxy-id TNG50-87-192324 \
  --overwrite \
  --download-retries 10 \
  --download-backoff-seconds 20
```

El flujo esperado es:

1. leer `PartType4/ParticleIDs` de `data/{galaxy_id}.cutout_phase2.hdf5`;
2. descargar/usar `offsets.{snapshot}.hdf5`;
3. ubicar el rango global de estrellas del subhalo;
4. descargar solo los chunks de `snapshot-{snapshot}.{file}.hdf5`
   necesarios;
5. leer `PartType4/ParticleIDs` y `PartType4/Potential`;
6. reordenar `Potential` al orden local del cutout usando `ParticleIDs`;
7. escribir `data/potential_cache/{galaxy_id}.stellar_potential.hdf5`.

Estado final con la API key local:

```text
offsets.87.hdf5                  -> descargado completo
TNG50-87-155298 chunks           -> 171, 172, 173
TNG50-87-192324 chunks           -> 216, 217, 218
TNG50-87-155298 cache            -> 3,710,124 Potential
TNG50-87-192324 cache            -> 3,923,066 Potential
```

Los archivos generados son:

```text
data/potential_cache/TNG50-87-155298.stellar_potential.hdf5
data/potential_cache/TNG50-87-192324.stellar_potential.hdf5
```

El downloader conserva `.part` e intenta reanudar con `Range` cuando se
relanza el comando. Además verifica el tamaño esperado antes de promover el
`.part` a HDF5 final, para no dejar archivos incompletos como válidos. Durante
la descarga se observaron `403` y `504` intermitentes para chunks grandes,
pero con reintentos extendidos se completaron los caches.

La corrida posterior del pipeline registró:

```text
potential_status     -> loaded
energy_definition    -> kinetic_plus_tng_potential_scaled_by_a
epsilon_definition   -> j_z_over_jc_energy_quantile
```

## Manifest para escalar a MaNGIA

Se añadió:

```bash
/home/andy/pythonprojects/datacubes/.venv/bin/python \
  scripts/build_unique_mangia_manifest.py
```

El manifest consolidado elimina vistas repetidas del catálogo local:

```text
outputs_catalog/mangia_unique_galaxies.csv
outputs_catalog/mangia_unique_galaxies_summary.json
```

Resultado actual: `10,051` filas de catálogo se reducen a `7,522` galaxias
únicas distribuidas entre snapshots `87` a `98`.
