# MIGRATION.md — Qué portar del v1 y qué reimplementar desde cero

> Documento operativo para Claude Code. Leer después de `CLAUDE.md` y
> antes de `specs/00_ROADMAP.md`. El repo v1 es `structural_labeling/`
> (proporcionado como referencia de solo lectura).

## Regla general

El v1 tiene dos partes de calidad muy distinta:

- **La fontanería (IO, unidades, geometría, SSP, manifiestos) funciona
  y está probada en producción.** Se PORTA, adaptando solo imports y
  contratos pydantic.
- **El núcleo científico (asignación de clases) NO implementa la
  metodología del artículo** (no hay GMM; las familias se asignan por
  perfiles radiales paramétricos reescalados al catálogo — la causa
  raíz diagnosticada en `docs/01_diagnostico.md`). Se REIMPLEMENTA
  siguiendo los specs.

Nunca copiar lógica del v1 que toque probabilidades de clase.

## Tabla de decisión por módulo v1

| Archivo v1 | Decisión | Destino v2 | Notas |
|---|---|---|---|
| `labeling/tng.py` | **PORTAR** | `io/tng_reader.py` | Lectura de cutouts HDF5. |
| `labeling/pipeline.py::_convert_truth_units` + `SNAP_A`, `HUBBLE_PARAM` | **PORTAR** | `io/units.py` | Conversión comoving→físico. Único bloque rescatable de pipeline.py. |
| `labeling/ssp.py` | **PORTAR** | `phase_b/ssp.py` | `particle_light_weights`, rejilla (age, Z) → M/L. |
| `labeling/geometry.py` | **PORTAR ÍNTEGRO** | `core/geometry.py` | `view_vector_from_index`, `icosahedron_positive_x_views`, `project_positions`, `center_and_rotate_faceon`, `deposit_to_grid`, `sample_grid_at_points`, `load_cube_geometry`, `weighted_quantile`. Fija la convención de orientación del spec 20. |
| `labeling/constants.py::AXIS_VIEWS, TNG_SIMULATION` | **PORTAR** | `core/constants.py` | Esquema de clases v2 se redefine (5 clases físicas + máscara separada; el eje "incierto" desaparece del tensor, principio del diagnóstico 19.5). |
| `labeling/manifest.py`, `build_manifest.py`, `run_matched_labeling.py` (parsing de matched_units) | **PORTAR** | `io/manifest.py` | Vinculación cubo ↔ subhalo ↔ vista ↔ pyPipe3D. |
| `download_tng_truth.py` | **PORTAR** | `cli/download.py` | Descarga de cutouts vía API TNG. |
| `labeling/models.py` (TNGTruth, ManifestRow, CubeGeometry) | **PORTAR con cambios** | `schemas/` | Migrar dataclasses → pydantic; `LabelProducts` se reemplaza por los contratos de `docs/04_contratos.md`. |
| `labeling/epsilon_baseline.py` | **PORTAR como baseline** | `baselines/epsilon_threshold.py` | Es el baseline del artículo. OJO: su `circularity_proxy` (v_φ/v_total) NO es la ε canónica; mantenerlo solo como baseline etiquetado como proxy. La ε del pipeline es j_z/j_c(E) del Extractor (spec 10). |
| `labeling/decomposition.py` | **DESCARTAR** | — | Familias por perfiles radiales + `iterative_scale_rows` (catálogo como constraint). Reemplazado por el GMM del spec 11. |
| `labeling/pipeline.py` (resto) | **DESCARTAR** | — | Monolito que mezcla fases A–D. En particular NO portar: `recalibrate_central_bulge_disk` (parche ad-hoc del centro), `rescale_families_to_targets` (constraint duro), `smooth_probabilities_spatially` dentro de `harden_labels` (parche cosmético del problema 19.2), `harden_labels`. |
| `labeling/config.py` | **DESCARTAR** | — | Sus knobs (boosts centrales, supresiones) corresponden a los parches descartados. Config nueva por módulo vía pydantic. |
| `calibrate_labeling.py` | **DESCARTAR** | — | Calibraba los parches. |
| `test_*.py` | **Rescatar fixtures** | `tests/` | Los datos sintéticos y el smoke del piloto son útiles; las aserciones sobre los parches no. |

## Qué se reimplementa desde cero (no existe en esta rama)

Según specs, en este orden (Hito 2 → 3 → 4):

1. **Extractor** (spec 10): ε = j_z/j_c(E) canónica con potencial
   (octree o catálogo), E, j_z, j_total, R, z en marco face-on.
   *El v1 no la calcula; solo tiene el proxy del baseline.*
2. **Classifier GMM** (spec 11 v2.1): feature-set `paper4d` default,
   `standard3d` alternativo, reordenamiento por energía, prior MORDOR.
   *No existe ningún GMM en el v1.*
3. **BarDetector** (spec 12): Fourier m=2 sobre partículas de disco.
   *El v1 usa un modelo geométrico cos(2φ) con parámetros del catálogo;
   descartado.*
4. **ArmDetector** (spec 13): residuales δΣ. *El v1 tiene una versión
   en `derive_substructure_probabilities`; la lógica de residuales
   axisimétricos puede consultarse como referencia, pero la
   reimplementación debe operar sobre P(disk) del GMM, no sobre las
   familias radiales.*
5. **LabelProjection** (spec 20): reusar `deposit_to_grid` y
   `project_positions` portados; la agregación normalizada por spaxel,
   N_eff (Kish) y las 4 variantes son nuevas.
6. **MaskBuilder** (spec 22), **QualityCheck + Packer** (spec 30).

## Datos externos nuevos

- **Catálogo MORDOR** (Zana et al. 2022, MNRAS 515, 1524; público para
  TNG50 z=0): descargar una vez; usado como prior del Classifier
  (spec 11 paso 4) y validación por subhalo (Classifier y BarDetector).
  Mapeo: bulge_frac = bulge + pseudo-bulge; disk_frac = thin + thick;
  other_frac = halo. Si un subhalo no está en el catálogo, fallback a
  Rodriguez-Gomez 2022 y registrar `prior_source` en metadatos.

## Ajuste al Hito 0–1 (scaffolding mínimo)

Para no gastar presupuesto del agente en infraestructura antes de tener
física: en Hito 0–1 solo `pyproject.toml`, `pytest` y la estructura de
paquetes. CI, `mypy`, `ruff`, `typer` y notebooks se incorporan al
cierre del Hito 2, cuando ya hay módulos reales que proteger.

## Criterio de éxito de la migración (antes de empezar Hito 2)

- [ ] Los módulos portados importan y pasan un smoke test con el piloto
      TNG50-87-141934-0-127 (leer cutout, convertir unidades, rotar
      face-on, depositar masa total a la grilla del cubo).
- [ ] El test de alineación del spec 20 (Spearman > 0.9 vs mapa de masa
      pyPipe3D, centroide < 1 spaxel) pasa con la convención portada.
- [ ] Ninguna función de `decomposition.py`, ni los parches listados de
      `pipeline.py`, aparecen en el repo v2.
