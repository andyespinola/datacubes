# Generación de Pseudo-etiquetas (Soft Labels)

Este paquete contiene los specs, documentos y una implementación inicial
ejecutable del proceso de generación de pseudo-etiquetas estructurales
`Y_int(i,j,c)` para cubos IFU MaNGIA/TNG50. No incluye entrenamiento de red
ni módulos auxiliares posteriores como input fotométrico o momentos h3/h4.

## Contenido

### Implementación actual
- `pyproject.toml` — paquete instalable en editable y comando `aperturenet-labels`
- `configs/default.yaml` — configuración local para las dos galaxias en `../data/`
- `src/aperturenet_labels/` — readers, Fase A, Fase B, Fase C y CLI
- `scripts/make_visual_validation.py` — figuras QA de mapas 2D, máscara y fracciones
- `scripts/make_segmentation_validation.py` — validación visual de segmentación dura/soft, confianza, margen y ambigüedad
- `scripts/extract_stellar_potential_cache.py` — cachea `PartType4/Potential` desde snapshots completos cuando TNG permite el acceso
- `scripts/build_unique_mangia_manifest.py` — consolida el catálogo MaNGIA/TNG en galaxias únicas para planificar el escalado
- `tests/` — validación de assets locales y smoke end-to-end liviano

### Migración desde v1
- `MIGRATION.md` — qué portar del repo `structural_labeling` (geometría, IO, SSP, manifiestos) y qué reimplementar (todo el núcleo de clasificación)

### Documentos de diseño (`docs/`)
- `01_diagnostico.md` — por qué un pipeline v2 (motivación)
- `02_principios.md` — cinco principios de diseño (separación física/render, etc.)
- `03_arquitectura.md` — arquitectura del pipeline
- `04_contratos.md` — contratos HDF5 entre módulos
- `05_data_inventory.md` — inventario validado de las dos galaxias locales nuevas
- `06_implementation_notes.md` — qué se implementó, assets TNG descargados para Hito 2, cómo correrlo y qué queda pendiente

### Specs técnicas (`specs/`)
- `00_ROADMAP.md` — plan de hitos
- `10_extractor.md` — partículas, posiciones, energías, circularidad
- `11_classifier.md` — **v2.1**: GMM con el vector 4D del artículo (ε, log R/R_eff, |z|/R_eff, E_norm), feature-set alternativo estándar 3D, reordenamiento por energía, prior MORDOR
- `12_bar_detector.md` — detección de barras vía Fourier m=2
- `13_arm_detector.md` — detección de brazos espirales por δΣ
- `20_label_projection.md` — proyección 3D→2D + agregación + N_eff + 4 variantes
- `22_mask_builder.md` — máscara de validez `M_val`
- `30_quality_check.md` — control de calidad final

### Lo NO incluido (intencionalmente)
- Validación de pseudo-etiquetas (`40_pseudolabel_validation.md`)
- Validación cinemática (`42_kinematic_validation.md`)
- Generación de imagen fotométrica de input (`25_image_provider.md`)
- Extracción de momentos h3/h4 con pPXF (`26_kinematic_moments.md`)
- `docs/05_validacion.md` (estrategia de validación)

Estos módulos son posteriores a la generación de las pseudo-etiquetas o
sirven a otros fines (inputs del modelo, validación) y se distribuyen por
separado.

## Notas sobre `20_label_projection.md`

Este spec **reemplaza** los antiguos `20_projector.md` y `21_aggregator.md`
del plan v2 original. Consolida proyección + agregación + cómputo de
`N_eff` y entrega las cuatro variantes del tensor de pseudo-etiquetas:
`mass_raw`, `mass_psf`, `lum_raw`, `lum_psf`. La antigua dependencia abierta
sobre la orientación de MaNGIA quedó resuelta en la implementación: se porta
la geometría base del v1 y luego se aplica una alineación D4 contra
`stellar_mass_density_log10` de pyPipe3D, registrada en la metadata de
`labels2d_v0.npz`.

## Orden de lectura recomendado

1. `README.md` (raíz) — visión global del pipeline.
2. `docs/05_data_inventory.md` — confirma los datos locales actuales y los assets phase2 descargados desde TNG.
3. `docs/06_implementation_notes.md` — estado implementado y comandos.
4. `MIGRATION.md` — frontera portar/reimplementar respecto al v1.
5. `docs/02_principios.md` — principios de diseño.
6. `docs/03_arquitectura.md` — diagrama de flujo y módulos.
7. `docs/04_contratos.md` — schemas HDF5 entre módulos.
8. `specs/00_ROADMAP.md` — hitos y dependencias entre specs.
9. Specs por orden numérico (10 → 13 → 20 → 22 → 30).
