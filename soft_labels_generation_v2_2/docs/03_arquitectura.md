# Arquitectura del pipeline v2

> Vista general de los módulos, fases, y diagrama de flujo.

## Visión general

El pipeline v2 está organizado en **3 fases** con **9 módulos**:

```
┌─────────────────────────────────────────────────────────────────┐
│  FASE A — FÍSICA (una vez por galaxia, invariante a orientación)│
│                                                                 │
│  TNG cutout                                                     │
│       ↓                                                         │
│  ┌─────────────┐    ┌─────────────────┐                         │
│  │ Extractor   │ →  │ particle_       │                         │
│  │             │    │ features.h5     │                         │
│  └─────────────┘    └─────────────────┘                         │
│       ↓                                                         │
│  ┌─────────────┐    ┌─────────────────┐                         │
│  │ Classifier  │ →  │ particle_labels │ ← prior: catálogo TNG   │
│  │   (GMM)     │    │ _initial.h5     │                         │
│  └─────────────┘    └─────────────────┘                         │
│       ↓                                                         │
│  ┌─────────────┐    ┌─────────────────┐                         │
│  │ BarDetector │ →  │ particle_labels │                         │
│  └─────────────┘    │ _with_bar.h5    │                         │
│       ↓             └─────────────────┘                         │
│  ┌─────────────┐    ┌─────────────────┐                         │
│  │ ArmDetector │ →  │ particle_labels │ ← producto persistente  │
│  └─────────────┘    │ _final.h5       │                         │
│                     └─────────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  FASE B — RENDERIZADO (una vez por orientación)                 │
│                                                                 │
│  ┌─────────────┐    ┌─────────────────┐                         │
│  │ Projector   │ →  │ projection_     │ ← orientación: θ, φ, ψ  │
│  │             │    │ raw_v{i}.npz    │                         │
│  └─────────────┘    └─────────────────┘                         │
│       ↓                                                         │
│  ┌─────────────┐    ┌─────────────────┐                         │
│  │ Aggregator  │ →  │ Y_int_v{i}.npz  │                         │
│  │             │    │  (mass + light) │                         │
│  └─────────────┘    └─────────────────┘                         │
│       ↓                                                         │
│  ┌─────────────┐    ┌─────────────────┐                         │
│  │ MaskBuilder │ →  │ M_valid_v{i}    │ ← cubo IFU para S/N     │
│  │             │    │ .npz            │                         │
│  └─────────────┘    └─────────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  FASE C — VALIDACIÓN Y EMPAQUETADO                              │
│                                                                 │
│  ┌─────────────┐    ┌─────────────────┐                         │
│  │ QualityCheck│ →  │ qa_report.json  │                         │
│  └─────────────┘    └─────────────────┘                         │
│       ↓                                                         │
│  ┌─────────────┐    ┌─────────────────┐                         │
│  │ Packer      │ →  │ dataset_entry   │ ← cubo IFU + pyPipe3D   │
│  │             │    │ _v{i}.h5 (final)│                         │
│  └─────────────┘    └─────────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Tabla de módulos

| Módulo | Fase | Spec | Entrada | Salida | Decisión |
|--------|------|------|---------|--------|----------|
| Extractor | A | [10](../specs/10_extractor.md) | TNG cutout | particle_features.h5 | Ninguna física, solo cálculo |
| Classifier | A | [11](../specs/11_classifier.md) | features | particle_labels_initial.h5 | P(bulge\|p), P(disk\|p), P(halo\|p) |
| BarDetector | A | [12](../specs/12_bar_detector.md) | features + initial | particle_labels_with_bar.h5 | P(bar\|p), redistribuye disk → bar |
| ArmDetector | A | [13](../specs/13_arm_detector.md) | + with_bar | particle_labels_final.h5 | P(arm\|p), redistribuye disk → arm |
| LabelProjection | B | [20](../specs/20_label_projection.md) | final + vista MaNGIA | labels2d_v{i}.npz (4 variantes + N_eff) | Geometría 3D → 2D + agregación |
| MaskBuilder | B | [22](../specs/22_mask_builder.md) | projection_raw + cubo | M_valid_v{i}.npz | Validez física + observacional |
| QualityCheck | C | [30](../specs/30_quality_check.md) | todo | qa_report.json | Métricas de calidad |
| Packer | C | [30](../specs/30_quality_check.md) | todo | dataset_entry_v{i}.h5 | Empaquetado final |

## Granularidad de ejecución

| Producto | Granularidad | Ejecuciones para 10K galaxias × 4 vistas |
|----------|--------------|------------------------------------------|
| particle_features | por galaxia | 10,000 |
| particle_labels_initial | por galaxia | 10,000 |
| particle_labels_with_bar | por galaxia | 10,000 |
| particle_labels_final | por galaxia | 10,000 |
| projection_raw | por galaxia × vista | 40,000 |
| Y_int | por galaxia × vista | 40,000 |
| M_valid | por galaxia × vista | 40,000 |
| qa_report | por galaxia × vista | 40,000 |
| dataset_entry | por galaxia × vista | 40,000 |

**Comparación con v1**:
- v1: 40,000 ejecuciones del pipeline completo (~35s cada una) → ~14 días
- v2: 10,000 ejecuciones de Fase A (~30s) + 40,000 de Fase B+C (~5s) → ~5.3 días

**Ganancia: ~2.6× más rápido en primera ejecución, mucho más rápido en re-ejecuciones parciales** (la mayoría de los cambios afectan solo un módulo).

## Layout en disco

```
data/
├── tng_cutouts/                  # input
│   ├── snapshot_087/
│   │   └── subhalo_141934.h5
│   └── ...
├── mangia/                       # input
│   ├── catalog.csv
│   └── cubes/
│       └── TNG50-87-141934-0-127.cube.fits.gz
├── pipe3d_maps/                  # input (ya derivados)
│   └── TNG50-87-141934-0-127/
│       ├── v_star.fits
│       └── ...
├── intermediate/                 # productos del pipeline
│   ├── phase_a/
│   │   └── TNG50-87-141934/
│   │       ├── particle_features.h5
│   │       ├── particle_labels_initial.h5
│   │       ├── particle_labels_with_bar.h5
│   │       └── particle_labels_final.h5
│   └── phase_b/
│       └── TNG50-87-141934/
│           ├── projection_raw_v0.npz
│           ├── projection_raw_v1.npz
│           ├── projection_raw_v2.npz
│           ├── projection_raw_v3.npz
│           ├── Y_int_v0.npz
│           ├── ...
│           ├── M_valid_v0.npz
│           └── ...
└── output/                       # productos finales
    ├── qa_reports/
    │   └── TNG50-87-141934_v0.json
    └── dataset_entries/
        └── TNG50-87-141934_v0.h5     # ← consumido por dataloader
```

## Configuración

Toda la configuración vive en YAML:

```yaml
# configs/default.yaml
extractor:
  align_radius_factor: 2.0
  potential_method: octree
  n_jc_bins: 200

classifier:
  method: gmm
  n_components: 3
  use_features: [epsilon, R_norm, z_norm]
  epsilon_init_thresholds:
    disk_min: 0.5
    bulge_max: 0.3
    halo_max: -0.3
  bulge_radial_max_kpc: 2.0
  seed: 42

bar_detector:
  epsilon_min: 0.3
  epsilon_max: 0.6
  z_max_kpc: 0.5
  a2_threshold: 0.3
  phi_tolerance_rad: 0.785

arm_detector:
  min_disk_prob: 0.3
  fine_grid_size: 256
  map_extent_kpc: 30.0
  residual_threshold: 0.3
  min_island_area: 20
  min_azimuthal_extent_deg: 30.0

projector:
  fov_arcsec: 30.0
  final_grid_size: 74
  oversample: 4

aggregator:
  normalize_per_spaxel: true
  add_background_class: true
  background_mass_threshold: 1.0e3

mask_builder:
  min_particles_per_spaxel: 30
  snr_window_angstrom: [5000.0, 5500.0]
  min_snr: 3.0
  min_island_area: 10
  closing_radius: 1

quality_check:
  fraction_tolerance: 0.10
  min_valid_fraction: 0.30
  max_uncertainty_p95: 0.5

packer:
  include_qa: true
  include_pipe3d: true
  compression: lzf
  cube_dtype: float32
  label_dtype: float32
```

Configuraciones especializadas (e.g. `configs/pilot.yaml`) sobrescriben valores específicos.
