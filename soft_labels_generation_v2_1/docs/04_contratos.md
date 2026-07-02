# Contratos de datos entre módulos

> Este documento define los **schemas HDF5** de los productos intermedios. Cualquier módulo que lea o escriba debe ajustarse exactamente a estos contratos.

## Convenciones generales

- **Tipos**: float32 para arrays grandes, float64 para escalares y metadata.
- **Compresión**: `lzf` para velocidad. `gzip` solo si tamaño es crítico.
- **Atributos**: cada dataset tiene attrs con `units`, `description`, `source_module`.
- **Versionado**: cada archivo tiene attr `schema_version` para futura compatibilidad.

## particle_features.h5

Producido por **Extractor**. Una galaxia → un archivo.

```
/                        attrs: {schema_version="1.0", source_module="extractor"}
├── metadata
│   attrs:
│       galaxy_id        (str)
│       snapshot         (int)
│       subhalo_id       (int)
│       n_particles      (int)
│       extracted_at     (ISO 8601)
│
├── kinematic
│   ├── epsilon          (N,) float32   units="dimensionless"
│   ├── R                (N,) float32   units="kpc"
│   ├── z                (N,) float32   units="kpc"
│   ├── E                (N,) float32   units="(km/s)^2"
│   ├── j_z              (N,) float32   units="kpc·km/s"
│   └── j_c              (N,) float32   units="kpc·km/s"
│
├── physical
│   ├── pos_aligned      (N, 3) float32 units="kpc"
│   ├── vel_aligned      (N, 3) float32 units="km/s"
│   ├── mass             (N,) float32   units="M_sun"
│   ├── age              (N,) float32   units="Gyr"
│   ├── metallicity      (N,) float32   units="Z"
│   └── light_g          (N,) float32   units="L_sun"
│
└── quality
    attrs:
        n_particles      (int)
        n_central        (int)
        L_total          (3,) float32
        R_eff_kpc        (float32)
        epsilon_mean     (float32)
        ...
```

## particle_labels_initial.h5

Producido por **Classifier**.

```
/                        attrs: {schema_version="1.0", source_module="classifier"}
├── metadata
│   attrs:
│       galaxy_id        (str)
│       n_particles      (int)
│       method_used      (str: "gmm" | "hard_thresholds_fallback")
│
├── P_class              (N, 3) float32   columns=["bulge", "disk", "halo"]
│                        attrs: column_names
│
├── gmm_params
│   ├── means            (3, n_features) float32
│   ├── covariances      (3, n_features, n_features) float32
│   └── weights          (3,) float32
│
└── quality
    attrs:
        bic, aic         (float32)
        converged        (bool)
        silhouette       (float32)
        fractions_recovered  (3,) float32
        fractions_catalog    (3,) float32 (or NaN)
```

## particle_labels_with_bar.h5

Producido por **BarDetector**. Mismo formato que initial pero con 4 columnas.

```
/
├── metadata
│   attrs:
│       galaxy_id, n_particles
│       has_bar          (bool)
│       bar_size_kpc     (float32)
│
├── P_class              (N, 4) float32   columns=["bulge", "disk", "bar", "halo"]
│
└── bar_diagnostics
    attrs:
        a2               (float32)
        phi_bar_rad      (float32)
        n_bar_particles  (int)
        bar_mass_fraction (float32)
```

## particle_labels_final.h5

Producido por **ArmDetector**. **Este es el producto final de la Fase A**.

```
/                        attrs: {schema_version="1.0", source_module="arm_detector"}
├── metadata
│   attrs:
│       galaxy_id, n_particles
│       phase_a_complete  (bool=True)
│       generated_at      (ISO 8601)
│
├── P_class              (N, 5) float32   columns=["bulge", "disk", "bar", "arm", "halo"]
│
├── arm_diagnostics
│   attrs:
│       n_crests, total_arm_area, arm_mass_fraction
│
└── full_pipeline_diagnostics    # acumula los de classifier y bar_detector
    attrs: ...
```

## projection_raw_v{view_id}.npz

Producido por **Projector**. Una galaxia × una orientación → un archivo.

```
NPZ fields:
    fine_mass            (296, 296, 5) float32   M_sun per fine spaxel per class
    fine_light           (296, 296, 5) float32   L_sun
    particle_count       (296, 296)    int32
    metadata             dict (saved as JSON in NPZ field)
        galaxy_id, view_id
        oversample, fov_arcsec
        orientation: {theta_deg, phi_deg, psi_deg, distance_mpc}
```

## Y_int_v{view_id}.npz

Producido por **Aggregator**.

```
NPZ fields:
    Y_int_mass           (74, 74, 5) float32   normalized to sum=1 per spaxel
    Y_int_light          (74, 74, 5) float32   normalized to sum=1 per spaxel
    raw_mass_per_class   (74, 74, 5) float32   M_sun
    raw_light_per_class  (74, 74, 5) float32   L_sun
    total_mass_per_spaxel  (74, 74) float32
    total_light_per_spaxel (74, 74) float32
    class_names          ["bulge", "disk", "bar", "arm", "halo"]
    metadata             dict
```

## M_valid_v{view_id}.npz

Producido por **MaskBuilder**.

```
NPZ fields:
    M_valid              (74, 74) bool
    M_criterion_A        (74, 74) bool
    M_criterion_B        (74, 74) bool
    M_criterion_C        (74, 74) bool
    diagnostics          dict
```

## dataset_entry_{galaxy_id}_{view_id}.h5

Producido por **Packer**. **Este es el archivo que el dataloader del entrenamiento consume.**

```
/                        attrs: {schema_version="1.0", pipeline_version="v2"}
├── metadata
│   attrs:
│       galaxy_id, view_id, snapshot, subhalo_id
│       orientation     (group) {theta_deg, phi_deg, psi_deg, distance_mpc}
│       generated_at    (ISO 8601)
│
├── inputs
│   ├── cube_ifu         (6603, 74, 74) float32   units="erg/s/cm^2/Å"
│   └── pipe3d_maps
│       ├── v_star       (74, 74) float32   units="km/s"
│       ├── sigma_star   (74, 74) float32   units="km/s"
│       ├── age          (74, 74) float32   units="Gyr"
│       ├── metallicity  (74, 74) float32   units="dex"
│       ├── mass         (74, 74) float32   units="log(M_sun/spaxel)"
│       └── av           (74, 74) float32   units="mag"
│
├── labels
│   ├── Y_int_mass       (74, 74, 5) float32
│   ├── Y_int_light      (74, 74, 5) float32
│   └── class_names      ["bulge", "disk", "bar", "arm", "halo"]
│
├── masks
│   └── M_valid          (74, 74) bool
│
└── qa
    attrs: ...
```

## Validación de schemas con pydantic

Todos los productos tienen un modelo pydantic en `src/aperturenet_labels/schemas/`:

```python
# schemas/particle_features.py
from pydantic import BaseModel, validator
import numpy as np

class ParticleFeaturesSchema(BaseModel):
    galaxy_id: str
    n_particles: int
    epsilon: np.ndarray
    # ... resto

    class Config:
        arbitrary_types_allowed = True

    @validator('epsilon')
    def epsilon_in_range(cls, v):
        assert v.min() >= -1.01 and v.max() <= 1.01
        return v
```

Cada lectura de un producto pasa por validación:

```python
def load_particle_features(path: Path) -> ParticleFeaturesSchema:
    with h5py.File(path, 'r') as f:
        data = {
            'galaxy_id': f['metadata'].attrs['galaxy_id'],
            'n_particles': int(f['metadata'].attrs['n_particles']),
            'epsilon': f['kinematic/epsilon'][:],
            # ...
        }
    return ParticleFeaturesSchema(**data)   # raises ValidationError si malformado
```

Esto garantiza que un módulo que produce datos malformados es detectado **inmediatamente** y no contamina los downstream.
