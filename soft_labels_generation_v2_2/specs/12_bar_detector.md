# Spec: BarDetector

> Módulo: `phase_a/bar_detector.py` · Hito: 2 · Depende de: Classifier

## Responsabilidad

Detectar partículas pertenecientes a la **barra galáctica** (si existe) mediante un criterio combinado de cinemática (ε intermedio + delgadez en z) y morfología (análisis de Fourier m=2 + radio de barra del catálogo). Reasigna probabilidad de `disk` a `bar`.

## Contrato de entrada

```python
class BarDetectorInput(BaseModel):
    features_path: Path                  # particle_features.h5
    initial_labels_path: Path            # particle_labels_initial.h5
    bar_meta: BarMeta                    # del catálogo TNG
    config: BarDetectorConfig

class BarMeta(BaseModel):
    has_bar: bool
    bar_size_kpc: Optional[float]        # R_bar del catálogo TNG
    bar_strength: Optional[float]        # A2 reportado en catálogo
    bar_angle_deg: Optional[float]       # ángulo de la barra (si reportado)

class BarDetectorConfig(BaseModel):
    epsilon_min: float = 0.3              # cinemática: ε_min para barra
    epsilon_max: float = 0.6              # cinemática: ε_max para barra
    z_max_kpc: float = 0.5                # cinemática: |z| máximo
    a2_threshold: float = 0.3             # morfología: A2 mínimo
    phi_tolerance_rad: float = 0.785      # morfología: ±π/4 alrededor del eje barra
```

## Contrato de salida

Añade columna `P_bar` al tensor de probabilidades, redistribuyendo desde `P_disk`.

```python
class IntermediateLabels(BaseModel):
    """Persisted as HDF5: particle_labels_with_bar.h5"""
    galaxy_id: str
    P_class: np.ndarray            # (N, 4): [bulge, disk, bar, halo]
    bar_diagnostics: dict
```

## Algoritmo

```
1. Si bar_meta.has_bar == False → P_bar = 0 para todas las partículas, return.
2. R_bar = bar_meta.bar_size_kpc
3. Análisis Fourier m=2 sobre densidad estelar face-on:
   - Filtrar partículas con R < R_bar
   - φ_p = atan2(y_p, x_p)
   - C2 = Σ_p m_p × exp(2i × φ_p) / Σ_p m_p
   - A2 = |C2|, φ_bar = arg(C2) / 2
4. Si A2 < a2_threshold → barra no detectable, P_bar = 0, return.
5. Criterio cinemático:
   is_kinematic = (ε > ε_min) & (ε < ε_max) & (|z| < z_max)
6. Criterio morfológico:
   φ_relative_to_bar = ((φ_p - φ_bar + π/2) % π) - π/2
   is_morphological = (R < R_bar) & (|φ_relative_to_bar| < phi_tolerance_rad)
7. P_bar = (is_kinematic & is_morphological).astype(float) × P_disk
8. P_disk_new = P_disk - P_bar
9. Concatenar nuevas columnas:
   P_class_new = stack([P_bulge, P_disk_new, P_bar, P_halo])
10. bar_diagnostics = {a2, phi_bar, n_bar_particles, bar_mass_fraction}
```

## Validación

### Test unitario

1. **Galaxia sin barra** (bar_meta.has_bar=False): `P_bar.sum() == 0` exactamente.
2. **Galaxia con barra sintética** (densidad m=2 fuerte): A2 detectado > 0.5, fracción de partículas en barra entre 5% y 30%.
3. **Conservación de probabilidad**: `P_class_new.sum(axis=1) ≈ 1` para todas las partículas.
4. **No tomar de bulge ni halo**: `P_bulge` y `P_halo` no cambian.

### Test de integración con piloto

Si el piloto tiene `has_bar=True` en el catálogo:
- A2 detectado debe ser comparable al `bar_strength` del catálogo (±20%)
- φ_bar debe ser coherente con `bar_angle_deg` si el catálogo lo provee

## Criterios de aceptación

- [ ] Tests unitarios pasan
- [ ] Sobre piloto: A2 dentro del ±20% del catálogo
- [ ] Sobre galaxia sin barra: cero falsos positivos
- [ ] `P_class.sum(axis=1) == 1` ±1e-6 para todas las partículas
- [ ] Tiempo < 5s por galaxia
- [ ] Logging incluye A2, n_bar, fracción de masa de barra

## Notas

- La detección se hace en el **frame face-on**, que ya está alineado por el Extractor.
- Si el catálogo no provee `bar_angle_deg`, lo derivamos del análisis de Fourier.
- Para galaxias muy poco masivas o mal resueltas, la detección puede ser ruidosa: si N_partículas con R < R_bar < 100, devolver P_bar = 0 con warning.

## Validación externa con MORDOR (añadido v2.1)

El catálogo MORDOR (Zana et al. 2022, TNG50, público) reporta por
subhalo: flag de barra, R_bar (R_Φ y R_peak) y fuerza A₂ (A₂,max y
A₂(<R_peak)). Validar sobre la muestra con cobertura MORDOR:

- Acuerdo del flag de barra (detectada/no detectada) > 85%.
- Para galaxias barradas en ambos: |R_bar propio − R_peak MORDOR|
  dentro del 30% y correlación de A₂ Spearman ρ > 0.6.

Esto sustituye la validación basada únicamente en el catálogo
morfológico de Rodriguez-Gomez para la barra.
