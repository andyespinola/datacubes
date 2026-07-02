# Spec: Extractor

> Módulo: `phase_a/extractor.py` · Hito: 2 · Dependencias previas: `io/tng_reader.py`, `io/mangia_reader.py`

## Responsabilidad

Tomar el cutout de TNG de una galaxia y producir un tensor estandarizado de **features físicas por partícula estelar**. **No clasifica**; solo computa cantidades intrínsecas.

## Contrato de entrada

```python
class ExtractorInput(BaseModel):
    cutout_path: Path                # cutout TNG (HDF5)
    subhalo_meta: SubhaloMeta        # del catálogo TNG
    snapshot_meta: SnapshotMeta      # snapshot completo (z, h, etc)
    config: ExtractorConfig
```

```python
class SubhaloMeta(BaseModel):
    subhalo_id: int
    snapshot: int
    center_pos: tuple[float, float, float]   # ckpc/h
    bulk_vel:   tuple[float, float, float]   # km/s
    r_eff_kpc:  float
    M_star:     float                        # M_sun

class ExtractorConfig(BaseModel):
    align_radius_factor: float = 2.0     # alinear con L en R < this × R_eff
    potential_method: Literal["octree", "catalog"] = "octree"
    n_jc_bins: int = 200
    age_unit: Literal["Gyr", "yr"] = "Gyr"
```

## Contrato de salida

```python
class ParticleFeatures(BaseModel):
    """Persisted as HDF5: particle_features.h5"""
    # Identificación
    galaxy_id: str                    # ej "TNG50-87-141934"
    n_particles: int

    # Features cinemáticas (todas arrays de shape (N,))
    epsilon: np.ndarray               # circularidad ε ∈ [-1, 1]
    R: np.ndarray                     # radio cilíndrico [kpc]
    z: np.ndarray                     # altura sobre plano [kpc]
    E: np.ndarray                     # energía específica [km^2/s^2]
    j_z: np.ndarray                   # momento angular z [kpc·km/s]
    j_c: np.ndarray                   # j_c(E) interpolado [kpc·km/s]
    j_total: np.ndarray               # |j| por partícula [kpc·km/s] (requerido por el feature-set standard3d del spec 11)

    # Features físicas
    pos_aligned: np.ndarray           # (N,3) en frame face-on [kpc]
    vel_aligned: np.ndarray           # (N,3) [km/s]
    mass: np.ndarray                  # masa estelar [M_sun]
    age: np.ndarray                   # edad [Gyr]
    metallicity: np.ndarray           # Z (no log)
    light_g: np.ndarray               # luminosidad banda g [L_sun]

    # Metadata para downstream
    R_eff_kpc: float                  # radio efectivo recomputado
    L_total: tuple[float, float, float]  # momento angular total usado para alinear
    quality: dict                     # ver "Quality metrics" abajo
```

## Algoritmo

```
1. Cargar partículas estelares del cutout (PartType4)
2. Centrar en subhalo: pos -= subhalo.center_pos
3. Quitar velocidad sistémica: vel -= subhalo.bulk_vel
4. Convertir unidades comoving → físicas usando snapshot.z y snapshot.h
5. Identificar partículas centrales: R < align_radius_factor × R_eff
6. Calcular L_total = Σ m_i × (r_i × v_i) sobre partículas centrales
7. Construir matriz de rotación que lleva L_total → eje z
8. Aplicar rotación a pos y vel → pos_aligned, vel_aligned
9. Calcular potencial Φ(r):
   - Si potential_method="catalog" y disponible: leer del catálogo TNG
   - Si potential_method="octree" o fallback: octree local con todas las
     partículas del cutout (estrellas + gas + DM)
10. Calcular E = ½v² + Φ
11. Calcular j_z = x*v_y - y*v_x  (en el frame alineado)
12. Calcular j_total = |r × v|
13. Construir j_c(E) por envolvente:
    a) Ordenar partículas por energía
    b) Para cada bin de E (n_jc_bins bins), tomar j_max en ese bin
    c) Suavizar con spline cúbico
14. Para cada partícula, interpolar j_c en su E → j_c_p
15. ε = j_z / j_c (clip a [-1, 1])
16. Computar age desde formation_time (a → cosmología → Gyr)
17. Computar light_g desde (mass, age, metallicity) usando grilla SSP
    (delegar a io/ssp_grid.py)
18. Calcular R_eff recomputado: radio que contiene 50% de la masa proyectada
19. Calcular quality metrics
20. Persistir como HDF5
```

## Cálculo del potencial — detalles

**Opción 1 (preferida): catálogo TNG**

Si el catálogo de circularidades de TNG (Rodriguez-Gomez et al. 2022) está disponible para el subhalo, usar directamente ε y E del catálogo. Ahorra tiempo y garantiza consistencia con la literatura.

```python
def load_circularities_from_catalog(subhalo_id, snapshot, catalog_path):
    """Returns (epsilon, E, j_z) per particle, or None if not available."""
    ...
```

**Opción 2: octree local**

Si no está disponible, computar el potencial con las partículas del cutout:

```python
from scipy.spatial import cKDTree

def compute_potential_octree(positions, masses, eps_softening=0.1):
    """
    Potential at each particle position from all particles
    (including gas + DM if provided).
    Uses tree-based summation; O(N log N).
    """
    tree = cKDTree(positions)
    # Gravitational potential with Plummer softening
    G = 4.302e-6  # kpc · (km/s)^2 / M_sun
    phi = np.zeros(len(positions))
    for i in range(len(positions)):
        d = np.linalg.norm(positions - positions[i], axis=1)
        d_soft = np.sqrt(d**2 + eps_softening**2)
        phi[i] = -G * np.sum(masses / d_soft)
        phi[i] -= -G * masses[i] / eps_softening  # remove self
    return phi
```

(Para partículas pesadas, paralelizar con joblib o vectorizar con tree code.)

## Quality metrics

```python
quality = {
    "n_particles": int,
    "n_central": int,                    # usadas para alinear
    "L_total_magnitude": float,          # |L_total| [kpc·km/s]
    "epsilon_mean": float,
    "epsilon_std": float,
    "epsilon_p7_fraction": float,        # fracción con ε > 0.7
    "epsilon_n3_fraction": float,        # fracción con ε < -0.3
    "potential_method_used": str,        # "catalog" o "octree"
    "compute_time_sec": float,
}
```

## Validación

### Test unitario (`tests/unit/test_extractor.py`)

1. Sintetizar una galaxia disco-puro de juguete (1000 partículas en órbitas circulares planas) → `ε ≈ 1` para todas.
2. Sintetizar bulbo-puro (1000 partículas en órbitas isotrópicas) → distribución de ε centrada en 0 con std ~0.3.
3. Galaxia mixta 50/50: histograma de ε debe ser bimodal.
4. Verificar conservación de masa: `Σ mass == M_star del subhalo` ±1%.
5. Verificar `R_eff` recomputado dentro del 10% del valor del catálogo.

### Test de integración con piloto

```bash
python -m aperturenet_labels.cli extract \
    --cutout data/pilot/TNG50-87-141934-cutout.h5 \
    --output /tmp/pilot_features.h5
```

Verificar:
- `quality.n_particles > 100_000` (galaxia bien resuelta)
- `quality.epsilon_p7_fraction` consistente con catálogo de descomposición (debería estar en el rango 0.4-0.7 para una galaxia disco)
- Tiempo < 60 segundos

### Comparación con ε de catálogo

Si el catálogo de Rodriguez-Gomez 2022 está disponible para el piloto:

```python
eps_ours = features.epsilon
eps_cat  = load_catalog_epsilons(subhalo_id=141934, snapshot=87)
rmse = np.sqrt(np.mean((eps_ours - eps_cat)**2))
assert rmse < 0.05, f"epsilon mismatch RMSE={rmse}"
```

## Criterios de aceptación del módulo

- [ ] Tests unitarios pasan
- [ ] Test de integración con piloto pasa
- [ ] `epsilon` dentro de RMSE 0.05 vs catálogo (si disponible)
- [ ] Tiempo < 60s por galaxia en máquina de referencia
- [ ] Output HDF5 valida contra el schema de `ParticleFeatures`
- [ ] Logging estructurado de cada paso
- [ ] Documentación inline (docstrings) completa
- [ ] `mypy` y `ruff` limpios

## Notas de implementación

- **No usar pandas** para los arrays grandes; numpy directo.
- **Usar `h5py` con compresión `lzf`** para los outputs (rápida y razonable).
- **Manejar correctamente las unidades**: TNG usa cgs y comoving; documentar conversiones en cada función.
- **Failure mode**: si el cutout no tiene partículas estelares o tiene < 100, levantar `InsufficientResolutionError` y skipear esa galaxia downstream.

## Referencias

- Abadi et al. 2003, ApJ 597:21 (formulación original de ε)
- Du et al. 2020, ApJ 895:139 (descomposición cinemática de TNG)
- Rodriguez-Gomez et al. 2022, MNRAS (catálogo de circularidades TNG50)
