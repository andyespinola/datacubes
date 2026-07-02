# Spec: ArmDetector

> Módulo: `phase_a/arm_detector.py` · Hito: 2 · Depende de: BarDetector

## Responsabilidad

Detectar partículas pertenecientes a **brazos espirales** mediante análisis de residuales sobre el modelo axisimétrico del disco. Solo opera sobre partículas con `P_disk > 0.3` y excluye la región de barra. Reasigna probabilidad de `disk` a `arm`.

## Contrato de entrada

```python
class ArmDetectorInput(BaseModel):
    features_path: Path                          # particle_features.h5
    intermediate_labels_path: Path               # particle_labels_with_bar.h5
    bar_meta: BarMeta                            # para excluir región de barra
    config: ArmDetectorConfig

class ArmDetectorConfig(BaseModel):
    min_disk_prob: float = 0.3              # umbral para considerar partícula
    fine_grid_size: int = 256               # resolución del mapa face-on
    map_extent_kpc: float = 30.0            # extensión del mapa face-on
    residual_threshold: float = 0.3         # δ_min para identificar cresta
    min_island_area: int = 20               # spaxels mínimos de una cresta
    min_azimuthal_extent_deg: float = 30.0  # extensión angular mínima
```

## Contrato de salida

Añade columna `P_arm` al tensor, redistribuyendo desde `P_disk`.

```python
class FinalLabels(BaseModel):
    """Persisted as HDF5: particle_labels_final.h5"""
    galaxy_id: str
    P_class: np.ndarray            # (N, 5): [bulge, disk, bar, arm, halo]
    arm_diagnostics: dict
```

## Algoritmo

```
1. Filtrar partículas con P_disk > min_disk_prob (ya descontando P_bar)
   → "disk dominated" particles
2. Si menos de 100 partículas dominantes en disco:
   P_arm = 0, return (sin estadística suficiente)
3. Construir mapa de densidad superficial face-on alta resolución:
   sigma_disk(x, y) = histogram2d(pos_x, pos_y,
                                   weights = mass × P_disk)
   sobre grilla de fine_grid_size² spaxels en map_extent_kpc²
4. Calcular perfil radial axisimétrico:
   R_grid(x, y) = sqrt(x² + y²)
   sigma_axisym(R) = mean(sigma_disk en anillos de R)
   sigma_axisym_2d(x, y) = interp(sigma_axisym, R_grid)
5. Residuales normalizados:
   delta(x, y) = (sigma_disk - sigma_axisym_2d) / max(sigma_axisym_2d, eps)
6. Identificar crestas espirales:
   spiral_mask = (delta > residual_threshold)
   spiral_mask = remove_small_islands(spiral_mask, min_island_area)
   spiral_mask = require_azimuthal_extent(spiral_mask, min_azimuthal_extent_deg)
7. Excluir región de barra:
   if bar_meta.has_bar:
       in_bar_region = R_grid < bar_meta.bar_size_kpc
       spiral_mask &= ~in_bar_region
8. Mapear partículas a la máscara:
   indices_x = digitize(pos_x, grid edges)
   indices_y = digitize(pos_y, grid edges)
   in_arm = spiral_mask[indices_y, indices_x]
9. P_arm = (in_arm & disk_dominated).astype(float) × P_disk
10. P_disk_new = P_disk - P_arm
11. P_class_new = stack([P_bulge, P_disk_new, P_bar, P_arm, P_halo])
12. arm_diagnostics = {n_crests, total_arm_area, arm_mass_fraction, mean_pitch_angle?}
```

## Detección de crestas — detalles

```python
def remove_small_islands(mask, min_area):
    """Quita islas con área < min_area usando connected components."""
    from scipy.ndimage import label
    labeled, n = label(mask)
    sizes = np.bincount(labeled.flat)
    sizes[0] = 0   # ignorar fondo
    keep = sizes >= min_area
    return keep[labeled]

def require_azimuthal_extent(mask, min_extent_deg):
    """Mantiene solo islas que se extienden > min_extent_deg azimutalmente."""
    from scipy.ndimage import label
    labeled, n = label(mask)
    keep = np.zeros(n + 1, dtype=bool)
    H, W = mask.shape
    for k in range(1, n + 1):
        ys, xs = np.where(labeled == k)
        # Centrar en el centro del mapa
        cx, cy = W // 2, H // 2
        phis = np.arctan2(ys - cy, xs - cx)
        extent_deg = np.degrees(phis.max() - phis.min())
        if extent_deg >= min_extent_deg:
            keep[k] = True
    return keep[labeled]
```

## Validación

### Test unitario

1. **Disco axisimétrico puro** (sin estructura espiral): `P_arm.sum() == 0` o muy pequeño.
2. **Disco con espiral sintética** (densidad de 2 brazos logarítmicos sobre disco exponencial): debe detectar 2 crestas, fracción de masa en arm entre 5% y 25%.
3. **Conservación**: `P_class_new.sum(axis=1) ≈ 1`.
4. **Barra excluida**: si bar_meta.has_bar=True, ninguna partícula con R < R_bar tiene P_arm > 0.

### Test de integración

Sobre piloto, comparar visualmente con la imagen del catálogo TNG (overlay de la máscara de brazos sobre el mapa de densidad estelar).

## Criterios de aceptación

- [ ] Tests unitarios pasan
- [ ] Disco axisimétrico → P_arm ≈ 0
- [ ] Disco con espirales sintéticas → 2 crestas detectadas
- [ ] No detecta brazos dentro de la región de barra
- [ ] Tiempo < 10s por galaxia
- [ ] Logging incluye n_crests, fracción de masa en brazos

## Notas y limitaciones conocidas

- **Galaxias face-on funcionan mejor que edge-on**. Como el frame ya está alineado face-on por el Extractor, esto no es problema.
- **Galaxias floculentas vs grand-design**: el algoritmo detecta mejor grand-design. Para floculentas, los parámetros pueden ser muy estrictos. Considerar un modo más laxo en versiones futuras.
- **Brazos viejos vs jóvenes**: actualmente no distinguimos. Una mejora futura sería ponderar también por edad de las partículas.
- **Pitch angle**: el algoritmo no lo calcula explícitamente. Es una métrica útil para validación cualitativa pero no es necesaria para el target.

## Referencias

- Sellwood & Carlberg 1984 (modelos clásicos de brazos espirales)
- Bottrell & Hani 2022, MNRAS 514:2821 (RealSim-IFS)
