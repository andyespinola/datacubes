# Spec: MaskBuilder

> Módulo: `phase_b/mask_builder.py` · Hito: 3 · Depende de: Projector, lectura del cubo IFU

## Responsabilidad

Construir la **máscara binaria de validez** `M_valid(74, 74)` que indica qué spaxels deben usarse en el entrenamiento. Combina criterios físicos (conteo de partículas), observacionales (S/N del cubo IFU) y geométricos (conectividad).

Esta es la **Mejora #2** del documento de análisis de etiquetas, llevada a su forma de módulo independiente.

## Contrato de entrada

```python
class MaskBuilderInput(BaseModel):
    projection_raw_path: Path           # para particle_count
    aggregated_path: Path               # para total_mass_per_spaxel
    cube_ifu_path: Path                 # cubo MaNGIA reconstruido
    config: MaskBuilderConfig

class MaskBuilderConfig(BaseModel):
    # Criterio A — conteo de partículas
    min_particles_per_spaxel: int = 30

    # Criterio B — señal observacional
    snr_window_angstrom: tuple[float, float] = (5000.0, 5500.0)
    min_snr: float = 3.0

    # Criterio C — conectividad
    min_island_area: int = 10
    closing_radius: int = 1
```

## Contrato de salida

```python
class ValidMask(BaseModel):
    """Persisted as NPZ: M_valid_v{view_id}.npz"""
    galaxy_id: str
    view_id: int

    M_valid: np.ndarray           # (74, 74) binary
    M_criterion_A: np.ndarray     # (74, 74) binary - particle count
    M_criterion_B: np.ndarray     # (74, 74) binary - S/N
    M_criterion_C: np.ndarray     # (74, 74) binary - connectivity
    diagnostics: dict
```

## Algoritmo

```
1. Cargar particle_count desde projection_raw (296×296)
2. Agregar a 74×74:
   particle_count_74 = particle_count.reshape(74,4,74,4).sum(axis=(1,3))
3. Criterio A: M_A = particle_count_74 >= min_particles_per_spaxel
4. Criterio B: cargar cubo IFU
   - Calcular S/N en ventana spectral (5000-5500 Å)
   - signal = mean(flux[mask_window]) por spaxel
   - noise = std(flux[mask_window]) por spaxel
   - snr = signal / noise
   - M_B = snr >= min_snr
5. Combinar: M_AB = M_A & M_B
6. Criterio C: conectividad espacial
   - labeled = label(M_AB)
   - Tomar solo la componente conexa más grande
   - Eliminar islas con área < min_island_area
   - Closing morfológico con radio=closing_radius para rellenar huecos
   → M_C
7. M_valid = M_C  (final)
8. diagnostics = {
       "n_valid_total": int(M_valid.sum()),
       "n_only_A_invalid": int((~M_A & M_B & M_C).sum()),
       "n_only_B_invalid": int((M_A & ~M_B & M_C).sum()),
       "n_dropped_by_C": int((M_AB & ~M_C).sum()),
       "fraction_valid": float(M_valid.mean()),
   }
```

## Validación

### Test unitario

1. **Particle count masivo**: si todos los spaxels tienen N_particles >> 30, M_A debe ser todo 1.
2. **S/N alto en todas partes**: M_B todo 1.
3. **Spaxels aislados**: si hay un spaxel valid_AB rodeado de invalid, debe ser eliminado por criterio C.
4. **Closing**: huecos de 1 spaxel rodeados de valid deben rellenarse.

### Test de integración con piloto

- M_valid debe descartar ~15-20% de los spaxels que el pipeline v1 considera válidos (los periféricos débiles, problema 19.1).
- M_valid debe mantener la región central conectada principal.
- Visualizar overlay sobre el cubo IFU para inspección manual.

## Criterios de aceptación

- [ ] Tests unitarios pasan
- [ ] Sobre piloto: ~15-20% de spaxels v1-válidos pasan a inválidos
- [ ] Componente conexa principal mantenida
- [ ] Tiempo < 2s por orientación

## Notas

- El cubo IFU ya está descargado para todos los mocks MaNGIA — no hay overhead extra.
- Para galaxias en MaNGA real (inferencia), no tendremos `particle_count`. La M_valid en inferencia se computa solo con criterios B y C.
- Si quisiéramos en el futuro hacer M_valid suave en lugar de binaria (peso por confianza), este es el módulo donde añadiríamos esa extensión.
