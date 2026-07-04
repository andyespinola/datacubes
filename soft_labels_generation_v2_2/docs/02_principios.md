# Cinco principios del rediseño

## P1 — Separación física vs renderizado

La decisión "qué componente estructural es esta partícula" es una propiedad **intrínseca** de la partícula, dependiente solo de su dinámica 3D. No cambia con el ángulo de vista, la PSF, el tamaño del spaxel ni la máscara de calidad.

La decisión "cómo aparece esta partícula en una vista observacional" es una propiedad **extrínseca**, dependiente de todos esos parámetros instrumentales.

El v1 entrelaza ambas. El v2 las separa en módulos distintos:
- **Classifier** (3D, una vez por galaxia)
- **Renderer = Projector + Aggregator** (2D, una vez por orientación)

**Consecuencia práctica**: las 4 orientaciones de una galaxia comparten la decisión física. Si más adelante queremos añadir 8 orientaciones por galaxia (data augmentation), la Fase A no se vuelve a correr.

## P2 — Probabilidad desde el origen

El v1 toma decisiones duras tempranas (`if epsilon > threshold`) y luego intenta suavizarlas. El v2 produce **probabilidades por partícula desde el primer paso**, mediante GMM sobre features cinemáticas + estructurales.

**Consecuencia práctica**:
- Las decisiones duras solo aparecen al final, opcionalmente, para visualización.
- La incertidumbre es una propiedad inherente del producto, no un post-procesamiento.
- Las clases se pueden mezclar: una partícula puede tener `P_disk = 0.6, P_bulge = 0.3, P_halo = 0.1`. Esto refleja la realidad física mejor que asignar una clase única.

## P3 — El catálogo es prior, no constraint

El catálogo morfológico oficial de TNG (Rodriguez-Gomez et al. 2022) reporta fracciones globales por galaxia: `bulge_fraction`, `disk_fraction`, `other_fraction`.

En v1, esas fracciones se imponen como **constraint duro** mediante reescalado iterativo. Esto causa el problema 19.3 (centro como disco): la fracción global de disco se "distribuye" sobre todas las partículas, incluyendo las centrales que cinemáticamente son bulbo.

En v2, el catálogo entra como **prior bayesiano** en la inicialización del GMM:

```python
weights_init = α * catalog_priors + (1 - α) * data_driven_init
```

con `α ∈ [0, 1]` configurable. El GMM puede ajustarse a la evidencia local cuando es fuerte, sin estar atado a las fracciones globales.

**Consecuencia práctica**: si una galaxia tiene un núcleo más bulboso de lo que el catálogo global sugiere, el GMM lo captura. Las fracciones recuperadas pueden discrepar del catálogo dentro del ±10% sin que sea un problema.

## P4 — Validación independiente por etapa

Cada módulo del pipeline debe poder **validarse de forma aislada** con métricas claras:

| Módulo | Validación independiente |
|--------|--------------------------|
| Extractor | ε computado vs ε del catálogo TNG (RMSE < 0.05) |
| Classifier | Fracciones recuperadas vs catálogo (±10%); BIC del GMM |
| BarDetector | A2 detectado vs A2 del catálogo (±20%) |
| ArmDetector | Inspección visual; consistencia con catálogo morfológico |
| Projector | Conservación de masa (±0.1%) |
| Aggregator | Suma a 1 por spaxel; conservación |
| MaskBuilder | ~15-20% spaxels v1-válidos descartados; componente conexa principal |

**Consecuencia práctica**: cuando algo se rompe, sabemos exactamente en qué módulo. La depuración deja de ser un misterio end-to-end.

## P5 — Decisiones reversibles

La arquitectura almacena **productos intermedios persistentes** que permiten revertir o re-ejecutar etapas tardías sin recomputar etapas tempranas:

```
particle_features.h5         ← producto del Extractor
particle_labels_initial.h5   ← producto del Classifier
particle_labels_with_bar.h5  ← producto del BarDetector
particle_labels_final.h5     ← producto del ArmDetector  ← FIN FASE A
projection_raw_v0.npz        ← producto del Projector (vista 0)
Y_int_v0.npz                 ← producto del Aggregator
M_valid_v0.npz               ← producto del MaskBuilder
dataset_entry_v0.h5          ← producto del Packer
```

**Consecuencias prácticas**:
- Para probar una nueva PSF: re-correr solo MaskBuilder + Packer.
- Para refinar la regla de detección de barras: re-correr BarDetector + ArmDetector + Phase B + Phase C, pero NO Extractor ni Classifier.
- Para experimentar con una nueva orientación: solo Phase B + Phase C.
- Para auditar: cualquier producto intermedio se puede inspeccionar directamente.

## Trade-offs aceptados

Estos principios tienen costos:

1. **Más espacio en disco**: cada producto intermedio se persiste. Estimación: ~50 MB por galaxia (Fase A) + ~30 MB por orientación (Fase B). Para 10K galaxias × 4 orientaciones: ~1.7 TB total. Aceptable con almacenamiento moderno.

2. **Más complejidad de orquestación**: hay 9 módulos en lugar de un script. Mitigado con el CLI (`aperturenet-labels run`) y configuración YAML.

3. **Curva de aprendizaje inicial**: contribuir requiere entender los contratos. Mitigado con docs/04_contratos.md y schemas pydantic con validación automática.

Los beneficios (validabilidad, iterabilidad, reproducibilidad) compensan ampliamente estos costos para un proyecto de tesis con horizonte de varios años.
