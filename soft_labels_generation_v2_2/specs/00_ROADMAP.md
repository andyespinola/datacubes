# Roadmap de implementación

> Léeme **después** de `CLAUDE.md` y **antes** de empezar a codificar.

## Estrategia general

Implementación **vertical antes que horizontal**: primero hacer un end-to-end mínimo que produzca *cualquier* salida sobre el piloto, después refinar cada módulo. Esto evita pasar tres semanas perfeccionando el Extractor sin saber si el resto del pipeline va a funcionar.

## Hitos

### Hito 0 — Setup del repo (medio día)

- Crear estructura de directorios listada en CLAUDE.md
- `pyproject.toml` con dependencias
- `pytest.ini`, `ruff.toml`, `mypy.ini`
- CI mínimo (GitHub Actions o equivalente): lint + test
- README con quickstart
- Verificar que `pip install -e .` funciona

**Criterio de aceptación**: `pytest` corre (aunque sin tests) sin errores.

### Hito 1 — Skeleton end-to-end (1 semana)

Implementar **versiones triviales** de los 9 módulos que producen output del shape correcto pero no necesariamente correcto físicamente. El objetivo es validar el flujo de datos completo y los contratos entre módulos.

Orden:

1. `io/tng_reader.py` — leer cutout del piloto (sin procesar nada)
2. `io/mangia_reader.py` — leer metadata del cubo del piloto
3. `phase_a/extractor.py` — devolver features dummy
4. `phase_a/classifier.py` — devolver P_class uniforme
5. `phase_a/bar_detector.py` — devolver P_bar = 0
6. `phase_a/arm_detector.py` — devolver P_arm = 0
7. `phase_b/label_projection.py` — rotación trivial + binning a la grilla del cubo con datos uniformes (spec 20 consolidado: proyección + agregación + N_eff)
8. `phase_b/mask_builder.py` — máscara = 1 en todas partes
9. `phase_c/quality_check.py` — devolver "todo OK"
10. `phase_c/packer.py` — empaquetar a HDF5 (contrato en spec 30)
12. `cli/run.py` — comando `aperturenet-labels run --pilot` que ejecuta todo

**Criterio de aceptación**: `aperturenet-labels run --pilot` produce un `dataset_entry.h5` con el shape correcto, en menos de 30 segundos. Aunque las etiquetas no tengan sentido físico aún.

### Hito 2 — Fase A real (2 semanas)

Reemplazar las versiones triviales de la Fase A por las implementaciones reales según las specs:

1. `extractor.py` — siguiendo `specs/10_extractor.md`
2. `classifier.py` — siguiendo `specs/11_classifier.md` (GMM + prior catálogo)
3. `bar_detector.py` — siguiendo `specs/12_bar_detector.md` (Fourier m=2)
4. `arm_detector.py` — siguiendo `specs/13_arm_detector.md` (residuales)

Validar contra el caso piloto comparando con el catálogo TNG oficial. La fracción global bulbo/disco/otro debe estar dentro del ±10% del catálogo.

**Criterio de aceptación**:
- ε computado coincide con catálogo TNG dentro de 0.05 RMS
- BIC del GMM finito y razonable
- Si el catálogo dice que el piloto tiene barra: BarDetector la detecta con A2 > 0.3
- Tests unitarios pasan
- Tiempo de Fase A: < 60s por galaxia

### Hito 3 — Fase B real (1.5 semanas)

Reemplazar las versiones triviales de la Fase B:

1. `label_projection.py` — siguiendo `specs/20_label_projection.md` (rotación a la vista MaNGIA + binning CIC + agregación ponderada + N_eff + 4 variantes)
2. `mask_builder.py` — siguiendo `specs/22_mask_builder.md` (criterios A+B+C)

**Criterio de aceptación**:
- Conservación de masa: `Σ Y_int_mass(i,j,c) ≈ M_*_total` dentro del 1%
- M_valid descarta ~15-20% de spaxels que el pipeline actual considera válidos (los periféricos)
- Tiempo de Fase B: < 10s por orientación

### Hito 4 — Fase C real (3 días)

1. `quality_check.py` — siguiendo `specs/30_quality_check.md`
2. `packer.py` — contrato de empaquetado incluido en `specs/30_quality_check.md` (HDF5 final con metadata)

**Criterio de aceptación**:
- `qa_report.json` contiene todas las métricas listadas en spec
- `dataset_entry.h5` se carga correctamente con la API de lectura
- Tamaño del archivo razonable (<100 MB por galaxia × orientación)

### Hito 5 — Validación cruzada y comparación con v1 (1 semana)

- Notebook `01_compare_with_v1.ipynb`: comparar lado a lado las etiquetas v1 vs v2 sobre el piloto
- Validar contra Galaxy Zoo 3D (si hay match con MaNGA real para esta galaxia)
- Documentar diferencias y justificar las correcciones

**Criterio de aceptación**: documento de comparación con figuras y métricas, listo para discusión con el director.

### Hito 6 — Escalado a la muestra completa (1 semana)

- Paralelización (joblib o dask)
- Manejo de errores robusto (galaxia que falla no detiene el resto)
- Logging estructurado para correr en lote
- Generación del dataset completo: ~10K galaxias × 4 orientaciones = 40K ejemplos

**Criterio de aceptación**:
- Procesa toda la muestra MaNGIA en < 1 semana en una máquina de 32 cores
- Tasa de error < 1% (galaxias que fallan)
- Métricas agregadas razonables sobre toda la muestra

## Tabla resumen

| Hito | Descripción | Duración | Acumulado |
|------|-------------|----------|-----------|
| 0 | Setup del repo | 0.5d | 0.5d |
| 1 | Skeleton end-to-end | 1 sem | 1.5 sem |
| 2 | Fase A real | 2 sem | 3.5 sem |
| 3 | Fase B real | 1.5 sem | 5 sem |
| 4 | Fase C real | 3 días | 5.5 sem |
| 5 | Validación cruzada | 1 sem | 6.5 sem |
| 6 | Escalado | 1 sem | 7.5 sem |

**Total estimado**: 7-8 semanas para tener el dataset v2 completo y validado.

## Reglas de oro

1. **No avances al siguiente hito sin que el anterior pase sus tests**.
2. **Si un módulo se complica, simplifica el alcance de ese módulo, no el de los siguientes**.
3. **El piloto es la fuente de verdad para validación rápida**. Cualquier cambio se prueba primero ahí.
4. **Cada módulo debe correr en aislamiento**: si quiero probar solo BarDetector con la salida del Classifier persistida, debe funcionar sin el resto del pipeline.
5. **Documenta decisiones de diseño que no estén en las specs**: si descubres algo que cambia el plan, escribe un ADR (Architecture Decision Record) en `docs/adrs/`.

## Entregables al final

- Repo con código + tests + CI
- Dataset MaNGIA completo en formato v2 (~40K HDF5 entries)
- Documento de validación (notebook 01) listo para director
- README con quickstart para que cualquiera lo use en otra muestra

## Riesgos y mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|--------------|------------|
| Cálculo de j_c(E) inestable o lento | Media | Usar catálogo TNG oficial cuando esté disponible; fallback a octree local |
| GMM no converge en algunas galaxias | Media | Detección + fallback a clasificación por umbrales puros |
| Detección de brazos espuria en galaxias sin brazos | Alta | Criterio de coherencia azimutal estricto; validar contra catálogo morfológico |
| Tiempo de Fase A escala mal | Baja | Profiling al final del Hito 2; paralelizar si es necesario |
| Algún cubo MaNGIA tiene metadata corrupta | Alta | Reader robusto que skipa galaxias problemáticas con log |
