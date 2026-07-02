# ApertureNet-S3 — Pipeline v2 de generación de etiquetas

> Este es el documento maestro que Claude Code debe leer **primero** antes de implementar nada.

## Qué es este proyecto

Este pipeline genera **etiquetas estructurales por spaxel** (bulbo / disco / barra / brazo / otro) para entrenar **ApertureNet-S3**, una red de segmentación que opera sobre cubos IFU (MaNGIA en entrenamiento, MaNGA en inferencia).

Las etiquetas se construyen desde la **verdad física de la simulación TNG50**, proyectada al plano observacional del mock MaNGIA.

## Por qué un pipeline v2 (y no parches al actual)

El pipeline actual ya funciona y produce `soft_mass`, `soft_light`, `hard_mass`, `hard_light`. Pero tiene seis problemas reportados (ver `docs/01_diagnostico.md`) cuya causa raíz común es: **mezcla en un solo flujo cuatro decisiones que deberían estar separadas** (física, calibración, geometría, observacional).

El pipeline v2 separa esas decisiones en módulos independientes con contratos explícitos. Esto:
- elimina los seis problemas de raíz, no por parches
- permite reusar la fase física entre las 4 orientaciones por galaxia (~2.6× más rápido)
- permite validación módulo por módulo
- permite iterar sobre un módulo sin re-ejecutar todo

## Arquitectura en una imagen

```
FASE A (FÍSICA, una vez por galaxia)
  TNG cutout → Extractor → Classifier → BarDetector + ArmDetector
                                                ↓
                                      particle_labels_final.h5

FASE B (RENDERIZADO, una vez por orientación)
  particle_labels_final + θ,φ → Projector → Aggregator → MaskBuilder
                                                ↓
                                Y_int_mass, Y_int_light, M_valid

FASE C (VALIDACIÓN Y EMPAQUETADO)
  todo lo anterior → QualityCheck → Packer → dataset_entry.h5
```

## Estructura del repo a crear

```
aperturenet_labels_v2/
├── CLAUDE.md                       # este archivo
├── MIGRATION.md                    # frontera portar/reimplementar vs v1
├── README.md                       # uso y quickstart
├── pyproject.toml                  # dependencias
├── docs/
│   ├── 01_diagnostico.md           # por qué v2
│   ├── 02_principios.md            # los 5 principios de diseño
│   ├── 03_arquitectura.md          # módulos, fases, diagrama
│   ├── 04_contratos.md             # esquemas HDF5 de productos
│   └── 05_validacion.md            # cómo validar cada módulo
├── specs/
│   ├── 00_ROADMAP.md               # orden de implementación
│   ├── 10_extractor.md             # spec módulo Extractor
│   ├── 11_classifier.md            # spec módulo Classifier
│   ├── 12_bar_detector.md          # spec BarDetector
│   ├── 13_arm_detector.md          # spec ArmDetector
│   ├── 20_label_projection.md      # spec proyección + agregación + N_eff (reemplaza 20_projector y 21_aggregator)
│   ├── 22_mask_builder.md          # spec MaskBuilder
│   └── 30_quality_check.md         # spec QualityCheck + empaquetado final (incluye al Packer)
├── src/aperturenet_labels/
│   ├── __init__.py
│   ├── io/                         # readers TNG y MaNGIA
│   ├── phase_a/                    # extractor, classifier, detectors
│   ├── phase_b/                    # label_projection (proyección+agregación), mask_builder
│   ├── phase_c/                    # quality_check + empaquetado final
│   ├── cli/                        # comandos de línea
│   └── schemas/                    # validación de productos intermedios
├── tests/
│   ├── unit/                       # un test por módulo
│   ├── integration/                # pipeline end-to-end mini
│   └── data/                       # fixtures pequeñas
├── notebooks/
│   ├── 00_smoke_test.ipynb         # verificación rápida con caso piloto
│   ├── 01_compare_with_v1.ipynb    # comparación visual con pipeline actual
│   └── 02_ablations.ipynb          # estudios de sensibilidad
└── configs/
    ├── default.yaml                # configuración base
    └── pilot.yaml                  # config para galaxia piloto
```

## Cómo usar este plan

1. **Lee `MIGRATION.md`** — define qué módulos se portan del repo v1 (`structural_labeling/`) y cuáles se reimplementan. No copies lógica de clases del v1.
2. **Sigue con `specs/00_ROADMAP.md`** — define el orden de implementación
2. **Para cada módulo, lee su spec en `specs/`** antes de codificar
3. **Implementa el módulo siguiendo la spec** y crea sus tests
4. **Valida con la galaxia piloto** TNG50-87-141934-0-127 antes de avanzar

Cada spec sigue el formato: contrato de entrada, contrato de salida, algoritmo, validación, criterios de aceptación.

## Caso piloto (siempre disponible)

```
plate-ifu: TNG50-87-141934-0-127
snapshot:  87
subhalo:   141934
view:      0
ifu:       127
shape:     (6603, 69, 69)
```

Cada módulo tiene un test de smoke que se corre contra esta galaxia y verifica que las salidas son razonables.

## Stack técnico

- Python 3.11
- numpy, scipy, scikit-learn, h5py, pyyaml
- astropy (FITS, WCS, unidades)
- pydantic (validación de configuraciones y contratos)
- typer (CLI)
- pytest + pytest-cov (testing)
- jupyterlab (notebooks de inspección)

NO usar PyTorch ni TensorFlow en este pipeline — esto es generación de datos, no entrenamiento. La parte de entrenamiento de ApertureNet-S3 es un repo separado.

## Principios de código

- **Funciones puras siempre que sea posible**. Cada módulo es esencialmente `f(input_h5, config) -> output_h5`.
- **Contratos explícitos** vía pydantic models. Si la entrada no cumple el schema, fallar con mensaje claro.
- **Productos intermedios persistentes** en HDF5. Cada paso guarda su salida.
- **Logging estructurado** (`structlog`). Cada paso loguea entrada, salida, métricas clave.
- **Reproducibilidad**: todos los stochastics usan seed configurable; default 42.
- **No optimizar prematuramente**: claridad > velocidad. Si un módulo es lento, paralelizar al final.

## Antes de hacer commit

- `pytest tests/unit/<modulo>` debe pasar
- `pytest tests/integration` debe pasar al final de cada fase
- `ruff check src/` sin errores
- `mypy src/` sin errores nuevos
- Notebook `00_smoke_test.ipynb` corre end-to-end sin error sobre el piloto

## Lo que NO está en scope

- Entrenamiento de ApertureNet-S3 (otro repo)
- El visor web (otro repo)
- Domain adaptation MaNGIA→MaNGA (otro repo)
- pyPipe3D (es upstream)

## Referencias clave

- Notas del director: `docs/notas_director/` (NOTA_DIRECTOR_*.md)
- Análisis de diseño: `docs/analisis_etiquetas_aperturenet.docx`
- Estado del arte: `docs/estado_del_arte_computacional.docx`
