# aperturenet_labels — pipeline v2

Generación de etiquetas estructurales por spaxel para el entrenamiento de **ApertureNet-S3** sobre cubos IFU MaNGIA.

**Estado (2026-06-10)**: Hitos 0–4 implementados y validados end-to-end sobre
los pilotos `TNG50-87-155298-0-127` y `TNG50-87-192324-0-127` (ver
`docs/adrs/ADR-001` para las decisiones y desviaciones documentadas).

## Quickstart

```bash
cd soft_labels_generation_v2_2
pip install -e ".[dev]"

# 1. Verificar instalación (33 tests unitarios)
pytest tests/unit -q

# 2. Correr sobre las 2 galaxias piloto (datos en ../data)
python -m aperturenet_labels.cli.main run --pilot

# 3. Tests de integración (alineación pyPipe3D + roundtrip)
pytest tests/integration -q
```

Productos: `../data/intermediate/phase_a/<galaxia>/` (features y etiquetas
por partícula), `../data/intermediate/phase_b/<galaxia>/` (proyecciones y
máscaras), `../data/output/dataset_entries/<galaxia>_v0.h5` (entrada final
del dataloader, 74×74 con padding centrado) y `../data/output/qa_reports/`.

## Documentación

- **Empezar por aquí**: [`CLAUDE.md`](./CLAUDE.md) — visión general del proyecto
- **Plan de implementación**: [`specs/00_ROADMAP.md`](./specs/00_ROADMAP.md)
- **Diseño detallado**: [`docs/`](./docs/)
- **Specs por módulo**: [`specs/`](./specs/)

## Comandos CLI principales

```bash
# Pipeline completo sobre una galaxia
aperturenet-labels run --galaxy-id TNG50-87-141934 --view 0

# Solo Fase A (física, una vez por galaxia)
aperturenet-labels phase-a --galaxy-id TNG50-87-141934

# Solo Fase B sobre Fase A ya computada (rápido, una vez por orientación)
aperturenet-labels phase-b --galaxy-id TNG50-87-141934 --view 0

# Procesar toda la muestra
aperturenet-labels batch --config configs/full_sample.yaml --workers 32

# Generar reporte de QA agregado sobre toda la muestra
aperturenet-labels qa-summary --output reports/qa_summary.html
```

## Para Claude Code

Este repo está diseñado para ser desarrollado iterativamente por Claude Code siguiendo el roadmap. La estrategia es:

1. **Hito 0**: setup del repo
2. **Hito 1**: skeleton end-to-end con módulos triviales
3. **Hito 2-4**: implementación real módulo por módulo
4. **Hito 5**: validación cruzada con pipeline v1
5. **Hito 6**: escalado a la muestra completa

Cada hito tiene criterios de aceptación claros en `specs/00_ROADMAP.md`.

## Caso piloto siempre disponible

```
TNG50-87-141934-0-127
```

Cualquier cambio se prueba primero aquí. Los datos del piloto están en `data/pilot/`.
