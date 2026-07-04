# 00 — Roadmap de implementación (v3)

> Léeme **después** de `MIGRATION.md` y **antes** de empezar a codificar.
> Cambios v2→v3: hito de capas EPN, escalera de ablación como entregable de
> primera clase, Etapa 1 conjunta MaNGIA+MaNGA.

## Estrategia general

**Vertical antes que horizontal**, igual que v2: primero un end-to-end mínimo
con módulos triviales, después los reales uno por uno. Con un añadido v3:
**la versión estándar (A0) se implementa antes que la familia EPN**, porque
sin baseline no hay ablación y sin ablación las innovaciones no se defienden.

## Hitos

| Hito | Descripción | Duración | Archivos clave |
|------|-------------|----------|----------------|
| 0 | Setup del repo | 0.5 día | `pyproject.toml`, estructura |
| 1 | Dataset v3 y DataLoader (con certezas, 4 variantes, size-agnostic) | 1.5 sem | [10_dataset.md](10_dataset.md) |
| 2 | Skeleton end-to-end con módulos triviales | 1 sem | `models/*.py` con `nn.Conv2d(1,1,1)` |
| 3 | Encoders + decoder reales, versión estándar (A0) | 2 sem | [20–23, 30](20_encoder_spectral.md) |
| 4 | Cabezas + pérdida anclada (50 v3) + primitivas EPN (45) + configs de la escalera A0–A4 | 2 sem | [40–43, 45, 50](45_evidence_layers.md) |
| 5 | Training loop + Etapa 1 (SSL conjunto MaNGIA+MaNGA) | 1 sem | [60_training.md](60_training.md) |
| 6 | Etapas 2–4 + escalera de ablación A0–A4 | 3.5 sem | continuación de 60 |
| 7 | Evaluación: métricas, S↔N_eff, conformal, GZ3D | 1.5 sem | [70_evaluation.md](70_evaluation.md) |

**Total estimado**: ~13 semanas hasta modelo entrenado, ablado y validado.

## Reglas de oro

1. **No avanzar al siguiente hito sin que el anterior pase sus tests.**
2. **Si un módulo se complica, simplifica el alcance de ese módulo, no el de
   los siguientes.**
3. **El piloto (`TNG50-87-141934-0-127`) es la fuente de verdad para
   validación rápida.** Cualquier cambio se prueba primero ahí.
4. **Cada módulo corre en aislamiento** con los shapes declarados en su spec.
5. **Documenta decisiones fuera de spec** como ADR en `docs/adrs/`.
6. **(Nueva, v3)** Toda primitiva EPN entra al modelo final solo si su nivel
   de la escalera de ablación lo justifica (criterios en
   [45_evidence_layers.md](45_evidence_layers.md)). Las innovaciones se ganan
   su lugar, no se asumen.
7. **(Nueva, v3)** Nada de tamaños espaciales hardcodeados (69, 72, 74) en
   código de modelo: todo deriva de `x.shape[-2:]`.

## Antes de hacer commit

- `pytest tests/unit/<modulo>` y, al cierre de cada hito, `tests/integration`.
- `ruff check src/`, `mypy src/` sin errores nuevos.
- Si el cambio afecta el comportamiento del modelo, una métrica de wandb debe
  reflejarlo.

## Lo que NO está en scope

- Generación de etiquetas (repo `galstructnet_labels`; v3 le pide el Cambio G,
  ver `MIGRATION.md`).
- Reducción de datos crudos del telescopio.
- Análisis científico downstream; despliegue web.
- Anexo A de 45 (noise-aware selective scan): experimental, fuera del camino
  crítico; solo si los Hitos 1–7 cierran con margen.

## Caso piloto siempre disponible

```
TNG50-87-141934-0-127
```

Un único `dataset_entry_*` v3 debe cargar, pasar por el modelo (A0 y A3) y
producir tensores válidos desde el Hito 2 (trivial) / Hito 4 (real).
