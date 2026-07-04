# GalStructNet-S3

Segmentación estructural probabilística por spaxel (bulge · disk · bar · arm ·
other) de cubos IFU MaNGA/MaNGIA, entrenada con etiquetas suaves derivadas de
TNG50 y con incertidumbre anclada en física medida. Núcleo: la
**Evidence-Propagating Network (EPN)** — dual-rail (señal, certeza), salida
Dirichlet con concentración supervisada por `α* = κ·N_eff·Y + 1`.

## Arranque

```bash
pip install -e ".[dev]"     # añade ".[mamba]" para el encoder espectral
pytest -q                   # los tests de referencia requieren torch;
                            # sin torch se saltan limpiamente
```

## Estructura

```
specs/        fuente de verdad (leer MIGRATION.md y 00_ROADMAP.md primero)
src/galstructnet_s3/
  data/       dataset v3 (señal, certeza) · collate size-agnostic    [Hito 1]
  models/     layers EPN · encoders · fusión · decoder · cabezas     [Hitos 3–4]
  losses/     KL Dirichlet anclada · dice · boundary · phys · total  [Hito 4]
  training/   trainer · currículo 4 etapas (Etapa 1 = MAE conjunto)  [Hitos 5–6]
  evaluation/ métricas · conformal · reporte de ablación             [Hito 7]
configs/      base + escalera de ablación A0–A4
tests/        conftest con entrada sintética del contrato v3 +
              tests ejecutables (referencia) + esqueletos numerados por spec
CLAUDE.md     guía operativa (reglas de oro, estado, orden de trabajo)
```

## Estado

Hito 0 (setup) completo. Implementaciones de referencia con tests: KL
Dirichlet anclada, descomposición TU/AU/EU, α-convolución PSF, L_phys,
L_boundary, scoring suaves, contrato de certeza/padding, cabeza MAE de la
Etapa 1. El resto: stubs con contrato + spec. Siguiente: **Hito 1**
(`specs/10_dataset.md`) + **Cambio G** en `galstructnet_labels`.
