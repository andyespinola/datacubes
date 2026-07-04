# CLAUDE.md — Guía operativa del repo `galstructnet_s3`

## Qué es esto

Implementación de **GalStructNet-S3**: segmentación estructural probabilística
por spaxel (bulge/disk/bar/arm/other) de cubos IFU MaNGA/MaNGIA, con
incertidumbre anclada en física. Núcleo doctoral: la **Evidence-Propagating
Network (EPN)** — todo tensor es (señal, certeza); la concentración Dirichlet
de salida se supervisa contra `α* = κ·N_eff·Y + 1` (N_eff = conteo efectivo de
Kish del pipeline de etiquetas TNG50).

## Fuente de verdad

`specs/` manda sobre cualquier otra cosa, incluido este archivo. Orden de
lectura: `specs/MIGRATION.md` → `specs/00_ROADMAP.md` →
`specs/45_evidence_layers.md` → resto en orden numérico. Cada módulo de
`src/` cita su spec y su Hito en el docstring.

## Estado actual (Hito 0 completado)

Implementado como **referencia transcrita del spec** (con tests ejecutables):

| Módulo | Qué | Tests |
|---|---|---|
| `models/heads/uncertainty.py` | Descomposición TU/AU/EU (formas cerradas) | `test_uncertainty.py` (valores exactos) |
| `losses/dirichlet.py` | `build_anchor`, `dirichlet_kl`, `anchored_seg_loss` | `test_loss.py` 01–04 |
| `losses/physics.py` | L_phys ponderada por masa (C7) | `test_loss.py` 07 |
| `models/heads/psf.py::PSFEvidenceModule` | α-convolución PSF | `test_psf_module.py` |
| `models/heads/boundary.py` | B-map + L_boundary | `test_boundary_metrics_data.py` |
| `evaluation/metrics.py::soft_{brier,nll}` | scoring rules suaves | ídem |
| `data/dataset.py::to_certainty`, `data/collate.py::pad_to_multiple` | contrato de certeza/padding | ídem |
| `training/curriculum.py::SpectralMAEHead`, `masked_spectral_loss` | Etapa 1 corregida (C6) | pendiente |

Todo lo demás son **stubs** con firma del contrato + `NotImplementedError` +
referencia al spec. `tests/` contiene además los esqueletos numerados (skip)
que replican las listas de tests de cada spec.

## Orden de trabajo (no negociar)

1. **Hito 1**: `data/` completo contra `specs/10`. Usa la fixture
   `synthetic_entry` (tests/conftest.py) — codifica el contrato v3
   ejecutable. Dependencia externa: **Cambio G** en `galstructnet_labels`
   (N_eff por ambas ponderaciones + variante PSF + h3/h4 empaquetados) —
   ver `specs/MIGRATION.md`.
2. **Hito 2**: skeleton end-to-end trivial sobre la fixture.
3. **Hito 3**: encoders/fusión/decoder **estándar (A0)**. La familia EPN NO
   se implementa aún: sin baseline no hay ablación.
4. **Hito 4**: cabezas + pérdida anclada + primitivas EPN (`specs/45`) +
   configs `configs/ablation_epn/`.
5. **Hitos 5–7**: training (Etapa 1 conjunta MaNGIA∪MaNGA), escalera A0–A4,
   evaluación + conformal.

## Reglas de oro

- No avanzar de hito sin que los tests del anterior pasen.
- **Cero tamaños espaciales hardcodeados** (69/72/74): todo de
  `x.shape[-2:]`. Hay grep de CI.
- `dirichlet_kl` SIEMPRE en float32 (ya implementado con autocast off).
- GroupNorm, nunca BatchNorm (rompe la semántica de certeza).
- No existe `L_unc` ni `kl_to_uniform` — si aparece en un diff, es un error
  (specs/50, C2).
- GZ3D particionado weak/val disjunto con `assert` en train y eval.
- Toda primitiva EPN se adopta solo si su nivel A0–A4 lo justifica
  (`specs/45`, criterios).
- `pipe3d_maps` canales 6–7 = **h3/h4 de pPXF** (no de pyPipe3D) — la
  distinción es factual y aparece en docstrings/figuras.
- Piloto real: `TNG50-87-141934-0-127`. En CI, la fixture sintética.

## Nombres

El sistema es **GalStructNet-S3**. Nunca "ApertureNet" (reservado a un
proyecto futuro distinto). Clases: `GalStructDataset`, `GalStructNetLossV3`.

## Comandos

```bash
pip install -e ".[dev]"        # + ".[mamba]" cuando toque el encoder espectral
pytest -q                      # sin torch: tests de referencia se saltan solos
ruff check src tests && mypy src
grep -rnE "\b(69|72|74)\b" src/ && echo "REVISAR literales" || true
```
