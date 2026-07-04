# MIGRATION — Specs `03_model/` v2 → v3

> **Renombre (2026-07-01):** el sistema se llama oficialmente **GalStructNet-S3** (antes ApertureNet-S3 en borradores). El nombre *ApertureNet* queda **reservado** para un proyecto futuro e independiente de imagen pura (sin IFU/espectroscopía) y no debe usarse en ningún material de esta tesis. Renombre aplicado en todo el paquete: repos `galstructnet_s3` / `galstructnet_labels`, clases `GalStructNetLossV3`, `GalStructDataset`.


> Léeme **antes** que el ROADMAP. Tabla módulo-por-módulo de qué se conserva,
> qué cambia y qué se descarta, con la trazabilidad a los hallazgos de la
> revisión de arquitectura (2026-06-12). Las referencias `Cx` y `§x` son a ese
> documento.

## Resumen de la transformación

v3 conserva el **backbone** (3 encoders + decoder + módulo PSF) y reescribe
**cabezas, pérdidas y fusión** para (a) corregir errores de la revisión
(C1–C9), (b) instalar la innovación doctoral — supervisión Dirichlet anclada
en `N_eff` + fusión modulada por precisión + capas EPN (spec nuevo 45) +
conformal, y (c) hacer el modelo size-agnostic y honesto en sus ablaciones.

## Tabla módulo por módulo

| Spec v2 | Estado v3 | Qué cambia | Origen |
|---|---|---|---|
| `00_ROADMAP` | Reescrito | Hito EPN; escalera A0–A4 como entregable; Etapa 1 conjunta; ~13 sem | §3, §4 |
| `10_dataset` | **Reescrito** | Pares (señal, certeza); 4 variantes `Y_*` + `N_eff_{raw,psf}`; h3/h4+err; target dual; size-agnostic; rest-frame; atol 1e-2 | C4, C5, C7, C9, §2.2 |
| `20_encoder_spectral` | Parche | Ablación Mamba vs Conv1D obligatoria; `return_sequence` (no pool en SSL); cache de embeddings; certeza colapsada a c̄_spec | §2.3 |
| `21_encoder_spatial` | Parche | Contrato de skips a 256; quitar test de equivariancia; `mult` dinámico | C8, C9, C5 |
| `22_encoder_physical` | **Reescrito** | 8 canales (+h3/h4); `PhysicalEncoderN` (NormConv) default; variante de propagación de momentos | C9, §2.2, 45 |
| `23_fusion` | **Reescrito** | Fusión por posición modulada por precisión (primaria); global → ablación A2 | §2.1, 45 |
| `30_decoder_unet` | **Reescrito** | Una sola implementación (sin la confusión narrada v2); tamaños dinámicos; variante NormConv | C9, C5, 45 |
| `40_head_segmentation` | **Reescrito** | Dos cabezas (masa/luz); `EvidenceHead` con techo de certeza; semántica de S | C3, §3.1, 45 |
| `41_head_boundary` | Parche | Reframing como regularizador; opera sobre prob_lum | §2.5 |
| `42_head_uncertainty` | **Reescrito** | Descomposición correcta por información mutua (TU/AU/EU); fórmula v2 era incorrecta | C1 |
| `43_module_psf` | **Reescrito** | Resolución de etiquetas (Cambio F anulado); modo α-convolución default | C4, §2.4, 45 |
| `45_evidence_layers` | **NUEVO** | Familia EPN: NormConv, EvidenceConv, PrecisionGatedFusion, EvidenceHead; invariantes y escalera | §3, §2.1, §2.4 |
| `50_loss` | **Reescrito** | `L_seg` = KL Dirichlet anclada en N_eff; `L_unc` eliminada; `L_phys` por masa; dual target; `L_PSF` en evidencia | C2, C3, C7, §3.1 |
| `60_training` | **Reescrito** | Etapa 1 MAE correcto + MaNGIA∪MaNGA; `L_weak` GZ3D particionado; escalera A0–A4 | C6, §4 |
| `70_evaluation` | **Reescrito** | ρ(S,N_eff); AUROC(EU); conformal; scoring rules suaves; partición GZ3D; tabla A0–A4 | §3, §4, C1, C7 |

**Reescritos** (8): 10, 22, 23, 30, 40, 42, 43, 50 + 00, 60, 70. **Parches**
(3): 20, 21, 41. **Nuevo** (1): 45.

## Cambios requeridos en el repo `galstructnet_labels` (Cambio G)

El modelo v3 pide al pipeline de etiquetas dos cosas que v2.1 casi tiene:

1. **`N_eff` con ambas ponderaciones.** Hoy el proyector (spec 20 labels,
   Paso 5) calcula Kish solo con peso de masa. Añadir el cálculo con peso de
   luminosidad (`N_eff_lum`). Es el mismo bucle con `w_lum`.
2. **Variante PSF de `N_eff`.** `N_eff_psf` por convolución del numerador y
   denominador de Kish con la PSF (o aproximación documentada: convolucionar
   `N_eff` directamente). Para ambas ponderaciones.
3. **Empaquetar h3/h4 (+err)** desde el spec 26 (inputs) al HDF5 de
   `dataset_entry` — ya se producen, solo se transportan.

Nada de esto toca la metodología GMM/MORDOR ya validada en v2.1; son tres
campos adicionales en el NPZ/HDF5 de salida. El "Cambio F" propuesto en el
spec 43 v2 (computar `Y_obs` on-the-fly) **se anula** (C4): las variantes
`Y_*_psf` ya existen.

## Orden de implementación (por hitos, ver ROADMAP)

1. **Hito 1** — `10_dataset` v3 + Cambio G en labels (en paralelo). Sin
   dataset no hay nada.
2. **Hito 2** — skeleton trivial (todos los módulos con `Conv2d(1,1,1)`),
   end-to-end sobre el piloto.
3. **Hito 3** — versión **estándar A0** de encoders (20, 21, 22-Std) + fusión
   (23-global) + decoder (30-std). El baseline existe antes que la familia.
4. **Hito 4** — cabezas (40 dual, 42 MI, 43) + pérdida anclada (50) +
   **primitivas EPN (45)** + configs A0–A4.
5. **Hitos 5–6** — training + etapas + escalera de ablación.
6. **Hito 7** — evaluación (70) con métricas firma + conformal + GZ3D.

## Reglas de oro nuevas (v3)

- Las innovaciones (anclaje, fusión por precisión, capas EPN, conformal) se
  **ganan su lugar** en la escalera A0–A4; ninguna se asume.
- Cero tamaños espaciales hardcodeados en código de modelo.
- La KL Dirichlet se computa en float32 aunque el resto sea bf16.
- GZ3D particionado weak/val con assert en train y en eval.

## Qué NO cambió (y por qué)

- **Mamba espectral**: se conserva como ingeniería (O(L) con campo receptivo
  global); su ablación vs Conv1D es obligatoria pero el default no cambia.
- **Swin+FPN, U-Net**: estándar, correctos, conservados.
- **Currículo de 4 etapas, MMD/DANN/consistency, mixed precision, clip 1.0,
  wandb**: estructura v2 intacta; solo se corrige la Etapa 1 y se añade
  `L_weak`.
- **Módulo PSF como diferenciador**: se conserva y se mejora (α-conv), no se
  reemplaza.
