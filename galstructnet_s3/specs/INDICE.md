# GalStructNet-S3 — Especificaciones del modelo (v3)

> **Renombre (2026-07-01):** el sistema se llama oficialmente **GalStructNet-S3** (antes ApertureNet-S3 en borradores). El nombre *ApertureNet* queda **reservado** para un proyecto futuro e independiente de imagen pura (sin IFU/espectroscopía) y no debe usarse en ningún material de esta tesis. Renombre aplicado en todo el paquete: repos `galstructnet_s3` / `galstructnet_labels`, clases `GalStructNetLossV3`, `GalStructDataset`.


> Paquete de specs de la carpeta `03_model/`, revisado tras la revisión de
> arquitectura 2026-06-12. Segmentación estructural probabilística por spaxel
> en cubos IFU MaNGA/MaNGIA, con incertidumbre como objetivo de primera clase.
>
> **Este paquete reescribe `03_model/`.** Las carpetas `01_label_generation`,
> `02_input_generation` y `04_validation` del paquete completo siguen vigentes,
> con el único cambio del **Cambio G** descrito en `MIGRATION.md` (N_eff por
> ambas ponderaciones + variante PSF + empaquetado de h3/h4).

---

## Orden de lectura

1. `MIGRATION.md` — qué cambió respecto a v2 y por qué (tabla módulo a módulo).
2. `00_ROADMAP.md` — plan de hitos (~13 semanas) y escalera de ablación.
3. `45_evidence_layers.md` — la familia de capas que define la red (innovación).
4. El resto en orden numérico.

---

## Contenido

| Spec | Estado | Tema |
|---|---|---|
| `MIGRATION.md` | nuevo | v2→v3, tabla de cambios, Cambio G en labels |
| `00_ROADMAP.md` | reescrito | Hitos, escalera A0–A4, Etapa 1 conjunta |
| `10_dataset.md` | reescrito | (señal, certeza); 4 variantes + N_eff; size-agnostic |
| `20_encoder_spectral.md` | parche | Mamba (ablación obligatoria; no-pool en SSL) |
| `21_encoder_spatial.md` | parche | Swin+FPN (skips 256; mult dinámico) |
| `22_encoder_physical.md` | reescrito | 8 canales; NormConv; momentos |
| `23_fusion.md` | reescrito | Fusión por posición modulada por precisión |
| `30_decoder_unet.md` | reescrito | FPN top-down; tamaños dinámicos; NormConv |
| `40_head_segmentation.md` | reescrito | Dirichlet dual; EvidenceHead |
| `41_head_boundary.md` | parche | Regularizador de fronteras (reframing) |
| `42_head_uncertainty.md` | reescrito | Descomposición correcta TU/AU/EU |
| `43_module_psf.md` | reescrito | PSF-aware; α-convolución |
| `45_evidence_layers.md` | **nuevo** | Red de Propagación de Evidencia (EPN) |
| `50_loss.md` | reescrito | KL Dirichlet anclada en N_eff; sin L_unc |
| `60_training.md` | reescrito | Currículo; SSL conjunto; L_weak; ablación |
| `70_evaluation.md` | reescrito | ρ(S,N_eff); conformal; scoring suaves; GZ3D |

---

## Las tres innovaciones (resumen)

1. **Targets Dirichlet anclados en N_eff** (45 §P4, 50, 70): la concentración
   de salida se supervisa contra el conteo efectivo de partículas (Kish) que
   generó cada etiqueta. Responde por construcción la crítica 2022–2024 a EDL
   (existe distribución de referencia frecuentista); extiende LUPI a "precisión
   de etiqueta privilegiada".
2. **Fusión multimodal modulada por precisión instrumental** (45 §P3, 23): el
   gating por modalidad se sesga con la precisión medida (IVAR, errores
   pyPipe3D), cerrando el lazo distribución-entra → distribución-sale.
3. **Conformal sobre el simplex** (70): conjuntos de predicción por spaxel con
   cobertura garantizada, Mondrian por clase y N_eff/radio.

Todas envueltas en la familia de capas EPN (spec 45) y validadas por la
escalera de ablación A0–A4 (60, 70).

---

## Repositorios

| Carpeta | Repositorio |
|---|---|
| `03_model/` (este paquete) | `galstructnet_s3` |
| `01/02/04` + Cambio G | `galstructnet_labels` |

## Caso piloto

`TNG50-87-141934-0-127` — debe cargar y pasar por el modelo (A0 y A3)
produciendo tensores válidos desde el Hito 2.
