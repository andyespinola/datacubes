# GalStructNet-S3 — Documento explicativo de la arquitectura

> Documento derivado de `specs/` (que **manda** sobre este archivo; ante
> discrepancia, gana el spec). Estado: Hitos 1–7 implementados y testeados
> contra la fixture sintética; entrenamiento pendiente de datos.
> Última actualización: 2026-07-03.

## 1. Qué hace el sistema

Segmentación estructural probabilística **por spaxel** de cubos IFU
MaNGA/MaNGIA en 5 clases (`bulge, disk, bar, arm, other`), con
incertidumbre anclada en física. Para cada spaxel el modelo no predice una
clase sino una **distribución de Dirichlet** Dir(α): la media α/S es la
segmentación probabilística y la concentración S = Σα es la confianza —
con unidades de pseudo-conteos comparables al conteo efectivo de
partículas (Kish) que generó la etiqueta en la simulación TNG50.

Las tres innovaciones (specs/INDICE):

1. **Anclaje Dirichlet en N_eff** — la concentración de salida se
   supervisa contra `α* = κ·N_eff·Y + 1`: existe una distribución de
   referencia frecuentista (responde la crítica 2022–2024 a EDL).
2. **Fusión multimodal modulada por precisión instrumental** — el peso de
   cada modalidad por spaxel se sesga con la precisión medida (IVAR,
   errores pyPipe3D).
3. **Conformal sobre el simplex** — conjuntos de predicción por spaxel con
   cobertura garantizada, Mondrian por clase × N_eff.

Todo envuelto en la familia de capas **EPN** (Evidence-Propagating
Network): cada tensor del modelo es un par (señal, certeza).

## 2. Convención (señal, certeza)

Toda activación espacial es `(x, c)` con `c ∈ [0, 1]` y dos garantías
duras (specs/45):

- **c = 0 ⟺ el valor se ignora exactamente** (sustituye al `nan_to_num`
  de v2: el valor bajo c=0 es irrelevante por construcción, no por
  saneo). Interpolar señal y certeza por separado viola esta garantía;
  todo upsampling de pares usa `interp(x·c)/interp(c)`.
- c decrece monótonamente con el error instrumental:
  `c = σ_ref² / (σ² + σ_ref²)`, con σ_ref = mediana del split de train.

Certeza por modalidad: `c_spec = snr/(snr+SNR_REF)` (escalar por spaxel),
`c_spat` del error fotométrico (o 1), `c_phys` de `pipe3d_err` (8 canales).
c=0 se fuerza fuera de `M_valid` y donde la señal original era NaN.

## 3. Flujo de datos

```
dataset_entry_{gid}_v{view}.h5
  cube (L,H,W) ──────────────► MambaSpectralEncoder ──► F_spec (256,H,W)
  + c_spec (1,H,W) ───────────────────────────────┐    [Conv1D dilatada =
  image (3,H,W) ────────────► SwinSpatialEncoder ─┼──► F_spat (256,H,W)
  + c_spat (3,H,W)                      + FPN     │    + skips ×3 (256)
  maps (8,H,W) ─────────────► PhysicalEncoder{Std,N} ► F_phys (64,H,W)
  + c_phys (8,H,W)                                │    + c_phys_out (1,H,W)
                                                  ▼
                    Fusion {concat | global | precision}
                    F_fused (384,H,W) + c_fused (1,H,W) + attn_w (3,H,W)
                                                  ▼
                    FPNDecoder{std,N} (+ skips, + c_skips)
                    hidden (256,H,W) + c_dec (1,H,W)
                                                  ▼
                    DualSegHeads {std | evidence}          ×2 targets
                    {alpha, prob, evidence, vacuity}_{mass,lum}
                                                  ▼
                    PSF {evidence: α-conv | prob: conv+renorm}
                    alpha_obs / prob_obs (plano observado)
```

Tamaños espaciales SIEMPRE dinámicos (`x.shape[-2:]`): los bundles MaNGA
van de 19 a 127 fibras. El encoder espacial declara `mult=32` y el dataset
paddea a múltiplo (certeza y máscara con 0/False: el padding se ignora por
construcción).

## 4. Módulos

| Módulo | Archivo | Qué hace |
|---|---|---|
| Encoder espectral | `models/encoders/spectral.py` | Cada spaxel es una secuencia de L≈6603 canales. Stem Conv1D (stride 2×2 → L/4, d_model=128) + 4 bloques **Mamba bidireccionales** (O(L), campo receptivo global). Pool + proyección a 256 en modo segmentación; `return_sequence=True` en Etapa 1 (sin pooling, C6). Contrafactual obligatorio: `DilatedConv1DEncoder` (ablación A8; también el camino CPU/CI). Sin mezcla espacial. |
| Encoder espacial | `models/encoders/spatial.py` | Swin-T (timm) + FPN top-down sobre la imagen g,r,i. Contexto morfológico global (única fuente de mezcla espacial de largo alcance). Devuelve features a resolución nativa + 3 skips a 256 canales para el decoder. Pesos ImageNet opcionales. |
| Encoder físico | `models/encoders/physical.py` | 8 mapas (v, σ, edad, Z, masa, Av de pyPipe3D + **h3/h4 de pPXF**). `Std` (A0): CNN 3×Conv-GN-GELU, anula señal con c=0. `N` (A1+): pila de NormConv2d — propaga (x, c). `MP` (momentos): experimento lateral. |
| Fusión | `models/fusion_global.py`, `fusion_precision.py` | `concat` (A2a), `global` = cross-attention v2 (A2b, baseline A0), **`precision` = PrecisionGatedFusion (A2c, default v3)**: por posición, M=3 tokens (uno por modalidad), atención con logits sesgados por g_m(log c̄_m), g_m init→0 (arranque neutral). Exporta `attn_w` — "en qué modalidad se apoyó el modelo". |
| Decoder | `models/decoder.py` | Refinador FPN top-down (no U-Net clásico: el bottleneck no está a resolución mínima). `std` y `normconv` (bloques NormConv sobre pares, upsampling normalizado por certeza). Emite `c_dec` para la cabeza. |
| Cabezas | `models/heads/segmentation.py` | **Duales** (masa/luz, C3): tronco compartido, proyecciones independientes. `std` (A0): α = 1 + softplus(proj). **`evidence` (P4)**: e = softplus(proj) ⊙ h(c_dec), h(u) = softplus(a·log u + b) — **techo de evidencia**: c_dec→0 ⇒ α→1 ⇒ ignorancia. (a, b) se persisten con nombre: son interpretables en la tesis. |
| PSF | `models/heads/psf.py` | `evidence` (default): α_obs = K_PSF ∗ (α−1) + 1 — la PSF redistribuye pseudo-conteos, prob_obs suma 1 automáticamente. `prob` (baseline A4): conv sobre prob + renormalización. |
| Incertidumbre | `models/heads/uncertainty.py` | Descomposición **TU/AU/EU por información mutua** (formas cerradas Dirichlet, C1). Vacuity K/S solo como diagnóstico. |

### Las 4 primitivas EPN (specs/45)

| | Primitiva | Invariante clave (testeado) |
|---|---|---|
| P1 | `NormConv2d` | agregación convexa ponderada por c + mezcla 1×1 con signo; ignorancia exacta; c_out convexa; equivariancia D4 |
| P2 | `EvidenceConv2d` | kernel estocástico ≥0 suma 1: conserva Σ evidencia en el interior; con kernel fijo = PSF reproduce la α-convolución |
| P3 | `PrecisionGatedFusion` | degradación selectiva (c̄_m→0 baja su peso); neutralidad inicial |
| P4 | `EvidenceHead` | S acotada por certeza; monotonía S(c); c=0 ⇒ vacuity 1 |

## 5. Pérdida (specs/50)

```
L_total = Σ_t∈{lum,mass} λ_t·[ L_seg(t) + 0.5·L_dice(t) + 0.4·L_PSF(t) ]
        + 0.3·L_boundary + 0.1·L_phys  (+ Etapa 3+: 0.2·L_consist + 0.2·L_weak)
                                        (+ Etapa 4: 0.5·L_MMD)
λ_lum = 1.0, λ_mass = 0.3
```

- **L_seg = KL(Dir(α*) ‖ Dir(α))** con α* = κ·N_eff·Y + 1 (forward por
  default; dirección y κ ∈ {0.25, 0.5, 1.0} se ablan). N_eff con cap p99.
  Siempre en float32 (autocast off local). Supervisa media Y
  concentración: sin `L_unc` ni ramping (eliminados, C2).
- L_dice: contrapeso del desbalance (barra/brazo ~5–10% de spaxels).
- L_boundary: regularizador de fronteras sobre prob_lum (τ=0.1).
- L_PSF: la misma KL anclada pero en el plano observado (α_obs vs
  Y_psf/N_eff_psf).
- L_phys: fracciones globales ponderadas POR MASA vs catálogo (C7).
- Máscaras: strict = M & ~M_unc_t (seg/dice/PSF); loose = M (resto).

Nota práctica (aprendida en el overfit del Hito 4): el gap inicial
S↔κ·N_eff genera un transitorio donde la evidencia colapsa antes de
crecer; el dial correcto es κ (y el init calibrado de b via
`scripts/init_evidence_scale.py`), no los pesos de los términos.

## 6. Currículo (specs/60)

| Etapa | Datos | Qué se entrena |
|---|---|---|
| 1 (50 ep) | MaNGIA ∪ MaNGA **sin etiquetas** | MAE espectral real (token de máscara en la entrada, reconstrucción por posición, sin pooling — C6). Cierra parte del gap de dominio desde el inicio. |
| 2 (100 ep) | MaNGIA (QA ok) | seg+dice (×2 targets) + phys. Éxito: IoU>0.5, **ρ(S,N_eff)>0.4**, ECE<0.15. |
| 3 (50 ep) | + MaNGA (GZ3D-weak + unlabeled) | + PSF + boundary + consist (KL simétrica bajo D4) + **L_weak** (BCE bar/arm vs voto GZ3D; partición weak/val disjunta con assert). |
| 4 (50 ep) | MaNGIA 50% + MaNGA 50% | + MMD sobre embeddings del decoder. Sanity: EU sube en MaNGA. |

bf16 + clip de gradiente 1.0 (imprescindible con Mamba). En la GPU local
(RTX 2060, Turing) no hay bf16: los tests corren fp32; bf16 queda para la
máquina de entrenamiento.

## 7. Escalera de ablación A0–A4

Las innovaciones **se ganan su lugar**; ninguna se asume. Un comando:
`python -m galstructnet_s3.cli ablate --ladder epn`.

| Nivel | Config | Pregunta |
|---|---|---|
| A0 | `A0_baseline` | baseline estándar (conv, attn global, EDL anclada) |
| A1 | `A1_normconv` | ¿la certeza en el tronco (P1) paga? |
| A2a/b/c | `A2*` | concat vs global vs precisión (P3) |
| A3 | `A3_evidence_head` | ¿el techo S≤f(c) (P4) mejora calibración/AUROC? |
| A4 | `A4_psf_{prob,evidence}` | ¿la α-convolución (P2) mejora nitidez? |

Criterio de adopción: A3 > A0 en soft-NLL **y** AUROC(EU); empatar en IoU
es aceptable (la familia compra incertidumbre, no accuracy). Dos
experimentos firma que una red estándar no puede hacer: barrido de ruido
en test actualizando c, y dropout de spaxels (c=0) — ambos en
`evaluation/validation.py`.

## 8. Salidas en inferencia (MaNGA)

Por spaxel y por target (masa/luz): `prob` (segmentación), `S`
(confianza en pseudo-conteos), **TU/AU/EU** (AU alta = frontera física
real; EU alta = fuera de dominio → triaje/abstención), vacuity
(diagnóstico), conjunto conformal (cobertura empírica en MaNGA: la
garantía formal se pierde con el shift y se reporta como tal), mapa
JS(p_mass‖p_lum) de discrepancia estructural masa/luz, `attn_w` por
modalidad y las certezas propagadas.

## 9. Pesos: qué existe y qué falta

**No hay pesos entrenados todavía** — el entrenamiento espera los
`dataset_entry` reales. El plan de pesos es:

- **Swin-T**: opcionalmente inicializado de ImageNet (timm,
  `pretrained=true` en config); el resto de la red **desde cero**.
- **Mamba, encoder físico, fusión, decoder, cabezas**: desde cero, vía el
  currículo de 4 etapas (la Etapa 1 autoentrena el encoder espectral).
- No se usa ningún foundation model externo: el dominio (espectros IFU +
  etiquetas de simulación) no tiene pretrained aplicable.
- Checkpoints: cada etapa parte del anterior; `(a, b)` de las
  EvidenceHeads se guardan con nombre. La escalera produce un checkpoint
  por nivel + `ladder_results.json` → tabla automática.

## 10. Mapa de código

```
src/galstructnet_s3/
  config.py                 carga YAML con _base_ y claves con punto
  cli.py                    train / ablate
  data/      dataset, transforms (D4, jitter, dropout h3/h4, rest-frame),
             collate (pad dinamico), stats (norm_stats.json)
  models/    model.py (contenedor + registry por config)
    encoders/ spectral (Mamba/Conv1D), spatial (Swin+FPN), physical (Std/N)
    layers/   normconv (P1), evidence (P2)
    fusion_global.py (concat, global) / fusion_precision.py (P3)
    decoder.py (FPN std/N)
    heads/    segmentation (std/evidence P4), psf, boundary, uncertainty
  losses/    dirichlet (ancla+KL), dice, psf, physics, domain (MMD), total
  training/  trainer (TrainerV3, Stage1Trainer), curriculum (MAE, consist, weak)
  evaluation/ metrics, conformal (Mondrian), validation (GZ3D, robustez),
             ablation_report (CSV+LaTeX)
```

Tests: `tests/unit/` (por módulo, replican las listas numeradas de cada
spec) + `tests/integration/` (skeleton e2e, A0 e2e + overfitting, overfit
EPN con ρ(S,N_eff), escalera desde configs, D4, CLI, robustez/GZ3D).
Fixture: `tests/conftest.py::synthetic_entry` — el contrato v3 ejecutable;
el piloto real `TNG50-87-141934-0-127` la sustituye fuera de CI.
