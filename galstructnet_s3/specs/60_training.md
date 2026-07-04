# 60 — Loop de entrenamiento y currículo (v3)

> Módulo: `training/trainer.py`, `training/curriculum.py` · Hito: 5–6.
> Cambios v3: Etapa 1 = masked spectral modeling **correcto** + MaNGIA∪MaNGA
> (C6 + §4); Etapa 3 incorpora `L_weak` con GZ3D particionado; la escalera de
> ablación A0–A4 es entregable de primera clase.

## Las cuatro etapas

| Etapa | Datos | Pérdidas activas | Épocas |
|---|---|---|---|
| 1 | **MaNGIA ∪ MaNGA** (sin etiquetas) | masked spectral (por posición) | 50 |
| 2 | MaNGIA filtrado por QA | seg+dice (×2 targets) + phys | 100 |
| 3 | MaNGIA + MaNGA(GZ3D-weak + unlabeled) | + PSF + boundary + consist + **weak** | 50 |
| 4 | MaNGIA + MaNGA (DA) | + MMD (o DANN) | 50 |

Cada etapa parte del checkpoint anterior; pesos rampados en las primeras
épocas. Splits **por galaxia** (las 4 vistas juntas) — leakage si no.

## Etapa 1 — Masked spectral modeling (corregido)

### Qué cambió (C6)

La v2 pooleaba el eje espectral antes de reconstruir: el decoder veía solo un
vector resumen de 256-d y el enmascarado del 30% apenas cambiaba la tarea
(autoencoder con cuello de botella global, sin gradiente útil de
"rellenar desde contexto"). v3 = MAE/BERT real:

```
cube  → [máscara de canales con token aprendible] → MambaSpectralEncoder
      → (N, L', d_model)  [SIN pooling]
      → proyección por posición a los canales originales
      → MSE solo en (canal, spaxel) enmascarados Y válidos
```

```python
class SpectralMAEHead(nn.Module):
    """Reconstrucción por posición desde la secuencia post-Mamba (no del
    pooling). Un token de máscara aprendible reemplaza los canales ocultos
    en la ENTRADA del encoder."""
    def __init__(self, d_model=128, patch=4):     # patch = downsample del encoder
        super().__init__()
        self.proj = nn.Linear(d_model, patch)     # reconstruye los `patch`
                                                  # canales que cada paso resume
    def forward(self, seq):                        # (N, L', d_model)
        return self.proj(seq).flatten(1)           # (N, L'·patch ≈ L)
```

```python
def masked_spectral_loss(rec, target, mask_lambda, mask_spatial):
    diff = (rec - target) ** 2
    m = mask_lambda.unsqueeze(-1).unsqueeze(-1) * mask_spatial.unsqueeze(1).float()
    return (diff * m).sum() / m.sum().clamp_min(1.0)
```

### Datos y por qué MaNGA entra aquí (§4)

MaNGIA (sin filtro QA) **∪** cubos MaNGA reales sin etiquetas
(`splits/manga_unlabeled.txt`). Es la forma más barata y estable de cerrar
parte del gap de dominio: el encoder espectral aprende representaciones
compartidas desde el inicio y no "ve por primera vez" un espectro real en la
Etapa 3. Reduce la carga de la Etapa 4. Riesgo bajo (no hay etiquetas que
corromper).

```yaml
stage_1:
  epochs: 50
  batch_size: 4
  optimizer: AdamW
  lr: 1.0e-4
  weight_decay: 0.01
  scheduler: cosine
  warmup_epochs: 5
  mask_ratio: 0.30
  data: [mangia_all, manga_unlabeled]   # mezcla 50/50 por batch
  restframe: true                        # ablar; reduce varianza Doppler
```

Éxito: loss de val baja/estanca; espectros reconstruidos razonables;
**embeddings de MaNGIA y MaNGA solapan** en una proyección 2D (UMAP) — chequeo
directo de que el SSL conjunto cierra dominio.

## Etapa 2 — Supervisado en MaNGIA (dual target)

Modelo completo; encoder espectral desde Etapa 1. Pérdida: `seg+dice` para
ambos targets (λ_mass=0.3) + `phys`. `PSF`/`boundary` aún off.

```yaml
stage_2:
  epochs: 100
  lr: 5.0e-5
  warmup_epochs: 10
  loss_weights: {seg: 1.0, dice: 0.5, phys: 0.1, boundary: 0.0, psf: 0.0,
                 lambda_mass: 0.3}
  kappa: 0.5            # ancla (ablar 0.25/0.5/1.0)
  augmentation: full
```

Datos: MaNGIA con `qa_status != fail`, split 70/15/15 por galaxia.
Éxito: accuracy argmax val > 70% en válidos; IoU mediano > 0.5 (salvo
"other"); **`ρ_Spearman(S, N_eff)` > 0.4** (la innovación funcionando);
ECE < 0.15.

## Etapa 3 — Semi-supervisado + supervisión débil GZ3D

```yaml
stage_3:
  epochs: 50
  lr: 1.0e-5
  loss_weights: {seg: 1.0, dice: 0.5, phys: 0.1, boundary: 0.3, psf: 0.4,
                 consist: 0.2, weak: 0.2, lambda_mass: 0.3}
  data: [mangia_qa, manga_gz3d_weak, manga_unlabeled]
```

### `L_weak` (GZ3D, la única señal de MaNGA con gradiente real en clases difíciles)

GZ3D aporta máscaras crowdsourced de **barras y brazos** para un subset MaNGA.
`L_weak` = BCE de `prob_bar`/`prob_arm` (en el plano observado, vía PSF)
contra la fracción de voto GZ3D, enmascarada a la cobertura de cada máscara.
Solo esas dos clases; el resto del simplex queda libre.

**Partición obligatoria (regla que faltaba en v2):** GZ3D se divide en
`manga_gz3d_weak` (entra a `L_weak`) y `manga_gz3d_val` (validación externa,
70 v3) **disjuntos por galaxia**. Usar GZ3D íntegro para ambas cosas
invalidaría la validación.

`L_consist`: dos forwards con augmentations D4 distintas → KL simétrica
(código v2).

Éxito: métricas MaNGIA-val no se degradan vs Etapa 2; predicciones MaNGA
razonables; acuerdo bar/arm con `manga_gz3d_val` mejora vs Etapa 2.

## Etapa 4 — Domain adaptation

Igual que v2: MMD primero (más simple/estable), DANN como plan B,
self-training solo como último recurso (filtrado por S alto **y** conjunto
conformal unitario, auditado contra GZ3D holdout — revisión §4).

```yaml
stage_4:
  epochs: 50
  lr: 5.0e-6
  loss_weights: {..., mmd: 0.5}
  data: [mangia_qa (50%), manga (50%)]
```

Éxito: MaNGIA-val no cae >5%; histogramas de confianza por clase similares
entre dominios; **EU sube en MaNGA** donde el gap importa (sanity de que la
epistémica corregida, 42 v3, mide dominio).

## Trainer

Estructura v2 conservada (`train_epoch`, `validate`, `_clip_gradients` a
norma 1.0 — crítico con Mamba, `run`, checkpointing, mixed precision bf16,
wandb). Añadidos v3:

```python
class TrainerV3(Trainer):
    def compute_loss(self, outputs, batch, stage):
        L = self.loss_fn(outputs, batch)               # 50 v3
        if stage >= 3:
            L["consist"] = consistency_loss(outputs, self._aug_forward(batch))
            if batch.get("gz3d_mask") is not None:
                L["weak"] = weak_gz3d_loss(outputs, batch)
            L["total"] = L["total"] + self.w_consist*L["consist"] \
                                    + self.w_weak*L.get("weak", 0.0)
        return L
```

KL en float32 local (50 v3); `assert (alpha >= 1).all()` antes de digamma.

## Escalera de ablación A0–A4 (entregable de primera clase)

`configs/ablation_epn/{A0,A1,A2a,A2b,A2c,A3,A4_prob,A4_evid}.yaml`. Cada una
fija las banderas de modelo (`physical`, `fusion`, `decoder`, head std/EPN,
`psf.mode`) y comparte semillas, split y presupuesto. El Trainer corre la
escalera con un solo comando (`cli ablate --ladder epn`). Resultados →
tabla automática en `evaluation/` (70 v3) con las métricas firma:
IoU/Dice, NLL/Brier, `ρ(S, N_eff)`, AUROC(EU→error), ECE, cobertura conformal,
y las dos curvas robustez (barrido de ruido, dropout de spaxels).

## Validación del trainer

Tests v2 conservados (1 epoch sin error; resume desde checkpoint; bf16 no
rompe overfitting; wandb registra términos) + v3:

- **Etapa 1 corregida**: con `mask_ratio=1.0` la loss es alta y baja al
  reducir el ratio (la tarea depende del enmascarado — lo que v2 no cumplía).
- **`ρ(S, N_eff)`** se loguea cada validación desde Etapa 2.
- **Partición GZ3D**: assert de disjunción `weak ∩ val = ∅` al construir
  loaders.

## Criterios de aceptación

- [ ] Cada etapa corre end-to-end; checkpoints cargan/guardan.
- [ ] Etapa 1 reconstruye por posición (no desde pooling) — test del ratio.
- [ ] `L_weak` solo sobre bar/arm; GZ3D particionado con assert.
- [ ] Escalera A0–A4 reproducible desde configs con un comando.
- [ ] bf16 + clip 1.0; wandb con los 6+ términos y `ρ(S,N_eff)`.
- [ ] Overfitting 1 sample < 200 iter (A0 y A3).

## Notas de implementación

- Gradient clipping con Mamba: imprescindible (nota v2).
- Validación cara: cada época al inicio, luego cada 5; early stopping a 10
  épocas sin mejora.
- Reproducibilidad bit-exacta imposible con CUDA+bf16; basta similitud de
  métricas finales.
- Si la Etapa 1 conjunta no solapa dominios (UMAP): subir proporción de MaNGA
  o añadir un término de alineación ligero (CORAL) — ADR.
