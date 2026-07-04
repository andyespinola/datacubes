# 50 — Función de pérdida unificada (v3): supervisión Dirichlet anclada

> Módulo: `losses/` · Hito: 4 · Depende de: cabezas (40–43 v3), dataset (10 v3).
> Cambios v3: `L_seg` pasa de Dirichlet-NLL a **KL entre Dirichlets anclada
> en N_eff** (núcleo de la innovación, revisión §3.1); `L_unc` y su ramping
> **se eliminan** (C2); `L_phys` ponderada por masa (C7); multi-tarea con dos
> cabezas (C3); `L_PSF` opcionalmente en espacio de evidencia.

## Visión general

```
L_total = Σ_t∈{lum,mass} λ_t · [ L_seg(t) + β·L_dice(t) + δ·L_PSF(t) ]
        + α·L_boundary + ε·L_phys + (Etapa 3+) w_c·L_consist + w_k·L_weak
```

| Término | Target | Ancla | Peso default |
|---|---|---|---|
| `L_seg(t)` | `Y_t_raw` | `N_eff_raw_t` | 1.0 (λ_lum=1.0, λ_mass=0.3) |
| `L_dice(t)` | `Y_t_raw` | — | 0.5 |
| `L_boundary` | `Y_lum_raw` | — | 0.3 (Etapa 3+) |
| `L_PSF(t)` | `Y_t_psf` | `N_eff_psf_t` | 0.4 (Etapa 3+) |
| `L_phys` | fracciones catálogo | ponderación masa/flujo | 0.1 |
| `L_consist` | self | — | 0.2 (Etapa 3+) |
| `L_weak` | GZ3D (bar/arm, MaNGA) | — | 0.2 (Etapa 3+) |
| ~~`L_unc`~~ | — | — | **eliminada** |

### Por qué desaparece `L_unc` (registro para la tesis)

El KL→uniforme sobre el argmax (i) destruía la información de las etiquetas
suaves (penalizaba evidencia legítima de la segunda clase), (ii) es el
regularizador cuya incertidumbre resultante la literatura 2022–2024 demostró
no fiel (Bengs; Jürgens; Shen — revisión C2), y (iii) era el único mecanismo
que controlaba S, dejándola sin semántica y exigiendo el ramping manual de
λ_KL. El ancla `α* = κ·N_eff·Y + 1` cumple esa función con referencia
frecuentista: S queda supervisada directamente, sin término extra ni ramping.

## L_seg — KL entre Dirichlets anclada

```python
# losses/dirichlet.py

def build_anchor(Y, n_eff, kappa=0.5, n_eff_cap=None, alpha0=1.0):
    """α* = κ·N_eff·Y + α0.  Cap en el p99 de train para que spaxels de
    concentración extrema no dominen la optimización."""
    if n_eff_cap is not None:
        n_eff = n_eff.clamp_max(n_eff_cap)
    return kappa * n_eff.unsqueeze(1) * Y + alpha0


def dirichlet_kl(a, b, mask):
    """KL( Dir(a) ‖ Dir(b) ), forma cerrada, enmascarada y promediada.

    KL = lnΓ(a0) − Σ lnΓ(a_c) − lnΓ(b0) + Σ lnΓ(b_c)
         + Σ (a_c − b_c)(ψ(a_c) − ψ(a0))
    """
    a0 = a.sum(1, keepdim=True); b0 = b.sum(1, keepdim=True)
    kl = (torch.lgamma(a0) - torch.lgamma(a).sum(1, keepdim=True)
          - torch.lgamma(b0) + torch.lgamma(b).sum(1, keepdim=True)
          + ((a - b) * (torch.digamma(a) - torch.digamma(a0)))
            .sum(1, keepdim=True)).squeeze(1)              # (B, H, W)
    m = mask.float()
    return (kl * m).sum() / m.sum().clamp_min(1.0)


def anchored_seg_loss(alpha_pred, Y_raw, n_eff_raw, mask,
                      kappa, n_eff_cap, direction="forward"):
    a_star = build_anchor(Y_raw, n_eff_raw, kappa, n_eff_cap)
    if direction == "forward":          # KL(ancla ‖ pred) — default
        return dirichlet_kl(a_star, alpha_pred, mask)
    return dirichlet_kl(alpha_pred, a_star, mask)          # "reverse"
```

- **Dirección**: `forward` (ancla‖pred) como default; `reverse` como config.
  La patología de gradientes de la verosimilitud Dirichlet en clases de baja
  probabilidad (Ryabinin et al. 2021) motiva ablar ambas; con K=5 el efecto
  es menor que en sus benchmarks, pero se mide, no se asume.
- **κ**: ablar {0.25, 0.5, 1.0}. Interpretación: κ=1 exige confianza al nivel
  de la estadística de partículas; κ<1 la descuenta (las etiquetas tienen
  además error sistemático del GMM, no solo de muestreo).
- **Cap**: `n_eff_cap = p99(N_eff_train)` precalculado en `norm_stats`.
- **Etiquetas duras (sanity)**: con Y one-hot y N_eff→∞ la KL forward se
  comporta como CE más un término de concentración — test 5.

## L_dice — sin cambios de fondo

Código v2 (Dice multiclase suave) sobre `prob_t` vs `Y_t_raw`. Sigue siendo
el contrapeso del desbalance (barra/brazo 5–10% de spaxels).

## L_boundary — sin cambios

MSE entre `B(prob_lum)` y `B(Y_lum_raw)` con τ=0.1 (41 v3: reframing como
regularizador, mismo cómputo). Solo target de luz (las fronteras científicas
se definen en luz).

## L_PSF — anclada en el plano observado

```python
# losses/psf.py
# modo 'evidence' (default con PSFEvidenceModule):
L_PSF(t) = dirichlet_kl( build_anchor(Y_t_psf, n_eff_psf_t, kappa, cap),
                         alpha_obs_t, mask )
# modo 'prob' (baseline A4): CE suave de v2 entre prob_obs y Y_t_psf
```

Los targets `Y_*_psf` y `N_eff_psf_*` vienen del proyector (C4): nada se
convoluciona en el DataLoader.

## L_phys — ponderada por masa (corrección C7)

```python
# losses/physics.py

def physics_constraint_loss(prob, mask, w_map, target_fractions, tol=0.05):
    """w_map: masa RAW por spaxel (w_phys_mass del dataset) para el target de
    masa; flujo en banda r para el de luz. Compara fracciones DE MASA contra
    fracciones de masa del catálogo — v2 comparaba fracciones de área contra
    fracciones de masa (incompatibles)."""
    w = (w_map * mask.float()).unsqueeze(1)                 # (B,1,H,W)
    frac = (prob * w).sum((2, 3)) / w.sum((1, 2, 3)).clamp_min(1e-6).unsqueeze(1)
    return ((frac - target_fractions).abs() - tol).clamp_min(0).mean()
```

Se omite (peso 0) para galaxias sin fracciones de catálogo — igual que v2.

## L_consist y L_weak (Etapa 3+, definidos en 60 v3)

`L_consist`: KL simétrica entre dos forwards con augmentations distintas
(código v2). `L_weak`: BCE de `prob_bar` y `prob_arm` contra fracciones de
voto GZ3D en el subset MaNGA débilmente etiquetado, enmascarada a la
cobertura GZ3D (60 v3 da el detalle y la partición).

## Máscaras (simplificación v3)

La gimnasia de doble pasada de v2 se sustituye por dos máscaras explícitas
en la firma:

```python
mask_strict = M & ~M_unc_t      # L_seg, L_dice, L_PSF (etiquetas confiables)
mask_loose  = M                  # L_boundary, L_phys
loss_fn(outputs, batch)          # construye ambas internamente por target
```

## La función combinadora

```python
# losses/total.py

class GalStructNetLossV3(nn.Module):
    def __init__(self, w_seg=1.0, w_dice=0.5, w_boundary=0.3, w_psf=0.4,
                 w_phys=0.1, w_consist=0.2, w_weak=0.2, lambda_mass=0.3,
                 kappa=0.5, n_eff_cap=None, kl_direction="forward",
                 psf_mode="evidence", boundary_tau=0.1, phys_tol=0.05):
        ...

    def forward(self, outputs, batch) -> dict:
        L = {}
        for t, lam in (("lum", 1.0), ("mass", self.lambda_mass)):
            ms = batch["M"] & ~batch[f"M_unc_{t}"]
            L[f"seg_{t}"]  = anchored_seg_loss(outputs[t]["alpha"],
                              batch[f"Y_{t}"], batch[f"n_eff_{t}"], ms,
                              self.kappa, self.n_eff_cap, self.kl_direction)
            L[f"dice_{t}"] = dice_loss_multiclass(outputs[t]["prob"],
                              batch[f"Y_{t}"], ms)
            if self.w_psf > 0 and f"alpha_obs_{t}" in outputs:
                L[f"psf_{t}"] = psf_loss(outputs, batch, t, ms,
                                         mode=self.psf_mode, ...)
        L["boundary"] = boundary_loss(outputs["boundary"], batch["Y_lum"],
                                      batch["M"], self.boundary_tau)
        L["phys"] = physics_constraint_loss(outputs["lum"]["prob"], batch["M"],
                       batch["w_phys_mass"], batch["target_fractions_mass"],
                       self.phys_tol)
        # consist / weak: añadidos por el Trainer en Etapa 3+ (60 v3)
        L["total"] = combine(L, self.weights)               # ver tabla
        return L
```

## Sanity de magnitudes al inicio del entrenamiento

Con el init calibrado de `b` (40 v3), órdenes esperados en el primer paso:

```
seg_lum:   O(1–10)   ← depende de κ y del gap S_init↔ancla; loguear SIEMPRE
dice_lum:  0.7–0.9
boundary:  0.05–0.2
psf_lum:   O(seg)
phys:      0.0–0.5
```

Regla v2 conservada: ningún término >10× ni <1% de los demás tras la primera
época; ajustar pesos, no la matemática.

## Validación

### Tests unitarios (`tests/unit/test_loss.py`)

1. **`dirichlet_kl(a, a) = 0`** exacto; `≥ 0` sobre aleatorios; simetría NO
   (test de que forward≠reverse).
2. **Ancla correcta**: `build_anchor` con N_eff=0 ⇒ α*=1 (spaxel sin
   estadística pide ignorancia — coherente con 45 §10).
3. **Gradiente hacia S**: con `prob_pred == Y` pero `S_pred ≠ S*`, el
   gradiente de `L_seg` sobre α es no nulo (la pérdida supervisa
   concentración, no solo media). Test clave de la innovación.
4. **Cap y máscara** funcionan; NaN-free en bf16 sobre tensores del piloto.
5. **Límite duro**: Y one-hot, N_eff grande ⇒ ranking de pérdidas coincide
   con CE (correlación de Spearman > 0.95 sobre predicciones aleatorias).
6. **C3**: `L_seg_mass` no genera gradiente en `head_lum.proj`.
7. **C7**: con prob uniforme y masa concentrada en una región de clase
   conocida, `L_phys` v3 difiere de la v2 en la dirección esperada (test
   construido).
8. **Backward de `total`** produce gradientes finitos en todos los
   parámetros.

## Criterios de aceptación

- [ ] Tests 1–8 pasan.
- [ ] Cero menciones a `kl_to_uniform`/`L_unc`/ramping en `losses/` (grep CI).
- [ ] Cada término logueado por separado en wandb (regla v2 conservada).
- [ ] Configs de la escalera (45) seleccionan κ, dirección y modo PSF sin
      tocar código.

## Notas de implementación

- `lgamma/digamma` en bf16 pierden precisión con α grandes: computar la KL
  en float32 (`autocast(enabled=False)` local) — coste despreciable.
- Si `seg` domina por el gap inicial de S: bajar κ antes que subir pesos de
  otros términos (la escala del ancla es el dial correcto).
- Detección de NaN por término — regla v2 conservada; el sospechoso típico
  v3 es `digamma` cerca de 0 (no debería ocurrir con α≥1: assert).
