# 70 — Evaluación y validación (v3)

> Módulo: `evaluation/metrics.py`, `evaluation/validation.py`,
> `evaluation/conformal.py` · Hito: 7.
> Cambios v3: métricas de la innovación (`ρ(S,N_eff)`, AUROC con EU correcta),
> conformal sobre el simplex, scoring rules para etiquetas suaves, partición
> GZ3D weak/val, tabla automática de la escalera A0–A4.

## Tres niveles (v2 conservado)

1. Métricas internas sobre MaNGIA-val (monitoreadas en entrenamiento).
2. Validación cruzada con catálogos externos (GZ3D-val, BUDDI-MaNGA).
3. Inspección cualitativa por astrónomos.

## Métricas internas

### Segmentación (v2 conservado, ahora por target)

`compute_iou_per_class`, `compute_dice_per_class`, `compute_pixel_accuracy`
sobre `M_valid & ~M_uncertain_t`, para `t ∈ {mass, lum}`. Argmax de `prob_t`.

### Calidad probabilística con etiquetas suaves (v3)

ECE-argmax (v2) **no basta** con etiquetas suaves: premia un modelo que acierta
la clase modal aunque calibre mal el resto del simplex. Añadir scoring rules
propias:

```python
def soft_brier(prob, Y_soft, mask):     # ‖prob − Y‖² promediado en clases
    d = ((prob - Y_soft) ** 2).sum(1)
    return (d * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

def soft_nll(prob, Y_soft, mask):       # −Σ Y·log p (CE suave)
    ce = -(Y_soft * torch.log(prob.clamp_min(1e-8))).sum(1)
    return (ce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
```

Reportar Brier y NLL suaves **además** de ECE. Son las métricas primarias de
la escalera (distinguen A3 de A0 donde IoU empata).

### La innovación, medida (v3) — núcleo de la defensa

```python
def concentration_calibration(alpha, n_eff, kappa, mask):
    """¿La concentración predicha trackea la estadística física de la etiqueta?
    Si la innovación funciona, S ≈ κ·N_eff."""
    S = alpha.sum(1)[mask]
    target = (kappa * n_eff)[mask]
    rho = spearman(S, target)                     # primario
    slope, r2 = ols(target, S)                    # secundario: pendiente≈1, R²
    return {"rho_S_neff": rho, "slope": slope, "r2": r2}
```

`ρ(S, N_eff)` se reporta por target y por estrato de radio. Es el enunciado
empírico que responde la crítica EDL: existe distribución de referencia y la
concentración la sigue.

### Descomposición de incertidumbre, validada (v3)

```python
def error_detection_auroc(EU, prob, Y_soft, mask):
    """¿La epistémica (información mutua, 42 v3) predice los errores del argmax?
    AUROC alto = la incertidumbre es útil para abstención/triaje."""
    err = (prob.argmax(1) != Y_soft.argmax(1))[mask].float()
    return auroc(EU.squeeze(1)[mask], err)        # primario para A3
```

Reportar AUROC con EU, con TU y con vacuity por separado — la tesis muestra
que EU (correcta) ≥ vacuity (lo que v2 llamaba epistémica) en detección de
error. Más: **diagrama de dispersión AU vs EU** sobre spaxels de frontera
(AU alta esperada) vs spaxels OOD inducidos (EU alta) — la figura que
demuestra que la descomposición v3 separa lo que la v2 confundía (C1).

### Conservación física (v3, corregida)

`compute_global_fractions` ahora **ponderada por masa/flujo** (C7), comparada
contra catálogo en fracciones de masa. Distancia L1 por galaxia.

## Conformal sobre el simplex (innovación #3, revisión §3.3)

```python
# evaluation/conformal.py

def calibrate_mondrian(probs_cal, Y_cal, mask_cal, strata, alpha=0.1):
    """Split-conformal Mondrian: umbral por estrato (clase × bin de N_eff o
    radio) para cobertura condicional. Score de no-conformidad: s = 1 − p_y
    (con y = argmax(Y_soft) del spaxel de calibración)."""
    q = {}
    for g in unique(strata):
        s = (1 - probs_cal.gather(1, Y_cal.argmax(1, keepdim=True)).squeeze(1))
        s = s[mask_cal & (strata == g)]
        q[g] = quantile(s, ceil((n+1)*(1-alpha))/n)          # corrección finita
    return q

def predict_sets(probs, strata, q):
    """Conjunto por spaxel: {c : prob_c ≥ 1 − q[estrato]}. Cobertura 1−α
    garantizada (intercambiabilidad) por estrato."""
    ...

def coverage_efficiency(sets, Y_soft, mask):
    cov  = (sets.gather(1, Y_soft.argmax(1, keepdim=True)).squeeze(1)[mask]).mean()
    size = sets.sum(1)[mask].float().mean()                  # eficiencia
    return cov, size
```

Estratos: clase × {bins de N_eff} (donde la estadística difiere) y × radio.
Calibrar en MaNGIA-val; reportar cobertura/eficiencia por estrato en
MaNGIA-test. **En MaNGA**: la intercambiabilidad se rompe (covariate shift
MaNGIA→MaNGA); reportar cobertura empírica en `manga_gz3d_val` (bar/arm) y
**discutir explícitamente** la pérdida de garantía — honestidad ante el
comité, no overclaim.

## Validación cruzada externa

### GZ3D (particionado, v3)

```python
def compare_with_gz3d(model, gz3d_val_path, loader_manga_val, n=500):
    """Acuerdo de prob_bar/prob_arm con máscaras GZ3D. SOLO sobre
    manga_gz3d_val (disjunto de manga_gz3d_weak usado en L_weak, 60 v3).
    Métricas: IoU bar, IoU arm, correlación de fracción de voto."""
```

Assert de disjunción `weak ∩ val = ∅` al cargar (espejo del de 60 v3).
BUDDI-MaNGA / PyMorph: comparación de fracciones B/T globales (luz) — distancia
de distribución, no por spaxel.

## Tabla automática de la escalera A0–A4

`evaluation/ablation_report.py` consume los checkpoints de
`configs/ablation_epn/*` y emite una tabla (CSV + LaTeX) con columnas:
IoU_med, Dice_med, soft-NLL, soft-Brier, ECE, ρ(S,N_eff), slope, AUROC(EU),
cobertura@90 / eficiencia, y las dos curvas de robustez resumidas a un
escalar (AUC de métrica-vs-ruido; caída de IoU @30% dropout). Una fila por
nivel; las celdas que deciden adopción (A3>A0 en NLL **y** AUROC) se marcan.

## Criterios de aceptación

- [ ] Métricas v2 (IoU/Dice/acc/ECE) + v3 (soft-NLL/Brier, ρ(S,N_eff),
      AUROC(EU), conformal) implementadas y testeadas sobre el piloto.
- [ ] Partición GZ3D con assert en evaluación y en entrenamiento.
- [ ] Conformal: cobertura empírica MaNGIA-test ≥ 1−α por estrato (test de
      que la calibración es correcta).
- [ ] `ablation_report.py` produce la tabla A0–A4 desde checkpoints.
- [ ] Figura AU-vs-EU (frontera vs OOD) generada — evidencia de C1 corregido.

## Notas de implementación

- AUROC/Spearman/quantile: `torchmetrics` o scipy; mantener en CPU sobre los
  spaxels válidos aplanados (cabe en memoria).
- Conformal es post-hoc: no requiere reentrenar; corre sobre logits guardados.
- `ρ(S, N_eff)` baja si κ está mal: es diagnóstico de κ, no solo de calidad
  del modelo — cruzar con el barrido de κ de la Etapa 2.
- Para la tesis: la tabla A0–A4 + la figura AU-vs-EU + la curva de robustez
  son las tres piezas que sostienen, respectivamente, "la familia EPN aporta",
  "la descomposición v3 es correcta" y "el anclaje da robustez a ruido".
