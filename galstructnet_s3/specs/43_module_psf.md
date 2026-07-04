# 43 — Módulo PSF-aware (v3)

> Módulo: `models/heads/psf.py` · Hito: 4 · Depende de: cabezas (40 v3),
> `EvidenceConv2d` ([45 §P2](45_evidence_layers.md)).
> Cambios v3: resolución del conflicto de etiquetas (C4 — el "Cambio F" v2
> queda anulado), modo α-convolución como default A4.

## Responsabilidad

Introduce el efecto del instrumento en el grafo: el modelo predice estructura
**intrínseca** (más nítida que la PSF) y este módulo la difumina con el
kernel de cada observación para compararla contra el plano observado. Sigue
siendo el diferenciador de GalStructNet-S3 frente a segmentadores estándar
(racional v2 íntegro: sin esto, la red aprende a reproducir el difuminado en
lugar de revertirlo, y no transfiere entre PSFs).

## Resolución del contrato de etiquetas (C4)

La v2 contenía tres afirmaciones incompatibles. Estado v3, definitivo:

- El proyector (labels repo, spec 20, Paso 6) **ya genera** `Y_*_raw` y
  `Y_*_psf` con la PSF correcta de cada vista.
- `L_seg` (50 v3) supervisa `prob_int` contra `Y_*_raw` + ancla `N_eff_raw`.
- `L_PSF` (50 v3) supervisa la salida de este módulo contra `Y_*_psf` +
  ancla `N_eff_psf`.
- **Nada se convoluciona on-the-fly en el DataLoader** (el Cambio F v2 se
  anula): evita recomputar lo ya hecho y el riesgo de doble convolución.

## Contratos

```python
# Entrada
alpha_int:  (B, K, H, W) ≥ 1        # de una cabeza (mass o lum)
psf_kernel: (B, Kh, Kw), suma 1     # por observación

# Salida
alpha_obs: (B, K, H, W) ≥ 1         # modo 'evidence'
prob_obs:  (B, K, H, W), suma 1     # ambos modos
```

## Dos modos (config `psf.mode`)

### `prob` — baseline (v2, conservado para ablación A4)

Convolución depthwise de `prob_int` por sample×clase (truco grouped-conv,
`groups=B·K`) + renormalización por la pérdida de masa del zero-padding.
Código v2 migrado tal cual.

### `evidence` — default v3 (α-convolución, 45 P2)

```python
class PSFEvidenceModule(nn.Module):
    """La PSF redistribuye pseudo-conteos: coherente con S ≈ κ·N_eff.

        α_obs = K_PSF * (α_int − 1) + 1
        prob_obs = α_obs / Σ α_obs        ← suma 1 automática, sin renormalizar
    """
    def forward(self, alpha_int, psf_kernel):
        B, K, H, W = alpha_int.shape
        e = (alpha_int - 1.0).view(1, B * K, H, W)
        psf = psf_kernel / psf_kernel.sum((-2, -1), keepdim=True).clamp_min(1e-12)
        w = psf.unsqueeze(1).expand(B, K, *psf.shape[-2:]).reshape(B*K, 1, *psf.shape[-2:])
        e_obs = F.conv2d(e, w, padding=(psf.shape[-2]//2, psf.shape[-1]//2),
                         groups=B * K).view(B, K, H, W)
        alpha_obs = e_obs + 1.0
        return alpha_obs, alpha_obs / alpha_obs.sum(1, keepdim=True)
```

Diferencias con `prob`: (i) la normalización es exacta por construcción;
(ii) `prob_obs` queda ponderada por la concentración local (los spaxels con
más evidencia pesan más en la mezcla post-PSF) — la historia generativa
correcta si S son conteos; (iii) habilita `L_PSF` como KL Dirichlet anclada
en espacio de evidencia (50 v3) en lugar de CE.

En bordes con zero-padding, parte de la evidencia sale del FOV: `S_obs`
decrece ahí. Es físico (el cubo termina), no bug; documentado y testeado.

## Validación

### Tests unitarios (`tests/unit/test_psf_module.py`)

1. **Shapes** (H,W dinámicos); B=1; kernels por-sample distintos.
2. **Suma a 1**: `prob_obs` (`atol=1e-6` en modo evidence; `1e-3` en prob).
3. **Identidad**: PSF = delta ⇒ `alpha_obs ≈ alpha_int` (atol 1e-5).
4. **Esparcimiento**: clase pura en un spaxel + PSF uniforme ⇒ evidencia
   repartida en el soporte; conservación interior
   `Σ e_obs = Σ e_int` (test 5 de 45).
5. **Regresión cruzada**: con α de baja concentración (S≈K+ε), `prob_obs`
   de ambos modos coinciden (atol 1e-3) — sanity de equivalencia en el
   límite.
6. **Gradiente fluye a `alpha_int`** en ambos modos.

## Criterios de aceptación

- [ ] Tests 1–6 pasan.
- [ ] Forward batch 4, kernel 11×11: < 100 ms.
- [ ] Selección por config; ambos modos en la escalera A4.
- [ ] El DataLoader v3 NO contiene `apply_psf_to_label` ni equivalente
      (grep en CI — refuerza C4).

## Notas de implementación

- El kernel viene como dato por observación (MaNGIA y MaNGA lo traen);
  normalizar en el dataset, re-normalizar aquí por seguridad — igual que v2.
- Kernel 11–13 px captura ~99% de la energía con FWHM ~5 px; no agrandar.
- Diagnóstico de entrenamiento (conservado de v2): si `prob_int` no resulta
  más nítida que las etiquetas, comparar `‖∇L_PSF/∇α_int‖` vs
  `‖∇L_seg/∇α_int‖` en wandb.
- Métrica de nitidez para A4 (70 v3): FWHM efectivo de la transición
  bulbo→disco en `prob_int` vs en `Y_raw` vs PSF — el modelo debe quedar
  entre etiqueta intrínseca y observada, más cerca de la primera.
