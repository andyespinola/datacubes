# 30 — Decoder (v3)

> Módulo: `models/decoder.py` · Hito: 3 (estándar) / 4 (variante NormConv).
> Cambios v3: una sola implementación (la confusión narrada del spec v2 se
> elimina — C9), tamaños dinámicos (C5), variante (señal, certeza) para A1.

## Responsabilidad

Toma `F_fused (B, 384, H, W)` (ya a resolución nativa) y las skip connections
multi-escala del encoder espacial, y produce `hidden (B, 256, H, W)` para las
cabezas. No es un U-Net clásico (el bottleneck no está a resolución mínima):
es un **refinador FPN top-down** — las skips, a resolución menor, se
upsamplean a la nativa y se mezclan progresivamente.

## Contratos

```python
# Entrada
features: (B, 384, H, W)                 # de fusion (23 v3)
skips:    list[(B, 256, H_i, W_i)] ×3    # del encoder espacial (21 v3),
                                         # resoluciones H/2ᵏ aprox.
c_fused:  (B, 1, H, W)                   # certeza fusionada (variante N)
c_skips:  list[(B, 1, H_i, W_i)] ×3      # avg-pool de c_spat por escala

# Salida
hidden: (B, 256, H, W)
c_dec:  (B, 1, H, W)                     # certeza para EvidenceHead (40 v3)
```

## Algoritmo (estándar, A0)

Única implementación — versión final del v2 sin la primera variante abortada:

```python
class FPNDecoder(nn.Module):
    """features ya está a alta resolución; cada skip se upsamplea al tamaño
    de features y se mezcla con un bloque Conv-GN-GELU ×2."""

    def __init__(self, in_ch=384, skip_ch=(256, 256, 256), out_ch=256, mid=256):
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch, mid, 1)
        self.blocks = nn.ModuleList(
            [self._block(mid + s, mid) for s in skip_ch])
        self.final = self._block(mid, out_ch, n=3)

    def _block(self, i, o, n=2):
        L = []
        for j in range(n):
            L += [nn.Conv2d(i if j == 0 else o, o, 3, padding=1),
                  nn.GroupNorm(8, o), nn.GELU()]
        return nn.Sequential(*L)

    def forward(self, features, skips):
        hw = features.shape[-2:]                      # ← dinámico, sin (69,69)
        x = self.in_proj(features)
        for skip, block in zip(skips, self.blocks):   # profundo → superficial
            s = F.interpolate(skip, size=hw, mode='bilinear',
                              align_corners=False)
            x = block(torch.cat([x, s], dim=1))
        return self.final(x)
```

## Variante NormConv (A1, `decoder: normconv`)

Mismos bloques con `NormConv2d` (45 P1) en lugar de Conv: cada bloque consume
y emite `(x, c)`; las skips entran como `(skip, c_skip↑)` y se concatenan en
señal y certeza. `c_dec` = certeza de salida del bloque final. En la versión
estándar, `c_dec := F.interpolate(c_fused)` pasa-través (interfaz uniforme).

```python
forward(features, skips, c_fused, c_skips) -> (hidden, c_dec)
```

Aproximación documentada: las skips vienen del Swin (estándar, sin certeza
propia); `c_skips` se construye con avg-pool de `c_spat` a cada escala — la
certeza fotométrica es el proxy de confiabilidad de esas features.

## Validación

### Tests unitarios (`tests/unit/test_decoder.py`)

1. **Shapes dinámicos**: probar (H,W) ∈ {(34,34), (72,72), (74,74)} → output
   `(B, 256, H, W)`. Cero literales espaciales en el módulo (grep en CI).
2. **Determinismo en eval; B=1.**
3. **Las skips contribuyen**: perturbar una skip cambia el output.
4. **(N) Ignorancia**: región con `c_fused=0` y `c_skips=0` ⇒ el output allí
   no depende de los valores de señal de esa región.
5. **(N) `c_dec ∈ [0,1]`**, 0 solo donde todo el soporte es 0.

### Test de overfitting (Hito 3)

Decoder + cabeza dummy overfittea 1 sample en < 200 iter (ambas variantes).

## Criterios de aceptación

- [ ] Tests pasan; forward batch 4 en GPU < 60 ms (estándar) / < 120 ms (N).
- [ ] Memoria pico < 2.5 GB con B=4.
- [ ] Selección por config `model.decoder: {std, normconv}`.

## Notas de implementación

- Si los gradientes hacia las skips son ~0 durante el entrenamiento, el
  modelo las ignora — loguear sus normas (nota v2, conservada).
- Si la memoria aprieta: `mid` 256→192 antes que tocar `out_ch`.
- La variante N duplica activaciones de este módulo (señal+certeza); sigue
  siendo <5% del presupuesto total (el Mamba domina).
