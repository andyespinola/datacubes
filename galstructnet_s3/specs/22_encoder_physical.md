# 22 — Encoder Físico (v3)

> Módulo: `models/encoders/physical.py` · Hito: 3 (estándar) / 4 (variantes
> EPN) · Depende de: dataset v3 (10), capas EPN (45 P1).
> Cambios v3: 8 canales (+h3/h4), versión NormConv como default A1+, variante
> de propagación de momentos (Nivel B). Refs: revisión C9, §2.2.

## Responsabilidad

Procesa los ocho mapas físicos (seis pyPipe3D + h3/h4 de pPXF, spec 26 de
inputs) **junto con sus certezas** y produce features espaciales que inyectan
cinemática y poblaciones estelares al modelo.

Los mapas contienen información que el encoder espectral tendría que
re-derivar ajustando espectros (reimplementar pyPipe3D/pPXF): v_star
(soporte rotacional), sigma_star (soporte de presión), age, metallicity,
mass, av — y, nuevos en v3, **h3/h4**: la correlación h3–V es la firma
cinemática directa de barras, la clase minoritaria más difícil. h3/h4 son
producto de pPXF (no de pyPipe3D) — distinción factual a conservar en todo
documento.

## Contratos

```python
# Entrada
maps:   (B, 8, H, W) f32   # v, sigma, age, Z, mass, av, h3, h4 (z-scoreados)
c_phys: (B, 8, H, W) f32   # certezas ∈ [0,1]; 0 donde NaN / ~M_valid / dropout

# Salida
features: (B, D_phys=64, H, W) f32
c_out:    (B, 1, H, W) f32          # certeza agregada (solo variantes EPN)
```

## Implementaciones (tres, seleccionables por config)

### `PhysicalEncoderStd` — baseline A0

La CNN de v2 (3 capas Conv 3×3 + GroupNorm + GELU, sin pooling), extendida a
`in_channels=8`. Ignora `c_phys` salvo para anular señal
(`x = x * (c_phys > 0)` — equivalente funcional del `nan_to_num` v2).
Devuelve `c_out = avg(c_phys)` para que la interfaz sea uniforme.

### `PhysicalEncoderN` — default A1+ (NormConv, 45 P1)

```python
class PhysicalEncoderN(nn.Module):
    def __init__(self, in_ch=8, d_out=64, hidden=32):
        super().__init__()
        self.l1 = NormConv2d(in_ch,    hidden,   k=3)
        self.l2 = NormConv2d(hidden,   hidden*2, k=3)
        self.l3 = NormConv2d(hidden*2, d_out,    k=1)

    def forward(self, maps, c_phys):
        x, c = self.l1(maps, c_phys)
        x, c = self.l2(x, c)
        x, c = self.l3(x, c)
        return x, c.mean(dim=1, keepdim=True)     # certeza agregada (B,1,H,W)
```

Garantías heredadas de P1: spaxels con `c=0` no contaminan a sus vecinos
(adiós interpolación implícita de NaNs); los bordes del FOV reciben certeza
menor automáticamente; equivariancia D4 exacta.

### `PhysicalEncoderMP` — Nivel B (propagación de momentos, opcional)

Variante de investigación (revisión §2.2): cada activación lleva `(μ, σ²)` y
las capas propagan ambos analíticamente (assumed density filtering: lineal
exacto + momentos de GELU bajo gaussiana, estilo Gast & Roth 2018 /
covarianza diagonal). Entrada: `μ = maps`, `σ² = pipe3d_err²` (sin z-score
del error: propagar en unidades normalizadas usando σ/σ_zscore).

```python
forward(mu, var) -> (mu_out (B,64,H,W), var_out (B,64,H,W))
# c_out := to_certainty(sqrt(var_out)) para interoperar con P3/P4
```

Es el único lugar del modelo donde se afirma semántica de varianza posterior
(la CNN es pequeña: el costo es ~2× este encoder, despreciable en el total).
Resultado publicable del capítulo: "la varianza de salida responde a la
varianza instrumental de entrada" — test 6.

## Validación

### Tests unitarios (`tests/unit/test_physical_encoder.py`)

1. **Shapes**: `(2,8,H,W)` → `(2,64,H,W)` (+ certeza) en las tres variantes,
   con H,W arbitrarios (probar 34×34 y 72×72).
2. **Ignorancia exacta (N)**: randomizar `maps` donde `c_phys=0` no cambia el
   output de `PhysicalEncoderN` (test 13 de 45 aplicado aquí).
3. **Std limpio**: input sin NaN ⇒ output sin NaN; con `c=0` en una región,
   esa región no produce NaN.
4. **Dropout de h3/h4**: `c_phys[:,6:8]=0` ⇒ output finito y distinto del
   caso con h3/h4 presentes (la información se usa cuando existe).
5. **Determinismo en eval; B=1.**
6. **(MP) Monotonía de varianza**: escalar `var_in` ×4 no disminuye
   `var_out` promedio (sanity de propagación).

### Test de overfitting (Hito 3/4)

Cada variante + cabeza dummy overfittea 1 sample del piloto.

## Criterios de aceptación

- [ ] Tests pasan para las tres variantes.
- [ ] `PhysicalEncoderN`: overhead < 2× vs Std en forward (sigue siendo
      despreciable en el total del modelo).
- [ ] Selección por config: `model.physical: {std, normconv, moments}`.
- [ ] Sin `nan_to_num` dentro del encoder (responsabilidad del dataset/c).

## Notas de implementación

- Resistir la tentación de añadir atención aquí — sigue válida la nota v2:
  los mapas son features precomputadas.
- Si el modelo overfittea en Etapa 2, `Dropout2d(0.1)` tras la primera capa
  (en N: sobre la señal, no sobre la certeza).
- `PhysicalEncoderMP` no entra a la escalera A0–A4; es un experimento lateral
  con bandera propia (`ablation: mp_encoder`) y su propio par de métricas
  (correlación var_out↔var_in; NLL del head con c de MP vs c heurística).
