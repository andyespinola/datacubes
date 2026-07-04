# 23 — Fusión multimodal (v3): por posición, modulada por precisión

> Módulo: `models/fusion_precision.py` (primaria), `models/fusion_global.py`
> (ablación) · Hito: 3 (global A0) / 4 (primaria) · Depende de: encoders
> 20–22, capas EPN ([45 §P3](45_evidence_layers.md)).
> Cambios v3: la cross-attention global de v2 deja de ser la fusión del
> modelo y pasa a variante de ablación A2. Refs: revisión §2.1.

## Responsabilidad

Combina las features de los tres encoders en un mapa unificado
`(B, 384, H, W)` **decidiendo por spaxel cuánto pesa cada modalidad**, con la
precisión instrumental medida como sesgo explícito de esa decisión. Devuelve
además la certeza fusionada `(B, 1, H, W)` que consumen el decoder y la
cabeza de evidencia.

## Por qué cambia respecto a v2 (honestidad arquitectónica)

La v2 declaraba la intuición correcta — "en alto S/N confiar en F_spec; en
bajo S/N apoyarse en F_spat/F_phys" — pero implementaba otra cosa: atención
global entre 4 761 tokens de query y 4 761 de key/value, donde el spaxel
(i,j) atiende a modalidades de (i',j') arbitrarios. Eso mezcla dos funciones
(gating de modalidad y contexto espacial, que ya aporta el Swin) y cuesta una
matriz de atención de 2.9 GB. La v3 implementa exactamente la intuición
declarada: **M=3 tokens por posición** (uno por modalidad), atención O(H·W·M²),
con los logits sesgados por la certeza medida de cada modalidad en ese spaxel.

## Contratos

```python
# Entrada
F_spat (B,256,H,W) · F_spec (B,256,H,W) · F_phys (B,64,H,W)
cbar = {"spat": (B,1,H,W), "spec": (B,1,H,W), "phys": (B,1,H,W)}
#  c̄_spat = avg(c_spat); c̄_spec = c_spec del dataset; c̄_phys = c_out del
#  encoder físico (variantes N/MP) o avg(c_phys) (Std).

# Salida
F_fused (B, 384, H, W) · c_fused (B, 1, H, W)
attn_w  (B, 3, H, W)            # pesos por modalidad — producto científico
```

## Algoritmo

La implementación de referencia es `PrecisionGatedFusion` de
[45 §P3](45_evidence_layers.md) (proyección por modalidad a d=384, query =
token espacial, sesgo aditivo `g_m(log c̄_m)` en los logits, residual con
F_spat, GroupNorm). Este spec añade los detalles de integración:

- **Exportar `attn_w`** siempre en eval (mapa "en qué se apoyó el modelo");
  en train, loguear su media por modalidad a wandb cada época.
- **`g_m` init→0**: arranque equivalente a fusión sin precisión (test de
  neutralidad, 45 §9). El gating se gana su efecto por gradiente.
- **`c_fused = Σ_m attn_w_m · c̄_m`** — promedio convexo; alimenta
  `UNetDecoder` (30 v3) y, tras el decoder, la `EvidenceHead` (40 v3).

## Variantes para la escalera de ablación

| Config | Clase | Nivel |
|---|---|---|
| `fusion: concat` | concat + Conv1×1 (la opción 1 descartada en v2; barata, baseline mínimo) | A2-a |
| `fusion: global` | `CrossAttentionFusion` de v2, código migrado tal cual (flash attention, residual F_spat) | A2-b / A0 |
| `fusion: precision` | `PrecisionGatedFusion` | A2-c (default v3) |

Las tres exponen la misma interfaz (devuelven `c_fused`; `concat` y `global`
la calculan como promedio simple de `cbar`).

## Validación

### Tests unitarios (`tests/unit/test_fusion.py`)

1. **Shapes** con H,W arbitrarios; B=1.
2. **Gradient flow a las tres modalidades** (test v2, conservado).
3. **Determinismo en eval.**
4. **Degradación selectiva** (45 §8): bajar `c̄_spec→0` en una región reduce
   monótonamente `attn_w[spec]` allí (estadístico, no exacto).
5. **Neutralidad inicial** (45 §9): con `g_m≡0`, salida == fusión sin
   precisión.
6. **Convexidad de `attn_w`**: suma 1, ≥0, por posición.
7. **Interfaz uniforme**: las tres variantes pasan 1–3 con la misma firma.

## Criterios de aceptación

- [ ] Tests 1–7 pasan.
- [ ] Forward `precision` sobre batch de 4 en A100: < 30 ms (es O(H·W·9)).
- [ ] Memoria pico < 1 GB con B=4 (vs ~3 GB de `global` sin flash).
- [ ] `attn_w` visible en wandb (imagen cada 10 épocas).

## Notas de implementación

- `attn_mask` aditivo de `nn.MultiheadAttention`: verificar broadcasting por
  cabeza (mismo sesgo para todas las cabezas; forma `(B·H·W·heads, 1, 3)` o
  broadcastable).
- Si en A2 la variante `global` ganara a `precision` en métricas de
  segmentación: investigar si lo que aporta es contexto espacial extra (no
  gating) — en ese caso la respuesta correcta es profundizar el Swin/FPN, no
  volver a la atención global; documentar como ADR.
- `log_clamped(c) = log(c.clamp_min(1e-4))` — certeza 0 no debe producir
  −inf en el gate; el −9.2 resultante ya sesga suficientemente.
