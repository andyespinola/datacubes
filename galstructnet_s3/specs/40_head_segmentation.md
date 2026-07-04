# 40 — Cabezas de segmentación (v3): Dirichlet dual con techo de certeza

> Módulo: `models/heads/segmentation.py` · Hito: 4 · Depende de: decoder
> (30 v3), capas EPN ([45 §P4](45_evidence_layers.md)).
> Cambios v3: dos cabezas (masa/luz, corrección C3), evidencia con techo de
> certeza, semántica explícita de S. La `DirichletSegHead` v2 se conserva
> como baseline A0.

## Responsabilidad

Producen, por spaxel, los parámetros `α` de una distribución de Dirichlet
sobre las 5 clases — una cabeza para el target ponderado por **masa** y otra
por **luz**. La media `α/S` es la segmentación probabilística; la
concentración `S = Σα` es la confianza, **con unidades**: pseudo-conteos
comparables a `κ·N_eff` (la pérdida anclada de 50 v3 los alinea).

## Por qué dos cabezas (y no una con dos targets)

La v2 aplicaba `dirichlet_nll(alpha, Y_light)` **y** `dirichlet_nll(alpha,
Y_mass)` sobre el mismo `alpha` (C3). Los dos targets difieren
sistemáticamente (la luz sobre-pesa brazos jóvenes; la masa, el bulbo viejo):
un α único se ve forzado a un promedio que no es ninguno y que infla la
incertidumbre aleatoria artificialmente. v3: tronco compartido, dos
proyecciones finales. La **discrepancia entre ambas cabezas** (`JS(p_mass ‖
p_lum)` por spaxel) se exporta como producto científico: mapa de dónde la
estructura en masa y en luz difieren.

## Contratos

```python
# Entrada
hidden: (B, 256, H, W)        # del decoder
c_dec:  (B, 1, H, W)          # certeza propagada (pasa-través en A0/Std)

# Salida (por target t ∈ {mass, lum})
{
  "alpha_t":    (B, 5, H, W)  ≥ 1,
  "prob_t":     (B, 5, H, W)  = α/S,
  "evidence_t": (B, 5, H, W)  = α − 1,
  "vacuity_t":  (B, 1, H, W)  = K/S        # diagnóstico; NO es "la epistémica"
}                                          # (descomposición correcta: 42 v3)
```

## Algoritmo

Implementación de referencia: `EvidenceHead` de [45 §P4](45_evidence_layers.md)
(`e = softplus(Conv1×1(hidden)) ⊙ h(c̄_dec)`, `h(u) = softplus(a·log u + b)`,
`α = 1 + e`). Este spec fija la composición:

```python
class DualSegHeads(nn.Module):
    def __init__(self, in_ch=256, n_classes=5, share_gate=True):
        super().__init__()
        self.head_mass = EvidenceHead(in_ch, n_classes)
        self.head_lum  = EvidenceHead(in_ch, n_classes)
        if share_gate:                       # (a, b) compartidos — ablar
            self.head_lum.a = self.head_mass.a
            self.head_lum.b = self.head_mass.b

    def forward(self, hidden, c_dec):
        return {"mass": self.head_mass(hidden, c_dec),
                "lum":  self.head_lum(hidden,  c_dec)}
```

- **Baseline A0**: `DirichletSegHead` v2 (softplus directo, sin techo de
  certeza), duplicada igualmente a dos cabezas. La dualidad es corrección,
  no innovación: aplica a todos los niveles de la escalera.
- **Init de `b`**: `scripts/init_evidence_scale.py` lo calibra para que
  `S` inicial ≈ `K + κ·mediana(N_eff_train)` — el modelo no arranca a órdenes
  de magnitud del ancla.

## Semántica de S (lo que el capítulo de la tesis afirma)

- `S → K` (evidencia 0): ignorancia; ocurre por construcción cuando
  `c_dec → 0` (test 10 de 45: ignorancia instrumental ⇒ ignorancia del
  modelo).
- `S ≈ κ·N_eff`: el modelo afirma tanta estadística como las partículas que
  generaron la etiqueta — el régimen sano tras entrenar (métrica
  `ρ_Spearman(S, N_eff)` en 70 v3).
- `S ≫ κ·N_eff`: sobreconfianza; la KL anclada lo penaliza directamente
  (no hace falta el KL→uniforme de v2 — eliminado, ver 50 v3).

## Validación

### Tests unitarios (`tests/unit/test_seg_heads.py`)

1. **Shapes** de los 8 tensores (2 targets × 4 salidas), H,W dinámicos.
2. **Invariantes**: `α ≥ 1`; `prob` suma 1 (`atol=1e-6`); `vacuity ∈ (0,1]`.
3. **Vacuity bajo ignorancia**: `c_dec = 0` ⇒ `α = 1` exacto, prob uniforme
   (45 §10).
4. **Monotonía en certeza**: con `hidden` fijo, `S` no decrece en `c_dec`
   (45 §11).
5. **Caso degenerado**: pesos a cero ⇒ prob ≈ uniforme.
6. **Independencia de cabezas**: gradiente de `L(alpha_mass)` no actualiza
   `head_lum.proj` (sí el tronco).
7. **Determinismo en eval; gradientes finitos.**

## Criterios de aceptación

- [ ] Tests 1–7 pasan (A0 y EPN).
- [ ] Forward despreciable (< 5 ms batch 4).
- [ ] `(a, b)` de cada cabeza persisten con nombre en el checkpoint (se
      interpretan en la tesis).
- [ ] Mapa de discrepancia masa/luz (`JS`) implementado en `evaluation/`.

## Notas de implementación

- Visualización: clase = `argmax(prob)`, con saturación reducida proporcional
  a la incertidumbre **total** (42 v3), no a vacuity — nota v2 corregida.
- Si tras Etapa 2 `S` queda plana (no trackea N_eff): primero revisar κ y el
  cap de N_eff (50 v3), después `share_gate=False`.
- No añadir capas aquí: la expresividad vive en el tronco; la cabeza es la
  interfaz probabilística.

## Referencias

- Sensoy et al. 2018 (EDL original — baseline A0).
- Revisión 2026-06-12 §3.1 (anclaje), C3 (dualidad); 45 §P4.
