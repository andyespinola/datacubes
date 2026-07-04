# 42 — Cabeza de incertidumbre (v3): descomposición por información mutua

> Módulo: `models/heads/uncertainty.py` · Hito: 4 · Depende de: cabezas de
> segmentación (40 v3). Sin parámetros entrenables.
> Cambios v3: la descomposición v2 era **incorrecta** (C1) — usaba la
> impureza de Gini de la media (una medida de incertidumbre TOTAL) como
> "aleatoria", vacuity como "epistémica", y un OR ad hoc. Todo eso se
> elimina. La pérdida `L_unc` asociada también desaparece (50 v3).

## Responsabilidad

Separa, desde los `α` de cada cabeza, los dos tipos de incertidumbre con la
descomposición estándar de teoría de la información sobre la Dirichlet:

```
TU = H[E[p]]            incertidumbre total      (entropía de la media)
AU = E[H[Cat(p)]]       aleatoria                (entropía esperada; forma cerrada)
EU = TU − AU            epistémica               (información mutua, ≥ 0)
```

`AU ≤ TU` siempre (Jensen: la entropía es cóncava), así que `EU ≥ 0` por
construcción — sin clamps semánticos.

### Por qué la v2 estaba mal (para el registro de la tesis)

`Σ p̄_c(1−p̄_c)` con `p̄ = α/S` es plana tanto cuando el spaxel es una mezcla
física real (aleatoria alta) como cuando el modelo es ignorante (α=(1,…,1),
epistémica alta): un spaxel OOD reportaba "aleatoria ≈ 1" siendo su
incertidumbre puramente epistémica. La descomposición por información mutua
separa exactamente esos dos casos: con α=(1,…,1), AU = ψ(K+1)−ψ(2) ≈ 1.28
nats y EU = ln K − AU ≈ 0.33 nats > 0; con α concentrada, ambas → 0.

## Contratos

```python
# Entrada
alpha: (B, K, H, W), todos ≥ 1        # de una cabeza (mass o lum)

# Salida (todas (B, 1, H, W), en nats; versión /ln(K) ∈ [0,1] para display)
{ "total": TU, "aleatoric": AU, "epistemic": EU,
  "vacuity": K/S }                    # diagnóstico de evidencia, aparte
```

## Algoritmo

```python
class UncertaintyDecomposition(nn.Module):
    """Formas cerradas sobre Dir(α). Sin parámetros."""

    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.K = n_classes

    def forward(self, alpha: Tensor) -> dict:
        S = alpha.sum(dim=1, keepdim=True)
        p = alpha / S

        # Total: entropía de la media
        TU = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1, keepdim=True)

        # Aleatoria: entropía esperada bajo la Dirichlet (forma cerrada)
        #   E[H] = −Σ_c (α_c/S)·(ψ(α_c + 1) − ψ(S + 1))
        AU = -(p * (torch.digamma(alpha + 1.0)
                    - torch.digamma(S + 1.0))).sum(dim=1, keepdim=True)

        # Epistémica: información mutua
        EU = (TU - AU).clamp_min(0.0)        # clamp solo numérico (≈1e-7)

        return {"total": TU, "aleatoric": AU, "epistemic": EU,
                "vacuity": self.K / S}
```

## Guía de interpretación (tabla para el astrónomo)

| Patrón | Lectura | Decisión |
|---|---|---|
| AU alta, EU baja | Mezcla física real (frontera disco/brazo, bulbo+barra superpuestos) | Tratar el spaxel como composición; no forzar clase única |
| EU alta (AU cualquiera) | El modelo no tiene respaldo: región fuera del soporte de entrenamiento, baja certeza instrumental, morfología rara | Revisión manual / excluir / candidato a re-etiquetado |
| Vacuity alta | Evidencia total baja (S≈K) — subconjunto de "EU alta" con causa instrumental probable (c̄ bajo) | Cruzar con mapas de certeza de entrada |
| Todo bajo | Predicción confiable | Usar directamente |

Visualización por defecto: `total` (normalizada /ln K) para el mapa general;
`EU` para auditoría de dominio (MaNGIA→MaNGA: EU debería subir en MaNGA donde
el gap importa); `AU` para el análisis de mezclas. La saturación de los mapas
de clase se reduce con `total` (corrige la nota v2 que usaba vacuity).

## Validación

### Tests unitarios (`tests/unit/test_uncertainty.py`)

1. **Shapes y rangos**: `TU ∈ [0, ln K]`, `AU ∈ [0, TU + 1e-6]`, `EU ≥ 0`,
   `vacuity ∈ (0, 1]`.
2. **Caso ignorante** `α = 1`: `TU = ln K`, `AU = ψ(K+1) − ψ(2)`
   (valor exacto, atol 1e-5), `EU = TU − AU > 0`, `vacuity = 1`.
3. **Caso concentrado** `α = (1000,1,1,1,1)`: `TU, AU, EU → 0`,
   `vacuity → 0`.
4. **Mezcla con mucha evidencia** `α = (500, 500, 1, 1, 1)`: `AU` alta
   (≈ ln 2), `EU ≈ 0` — el caso que la v2 no podía representar. Test clave.
5. **Jensen**: sobre α aleatorios, `AU ≤ TU` siempre.
6. **Sin parámetros; gradiente fluye hacia α; determinismo.**

## Criterios de aceptación

- [ ] Tests 1–6 pasan.
- [ ] Forward despreciable.
- [ ] `evaluation/` consume `EU` para el AUROC de detección de error
      (70 v3) y reporta las tres cantidades por estrato de N_eff.

## Notas de implementación

- Mantener todo en nats internamente; dividir entre `ln K` solo en
  visualización/reportes.
- No reintroducir pérdidas sobre estas cantidades: la incertidumbre se
  supervisa vía el ancla de 50 v3, no con regularizadores ad hoc (esa era la
  patología v2 — C2).

## Referencias

- Depeweg et al. (2018) ICML; Malinin & Gales (2018) NeurIPS — descomposición
  TU/AU/EU.
- Jürgens et al. (2024) ICML; Shen et al. (2024) NeurIPS — por qué vacuity
  como "epistémica" + KL→uniforme no es defendible (revisión C1/C2).
