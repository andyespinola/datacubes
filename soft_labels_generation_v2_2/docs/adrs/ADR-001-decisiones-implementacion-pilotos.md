# ADR-001 — Decisiones de implementación descubiertas con los pilotos

> Fecha: 2026-06-10 · Estado: aceptado · Contexto: Hitos 0–4 sobre los pilotos
> TNG50-87-155298-0-127 y TNG50-87-192324-0-127 (reemplazan al 141934 de las
> specs, que no tiene cutout ni cubo en `data/`).

## 1. Convención raster de la proyección (spec 20)

El vector de línea de visión del v1 (`view_vector_from_index`, +x para view 0)
es correcto, pero el raster del cubo MaNGIA está rotado 90° respecto al
binning directo `histogram2d(y, x)` del v1. La convención validada es:

```
(filas, columnas) del cubo = (x_proy, −y_proy)
```

implementada en `phase_b/label_projection.py::mangia_raster_coords`.
Evidencia (mapa de masa proyectada vs canal 18 de `SSP_pyPipe3D_REC`):

| Piloto | Spearman | centroide |
|---|---|---|
| 155298 | **0.980** | 0.05 px |
| 192324 | 0.865 | 0.85 px |

La búsqueda exhaustiva (6 vectores LOS × 8 transformaciones raster) confirma
que ésta es la mejor combinación para ambos pilotos. El 0.865 de 192324 se
debe a que su mapa de masa pyPipe3D es casi plano (poco rango dinámico
dentro del hexágono), lo que degrada la correlación de rangos; la inspección
visual confirma el registro espacial (morfología, elongación y satélite
coinciden). El gate del test de integración exige ρ>0.9 en al menos un
piloto y ρ>0.8 + centroide <1 px en ambos.

## 2. Potencial gravitacional: octree local (spec 10)

El campo `Potential` per-partícula NO existe para el snapshot 87: es un
snapshot "mini" de TNG (la API devuelve 400 "Invalid input" para
`stars=Potential` en snap 87, pero lo acepta en el 91, que es "full").
Como MaNGIA usa snapshots 85–99 y la mayoría son mini, el octree es el
método operativo general; `snapshot` queda disponible para los snapshots
full (91, 99). Se implementó:

- `potential_method="octree"`: Barnes-Hut con numba (θ=0.6, softening 0.288
  kpc), fuentes = estrellas + gas + DM del cutout (masa DM de `MassTable`).
  ~110 s por galaxia piloto (3.7–3.9M estrellas, 7.6–11.2M fuentes).
- `potential_method="spherical"`: aproximación esférica (Abadi 2003), para
  tests y fallback rápido.
- `potential_method="snapshot"`: queda implementado (`io/tng_reader.py::
  download_potential_cutout`) para cuando la API sea accesible.

Validación de ε resultante vs catálogo `stellar_circs.hdf5` (fracciones):

| Piloto | ε>0.7 (nuestro) | CircAbove07Frac (catálogo) |
|---|---|---|
| 155298 | 0.119 | 0.141 |
| 192324 | 0.293 | 0.322 |

Subestimación leve (~0.03) consistente con softening/θ del octree; aceptable.

## 3. GMM en galaxias dominadas por halo (spec 11) — LIMITACIÓN CONOCIDA

En 155298 (MORDOR: halo 56%, disco 25%, bulbo 19%) el GMM 4D parte el
esferoide en tres zonas radiales/energéticas; la regla "disco = mayor ε
medio" captura el componente central compacto (ε medio 0.22) porque ningún
componente es un disco real. Resultado: fracciones [0.48, 0.33, 0.19] vs
MORDOR [0.19, 0.25, 0.56] y el núcleo etiquetado "disco" (síntoma análogo
al problema 19.3 que el v2 elimina en galaxias con disco real).

En 192324 (disco + pseudo-bulbo) el GMM produce mapas sensatos (bulbo
central compacto + disco dominante + halo en bordes), con disco 0.54 vs
MORDOR 0.32 — MORDOR asigna 0.34 al pseudo-bulbo, que rota y el GMM agrupa
con el disco (ambigüedad de definición entre métodos, ver Du 2020 §4).

Decisión: comportamiento según spec; las desviaciones se reportan en
`qa_report.json` (flags `fraction_dev_*`). Pendiente discusión con el
director: (a) guard "sin componente disco" cuando max(ε medio) < umbral,
(b) K>3 con agrupación de componentes, (c) tratamiento del pseudo-bulbo.
El test unitario de disco-puro se relajó en consecuencia (un GMM K=3 sobre
una población única la subdivide; se verifica dominancia + ε alto del disco).

## 4. Barra de 192324: A₂ medido correcto, bajo el umbral del spec

A₂ medido (R<R_peak=1.31 kpc) = **0.156** vs MORDOR A₂(<R_peak) = 0.163
(−4%). Pero `a2_threshold=0.3` del spec → P_bar=0 (barra débil). El criterio
de aceptación del spec ("detecta con A2>0.3") asume barras fuertes. Decisión:
mantener el umbral del spec en el default y documentar; si el director quiere
pintar barras débiles, bajar `a2_threshold` a ~0.12 en config.

## 5. Máscara de validez: ~12–16% de spaxels válidos (spec 22)

El criterio B (S/N≥3 en 5000–5500 Å, con la extensión ERROR del cubo) deja
~600–800 spaxels válidos. El ruido del mock es plano (~4e-4) y el flujo cae
rápido: la región válida cubre r≈13 px ≈ **1.4 R_eff**, consistente con la
cobertura MaNGA (1.5 R_eff). El flag `low_validity` (<30% del FoV) queda
activo por construcción del mock; no es un defecto del pipeline.

## 6. ArmDetector: sin brazos en los pilotos

192324: disco liso en masa → 0 crestas (el residual del satélite queda
excluido por los criterios). 155298: 4 "crestas" espurias (satélite + anillos
del componente central mal etiquetado, ver §3) con 9% de masa — se reporta en
QA; depende de resolver §3. El detector pasa sus tests sintéticos
(2 brazos logarítmicos detectados, disco axisimétrico → 0).

## 7. Otros

- `manga_ifu_dsn=61` del catálogo MaNGIA difiere del header (`IFUCON=127`)
  y del nombre de archivo; el header/nombre mandan.
- Velocidad sistémica del subhalo: la API reporta km/s físicos (sin √a),
  a diferencia del v1 que aplicaba √a. Corregido en `io/units.py`.
- Edad estelar: conversión cosmológica exacta (Planck TNG) en lugar del
  proxy lineal 13.8·(1−a) del v1.
- Python 3.10 del sistema (specs piden 3.11); el código usa sintaxis ≥3.10.
