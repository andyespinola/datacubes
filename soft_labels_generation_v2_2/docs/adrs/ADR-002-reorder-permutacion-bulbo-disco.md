# ADR-002 — Asignación de roles GMM por permutación (fix inversión bulbo/disco)

> Fecha: 2026-07-04 · Estado: aceptado · Reemplaza: la regla de
> reordenamiento de componentes descrita en `specs/11_classifier.md` (cambio
> #3, "alineado al artículo IBERAMIA") y en `_reorder_components` v2.1.
> Contexto: muestra de 94 galaxias TNG50 (snapshots 87/88/89) procesada con
> el pipeline v2.2. Evidencia completa en
> `reports/investigacion_inversion_bulbo_disco.md`.

## Contexto y problema

El clasificador (spec 11) ajusta un GMM de 3 componentes sobre el vector 4D
del artículo IBERAMIA `(ε, log₁₀(R/R_eff+δ), |z|/R_eff, E_norm)` y luego
**asigna un rol** (bulge / disk / halo) a cada componente. La regla v2.1
(el "reordenamiento corregido" del artículo) era secuencial:

```
disk_k  = argmax_k( ε_medio[k] )                 # disco = el de mayor ε
resto   = los otros dos componentes
bulge_k = el de resto con menor E_norm           # bulbo = el más ligado
          (con log₁₀(R/R_eff) como desempate si |ΔE_norm| < 0.05)
halo_k  = el restante
```

**Defecto:** esta regla invierte las etiquetas de bulbo y disco cuando el
componente central compacto tiene un ε medio comparable o mayor que el del
disco extendido — algo que ocurre en dos situaciones frecuentes:

1. **Empate de ε** entre el componente central y el disco (p. ej.
   TNG50-88-312423: ε=0.41 vs 0.41). El `argmax` desempata por orden
   arbitrario del GMM y corona al componente CENTRAL como "disco"; entonces
   "bulbo = el más ligado del resto" recae forzosamente sobre el disco
   extendido real. **Una sola desambiguación arbitraria invierte ambas
   etiquetas.**
2. **Pseudo-bulbo rotante** (bulbo secular): el componente central tiene ε
   medio MAYOR que el disco (p. ej. TNG50-89-346164: ε=0.37 vs 0.10). La
   regla secuencial basada solo en ε es estructuralmente incapaz de
   distinguir "disco extendido" de "componente central compacto con algo de
   rotación".

**Prevalencia medida:** 22 de 94 galaxias (23%) a nivel de partículas;
17 de 94 tras proyectar a espaxels (métrica robusta ponderada por
probabilidad). El defecto es determinista (asignación de roles), no
estadístico (el ajuste GMM en sí es correcto en los tres casos).

**Verificación de que NO es proyección ni máscara:** el patrón existe ya a
nivel de partículas (Fase A), antes de proyectar; las simetrías D4 no lo
explican. El centro de las galaxias invertidas es cinemáticamente CALIENTE
(ε≈0, alta σ*, alta Σ*) pero quedaba etiquetado "disco".

## Decisión

Sustituir la asignación secuencial por **asignación conjunta de los tres
roles por permutación** (3! = 6 opciones), maximizando una puntuación por
rol sobre las medias GMM estandarizadas ENTRE los tres componentes de cada
galaxia:

```python
# medias GMM en espacio original, estandarizadas entre los 3 componentes:
#   m[k] = (means[k] - means.mean(0)) / means.std(0)   # columnas: ε, logR, |z|, E
s_bulge(k) = -m_logR[k] - m_E[k]              # compacto y ligado; SIN ε
s_disk(k)  =  m_ε[k] + m_logR[k] - m_z[k]     # rotante, extendido, delgado
s_halo(k)  =  m_z[k] - m_ε[k]                 # grueso, no rotante
asignación = argmax_{(b,d,h) ∈ permutaciones} [ s_bulge(b) + s_disk(d) + s_halo(h) ]
```

Tres propiedades que resuelven el defecto:

1. **Sin empates arbitrarios:** decide el conjunto de los tres roles a la
   vez, no un `argmax` secuencial.
2. **"bulbo" ya no exige ε bajo:** el bulbo se define por compacidad
   (log R menor) y ligadura (E menor), no por dispersión — el pseudo-bulbo
   central rotante queda correctamente en bulbo. Semántica observacional
   (concentración central de luz), consistente con MORDOR (Zana 2022) y con
   la inferencia MaNGA/GZ3D.
3. **"disco" exige extensión**, no solo rotación (término +log R).

Además se añade un **gate de QA radial** en `run_classifier`: si el radio
medio ponderado del bulbo supera al del disco (`r_bulge > r_disk/0.9`) se
emite `radial_inversion_flag=True` y un warning — habría detectado el 100%
de estos casos.

Se conserva la regla v2.1 solo para el feature-set `standard3d` (que no
tiene columnas de R/z para la nueva puntuación).

## Consecuencias

- **Corrección validada:** 17 → 0 inversiones en las 94 (0 regresiones,
  p < 10⁻⁶ binomial). Verificado con la métrica robusta, con el gate de
  producción, y visualmente (bulbo sobre el pico de σ*/Σ* en 312423, 340908,
  346164).
- **Contrato de Fase A modificado:** el orden de columnas de `P_class`
  cambia en las 25 galaxias donde la permutación difiere. La rama se registra
  como `reorder_rule_branch = "permutation_v2.2"` en `quality`.
- **Impacto aguas abajo:** BarDetector y ArmDetector operan sobre `P_disk`;
  en las galaxias antes invertidas ahora buscan barra/brazo sobre el DISCO
  correcto (antes lo hacían sobre el esferoide central). Por eso el reproceso
  fue completo (Fase A→B→C), no una simple permutación de columnas del entry
  final.
- **El artículo IBERAMIA debe revisarse:** su "reordenamiento corregido"
  (que resolvía bulbo/halo) es el origen de la inversión bulbo/disco. Ver
  `reports/revision_articulo_IBERAMIA.md`.

## Alternativas consideradas

- **Post-proceso sobre el entry final** (permutar columnas de Y): descartado
  — no recupera barra/brazo, que se tallaron sobre el componente equivocado.
- **Corte de energía por galaxia estilo MORDOR** (E_cut): más complejo, y la
  puntuación por permutación ya resuelve el caso con las medias GMM ya
  calculadas, sin re-ajustar.
- **auto-GMM con fusión jerárquica** (Du 2019/2020): sobre-ingeniería para
  el problema puntual; la asignación conjunta es el fix mínimo suficiente.

## Reproducibilidad

- Fix: `phase_a/classifier.py::_reorder_components` + test dorado
  `tests/unit/test_reorder_v22.py` (medias GMM reales de 312423/346164/464534).
- Prototipo (94 galaxias, sin re-ajustar GMM): `scripts/prototype_reorder_fix.py`.
- Cuantificación: `scripts/quantify_inversion.py` → `output/inversion_bulge_disk.csv`.
- Comparación pre/post: `scripts/compare_pre_post.py` +
  `scripts/report_pre_post.py` → `output/comparacion_pre_post_fix.{csv,pdf}`.
- Respaldo pre-fix: `/media/andy/Data/tng/mangia_flat_pre_fix_20260703/` (40 GB).
