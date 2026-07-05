# Revisión del artículo IBERAMIA — cambios a incorporar por los fixes v2.2

> 2026-07-04. Documento de trabajo para reescribir el artículo del método de
> clasificación estructural (el "artículo IBERAMIA" que fundamenta
> `specs/11_classifier.md`). Reúne, sección por sección del paper, qué
> afirmación cambia, la evidencia nueva y las figuras a regenerar.
> Referencias técnicas completas: `investigacion_inversion_bulbo_disco.md`,
> `docs/adrs/ADR-002-reorder-permutacion-bulbo-disco.md`.

## 0. Resumen ejecutivo (para el abstract / conclusiones)

El método publicado clasifica partículas estelares de galaxias simuladas
(TNG50) en bulbo/disco/halo mediante un GMM de 3 componentes sobre el vector
4D `(ε, log₁₀(R/R_eff+δ), |z|/R_eff, E_norm)`, con una regla de
reordenamiento de componentes para asignar los roles físicos. Al aplicar el
método a una muestra de 94 galaxias descubrimos que **esa regla de
reordenamiento invierte las etiquetas de bulbo y disco en el 23% de las
galaxias** (aquellas con bulbo compacto de rotación no nula o con empate de ε
entre el componente central y el disco). Corregimos la regla con una
**asignación conjunta de roles por permutación** y validamos el resultado
contra (i) el patrón pre/post, (ii) el potencial gravitatorio propio de la
simulación, y (iii) un barrido de sistemas en fusión. La corrección elimina
las inversiones (17→0 en el plano proyectado) sin regresiones, y la fidelidad
física del método queda confirmada de forma independiente.

## 1. El defecto en la regla de reordenamiento publicada

### 1.1 Qué dice el artículo

El artículo (spec 11, cambio #3) propone, tras ajustar el GMM, reordenar los
tres componentes así: **disco = el de mayor ε medio; bulbo = el más ligado
(E_norm menor) de los otros dos; halo = el restante** (con log₁₀(R/R_eff)
como desempate). Se presentó como corrección del problema bulbo↔halo del
método previo.

### 1.2 Por qué falla

La regla es **secuencial y basada en ε**. Falla en dos regímenes:

- **Empate de ε** entre el componente central compacto y el disco extendido:
  el `argmax` desempata por el orden arbitrario del GMM y puede coronar al
  componente CENTRAL como "disco"; entonces "bulbo = el más ligado del resto"
  cae sobre el disco real. Un solo desempate arbitrario invierte ambas
  etiquetas.
- **Pseudo-bulbo rotante** (bulbo secular): el componente central tiene ε
  medio MAYOR que el disco. La regla, que asume "más rotante = disco", falla
  por construcción.

Evidencia a nivel de partículas (medias GMM en espacio original, columnas
ε / log₁₀(R/R_eff) / |z|/R_eff / E_norm):

```
TNG50-88-312423 (EMPATE de ε):
  comp central : ε=+0.41  logR=-0.57  |z|=0.05  E=-0.70   → debe ser BULGE
  comp disco   : ε=+0.41  logR=+0.13  |z|=0.51  E=-0.43   → debe ser DISK
  → regla vieja: argmax(ε) arbitrario → central etiquetado "disk", disco "bulge"

TNG50-89-346164 (PSEUDO-BULBO rotante):
  comp central : ε=+0.37  ...  → la regla lo llama "disk" porque ε es mayor
  comp disco   : ε=+0.10  ...  → cae en "bulge"
```

Perfiles de ε(radio) confirman la inversión: en las galaxias afectadas el
centro (R<0.5 kpc) es cinemáticamente CALIENTE (ε≈0.0–0.1, alta σ*, alta Σ*)
pero quedaba etiquetado "disco", mientras el exterior rotante (ε creciente)
quedaba "bulbo".

### 1.3 Prevalencia

**23% de la muestra (22/94)** a nivel de partículas; **18%** (17/94) tras
proyectar a espaxels con la métrica robusta. No es un caso raro: afecta a
todas las galaxias con bulbo secular/pseudo-bulbo o con concentración central
de masa comparable en ε al disco.

## 2. La corrección

Reemplazar la asignación secuencial por **asignación conjunta por
permutación** (3! opciones), puntuando cada rol sobre las medias GMM
estandarizadas entre los tres componentes:

```
s_bulge(k) = -logR_hat[k] - E_hat[k]              # compacto y ligado; SIN ε
s_disk(k)  =  ε_hat[k] + logR_hat[k] - z_hat[k]   # rotante, extendido, delgado
s_halo(k)  =  z_hat[k] - ε_hat[k]                 # grueso, no rotante
asignación = argmax sobre permutaciones de la suma de los tres
```

Clave metodológica para el artículo: **el bulbo se define por compacidad +
ligadura, no por dispersión de velocidades** — así el pseudo-bulbo rotante
(que un observador MaNGA llama bulbo por su concentración de luz) queda
correctamente en la clase bulbo. Esto alinea el método con la semántica
observacional y con MORDOR (Zana et al. 2022), que trata el bulbo secular
como componente propio y nunca lo absorbe en disco.

## 3. Validaciones nuevas (material para la sección de resultados)

### 3.1 Pre-fix vs post-fix (94 galaxias)

| | PRE | POST |
|---|---|---|
| Inversiones bulbo/disco (radio ponderado por prob) | 17 | **0** |
| corregidas | — | 17 |
| regresiones | — | 0 |
| galaxias sin ningún cambio de etiqueta | — | 69 |

Significancia p < 10⁻⁶ (binomial). Las 69 sin cambio confirman que el fix
solo toca las galaxias con el defecto. Artefactos:
`output/comparacion_pre_post_fix.{csv,pdf}`.

### 3.2 Fidelidad del potencial: octree vs. potencial de TNG

El método calcula ε = j_z/j_c(E), y E = ½v² + Φ requiere el potencial Φ. En
los snapshots "mini" de TNG (donde vive el 97.8% de MaNGIA, ver §5) Φ no
está disponible y se calcula por **octree (Barnes-Hut)** sobre
estrellas+gas+DM. Para validar ese Φ, se procesó una galaxia del único
snapshot "full" del rango (91), que sí trae el `Potential` de TNG, por ambas
vías:

| Métrica (TNG50-91-571097, 161k estrellas) | Valor |
|---|---|
| ε correlación (ρ) octree vs TNG | **0.9872** |
| ε RMSE (escala [-1,1]) | 0.054 |
| acuerdo de etiquetas (argmax) | **97.5%** |

El octree reproduce el potencial de la simulación; las etiquetas de las 94
están bien fundadas en su base física. Los desacuerdos (2.5%) caen en la
frontera disco/bulbo (ε≈0.63), no son sistemáticos. Script:
`scripts/compare_octree_vs_tng.py`.

### 3.3 Sistemas en fusión (limitación honesta del esquema)

Barrido automático (Σ* doble pico + cross-check σ*): **2/93 galaxias (2%)**
son pares/fusiones reales (TNG50-88-365595, TNG50-89-372192), donde la
compañera se absorbe como brazo/disco porque el esquema de 5 clases no tiene
categoría merger. Es una limitación del esquema, no del fix. Script:
`scripts/detect_mergers.py` → `output/merger_sweep.csv`.

### 3.4 Patrón "pajarita" en sistemas inclinados/gruesos (2ª limitación honesta)

Barrido `detect_bowtie.py`: **3/92 galaxias (3%)** donde la proyección de la
membresía de componente 3D produce bulbo confiado FUERA del centro (disco
inclinado/de-canto o sistema grueso caliente). Las etiquetas 3D son
correctas; el mapa 2D está dominado por geometría de proyección. Es un límite
conocido de proyectar membresía cinemática (vs. brillo superficial). Refuerza
la elección de etiquetas blandas + incertidumbre. Detalle: `investigacion §12`.

## 4. Figuras a regenerar / añadir en el artículo

1. **Fig. inversión (mecanismo):** perfiles de ε(radio) y fracción de clase
   de una galaxia invertida, PRE vs POST (usar 312423). Muestra el centro
   caliente etiquetado disco (mal) → bulbo (bien).
2. **Fig. mosaico pre/post:** mapas argmax de las 17 corregidas lado a lado
   (ya generado: `output/comparacion_pre_post_fix.pdf`).
3. **Fig. evidencia física:** segmentación + σ*/Σ*/v* de 312423, 340908,
   346164 mostrando que el bulbo cae sobre el pico de σ*/Σ*.
4. **Fig. validación de potencial:** dispersión ε_octree vs ε_TNG
   (ρ=0.987) para la galaxia del snap 91.
5. **Tabla de resultados actualizada:** las fracciones por componente y las
   métricas de acuerdo del artículo deben recalcularse con el reordenamiento
   nuevo (las de las galaxias invertidas cambian).

## 5. Nota de infraestructura relevante para el método

MaNGIA usa snapshots TNG50 87–98; solo el 91 es "full" (con `Potential`). El
**97.8% de la muestra está en snapshots "mini" sin potencial per-partícula**,
por lo que el cálculo de ε depende del potencial por octree (validado en
§3.2). El artículo debe declarar que Φ se computa por octree para el grueso
de la muestra y que se validó contra el Φ de TNG en el subconjunto full.
Coste de memoria del octree: ~168 bytes/fuente (medido); la galaxia mayor
(102M fuentes) requiere ~23 GB de pico.

## 6. Afirmaciones del artículo que NO cambian

- El vector 4D y el uso del GMM de 3 componentes: correctos, se conservan.
- El prior/validación MORDOR (Zana 2022): correcto.
- `agreement` con el ε-threshold como métrica (no assert): correcto — de
  hecho el descubrimiento refuerza esto (las galaxias invertidas tenían
  agreement anómalamente bajo, 0.40–0.55, señal que el gate de QA ahora
  captura explícitamente).
- El ajuste GMM en sí: correcto en todos los casos; el defecto era solo la
  asignación de roles posterior.

## 7. Checklist para la reescritura

- [ ] Sección de método: reemplazar la regla de reordenamiento por la
      asignación por permutación (§2).
- [ ] Sección de resultados: añadir la validación pre/post (§3.1) y la del
      potencial (§3.2).
- [ ] Discusión: añadir la limitación de fusiones (§3.3) y la dependencia del
      octree + validación (§5).
- [ ] Recalcular todas las tablas de fracciones/acuerdo con el reordenamiento
      nuevo.
- [ ] Regenerar las figuras (§4).
- [ ] Abstract/conclusiones: incorporar el resumen (§0).
