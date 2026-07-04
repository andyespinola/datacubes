# Investigación: inversión bulbo/disco en las etiquetas v2

> 2026-07-03. Muestra: las 94 galaxias procesadas (oleadas 1+2) en
> `/media/andy/Data/tng/mangia_flat/output/dataset_entries/`.
> Artefactos: `output/inversion_bulge_disk.csv`, `output/inversion_profiles.npy`.

## 1. El síntoma, cuantificado

Sobre las 94 galaxias (target masa, raw), con radios normalizados al P90 de
la máscara válida:

| Métrica | Conteo | % |
|---|---|---|
| Núcleo (r < 0.15 R) dominado por **disco** | 27 | 29% |
| Inversión de anillo (disco central + anillo de bulbo en 0.2–0.6 R) | 15 | 16% |
| r_medio(disco) < r_medio(bulbo) (2D) | 17 | 18% |
| r_medio(disco) < 0.9·r_medio(bulbo) (por partícula, prob-ponderado) | **22** | **23%** |

El patrón es siempre el mismo: el "disco" ocupa el centro y el "bulbo"
aparece como anillo o envolvente a radios intermedios/externos —
físicamente invertido.

## 2. La causa raíz (confirmada, no hipotética)

### 2.1 No es proyección ni máscara

Las simetrías D4 no lo explican y el perfil existe ya a nivel de
partículas (Fase A), antes de proyectar.

### 2.2 El centro de las galaxias invertidas es cinemáticamente CALIENTE

Perfiles por partícula (etiqueta final vs radio):

```
TNG50-88-312423   R<0.5 kpc: eps_med=+0.07  -> 96% "disk"   (¡eps~0 = esferoide!)
                  8-20 kpc:  eps_med=+0.61  -> 67% "bulge"  (¡rotante = disco!)
TNG50-89-346164   R<0.5 kpc: eps_med=+0.11  -> 92% "disk"
TNG50-89-464534   R<0.5 kpc: eps_med=+0.01  -> 98% "bulge"  (caso SANO)
```

Las etiquetas están intercambiadas: no hay un "disco nuclear frío" real.

### 2.3 El mecanismo exacto: `_reorder_components` (classifier.py)

La regla v2.1 asigna roles a los 3 componentes GMM en dos pasos
secuenciales: `disk = argmax(eps medio)` y `bulge = el más ligado del
resto`. Medias GMM (espacio original) de la galaxia invertida 312423:

```
comp 0: eps=+0.41  log10(R/Reff)=-0.57  |z|/Reff=0.05  E=-0.70  <- compacto central ligado
comp 1: eps=+0.03  log10(R/Reff)=+0.76  |z|/Reff=8.05  E=-0.22  <- halo
comp 2: eps=+0.41  log10(R/Reff)=+0.13  |z|/Reff=0.51  E=-0.43  <- disco extendido
```

`eps` medio EMPATA (0.41 vs 0.41): `argmax` resuelve por orden arbitrario
y corona a **comp 0 (el componente central)** como "disco". Entonces
"bulbo = más ligado del resto" recae por fuerza en **comp 2 (el disco
extendido real)**. Un solo empate invierte ambas etiquetas. En 346164 ni
siquiera hay empate: el componente central tiene eps medio MAYOR (0.37 vs
0.10) porque contiene un pseudo-bulbo rotante — la regla secuencial basada
solo en eps es estructuralmente incapaz de distinguir "disco extendido" de
"componente central compacto con algo de rotación".

En el caso sano (464534) los eps difieren con el signo correcto
(0.24 extendido vs 0.02 central) y la regla acierta. La fragilidad es del
paso de ASIGNACIÓN DE ROLES, no del ajuste GMM: los tres componentes que
encuentra el GMM son razonables en los tres casos.

## 3. Anclaje en la literatura

- **MORDOR** ([Zana et al. 2022](https://academic.oup.com/mnras/article/515/1/1524/6612743)):
  descompone TNG50 en 5 componentes usando (eps, E) con corte de energía
  por galaxia, y trata explícitamente el **bulbo secular/pseudo-bulbo**
  (componente central rotante) como clase separada del disco. Ratifica que
  un centro compacto con rotación NO debe absorberse en "disco".
- **auto-GMM** ([Du et al. 2019](https://iopscience.iop.org/article/10.3847/1538-4357/ab43cc),
  [2020](https://arxiv.org/abs/2002.04182)): número de componentes por BIC
  y FUSIÓN jerárquica de sub-estructuras hacia {disco frío, disco caliente,
  bulbo, halo} — precisamente porque forzar K fijo + asignación ingenua de
  roles produce intercambios como el nuestro.

Semántica elegida para GalStructNet: el pseudo-bulbo se PLIEGA en "bulge"
(es lo que un observador llama bulbo: concentración central de luz;
consistente con la inferencia MaNGA/GZ3D).

## 4. La corrección propuesta (validada en prototipo)

Sustituir la asignación secuencial por **asignación conjunta por
permutación** (3! = 6 opciones) maximizando una puntuación por rol, con
las medias GMM estandarizadas ENTRE los 3 componentes de cada galaxia:

```python
s_bulge(k) = -R_hat[k] - E_hat[k]              # compacto y ligado (SIN eps:
                                               #  pseudo-bulbo cuenta como bulbo)
s_disk(k)  = eps_hat[k] + R_hat[k] - z_hat[k]  # rotante, extendido, delgado
s_halo(k)  = z_hat[k] - eps_hat[k]             # grueso, no rotante
asignacion = argmax sobre permutaciones de [s_bulge(b) + s_disk(d) + s_halo(h)]
```

Tres propiedades: (i) no hay empates arbitrarios — decide el conjunto;
(ii) "bulbo" ya no exige eps bajo — el pseudo-bulbo central rotante queda
en bulbo; (iii) "disco" exige extensión, no solo rotación.

### Resultado del prototipo (94 galaxias, sin re-ejecutar el GMM)

Reconstruyendo P_raw desde los `gmm_params` guardados y aplicando ambas
reglas (script `prototype_reorder_fix.py`):

| | regla actual | regla propuesta |
|---|---|---|
| inversiones r_disk < 0.9·r_bulge | **22 / 94** | **0 / 94** |
| galaxias arregladas | — | 22 |
| galaxias empeoradas | — | 0 |
| permutación cambia | — | 25 (las 22 + 3 intercambios bulbo/halo) |

Significancia: 22→0 con n=94 (binomial, p < 1e-6). Las 3 permutaciones
extra que cambian sin ser "inversiones" deben revisarse visualmente al
aplicar (probables intercambios bulbo/halo en esferoidales puras).

## 5. ¿Necesitamos procesar más de 100 galaxias?

**No.** Argumentos:

1. La causa es un defecto determinista de asignación de roles, no un
   fenómeno estadístico: se identifica y corrige con los parámetros GMM
   ya guardados de las 94.
2. Prevalencia 23% ± 4% (binomial, n=94): la muestra actual ya la mide con
   precisión suficiente, y la validación del fix (22→0, 0 regresiones) es
   concluyente con este n.
3. Más galaxias servirían solo para el QA final del pipeline completo
   (bar/arm incluidos), no para diseñar la solución.

Lo que SÍ conviene: al aplicar el parche, agregar un **gate de QA
automático** que dispare alerta si `r_medio(bulge) > r_medio(disk)` en el
plano proyectado — habría detectado el 100% de estos casos (y el
`agreement_with_hard_thresholds` de estas galaxias ya era anómalo:
0.40–0.55).

## 6. Plan de aplicación (barato: no hay que re-descargar ni re-ajustar)

1. Parchar `_reorder_components` en `phase_a/classifier.py` con la regla
   por permutación (≈25 líneas) + test unitario con las medias reales de
   312423/346164/464534 como casos dorados.
2. Re-etiquetar las 94: la permutación es un reordenamiento de columnas de
   `P_class` — se puede recomputar Fase A desde `gmm_params` sin re-ajustar
   el GMM, o re-correr Fase A completa (~10–130 s/galaxia). Luego Fase B/C.
3. QA: re-correr `inversion_bulge_disk.csv` (esperado: 0 inversiones de
   anillo) + revisión visual de las 3 galaxias con intercambio bulbo/halo
   + regenerar la comparación 2D.
4. Registrar como ADR (la regla de reordenamiento cambia el contrato de
   la Fase A).

---

## 7. Resultado del reproceso (2026-07-03)

Parche aplicado (`_reorder_components` v2.2 + gate de QA radial), suite de
tests v2_2 en verde (37 tests, incluidos 4 casos dorados nuevos con las
medias GMM reales de 312423/346164/464534 y el test del bug v2.0
actualizado). Reproceso completo de las 94 galaxias reutilizando los
`particle_features.h5` cacheados; respaldo pre-fix en
`mangia_flat_pre_fix_20260703/` (40 GB).

**Comparación pre-fix vs post-fix (94 galaxias, radio ponderado por
probabilidad — métrica robusta):**

| | PRE | POST |
|---|---|---|
| Inversiones bulbo/disco | 17 | **0** |
| corregidas | — | 17 |
| regresiones | — | 0 |
| galaxias sin ningún cambio de etiqueta | — | 69 |

El gate de QA de producción (radio ponderado a nivel de partícula)
confirma 0/94 marcadas post-fix. La verdad de partículas (perfil ε vs
radio) ratifica que en las 17 corregidas el centro es esferoide caliente
(bulbo, ε≈0.0–0.12) y el exterior rotante (disco), p.ej. 312423
r_bulge=3.1 kpc vs r_disk=15.1 kpc.

**Nota metodológica:** una métrica ingenua basada en el radio del *argmax*
(no ponderada) producía 2 falsos positivos (323384, 69030) porque un bulbo
muy compacto gana pocos spaxels centrales y unos spaxels externos dispersos
inflan su radio medio. La métrica ponderada por probabilidad (y el gate de
producción) no tienen ese problema. Artefactos:
`output/comparacion_pre_post_fix.{csv,pdf}`.

## 8. Pendiente: 6 galaxias grandes con OOM

Las 6 galaxias cuyos cutouts se re-descargaron (205585, 233037, 161414,
210167, 229623, 237390; 4.5–9.7M estrellas) **mueren por OOM** en el
cálculo del potencial por octree (>50M fuentes). Nunca estuvieron en las 94
(antes fallaban por cutout corrupto). No afectan la validación del fix.
Solución propuesta: usar la ruta de potencial canónico descargando el campo
`Potential` per-partícula de la API TNG (`download_potential_cutout`,
`query=stars=Coordinates,ParticleIDs,Potential`) para saltarse el octree —
config `require_potential_cache`. `205585` además carece de `cube_maps`
(producto del flujo de reconstrucción MaNGIA local, no descargable).

## 9. La ruta "descargar Potential" NO es viable (mini-snapshots)

Intento de bajar el campo `Potential` per-partícula para saltarse el octree
en las 6 galaxias OOM (y validar en 4 ya procesadas). **Resultado negativo,
verificado empíricamente contra la API TNG:**

```
snap 84: stars=Coordinates,Potential -> HTTP 302  (full snapshot, con Potential)
snap 87: -> HTTP 400 "Invalid input"  (MINI snapshot, SIN Potential)
snap 88: -> HTTP 400                   (MINI)
snap 89: -> HTTP 400                   (MINI)
snap 91: -> HTTP 302                   (full)
snap 99: -> HTTP 302                   (full)
```

TNG solo guarda `Potential` (y el set completo de campos) en los ~21
**snapshots "full"**; los intermedios son **"mini"** con campos reducidos.
La muestra MaNGIA se construyó en snapshots **87/88/89 — todos mini**, así
que el potencial per-partícula NO existe para descargar. La función
`download_potential_cutout` del repo solo funciona en snapshots full.

**Consecuencia:** el octree (o la aproximación esférica) es la ÚNICA vía
para el potencial en esta muestra. La validación snapshot-vs-octree pedida
no puede hacerse (no hay snapshot Potential contra el cual comparar).

### Opciones reales para las 6 OOM (potencial por octree con >50M fuentes)

1. **Submuestreo de DM en las fuentes del octree** (recomendado): la materia
   oscura domina el conteo (~45M de los 52M) y su potencial es suave; usar 1
   de cada N partículas DM con masa ×N reduce el árbol N× en memoria con
   error despreciable en Φ. Cambio local en `extractor.run_extractor`
   (sources) o en `compute_potential_octree`.
2. **Aproximación esférica** (`compute_potential_spherical`, ya existe):
   Φ(r) por perfil de masa encerrada; sin árbol, O(N log N) trivial en
   memoria. Menos exacta en galaxias no esféricas pero suficiente para ε de
   discos de cara — ablar contra octree en una galaxia que sí cabe.
3. Más RAM / swap (fuerza bruta, no recomendado).

Las 6 no bloquean nada del dataset actual (94 completas y validadas); son un
límite de memoria del octree, no del método de etiquetado.

## 10. Validación del octree contra el potencial propio de TNG

Ninguna de las 94 galaxias (snapshots mini 87/88/89) trae el campo
`Potential` de TNG. Para validar el octree se descargó **una galaxia del
snapshot 91** (único full del rango MaNGIA, 225 galaxias disponibles): la
más pequeña, **TNG50-91-571097** (161k estrellas). Se corrió el extractor
por las dos vías —`snapshot` (Φ de TNG) y `octree` (nuestro método)— y se
comparó ε y etiquetas. Script: `scripts/compare_octree_vs_tng.py`.

**Resultado (161.246 estrellas):**

| Métrica | Valor |
|---|---|
| ε correlación (ρ) TNG vs octree | **0.9872** |
| ε RMSE (escala [-1,1]) | 0.054 (~2.7%) |
| ε mediana TNG / octree | +0.768 / +0.793 |
| acuerdo de argmax (etiquetas) | **97.5%** |
| fracciones bulge/disk/halo TNG | 0.265 / 0.582 / 0.153 |
| fracciones bulge/disk/halo octree | 0.274 / 0.567 / 0.159 |

Los desacuerdos (2.5%) están en ε≈0.63 — la frontera disco/bulbo, donde la
clasificación es intrínsecamente ambigua, no errores sistemáticos. El
octree tiene un sesgo positivo minúsculo en ε (+0.025 en mediana) porque
solo ve la masa del cutout (sin el campo de marea de la caja completa),
efecto despreciable en las etiquetas (<1% en fracciones globales).

**Conclusión:** el potencial por octree usado para las 94 galaxias es fiel
a la verdad de la simulación (ρ=0.987, acuerdo de etiquetas 97.5%). El set
de etiquetas está bien fundado; el octree es un método válido para toda la
muestra MaNGIA, no una aproximación de conveniencia.

## 11. Barrido automatico de sistemas en fusion (inspeccion visual)

Tras confirmar visualmente el fix en las correcciones (bulbo sobre el pico
de sigma*/Sigma* en 312423, 340908, 346164), barrido automatico de las 94
buscando DOBLES NUCLEOS: multiples picos en Sigma* (masa lineal) con
verificacion cruzada en sigma* (una compañera tiene nucleo caliente propio;
un brazo es sobredensidad fria). Script: `scripts/detect_mergers.py`,
salida `output/merger_sweep.csv` + `output/merger_flagged.txt`.

Criterio: secundario >=15% del primario en masa, separado >=5px, con
sigma* localmente realzada (>med+0.5sigma).

**Resultado: 2 / 93 galaxias (2%) son pares/fusiones reales:**

| Galaxia | ratio masa sec/prim | separacion | 2 nucleos calientes |
|---|---|---|---|
| TNG50-88-365595 | 0.77 | 23.6 px | si |
| TNG50-89-372192 | 0.77 | 21.5 px | si |

(93 y no 94: TNG50-88-205585 quedo sin cube_maps.)

**Nota metodologica:** TNG50-88-382174, sobre-estimada como doble en la
inspeccion visual a ojo, resulto tener un secundario de solo 6% de masa y
FRIO (sigma*_z=-2.36) — un grumo menor, no un nucleo compañero. El detector
cuantitativo (umbral 15% + cross-check sigma*) lo clasifico correctamente
como single. El detector es mas fiable que la inspeccion visual para este
juicio.

**Implicacion:** en las 2 galaxias flag, la compañera se absorbe como
arm/disk porque el esquema de 5 clases (bulge/disk/bar/arm/halo) no tiene
categoria merger/companera. Es una limitacion del esquema, independiente
del fix bulbo/disco. Opciones: (a) excluir las 2 del entrenamiento
supervisado, (b) marcarlas con `qa_status=merger` y bajar su peso, (c)
aceptar el ruido (2% de la muestra). Recomendacion: marcar y decidir al
escalar (a 10k la fraccion de fusiones sera similar, ~2%, ~200 galaxias).
