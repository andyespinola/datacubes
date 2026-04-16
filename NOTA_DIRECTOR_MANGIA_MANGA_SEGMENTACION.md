# Nota Metodológica para Segmentación de Estructuras Galácticas con MaNGIA y MaNGA

## 1. Objetivo de esta nota

Esta nota reformula la propuesta metodológica del proyecto bajo una premisa más precisa:

- los mapas físicos 2D que se usarán como entrada no son mapas intrínsecos de simulación;
- son mapas derivados del cubo espectral mediante el mismo algoritmo de análisis, `pyPipe3D`, tanto para MaNGIA como para MaNGA.

Bajo esa premisa, el objetivo es evaluar si resulta metodológicamente sólido:

- preentrenar una red neuronal con cubos mock de MaNGIA;
- usar como entrada el espectro por spaxel, mapas físicos derivados y una imagen de la galaxia;
- y transferir luego el modelo a MaNGA para segmentación espacial de estructuras como bulbo, disco y brazos.

## 2. Verificación de la premisa

La premisa es válida, con una aclaración importante.

### 2.1. Lo que sí está verificado

La publicación de MaNGIA indica explícitamente que los cubos mock fueron procesados con `pyPipe3D`, siguiendo el mismo procedimiento usado para MaNGA.

La referencia principal es:

- Astronomy & Astrophysics, *MaNGIA: 10 000 mock galaxies for stellar population analysis*:
  https://www.aanda.org/articles/aa/full_html/2023/05/aa45509-22/aa45509-22.html

En el resumen accesible públicamente de esa publicación se indica que:

- produjeron cubos `MaNGA-like` para galaxias simuladas;
- los procesaron con el código de análisis estelar `pyPipe3D`;
- y así extrajeron mapas espaciales de historia de formación estelar, edad, metalicidad, masa y cinemática, siguiendo los mismos procedimientos usados en la release oficial de MaNGA.

También del lado observacional está verificado que MaNGA dispone de productos Pipe3D oficiales como catálogo de valor agregado.

Referencia oficial de SDSS:

- SDSS Value Added Catalogs, *MaNGA Pipe3D value added catalog: Spatially resolved and integrated properties of galaxies for DR17*:
  https://www.sdss.org/dr19/data_access/value-added-catalogs/

Allí se describe el VAC de MaNGA Pipe3D como un catálogo con propiedades de poblaciones estelares y líneas de emisión, medidas con el código de análisis IFU Pipe3D.

En consecuencia, sí es correcto afirmar que existen mapas físicos 2D derivados con el mismo algoritmo de análisis en ambos dominios:

- MaNGIA mock, analizado con `pyPipe3D`;
- MaNGA real, con productos Pipe3D oficiales.

### 2.2. La aclaración importante

En el entorno local actual, el archivo `cube_val` no corresponde a una salida de `pyPipe3D`.

El propio código local lo deja claro:

- [official_mangia/sin_ifu_clean.py](/home/andy/pythonprojects/cubes/official_mangia/sin_ifu_clean.py#L1858) describe `cube_val` como:
  “intrinsic and assigned values from the simulation”.
- [official_mangia/sin_ifu_clean.py](/home/andy/pythonprojects/cubes/official_mangia/sin_ifu_clean.py#L2231) genera esos mapas a partir de partículas estelares y propiedades asignadas de la simulación.

Además, el mismo código indica que el `cube.fits.gz` se genera con una máscara “necessary for pyPipe3D processing” en [official_mangia/sin_ifu_clean.py](/home/andy/pythonprojects/cubes/official_mangia/sin_ifu_clean.py#L1855), lo que implica que `cube` es la entrada preparada para Pipe3D, mientras que `cube_val` es otra cosa.

Por tanto, la premisa es válida si el proyecto usa:

- mapas derivados por `pyPipe3D` en MaNGIA;
- mapas derivados por `pyPipe3D` en MaNGA.

La premisa no sería válida si se mezclara:

- `cube_val` de MaNGIA;
- con mapas Pipe3D de MaNGA.

## 3. Reformulación correcta del problema

La versión metodológicamente consistente del proyecto es la siguiente:

- usar cubos espectrales `manga-like` de MaNGIA;
- procesarlos con `pyPipe3D`;
- obtener mapas físicos 2D derivados con el mismo algoritmo que se usa para MaNGA;
- construir entradas multicanal con:
  - espectro por spaxel,
  - mapas físicos derivados con `pyPipe3D`,
  - e imagen asociada;
- y entrenar una red de segmentación espacial que prediga la clase estructural de cada spaxel.

En esta formulación, los mapas físicos dejan de ser “información privilegiada de simulación” y pasan a ser:

- productos observacionales derivados;
- comparables entre mock y datos reales;
- y por tanto defendibles como entradas para aprendizaje por transferencia.

## 4. Qué aprende la red en este esquema

La tarea propuesta es una segmentación por spaxel.

Eso significa que, para cada posición espacial del cubo, la red debe predecir una etiqueta o probabilidad de pertenencia a una estructura, por ejemplo:

- bulbo;
- disco;
- brazos espirales;
- u otras clases morfológicas que se definan después.

Si el input contiene:

- el espectro completo de cada spaxel;
- mapas físicos 2D derivados con `pyPipe3D`;
- e imagen asociada;

entonces la red aprende simultáneamente:

- información espectral local;
- contexto espacial;
- correlaciones entre firma espectral y estructura galáctica;
- correlaciones entre propiedades físicas derivadas y morfología;
- y continuidad espacial entre regiones adyacentes.

Eso es especialmente valioso para distinguir:

- bulbo, típicamente más central y con propiedades estelares distintas;
- disco, más extendido y suave;
- brazos, más estructurados y localizados.

## 5. Por qué esta formulación es metodológicamente más sólida

Usar mapas derivados con el mismo algoritmo en ambos dominios mejora mucho la consistencia del problema.

La ventaja principal es que el modelo ya no se entrena con un conjunto de variables disponibles en simulación pero ausentes en observación. En su lugar, se entrena con observables derivados de forma homogénea.

Eso permite afirmar que la red recibe, en entrenamiento y en inferencia:

- variables físicamente análogas;
- construidas con el mismo tipo de pipeline;
- comparables en significado científico.

Esto no elimina por completo el desajuste de dominio entre MaNGIA y MaNGA, pero sí elimina una de las fuentes más graves de inconsistencia conceptual.

## 6. Qué diferencias siguen existiendo entre MaNGIA y MaNGA

Aun si ambos usan `pyPipe3D`, el problema de transferencia no desaparece.

Siguen existiendo diferencias importantes:

- tamaño espacial del cubo;
- cobertura del campo;
- tratamiento de borde;
- ruido y errores;
- máscaras de calidad;
- posibles diferencias en sampling espectral;
- diferencias instrumentales y observacionales;
- y diferencias físicas entre simulación y universo observado.

Por lo tanto, aunque los mapas Pipe3D sean comparables, sigue siendo necesario estandarizar la representación final del dataset.

## 7. La pregunta espacial clave: 69x69 frente a 74x74

En las pruebas locales realizadas hasta ahora:

- un cubo MaNGIA mock quedó en `69x69`;
- un `LOGCUBE` real de MaNGA quedó en `74x74`.

Ambos comparten aproximadamente la misma escala espacial por spaxel, pero no la misma caja espacial final.

Esto sigue siendo metodológicamente relevante incluso si los mapas físicos se obtienen con el mismo algoritmo.

La razón es que una red de segmentación aprende también:

- geometría del campo;
- proximidad al borde;
- extensión relativa del disco;
- ubicación de regiones externas;
- y patrones espaciales amplios.

Por eso, el hecho de que ambos dominios compartan `pyPipe3D` no resuelve por sí mismo el problema de entrenar en `69x69` e inferir en `74x74`.

## 8. Recomendación sobre la representación espacial

La recomendación metodológica sigue siendo:

- no entrenar el modelo final dejando los mocks en `69x69` como formato definitivo;
- adaptar los datos MaNGIA a la geometría final del dominio objetivo;
- idealmente trabajar con una representación final `74x74` compatible con MaNGA.

La forma más conservadora de hacerlo es:

- centrar el cubo mock;
- aplicar padding hasta `74x74`;
- conservar una máscara explícita de spaxels válidos y no válidos.

Esta solución es preferible a un remuestreo espacial agresivo porque:

- preserva mejor la información original;
- evita introducir suavizado artificial;
- minimiza deformaciones geométricas;
- y deja al modelo una convención espacial consistente para entrenamiento y transferencia.

## 9. Recomendación sobre el uso de los mapas físicos Pipe3D

Bajo la premisa verificada, sí es razonable usar los mapas físicos derivados como entradas del modelo.

Pero eso requiere mantener consistencia estricta en:

- algoritmo de derivación;
- resolución espacial final;
- unidades físicas;
- tratamiento de máscara;
- y definición de los canales disponibles.

En otras palabras, no basta con que ambos se llamen “edad”, “masa” o “metalicidad”. Es importante que estén:

- producidos por el mismo pipeline o por una variante plenamente equivalente;
- y representados de forma común dentro del tensor de entrada.

## 10. Recomendación sobre el dataset de entrenamiento

La propuesta más sólida para el dataset es la siguiente:

1. partir de cubos MaNGIA mock `manga-like`;
2. procesarlos con `pyPipe3D`;
3. extraer mapas 2D físicos derivados con ese mismo algoritmo;
4. adaptar el cubo y sus mapas a una representación final compatible con MaNGA, idealmente `74x74`;
5. incorporar una máscara de validez por spaxel;
6. usar como entrada:
   - espectro por spaxel,
   - mapas Pipe3D,
   - imagen asociada;
7. entrenar la red de segmentación sobre esa representación homogénea;
8. transferir y afinar luego con MaNGA real.

## 11. Qué cambia respecto de la versión anterior de la nota

La diferencia clave con la formulación anterior es conceptual:

- antes, existía el riesgo de mezclar mapas intrínsecos de simulación con mapas observacionales;
- ahora, la premisa se apoya en mapas derivados con el mismo algoritmo `pyPipe3D` en ambos dominios.

Eso fortalece mucho la justificación metodológica del proyecto.

Sin embargo, no cambia dos conclusiones fundamentales:

- la representación espacial aún debe homogenizarse para facilitar la transferencia;
- y la recomendación práctica sigue siendo adaptar los mocks a `74x74` si ese será el formato final de MaNGA.

## 12. Recomendación final

La recomendación metodológica final es:

- sí es científicamente defendible usar MaNGIA para preentrenar una red de segmentación por spaxel;
- sí es defendible usar mapas físicos 2D como entrada si esos mapas son los derivados por `pyPipe3D` en ambos dominios;
- no es recomendable mezclar `cube_val` con mapas Pipe3D como si fueran equivalentes;
- y, aun bajo esta premisa más fuerte, sigue siendo recomendable estandarizar el dataset final al formato espacial de MaNGA, idealmente `74x74` con máscara de spaxels válidos.

## 13. Síntesis ejecutiva

En resumen:

- la premisa queda verificada si hablamos de mapas derivados con `pyPipe3D`;
- MaNGIA y MaNGA pueden compararse de manera mucho más sólida bajo ese esquema;
- `cube_val` no debe confundirse con una salida de `pyPipe3D`;
- la estrategia de transferencia sigue siendo válida;
- y la recomendación práctica continúa siendo adaptar los mocks a la geometría final de MaNGA.

## 14. Referencias

- MaNGIA paper en A&A:
  https://www.aanda.org/articles/aa/full_html/2023/05/aa45509-22/aa45509-22.html

- Resumen institucional del paper MaNGIA:
  https://www.iac.es/en/science-and-technology/publications/mangia-10-000-mock-galaxies-stellar-population-analysis?base_route_name=entity.node.canonical&overridden_route_name=entity.node.canonical&page_manager_page=node_view&page_manager_page_variant=node_view-panels_variant-2&page_manager_page_variant_weight=-4

- SDSS Value Added Catalogs, MaNGA Pipe3D VAC:
  https://www.sdss.org/dr19/data_access/value-added-catalogs/

- Evidencia local sobre `cube_val` y `pyPipe3D`:
  [official_mangia/sin_ifu_clean.py](/home/andy/pythonprojects/cubes/official_mangia/sin_ifu_clean.py#L1849)
  [official_mangia/sin_ifu_clean.py](/home/andy/pythonprojects/cubes/official_mangia/sin_ifu_clean.py#L2231)
