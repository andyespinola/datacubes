# Nota Metodológica: Construcción de Etiquetas Bulbo/Disco/Barra/Brazos desde la Verdad de la Simulación

## 1. Objetivo de esta nota

Esta nota está dedicada exclusivamente a una pregunta metodológica central del proyecto:

- cómo construir etiquetas supervisadas por spaxel para estructuras galácticas;
- usando la verdad física disponible en la simulación;
- y manteniendo compatibilidad con una entrada observacional tipo MaNGIA/MaNGA.

La pregunta concreta es si resulta viable generar, para cada spaxel del cubo, una etiqueta estructural como:

- bulbo;
- disco;
- barra;
- brazos espirales;
- fondo/no válido;
- u otra clase residual.

La respuesta corta es sí, pero con una precisión importante:

- la simulación entrega la verdad física en 3D;
- la red aprenderá sobre un plano observado 2D;
- por lo tanto, las etiquetas deben construirse en dos etapas:
  - primero en el espacio físico de la simulación;
  - luego proyectadas al mismo plano y a la misma grilla espacial del cubo IFU.

## 2. Idea central

La ventaja científica de MaNGIA no es solamente que genera cubos `MaNGA-like`.

La ventaja decisiva es que, detrás de ese cubo observado, existe una galaxia simulada con mucha más información que la disponible en observaciones:

- posiciones 3D;
- velocidades 3D;
- masas;
- edades estelares;
- metalicidades;
- tasas de formación estelar;
- historia temporal;
- distribución de gas, estrellas y materia oscura;
- y catálogos suplementarios de morfología y cinemática.

Eso permite construir etiquetas con una base física mucho más fuerte que una clasificación puramente visual.

La estrategia correcta no es etiquetar "a ojo" el cubo reconstruido, sino:

1. identificar las componentes estructurales en la simulación;
2. proyectarlas al ángulo de visión del mock;
3. degradarlas a la resolución espacial del cubo;
4. y finalmente asignar una etiqueta o probabilidad a cada spaxel.

## 3. Qué representa realmente la galaxia en la simulación

IllustrisTNG no almacena una galaxia como una imagen terminada.

La galaxia está representada como una colección de elementos físicos dentro de un volumen cosmológico:

- celdas de gas;
- partículas de estrellas;
- partículas de materia oscura;
- agujeros negros;
- y catálogos de subhalos/halos.

La documentación oficial de TNG describe explícitamente el contenido de los snapshots:

- `PartType0`: gas;
- `PartType1`: materia oscura;
- `PartType4`: estrellas;
- `PartType5`: agujeros negros.

También deja explícito que los snapshots guardan coordenadas, masas, velocidades, metalicidad, SFR y otras propiedades por elemento.

Fuentes:

- TNG Project Description:
  https://www.tng-project.org/about/
- TNG Data Specifications:
  https://www.tng-project.org/data/docs/specifications/

En particular, la descripción oficial de TNG indica que el modelo incluye:

- enfriamiento y calentamiento radiativo del gas;
- formación estelar;
- evolución de poblaciones estelares y enriquecimiento químico;
- feedback estelar;
- formación y crecimiento de agujeros negros;
- feedback de agujeros negros;
- magnetohidrodinámica.

Eso significa que la simulación contiene los ingredientes físicos de los que emergen el disco, el bulbo, la barra y, en muchos casos, los brazos espirales.

## 4. Qué datos pueden usarse como verdad para construir etiquetas

### 4.1. Datos primarios de la simulación

Son los más importantes para este problema.

Para las estrellas, interesan especialmente:

- coordenadas 3D;
- velocidades 3D;
- masa;
- energía de ligadura o proxies cinemáticos;
- edad;
- metalicidad;
- luminosidad o espectro asignado, si se desea usar una ponderación por luz.

Para el gas, pueden ser útiles de forma secundaria:

- distribución de gas frío;
- SFR;
- metalicidad del gas;
- cinemática del gas.

Esto sirve sobre todo para refinar regiones de brazos o de formación estelar reciente, pero las etiquetas estructurales de bulbo/disco/barra deben definirse principalmente con la componente estelar.

### 4.2. Catálogos suplementarios oficiales de TNG

TNG publica además varios catálogos derivados que son muy útiles como apoyo metodológico.

Entre los más relevantes están:

- catálogo de circularidades estelares, momento angular y axis ratios;
- catálogos de barras;
- catálogos de morfología cinemática;
- imágenes sintéticas y medidas morfológicas;
- mocks ópticos e imágenes tipo SDSS.

Fuentes oficiales:

- circularidades estelares:
  https://www.tng-project.org/data/docs/specifications/
- catálogos de barras y morfologías cinemáticas:
  https://www.tng-project.org/data/docs/specifications/

Estos catálogos son muy valiosos, pero hay que usarlos con cuidado.

### 4.3. Qué sí aportan esos catálogos

Sí aportan:

- evidencia objetiva de que la galaxia es disky o bulge-dominated;
- fracciones globales de masa en componentes cinemáticas;
- indicadores globales de barra;
- tamaños y fortalezas de barra;
- imágenes sintéticas sobre las que puede hacerse validación.

### 4.4. Qué no resuelven por sí solos

En general, esos productos no equivalen automáticamente a una máscara final por spaxel lista para entrenar.

La razón es que muchas veces entregan:

- fracciones globales por galaxia;
- medidas resumidas;
- banderas de barra;
- o imágenes auxiliares;

pero no necesariamente una etiqueta final ya rasterizada en la grilla exacta del cubo MaNGIA.

Por eso, para un proyecto de segmentación espacial, la ruta más robusta sigue siendo:

- partir de la verdad por partícula o por mapa intrínseco de alta resolución;
- y reconstruir nosotros la máscara proyectada por spaxel.

## 5. Verificación de que esta estrategia es compatible con MaNGIA

La publicación de MaNGIA explica que los cubos `MaNGA-like` se generan a partir de galaxias TNG forward-modeladas al plano observacional, y luego se analizan con `pyPipe3D`.

Fuente principal:

- MaNGIA: 10 000 mock galaxies for stellar population analysis
  https://www.aanda.org/articles/aa/full_html/2023/05/aa45509-22/aa45509-22.html

Eso hace posible una separación metodológica muy importante:

- las entradas del modelo pueden ser observacionales o pseudo-observacionales;
- las etiquetas pueden venir de la verdad de simulación.

Esa separación es precisamente lo que vuelve valioso el esquema de aprendizaje supervisado.

En otras palabras:

- `input`: cubo mock + mapas derivados con el mismo pipeline que en MaNGA;
- `target`: máscara estructural construida desde la verdad física de TNG.

## 6. Principio metodológico clave: las etiquetas deben vivir en el mismo espacio que el input

La simulación vive en 3D.

La red verá:

- un cubo 3D espectral, pero proyectado en 2D espacial;
- mapas 2D derivados;
- una imagen 2D;
- y una máscara espacial por spaxel.

Por eso, aunque las estructuras se definan a partir de la verdad 3D, el target final debe expresarse como:

- una máscara 2D por spaxel;
- o mejor aún, un tensor de probabilidades por clase y por spaxel.

La regla práctica es:

- la física se define en 3D;
- la supervisión se entrena en 2D proyectado.

## 7. Pipeline propuesto para construir las etiquetas

### 7.1. Paso 1: vincular el cubo MaNGIA con su galaxia en TNG

Cada mock MaNGIA está asociado a:

- una simulación;
- un snapshot;
- un `subhalo_id`;
- y una vista o proyección concreta.

En nuestra convención local eso aparece en nombres como:

- `TNG50-87-141934-0-127`

donde, para este ejemplo:

- `87` corresponde al snapshot;
- `141934` al `subhalo_id`;
- `0` a la vista.

Ese vínculo es el punto de partida para ir desde el cubo observado hacia la verdad física.

### 7.2. Paso 2: extraer la verdad estelar de esa galaxia

Para la galaxia seleccionada se debe extraer, idealmente dentro de un radio físico razonable, la población estelar asociada al subhalo:

- coordenadas `x, y, z`;
- velocidades `vx, vy, vz`;
- masa estelar;
- edad;
- metalicidad;
- energía o proxy de binding, si se quiere usar circularidad orbital;
- luminosidad o peso en banda óptica, si se quiere proyectar por luz y no por masa.

Aquí hay una decisión metodológica importante:

- si las etiquetas deben representar estructura dinámica, la masa estelar es la mejor base;
- si deben representar estructura visible tal como la "vería" el telescopio, puede ser útil proyectar por luminosidad;
- en la práctica conviene guardar ambas versiones.

### 7.3. Paso 3: definir el sistema de referencia galáctico

Antes de clasificar partículas, la galaxia debe llevarse a un marco físico consistente:

- centrar en el centro del subhalo o en la partícula más ligada;
- sustraer la velocidad sistémica;
- alinear el eje `z` con el momento angular estelar total del disco;
- distinguir entre una vista intrínseca "face-on" para identificar estructura y la vista observacional usada por MaNGIA para entrenar.

Esto importa porque:

- bulbo y disco se separan mejor en un marco dinámico alineado;
- barra y brazos se detectan mejor en una proyección face-on;
- pero el target final debe llevarse luego a la proyección observacional del mock.

### 7.4. Paso 4: construir primero una descomposición física intrínseca

Aquí es donde se define, a nivel de partículas o de celdas de alta resolución, qué pertenece a cada componente.

### 7.4.1. Disco

La forma más estándar de identificar disco estelar en simulaciones es mediante soporte rotacional.

La documentación pública de TNG incluye un catálogo de circularidades estelares y recuerda una convención muy usada:

- estrellas con circularidad alta y positiva corresponden a una componente con soporte rotacional;
- por ejemplo, la fracción con `epsilon > 0.7` se usa como medida común de "disk stars".

Fuente:

- TNG Data Specifications, sección de circularidades:
  https://www.tng-project.org/data/docs/specifications/

Metodológicamente, el disco puede definirse como:

- estrellas con circularidad suficientemente positiva;
- órbitas coherentes con el momento angular principal;
- distribución espacial aplanada.

En una primera versión del proyecto, conviene agrupar como `disco`:

- disco frío;
- y disco cálido.

Eso reduce complejidad y produce una etiqueta más estable.

### 7.4.2. Bulbo

El bulbo puede definirse como la componente estelar:

- central;
- más dispersiva;
- menos soportada por rotación;
- con circularidad cercana a cero o baja;
- y morfología aproximadamente esferoidal o poco aplanada.

Con criterios cinemáticos, suele construirse como el complemento spheroidal del disco.

Con criterios morfológicos, también puede reforzarse usando:

- concentración radial;
- axis ratios;
- dominancia en el centro.

Si se desea simplificar para una primera versión:

- `bulbo = pseudo-bulge + bulge clásico`

es una convención razonable.

Si se desea mayor fineza después, el pseudo-bulge puede convertirse en una clase separada.

### 7.4.3. Barra

La barra no se define bien solo con circularidad.

Es una estructura:

- central;
- elongada;
- no axisimétrica;
- y embebida en el disco.

La documentación pública de TNG incluye catálogos de barras y propiedades de barra, basados en el análisis de Fourier de la densidad superficial estelar.

Fuente:

- TNG Data Specifications, sección de barras:
  https://www.tng-project.org/data/docs/specifications/

Ahí se documentan, entre otras cosas:

- bandera de galaxia barrada;
- tamaño de barra;
- fuerza de barra;
- amplitud `A2`.

Eso es muy útil como control global, pero para generar la máscara espacial por spaxel conviene reconstruir también la región de barra de forma explícita:

1. proyectar la galaxia face-on;
2. medir la densidad superficial estelar;
3. calcular la componente azimutal `m=2` o un indicador equivalente;
4. identificar una fase coherente en la región interna;
5. tomar el radio de barra `R_bar`;
6. definir como `barra` la sobredensidad elongada interna dentro de ese radio.

En otras palabras:

- el catálogo oficial de barra sirve como validación y prior;
- la máscara 2D final conviene reconstruirla sobre el mapa estelar proyectado.

### 7.4.4. Brazos espirales

Aquí está el punto más delicado.

No encontré evidencia en la documentación pública de TNG de un catálogo general, homogéneo y por partícula que ya entregue una máscara de brazos espirales para todo el conjunto que nos interesa. Eso es una inferencia basada en la revisión de los catálogos públicos listados en las especificaciones de TNG: sí aparecen catálogos de circularidad, descomposiciones cinemáticas, barras, imágenes sintéticas y morfologías globales, pero no una máscara pública estándar de brazos por spaxel equivalente a la que necesita este proyecto.

Fuente revisada:

- TNG Data Specifications:
  https://www.tng-project.org/data/docs/specifications/

Por eso, los brazos deben derivarse.

La forma más realista es:

1. tomar el mapa estelar o de luz proyectado face-on;
2. sustraer un modelo suave axisimétrico del disco;
3. identificar los residuales espirales coherentes;
4. excluir la región de barra interna;
5. definir como `brazos` las crestas espirales residuales.

Según el objetivo del proyecto, esto puede hacerse sobre:

- masa estelar superficial;
- luminosidad óptica;
- o una combinación con estrellas jóvenes/SFR, si se quiere enfatizar brazos visuales recientes.

Mi recomendación es separar dos conceptos:

- `brazo dinámico/estelar`: definido sobre masa o luz estelar;
- `región de formación estelar`: definida sobre SFR o población joven.

Para segmentación morfológica principal, el target debería ser el primero.

### 7.5. Paso 5: convertir la descomposición intrínseca en una máscara 2D de alta resolución

Una vez clasificadas partículas o elementos estelares, se proyectan al plano elegido.

Aquí es importante distinguir dos proyecciones:

- la proyección intrínseca face-on usada para detectar barra y brazos;
- la proyección observacional específica del mock MaNGIA usada para el entrenamiento.

La secuencia recomendada es:

1. detectar estructura en el sistema físico más estable;
2. conservar la identidad de las partículas clasificadas;
3. re-proyectar esas mismas partículas a la vista observacional del mock;
4. construir mapas 2D continuos de contribución por clase.

El resultado de este paso no debería ser todavía una clase dura, sino mapas de contribución:

- masa de bulbo por píxel fino;
- masa de disco por píxel fino;
- masa de barra por píxel fino;
- masa de brazos por píxel fino.

### 7.6. Paso 6: degradar esas etiquetas al plano observacional

Este paso es esencial.

La red no verá la galaxia intrínseca ideal, sino una versión observacional:

- con PSF;
- con tamaño de spaxel;
- con mezcla espacial;
- y con máscara geométrica.

Por eso, las etiquetas deben pasar por una degradación compatible con el cubo.

La receta más consistente es:

1. partir de mapas de alta resolución por clase;
2. convolverlos con la PSF espacial usada en el mock;
3. muestrearlos en la misma grilla del cubo;
4. calcular, para cada spaxel, la fracción de contribución de cada clase.

Esto evita una inconsistencia frecuente:

- usar inputs observacionalmente degradados;
- pero targets ideales y demasiado "afilados".

Si el input está mezclado por seeing y por tamaño de fibra/spaxel, el target debe reflejar también esa mezcla.

### 7.7. Paso 7: producir etiquetas suaves y luego, si se desea, etiquetas duras

La forma más robusta de etiquetar un spaxel no es con una sola clase binaria desde el comienzo.

Lo recomendable es generar primero:

- un vector de probabilidades o fracciones por clase.

Por ejemplo, para cada spaxel `(y, x)`:

- `p_bulbo(y, x)`
- `p_disco(y, x)`
- `p_barra(y, x)`
- `p_brazos(y, x)`
- `p_otro(y, x)`

con la condición de que la suma sea 1 en spaxels válidos.

Esto es mejor porque muchos spaxels, especialmente:

- en zonas de transición;
- cerca del centro;
- o después de la degradación por PSF;

son mezclas reales de más de una estructura.

La etiqueta dura se puede definir recién al final como:

- la clase de mayor probabilidad;

o bien con una regla adicional:

- si ninguna clase supera un umbral, marcar como `incierto`.

## 8. Qué datos entran realmente al pipeline de construcción de etiquetas

Conviene separar con claridad dos tipos de entrada:

- entrada para construir las etiquetas;
- entrada de la red neuronal.

### 8.1. Entrada para construir las etiquetas

Esta entrada viene de la simulación y del mock.

### A. Desde TNG

- `snapshot`
- `subhalo_id`
- posiciones estelares 3D
- velocidades estelares 3D
- masas estelares
- edades
- metalicidades
- luminosidades o pesos fotométricos, si están disponibles
- información cinemática auxiliar
- catálogos suplementarios de circularidad, barra o morfología, cuando existan

### B. Desde MaNGIA

- la vista o cámara usada para ese mock
- la geometría espacial del cubo
- la PSF adoptada
- el tamaño del spaxel
- la máscara espacial válida
- la correspondencia exacta entre la galaxia TNG y el cubo reconstruido

### 8.2. Entrada de la red neuronal

La red no debería ver directamente la verdad intrínseca de la simulación.

La red debería ver variables comparables entre MaNGIA y MaNGA.

La propuesta más consistente es:

- espectro completo por spaxel;
- mapas 2D derivados con `pyPipe3D`;
- una imagen 2D de referencia, por ejemplo reconstruida en banda V o colapsada;
- una máscara de spaxels válidos;
- y, opcionalmente, canales geométricos auxiliares como radio normalizado o coordenadas centradas.

En forma de tensor, una muestra podría representarse como:

- `X = [spectral cube, pipe3d maps, image channels, valid mask]`

y el target como:

- `Y = [p_bulbo, p_disco, p_barra, p_brazos, p_otro]`

o, en su versión dura:

- `Y_label in {0, 1, 2, 3, 4, 5}` por spaxel.

## 9. Definición práctica recomendada de clases para una primera versión

Para una primera fase del proyecto, conviene evitar una taxonomía demasiado fina.

La propuesta más robusta es:

- `0`: fondo / no válido
- `1`: bulbo
- `2`: disco
- `3`: barra
- `4`: brazos espirales
- `5`: incierto / otras estructuras

Esto tiene varias ventajas:

- reduce ambigüedad;
- permite entrenar con un número razonable de clases;
- deja margen para absorber regiones mixtas o poco confiables;
- y evita forzar decisiones artificiales en spaxels de frontera.

En una segunda fase podrían separarse:

- pseudo-bulge;
- disco frío vs disco cálido;
- anillos;
- regiones star-forming;
- halo interno.

Pero no es recomendable empezar con ese nivel de detalle.

## 10. Qué estructura concreta recomiendo usar para cada clase

### 10.1. Bulbo

Definición recomendada:

- componente central spheroidal;
- baja circularidad;
- baja dominancia de rotación;
- alta concentración.

Implementación inicial:

- usar un criterio cinemático de no-disco;
- restringir espacialmente al centro;
- combinar bulbo clásico y pseudo-bulge en una sola clase, salvo que el proyecto necesite separarlos.

### 10.2. Disco

Definición recomendada:

- estrellas con soporte rotacional;
- distribución aplanada;
- circularidad alta y positiva.

Implementación inicial:

- unir disco frío y disco cálido en una sola clase `disco`.

### 10.3. Barra

Definición recomendada:

- sobredensidad elongada en la región interna;
- inmersa en el disco;
- fase coherente de la componente `m=2`;
- radio compatible con un `R_bar` definido.

Implementación inicial:

- usar el bar flag y tamaño global como validación;
- construir la máscara espacial mediante análisis 2D de densidad estelar proyectada.

### 10.4. Brazos

Definición recomendada:

- sobredensidades espirales no axisimétricas;
- externas al radio de barra;
- contenidas dentro del disco.

Implementación inicial:

- modelar un disco suave;
- medir residuales;
- seleccionar crestas espirales coherentes;
- y asignar esos residuales como brazos.

## 11. Qué productos intermedios conviene guardar

Para que el pipeline sea auditable y científicamente defendible, conviene guardar no solo la máscara final, sino también productos intermedios:

- mapa de masa superficial total;
- mapa de masa superficial por clase;
- mapa de luz superficial por clase;
- máscara intrínseca face-on;
- máscara proyectada a la vista observacional;
- máscara ya degradada a la grilla IFU;
- probabilidades por clase y por spaxel;
- etiqueta dura final por spaxel;
- y métricas globales por galaxia.

Estas métricas globales pueden incluir:

- fracción de masa por clase;
- tamaño de barra;
- fuerza de barra;
- fracción de disco;
- fracción de bulbo;
- área total asignada a brazos.

Eso permite verificar que la máscara final no contradice lo que se sabe de la galaxia a escala global.

## 12. Dónde usar la simulación y dónde no usarla

Esta distinción es crucial para defender el proyecto.

### Sí usar la simulación para

- construir etiquetas supervisadas;
- cuantificar incertidumbre del target;
- validar consistencia física de las clases;
- estudiar sesgos del etiquetado.

### No usar la simulación como input oculto de inferencia

Cuando el modelo se aplique a MaNGA real, no existirá la verdad intrínseca.

Por lo tanto, no debe entrenarse un modelo que dependa como entrada de variables imposibles de observar directamente en MaNGA.

En consecuencia:

- la verdad de simulación debe usarse como `target`;
- no como un canal obligatorio del `input`.

Esa separación hace que la transferencia a MaNGA sea metodológicamente limpia.

## 13. Riesgos metodológicos y cómo mitigarlos

### Riesgo 1: clases demasiado rígidas

Problema:

- muchas regiones son mezclas reales de estructuras.

Mitigación:

- usar etiquetas suaves;
- usar una clase `incierto`;
- conservar fracciones por clase además de la clase dura.

### Riesgo 2: confundir barra con bulbo o con disco interno

Problema:

- la barra vive en la zona más difícil del campo.

Mitigación:

- combinar cinemática y morfología;
- usar tamaño y fuerza de barra como control;
- excluir la barra del dominio de brazos.

### Riesgo 3: sobreidentificar brazos a partir de regiones star-forming

Problema:

- no toda región joven o brillante es un brazo estructural.

Mitigación:

- definir brazos primero sobre estructura estelar;
- usar SFR solo como refinamiento secundario, no como criterio único.

### Riesgo 4: targets físicamente correctos pero observacionalmente incompatibles

Problema:

- el input está degradado por PSF y el target no.

Mitigación:

- degradar también las etiquetas;
- y trabajar con fracciones por spaxel.

## 14. Recomendación metodológica final

La forma más defendible de construir etiquetas para este proyecto es la siguiente:

1. usar la verdad estelar 3D de TNG para definir componentes físicas;
2. identificar disco y bulbo principalmente con criterios cinemáticos;
3. identificar barra con análisis de densidad superficial y modo `m=2`;
4. identificar brazos con residuales espirales sobre el disco proyectado;
5. proyectar todas las componentes a la misma vista observacional del mock;
6. degradar las máscaras a la PSF y a la grilla del cubo;
7. guardar probabilidades por clase por spaxel;
8. derivar de allí una etiqueta dura solo si hace falta.

Este esquema tiene una ventaja fuerte:

- las entradas del modelo siguen siendo comparables con MaNGA;
- pero la supervisión aprovecha toda la profundidad física de la simulación.

## 15. Conclusión

Sí es posible construir etiquetas bulbo/disco/barra/brazos desde la verdad de la simulación.

De hecho, esa es una de las mayores fortalezas científicas de MaNGIA frente a un problema supervisado de segmentación espacial.

Sin embargo, la etiqueta correcta no debe extraerse directamente como una "imagen final" de la simulación.

La estrategia correcta es:

- descomponer primero la galaxia en términos físicos;
- proyectar después esas componentes al plano observacional;
- y entrenar la red con targets por spaxel compatibles con la resolución y la geometría del cubo.

En ese marco, la simulación sirve para entregar:

- targets físicamente informados;
- consistentes;
- auditables;
- y más ricos que cualquier etiquetado puramente visual.

## 16. Fuentes utilizadas

- IllustrisTNG Project Description:
  https://www.tng-project.org/about/
- IllustrisTNG Data Specifications:
  https://www.tng-project.org/data/docs/specifications/
- MaNGIA paper:
  https://www.aanda.org/articles/aa/full_html/2023/05/aa45509-22/aa45509-22.html

## 17. Estado de verificación de las afirmaciones

Directamente verificado con fuentes:

- que TNG almacena snapshots con gas, estrellas, materia oscura y agujeros negros;
- que TNG publica catálogos de circularidades, barras y morfologías cinemáticas;
- que TNG50 resuelve estructuras internas como bulbos, barras y brazos;
- que MaNGIA forward-modela galaxias TNG al plano observacional y produce cubos comparables con MaNGA.

Inferencias metodológicas propuestas en esta nota:

- que la mejor forma de obtener etiquetas por spaxel es reconstruirlas desde partículas y mapas proyectados;
- que los brazos deben derivarse con una máscara propia si se busca una segmentación homogénea por spaxel;
- que conviene usar etiquetas suaves degradadas por PSF antes de pasar a etiquetas duras.

Estas inferencias no contradicen las fuentes revisadas, pero son una propuesta de diseño metodológico del proyecto, no un producto estándar ya entregado por MaNGIA o por TNG.

## 18. Trabajo realizado hasta ahora

En esta sección se resume, paso a paso, lo que ya se implementó y verificó en el entorno local del proyecto.

### 18.1. Vinculación entre MaNGIA y TNG

Ya se verificó la correspondencia entre el mock MaNGIA y su galaxia en TNG para el caso piloto:

- identificador canónico: `TNG50-87-141934-0-127`
- snapshot: `87`
- `subhalo_id`: `141934`
- vista: `0`
- IFU: `127`

También se verificó la correspondencia con el catálogo local de MaNGIA y se automatizó la lectura de `re_kpc` para la reconstrucción oficial del cubo.

### 18.2. Reconstrucción del cubo `MaNGA-like`

Se implementó y validó el wrapper local del flujo oficial de MaNGIA para producir el cubo `MaNGA-like` a partir del RSS.

Se generaron correctamente:

- el cubo principal `*.cube.fits.gz`
- el archivo compañero `*.cube_val.fits.gz`

Para el caso piloto, el cubo principal quedó con shape:

- `(6603, 69, 69)`

y con extensiones consistentes con el producto esperado:

- `PRIMARY`
- `ERROR`
- `MASK`
- `GAS`

### 18.3. Construcción del proyecto de etiquetado estructural

Se creó un proyecto independiente para el etiquetado estructural basado en verdad de simulación.

Ese proyecto ya incluye:

- construcción de manifiestos por galaxia;
- descarga de verdad TNG;
- carga del catálogo morfológico oficial;
- uso de una librería SSP local para pesos por luz;
- pipeline completo de etiquetado;
- guardado de productos finales y de QA.

### 18.4. Descarga y preparación de la verdad de TNG

Para la galaxia piloto ya se descargaron y validaron:

- el `cutout` estelar y de gas de TNG;
- los metadatos del subhalo;
- el catálogo suplementario oficial de morfologías cinemáticas y barras de TNG50-1.

Con esos datos se verificó la información global de la galaxia:

- fracción `disk_family`
- fracción `bulge_family`
- fracción `other_family`
- condición de galaxia barrada
- tamaño de barra
- fuerza de barra

### 18.5. Conversión física y normalización

El pipeline ya convierte la verdad de TNG a unidades físicas consistentes antes de etiquetar:

- posiciones físicas;
- velocidades físicas;
- masas estelares;
- edades estelares;
- metadatos del subhalo.

También se centra la galaxia en el sistema del subhalo y se rota a una vista `face-on` usando el momento angular estelar global.

### 18.6. Construcción de pesos por masa y por luz

Ya se implementaron dos versiones del target:

- `soft_mass`
- `soft_light`

La primera usa masa estelar como peso.

La segunda usa luminosidad estimada a partir de:

- masa;
- edad;
- metalicidad;
- y una grilla SSP local.

### 18.7. Descomposición intrínseca en familias

Ya se implementó una primera descomposición en familias estructurales:

- `bulbo`
- `disco`
- `other`

Esa descomposición usa:

- radio galactocéntrico;
- cuantiles ponderados por masa;
- fracciones objetivo del catálogo morfológico oficial;
- reescalado iterativo para respetar las fracciones globales.

### 18.8. Descomposición de subestructura dentro del disco

Dentro de la familia disco ya se implementó una separación preliminar entre:

- `disco`
- `barra`
- `brazos`

La barra se modela usando:

- tamaño de barra;
- fuerza de barra;
- coherencia angular del modo `m=2`.

Los brazos se modelan usando:

- mapa `face-on` de masa del disco;
- suavizado espacial;
- componente axisimétrica;
- residuales positivos espirales;
- y un refuerzo opcional con gas/SFR.

### 18.9. Proyección al plano observado de MaNGIA

Una vez inferida la identidad estructural de las partículas, el pipeline ya:

- reproyecta a la vista observacional concreta del mock;
- deposita cada clase en la grilla del cubo;
- aplica la PSF espacial;
- y remuestrea a la geometría final IFU.

### 18.10. Generación de etiquetas por spaxel

El pipeline ya produce, por galaxia:

- `soft_mass`
- `soft_light`
- `hard_mass`
- `hard_light`
- `confidence_mass`
- `confidence_light`
- `valid_mask`

También genera productos de QA:

- mapas `face-on`
- mapas observados por clase
- residuales del disco
- resumen global en JSON

### 18.11. Validación del caso piloto

Para la galaxia piloto ya se verificó que las fracciones globales recuperadas sean consistentes con el catálogo oficial.

La recuperación global quedó, dentro del footprint válido, muy cercana a los targets del catálogo para:

- `bulge_family`
- `disk_family`
- `other_family`

También quedó recuperado un tamaño de barra compatible con el valor objetivo del catálogo.

### 18.12. Integración con el visor

El visor web local ya fue extendido para mostrar:

- overlays de etiquetas estructurales;
- selección de clase;
- selección de tipo de etiqueta (`soft_mass`, `soft_light`, `hard_mass`, `hard_light`);
- opacidad del overlay;
- y probabilidades por spaxel.

También ya se puede inspeccionar visualmente:

- qué estructura domina en un spaxel;
- qué tan confiada es la clasificación dura;
- y cómo difieren masa y luz.

## 19. Problemas reportados durante la inspección actual

Durante la inspección visual y cuantitativa del caso piloto aparecieron varios problemas metodológicos que todavía no están resueltos.

### 19.1. Spaxels periféricos muy débiles todavía se clasifican como disco

Se detectaron spaxels que visualmente parecen estar fuera de la estructura útil de la galaxia, pero que el pipeline todavía considera válidos y termina clasificando como `disco`.

El ejemplo discutido explícitamente fue:

- `x=44`, `y=18`

En ese caso:

- el spaxel todavía pertenece al `valid_mask`;
- tiene señal numéricamente no nula;
- y tras la normalización por spaxel el disco queda como clase dominante.

Esto indica que el criterio actual de `valid_mask` es demasiado permisivo.

### 19.2. La etiqueta dura no usa coherencia espacial explícita

Se observó que spaxels vecinos, incluso contiguos en el centro, pueden quedar con clases distintas:

- uno como `bulbo`;
- el de al lado como `incierto_otro`;
- o varios vecinos como `disco` alrededor de una zona visualmente compacta.

Esto ocurre porque la etapa `hard`:

- no usa vecindad;
- no usa conectividad;
- no usa regularización espacial;
- y decide clase por spaxel de forma local.

### 19.3. El centro puede quedar dominado por disco o por incertidumbre

Se reportó que el centro de la galaxia no aparece sistemáticamente etiquetado como `bulbo`.

La inspección mostró que en el spaxel central más brillante:

- `bulbo` tiene probabilidad alta;
- pero `disco` queda ligeramente por encima;
- y la clase dura puede caer en `incierto_otro` por no superar el umbral mínimo.

Esto sugiere que el modelo actual:

- es demasiado simple en la competencia `bulbo vs disco`;
- y está muy influido por la fracción global de disco de la galaxia.

### 19.4. `hard_mass` y `hard_light` pueden discrepar

También se observó que algunos spaxels aparecen como:

- `incierto` o `other` en masa;
- pero `disco` en luz.

Eso no es un error de implementación, sino una consecuencia del diseño actual:

- `mass` pondera masa estelar;
- `light` pondera luminosidad;
- y una población puede dominar visualmente sin dominar en masa.

Aunque esto es físicamente razonable, todavía no existe una política formal de cómo usar esta discrepancia en la construcción final del dataset.

### 19.5. La clase `incierto_otro` mezcla dos significados distintos

En el esquema actual, `incierto_otro` absorbe simultáneamente:

- la componente física residual `other`;
- y los spaxels para los que la etiqueta dura no alcanza suficiente confianza.

Eso hace que la interpretación científica de esa clase sea ambigua.

### 19.6. El umbral duro actual puede ser demasiado permisivo en la periferia

La versión actual usa:

- `hard_label_min_prob = 0.50`
- `hard_label_margin = 0.15`

Al inspeccionar el caso piloto se vio que eso permite que varios spaxels periféricos entren como `disco`.

También se probó informalmente un aumento de umbral a `0.60`, que reduce algunos falsos positivos periféricos, pero al costo de convertir muchos spaxels previamente clasificados en:

- `incierto_otro`

Eso muestra que el problema no se resuelve solo subiendo el umbral.

## 20. Sugerencias de solución discutidas y todavía no implementadas

### 20.1. Hacer el `valid_mask` más estricto

La mejora con más impacto inmediato sería redefinir el `valid_mask` para que represente mejor el footprint observable útil de la galaxia.

La propuesta discutida es construirlo como:

- footprint instrumental
- más un criterio de señal mínima
- más limpieza espacial

Concretamente:

- usar el mapa colapsado 2D;
- suavizarlo;
- definir un umbral robusto de señal;
- eliminar islas pequeñas;
- conservar la componente conexa principal.

### 20.2. Separar `incierto` de `other`

Se discutió que la clase residual física no debería mezclarse con la falta de confianza.

Una mejora clara sería separar:

- `other` como clase física;
- `incierto` como estado de baja confianza;
- y `no_valido` como píxel fuera del footprint.

### 20.3. Comparar explícitamente varios umbrales duros `[implementado]`

En vez de fijar un único umbral, se sugirió producir varias versiones:

- `hard_mass_050`
- `hard_mass_055`
- `hard_mass_060`

y lo mismo para luz.

Esto permitiría inspeccionar visualmente y cuantitativamente qué umbral es más razonable.

Actualización:

- esta comparación ya quedó implementada en el pipeline;
- las variantes se guardan dentro de `*.labels.npz`;
- y el resumen de conteos por umbral queda registrado en `*.summary.json`.

En el caso piloto actual, el comportamiento observado es:

- al subir de `0.50` a `0.55` y `0.60`, disminuyen los spaxels clasificados como `disco`;
- al mismo tiempo aumenta la fracción de spaxels que pasan a `incierto`;
- por lo tanto, esta comparación confirma que subir el umbral sí vuelve más conservadora la etiqueta dura, pero no corrige por sí solo los problemas de fondo en periferia o zonas mixtas.

### 20.4. Añadir regularización espacial a las etiquetas duras

Se discutió implementar una etapa posterior de coherencia espacial para evitar cambios espurios entre vecinos.

Opciones propuestas:

- filtro modal por vecindad;
- eliminación de islas pequeñas;
- relleno de huecos;
- componente conexa principal por clase;
- o modelos más formales tipo `MRF/CRF`.

### 20.5. Mejorar el tratamiento del centro `bulbo vs disco`

Se discutió que el centro necesita una modelización más física.

Las mejoras propuestas fueron:

- hacer al bulbo más concentrado;
- debilitar la componente de disco en el núcleo;
- introducir una prior central para bulbo;
- o usar una descomposición más realista de perfil central.

### 20.6. Usar `soft labels` como producto principal y `hard labels` como derivado

Metodológicamente se acordó que:

- `soft_mass` sigue siendo el target principal más defendible;
- `soft_light` debe conservarse como target auxiliar o comparativo;
- `hard_*` debe verse como una versión simplificada para visualización o experimentación, no como la verdad final.

### 20.7. Exponer estas variantes directamente en el visor

También se sugirió que el visor muestre de forma más explícita:

- comparación lado a lado entre masa y luz;
- diferencias entre hard labels con distintos umbrales;
- la máscara válida actual y una máscara válida más estricta;
- mapas de confianza e incertidumbre.

## 21. Recomendación práctica para la siguiente iteración

Dado todo lo anterior, la siguiente iteración metodológica debería priorizar, en este orden:

1. redefinir `valid_mask`;
2. separar `incierto` de `other`;
3. implementar una versión espacialmente regularizada de las hard labels;
4. recalibrar el tratamiento central `bulbo vs disco`;
5. y solo después volver a ajustar umbrales duros.

Este orden es importante porque varios de los errores reportados no provienen del umbral por sí solo, sino de una combinación entre:

- máscara válida demasiado amplia;
- normalización local por spaxel;
- y ausencia de coherencia espacial posterior.

## 22. Aporte del artículo `RealSim-IFS` al problema de etiquetas

En esta etapa también se revisó el artículo:

- [stac1532.pdf](/home/andy/Documents/INAOE/stac1532.pdf)

correspondiente a:

- Bottrell, C. y Hani, M. H., *Realistic synthetic integral field spectroscopy with RealSim-IFS*, MNRAS 514, 2821–2838 (2022).

La conclusión es que el artículo sí aporta información útil para el proyecto, pero principalmente en el plano de:

- realismo observacional del mock;
- comparabilidad simulado-observado;
- y diseño del dataset para transferencia hacia MaNGA.

No aporta una receta directa ni una máscara ya construida para etiquetar por spaxel:

- `bulbo`
- `disco`
- `barra`
- `brazos`

### 22.1. Qué parte del artículo ya está alineada con lo que se hizo

El artículo enfatiza que, para transferir modelos desde datos sintéticos a datos reales, los mocks deben ser estadísticamente comparables al dominio observado.

Eso ya está reflejado en dos decisiones metodológicas que tomamos en este workspace:

- no quedarnos solo con el cubo oficial MaNGIA tal como sale del upstream;
- crear una representación más comparable con MaNGA mediante el proyecto `mangia_logcube_74x74`.

En ese sentido, el artículo respalda de forma conceptual:

- la necesidad de incorporar realism observacional;
- la importancia de respetar footprint, ruido, varianza, reconstrucción espacial y convención del producto final;
- y la idea de que los mismos métodos de análisis deben aplicarse sobre simulación y observación siempre que sea posible.

### 22.2. Qué información del artículo podría ayudarnos directamente a mejorar las etiquetas

Hay al menos tres aportes concretos que sí pueden incorporarse más adelante al pipeline de etiquetas.

#### A. Robustez frente a orientación

El artículo usa:

- cuatro sightlines por galaxia

para aumentar la cobertura del espacio de orientaciones sin redundancia fuerte.

Esto es útil para nuestro problema porque las etiquetas estructurales proyectadas dependen de:

- la inclinación;
- el ángulo de posición;
- la detectabilidad de barra;
- y la visibilidad de brazos.

Por lo tanto, una mejora futura razonable sería construir el dataset de entrenamiento incluyendo varias vistas por galaxia y no una sola proyección.

#### B. Criterio de calidad ligado a resolución numérica

El artículo señala que, cuando el número de partículas estelares baja demasiado, aumenta el desacuerdo entre medidas morfológicas y cinemáticas de fracción de bulbo.

Eso es muy relevante para nuestro pipeline porque sugiere que no toda galaxia simulada debería entrar al conjunto de entrenamiento con el mismo peso o sin filtro.

Una mejora concreta sería introducir:

- un umbral mínimo de calidad basado en número de partículas estelares;
- o una bandera de menor confianza para galaxias peor resueltas.

#### C. Compatibilidad entre cantidades ponderadas por masa y por luz

El artículo remarca que el forward modelling puede aplicarse a cubos de distintas cantidades ponderadas por:

- masa;
- luz;
- velocidad;
- edad;
- metalicidad;
- u otras magnitudes.

Eso es coherente con lo que ya hacemos al conservar en paralelo:

- `soft_mass`
- `soft_light`

y apoya la decisión de no colapsar ambas representaciones en una sola etiqueta sin distinguir su interpretación física.

### 22.3. Qué NO estamos usando del artículo en la construcción actual de etiquetas

En la implementación actual del pipeline de etiquetas no se está usando el artículo como fuente directa de reglas de clasificación morfológica.

En particular, no se está usando para:

- definir formalmente `bulbo`, `disco`, `barra` o `brazos`;
- generar una máscara morfológica por spaxel;
- ni sustituir el catálogo morfológico TNG o la reconstrucción desde la simulación base.

La fuente principal actual de verdad sigue siendo:

- la simulación TNG;
- el cutout del subhalo;
- sus metadatos;
- el catálogo morfológico oficial;
- y la proyección a la geometría observable del cubo.

### 22.4. Cómo debe describirse su rol metodológico

La forma correcta de citar el papel de este artículo dentro del proyecto es:

- como justificación del realismo observacional del mock;
- como respaldo para la comparabilidad con MaNGA;
- y como apoyo para decisiones de diseño del dataset y de transferencia simulado-observado.

No debe describirse como:

- la fuente directa de las etiquetas morfológicas;
- ni como un método que ya nos entregue la segmentación `bulbo/disco/barra/brazos`.

### 22.5. Recomendación práctica derivada de esta revisión

Después de revisar el artículo, las mejoras más defendibles para una siguiente fase serían:

1. incorporar explícitamente varias orientaciones por galaxia en el dataset final;
2. introducir un criterio de calidad basado en resolución estelar o número de partículas;
3. mantener la separación entre targets por masa y por luz;
4. seguir reforzando la comparabilidad observacional con MaNGA antes del entrenamiento final.

En síntesis:

- sí, el artículo es útil;
- sí, ya influye conceptualmente en lo que estamos construyendo;
- pero no es la fuente principal de las etiquetas estructurales.
