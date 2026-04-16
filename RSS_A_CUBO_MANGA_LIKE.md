# RSS a Cubo MaNGA-like

## Objetivo

Este documento explica:

- qué es un archivo RSS;
- cómo se reconstruye a partir de él un cubo espectral tipo MaNGA;
- por qué cada spaxel del cubo final contiene un espectro completo;
- qué hace en este proyecto el reconstruidor simple y qué hace el flujo oficial de MaNGIA.

La explicación está basada en los scripts locales [rss_to_cube_base.py](/home/andy/pythonprojects/cubes/rss_to_cube_base.py), [rss_to_cube_mangia_official.py](/home/andy/pythonprojects/cubes/rss_to_cube_mangia_official.py) y en el código upstream oficial [official_mangia/sin_ifu_clean.py](/home/andy/pythonprojects/cubes/official_mangia/sin_ifu_clean.py).

## 1. Qué es un RSS

RSS significa `Row-Stacked Spectra`.

La idea es simple:

- cada fibra del IFU registra un espectro;
- todos esos espectros se apilan en una matriz 2D;
- un eje corresponde a la longitud de onda;
- el otro eje corresponde a las fibras.

En otras palabras, un RSS no es todavía una imagen 3D regularizada en `(lambda, y, x)`. Es una colección de espectros medidos en posiciones discretas del plano del cielo, una posición por fibra.

Si lo pensamos físicamente:

- una fibra no mide un único número;
- una fibra mide flujo en muchas longitudes de onda;
- por lo tanto, cada fibra aporta una curva completa `F(lambda)`;
- además, cada fibra tiene una posición espacial asociada, por ejemplo `x_ifu` y `y_ifu`.

Eso significa que el RSS contiene dos tipos de información a la vez:

- información espectral: cómo cambia el flujo con la longitud de onda;
- información espacial: en qué punto del IFU fue tomada cada medición.

## 2. Qué suele haber dentro del RSS de MaNGIA

En el flujo oficial de este proyecto, el RSS contiene varias extensiones relevantes. El código upstream las lee en [official_mangia/sin_ifu_clean.py](/home/andy/pythonprojects/cubes/official_mangia/sin_ifu_clean.py#L1889):

- `rss[0]`: espectro principal por fibra;
- `rss[1]`: componente de gas;
- `rss[2]`: valores físicos por fibra;
- `rss[3]`, `rss[4]`, `rss[5]`: descomposiciones auxiliares;
- `rss[6]`: `x_ifu`;
- `rss[7]`: `y_ifu`.

Además, el header del RSS aporta metadatos necesarios para reconstruir el cubo:

- `NAXIS2`, `CRVAL2`, `CRPIX2`, `CDELT2`: definen el eje espectral;
- `IFUCON`: configuración del bundle;
- `PSF`, `REDSHIFT`, `KPCSEC`, `R`: metadatos instrumentales y cosmológicos.

En este proyecto, el script [rss_to_cube_base.py](/home/andy/pythonprojects/cubes/rss_to_cube_base.py) también está preparado para detectar variantes menos ordenadas del RSS:

- busca una HDU de flujo por nombre o por dimensionalidad;
- intenta reconstruir `wave` desde el header si no existe una extensión `WAVE`;
- corrige la orientación del arreglo para llevarlo a `(n_fibers, n_wave)`;
- colapsa `x_ifu` y `y_ifu` a un vector por fibra si vinieran en forma 2D.

## 3. Qué diferencia hay entre RSS y cubo

La diferencia clave es el sistema de muestreo espacial.

En un RSS:

- las observaciones viven en fibras;
- las fibras están en posiciones irregulares o semirregulares;
- no existe todavía una malla cartesiana de píxeles espaciales.

En un cubo espectral:

- los datos viven en una grilla regular `(x, y)`;
- cada celda espacial es un spaxel;
- para cada spaxel existe un espectro completo.

Por eso convertir RSS a cubo no es una simple reorganización de ejes. Es una reconstrucción espacial.

## 4. Qué es un spaxel

Un spaxel es un `spatial pixel`.

Cada spaxel representa una pequeña celda del plano del cielo o del plano IFU. En un cubo espectral, cada spaxel guarda un vector:

`[F(lambda_1), F(lambda_2), ..., F(lambda_N)]`

Así, el cubo completo puede pensarse como:

- dos dimensiones espaciales;
- una dimensión espectral.

## 5. Cómo se pasa del RSS a un cubo

### 5.1. Paso conceptual

La reconstrucción consiste en tomar espectros medidos en posiciones discretas y estimar, para cada punto de una malla regular, qué espectro le corresponde.

Eso requiere:

- conocer dónde está cada fibra;
- definir una grilla espacial de salida;
- decidir una regla de interpolación o mezcla;
- aplicar esa regla a todas las longitudes de onda.

### 5.2. Paso matemático

Si una fibra `k` está en la posición `(x_k, y_k)` y mide el espectro `F_k(lambda)`, entonces para un spaxel centrado en `(x, y)` se calcula un peso espacial `w_k(x, y)`.

Luego, el espectro del spaxel se estima como:

`F_spaxel(lambda) = sum_k [w_k(x, y) * F_k(lambda)] / sum_k [w_k(x, y)]`

La clave es esta:

- los pesos dependen de la geometría espacial;
- el espectro depende de la longitud de onda;
- la combinación se hace longitud de onda por longitud de onda;
- el resultado sigue siendo un espectro completo.

## 6. Por qué cada spaxel puede contener toda la información espectral

Esta es la pregunta más importante.

La respuesta corta es:

porque cada fibra ya trae un espectro completo, y la reconstrucción espacial combina espectros completos, no valores escalares sueltos.

Dicho de otro modo:

- una fibra no aporta solo brillo total;
- aporta una intensidad distinta en cada `lambda`;
- cuando varias fibras contribuyen a un spaxel, lo hacen en todas las longitudes de onda;
- por eso el spaxel final recibe una curva espectral completa.

No hay una “invención” de la dimensión espectral durante la reconstrucción. La dimensión espectral ya estaba en el RSS. Lo que se reconstruye es la distribución espacial.

La reconstrucción no crea espectros desde cero; redistribuye e interpola espectros ya medidos por las fibras sobre una grilla regular.

## 7. Qué hace el flujo oficial de MaNGIA

El wrapper local [rss_to_cube_mangia_official.py](/home/andy/pythonprojects/cubes/rss_to_cube_mangia_official.py) prepara y llama a `regrid()` del upstream oficial. Esa función vive en [official_mangia/sin_ifu_clean.py](/home/andy/pythonprojects/cubes/official_mangia/sin_ifu_clean.py#L1819).

El procedimiento oficial, resumido, es este:

### 7.1. Identificar el bundle IFU y sus parámetros

El wrapper lee `IFUCON` del RSS y lo convierte al identificador `n_fib` esperado por MaNGIA:

- 19 fibras -> `n_fib = 3`
- 37 fibras -> `n_fib = 4`
- 61 fibras -> `n_fib = 5`
- 91 fibras -> `n_fib = 6`
- 127 fibras -> `n_fib = 7`

Esto no significa que haya 7 fibras. Significa que el diseño del bundle corresponde al caso grande de 127 fibras.

### 7.2. Leer espectros, mapas y posiciones

El upstream abre el RSS y carga:

- `spec_ifu`: espectros principales por fibra;
- `spec_ifu_g`: espectros de gas;
- `spec_val`: valores físicos por fibra;
- `x_ifu`, `y_ifu`: coordenadas espaciales de las fibras.

Si el RSS contiene más entradas que el bundle final, el código selecciona solo las fibras necesarias mediante `fib_ind_final` en [official_mangia/sin_ifu_clean.py](/home/andy/pythonprojects/cubes/official_mangia/sin_ifu_clean.py#L1882). El patrón `idx`, `idx+127`, `idx+254` sugiere que el RSS puede incluir varios bloques de observación; esto es una inferencia razonable a partir del código.

### 7.3. Reconstruir el eje espectral

El eje de longitudes de onda no se guarda como lista explícita dentro de `regrid()`. Se reconstruye con:

- `CRVAL2`
- `CRPIX2`
- `CDELT2`
- `NAXIS2`

En [official_mangia/sin_ifu_clean.py](/home/andy/pythonprojects/cubes/official_mangia/sin_ifu_clean.py#L1919), el upstream hace:

`wl = np.arange(crval_w, crval_w + nw * cdelt_w, cdelt_w)`

Eso genera una longitud de onda para cada muestra espectral.

### 7.4. Ajustar resolución y ruido

Antes de la reconstrucción espacial, el flujo oficial:

- degrada la resolución de la componente de gas a la resolución MaNGA;
- opcionalmente añade ruido controlado por `R_eff` y por un `S/N` objetivo.

Por eso nuestro wrapper resuelve `re_kpc` desde [MaNGIA_catalog.fits](/home/andy/pythonprojects/cubes/MaNGIA_catalog.fits) cuando no se pasa `--r-eff` manualmente.

### 7.5. Definir la grilla espacial de salida

El upstream fija:

- `pix_s = 0.5` arcsec por spaxel;
- una extensión espacial `nl x nl` derivada del alcance de `x_ifu` y `y_ifu`;
- una geometría final regular compatible con un cubo tipo MaNGA.

En nuestro caso real, eso produjo un cubo principal con shape `(6603, 69, 69)` en [TNG50-87-141934-0-127.cube.fits.gz](/home/andy/pythonprojects/cubes/TNG50-87-141934-0-127.cube.fits.gz).

### 7.6. Reconstruir cada spaxel mediante pesos espaciales

Este es el corazón del algoritmo.

Para cada posición `(i, j)` de la malla espacial:

- se calcula el centro del spaxel;
- se recorren todas las fibras;
- se mide la distancia del centro del spaxel al centro de cada fibra;
- si la fibra está lo bastante cerca, aporta al spaxel;
- su contribución se pondera con un peso gaussiano.

En el upstream el peso es:

`Wg = exp(-(Rsp / sigma_rec)^2 / 2)`

con un corte espacial de contribución, tal como se ve en [official_mangia/sin_ifu_clean.py](/home/andy/pythonprojects/cubes/official_mangia/sin_ifu_clean.py#L1979).

Luego el código combina:

- el espectro principal;
- el error;
- la componente de gas;
- los mapas físicos auxiliares.

La normalización final es por la suma total de pesos `Wgt`.

### 7.7. Escribir el cubo final

El flujo oficial escribe dos archivos:

- `.cube.fits.gz`
- `.cube_val.fits.gz`

El primero contiene:

- `PRIMARY`: cubo espectral;
- `ERROR`;
- `MASK`;
- `GAS`.

El segundo contiene:

- mapas físicos e intrínsecos por spaxel;
- masa por edad;
- masa por edad y metalicidad;
- luminosidad por edad y metalicidad.

## 8. Qué hace el reconstruidor simple de este proyecto

El script [rss_to_cube_base.py](/home/andy/pythonprojects/cubes/rss_to_cube_base.py) implementa una versión mucho más simple, pensada para inspección, prototipado y validación.

Su procedimiento es:

- localizar la HDU de flujo;
- inferir o leer el eje de onda;
- detectar `x_ifu` y `y_ifu`;
- normalizar la orientación del flujo a `(n_fibers, n_wave)`;
- construir una grilla regular;
- calcular pesos gaussianos fijos en el plano espacial;
- combinar todas las fibras en cada spaxel mediante `einsum`.

La función principal de esta versión es [reconstruct_cube_simple()](/home/andy/pythonprojects/cubes/rss_to_cube_base.py#L283).

La lógica es conceptualmente la misma que en el flujo oficial:

- pesos espaciales por fibra;
- mezcla espectral por longitud de onda;
- escritura del cubo 3D.

La diferencia es que no intenta reproducir exactamente el producto oficial de MaNGIA.

## 9. Intuición física: qué “ve” realmente un spaxel reconstruido

Es importante no malinterpretar el cubo reconstruido.

Un spaxel del cubo:

- no es una medición directa independiente hecha por un detector cuadrado;
- es una estimación reconstruida a partir de varias fibras cercanas;
- hereda correlaciones con spaxels vecinos;
- depende del kernel espacial, del seeing, de la PSF y del esquema de ponderación.

Eso explica por qué dos spaxels adyacentes pueden tener espectros muy parecidos: ambos comparten parte de la misma información de entrada.

Por lo tanto:

- el cubo es una representación regularizada y muy útil;
- pero no debe interpretarse como si cada spaxel proviniera de una observación completamente independiente.

## 10. Por qué el cubo sigue siendo “manga-like”

Se lo llama `manga-like` porque el producto final adopta convenciones similares a MaNGA:

- grilla espacial regular;
- WCS espacial en RA/DEC;
- eje espectral explícito;
- extensiones `ERROR`, `MASK` y `GAS`;
- pixel scale y reconstrucción espacial análogas a un cubo de IFU.

En el upstream esto queda reflejado en el header del cubo:

- `CTYPE1 = RA---TAN`
- `CTYPE2 = DEC--TAN`
- `CUNIT3 = Wavelength [A]`
- `IFUCON`, `PSF`, `FOV`, `REDSHIFT`, `R`

Todo eso se escribe en [official_mangia/sin_ifu_clean.py](/home/andy/pythonprojects/cubes/official_mangia/sin_ifu_clean.py#L2017).

## 11. Ejemplo concreto en este proyecto

Para el archivo RSS:

- [RSS/TNG50-87-141934-0-127.cube_RSS.fits](/home/andy/pythonprojects/cubes/RSS/TNG50-87-141934-0-127.cube_RSS.fits)

el flujo oficial produjo:

- [TNG50-87-141934-0-127.cube.fits.gz](/home/andy/pythonprojects/cubes/TNG50-87-141934-0-127.cube.fits.gz)
- [TNG50-87-141934-0-127.cube_val.fits.gz](/home/andy/pythonprojects/cubes/TNG50-87-141934-0-127.cube_val.fits.gz)

con estas características principales:

- shape del cubo principal: `(6603, 69, 69)`;
- 6603 muestras espectrales;
- 69 x 69 spaxels espaciales;
- extensiones `PRIMARY`, `ERROR`, `MASK`, `GAS`.

## 12. Cómo leer un FITS: header vs data

En un archivo FITS, cada `HDU` tiene dos componentes principales:

- `header`: metadatos;
- `data`: arreglo numérico.

El `header` no contiene el cubo en sí. Contiene la descripción del cubo:

- qué representa cada eje;
- cuáles son las unidades;
- cuál es el tamaño del campo;
- cuál es la PSF;
- si el gas está sumado o separado;
- cómo interpretar espacial y espectralmente el arreglo.

El `data`, en cambio, es el contenido numérico real.

Para el cubo principal de este proyecto:

- [TNG50-87-141934-0-127.cube.fits.gz](/home/andy/pythonprojects/cubes/TNG50-87-141934-0-127.cube.fits.gz)

la HDU `PRIMARY` tiene:

- un `header` con keywords como `PSF`, `IFUCON`, `REDSHIFT`, `CTYPE1`, `CTYPE2`, `CRVAL3`, `CDELT3`, `UNITS`, `WGAS`;
- un `data` con shape `(6603, 69, 69)`.

Eso significa:

- 6603 muestras espectrales;
- 69 x 69 spaxels espaciales;
- para cada posición `(x, y)` existe un espectro de longitud 6603;
- para cada índice espectral `k` existe una imagen 2D del campo.

Además, las otras extensiones también siguen la misma idea:

- `ERROR`: header propio + arreglo 3D de errores;
- `MASK`: header propio + arreglo 3D de máscara;
- `GAS`: header propio + arreglo 3D de emisión gaseosa.

Por eso, cuando se abre un FITS, no basta con mirar solo el `data`. Hace falta leer el `header` para saber qué significa ese arreglo.

En Python, la separación se ve así:

```python
from astropy.io import fits

hdul = fits.open("/home/andy/pythonprojects/cubes/TNG50-87-141934-0-127.cube.fits.gz")

primary_header = hdul[0].header
primary_cube = hdul[0].data

error_header = hdul[1].header
error_cube = hdul[1].data

mask_header = hdul[2].header
mask_cube = hdul[2].data

gas_header = hdul[3].header
gas_cube = hdul[3].data
```

La regla práctica es:

- `header` dice cómo interpretar;
- `data` contiene los valores.

## 13. Resumen corto

Un RSS ya contiene toda la información espectral, pero muestreada en fibras.

La reconstrucción a cubo no inventa la dimensión espectral. Lo que hace es:

- tomar cada espectro por fibra;
- usar la posición espacial de esa fibra;
- interpolar esos espectros sobre una malla regular;
- asignar a cada spaxel un espectro completo.

Por eso cada spaxel del cubo final puede representar “la información espectral de ese punto”: no porque haya sido observado directamente como un píxel aislado, sino porque su espectro se estima combinando, con pesos espaciales, los espectros medidos por las fibras vecinas.
