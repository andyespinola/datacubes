# Proyecto `mangia_logcube_74x74`

## 1. Objetivo

Este proyecto toma un RSS de MaNGIA, ejecuta primero la reconstrucción oficial intacta y luego armoniza ese cubo a una convención **`MaNGA LOGCUBE-like 74x74`**.

El objetivo no es reemplazar el cubo oficial de MaNGIA, sino generar una segunda salida trazable y comparable con un `LOGCUBE` real de MaNGA para:

- comparación visual cubo a cubo;
- entrenamiento y transferencia de modelos hacia MaNGA;
- validación metodológica del dominio simulado frente al dominio observado.

## 2. Motivación

En el entorno actual:

- el cubo oficial MaNGIA de referencia sale en `69x69`;
- el `LOGCUBE` real de MaNGA usado como template está en `74x74`;
- ambos comparten una escala espacial cercana, pero no la misma caja espacial final;
- para comparación visual, ML y transferencia a MaNGA conviene una representación homogénea `74x74`.

Por eso este proyecto mantiene intacta la reconstrucción oficial y agrega una etapa explícita de armonización geométrica y espectral.

## 3. Entradas

Entradas principales:

- RSS FITS MaNGIA;
- `MaNGIA_catalog.fits`;
- template SSP del flujo oficial;
- `LOGCUBE` real de referencia.

Parámetros opcionales más importantes:

- `--include-gas`
- `--outdir`
- `--reference-logcube`
- `--keep-official`
- `--catalog`
- `--template-ssp-control`

## 4. Salidas

Por cada RSS, el flujo puede producir:

- cubo oficial MaNGIA de procedencia:
  - `<prefijo>.cube.fits.gz`
  - `<prefijo>.cube_val.fits.gz`
- cubo armonizado:
  - `<prefijo>.manga_logcube_74x74.fits.gz`
- resumen QA:
  - `<prefijo>.manga_logcube_74x74.summary.json`

## 5. Flujo

El pipeline completo es:

1. reconstrucción oficial MaNGIA;
2. lectura del cubo oficial;
3. lectura del `LOGCUBE` real de referencia;
4. armonización espacial a `74x74`;
5. armonización espectral al grid MaNGA;
6. construcción de `IVAR`;
7. remapeo de `MASK`;
8. escritura del FITS final;
9. validación automática contra el template MaNGA.

## 6. Convenciones del producto final

El producto final sigue este contrato:

- `PRIMARY`: solo metadatos y provenance;
- `FLUX`: cubo científico principal;
- `IVAR`: derivado desde `ERROR`;
- `MASK`: máscara 3D;
- `WAVE`: eje espectral 1D explícito;
- `GAS`: opcional, fuera del núcleo DRP-like;
- `BUNIT = 1E-17 erg/s/cm^2/Angstrom/spaxel`.

El cubo final hereda el grid espacial y el grid espectral del `LOGCUBE` real usado como template.

## 7. Cómo correrlo

Ejemplo típico:

```bash
python /home/andy/pythonprojects/cubes/mangia_logcube_74x74/build_mangia_logcube.py \
  /ruta/al/rss.fits \
  --reference-logcube /home/andy/pythonprojects/cubes/manga_compare_project/data/manga-7443-12703-LOGCUBE.fits.gz \
  --outdir /home/andy/pythonprojects/cubes/mangia_logcube_74x74/output \
  --keep-official
```

## 8. Limitaciones v1

La primera versión asume:

- comparabilidad fuerte con `LOGCUBE`, no reproducción completa del DRP de SDSS;
- `cube_val` queda fuera de alcance;
- la validación principal se hace sobre el caso `127-fiber`;
- el template MaNGA se toma de un `LOGCUBE` real local;
- la armonización se hace **después** del regrid oficial, no modificando el upstream vendorizado.
