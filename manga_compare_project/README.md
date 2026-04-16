# MaNGA Compare Project

Proyecto limpio para descargar un producto real de MaNGA y visualizarlo con la misma logica general con la que venimos inspeccionando los cubos mock `manga-like`.

## Objetivo

Este proyecto sirve para:

- descargar un `LOGCUBE`, `LINCUBE`, `LOGRSS` o `LINRSS` real de MaNGA;
- abrirlo en un visor web local;
- comparar visualmente un producto real de MaNGA con un cubo mock tipo MaNGIA si copias ambos en `data/`.

## Que soporta el visor

- cubos 3D MaNGA DRP reales con extension `FLUX`;
- RSS 2D MaNGA DRP reales con `FLUX`, `WAVE`, `XPOS`, `YPOS`;
- cubos mock con `PRIMARY` 3D como los generados en este workspace.

## Quick start

1. Crea el entorno:

```bash
bash scripts/bootstrap.sh
```

2. Descarga un ejemplo real de MaNGA:

```bash
bash scripts/download_sample.sh
```

Por defecto descarga:

- `plateifu = 7443-12703`
- `product = LOGCUBE`
- `release = dr17`
- `drpver = v3_1_1`

3. Lanza el visor:

```bash
bash scripts/run_viewer.sh
```

4. Abre:

```text
http://127.0.0.1:8010
```

## Cambiar entre cubo y RSS

Si quieres un RSS real en vez de un cubo:

```bash
MANGA_PRODUCT=LOGRSS bash scripts/download_sample.sh
```

o cambia `MANGA_PRODUCT` en `.env`.

## Comparar con tu mock

Si quieres comparar con un cubo mock `manga-like`, copia ese FITS dentro de `data/` tambien. El visor listara ambos archivos y podras cambiar entre ellos desde el selector.

## Estructura

- `download_manga_drp.py`: descarga un producto DRP real de MaNGA desde el SAS oficial.
- `manga_compare_viewer.py`: visor web local para cubos y RSS.
- `viewer_static/`: frontend del visor.
- `.env`: configuracion local.

## Nota tecnica sobre las fuentes oficiales

La URL del downloader sigue el patron publico del SAS de SDSS para MaNGA DRP, consistente con los enlaces de descarga mostrados por Marvin en DR17 para targets como `7443-12703`.

Referencias oficiales:

- SDSS Marvin galaxy page for 7443-12703:
  https://magrathea.sdss.org/marvin/galaxy/7443-12703/
- SDSS Data Access overview:
  https://www.sdss.org/dr19/data_access/
- SDSS sdss-access package:
  https://www.sdss.org/dr19/software/packages/sdss-access/
