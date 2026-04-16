# datacubes

Workspace para reconstruir, armonizar, visualizar y etiquetar cubos MaNGIA/MaNGA-like.

El punto de entrada detallado es [`GUIA_PROYECTOS_WORKSPACE.md`](GUIA_PROYECTOS_WORKSPACE.md). Esta raiz queda preparada para subirse a GitHub con codigo, documentacion y el catalogo MaNGIA, pero sin productos cientificos pesados.

## Proyectos incluidos

- `rss_to_cube_base.py`: reconstruccion simple RSS -> cubo para inspeccion y pruebas rapidas.
- `rss_to_cube_mangia_official.py`: wrapper local del flujo oficial MaNGIA.
- `cube_web_viewer.py`: visor web interactivo de cubos, espectros por spaxel y etiquetas estructurales.
- `structural_labeling/`: pipeline de etiquetas `bulbo/disco/barra/brazos/other/incierto` desde verdad de simulacion TNG.
- `mangia_logcube_74x74/`: armonizacion a un producto `MaNGA LOGCUBE-like 74x74`.
- `manga_compare_project/`: descarga y visualizacion comparativa de un cubo/RSS real de MaNGA.
- `deploy_mangia_10k/`: proyecto portable para reconstruccion batch a gran escala.

## Instalacion local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Visualizador principal

```bash
python cube_web_viewer.py
```

Luego abrir:

```text
http://127.0.0.1:8000
```

El visor soporta cubos con flujo en `PRIMARY` o en una HDU `FLUX`, y expone overlays de etiquetas:

- `soft_mass`
- `soft_light`
- `hard_mass` / `hard_light` default
- `hard_mass_050`, `hard_mass_055`, `hard_mass_060`
- `hard_light_050`, `hard_light_055`, `hard_light_060`

## Politica de datos para GitHub

Este repositorio esta pensado para versionar codigo y documentacion, no productos cientificos pesados. El archivo [`.gitignore`](.gitignore) excluye por defecto:

- cubos y RSS `*.fits` / `*.fits.gz`;
- cutouts y catalogos descargados `*.h5` / `*.hdf5`;
- salidas `*.npz`;
- paquetes `*.tar.gz`;
- carpetas `data/`, `cache/` y `output/` generadas por los proyectos.

Excepcion importante:

- `MaNGIA_catalog.fits` queda permitido explicitamente porque es el catalogo local necesario para resolver metadatos como `snapshot`, `subhalo_id`, `view` y `re_kpc`.

## Assets externos necesarios

Algunos flujos requieren datos que no se suben al repositorio:

- RSS MaNGIA de entrada.
- Cubos MaNGA reales descargados para comparacion.
- Cutouts TNG y catalogos suplementarios descargados con API key.
- El template SSP `MaStar_CB19.slog_1_5.fits.gz` usado por el flujo oficial MaNGIA.

Las rutas esperadas y comandos de cada flujo estan documentados en [`GUIA_PROYECTOS_WORKSPACE.md`](GUIA_PROYECTOS_WORKSPACE.md).



Si tienes GitHub CLI instalado y autenticado, se puede hacer en una sola secuencia:

```bash
gh repo create datacubes --private --source=. --remote=origin --push
```
