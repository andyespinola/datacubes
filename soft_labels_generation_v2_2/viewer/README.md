# Visor web de segmentación estructural (pipeline v2)

Visor interactivo de los `dataset_entry_*.h5` finales. Basado en la
arquitectura de `manga_compare_project/manga_compare_viewer.py` (Flask +
canvas + SVG, sin dependencias de frontend).

## Uso

```bash
cd soft_labels_generation_v2_2
python3 viewer/segmentation_viewer.py            # sirve en http://127.0.0.1:8020
# opciones: --port 8020 --data-dir ../data/output/dataset_entries
```

Requiere `flask` y que existan entries (`python -m aperturenet_labels.cli.main run --pilot`).

## Qué muestra

**Panel izquierdo** — selección de galaxia y de capa:
- **Segmentación**: clase dominante por spaxel (rojo=bulbo, azul=disco,
  naranja=barra, verde=brazo, violeta=halo); la opacidad es proporcional a la
  probabilidad máxima y los spaxels fuera de `M_valid` se atenúan.
- **P(clase)**: probabilidad blanda de una clase (chips para elegirla).
- **N_eff** (Kish), **M_valid**, **cubo medio** y **slice espectral** del cubo IFU.
- **Mapas pyPipe3D**: v★ (centrada en la mediana), σ★, edad, [Z/H], Σ★, Av.
- **Variante de etiqueta**: masa / luz × raw / PSF (las 4 variantes del spec 20).

**Panel central** — el mapa (origen abajo-izquierda, convención astronómica).
Clic en un spaxel para inspeccionarlo.

**Panel derecho** — inspector del spaxel seleccionado:
- barras con las 5 probabilidades de la variante activa,
- espectro completo del cubo IFU en ese spaxel,
- N_eff, pertenencia a M_valid y valores pyPipe3D puntuales.

El panel izquierdo muestra además el resumen QA del entry (status, flags,
conservación de masa, barra detectada, crestas de brazos).
