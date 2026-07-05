# Etiquetado masivo (10k+ galaxias) — instrucciones para Claude Code remoto

> Guía operativa para correr el pipeline de etiquetas estructurales v2.2
> (con el fix bulbo/disco de ADR-002) sobre las 10 000+ galaxias MaNGIA en el
> servidor remoto. **Léela entera antes de ejecutar nada.** Está pensada para
> que Claude Code la siga paso a paso, verificando en cada punto.

## 0. Contexto y principio rector

Los cubos IFU están en una unidad USB. **El etiquetado NO se hace solo con
los cubos**: la Fase A (física) necesita las partículas de la simulación TNG
(cutouts) y la materia oscura; la Fase B (renderizado) usa el cubo y los
mapas pyPipe3D. Antes de procesar hay que **inventariar** qué inputs existen
y **completar los que falten** (probablemente la materia oscura / phase2).

Regla de oro: **no lances el batch completo hasta que el inventario dé
COMPLETO para la mayoría y hayas hecho una prueba con `--limit`.**

## 1. Ubicaciones (BUSCAR y confirmar antes de empezar)

Busca en el servidor y anota las rutas reales (pueden variar):

| Qué | Dónde buscar | Notas |
|---|---|---|
| Repo | donde clonaste `soft_labels_generation_v2_2` | contiene `scripts/`, `src/` |
| Cubos + inputs TNG | USB, p. ej. `/run/media/aespinola/ADATA HM800/datacubes/` | **la ruta tiene un espacio → entre comillas siempre** |
| `MaNGIA_catalog.fits` | **en la USB `datacubes/`** (junto a los cubos) | 0.5 MB; da re_kpc y repeat_count |
| SSP template `MaStar_CB19.slog_1_5.fits.gz` | **en la USB `datacubes/`** | 10.6 MB |
| MORDOR `morphs_kinematic_bars.hdf5` | **en la USB `datacubes/`** | 50 MB; priors del clasificador |
| Disco local rápido para OUTPUT | NO la USB | ver §2 (necesita cientos de GB) |

Comando de búsqueda sugerido:
```bash
find / -name "MaStar_CB19.slog_1_5.fits.gz" 2>/dev/null
find / -name "MaNGIA_catalog.fits" 2>/dev/null
find / -name "morphs_kinematic_bars.hdf5" 2>/dev/null
ls "/run/media/aespinola/ADATA HM800/datacubes/" | head
```
**TODOS los inputs están en la USB `datacubes/`**: cubos, cutouts, phase2,
subhalo.json, cube_maps, y también el catálogo (`MaNGIA_catalog.fits`), el
MORDOR (`morphs_kinematic_bars.hdf5`) y el SSP (`MaStar_CB19...fits.gz`). Los
scripts los resuelven automáticamente desde ahí (prioridad: argumento > USB >
`aux/` del repo, que es solo un respaldo). Verifícalo:
```bash
ls "/run/media/aespinola/ADATA HM800/datacubes/" | grep -E "MaNGIA_catalog|morphs_kinematic|MaStar" 
```
Si alguno NO está en la USB, o lo copias ahí, o pásalo con `--ssp/--catalog/--mordor`.

## 2. Requisitos de disco y memoria (IMPORTANTE)

- **RAM:** cada galaxia grande usa hasta ~23 GB en el octree. Con 128 GB,
  usar `--workers 4` (4×23 = 92 GB, con holgura). No subas workers salvo que
  el inventario muestre galaxias mayormente pequeñas.
- **Disco (output):**
  - `dataset_entries` (SOLO etiquetas, por defecto): ~0.6 MB/galaxia →
    **~6 GB para 10k**. El cubo NO se embebe (es un producto independiente;
    el entry guarda una referencia `metadata.cube_file`). Con `--copy-cube`
    se embebería el cubo (~85 MB c/u → ~850 GB) — NO recomendado.
  - intermedios `phase_a` durante el proceso: ~0.5 GB/galaxia → **~5 TB si NO
    se limpian**. **Usa `--cleanup`**: los borra tras cada entry OK; el pico
    baja a ~(workers × 0.5 GB) ≈ 2 GB. El resume sigue funcionando (mira el
    entry final, no los intermedios).
  - Con `--cleanup` (default de etiquetas-solo) necesitas **~20 GB libres**,
    no 1 TB. El disco de output NO tiene que ser enorme.
- Aun así, el output NO debe ir a la USB (lenta).

## 3. Entorno Python

Python ≥ 3.10 con: `numpy scipy h5py astropy scikit-learn structlog numba
pydantic pyyaml typer matplotlib`. Instala el repo y verifica:
```bash
cd <repo>
pip install -e ".[dev]"            # o pip install numpy scipy h5py astropy scikit-learn structlog numba pydantic typer pyyaml
python -m pytest tests/unit -q     # debe pasar (incluye el fix v2.2)
```
Confirma que el fix está presente:
```bash
grep -n "permutation_v2.2" src/aperturenet_labels/phase_a/classifier.py
# debe aparecer; si no, el repo no tiene el fix — no proceses.
```

## 4. Inventario de inputs (SIEMPRE primero)

```bash
python scripts/inventory_inputs.py \
  --input-dir "/run/media/aespinola/ADATA HM800/datacubes" \
  --check-dm-in-cutout --out inventory.csv
# (el catálogo se autodetecta desde la USB; --catalog solo si está en otro sitio)
```
Lee el resumen. Estados posibles por galaxia:
- `COMPLETO` — listo para procesar.
- `ok_sin_maps_faseB_limitada` — procesa, pero sin mapas pyPipe3D la Fase B
  usa menos información (aceptable si es minoría).
- **`SIN_DM_potencial_sesgado`** — falta la materia oscura. **NO procesar así**
  (el potencial excluiría el halo DM, ~85% de la masa, y ε saldría sesgado).
  → ir al §5 a descargar phase2.
- `FALTA_cutout` / `FALTA_subhalo_json` / `FALTA_en_catalogo` — no procesables;
  quedarán descartadas automáticamente.

**Decisión:** si `has_dm` es alto (la mayoría tiene DM dentro del cutout o
phase2), salta al §6. Si muchas están `SIN_DM`, haz el §5.

## 4.5 Credenciales de TNG (solo si hay que descargar)

La descarga (§5) necesita tu API-Key de TNG. Créala una vez:
```bash
cp aux/.env.example aux/.env
# edita aux/.env y pon tu key de https://www.tng-project.org/users/profile/
```
`aux/.env` está en `.gitignore` (no se sube). Alternativamente:
`export TNG_API_KEY=...`. Si el server no tiene acceso a internet/TNG, avisa
al usuario: la descarga no será posible desde ahí.

## 5. Completar los inputs de TNG que falten (cutouts, subhalo.json, DM)

**No todos los cutouts ni la materia oscura están en el disco.** Los cubos y
`cube_maps` sí (son la reconstrucción MaNGIA local), pero el cutout de
partículas, el `subhalo.json` y el DM se descargan de la API de TNG. Un solo
script baja lo que falte por galaxia:

```bash
# Requiere API-Key de TNG (línea TNG_API_KEY=... en un .env, o export TNG_API_KEY).
python scripts/download_tng_inputs.py \
  --input-dir "/run/media/aespinola/ADATA HM800/datacubes" \
  --out-dir   <dir_escribible>   \   # si la USB es de solo-lectura, usa OTRO dir
  --env-file  aux/.env --what all --workers 4
```
Por galaxia con cubo, descarga solo lo ausente:
- `*.cutout.hdf5` (estrellas+gas) si no existe,
- `*.subhalo.json` (metadatos) si no existe,
- `*.cutout_phase2.hdf5` (DM, query mínima `dm=Coordinates`) si el cutout no
  trae `PartType1` y no hay phase2.

Notas:
- Idempotente/reanudable: salta lo ya presente. `--what {cutout,subhalo,dm}`
  para bajar solo un tipo.
- **Volumen:** el DM domina el peso (decenas–cientos de MB/galaxia). Para 10k
  puede ser **cientos de GB** — verifica ancho de banda y espacio antes.
- Si `--out-dir` ≠ USB, mueve luego los archivos descargados al MISMO
  directorio de los cubos (el pipeline los busca por nombre ahí), o corre el
  batch con un `--input-dir` que los contenga junto a los cubos.
- **Si el servidor no tiene acceso a la API de TNG**, avísale al usuario: sin
  cutouts no se puede etiquetar, y sin DM el potencial sale sesgado.

Vuelve a correr el inventario (§4) hasta que la mayoría esté `COMPLETO`. Ojo:
descargar cutouts de 10k galaxias puede tardar **horas**; lánzalo en
background (`nohup ... &`) y monitorea, igual que el batch (§7).

## 6. Prueba con pocas galaxias (OBLIGATORIO antes del batch completo)

```bash
python scripts/run_batch.py \
  --input-dir  "/run/media/aespinola/ADATA HM800/datacubes" \
  --output-dir /datos/labels_out \
  --workers 4 --timeout-sec 3600 --cleanup --limit 8 --no-qa
# SSP/catálogo/MORDOR se resuelven solos desde la USB (input-dir).
```
Verifica que:
- El manifest reporta un número razonable de "galaxias con inputs completos".
- Las 8 terminan en `ok` (no `error`/`timeout`).
- Se crearon 8 archivos en `/datos/labels_out/output/dataset_entries/`,
  pequeños (~0.6 MB, solo etiquetas — el cubo NO va embebido, es
  independiente; `metadata.cube_file` guarda el nombre del cubo).
- Un entry abre bien y su clasificador usó el fix:
```bash
python - <<'PY'
import h5py, glob
e = sorted(glob.glob("/datos/labels_out/output/dataset_entries/*.h5"))[0]
with h5py.File(e) as f:
    print("claves:", list(f["labels"].keys()))
PY
```
Si algo falla aquí, **para y diagnostica** (revisa `batch_progress.jsonl` y
corre `label_one.py` suelto para esa galaxia para ver el traceback completo).

## 7. Batch completo (desatendido, reanudable)

Lánzalo en background para que sobreviva a la sesión:
```bash
mkdir -p /datos/labels_out
nohup python scripts/run_batch.py \
  --input-dir  "/run/media/aespinola/ADATA HM800/datacubes" \
  --output-dir /datos/labels_out \
  --workers 4 --timeout-sec 3600 --cleanup \
  > /datos/labels_out/batch.log 2>&1 &
echo "PID $!"
```
- **Reanudable:** si se corta, vuelve a lanzar el MISMO comando; salta las
  galaxias que ya tienen entry.
- **Monitoreo:**
```bash
tail -f /datos/labels_out/batch.log                 # progreso + ETA
wc -l /datos/labels_out/output/dataset_entries/*.h5 2>/dev/null | tail -1
grep -c '"status": "ok"'    /datos/labels_out/batch_progress.jsonl
grep -c '"status": "error"' /datos/labels_out/batch_progress.jsonl
```
- Al terminar escribe `batch_summary.json` y corre los barridos de QA
  (inversión, fusiones, pajarita) sobre el resultado.

## 8. Fallos esperados y qué hacer

| Síntoma | Causa | Acción |
|---|---|---|
| `status: error`, `returncode: -9` | OOM del octree (galaxia enorme) | con 128 GB es raro; si pasa, baja `--workers` o procesa esas sueltas |
| `status: timeout` | galaxia muy grande > timeout | sube `--timeout-sec` y re-lanza (resume) |
| aviso "SIN materia oscura" en el log | phase2 no encontrado | §5 (descargar DM); ε de esas galaxias es poco fiable |
| `FALTA dependencia` al arrancar | falta SSP/catálogo/MORDOR | §1 (coloca los auxiliares en `aux/`) |
| muchas `descartadas: sin_cutout` | faltan cutouts de TNG | §5: `download_tng_inputs.py --what cutout` (o `all`); sin API de TNG, avisar al usuario |

## 9. QA post-proceso (automático, pero revísalo)

`run_batch.py` corre al final (salvo `--no-qa`):
- `quantify_inversion.py` → `inversion_bulge_disk.csv` (esperado: ~0 inversiones
  de anillo, porque el fix está aplicado).
- `detect_mergers.py` → `merger_sweep.csv` + `merger_flagged.txt` (~2% pares).
- `detect_bowtie.py` → `bowtie_sweep.csv` + `bowtie_flagged.txt` (~3% proyección
  inclinada).

Estas dos últimas listas marcan galaxias con etiquetas 2D dominadas por
geometría/fusión (NO son errores del clasificador; ver
`reports/00_INDICE_fix_bulbo_disco.md`). Repórtaselas al usuario para decidir
si se excluyen del entrenamiento.

## 10. Resumen de contexto para el usuario al terminar

Cuando acabes, informa: nº de entries generados, nº ok/error/timeout, cuántas
quedaron `SIN_DM` (si las hubo), y las listas de fusión/pajarita. El estado
completo está en `batch_summary.json` y `batch_progress.jsonl`.
