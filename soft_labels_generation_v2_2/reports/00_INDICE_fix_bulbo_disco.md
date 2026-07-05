# Índice maestro — Fix inversión bulbo/disco (v2.2) y validaciones

> 2026-07-04. Punto de entrada a TODA la documentación del hallazgo, la
> corrección y las validaciones sobre las etiquetas estructurales v2. Pensado
> como fuente única para (a) reescribir el artículo IBERAMIA y (b) el capítulo
> de metodología de la tesis.

## Qué pasó, en una frase

La regla de reordenamiento de componentes GMM del artículo/pipeline v2.1
invertía las etiquetas de **bulbo y disco en el 23% de las galaxias**; se
corrigió con una asignación conjunta de roles por permutación, se reprocesó
la muestra completa (94 galaxias) y se validó la corrección y la base física
del método de forma independiente.

## Documentos (orden de lectura recomendado)

| # | Documento | Contenido |
|---|---|---|
| 1 | [`revision_articulo_IBERAMIA.md`](revision_articulo_IBERAMIA.md) | **Para reescribir el paper.** Sección por sección: qué afirmación cambia, evidencia, figuras a regenerar, checklist. |
| 2 | [`../docs/adrs/ADR-002-reorder-permutacion-bulbo-disco.md`](../docs/adrs/ADR-002-reorder-permutacion-bulbo-disco.md) | **Decisión formal.** Contexto, mecanismo del defecto, la regla nueva, consecuencias, alternativas. |
| 3 | [`investigacion_inversion_bulbo_disco.md`](investigacion_inversion_bulbo_disco.md) | **Apéndice técnico completo** (11 secciones): cuantificación, causa raíz, literatura, prototipo, reproceso, validación octree-vs-TNG, barrido de fusiones. |
| 4 | [`../specs/11_classifier.md`](../specs/11_classifier.md) | Spec del clasificador actualizado a v2.2 (algoritmo de reordenamiento nuevo + gate de QA). |
| 5 | [`inspeccion_visual/`](inspeccion_visual/) | Paneles PNG por galaxia (segmentación + evidencia física σ*/Σ*/v*). |

## Resultados clave (números para citar)

| Resultado | Valor | Fuente |
|---|---|---|
| Prevalencia de la inversión | 23% (22/94) partículas · 18% (17/94) proyectado | inv. §1, §7 |
| Inversiones tras el fix | **0** (17→0, 0 regresiones, p<10⁻⁶) | inv. §7 |
| Galaxias sin ningún cambio | 69/94 | inv. §7 |
| Fidelidad octree vs potencial TNG | ρ(ε)=0.987, acuerdo etiquetas 97.5% | inv. §10 |
| Sistemas en fusión (limitación) | 2/93 (2%) | inv. §11 |
| Patrón pajarita / proyección inclinada (limitación) | 3/92 (3%) | inv. §12 |
| MaNGIA en snapshots sin Potential | 97.8% | inv. §9 |
| Memoria octree (galaxia mayor, 102M fuentes) | ~23 GB pico (168 B/fuente) | inv. §8 |

## Scripts reproducibles (en `../scripts/`)

| Script | Qué hace | Salida |
|---|---|---|
| `prototype_reorder_fix.py` | prueba la regla nueva sobre las 94 sin re-ajustar GMM | consola (22→0) |
| `quantify_inversion.py` | mide la inversión por perfil radial | `output/inversion_bulge_disk.csv` |
| `compare_pre_post.py` | compara backup pre-fix vs reproceso | `output/comparacion_pre_post_fix.csv` |
| `report_pre_post.py` | mosaico visual pre/post de las corregidas | `output/comparacion_pre_post_fix.pdf` |
| `compare_octree_vs_tng.py` | valida octree contra Potential de TNG (snap 91) | consola (ρ=0.987) |
| `detect_mergers.py` | barrido de sistemas en fusión | `output/merger_sweep.csv`, `merger_flagged.txt` |
| `detect_bowtie.py` | barrido de patrón pajarita (proyección inclinada) | `output/bowtie_sweep.csv`, `bowtie_flagged.txt` |
| `render_inspection.py` | paneles de inspección visual por galaxia | `reports/inspeccion_visual/*.png` |

## Artefactos de datos

- **Muestra corregida (94):** `/media/andy/Data/tng/mangia_flat/output/dataset_entries/*.h5` (branch `permutation_v2.2` en las 94).
- **Respaldo pre-fix (referencia):** `/media/andy/Data/tng/mangia_flat_pre_fix_20260703/` (40 GB).
- **Flags de fusión:** `output/merger_flagged.txt` (2 galaxias).
- **Assets snap-91 (validación potencial):** `/media/andy/Data/tng/snap91_validation/`.
- **Código:** `phase_a/classifier.py` (fix) + `tests/unit/test_reorder_v22.py` (casos dorados). Suite v2_2: 37 tests en verde.

## Estado y pendientes

- [x] Fix aplicado y testeado; 94 re-etiquetadas; 17→0 inversiones.
- [x] Base física validada (octree ≈ potencial TNG).
- [x] Inspección visual (fix confirmado sobre σ*/Σ*).
- [x] Barrido de fusiones (2 flags).
- [x] Documentación completa (este índice + 4 documentos + spec + figuras).
- [ ] **Artículo IBERAMIA:** reescribir según `revision_articulo_IBERAMIA.md`.
- [ ] Recalcular tablas de fracciones/acuerdo del artículo con el reordenamiento nuevo.
- [ ] Decidir tratamiento de las 2 fusiones (excluir / marcar / aceptar).
- [ ] 6 galaxias grandes con OOM del octree: pendientes para la máquina de 128 GB (no bloquean).
- [ ] 1 galaxia (TNG50-88-205585) sin `cube_maps` (producto de reconstrucción MaNGIA local).
