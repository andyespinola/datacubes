# 10 — Dataset y DataLoader (v3)

> Módulo: `data/dataset.py`, `data/transforms.py`, `data/collate.py` · Hito: 1.
> Cambios v3: pares (señal, certeza), cuatro variantes de etiqueta + `N_eff`,
> size-agnostic, target dual masa/luz, rest-frame opcional. Refs: revisión
> C4, C5, C7, C9, §2.2; spec [45](45_evidence_layers.md).

## Responsabilidad

Lee los `dataset_entry_*.h5` del repo `galstructnet_labels` y entrega samples
normalizados con **certezas instrumentales** y **etiquetas Dirichlet ancladas**
listas para el modelo. Es el primer módulo que se implementa.

## Contrato de entrada (HDF5 v3)

Un directorio con archivos HDF5, uno por galaxia × vista. Tamaño espacial
`(H, W)` **nativo del cubo de la vista** (varía con el bundle: MaNGA usa IFUs
de 19–127 fibras; MaNGIA replica las configuraciones). Nada asume 69 ni 74.

```
dataset_entry_{galaxy_id}_v{view}.h5
├── inputs/
│   ├── cube_ifu                  # (L, H, W) float32, L=6603 en MaNGIA
│   ├── snr_spec                  # (H, W) float32 — S/N mediano 5000–5500 Å
│   ├── cube_ivar                 # (L, H, W) float32 [OPCIONAL, flag
│   │                             #  carry_full_ivar; solo Anexo A de 45]
│   ├── image                     # (3, H, W) float32 (síntesis o SDSS)
│   ├── image_err                 # (3, H, W) float32 [opcional; 0 ⇒ c=1]
│   ├── pipe3d_maps               # (8, H, W): v, sigma, age, Z, mass, av, h3, h4
│   ├── pipe3d_err                # (8, H, W) errores 1σ (h3/h4 de pPXF, spec 26)
│   └── psf_kernel                # (K, K) float32, K impar, suma = 1
├── labels/                       # nombres alineados al proyector (spec 20 labels)
│   ├── Y_mass_raw, Y_mass_psf    # (5, H, W) float32, suman a 1 sobre clases
│   ├── Y_lum_raw,  Y_lum_psf     # (5, H, W) float32   [v2 'light' → v3 'lum']
│   ├── N_eff_raw_mass, N_eff_raw_lum     # (H, W) — Kish por ponderación
│   ├── N_eff_psf_mass, N_eff_psf_lum     # (H, W) — variante PSF (Cambio G)
│   └── class_names               # ['bulge','disk','bar','arm','other']
├── masks/
│   ├── M_valid                   # (H, W) bool
│   ├── M_uncertain_mass, M_uncertain_lum # (H, W) bool
├── qa/                           # atributos
│   └── target_fractions_{mass,lum}       # (5,) — catálogo TNG/MORDOR
```

**Dependencias hacia `galstructnet_labels` (Cambio G, ver MIGRATION):** h3/h4
(+err) empaquetados desde el spec 26; `N_eff` calculado con ambas
ponderaciones y en variante PSF; las cuatro variantes `Y_*` ya existen
(proyector, Paso 6) — solo se empaquetan, **no** se recomputa PSF on-the-fly
(C4: el "Cambio F" de 43 v2 queda anulado).

## Contrato de salida

```python
{
  # señal + certeza por modalidad (certezas ∈ [0,1]; 0 = ignorar)
  "cube":   (L, H, W) f32,   "c_spec": (1, H, W) f32,   # c̄_spec desde snr_spec
  "image":  (3, H, W) f32,   "c_spat": (3, H, W) f32,
  "maps":   (8, H, W) f32,   "c_phys": (8, H, W) f32,
  "psf":    (K, K) f32,
  # etiquetas: ambas ponderaciones, intrínsecas y observadas, + anclas
  "Y_mass": (5,H,W), "Y_mass_obs": (5,H,W), "n_eff_mass": (H,W), "n_eff_mass_obs": (H,W),
  "Y_lum":  (5,H,W), "Y_lum_obs":  (5,H,W), "n_eff_lum":  (H,W), "n_eff_lum_obs":  (H,W),
  # máscaras y física
  "M": (H,W) bool, "M_unc_mass": (H,W) bool, "M_unc_lum": (H,W) bool,
  "w_phys_mass": (H,W) f32,        # masa RAW (sin z-score) ≥0, para L_phys (C7)
  "target_fractions_mass": (5,), "target_fractions_lum": (5,),
  "galaxy_id": str, "view_id": int,
}
```

Las etiquetas se transponen a canales-primero una sola vez aquí. `Y` v2 (un
solo target) desaparece: el modelo v3 tiene dos cabezas (C3) y la pérdida
consume ambos juegos.

## Certeza desde precisión instrumental

Sustituye al `nan_to_num` de v2 (la garantía "c=0 ⇒ se ignora" la da
NormConv, spec 45 P1):

```python
def to_certainty(sigma, sigma_ref):           # ambos (C, H, W) / (C,1,1)
    c = sigma_ref**2 / (sigma**2 + sigma_ref**2)      # ∈ (0,1]; c(σ_ref)=0.5
    return c

# en __getitem__:
c_phys = to_certainty(pipe3d_err, SIGMA_REF_PHYS)     # (8,H,W)
c_spat = to_certainty(image_err,  SIGMA_REF_SPAT) if has_err else ones
c_spec = (snr_spec / (snr_spec + SNR_REF)).unsqueeze(0)
for c, x in ((c_phys, maps), (c_spat, image)):
    bad = ~M_valid | torch.isnan(x).any?  # por canal: c[ch][nan|~M]=0
    c[..., ~M_valid] = 0.0
    c[torch.isnan(x)] = 0.0
x = torch.nan_to_num(x, 0.0)                  # el valor da igual: c=0 lo anula
```

`SIGMA_REF_*` y `SNR_REF` = medianas del split de train, precalculadas por
`scripts/compute_norm_stats.py` junto a mean/std (un solo JSON versionado).

## Normalización

Idéntica a v2 (log1p + z-score por λ para el cubo; z-score por banda/por mapa;
PSF sin normalizar; etiquetas sin normalizar), con dos añadidos:

- `pipe3d_err` **no** se z-scorea: se consume solo vía `to_certainty`.
- `w_phys_mass` es el mapa de masa **crudo** (antes del z-score), clampeado a
  ≥0; se guarda aparte porque la versión normalizada de `maps[4]` no sirve
  como peso físico (C7).

### Rest-frame opcional (front-end físico, revisión §2.3)

Transformación `RestFrameShift` (flag `restframe: bool`, default off, ablar):
des-desplaza cada espectro usando `v_star` de pyPipe3D
(`λ_rest = λ_obs / (1 + v/c)`), re-muestreando a la rejilla común por
interpolación lineal. Elimina la varianza nuisance dominante (Doppler) antes
del encoder espectral. Donde `c_phys[v]` sea baja, no corregir (v poco
confiable): aplicar solo si `c_phys[0] > 0.5`.

## Augmentations

- `RandomDihedral`: igual que v2, **incluyendo todos los canales de certeza**
  (`c_spec`, `c_spat`, `c_phys`) y los ocho tensores de etiqueta/ancla en la
  misma transformación. PSF y escalares no rotan.
- `SpectralJitter`: ahora calibrado al ruido real — `σ_jitter` proporcional a
  `1/snr_spec` por spaxel en lugar de un 2% global.
- `ChannelDropout(p=0.15)` sobre los canales h3/h4 (índices 6–7): pone
  `c_phys[6:8]=0` (no toca la señal: c=0 basta). Robustez a su ausencia en
  inferencia MaNGA de bajo S/N.
- `PSFJitter`: igual que v2, solo Etapa 4.
- Prohibido (igual que v2): cualquier augmentación del eje λ que rompa su
  asimetría (salvo el rest-frame determinístico de arriba).

## Size-agnostic y collate

`pad_to_74` desaparece. Reglas:

```python
def pad_to_multiple(sample, mult=32):
    """Pad simétrico-aproximado de TODOS los tensores espaciales (señal,
    certeza, etiquetas, anclas, máscaras) al múltiplo `mult` requerido por el
    encoder espacial. La certeza y M_valid se rellenan con 0/False: el modelo
    ignora el padding por construcción (45 P1)."""

def collate_pad(batch):
    """Pad por batch al (H_max, W_max) del batch (ya múltiplos de `mult`).
    Devuelve además 'hw_native' por sample para recortar en evaluación."""
```

`mult` lo declara el encoder espacial (spec 21 v3) — el dataset lo lee de la
config, no lo conoce a priori. Ningún `(69, 69)` literal en este módulo.

## Splits

Por galaxia (las 4 vistas juntas), seeds fijas, archivos
`splits/{train,val,test}.txt` — igual que v2. Añadido v3: lista separada
`splits/manga_unlabeled.txt` (cubos MaNGA para Etapa 1/3) y
`splits/manga_gz3d_{weak,val}.txt` (partición disjunta de GZ3D, ver 70 v3).

## Validación

### Tests unitarios (`tests/unit/test_dataset.py`)

1. **Shapes**: piloto carga; cada tensor con el shape del contrato (H, W
   nativos del archivo, sin literales).
2. **Etiquetas suman a 1** en spaxels válidos para las 4 variantes,
   `atol=1e-2` (acumulación float32; aprendizaje v2.1 — C9).
3. **Certezas en rango**: `c_* ∈ [0,1]`; `c=0` exactamente donde
   `~M_valid` o NaN original.
4. **Anclas coherentes**: `N_eff ≥ 0`; `N_eff_psf` finito; en spaxels con
   `N_eff_raw == 0`, `M_valid` es False (consistencia con MaskBuilder).
5. **Padding**: `pad_to_multiple` deja `(H', W')` múltiplos de `mult`;
   certeza y M en el padding = 0/False; `collate_pad` con tamaños mixtos
   produce batch rectangular + `hw_native` correctos.
6. **Augmentations**: determinísticas con seed; `RandomDihedral` mantiene
   alineación señal↔certeza↔etiqueta (rotar y des-rotar = identidad).
7. **ChannelDropout** apaga `c_phys[6:8]` sin tocar `maps`.
8. **RestFrameShift**: con `v_star ≡ 0` es identidad; con v constante,
   desplaza una línea sintética el número correcto de canales.

### Smoke test

```bash
python -c "
from galstructnet_s3.data import GalStructDataset, collate_pad
ds = GalStructDataset('data/.../dataset_entries', 'train')
s = ds[0]
for k, v in s.items():
    print(k, tuple(v.shape) if hasattr(v, 'shape') else v)
"
```

## Criterios de aceptación

- [ ] Tests 1–8 pasan sobre el piloto.
- [ ] Carga < 1.5 s/sample en SSD (el contrato creció ~10% en bytes; el cubo
      sigue dominando).
- [ ] Sin literales 69/72/74 en `data/` (grep en CI).
- [ ] `compute_norm_stats.py` emite mean/std + `SIGMA_REF_*`/`SNR_REF` en un
      JSON único versionado.
- [ ] Sin leaks de memoria en 100 iteraciones.

## Notas de implementación

- Lazy loading, `num_workers≥4`, `pin_memory`, cuidado con handles HDF5 —
  igual que v2.
- Almacenar `cube_ifu` en float16 en disco es aceptable (la normalización
  log+z-score lo tolera); convierte a f32 al cargar. Recorta I/O ~2×.
- `cube_ivar` completo solo bajo `carry_full_ivar=True` (duplica el peso del
  archivo); por defecto basta `snr_spec`.
- Muestreo de 2 de las 4 vistas por época (`views_per_epoch: 2`) como opción
  de throughput: las vistas comparten partículas (consistencia inter-vista
  medida C_bulge=0.962), la redundancia lo permite.
