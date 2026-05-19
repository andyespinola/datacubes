# Spec: Image Provider

> Módulo: `phase_input/image_provider.py` · Hito: 4 · Depende de: `drpall` (MaNGA), `cube_ifu` (ambos modos)

## Responsabilidad

Producir, para cada entrada del catálogo, un tensor `image_synth` de shape
`(3, H, W)` con la imagen fotométrica multibanda **g, r, i** alineada a la
rejilla de spaxels del cubo IFU correspondiente. Operar en dos modos
intercambiables:

- **Modo MaNGIA (sintético)**: integra el cubo espectral a través de las
  curvas de transmisión SDSS g, r, i.
- **Modo MaNGA (real)**: descarga cutouts fotométricos de SDSS DR17 que
  contienen la galaxia, los re-proyecta al WCS del cubo IFU y los
  resamplea a la rejilla de spaxels.

Ambos modos producen tensores con el **mismo formato, escala física, unidades
y normalización** para que el modelo no pueda distinguir entre dominios
sintético y observacional por características superficiales de la imagen.

## Decisión de diseño: ¿por qué dos modos?

Para MaNGIA no existe imagen fotométrica observada; sintetizarla desde el
cubo es la única opción coherente con la naturaleza simulada del dato.

Para MaNGA real existe la alternativa de sintetizar la imagen integrando
el cubo igual que en MaNGIA, pero esta opción se descarta por tres
razones:

1. **Profundidad fotométrica**. Las imágenes SDSS reales tienen S/N
   integrado más alto que el cubo, porque la fotometría imaging usa más
   tiempo de exposición efectivo por píxel que la espectroscopía IFU.
2. **Resolución espacial**. SDSS imaging tiene pixel scale de
   $0.396\,\text{arcsec}$ con seeing típico $\sim 1.3\,\text{arcsec}$; el
   cubo MaNGA tiene $0.5\,\text{arcsec/spaxel}$ con FWHM de PSF
   $\sim 2.5\,\text{arcsec}$.
3. **Convención del survey**. El DRP de MaNGA y los catálogos morfológicos
   asociados (Galaxy Zoo 3D, Domínguez Sánchez et al. 2022, BUDDI-MaNGA)
   usan imágenes SDSS reales, no las recuperadas del cubo. Mantener esta
   convención facilita el cruce con catálogos auxiliares en inferencia
   y validación.

Esta asimetría introduce un *domain shift* menor entre MaNGIA y MaNGA en
la rama fotométrica. Se evalúa explícitamente en QA (sección de
validación abajo) y se documenta como limitación conocida en el modelo.

## Contrato de entrada

```python
class ImageProviderInput(BaseModel):
    mode: Literal["mangia", "manga"]    # determina origen y procedimiento
    galaxy_id: str                       # MaNGIA: TNG50-... | MaNGA: PLATE-IFU
    cube_path: Path                      # FITS del cubo IFU (siempre requerido)
    drpall_row: dict | None = None       # requerido solo en modo "manga"
    cache_dir: Path | None = None        # cache de descargas SDSS

    # Solo en modo "mangia"
    view_id: int | None = None           # orientación 0..3

    config: ImageProviderConfig


class ImageProviderConfig(BaseModel):
    # Filtros y curvas de transmisión
    filter_set: tuple[str, ...] = ("sdss2010-g", "sdss2010-r", "sdss2010-i")

    # Salida
    output_shape: tuple[int, int] = (69, 69)   # MaNGIA nativo; MaNGA puede ser (74, 74)
    output_dtype: str = "float32"

    # Unidad común de salida
    output_unit: Literal["nanomaggie", "ab_flux", "ab_mag_arcsec2"] = "nanomaggie"

    # Solo en modo MaNGIA: nivel de ruido fotométrico simulado a añadir
    add_synthetic_noise: bool = False
    noise_sigma_relative: float = 0.0     # 0 = sin ruido

    # Solo en modo MaNGA: márgenes y servicios
    sdss_cutout_size_arcsec: float = 80.0  # margen sobre el FOV del IFU
    sdss_source: Literal["sas", "skyserver", "astroquery"] = "astroquery"
    sdss_data_release: str = "DR17"

    # Manejo de errores
    on_missing_band: Literal["skip", "interpolate", "fail"] = "fail"
    on_wcs_failure: Literal["fallback_synthesis", "fail"] = "fallback_synthesis"
```

## Contrato de salida

```python
class ProvidedImage(BaseModel):
    """Persisted as NPZ: image_g{galaxy_id}_v{view_id}.npz"""
    galaxy_id: str
    view_id: int | None              # None en modo MaNGA

    image: np.ndarray                # (3, H, W) float32
    band_names: list[str]            # ['g', 'r', 'i']
    unit: str                        # mirror del config

    # Metadatos de proveniencia
    source: Literal["synthesized", "sdss_real"]
    wcs_aligned: bool                # True si las celdas coinciden con el cubo
    fwhm_psf_arcsec: float | None    # PSF efectiva (solo en MaNGA real)
    n_bands_imputed: int             # bandas que tuvieron que interpolarse
```

## Algoritmo: modo MaNGIA (síntesis desde el cubo)

```
1. Cargar cube_flux y cube_wave del FITS:
     cube_flux: (n_wave, n_y, n_x) en unidades de
                10⁻¹⁷ erg/s/cm²/Å/spaxel    (convención MaNGIA)
     cube_wave: (n_wave,) en Å

2. Cargar curvas de transmisión:
     filters = speclite.filters.load_filters(*filter_set)
     # FilterSequence con .effective_wavelengths, .ab_zeropoint, etc.

3. Para cada banda b en filters:
     a. Interpolar T_b(λ) a la rejilla cube_wave
     b. Integrar el cubo a través del filtro:

          F_b(x, y) = ∫ cube_flux(λ, x, y) T_b(λ) dλ / ∫ T_b(λ) dλ

        en unidades de 10⁻¹⁷ erg/s/cm²/Å (densidad de flujo promedio en la
        banda).

     c. Convertir a unidad común de salida (output_unit):
          - "nanomaggie":     F_b * unit_to_nanomaggie(b)
          - "ab_flux":        F_b * convert_to_ab_flux(b)
          - "ab_mag_arcsec2": -2.5 log10(F_b / pixel_area_arcsec2) + ZP

4. Si add_synthetic_noise:
     ruido_b(x, y) ~ N(0, noise_sigma_relative * F_b(x, y))
     F_b ← F_b + ruido_b

5. Apilar bandas: image = stack([F_g, F_r, F_i], axis=0)  → (3, n_y, n_x)

6. Si output_shape != image.shape[1:]:
     image = pad_or_crop(image, output_shape)
     # MaNGIA es nativamente 69×69, no requiere ajuste

7. Guardar NPZ con metadatos
```

## Algoritmo: modo MaNGA (cutouts SDSS reales)

```
1. Leer drpall_row para extraer:
     ra, dec   : centro de la galaxia (deg)
     plateifu  : identificador único
     ifusize   : tamaño del bundle (19, 37, 61, 91 o 127)
     mngtarg1  : selección de muestra

2. Calcular tamaño de cutout:
     ifu_fov_arcsec = ifu_diameter_table[ifusize]  # típico 12–32 arcsec
     cutout_size = max(sdss_cutout_size_arcsec, ifu_fov_arcsec * 2.5)

3. Para cada banda b en filter_set:
     a. Descargar imagen SDSS centrada en (ra, dec) con tamaño cutout_size:

         if sdss_source == "astroquery":
             from astroquery.sdss import SDSS
             pos = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
             query = SDSS.query_region(pos, radius=cutout_size/2*u.arcsec,
                                       data_release=17)
             frames = SDSS.get_images(matches=query, band=b[-1],
                                      data_release=17)

         elif sdss_source == "skyserver":
             url = (f"http://skyserver.sdss.org/dr17/SkyServerWS/"
                    f"ImgCutout/getfits?ra={ra}&dec={dec}"
                    f"&width={cutout_size_pix}&height={cutout_size_pix}"
                    f"&filter={b[-1]}")

         elif sdss_source == "sas":
             # Descarga directa del frame que contiene la posición
             # vía SDSS SAS (sas.sdss.org/dr17/eboss/photoObj/frames/...)

     b. Validar WCS del cutout: WCS(header) debe ser consistente

     c. Cachear en cache_dir/{plateifu}/{b}.fits para reuso

4. Leer header WCS del cubo MaNGA:
     wcs_manga = WCS(cube_fits['FLUX'].header).celestial
     shape_target = (cube_fits['FLUX'].data.shape[1],
                     cube_fits['FLUX'].data.shape[2])
     # típicamente (74, 74) para MaNGA primaria

5. Para cada banda b:
     a. Cargar cutout SDSS con su WCS:
           wcs_sdss = WCS(sdss_frame.header).celestial

     b. Re-proyectar al WCS del cubo MaNGA:
           from reproject import reproject_interp
           reprojected_b, _ = reproject_interp(
               (sdss_frame.data, wcs_sdss),
               wcs_manga, shape_out=shape_target,
           )

     c. Verificar conservación de flujo total:
           |sum(reprojected_b) - sum(sdss_frame.data) * pixel_ratio| / sum < 0.05

     d. Convertir de nanomaggies SDSS a output_unit (mismo procedimiento
        que en modo MaNGIA, asegurando equivalencia numérica)

6. Apilar bandas: image = stack([rep_g, rep_r, rep_i], axis=0)

7. Si output_shape != image.shape[1:]:
     image = pad_or_crop(image, output_shape)

8. Guardar NPZ con metadatos (incluyendo fwhm_psf_arcsec del cutout SDSS)
```

## Procesamiento por catálogo completo

```python
class CatalogImageBuilder:
    """Orquestador para generar imágenes de todo el catálogo."""

    def build_mangia_catalog(
        self,
        mangia_root: Path,
        output_dir: Path,
        config: ImageProviderConfig,
        n_workers: int = 8,
    ) -> CatalogReport:
        """
        Procesa las ~10 000 galaxias de MaNGIA × 4 orientaciones = ~40 000
        imágenes sintéticas. Cada una toma ~50 ms en CPU.
        Tiempo total estimado: ~40 min con 8 workers.
        """
        entries = list((mangia_root / "cubes").glob("*.fits"))
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(self._process_mangia_one, e, output_dir, config)
                for e in entries
            ]
            results = [f.result() for f in tqdm(futures)]
        return CatalogReport.from_results(results)

    def build_manga_catalog(
        self,
        drpall_path: Path,
        cubes_dir: Path,
        output_dir: Path,
        config: ImageProviderConfig,
        n_workers: int = 4,                   # menor por límites de red
    ) -> CatalogReport:
        """
        Procesa las ~10 000 galaxias de MaNGA. Cada una requiere:
        - 3 descargas SDSS (~1–3 s por banda con caché frío)
        - re-proyección WCS (~0.5 s)
        - persistencia (~0.1 s)

        Tiempo total estimado: ~6 h con 4 workers en caché frío,
        ~30 min con caché caliente. Descarga acumulada ~5–10 GB.
        """
        drpall = Table.read(drpall_path)
        # Filtrar entradas problemáticas
        drpall = drpall[drpall['mngtarg1'] > 0]    # solo muestra principal
        drpall = drpall[~drpall['plateifu'].isna()]

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(self._process_manga_one, row, cubes_dir,
                            output_dir, config)
                for row in drpall
            ]
            results = [f.result() for f in tqdm(futures)]
        return CatalogReport.from_results(results)
```

## Manifest del catálogo

Cada llamada a `build_*_catalog` genera un manifest CSV con una fila por
entrada procesada:

```
galaxy_id,view_id,mode,status,source,fwhm_psf,n_bands_imputed,wcs_aligned,output_path
TNG50-87-141934-0-127,0,mangia,ok,synthesized,,0,True,images/mangia/TNG50-87-141934-0-127_v0.npz
TNG50-87-141934-0-127,1,mangia,ok,synthesized,,0,True,images/mangia/TNG50-87-141934-0-127_v1.npz
...
8485-1901,,manga,ok,sdss_real,1.31,0,True,images/manga/8485-1901.npz
8485-1902,,manga,ok,sdss_real,1.42,0,True,images/manga/8485-1902.npz
8485-1903,,manga,fallback,synthesized,,0,False,images/manga/8485-1903.npz
...
```

El estado `fallback` se asigna cuando la descarga SDSS falla y la
política `on_wcs_failure="fallback_synthesis"` reemplaza la imagen real
por una sintetizada del cubo. Estos casos se reportan separadamente para
auditoría.

## Notas de implementación

### Unidad común de salida

Por defecto se usa **nanomaggies por spaxel**, que es la unidad nativa de
las imágenes SDSS calibradas y se traduce fácilmente desde el cubo:

$$
F_{\text{nMgy}} = F_{\text{erg/s/cm}^2\text{/Å}}\cdot
                  \frac{\lambda_{\text{eff}}^2}{c}\cdot 10^{9}\cdot
                  10^{-(48.6 - ZP_{AB})/2.5}.
$$

`speclite` proporciona la conversión exacta vía
`filter.convolve_with_function()` y `filter.get_ab_maggies()`. No
hay que reimplementarlo desde cero.

### Cache de descargas SDSS

Las descargas SDSS deben cachearse agresivamente:
- Ruta: `cache_dir/{plate}/{plateifu}_{band}.fits`
- TTL infinito (los datos SDSS DR17 no cambian).
- Validación: comparar `MD5` del archivo descargado contra `MD5SUM` del
  servidor SDSS para detectar descargas corruptas.

Con caché caliente, el procesamiento de las ~10 000 galaxias se reduce
de ~6 horas a ~30 minutos.

### PSF matching (opcional)

Para experimentos donde la consistencia de PSF entre MaNGIA y MaNGA es
crítica, se puede degradar las imágenes SDSS reales a la PSF de MaNGA
mediante convolución gaussiana:

```python
sigma_match = sqrt(fwhm_manga**2 - fwhm_sdss**2) / 2.355
image_matched = gaussian_filter(image, sigma=sigma_match, axes=(1, 2))
```

Esta funcionalidad se controla por flag separado
(`match_psf_to_manga: bool = False`) y queda fuera del scope de la
versión inicial del módulo.

## Validación

### Tests unitarios

1. **Carga de filtros**:
   `filters = load_filters('sdss2010-g', 'sdss2010-r', 'sdss2010-i')`
   no lanza excepción; cada filtro tiene `effective_wavelength` razonable.

2. **Síntesis sobre SSP conocido**:
   Para un espectro SSP de edad 5 Gyr, $Z = Z_\odot$, comparar la magnitud
   sintetizada con la magnitud reportada por la biblioteca SSP nativa.
   Tolerancia: $\le 0.05\,\text{mag}$ por banda.

3. **Conservación de flujo en re-proyección**:
   Aplicar `reproject_interp` a una imagen sintética de prueba; el flujo
   total debe conservarse dentro del $\le 5\%$ (las pérdidas pequeñas son
   atribuibles a píxeles de borde y a interpolación).

4. **Equivalencia de unidades**:
   Sintetizar la misma fuente puntual en modo MaNGIA y descargar la misma
   fuente en modo MaNGA real (galaxia de prueba); ambas imágenes deben
   tener la misma magnitud integrada dentro de $\le 0.3\,\text{mag}$.

5. **Robustez frente a banda faltante**:
   Con `on_missing_band="interpolate"`, si una banda SDSS no puede
   descargarse, interpolar linealmente entre las dos bandas adyacentes.
   Verificar que el flag `n_bands_imputed` se actualiza correctamente.

### Test de integración con piloto

- **Piloto MaNGIA** (TNG50-87-141934-0-127, vistas 0–3):
  Sintetizar las 4 imágenes; verificar `image.shape == (3, 69, 69)`,
  `image.sum()` consistente entre orientaciones dentro de $\le 30\%$
  (variación esperada por geometría), `image >= 0` (sin píxeles negativos
  espurios).

- **Piloto MaNGA real** (galaxia 8485-1901 si está disponible, o
  cualquier galaxia con shape conocido):
  Descargar las 3 bandas, re-proyectar, verificar
  `image.shape == (3, 74, 74)`, WCS alineado con el cubo (centro a $\le
  0.1\,\text{arcsec}$).

- **Test cruzado MaNGIA vs.\ MaNGA sintetizado**:
  Para una galaxia MaNGA específica, generar la imagen tanto por el modo
  MaNGA (descarga SDSS) como por el modo MaNGIA aplicado a su propio cubo
  (síntesis desde el cubo). Comparar perfiles radiales: deben coincidir
  cualitativamente dentro de $\le 30\%$, con la imagen sintetizada
  típicamente más suave y con menor profundidad efectiva.

### Tests de catálogo completo

Después de procesar el catálogo:

- $\ge 99\%$ de las entradas MaNGIA deben tener `status="ok"`.
- $\ge 95\%$ de las entradas MaNGA deben tener `status="ok"` (admite
  algunos fallbacks por galaxias en bordes de placas SDSS).
- Las imágenes con `n_bands_imputed > 0` no deben superar el $1\%$ del
  catálogo.
- Distribución de magnitudes integradas en cada banda razonable
  (sin colas extremas que indiquen errores de unidad).

## Edge cases

| Caso | Síntoma | Tratamiento |
|---|---|---|
| Galaxia MaNGA en borde de placa SDSS | Frame parcial, NaN en bordes | Padding con cero, marcar `wcs_aligned=False` |
| Cubo MaNGIA con flujo negativo (ruido del proceso de mock) | Píxeles negativos en `image` | Clip a cero antes de guardar |
| Banda SDSS no disponible para una galaxia | Descarga falla con 404 | Política `on_missing_band` |
| WCS del cubo MaNGA corrupto | `WCS(header)` lanza excepción | Fallback a síntesis o `fail` según política |
| Galaxia muy extendida que excede `cutout_size` | Brillo en borde no nulo | Aumentar `cutout_size` dinámicamente |
| Cache corrupto | Lectura del FITS falla | Re-descargar, invalidar cache |
| Galaxia con $z > 0.15$ donde MaNGA no observa | No aplica (todos en muestra MaNGA tienen $z<0.15$) | --- |
| Galaxia con vecino brillante cercano (estrella) | Imagen contaminada en una banda | Reportar en QA; no excluir automáticamente |

## Criterios de aceptación

- [ ] Modo MaNGIA produce imágenes para las 4 vistas del piloto en
      $\le 30\,\text{s}$ totales.
- [ ] Modo MaNGA descarga, re-proyecta y persiste una galaxia de prueba en
      $\le 60\,\text{s}$ con caché frío.
- [ ] Tests unitarios pasan.
- [ ] El manifest del catálogo se genera correctamente con las columnas
      requeridas.
- [ ] Equivalencia de unidades MaNGIA vs.\ MaNGA verificada en la galaxia
      de test cruzado.
- [ ] Caché persistente funciona: segunda ejecución sobre la misma
      galaxia MaNGA tarda $\le 1\,\text{s}$.
- [ ] $\ge 99\%$ de éxito en catálogo MaNGIA, $\ge 95\%$ en catálogo MaNGA.
- [ ] Documentación de limitaciones (asimetría sintético/real) está
      visible en el README del módulo.

## Integración con el Packer

El Packer (módulo final del pipeline que ensambla el HDF5 de entrada del
modelo) consume el NPZ producido por este módulo y lo guarda como
`inputs/image_synth` en el HDF5 final, con la misma shape contratada en
spec `10_dataset.md` (`(3, 69, 69)` para MaNGIA, `(3, 74, 74)` para MaNGA).

El Packer también debe:
- Verificar consistencia entre el `galaxy_id`/`view_id` del NPZ y el del
  HDF5 que está construyendo.
- Copiar los metadatos relevantes (`source`, `fwhm_psf_arcsec`,
  `n_bands_imputed`) como atributos HDF5 bajo `inputs/image_synth.attrs`.
- Marcar entradas con `wcs_aligned=False` en el campo `qa/flags` del HDF5
  para que el entrenamiento pueda filtrarlas si se desea.

## Dependencias nuevas

Agregar al `pyproject.toml` del pipeline de etiquetado:

```toml
[project.optional-dependencies]
imaging = [
    "speclite>=0.18",
    "astroquery>=0.4.7",
    "reproject>=0.13",
    "scipy>=1.10",        # para gaussian_filter, ndimage
]
```

`speclite` ya estaba listada en el plan maestro para la síntesis MaNGIA.
`astroquery` y `reproject` son nuevas dependencias para el modo MaNGA real.

## Estimaciones de tiempo y storage

| Tarea | Tiempo (8 workers) | Almacenamiento |
|---|---|---|
| Catálogo MaNGIA (10 000 × 4 vistas) | $\sim 40\,\text{min}$ | $\sim 4\,\text{GB}$ NPZ |
| Catálogo MaNGA (10 000, caché frío) | $\sim 6\,\text{h}$ | $\sim 8\,\text{GB}$ FITS cache + $\sim 1\,\text{GB}$ NPZ |
| Catálogo MaNGA (caché caliente) | $\sim 30\,\text{min}$ | mismo |
| Storage acumulado para ambos catálogos completos | --- | $\sim 13\,\text{GB}$ |

## Roadmap de implementación

1. **Fase 1 (1 día)**: implementación del modo MaNGIA con tests unitarios y
   piloto. Esta parte solo requiere `speclite` y se valida contra el caso
   piloto ya existente.
2. **Fase 2 (1 día)**: implementación del modo MaNGA real para una sola
   galaxia con caché manual; verificación de WCS y unidades.
3. **Fase 3 (medio día)**: orquestación de catálogo completo, manifest,
   paralelización.
4. **Fase 4 (1 día)**: integración con el Packer y regeneración de los
   HDF5 del piloto.

Total estimado: **3.5 días** de desarrollo + tiempo de cómputo de
ejecución de catálogo.
