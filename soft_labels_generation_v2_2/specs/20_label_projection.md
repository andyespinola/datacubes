# Spec: Proyección de Etiquetas 3D → 2D

> Módulo: `phase_b/label_projection.py` · Hito: 3 · Depende de: Classifier + BarDetector + ArmDetector (probabilidades por partícula), Extractor (posiciones, frame), metadatos de orientación de MaNGIA

## Responsabilidad

Transformar las probabilidades estructurales **por partícula** —producidas
por el Classifier, el BarDetector y el ArmDetector— en los tensores de
pseudo-etiquetas **2D por spaxel** `Y_int(i, j, c)`, proyectando las
partículas a la geometría de cada una de las 4 orientaciones que MaNGIA
provee por galaxia.

Este es el paso que materializa las pseudo-etiquetas que el modelo usará
como objetivo de entrenamiento y que el spec de validación
(`40_pseudolabel_validation.md`) consume. **Sin este módulo no existen
proyecciones**; los módulos aguas abajo lo asumen ya ejecutado.

El módulo entrega, por galaxia y por vista:

- Cuatro tensores `Y_int`: `mass_raw`, `mass_psf`, `lum_raw`, `lum_psf`.
- El mapa de conteo efectivo de partículas `N_eff` (insumo del MaskBuilder).
- El mapa de conteo bruto de partículas por spaxel.
- Metadatos de la vista (orientación, PSF).

## Alcance: qué hace y qué NO hace

**Hace**: rotación de partículas a la orientación de cada vista, binning
espacial a la rejilla de spaxels, agregación ponderada de probabilidades,
convolución por PSF, cómputo de `N_eff`.

**No hace**:
- No clasifica partículas — eso es del Classifier/Bar/Arm (specs 11–13).
- No construye la máscara de validez `M_val` — eso es del MaskBuilder
  (spec 22), que consume el `N_eff` que este módulo produce.
- No genera la imagen fotométrica — eso es del Image Provider (spec 25).
- No extrae momentos cinemáticos — eso es del módulo 26.

## Suposiciones sobre los datos

Esta sección es deliberadamente explícita, porque la correcta ejecución
del módulo depende de que estos supuestos se cumplan.

| Supuesto | Detalle | De dónde viene |
|---|---|---|
| Probabilidades por partícula completas | `P_i(c)` sobre las 5 clases, normalizadas, una por partícula | Classifier + BarDetector + ArmDetector |
| Posiciones en el frame centrado | `r_i` recentradas en el centro de la galaxia, con velocidad sistémica sustraída | Extractor |
| Masa, edad y metalicidad por partícula | Necesarias: masa para la variante `mass`, edad+metalicidad para la variante `lum` | Extractor / catálogo TNG |
| Orientaciones de las 4 vistas MaNGIA | Vector de línea de visión por índice de vista (convención v1 portada) | `geometry.py` portado (ver sección final) |
| Rejilla de spaxels por vista | `(H, W)` y escala espacial, idénticos al cubo MaNGIA de esa vista | Cabecera del cubo MaNGIA |
| Kernel de PSF por vista | FWHM de la PSF, para la variante `psf` | Cabecera del cubo MaNGIA |
| Radio de cobertura `R_cov` | `mín(1.5 R_eff, R_IFU)` o `2.5 R_eff` según muestra | Definido por el pipeline (Criterio D) |

### Nota crítica sobre la orientación de las vistas

La proyección **debe usar exactamente la misma orientación** con la que
MaNGIA generó el cubo de cada vista. Si la orientación de la proyección de
etiquetas no coincide con la del cubo, las pseudo-etiquetas quedarán
espacialmente desalineadas respecto a los inputs del modelo, invalidando
el par (input, etiqueta).

Por tanto, el módulo **no elige las orientaciones libremente**: usa la
convención validada del pipeline v1 (vector de línea de visión por
índice de vista; ver la sección "Convención de orientación (RESUELTA)"
al final de este spec y `MIGRATION.md`), y la verifica con el test de
alineación obligatorio contra el mapa de masa de pyPipe3D.

## Contrato de entrada

```python
class LabelProjectionInput(BaseModel):
    galaxy_id: str

    # --- Datos por partícula (en el frame centrado del Extractor) ---
    positions: np.ndarray              # (N, 3) posiciones recentradas, kpc
    velocities: np.ndarray             # (N, 3) velocidades, km/s
    mass: np.ndarray                   # (N,) masa estelar, M_sun
    age: np.ndarray                    # (N,) edad estelar, Gyr
    metallicity: np.ndarray            # (N,) metalicidad [Z/H]
    p_class: np.ndarray                # (N, 5) P_i(c), normalizada
    class_names: tuple[str, ...] = ("bulge", "disk", "bar", "arm", "other")

    # --- Definición de las vistas (de MaNGIA) ---
    views: dict[int, ViewDefinition]   # {view_id: definición}

    # --- Restricción de cobertura ---
    r_eff: float                       # radio efectivo, kpc
    r_cov: float                       # radio de cobertura, kpc

    config: LabelProjectionConfig


class ViewDefinition(BaseModel):
    """Definición geométrica de una vista, derivada de MaNGIA."""
    rotation_matrix: np.ndarray        # (3, 3) rotación al marco de la vista
    grid_shape: tuple[int, int]        # (H, W) rejilla de spaxels
    spaxel_scale_kpc: float            # escala espacial por spaxel, kpc
    fwhm_psf_kpc: float                # FWHM de la PSF en kpc
    inclination_deg: float             # inclinación resultante de la vista


class LabelProjectionConfig(BaseModel):
    # Biblioteca SSP para la ponderación por luminosidad
    ssp_library_path: Path
    luminosity_band: Literal["g", "r", "i", "pseudoV"] = "r"

    # Estrategia de binning
    binning: Literal["nearest", "cic"] = "cic"   # cic = cloud-in-cell

    # Numérico
    epsilon: float = 1e-8              # estabilidad en divisiones

    # Variantes a producir
    produce_variants: tuple[str, ...] = (
        "mass_raw", "mass_psf", "lum_raw", "lum_psf",
    )
```

## Contrato de salida

```python
class ProjectedLabels(BaseModel):
    """Persisted as NPZ: labels2d_g{galaxy_id}_v{view_id}.npz"""
    galaxy_id: str
    view_id: int

    # Cuatro variantes del tensor de pseudo-etiquetas, cada una (5, H, W)
    Y_mass_raw: np.ndarray
    Y_mass_psf: np.ndarray
    Y_lum_raw: np.ndarray
    Y_lum_psf: np.ndarray

    # Insumos para el MaskBuilder
    n_eff: np.ndarray                  # (H, W) conteo efectivo (Kish)
    n_particles_map: np.ndarray        # (H, W) conteo bruto de partículas

    # Metadatos de la vista
    inclination_deg: float
    fwhm_psf_kpc: float
    n_particles_within_rcov: int
    fraction_clipped_by_rcov: float    # fracción de masa fuera de R_cov
```

## Algoritmo

### Paso 1 — Selección por cobertura

```
Calcular el radio esférico de cada partícula en el frame centrado:
    r_i = || positions[i] ||

Construir la máscara de cobertura:
    within_rcov = (r_i <= r_cov)

Solo las partículas dentro de R_cov se proyectan. Esto refleja que MaNGA
real no observa más allá de esa cobertura.
```

### Paso 2 — Ponderación por luminosidad (preparación de la variante `lum`)

```
Para cada partícula, calcular la luminosidad en la banda elegida:

    L_i = mass[i] * ell_b(age[i], metallicity[i])

donde ell_b(t, Z) es la luminosidad por unidad de masa de un SSP de edad t
y metalicidad Z, integrada a través del filtro de la banda. Se obtiene
interpolando la biblioteca SSP sobre una rejilla precalculada (age, Z).

Esto da dos vectores de peso por partícula:
    w_mass[i] = mass[i]
    w_lum[i]  = L_i
```

### Paso 3 — Rotación a la orientación de la vista

```
Para cada vista q con su matriz de rotación R(q):

    r_rot[i] = R(q) @ positions[i]

    u_i = r_rot[i, 0]      # coordenada en el plano del cielo (eje 1)
    v_i = r_rot[i, 1]      # coordenada en el plano del cielo (eje 2)
    l_i = r_rot[i, 2]      # coordenada de línea de visión

R(q) DEBE ser la misma rotación con la que MaNGIA generó el cubo de la
vista q (ver nota crítica en Suposiciones).
```

### Paso 4 — Binning y agregación

```
Mapear (u_i, v_i) a índices de spaxel usando spaxel_scale_kpc y el centro
de la rejilla (H, W).

Para cada spaxel s y clase c, agregar como promedio ponderado:

    Y_raw(s, c) = Σ_i w_i · P_i(c) · K_s(u_i, v_i)
                  ─────────────────────────────────
                  Σ_i w_i · K_s(u_i, v_i) + epsilon

donde K_s es el núcleo de binning:
  - binning="nearest": K_s = 1 si la partícula cae en el spaxel s, 0 si no.
  - binning="cic" (cloud-in-cell): la partícula reparte su peso entre los
    4 spaxels vecinos según la distancia bilineal. Recomendado: reduce el
    ruido de discretización.

Se ejecuta dos veces, con w = w_mass y con w = w_lum, produciendo
Y_mass_raw e Y_lum_raw. Por construcción Σ_c Y_raw(s,c) = 1 en spaxels
con peso suficiente.
```

### Paso 5 — Conteo efectivo de partículas (Kish)

```
Durante la agregación, calcular por spaxel el tamaño efectivo de muestra:

    N_eff(s) = [ Σ_i w_i · K_s(i) ]²  /  [ Σ_i w_i² · K_s(i)² + epsilon ]

y el conteo bruto:

    n_particles_map(s) = Σ_i K_s(i)

N_eff es un insumo directo del MaskBuilder (Criterio de muestreo). Se
calcula con la ponderación por masa.
```

### Paso 6 — Convolución por PSF

```
Para producir las variantes psf, convolucionar cada canal de clase del
tensor raw con un núcleo gaussiano:

    Y_psf(·, c) = K_PSF * Y_raw(·, c)

    sigma_psf = fwhm_psf_kpc / 2.355 / spaxel_scale_kpc   [en spaxels]

La convolución se aplica canal por canal. Tras convolucionar, renormalizar
para que Σ_c Y_psf(s,c) = 1 en cada spaxel.

Se obtienen así Y_mass_psf e Y_lum_psf.
```

### Paso 7 — Empaquetado

```
Guardar las 4 variantes, n_eff, n_particles_map y los metadatos de la
vista en el NPZ de salida.
```

### Pseudocódigo integrado

```
ENTRADA: datos por partícula, definición de las 4 vistas, r_cov
SALIDA:  4 tensores Y_int + n_eff por vista

1.  within_rcov ← (||positions|| <= r_cov)
2.  w_mass ← mass
3.  L ← mass * ell_b(age, metallicity)        # interpola SSP
4.  w_lum ← L
5.  PARA cada vista q:
6.      r_rot ← R(q) @ positions
7.      (u, v, l) ← componentes de r_rot
8.      idx ← mapear (u, v) a índices de spaxel
9.      Y_mass_raw ← agregar(p_class, w_mass, idx, within_rcov)
10.     Y_lum_raw  ← agregar(p_class, w_lum,  idx, within_rcov)
11.     n_eff ← kish(w_mass, idx, within_rcov)
12.     n_particles_map ← contar(idx, within_rcov)
13.     Y_mass_psf ← renormalizar(convolución_psf(Y_mass_raw))
14.     Y_lum_psf  ← renormalizar(convolución_psf(Y_lum_raw))
15.     guardar NPZ de la vista q
```

## Procesamiento por catálogo

```python
class CatalogProjectionBuilder:
    def build_catalog(
        self,
        entries: Iterable[LabelProjectionInput],
        output_dir: Path,
        n_workers: int = 8,
    ) -> CatalogProjectionReport:
        """
        Proyecta las pseudo-etiquetas de todo el catálogo.
        ~10 000 galaxias × 4 vistas. Cada vista ~5-20 s (binning + PSF).
        Estimación: ~1-3 h con 8 workers para todo MaNGIA.
        """
        ...
```

El manifest CSV registra una fila por (galaxia, vista):

```
galaxy_id,view_id,status,n_particles_within_rcov,fraction_clipped,inclination_deg,output_path
TNG50-87-141934-0-127,0,ok,118540,0.07,18.3,labels2d/TNG50-87-141934-0-127_v0.npz
...
```

## Edge cases

| Caso | Síntoma | Tratamiento |
|---|---|---|
| Spaxel sin partículas | División por cero en la agregación | `Y_raw(s,·)=0`; el MaskBuilder lo marcará inválido |
| Partícula exactamente en el borde de un spaxel | Ambigüedad de asignación | `cic` reparte el peso; con `nearest`, regla determinista de redondeo |
| Galaxia con casi toda la masa fuera de R_cov | `fraction_clipped` alta | Proyectar igual; reportar para revisión en QA |
| Orientación MaNGIA no disponible | No se puede construir `R(q)` | Fallar con error explícito; no inventar la orientación |
| PSF con FWHM cero o ausente | Variante psf indefinida | Usar `Y_psf = Y_raw`; marcar en metadatos |
| `p_class` no normalizada | Tensores de salida no suman 1 | Renormalizar a la entrada; emitir advertencia |
| Edad o metalicidad fuera de la rejilla SSP | Interpolación de `ell_b` falla | Recortar al borde de la rejilla SSP; registrar |

## Validación (tests)

### Tests unitarios

1. **Conservación de masa**: la suma de `w_mass` agregada sobre todos los
   spaxels debe igualar la masa total de partículas dentro de `R_cov`,
   dentro del 0.1 %.
2. **Normalización**: `Σ_c Y_raw(s,c) = 1` en todo spaxel con peso > 0.
3. **Normalización tras PSF**: `Σ_c Y_psf(s,c) = 1` tras renormalizar.
4. **Rotación identidad**: con `R(q) = I`, la proyección debe coincidir
   con el binning directo de las coordenadas `(x, y)`.
5. **Kish**: para un spaxel con `n` partículas de peso uniforme,
   `N_eff = n`; con pesos muy desiguales, `N_eff < n`.
6. **Cobertura**: partículas con `r_i > r_cov` no deben contribuir a
   ningún spaxel.
7. **Invarianza de clase**: rotar la galaxia no cambia las fracciones de
   clase globales (las `P_i(c)` no dependen de la orientación).

### Test de integración con el piloto

Ejecutar el módulo sobre el caso piloto (`TNG50-87-141934-0-127`):

- Producir las 4 vistas, cada una con sus 4 variantes de `Y_int`,
  `n_eff` y `n_particles_map`.
- Verificar shapes `(5, H, W)` consistentes con la rejilla de la vista.
- Verificar que las fracciones de clase globales son consistentes entre
  las 4 vistas dentro del 5 % (las `P_i(c)` son invariantes; solo varía
  el clipping de apertura).
- Verificar que el NPZ y el manifest se escriben correctamente.

## Criterios de aceptación

- [ ] El módulo produce las 4 variantes de `Y_int` para las 4 vistas del
      piloto.
- [ ] Tests unitarios pasan, en particular conservación de masa y
      normalización.
- [ ] `n_eff` se exporta y es consumible por el MaskBuilder.
- [ ] La proyección usa las orientaciones de MaNGIA, no orientaciones
      arbitrarias.
- [ ] El manifest del catálogo se genera con las columnas contratadas.
- [ ] Ningún tensor de salida contiene NaN; los spaxels sin partículas
      quedan en cero, no en NaN.
- [ ] El test de alineación contra el mapa de masa de pyPipe3D pasa
      (Spearman > 0.9, centroide < 1 spaxel) con la convención portada.

## Dependencias

```toml
# ya en pyproject.toml del pipeline de etiquetado
numpy
scipy            # convolución gaussiana, interpolación SSP
```

No se introducen dependencias nuevas. Conforme a la regla del proyecto,
**no** se usa PyTorch, TensorFlow ni JAX en el pipeline de etiquetado.

La biblioteca SSP (MILES) requerida para la ponderación por luminosidad es
la misma que usa el módulo `25_image_provider.md`; se comparte la rejilla
precalculada `(age, Z) → ell_b`.

## Relación con otros módulos

- **Entrada**: consume `p_class` del Classifier + BarDetector + ArmDetector,
  y posiciones/masa/edad/metalicidad del Extractor.
- **Salida hacia el MaskBuilder**: `n_eff` y `n_particles_map` son insumos
  directos del Criterio de muestreo del MaskBuilder (spec 22).
- **Salida hacia el Packer**: las 4 variantes de `Y_int` se empaquetan en
  el HDF5 final como el objetivo de entrenamiento del modelo.
- **Salida hacia la validación**: el spec `40_pseudolabel_validation.md`
  consume `Y_int` (variante fijada por `config.label_variant`) y los
  metadatos de vista que este módulo produce.

## Roadmap de implementación

1. **Fase 1 (1 día)**: rotación, binning `nearest` y agregación ponderada
   por masa; tests de conservación y normalización sobre el piloto.
2. **Fase 2 (medio día)**: binning `cic` y cómputo de `N_eff`.
3. **Fase 3 (medio día)**: ponderación por luminosidad (integración SSP) y
   variante `lum`.
4. **Fase 4 (medio día)**: convolución por PSF y renormalización; variantes
   `psf`.
5. **Fase 5 (medio día)**: orquestación de catálogo, manifest, NPZ.

Total estimado: **3 días** de desarrollo, más el tiempo de cómputo de
ejecución sobre el catálogo (~1-3 h).

## Convención de orientación (RESUELTA — antes "dependencia abierta")

La convención que MaNGIA usa por vista ya está implementada y validada
en el pipeline v1 (`labeling/geometry.py` + `labeling/constants.py`),
con la que se generaron los resultados del artículo. **Portar ese código
tal cual** (ver `MIGRATION.md`); no reimplementar la geometría.

La vista se define por un **vector de línea de visión** (no por matriz
de rotación ni ángulos), construido así:

1. `view_vector_from_index(view, repeat_count)`:
   - Si `repeat_count ≤ 3`: vistas axiales `AXIS_VIEWS = ((1,0,0),
     (0,1,0), (0,0,1))`, indexadas por `view` ∈ {0,1,2}, en el marco de
     coordenadas de la simulación (NO en el marco face-on).
   - Si `repeat_count > 3`: conjunto isotrópico de 6 vistas derivado de
     los vértices de un icosaedro rotado para que un vértice apunte a
     +x, filtrado al hemisferio x ≥ 0 y ordenado por (−x, −y, −z)
     (`icosahedron_positive_x_views()`).
2. `project_positions(positions, view_vector)` construye la base
   ortonormal del plano del cielo: `los = v̂`,
   `x_axis = normalize(ẑ × los)` (fallback ŷ si los ∥ ẑ),
   `y_axis = normalize(los × x_axis)`; las coordenadas proyectadas son
   los productos punto contra esa base.

`ViewDefinition` por lo tanto reemplaza `rotation_matrix` por:

```python
class ViewDefinition(BaseModel):
    view_vector: tuple[float, float, float]   # línea de visión, marco simulación
    grid_shape: tuple[int, int]
    spaxel_scale_arcsec: float
    kpc_per_arcsec: float
    fwhm_psf_arcsec: float
```

**Importante**: las posiciones que se proyectan son las del marco de la
simulación centradas en el subhalo (`pos - subhalo_pos`), no las del
marco face-on. La rotación face-on (alineación con el momento angular)
solo se usa en Fase A para las features del Classifier y para los
detectores de barra/brazos.

**Test de alineación obligatorio** (sustituye a la confirmación con el
equipo de MaNGIA): para el piloto, proyectar la masa estelar con esta
convención y verificar contra el mapa de masa de pyPipe3D del cubo de la
misma vista — correlación espacial de Spearman > 0.9 y desplazamiento
del centroide < 1 spaxel. Si este test falla, la convención portada no
corresponde a esa muestra y SÍ hay que escalar al equipo de MaNGIA.
