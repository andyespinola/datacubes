# Spec: QualityCheck

> Módulo: `phase_c/quality_check.py` · Hito: 4 · Depende de: todos los módulos previos

## Responsabilidad

Producir un **reporte estructurado de calidad por galaxia × orientación** que permita:
- detectar galaxias problemáticas para excluirlas del entrenamiento
- generar estadísticas agregadas sobre toda la muestra
- diagnosticar regresiones cuando se modifique el pipeline

## Contrato de entrada

```python
class QualityCheckInput(BaseModel):
    galaxy_id: str
    view_id: int
    features_path: Path
    final_labels_path: Path
    aggregated_path: Path
    valid_mask_path: Path
    catalog_morpho: dict           # del catálogo TNG
    config: QualityCheckConfig

class QualityCheckConfig(BaseModel):
    fraction_tolerance: float = 0.10        # ±10% vs catálogo
    min_valid_fraction: float = 0.30        # M_valid debe cubrir ≥30% de spaxels
    max_uncertainty_p95: float = 0.5        # p95(1 - max(P_class)) ≤ 0.5
```

## Contrato de salida

```python
class QualityReport(BaseModel):
    """Persisted as JSON: qa_report_{galaxy_id}_{view_id}.json"""
    galaxy_id: str
    view_id: int
    timestamp: str

    # Status global
    status: Literal["pass", "warning", "fail"]
    flags: list[str]               # razones si warning/fail

    # Conservación
    mass_conservation_error: float          # |M_aggregated - M_features| / M_features
    light_conservation_error: float

    # Comparación con catálogo
    fractions_recovered: dict      # {bulge, disk, bar, arm, halo}
    fractions_catalog: dict
    fraction_deviations: dict

    # Calidad espacial
    n_spaxels_valid: int
    fraction_spaxels_valid: float
    largest_connected_component_fraction: float

    # Calidad probabilística
    mean_max_probability: float
    p95_uncertainty: float

    # Métricas físicas
    bar_detected: bool
    bar_a2: Optional[float]
    n_arm_crests: int

    # Tiempos por módulo
    time_per_module_sec: dict
```

## Algoritmo

```
1. Cargar todos los productos intermedios
2. Verificar conservación de masa:
   M_features = features.mass.sum()
   M_aggregated = aggregated.raw_mass_per_class.sum()
   mass_conservation_error = |M_aggregated - M_features| / M_features
3. Calcular fracciones recuperadas:
   fractions_recovered = {
       c: (raw_mass_per_class[:,:,c] * M_valid).sum() / total_mass
       for c in classes
   }
4. Comparar con fracciones del catálogo:
   for c in catalog:
       deviation = |recovered[c] - catalog[c]| / catalog[c]
5. Verificar M_valid:
   - n_valid >= 0.30 × 74×74? sino → flag "low_validity"
6. Verificar componente conexa:
   - largest CC / total valid >= 0.95? sino → flag "fragmented_mask"
7. Verificar incertidumbre:
   - p95(1 - max(P_class)) <= 0.5? sino → flag "high_uncertainty"
8. Status:
   - fail: mass_conservation > 0.05 OR cualquier criterio crítico
   - warning: cualquier flag no crítico
   - pass: ningún flag
9. Guardar JSON
```

## Criterios de aceptación

- [ ] Reporte JSON válido y bien-formed
- [ ] Tiempo < 1s
- [ ] Status correcto en casos de prueba conocidos

---

# Spec: Packer

> Módulo: `phase_c/packer.py` · Hito: 4 · Depende de: QualityCheck (opcional)

## Responsabilidad

Empaquetar todos los productos finales en un **único archivo HDF5 por galaxia × orientación**, listo para ser consumido por el dataloader de ApertureNet-S3 durante el entrenamiento.

## Contrato de entrada

```python
class PackerInput(BaseModel):
    galaxy_id: str
    view_id: int
    aggregated_path: Path
    valid_mask_path: Path
    cube_ifu_path: Path
    pipe3d_maps_paths: dict        # {v_star, sigma_star, age, metallicity, mass, av}
    qa_report_path: Optional[Path]
    config: PackerConfig

class PackerConfig(BaseModel):
    include_qa: bool = True
    include_pipe3d: bool = True
    compression: str = "lzf"
    cube_dtype: str = "float32"
    label_dtype: str = "float32"
```

## Contrato de salida

```python
class DatasetEntry:
    """
    Final HDF5 file: dataset_entry_{galaxy_id}_{view_id}.h5

    Structure:
        /metadata
            galaxy_id (str)
            view_id (int)
            snapshot, subhalo_id (int)
            orientation (theta, phi, psi, distance) (group)
            timestamp (str)
            pipeline_version (str)

        /inputs
            cube_ifu (6603, 74, 74) float32
            pipe3d_maps/
                v_star      (74, 74) float32
                sigma_star  (74, 74) float32
                age         (74, 74) float32
                metallicity (74, 74) float32
                mass        (74, 74) float32
                av          (74, 74) float32

        /labels
            Y_int_mass  (74, 74, 5) float32   # primary target (multi-task)
            Y_int_light (74, 74, 5) float32   # secondary target (multi-task)
            class_names ["bulge", "disk", "bar", "arm", "halo"] (str array)

        /masks
            M_valid     (74, 74) bool

        /qa
            status (str)
            mass_conservation_error (float)
            ... (resto del QA report)
    """
```

## Algoritmo

```
1. Leer metadata de todos los productos intermedios
2. Cargar cubo IFU y verificar shape (6603, 74, 74) o adaptar si 69×69 vía padding
3. Cargar mapas pyPipe3D si include_pipe3d=True
4. Cargar Y_int_mass, Y_int_light, M_valid
5. Cargar QA report si include_qa=True
6. Crear archivo HDF5 con estructura definida
7. Escribir cada grupo con compresión configurada
8. Añadir attrs con metadata
9. Verificar lectura: abrir el archivo y validar estructura
```

## Notas

- **Padding 69→74**: si el cubo viene en 69×69 (caso piloto actual), aplicar padding centrado a 74×74. Esto es decisión metodológica recomendada por el director (sección 8 de `NOTA_DIRECTOR_MANGIA_MANGA_SEGMENTACION.md`).
- **Tipos de datos**: float32 es suficiente para todo. float64 desperdiciaría espacio (los archivos son grandes).
- **Compresión lzf**: rápida y razonable para arrays float32. gzip es más compacto pero más lento; preferir lzf por velocidad.

## Criterios de aceptación

- [ ] HDF5 válido (h5py puede abrirlo sin error)
- [ ] Contiene todos los grupos requeridos
- [ ] Tamaño razonable: < 100 MB por entrada
- [ ] Tiempo < 5s
- [ ] Test de roundtrip: leer y verificar que coincide con los inputs
