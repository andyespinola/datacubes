# Spec: Classifier (v2.1 — alineado al artículo IBERAMIA)

> Módulo: `phase_a/classifier.py` · Hito: 2 · Depende de: Extractor

## Cambios respecto a v2.0 de este spec

| # | Cambio | Motivo |
|---|--------|--------|
| 1 | Features: 3D → **4D del artículo** (ε, log₁₀(R/R_eff+δ), \|z\|/R_eff, E_norm) | El artículo validó este vector en 500 galaxias (Test A 93.3%, Test B 99.0%). El spec anterior no coincidía con la metodología publicada. |
| 2 | Feature-set alternativo `standard3d` (ε, j_p/j_c, e/\|e\|_max) configurable | Espacio estándar de la literatura (Doménech-Moral 2012; Obreja 2018; Du 2019/2020; Zana 2022; Proctor 2024). Permite comparación 1:1 con catálogos públicos sobre TNG50. |
| 3 | Reordenamiento de componentes **corregido** | El criterio anterior (ordenar las 3 por ε) confunde bulbo y halo. Regla nueva: disco = ε medio más alto; entre las restantes, bulbo = la más ligada (E_norm), con R/R_eff como criterio secundario. |
| 4 | Prior y validación con **catálogo MORDOR** (Zana et al. 2022, público, TNG50) | Fracciones por componente más informativas que las 3 fracciones globales de Rodriguez-Gomez 2022. |
| 5 | `agreement` con umbrales duros pasa de **assert a métrica reportada** | El artículo demostró que la discrepancia con el ε-threshold es la virtud del GMM (92.9% vs 40.9% de éxito cinemático). Forzar acuerdo >80% empuja al modelo hacia el baseline malo. |
| 6 | `bulge_radial_max_kpc` absoluto → `bulge_radial_max_reff` relativo | Un radio fijo en kpc sesga la inicialización según el tamaño de la galaxia. |

## Responsabilidad

Asignar a cada partícula estelar una **distribución de probabilidad** sobre las clases primarias `{bulge, disk, halo}`. **No identifica barra ni brazos** (specs 12–13 las extraen del disco después).

## Contrato de entrada

```python
class ClassifierInput(BaseModel):
    features_path: Path                          # particle_features.h5
    catalog_priors: Optional[CatalogPriors]      # MORDOR o Rodriguez-Gomez
    config: ClassifierConfig

class CatalogPriors(BaseModel):
    source: Literal["mordor", "rodriguez_gomez", "none"]
    bulge_frac: float        # MORDOR: bulge + pseudo-bulge
    disk_frac: float         # MORDOR: thin + thick disc
    other_frac: float        # MORDOR: halo
    confidence: float = 0.5  # peso del prior α ∈ [0,1]; 0.5 por defecto, NO 1.0

class ClassifierConfig(BaseModel):
    method: Literal["gmm", "hard_thresholds"] = "gmm"
    n_components: int = 3                        # bulge, disk, halo
    feature_set: Literal["paper4d", "standard3d"] = "paper4d"
    log_radius_delta: float = 0.05               # δ en log10(R/R_eff + δ)
    epsilon_init_thresholds: dict = {
        "disk_min": 0.5,
        "bulge_max": 0.3,
        "halo_max": -0.3,
    }
    bulge_radial_max_reff: float = 1.0           # bulbo inicial solo si R < esto × R_eff
    seed: int = 42
```

## Definición de los feature-sets

### `paper4d` (default — el vector validado del artículo)

```python
delta = config.log_radius_delta
X = np.stack([
    features.epsilon,                                  # ε = j_z / j_c(E)
    np.log10(features.R / features.R_eff_kpc + delta), # estructura radial (log)
    np.abs(features.z) / features.R_eff_kpc,           # estructura vertical
    features.E / np.abs(features.E).max(),             # E_norm ∈ [-1, 0)
], axis=1)
```

Notas:
- `epsilon` es la circularidad canónica ε = j_z/j_c(E) que produce el
  Extractor (spec 10), NO el proxy v_φ/v_total del pipeline v1.
- `E_norm` usa la energía específica total (cinética + potencial) del
  Extractor, normalizada por |E|_max (la partícula más ligada). Coincide
  con e/|e|_max de la literatura.

### `standard3d` (alternativo — espacio estándar de la literatura)

```python
j_p = np.sqrt(np.maximum(features.j_total**2 - features.j_z**2, 0.0))
X = np.stack([
    features.epsilon,                       # j_z / j_c(E)
    j_p / features.j_c,                     # momento angular no-azimutal
    features.E / np.abs(features.E).max(),  # e / |e|_max
], axis=1)
```

Requiere que el Extractor exporte también `j_total` (norma de **j** por
partícula); añadirlo al contrato del spec 10 si no está.

**Uso previsto de `standard3d`**: corridas de comparación contra los
catálogos públicos auto-GMM (Du et al. 2020) y MORDOR (Zana et al. 2022)
sobre TNG50, y estudio de ablación (¿qué aporta j_p/j_c que no aporten
R, z?). El producto final del catálogo usa `paper4d` salvo que la
ablación demuestre lo contrario.

## Algoritmo

### Paso 1 — Construir X según `feature_set` y estandarizar

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```

### Paso 2 — Inicialización por umbrales físicos

```python
init_labels = np.full(N, -1)
init_labels[features.epsilon > config.epsilon_init_thresholds["disk_min"]] = 1   # disco
init_labels[
    (features.epsilon < config.epsilon_init_thresholds["bulge_max"]) &
    (features.R < config.bulge_radial_max_reff * features.R_eff_kpc)
] = 0   # bulbo (central, relativo a R_eff)
init_labels[features.epsilon < config.epsilon_init_thresholds["halo_max"]] = 2  # halo

unassigned = init_labels == -1
init_labels[unassigned] = nearest_class_by_distance(
    X_scaled[unassigned], X_scaled[~unassigned], init_labels[~unassigned]
)
```

### Paso 3 — Medias iniciales

```python
means_init = np.array([X_scaled[init_labels == k].mean(axis=0) for k in range(3)])
```

### Paso 4 — Pesos iniciales con prior del catálogo

Fuente preferida: **MORDOR** (Zana et al. 2022), catálogo público sobre
TNG50 con fracciones de masa por subhalo para thin disc, thick disc,
bulge, pseudo-bulge y halo. Mapeo al esquema K=3:

```
bulge_frac = bulge + pseudo_bulge
disk_frac  = thin_disc + thick_disc
other_frac = halo
```

Fallback: Rodriguez-Gomez et al. 2022 (bulge/disk/other globales).

```python
if catalog_priors is not None and catalog_priors.source != "none":
    α = catalog_priors.confidence          # default 0.5, nunca 1.0
    weights_data = np.bincount(init_labels, minlength=3) / N
    weights_init = α * np.array([catalog_priors.bulge_frac,
                                  catalog_priors.disk_frac,
                                  catalog_priors.other_frac]) \
                 + (1 - α) * weights_data
    weights_init /= weights_init.sum()
else:
    weights_init = np.bincount(init_labels, minlength=3) / N
```

El prior **solo inicializa**; el EM es libre de alejarse. No existe
ningún paso de reescalado posterior a las fracciones del catálogo
(principio P3: prior, no constraint).

### Paso 5 — Ajustar GMM

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    weights_init=weights_init,
    means_init=means_init,
    max_iter=200,
    tol=1e-4,
    reg_covar=1e-6,
    random_state=config.seed,
)
gmm.fit(X_scaled)
P_class = gmm.predict_proba(X_scaled)   # (N, 3) en orden arbitrario del GMM
```

### Paso 6 — Reordenamiento de componentes (CORREGIDO)

El GMM no garantiza el orden. La regla v2.0 (ordenar las tres por ε)
es **incorrecta**: bulbo y halo tienen ambos ε medio bajo y se
confunden. Regla v2.1, consistente con el criterio energético de
MORDOR (el bulbo es la población más ligada):

```python
# Medias en el espacio ORIGINAL (des-estandarizar):
means_orig = scaler.inverse_transform(gmm.means_)   # (3, n_features)

eps_col = 0                       # columna de ε en ambos feature-sets
e_col   = X.shape[1] - 1          # E_norm es la última columna
r_col   = 1                       # log-radio (paper4d) o j_p/j_c (standard3d)

# 1. Disco = componente con ε medio MÁS ALTO
disk_k = int(np.argmax(means_orig[:, eps_col]))

# 2. Entre las dos restantes: bulbo = la MÁS LIGADA (E_norm más negativa).
rest = [k for k in range(3) if k != disk_k]
if config.feature_set == "paper4d":
    # Criterio primario: energía. Si |ΔE_norm| < 0.05 (degenerado),
    # criterio secundario: bulbo = menor log10(R/R_eff + δ).
    e0, e1 = means_orig[rest[0], e_col], means_orig[rest[1], e_col]
    if abs(e0 - e1) >= 0.05:
        bulge_k = rest[0] if e0 < e1 else rest[1]
    else:
        bulge_k = rest[int(np.argmin([means_orig[rest[0], r_col],
                                      means_orig[rest[1], r_col]]))]
else:  # standard3d
    bulge_k = rest[int(np.argmin([means_orig[rest[0], e_col],
                                  means_orig[rest[1], e_col]]))]
halo_k = [k for k in rest if k != bulge_k][0]

P_class = P_class[:, [bulge_k, disk_k, halo_k]]   # → [P_bulge, P_disk, P_halo]
```

Prohibido: cualquier variante que distinga bulbo de halo solo por ε.

### Paso 7 — Métricas de calidad

```python
quality = {
    "method_used": config.method,
    "feature_set": config.feature_set,
    "bic": gmm.bic(X_scaled),
    "aic": gmm.aic(X_scaled),
    "n_iter": gmm.n_iter_,
    "converged": gmm.converged_,
    "fractions_recovered": {...},          # ponderadas por MASA, no por conteo
    "fractions_catalog": {...},            # del prior usado (mordor / rg2022)
    "delta_fractions_vs_catalog": {...},   # recuperada - catálogo, por clase
    "agreement_with_hard_thresholds": float,   # MÉTRICA REPORTADA, sin assert
    "max_uncertainty_per_particle": {"mean": ..., "p95": ...},
    "component_means_original_space": means_orig.tolist(),
    "reorder_rule_branch": "energy" | "radius_tiebreak",
}
```

Notas:
- `fractions_recovered` se pondera por masa estelar:
  `Σᵢ mᵢ·P_class[i,c] / Σᵢ mᵢ` — las fracciones de los catálogos son de
  masa, comparar conteos contra masas es un error.
- `silhouette_score` es opcional y, si se computa, hacerlo sobre una
  submuestra ≤ 50k partículas (es O(N²)).

### Fallback si el GMM no converge

```python
if not gmm.converged_:
    log.warning("GMM no convergió; fallback a umbrales duros")
    P_class = hard_threshold_classification(features)   # baseline del artículo
    quality["method_used"] = "hard_thresholds_fallback"
```

El fallback usa la ε canónica (j_z/j_c), no el proxy v_φ/v_total del v1.

## Validación

### Tests unitarios (sintéticos)

1. Galaxia disco-puro: `P_class[:,1] > 0.9` para >90% de partículas.
2. Galaxia bulbo-puro: análogo, columna 0.
3. Mixta 50/50: fracciones recuperadas dentro del 10%.
4. Suma a 1: `P_class.sum(axis=1) ≈ 1`.
5. **Reordenamiento**: construir un caso con bulbo ligado y halo difuso
   ambos con ε≈0; verificar que el remap NO los intercambia (este test
   habría detectado el bug de v2.0).
6. Determinismo: misma seed → mismas P_class bit a bit.

### Test de integración con el piloto

- `quality.converged == True`
- `|delta_fractions_vs_catalog| ≤ 0.10` por clase contra **MORDOR**
  (subhalo 141934, snapshot 87). Si MORDOR no cubre el subhalo, usar
  Rodriguez-Gomez 2022 y registrarlo.
- Histograma de P_disk bimodal (visual, notebook).
- `agreement_with_hard_thresholds` se REPORTA. Valores de referencia del
  artículo: el GMM-4D discrepa del ε-threshold precisamente en las
  regiones donde el umbral falla cinemáticamente. Un acuerdo muy alto
  (>0.95) es sospechoso (el GMM colapsó al umbral); uno muy bajo (<0.4)
  amerita inspección visual. Ninguno de los dos es fallo automático.

### Validación poblacional (Hito 5, no bloquea Hito 2)

Sobre la muestra con cobertura MORDOR:
- Correlación por galaxia entre `disk_frac` recuperada y
  `thin+thick` de MORDOR: Spearman ρ > 0.7 esperado.
- Los umbrales de aceptación cinemática vienen del artículo:
  Test A ≥ 90% de la muestra, Test B ≥ 95% (valores logrados: 93.3% y
  99.0%). Estos se evalúan en el spec de QC, no aquí.

## Criterios de aceptación

- [ ] Tests unitarios 1–6 pasan (incluido el test de reordenamiento)
- [ ] Integración con piloto pasa (fracciones ±10% vs MORDOR/RG2022)
- [ ] `feature_set` conmuta entre `paper4d` y `standard3d` sin cambios de código
- [ ] No existe ningún reescalado a fracciones del catálogo en el módulo
- [ ] Tiempo < 30 s por galaxia (paper4d, ~10⁵–10⁶ partículas)
- [ ] Output HDF5 valida contra schema; `mypy` y `ruff` limpios

## Decisiones de diseño documentadas

**¿Por qué el 4D del artículo como default y no el 3D estándar?** El 4D
ya está validado cinemáticamente en 500 galaxias (Tests A/B del
artículo) y mantiene coherencia paper↔tesis. Las features espaciales
(R, |z|) además inducen coherencia espacial en los mapas proyectados,
deseable para etiquetas de segmentación. El 3D estándar se conserva
como modo de comparación/ablación porque es lo que usan los catálogos
externos.

**¿Por qué MORDOR como prior y validación?** Es público, cubre TNG50
z=0 completo, da fracciones por componente más granulares que
Rodriguez-Gomez, y su criterio energético para el bulbo es el mismo que
adopta la regla de reordenamiento. Además su flag de barra, A₂ y R_bar
sirven al spec 12 (BarDetector) como validación por subhalo.

**¿Por qué no edad/metalicidad en el GMM?** Ningún método estándar de
descomposición dinámica las usa; introducen sesgos por historias de
formación específicas. Sin cambios respecto a v2.0.

## Referencias

- Abadi et al. 2003, ApJ 597, 21 (circularidad ε)
- Doménech-Moral et al. 2012, MNRAS 421, 2510 (espacio ε, j_p/j_c, e/|e|_max)
- Obreja et al. 2018, MNRAS 477, 4915 (gsf, GMM)
- Du et al. 2019, ApJ 884, 129; Du et al. 2020, ApJ 895, 139 (auto-GMM en TNG)
- Zana et al. 2022, MNRAS 515, 1524 (MORDOR, catálogo TNG50, DOI 10.1093/mnras/stac1708)
- Proctor et al. 2024, MNRAS 527, 2624 (GMM en EAGLE, validación vs κ_co)
- Rodriguez-Gomez et al. 2022 (fracciones globales, fallback)
