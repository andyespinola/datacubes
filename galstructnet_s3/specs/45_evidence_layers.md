# 45 — Capas de Propagación de Evidencia (EPN)

> Módulo: `models/layers/normconv.py`, `models/layers/evidence.py`,
> `models/fusion_precision.py` · Hito: 3–4 (después del skeleton estándar del
> Hito 2, para que el baseline de ablación exista) · Depende de: dataset v3
> (10, con `cube_ivar`, `pipe3d_err`, `N_eff`), revisión de arquitectura
> 2026-06-12 (§2.1–2.4, §3).

## Responsabilidad

Define la familia de capas que convierte a GalStructNet-S3 en una **Red de
Propagación de Evidencia**: todo tensor del modelo es un par `(señal, certeza)`
y el camino de salida opera sobre **campos de evidencia** (pseudo-conteos no
negativos). La certeza de entrada proviene de precisiones instrumentales
medidas (IVAR del cubo, mapas de error pyPipe3D, `M_valid`); la concentración
Dirichlet de salida emerge de la certeza propagada y se supervisa contra la
precisión medida de la etiqueta (`κ·N_eff`).

Invariante de identidad del sistema: **entrada, salida y etiqueta comparten
unidad (pseudo-conteos / precisión medida)**. Cada capa de esta familia
declara una reducción al caso estándar y un invariante testeable.

Cuatro primitivas:

| Primitiva | Archivo | Sustituye a / se inserta en |
|---|---|---|
| P1 `NormConv2d` | `layers/normconv.py` | Convs de `22_encoder_physical`, `30_decoder_unet` |
| P2 `EvidenceConv2d` | `layers/evidence.py` | Generaliza `43_module_psf` (kernel fijo = caso particular) |
| P3 `PrecisionGatedFusion` | `fusion_precision.py` | Reemplaza `23_fusion` (la global queda como ablación) |
| P4 `EvidenceHead` | `heads/segmentation.py` (mod.) | Modifica `40_head_segmentation` |

Fuera de scope de este spec: el encoder Mamba permanece estándar (la certeza
espectral entra colapsada a S/N por spaxel vía P3). La extensión
*noise-aware selective scan* es el Anexo A, experimental y fuera del camino
crítico.

## Convención (señal, certeza) y mapeo desde datos

Toda activación espacial es `(x, c)` con `x: (B, C, H, W)` y `c: (B, C, H, W)`,
`c ∈ [0, 1]`. Semántica operacional: `c` es un **peso de confianza** con dos
garantías duras — `c = 0` ⟺ el valor se ignora exactamente; `c` decrece
monótonamente con el error instrumental. **No** se afirma que `c` sea una
varianza posterior exacta (esa afirmación vive solo en la variante Nivel B de
propagación de momentos del encoder físico; ver revisión §2.2).

Mapeo en `data/dataset.py` (sustituye a `nan_to_num`):

```python
# Por canal: sigma_ref = mediana del error en train (precalculada en norm_stats)
c = sigma_ref**2 / (sigma**2 + sigma_ref**2)        # ∈ (0, 1]; c(sigma_ref)=0.5
# Equivalente con IVAR: c = ivar / (ivar + ivar_ref)
c[~M_valid] = 0.0
c[torch.isnan(x)] = 0.0; x = torch.nan_to_num(x, 0.0)   # el valor da igual: c=0
```

- `maps` → `c_phys (6,H,W)` desde `pipe3d_err` (+2 canales si h3/h4 entran).
- `image` → `c_spat (3,H,W)` desde el mapa de ruido fotométrico (o 1.0 si falta).
- `cube` → escalar por spaxel `c̄_spec (1,H,W)` = mediana_λ de `ivar·σ_ref²`
  en la ventana 5000–5500 Å (el per-λ completo solo lo usa el Anexo A).

## P1 — `NormConv2d` (convolución normalizada por certeza)

### Contrato

```python
forward(x: (B,Cin,H,W), c: (B,Cin,H,W)) -> (y: (B,Cout,H,W), c_out: (B,Cout,H,W))
```

### Algoritmo

```python
class NormConv2d(nn.Module):
    """Agregación espacial ponderada por certeza + mezcla puntual con signo.

    La aplicabilidad A es no negativa (softplus): la agregación espacial es
    un promedio convexo ponderado por c. La expresividad con signo se
    restituye en la mezcla puntual 1×1 posterior.
    """
    def __init__(self, c_in, c_out, k=3, eps=1e-4):
        super().__init__()
        self.theta_A = nn.Parameter(torch.randn(c_out, c_in, k, k) * 0.1)
        self.mix     = nn.Conv2d(c_out, c_out, 1)          # con signo
        self.norm    = nn.GroupNorm(8, c_out)
        self.act     = nn.GELU()
        self.eps     = eps
        self.pad     = k // 2

    def forward(self, x, c):
        A   = F.softplus(self.theta_A)                      # ≥ 0
        num = F.conv2d(x * c, A, padding=self.pad)
        den = F.conv2d(c,     A, padding=self.pad)
        z   = num / (den + self.eps)                        # señal normalizada
        ones  = torch.ones_like(c)
        c_out = F.conv2d(c, A, padding=self.pad) / \
                (F.conv2d(ones, A, padding=self.pad) + self.eps)
        y = self.act(self.norm(self.mix(z)))
        return y, c_out                                     # puntuales no alteran c (aprox.)
```

### Propiedades (cada una es un test)

1. **Ignorancia exacta**: la salida es invariante al valor de `x` donde `c=0`.
2. **Reducción**: con `c ≡ 1`, `z` = convolución con el kernel no negativo
   normalizado (suavizado estándar); el par (z, mix 1×1) recupera la
   expresividad de una conv estándar de soporte k.
3. **Certeza convexa y monótona**: `0 ≤ c_out ≤ max_vecindad(c)`;
   `c_out = 0` ⟺ todo el soporte tiene `c = 0`. Con zero-padding, los bordes
   reciben `c_out` menor automáticamente (físicamente correcto: el cubo
   termina ahí).
4. **Equivariancia D4 exacta** (es una convolución): rota la entrada ⇒ rota
   la salida. (A diferencia del Swin, aquí el test sí es válido.)

## P2 — `EvidenceConv2d` (redistribución de pseudo-conteos)

### Contrato

```python
forward(e: (B,K,H,W) ≥ 0) -> e': (B,K,H,W) ≥ 0
# Variante D (default, depthwise): conserva Σ_espacial e por clase.
# Variante M (mixing): kernel (K,K,k,k) ≥ 0 con columnas normalizadas;
#                      conserva solo la evidencia total. Solo para ablación.
```

### Algoritmo (variante D)

```python
class EvidenceConv2d(nn.Module):
    """Kernel estocástico por clase: K⁺ ≥ 0, Σ_espacial K⁺ = 1.
    La convolución es entonces redistribución de conteos (masa conservada
    en el interior). Con kernel fijo = PSF, reproduce el módulo PSF en
    espacio de evidencia (revisión §2.4)."""
    def __init__(self, n_ch, k, fixed_kernel: Tensor | None = None):
        super().__init__()
        if fixed_kernel is not None:                        # modo PSF
            self.register_buffer('theta', fixed_kernel.log())  # ya ≥0, suma 1
            self.fixed = True
        else:
            self.theta = nn.Parameter(torch.zeros(n_ch, 1, k, k))
            self.fixed = False
        self.pad = k // 2

    def kernel(self):
        Kp = self.theta.exp() if self.fixed else F.softplus(self.theta)
        return Kp / Kp.sum(dim=(-2, -1), keepdim=True)      # estocástico

    def forward(self, e):
        Kp = self.kernel().to(e.dtype)
        n_ch = e.shape[1]
        return F.conv2d(e, Kp.expand(n_ch, 1, *Kp.shape[-2:]) if self.fixed
                        else Kp, padding=self.pad, groups=n_ch)
```

Aplicación al módulo PSF (sustituye la conv-sobre-prob + renormalización):

```python
alpha_obs = EvidenceConv2d(K, k, fixed_kernel=psf)(alpha_int - 1.0) + 1.0
prob_obs  = alpha_obs / alpha_obs.sum(dim=1, keepdim=True)   # suma 1 automática
```

(El kernel PSF es por muestra: usar el truco grouped-conv de `43` con
`groups=B·K`; el código de arriba muestra la semántica, no el batching.)

### Propiedades

5. **Conservación de evidencia (interior)**: delta de evidencia colocada
   lejos del borde ⇒ `Σ_s e'(s,c) = Σ_s e(s,c)` con `atol=1e-5` (float32).
   En bordes con zero-padding la masa que sale del FOV se pierde —
   documentado, no es bug. Test sobre región interior.
6. **Identidad**: `K⁺ = δ` ⇒ `e' = e`.
7. **Equivalencia PSF**: con `fixed_kernel = K_PSF`, la salida coincide con
   la α-convolución de la revisión §2.4 (test de regresión contra
   implementación de referencia con numpy).

## P3 — `PrecisionGatedFusion` (fusión por posición modulada por precisión)

### Contrato

```python
forward(F_spat (B,256,H,W), F_spec (B,256,H,W), F_phys (B,64,H,W),
        cbar: dict[str, (B,1,H,W)])            # certezas escalares por modalidad
  -> (F_fused (B,384,H,W), c_fused (B,1,H,W))
```

### Algoritmo

Por posición `(i,j)`: M=3 tokens (una entrada por modalidad), atención sobre
M tokens — no sobre los 4 761 spaxels. Costo O(H·W·M²) ≈ trivial.

```python
class PrecisionGatedFusion(nn.Module):
    def __init__(self, d_spat=256, d_spec=256, d_phys=64, d=384, n_heads=4):
        super().__init__()
        self.proj = nn.ModuleDict({
            'spat': nn.Conv2d(d_spat, d, 1),
            'spec': nn.Conv2d(d_spec, d, 1),
            'phys': nn.Conv2d(d_phys, d, 1)})
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.gate = nn.ModuleDict({m: make_mlp(1, 16, 1) for m in
                                   ('spat', 'spec', 'phys')})   # g_m, init→0
        self.out  = nn.Conv2d(d, d, 1)
        self.norm = nn.GroupNorm(8, d)

    def forward(self, F_spat, F_spec, F_phys, cbar):
        toks = {m: self.proj[m](f) for m, f in
                zip(('spat','spec','phys'), (F_spat, F_spec, F_phys))}
        # (B, d, H, W) -> (B·H·W, M, d): cada posición es una "frase" de M tokens
        seq  = torch.stack([flatten_pos(toks[m]) for m in MODS], dim=1)
        bias = torch.stack([self.gate[m](log_clamped(cbar[m])).flatten()
                            for m in MODS], dim=1)          # (B·H·W, M)
        # query = token espacial; sesgo aditivo de precisión en los logits
        q = seq[:, 0:1]
        attn_mask = -bias.unsqueeze(1)        # se suma a los logits (M keys)
        fused, w = self.attn(q, seq, seq, attn_mask=attn_mask,
                             need_weights=True, average_attn_weights=True)
        F_f = unflatten_pos(fused.squeeze(1))               # (B, d, H, W)
        c_f = (w.squeeze(1) * torch.stack(
                   [flatten_pos(cbar[m]).squeeze(-1) for m in MODS], dim=1)
              ).sum(dim=1)                                   # certeza fusionada
        F_f = self.norm(self.out(F_f) + self.proj['spat'](F_spat))
        return F_f, unflatten_pos_scalar(c_f)
```

- `g_m` se inicializa a salida ≈ 0: al inicio del entrenamiento el gating no
  perturba (equivale a atención de modalidades sin precisión); el sesgo se
  aprende.
- Los pesos de atención `w` por spaxel son un **producto científico**: el mapa
  "en qué modalidad se apoyó el modelo" se exporta en evaluación.
- La cross-attention global de `23_fusion.md` se conserva en el repo como
  `CrossAttentionFusion` solo para la ablación A2.

### Propiedades

8. **Degradación selectiva**: poner `c̄_spec → 0` en una región debe reducir
   monótonamente el peso de atención de `spec` allí (test estadístico sobre
   `w`, no igualdad exacta).
9. **Neutralidad inicial**: con `g_m ≡ 0` (init), la salida coincide con la
   fusión sin precisión (test de regresión al construir).

## P4 — `EvidenceHead` (emergencia de la Dirichlet)

### Contrato

```python
forward(hidden (B,256,H,W), c_dec (B,1,H,W)) ->
  {alpha (B,K,H,W) ≥ 1, prob, evidence, vacuity}
```

### Algoritmo

```python
class EvidenceHead(nn.Module):
    """e = e_net ⊙ h(c̄): el techo de evidencia escala con la certeza
    propagada; la red puede emitir menos evidencia que el techo (regiones de
    feature-space no vistas), nunca más sin respaldo instrumental.
    c̄→0 ⇒ α→1 ⇒ vacuity=1: ignorancia instrumental ⇒ ignorancia del modelo."""
    def __init__(self, in_ch=256, n_classes=5):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, n_classes, 1)
        self.a = nn.Parameter(torch.tensor(1.0))   # h(u)=softplus(a·log u + b)
        self.b = nn.Parameter(torch.tensor(4.0))   # init: h(1)=softplus(4)≈4 → S~K+K·4

    def forward(self, hidden, c_dec):
        e_net = F.softplus(self.proj(hidden))
        h = F.softplus(self.a * torch.log(c_dec.clamp_min(1e-6)) + self.b)
        e = e_net * h
        alpha = e + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        return {'alpha': alpha, 'prob': alpha / S, 'evidence': e,
                'vacuity': alpha.shape[1] / S}
```

Inicialización de `b`: calibrar para que `S` inicial caiga en el orden de
`κ·mediana(N_eff)` del train (script en `scripts/init_evidence_scale.py`);
evita arrancar a órdenes de magnitud del target ancla de `50_loss` v3.

La supervisión de `S` la hace la KL Dirichlet anclada
(`L_seg = KL(Dir(κ·N_eff·Y + 1) ‖ Dir(α))`, spec 50 v3); este spec solo
garantiza el **sesgo inductivo arquitectónico** (S acotada por certeza). Dos
cabezas (mass/light) según corrección C3 de la revisión: dos `proj`, mismo
`(a, b)` compartido u independiente (ablar).

### Propiedades

10. **Vacuity bajo ignorancia**: `c_dec = 0` ⇒ `α = 1` exacto ⇒ vacuity = 1,
    `prob` uniforme. Test directo.
11. **Monotonía**: con `hidden` fijo, `S` es no decreciente en `c_dec`.
12. **α ≥ 1**, `prob` suma 1 (`atol=1e-6`), gradientes finitos — heredados
    de `40` v2.

## Integración: mapa de inserciones

```
dataset v3 ──(x, c) por modalidad──► PhysicalEncoder-N   (P1, sustituye 22)
                                  ► SwinSpatialEncoder    (estándar; c_spat se
                                                           agrega por escala con
                                                           avg-pool para skips)
                                  ► MambaSpectralEncoder  (estándar; c̄_spec
                                                           escalar a P3)
encoders ──► PrecisionGatedFusion (P3, sustituye 23) ──► UNetDecoder-N
            (P1 en UpBlocks; skips llevan (z, c); c de skips Swin = avg-pool
             de c_spat a cada escala — aproximación documentada)
decoder ──► EvidenceHead ×2 (P4, mod. 40) ──► EvidenceConv2d[PSF fijo]
            (P2, sustituye 43) ──► prob_obs
```

El tronco Swin y el Mamba **no** migran a la familia en v3 (riesgo/beneficio
desfavorable); su certeza entra agregada en P3 y en P4 vía `c_dec` (promedio
ponderado de las certezas de skip en el decoder).

## Validación

### Tests unitarios (`tests/unit/test_evidence_layers.py`)

Los 12 numerados arriba, más:

13. **Sustitución de nan_to_num**: muestra del piloto con NaN inyectados en
    `maps` → mismo output que la muestra limpia con `c=0` en esas posiciones
    (la garantía operacional #1 aplicada end-to-end al encoder físico).
14. **Conservación bajo bf16**: test 5 con `atol=1e-2` en mixed precision.
15. **Shapes y B=1** para las cuatro primitivas.

### Test de overfitting (Hito 3/4)

Encoder físico-N + EvidenceHead sobre 1 muestra del piloto: la KL anclada debe
bajar monotónicamente y `ρ_Spearman(S, κ·N_eff)` sobre los spaxels válidos del
piloto debe volverse positiva en < 500 iteraciones.

## Plan de ablación (la escalera que defiende la familia)

Mismo split, mismas semillas, métricas fijas: IoU/Dice, NLL/Brier contra
etiquetas suaves, `ρ_Spearman(S_pred, N_eff)` en val, AUROC de detección de
error con EU (información mutua, spec 42 v3), ECE.

| Nivel | Configuración | Pregunta que responde |
|---|---|---|
| A0 | Specs v2 estándar (conv, concat o cross-attn global, EDL anclada) | baseline |
| A1 | + NormConv en encoder físico y decoder (P1) | ¿la certeza en el tronco paga? |
| A2 | + PrecisionGatedFusion (P3) vs concat vs global | ¿el gating por precisión paga? |
| A3 | + EvidenceHead con techo de certeza (P4) | ¿el sesgo S≤f(c) mejora calibración/AUROC? |
| A4 | + PSF en espacio de evidencia (P2) vs prob-conv | ¿la α-convolución mejora nitidez intrínseca? |

Dos experimentos firma de la familia (ninguna red estándar puede hacerlos bien
sin reentrenar):

- **Barrido de ruido en test**: degradar el cubo/mapas con ruido consistente
  con IVAR escalado ×{1, 2, 4, 8} actualizando `c`. EPN debe degradar
  gracefully (NLL/ECE planos hasta ×4); A0 no tiene cómo saberlo. Curva
  métrica-vs-ruido por nivel de ablación.
- **Dropout de spaxels**: apagar aleatoriamente 10–40% de spaxels (`c=0`) vs
  el baseline con `nan_to_num`. Reportar caída de IoU; EPN debería dominar.

Criterio de adopción: A3 debe superar a A0 en NLL **y** AUROC de error; si
solo empata en IoU es aceptable (la familia compra incertidumbre, no
necesariamente accuracy).

## Criterios de aceptación

- [ ] Tests 1–15 pasan en CPU y GPU.
- [ ] Forward del modelo completo con P1–P4 sobre el piloto produce shapes
      correctos y gradientes finitos en bf16.
- [ ] Overhead de tiempo vs v2 estándar < 25% por step (la certeza duplica
      activaciones del tronco físico/decoder, no del Mamba).
- [ ] Mapa de pesos de atención por modalidad exportado en evaluación.
- [ ] Escalera A0–A4 reproducible desde configs (`configs/ablation_epn/*.yaml`).

## Notas de implementación

- `eps`: 1e-4 en NormConv (den), 1e-6 en logs. Con bf16, clampear `den` antes
  de dividir.
- No usar BatchNorm en ningún punto de la familia (rompe la semántica de c
  con batches pequeños); GroupNorm como en el resto del modelo.
- `attn_mask` aditivo de `nn.MultiheadAttention` espera forma
  `(B·n_heads, L_q, L_k)` o broadcastable; verificar el broadcasting del sesgo
  por cabeza (mismo sesgo para todas las cabezas es lo correcto aquí).
- Persistir `(a, b)` de cada `EvidenceHead` en el checkpoint con nombre — son
  los dos números que el capítulo de la tesis interpreta.
- Si A1 muestra inestabilidad temprana: congelar `theta_A` a init uniforme
  las primeras 2 épocas (equivale a promedio ponderado por c puro) y liberar.

## Anexo A (experimental, fuera del camino crítico) — Selective scan consciente de ruido

La selectividad de Mamba ya es dependiente de la entrada; la extensión natural
es modular el paso de actualización por la certeza espectral por canal:
`Δ_t' = Δ_t · φ(c_t)` con `φ` monótona en [0,1] (o equivalentemente atenuar
`B_t·x_t`). Canales λ ruidosos actualizan menos el estado; el estado retiene
el contexto de canales limpios. Riesgo: interactúa con la parametrización
interna de `mamba_ssm` (requiere fork del kernel o pre-escalado de la entrada
como aproximación `x_t' = c_t·x_t`, que es la versión barata a probar
primero). Si funciona, es publicable por separado; si no, no bloquea nada.
Bandera de config: `spectral.noise_aware_scan: {off, prescale, full}`.

## Referencias

- Knutsson, Westin (1993). *Normalized and Differential Convolution.* CVPR — agregación señal/certeza clásica.
- Eldesokey, Felsberg, Khan (2019). *Confidence Propagation through CNNs for Guided Sparse Depth Regression.* TPAMI; y (2020) *Uncertainty-Aware CNNs.* CVPR — versión profunda y aprendible.
- Wang, Yeung (2016). *Natural-Parameter Networks.* NeurIPS — activaciones como distribuciones de familia exponencial.
- Gast, Roth (2018). *Lightweight Probabilistic Deep Networks.* CVPR — propagación de momentos (Nivel B del encoder físico).
- Sensoy et al. (2018) NeurIPS; Charpentier et al. (2020) NeurIPS; Ryabinin et al. (2021) NeurIPS; Jürgens et al. (2024) ICML — linaje y crítica EDL; motivación del anclaje en N_eff (revisión §3.1).
- Revisión de arquitectura GalStructNet-S3, 2026-06-12 — §2.1, §2.2, §2.4, §3, C1–C9.
