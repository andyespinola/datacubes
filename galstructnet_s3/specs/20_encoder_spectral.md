# 20 — Encoder Espectral (Mamba)

> Módulo: `models/encoders/spectral.py` · Hito: 3 · Depende de: nada del modelo,
> solo del dataset (Hito 1).
>
> **Parche v3** (resto del spec v2 vigente — Mamba se conserva como elección
> de ingeniería, no como contribución; revisión §2.3): (a) la ablación
> **Mamba vs Conv1D dilatada vs downsample agresivo (stride 8–16)** es
> OBLIGATORIA, no opcional — nivel A8 de la escalera; si la Conv1D empata,
> gana por simplicidad. (b) En la Etapa 1 (masked spectral, 60 v3) el encoder
> **NO poolea** el eje espectral: conserva la secuencia `(N, L', d_model)`
> para `SpectralMAEHead`; el `mean pooling` + `proj_out` quedan exclusivos del
> modo segmentación (añadir flag `return_sequence: bool`). (c) Plan de memoria
> documentado: congelar el encoder tras Etapa 1 y **cachear embeddings**
> `(B,256,H,W)` en disco para Etapas 2–3; descongelar solo en fine-tuning final
> si la ablación lo justifica. (d) La certeza espectral entra al modelo
> colapsada a `c̄_spec` (S/N por spaxel) vía la fusión (23 v3); el uso de IVAR
> per-λ dentro del scan es el Anexo A de 45 (experimental).

## Responsabilidad

Procesa el cubo espectral `(B, 6603, 69, 69)` y produce un mapa de features
espaciales `(B, D_spec, 69, 69)` donde cada posición `(i, j)` contiene un
embedding que resume la información espectral del spaxel correspondiente.

**Conceptualmente**: cada spaxel tiene un espectro de 6 603 canales que
queremos comprimir en un vector de dimensión `D_spec ≈ 256` que capture las
firmas físicas relevantes (líneas de absorción, continuo, líneas de emisión,
desplazamiento Doppler).

## Por qué Mamba (no Transformer ni CNN)

Tres opciones se consideraron:

1. **Transformer (self-attention)**: complejidad `O(L²) = O(43.6M)` operaciones
   por spaxel × 4761 spaxels = inviable.

2. **CNN 1D**: complejidad lineal pero **campo receptivo limitado** —
   capturar la relación entre la línea de Ca II (3933 Å) y la de Hα (6563 Å)
   requeriría muchas capas de convolución, perdiendo eficiencia.

3. **Mamba (state-space model)**: complejidad lineal `O(L)`, pero con campo
   receptivo global gracias a la recurrencia. **Esta es la elegida**.

Mamba es un modelo de espacio de estados selectivo (`SSM`) que mantiene un
estado oculto que se actualiza en cada paso de la secuencia. El estado tiene
información comprimida de toda la historia previa, lo que da al modelo memoria
larga sin el costo cuadrático de atención.

## Contrato de entrada

```python
cube: torch.Tensor  # shape (B, L, H, W), dtype float32
                    # L = 6603 (canales espectrales)
                    # H = W = 69 (spaxels)
                    # ya normalizado por el Dataset (log + z-score)
```

## Contrato de salida

```python
features: torch.Tensor  # shape (B, D_spec, H, W), dtype float32
                        # D_spec = 256 (configurable)
                        # H, W preservados (no hay pooling espacial)
```

## Algoritmo

```python
class MambaSpectralEncoder(nn.Module):
    """
    Procesa cada spaxel como una secuencia espectral independiente.

    Estructura:
      1. Proyección de entrada: Conv1D para reducir dimensionalidad inicial.
      2. N bloques Mamba que actualizan el estado oculto a lo largo de λ.
      3. Pooling sobre el eje espectral (mean pooling).
      4. Proyección de salida a D_spec.

    No hay mezcla espacial — esa es responsabilidad del encoder espacial.
    """

    def __init__(
        self,
        d_model: int = 128,        # dimensión interna del Mamba
        d_out: int = 256,          # dimensión de salida (D_spec)
        n_layers: int = 4,         # número de bloques Mamba
        d_state: int = 16,         # dimensión del estado del SSM
        d_conv: int = 4,           # tamaño del kernel conv interno
        expand: int = 2,           # factor de expansión en Mamba
        downsample_factor: int = 4,  # reduce L de 6603 a ~1650 antes de Mamba
    ):
        super().__init__()
        from mamba_ssm import Mamba

        # Reducción inicial del eje espectral por convolución con stride
        # Esto reduce la longitud efectiva sin perder mucha información
        self.proj_in = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )

        # Bloques Mamba (bidireccionales)
        self.blocks = nn.ModuleList([
            BidirectionalMambaBlock(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Pooling espectral + proyección final
        self.proj_out = nn.Linear(d_model, d_out)

    def forward(self, cube: Tensor) -> Tensor:
        B, L, H, W = cube.shape

        # Reordenar: tratar cada spaxel como una secuencia 1D independiente
        # (B, L, H, W) → (B*H*W, 1, L)
        x = rearrange(cube, 'b l h w -> (b h w) 1 l')

        # Proyección de entrada: reduce L y eleva canales
        # (B*H*W, 1, L) → (B*H*W, d_model, L/4)
        x = self.proj_in(x)

        # Mamba espera (N, L, D)
        x = rearrange(x, 'n d l -> n l d')

        # Bloques Mamba
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Pooling sobre el eje espectral
        # (N, L/4, d_model) → (N, d_model)
        x = x.mean(dim=1)

        # Proyección final
        # (N, d_model) → (N, d_out)
        x = self.proj_out(x)

        # Reordenar a mapa espacial
        # (B*H*W, d_out) → (B, d_out, H, W)
        return rearrange(x, '(b h w) d -> b d h w', b=B, h=H, w=W)


class BidirectionalMambaBlock(nn.Module):
    """
    Mamba forward + Mamba backward, sumados.

    El espectro tiene información asimétrica (líneas más al rojo de Hα son
    distintas a las más al azul), pero la dependencia entre líneas separadas
    es simétrica (Ca II azul puede informar sobre Hα rojo y viceversa).
    """

    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        from mamba_ssm import Mamba
        self.fwd = Mamba(d_model=d_model, d_state=d_state,
                         d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state,
                         d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (N, L, D)
        out_fwd = self.fwd(x)
        out_bwd = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        return self.norm(x + out_fwd + out_bwd)
```

### Por qué bidireccional

Una línea de absorción de Hα en 6563 Å se interpreta de forma distinta según el
contexto de la línea de Ca II en 3933 Å (azul). Una pasada solo forward
acumula información del azul al rojo; la pasada backward complementa con la
dirección opuesta. La suma combina ambos contextos.

### Por qué downsampling inicial

L=6603 es muy largo incluso para Mamba. El primer paso de convolución 1D con
`stride=2` aplicado dos veces lleva L a ~1650, manteniendo casi toda la
información (los espectros son suaves a escala de pocos Å). Sin esto, el
entrenamiento se vuelve impracticablemente lento.

## Memoria y rendimiento

Para `B=4, L=6603, H=W=69, D_spec=256`:

- Tensor de entrada: `4 × 6603 × 69 × 69 × 4 B = 502 MB` (float32)
- Después del downsampling 4×: `4 × 1650 × 69 × 69 × 128 × 4 B = 16 GB`
  → necesitamos procesar por chunks o usar gradient checkpointing
- Salida: `4 × 256 × 69 × 69 × 4 B = 19 MB`

**Recomendación**: gradient checkpointing en cada bloque Mamba. Esto multiplica
el tiempo de forward por ~1.3× pero reduce la memoria de activaciones por ~`n_layers`.

```python
from torch.utils.checkpoint import checkpoint

for block in self.blocks:
    x = checkpoint(block, x, use_reentrant=False)
```

## Validación

### Test unitario (`tests/unit/test_spectral_encoder.py`)

1. **Shape de salida correcto**: input `(2, 6603, 69, 69)` → output `(2, 256, 69, 69)`.
2. **Equivariance espacial**: rotando el input 90°, el output también rota 90°.
   Esto se cumple porque cada spaxel se procesa independientemente.
3. **Cero gradiente espacial entre spaxels**: el gradiente del output en `(i, j)`
   con respecto al input en `(i', j')` debe ser cero para `(i', j') ≠ (i, j)`.
4. **Forward determinístico en eval**: dos forwards consecutivos con el mismo
   input producen el mismo output (con `model.eval()` y sin dropout).
5. **Forward funciona con `B=1`**: caso edge.

### Test de overfitting (parte del Hito 3)

El encoder solo, conectado a una cabeza de clasificación dummy, debe poder
overfittear a 1 sample del piloto en < 100 iteraciones. Loss debe bajar
monotónicamente.

## Criterios de aceptación

- [ ] Tests unitarios pasan.
- [ ] Output tiene shape `(B, D_spec, H, W)`.
- [ ] Forward sobre 1 sample del piloto en CPU: < 30 segundos (sanity).
- [ ] Forward sobre batch de 4 en GPU A100: < 2 segundos.
- [ ] Memoria pico en GPU < 20 GB con `B=4` y gradient checkpointing.
- [ ] Backward pass produce gradientes finitos (no NaN, no Inf) en todos los
      parámetros.

## Notas de implementación

- **Dependencia**: `pip install mamba-ssm causal-conv1d`. Requiere CUDA. En
  desarrollo en CPU (sin GPU), reemplazar `Mamba` con un placeholder
  (e.g., `nn.GRU`) controlado por flag de config — esto permite que los tests
  unitarios corran en CI sin GPU.

- **Bidireccionalidad costosa**: si la memoria es problema, eliminar la rama
  backward. La pérdida de calidad es probablemente < 5%, pero a cambio
  reducimos memoria y tiempo casi a la mitad.

- **Stateful inference**: en inferencia podemos procesar la secuencia en chunks
  manteniendo el estado del SSM. Esto reduciría memoria a costa de complejidad
  de implementación. **No lo hagas en Hito 3** — solo si el Hito 5 muestra
  que la memoria es un problema real.

- **Alternative**: si Mamba resulta inestable o difícil de instalar, el
  fallback es un encoder basado en `S4` o incluso un Conv1D profundo con
  dilataciones progresivas. Esto está fuera del scope del Hito 3.
