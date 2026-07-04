# 21 — Encoder Espacial (Swin Transformer)

> Módulo: `models/encoders/spatial.py` · Hito: 3 · Sin dependencias internas.
>
> **Parche v3** (resto del spec v2 vigente): (a) el contrato de skips declara
> 256 canales (lo que el decoder espera; el código FPN ya los produce — C9);
> (b) el test de "equivariancia bajo rotación 90°" se ELIMINA — Swin con
> ventanas + padding asimétrico + patch-merging no es equivariante y el test
> fallaría siempre (C8); en su lugar, test de consistencia estadística del
> modelo completo bajo D4 tras des-rotar (en `tests/integration`); (c) el
> módulo expone `mult` (múltiplo espacial requerido = `patch_size ·
> 2^(n_stages-1)`) para que el dataset paddee dinámicamente — sin literales
> 69/72/74 (C5); el padding interno a múltiplo se calcula de `x.shape[-2:]`,
> no de un tamaño fijo. La concatenación opcional de mapas pyPipe3D como
> canales 4–9 sigue descartada en Hito 3 (el encoder físico los maneja).

## Responsabilidad

Procesa la imagen sintética (3 canales, RGB sintético) y produce un mapa de
features espaciales `(B, D_spat, 69, 69)` que captura **contexto morfológico
global**: dónde estamos en la galaxia, cuál es la geometría del bulbo, dónde
están los brazos.

A diferencia del encoder espectral, este encoder **sí mezcla información entre
spaxels vecinos** — esa es su razón de existir. Sin él, el modelo no podría
aprender que "este spaxel está en la periferia del disco" porque eso requiere
mirar a los vecinos.

## Por qué Swin Transformer (no CNN, no ViT plano)

Tres opciones se consideraron:

1. **CNN clásica (ResNet)**: campo receptivo crece linealmente con
   profundidad. Para que cada spaxel "vea" toda la galaxia (69 píxeles de
   diámetro), necesitamos muchas capas, lo que diluye la información local.

2. **ViT plano**: divide la imagen en parches y aplica self-attention global.
   Funciona pero pierde la jerarquía multi-escala que es importante para
   morfología (brazos finos vs disco extendido).

3. **Swin Transformer**: atención local en ventanas + ventanas desplazadas
   entre capas + jerarquía multi-escala (4 etapas con downsampling). Combina
   lo mejor: contexto local barato, contexto global gradual, multi-escala
   natural. **Esta es la elegida.**

Ya hay implementaciones probadas en `timm`. No reinventamos la rueda.

## Contrato de entrada

```python
image: torch.Tensor  # shape (B, 3, 69, 69), dtype float32
                     # canales: g, r, i sintéticos (banda SDSS)
                     # ya normalizada por el Dataset (z-score por banda)
```

## Contrato de salida

```python
features: torch.Tensor  # shape (B, D_spat, 69, 69), dtype float32
                        # D_spat = 256 (configurable)
                        # H, W preservados (interpolación de vuelta a 69×69)
skip_connections: list  # 3 tensores de skip para el decoder, TODOS a d_out=256
                        # (tras lateral conv del FPN; el decoder 30 v3 los
                        #  espera así):
                        #   skip[0]: (B, 256, H/2,  W/2)    alto detalle
                        #   skip[1]: (B, 256, H/4,  W/4)    medio
                        #   skip[2]: (B, 256, H/8,  W/8)    contexto
                        # (resoluciones aproximadas; el decoder interpola)
```

Las skip connections se exponen para que el decoder U-Net pueda usarlas.
**Esto es importante**: el decoder no recibe directamente del encoder, recibe
del fusion layer, pero las skips vienen de aquí.

## Algoritmo

```python
class SwinSpatialEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        d_out: int = 256,
        img_size: int = 69,             # input nativo MaNGIA
        embed_dim: int = 96,
        depths: tuple = (2, 2, 6, 2),
        num_heads: tuple = (3, 6, 12, 24),
        window_size: int = 6,
    ):
        super().__init__()
        # Padding interno: Swin requiere img_size divisible por
        # patch_size × 2^(num_stages-1) × window_size
        # Para nuestros parámetros: 4 × 8 × 6 = 192 → demasiado.
        # Solución: pad a 72 (divisible por 4 y 6) y procesar.
        self.input_pad = (1, 2, 1, 2)   # F.pad ordering: left, right, top, bottom
        self.img_size_padded = img_size + 3   # 72

        from timm.models.swin_transformer import SwinTransformer
        self.swin = SwinTransformer(
            img_size=self.img_size_padded,
            patch_size=4,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            features_only=True,         # devuelve los 4 stages, no clasificación
            out_indices=(0, 1, 2, 3),
        )

        # Cabezas para alinear cada stage a un canal output común
        # Stages tendrán canales: 96, 192, 384, 768 (con embed_dim=96)
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, d_out, 1) for c in (96, 192, 384, 768)
        ])

        self.fpn_smooth = nn.ModuleList([
            nn.Conv2d(d_out, d_out, 3, padding=1) for _ in range(4)
        ])

    def forward(self, image: Tensor) -> tuple[Tensor, list[Tensor]]:
        B = image.shape[0]
        # Pad input a tamaño compatible con Swin
        x = F.pad(image, self.input_pad, mode='constant', value=0.0)

        # Forward Swin: lista de 4 features
        feats = self.swin(x)
        # feats[i] tiene shape (B, C_i, H_i, W_i)

        # Aplicar lateral conv y construir FPN top-down
        lat = [self.lateral[i](feats[i]) for i in range(4)]

        # Top-down (FPN style)
        out = [None] * 4
        out[3] = self.fpn_smooth[3](lat[3])
        for i in range(2, -1, -1):
            up = F.interpolate(out[i+1], size=lat[i].shape[-2:],
                               mode='bilinear', align_corners=False)
            out[i] = self.fpn_smooth[i](lat[i] + up)

        # El nivel más fino out[0] es el que vamos a usar como features finales
        # pero todavía está a tamaño 18×18. Hay que upsamplear a 69×69.
        features = F.interpolate(out[0], size=(69, 69),
                                 mode='bilinear', align_corners=False)

        # Skip connections para el decoder: niveles intermedios sin upsample final
        # Convertimos los tamaños internos al tamaño nativo (sin el padding)
        skip_connections = []
        for i in range(3):  # solo 3 skips, no incluimos el más profundo
            skip_size = (69 // (2 ** (i + 1)) + 1, 69 // (2 ** (i + 1)) + 1)
            # interpolar a tamaño compatible con upsamples del decoder
            s = F.interpolate(out[i], size=skip_size,
                              mode='bilinear', align_corners=False)
            skip_connections.append(s)

        return features, skip_connections
```

## Detalles importantes

### El padding 69 → 72

Swin Transformer requiere que `img_size` sea divisible por
`patch_size × 2^(num_stages-1) × window_size`. Con nuestros parámetros eso
sería 4 × 8 × 6 = 192, que no funciona para 69. Pero después de patchify
(div por 4) y bajar 2 niveles más (div por 4 más), el feature map está a
9×9 en el stage más profundo. La constraint real es que 69 sea divisible
por `patch_size × 2^(num_stages-1) = 4 × 8 = 32` después del patch, lo cual
no se cumple — necesitamos 72 para pasar 72 → 18 → 9.

Pad asimétrico (1, 2, 1, 2) → 72 funciona. El padding cae en spaxels de fondo
del HDF5 (que ya son cero) así que no introducimos artefactos.

### Por qué FPN

Sin FPN, las features finales del Swin a tamaño 9×9 perdieron demasiada
resolución espacial para ser útiles a nivel de spaxel. El FPN top-down
combina información semántica de niveles profundos con resolución de niveles
superficiales, dando features ricas a alta resolución.

### Concatenar pyPipe3D maps como canales adicionales (opcional)

En el plan original se mencionó concatenar los 6 mapas pyPipe3D a la imagen
sintética como canales 4-9. **No lo hagamos en Hito 3.** Razones:

1. El encoder físico se encarga de los pyPipe3D maps específicamente.
2. Concatenarlos aquí complica el dominio de la imagen sintética y la
   transferencia MaNGIA → MaNGA (los maps tienen escalas distintas).
3. La fusion layer ya los combina downstream.

**Si en Hito 5 vemos que el modelo no usa bien la información morfológica**,
esta es la primera ablation a probar.

## Validación

### Tests unitarios (`tests/unit/test_spatial_encoder.py`)

1. **Shape de salida correcto**: input `(2, 3, 69, 69)` → output features
   `(2, 256, 69, 69)` y 3 skip connections con shapes razonables.
2. **Determinismo**: en eval, dos forwards iguales dan output igual.
3. **(ELIMINADO en v3 — C8)** El test de equivariancia 90° del encoder
   espacial se retira: Swin no es equivariante. La invariancia que importa
   (modelo completo bajo D4 + des-rotación, con tolerancia) se prueba en
   `tests/integration/test_d4_consistency.py`.
4. **Sensibilidad espacial**: cambiar el píxel `(34, 34)` del input debe
   afectar features en una vecindad amplia (`> 5×5`), confirmando que el
   encoder mezcla información espacial.

### Test de carga de pesos pretrained (opcional)

Swin tiene pesos preentrenados en ImageNet disponibles en `timm`. Carga estos
pesos como inicialización:

```python
from timm import create_model
swin_pretrained = create_model('swin_tiny_patch4_window7_224',
                                pretrained=True, in_chans=3)
# Adaptar pesos al `img_size=72, window_size=6` (algunos pesos no transfieren)
```

Esto NO es requisito del Hito 3, pero acelera convergencia en Etapa 1.

## Criterios de aceptación

- [ ] Tests unitarios pasan.
- [ ] Output features tiene shape `(B, 256, 69, 69)`.
- [ ] Tres skip connections con shapes correctos para el decoder.
- [ ] Forward sobre batch de 4 en GPU A100: < 200 ms.
- [ ] Memoria pico en GPU: < 4 GB con `B=4`.
- [ ] Backward produce gradientes finitos.

## Notas de implementación

- **Dependencia**: `pip install timm`. Maduro y estable.
- **Si timm no funciona**: implementar Swin a mano es un proyecto en sí mismo.
  Alternativa más simple: ResNet-50 con dilated convolutions. Plan B.
- **Window size = 6**: elegido para que 72 sea divisible por 6 (12 ventanas
  por dimensión). Window size = 7 (default de Swin) requiere 56, 84, 112,
  no 72. Si quieres usar pesos pretrained, esto es un problema.
