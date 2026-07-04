"""Encoder espacial (Swin-T + FPN). Spec: specs/21_encoder_spatial.md - Hito 3.

v3: skips a 256 canales (contrato corregido C9); expone `mult` (multiplo
espacial requerido) para el dataset; padding interno dinamico desde
x.shape[-2:] (C5). Sin test de equivariancia (C8): la consistencia D4 del
modelo completo vive en tests/integration.

Contrato:
    forward(image (B,3,H,W)) -> (features (B,d_out,H,W),
                                 skips [(B,d_out,H/2,W/2), (/4), (/8)])
El input llega del dataset ya paddeado a multiplo de `mult` (= patch_size *
2**(n_stages-1) = 32 con los defaults); si no lo esta, este modulo paddea
internamente y recorta al final (size-agnostic, cero literales).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwinSpatialEncoder(nn.Module):
    def __init__(self, d_out: int = 256, in_channels: int = 3,
                 embed_dim: int = 96, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), window_size: int = 8,
                 patch_size: int = 4, pretrained: bool = False,
                 img_size: int = 224):
        super().__init__()
        from timm.models.swin_transformer import SwinTransformer

        self.patch_size = patch_size
        self.n_stages = len(depths)
        self._mult = patch_size * 2 ** (self.n_stages - 1)
        self.window_size = window_size

        # img_size es solo el tamano de construccion de timm (tablas de bias
        # relativas); el forward acepta cualquier multiplo via strict_img_size
        # =False y padding dinamico propio.
        self.swin = SwinTransformer(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, num_classes=0, global_pool="")
        self.swin.patch_embed.strict_img_size = False  # type: ignore[assignment]
        self.swin.patch_embed.dynamic_img_pad = True  # type: ignore[assignment]
        self._cur_size: tuple[int, int] | None = None

        chans = [embed_dim * 2 ** i for i in range(self.n_stages)]
        self.lateral = nn.ModuleList([nn.Conv2d(c, d_out, 1) for c in chans])
        self.fpn_smooth = nn.ModuleList(
            [nn.Conv2d(d_out, d_out, 3, padding=1) for _ in chans])

        if pretrained:
            self._load_pretrained(embed_dim, depths, num_heads, in_channels)

    @property
    def mult(self) -> int:
        """Multiplo espacial requerido = patch_size * 2**(n_stages-1)."""
        return self._mult

    def _load_pretrained(self, embed_dim, depths, num_heads, in_chans) -> None:
        """Init desde Swin-T ImageNet (specs/21, opcional). Los pesos de
        posicion relativa dependen de window_size: timm los re-interpola."""
        import timm
        ref = timm.create_model("swin_tiny_patch4_window7_224",
                                pretrained=True, in_chans=in_chans)
        state = ref.state_dict()
        missing = self.swin.load_state_dict(state, strict=False)
        del ref
        self._pretrained_missing = len(missing.missing_keys)

    def _stage_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward por stages de timm devolviendo los 4 mapas NCHW."""
        feats = []
        x = self.swin.patch_embed(x)
        for layer in self.swin.layers:
            # timm SwinTransformerStage: downsample primero, luego bloques
            x = layer(x)
            feats.append(x.permute(0, 3, 1, 2).contiguous())
        return feats

    def _sync_input_size(self, hw: tuple[int, int]) -> None:
        """Reconstruye las mascaras SW-MSA de timm cuando cambia el tamano
        (size-agnostic, C5). Los bundles MaNGA son pocos tamanos distintos:
        la reconstruccion se amortiza; los parametros no cambian."""
        if hw != self._cur_size:
            # la reinterpolacion de rel-pos-bias de timm mezcla devices si el
            # modulo vive en CUDA: redimensionar en CPU y volver
            dev = next(self.swin.parameters()).device
            if dev.type != "cpu":
                self.swin.to("cpu")
            self.swin.set_input_size(img_size=hw,
                                     window_size=(self.window_size,
                                                  self.window_size),
                                     always_partition=True)
            if dev.type != "cpu":
                self.swin.to(dev)
            self._cur_size = hw

    def forward(self, image: torch.Tensor):
        H, W = image.shape[-2:]
        ph, pw = (-H) % self._mult, (-W) % self._mult
        x = F.pad(image, (0, pw, 0, ph)) if (ph or pw) else image

        Hp, Wp = x.shape[-2:]
        self._sync_input_size((int(Hp), int(Wp)))
        feats = self._stage_features(x)
        lat = [self.lateral[i](f) for i, f in enumerate(feats)]

        out: list[torch.Tensor] = [torch.empty(0)] * len(lat)
        out[-1] = self.fpn_smooth[-1](lat[-1])
        for i in range(len(lat) - 2, -1, -1):
            up = F.interpolate(out[i + 1], size=lat[i].shape[-2:],
                               mode="bilinear", align_corners=False)
            out[i] = self.fpn_smooth[i](lat[i] + up)

        features = F.interpolate(out[0], size=(H, W), mode="bilinear",
                                 align_corners=False)
        # skips a resoluciones ~H/2^k del tamano NATIVO (el decoder interpola)
        skips = []
        for i in range(3):
            size = (max(1, H // 2 ** (i + 1)), max(1, W // 2 ** (i + 1)))
            skips.append(F.interpolate(out[i], size=size, mode="bilinear",
                                       align_corners=False))
        return features, skips
