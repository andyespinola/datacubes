"""TrainerV3. Spec: specs/60_training.md - Hito 5.

Estructura v2 conservada (train_epoch, validate, clip a norma 1.0 -
imprescindible con Mamba, run, checkpointing, mixed precision, wandb).
Anadidos v3: KL en fp32 (interno a dirichlet_kl), rho(S, N_eff) cada
validacion, consist/weak en Etapa 3+ (los pesos vienen de la config).

Precision: 'bf16' solo si la GPU lo soporta (Ampere+); si no, cae a fp32
con aviso (la RTX 2060 local no tiene bf16; la maquina de entrenamiento si).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..losses.total import GalStructNetLossV3
from ..losses.domain import mmd_loss, pooled_embeddings
from .curriculum import consistency_loss, weak_gz3d_loss


def spearman_rho(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman sin scipy (para el loop de validacion)."""
    def rank(v: torch.Tensor) -> torch.Tensor:
        r = torch.empty_like(v)
        r[v.argsort()] = torch.arange(len(v), dtype=v.dtype, device=v.device)
        return r
    if len(x) < 2:
        return float("nan")
    rx, ry = rank(x.flatten().float()), rank(y.flatten().float())
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (rx.norm() * ry.norm()).clamp_min(1e-12)
    return float((rx * ry).sum() / denom)


class _NullLogger:
    def log(self, *_args, **_kw) -> None: ...
    def finish(self) -> None: ...


class TrainerV3:
    """Loop de entrenamiento supervisado (Etapas 2-4). La Etapa 1 (MAE)
    usa `Stage1Trainer` con SpectralMAEPretrainer (curriculum.py)."""

    def __init__(self, model: nn.Module, loss_fn: GalStructNetLossV3,
                 cfg: dict, train_loader, val_loader,
                 device: str | None = None, logger: Any = None,
                 ckpt_dir: str | Path = "checkpoints"):
        tcfg = cfg.get("training", {})
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.stage = int(tcfg.get("stage", 2))
        self.epochs = int(tcfg.get("epochs", 1))
        self.clip_norm = float(tcfg.get("clip_norm", 1.0))
        weights = cfg.get("loss", {}).get("weights", {})
        self.w_consist = float(weights.get("consist", 0.0))
        self.w_weak = float(weights.get("weak", 0.0))
        self.w_mmd = float(weights.get("mmd", 0.0))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger if logger is not None else _NullLogger()
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=float(tcfg.get("lr", 5e-5)),
                                     weight_decay=float(
                                         tcfg.get("weight_decay", 0.01)))
        warmup = int(tcfg.get("warmup_epochs", 0))
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda e: min(1.0, (e + 1) / max(1, warmup)))
        self.autocast_dtype = self._resolve_precision(
            str(tcfg.get("precision", "bf16")))
        self.epoch = 0

    def _resolve_precision(self, precision: str) -> torch.dtype | None:
        if precision == "bf16":
            if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            warnings.warn("bf16 no soportado en este dispositivo: fp32",
                          stacklevel=2)
            return None
        if precision == "fp16":
            return torch.float16
        return None

    def _autocast(self):
        if self.autocast_dtype is None or self.device == "cpu":
            import contextlib
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.autocast_dtype)

    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    # -- pasos ---------------------------------------------------------------

    def compute_loss(self, outputs: dict, batch: dict) -> dict:
        L = self.loss_fn(outputs, batch)
        if self.stage >= 3:
            if "outputs_aug" in batch:                 # segundo forward D4
                L["consist"] = consistency_loss(
                    outputs["lum"]["prob"],
                    batch["outputs_aug"]["lum"]["prob"], batch["M"])
                L["total"] = L["total"] + self.w_consist * L["consist"]
            if batch.get("gz3d_mask") is not None:
                L["weak"] = weak_gz3d_loss(outputs, batch)
                L["total"] = L["total"] + self.w_weak * L["weak"]
        if self.stage >= 4 and self.w_mmd > 0 and "domain" in batch:
            emb = pooled_embeddings(outputs["hidden"], batch["M"])
            dom = batch["domain"].bool()
            L["mmd"] = mmd_loss(emb[~dom], emb[dom])   # MaNGIA vs MaNGA
            L["total"] = L["total"] + self.w_mmd * L["mmd"]
        return L

    def train_epoch(self) -> dict:
        self.model.train()
        agg: dict[str, float] = {}
        n = 0
        for batch in self.train_loader:
            batch = self._to_device(batch)
            self.opt.zero_grad()
            with self._autocast():
                out = self.model(batch)
                L = self.compute_loss(out, batch)
            L["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.clip_norm)
            self.opt.step()
            n += 1
            for k, v in L.items():
                agg[k] = agg.get(k, 0.0) + float(v)
        return {f"train/{k}": v / max(1, n) for k, v in agg.items()}

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        agg: dict[str, float] = {}
        S_all, neff_all = [], []
        n = 0
        for batch in self.val_loader:
            batch = self._to_device(batch)
            with self._autocast():
                out = self.model(batch)
                L = self.loss_fn(out, batch)
            n += 1
            for k, v in L.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            ms = batch["M"] & ~batch["M_unc_lum"]
            S_all.append(out["lum"]["alpha"].sum(1)[ms].float().cpu())
            neff_all.append(batch["n_eff_lum"][ms].float().cpu())
        metrics = {f"val/{k}": v / max(1, n) for k, v in agg.items()}
        # la innovacion, monitoreada SIEMPRE (specs/60)
        metrics["val/rho_S_neff"] = spearman_rho(torch.cat(S_all),
                                                 torch.cat(neff_all))
        return metrics

    # -- ciclo ----------------------------------------------------------------

    def run(self) -> dict:
        last: dict = {}
        for _ in range(self.epochs):
            if hasattr(self.train_loader.dataset, "set_epoch"):
                self.train_loader.dataset.set_epoch(self.epoch)
            train_m = self.train_epoch()
            val_m = self.validate()
            self.sched.step()
            last = {**train_m, **val_m, "epoch": self.epoch}
            self.logger.log(last)
            self.epoch += 1                    # epocas COMPLETADAS
            self.save_checkpoint(self.ckpt_dir / "last.pt")
        return last

    # -- checkpointing ---------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save({"model": self.model.state_dict(),
                    "opt": self.opt.state_dict(),
                    "sched": self.sched.state_dict(),
                    "epoch": self.epoch,
                    "cfg": self.cfg}, path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.sched.load_state_dict(ckpt["sched"])
        self.epoch = int(ckpt["epoch"])


class Stage1Trainer:
    """Etapa 1: masked spectral modeling conjunto MaNGIA U MaNGA (specs/60).

    Recibe el `SpectralMAEPretrainer` (curriculum.py) y loaders que entregan
    batches con 'cube' y 'M'. El exito se evalua fuera (val baja + UMAP de
    embeddings MaNGIA/MaNGA solapados)."""

    def __init__(self, pretrainer: nn.Module, cfg: dict, train_loader,
                 val_loader, device: str | None = None, logger: Any = None,
                 ckpt_dir: str | Path = "checkpoints"):
        tcfg = cfg.get("training", {})
        self.pre = pretrainer
        self.epochs = int(tcfg.get("epochs", 1))
        self.clip_norm = float(tcfg.get("clip_norm", 1.0))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger if logger is not None else _NullLogger()
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.pre.to(self.device)
        self.opt = torch.optim.AdamW(self.pre.parameters(),
                                     lr=float(tcfg.get("lr", 1e-4)),
                                     weight_decay=float(
                                         tcfg.get("weight_decay", 0.01)))

    def run(self) -> dict:
        last: dict = {}
        for epoch in range(self.epochs):
            self.pre.train()
            tot, n = 0.0, 0
            for batch in self.train_loader:
                cube = batch["cube"].to(self.device)
                M = batch["M"].to(self.device)
                self.opt.zero_grad()
                out = self.pre(cube, M)
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.pre.parameters(),
                                               self.clip_norm)
                self.opt.step()
                tot += float(out["loss"])
                n += 1
            self.pre.eval()
            vtot, vn = 0.0, 0
            with torch.no_grad():
                for batch in self.val_loader:
                    out = self.pre(batch["cube"].to(self.device),
                                   batch["M"].to(self.device))
                    vtot += float(out["loss"])
                    vn += 1
            last = {"epoch": epoch, "train/mae": tot / max(1, n),
                    "val/mae": vtot / max(1, vn)}
            self.logger.log(last)
            torch.save({"pretrainer": self.pre.state_dict(), "epoch": epoch},
                       self.ckpt_dir / "stage1_last.pt")
        return last
