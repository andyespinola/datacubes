"""CLI de entrenamiento y escalera de ablacion. Specs/60 - Hitos 5-6.

    python -m galstructnet_s3.cli train --config configs/base.yaml
    python -m galstructnet_s3.cli train --config ... --stage 1
    python -m galstructnet_s3.cli ablate --ladder epn [--epochs N]

La escalera corre las configs de configs/ablation_epn/ con las mismas
semillas, split y presupuesto (specs/60 'Escalera'); los resultados los
consume evaluation/ablation_report.py (Hito 7).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import GalStructDataset, collate_pad
from .data.dataset import check_gz3d_partition
from .losses.total import GalStructNetLossV3
from .models.model import build_model
from .training.curriculum import SpectralMAEPretrainer
from .training.trainer import Stage1Trainer, TrainerV3


def make_logger(cfg: dict):
    """wandb si esta disponible y habilitado; si no, logger nulo."""
    lcfg = cfg.get("logging", {})
    if not lcfg.get("wandb", False):
        return None
    import wandb
    return wandb.init(project=lcfg.get("project", "galstructnet-s3"),
                      mode=lcfg.get("mode", "offline"), config=cfg)


def _seed(cfg: dict) -> None:
    seed = int(cfg.get("training", {}).get("seed", 42))
    torch.manual_seed(seed)


def make_loaders(cfg: dict, mult: int) -> tuple[DataLoader, DataLoader]:
    dcfg = cfg.get("data", {})
    tcfg = cfg.get("training", {})
    common = dict(root=dcfg["root"], mult=mult,
                  restframe=dcfg.get("restframe", False),
                  carry_full_ivar=dcfg.get("carry_full_ivar", False),
                  views_per_epoch=dcfg.get("views_per_epoch", 4),
                  seed=tcfg.get("seed", 42))
    train_ds = GalStructDataset(split="train", augment=True, **common)
    val_ds = GalStructDataset(split="val", augment=False, **common)
    kw = dict(batch_size=int(tcfg.get("batch_size", 4)),
              collate_fn=collate_pad, num_workers=dcfg.get("num_workers", 4),
              pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw))


def cmd_train(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    if args.stage is not None:
        cfg.setdefault("training", {})["stage"] = args.stage
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    _seed(cfg)
    stage = int(cfg.get("training", {}).get("stage", 2))
    logger = make_logger(cfg)
    ckpt_dir = Path(args.ckpt_dir)

    if stage == 1:
        from .models.encoders.spectral import build_spectral_encoder
        scfg = dict(cfg["model"].get("spectral", {}))
        backbone = scfg.pop("backbone", "mamba")
        scfg["return_sequence"] = True
        encoder = build_spectral_encoder(backbone, **scfg)
        pre = SpectralMAEPretrainer(
            encoder, mask_ratio=cfg.get("training", {}).get("mask_ratio", 0.30))
        # mult=1: la Etapa 1 no pasa por el Swin (solo cubo)
        train_dl, val_dl = make_loaders(cfg, mult=1)
        trainer: TrainerV3 | Stage1Trainer = Stage1Trainer(
            pre, cfg, train_dl, val_dl, logger=logger, ckpt_dir=ckpt_dir)
        return trainer.run()

    model = build_model(cfg)
    mult = getattr(model.spatial, "mult", 32)
    if stage >= 3:
        check_gz3d_partition(cfg["data"]["root"])   # assert (specs/60)
    train_dl, val_dl = make_loaders(cfg, mult=mult)
    lcfg = cfg.get("loss", {})
    weights = dict(lcfg.get("weights", {}))
    loss_fn = GalStructNetLossV3(
        w_seg=weights.get("seg", 1.0), w_dice=weights.get("dice", 0.5),
        w_boundary=weights.get("boundary", 0.3 if stage >= 3 else 0.0),
        w_psf=weights.get("psf", 0.4 if stage >= 3 else 0.0),
        w_phys=weights.get("phys", 0.1),
        lambda_mass=weights.get("lambda_mass", 0.3),
        kappa=lcfg.get("kappa", 0.5),
        n_eff_cap=_resolve_cap(lcfg.get("n_eff_cap"), cfg),
        kl_direction=lcfg.get("kl_direction", "forward"),
        psf_mode=cfg["model"].get("psf_mode", "evidence"))
    trainer = TrainerV3(model, loss_fn, cfg, train_dl, val_dl,
                        logger=logger, ckpt_dir=ckpt_dir)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    return trainer.run()


def _resolve_cap(cap, cfg: dict) -> float | None:
    """'p99' se lee de norm_stats.json (specs/50 'Cap')."""
    if cap is None or cap == "none":
        return None
    if cap == "p99":
        stats_path = Path(cfg["data"]["root"]) / "norm_stats.json"
        if not stats_path.exists():
            return None
        stats = json.loads(stats_path.read_text())
        return float(stats["n_eff_cap"]["raw_lum"])
    return float(cap)


def cmd_ablate(args: argparse.Namespace) -> dict:
    """Corre la escalera A0-A4 con un solo comando (specs/60): mismas
    semillas, split y presupuesto para todos los niveles."""
    assert args.ladder == "epn"
    results: dict[str, dict] = {}
    out_dir = Path(args.ckpt_dir)
    for path in sorted(glob.glob(f"{args.configs_dir}/*.yaml")):
        name = Path(path).stem
        sub = argparse.Namespace(config=path, stage=args.stage,
                                 epochs=args.epochs,
                                 ckpt_dir=out_dir / name, resume=None)
        results[name] = cmd_train(sub)
    (out_dir / "ladder_results.json").write_text(
        json.dumps(results, indent=1, default=float))
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="galstructnet_s3")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--stage", type=int, default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--ckpt-dir", default="checkpoints")
    p_train.add_argument("--resume", default=None)
    p_train.set_defaults(fn=cmd_train)

    p_abl = sub.add_parser("ablate")
    p_abl.add_argument("--ladder", default="epn")
    p_abl.add_argument("--configs-dir", default="configs/ablation_epn")
    p_abl.add_argument("--stage", type=int, default=None)
    p_abl.add_argument("--epochs", type=int, default=None)
    p_abl.add_argument("--ckpt-dir", default="checkpoints/ablation")
    p_abl.set_defaults(fn=cmd_ablate)

    args = ap.parse_args(argv)
    metrics = args.fn(args)
    print(json.dumps(metrics, indent=1, default=float))


if __name__ == "__main__":
    main()
