"""Metricas v3. Spec: specs/70_evaluation.md - Hito 7.

Tres familias: segmentacion (IoU/Dice/acc por target), calidad
probabilistica con etiquetas suaves (soft-Brier/NLL + ECE), y las metricas
de la innovacion (rho(S, kappa*N_eff), AUROC de deteccion de error con EU).
Todo opera sobre spaxels validos aplanados en CPU (specs/70 notas).
"""
from __future__ import annotations

import torch


# -- scoring rules suaves (referencia del spec) -------------------------------

def soft_brier(prob, Y_soft, mask) -> torch.Tensor:
    d = ((prob - Y_soft) ** 2).sum(1)
    return (d * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def soft_nll(prob, Y_soft, mask) -> torch.Tensor:
    ce = -(Y_soft * torch.log(prob.clamp_min(1e-8))).sum(1)
    return (ce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


# -- segmentacion (v2 conservado, por target) ---------------------------------

def compute_iou_per_class(prob, Y_soft, mask, n_classes: int = 5) -> torch.Tensor:
    """IoU del argmax por clase; NaN si la clase no aparece."""
    pred = prob.argmax(1)[mask]
    true = Y_soft.argmax(1)[mask]
    ious = torch.full((n_classes,), float("nan"))
    for c in range(n_classes):
        inter = ((pred == c) & (true == c)).sum()
        union = ((pred == c) | (true == c)).sum()
        if union > 0:
            ious[c] = inter.float() / union.float()
    return ious


def compute_dice_per_class(prob, Y_soft, mask, n_classes: int = 5) -> torch.Tensor:
    pred = prob.argmax(1)[mask]
    true = Y_soft.argmax(1)[mask]
    dices = torch.full((n_classes,), float("nan"))
    for c in range(n_classes):
        inter = ((pred == c) & (true == c)).sum().float()
        total = (pred == c).sum().float() + (true == c).sum().float()
        if total > 0:
            dices[c] = 2.0 * inter / total
    return dices


def compute_pixel_accuracy(prob, Y_soft, mask) -> float:
    pred = prob.argmax(1)[mask]
    true = Y_soft.argmax(1)[mask]
    return float((pred == true).float().mean()) if len(pred) else float("nan")


def ece_argmax(prob, Y_soft, mask, n_bins: int = 15) -> float:
    """ECE clasico del argmax. NO basta con etiquetas suaves (specs/70):
    reportar siempre junto a soft_brier/soft_nll."""
    conf = prob.max(1).values[mask]
    correct = (prob.argmax(1) == Y_soft.argmax(1))[mask].float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(conf)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        sel = (conf > edges[i]) & (conf <= edges[i + 1])
        if sel.any():
            ece += float(sel.sum()) / n * abs(float(correct[sel].mean())
                                              - float(conf[sel].mean()))
    return ece


# -- la innovacion, medida (specs/70) ------------------------------------------

def concentration_calibration(alpha, n_eff, kappa, mask) -> dict:
    """La concentracion predicha trackea la estadistica fisica de la
    etiqueta? Si la innovacion funciona, S ~ kappa*N_eff."""
    from scipy.stats import linregress, spearmanr

    S = alpha.sum(1)[mask].float().cpu().numpy()
    target = (kappa * n_eff)[mask].float().cpu().numpy()
    if len(S) < 3:
        return {"rho_S_neff": float("nan"), "slope": float("nan"),
                "r2": float("nan")}
    rho = float(spearmanr(S, target).statistic)
    fit = linregress(target, S)
    return {"rho_S_neff": rho, "slope": float(fit.slope),
            "r2": float(fit.rvalue ** 2)}


def error_detection_auroc(EU, prob, Y_soft, mask) -> float:
    """La epistemica (informacion mutua, specs/42) predice los errores del
    argmax? AUROC alto = incertidumbre util para abstencion/triaje."""
    from torchmetrics.functional.classification import binary_auroc

    err = (prob.argmax(1) != Y_soft.argmax(1))[mask].float()
    scores = EU.squeeze(1)[mask].float()
    if err.min() == err.max():                    # sin ambas clases
        return float("nan")
    return float(binary_auroc(scores, err.long()))


def js_divergence_map(prob_a, prob_b) -> torch.Tensor:
    """JS(p_mass || p_lum) por spaxel (B, H, W): mapa de donde la
    estructura en masa y en luz difieren (producto cientifico, specs/40)."""
    pa = prob_a.clamp_min(1e-8)
    pb = prob_b.clamp_min(1e-8)
    m = 0.5 * (pa + pb)
    kl_am = (pa * (pa.log() - m.log())).sum(1)
    kl_bm = (pb * (pb.log() - m.log())).sum(1)
    return 0.5 * (kl_am + kl_bm)


def compute_global_fractions(prob, mask, w_map) -> torch.Tensor:
    """Fracciones globales PONDERADAS por masa/flujo (C7): comparables a
    las fracciones de masa del catalogo. (B, K)."""
    w = (w_map * mask.float()).unsqueeze(1)
    return (prob * w).sum((2, 3)) / w.sum((1, 2, 3)).clamp_min(1e-6).unsqueeze(1)
