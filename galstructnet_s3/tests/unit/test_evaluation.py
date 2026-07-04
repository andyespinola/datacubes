"""Tests de specs/70 (Hito 7): metricas, conformal, reporte, robustez."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from galstructnet_s3.evaluation.ablation_report import build_report
from galstructnet_s3.evaluation.conformal import (calibrate_mondrian,
                                                  coverage_efficiency,
                                                  make_strata, predict_sets)
from galstructnet_s3.evaluation.metrics import (compute_dice_per_class,
                                                compute_global_fractions,
                                                compute_iou_per_class,
                                                compute_pixel_accuracy,
                                                concentration_calibration,
                                                ece_argmax,
                                                error_detection_auroc,
                                                js_divergence_map)


def _perfect(B=1, K=5, H=8, W=8, seed=0):
    torch.manual_seed(seed)
    Y = torch.softmax(torch.randn(B, K, H, W) * 3, dim=1)
    m = torch.ones(B, H, W, dtype=torch.bool)
    return Y, m


def test_segmentacion_perfecta_y_degradada():
    Y, m = _perfect()
    assert compute_pixel_accuracy(Y, Y, m) == 1.0
    ious = compute_iou_per_class(Y, Y, m)
    dices = compute_dice_per_class(Y, Y, m)
    assert torch.nan_to_num(ious, 1.0).min() == 1.0
    assert torch.nan_to_num(dices, 1.0).min() == 1.0
    wrong = torch.roll(Y, 1, dims=1)
    assert compute_pixel_accuracy(wrong, Y, m) < 0.2


def test_ece_calibracion_perfecta_baja():
    Y, m = _perfect()
    assert ece_argmax(Y, Y, m) < 0.35        # confianza == accuracy en argmax
    over = torch.zeros_like(Y)
    over[:, 0] = 0.99
    over[:, 1:] = 0.0025                     # sobreconfiado y errado
    assert ece_argmax(over, Y, m) > ece_argmax(Y, Y, m)


def test_concentration_calibration_detecta_tracking():
    torch.manual_seed(0)
    B, K, H, W = 1, 5, 12, 12
    n_eff = torch.rand(B, H, W) * 100 + 5
    m = torch.ones(B, H, W, dtype=torch.bool)
    prob = torch.softmax(torch.randn(B, K, H, W), dim=1)
    alpha_good = prob * (0.5 * n_eff).unsqueeze(1)
    alpha_good = alpha_good.clamp_min(1e-3) + 1.0
    good = concentration_calibration(alpha_good, n_eff, 0.5, m)
    assert good["rho_S_neff"] > 0.95
    assert 0.8 < good["slope"] < 1.2
    alpha_flat = prob * 10 + 1.0
    flat = concentration_calibration(alpha_flat, n_eff, 0.5, m)
    assert abs(flat["rho_S_neff"]) < 0.5


def test_error_detection_auroc_eu_util():
    torch.manual_seed(1)
    B, K, H, W = 1, 5, 10, 10
    Y, m = _perfect(B, K, H, W, seed=1)
    prob = Y.clone()
    # inyectar errores en la mitad izquierda y darles EU alta
    prob[:, :, :, :5] = torch.roll(Y[:, :, :, :5], 1, dims=1)
    EU = torch.zeros(B, 1, H, W)
    EU[:, :, :, :5] = 1.0
    assert error_detection_auroc(EU, prob, Y, m) > 0.95
    EU_inv = 1.0 - EU
    assert error_detection_auroc(EU_inv, prob, Y, m) < 0.05


def test_js_map_cero_si_iguales():
    Y, _ = _perfect()
    assert js_divergence_map(Y, Y).abs().max() < 1e-7
    assert js_divergence_map(Y, torch.roll(Y, 1, 1)).mean() > 0.01


def test_global_fractions_ponderadas():
    B, K, H, W = 1, 5, 4, 4
    prob = torch.zeros(B, K, H, W)
    prob[:, 0, :2] = 1.0                    # bulge arriba
    prob[:, 1, 2:] = 1.0                    # disk abajo
    m = torch.ones(B, H, W, dtype=torch.bool)
    w = torch.zeros(B, H, W)
    w[:, :2] = 3.0                          # masa concentrada arriba
    w[:, 2:] = 1.0
    fr = compute_global_fractions(prob, m, w)
    assert torch.allclose(fr[0, :2], torch.tensor([0.75, 0.25]), atol=1e-5)


# -- conformal -----------------------------------------------------------------

def _conformal_data(N=4000, K=5, seed=0):
    """Predicciones ruidosas pero informativas sobre etiquetas one-hot."""
    torch.manual_seed(seed)
    y = torch.randint(0, K, (N,))
    logits = torch.randn(N, K)
    logits[torch.arange(N), y] += 2.0
    prob = torch.softmax(logits, dim=1)
    # forma (1, K, 1, N) para la interfaz espacial
    prob = prob.T.reshape(1, K, 1, N)
    Y = torch.nn.functional.one_hot(y, K).float().T.reshape(1, K, 1, N)
    n_eff = torch.rand(1, 1, N) * 200
    m = torch.ones(1, 1, N, dtype=torch.bool)
    return prob, Y, n_eff, m


def test_conformal_cobertura_90():
    prob_cal, Y_cal, neff_cal, m_cal = _conformal_data(seed=0)
    prob_te, Y_te, neff_te, m_te = _conformal_data(seed=1)
    st_cal = make_strata(Y_cal, neff_cal)
    st_te = make_strata(Y_te, neff_te)
    q = calibrate_mondrian(prob_cal, Y_cal, m_cal, st_cal, alpha=0.1)
    sets = predict_sets(prob_te, st_te, q)
    cov, size = coverage_efficiency(sets, Y_te, m_te)
    assert cov >= 0.88, cov                  # 1-alpha con margen de muestreo
    assert 1.0 <= size < 4.0, size           # eficiencia: no degenerado
    # conjunto nunca vacio
    assert (sets.sum(1) >= 1).all()


def test_conformal_alpha_mas_estricto_conjuntos_mas_grandes():
    prob_cal, Y_cal, neff, m = _conformal_data(seed=2)
    st = make_strata(Y_cal, neff)
    q10 = calibrate_mondrian(prob_cal, Y_cal, m, st, alpha=0.1)
    q01 = calibrate_mondrian(prob_cal, Y_cal, m, st, alpha=0.01)
    s10 = predict_sets(prob_cal, st, q10)
    s01 = predict_sets(prob_cal, st, q01)
    assert s01.sum() >= s10.sum()


# -- reporte -------------------------------------------------------------------

def test_ablation_report_emite_csv_y_tex(tmp_path):
    results = {"A0_baseline": {"val/total": 1.5, "val/rho_S_neff": 0.2},
               "A3_evidence": {"val/total": 1.2, "val/rho_S_neff": 0.6}}
    rp = tmp_path / "ladder_results.json"
    rp.write_text(json.dumps(results))
    csv = build_report(rp)
    assert "A0_baseline" in csv and "A3_evidence" in csv
    assert (tmp_path / "ablation_table.csv").exists()
    tex = (tmp_path / "ablation_table.tex").read_text()
    assert "\\toprule" in tex and "rho" in tex.replace("\\_", "_")
