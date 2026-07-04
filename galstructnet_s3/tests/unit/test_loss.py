"""Tests de specs/50_loss.md - los de referencia EJECUTABLES; el resto TODO."""
import pytest

torch = pytest.importorskip("torch")
from galstructnet_s3.losses.dirichlet import build_anchor, dirichlet_kl, anchored_seg_loss
from galstructnet_s3.losses.physics import physics_constraint_loss


def _rand_alpha(B=2, K=5, H=6, W=6, scale=30.0, seed=0):
    torch.manual_seed(seed)
    return 1.0 + torch.rand(B, K, H, W) * scale


def test_01_kl_identidad_y_no_negatividad():
    a = _rand_alpha()
    m = torch.ones(a.shape[0], *a.shape[-2:], dtype=torch.bool)
    assert dirichlet_kl(a, a, m).abs().item() < 1e-5
    b = _rand_alpha(seed=1)
    assert dirichlet_kl(a, b, m).item() >= 0
    assert abs(dirichlet_kl(a, b, m).item() - dirichlet_kl(b, a, m).item()) > 1e-6


def test_02_ancla_con_neff_cero_es_ignorancia():
    Y = torch.softmax(torch.randn(2, 5, 4, 4), dim=1)
    n_eff = torch.zeros(2, 4, 4)
    a_star = build_anchor(Y, n_eff, kappa=0.5)
    assert torch.allclose(a_star, torch.ones_like(a_star))


def test_03_gradiente_hacia_S():
    """Con prob_pred == Y pero S distinta del ancla, hay gradiente (innovacion)."""
    Y = torch.softmax(torch.randn(1, 5, 4, 4), dim=1)
    n_eff = torch.full((1, 4, 4), 100.0)
    S_wrong = torch.tensor(8.0, requires_grad=True)
    alpha = Y * S_wrong  # media exacta, concentracion equivocada
    alpha = alpha - alpha.min() + 1.0  # asegurar >=1 manteniendo grad
    m = torch.ones(1, 4, 4, dtype=torch.bool)
    loss = anchored_seg_loss(alpha, Y, n_eff, m, kappa=0.5)
    loss.backward()
    assert S_wrong.grad is not None and S_wrong.grad.abs().item() > 0


def test_04_cap_y_mascara():
    Y = torch.softmax(torch.randn(1, 5, 3, 3), dim=1)
    n_eff = torch.full((1, 3, 3), 1e6)
    a_star = build_anchor(Y, n_eff, kappa=1.0, n_eff_cap=100.0)
    assert a_star.sum(1).max().item() <= 1.0 * 100.0 + 5 + 1e-3


def test_07_lphys_ponderada_por_masa_direccion():
    prob = torch.zeros(1, 5, 4, 4)
    prob[:, 0] = 1.0                                           # todo bulge
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    w = torch.zeros(1, 4, 4)
    w[:, :2] = 10.0                                            # masa arriba
    tgt = torch.tensor([[1.0, 0, 0, 0, 0]])
    assert physics_constraint_loss(prob, mask, w, tgt).item() < 1e-6
    tgt2 = torch.tensor([[0.0, 1.0, 0, 0, 0]])
    # discrepancia total en 2 de 5 clases: (0.95+0.95)/5 = 0.38 (el spec
    # promedia sobre clases); el umbral comprueba direccion, no magnitud
    assert physics_constraint_loss(prob, mask, w, tgt2).item() > 0.3


def test_05_limite_duro_ranking_ce():
    """Y one-hot, N_eff grande: el ranking POR SPAXEL de la KL forward
    coincide con el de la CE (Spearman > 0.95, predicciones aleatorias)."""
    from scipy.stats import spearmanr
    torch.manual_seed(3)
    N = 500
    y_idx = torch.randint(0, 5, (N,))
    Y = torch.nn.functional.one_hot(y_idx, 5).float()          # (N, 5)
    a_star = 1.0 * 5000.0 * Y + 1.0                            # kappa=1
    # concentracion S FIJA entre predicciones: la KL es "CE + termino de
    # concentracion" (specs/50); a S constante el ranking debe ser el de CE
    e = torch.rand(N, 5)
    alpha = 1.0 + 20.0 * e / e.sum(1, keepdim=True)
    prob = alpha / alpha.sum(1, keepdim=True)

    # KL(Dir(a*) || Dir(alpha)) por spaxel, forma cerrada (specs/50)
    a0, b0 = a_star.sum(1), alpha.sum(1)
    kl = (torch.lgamma(a0) - torch.lgamma(a_star).sum(1)
          - torch.lgamma(b0) + torch.lgamma(alpha).sum(1)
          + ((a_star - alpha)
             * (torch.digamma(a_star)
                - torch.digamma(a0).unsqueeze(1))).sum(1))
    ce = -(Y * prob.clamp_min(1e-8).log()).sum(1)
    rho = spearmanr(kl.numpy(), ce.numpy()).statistic
    assert rho > 0.95, rho


def test_06_c3_seg_mass_no_toca_head_lum():
    from galstructnet_s3.losses.total import GalStructNetLossV3
    from galstructnet_s3.models.heads.segmentation import DualSegHeads
    torch.manual_seed(0)
    B, H, W = 1, 8, 8
    heads = DualSegHeads(kind="evidence")
    hidden = torch.randn(B, 256, H, W)
    c_dec = torch.rand(B, 1, H, W)
    out = heads(hidden, c_dec)
    batch = _fake_batch(B, H, W)
    loss_fn = GalStructNetLossV3(w_boundary=0.0, w_phys=0.0, w_psf=0.0)
    L = loss_fn({**out, "boundary": out["lum"]["prob"]}, batch)
    L["seg_mass"].backward(retain_graph=True)
    g_lum = heads.head_lum.proj.weight.grad
    assert g_lum is None or g_lum.abs().sum() == 0
    assert heads.head_mass.proj.weight.grad is not None


def _fake_batch(B, H, W, seed=1):
    torch.manual_seed(seed)
    Y = torch.softmax(torch.randn(B, 5, H, W), dim=1)
    batch = {"M": torch.ones(B, H, W, dtype=torch.bool),
             "w_phys_mass": torch.rand(B, H, W),
             "target_fractions_mass": torch.softmax(torch.randn(B, 5), -1)}
    for t in ("mass", "lum"):
        batch[f"Y_{t}"] = Y.clone()
        batch[f"Y_{t}_obs"] = Y.clone()
        batch[f"n_eff_{t}"] = torch.rand(B, H, W) * 100
        batch[f"n_eff_{t}_obs"] = torch.rand(B, H, W) * 90
        batch[f"M_unc_{t}"] = torch.zeros(B, H, W, dtype=torch.bool)
    return batch


def test_08_backward_total_gradientes_finitos():
    from galstructnet_s3.losses.total import GalStructNetLossV3
    from galstructnet_s3.models.heads.psf import PSFEvidenceModule
    from galstructnet_s3.models.heads.segmentation import DualSegHeads
    torch.manual_seed(0)
    B, H, W = 2, 12, 12
    heads = DualSegHeads(kind="evidence")
    hidden = torch.randn(B, 256, H, W, requires_grad=True)
    out = heads(hidden, torch.rand(B, 1, H, W))
    out["boundary"] = out["lum"]["prob"]
    psf = torch.rand(B, 5, 5)
    psf = psf / psf.sum((-2, -1), keepdim=True)
    for t in ("mass", "lum"):
        a_obs, p_obs = PSFEvidenceModule()(out[t]["alpha"], psf)
        out[f"alpha_obs_{t}"] = a_obs
        out[f"prob_obs_{t}"] = p_obs
    batch = _fake_batch(B, H, W)
    L = GalStructNetLossV3()(out, batch)
    assert {"seg_lum", "dice_lum", "psf_lum", "seg_mass", "dice_mass",
            "psf_mass", "boundary", "phys", "total"} <= set(L)
    L["total"].backward()
    assert torch.isfinite(hidden.grad).all()
    for n, p in heads.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
