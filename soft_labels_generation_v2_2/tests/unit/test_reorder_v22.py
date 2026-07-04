"""Casos dorados de la regla de asignacion por permutacion (v2.2).

Medias GMM REALES (espacio original [eps, log10(R/Reff), |z|/Reff, E_norm])
de tres galaxias de la muestra de 94 — dos invertidas por la regla v2.1 y
una sana. Ver reports/investigacion_inversion_bulbo_disco.md.
"""
import numpy as np

from aperturenet_labels.phase_a.classifier import _reorder_components


def _perm_from(P_out, P_in):
    """Recupera la permutacion aplicada comparando columnas."""
    perm = []
    for i in range(3):
        for k in range(3):
            if np.allclose(P_out[:, i], P_in[:, k]):
                perm.append(k)
                break
    return tuple(perm)


def _run(means):
    P = np.eye(3)[np.array([0, 1, 2, 0, 1, 2])].astype(float)  # trazador
    out, branch = _reorder_components(P, np.asarray(means), "paper4d")
    return _perm_from(out, P), branch


def test_312423_empate_eps_no_invierte():
    # v2.1: empate eps 0.41 vs 0.41 -> coronaba comp0 (central) como disco
    means = [[+0.41, -0.57, 0.05, -0.70],   # compacto central ligado = BULGE
             [+0.03, +0.76, 8.05, -0.22],   # halo
             [+0.41, +0.13, 0.51, -0.43]]   # disco extendido = DISK
    (b, d, h), branch = _run(means)
    assert (b, d, h) == (0, 2, 1)
    assert branch == "permutation_v2.2"


def test_346164_pseudobulbo_rotante_es_bulbo():
    # v2.1: comp0 (central, eps=0.37 > 0.10) quedaba como disco
    means = [[+0.37, -0.39, 0.11, -0.69],
             [+0.10, +0.80, 5.44, -0.24],
             [+0.10, +0.19, 0.84, -0.43]]
    (b, d, h), _ = _run(means)
    assert (b, d, h) == (0, 2, 1)


def test_464534_caso_sano_sin_regresion():
    # v2.1 acertaba aqui; la v2.2 debe elegir lo mismo
    means = [[+0.02, -0.42, 0.09, -0.68],
             [-0.14, +0.67, 4.04, -0.24],
             [+0.24, +0.23, 0.68, -0.43]]
    (b, d, h), _ = _run(means)
    assert (b, d, h) == (0, 2, 1)


def test_disco_dominante_clasico():
    # configuracion arquetipica sin ambiguedad
    means = [[+0.05, -0.50, 0.10, -0.80],   # bulbo
             [+0.75, +0.30, 0.15, -0.45],   # disco
             [-0.02, +0.70, 4.00, -0.20]]   # halo
    (b, d, h), _ = _run(means)
    assert (b, d, h) == (0, 1, 2)
