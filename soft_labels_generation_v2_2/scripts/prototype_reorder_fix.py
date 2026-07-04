# Prototipo de la regla de asignacion de roles GMM corregida, evaluada a
# nivel de particulas sobre las 94 galaxias procesadas (sin re-correr nada).
#
# Regla ACTUAL (v2.1): disk = argmax(mean eps); bulge = mas ligado del resto
#   -> falla cuando el componente compacto central tiene eps medio comparable
#      al del disco extendido (empates resueltos por orden arbitrario).
# Regla PROPUESTA: asignacion conjunta por permutacion (3! = 6) maximizando
#   s_bulge(k) = -R_hat(k) - E_hat(k)          (compacto y ligado; SIN eps:
#                                               el pseudobulbo rotante cuenta
#                                               como bulbo, semantica
#                                               observacional/MORDOR)
#   s_disk(k)  = eps_hat(k) + R_hat(k) - z_hat(k)  (rotante, extendido, delgado)
#   s_halo(k)  = z_hat(k) - eps_hat(k)             (grueso, no rotante)
# con features estandarizadas ENTRE los 3 componentes de cada galaxia.
from itertools import permutations
from pathlib import Path

import h5py
import numpy as np

BASE = Path("/media/andy/Data/tng/mangia_flat/intermediate/phase_a")
EPS, LOGR, ZH, EN = 0, 1, 2, 3


def old_rule(means):
    """Replica _reorder_components v2.1 (feature_set=paper4d)."""
    disk_k = int(np.argmax(means[:, EPS]))
    rest = [k for k in range(3) if k != disk_k]
    e0, e1 = means[rest[0], EN], means[rest[1], EN]
    if abs(e0 - e1) >= 0.05:
        bulge_k = rest[0] if e0 < e1 else rest[1]
    else:
        bulge_k = rest[int(np.argmin([means[rest[0], LOGR],
                                      means[rest[1], LOGR]]))]
    halo_k = [k for k in rest if k != bulge_k][0]
    return (bulge_k, disk_k, halo_k)


def new_rule(means):
    m = (means - means.mean(0)) / np.maximum(means.std(0), 1e-9)
    s_b = -m[:, LOGR] - m[:, EN]
    s_d = m[:, EPS] + m[:, LOGR] - m[:, ZH]
    s_h = m[:, ZH] - m[:, EPS]
    best, best_perm = -np.inf, None
    for b, d, h in permutations(range(3)):
        tot = s_b[b] + s_d[d] + s_h[h]
        if tot > best:
            best, best_perm = tot, (b, d, h)
    return best_perm


def r_medios(P_raw, perm, R):
    """r medio ponderado por prob de bulge y disk bajo la permutacion."""
    pb, pd = P_raw[:, perm[0]], P_raw[:, perm[1]]
    rb = (R * pb).sum() / max(pb.sum(), 1e-9)
    rd = (R * pd).sum() / max(pd.sum(), 1e-9)
    return rb, rd


res = {"total": 0, "sin_gmm": 0, "perm_cambia": 0,
       "inv_antes": 0, "inv_despues": 0, "arregladas": [], "rotas": []}
for gdir in sorted(BASE.iterdir()):
    gid = gdir.name
    f_init = gdir / "particle_labels_initial.h5"
    f_feat = gdir / "particle_features.h5"
    if not (f_init.exists() and f_feat.exists()):
        continue
    with h5py.File(f_init) as f:
        if "gmm_params" not in f:
            res["sin_gmm"] += 1
            continue
        means = f["gmm_params/means_original_space"][()]
        P_cls = f["P_class"][()]            # ya reordenada [b, d, h]
    with h5py.File(f_feat) as f:
        R = f["kinematic/R"][()][: len(P_cls)]
        reff = np.nan
        for grp in ("metadata", "quality"):
            if grp in f:
                for k, v in f[grp].attrs.items():
                    if "eff" in k.lower():
                        reff = float(v)
    res["total"] += 1

    perm_old = old_rule(means)
    # reconstruir P_raw: P_cls[:, i] = P_raw[:, perm_old[i]]
    P_raw = np.empty_like(P_cls)
    for i, k in enumerate(perm_old):
        P_raw[:, k] = P_cls[:, i]
    perm_new = new_rule(means)

    rb_o, rd_o = r_medios(P_raw, perm_old, R)
    rb_n, rd_n = r_medios(P_raw, perm_new, R)
    inv_o = rd_o < rb_o * 0.9          # disco claramente mas interno que bulbo
    inv_n = rd_n < rb_n * 0.9
    res["inv_antes"] += inv_o
    res["inv_despues"] += inv_n
    if perm_new != perm_old:
        res["perm_cambia"] += 1
    if inv_o and not inv_n:
        res["arregladas"].append(gid)
    if inv_n and not inv_o:
        res["rotas"].append(gid)

print(f"galaxias con GMM: {res['total']} (fallback sin GMM: {res['sin_gmm']})")
print(f"permutacion cambia con la regla nueva: {res['perm_cambia']}")
print(f"inversion r_disk<r_bulge  ANTES: {res['inv_antes']}   "
      f"DESPUES: {res['inv_despues']}")
print(f"arregladas ({len(res['arregladas'])}): {res['arregladas']}")
print(f"rotas nuevas ({len(res['rotas'])}): {res['rotas']}")
