"""Cálculo del potencial gravitacional para el Extractor (spec 10).

Métodos:
- "snapshot": campo Potential per-partícula descargado de la API TNG (exacto).
- "octree": Barnes-Hut O(N log N) con numba (estrellas+gas+DM del cutout).
- "spherical": aproximación esféricamente simétrica
  Φ(r) = -G [ M(<r)/r + Σ_{r_j>r} m_j/r_j ]  — estándar en descomposiciones
  cinemáticas (Abadi 2003); rápida y suficiente para j_c(E) si el halo domina.
"""
from __future__ import annotations

import numpy as np

from ..core.constants import G_KPC_KMS2_MSUN

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def deco(f):
            return f

        return deco if args and callable(args[0]) is False or kwargs else (args[0] if args else deco)

    prange = range  # type: ignore


def compute_potential_spherical(
    target_pos: np.ndarray,
    source_pos: np.ndarray,
    source_mass: np.ndarray,
) -> np.ndarray:
    """Potencial esférico equivalente alrededor del origen (frame centrado)."""
    r_src = np.linalg.norm(source_pos, axis=1)
    r_tgt = np.linalg.norm(target_pos, axis=1)
    order = np.argsort(r_src)
    r_sorted = r_src[order]
    m_sorted = source_mass[order]
    m_cum = np.cumsum(m_sorted)  # M(<r) inclusivo
    # término exterior: Σ_{r_j > r} m_j / r_j  (acumulado desde afuera)
    over_r = m_sorted / np.clip(r_sorted, 1e-6, None)
    outer_cum = np.concatenate([np.cumsum(over_r[::-1])[::-1], [0.0]])
    idx = np.searchsorted(r_sorted, r_tgt, side="right")
    m_inner = np.where(idx > 0, m_cum[np.clip(idx - 1, 0, None)], 0.0)
    phi = -G_KPC_KMS2_MSUN * (
        m_inner / np.clip(r_tgt, 1e-3, None) + outer_cum[idx]
    )
    return phi


if HAS_NUMBA:

    @njit(cache=True)
    def _build_tree(pos, mass, leafsize):
        """Octree lineal en arrays. Devuelve nodos (center, half, mass, com,
        first_child, n_children, leaf_start, leaf_count) y orden de partículas."""
        n = pos.shape[0]
        # cota superior generosa de nodos
        max_nodes = max(64, int(4 * n / max(leafsize, 1) + 64) * 8)
        centers = np.zeros((max_nodes, 3))
        halves = np.zeros(max_nodes)
        masses = np.zeros(max_nodes)
        coms = np.zeros((max_nodes, 3))
        first_child = np.full(max_nodes, -1, np.int64)
        n_children = np.zeros(max_nodes, np.int64)
        leaf_start = np.full(max_nodes, -1, np.int64)
        leaf_count = np.zeros(max_nodes, np.int64)

        order = np.arange(n)
        # bounding cube
        lo = np.array([pos[:, 0].min(), pos[:, 1].min(), pos[:, 2].min()])
        hi = np.array([pos[:, 0].max(), pos[:, 1].max(), pos[:, 2].max()])
        c0 = (lo + hi) / 2.0
        h0 = (hi - lo).max() / 2.0 + 1e-3

        # pila de trabajo: (node_id, start, end) sobre 'order'
        node_count = 1
        centers[0] = c0
        halves[0] = h0
        stack_node = np.zeros(max_nodes, np.int64)
        stack_lo = np.zeros(max_nodes, np.int64)
        stack_hi = np.zeros(max_nodes, np.int64)
        sp = 0
        stack_node[0] = 0
        stack_lo[0] = 0
        stack_hi[0] = n
        sp = 1
        scratch = np.empty(n, np.int64)

        while sp > 0:
            sp -= 1
            nid = stack_node[sp]
            s = stack_lo[sp]
            e = stack_hi[sp]
            cnt = e - s
            # masa y com del nodo
            mtot = 0.0
            cx = 0.0
            cy = 0.0
            cz = 0.0
            for k in range(s, e):
                i = order[k]
                mtot += mass[i]
                cx += pos[i, 0] * mass[i]
                cy += pos[i, 1] * mass[i]
                cz += pos[i, 2] * mass[i]
            masses[nid] = mtot
            if mtot > 0:
                coms[nid, 0] = cx / mtot
                coms[nid, 1] = cy / mtot
                coms[nid, 2] = cz / mtot
            if cnt <= leafsize:
                leaf_start[nid] = s
                leaf_count[nid] = cnt
                continue
            # particionar en 8 octantes (counting sort estable)
            counts = np.zeros(8, np.int64)
            octs = np.empty(cnt, np.int64)
            ccx = centers[nid, 0]
            ccy = centers[nid, 1]
            ccz = centers[nid, 2]
            for k in range(cnt):
                i = order[s + k]
                q = 0
                if pos[i, 0] >= ccx:
                    q += 4
                if pos[i, 1] >= ccy:
                    q += 2
                if pos[i, 2] >= ccz:
                    q += 1
                octs[k] = q
                counts[q] += 1
            offs = np.zeros(8, np.int64)
            acc = 0
            for q in range(8):
                offs[q] = acc
                acc += counts[q]
            for k in range(cnt):
                scratch[s + offs[octs[k]]] = order[s + k]
                offs[octs[k]] += 1
            for k in range(s, e):
                order[k] = scratch[k]
            # crear hijos no vacíos
            first_child[nid] = node_count
            half_child = halves[nid] / 2.0
            acc = 0
            for q in range(8):
                if counts[q] == 0:
                    continue
                cid = node_count
                node_count += 1
                n_children[nid] += 1
                sx = 1.0 if (q & 4) else -1.0
                sy = 1.0 if (q & 2) else -1.0
                sz = 1.0 if (q & 1) else -1.0
                centers[cid, 0] = ccx + sx * half_child
                centers[cid, 1] = ccy + sy * half_child
                centers[cid, 2] = ccz + sz * half_child
                halves[cid] = half_child
                stack_node[sp] = cid
                stack_lo[sp] = s + acc
                stack_hi[sp] = s + acc + counts[q]
                sp += 1
                acc += counts[q]

        return (
            centers[:node_count],
            halves[:node_count],
            masses[:node_count],
            coms[:node_count],
            first_child[:node_count],
            n_children[:node_count],
            leaf_start[:node_count],
            leaf_count[:node_count],
            order,
        )

    @njit(parallel=True, cache=True)
    def _evaluate_potential(
        targets,
        pos,
        mass,
        centers,
        halves,
        masses,
        coms,
        first_child,
        n_children,
        leaf_start,
        leaf_count,
        order,
        theta,
        eps2,
        G,
    ):
        nt = targets.shape[0]
        phi = np.zeros(nt)
        for t in prange(nt):
            tx = targets[t, 0]
            ty = targets[t, 1]
            tz = targets[t, 2]
            acc = 0.0
            stack = np.empty(512, np.int64)
            sp = 0
            stack[0] = 0
            sp = 1
            while sp > 0:
                sp -= 1
                nid = stack[sp]
                if masses[nid] <= 0.0:
                    continue
                dx = coms[nid, 0] - tx
                dy = coms[nid, 1] - ty
                dz = coms[nid, 2] - tz
                d2 = dx * dx + dy * dy + dz * dz
                size = 2.0 * halves[nid]
                if leaf_start[nid] >= 0:
                    s = leaf_start[nid]
                    e = s + leaf_count[nid]
                    for k in range(s, e):
                        i = order[k]
                        ddx = pos[i, 0] - tx
                        ddy = pos[i, 1] - ty
                        ddz = pos[i, 2] - tz
                        dd2 = ddx * ddx + ddy * ddy + ddz * ddz
                        acc -= G * mass[i] / np.sqrt(dd2 + eps2)
                elif size * size < theta * theta * d2:
                    acc -= G * masses[nid] / np.sqrt(d2 + eps2)
                else:
                    fc = first_child[nid]
                    for c in range(n_children[nid]):
                        stack[sp] = fc + c
                        sp += 1
            phi[t] = acc
        return phi

    def compute_potential_octree(
        target_pos: np.ndarray,
        source_pos: np.ndarray,
        source_mass: np.ndarray,
        theta: float = 0.6,
        softening: float = 0.288,
        leafsize: int = 32,
    ) -> np.ndarray:
        tree = _build_tree(
            np.ascontiguousarray(source_pos, dtype=np.float64),
            np.ascontiguousarray(source_mass, dtype=np.float64),
            leafsize,
        )
        return _evaluate_potential(
            np.ascontiguousarray(target_pos, dtype=np.float64),
            np.ascontiguousarray(source_pos, dtype=np.float64),
            np.ascontiguousarray(source_mass, dtype=np.float64),
            *tree,
            theta,
            softening**2,
            G_KPC_KMS2_MSUN,
        )

else:  # pragma: no cover

    def compute_potential_octree(*args, **kwargs):  # type: ignore
        raise ImportError("numba no disponible; usa potential_method='spherical'")
