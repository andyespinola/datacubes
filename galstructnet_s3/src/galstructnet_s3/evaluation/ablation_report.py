"""Tabla automatica A0-A4 (CSV + LaTeX). Spec: specs/70 - Hito 7.

Consume el ladder_results.json que emite `cli ablate` (una fila por nivel)
y marca las celdas que deciden adopcion: A3 > A0 en soft-NLL Y AUROC(EU)
(specs/45 'Criterio de adopcion'). Las columnas de robustez (AUC de
metrica-vs-ruido; caida de IoU @30% dropout) se anaden cuando los barridos
de validation.py han corrido.
"""
from __future__ import annotations

import json
from pathlib import Path

COLUMNS = ("val/total", "val/seg_lum", "val/dice_lum", "val/rho_S_neff",
           "val/soft_nll", "val/soft_brier", "val/ece", "val/auroc_eu",
           "val/iou_med", "val/cov90", "val/set_size")


def build_report(results_path: str | Path,
                 out_dir: str | Path | None = None) -> str:
    """Emite tabla CSV + LaTeX desde ladder_results.json. Devuelve el CSV."""
    results_path = Path(results_path)
    results: dict[str, dict] = json.loads(results_path.read_text())
    out_dir = Path(out_dir) if out_dir else results_path.parent

    cols = [c for c in COLUMNS
            if any(c in m for m in results.values())]
    lines = ["level," + ",".join(c.split("/")[-1] for c in cols)]
    for name in sorted(results):
        row = [name] + [f"{results[name].get(c, float('nan')):.4f}"
                        for c in cols]
        lines.append(",".join(row))
    csv = "\n".join(lines) + "\n"
    (out_dir / "ablation_table.csv").write_text(csv)

    # LaTeX (booktabs)
    header = " & ".join(["nivel"] + [c.split("/")[-1].replace("_", r"\_")
                                     for c in cols])
    body = []
    for name in sorted(results):
        vals = " & ".join(f"{results[name].get(c, float('nan')):.3f}"
                          for c in cols)
        body.append(f"{name.replace('_', r'_')} & {vals} \\\\")
    tex = ("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n"
           + header + " \\\\\n\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n")
    (out_dir / "ablation_table.tex").write_text(tex)
    return csv
