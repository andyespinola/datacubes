from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from summarize_metrics import main


class SummarizeMetricsTest(unittest.TestCase):
    def test_writes_csv_and_markdown_from_absolute_glob(self) -> None:
        with tempfile.TemporaryDirectory(prefix="orientation-summary-") as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            for galaxy_id, accepted, cglobal, reasons in (
                ("TNG50-90-1", True, 0.82, []),
                ("TNG50-90-2", False, 0.41, ["Cglobal<0.60", "C_disco<0.70"]),
            ):
                galaxy_dir = outputs / galaxy_id
                galaxy_dir.mkdir(parents=True)
                (galaxy_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "galaxy_id": galaxy_id,
                            "snapshot": 90,
                            "subhalo_id": int(galaxy_id.rsplit("-", 1)[1]),
                            "accepted": accepted,
                            "fvalid": 0.75,
                            "Cglobal": cglobal,
                            "variant": "Y_lum_psf",
                            "failure_reasons": reasons,
                            "classes": {"bulbo": cglobal - 0.05, "disco": cglobal + 0.02},
                        }
                    ),
                    encoding="utf-8",
                )

            out_csv = root / "summary.csv"
            out_md = root / "summary.md"
            main(
                [
                    "--metrics-glob",
                    str(outputs / "*" / "metrics.json"),
                    "--out",
                    str(out_csv),
                    "--report",
                    str(out_md),
                ]
            )

            self.assertTrue(out_csv.exists())
            self.assertTrue(out_md.exists())
            with out_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["galaxy_id"], "TNG50-90-1")
            self.assertIn("C_bulbo", rows[0])
            report = out_md.read_text(encoding="utf-8")
            self.assertIn("Tasa de aceptación: 50.0 %", report)
            self.assertIn("`Cglobal<0.60`", report)


if __name__ == "__main__":
    unittest.main()
