from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from validation.kinematic import (
    KinematicMomentMaps,
    KinematicValidationConfig,
    KinematicValidationInput,
    build_success_report,
    validate_kinematic_unit,
)
from validation.run_kinematic_validation import main
from validation.run_test_a_sensitivity import main as sensitivity_main


def synthetic_unit(with_bar: bool = True, with_moments: bool = False) -> KinematicValidationInput:
    h, w = 12, 12
    yy, xx = np.indices((h, w))
    radius = np.sqrt((xx - 5.5) ** 2 + (yy - 5.5) ** 2)
    y = np.zeros((5, h, w), dtype=np.float32)
    bulge = radius < 3
    disk = radius >= 3
    bar = (np.abs(yy - 5.5) <= 1.5) & (radius < 4.5)
    y[0, bulge] = 0.85
    y[1, disk] = 0.85
    y[2, bar] = 0.85 if with_bar else 0.0
    y[1, bar] = 0.10 if with_bar else y[1, bar]
    y[4] = np.clip(1.0 - np.sum(y[:4], axis=0), 0.0, 1.0)
    y /= np.clip(np.sum(y, axis=0, keepdims=True), 1e-6, None)

    v = (xx - 5.5).astype(np.float32) * 35.0
    sigma = np.full((h, w), 90.0, dtype=np.float32)
    sigma[bulge] = 180.0
    sigma[bar] = 135.0
    moments = None
    if with_moments:
        h3 = -v / np.nanmax(v)
        moments = KinematicMomentMaps(h3=h3.astype(np.float32), h4=np.zeros_like(h3), quality_mask=np.ones_like(v, dtype=bool))
    return KinematicValidationInput(
        unit_id="TNG50-87-141934-0",
        galaxy_id="TNG50-87-141934",
        canonical_id="TNG50-87-141934-0-37",
        view_id=0,
        y_int=y,
        m_val=np.ones((h, w), dtype=bool),
        v_star=v,
        sigma_star=sigma,
        r_bar=3.0 if with_bar else None,
        kinematic_moments=moments,
    )


def contaminated_reference_unit() -> KinematicValidationInput:
    h, w = 12, 12
    yy, xx = np.indices((h, w))
    radius = np.sqrt((xx - 5.5) ** 2 + (yy - 5.5) ** 2)
    y = np.zeros((5, h, w), dtype=np.float32)
    y[0, radius < 2.0] = 0.90
    y[1, (radius >= 2.0) & (radius < 4.0)] = 0.90
    y[4, radius >= 4.0] = 0.90

    v = (xx - 5.5).astype(np.float32) * 35.0
    sigma = np.full((h, w), 90.0, dtype=np.float32)
    sigma[radius < 2.0] = 180.0
    return KinematicValidationInput(
        unit_id="TNG50-87-141934-0",
        galaxy_id="TNG50-87-141934",
        canonical_id="TNG50-87-141934-0-37",
        view_id=0,
        y_int=y,
        m_val=np.ones((h, w), dtype=bool),
        v_star=v,
        sigma_star=sigma,
        r_bar=None,
    )


class KinematicValidationTests(unittest.TestCase):
    def test_validates_ab_without_penalizing_missing_h3h4(self) -> None:
        result = validate_kinematic_unit(synthetic_unit(with_bar=False), KinematicValidationConfig(min_spaxels_for_test=10))

        self.assertEqual(result.test_a_rotation, "PASS")
        self.assertEqual(result.rotation_test_mode, "contrast")
        self.assertEqual(result.rotation_reference_mode, "bulge_other")
        self.assertAlmostEqual(result.velocity_center_median, 0.0, places=5)
        self.assertGreater(result.v_over_sigma_ratio, 1.10)
        self.assertEqual(result.test_b_dispersion, "PASS")
        self.assertEqual(result.test_c_bar_sigma, "N/A")
        self.assertEqual(result.test_d_h3_signature, "N/A")
        self.assertTrue(result.passes)
        self.assertEqual(result.coherence_score, 1.0)

    def test_bar_and_h3_tests_can_pass(self) -> None:
        result = validate_kinematic_unit(
            synthetic_unit(with_bar=True, with_moments=True),
            KinematicValidationConfig(dominant_class_threshold=0.4, min_spaxels_for_test=10),
        )

        self.assertEqual(result.test_c_bar_sigma, "PASS")
        self.assertEqual(result.test_d_h3_signature, "PASS")
        self.assertTrue(result.h3h4_used)

    def test_success_report_uses_only_applicable_tests(self) -> None:
        config = KinematicValidationConfig(dominant_class_threshold=0.4, min_spaxels_for_test=10)
        results = [
            validate_kinematic_unit(synthetic_unit(with_bar=False), config),
            validate_kinematic_unit(synthetic_unit(with_bar=True, with_moments=True), config),
        ]

        report = build_success_report(results)

        self.assertEqual(report.n_units_total, 2)
        self.assertEqual(report.n_applicable_test_d, 1)
        self.assertEqual(report.success_rate_test_d, 100.0)

    def test_spearman_rotation_mode_remains_available(self) -> None:
        result = validate_kinematic_unit(
            synthetic_unit(with_bar=False),
            KinematicValidationConfig(rotation_test_mode="spearman", min_spaxels_for_test=10),
        )

        self.assertEqual(result.rotation_test_mode, "spearman")
        self.assertIsNotNone(result.rho_disk)
        self.assertEqual(result.test_a_rotation, "PASS")

    def test_bulge_reference_excludes_other_contamination(self) -> None:
        unit = contaminated_reference_unit()
        default_result = validate_kinematic_unit(unit, KinematicValidationConfig(min_spaxels_for_test=10))
        bulge_result = validate_kinematic_unit(
            unit,
            KinematicValidationConfig(rotation_reference_mode="bulge", min_spaxels_for_test=10),
        )

        self.assertEqual(default_result.test_a_rotation, "FAIL")
        self.assertGreater(default_result.n_other_spaxels, default_result.n_bulge_spaxels)
        self.assertEqual(bulge_result.test_a_rotation, "PASS")
        self.assertEqual(bulge_result.n_reference_spaxels, bulge_result.n_bulge_spaxels)

    def test_central_reference_is_available_for_test_a(self) -> None:
        result = validate_kinematic_unit(
            synthetic_unit(with_bar=False),
            KinematicValidationConfig(
                rotation_reference_mode="central",
                central_reference_radius_fraction=0.30,
                min_spaxels_for_test=10,
            ),
        )

        self.assertEqual(result.rotation_reference_mode, "central")
        self.assertGreaterEqual(result.n_reference_spaxels, 10)
        self.assertEqual(result.test_a_rotation, "PASS")

    def test_velocity_is_centered_before_vsigma(self) -> None:
        unit = synthetic_unit(with_bar=False)
        shifted = KinematicValidationInput(
            unit_id=unit.unit_id,
            galaxy_id=unit.galaxy_id,
            canonical_id=unit.canonical_id,
            view_id=unit.view_id,
            y_int=unit.y_int,
            m_val=unit.m_val,
            v_star=unit.v_star + 32000.0,
            sigma_star=unit.sigma_star,
            r_bar=unit.r_bar,
        )

        result = validate_kinematic_unit(shifted, KinematicValidationConfig(min_spaxels_for_test=10))

        self.assertGreater(result.velocity_center_median, 30000.0)
        self.assertLess(result.v_over_sigma_global_median, 2.0)
        self.assertEqual(result.test_a_rotation, "PASS")

    def test_cli_writes_expected_outputs_from_npz_maps(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kin-val-cli-") as tmp:
            root = Path(tmp)
            labels_dir = root / "labels"
            outdir = root / "validation"
            labels_dir.mkdir()
            unit = synthetic_unit(with_bar=False)
            labels_path = labels_dir / f"{unit.canonical_id}.labels.npz"
            summary_path = labels_dir / f"{unit.canonical_id}.summary.json"
            np.savez_compressed(
                labels_path,
                soft_mass=np.concatenate([np.zeros((1, *unit.m_val.shape), dtype=np.float32), unit.y_int, np.zeros((1, *unit.m_val.shape), dtype=np.float32)]),
                soft_light=np.concatenate([np.zeros((1, *unit.m_val.shape), dtype=np.float32), unit.y_int, np.zeros((1, *unit.m_val.shape), dtype=np.float32)]),
                hard_mass=np.zeros(unit.m_val.shape, dtype=np.int16),
                hard_light=np.zeros(unit.m_val.shape, dtype=np.int16),
                valid_mask=unit.m_val,
                class_names=np.array(["no_valido", "bulbo", "disco", "barra", "brazos", "other", "incierto"]),
            )
            summary_path.write_text(json.dumps({"bar_metadata": {"barred_target": False}}), encoding="utf-8")
            maps_path = root / "maps.npz"
            np.savez_compressed(maps_path, V=unit.v_star, SIGMA=unit.sigma_star)
            matched_path = root / "matched_units.csv"
            with matched_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "unit_id",
                        "galaxy_id",
                        "canonical_id",
                        "view",
                        "cube_path",
                        "maps2d_path",
                        "maps2d_format",
                        "v_map_key",
                        "sigma_map_key",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "unit_id": unit.unit_id,
                        "galaxy_id": unit.galaxy_id,
                        "canonical_id": unit.canonical_id,
                        "view": unit.view_id,
                        "cube_path": "",
                        "maps2d_path": str(maps_path),
                        "maps2d_format": "npz",
                        "v_map_key": "V",
                        "sigma_map_key": "SIGMA",
                    }
                )

            code = main(
                [
                    "--matched-units",
                    str(matched_path),
                    "--labels-dir",
                    str(labels_dir),
                    "--outdir",
                    str(outdir),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((outdir / "kinematic_validation_units.csv").exists())
            self.assertTrue((outdir / "kinematic_validation_report.json").exists())
            self.assertTrue((outdir / "kinematic_validation_report.md").exists())
            self.assertTrue((outdir / "coherence_score_histogram.png").exists())
            self.assertTrue((outdir / "test_a_diagnostics.csv").exists())
            self.assertTrue((outdir / "test_a_summary_by_view.csv").exists())
            self.assertTrue((outdir / "test_a_summary_by_global_vsigma.csv").exists())
            self.assertTrue((outdir / "test_a_extreme_pass_fail.md").exists())

    def test_sensitivity_runner_writes_combined_summary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kin-val-sensitivity-") as tmp:
            root = Path(tmp)
            labels_dir = root / "labels"
            outdir = root / "sensitivity"
            labels_dir.mkdir()
            unit = synthetic_unit(with_bar=False)
            labels_path = labels_dir / f"{unit.canonical_id}.labels.npz"
            summary_path = labels_dir / f"{unit.canonical_id}.summary.json"
            np.savez_compressed(
                labels_path,
                soft_mass=np.concatenate([np.zeros((1, *unit.m_val.shape), dtype=np.float32), unit.y_int, np.zeros((1, *unit.m_val.shape), dtype=np.float32)]),
                soft_light=np.concatenate([np.zeros((1, *unit.m_val.shape), dtype=np.float32), unit.y_int, np.zeros((1, *unit.m_val.shape), dtype=np.float32)]),
                hard_mass=np.zeros(unit.m_val.shape, dtype=np.int16),
                hard_light=np.zeros(unit.m_val.shape, dtype=np.int16),
                valid_mask=unit.m_val,
                class_names=np.array(["no_valido", "bulbo", "disco", "barra", "brazos", "other", "incierto"]),
            )
            summary_path.write_text(json.dumps({"bar_metadata": {"barred_target": False}}), encoding="utf-8")
            maps_path = root / "maps.npz"
            np.savez_compressed(maps_path, V=unit.v_star, SIGMA=unit.sigma_star)
            matched_path = root / "matched_units.csv"
            with matched_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "unit_id",
                        "galaxy_id",
                        "canonical_id",
                        "view",
                        "cube_path",
                        "maps2d_path",
                        "maps2d_format",
                        "v_map_key",
                        "sigma_map_key",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "unit_id": unit.unit_id,
                        "galaxy_id": unit.galaxy_id,
                        "canonical_id": unit.canonical_id,
                        "view": unit.view_id,
                        "cube_path": "",
                        "maps2d_path": str(maps_path),
                        "maps2d_format": "npz",
                        "v_map_key": "V",
                        "sigma_map_key": "SIGMA",
                    }
                )

            code = sensitivity_main(
                [
                    "--matched-units",
                    str(matched_path),
                    "--labels-dir",
                    str(labels_dir),
                    "--outdir",
                    str(outdir),
                    "--thresholds",
                    "1.00,1.20",
                    "--min-spaxels-for-test",
                    "10",
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((outdir / "ratio_1p00" / "kinematic_validation_report.json").exists())
            self.assertTrue((outdir / "ratio_1p20" / "kinematic_validation_report.json").exists())
            summary_csv = outdir / "test_a_sensitivity_summary.csv"
            summary_md = outdir / "test_a_sensitivity_summary.md"
            self.assertTrue(summary_csv.exists())
            self.assertTrue(summary_md.exists())
            with summary_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["threshold"] for row in rows], ["1.0", "1.2"])
            self.assertEqual([row["reference_mode"] for row in rows], ["central", "central"])


if __name__ == "__main__":
    unittest.main()
