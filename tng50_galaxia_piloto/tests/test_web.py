from __future__ import annotations

import numpy as np
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot_viewer.web import create_app


class FakeSelection:
    def __init__(self, label, quantity, quantity_label, selected_count, sampled_count, packed):
        self.label = label
        self.quantity = quantity
        self.quantity_label = quantity_label
        self.selected_count = selected_count
        self.sampled_count = sampled_count
        self.packed = packed


class FakeStore:
    def get_config(self):
        return {
            "summary": {"canonical_id": "fake"},
            "quantities": {"stars": ["mass"], "gas": ["mass"]},
            "quantity_meta": {
                "stars": {"mass": {"label": "Mass", "unit": "u"}},
                "gas": {"mass": {"label": "Mass", "unit": "u"}},
            },
            "views": {"faceon": "Face-on", "edgeon": "Edge-on"},
            "defaults": {
                "component": "stars",
                "quantity": "mass",
                "view": "faceon",
                "radius_kpc": 10,
                "bins": 32,
                "profile_bins": 16,
                "cloud_detail": "equilibrada",
            },
            "limits": {"radius_kpc": {"min": 8, "max": 80}},
            "cloud_presets": {
                "rapida": {"label": "Rapida", "max_stars": 1000, "max_gas": 500},
                "equilibrada": {"label": "Equilibrada", "max_stars": 2000, "max_gas": 1000},
                "completa": {"label": "Completa", "max_stars": 0, "max_gas": 0},
            },
        }

    def get_map(self, **_kwargs):
        return {
            "component": "stars",
            "quantity": "mass",
            "view": "faceon",
            "radius_kpc": 10.0,
            "bins": 32,
            "label": "Mass",
            "unit": "u",
            "axis_labels": {"x": "x", "y": "y"},
            "extent": {"xmin": -10, "xmax": 10, "ymin": -10, "ymax": 10},
            "vmin": 0.0,
            "vmax": 1.0,
            "data": [[0.0, 1.0], [None, 0.5]],
        }

    def get_profile(self, **_kwargs):
        return {
            "component": "stars",
            "quantity": "mass",
            "label": "Profile",
            "unit": "u",
            "radius_kpc": 10.0,
            "x": [0.0, 1.0],
            "y": [0.5, 0.2],
        }

    def get_particle_cloud(self, star_quantity="mass", gas_quantity="mass", **_kwargs):
        return {
            "radius_kpc": 10.0,
            "halfmass_radius_kpc": 4.2,
            "render_backend": "webgl",
            "stars": {
                "label": "Estrellas",
                "quantity": star_quantity,
                "quantity_label": "Mass",
                "selected_count": 20,
                "sampled_count": 4,
            },
            "gas": {
                "label": "Gas",
                "quantity": gas_quantity,
                "quantity_label": "Mass",
                "selected_count": 10,
                "sampled_count": 2,
            },
        }

    def get_particle_selection(self, component, quantity="mass", **_kwargs):
        if component == "stars":
            array = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.5]], dtype=np.float32)
            return FakeSelection("Estrellas", quantity, "Mass", 20, 2, array)
        array = np.array([[0.0, 0.0, 1.0, 0.8]], dtype=np.float32)
        return FakeSelection("Gas", quantity, "Mass", 10, 1, array)


class WebTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(FakeStore()).test_client()

    def test_health_endpoint(self):
        response = self.app.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_config_endpoint(self):
        response = self.app.get("/api/config")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["summary"]["canonical_id"], "fake")

    def test_map_endpoint(self):
        response = self.app.get("/api/map?component=stars&quantity=mass&view=faceon")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["label"], "Mass")

    def test_particles_3d_endpoint(self):
        response = self.app.get("/api/particles3d?radius_kpc=10&max_stars=4&max_gas=2&star_quantity=metallicity")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["stars"]["sampled_count"], 4)
        self.assertEqual(response.get_json()["stars"]["quantity"], "metallicity")

    def test_particles_3d_buffer_endpoint(self):
        response = self.app.get("/api/particles3d/buffer?component=stars&quantity=metallicity&radius_kpc=10&max_points=4")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Float-Stride"], "4")
        self.assertEqual(response.headers["X-Quantity"], "metallicity")
        self.assertGreater(len(response.data), 0)


if __name__ == "__main__":
    unittest.main()
