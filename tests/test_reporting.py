"""Regression checks for native-resolution inspection reporting.

Run from the project root with:
    .venv\\Scripts\\python.exe -B tests\\test_reporting.py
"""
from __future__ import annotations

import csv
import json
import sys
import unittest
from io import StringIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import selection_for_checkpoint  # noqa: E402
from surfaceguard.reporting import (  # noqa: E402
    build_inspection,
    export_payload,
    inspection_html_report,
    regions_csv,
    resize_labels_to_original,
    resize_probabilities_to_original,
)


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.model_labels = np.array([[1, 1], [0, 2]], dtype=np.uint8)
        self.original = np.zeros((4, 6, 3), dtype=np.uint8)
        self.probabilities = np.zeros((4, 2, 2), dtype=np.float32)
        self.probabilities[0] = 1.0
        self.probabilities[1, 0, 0] = 0.9
        self.probabilities[0, 0, 0] = 0.1
        self.probabilities[1, 0, 1] = 0.9
        self.probabilities[0, 0, 1] = 0.1
        self.probabilities[2, 1, 1] = 0.8
        self.probabilities[0, 1, 1] = 0.2

    def test_label_resize_preserves_ids(self):
        resized = resize_labels_to_original(self.model_labels, self.original.shape[:2])
        self.assertEqual(resized.shape, self.original.shape[:2])
        self.assertEqual(set(np.unique(resized).tolist()), {0, 1, 2})
        self.assertEqual(int((resized == 1).sum()), 12)
        self.assertEqual(int((resized == 2).sum()), 6)

    def test_probability_resize_is_normalized(self):
        resized = resize_probabilities_to_original(self.probabilities, self.original.shape[:2])
        self.assertEqual(resized.shape, (4, 4, 6))
        self.assertTrue(np.allclose(resized.sum(axis=0), 1.0))
        self.assertGreater(float(resized[1].max()), 0.5)

    def test_regions_csv_uses_native_resolution(self):
        inspection = build_inspection(
            self.original,
            self.model_labels,
            self.probabilities,
            "native_sample.png",
            "Hybrid",
        )
        self.assertEqual(inspection["visualization_resolution"], {"width": 6, "height": 4})
        self.assertEqual(inspection["inference_resolution"], {"width": 2, "height": 2})
        self.assertEqual(inspection["affected_area_pixels"], 18)
        self.assertEqual(len(inspection["regions"]), 2)
        self.assertTrue(np.any(inspection["annotated_overlay"] != inspection["native_overlay"]))

        rows = list(csv.DictReader(StringIO(regions_csv(inspection).decode("utf-8"))))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["image_filename"] for row in rows}, {"native_sample.png"})
        self.assertEqual(sum(int(row["defect_area_pixels"]) for row in rows), 18)
        self.assertAlmostEqual(sum(float(row["percent_of_original_image"]) for row in rows), 75.0, places=4)
        self.assertTrue(all(row["mean_class_probability"] for row in rows))

    def test_html_report_embeds_all_images_and_payload_is_json_safe(self):
        inspection = build_inspection(
            self.original,
            self.model_labels,
            self.probabilities,
            "native_sample.png",
            "Hybrid",
        )
        report = inspection_html_report(
            inspection,
            {"detection_method": "Hybrid", "model": "TinyUNet"},
            severity="Low",
        )
        self.assertIn("SurfaceGuard", report)
        self.assertIn("MODEL PREDICTION", report)
        self.assertIn("native_sample.png", report)
        self.assertGreaterEqual(report.count("data:image/png;base64,"), 3)
        payload = export_payload(inspection, severity="Low")
        self.assertNotIn("original_rgb", payload)
        self.assertNotIn("native_labels", payload)
        json.dumps(payload)

    def test_published_checkpoints_resolve_to_their_own_selection_files(self):
        self.assertEqual(selection_for_checkpoint("models/best.pt").name, "selection.json")
        self.assertEqual(
            selection_for_checkpoint("models/best_perclass.pt").name,
            "best_perclass_selection.json",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
