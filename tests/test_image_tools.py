import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visual_coding_agent_harness.tools.image_atomic import (
    crop_region,
    enhance_image,
    threshold_image,
    zoom_region,
)


class ImageAtomicToolsTest(unittest.TestCase):
    def test_crop_region_accepts_normalized_bbox_and_writes_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            output_path = Path(tmp) / "crop.png"
            Image.new("RGB", (100, 80), color=(255, 0, 0)).save(image_path)

            result = crop_region(
                image_path=str(image_path),
                bbox=[100, 250, 700, 750],
                output_path=str(output_path),
            )

            with Image.open(output_path) as cropped:
                self.assertEqual(cropped.size, (60, 40))
            self.assertEqual(result["output_artifacts"], [str(output_path)])
            self.assertEqual(result["regions"][0]["bbox"], [100, 250, 700, 750])
            self.assertEqual(result["regions"][0]["pixel_bbox"], [10, 20, 70, 60])

    def test_zoom_region_crops_then_resizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            output_path = Path(tmp) / "zoom.png"
            Image.new("RGB", (100, 80), color=(0, 255, 0)).save(image_path)

            result = zoom_region(
                image_path=str(image_path),
                bbox=[0, 0, 500, 500],
                output_path=str(output_path),
                scale=2,
            )

            with Image.open(output_path) as zoomed:
                self.assertEqual(zoomed.size, (100, 80))
            self.assertIn("zoomed", result["claim"])

    def test_threshold_and_enhance_write_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            threshold_path = Path(tmp) / "threshold.png"
            enhance_path = Path(tmp) / "enhanced.png"
            Image.new("RGB", (20, 20), color=(120, 120, 120)).save(image_path)

            threshold_result = threshold_image(str(image_path), str(threshold_path), threshold=100)
            enhance_result = enhance_image(str(image_path), str(enhance_path), sharpness=1.5, contrast=1.2)

            self.assertTrue(threshold_path.exists())
            self.assertTrue(enhance_path.exists())
            self.assertEqual(threshold_result["output_artifacts"], [str(threshold_path)])
            self.assertEqual(enhance_result["output_artifacts"], [str(enhance_path)])


if __name__ == "__main__":
    unittest.main()
