import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visual_coding_agent_harness.protocol import ToolRequest
from visual_coding_agent_harness.tools.traditional import build_traditional_registry


class TraditionalRegistryTest(unittest.TestCase):
    def test_build_traditional_registry_executes_image_tools_by_name(self):
        registry = build_traditional_registry()

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            crop_path = Path(tmp) / "crop.png"
            zoom_path = Path(tmp) / "zoom.png"
            Image.new("RGB", (100, 80), color=(0, 0, 255)).save(image_path)

            results = registry.execute_batch(
                [
                    ToolRequest(
                        tool="crop_region",
                        arguments={
                            "image_path": str(image_path),
                            "bbox": [0, 0, 500, 500],
                            "output_path": str(crop_path),
                        },
                        request_id="crop_1",
                        caller="spatial_worker",
                    ),
                    ToolRequest(
                        tool="zoom_region",
                        arguments={
                            "image_path": str(image_path),
                            "bbox": [0, 0, 500, 500],
                            "output_path": str(zoom_path),
                            "scale": 2,
                        },
                        request_id="zoom_1",
                        caller="spatial_worker",
                    ),
                ]
            )

            self.assertEqual([result.request_id for result in results], ["crop_1", "zoom_1"])
            self.assertTrue(crop_path.exists())
            self.assertTrue(zoom_path.exists())


if __name__ == "__main__":
    unittest.main()
