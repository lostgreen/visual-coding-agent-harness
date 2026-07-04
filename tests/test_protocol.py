import unittest

from visual_coding_agent_harness.legacy.core.protocol import ToolRequest, ToolResult
from visual_coding_agent_harness.legacy.core.registry import ToolRegistry, tool


class ProtocolTest(unittest.TestCase):
    def test_tool_request_and_result_round_trip(self):
        request = ToolRequest(
            tool="crop_region",
            arguments={"image_path": "input/frame.jpg", "bbox": [100, 100, 500, 500]},
            request_id="req_001",
            caller="spatial_worker",
        )

        encoded = request.to_dict()
        decoded = ToolRequest.from_dict(encoded)

        self.assertEqual(decoded.tool, "crop_region")
        self.assertEqual(decoded.arguments["bbox"], [100, 100, 500, 500])
        self.assertEqual(decoded.request_id, "req_001")
        self.assertEqual(decoded.caller, "spatial_worker")

        result = ToolResult.from_mapping(
            request=decoded,
            output={
                "claim": "Cropped region saved.",
                "confidence": 1.0,
                "output_artifacts": ["artifacts/crops/frame_crop.png"],
                "regions": [{"bbox": [100, 100, 500, 500]}],
            },
        )

        self.assertEqual(result.tool, "crop_region")
        self.assertEqual(result.request_id, "req_001")
        self.assertEqual(result.output_artifacts, ["artifacts/crops/frame_crop.png"])
        self.assertEqual(result.regions[0]["bbox"], [100, 100, 500, 500])

    def test_registry_executes_batch_requests_in_order(self):
        registry = ToolRegistry()

        @tool(name="echo", description="Echo a claim.")
        def echo(claim: str):
            return {"claim": claim, "confidence": 1.0}

        registry.register(echo)

        results = registry.execute_batch(
            [
                ToolRequest(tool="echo", arguments={"claim": "first"}, request_id="r1"),
                ToolRequest(tool="echo", arguments={"claim": "second"}, request_id="r2"),
            ]
        )

        self.assertEqual([result.request_id for result in results], ["r1", "r2"])
        self.assertEqual([result.claim for result in results], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
